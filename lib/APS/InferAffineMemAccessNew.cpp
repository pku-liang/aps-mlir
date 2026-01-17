#include "APS/PassDetail.h"
#include "APS/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "infer-affine-mem-access-new"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::affine;
using namespace mlir::memref;

namespace {

// Check if a value is an affine.for induction variable
static bool isAffineForInductionVar(Value val) {
  if (auto blockArg = llvm::dyn_cast<BlockArgument>(val)) {
    if (auto forOp = dyn_cast<AffineForOp>(blockArg.getOwner()->getParentOp()))
      return forOp.getInductionVar() == blockArg;
  }
  return false;
}

// Check if a value is defined outside all enclosing affine.for loops
static bool isLoopInvariant(Value val, Operation *contextOp) {
  Operation *current = contextOp->getParentOp();
  while (current) {
    if (auto forOp = dyn_cast<AffineForOp>(current)) {
      if (auto *defOp = val.getDefiningOp()) {
        if (forOp.getRegion().isAncestor(defOp->getParentRegion()))
          return false;
      }
      if (auto blockArg = llvm::dyn_cast<BlockArgument>(val)) {
        if (forOp.getRegion().isAncestor(blockArg.getOwner()->getParent()))
          return false;
      }
    }
    current = current->getParentOp();
  }
  return true;
}

// Recursively clone an arithmetic expression tree, converting to index type.
// Returns the cloned value in index type, or nullptr if we can't handle it.
static Value cloneAsIndexType(Value val, OpBuilder &builder,
                              DenseMap<Value, Value> &cache,
                              Operation *contextOp) {
  // Check cache first
  auto it = cache.find(val);
  if (it != cache.end())
    return it->second;

  Value result;
  Location loc = val.getLoc();
  Type indexType = builder.getIndexType();

  // Handle casts FIRST - trace through them regardless of type
  if (auto castOp = val.getDefiningOp<arith::IndexCastOp>()) {
    result = cloneAsIndexType(castOp.getIn(), builder, cache, contextOp);
  }
  else if (auto extOp = val.getDefiningOp<arith::ExtSIOp>()) {
    result = cloneAsIndexType(extOp.getIn(), builder, cache, contextOp);
  }
  else if (auto extOp = val.getDefiningOp<arith::ExtUIOp>()) {
    result = cloneAsIndexType(extOp.getIn(), builder, cache, contextOp);
  }
  else if (auto truncOp = val.getDefiningOp<arith::TruncIOp>()) {
    result = cloneAsIndexType(truncOp.getIn(), builder, cache, contextOp);
  }
  // Handle constants
  else if (auto constOp = val.getDefiningOp<arith::ConstantIntOp>()) {
    result = builder.create<arith::ConstantIndexOp>(loc, constOp.value());
  }
  else if (auto constOp = val.getDefiningOp<arith::ConstantIndexOp>()) {
    result = val;
  }
  // Handle binary arithmetic - clone with index type operands
  else if (auto addOp = val.getDefiningOp<arith::AddIOp>()) {
    Value lhs = cloneAsIndexType(addOp.getLhs(), builder, cache, contextOp);
    Value rhs = cloneAsIndexType(addOp.getRhs(), builder, cache, contextOp);
    if (lhs && rhs)
      result = builder.create<arith::AddIOp>(loc, lhs, rhs);
  }
  else if (auto mulOp = val.getDefiningOp<arith::MulIOp>()) {
    Value lhs = cloneAsIndexType(mulOp.getLhs(), builder, cache, contextOp);
    Value rhs = cloneAsIndexType(mulOp.getRhs(), builder, cache, contextOp);
    if (lhs && rhs)
      result = builder.create<arith::MulIOp>(loc, lhs, rhs);
  }
  else if (auto subOp = val.getDefiningOp<arith::SubIOp>()) {
    Value lhs = cloneAsIndexType(subOp.getLhs(), builder, cache, contextOp);
    Value rhs = cloneAsIndexType(subOp.getRhs(), builder, cache, contextOp);
    if (lhs && rhs)
      result = builder.create<arith::SubIOp>(loc, lhs, rhs);
  }
  // If already index type (e.g., loop IV), return as-is
  else if (val.getType().isIndex()) {
    result = val;
  }
  // For values that are loop-invariant, cast them to index
  else if (val.getType().isIntOrIndex() && isLoopInvariant(val, contextOp)) {
    result = builder.create<arith::IndexCastOp>(loc, indexType, val);
  }
  // Unknown - give up
  else {
    LLVM_DEBUG(llvm::dbgs() << "Cannot clone to index type: " << val << "\n");
    return nullptr;
  }

  cache[val] = result;
  return result;
}

// Result of building an affine expression
struct AffineResult {
  bool success = false;
  AffineExpr expr;
  SmallVector<Value> dims;    // Induction variables (ordered)
  SmallVector<Value> symbols; // Loop-invariant values (ordered)
};

// Build an AffineExpr from a clean index-type value
static AffineResult buildAffineExpr(Value val, Operation *contextOp,
                                    MLIRContext *ctx,
                                    DenseMap<Value, unsigned> &dimMap,
                                    DenseMap<Value, unsigned> &symMap,
                                    SmallVectorImpl<Value> &dimList,
                                    SmallVectorImpl<Value> &symList) {
  AffineResult result;

  if (!val.getType().isIndex()) {
    LLVM_DEBUG(llvm::dbgs() << "Not index type: " << val << "\n");
    return result;
  }

  // Constant
  if (auto constOp = val.getDefiningOp<arith::ConstantIndexOp>()) {
    result.success = true;
    result.expr = getAffineConstantExpr(constOp.value(), ctx);
    return result;
  }

  // Induction variable -> dimension
  if (isAffineForInductionVar(val)) {
    unsigned dimIdx;
    auto it = dimMap.find(val);
    if (it != dimMap.end()) {
      dimIdx = it->second;
    } else {
      dimIdx = dimList.size();
      dimMap[val] = dimIdx;
      dimList.push_back(val);
    }
    result.success = true;
    result.expr = getAffineDimExpr(dimIdx, ctx);
    return result;
  }

  // Addition
  if (auto addOp = val.getDefiningOp<arith::AddIOp>()) {
    auto lhs = buildAffineExpr(addOp.getLhs(), contextOp, ctx, dimMap, symMap, dimList, symList);
    auto rhs = buildAffineExpr(addOp.getRhs(), contextOp, ctx, dimMap, symMap, dimList, symList);
    if (lhs.success && rhs.success) {
      result.success = true;
      result.expr = lhs.expr + rhs.expr;
      return result;
    }
  }

  // Multiplication - affine only allows dim*const or const*dim
  if (auto mulOp = val.getDefiningOp<arith::MulIOp>()) {
    auto lhs = buildAffineExpr(mulOp.getLhs(), contextOp, ctx, dimMap, symMap, dimList, symList);
    auto rhs = buildAffineExpr(mulOp.getRhs(), contextOp, ctx, dimMap, symMap, dimList, symList);
    if (lhs.success && rhs.success) {
      result.success = true;
      result.expr = lhs.expr * rhs.expr;
      return result;
    }
  }

  // Subtraction
  if (auto subOp = val.getDefiningOp<arith::SubIOp>()) {
    auto lhs = buildAffineExpr(subOp.getLhs(), contextOp, ctx, dimMap, symMap, dimList, symList);
    auto rhs = buildAffineExpr(subOp.getRhs(), contextOp, ctx, dimMap, symMap, dimList, symList);
    if (lhs.success && rhs.success) {
      result.success = true;
      result.expr = lhs.expr - rhs.expr;
      return result;
    }
  }

  // Loop-invariant value -> symbol
  if (isLoopInvariant(val, contextOp)) {
    unsigned symIdx;
    auto it = symMap.find(val);
    if (it != symMap.end()) {
      symIdx = it->second;
    } else {
      symIdx = symList.size();
      symMap[val] = symIdx;
      symList.push_back(val);
    }
    result.success = true;
    result.expr = getAffineSymbolExpr(symIdx, ctx);
    return result;
  }

  LLVM_DEBUG(llvm::dbgs() << "Cannot build affine expr for: " << val << "\n");
  return result;
}

// Pattern to convert memref.load to affine.load
struct ConvertLoadPattern : public OpRewritePattern<memref::LoadOp> {
  using OpRewritePattern<memref::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    if (loadOp.getIndices().size() != 1)
      return failure();

    Value origIndex = loadOp.getIndices()[0];
    MLIRContext *ctx = rewriter.getContext();

    // Step 1: Clone the index computation in index type
    DenseMap<Value, Value> cloneCache;
    rewriter.setInsertionPoint(loadOp);
    Value indexVal = cloneAsIndexType(origIndex, rewriter, cloneCache, loadOp);
    if (!indexVal) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to clone index as index type\n");
      return failure();
    }

    // Step 2: Build affine expression from the clean index
    DenseMap<Value, unsigned> dimMap, symMap;
    SmallVector<Value> dimList, symList;
    auto affineResult = buildAffineExpr(indexVal, loadOp, ctx, dimMap, symMap, dimList, symList);
    if (!affineResult.success) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to build affine expr\n");
      return failure();
    }

    // Step 3: Check if the expression is a valid affine expression
    // (affine expressions can have dim*const but not dim*dim)
    if (!affineResult.expr.isPureAffine()) {
      LLVM_DEBUG(llvm::dbgs() << "Expression is not pure affine: " << affineResult.expr << "\n");
      return failure();
    }

    // Step 4: Create the affine map and affine.load
    AffineMap map = AffineMap::get(dimList.size(), symList.size(), affineResult.expr);

    SmallVector<Value> mapOperands;
    mapOperands.append(dimList);
    mapOperands.append(symList);

    auto affineLoad = rewriter.create<AffineLoadOp>(
        loadOp.getLoc(), loadOp.getMemRef(), map, mapOperands);

    rewriter.replaceOp(loadOp, affineLoad.getResult());

    LLVM_DEBUG(llvm::dbgs() << "Converted to affine.load with map: " << map << "\n");
    return success();
  }
};

// Pattern to convert memref.store to affine.store
struct ConvertStorePattern : public OpRewritePattern<memref::StoreOp> {
  using OpRewritePattern<memref::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    if (storeOp.getIndices().size() != 1)
      return failure();

    Value origIndex = storeOp.getIndices()[0];
    MLIRContext *ctx = rewriter.getContext();

    // Step 1: Clone the index computation in index type
    DenseMap<Value, Value> cloneCache;
    rewriter.setInsertionPoint(storeOp);
    Value indexVal = cloneAsIndexType(origIndex, rewriter, cloneCache, storeOp);
    if (!indexVal) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to clone index as index type\n");
      return failure();
    }

    // Step 2: Build affine expression from the clean index
    DenseMap<Value, unsigned> dimMap, symMap;
    SmallVector<Value> dimList, symList;
    auto affineResult = buildAffineExpr(indexVal, storeOp, ctx, dimMap, symMap, dimList, symList);
    if (!affineResult.success) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to build affine expr\n");
      return failure();
    }

    // Step 3: Check if the expression is a valid affine expression
    if (!affineResult.expr.isPureAffine()) {
      LLVM_DEBUG(llvm::dbgs() << "Expression is not pure affine: " << affineResult.expr << "\n");
      return failure();
    }

    // Step 4: Create the affine map and affine.store
    AffineMap map = AffineMap::get(dimList.size(), symList.size(), affineResult.expr);

    SmallVector<Value> mapOperands;
    mapOperands.append(dimList);
    mapOperands.append(symList);

    rewriter.create<AffineStoreOp>(
        storeOp.getLoc(), storeOp.getValue(), storeOp.getMemRef(), map, mapOperands);

    rewriter.eraseOp(storeOp);

    LLVM_DEBUG(llvm::dbgs() << "Converted to affine.store with map: " << map << "\n");
    return success();
  }
};

struct InferAffineMemAccessNewPass
    : public InferAffineMemAccessNewBase<InferAffineMemAccessNewPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertLoadPattern, ConvertStorePattern>(&getContext());

    GreedyRewriteConfig config;
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
std::unique_ptr<OperationPass<func::FuncOp>> createInferAffineMemAccessNewPass() {
  return std::make_unique<InferAffineMemAccessNewPass>();
}
} // namespace mlir
