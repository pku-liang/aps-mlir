#include "APS/PassDetail.h"
#include "APS/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "infer-affine-mem-access"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::affine;
using namespace mlir::memref;

namespace {

// Helper to check if a value is a constant integer
static std::optional<int64_t> getConstantIntValue(Value val) {
  if (auto constOp = val.getDefiningOp<arith::ConstantIntOp>())
    return constOp.value();
  if (auto constOp = val.getDefiningOp<arith::ConstantIndexOp>())
    return constOp.value();
  return std::nullopt;
}

// Helper to check if a value is an affine.for induction variable
static bool isAffineForInductionVar(Value val) {
  if (auto blockArg = llvm::dyn_cast<BlockArgument>(val)) {
    if (auto forOp = dyn_cast<AffineForOp>(blockArg.getOwner()->getParentOp()))
      return forOp.getInductionVar() == blockArg;
  }
  return false;
}

// Try to express a value as a linear combination of induction variables + symbols + constant
// Returns: {found, inductionVars, coefficients, symbols, symbolCoefficients, constant}
struct MultiDimAffineInfo {
  bool found = false;
  SmallVector<Value> inductionVars;         // List of IVs [i, j, ...]
  SmallVector<int64_t> coefficients;        // Coefficients for IVs [a, b, ...]
  SmallVector<Value> symbols;               // Loop-invariant values [s0, s1, ...]
  SmallVector<int64_t> symbolCoefficients;  // Coefficients for symbols [c, d, ...]
  int64_t constant = 0;                     // Constant offset
};

// Try to express a value as a*x + b where x is an induction variable
// Returns: {found, inductionVar, a, b}
struct AffineExprInfo {
  bool found = false;
  Value inductionVar;
  int64_t multiplier = 1;  // 'a' coefficient
  int64_t offset = 0;      // 'b' constant
};

// Helper to trace back through index_cast and type conversions to find if a value comes from an IV
static Value findInductionVar(Value val) {
  // First check if val itself is an IV
  if (isAffineForInductionVar(val))
    return val;

  // Trace through index_cast and other cast operations
  Value current = val;
  while (true) {
    if (auto castOp = current.getDefiningOp<arith::IndexCastOp>()) {
      current = castOp.getIn();
      if (isAffineForInductionVar(current))
        return current;
      continue;
    }

    // Also trace through ExtSI/ExtUI for sign/zero extension
    if (auto extOp = current.getDefiningOp<arith::ExtSIOp>()) {
      current = extOp.getIn();
      if (isAffineForInductionVar(current))
        return current;
      continue;
    }

    if (auto extOp = current.getDefiningOp<arith::ExtUIOp>()) {
      current = extOp.getIn();
      if (isAffineForInductionVar(current))
        return current;
      continue;
    }

    break;
  }

  return Value();
}

// Try to evaluate a value as a compile-time constant by tracing through arithmetic ops
static std::optional<int64_t> tryEvaluateConstant(Value val) {
  // Direct constant check
  if (auto constVal = getConstantIntValue(val))
    return constVal;

  // Trace through casts
  if (auto castOp = val.getDefiningOp<arith::IndexCastOp>())
    return tryEvaluateConstant(castOp.getIn());
  if (auto extOp = val.getDefiningOp<arith::ExtSIOp>())
    return tryEvaluateConstant(extOp.getIn());
  if (auto extOp = val.getDefiningOp<arith::ExtUIOp>())
    return tryEvaluateConstant(extOp.getIn());
  if (auto truncOp = val.getDefiningOp<arith::TruncIOp>())
    return tryEvaluateConstant(truncOp.getIn());

  // Evaluate binary operations
  if (auto addOp = val.getDefiningOp<arith::AddIOp>()) {
    auto lhs = tryEvaluateConstant(addOp.getLhs());
    auto rhs = tryEvaluateConstant(addOp.getRhs());
    if (lhs && rhs)
      return *lhs + *rhs;
  }

  if (auto mulOp = val.getDefiningOp<arith::MulIOp>()) {
    auto lhs = tryEvaluateConstant(mulOp.getLhs());
    auto rhs = tryEvaluateConstant(mulOp.getRhs());
    if (lhs && rhs)
      return *lhs * *rhs;
  }

  if (auto subOp = val.getDefiningOp<arith::SubIOp>()) {
    auto lhs = tryEvaluateConstant(subOp.getLhs());
    auto rhs = tryEvaluateConstant(subOp.getRhs());
    if (lhs && rhs)
      return *lhs - *rhs;
  }

  return std::nullopt;
}

// Helper to check if a value is defined outside all enclosing affine.for loops
static bool isLoopInvariant(Value val, Operation *contextOp) {
  // Find all enclosing affine.for operations
  Operation *current = contextOp->getParentOp();
  while (current) {
    if (auto forOp = dyn_cast<AffineForOp>(current)) {
      // Check if val is defined inside this loop's region
      if (auto *defOp = val.getDefiningOp()) {
        if (forOp.getRegion().isAncestor(defOp->getParentRegion()))
          return false;  // Defined inside the loop, not invariant
      }
      if (auto blockArg = llvm::dyn_cast<BlockArgument>(val)) {
        if (forOp.getRegion().isAncestor(blockArg.getOwner()->getParent()))
          return false;  // Block argument inside the loop
      }
    }
    current = current->getParentOp();
  }
  return true;  // Defined outside all loops
}

// Analyze multi-dimensional affine expression: a*i + b*j + ... + c*s0 + d*s1 + ... + e
static MultiDimAffineInfo analyzeMultiDimIndex(Value index, Operation *contextOp) {
  MultiDimAffineInfo info;

  // Trace through index_cast operations
  Value current = index;
  while (auto castOp = current.getDefiningOp<arith::IndexCastOp>()) {
    current = castOp.getIn();
  }

  // Map from induction variable to its coefficient
  llvm::DenseMap<Value, int64_t> ivCoefficients;
  // Map from symbol (loop-invariant value) to its coefficient
  llvm::DenseMap<Value, int64_t> symbolCoefficients;
  int64_t constantTerm = 0;

  // Recursively decompose additions
  std::function<bool(Value)> decomposeAdditions = [&](Value val) -> bool {
    // Trace through index_cast and other cast operations
    while (true) {
      if (auto castOp = val.getDefiningOp<arith::IndexCastOp>()) {
        val = castOp.getIn();
        continue;
      }
      if (auto extOp = val.getDefiningOp<arith::ExtSIOp>()) {
        val = extOp.getIn();
        continue;
      }
      if (auto extOp = val.getDefiningOp<arith::ExtUIOp>()) {
        val = extOp.getIn();
        continue;
      }
      if (auto truncOp = val.getDefiningOp<arith::TruncIOp>()) {
        val = truncOp.getIn();
        continue;
      }
      break;
    }

    // Check if it's a constant
    if (auto constVal = getConstantIntValue(val)) {
      constantTerm += *constVal;
      return true;
    }

    // Check if it's an IV directly (coefficient = 1)
    if (Value iv = findInductionVar(val)) {
      ivCoefficients[iv] = ivCoefficients.lookup(iv) + 1;
      return true;
    }

    // Check for multiplication: a * IV or a * symbol
    if (auto mulOp = val.getDefiningOp<arith::MulIOp>()) {
      Value lhs = mulOp.getLhs();
      Value rhs = mulOp.getRhs();

      // Try lhs = constant, rhs = IV
      if (auto coeff = tryEvaluateConstant(lhs)) {
        if (Value iv = findInductionVar(rhs)) {
          ivCoefficients[iv] = ivCoefficients.lookup(iv) + *coeff;
          return true;
        }
      }

      // Try lhs = IV, rhs = constant
      if (auto coeff = tryEvaluateConstant(rhs)) {
        if (Value iv = findInductionVar(lhs)) {
          ivCoefficients[iv] = ivCoefficients.lookup(iv) + *coeff;
          return true;
        }
      }

      // Try lhs = constant, rhs = loop-invariant symbol
      // Trace through casts on rhs to find the underlying loop-invariant value
      if (auto coeff = tryEvaluateConstant(lhs)) {
        Value rhsTraced = rhs;
        while (true) {
          if (auto castOp = rhsTraced.getDefiningOp<arith::IndexCastOp>()) {
            rhsTraced = castOp.getIn();
          } else if (auto extOp = rhsTraced.getDefiningOp<arith::ExtSIOp>()) {
            rhsTraced = extOp.getIn();
          } else if (auto extOp = rhsTraced.getDefiningOp<arith::ExtUIOp>()) {
            rhsTraced = extOp.getIn();
          } else if (auto truncOp = rhsTraced.getDefiningOp<arith::TruncIOp>()) {
            rhsTraced = truncOp.getIn();
          } else {
            break;
          }
        }
        if (isLoopInvariant(rhsTraced, contextOp)) {
          symbolCoefficients[rhsTraced] = symbolCoefficients.lookup(rhsTraced) + *coeff;
          return true;
        }
      }

      // Try lhs = loop-invariant symbol, rhs = constant
      // Trace through casts on lhs to find the underlying loop-invariant value
      if (auto coeff = tryEvaluateConstant(rhs)) {
        Value lhsTraced = lhs;
        while (true) {
          if (auto castOp = lhsTraced.getDefiningOp<arith::IndexCastOp>()) {
            lhsTraced = castOp.getIn();
          } else if (auto extOp = lhsTraced.getDefiningOp<arith::ExtSIOp>()) {
            lhsTraced = extOp.getIn();
          } else if (auto extOp = lhsTraced.getDefiningOp<arith::ExtUIOp>()) {
            lhsTraced = extOp.getIn();
          } else if (auto truncOp = lhsTraced.getDefiningOp<arith::TruncIOp>()) {
            lhsTraced = truncOp.getIn();
          } else {
            break;
          }
        }
        if (isLoopInvariant(lhsTraced, contextOp)) {
          symbolCoefficients[lhsTraced] = symbolCoefficients.lookup(lhsTraced) + *coeff;
          return true;
        }
      }
    }

    // Check for addition: decompose both sides
    if (auto addOp = val.getDefiningOp<arith::AddIOp>()) {
      return decomposeAdditions(addOp.getLhs()) &&
             decomposeAdditions(addOp.getRhs());
    }

    // If nothing else matched, check if it's a loop-invariant symbol (coefficient = 1)
    if (isLoopInvariant(val, contextOp)) {
      symbolCoefficients[val] = symbolCoefficients.lookup(val) + 1;
      return true;
    }

    return false;
  };

  // Try to decompose the expression
  if (!decomposeAdditions(current)) {
    info.found = false;
    return info;
  }

  // Must have at least one IV for affine analysis to be useful
  if (ivCoefficients.empty()) {
    info.found = false;
    return info;
  }

  // Build the result
  info.found = true;
  info.constant = constantTerm;
  for (auto &pair : ivCoefficients) {
    info.inductionVars.push_back(pair.first);
    info.coefficients.push_back(pair.second);
  }
  for (auto &pair : symbolCoefficients) {
    info.symbols.push_back(pair.first);
    info.symbolCoefficients.push_back(pair.second);
  }

  return info;
}

// Trace through index_cast, muli, addi to find affine pattern
static AffineExprInfo analyzeIndex(Value index) {
  AffineExprInfo info;

  // Trace through index_cast and other cast operations to find the computation
  Value current = index;
  while (true) {
    if (auto castOp = current.getDefiningOp<arith::IndexCastOp>()) {
      current = castOp.getIn();
      continue;
    }
    if (auto extOp = current.getDefiningOp<arith::ExtSIOp>()) {
      current = extOp.getIn();
      continue;
    }
    if (auto extOp = current.getDefiningOp<arith::ExtUIOp>()) {
      current = extOp.getIn();
      continue;
    }
    if (auto truncOp = current.getDefiningOp<arith::TruncIOp>()) {
      current = truncOp.getIn();
      continue;
    }
    break;
  }

  // Check if it's directly an induction variable: x
  if (Value iv = findInductionVar(current)) {
    info.found = true;
    info.inductionVar = iv;
    info.multiplier = 1;
    info.offset = 0;
    return info;
  }

  // Check for pattern: x + b
  if (auto addOp = current.getDefiningOp<arith::AddIOp>()) {
    Value lhs = addOp.getLhs();
    Value rhs = addOp.getRhs();

    // Try lhs = x, rhs = b
    if (Value iv = findInductionVar(lhs)) {
      if (auto offset = getConstantIntValue(rhs)) {
        info.found = true;
        info.inductionVar = iv;
        info.multiplier = 1;
        info.offset = *offset;
        return info;
      }
    }

    // Try lhs = b, rhs = x
    if (Value iv = findInductionVar(rhs)) {
      if (auto offset = getConstantIntValue(lhs)) {
        info.found = true;
        info.inductionVar = iv;
        info.multiplier = 1;
        info.offset = *offset;
        return info;
      }
    }

    // Try lhs = a*x, rhs = b
    if (auto mulOp = lhs.getDefiningOp<arith::MulIOp>()) {
      auto subInfo = analyzeIndex(lhs);
      if (subInfo.found && subInfo.offset == 0) {
        if (auto offset = getConstantIntValue(rhs)) {
          info.found = true;
          info.inductionVar = subInfo.inductionVar;
          info.multiplier = subInfo.multiplier;
          info.offset = *offset;
          return info;
        }
      }
    }

    // Try lhs = b, rhs = a*x
    if (auto mulOp = rhs.getDefiningOp<arith::MulIOp>()) {
      auto subInfo = analyzeIndex(rhs);
      if (subInfo.found && subInfo.offset == 0) {
        if (auto offset = getConstantIntValue(lhs)) {
          info.found = true;
          info.inductionVar = subInfo.inductionVar;
          info.multiplier = subInfo.multiplier;
          info.offset = *offset;
          return info;
        }
      }
    }
  }

  // Check for pattern: a * x
  if (auto mulOp = current.getDefiningOp<arith::MulIOp>()) {
    Value lhs = mulOp.getLhs();
    Value rhs = mulOp.getRhs();

    // Try lhs = x, rhs = a (with potential casts on lhs)
    if (Value iv = findInductionVar(lhs)) {
      if (auto multiplier = getConstantIntValue(rhs)) {
        info.found = true;
        info.inductionVar = iv;
        info.multiplier = *multiplier;
        info.offset = 0;
        LLVM_DEBUG(llvm::dbgs() << "Found pattern: " << *multiplier << " * IV (lhs)\n");
        return info;
      }
    }

    // Try lhs = a, rhs = x (with potential casts on rhs)
    if (Value iv = findInductionVar(rhs)) {
      if (auto multiplier = getConstantIntValue(lhs)) {
        info.found = true;
        info.inductionVar = iv;
        info.multiplier = *multiplier;
        info.offset = 0;
        LLVM_DEBUG(llvm::dbgs() << "Found pattern: " << *multiplier << " * IV (rhs)\n");
        return info;
      }
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Could not analyze index: " << index << "\n");
  return info;
}

// Helper to convert a value to index type if needed.
// Important: Insert the cast right after the value's definition to keep it outside loops.
static Value ensureIndexType(Value val, PatternRewriter &rewriter) {
  if (val.getType().isIndex())
    return val;

  // Find the insertion point: right after the defining op (or at function start for block args)
  OpBuilder::InsertionGuard guard(rewriter);
  if (auto *defOp = val.getDefiningOp()) {
    rewriter.setInsertionPointAfter(defOp);
  } else if (auto blockArg = llvm::dyn_cast<BlockArgument>(val)) {
    // For block arguments, insert at the beginning of the block
    rewriter.setInsertionPointToStart(blockArg.getOwner());
  }

  return rewriter.create<arith::IndexCastOp>(val.getLoc(), rewriter.getIndexType(), val);
}

// Pattern to convert memref.load with affine index to affine.load
struct InferAffineLoadPattern : public OpRewritePattern<memref::LoadOp> {
  using OpRewritePattern<memref::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    // Only handle single-dimensional loads for now
    if (loadOp.getIndices().size() != 1)
      return failure();

    Value index = loadOp.getIndices()[0];

    // Try multi-dimensional analysis with symbol support
    MultiDimAffineInfo multiInfo = analyzeMultiDimIndex(index, loadOp);
    if (multiInfo.found) {
      LLVM_DEBUG(llvm::dbgs() << "Found multi-dim affine pattern with "
                              << multiInfo.inductionVars.size() << " IVs and "
                              << multiInfo.symbols.size() << " symbols for: "
                              << loadOp << "\n");

      // Build affine map: (d0, d1, ...)[s0, s1, ...] -> (c0*d0 + c1*d1 + ... + e0*s0 + e1*s1 + ... + constant)
      unsigned numDims = multiInfo.inductionVars.size();
      unsigned numSymbols = multiInfo.symbols.size();

      AffineExpr affineExpr = rewriter.getAffineConstantExpr(multiInfo.constant);

      // Add dimension terms (IVs)
      for (unsigned i = 0; i < numDims; ++i) {
        AffineExpr dimExpr = rewriter.getAffineDimExpr(i);
        affineExpr = affineExpr + dimExpr * multiInfo.coefficients[i];
      }

      // Add symbol terms (loop-invariant values)
      for (unsigned i = 0; i < numSymbols; ++i) {
        AffineExpr symExpr = rewriter.getAffineSymbolExpr(i);
        affineExpr = affineExpr + symExpr * multiInfo.symbolCoefficients[i];
      }

      AffineMap map = AffineMap::get(numDims, numSymbols, affineExpr);

      // Collect map operands: first IVs (dims), then symbols
      SmallVector<Value> mapOperands;
      for (Value iv : multiInfo.inductionVars) {
        mapOperands.push_back(iv);  // IVs are already index type
      }
      for (Value sym : multiInfo.symbols) {
        mapOperands.push_back(ensureIndexType(sym, rewriter));
      }

      // Create affine.load with all operands
      auto affineLoad = rewriter.create<AffineLoadOp>(
          loadOp.getLoc(), loadOp.getMemRef(), map, mapOperands);

      rewriter.replaceOp(loadOp, affineLoad.getResult());

      LLVM_DEBUG(llvm::dbgs() << "Replaced with affine.load using map: " << map << "\n");
      return success();
    }

    // Fallback to single-dimensional analysis
    AffineExprInfo info = analyzeIndex(index);
    if (!info.found) {
      // Last resort: check if index is a pure constant expression
      if (auto constIndex = tryEvaluateConstant(index)) {
        LLVM_DEBUG(llvm::dbgs() << "Found constant index " << *constIndex
                                << " for: " << loadOp << "\n");

        // Build affine map: () -> (constant)
        AffineExpr affineExpr = rewriter.getAffineConstantExpr(*constIndex);
        AffineMap map = AffineMap::get(0, 0, affineExpr);

        // Create affine.load with constant index (no IV needed)
        auto affineLoad = rewriter.create<AffineLoadOp>(
            loadOp.getLoc(), loadOp.getMemRef(), map, ValueRange{});

        rewriter.replaceOp(loadOp, affineLoad.getResult());

        LLVM_DEBUG(llvm::dbgs() << "Replaced with affine.load using constant map: "
                                << map << "\n");
        return success();
      }

      LLVM_DEBUG(llvm::dbgs() << "Could not infer affine pattern for: " << loadOp << "\n");
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "Found affine pattern: " << info.multiplier
                            << "*x + " << info.offset << " for: " << loadOp << "\n");

    // Build affine map: (d0) -> (a * d0 + b)
    AffineExpr dimExpr = rewriter.getAffineDimExpr(0);
    AffineExpr affineExpr = dimExpr * info.multiplier + info.offset;
    AffineMap map = AffineMap::get(1, 0, affineExpr);

    // Create affine.load with the induction variable
    auto affineLoad = rewriter.create<AffineLoadOp>(
        loadOp.getLoc(), loadOp.getMemRef(), map, ValueRange{info.inductionVar});

    rewriter.replaceOp(loadOp, affineLoad.getResult());

    LLVM_DEBUG(llvm::dbgs() << "Replaced with affine.load using map: " << map << "\n");
    return success();
  }
};

// Pattern to convert memref.store with affine index to affine.store
struct InferAffineStorePattern : public OpRewritePattern<memref::StoreOp> {
  using OpRewritePattern<memref::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    // Only handle single-dimensional stores for now
    if (storeOp.getIndices().size() != 1)
      return failure();

    Value index = storeOp.getIndices()[0];

    // Try multi-dimensional analysis with symbol support
    MultiDimAffineInfo multiInfo = analyzeMultiDimIndex(index, storeOp);
    if (multiInfo.found) {
      LLVM_DEBUG(llvm::dbgs() << "Found multi-dim affine pattern with "
                              << multiInfo.inductionVars.size() << " IVs and "
                              << multiInfo.symbols.size() << " symbols for: "
                              << storeOp << "\n");

      // Build affine map: (d0, d1, ...)[s0, s1, ...] -> (c0*d0 + c1*d1 + ... + e0*s0 + e1*s1 + ... + constant)
      unsigned numDims = multiInfo.inductionVars.size();
      unsigned numSymbols = multiInfo.symbols.size();

      AffineExpr affineExpr = rewriter.getAffineConstantExpr(multiInfo.constant);

      // Add dimension terms (IVs)
      for (unsigned i = 0; i < numDims; ++i) {
        AffineExpr dimExpr = rewriter.getAffineDimExpr(i);
        affineExpr = affineExpr + dimExpr * multiInfo.coefficients[i];
      }

      // Add symbol terms (loop-invariant values)
      for (unsigned i = 0; i < numSymbols; ++i) {
        AffineExpr symExpr = rewriter.getAffineSymbolExpr(i);
        affineExpr = affineExpr + symExpr * multiInfo.symbolCoefficients[i];
      }

      AffineMap map = AffineMap::get(numDims, numSymbols, affineExpr);

      // Collect map operands: first IVs (dims), then symbols
      SmallVector<Value> mapOperands;
      for (Value iv : multiInfo.inductionVars) {
        mapOperands.push_back(iv);  // IVs are already index type
      }
      for (Value sym : multiInfo.symbols) {
        mapOperands.push_back(ensureIndexType(sym, rewriter));
      }

      // Create affine.store with all operands
      rewriter.create<AffineStoreOp>(
          storeOp.getLoc(), storeOp.getValue(), storeOp.getMemRef(),
          map, mapOperands);

      rewriter.eraseOp(storeOp);

      LLVM_DEBUG(llvm::dbgs() << "Replaced with affine.store using map: " << map << "\n");
      return success();
    }

    // Fallback to single-dimensional analysis
    AffineExprInfo info = analyzeIndex(index);
    if (!info.found) {
      // Last resort: check if index is a pure constant expression
      if (auto constIndex = tryEvaluateConstant(index)) {
        LLVM_DEBUG(llvm::dbgs() << "Found constant index " << *constIndex
                                << " for: " << storeOp << "\n");

        // Build affine map: () -> (constant)
        AffineExpr affineExpr = rewriter.getAffineConstantExpr(*constIndex);
        AffineMap map = AffineMap::get(0, 0, affineExpr);

        // Create affine.store with constant index (no IV needed)
        rewriter.create<AffineStoreOp>(
            storeOp.getLoc(), storeOp.getValue(), storeOp.getMemRef(),
            map, ValueRange{});

        rewriter.eraseOp(storeOp);

        LLVM_DEBUG(llvm::dbgs() << "Replaced with affine.store using constant map: "
                                << map << "\n");
        return success();
      }

      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "Found affine pattern: " << info.multiplier
                            << "*x + " << info.offset << " for: " << storeOp << "\n");

    // Build affine map: (d0) -> (a * d0 + b)
    AffineExpr dimExpr = rewriter.getAffineDimExpr(0);
    AffineExpr affineExpr = dimExpr * info.multiplier + info.offset;
    AffineMap map = AffineMap::get(1, 0, affineExpr);

    // Create affine.store with the induction variable
    rewriter.create<AffineStoreOp>(
        storeOp.getLoc(), storeOp.getValue(), storeOp.getMemRef(),
        map, ValueRange{info.inductionVar});

    rewriter.eraseOp(storeOp);

    LLVM_DEBUG(llvm::dbgs() << "Replaced with affine.store using map: " << map << "\n");
    return success();
  }
};

struct InferAffineMemAccessPass
    : public InferAffineMemAccessBase<InferAffineMemAccessPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<InferAffineLoadPattern, InferAffineStorePattern>(&getContext());

    GreedyRewriteConfig config;
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createInferAffineMemAccessPass() {
  return std::make_unique<InferAffineMemAccessPass>();
}
} // namespace mlir
