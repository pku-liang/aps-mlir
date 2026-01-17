//===- NormalizeMemrefIndices.cpp - Normalize memref indices for affine ---===//
//
// This pass normalizes memref load/store indices to be compatible with
// affine-raise-from-memref pass. Specifically, it pushes index_cast operations
// down through arithmetic operations so that the arithmetic is performed in
// index type rather than on integers that are then cast to index.
//
// Before:
//   %0 = arith.muli %iv, %c4 : i32
//   %1 = arith.index_cast %0 : i32 to index
//   memref.store %val, %mem[%1]
//
// After:
//   %0 = arith.index_cast %iv : i32 to index
//   %1 = arith.index_cast %c4 : i32 to index
//   %2 = arith.muli %0, %1 : index
//   memref.store %val, %mem[%2]
//
//===----------------------------------------------------------------------===//

#include "TOR/PassDetail.h"
#include "TOR/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "normalize-memref-indices"

using namespace mlir;

namespace {

// Helper to cast a value to index type
Value castToIndex(OpBuilder &builder, Location loc, Value value) {
  if (value.getType().isIndex())
    return value;
  return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), value);
}

// Pattern to normalize index_cast(arith_op(a, b)) -> arith_op(index_cast(a), index_cast(b))
struct NormalizeIndexCastPattern : public OpRewritePattern<arith::IndexCastOp> {
  using OpRewritePattern<arith::IndexCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::IndexCastOp castOp,
                                PatternRewriter &rewriter) const override {
    // Only process casts to index type
    if (!castOp.getType().isIndex())
      return failure();

    Value input = castOp.getIn();
    Operation *defOp = input.getDefiningOp();
    if (!defOp)
      return failure();

    Location loc = castOp.getLoc();

    // Handle binary arithmetic operations
    if (auto addOp = dyn_cast<arith::AddIOp>(defOp)) {
      Value lhs = castToIndex(rewriter, loc, addOp.getLhs());
      Value rhs = castToIndex(rewriter, loc, addOp.getRhs());
      Value newAdd = rewriter.create<arith::AddIOp>(loc, lhs, rhs);
      rewriter.replaceOp(castOp, newAdd);
      return success();
    }

    if (auto mulOp = dyn_cast<arith::MulIOp>(defOp)) {
      Value lhs = castToIndex(rewriter, loc, mulOp.getLhs());
      Value rhs = castToIndex(rewriter, loc, mulOp.getRhs());
      Value newMul = rewriter.create<arith::MulIOp>(loc, lhs, rhs);
      rewriter.replaceOp(castOp, newMul);
      return success();
    }

    if (auto subOp = dyn_cast<arith::SubIOp>(defOp)) {
      Value lhs = castToIndex(rewriter, loc, subOp.getLhs());
      Value rhs = castToIndex(rewriter, loc, subOp.getRhs());
      Value newSub = rewriter.create<arith::SubIOp>(loc, lhs, rhs);
      rewriter.replaceOp(castOp, newSub);
      return success();
    }

    // Handle chained index_cast
    if (auto innerCast = dyn_cast<arith::IndexCastOp>(defOp)) {
      // index_cast(index_cast(x)) -> index_cast(x)
      Value newCast = castToIndex(rewriter, loc, innerCast.getIn());
      rewriter.replaceOp(castOp, newCast);
      return success();
    }

    return failure();
  }
};

// Pattern to convert integer constants used in index computations to index constants
struct ConvertConstantsToIndexPattern : public RewritePattern {
  ConvertConstantsToIndexPattern(MLIRContext *context)
      : RewritePattern(arith::ConstantOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto constOp = cast<arith::ConstantOp>(op);

    // Check if this constant is used in index computations
    bool usedInIndexComputation = false;
    for (Operation *user : constOp->getUsers()) {
      if (isa<arith::IndexCastOp>(user)) {
        usedInIndexComputation = true;
        break;
      }
      // Check if used in arithmetic operations that feed into index_cast
      if (isa<arith::AddIOp, arith::MulIOp, arith::SubIOp>(user)) {
        for (Operation *userUser : user->getUsers()) {
          if (isa<arith::IndexCastOp>(userUser)) {
            usedInIndexComputation = true;
            break;
          }
        }
      }
    }

    if (!usedInIndexComputation)
      return failure();

    // Convert integer constant to index constant
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
      if (!constOp.getType().isIndex()) {
        auto indexConst = rewriter.create<arith::ConstantIndexOp>(
            constOp.getLoc(), intAttr.getInt());

        // Replace uses that need index type
        for (OpOperand &use : llvm::make_early_inc_range(constOp->getUses())) {
          if (isa<arith::IndexCastOp>(use.getOwner())) {
            use.set(indexConst);
          }
        }

        return success();
      }
    }

    return failure();
  }
};

struct NormalizeMemrefIndicesPass
    : public NormalizeMemrefIndicesBase<NormalizeMemrefIndicesPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<NormalizeIndexCastPattern>(&getContext());
    patterns.add<ConvertConstantsToIndexPattern>(&getContext());

    GreedyRewriteConfig config;
    // Default maxIterations is 10, which should be sufficient

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createNormalizeMemrefIndicesPass() {
  return std::make_unique<NormalizeMemrefIndicesPass>();
}
} // namespace mlir
