#include "APS/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "scf-for-index-cast"

#include "APS/Passes.h"
#include "APS/PassDetail.h"

namespace {

using namespace mlir;
using namespace mlir::arith;

// Helper function to cast a value to index type
Value castToIndex(OpBuilder &builder, Location loc, Value val) {
  if (val.getType().isIndex()) {
    return val;
  }
  return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), val);
}

// Pattern to convert scf.for loop bounds to index type
struct SCFForIndexCastPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    // Check if the loop already uses index type
    if (forOp.getLowerBound().getType().isIndex() &&
        forOp.getUpperBound().getType().isIndex() &&
        forOp.getStep().getType().isIndex()) {
      return failure(); // Already using index type
    }

    Location loc = forOp.getLoc();

    // Cast loop bounds to index type
    Value lowerBound = castToIndex(rewriter, loc, forOp.getLowerBound());
    Value upperBound = castToIndex(rewriter, loc, forOp.getUpperBound());
    Value step = castToIndex(rewriter, loc, forOp.getStep());

    // Get the old induction variable
    Value oldIV = forOp.getInductionVar();
    Type oldIVType = oldIV.getType();

    // Create new scf.for with index-typed bounds
    auto newForOp = rewriter.create<scf::ForOp>(
        loc, lowerBound, upperBound, step, forOp.getInitArgs(),
        [&](OpBuilder &builder, Location loc, Value newIV, ValueRange iterArgs) {
          // If the old IV was not index type, cast the new IV back
          Value ivToUse = newIV;
          if (!oldIVType.isIndex()) {
            ivToUse = builder.create<arith::IndexCastOp>(loc, oldIVType, newIV);
          }

          // Clone the body operations
          IRMapping mapping;
          mapping.map(oldIV, ivToUse);
          for (auto [oldArg, newArg] : llvm::zip(forOp.getRegionIterArgs(), iterArgs)) {
            mapping.map(oldArg, newArg);
          }

          for (auto &op : forOp.getBody()->without_terminator()) {
            builder.clone(op, mapping);
          }

          // Handle the yield operation
          if (auto yieldOp = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator())) {
            SmallVector<Value> yieldOperands;
            for (Value operand : yieldOp.getOperands()) {
              yieldOperands.push_back(mapping.lookup(operand));
            }
            builder.create<scf::YieldOp>(loc, yieldOperands);
          } else {
            builder.create<scf::YieldOp>(loc, ValueRange{});
          }
        });

    // Copy attributes
    newForOp->setAttrs(forOp->getAttrs());

    // Replace the old loop
    rewriter.replaceOp(forOp, newForOp.getResults());

    LLVM_DEBUG(llvm::dbgs() << "Converted scf.for to use index type\n");
    return success();
  }
};

struct SCFForIndexCastPass : SCFForIndexCastBase<SCFForIndexCastPass> {
  void runOnOperation() override {
    auto op = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<SCFForIndexCastPattern>(&getContext());
    GreedyRewriteConfig config;
    // Use AnyOp to process nested loops created during rewriting
    config.setStrictness(GreedyRewriteStrictness::AnyOp);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
  std::unique_ptr<OperationPass<func::FuncOp>> createSCFForIndexCastPass() {
    return std::make_unique<SCFForIndexCastPass>();
  }
}
