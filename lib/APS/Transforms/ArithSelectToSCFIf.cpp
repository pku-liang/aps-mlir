//===- ArithSelectToSCFIf.cpp - Convert arith.select to scf.if -----------===//
//
// This pass converts arith.select operations to scf.if operations.
// This is useful when the backend prefers explicit control flow over
// conditional selection, or for pattern matching with control-flow heavy code.
//
//===----------------------------------------------------------------------===//

#include "APS/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {

#define GEN_PASS_DEF_ARITHSELECTTOSCFIF
#include "APS/Passes.h.inc"

namespace {

/// Pattern to convert arith.select to scf.if
///
/// arith.select chooses between two values based on a condition.
/// scf.if provides explicit control flow with then/else branches.
///
/// Example transformation:
///   %result = arith.select %cond, %true_val, %false_val : i32
/// becomes:
///   %result = scf.if %cond -> i32 {
///     scf.yield %true_val : i32
///   } else {
///     scf.yield %false_val : i32
///   }
///
struct ArithSelectToSCFIfPattern : public OpRewritePattern<arith::SelectOp> {
  using OpRewritePattern<arith::SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SelectOp selectOp,
                                PatternRewriter &rewriter) const override {
    auto loc = selectOp.getLoc();
    auto condition = selectOp.getCondition();
    auto trueValue = selectOp.getTrueValue();
    auto falseValue = selectOp.getFalseValue();
    auto resultType = selectOp.getType();

    // Create scf.if with result type
    auto ifOp = rewriter.create<scf::IfOp>(
        loc, resultType, condition,
        /*withElseRegion=*/true);

    // Build then region: yield true value
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      rewriter.create<scf::YieldOp>(loc, trueValue);
    }

    // Build else region: yield false value
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
      rewriter.create<scf::YieldOp>(loc, falseValue);
    }

    // Replace the select operation with the if operation
    rewriter.replaceOp(selectOp, ifOp.getResults());
    return success();
  }
};

struct ArithSelectToSCFIfPass
    : public impl::ArithSelectToSCFIfBase<ArithSelectToSCFIfPass> {
  void runOnOperation() override {
    auto *context = &getContext();
    auto op = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<ArithSelectToSCFIfPattern>(context);

    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createArithSelectToSCFIfPass() {
  return std::make_unique<ArithSelectToSCFIfPass>();
}

} // namespace mlir
