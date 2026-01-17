//===- CombExtractToArithTrunc.cpp - Convert comb.extract to arith.trunci -===//
//
// This pass converts comb.extract operations to arith.trunci operations.
// This is useful for lowering to standard dialects that don't have comb dialect.
//
//===----------------------------------------------------------------------===//

#include "APS/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {

#define GEN_PASS_DEF_COMBEXTRACTTOARITHTRUNC
#include "APS/Passes.h.inc"

namespace {

/// Pattern to convert comb.extract to arith.trunci
///
/// comb.extract extracts a range of bits from an integer.
/// When extracting from bit 0, this is equivalent to arith.trunci + optional shift.
///
/// Example 1 (simple truncation):
///   %result = comb.extract %input from 0 : (i32) -> i8
/// becomes:
///   %result = arith.trunci %input : i32 to i8
///
/// Example 2 (extraction from middle):
///   %result = comb.extract %input from 8 : (i32) -> i8
/// becomes:
///   %shifted = arith.shrui %input, 8 : i32
///   %result = arith.trunci %shifted : i32 to i8
///
struct CombExtractToArithTruncPattern
    : public OpRewritePattern<circt::comb::ExtractOp> {
  using OpRewritePattern<circt::comb::ExtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(circt::comb::ExtractOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto input = op.getInput();
    auto inputType = llvm::dyn_cast<IntegerType>(input.getType());
    auto resultType = llvm::dyn_cast<IntegerType>(op.getType());

    if (!inputType || !resultType)
      return failure();

    uint32_t lowBit = op.getLowBit();
    uint32_t inputWidth = inputType.getWidth();
    uint32_t resultWidth = resultType.getWidth();

    // Case 1: Extract from bit 0 - simple truncation
    if (lowBit == 0) {
      // comb.extract %input from 0 : (iN) -> iM
      // =>
      // arith.trunci %input : iN to iM
      rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, resultType, input);
      return success();
    }

    // Case 2: Extract from middle - shift then truncate
    // comb.extract %input from K : (iN) -> iM
    // =>
    // %shifted = arith.shrui %input, K : iN
    // %result = arith.trunci %shifted : iN to iM

    auto shiftAmount = rewriter.create<arith::ConstantIntOp>(
        loc, lowBit, inputType.getWidth());

    // Determine signedness - use unsigned shift for now
    // (comb.extract is logically unsigned)
    auto shifted = rewriter.create<arith::ShRUIOp>(loc, input, shiftAmount);

    rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, resultType, shifted);
    return success();
  }
};

struct CombExtractToArithTruncPass
    : public impl::CombExtractToArithTruncBase<CombExtractToArithTruncPass> {
  void runOnOperation() override {
    auto *context = &getContext();
    auto op = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<CombExtractToArithTruncPattern>(context);

    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createCombExtractToArithTruncPass() {
  return std::make_unique<CombExtractToArithTruncPass>();
}

} // namespace mlir
