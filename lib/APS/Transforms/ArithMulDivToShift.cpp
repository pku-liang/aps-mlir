//===- ArithMulDivToShift.cpp - Convert arith mul/div to shift ops --------===//
//
// This pass converts arith.muli and arith.divui/divsi operations to shift
// operations when the second operand is a constant power of 2.
//
// Transformations:
// - arith.muli %x, %pow2  => arith.shli %x, log2(%pow2)
// - arith.divui %x, %pow2 => arith.shrui %x, log2(%pow2)
// - arith.divsi %x, %pow2 => (signed shift with bias for negative values)
//
//===----------------------------------------------------------------------===//

#include "APS/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {

#define GEN_PASS_DEF_ARITHMULDIVTOSHIFT
#include "APS/Passes.h.inc"

namespace {

/// Check if a value is a constant power of 2 and return the log2 value.
/// Returns -1 if not a constant power of 2.
/// Handles both arith.constant for integers and index types.
static int64_t getLog2IfPowerOf2(Value value) {
  int64_t val = 0;

  // Try ConstantIntOp first (for integer types)
  if (auto constIntOp = value.getDefiningOp<arith::ConstantIntOp>()) {
    val = constIntOp.value();
  }
  // Try ConstantIndexOp (for index type)
  else if (auto constIndexOp = value.getDefiningOp<arith::ConstantIndexOp>()) {
    val = constIndexOp.value();
  }
  // Try general ConstantOp
  else if (auto constOp = value.getDefiningOp<arith::ConstantOp>()) {
    auto attr = llvm::dyn_cast<IntegerAttr>(constOp.getValue());
    if (!attr)
      return -1;
    val = attr.getInt();
  } else {
    return -1;
  }

  if (val <= 0)
    return -1;

  // Check if it's a power of 2: (val & (val - 1)) == 0
  if ((val & (val - 1)) != 0)
    return -1;

  // Calculate log2
  int64_t log2Val = 0;
  while ((1LL << log2Val) < val)
    log2Val++;

  return log2Val;
}

/// Create a constant with the given value matching the type of the original value.
static Value createConstant(PatternRewriter &rewriter, Location loc, Type type,
                            int64_t value) {
  if (type.isIndex()) {
    return rewriter.create<arith::ConstantIndexOp>(loc, value);
  }
  return rewriter.create<arith::ConstantIntOp>(loc, type, value);
}

/// Pattern to convert arith.muli with power of 2 to arith.shli
///
/// Example:
///   %result = arith.muli %x, %c8 : i32  (where %c8 = 8 = 2^3)
/// becomes:
///   %c3 = arith.constant 3 : i32
///   %result = arith.shli %x, %c3 : i32
///
struct MuliToShliPattern : public OpRewritePattern<arith::MulIOp> {
  using OpRewritePattern<arith::MulIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::MulIOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();

    // Check if RHS is a power of 2
    int64_t log2Val = getLog2IfPowerOf2(rhs);
    if (log2Val < 0) {
      // Try LHS (multiplication is commutative)
      log2Val = getLog2IfPowerOf2(lhs);
      if (log2Val < 0)
        return failure();
      std::swap(lhs, rhs);
    }

    // Replace: muli %x, 2^n => shli %x, n
    auto resultType = op.getType();
    auto shiftAmount = createConstant(rewriter, loc, resultType, log2Val);
    rewriter.replaceOpWithNewOp<arith::ShLIOp>(op, lhs, shiftAmount);
    return success();
  }
};

/// Pattern to convert arith.divui with power of 2 to arith.shrui
///
/// Example:
///   %result = arith.divui %x, %c8 : i32  (where %c8 = 8 = 2^3)
/// becomes:
///   %c3 = arith.constant 3 : i32
///   %result = arith.shrui %x, %c3 : i32
///
struct DivuiToShruiPattern : public OpRewritePattern<arith::DivUIOp> {
  using OpRewritePattern<arith::DivUIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::DivUIOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();

    // Check if RHS is a power of 2
    int64_t log2Val = getLog2IfPowerOf2(rhs);
    if (log2Val < 0)
      return failure();

    // Replace: divui %x, 2^n => shrui %x, n
    auto resultType = op.getType();
    auto shiftAmount = createConstant(rewriter, loc, resultType, log2Val);
    rewriter.replaceOpWithNewOp<arith::ShRUIOp>(op, lhs, shiftAmount);
    return success();
  }
};

/// Pattern to convert arith.divsi with power of 2 to arith.shrsi
///
/// Example:
///   %result = arith.divsi %x, %c8 : i32  (where %c8 = 8 = 2^3)
/// becomes:
///   %c3 = arith.constant 3 : i32
///   %result = arith.shrsi %x, %c3 : i32
///
struct DivsiToShrsiPattern : public OpRewritePattern<arith::DivSIOp> {
  using OpRewritePattern<arith::DivSIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::DivSIOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();

    // Check if RHS is a power of 2
    int64_t log2Val = getLog2IfPowerOf2(rhs);
    if (log2Val < 0)
      return failure();

    // Replace: divsi %x, 2^n => shrsi %x, n
    auto resultType = op.getType();
    auto shiftAmount = createConstant(rewriter, loc, resultType, log2Val);
    rewriter.replaceOpWithNewOp<arith::ShRSIOp>(op, lhs, shiftAmount);
    return success();
  }
};

struct ArithMulDivToShiftPass
    : public impl::ArithMulDivToShiftBase<ArithMulDivToShiftPass> {
  void runOnOperation() override {
    auto *context = &getContext();
    auto op = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<MuliToShliPattern, DivuiToShruiPattern, DivsiToShrsiPattern>(
        context);

    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createArithMulDivToShiftPass() {
  return std::make_unique<ArithMulDivToShiftPass>();
}

} // namespace mlir
