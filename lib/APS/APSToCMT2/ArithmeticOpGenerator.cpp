//===- ArithmeticOpGenerator.cpp - Arithmetic Operation Generator ----------===//
//
// This file implements the arithmetic operation generator for TOR functions
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "APS/BBHandler.h"
#include "circt/Dialect/Cmt2/ECMT2/Signal.h"
#include "TOR/TOR.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir {

using namespace mlir::tor;
using namespace circt::cmt2::ecmt2;
using namespace circt::cmt2::ecmt2::stl;
using namespace circt::firrtl;

LogicalResult ArithmeticOpGenerator::generateRule(Operation *op, mlir::OpBuilder &b,
                                                Location loc, int64_t slot,
                                                llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  if (auto addOp = dyn_cast<tor::AddIOp>(op)) {
    auto lhs = getValueInRule(addOp.getLhs(), op, 0, b, localMap, loc);
    auto rhs = getValueInRule(addOp.getRhs(), op, 1, b, localMap, loc);
    if (failed(lhs) || failed(rhs))
      return failure();
    return performArithmeticOp(b, loc, *lhs, *rhs, addOp.getResult(), "add", localMap);
  } else if (auto subOp = dyn_cast<tor::SubIOp>(op)) {
    auto lhs = getValueInRule(subOp.getLhs(), op, 0, b, localMap, loc);
    auto rhs = getValueInRule(subOp.getRhs(), op, 1, b, localMap, loc);
    if (failed(lhs) || failed(rhs))
      return failure();
    return performArithmeticOp(b, loc, *lhs, *rhs, subOp.getResult(), "sub", localMap);
  } else if (auto mulOp = dyn_cast<tor::MulIOp>(op)) {
    auto lhs = getValueInRule(mulOp.getLhs(), op, 0, b, localMap, loc);
    auto rhs = getValueInRule(mulOp.getRhs(), op, 1, b, localMap, loc);
    if (failed(lhs) || failed(rhs))
      return failure();
    return performArithmeticOp(b, loc, *lhs, *rhs, mulOp.getResult(), "mul", localMap);
  } else if (auto cmpOp = dyn_cast<tor::CmpIOp>(op)) {
    auto lhs = getValueInRule(cmpOp.getLhs(), op, 0, b, localMap, loc);
    auto rhs = getValueInRule(cmpOp.getRhs(), op, 1, b, localMap, loc);
    if (failed(lhs) || failed(rhs))
      return failure();
    return performComparisonOp(b, loc, *lhs, *rhs, cmpOp.getResult(), cmpOp.getPredicate(), localMap);
  } else if (auto selectOp = dyn_cast<arith::SelectOp>(op)) {
    auto condition = getValueInRule(selectOp.getCondition(), op, 0, b, localMap, loc);
    auto trueValue = getValueInRule(selectOp.getTrueValue(), op, 1, b, localMap, loc);
    auto falseValue = getValueInRule(selectOp.getFalseValue(), op, 2, b, localMap, loc);
    if (failed(condition) || failed(trueValue) || failed(falseValue))
      return failure();
    return performSelectOp(b, loc, *condition, *trueValue, *falseValue, selectOp.getResult(), localMap);
  } else if (auto extuiOp = dyn_cast<arith::ExtUIOp>(op)) {
    auto input = getValueInRule(extuiOp.getIn(), op, 0, b, localMap, loc);
    if (failed(input))
      return failure();
    return performExtUIOp(b, loc, *input, extuiOp.getResult(), localMap);
  } else if (auto trunciOp = dyn_cast<arith::TruncIOp>(op)) {
    auto input = getValueInRule(trunciOp.getIn(), op, 0, b, localMap, loc);
    if (failed(input))
      return failure();
    return performTruncIOp(b, loc, *input, trunciOp.getResult(), localMap);
  } else if (auto extractOp = dyn_cast<circt::comb::ExtractOp>(op)) {
    auto input = getValueInRule(extractOp.getInput(), op, 0, b, localMap, loc);
    if (failed(input))
      return failure();
    return performExtractOp(b, loc, *input, extractOp.getLowBit(), extractOp.getResult(), localMap);
  } else if (auto shliOp = dyn_cast<arith::ShLIOp>(op)) {
    auto lhs = getValueInRule(shliOp.getLhs(), op, 0, b, localMap, loc);
    auto rhs = getValueInRule(shliOp.getRhs(), op, 1, b, localMap, loc);
    if (failed(lhs) || failed(rhs))
      return failure();
    return performShiftOp(b, loc, *lhs, *rhs, shliOp.getResult(), "shl", localMap);
  } else if (auto shruiOp = dyn_cast<arith::ShRUIOp>(op)) {
    auto lhs = getValueInRule(shruiOp.getLhs(), op, 0, b, localMap, loc);
    auto rhs = getValueInRule(shruiOp.getRhs(), op, 1, b, localMap, loc);
    if (failed(lhs) || failed(rhs))
      return failure();
    return performShiftOp(b, loc, *lhs, *rhs, shruiOp.getResult(), "shrui", localMap);
  } else if (auto shrsiOp = dyn_cast<arith::ShRSIOp>(op)) {
    auto lhs = getValueInRule(shrsiOp.getLhs(), op, 0, b, localMap, loc);
    auto rhs = getValueInRule(shrsiOp.getRhs(), op, 1, b, localMap, loc);
    if (failed(lhs) || failed(rhs))
      return failure();
    return performShiftOp(b, loc, *lhs, *rhs, shrsiOp.getResult(), "shrsi", localMap);
  } else if (auto andiOp = dyn_cast<arith::AndIOp>(op)) {
    auto lhs = getValueInRule(andiOp.getLhs(), op, 0, b, localMap, loc);
    auto rhs = getValueInRule(andiOp.getRhs(), op, 1, b, localMap, loc);
    if (failed(lhs) || failed(rhs))
      return failure();
    return performBitwiseOp(b, loc, *lhs, *rhs, andiOp.getResult(), "and", localMap);
  } else if (auto oriOp = dyn_cast<arith::OrIOp>(op)) {
    auto lhs = getValueInRule(oriOp.getLhs(), op, 0, b, localMap, loc);
    auto rhs = getValueInRule(oriOp.getRhs(), op, 1, b, localMap, loc);
    if (failed(lhs) || failed(rhs))
      return failure();
    return performBitwiseOp(b, loc, *lhs, *rhs, oriOp.getResult(), "or", localMap);
  } else if (auto xoriOp = dyn_cast<arith::XOrIOp>(op)) {
    auto lhs = getValueInRule(xoriOp.getLhs(), op, 0, b, localMap, loc);
    auto rhs = getValueInRule(xoriOp.getRhs(), op, 1, b, localMap, loc);
    if (failed(lhs) || failed(rhs))
      return failure();
    return performBitwiseOp(b, loc, *lhs, *rhs, xoriOp.getResult(), "xor", localMap);
  } else if (auto extsiOp = dyn_cast<arith::ExtSIOp>(op)) {
    auto input = getValueInRule(extsiOp.getIn(), op, 0, b, localMap, loc);
    if (failed(input))
      return failure();
    return performExtSIOp(b, loc, *input, extsiOp.getResult(), localMap);
  }

  return failure();
}

bool ArithmeticOpGenerator::canHandle(Operation *op) const {
  return isa<tor::AddIOp, tor::SubIOp, tor::MulIOp, tor::CmpIOp, arith::SelectOp,
             arith::ExtUIOp, arith::ExtSIOp, arith::TruncIOp, circt::comb::ExtractOp,
             arith::ShLIOp, arith::ShRUIOp, arith::ShRSIOp,
             arith::AndIOp, arith::OrIOp, arith::XOrIOp>(op);
}

LogicalResult ArithmeticOpGenerator::performArithmeticOp(mlir::OpBuilder &b, Location loc,
                                                       mlir::Value lhs, mlir::Value rhs,
                                                       mlir::Value result, StringRef opName,
                                                       llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  // Determine result width based on operation type
  auto requiredWidth = cast<IntegerType>(result.getType()).getWidth();

  // Perform the arithmetic operation using Signal abstraction
  Signal lhsSignal(lhs, &b, loc);
  Signal rhsSignal(rhs, &b, loc);

  Signal resultSignal(lhs, &b, loc); // dummy init
  if (opName == "add") {
    resultSignal = lhsSignal + rhsSignal;
  } else if (opName == "sub") {
    resultSignal = lhsSignal - rhsSignal;
  } else if (opName == "mul") {
    resultSignal = lhsSignal * rhsSignal;
  } else {
    return failure();
  }

  auto firrtlWidth = resultSignal.getWidth();
  Signal resultSignalWidthFix = resultSignal;
  if (firrtlWidth > requiredWidth) {
    resultSignalWidthFix = resultSignal.bits(requiredWidth - 1, 0);
  } else if (firrtlWidth < requiredWidth) {
    resultSignalWidthFix = resultSignal.pad(requiredWidth);
  }

  localMap[result] = resultSignalWidthFix.getValue();
  return success();
}

LogicalResult ArithmeticOpGenerator::performComparisonOp(mlir::OpBuilder &b, Location loc,
                                                        mlir::Value lhs, mlir::Value rhs,
                                                        mlir::Value result,
                                                        mlir::tor::CmpIPredicate predicate,
                                                        llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  // Create Signal wrappers
  Signal lhsSignal(lhs, &b, loc);
  Signal rhsSignal(rhs, &b, loc);

  // Get operand widths from FIRRTL Signal
  auto lhsWidth = lhsSignal.getWidth();
  auto rhsWidth = rhsSignal.getWidth();

  // Match widths to the maximum (like arith.cmpi expects)
  auto maxWidth = std::max(lhsWidth, rhsWidth);
  if (lhsWidth < maxWidth) {
    lhsSignal = lhsSignal.pad(maxWidth);
  }
  if (rhsWidth < maxWidth) {
    rhsSignal = rhsSignal.pad(maxWidth);
  }

  Signal resultSignal(lhs, &b, loc); // dummy init

  // Map predicate to Signal comparison operators
  switch (predicate) {
    case mlir::tor::CmpIPredicate::eq:
      resultSignal = lhsSignal == rhsSignal;
      break;
    case mlir::tor::CmpIPredicate::ne:
      resultSignal = lhsSignal != rhsSignal;
      break;
    case mlir::tor::CmpIPredicate::slt:
    case mlir::tor::CmpIPredicate::ult:
      resultSignal = lhsSignal < rhsSignal;
      break;
    case mlir::tor::CmpIPredicate::sle:
    case mlir::tor::CmpIPredicate::ule:
      resultSignal = lhsSignal <= rhsSignal;
      break;
    case mlir::tor::CmpIPredicate::sgt:
    case mlir::tor::CmpIPredicate::ugt:
      resultSignal = lhsSignal > rhsSignal;
      break;
    case mlir::tor::CmpIPredicate::sge:
    case mlir::tor::CmpIPredicate::uge:
      resultSignal = lhsSignal >= rhsSignal;
      break;
    default:
      return failure();
  }

  // Get required result width from the TOR operation result type
  // The result type should be an integer type (typically i1 for comparisons)
  auto requiredWidth = dyn_cast<IntegerType>(result.getType()).getWidth();

  auto firrtlWidth = resultSignal.getWidth();

  Signal resultSignalWidthFix = resultSignal;
  if (firrtlWidth > requiredWidth) {
    resultSignalWidthFix = resultSignal.bits(requiredWidth - 1, 0);
  } else if (firrtlWidth < requiredWidth) {
    resultSignalWidthFix = resultSignal.pad(requiredWidth);
  }

  localMap[result] = resultSignalWidthFix.getValue();
  return success();
}

LogicalResult ArithmeticOpGenerator::performSelectOp(mlir::OpBuilder &b, Location loc,
                                                     mlir::Value condition, mlir::Value trueValue,
                                                     mlir::Value falseValue, mlir::Value result,
                                                     llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  // Create Signal wrappers
  Signal condSignal(condition, &b, loc);
  Signal trueSignal(trueValue, &b, loc);
  Signal falseSignal(falseValue, &b, loc);

  // Get operand widths from FIRRTL Signal
  auto trueWidth = trueSignal.getWidth();
  auto falseWidth = falseSignal.getWidth();

  // Match widths to the maximum (ensure operands have same width for mux)
  auto maxWidth = std::max(trueWidth, falseWidth);
  if (trueWidth < maxWidth) {
    trueSignal = trueSignal.pad(maxWidth);
  }
  if (falseWidth < maxWidth) {
    falseSignal = falseSignal.pad(maxWidth);
  }

  // Perform mux operation: if condition is true, select trueSignal, else select falseSignal
  // Signal::mux signature is: condition.mux(trueVal, falseVal)
  Signal resultSignal = condSignal.mux(trueSignal, falseSignal);

  // Get required result width from the operation result type
  auto requiredWidth = cast<IntegerType>(result.getType()).getWidth();

  auto firrtlWidth = resultSignal.getWidth();

  Signal resultSignalWidthFix = resultSignal;
  if (firrtlWidth > requiredWidth) {
    resultSignalWidthFix = resultSignal.bits(requiredWidth - 1, 0);
  } else if (firrtlWidth < requiredWidth) {
    resultSignalWidthFix = resultSignal.pad(requiredWidth);
  }

  localMap[result] = resultSignalWidthFix.getValue();
  return success();
}

LogicalResult ArithmeticOpGenerator::performExtUIOp(mlir::OpBuilder &b, Location loc,
                                                    mlir::Value input, mlir::Value result,
                                                    llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  // Create Signal wrapper for input
  Signal inputSignal(input, &b, loc);

  // Get result width for extension
  auto requiredWidth = cast<IntegerType>(result.getType()).getWidth();

  // Perform zero-extension (unsigned extension) using pad
  Signal resultSignal = inputSignal.pad(requiredWidth);

  localMap[result] = resultSignal.getValue();
  return success();
}

LogicalResult ArithmeticOpGenerator::performTruncIOp(mlir::OpBuilder &b, Location loc,
                                                     mlir::Value input, mlir::Value result,
                                                     llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  // Create Signal wrapper for input
  Signal inputSignal(input, &b, loc);

  // Get result width for truncation
  auto requiredWidth = cast<IntegerType>(result.getType()).getWidth();

  // Perform truncation by extracting lower bits
  Signal resultSignal = inputSignal.bits(requiredWidth - 1, 0);

  localMap[result] = resultSignal.getValue();
  return success();
}

LogicalResult ArithmeticOpGenerator::performExtractOp(mlir::OpBuilder &b, Location loc,
                                                      mlir::Value input, unsigned lowBit,
                                                      mlir::Value result,
                                                      llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  // Create Signal wrapper for input
  Signal inputSignal(input, &b, loc);

  // Get result width to determine high bit
  auto resultWidth = cast<IntegerType>(result.getType()).getWidth();
  auto highBit = lowBit + resultWidth - 1;

  // Extract bits using Signal::bits(high, low)
  Signal resultSignal = inputSignal.bits(highBit, lowBit);

  localMap[result] = resultSignal.getValue();
  return success();
}

/// Try to get a constant integer value from a FIRRTL value.
/// Returns std::nullopt if not a constant.
static std::optional<int64_t> getConstantValue(mlir::Value value) {
  // Check if it's a FIRRTL constant
  if (auto constOp = value.getDefiningOp<circt::firrtl::ConstantOp>()) {
    return constOp.getValue().getSExtValue();
  }
  return std::nullopt;
}

LogicalResult ArithmeticOpGenerator::performShiftOp(mlir::OpBuilder &b, Location loc,
                                                     mlir::Value lhs, mlir::Value rhs,
                                                     mlir::Value result, StringRef opName,
                                                     llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  // Create Signal wrappers
  Signal lhsSignal(lhs, &b, loc);

  // Get result width
  auto requiredWidth = cast<IntegerType>(result.getType()).getWidth();

  // Check if shift amount is a constant
  auto constShiftAmount = getConstantValue(rhs);

  // Create FIRRTL shift operations using the underlying values
  mlir::Value shiftResult;

  if (constShiftAmount.has_value()) {
    // Constant shift - use ShlPrimOp/ShrPrimOp
    int64_t shiftAmt = *constShiftAmount;

    if (opName == "shl") {
      // Constant left shift
      shiftResult = b.create<circt::firrtl::ShlPrimOp>(loc, lhs, shiftAmt);
    } else if (opName == "shrui" || opName == "shrsi") {
      // Constant right shift (both logical and arithmetic use ShrPrimOp)
      shiftResult = b.create<circt::firrtl::ShrPrimOp>(loc, lhs, shiftAmt);
    } else {
      return failure();
    }
  } else {
    // Dynamic shift - use DShlPrimOp/DShrPrimOp
    if (opName == "shl") {
      // Dynamic left shift: dshl
      shiftResult = b.create<circt::firrtl::DShlPrimOp>(loc, lhs, rhs);
    } else if (opName == "shrui") {
      // Dynamic logical right shift: dshr
      shiftResult = b.create<circt::firrtl::DShrPrimOp>(loc, lhs, rhs);
    } else if (opName == "shrsi") {
      // Dynamic arithmetic right shift: for signed, we need to ensure proper sign extension
      // FIRRTL's dshr on signed values performs arithmetic shift
      shiftResult = b.create<circt::firrtl::DShrPrimOp>(loc, lhs, rhs);
    } else {
      return failure();
    }
  }

  // Wrap result in Signal for width adjustment
  Signal resultSignal(shiftResult, &b, loc);
  auto firrtlWidth = resultSignal.getWidth();
  Signal resultSignalWidthFix = resultSignal;

  if (firrtlWidth > requiredWidth) {
    resultSignalWidthFix = resultSignal.bits(requiredWidth - 1, 0);
  } else if (firrtlWidth < requiredWidth) {
    resultSignalWidthFix = resultSignal.pad(requiredWidth);
  }

  localMap[result] = resultSignalWidthFix.getValue();
  return success();
}

LogicalResult ArithmeticOpGenerator::performBitwiseOp(mlir::OpBuilder &b, Location loc,
                                                       mlir::Value lhs, mlir::Value rhs,
                                                       mlir::Value result, StringRef opName,
                                                       llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  // Create Signal wrappers
  Signal lhsSignal(lhs, &b, loc);
  Signal rhsSignal(rhs, &b, loc);

  // Get operand widths
  auto lhsWidth = lhsSignal.getWidth();
  auto rhsWidth = rhsSignal.getWidth();

  // Match widths to the maximum
  auto maxWidth = std::max(lhsWidth, rhsWidth);
  if (lhsWidth < maxWidth) {
    lhsSignal = lhsSignal.pad(maxWidth);
  }
  if (rhsWidth < maxWidth) {
    rhsSignal = rhsSignal.pad(maxWidth);
  }

  // Get result width
  auto requiredWidth = cast<IntegerType>(result.getType()).getWidth();

  Signal resultSignal(lhs, &b, loc); // dummy init

  if (opName == "and") {
    // Bitwise AND
    resultSignal = lhsSignal & rhsSignal;
  } else if (opName == "or") {
    // Bitwise OR
    resultSignal = lhsSignal | rhsSignal;
  } else if (opName == "xor") {
    // Bitwise XOR
    resultSignal = lhsSignal ^ rhsSignal;
  } else {
    return failure();
  }

  // Adjust width if needed
  auto firrtlWidth = resultSignal.getWidth();
  Signal resultSignalWidthFix = resultSignal;

  if (firrtlWidth > requiredWidth) {
    resultSignalWidthFix = resultSignal.bits(requiredWidth - 1, 0);
  } else if (firrtlWidth < requiredWidth) {
    resultSignalWidthFix = resultSignal.pad(requiredWidth);
  }

  localMap[result] = resultSignalWidthFix.getValue();
  return success();
}

LogicalResult ArithmeticOpGenerator::performExtSIOp(mlir::OpBuilder &b, Location loc,
                                                     mlir::Value input, mlir::Value result,
                                                     llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  // Create Signal wrapper for input
  Signal inputSignal(input, &b, loc);

  // Get input and result widths
  auto inputWidth = inputSignal.getWidth();
  auto requiredWidth = cast<IntegerType>(result.getType()).getWidth();

  // Perform sign-extension
  // In FIRRTL, we need to treat the input as signed and extend it
  // We can achieve this by getting the sign bit and using it to fill the extended bits

  if (requiredWidth <= inputWidth) {
    // No extension needed, just truncate if necessary
    Signal resultSignal = inputSignal.bits(requiredWidth - 1, 0);
    localMap[result] = resultSignal.getValue();
    return success();
  }

  // Create FIRRTL operations for sign extension
  // For proper sign extension in FIRRTL: convert to SInt, pad, then convert back to UInt

  // Create signed version type
  auto signedType = circt::firrtl::SIntType::get(b.getContext(), inputWidth);

  // Cast to SInt, pad to required width, then cast back to UInt
  auto asSInt = b.create<circt::firrtl::AsSIntPrimOp>(loc, signedType, input);
  auto paddedType = circt::firrtl::SIntType::get(b.getContext(), requiredWidth);
  auto padded = b.create<circt::firrtl::PadPrimOp>(loc, paddedType, asSInt, requiredWidth);
  auto resultType = circt::firrtl::UIntType::get(b.getContext(), requiredWidth);
  auto asUInt = b.create<circt::firrtl::AsUIntPrimOp>(loc, resultType, padded);

  localMap[result] = asUInt.getResult();
  return success();
}

} // namespace mlir