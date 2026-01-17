//===- RegisterOpGenerator.cpp - Register Operation Generator --------------===//
//
// This file implements the register operation generator for TOR functions
//
//===----------------------------------------------------------------------===//

#include "APS/BBHandler.h"
#include "circt/Dialect/Cmt2/ECMT2/Signal.h"
#include "APS/APSOps.h"

namespace mlir {

using namespace mlir;
using namespace mlir::tor;
using namespace circt::cmt2::ecmt2;
using namespace circt::cmt2::ecmt2::stl;
using namespace circt::firrtl;

LogicalResult RegisterOpGenerator::generateRule(Operation *op, mlir::OpBuilder &b,
                                              Location loc, int64_t slot,
                                              llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  if (auto readRf = dyn_cast<aps::CpuRfRead>(op)) {
    return generateCpuRfRead(readRf, b, loc, slot, localMap);
  } else if (auto writeRf = dyn_cast<aps::CpuRfWrite>(op)) {
    return generateCpuRfWrite(writeRf, b, loc, slot, localMap);
  }

  return failure();
}

bool RegisterOpGenerator::canHandle(Operation *op) const {
  return isa<aps::CpuRfRead, aps::CpuRfWrite>(op);
}

LogicalResult RegisterOpGenerator::generateCpuRfRead(aps::CpuRfRead op, mlir::OpBuilder &b,
                                                   Location loc, int64_t slot,
                                                   llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  // PANIC if not in first time slot
  if (slot != 0) {
    op.emitError("aps.readrf operations must appear only in the first "
                 "time slot (slot 0), but found in slot ")
        << slot;
    llvm::report_fatal_error("readrf must be in first time slot");
  }

  // Get the function argument that this readrf is reading from
  Value regArg = op.getRs();

  // Map function arguments to rs1/rs2 based on their position in the function
  int64_t argIndex = -1;
  for (unsigned i = 0; i < bbHandler->getFuncOp().getNumArguments(); ++i) {
    if (bbHandler->getFuncOp().getArgument(i) == regArg) {
      argIndex = i;
      break;
    }
  }

  if (!cachedRoCCCmdBundle || !regRdInstance) {
    op.emitError("RoCC command bundle or register instance not available");
    return failure();
  }

  auto instruction = Bundle(cachedRoCCCmdBundle, &b, loc);
  Signal regValue = argIndex == 0 ? instruction["rs1data"] : instruction["rs2data"];
  localMap[op.getResult()] = regValue.getValue();
  return success();
}

LogicalResult RegisterOpGenerator::generateCpuRfWrite(aps::CpuRfWrite op, mlir::OpBuilder &b,
                                                    Location loc, int64_t slot,
                                                    llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  auto rdvalue = getValueInRule(op.getValue(), op.getOperation(), 1, b, localMap, loc);
  if (failed(rdvalue))
    return failure();

  if (!regRdInstance) {
    op.emitError("register instance not available");
    return failure();
  }

  auto rd = regRdInstance->callValue("read", b)[0];
  llvm::SmallVector<mlir::Value> bundleFields = {rd, *rdvalue};
  auto bundleType = BundleType::get(
      b.getContext(),
      {
          BundleType::BundleElement{b.getStringAttr("rd"), false,
                                    UIntType::get(b.getContext(), 5)},
          BundleType::BundleElement{b.getStringAttr("rddata"), false,
                                    UIntType::get(b.getContext(), 32)},
      });

  auto bundleValue = b.create<BundleCreateOp>(loc, bundleType, bundleFields);

  if (!bbHandler->getRoccInstance()) {
    op.emitError("RoCC instance not available");
    return failure();
  }

  bbHandler->getRoccInstance()->callMethod("resp_from_user", {bundleValue}, b);
  // writerf doesn't produce a result, just performs the write
  return success();
}

} // namespace mlir