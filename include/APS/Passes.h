#ifndef APS_PASSES_H
#define APS_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include <memory>
#include "TOR/TOR.h"

namespace mlir {

std::unique_ptr<OperationPass<mlir::func::FuncOp>> createSCFForIndexCastPass();
std::unique_ptr<OperationPass<mlir::func::FuncOp>> createAPSMemToMemRefPass();
std::unique_ptr<OperationPass<mlir::func::FuncOp>> createAffineMemToAPSMemPass();
std::unique_ptr<OperationPass<mlir::func::FuncOp>> createMemRefToAPSMemPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createMemoryMapPass();
std::unique_ptr<Pass> createInferAffineMemAccessPass();
std::unique_ptr<OperationPass<mlir::func::FuncOp>> createInferAffineMemAccessNewPass();
std::unique_ptr<OperationPass<mlir::tor::FuncOp>> createAPSMemLoadDuplicationPass();
std::unique_ptr<OperationPass<mlir::tor::DesignOp>> createAPSSplitMemoryOpsPass();
std::unique_ptr<Pass> createAPSHoistReadRFPass();
std::unique_ptr<OperationPass<mlir::func::FuncOp>> createAPSScalarMemToGlobalPass();
std::unique_ptr<Pass> createAPSToCMT2GenPass();
std::unique_ptr<Pass> createAPSToStandardPass();
std::unique_ptr<Pass> createCombExtractToArithTruncPass();
std::unique_ptr<Pass> createArithSelectToSCFIfPass();
std::unique_ptr<Pass> createArithMulDivToShiftPass();

#define GEN_PASS_REGISTRATION
#include "APS/Passes.h.inc"

} // namespace mlir

#endif // APS_PASSES_H
