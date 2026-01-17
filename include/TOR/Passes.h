#ifndef TOR_PASSES_H
#define TOR_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "TOR/TOR.h"
#include <limits>
#include <memory>

namespace mlir {
    std::unique_ptr<OperationPass<mlir::tor::DesignOp>> createTORSchedulePass();

    std::unique_ptr<OperationPass<mlir::tor::DesignOp>> createTORTimeGraphPass();

    std::unique_ptr<OperationPass<mlir::ModuleOp>> createTORSplitPass();

    std::unique_ptr<OperationPass<tor::DesignOp>> createMAxiBurstInferPass();

    std::unique_ptr<OperationPass<mlir::tor::DesignOp>> createSCFToTORPass();

    std::unique_ptr<OperationPass<mlir::tor::DesignOp>> createFuncMemrefExtractPass();

    std::unique_ptr<OperationPass<mlir::tor::DesignOp>> createSCFIterArgsPass();

    std::unique_ptr<OperationPass<mlir::tor::DesignOp>> createSCFDumpPass();

    std::unique_ptr<OperationPass<mlir::ModuleOp>> createConvertInputPass();

    std::unique_ptr<OperationPass<mlir::ModuleOp>> createConvertStreamPass();

    std::unique_ptr<OperationPass<mlir::tor::DesignOp>> createTORDumpPass();

    std::unique_ptr<OperationPass<mlir::func::FuncOp>> createArrayPartitionPass();

    std::unique_ptr<OperationPass<mlir::ModuleOp>> createNewArrayPartitionPass();

    std::unique_ptr<OperationPass<mlir::ModuleOp>> createRemoveRedundantAccessPass();
    
    std::unique_ptr<OperationPass<mlir::ModuleOp>> createKrnlLowerPass();

    std::unique_ptr<OperationPass<mlir::func::FuncOp>> createUnificationIndexCastPass();

    std::unique_ptr<OperationPass<mlir::func::FuncOp>> createFuseLoopPass();

    std::unique_ptr<Pass> createAffineForLoweringPass();

    std::unique_ptr<OperationPass<mlir::ModuleOp>> createHlsUnrollPass();

    std::unique_ptr<OperationPass<mlir::ModuleOp>> createIgnoreAliasingLICMPass();

    std::unique_ptr<OperationPass<mlir::ModuleOp>> createMem2IterArgsPass();

    std::unique_ptr<OperationPass<mlir::ModuleOp>> createNewArrayPartitionPass();

    std::unique_ptr<Pass> createNormalizeMemrefIndicesPass();

    std::unique_ptr<OperationPass<mlir::ModuleOp>> createArrayOptPass();

    std::unique_ptr<OperationPass<mlir::ModuleOp>> createCountCyclesPass();

    std::unique_ptr<OperationPass<mlir::ModuleOp>> createGenPragmaReportPass();

    std::unique_ptr<OperationPass<mlir::ModuleOp>> createAddPragmaPass();
    
    std::unique_ptr<OperationPass<mlir::tor::DesignOp>> createTORFusePass();

    std::unique_ptr<OperationPass<mlir::func::FuncOp>> createLoopFlattenPass();

    std::unique_ptr<OperationPass<mlir::func::FuncOp>> createExpressionBalancePass();

    std::unique_ptr<OperationPass<mlir::func::FuncOp>> createLoopMergePass();
    
    std::unique_ptr<OperationPass<mlir::func::FuncOp>> createLoopTripcountPass();

    std::unique_ptr<OperationPass<mlir::func::FuncOp>> createMemrefReusePass();
    
    std::unique_ptr<OperationPass<mlir::ModuleOp>> createMemrefGlobalToConstantPass();

    std::unique_ptr<OperationPass<mlir::ModuleOp>> createStructSplitPass();
    std::unique_ptr<OperationPass<mlir::ModuleOp>> createArrayUseOffsetPass();

    std::unique_ptr<OperationPass<mlir::ModuleOp>> createDemangleFuncNamePass();
    
    std::unique_ptr<OperationPass<mlir::ModuleOp>> createONNXLowerPass();

    std::unique_ptr<OperationPass<mlir::ModuleOp>> createConvertMathToCallPass();
    std::unique_ptr<OperationPass<mlir::ModuleOp>> createTopFuncArg2GlobalPass();

    std::unique_ptr<Pass> createExtractStridedMetadataPass();
    std::unique_ptr<Pass> createReinterpretCastPass();
    std::unique_ptr<Pass> createEmbedWeightsAndOptimizeLoadPass();
    std::unique_ptr<OperationPass<mlir::ModuleOp>> createTosaReshapeNoDataLayoutPass();
    std::unique_ptr<OperationPass<mlir::ModuleOp>> createRemoveFuncArgPass();
    std::unique_ptr<OperationPass<mlir::ModuleOp>> createMinMaxToCmpSelectPass();
    std::unique_ptr<OperationPass<mlir::ModuleOp>> createLinalgConvAttrPass();

    std::unique_ptr<Pass> createRaiseSCFToAffinePass();

#define GEN_PASS_REGISTRATION

#include "TOR/Passes.h.inc"

}
#endif // TOR_PASSES_H
