#ifndef APS_TRANSFORMS_APSTOSTANDARD_H
#define APS_TRANSFORMS_APSTOSTANDARD_H

#include "APS/APSOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

namespace mlir {
class DialectRegistry;
class ModuleOp;
class Type;

namespace func {
class FuncOp;
} // namespace func
} // namespace mlir

namespace mlir::aps {

// Helper structure to store global memref information
struct GlobalMemrefInfo {
  mlir::MemRefType type;
  mlir::Location loc;

  GlobalMemrefInfo(mlir::MemRefType t, mlir::Location l) : type(t), loc(l) {}
};

class APSToStandardPass
    : public mlir::PassWrapper<APSToStandardPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(APSToStandardPass)

  llvm::StringRef getArgument() const final;
  llvm::StringRef getDescription() const final;
  void getDependentDialects(mlir::DialectRegistry &registry) const final;
  void runOnOperation() final;

private:

  mlir::LogicalResult
  simplifyFunction(mlir::func::FuncOp func,
                   llvm::SmallVector<::aps::CpuRfRead> &rfReads,
                   llvm::SmallVector<::aps::CpuRfWrite> &rfWrites,
                   llvm::SmallVector<::aps::MemBurstLoad> &burstLoads,
                   llvm::SmallVector<::aps::MemBurstStore> &burstStores,
				   std::optional<unsigned> cpuMemArgIndex,
                   const llvm::DenseMap<llvm::StringRef, GlobalMemrefInfo> &requiredGlobals,
                   llvm::DenseMap<llvm::StringRef, unsigned> &globalArgIndices);
  mlir::LogicalResult lowerFunction(mlir::func::FuncOp func,
                                    std::optional<unsigned> cpuMemArgIndex,
                                    const llvm::DenseMap<llvm::StringRef, GlobalMemrefInfo> &requiredGlobals,
                                    llvm::DenseMap<llvm::StringRef, unsigned> &globalArgIndices);

  mlir::LogicalResult updateCalls(
      mlir::func::FuncOp func,
      const llvm::DenseMap<mlir::func::FuncOp, std::optional<mlir::Type>>
          &cpuMemTypes,
      const llvm::DenseMap<mlir::func::FuncOp, unsigned> &cpuMemArgIndices,
      llvm::DenseMap<llvm::StringRef, mlir::func::FuncOp> &funcByName);

  mlir::LogicalResult updateCallsWithGlobals(
      mlir::func::FuncOp func,
      const llvm::DenseMap<mlir::func::FuncOp, llvm::DenseMap<llvm::StringRef, unsigned>> &funcGlobalArgIndices,
      llvm::DenseMap<llvm::StringRef, mlir::func::FuncOp> &funcByName);
};

/// Create a pass that lowers APS dialect operations to standard MLIR dialects.
std::unique_ptr<mlir::Pass> createAPSToStandardPass();

} // namespace mlir::aps

#endif // APS_TRANSFORMS_APSTOSTANDARD_H
