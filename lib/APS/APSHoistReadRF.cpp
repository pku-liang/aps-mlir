//===- APSHoistReadRF.cpp - Hoist aps.readrf to function entry -----------===//
//
// This pass moves all aps.readrf operations to the entry block of their
// containing function. This ensures they can be scheduled at cycle 0.
//
//===----------------------------------------------------------------------===//

#include "APS/APSDialect.h"
#include "APS/APSOps.h"
#include "APS/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "aps-hoist-readrf"

namespace mlir {

#define GEN_PASS_DEF_APSHOISTREADRF
#include "APS/Passes.h.inc"

namespace {

struct APSHoistReadRFPass
    : public impl::APSHoistReadRFBase<APSHoistReadRFPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    hoistReadRFInFunction(funcOp);
  }

private:
  void hoistReadRFInFunction(func::FuncOp funcOp) {
    // Get the entry block (first block in the function region)
    if (funcOp.getBody().empty())
      return;

    Block &entryBlock = funcOp.getBody().front();

    // Collect all readrf operations in the function
    llvm::SmallVector<aps::CpuRfRead, 4> readRfOps;
    funcOp.walk([&](aps::CpuRfRead readRfOp) {
      readRfOps.push_back(readRfOp);
    });

    if (readRfOps.empty())
      return;

    llvm::errs() << "Found " << readRfOps.size()
                 << " aps.readrf operations in function "
                 << funcOp.getName() << "\n";

    // Find the position of the first non-readrf operation in the entry block
    // We'll move all readrf ops before this point
    Operation *firstNonReadRf = nullptr;
    for (auto &op : entryBlock) {
      if (!llvm::isa<aps::CpuRfRead>(op)) {
        firstNonReadRf = &op;
        break;
      }
    }

    if (!firstNonReadRf) {
      // All operations are readrf, nothing to do
      return;
    }

    // Move each readrf operation to the beginning of the entry block
    OpBuilder builder(&getContext());
    for (auto readRfOp : readRfOps) {
      // Check if this readrf is already before firstNonReadRf
      bool needsMove = false;
      for (auto it = firstNonReadRf->getIterator(); it != entryBlock.end(); ++it) {
        if (&*it == readRfOp.getOperation()) {
          needsMove = true;
          break;
        }
      }

      if (needsMove) {
        llvm::errs() << "  Moving readrf before other operations\n";
        // Move the operation before firstNonReadRf
        readRfOp->moveBefore(firstNonReadRf);
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> createAPSHoistReadRFPass() {
  return std::make_unique<APSHoistReadRFPass>();
}

} // namespace mlir
