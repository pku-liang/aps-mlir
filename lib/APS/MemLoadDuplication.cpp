#include "APS/PassDetail.h"
#include "APS/Passes.h"
#include "APS/APSDialect.h"
#include "APS/APSOps.h"
#include "TOR/TORDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "aps-memload-duplication"

using namespace mlir;
using namespace aps;

namespace {

// Helper to get scheduling attributes
static std::optional<int64_t> getStarttime(Operation *op) {
  if (auto attr = op->getAttrOfType<IntegerAttr>("starttime"))
    return attr.getInt();
  return std::nullopt;
}

// Check if there's a memory conflict at a given time
// Returns true if there's ANY read/write to the same memref at that cycle
// (regardless of index - a[1] or a[2] doesn't matter, any access counts)
static bool hasMemoryConflict(Value memref, int64_t cycle, Block *block,
                               Operation *excludeOp) {
  for (auto &op : *block) {
    if (&op == excludeOp)
      continue;

    auto starttime = getStarttime(&op);
    if (!starttime)
      continue;

    // Check if op starts at this cycle (memory operation happens at starttime)
    if (*starttime != cycle)
      continue;

    // Check if it accesses the same memref (any index)
    if (auto memload = dyn_cast<aps::MemLoad>(&op)) {
      if (memload.getMemref() == memref)
        return true;
    } else if (auto memstore = dyn_cast<aps::MemStore>(&op)) {
      if (memstore.getMemref() == memref)
        return true;
    }
  }
  return false;
}

struct MemLoadDuplicationPass : public APSMemLoadDuplicationBase<MemLoadDuplicationPass> {
  void runOnOperation() override {
    auto func = getOperation();

    LLVM_DEBUG(llvm::dbgs() << "\n=== MemLoad Duplication Pass ===\n");

    // Collect all memload ops and their users
    SmallVector<aps::MemLoad> memloads;
    func.walk([&](aps::MemLoad memload) {
      memloads.push_back(memload);
    });

    int duplicateCount = 0;
    int fifosSaved = 0;

    for (auto memload : memloads) {
      auto memloadStarttime = getStarttime(memload);
      if (!memloadStarttime) {
        LLVM_DEBUG(llvm::dbgs() << "Skipping memload without starttime\n");
        continue;
      }

      Value memref = memload.getMemref();
      auto indices = memload.getIndices();
      Block *block = memload->getBlock();

      // Analyze all users of this memload
      SmallVector<std::pair<Operation*, int64_t>> usersToFix;

      for (OpOperand &use : memload->getUses()) {
        Operation *user = use.getOwner();
        auto userStarttime = getStarttime(user);

        if (!userStarttime) {
          LLVM_DEBUG(llvm::dbgs() << "User has no starttime, skipping\n");
          continue;
        }

        // Check if user starts at a different time than memload
        if (*userStarttime != *memloadStarttime) {
          LLVM_DEBUG(llvm::dbgs() << "Found user at different time: memload@"
                                  << *memloadStarttime << " user@" << *userStarttime << "\n");

          // Check if there's a memory conflict at the user's start time
          if (!hasMemoryConflict(memref, *userStarttime, block, memload.getOperation())) {
            usersToFix.push_back({user, *userStarttime});
            LLVM_DEBUG(llvm::dbgs() << "  -> Can duplicate! No conflict at cycle "
                                    << *userStarttime << "\n");
          } else {
            LLVM_DEBUG(llvm::dbgs() << "  -> Cannot duplicate, memory conflict at cycle "
                                    << *userStarttime << "\n");
          }
        }
      }

      // For each unique time where we can duplicate, create one duplicate memload
      llvm::DenseMap<int64_t, aps::MemLoad> timeToClone;

      for (auto [user, time] : usersToFix) {
        if (timeToClone.count(time))
          continue;

        // Create a duplicate memload at this time
        // IMPORTANT: Insert BEFORE the user operation
        OpBuilder builder(user);
        auto newMemload = builder.create<aps::MemLoad>(
            memload.getLoc(),
            memload.getResult().getType(),
            memref,
            indices);

        // Copy scheduling attributes
        newMemload->setAttr("starttime", builder.getI32IntegerAttr(time));
        newMemload->setAttr("endtime", builder.getI32IntegerAttr(time + 1));

        // Copy other attributes if present
        if (auto slot = memload->getAttr("slot"))
          newMemload->setAttr("slot", slot);
        if (auto dump = memload->getAttr("dump"))
          newMemload->setAttr("dump", dump);

        timeToClone[time] = newMemload;
        duplicateCount++;

        LLVM_DEBUG(llvm::dbgs() << "Created duplicate memload at time " << time << "\n");
      }

      // Update users to use the duplicated memloads
      for (auto [user, time] : usersToFix) {
        if (auto newMemload = timeToClone.lookup(time)) {
          // Find which operand uses the memload and replace it
          for (OpOperand &operand : user->getOpOperands()) {
            if (operand.get() == memload.getResult()) {
              operand.set(newMemload.getResult());
              fifosSaved++;
              LLVM_DEBUG(llvm::dbgs() << "Updated user to use duplicated memload (saved FIFO)\n");
            }
          }
        }
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "\n=== Summary ===\n");
    LLVM_DEBUG(llvm::dbgs() << "Duplicated memloads: " << duplicateCount << "\n");
    LLVM_DEBUG(llvm::dbgs() << "FIFOs saved: " << fifosSaved << "\n");

    if (duplicateCount > 0) {
      llvm::errs() << "[MemLoadDuplication] Saved " << fifosSaved
                   << " FIFOs by duplicating " << duplicateCount << " memory loads\n";
    }
  }
};

} // namespace

namespace mlir {
std::unique_ptr<OperationPass<tor::FuncOp>> createAPSMemLoadDuplicationPass() {
  return std::make_unique<MemLoadDuplicationPass>();
}
} // namespace mlir
