#include "TOR/Passes.h"
#include "TOR/TOR.h"
#include "APS/APSOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/Support/Debug.h"

#include <set>

#define DEBUG_TYPE "aps-split-memory-ops"

namespace {
using namespace mlir;

/// Helper: Get ref_time pair from operation
std::pair<int, int> getRefTimePair(Operation *op) {
  auto startAttr = op->getAttrOfType<IntegerAttr>("ref_starttime");
  auto endAttr = op->getAttrOfType<IntegerAttr>("ref_endtime");

  if (!startAttr || !endAttr)
    return {-1, -1};

  return {startAttr.getInt(), endAttr.getInt()};
}

/// Helper: Set ref_time attributes
void setRefTimePair(Operation *op, int startTime, int endTime) {
  auto ctx = op->getContext();
  op->setAttr("ref_starttime",
              IntegerAttr::get(IntegerType::get(ctx, 32), startTime));
  op->setAttr("ref_endtime",
              IntegerAttr::get(IntegerType::get(ctx, 32), endTime));
}

/// Helper: Check if a value comes from _cpu_memory
bool isFromCpuMemory(Value memref) {
  // Trace back to see if this memref comes from a get_global with "_cpu_memory"
  auto defOp = memref.getDefiningOp();
  if (!defOp)
    return false;

  if (auto getGlobal = dyn_cast<memref::GetGlobalOp>(defOp)) {
    StringRef name = getGlobal.getName();
    return name.contains("_cpu_memory");
  }

  return false;
}

/// Split aps.memload into aps.itfc.load_req + aps.itfc.load_collect
struct SplitMemLoadPattern : public OpRewritePattern<aps::MemLoad> {
  using OpRewritePattern<aps::MemLoad>::OpRewritePattern;

  LogicalResult matchAndRewrite(aps::MemLoad loadOp,
                                PatternRewriter &rewriter) const override {
    llvm::dbgs() << "DEBUG: SplitMemLoadPattern::matchAndRewrite called\n";

    // Only process if has ref_* attributes
    if (!loadOp->hasAttr("ref_starttime") || !loadOp->hasAttr("ref_endtime")) {
      llvm::dbgs() << "DEBUG: aps.memload missing ref_* attributes, skipping\n";
      return failure();
    }

    auto refTime = getRefTimePair(loadOp.getOperation());
    int A = refTime.first;
    int B = refTime.second;

    llvm::dbgs() << "\n=== Splitting aps.memload ===\n";
    llvm::dbgs() << "  Original ref_time: (" << A << ", " << B << ")\n";

    // Create aps.itfc.load_req: ref_starttime=A, ref_endtime=B
    rewriter.setInsertionPoint(loadOp);
    auto reqOp = rewriter.create<aps::ItfcLoadReq>(
        loadOp.getLoc(),
        loadOp.getResult().getType(),
        loadOp.getMemref(),
        loadOp.getIndices());

    setRefTimePair(reqOp, A, B);
    llvm::dbgs() << "  Created aps.itfc.load_req: (" << A << ", " << B << ")\n";

    // Create aps.itfc.load_collect: ref_starttime=B, ref_endtime=B+1
    auto collectOp = rewriter.create<aps::ItfcLoadCollect>(
        loadOp.getLoc(),
        loadOp.getResult().getType(),
        reqOp.getResult());

    setRefTimePair(collectOp, B, B + 1);
    llvm::dbgs() << "  Created aps.itfc.load_collect: (" << B << ", " << (B + 1) << ")\n";

    // Replace uses and erase original
    rewriter.replaceOp(loadOp, collectOp.getResult());

    llvm::dbgs() << "=== Split complete ===\n\n";
    return success();
  }
};

/// Split aps.memstore into aps.itfc.store_req + aps.itfc.store_collect
struct SplitMemStorePattern : public OpRewritePattern<aps::MemStore> {
  using OpRewritePattern<aps::MemStore>::OpRewritePattern;

  LogicalResult matchAndRewrite(aps::MemStore storeOp,
                                PatternRewriter &rewriter) const override {
    llvm::dbgs() << "DEBUG: SplitMemStorePattern::matchAndRewrite called\n";

    // Only process if has ref_* attributes
    if (!storeOp->hasAttr("ref_starttime") || !storeOp->hasAttr("ref_endtime")) {
      llvm::dbgs() << "DEBUG: aps.memstore missing ref_* attributes, skipping\n";
      return failure();
    }

    auto refTime = getRefTimePair(storeOp.getOperation());
    int A = refTime.first;
    int B = refTime.second;

    llvm::dbgs() << "\n=== Splitting aps.memstore ===\n";
    llvm::dbgs() << "  Original ref_time: (" << A << ", " << B << ")\n";

    // Create aps.itfc.store_req: ref_starttime=A, ref_endtime=B
    rewriter.setInsertionPoint(storeOp);
    auto reqOp = rewriter.create<aps::ItfcStoreReq>(
        storeOp.getLoc(),
        rewriter.getNoneType(),
        storeOp.getValue(),
        storeOp.getMemref(),
        storeOp.getIndices());

    setRefTimePair(reqOp, A, B);
    llvm::dbgs() << "  Created aps.itfc.store_req: (" << A << ", " << B << ")\n";

    // Create aps.itfc.store_collect: ref_starttime=B, ref_endtime=B+1
    auto collectOp = rewriter.create<aps::ItfcStoreCollect>(
        storeOp.getLoc(),
        reqOp.getResult());

    setRefTimePair(collectOp, B, B + 1);
    llvm::dbgs() << "  Created aps.itfc.store_collect: (" << B << ", " << (B + 1) << ")\n";

    // Erase original store
    rewriter.eraseOp(storeOp);

    llvm::dbgs() << "=== Split complete ===\n\n";
    return success();
  }
};

struct APSSplitMemoryOpsPass
    : public PassWrapper<APSSplitMemoryOpsPass, OperationPass<tor::DesignOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(APSSplitMemoryOpsPass)

  StringRef getArgument() const final { return "aps-split-memory-ops"; }
  StringRef getDescription() const final {
    return "Split APS memory operations into request-collect pairs";
  }

  void runOnOperation() override {
    tor::DesignOp designOp = getOperation();

    // Process each function
    for (auto funcOp : designOp.getOps<tor::FuncOp>()) {
      // Only process scheduled functions
      if (!funcOp->hasAttr("scheduled")) {
        llvm::dbgs() << "Skipping unscheduled function: " << funcOp.getName() << "\n";
        continue;
      }

      llvm::dbgs() << "\n============================================\n";
      llvm::dbgs() << "Processing scheduled function: " << funcOp.getName() << "\n";
      llvm::dbgs() << "============================================\n";

      // Count operations before transformation
      int memloadCount = 0, memstoreCount = 0, burstloadCount = 0, burststoreCount = 0;
      funcOp.walk([&](Operation *op) {
        if (isa<aps::MemLoad>(op)) {
          memloadCount++;
          llvm::dbgs() << "Found aps.memload with ref_* attrs: "
                       << op->hasAttr("ref_starttime") << "\n";
        }
        if (isa<aps::MemStore>(op)) {
          memstoreCount++;
          llvm::dbgs() << "Found aps.memstore with ref_* attrs: "
                       << op->hasAttr("ref_starttime") << "\n";
        }
        if (isa<aps::MemBurstLoad>(op)) {
          burstloadCount++;
          llvm::dbgs() << "Found aps.memburstload with ref_* attrs: "
                       << op->hasAttr("ref_starttime") << "\n";
        }
        if (isa<aps::MemBurstStore>(op)) {
          burststoreCount++;
          llvm::dbgs() << "Found aps.memburststore with ref_* attrs: "
                       << op->hasAttr("ref_starttime") << "\n";
        }
      });
      llvm::dbgs() << "Total aps.memload ops: " << memloadCount << "\n";
      llvm::dbgs() << "Total aps.memstore ops: " << memstoreCount << "\n";
      llvm::dbgs() << "Total aps.memburstload ops: " << burstloadCount << "\n";
      llvm::dbgs() << "Total aps.memburststore ops: " << burststoreCount << "\n";

      // Manually transform each operation
      llvm::dbgs() << "Starting manual transformation...\n";
      IRRewriter rewriter(&getContext());

      // Collect operations to transform (can't modify while walking)
      SmallVector<aps::MemLoad> memloads;
      SmallVector<aps::MemLoad> spmloads;
      SmallVector<aps::MemStore> memstores;
      SmallVector<aps::MemBurstLoad> burstloads;
      SmallVector<aps::MemBurstStore> burststores;

      funcOp.walk([&](Operation *op) {
        if (auto loadOp = dyn_cast<aps::MemLoad>(op)) {
          if (loadOp->hasAttr("ref_starttime") && loadOp->hasAttr("ref_endtime")) {
            // Only transform if memref comes from _cpu_memory
            if (isFromCpuMemory(loadOp.getMemref())) {
              memloads.push_back(loadOp);
            } else {
              spmloads.push_back(loadOp);
            }
          }
        }
        if (auto storeOp = dyn_cast<aps::MemStore>(op)) {
          if (storeOp->hasAttr("ref_starttime") && storeOp->hasAttr("ref_endtime")) {
            // Only transform if memref comes from _cpu_memory
            if (isFromCpuMemory(storeOp.getMemref())) {
              memstores.push_back(storeOp);
            } else {
              llvm::dbgs() << "  Skipping aps.memstore (not from _cpu_memory)\n";
            }
          }
        }
        if (auto burstOp = dyn_cast<aps::MemBurstLoad>(op)) {
          if (burstOp->hasAttr("ref_starttime") && burstOp->hasAttr("ref_endtime")) {
            burstloads.push_back(burstOp);
          }
        }
        if (auto burstOp = dyn_cast<aps::MemBurstStore>(op)) {
          if (burstOp->hasAttr("ref_starttime") && burstOp->hasAttr("ref_endtime")) {
            burststores.push_back(burstOp);
          }
        }
      });

      // Transform memloads
      for (auto loadOp : memloads) {
        auto refTime = getRefTimePair(loadOp.getOperation());
        int A = refTime.first;
        int B = refTime.second;

        llvm::dbgs() << "\n=== Splitting aps.memload (cpu) ===\n";
        llvm::dbgs() << "  Original ref_time: (" << A << ", " << B << ")\n";

        rewriter.setInsertionPoint(loadOp);
        auto reqOp = rewriter.create<aps::ItfcLoadReq>(
            loadOp.getLoc(),
            loadOp.getResult().getType(),
            loadOp.getMemref(),
            loadOp.getIndices());

        // Copy all attributes from original operation first
        for (auto attr : loadOp->getAttrs()) {
          reqOp->setAttr(attr.getName(), attr.getValue());
        }
        // Then set the specific ref_time for request op
        setRefTimePair(reqOp, A, B);
        llvm::dbgs() << "  Created aps.itfc.load_req: (" << A << ", " << B << ")\n";

        auto collectOp = rewriter.create<aps::ItfcLoadCollect>(
            loadOp.getLoc(),
            loadOp.getResult().getType(),
            reqOp.getResult());

        // Copy all attributes from original operation first
        for (auto attr : loadOp->getAttrs()) {
          collectOp->setAttr(attr.getName(), attr.getValue());
        }
        // Then set the specific ref_time for collect op
        setRefTimePair(collectOp, B, B + 1);
        llvm::dbgs() << "  Created aps.itfc.load_collect: (" << B << ", " << (B + 1) << ")\n";

        rewriter.replaceOp(loadOp, collectOp.getResult());
        llvm::dbgs() << "=== Split complete ===\n\n";
      }

      // Transform spm memloads
      for (auto spmLoadOp : spmloads) {
        auto refTime = getRefTimePair(spmLoadOp.getOperation());
        int A = refTime.first;
        int B = refTime.second;

        llvm::dbgs() << "\n=== Splitting aps.memload (spm) ===\n";
        llvm::dbgs() << "  Original ref_time: (" << A << ", " << B << ")\n";

        rewriter.setInsertionPoint(spmLoadOp);
        auto reqOp = rewriter.create<aps::SpmLoadReq>(
            spmLoadOp.getLoc(),
            spmLoadOp.getResult().getType(),
            spmLoadOp.getMemref(),
            spmLoadOp.getIndices());

        // Copy all attributes from original operation first
        for (auto attr : spmLoadOp->getAttrs()) {
          reqOp->setAttr(attr.getName(), attr.getValue());
        }
        // Then set the specific ref_time for request op
        setRefTimePair(reqOp, A, B);
        llvm::dbgs() << "  Created aps.itfc.load_req: (" << A << ", " << B << ")\n";

        auto collectOp = rewriter.create<aps::SpmLoadCollect>(
            spmLoadOp.getLoc(),
            spmLoadOp.getResult().getType(),
            reqOp.getResult());

        // Copy all attributes from original operation first
        for (auto attr : spmLoadOp->getAttrs()) {
          collectOp->setAttr(attr.getName(), attr.getValue());
        }
        // Then set the specific ref_time for collect op
        setRefTimePair(collectOp, B, B + 1);
        llvm::dbgs() << "  Created aps.spm.load_collect: (" << B << ", " << (B + 1) << ")\n";

        rewriter.replaceOp(spmLoadOp, collectOp.getResult());
        llvm::dbgs() << "=== Split complete ===\n\n";
      }

      // Transform memstores
      for (auto storeOp : memstores) {
        auto refTime = getRefTimePair(storeOp.getOperation());
        int A = refTime.first;
        int B = refTime.second;

        llvm::dbgs() << "\n=== Splitting aps.memstore ===\n";
        llvm::dbgs() << "  Original ref_time: (" << A << ", " << B << ")\n";

        rewriter.setInsertionPoint(storeOp);
        auto reqOp = rewriter.create<aps::ItfcStoreReq>(
            storeOp.getLoc(),
            rewriter.getNoneType(),
            storeOp.getValue(),
            storeOp.getMemref(),
            storeOp.getIndices());

        // Copy all attributes from original operation first
        for (auto attr : storeOp->getAttrs()) {
          reqOp->setAttr(attr.getName(), attr.getValue());
        }
        // Then set the specific ref_time for request op
        setRefTimePair(reqOp, A, B);
        llvm::dbgs() << "  Created aps.itfc.store_req: (" << A << ", " << B << ")\n";

        auto collectOp = rewriter.create<aps::ItfcStoreCollect>(
            storeOp.getLoc(),
            reqOp.getResult());

        // Copy all attributes from original operation first
        for (auto attr : storeOp->getAttrs()) {
          collectOp->setAttr(attr.getName(), attr.getValue());
        }
        // Then set the specific ref_time for collect op
        setRefTimePair(collectOp, B, B + 1);
        llvm::dbgs() << "  Created aps.itfc.store_collect: (" << B << ", " << (B + 1) << ")\n";

        rewriter.eraseOp(storeOp);
        llvm::dbgs() << "=== Split complete ===\n\n";
      }

      // Transform burst loads
      for (auto burstOp : burstloads) {
        auto refTime = getRefTimePair(burstOp.getOperation());
        int A = refTime.first;
        int B = refTime.second;

        llvm::dbgs() << "\n=== Splitting aps.memburstload ===\n";
        llvm::dbgs() << "  Original ref_time: (" << A << ", " << B << ")\n";

        rewriter.setInsertionPoint(burstOp);
        auto reqOp = rewriter.create<aps::ItfcBurstLoadReq>(
            burstOp.getLoc(),
            rewriter.getNoneType(),
            burstOp.getCpuAddr(),
            burstOp.getMemrefs(),
            burstOp.getStart(),
            burstOp.getLength());

        // Copy all attributes from original operation first
        for (auto attr : burstOp->getAttrs()) {
          reqOp->setAttr(attr.getName(), attr.getValue());
        }
        // Then set the specific ref_time for request op
        setRefTimePair(reqOp, A, B);
        llvm::dbgs() << "  Created aps.itfc.burst_load_req: (" << A << ", " << B << ")\n";

        auto collectOp = rewriter.create<aps::ItfcBurstLoadCollect>(
            burstOp.getLoc(),
            reqOp.getResult());

        // Copy all attributes from original operation first
        for (auto attr : burstOp->getAttrs()) {
          collectOp->setAttr(attr.getName(), attr.getValue());
        }
        // Then set the specific ref_time for collect op
        setRefTimePair(collectOp, B, B + 1);
        llvm::dbgs() << "  Created aps.itfc.burst_load_collect: (" << B << ", " << (B + 1) << ")\n";

        rewriter.eraseOp(burstOp);
        llvm::dbgs() << "=== Split complete ===\n\n";
      }

      // Transform burst stores
      for (auto burstOp : burststores) {
        auto refTime = getRefTimePair(burstOp.getOperation());
        int A = refTime.first;
        int B = refTime.second;

        llvm::dbgs() << "\n=== Splitting aps.memburststore ===\n";
        llvm::dbgs() << "  Original ref_time: (" << A << ", " << B << ")\n";

        rewriter.setInsertionPoint(burstOp);
        auto reqOp = rewriter.create<aps::ItfcBurstStoreReq>(
            burstOp.getLoc(),
            rewriter.getNoneType(),
            burstOp.getMemrefs(),
            burstOp.getStart(),
            burstOp.getCpuAddr(),
            burstOp.getLength());

        // Copy all attributes from original operation first
        for (auto attr : burstOp->getAttrs()) {
          reqOp->setAttr(attr.getName(), attr.getValue());
        }
        // Then set the specific ref_time for request op
        setRefTimePair(reqOp, A, B);
        llvm::dbgs() << "  Created aps.itfc.burst_store_req: (" << A << ", " << B << ")\n";

        auto collectOp = rewriter.create<aps::ItfcBurstStoreCollect>(
            burstOp.getLoc(),
            reqOp.getResult());

        // Copy all attributes from original operation first
        for (auto attr : burstOp->getAttrs()) {
          collectOp->setAttr(attr.getName(), attr.getValue());
        }
        // Then set the specific ref_time for collect op
        setRefTimePair(collectOp, B, B + 1);
        llvm::dbgs() << "  Created aps.itfc.burst_store_collect: (" << B << ", " << (B + 1) << ")\n";

        rewriter.eraseOp(burstOp);
        llvm::dbgs() << "=== Split complete ===\n\n";
      }

      llvm::dbgs() << "Transformed " << memloads.size() << " loads, "
                   << memstores.size() << " stores, "
                   << burstloads.size() << " burst loads, "
                   << burststores.size() << " burst stores\n";

      llvm::dbgs() << "Completed function: " << funcOp.getName() << "\n\n";
    }
  }
};

} // namespace

namespace mlir {
std::unique_ptr<OperationPass<tor::DesignOp>> createAPSSplitMemoryOpsPass() {
  return std::make_unique<APSSplitMemoryOpsPass>();
}
} // namespace mlir
