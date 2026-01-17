#include "APS/Passes.h"
#include "APS/PassDetail.h"
#include "APS/APSOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "aps-scalar-mem-to-global"

namespace {

using namespace mlir;
using namespace mlir::memref;

/// Helper function to check if a memref type is a single-element memref (memref<1xT>)
static bool isSingleElementMemRef(MemRefType memrefType) {
  if (!memrefType)
    return false;

  auto shape = memrefType.getShape();
  // Check for memref<1xT> - rank 1 with size 1
  return shape.size() == 1 && shape[0] == 1;
}

/// Helper function to trace a memref value back to its get_global operation
/// Returns the global name if found, empty string otherwise
static std::string getGlobalNameForMemRef(Value memref) {
  // Trace back through the def-use chain
  Value current = memref;

  while (current) {
    // Check if this is a memref.get_global operation
    if (auto getGlobalOp = current.getDefiningOp<GetGlobalOp>()) {
      return getGlobalOp.getName().str();
    }

    // Check if this is a block argument (can't trace further)
    if (llvm::isa<BlockArgument>(current)) {
      return "";
    }

    // Check if the defining operation has a single result that we can trace
    Operation *defOp = current.getDefiningOp();
    if (!defOp)
      return "";

    // For now, only handle direct get_global
    // Could extend to handle subview, cast, etc. if needed
    return "";
  }

  return "";
}

/// Pattern to convert aps.memload on memref<1xT> to aps.globalload
struct ScalarMemLoadToGlobalLoadPattern : public OpRewritePattern<aps::MemLoad> {
  using OpRewritePattern<aps::MemLoad>::OpRewritePattern;

  LogicalResult matchAndRewrite(aps::MemLoad loadOp,
                                PatternRewriter &rewriter) const override {
    Value memref = loadOp.getMemref();
    auto memrefType = llvm::dyn_cast<MemRefType>(memref.getType());

    // Check if this is a single-element memref
    if (!isSingleElementMemRef(memrefType)) {
      return failure();
    }

    // Trace back to find the global name
    std::string globalName = getGlobalNameForMemRef(memref);
    if (globalName.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "Could not find global name for memref<1xT>\n");
      return failure();
    }

    Location loc = loadOp.getLoc();
    Type resultType = loadOp.getResult().getType();

    // Create symbol reference attribute
    auto symbolRef = FlatSymbolRefAttr::get(rewriter.getContext(), globalName);

    // Create aps.globalload
    auto globalLoadOp = rewriter.create<aps::GlobalLoad>(loc, resultType, symbolRef);

    // Replace the memload
    rewriter.replaceOp(loadOp, globalLoadOp.getResult());

    LLVM_DEBUG(llvm::dbgs() << "Converted aps.memload on memref<1x"
               << resultType << "> to aps.globalload @" << globalName << "\n");
    return success();
  }
};

/// Pattern to convert aps.memstore on memref<1xT> to aps.globalstore
struct ScalarMemStoreToGlobalStorePattern : public OpRewritePattern<aps::MemStore> {
  using OpRewritePattern<aps::MemStore>::OpRewritePattern;

  LogicalResult matchAndRewrite(aps::MemStore storeOp,
                                PatternRewriter &rewriter) const override {
    Value memref = storeOp.getMemref();
    auto memrefType = llvm::dyn_cast<MemRefType>(memref.getType());

    // Check if this is a single-element memref
    if (!isSingleElementMemRef(memrefType)) {
      return failure();
    }

    // Trace back to find the global name
    std::string globalName = getGlobalNameForMemRef(memref);
    if (globalName.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "Could not find global name for memref<1xT>\n");
      return failure();
    }

    Location loc = storeOp.getLoc();
    Value value = storeOp.getValue();

    // Create symbol reference attribute
    auto symbolRef = FlatSymbolRefAttr::get(rewriter.getContext(), globalName);

    // Create aps.globalstore
    rewriter.create<aps::GlobalStore>(loc, value, symbolRef);

    // Erase the memstore
    rewriter.eraseOp(storeOp);

    LLVM_DEBUG(llvm::dbgs() << "Converted aps.memstore on memref<1x"
               << value.getType() << "> to aps.globalstore @" << globalName << "\n");
    return success();
  }
};

struct APSScalarMemToGlobalPass : APSScalarMemToGlobalBase<APSScalarMemToGlobalPass> {
  void runOnOperation() override {
    auto op = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<ScalarMemLoadToGlobalLoadPattern, ScalarMemStoreToGlobalStorePattern>(
        &getContext());
    GreedyRewriteConfig config;
    config.setStrictness(GreedyRewriteStrictness::ExistingOps);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
std::unique_ptr<OperationPass<func::FuncOp>> createAPSScalarMemToGlobalPass() {
  return std::make_unique<APSScalarMemToGlobalPass>();
}
} // namespace mlir
