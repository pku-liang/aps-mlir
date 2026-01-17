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

#define DEBUG_TYPE "memref-to-aps-mem"

namespace {

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::memref;

// Helper function to cast indices from index type to i32 type
SmallVector<Value> castIndicesToI32(OpBuilder &builder, Location loc,
                                     ValueRange indices) {
  SmallVector<Value> i32CastedIndices;
  auto i32Type = builder.getI32Type();

  for (Value idx : indices) {
    // memref operations always use index type, so we cast from index to i32
    if (idx.getType().isIndex()) {
      auto casted = builder.create<IndexCastOp>(loc, i32Type, idx);
      i32CastedIndices.push_back(casted);
    } else {
      // Should not happen for memref operations, but handle it anyway
      i32CastedIndices.push_back(idx);
    }
  }
  return i32CastedIndices;
}

// Pattern to convert memref.load to aps.memload
struct MemRefLoadToAPSMemLoadPattern : public OpRewritePattern<LoadOp> {
  using OpRewritePattern<LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    Location loc = loadOp.getLoc();

    // Get the indices from memref.load
    SmallVector<Value> indices(loadOp.getIndices().begin(),
                               loadOp.getIndices().end());

    // Cast indices from index to i32 type
    SmallVector<Value> i32CastedIndices =
        castIndicesToI32(rewriter, loc, indices);

    // Get the result type from the original load
    Type resultType = loadOp.getResult().getType();

    // Create aps.memload with i32-typed indices
    auto apsLoadOp = rewriter.create<aps::MemLoad>(
        loc, resultType, loadOp.getMemRef(), i32CastedIndices);

    // Replace the memref.load
    rewriter.replaceOp(loadOp, apsLoadOp.getResult());

    LLVM_DEBUG(llvm::dbgs() << "Converted memref.load to aps.memload\n");
    return success();
  }
};

// Pattern to convert memref.store to aps.memstore
struct MemRefStoreToAPSMemStorePattern : public OpRewritePattern<StoreOp> {
  using OpRewritePattern<StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    Location loc = storeOp.getLoc();

    // Get the indices from memref.store
    SmallVector<Value> indices(storeOp.getIndices().begin(),
                               storeOp.getIndices().end());

    // Cast indices from index to i32 type
    SmallVector<Value> i32CastedIndices =
        castIndicesToI32(rewriter, loc, indices);

    // Create aps.memstore with i32-typed indices
    rewriter.create<aps::MemStore>(loc, storeOp.getValue(),
                                   storeOp.getMemRef(), i32CastedIndices);

    // Erase the memref.store
    rewriter.eraseOp(storeOp);

    LLVM_DEBUG(llvm::dbgs() << "Converted memref.store to aps.memstore\n");
    return success();
  }
};

struct MemRefToAPSMemPass : MemRefMemToAPSMemBase<MemRefToAPSMemPass> {
  void runOnOperation() override {
    auto op = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<MemRefLoadToAPSMemLoadPattern, MemRefStoreToAPSMemStorePattern>(
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
std::unique_ptr<OperationPass<func::FuncOp>> createMemRefToAPSMemPass() {
  return std::make_unique<MemRefToAPSMemPass>();
}
} // namespace mlir
