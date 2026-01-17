#include "APS/APSDialect.h"
#include "APS/APSOps.h"

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"

#include "APS/APSDialect.cpp.inc"
#define GET_OP_CLASSES
#include "APS/APS.cpp.inc"

using namespace mlir;
using namespace aps;

void APSDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "APS/APS.cpp.inc"
  >();
}

//===----------------------------------------------------------------------===//
// MemoryMapOp
//===----------------------------------------------------------------------===//

ParseResult MemoryMapOp::parse(OpAsmParser &parser, OperationState &result) {
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();

  return success();
}

void MemoryMapOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

//===----------------------------------------------------------------------===//
// GlobalLoadOp Canonicalization
//===----------------------------------------------------------------------===//

namespace {
/// Fold GlobalLoad that immediately follows a GlobalStore to the same global.
/// Pattern: aps.globalstore %v, @x followed by %y = aps.globalload @x => replace %y with %v
///
/// This pattern enables store->load fusion after loop unrolling, converting:
///   aps.globalstore %1, @count
///   %2 = aps.globalload @count
/// into:
///   aps.globalstore %1, @count
///   (use %1 directly instead of %2)
struct FoldGlobalLoadAfterStore : public OpRewritePattern<GlobalLoad> {
  using OpRewritePattern<GlobalLoad>::OpRewritePattern;

  LogicalResult matchAndRewrite(GlobalLoad loadOp,
                                 PatternRewriter &rewriter) const override {
    // Get the global symbol being loaded
    auto loadSymbol = loadOp.getGlobalName();

    // Walk backwards from the load to find the most recent store to the same global
    Operation *op = loadOp.getOperation();
    Block *block = op->getBlock();

    // Iterate backwards through the block from the load operation
    auto it = Block::iterator(op);
    if (it == block->begin())
      return failure();

    --it;
    while (true) {
      Operation *prevOp = &*it;

      // IMPORTANT: Stop at operations with regions (loops, conditionals, etc.)
      // We cannot fold across region boundaries because the control flow
      // means the store might not dominate the load
      if (prevOp->getNumRegions() > 0) {
        // Found a region-bearing operation (loop, conditional, etc.)
        // Cannot safely fold across this boundary
        return failure();
      }

      // Check if this is a GlobalStore to the same global
      if (auto storeOp = dyn_cast<GlobalStore>(prevOp)) {
        if (storeOp.getGlobalName() == loadSymbol) {
          // Found matching store! Replace load with the stored value
          rewriter.replaceOp(loadOp, storeOp.getValue());
          return success();
        }
      }

      // Check if this operation might interfere with the global
      // Only stop if it's another GlobalLoad/GlobalStore to the SAME global
      // (memory operations to other globals or memrefs don't interfere)
      if (auto otherLoadOp = dyn_cast<GlobalLoad>(prevOp)) {
        if (otherLoadOp.getGlobalName() == loadSymbol) {
          // Another load to same global - safe to continue (it doesn't modify the value)
        }
      } else if (auto otherStoreOp = dyn_cast<GlobalStore>(prevOp)) {
        if (otherStoreOp.getGlobalName() == loadSymbol) {
          // Another store to same global - already handled above, shouldn't reach here
          return failure();
        }
      }
      // Note: We don't check for general memory effects because:
      // 1. GlobalLoad/GlobalStore operations are specific to named globals
      // 2. memref.load/store operations operate on different memory locations
      // 3. Only GlobalLoad/GlobalStore to the same symbol can interfere

      // Move to previous operation
      if (it == block->begin())
        return failure();
      --it;
    }

    return failure();
  }
};

/// Remove GlobalStore that is immediately followed by another GlobalStore to the same global.
/// Pattern: aps.globalstore %v1, @x followed by aps.globalstore %v2, @x => remove first store
///
/// This pattern removes redundant stores after loop unrolling:
///   aps.globalstore %1, @count
///   aps.globalstore %2, @count
/// becomes:
///   aps.globalstore %2, @count  (first store removed since it's overwritten)
struct RemoveDeadGlobalStore : public OpRewritePattern<GlobalStore> {
  using OpRewritePattern<GlobalStore>::OpRewritePattern;

  LogicalResult matchAndRewrite(GlobalStore storeOp,
                                 PatternRewriter &rewriter) const override {
    auto storeSymbol = storeOp.getGlobalName();

    // Walk forward from this store to find if another store to same global follows
    Operation *op = storeOp.getOperation();
    Block *block = op->getBlock();

    auto it = Block::iterator(op);
    ++it; // Move past current store

    if (it == block->end())
      return failure();

    // Scan forward to find the next GlobalStore to the same global
    while (it != block->end()) {
      Operation *nextOp = &*it;

      // IMPORTANT: Stop at operations with regions (loops, conditionals, etc.)
      // We cannot remove stores if a region exists between this store and the next,
      // because the region might read the value before it's overwritten
      if (nextOp->getNumRegions() > 0) {
        // Found a region-bearing operation (loop, conditional, etc.)
        // Cannot safely remove this store
        return failure();
      }

      // Check if this is a GlobalStore to the same global
      if (auto nextStoreOp = dyn_cast<GlobalStore>(nextOp)) {
        if (nextStoreOp.getGlobalName() == storeSymbol) {
          // Found another store to same global! Remove the current (earlier) store
          rewriter.eraseOp(storeOp);
          return success();
        }
      }

      // Check if this is a GlobalLoad to the same global - can't remove store
      if (auto nextLoadOp = dyn_cast<GlobalLoad>(nextOp)) {
        if (nextLoadOp.getGlobalName() == storeSymbol) {
          // Load uses this store, can't remove it
          return failure();
        }
      }

      // Note: memref operations don't interfere with named globals
      ++it;
    }

    return failure();
  }
};
} // namespace

void aps::GlobalLoad::getCanonicalizationPatterns(RewritePatternSet &results,
                                                    MLIRContext *context) {
  results.add<FoldGlobalLoadAfterStore>(context);
}

void aps::GlobalStore::getCanonicalizationPatterns(RewritePatternSet &results,
                                                     MLIRContext *context) {
  results.add<RemoveDeadGlobalStore>(context);
}

//===----------------------------------------------------------------------===//
// MemLoad Canonicalization
//===----------------------------------------------------------------------===//

namespace {
/// Helper function to check if two memory accesses are to the same location.
/// Returns true if both memref and all indices are identical.
static bool isSameMemoryLocation(Value memref1, ValueRange indices1,
                                  Value memref2, ValueRange indices2) {
  // First check if memrefs are the same
  if (memref1 != memref2)
    return false;

  // Check if index counts match
  if (indices1.size() != indices2.size())
    return false;

  // Check if all indices are identical
  for (size_t i = 0; i < indices1.size(); ++i) {
    if (indices1[i] != indices2[i])
      return false;
  }

  return true;
}

/// Fold MemLoad that immediately follows a MemStore to the same memory location.
/// Pattern: aps.memstore %v, %mem[%i] followed by %y = aps.memload %mem[%i]
///          => replace %y with %v
///
/// This pattern enables store->load fusion after loop unrolling, similar to
/// GlobalLoad/GlobalStore canonicalization but considers both memref and indices.
///
/// Example transformation:
///   aps.memstore %1, %mem[%c0] : i32, memref<16xi32>, index
///   %2 = aps.memload %mem[%c0] : memref<16xi32>, index -> i32
/// becomes:
///   aps.memstore %1, %mem[%c0] : i32, memref<16xi32>, index
///   (use %1 directly instead of %2)
struct FoldMemLoadAfterStore : public OpRewritePattern<MemLoad> {
  using OpRewritePattern<MemLoad>::OpRewritePattern;

  LogicalResult matchAndRewrite(MemLoad loadOp,
                                 PatternRewriter &rewriter) const override {
    Value loadMemref = loadOp.getMemref();
    ValueRange loadIndices = loadOp.getIndices();

    // Walk backwards from the load to find the most recent store to the same location
    Operation *op = loadOp.getOperation();
    Block *block = op->getBlock();

    auto it = Block::iterator(op);
    if (it == block->begin())
      return failure();

    --it;
    while (true) {
      Operation *prevOp = &*it;

      // IMPORTANT: Stop at operations with regions (loops, conditionals, etc.)
      // We cannot fold across region boundaries because control flow
      // means the store might not dominate the load
      if (prevOp->getNumRegions() > 0) {
        return failure();
      }

      // Check if this is a MemStore to the same memory location
      if (auto storeOp = dyn_cast<MemStore>(prevOp)) {
        Value storeMemref = storeOp.getMemref();
        ValueRange storeIndices = storeOp.getIndices();

        if (isSameMemoryLocation(loadMemref, loadIndices,
                                  storeMemref, storeIndices)) {
          // Found matching store! Replace load with the stored value
          rewriter.replaceOp(loadOp, storeOp.getValue());
          return success();
        }

        // Different location - check for potential aliasing
        // Conservative: if same memref but different/unknown indices, stop
        if (storeMemref == loadMemref) {
          // Same memref but different indices - might alias, cannot optimize
          return failure();
        }
      }

      // Check if this is another MemLoad to the same location
      if (auto otherLoadOp = dyn_cast<MemLoad>(prevOp)) {
        Value otherMemref = otherLoadOp.getMemref();
        ValueRange otherIndices = otherLoadOp.getIndices();

        if (isSameMemoryLocation(loadMemref, loadIndices,
                                  otherMemref, otherIndices)) {
          // Another load to same location - safe to continue (doesn't modify)
          // Continue searching for earlier store
        }
      }

      // Check for operations that might have memory side effects
      if (prevOp->hasTrait<OpTrait::HasRecursiveMemoryEffects>() ||
          !prevOp->getRegions().empty()) {
        // Operation might have memory effects, stop search
        return failure();
      }

      // Move to previous operation
      if (it == block->begin())
        return failure();
      --it;
    }

    return failure();
  }
};

/// Remove MemStore that is immediately followed by another MemStore to the same location.
/// Pattern: aps.memstore %v1, %mem[%i] followed by aps.memstore %v2, %mem[%i]
///          => remove first store
///
/// This pattern removes redundant stores after loop unrolling.
///
/// Example transformation:
///   aps.memstore %1, %mem[%c0] : i32, memref<16xi32>, index
///   aps.memstore %2, %mem[%c0] : i32, memref<16xi32>, index
/// becomes:
///   aps.memstore %2, %mem[%c0] : i32, memref<16xi32>, index
///   (first store removed since it's immediately overwritten)
struct RemoveDeadMemStore : public OpRewritePattern<MemStore> {
  using OpRewritePattern<MemStore>::OpRewritePattern;

  LogicalResult matchAndRewrite(MemStore storeOp,
                                 PatternRewriter &rewriter) const override {
    Value storeMemref = storeOp.getMemref();
    ValueRange storeIndices = storeOp.getIndices();

    // Walk forward from this store to find if another store to same location follows
    Operation *op = storeOp.getOperation();
    Block *block = op->getBlock();

    auto it = Block::iterator(op);
    ++it; // Move past current store

    if (it == block->end())
      return failure();

    // Scan forward to find the next MemStore to the same location
    while (it != block->end()) {
      Operation *nextOp = &*it;

      // IMPORTANT: Stop at operations with regions (loops, conditionals, etc.)
      // We cannot remove stores if a region exists between stores,
      // because the region might read the value before it's overwritten
      if (nextOp->getNumRegions() > 0) {
        return failure();
      }

      // Check if this is a MemStore to the same memory location
      if (auto nextStoreOp = dyn_cast<MemStore>(nextOp)) {
        Value nextMemref = nextStoreOp.getMemref();
        ValueRange nextIndices = nextStoreOp.getIndices();

        if (isSameMemoryLocation(storeMemref, storeIndices,
                                  nextMemref, nextIndices)) {
          // Found another store to same location! Remove current (earlier) store
          rewriter.eraseOp(storeOp);
          return success();
        }

        // Different location in same memref - might affect aliasing, stop
        if (nextMemref == storeMemref) {
          return failure();
        }
      }

      // Check if this is a MemLoad to the same location - can't remove store
      if (auto nextLoadOp = dyn_cast<MemLoad>(nextOp)) {
        Value nextMemref = nextLoadOp.getMemref();
        ValueRange nextIndices = nextLoadOp.getIndices();

        if (isSameMemoryLocation(storeMemref, storeIndices,
                                  nextMemref, nextIndices)) {
          // Load uses this store, can't remove it
          return failure();
        }

        // Different location in same memref - might be reading nearby, stop
        if (nextMemref == storeMemref) {
          return failure();
        }
      }

      // Check for operations with potential memory effects
      if (nextOp->hasTrait<OpTrait::HasRecursiveMemoryEffects>() ||
          !nextOp->getRegions().empty()) {
        return failure();
      }

      ++it;
    }

    return failure();
  }
};
} // namespace

void aps::MemLoad::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.add<FoldMemLoadAfterStore>(context);
}

void aps::MemStore::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.add<RemoveDeadMemStore>(context);
}

// Force template instantiation for TypeID
namespace mlir::detail {
template struct TypeIDResolver<aps::APSDialect, void>;
}