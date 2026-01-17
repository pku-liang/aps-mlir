#include "APS/Passes.h"
#include "APS/APSOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "memory-map"

namespace {
using namespace mlir;
using namespace mlir::memref;

/// Structure to hold information about a partitioned memref group
struct MemrefGroup {
  std::string originalName;
  llvm::SmallVector<GlobalOp, 4> banks;
  uint32_t cyclicMode = 0;  // 1 = cyclic, 0 = block
  uint32_t numBanks = 1;

  MemrefGroup() = default;
  MemrefGroup(StringRef name) : originalName(name.str()) {}
};

/// Pass to generate memory map for global memrefs after array partition
struct MemoryMapPass : public PassWrapper<MemoryMapPass, OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MemoryMapPass)

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    OpBuilder builder(moduleOp.getContext());

    // Collect all global memrefs and group partitioned banks
    llvm::StringMap<MemrefGroup> memrefGroups;
    collectMemrefGroups(moduleOp, memrefGroups);

    // If no memrefs found, nothing to do
    if (memrefGroups.empty()) {
      return;
    }

    // Create memory map operation
    createMemoryMap(moduleOp, builder, memrefGroups);
  }

  StringRef getArgument() const final { return "memory-map"; }
  StringRef getDescription() const final {
    return "Generate memory map for global memrefs after array partition";
  }

private:
  /// Collect all global memrefs and group by original name
  void collectMemrefGroups(ModuleOp moduleOp,
                          llvm::StringMap<MemrefGroup> &groups) {
    moduleOp.walk([&](GlobalOp globalOp) {
      // Get original variable name from var_name attribute
      auto varNameAttr = globalOp->getAttrOfType<StringAttr>("var_name");
      if (!varNameAttr) {
        return;
      }

      std::string varName = varNameAttr.getValue().str();

      // Initialize group if not exists
      if (groups.find(varName) == groups.end()) {
        groups[varName] = MemrefGroup(varName);

        // Read partition metadata if present
        if (auto cyclicAttr = globalOp->getAttrOfType<ArrayAttr>("partition_cyclic_array")) {
          if (cyclicAttr.size() > 0) {
            auto cyclicVal = llvm::dyn_cast<IntegerAttr>(cyclicAttr[0]);
            if (cyclicVal) {
              groups[varName].cyclicMode = cyclicVal.getValue().getZExtValue();
            }
          }
        }

        if (auto factorAttr = globalOp->getAttrOfType<ArrayAttr>("partition_factor_array")) {
          if (factorAttr.size() > 0) {
            auto factorVal = llvm::dyn_cast<IntegerAttr>(factorAttr[0]);
            if (factorVal) {
              groups[varName].numBanks = factorVal.getValue().getZExtValue();
            }
          }
        }
      }

      // Add this global to the group
      groups[varName].banks.push_back(globalOp);
    });
  }

  /// Create the memory map operation
  void createMemoryMap(ModuleOp moduleOp, OpBuilder &builder,
                      llvm::StringMap<MemrefGroup> &groups) {
    // Insert memory map at the beginning of the module
    builder.setInsertionPointToStart(moduleOp.getBody());

    auto memoryMapOp = builder.create<aps::MemoryMapOp>(moduleOp.getLoc());
    Block *mapBody = builder.createBlock(&memoryMapOp.getRegion());
    builder.setInsertionPointToStart(mapBody);

    // Track current address
    uint32_t currentAddress = 0;

    // Process each memref group in a consistent order
    llvm::SmallVector<StringRef, 8> groupNames;
    for (auto &entry : groups) {
      groupNames.push_back(entry.getKey());
    }
    llvm::sort(groupNames);

    for (StringRef groupName : groupNames) {
      auto &group = groups[groupName];

      // Calculate bank size (use first bank's size)
      uint32_t bankSize = 0;
      if (!group.banks.empty()) {
        auto memrefType = llvm::cast<MemRefType>(group.banks[0].getType());
        uint64_t numElements = memrefType.getNumElements();
        // if (numElements == 1) {
        //   continue; // Skip single-element memrefs
        // }
        uint32_t elementSize = memrefType.getElementTypeBitWidth() / 8;
        bankSize = numElements * elementSize;
      }

      // Create array of bank symbol attributes
      llvm::SmallVector<Attribute, 4> bankSymbols;
      for (auto globalOp : group.banks) {
        bankSymbols.push_back(FlatSymbolRefAttr::get(globalOp.getSymNameAttr()));
      }

      // Get actual number of banks (may differ from factor due to partitioning)
      uint32_t actualNumBanks = group.banks.size();
      
      // Create mem_entry operation
      builder.create<aps::MemEntryOp>(
        moduleOp.getLoc(),
        builder.getStringAttr(group.originalName),
        builder.getArrayAttr(bankSymbols),
        builder.getUI32IntegerAttr(currentAddress),
        builder.getUI32IntegerAttr(bankSize),
        builder.getUI32IntegerAttr(actualNumBanks),
        builder.getUI32IntegerAttr(group.cyclicMode)
      );

      // Update address for next group
      currentAddress += bankSize * actualNumBanks;

      // Align to next power-of-2 boundary to ensure each array occupies 2^n bytes
      // This prevents burst accesses from crossing into other arrays
      uint32_t totalSize = bankSize * actualNumBanks;
      if (totalSize > 0) {
        // Find next power of 2 >= totalSize
        uint32_t alignedSize = 1;
        while (alignedSize < totalSize) {
          alignedSize <<= 1;
        }
        // Align currentAddress to this power-of-2 boundary
        currentAddress = ((currentAddress + alignedSize - 1) / alignedSize) * alignedSize;
      }

      LLVM_DEBUG(llvm::dbgs() << "Memory map entry: " << group.originalName
                             << " at 0x" << llvm::utohexstr(currentAddress - bankSize * actualNumBanks)
                             << " size=" << bankSize
                             << " banks=" << actualNumBanks
                             << " cyclic=" << group.cyclicMode << "\n");
    }

    // Create terminator
    builder.create<aps::MemFinishOp>(moduleOp.getLoc());
  }
};

} // namespace

namespace mlir {
std::unique_ptr<OperationPass<mlir::ModuleOp>> createMemoryMapPass() {
  return std::make_unique<MemoryMapPass>();
}
} // namespace mlir
