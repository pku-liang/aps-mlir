//===- APSToCMT2GenPass.cpp - Generate CMT2 hardware from APS TOR --------===//
//
// This pass generates a hierarchical CMT2-based scratchpad memory pool from
// aps.memorymap with burst access support.
//
// Architecture:
//   ScratchpadMemoryPool (top-level)
//     ├─ burst_read(addr: u64) -> u64
//     ├─ burst_write(addr: u64, data: u64)
//     ├─ memory_mem_a (submodule per memory entry)
//     │    ├─ burst_read(addr: u64) -> u64
//     │    ├─ burst_write(addr: u64, data: u64)
//     │    ├─ mem_a_bank0 (Mem1r1w instance)
//     │    └─ mem_a_bank1, ...
//     └─ memory_mem_b (submodule)
//          └─ ...
//
// Features:
//   - Hierarchical module structure with proper instantiation
//   - Address decoding for routing burst accesses to correct memory entry
//   - Bank selection logic (cyclic vs block partitioning)
//   - Data bit extraction/placement for 64-bit burst bus
//   - Support for u8/u16/u24/u32/u40/u48/u56/u64 data widths
//   - Bank conflict validation (requires num_banks × data_width >= 64)
//   - cmt2.call operations for method invocation
//
//===----------------------------------------------------------------------===//

#include "APS/APSToCMT2.h"

#define DEBUG_TYPE "aps-memory-pool-gen"

namespace mlir {

using namespace mlir;
using namespace mlir::tor;
using namespace circt::cmt2::ecmt2;
using namespace circt::cmt2::ecmt2::stl;
using namespace circt::firrtl;

void APSToCMT2GenPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<aps::APSDialect>();
  registry.insert<circt::cmt2::Cmt2Dialect>();
  registry.insert<FIRRTLDialect>();
}

void APSToCMT2GenPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  llvm::dbgs() << "DEBUG: APSToCMT2GenPass::runOnOperation() started\n";

  // Find the aps.memorymap operation
  aps::MemoryMapOp memoryMapOp;
  moduleOp.walk([&](aps::MemoryMapOp op) {
    memoryMapOp = op;
    return WalkResult::interrupt();
  });

  // Initialize CMT2 module library
  auto &library = ModuleLibrary::getInstance();
  llvm::dbgs() << "DEBUG: Initializing CMT2 module library\n";

  // Try to find the module library manifest
  // First try: relative to build directory
  llvm::SmallString<256> manifestPath;
  llvm::sys::path::append(manifestPath,
                          "circt/lib/Dialect/Cmt2/ModuleLibrary/manifest.yaml");

  // Set library base path (parent of manifest)
  if (llvm::sys::fs::exists(manifestPath)) {
    llvm::dbgs() << "DEBUG: Found manifest at " << manifestPath << "\n";
    llvm::SmallString<256> libraryPath =
        llvm::sys::path::parent_path(manifestPath);
    library.setLibraryPath(libraryPath);

    // Load the manifest
    if (mlir::failed(library.loadManifest(manifestPath))) {
      llvm::dbgs() << "DEBUG: Failed to load module library manifest\n";
      moduleOp.emitWarning()
          << "Failed to load module library manifest from " << manifestPath;
      return;
    }
    llvm::dbgs() << "DEBUG: Successfully loaded module library manifest\n";
  } else {
    llvm::dbgs() << "DEBUG: Module library manifest not found at "
                 << manifestPath << "\n";
    moduleOp.emitWarning() << "Module library manifest not found. "
                           << "External FIRRTL modules may not work correctly.";
    return;
  }

  MLIRContext *context = moduleOp.getContext();
  Circuit circuit("MemoryPool", *context);

  addBurstMemoryInterface(circuit);
  addRoccAndHellaMemoryInterface(circuit);
  auto memoryPoolResult = generateMemoryPool(circuit, moduleOp, memoryMapOp);
  Module *poolModule = memoryPoolResult.poolModule;
  memEntryMap = std::move(memoryPoolResult.memEntryMap);

  auto glblRegister = generateGlobalRegisterList(circuit, moduleOp, memoryMapOp);

  // Generate RoCC adapter module
  llvm::SmallVector<unsigned long, 4> opcodes = {}; // Example opcodes
  moduleOp.walk([&](tor::FuncOp funcOp) {
    auto opcode = funcOp->getAttrOfType<IntegerAttr>("opcode").getInt();
    auto funct7 = funcOp->getAttrOfType<IntegerAttr>("funct7").getInt();
    opcodes.push_back((opcode << 8) + funct7);
  });
  auto *roccAdapterModule = generateRoCCAdapter(circuit, opcodes);

  // // Generate memory translator module
  auto *memoryAdapterModule = generateMemoryAdapter(circuit);

  // Generate rule-based main module for TOR functions
  auto instances = generateRuleBasedMainModule(
      moduleOp, circuit, poolModule, roccAdapterModule, memoryAdapterModule, glblRegister);
  auto *mainModule = instances.mainModule;
  auto *poolInstance = instances.poolInstance;
  auto *roccInstance = instances.roccInstance;
  auto *hellaMemInstance = instances.hellaMemInstance;

  // Add burst read/write methods to expose memory pool functionality
  addBurstMethodsToMainModule(mainModule, poolInstance);

  addRoCCAndMemoryMethodToMainModule(mainModule, roccInstance,
                                     hellaMemInstance);

  auto generatedModule = circuit.generateMLIR();
  if (!generatedModule) {
    moduleOp.emitError() << "failed to materialize MLIR";
    return;
  }

  // Preserve all original Ops
  auto &targetBlock = moduleOp.getBodyRegion().front();
  targetBlock.clear();
  auto &generatedOps = generatedModule->getBodyRegion().front().getOperations();

  for (auto &op : llvm::make_early_inc_range(generatedOps)) {
    if (op.hasTrait<mlir::OpTrait::IsTerminator>())
      continue;
    targetBlock.push_back(op.clone());
  }
}

// ============================================================================
//
//                            MAIN LOGIC GENERATION
//
// ============================================================================

/// Generate rule-based main module for TOR functions
MainModuleInstances APSToCMT2GenPass::generateRuleBasedMainModule(
    ModuleOp moduleOp, Circuit &circuit, Module *poolModule, Module *roccModule,
    Module *hellaMemModule, SmallVector<std::tuple<std::string, int8_t>, 8> glblRegister) {
  // Create main module in the same circuit
  auto *mainModule = circuit.addModule("main");

  // Find all TOR functions that need rule generation
  llvm::SmallVector<tor::FuncOp, 4> torFuncs;
  moduleOp.walk([&](tor::FuncOp funcOp) { torFuncs.push_back(funcOp); });

  // Add clock and reset arguments
  auto mainClk = mainModule->addClockArgument("clk");
  auto mainRst = mainModule->addResetArgument("rst");
  auto *burstControllerItfcDecl =
      mainModule->defineInterfaceDecl("dma", "BurstDMAController");
  mainModule->defineInterfaceDecl("rocc_resp", "roccRespItfc");
  mainModule->defineInterfaceDecl("hella_cmd", "hellaCmdItfc");

  auto &builder = mainModule->getBuilder();
  auto savedIP = builder.saveInsertionPoint();
  for (auto &[regName, regWidth] : glblRegister) {
    auto *regMod = STLLibrary::createRegModule(regWidth, 0, circuit);
    mainModule->addInstance("glbl_reg_" + regName, regMod,
                            {mainClk.getValue(), mainRst.getValue()});
  }
  builder.restoreInsertionPoint(savedIP);

  // Add scratchpad pool instance - use the pool module we created earlier
  auto *poolInstance = mainModule->addInstance(
      "scratchpad_pool", poolModule, {mainClk.getValue(), mainRst.getValue()});
  auto *roccInstance = mainModule->addInstance(
      "rocc_adapter", roccModule, {mainClk.getValue(), mainRst.getValue()},
      {{"rocc_resp", "rocc_resp"}});
  auto *hellaMemInstance = mainModule->addInstance(
      "hellacache_adapter", hellaMemModule,
      {mainClk.getValue(), mainRst.getValue()}, {{"hella_cmd", "hella_cmd"}});

  // For each TOR function, generate rules
  // Extract opcodes from the function or use defaults
  // For scalar.mlir, we need to determine the appropriate opcode
  for (auto funcOp : torFuncs) {
    auto opcode = funcOp->getAttrOfType<IntegerAttr>("opcode").getInt();
    auto funct7 = funcOp->getAttrOfType<IntegerAttr>("funct7").getInt();
    generateRulesForFunction(mainModule, funcOp, poolInstance, roccInstance,
                             hellaMemInstance, burstControllerItfcDecl, circuit,
                             mainClk, mainRst, (opcode << 8) + funct7);
                             // we now use both opcode and funct7 to identify functions
  }

  return {mainModule, poolInstance, roccInstance, hellaMemInstance};
}

} // namespace mlir

std::unique_ptr<mlir::Pass> mlir::createAPSToCMT2GenPass() {
  return std::make_unique<mlir::APSToCMT2GenPass>();
}
