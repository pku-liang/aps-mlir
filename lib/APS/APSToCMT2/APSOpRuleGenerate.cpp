//===- RuleGeneration.cpp - Rule Generation for TOR Functions -------------===//
//
// This file implements the rule generation functionality for TOR functions
// that was previously in APSToCMT2GenPass.cpp
//
// REFACTORED: Now uses BBHandler for object-oriented basic block management
// and LoopHandler for FIFO-based loop coordination
//
//===----------------------------------------------------------------------===//

#include "APS/APSToCMT2.h"
#include "APS/BlockHandler.h"
#include "APS/BBHandler.h"
#include "APS/LoopHandler.h"

namespace mlir {

using namespace mlir;
using namespace mlir::tor;
using namespace circt::cmt2::ecmt2;
using namespace circt::cmt2::ecmt2::stl;
using namespace circt::firrtl;

//===----------------------------------------------------------------------===//
// Main Rule Generation Function - Refactored to use BBHandler and LoopHandler
//===----------------------------------------------------------------------===//

/// Generate rules for a specific TOR function - proper implementation from
/// rulegenpass.cpp
void APSToCMT2GenPass::generateRulesForFunction(
    Module *mainModule, tor::FuncOp funcOp, Instance *poolInstance,
    Instance *roccInstance, Instance *hellaMemInstance, InterfaceDecl *dmaItfc,
    Circuit &circuit, Clock mainClk, Reset mainRst, unsigned long opcode) {

  // First, check if function contains loops - if so, use LoopHandler
  bool hasLoops = false;
  funcOp.walk([&](tor::ForOp forOp) {
    hasLoops = true;
    return WalkResult::interrupt();
  });

  // Unified entry point: Always use BlockHandler, regardless of whether there are loops
  llvm::dbgs() << "[RuleGen] Function " << funcOp.getName()
               << " - using unified BlockHandler for all processing\n";

  // Create reg_rd register instance (one per opcode, shared across all blocks)
  auto &builder = mainModule->getBuilder();
  auto savedIP = builder.saveInsertionPoint();
  auto *regRdMod = STLLibrary::createRegModule(5, 0, circuit);
  Instance *regRdInstance = mainModule->addInstance("reg_rd_" + (std::ostringstream() << std::hex << std::setw(4) << std::setfill('0') << opcode).str(), regRdMod,
                                                    {mainClk.getValue(), mainRst.getValue()});
  builder.restoreInsertionPoint(savedIP);
  llvm::dbgs() << "[RuleGen] Created shared reg_rd instance: reg_rd_" << opcode << "\n";

  // For top-level function processing, we don't have external FIFOs
  // Token FIFOs and value FIFOs will be created internally by BlockHandler
  Instance *topLevelInputTokenFIFO = nullptr;
  Instance *topLevelOutputTokenFIFO = nullptr;
  llvm::DenseMap<Value, Instance*> topLevelInputFIFOs;  // Empty for top level
  llvm::DenseMap<Value, llvm::SmallVector<std::pair<BlockInfo*, Instance*>, 4>> topLevelOutputFIFOs; // Empty for top level

  // Create BlockHandler to manage all blocks with FIFO coordination
  // BlockHandler will internally delegate to specialized handlers (LoopHandler, BBHandler) as needed
  BlockHandler blockHandler(this, mainModule, funcOp, poolInstance,
                           roccInstance, hellaMemInstance, dmaItfc, circuit,
                           mainClk, mainRst, opcode, regRdInstance,
                           topLevelInputTokenFIFO, topLevelOutputTokenFIFO,
                           topLevelInputFIFOs, topLevelOutputFIFOs);

  // Process all blocks - BlockHandler will handle loops, basic blocks, conditionals, etc.
  if (failed(blockHandler.processFunctionAsBlocks())) {
    funcOp.emitError("failed to process blocks for rule generation");
    return;
  }

  llvm::dbgs() << "[BlockHandler] Successfully processed all blocks for function "
               << funcOp.getName() << "\n";
}

} // namespace mlir