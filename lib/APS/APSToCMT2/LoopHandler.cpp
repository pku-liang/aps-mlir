//===- LoopHandler.cpp - Canonical Loop Handler with Block Coordination ---===//
//
// This file implements the canonical loop handler using Signal EDSL for
// clean hardware generation with proper token flow coordination
//
//===----------------------------------------------------------------------===//

#include "APS/LoopHandler.h"
#include "APS/APSOps.h"
#include "APS/BBHandler.h"
#include "APS/BlockHandler.h"
#include "circt/Dialect/Cmt2/ECMT2/SignalHelpers.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {

using namespace mlir;
using namespace mlir::tor;
using namespace circt::cmt2::ecmt2;
using namespace circt::cmt2::ecmt2::stl;
using namespace circt::firrtl;

//===----------------------------------------------------------------------===//
// Canonical LoopHandler Implementation with Signal EDSL
//===----------------------------------------------------------------------===//

LoopHandler::LoopHandler(APSToCMT2GenPass *pass, Module *mainModule,
                         tor::FuncOp funcOp, Instance *poolInstance,
                         Instance *roccInstance, Instance *hellaMemInstance,
                         InterfaceDecl *dmaItfc, Circuit &circuit,
                         Clock mainClk, Reset mainRst, unsigned long opcode, Instance *regRdInstance,
                         Instance *input_token_fifo,
                         Instance *output_token_fifo,
                         llvm::DenseMap<Value, Instance *> &input_fifos,
                         llvm::DenseMap<Value, llvm::SmallVector<std::pair<BlockInfo*, Instance*>, 4>> &output_fifos,
                         const std::string &namePrefix)
    : BlockHandler(pass, mainModule, funcOp, poolInstance, roccInstance,
                   hellaMemInstance, dmaItfc, circuit, mainClk, mainRst, opcode, regRdInstance,
                   input_token_fifo, output_token_fifo, input_fifos,
                   output_fifos, namePrefix) {}

LogicalResult LoopHandler::processLoopBlock(BlockInfo &loopBlock) {
  // Process a single loop block following Blockgen.md canonical pattern
  // entry → body → next with proper token coordination

  // 1. Extract the tor.for operation from this block segment
  tor::ForOp forOp = nullptr;
  for (Operation *op : loopBlock.operations) {
    if (auto candidate = dyn_cast<tor::ForOp>(op)) {
      forOp = candidate;
      break;
    }
  }

  if (!forOp) {
    llvm::dbgs() << "[LoopHandler] ERROR: Loop block segment contains " << loopBlock.operations.size() << " operations:\n";
    for (Operation *op : loopBlock.operations) {
      llvm::dbgs() << "  - " << op->getName() << "\n";
    }
    llvm::report_fatal_error("Loop block segment does not contain tor.for operation");
  }

  // 2. Initialize the single loop with proper hierarchical name
  // Use namePrefix (inherited from BlockHandler) instead of loopBlock.blockName to avoid duplication
  std::string loop_name = namePrefix + "loop";
  loop.initialize(forOp, loop_name);

  // 3. Extract loop control information
  loop.inductionVar = forOp.getInductionVar();
  loop.lowerBound = forOp.getLowerBound();
  loop.upperBound = forOp.getUpperBound();
  loop.step = forOp.getStep();

  // Extract iter_args if present
  for (Value iterArg : forOp.getRegionIterArgs()) {
    loop.iterArgs.push_back(iterArg);
    loop.iterArgTypes.push_back(iterArg.getType());
  }

  // 4. Create simplified loop infrastructure (FIFOs)
  if (failed(createLoopInfrastructure()))
    return failure();

  // 5. Process loop body operations using BBHandler with token coordination
  if (failed(processLoopBodyOperations(forOp, loopBlock)))
    return failure();

  // 6. Generate canonical loop rules (entry → body → next)
  if (failed(generateCanonicalLoopRules(loopBlock)))
    return failure();

  return success();
}

LogicalResult LoopHandler::processBlock(BlockInfo &block) {
  llvm::dbgs() << "[LoopHandler] Processing block " << block.blockId
               << " as loop block\n";

  // Delegate to processLoopBlock for loop-specific processing
  return processLoopBlock(block);
}

LogicalResult LoopHandler::generateCanonicalLoopRules(BlockInfo &loopBlock) {
  // Generate exactly 2 rules per Blockgen.md: entry rule and next rule
  llvm::dbgs() << "[LoopHandler] Generating canonical loop rules (entry + "
                  "next) for loop "
               << loop.loopName << "\n";

  // Create entry rule - handles loop initialization and first iteration
  if (failed(generateLoopEntryRule(loopBlock)))
    return failure();

  // Create next rule - handles loop iteration and termination
  if (failed(generateLoopNextRule(loopBlock)))
    return failure();

  return success();
}

LogicalResult LoopHandler::generateLoopEntryRule(BlockInfo &loopBlock) {
  // Per Blockgen.md: Create entry rule that handles loop initialization
  // Use loop name as distinguisher, not loop ID
  auto *rule = mainModule->addRule(loop.loopName + "_entry_rule");

  rule->guard([&](mlir::OpBuilder &b) {
    auto loc = b.getUnknownLoc();
    // Per Blockgen.md: use 1'b1 - coordination handled automatically by CMT
    auto alwaysTrue = UInt::constant(1, 1, b, loc);
    b.create<circt::cmt2::ReturnOp>(loc, alwaysTrue.getValue());
  });

  rule->body([&](mlir::OpBuilder &b) {
    auto loc = b.getUnknownLoc();

    llvm::dbgs() << "[LoopHandler] Generating entry rule for loop "
                 << loop.loopName << "\n";

    // 1. Dequeue token from previous block (token input fifo)
    if (inputTokenFIFO) {
      auto prevToken = inputTokenFIFO->callMethod("deq", {}, b);
      llvm::dbgs() << "[LoopHandler] Dequeued token from previous block\n";
    } else {
      llvm::dbgs() << "[LoopHandler] No input token FIFO (top-level loop)\n";
    }

    // 2. Handle cross-block value consumption from input fifos
    // IMPORTANT: Only dequeue values that are actually used by this loop
    // (either by the loop operations themselves or by the loop body via loop-to-body FIFOs)
    for (auto &[value, fifo] : input_fifos) {
      if (!fifo)
        continue;

      // Skip constants - they don't need to be dequeued
      if (value.getDefiningOp<arith::ConstantOp>()) {
        llvm::dbgs() << "[LoopHandler] Skipping constant value in entry rule\n";
        continue;
      }

      // Check if this value has a corresponding loop-to-body FIFO or state register
      // If not, it means this value is NOT used by the loop - skip dequeuing
      bool hasLoopToBodyFifo = loop.loop_to_body_fifos.count(value) > 0;
      bool hasStateRegister = loop.input_state_registers.count(value) > 0;

      if (!hasLoopToBodyFifo && !hasStateRegister) {
        llvm::dbgs() << "[LoopHandler] Skipping input FIFO - value not used by loop\n";
        continue;
      }

      // Dequeue only if value is actually used by the loop
      auto dequeuedValue = fifo->callMethod("deq", {}, b)[0];
      llvm::dbgs() << "[LoopHandler] Dequeued cross-block value from input FIFO\n";

      // Write to state register for persistent storage (used by next rule)
      if (hasStateRegister) {
        Instance *stateReg = loop.input_state_registers[value];
        stateReg->callMethod("write", {dequeuedValue}, b);
        llvm::dbgs() << "[LoopHandler] Wrote dequeued value to state register\n";
      }

      // Enqueue to loop-to-body FIFO for first iteration
      if (hasLoopToBodyFifo) {
        Instance *loopToBodyFifo = loop.loop_to_body_fifos[value];
        loopToBodyFifo->callMethod("enq", {dequeuedValue}, b);
        llvm::dbgs() << "[LoopHandler] Enqueued value to loop-to-body FIFO for first iteration\n";
      }
    }

    // 3. Initialize loop state in loop carry fifo
    // Pack state: [counter][bound][step][iter_args...]
    // Convert all values to FIRRTL types before creating Signals
    auto convertToFIRRTL = [&](mlir::Value val) -> mlir::Value {
      if (isa<circt::firrtl::FIRRTLBaseType>(val.getType())) {
        llvm::report_fatal_error("LoopHandler: loop boundary is not a constant!");
      }
      if (auto constOp = val.getDefiningOp<arith::ConstantOp>()) {
        auto intAttr = mlir::cast<IntegerAttr>(constOp.getValueAttr());
        unsigned width = mlir::cast<IntegerType>(intAttr.getType()).getWidth();
        return UInt::constant(intAttr.getValue().getZExtValue(), width, b, loc).getValue();
      }
      llvm::report_fatal_error("LoopHandler: Cannot convert non-constant MLIR type to FIRRTL");
    };

    // high: lowerBound (start), medium: upperBound (stop), low: step
    // TODO: we only handle increment bound here...
    Signal loopState(convertToFIRRTL(loop.lowerBound), &b, loc);
    loopState = loopState.cat(Signal(convertToFIRRTL(loop.upperBound), &b, loc));
    loopState = loopState.cat(Signal(convertToFIRRTL(loop.step), &b, loc));

    // TODO: ignore for now..
    // for (unsigned i = 0; i < loop.iterArgs.size(); i++) {
    //   loopState = loopState.cat(Signal(convertToFIRRTL(loop.iterArgs[i]), &b, loc));
    // }

    loop.loop_state_fifo->callMethod("enq", {loopState.getValue()}, b);
    llvm::dbgs() << "[LoopHandler] Initialized loop state FIFO\n";

    // 4. Extract and enqueue loop variables for the loop body
    // Extract induction variable from loop state and enqueue to its FIFO
    if (loop.inductionVar && inductionVarFIFO) {
      // Extract induction variable from loop state (first 32 bits)
      Signal stateSig(loopState.getValue(), &b, loc);
      auto inductionVar = stateSig.bits(95, 64);
      inductionVarFIFO->callMethod("enq", {inductionVar.getValue()}, b);
      llvm::dbgs() << "[LoopHandler] Enqueued induction variable to its FIFO\n";
    }

    // 6. Signal loop body to start using token coordination
    // Per Blockgen.md: enq token to body for loop body execution
    auto startToken = UInt::constant(1, 1, b, loc);
    if (loop.token_fifos.to_body) {
      loop.token_fifos.to_body->callMethod("enq", {startToken.getValue()}, b);
      llvm::dbgs()
          << "[LoopHandler] Signaled loop body to start via token FIFO\n";
    }

    b.create<circt::cmt2::ReturnOp>(loc);
  });

  rule->finalize();
  llvm::dbgs() << "[LoopHandler] Generated entry rule for loop "
               << loop.loopName << "\n";
  return success();
}

LogicalResult LoopHandler::generateLoopNextRule(BlockInfo &loopBlock) {
  auto *rule = mainModule->addRule(loop.loopName + "_next_rule");

  rule->guard([&](mlir::OpBuilder &b) {
    auto loc = b.getUnknownLoc();
    // Always return 1'b1 - coordination handled by FIFO availability
    auto alwaysReady = UInt::constant(1, 1, b, loc);
    b.create<circt::cmt2::ReturnOp>(loc, alwaysReady.getValue());
  });

  rule->body([&](mlir::OpBuilder &b) {
    auto loc = b.getUnknownLoc();

    // Canonical next rule: handle iteration logic and loop termination
    llvm::dbgs()
        << "[LoopHandler] Generating canonical next rule body for loop "
        << loop.loopName << "\n";

    // 1. Dequeue body completion token from body_to_next fifo
    if (loop.token_fifos.body_to_next) {
      auto bodyCompleteToken =
          loop.token_fifos.body_to_next->callMethod("deq", {}, b);
      llvm::dbgs() << "[LoopHandler] Dequeued body completion token\n";
    }

    // 2. Dequeue loop state from previous iteration
    if (loop.loop_state_fifo) {
      auto loopState = loop.loop_state_fifo->callMethod("deq", {}, b)[0];

      // Extract state components using Signal operations
      // [counter:32][bound:32][step:32][iter_arg0...]
      Signal stateSig(loopState, &b, loc);

      auto currentCounter = stateSig.bits(95, 64); // Extract counter (bits 31:0)
      auto upperBound = stateSig.bits(63, 32);    // Extract bound (bits 63:32)
      auto step = stateSig.bits(31, 0);          // Extract step (bits 95:64)

      // Extract iter_args
      llvm::SmallVector<mlir::Value> iterArgs;
      unsigned stateOffset = 96;
      for (unsigned i = 0; i < loop.iterArgs.size(); i++) {
        unsigned width = getBitWidth(loop.iterArgTypes[i]);
        unsigned highBit = stateOffset + width - 1;
        auto iterArg = stateSig.bits(highBit, stateOffset);
        iterArgs.push_back(iterArg.getValue());
        stateOffset += width;
      }

      // 4. Canonical loop decision:
      // If shouldContinue: increment counter and continue loop
      // If not shouldContinue: exit loop and pass control to next block
      auto nextCounter = currentCounter + step;

      // 3. Check if loop should continue: counter <= upper_bound
      auto shouldContinue = nextCounter <= upperBound;
      llvm::dbgs()
          << "[LoopHandler] Next rule: checking if counter < upper_bound\n";

      // 5. Update loop state and either continue or exit using ECMT2 If
      // construct for proper signal-based conditional execution
      If(
          shouldContinue,
          // Then branch: Continue looping
          [&](mlir::OpBuilder &b) {
            llvm::dbgs() << "[LoopHandler] Next rule: continuing loop, counter "
                            "updated\n";

            // Pack updated state: [nextCounter][upperBound][step][iterArgs...]
            Signal updatedState(nextCounter.getValue(), &b, loc);
            updatedState = updatedState.cat(upperBound);
            updatedState = updatedState.cat(step);

            // Add updated iter_args (for now, just pass through)
            for (auto iterArg : iterArgs) {
              updatedState = updatedState.cat(Signal(iterArg, &b, loc));
            }

            // Enqueue updated state for next iteration
            loop.loop_state_fifo->callMethod("enq", {updatedState.bits(stateOffset - 1, 0).getValue()},
                                             b);

            // Read from state registers and enqueue to loop-to-body FIFOs
            // This provides the cross-block values for the next iteration
            for (auto &[value, fifo] : input_fifos) {
              // Skip constants - they don't have state registers
              if (value.getDefiningOp<arith::ConstantOp>()) {
                continue;
              }

              if (fifo && loop.input_state_registers.count(value) && loop.loop_to_body_fifos.count(value)) {
                Instance *stateReg = loop.input_state_registers[value];
                auto storedValue = stateReg->callMethod("read", {}, b)[0];
                Instance *loopToBodyFifo = loop.loop_to_body_fifos[value];
                loopToBodyFifo->callMethod("enq", {storedValue}, b);
                llvm::dbgs() << "[LoopHandler] Next rule: read from state register and enqueued to loop-to-body FIFO\n";
              }
            }

            if (loop.inductionVar && inductionVarFIFO) {
              // Extract induction variable from loop state (first 32 bits)
              Signal stateSig(updatedState.getValue(), &b, loc);
              auto inductionVar = nextCounter.bits(31, 0);
              inductionVarFIFO->callMethod("enq", {inductionVar.getValue()}, b);
              llvm::dbgs() << "[LoopHandler] Enqueued induction variable to its FIFO\n";
            }

            // Signal next iteration via token FIFO coordination
            if (loop.token_fifos.to_body) {
              auto continueToken = UInt::constant(1, 1, b, loc);
              loop.token_fifos.to_body->callMethod(
                  "enq", {continueToken.getValue()}, b);
              llvm::dbgs() << "[LoopHandler] Next rule: signaling next "
                              "iteration via token FIFO\n";
            }
          },
          // Else branch: Loop complete
          [&](mlir::OpBuilder &b) {
            llvm::dbgs()
                << "[LoopHandler] Next rule: loop complete, signaling exit\n";

            // Signal loop completion directly to next block via output token FIFO
            // (no intermediate next_to_exit FIFO needed)
            if (outputTokenFIFO) {
              auto outputExitToken = UInt::constant(1, 1, b, loc);
              outputTokenFIFO->callMethod("enq", {outputExitToken.getValue()}, b);
              llvm::dbgs() << "[LoopHandler] Next rule: enqueued output token to next block\n";
            } else {
              llvm::dbgs() << "[LoopHandler] No output token FIFO (top-level loop exit)\n";
            }

            // Pass loop results to next block via output FIFOs
            // For now, just pass the iter_args as results
            for (unsigned i = 0; i < loop.iterArgs.size(); i++) {
              llvm::dbgs() << "[LoopHandler] Passing iter_arg " << i
                           << " to next block\n";
            }
          },
          b, loc);
    } 
    
    b.create<circt::cmt2::ReturnOp>(loc);

  });

  rule->finalize();

  llvm::dbgs() << "[LoopHandler] Generated canonical loop next rule for loop "
              << loop.loopName << "\n";
  return success();
}

LogicalResult LoopHandler::createLoopInfrastructure() {
  llvm::dbgs() << "[LoopHandler] Creating loop infrastructure for loop "
               << loop.loopName << "\n";

  auto &builder = mainModule->getBuilder();
  auto savedIP = builder.saveInsertionPoint();

  // Create token FIFOs for canonical loop coordination (entry → body → next)
  // Entry -> Body: signals that loop body can start
  auto *entryToBodyMod = STLLibrary::createFIFO2IModule(1, circuit);
  builder.restoreInsertionPoint(savedIP);
  std::string entryToBodyName = loop.loopName + "_token_entry_to_body";
  loop.token_fifos.to_body =
      mainModule->addInstance(entryToBodyName, entryToBodyMod,
                              {mainClk.getValue(), mainRst.getValue()});
  llvm::dbgs() << "[LoopHandler] Created entry-to-body token FIFO: "
               << entryToBodyName << "\n";

  // Body -> Next: signals that body execution is complete
  auto *bodyToNextMod = STLLibrary::createFIFO2IModule(1, circuit);
  builder.restoreInsertionPoint(savedIP);
  std::string bodyToNextName = loop.loopName + "_token_body_to_next";
  loop.token_fifos.body_to_next = mainModule->addInstance(
      bodyToNextName, bodyToNextMod, {mainClk.getValue(), mainRst.getValue()});
  llvm::dbgs() << "[LoopHandler] Created body-to-next token FIFO: "
               << bodyToNextName << "\n";

  // Note: next_to_exit is NOT created - we use the loop block's outputTokenFIFO instead
  // This connects the loop's exit directly to the next block
  loop.token_fifos.next_to_exit = nullptr;

  // Create single loop state FIFO for carrying iteration state
  // Calculate total bit width: counter(32) + bound(32) + step(32) + iter_args
  unsigned stateWidth = 32 + 32 + 32; // counter + bound + step
  for (unsigned i = 0; i < loop.iterArgs.size(); i++) {
    stateWidth += getBitWidth(loop.iterArgTypes[i]);
  }

  auto *stateMod = STLLibrary::createFIFO2IModule(stateWidth, circuit);
  builder.restoreInsertionPoint(savedIP);
  std::string stateName = loop.loopName + "_state_fifo";
  loop.loop_state_fifo = mainModule->addInstance(
      stateName, stateMod, {mainClk.getValue(), mainRst.getValue()});
  llvm::dbgs() << "[LoopHandler] Created loop state FIFO: " << stateName
               << " (width=" << stateWidth << ")\n";

  // Create FIFO for induction variable that will be used by the entry rule
  // Only create it if the induction variable is actually used in the loop body
  if (loop.inductionVar) {
    // Get the loop body to check if induction variable is used
    Block *loopBody = loop.forOp ? loop.forOp.getBody() : nullptr;
    if (loopBody && isValueUsedInLoopBody(loop.inductionVar, loopBody)) {
      llvm::dbgs() << "[LoopHandler] Creating FIFO for induction variable (used in loop body)\n";
      auto *indVarMod = STLLibrary::createFIFO2IModule(
          getBitWidth(loop.inductionVar.getType()), circuit);
      builder.restoreInsertionPoint(savedIP);
      std::string indVarName = loop.loopName + "_induction_var";
      inductionVarFIFO = mainModule->addInstance(
          indVarName, indVarMod, {mainClk.getValue(), mainRst.getValue()});
      llvm::dbgs() << "[LoopHandler] Created induction variable FIFO: " << indVarName << "\n";
    } else {
      llvm::dbgs() << "[LoopHandler] Skipping induction variable FIFO creation (not used in loop body)\n";
    }
  }

  // Create state registers and loop-to-body FIFOs for input values
  // Only create for values that are actually used by the loop body
  // State registers: store values for reuse across iterations (used in next rule)
  // Loop-to-body FIFOs: separate FIFOs that loop body reads from (fed by entry/next rules)
  Block *loopBody = loop.forOp.getBody();
  llvm::dbgs() << "[LoopHandler] Creating state registers and loop-to-body FIFOs for input values used in loop body\n";
  for (auto &[value, fifo] : input_fifos) {
    if (fifo) {
      // Skip constants - they don't need FIFOs since there will be no readers
      if (value.getDefiningOp<arith::ConstantOp>()) {
        llvm::dbgs() << "[LoopHandler] Skipping constant value (constants don't need FIFOs)\n";
        continue;
      }

      // Check if this value is actually used in the loop body
      if (!isValueUsedInLoopBody(value, loopBody)) {
        llvm::dbgs() << "[LoopHandler] Skipping value (not used in loop body)\n";
        continue;
      }

      unsigned bitWidth = getBitWidth(value.getType());

      // Create state register for persistent storage across iterations
      auto *regMod = STLLibrary::createRegModule(bitWidth, 0, circuit);
      builder.restoreInsertionPoint(savedIP);
      std::string regName = loop.loopName + "_input_state_reg_" + std::to_string(loop.input_state_registers.size());
      Instance *regInstance = mainModule->addInstance(
          regName, regMod, {mainClk.getValue(), mainRst.getValue()});
      loop.input_state_registers[value] = regInstance;
      llvm::dbgs() << "[LoopHandler] Created state register: " << regName << " (width=" << bitWidth << ")\n";

      // Create loop-to-body FIFO for loop body to consume from
      auto *loopToBodyFifoMod = STLLibrary::createFIFO2IModule(bitWidth, circuit);
      builder.restoreInsertionPoint(savedIP);
      std::string loopToBodyFifoName = loop.loopName + "_loop_to_body_fifo_" + std::to_string(loop.loop_to_body_fifos.size());
      Instance *loopToBodyFifo = mainModule->addInstance(
          loopToBodyFifoName, loopToBodyFifoMod, {mainClk.getValue(), mainRst.getValue()});
      loop.loop_to_body_fifos[value] = loopToBodyFifo;
      llvm::dbgs() << "[LoopHandler] Created loop-to-body FIFO: " << loopToBodyFifoName << " (width=" << bitWidth << ")\n";
    }
  }

  return success();
}

unsigned LoopHandler::getBitWidth(mlir::Type type) {
  if (auto intType = dyn_cast<mlir::IntegerType>(type)) {
    return intType.getWidth();
  }
  return 32; // Default width
}

bool LoopHandler::isValueUsedInLoopBody(Value value, Block *loopBody) {
  if (!value || !loopBody) return false;

  // Walk through all operations in the loop body
  for (Operation &op : loopBody->getOperations()) {
    // Check if this operation uses the value
    for (Value operand : op.getOperands()) {
      if (operand == value) {
        return true;
      }
    }

    // Recursively check nested operations
    if (op.getNumRegions() > 0) {
      for (Region &region : op.getRegions()) {
        for (Block &block : region.getBlocks()) {
          if (isValueUsedInLoopBody(value, &block)) {
            return true;
          }
        }
      }
    }
  }

  return false;
}

LogicalResult LoopHandler::processLoopBodyOperations(tor::ForOp forOp, BlockInfo &loopBlock) {
  llvm::dbgs() << "[LoopHandler] Processing loop body operations for loop " << loop.loopName << "\n";

  // Get the loop body block
  Block *loopBody = forOp.getBody();
  if (!loopBody) {
    return forOp.emitError("loop has no body block");
  }

  // Create input fifos that include loop variables for the loop body
  // Use loop-to-body FIFOs instead of external input_fifos to hide cross-block FIFOs from subblocks
  llvm::DenseMap<Value, Instance*> loopBodyInputFIFOs = loop.loop_to_body_fifos;

  // Add the induction variable FIFO (created in createLoopInfrastructure) to the input map
  if (loop.inductionVar && inductionVarFIFO) {
    llvm::dbgs() << "[LoopHandler] Adding induction variable FIFO to loop body inputs\n";
    loopBodyInputFIFOs[loop.inductionVar] = inductionVarFIFO;
  }

  // Use BlockHandler's processLoopBodyAsBlocks for proper loop body processing
  // This will handle block segmentation, dataflow analysis, and rule generation
  BlockHandler loopBodyHandler(
      pass, mainModule, funcOp, poolInstance, roccInstance,
      hellaMemInstance, dmaItfc, circuit, mainClk, mainRst, opcode, regRdInstance,
      loop.token_fifos.to_body,      // Input token: signals body can start
      loop.token_fifos.body_to_next, // Output token: signals body completion
      loopBodyInputFIFOs,            // Input data FIFOs (including loop variables)
      output_fifos,                  // Output data FIFOs (to loop handler)
      loop.loopName + "_"            // Name prefix for nested blocks
  );

  llvm::dbgs() << "[LoopHandler] Processing loop body using BlockHandler::processLoopBodyAsBlocks\n";

  if (failed(loopBodyHandler.processLoopBodyAsBlocks(forOp))) {
    llvm::dbgs() << "[LoopHandler] Failed to process loop body operations\n";
    return failure();
  }

  llvm::dbgs() << "[LoopHandler] Successfully processed loop body operations\n";
  return success();
}

} // namespace mlir