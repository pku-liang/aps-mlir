//===- LoopHandler.h - Loop Handler for FIFO-based Block Coordination -*- C++ -*-===//
//
// This file declares the loop handler that processes loops first, then
// delegates blocks to BBHandler using FIFO-based coordination
//
//===----------------------------------------------------------------------===//

#ifndef APS_LOOPHANDLER_H
#define APS_LOOPHANDLER_H

#include "APSToCMT2.h"
#include "APS/APSOps.h"
#include "APS/BlockHandler.h"
#include "circt/Dialect/Cmt2/ECMT2/Instance.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {

using namespace mlir;
using namespace mlir::tor;
using namespace circt::cmt2::ecmt2;
using namespace circt::cmt2::ecmt2::stl;
using namespace circt::firrtl;

//===----------------------------------------------------------------------===//
// Loop Data Structures
//===----------------------------------------------------------------------===//

/// Represents a loop with unified block-based coordination
struct LoopInfo {
  tor::ForOp forOp;
  std::string loopName;  // Hierarchical name for nested loops (e.g., "loop_0", "loop_0_body_1")

  // Loop control values
  Value inductionVar;
  Value lowerBound, upperBound, step;
  llvm::SmallVector<Value> iterArgs;
  llvm::SmallVector<Type> iterArgTypes;

  // Loop body blocks (using standard BlockInfo from BlockHandler)
  llvm::SmallVector<BlockInfo, 4> blocks;
  llvm::DenseMap<unsigned, BlockInfo *> blockMap;
  llvm::DenseMap<Block *, BlockInfo *> mlirBlockMap;

  // Token FIFOs for canonical loop coordination (entry → body → next)
  struct TokenFIFOs {
    Instance *to_body; // Entry rule signals body can start
    Instance *body_to_next;  // Body signals completion to next rule
    Instance *next_to_exit;  // Next rule signals loop completion
  } token_fifos;

  // State management FIFOs
  Instance *loop_state_fifo;  // Carries (counter, bound, step, iter_args)
  Instance *loop_result_fifo; // Final iter_args results
  llvm::SmallVector<Instance *, 4> iter_arg_fifos; // Individual iter_arg FIFOs

  // State registers: persistent storage for input values that need to be accessed by both entry and next rules
  // Maps input Value to Register instance for consistent storage across iterations
  llvm::DenseMap<Value, Instance *> input_state_registers;

  // Loop-to-body FIFOs: separate FIFOs for cross-block values that loop body consumes from
  // Entry rule and next rule enqueue to these; loop body dequeues from these
  // This hides the external input_fifos from the loop body subblocks
  llvm::DenseMap<Value, Instance *> loop_to_body_fifos;

  LoopInfo() : loopName("uninitialized"),
        token_fifos{nullptr, nullptr, nullptr},
        loop_state_fifo(nullptr), loop_result_fifo(nullptr) {}

  // Initialize with actual loop information
  void initialize(tor::ForOp forOp, const std::string &name) {
    this->forOp = forOp;
    this->loopName = name;
  }
};

//===----------------------------------------------------------------------===//
// Loop Handler
//===----------------------------------------------------------------------===//

/// Specialized loop handler that derives from BlockHandler
/// Handles loop control structure (entry → body → next) while integrating
/// with the unified block system and producer-responsible FIFO coordination
class LoopHandler : public BlockHandler {
public:
  LoopHandler(APSToCMT2GenPass *pass, Module *mainModule, tor::FuncOp funcOp,
              Instance *poolInstance, Instance *roccInstance,
              Instance *hellaMemInstance, InterfaceDecl *dmaItfc,
              Circuit &circuit, Clock mainClk, Reset mainRst,
              unsigned long opcode, Instance *regRdInstance, Instance *input_token_fifo,
              Instance *output_token_fifo,
              llvm::DenseMap<Value, Instance *> &input_fifos,
              llvm::DenseMap<Value, llvm::SmallVector<std::pair<BlockInfo*, Instance*>, 4>> &output_fifos,
              const std::string &namePrefix = "");

  /// Process a loop block within the unified block system
  LogicalResult processLoopBlock(BlockInfo &loopBlock);

  /// Get the loop that this handler processes
  const LoopInfo &getLoop() const { return loop; }

protected:
  // Override processBlock to handle loop specialization
  LogicalResult processBlock(BlockInfo &block) override;

private:
  // Single loop that this handler processes
  LoopInfo loop;

  //===--------------------------------------------------------------------===//
  // Rule Generation
  //===--------------------------------------------------------------------===//

  /// Generate canonical loop rules following Blockgen.md (entry → body → next)
  LogicalResult generateCanonicalLoopRules(BlockInfo &loopBlock);

  /// Helper: Generate loop entry rule (token coordination and state init)
  LogicalResult generateLoopEntryRule(BlockInfo &loopBlock);

  /// Helper: Generate loop next rule (termination check and state update)
  LogicalResult generateLoopNextRule(BlockInfo &loopBlock);

  /// Create loop infrastructure (FIFOs for coordination)
  LogicalResult createLoopInfrastructure();

  //===--------------------------------------------------------------------===//
  // Utility Methods
  //===--------------------------------------------------------------------===//

  /// Get bit width for a type
  unsigned getBitWidth(mlir::Type type);

  /// Process loop body operations using BlockHandler with token coordination
  LogicalResult processLoopBodyOperations(tor::ForOp forOp, BlockInfo &loopBlock);

  /// Check if a value is used in the loop body
  bool isValueUsedInLoopBody(Value value, Block *loopBody);

private:
  // Induction variable FIFO - created in infrastructure, used by entry rule
  Instance *inductionVarFIFO = nullptr;
};

} // namespace mlir

#endif // APS_LOOPHANDLER_H