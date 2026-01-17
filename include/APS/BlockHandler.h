//===- BlockHandler.h - Unified Block Handler -*- C++ -*-===//
//
// This file declares the unified block handler that treats all control flow
// as blocks with producer-responsible FIFO coordination
//
//===----------------------------------------------------------------------===//

#ifndef APS_BLOCKHANDLER_H
#define APS_BLOCKHANDLER_H

#include "APS/APSOps.h"
#include "APS/APSToCMT2.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {

using namespace mlir;
using namespace mlir::tor;
using namespace circt::cmt2::ecmt2;
using namespace circt::cmt2::ecmt2::stl;
using namespace circt::firrtl;

//===----------------------------------------------------------------------===//
// Block Types
//===----------------------------------------------------------------------===//

enum class BlockType {
  REGULAR,           // Sequential operations
  LOOP_HEADER,       // Loop initialization (tor.for)
  LOOP_BODY,         // Loop iteration body
  LOOP_EXIT,         // Loop termination
  CONDITIONAL_THEN,  // If-then branch
  CONDITIONAL_ELSE,  // If-else branch
  CONDITIONAL_EXIT   // After conditional
};

//===----------------------------------------------------------------------===//
// Block Information
//===----------------------------------------------------------------------===//

/// Represents a unified block with FIFO-based communication
struct BlockInfo {
  unsigned blockId;
  std::string blockName;  // Hierarchical name for nested blocks
  Block* mlirBlock;       // Parent MLIR block (for context)
  BlockType type;
  int64_t startTime, endTime;

  // Operations belonging to this specific block segment
  llvm::SmallVector<Operation*> operations;

  // Values produced/consumed by this block
  llvm::SmallVector<Value> producedValues;
  llvm::SmallVector<Value> consumedValues;

  // Cross-block communication - producer creates these
  // For each produced value, map to list of (consumer_block, FIFO) pairs
  // One value may have multiple consumers, each needs its own FIFO
  llvm::DenseMap<Value, llvm::SmallVector<std::pair<BlockInfo*, Instance*>, 4>> output_fifos;
  llvm::DenseMap<Value, Instance*> input_fifos;   // Values this block consumes

  // Block execution coordination - unified token system
  Instance* input_token_fifo;     // Token coordination (prev block complete -> this block ready)
  Instance* output_token_fifo;    // Token to next block (this block complete -> next block ready)

  // Block-specific data (union-like pattern)
  bool is_loop_block;
  bool is_conditional_block;

  BlockInfo(unsigned blockId, const std::string& blockName, Block* block, BlockType type)
    : blockId(blockId), blockName(blockName), mlirBlock(block), type(type),
      startTime(-1), endTime(-1), input_token_fifo(nullptr), output_token_fifo(nullptr),
      is_loop_block(false), is_conditional_block(false) {}
};

/// Cross-block value flow information
struct CrossBlockValueFlow {
  Value value;
  BlockInfo* producer_block;
  BlockInfo* consumer_block;
  Instance* fifo;
};

//===----------------------------------------------------------------------===//
// Block Handler Base Class
//===----------------------------------------------------------------------===//

/// Unified block handler with producer-responsible FIFO coordination
class BlockHandler {
public:
  BlockHandler(APSToCMT2GenPass *pass, Module *mainModule, tor::FuncOp funcOp,
               Instance *poolInstance, Instance *roccInstance,
               Instance *hellaMemInstance, InterfaceDecl *dmaItfc,
               Circuit &circuit, Clock mainClk, Reset mainRst,
               unsigned long opcode, Instance *regRdInstance,
               Instance *inputTokenFIFO, Instance *outputTokenFIFO,
              llvm::DenseMap<Value, Instance*> &input_fifos,
              llvm::DenseMap<Value, llvm::SmallVector<std::pair<BlockInfo*, Instance*>, 4>> &output_fifos,
              const std::string &namePrefix = "");

  /// Process all blocks in the function
  LogicalResult processFunctionAsBlocks();

  LogicalResult processLoopBodyAsBlocks(tor::ForOp loopOp);

  /// Process a specific block (virtual for specialization)
  virtual LogicalResult processBlock(BlockInfo& block);

  /// Create producer FIFO for a value with def-use naming
  Instance* createProducerFIFO(Value value, unsigned producerBlockId, unsigned consumerBlockId, unsigned counter);

  /// Find all consumers of a value
  llvm::SmallVector<BlockInfo*> findValueConsumers(Value value);

protected:
  // Core components
  APSToCMT2GenPass *pass;
  Module *mainModule;
  tor::FuncOp funcOp;
  Instance *poolInstance;
  Instance *roccInstance;
  Instance *hellaMemInstance;
  Instance *regRdInstance;  // Shared reg_rd register for all blocks
  InterfaceDecl *dmaItfc;
  Circuit &circuit;
  Clock mainClk;
  Reset mainRst;
  unsigned long opcode;

  // Name prefix for hierarchical naming (e.g., "43_" for opcode, "43_loop_1_" for nested)
  std::string namePrefix;

  // External token FIFOs for top-level block coordination
  Instance *inputTokenFIFO;
  Instance *outputTokenFIFO;

  // Block information
  llvm::SmallVector<BlockInfo, 4> blocks;
  llvm::DenseMap<unsigned, BlockInfo*> blockMap;
  llvm::DenseMap<Block*, BlockInfo*> mlirBlockMap;

  // Cross-block value flows
  llvm::SmallVector<CrossBlockValueFlow> crossBlockFlows;
  
  // Unified token FIFOs for cross-block coordination (block i -> block i+1)
  llvm::DenseMap<std::pair<unsigned, unsigned>, Instance*> unifiedTokenFIFOs;

  // Input and Output FIFOs of the blocks, via input from parent
  llvm::DenseMap<Value, Instance*> input_fifos;
  // Output FIFOs: for each value, list of (consumer_block, FIFO) pairs
  llvm::DenseMap<Value, llvm::SmallVector<std::pair<BlockInfo*, Instance*>, 4>> output_fifos;

  //===--------------------------------------------------------------------===//
  // Input Distribution Infrastructure (for sub-blocks)
  //===--------------------------------------------------------------------===//

  // Maps: input_value -> sub_block_index -> dedicated FIFO for that sub-block
  llvm::DenseMap<Value, llvm::DenseMap<unsigned, Instance*>> input_distribution_fifos;

  // Flag: whether this block needs input distribution rule
  bool needsInputDistribution = false;

  // Token FIFO: parent -> distribution rule (only if distribution needed)
  Instance* input_distribution_token_fifo = nullptr;

  //===--------------------------------------------------------------------===//
  // Block Analysis
  //===--------------------------------------------------------------------===//

  /// Identify all blocks in the function
  LogicalResult identifyBlocksByFuncOp();

  LogicalResult identifyBlocksByLoop(tor::ForOp loopOp);

  /// Segment a block into blocks based on control flow with FIFO propagation
  LogicalResult segmentBlockIntoBlocks(Block *mlirBlock, unsigned &blockId);

  /// Analyze a single operation within a block
  void analyzeOperationInBlock(Operation *op, BlockInfo &block);

  /// Analyze data flow between blocks
  LogicalResult analyzeCrossBlockDataflow();

  /// Determine block types (regular, loop, conditional, etc.)
  BlockType determineBlockType(Block* block);

  /// Check if block contains loop operations
  bool containsLoop(Block* block);

  /// Check if block contains conditional operations
  bool containsConditional(Block* block);

  //===--------------------------------------------------------------------===//
  // FIFO Infrastructure
  //===--------------------------------------------------------------------===//

  /// Create all producer FIFOs for cross-block communication
  LogicalResult createProducerFIFOs();

  /// Get unique FIFO name
  std::string getFIFOName(StringRef prefix, unsigned blockId, StringRef suffix = "");

  //===--------------------------------------------------------------------===//
  // Input Distribution (for sub-blocks with shared inputs)
  //===--------------------------------------------------------------------===//

  /// Analyze if input distribution is needed for sub-blocks
  LogicalResult analyzeInputDistributionNeeds();

  /// Create input distribution infrastructure (FIFOs and token coordination)
  LogicalResult createInputDistributionInfrastructure();

  /// Generate input distribution rule (dequeue once, enqueue to all sub-blocks)
  LogicalResult generateInputDistributionRule();

  //===--------------------------------------------------------------------===//
  // Rule Generation
  //===--------------------------------------------------------------------===//

  /// Process all blocks through specialized handlers (BBHandler/LoopHandler)
  LogicalResult processAllBlocks();

  /// Process a regular block using internal BB logic
  LogicalResult processRegularBlockWithBBHandler(BlockInfo& block);

  /// Create intra-block coordination FIFOs (ready/complete for slot-to-slot)
  LogicalResult createBlockTokenFIFOs();

  //===--------------------------------------------------------------------===//
  // Utility Methods
  //===--------------------------------------------------------------------===//

  /// Get bit width for FIFO sizing
  unsigned getBitWidth(mlir::Type type);

  /// Check if value is used in target block
  bool isValueUsedInBlock(Value value, BlockInfo& targetBlock);

  /// Check if a value comes from a virtual operation (doesn't need FIFO)
  bool isVirtualValue(Value value);

  /// Generate hierarchical block name for nested blocks
  std::string generateBlockName(unsigned blockId, BlockType type, const std::string& parentName = "");
};

} // namespace mlir

#endif // APS_BLOCKHANDLER_H