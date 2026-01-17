//===- BBHandler.h - Basic Block Handler for Rule Generation -*- C++ -*-===//
//
// This file declares the object-oriented basic block handling for TOR
// function rule generation
//
//===----------------------------------------------------------------------===//

#ifndef APS_BBHANDLER_H
#define APS_BBHANDLER_H

#include "APS/APSToCMT2.h"
#include "APS/APSOps.h"
#include "APS/BlockHandler.h"
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
// Forward Declarations
//===----------------------------------------------------------------------===//

class OperationGenerator;
class ArithmeticOpGenerator;
class MemoryOpGenerator;
class InterfaceOpGenerator;
class RegisterOpGenerator;

//===----------------------------------------------------------------------===//
// Basic Block Handler
//===----------------------------------------------------------------------===//

/// Handles all basic block related operations for rule generation
class BBHandler {
public:
  BBHandler(APSToCMT2GenPass *pass, Module *mainModule, tor::FuncOp funcOp,
            Instance *poolInstance, Instance *roccInstance,
            Instance *hellaMemInstance, Instance *regRdInstance,
            InterfaceDecl *dmaItfc, Circuit &circuit, Clock mainClk, Reset mainRst,
            unsigned long opcode);

  /// Main entry point - analyze and generate rules for all basic blocks
  LogicalResult processBasicBlocks();

  /// Get the slot order after analysis
  const llvm::SmallVector<int64_t, 8> &getSlotOrder() const {
    return slotOrder;
  }

  /// Get slot map for external access
  const llvm::DenseMap<int64_t, SlotInfo> &getSlotMap() const {
    return slotMap;
  }

  /// Get cross-slot FIFOs for external access
  const llvm::DenseMap<mlir::Value, llvm::SmallVector<CrossSlotFIFO *>> &
  getCrossSlotFIFOs() const {
    return crossSlotFIFOs;
  }
  llvm::DenseMap<mlir::Value, llvm::SmallVector<CrossSlotFIFO *>> &
  getCrossSlotFIFOs() {
    return crossSlotFIFOs;
  }

  /// Get function operation
  tor::FuncOp getFuncOp() const { return funcOp; }

  /// Get module instances
  Instance *getPoolInstance() const { return poolInstance; }
  Instance *getRoccInstance() const { return roccInstance; }
  Instance *getHellaMemInstance() const { return hellaMemInstance; }
  InterfaceDecl *getDmaInterface() const { return dmaItfc; }

  /// Get memory entry map from pass
  const llvm::DenseMap<llvm::StringRef, MemoryEntryInfo> &
  getMemEntryMap() const { return pass->memEntryMap; };

  /// Process a single basic block using BlockInfo
  LogicalResult processBasicBlock(BlockInfo& block);

  /// Process a single operation within a basic block
  LogicalResult processOperationInBlock(Operation *op, mlir::OpBuilder &b, 
                                        mlir::Location loc,
                                        llvm::DenseMap<Value, Instance*> &inputFIFOs,
                                        llvm::DenseMap<Value, Instance*> &outputFIFOs);

private:
  // Core components
  APSToCMT2GenPass *pass;
  Module *mainModule;
  tor::FuncOp funcOp;
  Instance *poolInstance;
  Instance *roccInstance;
  Instance *hellaMemInstance;
  InterfaceDecl *dmaItfc;
  Circuit &circuit;
  Clock mainClk;
  Reset mainRst;
  unsigned long opcode;

  // Current block being processed (set by processBasicBlock)
  BlockInfo* currentBlock = nullptr;

  // Basic block analysis data
  llvm::DenseMap<int64_t, SlotInfo> slotMap;
  llvm::SmallVector<int64_t, 8> slotOrder;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<CrossSlotFIFO *>>
      crossSlotFIFOs;
  llvm::DenseMap<std::pair<int64_t, int64_t>, unsigned> fifoCounts;
  llvm::SmallVector<std::unique_ptr<CrossSlotFIFO>, 8> fifoStorage;

  // Operation generators
  std::unique_ptr<ArithmeticOpGenerator> arithmeticGen;
  std::unique_ptr<MemoryOpGenerator> memoryGen;
  std::unique_ptr<InterfaceOpGenerator> interfaceGen;
  std::unique_ptr<RegisterOpGenerator> registerGen;

  // RoCC command bundle caching
  mlir::Value cachedRoCCCmdBundle;
  Instance *regRdInstance = nullptr;

  // Stage token FIFOs for synchronization
  llvm::DenseMap<int64_t, Instance *> stageTokenFifos;

  // Rule precedence pairs
  llvm::SmallVector<std::pair<std::string, std::string>, 4> precedencePairs;

  //===--------------------------------------------------------------------===//
  // Analysis Phase
  //===--------------------------------------------------------------------===//

  /// Collect operations by time slot from the function body
  LogicalResult collectOperationsBySlot();

  /// Collect operations from a specific list by time slot (for single basic block)
  LogicalResult collectOperationsFromList(llvm::SmallVector<Operation*> &operations);

  /// Validate that all operations are supported
  LogicalResult validateOperations();

  /// Build cross-slot FIFO mapping for value communication (old interface)
  LogicalResult buildCrossSlotFIFOs();

  /// Create token FIFOs for stage synchronization (old interface)
  LogicalResult createTokenFIFOs();

  /// Instantiate cross-slot FIFO hardware modules
  LogicalResult instantiateCrossSlotFIFOs();

  //===--------------------------------------------------------------------===//
  // Rule Generation Phase
  //===--------------------------------------------------------------------===//

  /// Generate rules for each time slot
  LogicalResult generateSlotRules();

  /// Handle RoCC command bundle in slot 0
  LogicalResult handleRoCCCommandBundle(mlir::OpBuilder &b, Location loc);

  /// Handle token synchronization between stages
  LogicalResult handleTokenSynchronization(mlir::OpBuilder &b, Location loc,
                                           int64_t slot);

  /// Write token to next stage at the end
  LogicalResult writeTokenToNextStage(mlir::OpBuilder &b, Location loc,
                                      int64_t slot);

  LogicalResult
  generateRuleForOperation(Operation *op, mlir::OpBuilder &b, Location loc,
                           int64_t slot,
                           llvm::DenseMap<mlir::Value, mlir::Value> &localMap);

  //===--------------------------------------------------------------------===//
  // Utility Methods
  //===--------------------------------------------------------------------===//

  public:
  /// Get slot for an operation
  std::optional<int64_t> getSlotForOp(Operation *op);

  /// Check if an operation is a control flow boundary
  bool isControlFlowBoundary(Operation *op);

  /// Convert MLIR type to FIRRTL type
  mlir::Type toFirrtlType(mlir::Type type, mlir::MLIRContext *ctx);

  /// Round up to power of 2
  unsigned int roundUpToPowerOf2(unsigned int n);

  /// Calculate log2 (floor) of a number
  unsigned int log2Floor(unsigned int n);
};

//===----------------------------------------------------------------------===//
// Operation Generator Base Class
//===----------------------------------------------------------------------===//

/// Base class for operation-specific generators
class OperationGenerator {
public:
  OperationGenerator(BBHandler *bbHandler) : bbHandler(bbHandler) {}
  virtual ~OperationGenerator() = default;

  /// Generate rule implementation for this operation
  virtual LogicalResult
  generateRule(Operation *op, mlir::OpBuilder &b, Location loc, int64_t slot,
               llvm::DenseMap<mlir::Value, mlir::Value> &localMap) = 0;

  /// Check if this generator can handle the operation
  virtual bool canHandle(Operation *op) const = 0;

protected:
  BBHandler *bbHandler;

  /// Helper to get value in current rule context
  virtual FailureOr<mlir::Value>
  getValueInRule(mlir::Value v, Operation *currentOp, unsigned operandIndex,
                 mlir::OpBuilder &b,
                 llvm::DenseMap<mlir::Value, mlir::Value> &localMap,
                 Location loc);
};

//===----------------------------------------------------------------------===//
// Arithmetic Operation Generator
//===----------------------------------------------------------------------===//

/// Handles arithmetic operations (add, sub, mul)
class ArithmeticOpGenerator : public OperationGenerator {
public:
  ArithmeticOpGenerator(BBHandler *bbHandler) : OperationGenerator(bbHandler) {}

  LogicalResult
  generateRule(Operation *op, mlir::OpBuilder &b, Location loc, int64_t slot,
               llvm::DenseMap<mlir::Value, mlir::Value> &localMap) override;

  bool canHandle(Operation *op) const override;

private:
  /// Perform arithmetic operation using Signal abstraction
  LogicalResult
  performArithmeticOp(mlir::OpBuilder &b, Location loc, mlir::Value lhs,
                      mlir::Value rhs, mlir::Value result, StringRef opName,
                      llvm::DenseMap<mlir::Value, mlir::Value> &localMap);

  /// Perform comparison operation using Signal abstraction
  LogicalResult
  performComparisonOp(mlir::OpBuilder &b, Location loc, mlir::Value lhs,
                      mlir::Value rhs, mlir::Value result,
                      mlir::tor::CmpIPredicate predicate,
                      llvm::DenseMap<mlir::Value, mlir::Value> &localMap);

  /// Perform select operation using Signal abstraction (mux)
  LogicalResult
  performSelectOp(mlir::OpBuilder &b, Location loc, mlir::Value condition,
                  mlir::Value trueValue, mlir::Value falseValue,
                  mlir::Value result,
                  llvm::DenseMap<mlir::Value, mlir::Value> &localMap);

  /// Perform unsigned extension operation using Signal abstraction (pad)
  LogicalResult
  performExtUIOp(mlir::OpBuilder &b, Location loc, mlir::Value input,
                 mlir::Value result,
                 llvm::DenseMap<mlir::Value, mlir::Value> &localMap);

  /// Perform truncation operation using Signal abstraction (bits)
  LogicalResult
  performTruncIOp(mlir::OpBuilder &b, Location loc, mlir::Value input,
                  mlir::Value result,
                  llvm::DenseMap<mlir::Value, mlir::Value> &localMap);

  /// Perform bit extraction operation using Signal abstraction (bits)
  LogicalResult
  performExtractOp(mlir::OpBuilder &b, Location loc, mlir::Value input,
                   unsigned lowBit, mlir::Value result,
                   llvm::DenseMap<mlir::Value, mlir::Value> &localMap);

  /// Perform shift operation using Signal abstraction (<<, >>, dshr)
  LogicalResult
  performShiftOp(mlir::OpBuilder &b, Location loc, mlir::Value lhs,
                 mlir::Value rhs, mlir::Value result, StringRef opName,
                 llvm::DenseMap<mlir::Value, mlir::Value> &localMap);

  /// Perform bitwise operation using Signal abstraction (&, |, ^)
  LogicalResult
  performBitwiseOp(mlir::OpBuilder &b, Location loc, mlir::Value lhs,
                   mlir::Value rhs, mlir::Value result, StringRef opName,
                   llvm::DenseMap<mlir::Value, mlir::Value> &localMap);

  /// Perform signed extension operation using FIRRTL asSInt/pad/asUInt
  LogicalResult
  performExtSIOp(mlir::OpBuilder &b, Location loc, mlir::Value input,
                 mlir::Value result,
                 llvm::DenseMap<mlir::Value, mlir::Value> &localMap);
};

//===----------------------------------------------------------------------===//
// Memory Operation Generator
//===----------------------------------------------------------------------===//

/// Handles memory operations (load, store, burst operations)
class MemoryOpGenerator : public OperationGenerator {
public:
  MemoryOpGenerator(BBHandler *bbHandler) : OperationGenerator(bbHandler) {}

  LogicalResult
  generateRule(Operation *op, mlir::OpBuilder &b, Location loc, int64_t slot,
               llvm::DenseMap<mlir::Value, mlir::Value> &localMap) override;

  bool canHandle(Operation *op) const override;

private:
  /// Handle SPM load request (first phase of split memory load)
  LogicalResult
  generateSpmLoadReq(aps::SpmLoadReq op, mlir::OpBuilder &b, Location loc,
                     int64_t slot,
                     llvm::DenseMap<mlir::Value, mlir::Value> &localMap);

  /// Handle SPM load collect (second phase of split memory load)
  LogicalResult
  generateSpmLoadCollect(aps::SpmLoadCollect op, mlir::OpBuilder &b, Location loc,
                         int64_t slot,
                         llvm::DenseMap<mlir::Value, mlir::Value> &localMap);

  /// Handle regular memory store
  LogicalResult
  generateMemStore(aps::MemStore op, mlir::OpBuilder &b, Location loc,
                   int64_t slot,
                   llvm::DenseMap<mlir::Value, mlir::Value> &localMap);

  /// Handle global memory load
  LogicalResult
  generateGlobalMemLoad(aps::GlobalLoad op, mlir::OpBuilder &b, Location loc,
                  int64_t slot,
                  llvm::DenseMap<mlir::Value, mlir::Value> &localMap);

  /// Handle global memory store
  LogicalResult
  generateGlobalMemStore(aps::GlobalStore op, mlir::OpBuilder &b, Location loc,
                   int64_t slot,
                   llvm::DenseMap<mlir::Value, mlir::Value> &localMap);

  /// Handle burst load request
  LogicalResult
  generateBurstLoadReq(aps::ItfcBurstLoadReq op, mlir::OpBuilder &b,
                       Location loc, int64_t slot,
                       llvm::DenseMap<mlir::Value, mlir::Value> &localMap);

  /// Handle burst load collect
  LogicalResult
  generateBurstLoadCollect(aps::ItfcBurstLoadCollect op, mlir::OpBuilder &b,
                           Location loc, int64_t slot,
                           llvm::DenseMap<mlir::Value, mlir::Value> &localMap);

  /// Handle burst store request
  LogicalResult
  generateBurstStoreReq(aps::ItfcBurstStoreReq op, mlir::OpBuilder &b,
                        Location loc, int64_t slot,
                        llvm::DenseMap<mlir::Value, mlir::Value> &localMap);

  /// Handle burst store collect
  LogicalResult
  generateBurstStoreCollect(aps::ItfcBurstStoreCollect op, mlir::OpBuilder &b,
                            Location loc, int64_t slot,
                            llvm::DenseMap<mlir::Value, mlir::Value> &localMap);
};

//===----------------------------------------------------------------------===//
// Interface Operation Generator
//===----------------------------------------------------------------------===//

/// Handles interface operations (load/store requests and collects)
class InterfaceOpGenerator : public OperationGenerator {
public:
  InterfaceOpGenerator(BBHandler *bbHandler) : OperationGenerator(bbHandler) {}

  LogicalResult
  generateRule(Operation *op, mlir::OpBuilder &b, Location loc, int64_t slot,
               llvm::DenseMap<mlir::Value, mlir::Value> &localMap) override;

  bool canHandle(Operation *op) const override;

private:
  /// Handle regular interface load request
  LogicalResult
  generateItfcLoadReq(aps::ItfcLoadReq op, mlir::OpBuilder &b, Location loc,
                      int64_t slot,
                      llvm::DenseMap<mlir::Value, mlir::Value> &localMap);

  /// Handle regular interface load collect
  LogicalResult
  generateItfcLoadCollect(aps::ItfcLoadCollect op, mlir::OpBuilder &b,
                          Location loc, int64_t slot,
                          llvm::DenseMap<mlir::Value, mlir::Value> &localMap);

  /// Handle regular interface store request
  LogicalResult
  generateItfcStoreReq(aps::ItfcStoreReq op, mlir::OpBuilder &b, Location loc,
                       int64_t slot,
                       llvm::DenseMap<mlir::Value, mlir::Value> &localMap);

  /// Handle regular interface store collect
  LogicalResult
  generateItfcStoreCollect(aps::ItfcStoreCollect op, mlir::OpBuilder &b,
                           Location loc, int64_t slot,
                           llvm::DenseMap<mlir::Value, mlir::Value> &localMap);
};

//===----------------------------------------------------------------------===//
// Register Operation Generator
//===----------------------------------------------------------------------===//

/// Handles register file operations (readrf, writerf)
class RegisterOpGenerator : public OperationGenerator {
public:
  RegisterOpGenerator(BBHandler *bbHandler) : OperationGenerator(bbHandler) {}

  LogicalResult
  generateRule(Operation *op, mlir::OpBuilder &b, Location loc, int64_t slot,
               llvm::DenseMap<mlir::Value, mlir::Value> &localMap) override;

  bool canHandle(Operation *op) const override;

  /// Set cached RoCC command bundle for register operations
  void setCachedRoCCCmdBundle(mlir::Value bundle) {
    cachedRoCCCmdBundle = bundle;
  }
  void setRegRdInstance(Instance *instance) { regRdInstance = instance; }

private:
  mlir::Value cachedRoCCCmdBundle;
  Instance *regRdInstance = nullptr;

  /// Handle register file read
  LogicalResult
  generateCpuRfRead(aps::CpuRfRead op, mlir::OpBuilder &b, Location loc,
                    int64_t slot,
                    llvm::DenseMap<mlir::Value, mlir::Value> &localMap);

  /// Handle register file write
  LogicalResult
  generateCpuRfWrite(aps::CpuRfWrite op, mlir::OpBuilder &b, Location loc,
                     int64_t slot,
                     llvm::DenseMap<mlir::Value, mlir::Value> &localMap);
};

} // namespace mlir

#endif // APS_BBHANDLER_H