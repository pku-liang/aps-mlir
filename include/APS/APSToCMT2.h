//===- RuleGeneration.h - Rule Generation for TOR Functions -----*- C++ -*-===//
//
// This file declares the rule generation functionality for TOR functions
// that was previously in APSToCMT2GenPass.cpp
//
//===----------------------------------------------------------------------===//

#ifndef APS_RULEGENERATION_H
#define APS_RULEGENERATION_H

#include "APS/APSOps.h"
#include "APS/Passes.h"
#include "TOR/TOR.h"
#include "circt/Dialect/Cmt2/Cmt2Dialect.h"
#include "circt/Dialect/Cmt2/ECMT2/Circuit.h"
#include "circt/Dialect/Cmt2/ECMT2/FunctionLike.h"
#include "circt/Dialect/Cmt2/ECMT2/Instance.h"
#include "circt/Dialect/Cmt2/ECMT2/Interface.h"
#include "circt/Dialect/Cmt2/ECMT2/Module.h"
#include "circt/Dialect/Cmt2/ECMT2/ModuleLibrary.h"
#include "circt/Dialect/Cmt2/ECMT2/STLLibrary.h"
#include "circt/Dialect/Cmt2/ECMT2/Signal.h"
#include "circt/Dialect/Cmt2/ECMT2/SignalHelpers.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "mlir-c/Support.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <iomanip>

namespace mlir {

using namespace mlir;
using namespace mlir::tor;
using namespace circt::cmt2::ecmt2;
using namespace circt::cmt2::ecmt2::stl;
using namespace circt::firrtl;

/// Information about operations in a time slot
struct SlotInfo {
  llvm::SmallVector<mlir::Operation *, 4> ops;
};

/// Cross-slot FIFO for communicating values between pipeline stages
struct CrossSlotFIFO {
  mlir::Value producerValue;        // The SSA value being communicated
  int64_t producerSlot = 0;         // Source time slot
  int64_t consumerSlot = 0;         // Destination time slot
  std::string instanceName;         // FIFO instance name
  mlir::Type firType;               // FIRRTL type for the value
  Instance *fifoInstance = nullptr; // FIFO module instance
  llvm::SmallVector<std::pair<mlir::Operation *, unsigned>>
      consumers; // (consumerOp, operandIndex) pairs
};

/// Helper struct for memory entry information
struct MemoryEntryInfo {
  std::string name;
  uint32_t baseAddress;
  uint32_t bankSize;
  uint32_t numBanks;
  bool isCyclic;
  int dataWidth;
  int addrWidth;
  int depth;
  llvm::SmallVector<Instance *, 4> bankInstances;
};

/// Result struct for generateMemoryPool containing module and memory entry map
struct MemoryPoolResult {
  Module *poolModule;
  llvm::DenseMap<llvm::StringRef, MemoryEntryInfo> memEntryMap;
};

// Struct to hold all instances returned by generateRuleBasedMainModule
struct MainModuleInstances {
  Module *mainModule;
  Instance *poolInstance;
  Instance *roccInstance;
  Instance *hellaMemInstance;
};

struct APSToCMT2GenPass
    : public PassWrapper<APSToCMT2GenPass, OperationPass<mlir::ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(APSToCMT2GenPass)

  void getDependentDialects(DialectRegistry &registry) const override;

  void runOnOperation() override;

private:
  /// Get memref.global by symbol name
  memref::GlobalOp getGlobalMemRef(mlir::Operation *scope,
                                   StringRef symbolName);

  /// Extract data width and address width from a memref type
  bool extractMemoryParameters(memref::GlobalOp globalOp, int &dataWidth,
                               int &addrWidth, int &depth);

  void addBurstMemoryInterface(Circuit &circuit);

  void addRoccAndHellaMemoryInterface(Circuit &circuit);

  /// Generate a bank wrapper module that encapsulates bank selection and data
  /// alignment
  Module *generateBankWrapperModule(const MemoryEntryInfo &entryInfo,
                                    Circuit &circuit, size_t bankIdx,
                                    ExternalModule *memMod, Clock clk,
                                    Reset rst, bool burstEnable);

  Module *
  generateMemoryEntryModule(const MemoryEntryInfo &entryInfo, Circuit &circuit,
                            Clock clk, Reset rst,
                            const llvm::SmallVector<std::string, 4> &bankNames);

  /// Generate burst access logic (address decoding and bank selection)
  void generateBurstAccessLogic(
      Module *poolModule,
      const llvm::SmallVector<MemoryEntryInfo> &memEntryInfos, Circuit &circuit,
      Clock clk, Reset rst);

  void addRoCCAndMemoryMethodToMainModule(Module *mainModule,
                                          Instance *roccInstance,
                                          Instance *hellaMemInstance);

  MainModuleInstances generateRuleBasedMainModule(ModuleOp moduleOp,
                                                  Circuit &circuit,
                                                  Module *poolModule,
                                                  Module *roccModule,
                                                  Module *hellaMemModule,
                                                  SmallVector<std::tuple<std::string, int8_t>, 8> glblRegister);
  /// Generate the CMT2 memory pool module
  MemoryPoolResult generateMemoryPool(Circuit &circuit, ModuleOp moduleOp,
                                      aps::MemoryMapOp memoryMapOp);

  /// Generate the CMT2 memory pool module
  SmallVector<std::tuple<std::string, int8_t>, 8> generateGlobalRegisterList(Circuit &circuit, ModuleOp moduleOp,
                                      aps::MemoryMapOp memoryMapOp);

  /// Add burst read/write methods to main module
  void addBurstMethodsToMainModule(Module *mainModule, Instance *poolInstance);

  BundleType getHellaRespBundleType(Builder &builder);

  BundleType getHellaUserCmdBundleType(Builder &builder);

  BundleType getHellaCmdBundleType(Builder &builder);
  
  BundleType getHellaUserRespBundleType(Builder &builder);

  /// Generate Memory Translator module that bridges HellaCache interface with
  /// User Memory Protocol
  Module *generateMemoryAdapter(Circuit &circuit);

  BundleType getRoccCmdBundleType(Builder &builder);

  BundleType getRoccRespBundleType(Builder &builder);

  Module *
  generateRoCCAdapter(Circuit &circuit,
                      const llvm::SmallVector<unsigned long, 4> &opcodes);
public:
  // Memory entry map for fast lookup by name
  llvm::DenseMap<llvm::StringRef, MemoryEntryInfo> memEntryMap;

  /// Generate rules for a specific TOR function
  /// This function handles:
  /// - Pipeline stage analysis and scheduling
  /// - Cross-slot FIFO generation for data communication
  /// - Rule generation for each time slot
  /// - Memory interface handling
  /// - RoCC instruction processing
  void generateRulesForFunction(Module *mainModule, tor::FuncOp funcOp,
                                Instance *poolInstance, Instance *roccInstance,
                                Instance *hellaMemInstance,
                                InterfaceDecl *dmaItfc, Circuit &circuit,
                                Clock mainClk, Reset mainRst,
                                unsigned long opcode);

  StringRef getArgument() const final { return "aps-to-cmt2-gen"; }
  StringRef getDescription() const final {
    return "Generate CMT2 hardware from APS TOR operations";
  }
};
} // namespace mlir

#endif // APS_RULEGENERATION_H