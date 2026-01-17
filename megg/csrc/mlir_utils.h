#pragma once
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"
#include "mlir_ir_builder.h"
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <unordered_map>
#include <vector>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

// Include dialect extensions to register inliner and other interfaces
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace megg {

// Wrapper class for mlir::Operation* to avoid ownership issues with nanobind
// This class does NOT own the operation pointer - MLIR owns it
class MLIROperationRef {
public:
  MLIROperationRef(mlir::Operation *op = nullptr) : op_(op) {}

  mlir::Operation *get() const { return op_; }
  mlir::Operation *operator->() const { return op_; }
  mlir::Operation &operator*() const { return *op_; }
  bool valid() const { return op_ != nullptr; }

private:
  mlir::Operation *op_; // Non-owning pointer
};

enum class OperationType {
  // Module
  MODULE,

  // Function operations
  FUNC_FUNC,
  FUNC_CALL,
  FUNC_RETURN,

  // Arithmetic operations
  ARITH_ADD,
  ARITH_SUB,
  ARITH_MUL,
  ARITH_DIV,
  ARITH_REM,
  ARITH_ADDF,
  ARITH_SUBF,
  ARITH_MULF,
  ARITH_DIVF,
  ARITH_REMF,
  ARITH_CONSTANT,
  ARITH_CMPI,
  ARITH_CMPF,
  ARITH_NEG,
  ARITH_NEGF,
  ARITH_SELECT,

  // Logical operations
  ARITH_ANDI,
  ARITH_ORI,
  ARITH_XORI,

  // Cast operations
  ARITH_INDEX_CAST,
  ARITH_SITOFP,
  ARITH_UITOFP,
  ARITH_FPTOSI,
  ARITH_FPTOUI,
  ARITH_EXTSI,
  ARITH_EXTUI,
  ARITH_TRUNCI,
  ARITH_BITCAST,

  // Affine operations
  AFFINE_FOR,
  AFFINE_IF,
  AFFINE_LOAD,
  AFFINE_STORE,
  AFFINE_YIELD,
  AFFINE_APPLY,

  // SCF operations
  SCF_FOR,
  SCF_IF,
  SCF_WHILE,
  SCF_YIELD,
  SCF_CONDITION,

  // Memory operations
  MEMREF_ALLOC,
  MEMREF_DEALLOC,
  MEMREF_LOAD,
  MEMREF_STORE,
  MEMREF_CAST,
  MEMREF_GET_GLOBAL,
  MEMREF_GLOBAL,
  MEMREF_ALLOCA = 52,

  // LLVM operations
  LLVM_CALL,
  LLVM_RETURN,

  // Shift operations (explicitly numbered to match Python)
  ARITH_SHLI = 60,
  ARITH_SHRSI = 61,
  ARITH_SHRUI = 62,
 

  // Unknown
  UNKNOWN = 99
};

class MLIRModule {
public:
  MLIRModule() { initializeContext(); }
  bool parseFromString(const std::string &mlirText, bool applyInlining = true);
  bool parseFromFile(const std::string &filepath, bool applyInlining = true);
  std::string toString() const;
  std::vector<MLIROperationRef> getOperations();
  std::vector<MLIROperationRef> getFunctions();
  mlir::ModuleOp getModule();
  mlir::MLIRContext *getContext();
  void appendToModule(mlir::Operation *op);
  bool applyInlining();  // Apply inliner pass to module
  bool validate() {
    return mlir::succeeded(mlir::verify(module->getOperation()));
  }
  void genLoopName();
  std::pair<std::unique_ptr<MLIRModule>, std::unordered_map<mlir::Operation*, mlir::Operation*>> cloneWithMapping();

  // Set module from an existing ModuleOp (for casting)
  void setModule(mlir::ModuleOp moduleOp) {
    module = mlir::OwningOpRef<mlir::ModuleOp>(moduleOp);
  }

private:
  void initializeContext();
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};

class CustomPass {
public:
  explicit CustomPass(std::string name) : PassName(name) {}
  virtual ~CustomPass() = default;
  virtual bool run(mlir::ModuleOp module) = 0;
  std::string getName() const { return PassName; };

private:
  std::string PassName;
};

class PassManager {
public:
  explicit PassManager(MLIRModule &module) : mlirModule(module) {}
  ~PassManager() = default;

  // Delete copy constructor and copy assignment
  PassManager(const PassManager &) = delete;
  PassManager &operator=(const PassManager &) = delete;
  PassManager(PassManager &&) = delete;
  PassManager &operator=(PassManager &&) = delete;

  void registerAffineLoopUnrollPasses(mlir::Operation *functionOp,
                                      mlir::Operation *loopOp,
                                      uint64_t unrollSize);
  void registerAffineLoopUnrollJamPasses(mlir::Operation *functionOp,
                                         mlir::Operation *loopOp,
                                         uint64_t unrollJamSize);
  void registerAffineLoopTilingPass(mlir::Operation *functionOp,
                                    mlir::Operation *loopOp,
                                    uint64_t defaultTileSize = 32);
  void registerCustomPass(std::unique_ptr<CustomPass> pass);
  bool runPass(const std::string &passName);
  bool runAllPass();

  // Verify the module is valid
  bool verifyModule();
  void clearCustomPasses();

private:
  MLIRModule &mlirModule;
  std::unordered_map<std::string, std::unique_ptr<CustomPass>> customPasses;
};

} // namespace megg