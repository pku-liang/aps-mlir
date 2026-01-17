#include "mlir_utils.h"
#include "loop_raise.h"
#include "loop_tiling.h"
#include "loop_unroll.h"
#include "loop_unroll_jam.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/Support/Casting.h"
#include <stdexcept>
namespace nb = nanobind;
using namespace nb::literals;

namespace megg {

void MLIRModule::initializeContext() {
  context = std::make_unique<mlir::MLIRContext>();

  // Create dialect registry and register all necessary dialects
  mlir::DialectRegistry registry;

  // Register core dialects
  registry.insert<mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                  mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::LLVM::LLVMDialect>();

  // Register dialect extensions and interfaces (including inliner interfaces)
  // This is CRITICAL for avoiding "DialectInlinerInterface not implemented"
  // errors
  mlir::func::registerInlinerExtension(registry);
  mlir::LLVM::registerInlinerInterface(registry);

  // Apply registry to context
  context->appendDialectRegistry(registry);

  // Load all registered dialects with their extensions
  context->loadAllAvailableDialects();

  context->enableMultithreading();
  module = mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
}

bool MLIRModule::parseFromString(const std::string &mlirText,
                                 bool applyInliningFlag) {
  llvm::SourceMgr sourceMgr;
  auto memBuffer = llvm::MemoryBuffer::getMemBuffer(mlirText);
  sourceMgr.AddNewSourceBuffer(std::move(memBuffer), llvm::SMLoc());

  // Parse with a fresh context that has all dialects loaded
  mlir::OwningOpRef<mlir::ModuleOp> parsedModule =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, context.get());

  if (!parsedModule || mlir::failed(mlir::verify(*parsedModule))) {
    return false;
  }
  std::cout << "parse successful" << std::endl;
  module = std::move(parsedModule);

  if (applyInliningFlag) {
    std::cout << "Attempting to apply inlining..." << std::endl;
    if (!applyInlining()) {
      std::cerr << "Warning: inlining pass failed - this may be due to missing "
                   "dialect interfaces"
                << std::endl;
      std::cerr << "Continuing without inlining..." << std::endl;
      // Don't fail completely - module is still valid
    } else {
      std::cout << "Inlining applied successfully" << std::endl;
    }
  }
  //   genLoopName();
  return true;
}

bool MLIRModule::parseFromFile(const std::string &filepath,
                               bool applyInliningFlag) {
  std::string errorMessage;
  auto fileBuffer = mlir::openInputFile(filepath, &errorMessage);
  if (!fileBuffer) {
    return false;
  }
  return parseFromString(fileBuffer->getBuffer().str(), applyInliningFlag);
}

std::string MLIRModule::toString() const {
  if (!module) {
    return "";
  }
  std::string result;
  llvm::raw_string_ostream stream(result);

  // Use default OpPrintingFlags (empty constructor gives pretty form)
  // DO NOT set printGenericOpForm - default is already pretty form
  mlir::OpPrintingFlags flags;

  module.get()->print(stream, flags);
  return result;
}

std::vector<MLIROperationRef> MLIRModule::getOperations() {
  std::vector<MLIROperationRef> operations;
  if (!module) {
    return operations;
  }

  module.get()->walk([&operations](mlir::Operation *op) {
    operations.push_back(MLIROperationRef(op));
  });

  return operations;
}

std::vector<MLIROperationRef> MLIRModule::getFunctions() {
  std::vector<MLIROperationRef> functions;
  if (!module) {
    return functions;
  }

  module.get()->walk([&functions](mlir::Operation *op) {
    if (op->getName().getStringRef() == "func.func") {
      functions.push_back(MLIROperationRef(op));
    }
  });

  return functions;
}

mlir::ModuleOp MLIRModule::getModule() {
  return module ? module.get() : nullptr;
}

mlir::MLIRContext *MLIRModule::getContext() { return context.get(); }


void MLIRModule::appendToModule(mlir::Operation *op) {
  if (module && op) {
    // Clone the operation instead of moving it to avoid use-def chain issues
    // This creates a deep copy of the operation and all its nested
    // regions/blocks
    mlir::Operation *cloned = op->clone();

    // Add the cloned operation to the module's body
    module.get().getBody()->push_back(cloned);
  }
}

bool MLIRModule::applyInlining() {
  if (!module)
    return false;
  mlir::PassManager pm(context.get());
  pm.addPass(mlir::createInlinerPass());
  if (mlir::failed(pm.run(module.get())))
    return false;

  return true;
}

std::pair<std::unique_ptr<MLIRModule>,
          std::unordered_map<mlir::Operation *, mlir::Operation *>>
MLIRModule::cloneWithMapping() {
  if (!module) {
    return std::make_pair(
        nullptr, std::unordered_map<mlir::Operation *, mlir::Operation *>{});
  }

  // Create a new MLIRModule for the cloned module
  auto clonedMLIRModule = std::make_unique<MLIRModule>();

  // Clone the module operation (clone() returns ModuleOp value)
  mlir::ModuleOp clonedModuleOp = module->clone();

  // Replace the new module's content with the cloned one
  clonedMLIRModule->module = mlir::OwningOpRef<mlir::ModuleOp>(clonedModuleOp);

  // Build the operation mapping by walking both modules in parallel
  // We use structural matching based on position and dump string
  std::unordered_map<mlir::Operation *, mlir::Operation *> opMap;

  llvm::SmallVector<mlir::Operation *> oldOps;
  llvm::SmallVector<mlir::Operation *> newOps;

  module->walk([&](mlir::Operation *op) { oldOps.push_back(op); });

  clonedMLIRModule->module->walk(
      [&](mlir::Operation *op) { newOps.push_back(op); });

  assert(oldOps.size() == newOps.size());
  for (size_t i = 0; i < oldOps.size(); ++i) {
    //   std::cerr << "Old" << oldOps[i] <<"\nNew"<<newOps[i];
    opMap[oldOps[i]] = newOps[i];
  }

  return {std::move(clonedMLIRModule), std::move(opMap)};
}

void PassManager::registerAffineLoopUnrollPasses(mlir::Operation *functionOp,
                                                 mlir::Operation *loopOp,
                                                 uint64_t unrollSize) {
  auto loopUnroller = std::make_unique<LoopUnroller>(
      static_cast<int>(unrollSize), functionOp, loopOp);
  registerCustomPass(std::move(loopUnroller));
}

void PassManager::registerAffineLoopUnrollJamPasses(mlir::Operation *functionOp,
                                                    mlir::Operation *loopOp,
                                                    uint64_t unrollJamSize) {
  auto loopUnrollJammer = std::make_unique<LoopUnrollJammer>(
      static_cast<int>(unrollJamSize), functionOp, loopOp);
  registerCustomPass(std::move(loopUnrollJammer));
}

void PassManager::registerAffineLoopTilingPass(mlir::Operation *functionOp,
                                               mlir::Operation *loopOp,
                                               uint64_t tileSize) {
  auto tiler = std::make_unique<LoopTiler>(functionOp, tileSize);
  if (loopOp != nullptr) {
    tiler->addTileConfig(loopOp, tileSize);
  }
  registerCustomPass(std::move(tiler));
}

void PassManager::registerCustomPass(std::unique_ptr<CustomPass> pass) {
  if (pass) {
    std::string name = pass->getName();
    customPasses[name] = std::move(pass);
  }
}

bool PassManager::runPass(const std::string &passName) {
  auto it = customPasses.find(passName);
  bool modify = false;
  if (it != customPasses.end()) {
    modify |= it->second->run(mlirModule.getModule());
  }
  return modify;
}

bool PassManager::runAllPass() {
  bool modify = false;
  for (auto &pass : customPasses) {
    modify |= pass.second->run(mlirModule.getModule());
  }
  return modify;
}

bool PassManager::verifyModule() {
  auto module = mlirModule.getModule();
  if (!module) {
    return false;
  }
  return mlir::succeeded(mlir::verify(module));
}

void PassManager::clearCustomPasses() { customPasses.clear(); }

// Helper function to get operation type from operation name
OperationType getOperationType(const std::string &opName) {
  // Module
  if (opName == "builtin.module")
    return OperationType::MODULE;

  // Function operations
  if (opName == "func.func")
    return OperationType::FUNC_FUNC;
  if (opName == "func.call")
    return OperationType::FUNC_CALL;
  if (opName == "func.return")
    return OperationType::FUNC_RETURN;

  // Arithmetic operations
  if (opName == "arith.addi")
    return OperationType::ARITH_ADD;
  if (opName == "arith.subi")
    return OperationType::ARITH_SUB;
  if (opName == "arith.muli")
    return OperationType::ARITH_MUL;
  if (opName == "arith.divi" || opName == "arith.divui" ||
      opName == "arith.divsi")
    return OperationType::ARITH_DIV;
  if (opName == "arith.remi" || opName == "arith.remui" ||
      opName == "arith.remsi")
    return OperationType::ARITH_REM;
  if (opName == "arith.addf")
    return OperationType::ARITH_ADDF;
  if (opName == "arith.subf")
    return OperationType::ARITH_SUBF;
  if (opName == "arith.mulf")
    return OperationType::ARITH_MULF;
  if (opName == "arith.divf")
    return OperationType::ARITH_DIVF;
  if (opName == "arith.remf")
    return OperationType::ARITH_REMF;
  if (opName == "arith.constant")
    return OperationType::ARITH_CONSTANT;
  if (opName == "arith.cmpi")
    return OperationType::ARITH_CMPI;
  if (opName == "arith.cmpf")
    return OperationType::ARITH_CMPF;
  if (opName == "arith.negf")
    return OperationType::ARITH_NEGF;
  if (opName == "arith.negi")
    return OperationType::ARITH_NEG;
  if (opName == "arith.select")
    return OperationType::ARITH_SELECT;

  // Logical operations
  if (opName == "arith.andi")
    return OperationType::ARITH_ANDI;
  if (opName == "arith.ori")
    return OperationType::ARITH_ORI;
  if (opName == "arith.xori")
    return OperationType::ARITH_XORI;

  // Shift operations
  if (opName == "arith.shli")
    return OperationType::ARITH_SHLI;
  if (opName == "arith.shrsi")
    return OperationType::ARITH_SHRSI;
  if (opName == "arith.shrui")
    return OperationType::ARITH_SHRUI;

  // Cast operations
  if (opName == "arith.index_cast" || opName == "arith.index_castui")
    return OperationType::ARITH_INDEX_CAST;
  if (opName == "arith.sitofp")
    return OperationType::ARITH_SITOFP;
  if (opName == "arith.uitofp")
    return OperationType::ARITH_UITOFP;
  if (opName == "arith.fptosi")
    return OperationType::ARITH_FPTOSI;
  if (opName == "arith.fptoui")
    return OperationType::ARITH_FPTOUI;
  if (opName == "arith.extsi")
    return OperationType::ARITH_EXTSI;
  if (opName == "arith.extui")
    return OperationType::ARITH_EXTUI;
  if (opName == "arith.trunci")
    return OperationType::ARITH_TRUNCI;
  if (opName == "arith.bitcast")
    return OperationType::ARITH_BITCAST;

  // Affine operations
  if (opName == "affine.for")
    return OperationType::AFFINE_FOR;
  if (opName == "affine.if")
    return OperationType::AFFINE_IF;
  if (opName == "affine.load")
    return OperationType::AFFINE_LOAD;
  if (opName == "affine.store")
    return OperationType::AFFINE_STORE;
  if (opName == "affine.yield")
    return OperationType::AFFINE_YIELD;
  if (opName == "affine.apply")
    return OperationType::AFFINE_APPLY;

  // SCF operations
  if (opName == "scf.for")
    return OperationType::SCF_FOR;
  if (opName == "scf.if")
    return OperationType::SCF_IF;
  if (opName == "scf.while")
    return OperationType::SCF_WHILE;
  if (opName == "scf.yield")
    return OperationType::SCF_YIELD;
  if (opName == "scf.condition")
    return OperationType::SCF_CONDITION;

  // Memory operations
  if (opName == "memref.alloc")
    return OperationType::MEMREF_ALLOC;
  if (opName == "memref.alloca")
    return OperationType::MEMREF_ALLOCA;
  if (opName == "memref.dealloc")
    return OperationType::MEMREF_DEALLOC;
  if (opName == "memref.load")
    return OperationType::MEMREF_LOAD;
  if (opName == "memref.store")
    return OperationType::MEMREF_STORE;
  if (opName == "memref.cast")
    return OperationType::MEMREF_CAST;
  if (opName == "memref.get_global")
    return OperationType::MEMREF_GET_GLOBAL;
  if (opName == "memref.global")
    return OperationType::MEMREF_GLOBAL;

  // LLVM operations
  if (opName == "llvm.call")
    return OperationType::LLVM_CALL;
  if (opName == "llvm.return")
    return OperationType::LLVM_RETURN;

  return OperationType::UNKNOWN;
}

} // namespace megg

NB_MODULE(mlir_utils, m) {
  m.doc() = "MLIR utilities module for Megg compiler";

  // Context class binding
  nb::class_<mlir::MLIRContext>(m, "MLIRContext")
      .def("__repr__", [](mlir::MLIRContext &ctx) {
        return "<MLIRContext at " +
               std::to_string(reinterpret_cast<uintptr_t>(&ctx)) + ">";
      });

  // Type class bindings
  nb::class_<mlir::Type>(m, "MLIRType")
      .def(
          "__str__",
          [](mlir::Type &type) {
            std::string result;
            llvm::raw_string_ostream stream(result);
            type.print(stream);
            return result;
          },
          "Get string representation of the type");

  // FunctionType class bindings
  nb::class_<mlir::FunctionType, mlir::Type>(m, "MLIRFunctionType")
      .def(
          "get_inputs",
          [](mlir::FunctionType &funcType) {
            std::vector<mlir::Type> inputs;
            for (auto inputType : funcType.getInputs()) {
              inputs.push_back(inputType);
            }
            return inputs;
          },
          "Get input types")
      .def(
          "get_results",
          [](mlir::FunctionType &funcType) {
            std::vector<mlir::Type> results;
            for (auto resultType : funcType.getResults()) {
              results.push_back(resultType);
            }
            return results;
          },
          "Get result types")
      .def("get_num_inputs", &mlir::FunctionType::getNumInputs,
           "Get number of inputs")
      .def("get_num_results", &mlir::FunctionType::getNumResults,
           "Get number of results");

  // Value class bindings (for operands and results)
  nb::class_<mlir::Value>(m, "MLIRValue")
      .def(
          "__str__",
          [](mlir::Value &val) {
            std::string result;
            llvm::raw_string_ostream stream(result);
            val.print(stream);
            return result;
          },
          "Get string representation of the value")
      .def(
          "get_type_str",
          [](mlir::Value &val) {
            std::string result;
            llvm::raw_string_ostream stream(result);
            val.getType().print(stream);
            return result;
          },
          "Get type as string")
      .def(
          "get_type",
          [](mlir::Value &val) -> mlir::Type { return val.getType(); },
          "Get type as MLIRType object")
      .def(
          "get_ptr",
          [](mlir::Value &val) -> uintptr_t {
            return reinterpret_cast<uintptr_t>(val.getAsOpaquePointer());
          },
          "Get underlying pointer as integer for equality comparison")
      .def(
          "get_defining_op",
          [](mlir::Value &val) -> megg::MLIROperationRef {
            return megg::MLIROperationRef(val.getDefiningOp());
          },
          nb::rv_policy::reference_internal,
          "Get the operation that defines this value")
      .def_prop_ro(
          "owner",
          [](mlir::Value &val) -> megg::MLIROperationRef {
            return megg::MLIROperationRef(val.getDefiningOp());
          },
          nb::rv_policy::reference_internal,
          "Get the operation that owns/defines this value");

  // Region class bindings
  nb::class_<mlir::Region>(m, "MLIRRegion")
      .def(
          "__len__",
          [](mlir::Region &region) {
            return std::distance(region.begin(), region.end());
          },
          "Get number of blocks in the region")
      .def("empty", &mlir::Region::empty, "Check if region is empty")
      .def(
          "get_blocks",
          [](mlir::Region &region) {
            std::vector<mlir::Block *> blocks;
            for (mlir::Block &block : region) {
              blocks.push_back(&block);
            }
            return blocks;
          },
          nb::rv_policy::reference_internal, "Get all blocks in the region");

  // Block class bindings
  nb::class_<mlir::Block>(m, "MLIRBlock")
      .def(
          "__len__",
          [](mlir::Block &block) {
            return std::distance(block.begin(), block.end());
          },
          "Get number of operations in the block")
      .def(
          "__str__",
          [](mlir::Block &block) {
            std::string result;
            llvm::raw_string_ostream stream(result);
            block.print(stream);
            return result;
          },
          "print block")
      .def("empty", &mlir::Block::empty, "Check if block is empty")
      .def(
          "get_operations",
          [](mlir::Block &block) {
            std::vector<megg::MLIROperationRef> ops;
            for (mlir::Operation &op : block) {
              ops.push_back(megg::MLIROperationRef(&op));
            }
            return ops;
          },
          nb::rv_policy::reference_internal, "Get all operations in the block")
      .def(
          "get_arguments",
          [](mlir::Block &block) {
            std::vector<mlir::Value> args;
            for (unsigned i = 0; i < block.getNumArguments(); ++i) {
              args.push_back(block.getArgument(i));
            }
            return args;
          },
          nb::rv_policy::reference_internal, "Get all block arguments")
      .def(
          "get_terminator",
          [](mlir::Block &block) -> megg::MLIROperationRef {
            return megg::MLIROperationRef(block.getTerminator());
          },
          nb::rv_policy::reference_internal,
          "Get the terminator operation of the block")
      .def(
          "get_parent_op",
          [](mlir::Block &block) -> megg::MLIROperationRef {
            return megg::MLIROperationRef(block.getParentOp());
          },
          nb::rv_policy::reference_internal,
          "Get the parent operation that contains this block");

  nb::class_<megg::MLIROperationRef>(m, "MLIROperation")
      .def(
          "__str__",
          [](const megg::MLIROperationRef &ref) {
            if (!ref.valid())
              return std::string("<null operation>");
            std::string result;
            llvm::raw_string_ostream stream(result);
            ref.get()->print(stream);
            return result;
          },
          "Get string representation of the operation")
      .def(
          "get_name",
          [](const megg::MLIROperationRef &ref) {
            if (!ref.valid())
              return std::string("");
            return ref.get()->getName().getStringRef().str();
          },
          "Get operation name")
      .def(
          "erase",
          [](megg::MLIROperationRef &ref) {
            if (ref.valid()) {
              ref.get()->erase();
            }
          },
          "Erase this operation from its parent block")
      .def(
          "get_symbol_name",
          [](const megg::MLIROperationRef &ref) -> std::string {
            if (!ref.valid())
              return "";
            if (auto symName =
                    ref.get()->getAttrOfType<mlir::StringAttr>("sym_name")) {
              return symName.getValue().str();
            }
            return "";
          },
          "Get symbol name")
      .def(
          "get_type",
          [](const megg::MLIROperationRef &ref) -> int {
            if (!ref.valid())
              return static_cast<int>(megg::OperationType::UNKNOWN);
            return static_cast<int>(megg::getOperationType(
                ref.get()->getName().getStringRef().str()));
          },
          "Get operation type as integer")
      .def(
          "get_num_operands",
          [](const megg::MLIROperationRef &ref) -> size_t {
            if (!ref.valid())
              return 0;
            return ref.get()->getNumOperands();
          },
          "Get number of operands")
      .def(
          "get_num_results",
          [](const megg::MLIROperationRef &ref) -> size_t {
            if (!ref.valid())
              return 0;
            return ref.get()->getNumResults();
          },
          "Get number of results")
      .def(
          "get_operands",
          [](const megg::MLIROperationRef &ref) {
            std::vector<mlir::Value> operands;
            if (!ref.valid())
              return operands;
            for (auto operand : ref.get()->getOperands()) {
              operands.push_back(operand);
            }
            return operands;
          },
          nb::rv_policy::reference_internal, "Get all operands")
      .def(
          "get_results",
          [](const megg::MLIROperationRef &ref) {
            std::vector<mlir::Value> results;
            if (!ref.valid())
              return results;
            for (auto result : ref.get()->getResults()) {
              results.push_back(result);
            }
            return results;
          },
          nb::rv_policy::reference_internal, "Get all results")
      .def(
          "get_num_regions",
          [](const megg::MLIROperationRef &ref) -> size_t {
            if (!ref.valid())
              return 0;
            return ref.get()->getNumRegions();
          },
          "Get number of regions")
      .def(
          "get_regions",
          [](const megg::MLIROperationRef &ref) {
            std::vector<mlir::Region *> regions;
            if (!ref.valid())
              return regions;
            for (mlir::Region &region : ref.get()->getRegions()) {
              regions.push_back(&region);
            }
            return regions;
          },
          nb::rv_policy::reference_internal, "Get all regions")
      .def(
          "has_regions",
          [](const megg::MLIROperationRef &ref) {
            if (!ref.valid())
              return false;
            return ref.get()->getNumRegions() > 0;
          },
          "Check if operation has regions")
      .def(
          "is_terminator",
          [](const megg::MLIROperationRef &ref) {
            if (!ref.valid())
              return false;
            return ref.get()->hasTrait<mlir::OpTrait::IsTerminator>();
          },
          "Check if operation is a terminator")
      .def(
          "get_parent_op",
          [](const megg::MLIROperationRef &ref) -> megg::MLIROperationRef {
            if (!ref.valid())
              return megg::MLIROperationRef(nullptr);
            return megg::MLIROperationRef(ref.get()->getParentOp());
          },
          nb::rv_policy::reference_internal, "Get parent operation")
      .def(
          "get_block",
          [](const megg::MLIROperationRef &ref) -> mlir::Block * {
            if (!ref.valid())
              return nullptr;
            return ref.get()->getBlock();
          },
          nb::rv_policy::reference_internal, "Get containing block")
      .def(
          "get_attr",
          [](const megg::MLIROperationRef &ref,
             const std::string &name) -> nb::object {
            if (!ref.valid())
              return nb::none();
            auto attr = ref.get()->getAttr(name);
            if (!attr) {
              return nb::none();
            }

            // Handle different attribute types
            if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr)) {
              // Return integer value
              return nb::cast(intAttr.getInt());
            } else if (auto floatAttr = llvm::dyn_cast<mlir::FloatAttr>(attr)) {
              // Return float value
              return nb::cast(floatAttr.getValueAsDouble());
            } else if (auto strAttr = llvm::dyn_cast<mlir::StringAttr>(attr)) {
              // Return string value
              return nb::cast(strAttr.getValue().str());
            } else if (auto boolAttr = llvm::dyn_cast<mlir::BoolAttr>(attr)) {
              // Return boolean value
              return nb::cast(boolAttr.getValue());
            } else if (auto affineMapAttr =
                           llvm::dyn_cast<mlir::AffineMapAttr>(attr)) {
              // Special handling for AffineMapAttr (used by affine.for bounds)
              auto map = affineMapAttr.getValue();
              nb::dict result;

              // Extract map information
              result["num_dims"] = map.getNumDims();
              result["num_symbols"] = map.getNumSymbols();
              result["num_results"] = map.getNumResults();

              // Get string representation of the map
              std::string mapStr;
              llvm::raw_string_ostream stream(mapStr);
              map.print(stream);
              result["map_str"] = mapStr;

              // Check if it's a constant map (e.g., affine_map<() -> (0)>)
              if (map.isConstant()) {
                std::vector<int64_t> constants;
                for (auto expr : map.getResults()) {
                  if (auto constExpr =
                          llvm::dyn_cast<mlir::AffineConstantExpr>(expr)) {
                    constants.push_back(constExpr.getValue());
                  }
                }
                result["constants"] = constants;
                result["is_constant"] = true;
              } else {
                result["is_constant"] = false;
              }

              return nb::cast(result);
            } else if (auto denseI32ArrayAttr =
                           llvm::dyn_cast<mlir::DenseI32ArrayAttr>(attr)) {
              // Special handling for DenseI32ArrayAttr (used by
              // operandSegmentSizes in affine.for)
              std::vector<int32_t> values;
              for (auto val : denseI32ArrayAttr.asArrayRef()) {
                values.push_back(val);
              }
              return nb::cast(values);
            } else if (auto denseI64ArrayAttr =
                           llvm::dyn_cast<mlir::DenseI64ArrayAttr>(attr)) {
              // Special handling for DenseI64ArrayAttr
              std::vector<int64_t> values;
              for (auto val : denseI64ArrayAttr.asArrayRef()) {
                values.push_back(val);
              }
              return nb::cast(values);
            } else if (auto denseIntAttr =
                           llvm::dyn_cast<mlir::DenseIntElementsAttr>(attr)) {
              // Special handling for DenseIntElementsAttr
              std::vector<int64_t> values;
              for (auto val : denseIntAttr.getValues<int64_t>()) {
                values.push_back(val);
              }
              return nb::cast(values);
            } else if (auto denseAttr =
                           llvm::dyn_cast<mlir::DenseElementsAttr>(attr)) {
              // Handle general DenseElementsAttr
              if (denseAttr.getElementType().isInteger(32)) {
                std::vector<int32_t> values;
                for (auto val : denseAttr.getValues<int32_t>()) {
                  values.push_back(val);
                }
                return nb::cast(values);
              } else {
                // Return string representation for other dense types
                std::string result;
                llvm::raw_string_ostream stream(result);
                attr.print(stream);
                return nb::cast(result);
              }
            } else {
              // Return string representation for other types
              std::string result;
              llvm::raw_string_ostream stream(result);
              attr.print(stream);
              return nb::cast(result);
            }
          },
          "Get attribute value by name")
      .def(
          "has_attr",
          [](const megg::MLIROperationRef &ref, const std::string &name) {
            if (!ref.valid())
              return false;
            return ref.get()->hasAttr(name);
          },
          "Check if operation has an attribute")
      .def(
          "get_attr_names",
          [](const megg::MLIROperationRef &ref) {
            std::vector<std::string> names;
            if (!ref.valid())
              return names;
            for (auto &namedAttr : ref.get()->getAttrs()) {
              names.push_back(namedAttr.getName().str());
            }
            return names;
          },
          "Get all attribute names")
      .def(
          "get_scf_loop_bounds",
          [](const megg::MLIROperationRef &ref) -> nb::object {
            if (!ref.valid())
              return nb::none();

            // Check if this is an scf.for operation
            auto forOp = llvm::dyn_cast<mlir::scf::ForOp>(ref.get());
            if (!forOp)
              return nb::none();

            // Helper to get constant index value
            auto getConstantIndex =
                [](mlir::Value val) -> std::optional<int64_t> {
              if (auto constOp =
                      val.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
                return constOp.value();
              }
              if (auto constOp =
                      val.getDefiningOp<mlir::arith::ConstantIntOp>()) {
                return constOp.value();
              }
              return std::nullopt;
            };

            // Get lower bound
            auto lowerOpt = getConstantIndex(forOp.getLowerBound());
            if (!lowerOpt)
              return nb::none();

            // Get upper bound
            auto upperOpt = getConstantIndex(forOp.getUpperBound());
            if (!upperOpt)
              return nb::none();

            // Get step
            std::optional<int64_t> stepOpt;
            if (auto constantStep = forOp.getConstantStep()) {
              stepOpt = constantStep->getSExtValue();
            } else {
              stepOpt = getConstantIndex(forOp.getStep());
            }

            if (!stepOpt)
              return nb::none();

            // Return as dictionary with lower, upper, step
            nb::dict result;
            result["lower"] = *lowerOpt;
            result["upper"] = *upperOpt;
            result["step"] = *stepOpt;
            return result;
          },
          "Get loop bounds for SCF for loop as dict with 'lower', 'upper', "
          "'step' keys (returns None if not constant or not scf.for)")
      .def(
          "get_function_type",
          [](const megg::MLIROperationRef &ref) -> nb::object {
            if (!ref.valid())
              return nb::none();
            // Check if this is a function operation
            if (auto funcOp = llvm::dyn_cast<mlir::func::FuncOp>(ref.get())) {
              // Return the function type directly
              return nb::cast(funcOp.getFunctionType());
            }
            return nb::none();
          },
          "Get FunctionType if this is a function operation")
      .def(
          "cast_to_module",
          [](const megg::MLIROperationRef &ref) -> megg::MLIRModule* {
            if (!ref.valid())
              return nullptr;

            // Check if this is a module operation
            auto moduleOp = llvm::dyn_cast<mlir::ModuleOp>(ref.get());
            if (!moduleOp)
              return nullptr;

            // Create a new MLIRModule wrapper
            auto* module = new megg::MLIRModule();
            module->setModule(moduleOp);
            return module;
          },
          nb::rv_policy::take_ownership,
          "Cast this operation to MLIRModule if it's a module operation (returns None if not a module)")
      .def(
          "clone_with_mapping",
          [](const megg::MLIROperationRef &ref) -> nb::object {
            if (!ref.valid())
              return nb::none();

            // Clone the operation with IRMapping
            mlir::IRMapping mapper;
            auto* clonedOp = ref.get()->clone(mapper);

            // Build operation map by walking both operations
            std::unordered_map<mlir::Operation*, mlir::Operation*> opMap;

            // Walk the original operation and its nested ops
            ref.get()->walk([&](mlir::Operation *originalOp) {
              if (auto mappedOp = mapper.lookupOrNull(originalOp)) {
                opMap[originalOp] = mappedOp;
              }
            });

            // Convert the C++ map to a Python dict
            nb::dict pyMap;
            for (const auto &[oldOp, newOp] : opMap) {
              pyMap[nb::cast(megg::MLIROperationRef(oldOp))] =
                  nb::cast(megg::MLIROperationRef(newOp));
            }

            // Return tuple of (cloned_op, operation_map)
            return nb::make_tuple(
                nb::cast(megg::MLIROperationRef(clonedOp)), pyMap);
          },
          "Clone this operation and return (cloned_op, operation_map) mapping original ops to cloned ops")
      .def(
          "__hash__",
          [](const megg::MLIROperationRef &ref) -> size_t {
            // Hash based on the underlying C++ pointer value
            return std::hash<mlir::Operation*>{}(ref.get());
          },
          "Hash based on underlying operation pointer")
      .def(
          "__eq__",
          [](const megg::MLIROperationRef &ref, const megg::MLIROperationRef &other) -> bool {
            // Compare based on the underlying C++ pointer
            return ref.get() == other.get();
          },
          "Compare based on underlying operation pointer")
      .def(
          "get_ptr",
          [](const megg::MLIROperationRef &ref) -> uintptr_t {
            // Return pointer value as integer for debugging
            return reinterpret_cast<uintptr_t>(ref.get());
          },
          "Get pointer value as integer for debugging");

  // MLIRModule class bindings
  nb::class_<megg::MLIRModule>(m, "MLIRModule")
      .def(nb::init<>())
      .def("parse_from_string", &megg::MLIRModule::parseFromString,
           "mlir_text"_a, "apply_inlining"_a = true,
           "Parse MLIR module from string, optionally apply inlining")
      .def("parse_from_file", &megg::MLIRModule::parseFromFile, "filepath"_a,
           "apply_inlining"_a = true,
           "Parse MLIR module from file, optionally apply inlining")
      .def("to_string", &megg::MLIRModule::toString,
           "Get the module as MLIR text string")
      .def("get_operations", &megg::MLIRModule::getOperations,
           nb::rv_policy::reference_internal,
           "Get all operations in the module as a list")
      .def("get_functions", &megg::MLIRModule::getFunctions,
           nb::rv_policy::reference_internal,
           "Get all function operations in the module")
      .def("get_context", &megg::MLIRModule::getContext,
           nb::rv_policy::reference_internal, "Get the MLIR context")
      .def("get_module", &megg::MLIRModule::getModule,
           nb::rv_policy::reference_internal, "Get the MLIR module operation")
      .def(
          "append_to_module",
          [](megg::MLIRModule &self, nb::object op) {
            // Support both raw mlir::Operation* and MLIROperationRef
            if (nb::isinstance<megg::MLIROperationRef>(op)) {
              auto ref = nb::cast<megg::MLIROperationRef>(op);
              self.appendToModule(ref.get());
            } else if (nb::isinstance<mlir::Operation *>(op)) {
              self.appendToModule(nb::cast<mlir::Operation *>(op));
            } else {
              throw std::runtime_error("append_to_module expects "
                                       "MLIROperationRef or mlir::Operation*");
            }
          },
          "op"_a, "Append operation to module body")
      .def("apply_inlining", &megg::MLIRModule::applyInlining,
           "Apply MLIR inliner pass to inline function calls")
      .def("validate", &megg::MLIRModule::validate,
           nb::rv_policy::reference_internal, "check module is valide")
      .def(
          "clone_with_mapping",
          [](megg::MLIRModule &self) -> nb::object {
            auto [clonedModule, opMap] = self.cloneWithMapping();

            if (!clonedModule) {
              return nb::none();
            }

            // Convert the C++ map to a Python dict
            nb::dict pyMap;
            for (const auto &[oldOp, newOp] : opMap) {
              //   std::cerr << oldOp << "\n" << newOp << "\n";
              pyMap[nb::cast(megg::MLIROperationRef(oldOp))] =
                  nb::cast(megg::MLIROperationRef(newOp));
            }

            // Release ownership from unique_ptr and let nanobind manage it
            auto *rawModule = clonedModule.release();

            // Return tuple of (cloned_module, operation_map)
            return nb::make_tuple(
                nb::cast(rawModule, nb::rv_policy::take_ownership), pyMap);
          },
          "Clone the module and return (cloned_module, operation_map) where "
          "operation_map "
          "maps old operations to their clones");

  // PassManager class bindings
  nb::class_<megg::PassManager>(m, "PassManager")
      .def(nb::init<megg::MLIRModule &>(), "mlir_module"_a,
           "Create a pass manager for the given MLIR module")
      .def(
          "register_affine_loop_unroll_passes",
          [](megg::PassManager &pm, const megg::MLIROperationRef *functionOp,
             const megg::MLIROperationRef *loopOp, uint64_t unrollFactor) {
            mlir::Operation *funcPtr = functionOp ? functionOp->get() : nullptr;
            mlir::Operation *loopPtr = loopOp ? loopOp->get() : nullptr;
            pm.registerAffineLoopUnrollPasses(funcPtr, loopPtr, unrollFactor);
          },
          "function_op"_a = nullptr, "loop_op"_a = nullptr,
          "unroll_factor"_a = 4,
          "Register affine loop unrolling passes targeting a specific loop "
          "operation")
      .def(
          "register_affine_loop_unroll_jam_passes",
          [](megg::PassManager &pm, const megg::MLIROperationRef *functionOp,
             const megg::MLIROperationRef *loopOp, uint64_t unrollJamFactor) {
            mlir::Operation *funcPtr = functionOp ? functionOp->get() : nullptr;
            mlir::Operation *loopPtr = loopOp ? loopOp->get() : nullptr;
            pm.registerAffineLoopUnrollJamPasses(funcPtr, loopPtr,
                                                 unrollJamFactor);
          },
          "function_op"_a = nullptr, "loop_op"_a = nullptr,
          "unroll_jam_factor"_a = 4,
          "Register affine loop unroll-and-jam passes targeting a specific "
          "loop operation")
      .def(
          "register_affine_loop_tiling_pass",
          [](megg::PassManager &pm, const megg::MLIROperationRef *functionOp,
             const megg::MLIROperationRef *loopOp, uint64_t defaultTileSize) {
            mlir::Operation *funcPtr = functionOp ? functionOp->get() : nullptr;
            mlir::Operation *loopPtr = loopOp ? loopOp->get() : nullptr;
            pm.registerAffineLoopTilingPass(funcPtr, loopPtr, defaultTileSize);
          },
          "function_op"_a = nullptr, "loop_op"_a = nullptr,
          "default_tile_size"_a = 32,
          "Register affine loop tiling pass targeting an optional loop "
          "operation")
      .def("run_pass", &megg::PassManager::runPass, "pass_name"_a,
           "Run a specific registered pass by name")
      .def("run_all_pass", &megg::PassManager::runAllPass,
           "run all registerd passes")
      .def("verify_module", &megg::PassManager::verifyModule,
           "Verify the MLIR module is valid")
      .def("clear_custom_passes", &megg::PassManager::clearCustomPasses,
           "Clear all registered custom passes");

  // InsertPoint opaque type
  nb::class_<mlir::OpBuilder::InsertPoint>(m, "InsertPoint")
      .def(nb::init<>(), "Create an invalid insertion point");

  // IR Builder class bindings
  nb::class_<megg::MLIRIRBuilder>(m, "MLIRIRBuilder")
      .def(nb::init<>(), "Create a new MLIR IR builder with the given context")
      .def(
          "create_function",
          [](megg::MLIRIRBuilder &self, const std::string &name,
             const std::vector<mlir::Type> &arg_types,
             const std::vector<mlir::Type> &result_types) -> nb::tuple {
            auto [func, block] =
                self.createFunction(name, arg_types, result_types);
            // Wrap the operation pointer in MLIROperationRef
            megg::MLIROperationRef funcRef(func.getOperation());
            return nb::make_tuple(funcRef, block);
          },
          "name"_a, "arg_types"_a, "result_types"_a,
          nb::rv_policy::reference_internal,
          "Create a function with given name, argument types, and result "
          "types. Returns (func_op, entry_block)")

      // Type creation methods
      .def("get_i1_type", &megg::MLIRIRBuilder::getI1Type, "Get i1 type")
      .def("get_i32_type", &megg::MLIRIRBuilder::getI32Type, "Get i32 type")
      .def("get_i64_type", &megg::MLIRIRBuilder::getI64Type, "Get i64 type")
      .def("get_f32_type", &megg::MLIRIRBuilder::getF32Type, "Get f32 type")
      .def("get_f64_type", &megg::MLIRIRBuilder::getF64Type, "Get f64 type")
      .def("get_index_type", &megg::MLIRIRBuilder::getIndexType,
           "Get index type")
      .def("get_integer_type", &megg::MLIRIRBuilder::getIntegerType, "width"_a,
           "Get integer type with specified width")
      .def("get_memref_type", &megg::MLIRIRBuilder::getMemRefType, "shape"_a,
           "element_type"_a, "Get memref type with shape and element type")
      .def("get_function_type", &megg::MLIRIRBuilder::getFunctionType,
           "inputs"_a, "results"_a,
           "Get function type with input and result types")

      // LLVM dialect types
      .def("get_llvm_ptr_type", &megg::MLIRIRBuilder::getLLVMPtrType,
           "Get LLVM opaque pointer type (!llvm.ptr)")

      // LLVM dialect operations
      .def("create_ptrtoint", &megg::MLIRIRBuilder::createPtrToInt,
           "ptr"_a, "int_type"_a,
           "Create llvm.ptrtoint operation to convert pointer to integer")

      // Constant creation
      .def("create_constant_i32", &megg::MLIRIRBuilder::createConstantI32,
           "value"_a, "Create i32 constant")
      .def("create_constant_i64", &megg::MLIRIRBuilder::createConstantI64,
           "value"_a, "Create i64 constant")
      .def("create_constant_f32", &megg::MLIRIRBuilder::createConstantF32,
           "value"_a, "Create f32 constant")
      .def("create_constant_f64", &megg::MLIRIRBuilder::createConstantF64,
           "value"_a, "Create f64 constant")
      .def("create_constant_index", &megg::MLIRIRBuilder::createConstantIndex,
           "value"_a, "Create index constant")

      // Arithmetic operations
      .def("create_add_i", &megg::MLIRIRBuilder::createAddI, "lhs"_a, "rhs"_a,
           "Create integer addition")
      .def("create_sub_i", &megg::MLIRIRBuilder::createSubI, "lhs"_a, "rhs"_a,
           "Create integer subtraction")
      .def("create_mul_i", &megg::MLIRIRBuilder::createMulI, "lhs"_a, "rhs"_a,
           "Create integer multiplication")
      .def("create_div_si", &megg::MLIRIRBuilder::createDivSI, "lhs"_a, "rhs"_a,
           "Create signed integer division")
      .def("create_div_ui", &megg::MLIRIRBuilder::createDivUI, "lhs"_a, "rhs"_a,
           "Create unsigned integer division")
      .def("create_rem_si", &megg::MLIRIRBuilder::createRemSI, "lhs"_a, "rhs"_a,
           "Create signed integer remainder")
      .def("create_rem_ui", &megg::MLIRIRBuilder::createRemUI, "lhs"_a, "rhs"_a,
           "Create unsigned integer remainder")
      .def("create_add_f", &megg::MLIRIRBuilder::createAddF, "lhs"_a, "rhs"_a,
           "Create float addition")
      .def("create_sub_f", &megg::MLIRIRBuilder::createSubF, "lhs"_a, "rhs"_a,
           "Create float subtraction")
      .def("create_mul_f", &megg::MLIRIRBuilder::createMulF, "lhs"_a, "rhs"_a,
           "Create float multiplication")
      .def("create_div_f", &megg::MLIRIRBuilder::createDivF, "lhs"_a, "rhs"_a,
           "Create float division")
      .def("create_rem_f", &megg::MLIRIRBuilder::createRemF, "lhs"_a, "rhs"_a,
           "Create float remainder")

      // Logical operations (bitwise)
      .def("create_and_i", &megg::MLIRIRBuilder::createAndI, "lhs"_a, "rhs"_a,
           "Create bitwise AND")
      .def("create_or_i", &megg::MLIRIRBuilder::createOrI, "lhs"_a, "rhs"_a,
           "Create bitwise OR")
      .def("create_xor_i", &megg::MLIRIRBuilder::createXorI, "lhs"_a, "rhs"_a,
           "Create bitwise XOR")

      // Shift operations
      .def("create_shl_i", &megg::MLIRIRBuilder::createShlI, "lhs"_a, "rhs"_a,
           "Create shift left")
      .def("create_shrsi_i", &megg::MLIRIRBuilder::createShrSII, "lhs"_a,
           "rhs"_a, "Create shift right signed")
      .def("create_shrui_i", &megg::MLIRIRBuilder::createShrUII, "lhs"_a,
           "rhs"_a, "Create shift right unsigned")

      // Comparison operations
      .def("create_cmpi", &megg::MLIRIRBuilder::createCmpI, "predicate"_a,
           "lhs"_a, "rhs"_a, "Create integer comparison")
      .def("create_cmpf", &megg::MLIRIRBuilder::createCmpF, "predicate"_a,
           "lhs"_a, "rhs"_a, "Create float comparison")

      // Select operation (ternary conditional)
      .def("create_select", &megg::MLIRIRBuilder::createSelect, "condition"_a,
           "true_value"_a, "false_value"_a,
           "Create select (ternary conditional)")

      // Cast operations
      .def("create_index_cast", &megg::MLIRIRBuilder::createIndexCast,
           "value"_a, "target_type"_a, "Create index cast")
      .def("create_si_to_fp", &megg::MLIRIRBuilder::createSIToFP, "value"_a,
           "target_type"_a, "Create signed int to float cast")
      .def("create_fp_to_si", &megg::MLIRIRBuilder::createFPToSI, "value"_a,
           "target_type"_a, "Create float to signed int cast")
      .def("create_fp_to_ui", &megg::MLIRIRBuilder::createFPToUI, "value"_a,
           "target_type"_a, "Create float to unsigned int cast")
      .def("create_ui_to_fp", &megg::MLIRIRBuilder::createUIToFP, "value"_a,
           "target_type"_a, "Create unsigned int to float cast")
      .def("create_ext_si", &megg::MLIRIRBuilder::createExtSI, "value"_a,
           "target_type"_a, "Create sign extension")
      .def("create_ext_ui", &megg::MLIRIRBuilder::createExtUI, "value"_a,
           "target_type"_a, "Create zero extension")
      .def("create_trunc_i", &megg::MLIRIRBuilder::createTruncI, "value"_a,
           "target_type"_a, "Create integer truncation")
      .def("create_bitcast", &megg::MLIRIRBuilder::createBitcast, "value"_a,
           "target_type"_a, "Create bitcast")

      // Control flow
      .def(
          "create_return",
          [](megg::MLIRIRBuilder &self,
             const std::vector<mlir::Value> &values) -> megg::MLIROperationRef {
            return megg::MLIROperationRef(self.createReturn(values));
          },
          "values"_a = std::vector<mlir::Value>{},
          nb::rv_policy::reference_internal, "Create return operation")
      .def(
          "create_func_return",
          [](megg::MLIRIRBuilder &self,
             const std::vector<mlir::Value> &values) -> megg::MLIROperationRef {
            return megg::MLIROperationRef(self.createFuncReturn(values));
          },
          "values"_a = std::vector<mlir::Value>{},
          nb::rv_policy::reference_internal, "Create func.return operation")

      // Affine operations
      .def(
          "create_affine_for",
          [](megg::MLIRIRBuilder &self, int64_t lb, int64_t ub,
             int64_t step) -> megg::MLIROperationRef {
            return megg::MLIROperationRef(
                self.createAffineFor(lb, ub, step).getOperation());
          },
          "lower_bound"_a, "upper_bound"_a, "step"_a = 1,
          nb::rv_policy::reference_internal, "Create affine for loop")
      .def(
          "create_affine_yield",
          [](megg::MLIRIRBuilder &self,
             const std::vector<mlir::Value> &values) -> megg::MLIROperationRef {
            return megg::MLIROperationRef(self.createAffineYield(values));
          },
          "values"_a = std::vector<mlir::Value>{},
          nb::rv_policy::reference_internal, "Create affine yield")

      // SCF operations
      .def(
          "create_scf_for",
          [](megg::MLIRIRBuilder &self, mlir::Value lb, mlir::Value ub,
             mlir::Value step, const std::vector<mlir::Value> &iter_args)
              -> megg::MLIROperationRef {
            return megg::MLIROperationRef(
                self.createSCFFor(lb, ub, step, iter_args).getOperation());
          },
          "lower_bound"_a, "upper_bound"_a, "step"_a,
          "iter_args"_a = std::vector<mlir::Value>{},
          nb::rv_policy::reference_internal, "Create SCF for loop")
      .def(
          "create_scf_if",
          [](megg::MLIRIRBuilder &self, mlir::Value condition,
             const std::vector<mlir::Type> &result_types,
             bool has_else) -> megg::MLIROperationRef {
            return megg::MLIROperationRef(
                self.createSCFIf(condition, result_types, has_else)
                    .getOperation());
          },
          "condition"_a, "result_types"_a = std::vector<mlir::Type>{},
          "has_else"_a = false, nb::rv_policy::reference_internal,
          "Create SCF if operation")
      .def(
          "create_scf_while",
          [](megg::MLIRIRBuilder &self,
             const std::vector<mlir::Type> &result_types,
             const std::vector<mlir::Value> &init_vals)
              -> megg::MLIROperationRef {
            return megg::MLIROperationRef(
                self.createSCFWhile(result_types, init_vals).getOperation());
          },
          "result_types"_a, "init_vals"_a, nb::rv_policy::reference_internal,
          "Create SCF while operation")
      .def(
          "create_scf_yield",
          [](megg::MLIRIRBuilder &self,
             const std::vector<mlir::Value> &values) -> megg::MLIROperationRef {
            return megg::MLIROperationRef(self.createSCFYield(values));
          },
          "values"_a = std::vector<mlir::Value>{},
          nb::rv_policy::reference_internal, "Create SCF yield")
      .def(
          "create_scf_condition",
          [](megg::MLIRIRBuilder &self, mlir::Value condition,
             const std::vector<mlir::Value> &values) -> megg::MLIROperationRef {
            return megg::MLIROperationRef(
                self.createSCFCondition(condition, values));
          },
          "condition"_a, "values"_a = std::vector<mlir::Value>{},
          nb::rv_policy::reference_internal, "Create SCF condition")

      // Memory operations
      .def(
          "create_memref_alloc",
          [](megg::MLIRIRBuilder &self, mlir::Type type) {
            // Cast the generic Type to MemRefType
            auto memrefType = llvm::cast<mlir::MemRefType>(type);
            return self.createMemRefAlloc(memrefType);
          },
          "type"_a, "Create memref allocation (heap)")
      .def(
          "create_memref_alloca",
          [](megg::MLIRIRBuilder &self, mlir::Type type) {
            // Cast the generic Type to MemRefType
            auto memrefType = llvm::cast<mlir::MemRefType>(type);
            return self.createMemRefAlloca(memrefType);
          },
          "type"_a, "Create memref allocation (stack/alloca)")
      .def("create_memref_store", &megg::MLIRIRBuilder::createMemRefStore,
           "value"_a, "memref"_a, "indices"_a, "Store value to memref")
      .def("create_memref_load", &megg::MLIRIRBuilder::createMemRefLoad,
           "memref"_a, "indices"_a, "Load value from memref")
      .def("create_memref_get_global",
           &megg::MLIRIRBuilder::createMemRefGetGlobal, "memref_type"_a,
           "symbol_name"_a, "Get global memref")
      .def(
          "create_memref_extract_aligned_pointer_as_index",
          &megg::MLIRIRBuilder::createMemRefExtractAlignedPointerAsIndex,
          "memref"_a,
          "Expose memref.extract_aligned_pointer_as_index for pointer lowering")

      // LLVM operations
      .def("create_inline_asm", &megg::MLIRIRBuilder::createInlineAsm,
           "asm_string"_a, "constraints"_a, "operands"_a, "result_type"_a,
           "has_side_effects"_a = false,
           "Create LLVM inline assembly operation")

      // Insertion point control
      .def(
          "set_insertion_point_to_start_func",
          [](megg::MLIRIRBuilder &self, const megg::MLIROperationRef &ref) {
            if (!ref.valid())
              return;
            if (auto funcOp = llvm::dyn_cast<mlir::func::FuncOp>(ref.get())) {
              self.setInsertionPointToStart(funcOp);
            }
          },
          "func"_a, "Set insertion point to start of function")
      .def(
          "set_insertion_point_to_end_func",
          [](megg::MLIRIRBuilder &self, const megg::MLIROperationRef &ref) {
            if (!ref.valid())
              return;
            if (auto funcOp = llvm::dyn_cast<mlir::func::FuncOp>(ref.get())) {
              self.setInsertionPointToEnd(funcOp);
            }
          },
          "func"_a, "Set insertion point to end of function")
      .def(
          "set_insertion_point",
          [](megg::MLIRIRBuilder &self, const megg::MLIROperationRef &ref) {
            if (!ref.valid())
              return;
            self.setInsertionPoint(ref.get());
          },
          "op"_a, "Set insertion point before operation")
      .def(
          "set_insertion_point_after",
          [](megg::MLIRIRBuilder &self, const megg::MLIROperationRef &ref) {
            if (!ref.valid())
              return;
            self.setInsertionPointAfter(ref.get());
          },
          "op"_a, "Set insertion point after operation")

      // Block operations
      .def("create_block", &megg::MLIRIRBuilder::createBlock, "region"_a,
           "arg_types"_a = std::vector<mlir::Type>{},
           nb::rv_policy::reference_internal,
           "Create a new block in the region")
      .def(
          "set_insertion_point_to_start_block",
          [](megg::MLIRIRBuilder &self, mlir::Block *block) {
            self.setInsertionPointToStart(block);
          },
          "block"_a, "Set insertion point to start of block")
      .def(
          "set_insertion_point_to_end_block",
          [](megg::MLIRIRBuilder &self, mlir::Block *block) {
            self.setInsertionPointToEnd(block);
          },
          "block"_a, "Set insertion point to end of block")
      .def("save_insertion_point", &megg::MLIRIRBuilder::saveInsertionPoint,
           "Save current insertion point")
      .def("restore_insertion_point",
           &megg::MLIRIRBuilder::restoreInsertionPoint, "ip"_a,
           "Restore saved insertion point")
      .def("set_context", &megg::MLIRIRBuilder::setContext, "newcontext"_a,
           nb::rv_policy::reference_internal);
}
