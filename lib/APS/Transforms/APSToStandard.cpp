#include "APS/Transforms/APSToStandard.h"
#include "APS/APSOps.h"
#include "APS/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <functional>
#include <optional>

using namespace mlir;

namespace aps_dialect = ::aps;

namespace {

static constexpr unsigned kCpuMemorySpace = 0;

static FailureOr<Value> materializeIndex(Value value, Location loc,
                                         OpBuilder &builder) {
  if (llvm::isa<IndexType>(value.getType()))
    return value;

  if (auto intTy = llvm::dyn_cast<IntegerType>(value.getType())) {
    auto cast =
        builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), value);
    return cast.getResult();
  }

  return failure();
}

static LogicalResult mergeElementType(std::optional<Type> &target,
                                      Type candidate, Operation *diagOp,
                                      func::FuncOp context) {
  if (!target) {
    target = candidate;
    return success();
  }

  // Note: CPU memory is byte-addressable, so different element types
  // for different burst operations is perfectly valid. We don't enforce
  // type consistency here.
  // The type stored in 'target' is used for determining function parameter
  // types, but each memref can have its own element type.

  return success();
}

static LogicalResult mergeElementType(std::optional<Type> &target,
                                      std::optional<Type> candidate,
                                      func::FuncOp context) {
  if (!candidate)
    return success();
  if (!target) {
    target = candidate;
    return success();
  }
  if (*target != *candidate) {
    context.emitError("CPU memory element types differ across calls");
    return failure();
  }
  return success();
}

static FailureOr<std::optional<Type>>
computeLocalCpuElementType(func::FuncOp func) {
  std::optional<Type> elementType;
  WalkResult walk = func.walk([&](Operation *op) {
    if (auto burst = dyn_cast<aps_dialect::MemBurstLoad>(op)) {
      // Get first memref from variadic list
      if (burst.getMemrefs().empty()) {
        burst.emitError("expected at least one memref for burst destination");
        return WalkResult::interrupt();
      }
      auto memType = llvm::dyn_cast<MemRefType>(burst.getMemrefs()[0].getType());
      if (!memType) {
        burst.emitError("expected memref type for burst destination");
        return WalkResult::interrupt();
      }
      if (memType.getRank() != 1) {
        burst.emitError(
            "APSToStandard currently supports only 1-D memrefs for bursts");
        return WalkResult::interrupt();
      }
      if (failed(mergeElementType(elementType, memType.getElementType(), burst,
                                  func)))
        return WalkResult::interrupt();
    } else if (auto burst = dyn_cast<aps_dialect::MemBurstStore>(op)) {
      // Get first memref from variadic list
      if (burst.getMemrefs().empty()) {
        burst.emitError("expected at least one memref for burst destination");
        return WalkResult::interrupt();
      }
      auto memType = llvm::dyn_cast<MemRefType>(burst.getMemrefs()[0].getType());
      if (!memType) {
        burst.emitError("expected memref type for burst source");
        return WalkResult::interrupt();
      }
      if (memType.getRank() != 1) {
        burst.emitError(
            "APSToStandard currently supports only 1-D memrefs for bursts");
        return WalkResult::interrupt();
      }
      if (failed(mergeElementType(elementType, memType.getElementType(), burst,
                                  func)))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (walk.wasInterrupted())
    return failure();
  return elementType;
}

static constexpr StringLiteral kCpuMemArgAttr("aps.cpu_mem");

static std::optional<unsigned> findCpuMemArgIndex(func::FuncOp func) {
  for (auto [idx, arg] : llvm::enumerate(func.getArguments())) {
    if (func.getArgAttr(idx, kCpuMemArgAttr))
      return static_cast<unsigned>(idx);
  }
  return std::nullopt;
}

static FailureOr<unsigned> ensureCpuMemArgument(func::FuncOp func,
                                                Type elementType) {
  MLIRContext *ctx = func.getContext();
  auto memRefType = MemRefType::get(
      {ShapedType::kDynamic}, elementType, nullptr,
      IntegerAttr::get(IntegerType::get(ctx, 64), kCpuMemorySpace));

  if (auto existingIndex = findCpuMemArgIndex(func)) {
    auto existingType =
        llvm::dyn_cast<MemRefType>(func.getArgument(*existingIndex).getType());
    if (!existingType)
      return func.emitError("expected CPU memory argument to have memref type"),
             failure();
    if (existingType != memRefType)
      return func.emitError("existing CPU memory argument type mismatch"),
             failure();
    return *existingIndex;
  }

  for (auto [idx, arg] : llvm::enumerate(func.getArguments())) {
    if (arg.getType() == memRefType) {
      func.setArgAttr(idx, StringAttr::get(ctx, kCpuMemArgAttr),
                      UnitAttr::get(ctx));
      return static_cast<unsigned>(idx);
    }
  }

  unsigned insertIndex = func.getNumArguments();
  NamedAttrList argAttrs;
  argAttrs.append(StringAttr::get(ctx, kCpuMemArgAttr), UnitAttr::get(ctx));
  DictionaryAttr attrs = argAttrs.getDictionary(ctx);
  if (failed(
          func.insertArgument(insertIndex, memRefType, attrs, func.getLoc())))
    return failure();
  return insertIndex;
}

} // namespace

llvm::StringRef mlir::aps::APSToStandardPass::getArgument() const {
  return "aps-to-standard";
}

llvm::StringRef mlir::aps::APSToStandardPass::getDescription() const {
  return "Convert APS dialect operations to standard MLIR dialects";
}

void mlir::aps::APSToStandardPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry
      .insert<arith::ArithDialect, memref::MemRefDialect, scf::SCFDialect>();
}

/// Simplify function by removing hardware-specific operations
/// This implements Phase 1 (partial) and Phase 2 of the transformation
mlir::LogicalResult mlir::aps::APSToStandardPass::simplifyFunction(
    mlir::func::FuncOp func, SmallVector<aps_dialect::CpuRfRead> &rfReads,
    SmallVector<aps_dialect::CpuRfWrite> &rfWrites,
    SmallVector<aps_dialect::MemBurstLoad> &burstLoads,
    SmallVector<aps_dialect::MemBurstStore> &burstStores,
    std::optional<unsigned> cpuMemArgIndex,
    const llvm::DenseMap<StringRef, GlobalMemrefInfo> &requiredGlobals,
    llvm::DenseMap<StringRef, unsigned> &globalArgIndices) {
  OpBuilder builder(func);
  MLIRContext *ctx = func.getContext();
  unsigned insertIndex = func.getNumArguments();

  // ============================================================================
  // Phase 1: Add Global Variable Parameters
  // ============================================================================
  // Add function parameters for all required globals (including those needed by callees)
  // This transforms: memref.get_global @mem_a -> function parameter %mem_a
  llvm::DenseMap<StringRef, BlockArgument> globalToArg;

  // Add parameters for all required globals in a deterministic order
  SmallVector<StringRef> sortedGlobals;
  for (auto &entry : requiredGlobals) {
    sortedGlobals.push_back(entry.first);
  }
  llvm::sort(sortedGlobals);

  for (StringRef globalName : sortedGlobals) {
    auto it = requiredGlobals.find(globalName);
    if (it == requiredGlobals.end()) {
      return func.emitError("global not found in requiredGlobals"), failure();
    }
    auto memrefType = it->second.type;
    auto loc = it->second.loc;

    // Add parameter with attribute marking it as a global memref
    NamedAttrList attrs;
    attrs.append(StringAttr::get(ctx, "aps.global_memref"),
                 StringAttr::get(ctx, globalName));
    DictionaryAttr argAttrs = attrs.getDictionary(ctx);

    if (failed(func.insertArgument(insertIndex, memrefType, argAttrs, loc))) {
      return failure();
    }

    BlockArgument newArg = func.getArgument(insertIndex);
    globalToArg[globalName] = newArg;
    globalArgIndices[globalName] = insertIndex;
    insertIndex++;
  }

  // ============================================================================
  // Phase 2: Replace memref.get_global with Parameters
  // ============================================================================
  SmallVector<memref::GetGlobalOp> getGlobals;
  func.walk([&](memref::GetGlobalOp getGlobal) {
    getGlobals.push_back(getGlobal);
  });

  for (auto getGlobal : getGlobals) {
    StringRef globalName = getGlobal.getName();
    auto it = globalToArg.find(globalName);
    if (it != globalToArg.end()) {
      getGlobal.getResult().replaceAllUsesWith(it->second);
      getGlobal.erase();
    } else {
      return getGlobal.emitError("global '" + globalName + "' not found in required globals"), failure();
    }
  }

  // ============================================================================
  // Phase 1 & 2: Process aps.readrf Operations
  // ============================================================================
  // For readrf used in burst operations, infer array type from scratchpad memref
  // For other readrf, keep as scalar type
  // Add parameters and replace readrf operations
  llvm::DenseMap<Value, BlockArgument> rfReadToArg;

  for (auto read : rfReads) {
    // Check if this readrf result is ONLY used in burst operations
    // If it has any non-burst uses (like bit extraction), keep it as scalar
    MemRefType arrayType;
    bool hasOnlyBurstUses = true;
    bool hasBurstUse = false;

    for (auto &use : read.getResult().getUses()) {
      Operation *user = use.getOwner();
      if (auto burstLoad = dyn_cast<aps_dialect::MemBurstLoad>(user)) {
        // Get the scratchpad memref type (e.g., memref<16xi32>)
        if (!arrayType && !burstLoad.getMemrefs().empty()) {
          arrayType = llvm::dyn_cast<MemRefType>(burstLoad.getMemrefs()[0].getType());
        }
        hasBurstUse = true;
      } else if (auto burstStore = dyn_cast<aps_dialect::MemBurstStore>(user)) {
        // Get the scratchpad memref type
        if (!arrayType && !burstStore.getMemrefs().empty()) {
          arrayType = llvm::dyn_cast<MemRefType>(burstStore.getMemrefs()[0].getType());
        }
        hasBurstUse = true;
      } else {
        // Non-burst use detected (e.g., bit extraction, arithmetic)
        hasOnlyBurstUses = false;
      }
    }

    Type paramType;
    NamedAttrList attrs;

    // Only infer array type if:
    // 1. Has at least one burst use AND
    // 2. Has ONLY burst uses (no other operations like bit extraction)
    if (arrayType && hasBurstUse && hasOnlyBurstUses) {
      // This readrf represents an array pointer
      // Use the same type as the scratchpad (e.g., memref<16xi32>)
      paramType = arrayType;
      attrs.append(StringAttr::get(ctx, "aps.array_ptr"), UnitAttr::get(ctx));
    } else {
      // Scalar value, keep original type
      // This handles: pure scalars, mixed uses, or non-burst uses
      paramType = read.getResult().getType();
    }

    // Add the parameter
    DictionaryAttr argAttrs = attrs.getDictionary(ctx);
    if (failed(func.insertArgument(insertIndex, paramType, argAttrs, read.getLoc()))) {
      return failure();
    }
    BlockArgument newArg = func.getArgument(insertIndex);
    rfReadToArg[read.getResult()] = newArg;
    insertIndex++;
  }

  // Replace all uses of readrf results with new parameters
  for (auto read : rfReads) {
    auto it = rfReadToArg.find(read.getResult());
    if (it != rfReadToArg.end()) {
      read.getResult().replaceAllUsesWith(it->second);
    }
  }

  // Erase all readrf operations (replaced by parameters)
  for (auto read : rfReads) {
    read.erase();
  }

  // ============================================================================
  // Phase 2: Delete Burst Operations
  // ============================================================================
  // Burst operations only provided type hints for readrf inference
  // In high-level semantics, arrays are passed as pointers - no DMA transfers
  for (auto burstLoad : burstLoads) {
    burstLoad.erase();
  }
  for (auto burstStore : burstStores) {
    burstStore.erase();
  }

  // Clear vectors
  rfReads.clear();
  burstLoads.clear();
  burstStores.clear();

  // Note: Function signature is already updated by insertArgument() calls above

  // ============================================================================
  // Phase 4: Dead Code Elimination
  // ============================================================================
  // Remove operations that have no users after deleting burst operations
  // This includes arith.constant for addresses, unused index calculations, etc.
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> toErase;

    func.walk([&](Operation *op) {
      // Skip terminator operations and operations with side effects on memory
      if (op->hasTrait<OpTrait::IsTerminator>() ||
          isa<func::FuncOp>(op) ||
          isa<memref::StoreOp>(op) ||
          isa<aps_dialect::MemStore>(op) ||
          isa<aps_dialect::CpuRfWrite>(op)) {
        return;
      }

      // Check if the operation is trivially dead (no users and no side effects)
      if (isOpTriviallyDead(op)) {
        toErase.push_back(op);
        changed = true;
      }
    });

    for (auto *op : toErase) {
      op->erase();
    }
  }

  return success();
}

/// Lower APS operations to standard MLIR dialects
/// This implements Phase 2 (operation lowering) and Phase 1 (return type modification)
LogicalResult mlir::aps::APSToStandardPass::lowerFunction(
    func::FuncOp func, std::optional<unsigned> cpuMemArgIndex,
    const llvm::DenseMap<StringRef, GlobalMemrefInfo> &requiredGlobals,
    llvm::DenseMap<StringRef, unsigned> &globalArgIndices) {
  OpBuilder builder(func);

  // Collect all APS operations in the function
  SmallVector<aps_dialect::MemDeclare> memDeclares;
  SmallVector<aps_dialect::MemLoad> memLoads;
  SmallVector<aps_dialect::MemStore> memStores;
  SmallVector<aps_dialect::MemBurstLoad> burstLoads;
  SmallVector<aps_dialect::MemBurstStore> burstStores;
  SmallVector<aps_dialect::CpuRfRead> rfReads;
  SmallVector<aps_dialect::CpuRfWrite> rfWrites;

  func.walk([&](Operation *op) {
    if (auto declare = dyn_cast<aps_dialect::MemDeclare>(op)) {
      memDeclares.push_back(declare);
    } else if (auto load = dyn_cast<aps_dialect::MemLoad>(op)) {
      memLoads.push_back(load);
    } else if (auto store = dyn_cast<aps_dialect::MemStore>(op)) {
      memStores.push_back(store);
    } else if (auto burstLoad = dyn_cast<aps_dialect::MemBurstLoad>(op)) {
      burstLoads.push_back(burstLoad);
    } else if (auto burstStore = dyn_cast<aps_dialect::MemBurstStore>(op)) {
      burstStores.push_back(burstStore);
    } else if (auto read = dyn_cast<aps_dialect::CpuRfRead>(op)) {
      rfReads.push_back(read);
    } else if (auto write = dyn_cast<aps_dialect::CpuRfWrite>(op)) {
      rfWrites.push_back(write);
    }
  });

  // ============================================================================
  // Phase 2: Lower aps.memdeclare -> memref.alloc
  // ============================================================================
  for (auto declare : memDeclares) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(declare);

    auto memType = llvm::dyn_cast<MemRefType>(declare.getResult().getType());
    if (!memType) {
      declare.emitError("expected memref type for aps.memdeclare result");
      return failure();
    }

    if (!memType.hasStaticShape()) {
      declare.emitError(
          "dynamic memref shapes are not supported in aps.memdeclare yet");
      return failure();
    }

    auto alloc = builder.create<memref::AllocOp>(declare.getLoc(), memType);
    declare.getResult().replaceAllUsesWith(alloc);
    declare.erase();
  }

  // ============================================================================
  // Phase 2: Lower aps.memload -> memref.load
  // ============================================================================
  for (auto load : memLoads) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(load);

    SmallVector<Value> indices;
    indices.reserve(load.getIndices().size());
    for (Value index : load.getIndices()) {
      auto casted = materializeIndex(index, load.getLoc(), builder);
      if (failed(casted)) {
        load.emitError("expected integer or index type for memload index");
        return failure();
      }
      indices.push_back(*casted);
    }

    auto stdLoad = builder.create<memref::LoadOp>(load.getLoc(),
                                                  load.getMemref(), indices);
    load.replaceAllUsesWith(stdLoad.getResult());
    load.erase();
  }

  // ============================================================================
  // Phase 2: Lower aps.memstore -> memref.store
  // ============================================================================
  for (auto store : memStores) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(store);

    SmallVector<Value> indices;
    indices.reserve(store.getIndices().size());
    for (Value index : store.getIndices()) {
      auto casted = materializeIndex(index, store.getLoc(), builder);
      if (failed(casted)) {
        store.emitError("expected integer or index type for memstore index");
        return failure();
      }
      indices.push_back(*casted);
    }

    builder.create<memref::StoreOp>(store.getLoc(), store.getValue(),
                                    store.getMemref(), indices);
    store.erase();
  }

  // ============================================================================
  // Phase 1 & 2: Simplify Function (Remove Hardware Info)
  // ============================================================================
  // This handles:
  // - Adding global variable parameters
  // - Processing readrf (adding parameters based on type inference)
  // - Deleting burst operations
  // - Dead code elimination
  if (failed(simplifyFunction(func, rfReads, rfWrites, burstLoads, burstStores, cpuMemArgIndex, requiredGlobals, globalArgIndices))) {
    return failure();
  }

  // ============================================================================
  // Phase 1 & 2: Process aps.writerf -> Return Values
  // ============================================================================
  // Transform: aps.writerf %rd, %value -> return %value
  // Hardware semantics: Write to register file
  // Software semantics: Function return value
  SmallVector<Value> writerValues;
  writerValues.reserve(rfWrites.size());
  for (auto write : rfWrites) {
    // Debug: Check the actual type before calling getValue()
    Value valueOperand = write->getOperand(1);  // Second operand is the value
    llvm::errs() << "DEBUG: writerf value operand type: ";
    valueOperand.getType().print(llvm::errs());
    llvm::errs() << "\n";

    writerValues.push_back(valueOperand);
    write.erase();
  }

  if (!writerValues.empty()) {
    bool hasReturn = false;
    func.walk([&](func::ReturnOp ret) {
      hasReturn = true;
      SmallVector<Value> newOperands(ret.getOperands().begin(),
                                     ret.getOperands().end());
      newOperands.append(writerValues.begin(), writerValues.end());
      ret->setOperands(newOperands);
    });

    if (!hasReturn) {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(&func.getBody().back());
      builder.create<func::ReturnOp>(func.getLoc(), writerValues);
    }

    auto funcType = func.getFunctionType();
    SmallVector<Type> inputTypes(funcType.getInputs().begin(),
                                 funcType.getInputs().end());
    SmallVector<Type> resultTypes(funcType.getResults().begin(),
                                  funcType.getResults().end());
    for (Value v : writerValues)
      resultTypes.push_back(v.getType());
    auto newType =
        FunctionType::get(func.getContext(), inputTypes, resultTypes);
    func.setFunctionType(newType);
  }

  return success();
}

LogicalResult mlir::aps::APSToStandardPass::updateCalls(
    func::FuncOp func,
    const llvm::DenseMap<func::FuncOp, std::optional<Type>> &cpuMemTypes,
    const llvm::DenseMap<func::FuncOp, unsigned> &cpuMemArgIndices,
    llvm::DenseMap<StringRef, func::FuncOp> &funcByName) {
  Value callerCpuMem;
  auto callerRequiredIt = cpuMemTypes.find(func);
  if (callerRequiredIt != cpuMemTypes.end() && callerRequiredIt->second) {
    auto callerCpuIt = cpuMemArgIndices.find(func);
    if (callerCpuIt == cpuMemArgIndices.end())
      return func.emitError(
                 "CPU memory argument index missing for function requiring it"),
             failure();
    callerCpuMem = func.getArgument(callerCpuIt->second);
  }

  SmallVector<func::CallOp> callOps;
  func.walk([&](func::CallOp call) { callOps.push_back(call); });

  for (auto call : callOps) {
    auto calleeAttr = call.getCalleeAttr();
    auto it = funcByName.find(calleeAttr.getValue());
    if (it == funcByName.end())
      continue;
    func::FuncOp callee = it->second;

    auto calleeCpuIt = cpuMemTypes.find(callee);
    if (calleeCpuIt == cpuMemTypes.end() || !calleeCpuIt->second)
      continue;

    auto calleeType = callee.getFunctionType();
    auto calleeCpuIdxIt = cpuMemArgIndices.find(callee);
    if (calleeCpuIdxIt == cpuMemArgIndices.end())
      continue;
    unsigned cpuMemIdx = calleeCpuIdxIt->second;

    bool alreadyUpdated =
        call.getNumOperands() > cpuMemIdx &&
        call.getOperand(cpuMemIdx).getType() == calleeType.getInput(cpuMemIdx);
    if (alreadyUpdated)
      continue;

    if (!callerCpuMem)
      return call.emitError(
                 "callee requires CPU memory argument but caller has none"),
             failure();

    SmallVector<Value> operands(call.getArgOperands().begin(),
                                call.getArgOperands().end());
    operands.push_back(callerCpuMem);

    OpBuilder builder(call);
    auto newCall = builder.create<func::CallOp>(
        call.getLoc(), calleeType.getResults(), calleeAttr, operands);
    for (auto [oldRes, newRes] :
         llvm::zip(call.getResults(), newCall.getResults()))
      oldRes.replaceAllUsesWith(newRes);
    for (auto namedAttr : call->getAttrs()) {
      if (namedAttr.getName() == "callee")
        continue;
      newCall->setAttr(namedAttr.getName(), namedAttr.getValue());
    }
    call.erase();
  }

  return success();
}

LogicalResult mlir::aps::APSToStandardPass::updateCallsWithGlobals(
    func::FuncOp func,
    const llvm::DenseMap<func::FuncOp, llvm::DenseMap<StringRef, unsigned>> &funcGlobalArgIndices,
    llvm::DenseMap<StringRef, func::FuncOp> &funcByName) {

  // Get the caller's global argument indices
  auto callerIt = funcGlobalArgIndices.find(func);
  if (callerIt == funcGlobalArgIndices.end()) {
    // Caller has no globals, no need to update calls
    return success();
  }
  const auto &callerGlobalIndices = callerIt->second;

  SmallVector<func::CallOp> callOps;
  func.walk([&](func::CallOp call) { callOps.push_back(call); });

  for (auto call : callOps) {
    auto calleeAttr = call.getCalleeAttr();
    auto it = funcByName.find(calleeAttr.getValue());
    if (it == funcByName.end())
      continue;
    func::FuncOp callee = it->second;

    // Get the callee's global argument indices
    auto calleeIt = funcGlobalArgIndices.find(callee);
    if (calleeIt == funcGlobalArgIndices.end() || calleeIt->second.empty())
      continue;
    const auto &calleeGlobalIndices = calleeIt->second;

    // Build new operands list: original operands + global arguments
    SmallVector<Value> newOperands(call.getArgOperands().begin(),
                                   call.getArgOperands().end());

    // For each global the callee needs, find it in the caller's arguments
    SmallVector<StringRef> sortedCalleeGlobals;
    for (auto &entry : calleeGlobalIndices) {
      sortedCalleeGlobals.push_back(entry.first);
    }
    llvm::sort(sortedCalleeGlobals);

    for (StringRef globalName : sortedCalleeGlobals) {
      auto callerIt = callerGlobalIndices.find(globalName);
      if (callerIt == callerGlobalIndices.end()) {
        return call.emitError("caller does not have required global '" + globalName.str() + "' for callee '" + calleeAttr.getValue().str() + "'"),
               failure();
      }

      // Pass the caller's global argument to the callee
      Value globalArg = func.getArgument(callerIt->second);
      newOperands.push_back(globalArg);
    }

    // Create new call with updated operands
    OpBuilder builder(call);
    auto calleeType = callee.getFunctionType();
    auto newCall = builder.create<func::CallOp>(
        call.getLoc(), calleeType.getResults(), calleeAttr, newOperands);

    // Replace results
    for (auto [oldRes, newRes] :
         llvm::zip(call.getResults(), newCall.getResults()))
      oldRes.replaceAllUsesWith(newRes);

    // Copy attributes (except callee)
    for (auto namedAttr : call->getAttrs()) {
      if (namedAttr.getName() == "callee")
        continue;
      newCall->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    call.erase();
  }

  return success();
}

void mlir::aps::APSToStandardPass::runOnOperation() {
  ModuleOp module = getOperation();

  // ============================================================================
  // Phase 0: Global Analysis
  // ============================================================================
  // Collect all functions and build analysis structures
  SmallVector<func::FuncOp> functions;
  for (auto func : module.getOps<func::FuncOp>())
    functions.push_back(func);

  llvm::DenseMap<StringRef, func::FuncOp> funcByName;
  for (auto func : functions)
    funcByName[func.getSymName()] = func;

  // Build call graph: func -> callees
  llvm::DenseMap<func::FuncOp, SmallVector<func::FuncOp>> callees;
  for (auto func : functions) {
    func.walk([&](func::CallOp call) {
      auto calleeAttr = call.getCalleeAttr();
      auto it = funcByName.find(calleeAttr.getValue());
      if (it == funcByName.end())
        return;
      callees[func].push_back(it->second);
    });
  }

  // Collect local global usage for each function
  llvm::DenseMap<func::FuncOp, llvm::DenseMap<StringRef, GlobalMemrefInfo>> localGlobals;
  for (auto func : functions) {
    llvm::DenseMap<StringRef, GlobalMemrefInfo> globals;
    func.walk([&](memref::GetGlobalOp getGlobal) {
      StringRef globalName = getGlobal.getName();
      if (auto memrefType = llvm::dyn_cast<MemRefType>(getGlobal.getResult().getType())) {
        globals.insert({globalName, GlobalMemrefInfo(memrefType, getGlobal.getLoc())});
      }
    });
    localGlobals[func] = std::move(globals);
  }

  // Analyze CPU memory types (for legacy burst operations - will be removed)
  llvm::DenseMap<func::FuncOp, std::optional<Type>> localCpuTypes;
  for (auto func : functions) {
    auto localOrErr = computeLocalCpuElementType(func);
    if (failed(localOrErr)) {
      signalPassFailure();
      return;
    }
    localCpuTypes[func] = *localOrErr;
  }

  // Build topological order (callees before callers)
  llvm::DenseSet<func::FuncOp> visited;
  llvm::DenseSet<func::FuncOp> inStack;
  SmallVector<func::FuncOp> order;
  std::function<LogicalResult(func::FuncOp)> dfs =
      [&](func::FuncOp current) -> LogicalResult {
    if (visited.contains(current))
      return success();
    if (inStack.contains(current))
      return current.emitError(
                 "recursive call graphs are not supported by APSToStandard"),
             failure();

    inStack.insert(current);
    for (auto callee : callees[current])
      if (failed(dfs(callee)))
        return failure();
    inStack.erase(current);
    visited.insert(current);
    order.push_back(current);
    return success();
  };

  for (auto func : functions)
    if (failed(dfs(func))) {
      signalPassFailure();
      return;
    }

  // Propagate global requirements through call graph (bottom-up)
  llvm::DenseMap<func::FuncOp, llvm::DenseMap<StringRef, GlobalMemrefInfo>> requiredGlobals;
  for (auto func : order) {
    // Start with local globals
    auto globals = localGlobals[func];

    // Merge in globals from all callees
    for (auto callee : callees[func]) {
      auto calleeIt = requiredGlobals.find(callee);
      if (calleeIt != requiredGlobals.end()) {
        for (auto &entry : calleeIt->second) {
          if (!globals.count(entry.first)) {
            globals.insert(entry);
          }
        }
      }
    }

    requiredGlobals[func] = std::move(globals);
  }

  llvm::DenseMap<func::FuncOp, std::optional<Type>> requiredCpuTypes;
  llvm::DenseMap<func::FuncOp, unsigned> cpuMemArgIndices;
  llvm::DenseMap<func::FuncOp, llvm::DenseMap<StringRef, unsigned>> funcGlobalArgIndices;

  // ============================================================================
  // Phase 1: Function Signature Transformation
  // Phase 2: Operation Replacement & Elimination
  // Phase 3: Call Site Update
  // ============================================================================
  // Process each function in topological order (callees before callers)
  for (auto func : order) {
    // Handle CPU memory types (legacy)
    std::optional<Type> required = localCpuTypes[func];
    bool mergeFailed = false;
    for (auto callee : callees[func]) {
      auto calleeType = requiredCpuTypes.lookup(callee);
      if (failed(mergeElementType(required, calleeType, func))) {
        mergeFailed = true;
        break;
      }
    }
    if (mergeFailed) {
      signalPassFailure();
      return;
    }

    requiredCpuTypes[func] = required;

    std::optional<unsigned> cpuArgIndex;
    if (required) {
      auto idxOrErr = ensureCpuMemArgument(func, *required);
      if (failed(idxOrErr)) {
        signalPassFailure();
        return;
      }
      cpuMemArgIndices[func] = *idxOrErr;
      cpuArgIndex = *idxOrErr;
    }

    // Phase 1 & 2: Lower function (transforms signature and replaces operations)
    llvm::DenseMap<StringRef, unsigned> globalArgIndices;
    if (failed(lowerFunction(func, cpuArgIndex, requiredGlobals[func], globalArgIndices))) {
      signalPassFailure();
      return;
    }
    funcGlobalArgIndices[func] = globalArgIndices;

    // Phase 3: Update call sites for CPU memory (legacy)
    if (failed(updateCalls(func, requiredCpuTypes, cpuMemArgIndices, funcByName))) {
      signalPassFailure();
      return;
    }

    // Phase 3: Update call sites for globals
    if (failed(updateCallsWithGlobals(func, funcGlobalArgIndices, funcByName))) {
      signalPassFailure();
      return;
    }
  }

  // ============================================================================
  // Phase 4: Dead Code Elimination (already done in simplifyFunction)
  // ============================================================================

  // ============================================================================
  // Phase 5: Module Cleanup
  // ============================================================================
  // Remove all global variable declarations (they are now function parameters)
  SmallVector<memref::GlobalOp> globalOps;
  for (auto &op : llvm::make_early_inc_range(module.getOps())) {
    if (auto globalOp = dyn_cast<memref::GlobalOp>(op)) {
      globalOps.push_back(globalOp);
    }
  }

  for (auto globalOp : globalOps) {
    globalOp.erase();
  }

  // ============================================================================
  // Phase 6: Remove Unused Function Parameters
  // ============================================================================
  // After transformation, some parameters are no longer used:
  // - Original rs1, rs2, rd parameters (replaced by readrf/writerf)
  // - cpu_mem parameter (burst operations removed)
  // - Duplicate array pointers (when readrf array ptr == global array)
  for (auto func : functions) {
    // Find unused parameters
    SmallVector<unsigned> unusedIndices;
    for (auto [idx, arg] : llvm::enumerate(func.getArguments())) {
      if (arg.use_empty()) {
        unusedIndices.push_back(idx);
      }
    }

    // Erase in reverse order to maintain indices
    for (auto it = unusedIndices.rbegin(); it != unusedIndices.rend(); ++it) {
      if (failed(func.eraseArgument(*it))) {
        // This shouldn't fail, but check anyway
        func.emitWarning("Failed to erase unused argument");
      }
    }

    // Update function type
    auto funcType = func.getFunctionType();
    SmallVector<Type> newInputTypes;
    for (auto arg : func.getArguments()) {
      newInputTypes.push_back(arg.getType());
    }
    auto newType = FunctionType::get(
      func.getContext(), newInputTypes, funcType.getResults()
    );
    func.setFunctionType(newType);
  }
}

std::unique_ptr<mlir::Pass> mlir::aps::createAPSToStandardPass() {
  return std::make_unique<APSToStandardPass>();
}

// Wrapper function in mlir namespace for tablegen-generated pass registration
std::unique_ptr<mlir::Pass> mlir::createAPSToStandardPass() {
  return mlir::aps::createAPSToStandardPass();
}
