//===- MemoryOpGenerator.cpp - Memory Operation Generator
//------------------===//
//
// This file implements the memory operation generator for TOR functions
//
//===----------------------------------------------------------------------===//

#include "APS/APSOps.h"
#include "APS/BBHandler.h"
#include "circt/Dialect/Cmt2/ECMT2/Signal.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

namespace mlir {

using namespace mlir;
using namespace mlir::tor;
using namespace circt::cmt2::ecmt2;
using namespace circt::cmt2::ecmt2::stl;
using namespace circt::firrtl;

LogicalResult MemoryOpGenerator::generateRule(
    Operation *op, mlir::OpBuilder &b, Location loc, int64_t slot,
    llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  if (auto spmLoadReq = dyn_cast<aps::SpmLoadReq>(op)) {
    return generateSpmLoadReq(spmLoadReq, b, loc, slot, localMap);
  } else if (auto spmLoadCollect = dyn_cast<aps::SpmLoadCollect>(op)) {
    return generateSpmLoadCollect(spmLoadCollect, b, loc, slot, localMap);
  } else if (auto memStore = dyn_cast<aps::MemStore>(op)) {
    return generateMemStore(memStore, b, loc, slot, localMap);
  } else if (auto burstLoadReq = dyn_cast<aps::ItfcBurstLoadReq>(op)) {
    return generateBurstLoadReq(burstLoadReq, b, loc, slot, localMap);
  } else if (auto burstLoadCollect = dyn_cast<aps::ItfcBurstLoadCollect>(op)) {
    return generateBurstLoadCollect(burstLoadCollect, b, loc, slot, localMap);
  } else if (auto burstStoreReq = dyn_cast<aps::ItfcBurstStoreReq>(op)) {
    return generateBurstStoreReq(burstStoreReq, b, loc, slot, localMap);
  } else if (auto burstStoreCollect =
                 dyn_cast<aps::ItfcBurstStoreCollect>(op)) {
    return generateBurstStoreCollect(burstStoreCollect, b, loc, slot, localMap);
  } else if (auto globalLoad = dyn_cast<aps::GlobalLoad>(op)) {
    return generateGlobalMemLoad(globalLoad, b, loc, slot, localMap);
  } else if (auto globalStore = dyn_cast<aps::GlobalStore>(op)) {
    return generateGlobalMemStore(globalStore, b, loc, slot, localMap);
  } else if (isa<memref::GetGlobalOp>(op)) {
    return success();
  }

  return failure();
}

bool MemoryOpGenerator::canHandle(Operation *op) const {
  return isa<aps::MemStore, aps::ItfcBurstLoadReq,
             aps::ItfcBurstLoadCollect, aps::ItfcBurstStoreReq,
             aps::ItfcBurstStoreCollect, memref::GetGlobalOp,
             aps::GlobalStore, aps::GlobalLoad,
             aps::SpmLoadReq, aps::SpmLoadCollect>(op);
}

LogicalResult MemoryOpGenerator::generateGlobalMemLoad(
    aps::GlobalLoad op, mlir::OpBuilder &b, Location loc, int64_t slot,
    llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  auto width = llvm::dyn_cast<mlir::IntegerType>(op.getResult().getType()).getWidth();
  
  StringRef globalName = op.getGlobalName();
  auto glblRegName = std::string("glbl_reg_").append(globalName);

  // Call the appropriate scratchpad pool bank read method
  auto callResult = b.create<circt::cmt2::CallOp>(
      loc, circt::firrtl::UIntType::get(b.getContext(), width),
      mlir::ValueRange{},
      mlir::SymbolRefAttr::get(b.getContext(), glblRegName),
      mlir::SymbolRefAttr::get(b.getContext(), "read"),
      mlir::ArrayAttr(), mlir::ArrayAttr());

  localMap[op.getResult()] = callResult.getResult(0);
  return success();
}

LogicalResult MemoryOpGenerator::generateGlobalMemStore(
    aps::GlobalStore op, mlir::OpBuilder &b, Location loc, int64_t slot,
    llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  StringRef globalName = op.getGlobalName();
  auto glblRegName = std::string("glbl_reg_").append(globalName);

  auto data = getValueInRule(op.getValue(), op.getOperation(), 0, b,
                             localMap, loc);
  // Call the appropriate scratchpad pool bank read method
  b.create<circt::cmt2::CallOp>(
      loc, mlir::ValueRange{},
      mlir::ValueRange{*data},
      mlir::SymbolRefAttr::get(b.getContext(), glblRegName),
      mlir::SymbolRefAttr::get(b.getContext(), "write"),
      mlir::ArrayAttr(), mlir::ArrayAttr());

  return success();
}

LogicalResult MemoryOpGenerator::generateSpmLoadReq(
    aps::SpmLoadReq op, mlir::OpBuilder &b, Location loc, int64_t slot,
    llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  // PANIC if no indices provided
  if (op.getIndices().empty()) {
    op.emitError("SPM load request operation must have at least one index");
    llvm::report_fatal_error("SPM load request requires address indices");
  }

  auto addr = getValueInRule(op.getIndices()[0], op.getOperation(), 1, b,
                             localMap, loc);
  if (failed(addr)) {
    op.emitError("Failed to get address for SPM load request");
    llvm::report_fatal_error("SPM load request address resolution failed");
  }

  // Get the memory reference and check if it comes from memref.get_global
  Value memRef = op.getMemref();
  Operation *defOp = memRef.getDefiningOp();

  std::string memoryBankRule;
  if (auto getGlobalOp = dyn_cast<memref::GetGlobalOp>(defOp)) {
    // Extract the global symbol name and build the rule name
    StringRef globalName = getGlobalOp.getName();
    // Convert @mem_a_0 -> mem_a_0_read_0 (request phase)
    memoryBankRule = (globalName + "_read_0").str();
    llvm::dbgs() << "DEBUG: SPM load request from global " << globalName
                 << " using rule " << memoryBankRule << "\n";
  }

  // Get the memref type to determine array size and element type
  auto memrefType = dyn_cast<mlir::MemRefType>(memRef.getType());
  if (!memrefType) {
    op.emitError("Memory reference is not memref type");
    llvm::report_fatal_error("SpmLoadReq: invalid memref type");
  }

  // Calculate required address bit width from array size: ceil(log2(size))
  auto shape = memrefType.getShape();
  if (shape.empty() || shape[0] <= 0) {
    op.emitError("Memref must have valid array size");
    llvm::report_fatal_error("SpmLoadReq: invalid memref shape");
  }
  int64_t arraySize = shape[0];
  unsigned addrWidth = arraySize <= 1 ? 1 : (unsigned)std::ceil(std::log2(arraySize));

  llvm::dbgs() << "DEBUG: SPM load request - array size: " << arraySize
               << ", addr width: " << addrWidth << "\n";

  // Truncate address to required bit width using Signal
  auto addrSignal = Signal(*addr, &b, loc).bits(addrWidth - 1, 0);

  // Call the scratchpad pool bank read_0 method (request phase - sends address)
  b.create<circt::cmt2::CallOp>(
      loc, mlir::TypeRange{},
      mlir::ValueRange{addrSignal.getValue()},
      mlir::SymbolRefAttr::get(b.getContext(), "scratchpad_pool"),
      mlir::SymbolRefAttr::get(b.getContext(), memoryBankRule),
      mlir::ArrayAttr(), mlir::ArrayAttr());

  // Store dummy token - the actual data will be collected in SpmLoadCollect
  localMap[op.getResult()] = UInt::constant(1, 1, b, loc).getValue();
  return success();
}

LogicalResult MemoryOpGenerator::generateSpmLoadCollect(
    aps::SpmLoadCollect op, mlir::OpBuilder &b, Location loc, int64_t slot,
    llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  // Get the request token's defining operation to find the memory bank info
  Value request = op.getRequest();
  auto reqOp = request.getDefiningOp<aps::SpmLoadReq>();
  if (!reqOp) {
    op.emitError("SPM load collect must have SpmLoadReq as input");
    llvm::report_fatal_error("SpmLoadCollect: invalid request token");
  }

  // Get the memory reference from the request operation
  Value memRef = reqOp.getMemref();
  Operation *defOp = memRef.getDefiningOp();

  std::string memoryBankRule;
  if (auto getGlobalOp = dyn_cast<memref::GetGlobalOp>(defOp)) {
    // Extract the global symbol name and build the rule name
    StringRef globalName = getGlobalOp.getName();
    // Convert @mem_a_0 -> mem_a_0_read_1 (collect phase)
    memoryBankRule = (globalName + "_read_1").str();
    llvm::dbgs() << "DEBUG: SPM load collect from global " << globalName
                 << " using rule " << memoryBankRule << "\n";
  }

  // Get the memref type to determine element type for result width
  auto memrefType = dyn_cast<mlir::MemRefType>(memRef.getType());
  if (!memrefType) {
    op.emitError("Memory reference is not memref type");
    llvm::report_fatal_error("SpmLoadCollect: invalid memref type");
  }

  // Get element type for result width
  Type elementType = memrefType.getElementType();
  auto intType = dyn_cast<mlir::IntegerType>(elementType);
  if (!intType) {
    op.emitError("Memref element type is not integer");
    llvm::report_fatal_error("SpmLoadCollect: memref element must be integer type");
  }
  unsigned resultWidth = intType.getWidth();

  llvm::dbgs() << "DEBUG: SPM load collect - result width: " << resultWidth << "\n";

  // Call the scratchpad pool bank read_1 method (collect phase - returns data)
  auto callResult = b.create<circt::cmt2::CallOp>(
      loc, circt::firrtl::UIntType::get(b.getContext(), resultWidth),
      mlir::ValueRange{},
      mlir::SymbolRefAttr::get(b.getContext(), "scratchpad_pool"),
      mlir::SymbolRefAttr::get(b.getContext(), memoryBankRule),
      mlir::ArrayAttr(), mlir::ArrayAttr());

  localMap[op.getResult()] = callResult.getResult(0);
  return success();
}

LogicalResult MemoryOpGenerator::generateMemStore(
    aps::MemStore op, mlir::OpBuilder &b, Location loc, int64_t slot,
    llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  // PANIC if no indices provided
  if (op.getIndices().empty()) {
    op.emitError("Memory store operation must have at least one index");
    llvm::report_fatal_error("Memory store requires address indices");
  }

  // Get the value to store
  auto value = getValueInRule(op.getValue(), op.getOperation(), 0, b,
                              localMap, loc);
  if (failed(value)) {
    op.emitError("Failed to get value for memory store");
    llvm::report_fatal_error("Memory store value resolution failed");
  }

  // Get the address
  auto addr = getValueInRule(op.getIndices()[0], op.getOperation(), 2, b,
                             localMap, loc);
  if (failed(addr)) {
    op.emitError("Failed to get address for memory store");
    llvm::report_fatal_error("Memory store address resolution failed");
  }

  // Get the memory reference and check if it comes from memref.get_global
  Value memRef = op.getMemref();
  Operation *defOp = memRef.getDefiningOp();

  std::string memoryBankRule;
  if (auto getGlobalOp = dyn_cast<memref::GetGlobalOp>(defOp)) {
    // Extract the global symbol name and build the rule name
    StringRef globalName = getGlobalOp.getName();
    // Convert @mem_a_0 -> mem_a_0_write
    memoryBankRule = (globalName + "_write").str();
    llvm::dbgs() << "DEBUG: Memory store to global " << globalName
                 << " using rule " << memoryBankRule << "\n";
  }

  // Get the memref type to determine array size and element type
  auto memrefType = dyn_cast<mlir::MemRefType>(memRef.getType());
  if (!memrefType) {
    op.emitError("Memory reference is not memref type");
    llvm::report_fatal_error("MemStore: invalid memref type");
  }

  // Get element type for data width
  Type elementType = memrefType.getElementType();
  auto intType = dyn_cast<mlir::IntegerType>(elementType);
  if (!intType) {
    op.emitError("Memref element type is not integer");
    llvm::report_fatal_error("MemStore: memref element must be integer type");
  }
  unsigned dataWidth = intType.getWidth();

  // Calculate required address bit width from array size: ceil(log2(size))
  auto shape = memrefType.getShape();
  if (shape.empty() || shape[0] <= 0) {
    op.emitError("Memref must have valid array size");
    llvm::report_fatal_error("MemStore: invalid memref shape");
  }
  int64_t arraySize = shape[0];
  unsigned addrWidth = arraySize <= 1 ? 1 : (unsigned)std::ceil(std::log2(arraySize));

  llvm::dbgs() << "DEBUG: Memory store - array size: " << arraySize
               << ", addr width: " << addrWidth
               << ", data width: " << dataWidth << "\n";

  // Truncate address to required bit width using Signal
  auto addrSignal = Signal(*addr, &b, loc).bits(addrWidth - 1, 0);

  // Call the appropriate scratchpad pool bank write method
  b.create<circt::cmt2::CallOp>(
      loc, TypeRange{}, // No return value for write
      mlir::ValueRange{addrSignal.getValue(), *value},
      mlir::SymbolRefAttr::get(b.getContext(), "scratchpad_pool"),
      mlir::SymbolRefAttr::get(b.getContext(), memoryBankRule),
      mlir::ArrayAttr(), mlir::ArrayAttr());

  return success();
}

LogicalResult MemoryOpGenerator::generateBurstLoadReq(
    aps::ItfcBurstLoadReq op, mlir::OpBuilder &b, Location loc, int64_t slot,
    llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  Value cpuAddr = op.getCpuAddr();
  Value memRef = op.getMemrefs()[0];
  Value start = op.getStart();
  llvm::dbgs() << start.getType();
  Value numOfElements = op.getLength();

  // Calculate operand indices accounting for variadic memrefs
  // BurstLoadReq: cpu_addr(0), memrefs(1..N), start(1+N), length(2+N)
  unsigned numMemrefs = op.getMemrefs().size();
  unsigned cpuAddrOperandId = 0;
  unsigned startOperandId = 1 + numMemrefs;
  unsigned lengthOperandId = 2 + numMemrefs;

  // Get the global memory reference name
  auto getGlobalOp = dyn_cast<memref::GetGlobalOp>(memRef.getDefiningOp());
  if (!getGlobalOp) {
    op.emitError("Burst load request must use global memory reference");
    return failure();
  }
  auto globalName = getGlobalOp.getName().str();

  // Find the corresponding memory entry using the pre-built map with prefix matching
  const MemoryEntryInfo *targetMemEntry = nullptr;
  if (!globalName.empty()) {
    auto &memEntryMap = bbHandler->getMemEntryMap();
    
    // First try exact match
    auto exactIt = memEntryMap.find(globalName);
    if (exactIt != memEntryMap.end()) {
      targetMemEntry = &exactIt->second;
    } else {
      // If exact match fails, try prefix matching with underscore
      llvm::SmallVector<const MemoryEntryInfo *, 4> matchingEntries;
      
      for (auto &entry : memEntryMap) {
        std::string key = entry.first.str();
        // Check if the map key starts with globalName + "_"
        if (globalName.rfind( key + "_", 0) == 0) {
          targetMemEntry = &entry.second;
        }
      }
    }
  }

  if (!targetMemEntry) {
    op.emitError("Failed to find target memory entry!");
    return failure();
  }

  // Get element type and size from the memory entry info
  int elementSizeBytes =
      (targetMemEntry->dataWidth + 7) / 8; // Convert bits to bytes, rounding up

  // Calculate localAddr: baseAddress + (start * numOfElements *
  // elementSizeBytes)
  uint32_t baseAddress = targetMemEntry->baseAddress;

  // Calculate offset: start * numOfElements * elementSizeBytes
  auto baseAddrConst = UInt::constant(baseAddress, 32, b, loc);

  // Use getValueInRule to get start with proper FIRRTL conversion
  auto startValue = getValueInRule(start, op.getOperation(), startOperandId, b, localMap, loc);
  if (failed(startValue)) {
    op.emitError("Failed to get start for burst load");
    llvm::report_fatal_error("Burst load start resolution failed");
  }

  auto startSig = Signal(*startValue, &b, loc).bits(31, 0);
  auto elementSizeBytesConst = UInt::constant(elementSizeBytes, 32, b, loc);
  Signal localAddr = baseAddrConst + startSig * elementSizeBytesConst;

  // Calculate total burst length: elementSizeBytes * numOfElements, rounded up
  // to nearest power of 2
  auto numElementsOp = numOfElements.getDefiningOp<arith::ConstantOp>();
  if (!numElementsOp) {
    op.emitError("Number of elements must be a constant");
    return failure();
  }
  auto numElementsAttr = numElementsOp.getValue();
  auto numElements =
      dyn_cast<IntegerAttr>(numElementsAttr).getValue().getZExtValue();

  uint64_t totalBurstLength = (uint64_t)elementSizeBytes * numElements;
  uint32_t roundedTotalBurstLength =
      bbHandler->roundUpToPowerOf2((uint32_t)totalBurstLength);
  // TileLink size field expects log2 of transfer size (N where size = 2^N)
  uint32_t tlSizeField = bbHandler->log2Floor(roundedTotalBurstLength);
  if (tlSizeField == 0) {
    op->emitError("Got 0 when attempt to call burst load!");
  }
  auto realCpuLength = UInt::constant(tlSizeField, 32, b, loc);

  // Use getValueInRule to get cpuAddr with proper FIRRTL conversion
  auto cpuAddrValue = getValueInRule(cpuAddr, op.getOperation(), cpuAddrOperandId, b, localMap, loc);
  if (failed(cpuAddrValue)) {
    op.emitError("Failed to get cpuAddr for burst load");
    llvm::report_fatal_error("Burst load cpuAddr resolution failed");
  }

  // Call DMA interface
  auto dmaItfc = bbHandler->getDmaInterface();
  if (!dmaItfc) {
    op.emitError("DMA interface not available");
    return failure();
  }

  auto tl_id = op->getAttrOfType<IntegerAttr>("tl_channel").getInt();

  auto strideXAttr = op->getAttrOfType<IntegerAttr>("stride_x");
  int64_t strideX = 0;
  if (strideXAttr) {
    strideX = strideXAttr.getInt();
  }
  auto strideXValue = UInt::constant(strideX, 8, b, loc);
  auto strideYAttr = op->getAttrOfType<IntegerAttr>("stride_y");
  int64_t strideY = 0;
  if (strideYAttr) {
    strideY = strideYAttr.getInt();
  }
  auto strideYValue = UInt::constant(strideY, 8, b, loc);

  dmaItfc->callMethod("cpu_to_isax_ch" + std::to_string(tl_id),
                      {*cpuAddrValue, localAddr.bits(31, 0).getValue(),
                       realCpuLength.bits(3, 0).getValue(),
                       strideXValue.getValue(),
                       strideYValue.getValue()},
                      b);

  localMap[op.getResult()] = UInt::constant(1, 1, b, loc).getValue();
  return success();
}

LogicalResult MemoryOpGenerator::generateBurstLoadCollect(
    aps::ItfcBurstLoadCollect op, mlir::OpBuilder &b, Location loc,
    int64_t slot, llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  auto dmaItfc = bbHandler->getDmaInterface();
  if (dmaItfc) {
    auto tl_id = op->getAttrOfType<IntegerAttr>("tl_channel").getInt();
    dmaItfc->callMethod("poll_for_idle_ch" + std::to_string(tl_id), {}, b);
  }
  return success();
}

LogicalResult MemoryOpGenerator::generateBurstStoreReq(
    aps::ItfcBurstStoreReq op, mlir::OpBuilder &b, Location loc, int64_t slot,
    llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  Value cpuAddr = op.getCpuAddr();
  Value memRef = op.getMemrefs()[0];
  Value start = op.getStart();
  Value numOfElements = op.getLength();

  // Calculate operand indices accounting for variadic memrefs
  // BurstStoreReq: memrefs(0..N-1), start(N), cpu_addr(N+1), length(N+2)
  unsigned numMemrefs = op.getMemrefs().size();
  unsigned startOperandId = numMemrefs;
  unsigned cpuAddrOperandId = numMemrefs + 1;
  unsigned lengthOperandId = numMemrefs + 2;

  // Get the global memory reference name
  auto getGlobalOp = dyn_cast<memref::GetGlobalOp>(memRef.getDefiningOp());
  if (!getGlobalOp) {
    op.emitError("Burst store request must use global memory reference");
    return failure();
  }
  auto globalName = getGlobalOp.getName().str();

  // Find the corresponding memory entry using the pre-built map with prefix matching
  const MemoryEntryInfo *targetMemEntry = nullptr;
  if (!globalName.empty()) {
    auto &memEntryMap = bbHandler->getMemEntryMap();
    
    // First try exact match
    auto exactIt = memEntryMap.find(globalName);
    if (exactIt != memEntryMap.end()) {
      targetMemEntry = &exactIt->second;
    } else {
      // If exact match fails, try prefix matching with underscore
      llvm::SmallVector<const MemoryEntryInfo *, 4> matchingEntries;
      
      for (auto &entry : memEntryMap) {
        std::string key = entry.first.str();
        // Check if the map key starts with globalName + "_"
        if (globalName.rfind( key + "_", 0) == 0) {
          targetMemEntry = &entry.second;
        }
      }
    }
  }

  if (!targetMemEntry) {
    op.emitError("Failed to find target memory entry!");
    return failure();
  }

  // Get element type and size from the memory entry info
  int elementSizeBytes =
      (targetMemEntry->dataWidth + 7) / 8; // Convert bits to bytes, rounding up

  // Calculate localAddr: baseAddress + (start * numOfElements *
  // elementSizeBytes)
  uint32_t baseAddress = targetMemEntry->baseAddress;

  // Use getValueInRule to get start with proper FIRRTL conversion
  auto startValue = getValueInRule(start, op.getOperation(), startOperandId, b, localMap, loc);
  if (failed(startValue)) {
    op.emitError("Failed to get start for burst store");
    llvm::report_fatal_error("Burst store start resolution failed");
  }

  // Calculate offset: start * numOfElements * elementSizeBytes
  auto baseAddrConst = UInt::constant(baseAddress, 32, b, loc);
  auto startSig = Signal(*startValue, &b, loc);
  auto elementSizeBytesConst = UInt::constant(elementSizeBytes, 32, b, loc);
  Signal localAddr = baseAddrConst + startSig * elementSizeBytesConst;

  // Calculate total burst length: elementSizeBytes * numOfElements, rounded up
  // to nearest power of 2
  auto numElementsOp = numOfElements.getDefiningOp<arith::ConstantOp>();
  if (!numElementsOp) {
    op.emitError("Number of elements must be a constant");
    return failure();
  }
  auto numElementsAttr = numElementsOp.getValue();
  auto numElements =
      dyn_cast<IntegerAttr>(numElementsAttr).getValue().getZExtValue();

  uint64_t totalBurstLength = (uint64_t)elementSizeBytes * numElements;
  uint32_t roundedTotalBurstLength =
      bbHandler->roundUpToPowerOf2((uint32_t)totalBurstLength);
  // TileLink size field expects log2 of transfer size (N where size = 2^N)
  uint32_t tlSizeField = bbHandler->log2Floor(roundedTotalBurstLength);
  if (tlSizeField == 0) {
    op->emitError("Got 0 when attempt to call burst store!");
  }
  auto realCpuLength = UInt::constant(tlSizeField, 32, b, loc);

  // Use getValueInRule to get cpuAddr with proper FIRRTL conversion
  auto cpuAddrValue = getValueInRule(cpuAddr, op.getOperation(), cpuAddrOperandId, b, localMap, loc);
  if (failed(cpuAddrValue)) {
    op.emitError("Failed to get cpuAddr for burst store");
    llvm::report_fatal_error("Burst store cpuAddr resolution failed");
  }

  // Call DMA interface
  auto dmaItfc = bbHandler->getDmaInterface();
  if (!dmaItfc) {
    op.emitError("DMA interface not available");
    return failure();
  }

  auto tl_id = op->getAttrOfType<IntegerAttr>("tl_channel").getInt();
  auto strideXAttr = op->getAttrOfType<IntegerAttr>("stride_x");
  int64_t strideX = 0;
  if (strideXAttr) {
    strideX = strideXAttr.getInt();
  }
  auto strideXValue = UInt::constant(strideX, 8, b, loc);
  auto strideYAttr = op->getAttrOfType<IntegerAttr>("stride_y");
  int64_t strideY = 0;
  if (strideYAttr) {
    strideY = strideYAttr.getInt();
  }
  auto strideYValue = UInt::constant(strideY, 8, b, loc);

  dmaItfc->callMethod("isax_to_cpu_ch" + std::to_string(tl_id),
                      {*cpuAddrValue, localAddr.bits(31, 0).getValue(),
                       realCpuLength.bits(3, 0).getValue(),
                      strideXValue.getValue(),
                      strideYValue.getValue()},
                      b);

  localMap[op.getResult()] = UInt::constant(1, 1, b, loc).getValue();
  return success();
}

LogicalResult MemoryOpGenerator::generateBurstStoreCollect(
    aps::ItfcBurstStoreCollect op, mlir::OpBuilder &b, Location loc,
    int64_t slot, llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  auto dmaItfc = bbHandler->getDmaInterface();
  if (dmaItfc) {
    auto tl_id = op->getAttrOfType<IntegerAttr>("tl_channel").getInt();
    dmaItfc->callMethod("poll_for_idle_ch" + std::to_string(tl_id), {}, b);
  }
  return success();
}

} // namespace mlir