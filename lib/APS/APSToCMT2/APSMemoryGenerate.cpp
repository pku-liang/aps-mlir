#include "APS/APSToCMT2.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "aps-memory-pool-gen"

namespace mlir {

using namespace mlir;
using namespace mlir::tor;
using namespace circt::cmt2::ecmt2;
using namespace circt::cmt2::ecmt2::stl;
using namespace circt::firrtl;

/// Extract data width and address width from a memref type
bool APSToCMT2GenPass::extractMemoryParameters(memref::GlobalOp globalOp,
                                               int &dataWidth, int &addrWidth,
                                               int &depth) {
  if (!globalOp) {
    return false;
  }

  auto memrefType = globalOp.getType();
  if (!memrefType) {
    return false;
  }

  // Get element type and extract bit width
  auto elementType = memrefType.getElementType();
  if (auto intType = llvm::dyn_cast<IntegerType>(elementType)) {
    dataWidth = intType.getWidth();
  } else if (auto floatType = llvm::dyn_cast<FloatType>(elementType)) {
    dataWidth = floatType.getWidth();
  } else {
    return false;
  }

  // Get memory depth (number of elements)
  if (memrefType.getRank() != 1) {
    // Only support 1D memrefs for now
    if (memrefType.getRank() == 0) {
      addrWidth = 1;
      depth = 2;
      return true;
    }
  }

  depth = memrefType.getShape()[0];

  // Calculate address width: ceil(log2(depth)), make sure addrWidth are positive
  addrWidth = llvm::Log2_32_Ceil(depth);

  return true;
}

// ============================================================================
//
//                                BANK WRAPPER
//
// ============================================================================

/// Generate a bank wrapper module that encapsulates bank selection and data
/// alignment
Module *APSToCMT2GenPass::generateBankWrapperModule(
    const MemoryEntryInfo &entryInfo, Circuit &circuit, size_t bankIdx,
    ExternalModule *memMod, Clock clk, Reset rst, bool burstEnable) {
  std::string wrapperName =
      "BankWrapper_" + entryInfo.name + "_" + std::to_string(bankIdx);
  auto *wrapper = circuit.addModule(wrapperName);

  // Add clock and reset
  Clock wrapperClk = wrapper->addClockArgument("clk");
  Reset wrapperRst = wrapper->addResetArgument("rst");

  auto &builder = wrapper->getBuilder();
  auto loc = wrapper->getLoc();
  auto u64Type = UIntType::get(builder.getContext(), 64);
  auto u32Type = UIntType::get(builder.getContext(), 32);

  uint32_t elementsPerBurst = 64 / entryInfo.dataWidth;

  // Create the actual memory bank instance inside this wrapper
  auto *bankInst = wrapper->addInstance(
      "mem_bank", memMod, {wrapperClk.getValue(), wrapperRst.getValue()});

  if (burstEnable) {
    // Create wire_default modules for enable, data, and addr
    // Save insertion point before creating wire modules
    auto savedIPForWires = builder.saveInsertionPoint();

    auto wireEnableMod = STLLibrary::createWireDefaultModule(1, 0, circuit);
    auto wireDataMod =
        STLLibrary::createWireModule(entryInfo.dataWidth, circuit);
    auto wireAddrMod =
        STLLibrary::createWireModule(entryInfo.addrWidth, circuit);

    // Restore insertion point back to wrapper
    builder.restoreInsertionPoint(savedIPForWires);

    // Create wire_default instances in the wrapper
    auto *writeEnableWire =
        wrapper->addInstance("write_enable_wire", wireEnableMod, {});
    auto *writeDataWire =
        wrapper->addInstance("write_data_wire", wireDataMod, {});
    auto *writeAddrWire =
        wrapper->addInstance("write_addr_wire", wireAddrMod, {});

    // burst_read_0 method: calculates local addr, calls rd0 to initiate read
    auto *burstRead0 =
        wrapper->addMethod("burst_read_0", {{"addr", u32Type}}, {});

    burstRead0->guard([&](mlir::OpBuilder &guardBuilder,
                          llvm::ArrayRef<mlir::BlockArgument> args) {
      auto trueVal = UInt::constant(1, 1, guardBuilder, wrapper->getLoc());
      guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueVal.getValue());
    });

    burstRead0->body([&](mlir::OpBuilder &bodyBuilder,
                         llvm::ArrayRef<mlir::BlockArgument> args) {
      auto addr = Signal(args[0], &bodyBuilder, wrapper->getLoc());

      auto elementSizeConst =
          UInt::constant(entryInfo.dataWidth / 8, 32, bodyBuilder,
                         wrapper->getLoc()); // Element size in bytes
      auto numBanksConst = UInt::constant(entryInfo.numBanks, 32, bodyBuilder,
                                          wrapper->getLoc());
      auto myBankConst =
          UInt::constant(bankIdx, 32, bodyBuilder, wrapper->getLoc());

      // element_idx = addr / element_size
      auto elementIdx = addr / elementSizeConst;
      // start_bank_idx = element_idx % num_banks
      auto startBankIdx = elementIdx % numBanksConst;

      // position = (my_bank - start_bank_idx + num_banks) % num_banks
      auto position =
          (myBankConst - startBankIdx + numBanksConst) % numBanksConst;

      // Calculate local address: my_element_idx = element_idx + position;
      // local_addr = my_element_idx / num_banks
      auto myElementIdx = elementIdx + position;
      auto localAddr = myElementIdx / numBanksConst;
      auto localAddrTrunc = localAddr.bits(entryInfo.addrWidth - 1, 0);

      // Call rd0 to initiate read
      bankInst->callMethod("rd0", {localAddrTrunc.getValue()}, bodyBuilder);

      bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
    });

    burstRead0->finalize();

    // burst_read_1 method: gets data from rd1, aligns based on addr, returns
    auto *burstRead1 =
        wrapper->addMethod("burst_read_1", {{"addr", u32Type}}, {u64Type});

    burstRead1->guard([&](mlir::OpBuilder &guardBuilder,
                          llvm::ArrayRef<mlir::BlockArgument> args) {
      auto trueVal = UInt::constant(1, 1, guardBuilder, wrapper->getLoc());
      guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueVal.getValue());
    });

    burstRead1->body([&](mlir::OpBuilder &bodyBuilder,
                         llvm::ArrayRef<mlir::BlockArgument> args) {
      auto addr = Signal(args[0], &bodyBuilder, wrapper->getLoc());

      auto elementSizeConst =
          UInt::constant(entryInfo.dataWidth / 8, 32, bodyBuilder,
                         wrapper->getLoc()); // Element size in bytes
      auto numBanksConst = UInt::constant(entryInfo.numBanks, 32, bodyBuilder,
                                          wrapper->getLoc());
      auto myBankConst =
          UInt::constant(bankIdx, 32, bodyBuilder, wrapper->getLoc());
      auto elementsPerBurstConst =
          UInt::constant(elementsPerBurst, 32, bodyBuilder, wrapper->getLoc());

      // Recalculate position and isMine from addr
      // element_idx = addr / element_size
      auto elementIdx = addr / elementSizeConst;
      // start_bank_idx = element_idx % num_banks
      auto startBankIdx = elementIdx % numBanksConst;

      // position = (my_bank - start_bank_idx + num_banks) % num_banks
      auto position =
          (myBankConst - startBankIdx + numBanksConst) % numBanksConst;
      auto isMine = position < elementsPerBurstConst;

      // Get data from rd1
      auto bankDataValues = bankInst->callValue("rd1", bodyBuilder);
      auto rawData = Signal(bankDataValues[0], &bodyBuilder, loc);

      // Calculate bit position: position * data_width
      auto elementOffsetInBurst = position;

      // Generate aligned data for each possible offset position
      mlir::Value alignedData;

      if (entryInfo.dataWidth == 64) {
        // Full 64-bit, no padding needed
        alignedData = rawData.getValue();
      } else {
        // Generate padded data for each possible bit position and mux
        llvm::SmallVector<mlir::Value, 4> positionedDataValues;

        for (uint32_t elemOffset = 0; elemOffset < elementsPerBurst;
             ++elemOffset) {
          uint32_t bitShift = elemOffset * entryInfo.dataWidth;
          uint32_t leftPadWidth = 64 - bitShift - entryInfo.dataWidth;
          uint32_t rightPadWidth = bitShift;
          auto padded = rawData;
          if (leftPadWidth > 0) {
            auto zeroLeft = UInt::constant(0, leftPadWidth, bodyBuilder, loc);
            padded = zeroLeft.cat(padded);
          }
          if (rightPadWidth > 0) {
            auto zeroRight = UInt::constant(0, rightPadWidth, bodyBuilder, loc);
            padded = padded.cat(zeroRight);
          }
          positionedDataValues.push_back(padded.getValue());
        }

        // Mux to select correct positioned data
        alignedData = positionedDataValues[0];
        for (uint32_t i = 1; i < elementsPerBurst; ++i) {
          auto offsetConst = UInt::constant(i, 32, bodyBuilder, loc);
          auto isThisOffset = elementOffsetInBurst == offsetConst;
          alignedData =
              isThisOffset
                  .mux(Signal(positionedDataValues[i], &bodyBuilder, loc),
                       Signal(alignedData, &bodyBuilder, loc))
                  .getValue();
        }
      }

      // Return aligned data if mine, else 0
      auto zeroData = UInt::constant(0, 64, bodyBuilder, wrapper->getLoc());
      auto resultOp = isMine.mux(
          Signal(alignedData, &bodyBuilder, wrapper->getLoc()), zeroData);

      bodyBuilder.create<circt::cmt2::ReturnOp>(
          loc, mlir::ValueRange{resultOp.getValue()});
    });

    burstRead1->finalize();

    // burst_write method
    auto *burstWrite = wrapper->addMethod(
        "burst_write", {{"addr", u32Type}, {"data", u64Type}}, {});

    burstWrite->guard([&](mlir::OpBuilder &guardBuilder,
                          llvm::ArrayRef<mlir::BlockArgument> args) {
      auto trueConst = UInt::constant(1, 1, guardBuilder, loc);
      guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getValue());
    });

    burstWrite->body([&](mlir::OpBuilder &bodyBuilder,
                         llvm::ArrayRef<mlir::BlockArgument> args) {
      auto addr = Signal(args[0], &bodyBuilder, wrapper->getLoc());
      auto data = Signal(args[1], &bodyBuilder, wrapper->getLoc());

      // Helper: Create constants using Signal (32-bit for address calculations)
      auto elementSizeConst =
          UInt::constant(entryInfo.dataWidth / 8, 32, bodyBuilder,
                         wrapper->getLoc()); // Element size in bytes
      auto numBanksConst = UInt::constant(entryInfo.numBanks, 32, bodyBuilder,
                                          wrapper->getLoc());
      auto myBankConst =
          UInt::constant(bankIdx, 32, bodyBuilder, wrapper->getLoc());
      auto elementsPerBurstConst =
          UInt::constant(elementsPerBurst, 32, bodyBuilder, wrapper->getLoc());

      // Helper 1: element_idx = addr / element_size
      auto elementIdx = addr / elementSizeConst;

      // Helper 2: start_bank_idx = element_idx % num_banks
      auto startBankIdx = elementIdx % numBanksConst;

      // Helper 3: Calculate position and check participation
      auto position =
          ((myBankConst - startBankIdx + numBanksConst) % numBanksConst);
      auto isMine = position < elementsPerBurstConst;

      // Helper 4: Calculate local address
      auto myElementIdx = elementIdx + position;
      auto localAddr = myElementIdx / numBanksConst;
      auto localAddrTrunc = localAddr.bits(entryInfo.addrWidth - 1, 0);

      // Helper 5: Calculate data slice position
      auto elementOffsetInBurst = position;

      // Extract all possible data slices and mux using Signal
      llvm::SmallVector<Signal, 4> dataSlices;

      for (uint32_t elemOffset = 0; elemOffset < elementsPerBurst;
           ++elemOffset) {
        uint32_t bitStart = elemOffset * entryInfo.dataWidth;
        uint32_t bitEnd = bitStart + entryInfo.dataWidth - 1;
        auto slice = data.bits(bitEnd, bitStart);
        dataSlices.push_back(slice);
      }

      auto myData = dataSlices[0];
      for (uint32_t i = 1; i < elementsPerBurst; ++i) {
        auto offsetConst =
            UInt::constant(i, 32, bodyBuilder, wrapper->getLoc());
        auto isThisOffset = (elementOffsetInBurst == offsetConst);
        myData = isThisOffset.mux(dataSlices[i], myData);
      }

      // Write to wire instances using callMethod
      // The wires will be written conditionally based on isMine
      auto trueConst = UInt::constant(1, 1, bodyBuilder, wrapper->getLoc());
      auto falseConst = UInt::constant(0, 1, bodyBuilder, wrapper->getLoc());

      // Mux to select enable value based on isMine
      auto enableValue = isMine.mux(trueConst, falseConst);

      // Call write methods on wire instances
      writeEnableWire->callMethod("write", {enableValue.getValue()},
                                  bodyBuilder);
      writeDataWire->callMethod("write", {myData.getValue()}, bodyBuilder);
      writeAddrWire->callMethod("write", {localAddrTrunc.getValue()},
                                bodyBuilder);

      bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
    });

    burstWrite->finalize();

    // Create a rule that reads from wires and conditionally writes to bank
    // NOTE: This currently crashes because callValue on regular CMT2 Module
    // instances returns empty results. This is a bug in ECMT2 Instance.cpp that
    // needs to be fixed.
    auto *writeRule = wrapper->addRule("do_bank_write");

    writeRule->guard([&](mlir::OpBuilder &guardBuilder) {
      auto trueConst = UInt::constant(1, 1, guardBuilder, wrapper->getLoc());
      auto enableValues =
          Signal(writeEnableWire->callValue("read", guardBuilder)[0],
                &guardBuilder, loc);
      auto isEnabled = enableValues == trueConst;
      guardBuilder.create<circt::cmt2::ReturnOp>(loc, isEnabled.getValue());
    });

    writeRule->body([&, bankInst](mlir::OpBuilder &bodyBuilder) {
      // Read data and address from wires and write to bank
      auto dataValues = writeDataWire->callValue("read", bodyBuilder);
      auto addrValues = writeAddrWire->callValue("read", bodyBuilder);

      bankInst->callMethod("write", {dataValues[0], addrValues[0]}, bodyBuilder);

      bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
    });

    writeRule->finalize();
  }

  // Direct bank-level read methods (no burst translation), exposes native port
  // Split into rd0 (request) and rd1 (response) for 2-cycle memory
  auto bankAddrType = UIntType::get(builder.getContext(), entryInfo.addrWidth);
  auto bankDataType = UIntType::get(builder.getContext(), entryInfo.dataWidth);

  // bank_read_0: Initiate read request with address (calls rd0)
  auto *directReadMethod0 =
      wrapper->addMethod("bank_read_0", {{"addr", bankAddrType}}, {});

  directReadMethod0->guard([&](mlir::OpBuilder &guardBuilder,
                               llvm::ArrayRef<mlir::BlockArgument> /*args*/) {
    auto trueConst = UInt::constant(1, 1, guardBuilder, wrapper->getLoc());
    guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getValue());
  });

  directReadMethod0->body(
      [&, bankInst](mlir::OpBuilder &bodyBuilder,
                    llvm::ArrayRef<mlir::BlockArgument> args) {
        auto addr = Signal(args[0], &bodyBuilder, wrapper->getLoc());

        // Call rd0 to initiate read request
        bankInst->callMethod("rd0", {addr.getValue()}, bodyBuilder);
        bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
      });

  directReadMethod0->finalize();

  // bank_read_1: Get read data (calls rd1 value)
  auto *directReadValue1 =
      wrapper->addValue("bank_read_1", {bankDataType});

  directReadValue1->guard([&](mlir::OpBuilder &guardBuilder) {
    auto trueConst = UInt::constant(1, 1, guardBuilder, wrapper->getLoc());
    guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getValue());
  });

  directReadValue1->body([&, bankInst](mlir::OpBuilder &bodyBuilder) {
    // Call rd1 value to get read data
    auto readValues = bankInst->callValue("rd1", bodyBuilder);
    bodyBuilder.create<circt::cmt2::ReturnOp>(loc, readValues[0]);
  });

  directReadValue1->finalize();

  auto *directWriteMethod = wrapper->addMethod(
      "bank_write", {{"addr", bankAddrType}, {"data", bankDataType}}, {});

  directWriteMethod->guard([&](mlir::OpBuilder &guardBuilder,
                               llvm::ArrayRef<mlir::BlockArgument> /*args*/) {
    auto trueConst = UInt::constant(1, 1, guardBuilder, wrapper->getLoc());
    guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getValue());
  });

  directWriteMethod->body(
      [&, bankInst](mlir::OpBuilder &bodyBuilder,
                    llvm::ArrayRef<mlir::BlockArgument> args) {
        auto addr = Signal(args[0], &bodyBuilder, wrapper->getLoc());
        auto data = Signal(args[1], &bodyBuilder, wrapper->getLoc());

        // Directly drive the memory bank write ports.
        bankInst->callMethod("write", {data.getValue(), addr.getValue()},
                             bodyBuilder);
        bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
      });

  directWriteMethod->finalize();
  
  if (burstEnable) {
    // Ensure burst accesses get scheduled ahead of direct bank accesses.
    wrapper->setPrecedence(
        {{"burst_read_0", "burst_read_1"}, {"burst_read_1", "bank_read_0"},
         {"bank_read_0", "bank_read_1"}, {"burst_write", "bank_write"}});
  }

  return wrapper;
}

// ============================================================================
//
//                   MEMORY ENTRY (CONTAIN BANKWRAPPER)
//
// ============================================================================

/// Generate memory entry submodule for a single memory entry
Module *APSToCMT2GenPass::generateMemoryEntryModule(
    const MemoryEntryInfo &entryInfo, Circuit &circuit, Clock clk, Reset rst,
    const llvm::SmallVector<std::string, 4> &bankNames) {
  // Validate configuration to avoid bank conflicts in burst access
  // Constraint: num_banks * data_width >= 64
  // If < 64, multiple elements in a 64-bit burst map to the same bank
  // (conflict!)
  bool burstEnable = true;
  if (entryInfo.isCyclic) {
    uint32_t totalBankWidth = entryInfo.numBanks * entryInfo.dataWidth;
    if (totalBankWidth < 64) {
      llvm::errs()
          << "  ERROR: Cyclic partition configuration causes bank conflicts!\n";
      llvm::errs() << "    Entry: " << entryInfo.name << "\n";
      llvm::errs() << "    Config: " << entryInfo.numBanks << " banks × "
                   << entryInfo.dataWidth << " bits = " << totalBankWidth
                   << " bits\n";
      llvm::errs() << "    A 64-bit burst contains "
                   << (64 / entryInfo.dataWidth) << " elements\n";
      llvm::errs() << "    Elements per bank: "
                   << ((64 / entryInfo.dataWidth + entryInfo.numBanks - 1) /
                       entryInfo.numBanks)
                   << " (CONFLICT!)\n";
      llvm::errs() << "    Requirement: num_banks × data_width >= 64\n";
      llvm::errs()
          << "    Valid examples: 8×8, 4×16, 2×32, 1×64, 4×32, 8×16, etc.\n";
      burstEnable = false;
    }
  }
  
  // Single element (access via globalload/global store) has been moved out
  // if (entryInfo.addrWidth == 0) {
  //   // only one element
  //   burstEnable = false;
  // }

  // Create submodule for this memory entry
  std::string moduleName = "memory_" + entryInfo.name;
  auto *entryModule = circuit.addModule(moduleName);

  // Add clock and reset arguments
  Clock subClk = entryModule->addClockArgument("clk");
  Reset subRst = entryModule->addResetArgument("rst");

  auto &builder = entryModule->getBuilder();
  auto loc = entryModule->getLoc();

  auto u64Type = UIntType::get(builder.getContext(), 64);
  auto u32Type = UIntType::get(builder.getContext(), 32);

  // Create external module declaration at circuit level (BEFORE creating
  // instances) Save insertion point first
  auto savedIP = builder.saveInsertionPoint();

  llvm::StringMap<int64_t> memParams;
  memParams["data_width"] = entryInfo.dataWidth;
  memParams["addr_width"] = entryInfo.addrWidth;
  memParams["depth"] = entryInfo.depth;

  auto memMod = STLLibrary::createMem1r1w1cARegModule(
      entryInfo.dataWidth, entryInfo.addrWidth, entryInfo.depth, circuit);

  // Restore insertion point back to entry module
  builder.restoreInsertionPoint(savedIP);

  // Create bank wrapper modules (this changes insertion point to each wrapper)
  llvm::SmallVector<Module *, 4> wrapperModules;
  for (size_t i = 0; i < entryInfo.numBanks; ++i) {
    auto *wrapperMod = generateBankWrapperModule(entryInfo, circuit, i, memMod,
                                                 subClk, subRst, burstEnable);
    wrapperModules.push_back(wrapperMod);
  }

  // Restore insertion point back to entryModule before adding instances/methods
  builder.restoreInsertionPoint(savedIP);

  // Create wrapper instances IN THIS SUBMODULE
  llvm::SmallVector<Instance *, 4> wrapperInstances;
  for (size_t i = 0; i < entryInfo.numBanks; ++i) {
    std::string wrapperName = "bank_wrap_" + std::to_string(i);
    auto *wrapperInst = entryModule->addInstance(
        wrapperName, wrapperModules[i], {subClk.getValue(), subRst.getValue()});
    wrapperInstances.push_back(wrapperInst);
  }

  if (burstEnable) {
    // Create burst_read_0 method: forwards addr to all bank wrappers' burst_read_0
    auto *burstRead0 =
        entryModule->addMethod("burst_read_0", {{"addr", u32Type}}, {});

    burstRead0->guard([&](mlir::OpBuilder &guardBuilder,
                          llvm::ArrayRef<mlir::BlockArgument> args) {
      auto trueConst = UInt::constant(1, 1, guardBuilder, entryModule->getLoc());
      guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getValue());
    });

    burstRead0->body([&](mlir::OpBuilder &bodyBuilder,
                         llvm::ArrayRef<mlir::BlockArgument> args) {
      auto addr = Signal(args[0], &bodyBuilder, entryModule->getLoc());

      // Forward addr to all bank wrappers' burst_read_0
      auto methodSymbol =
          mlir::FlatSymbolRefAttr::get(bodyBuilder.getContext(), "burst_read_0");
      for (size_t i = 0; i < entryInfo.numBanks; ++i) {
        auto calleeSymbol = mlir::FlatSymbolRefAttr::get(
            bodyBuilder.getContext(), wrapperInstances[i]->getName());
        bodyBuilder.create<circt::cmt2::CallOp>(
            loc, mlir::TypeRange{}, mlir::ValueRange{addr.getValue()},
            calleeSymbol, methodSymbol, nullptr, nullptr);
      }

      bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
    });

    burstRead0->finalize();

    // Create burst_read_1 method: forwards addr to all wrappers' burst_read_1, ORs results
    auto *burstRead1 =
        entryModule->addMethod("burst_read_1", {{"addr", u32Type}}, {u64Type});

    burstRead1->guard([&](mlir::OpBuilder &guardBuilder,
                          llvm::ArrayRef<mlir::BlockArgument> args) {
      auto trueConst = UInt::constant(1, 1, guardBuilder, entryModule->getLoc());
      guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getValue());
    });

    burstRead1->body([&](mlir::OpBuilder &bodyBuilder,
                         llvm::ArrayRef<mlir::BlockArgument> args) {
      auto addr = Signal(args[0], &bodyBuilder, entryModule->getLoc());

      // Call all wrappers' burst_read_1 and OR results
      auto calleeSymbol0 = mlir::FlatSymbolRefAttr::get(
          bodyBuilder.getContext(), wrapperInstances[0]->getName());
      auto methodSymbol =
          mlir::FlatSymbolRefAttr::get(bodyBuilder.getContext(), "burst_read_1");
      auto callOp0 = bodyBuilder.create<circt::cmt2::CallOp>(
          loc, mlir::TypeRange{u64Type}, mlir::ValueRange{addr.getValue()},
          calleeSymbol0, methodSymbol, nullptr, nullptr);
      auto result =
          Signal(callOp0.getResult(0), &bodyBuilder, entryModule->getLoc());

      // OR together all other wrapper outputs
      for (size_t i = 1; i < entryInfo.numBanks; ++i) {
        auto calleeSymbol = mlir::FlatSymbolRefAttr::get(
            bodyBuilder.getContext(), wrapperInstances[i]->getName());
        auto callOp = bodyBuilder.create<circt::cmt2::CallOp>(
            loc, mlir::TypeRange{u64Type}, mlir::ValueRange{addr.getValue()},
            calleeSymbol, methodSymbol, nullptr, nullptr);
        auto data =
            Signal(callOp.getResult(0), &bodyBuilder, entryModule->getLoc());
        result = result | data;
      }

      bodyBuilder.create<circt::cmt2::ReturnOp>(loc, result.getValue());
    });

    burstRead1->finalize();
    // Create burst_write method: forwards data to all banks
    auto *burstWrite = entryModule->addMethod(
        "burst_write", {{"addr", u32Type}, {"data", u64Type}}, {});

    burstWrite->guard([&](mlir::OpBuilder &guardBuilder,
                          llvm::ArrayRef<mlir::BlockArgument> args) {
      auto trueConst = UInt::constant(1, 1, guardBuilder, entryModule->getLoc());
      guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getValue());
    });

    burstWrite->body([&](mlir::OpBuilder &bodyBuilder,
                        llvm::ArrayRef<mlir::BlockArgument> args) {
      auto addr = Signal(args[0], &bodyBuilder, entryModule->getLoc());
      auto data = Signal(args[1], &bodyBuilder, entryModule->getLoc());

      // Simple broadcast to all bank wrappers using CallOp
      // Each wrapper decides if it should write based on address
      auto methodSymbol =
          mlir::FlatSymbolRefAttr::get(bodyBuilder.getContext(), "burst_write");
      for (size_t i = 0; i < entryInfo.numBanks; ++i) {
        auto calleeSymbol = mlir::FlatSymbolRefAttr::get(
            bodyBuilder.getContext(), wrapperInstances[i]->getName());
        bodyBuilder.create<circt::cmt2::CallOp>(
            loc, mlir::TypeRange{},
            mlir::ValueRange{addr.getValue(), data.getValue()}, calleeSymbol,
            methodSymbol, nullptr, nullptr);
      }

      bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
    });

    burstWrite->finalize();
  } else {
    // dummy burst_read_0 method, do nothing
    auto *burstRead0 =
        entryModule->addMethod("burst_read_0", {{"addr", u32Type}}, {});
    burstRead0->guard([&](mlir::OpBuilder &guardBuilder,
                          llvm::ArrayRef<mlir::BlockArgument> args) {
      auto trueConst = UInt::constant(1, 1, guardBuilder, entryModule->getLoc());
      guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getValue());
    });
    burstRead0->body([&](mlir::OpBuilder &bodyBuilder,
                         llvm::ArrayRef<mlir::BlockArgument> args) {
      bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
    });
    burstRead0->finalize();

    // dummy burst_read_1 method, return a dummy value 0
    auto *burstRead1 =
        entryModule->addMethod("burst_read_1", {{"addr", u32Type}}, {u64Type});
    burstRead1->guard([&](mlir::OpBuilder &guardBuilder,
                          llvm::ArrayRef<mlir::BlockArgument> args) {
      auto trueConst = UInt::constant(1, 1, guardBuilder, entryModule->getLoc());
      guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getValue());
    });
    burstRead1->body([&](mlir::OpBuilder &bodyBuilder,
                         llvm::ArrayRef<mlir::BlockArgument> args) {
      auto dummyval = UInt::constant(0, 64, bodyBuilder, entryModule->getLoc());
      bodyBuilder.create<circt::cmt2::ReturnOp>(loc, dummyval.getValue());
    });
    burstRead1->finalize();

    // dummy burst write method, do nothing
    auto *burstWrite = entryModule->addMethod(
        "burst_write", {{"addr", u32Type}, {"data", u64Type}}, {});

    burstWrite->guard([&](mlir::OpBuilder &guardBuilder,
                          llvm::ArrayRef<mlir::BlockArgument> args) {
      auto trueConst = UInt::constant(1, 1, guardBuilder, entryModule->getLoc());
      guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getValue());
    });

    burstWrite->body([&](mlir::OpBuilder &bodyBuilder,
                        llvm::ArrayRef<mlir::BlockArgument> args) {
      bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
    });

    burstWrite->finalize();
  }

  // Expose per-bank direct read/write methods that bypass burst translation.
  // Read is split into bank_read_0 (request) and bank_read_1 (response) for 2-cycle memory.
  auto bankAddrType = UIntType::get(builder.getContext(), entryInfo.addrWidth);
  auto bankDataType = UIntType::get(builder.getContext(), entryInfo.dataWidth);

  for (size_t bankIdx = 0; bankIdx < entryInfo.numBanks; ++bankIdx) {
    std::string bankRead0Name = "bank_read_0_" + std::to_string(bankIdx);
    std::string bankRead1Name = "bank_read_1_" + std::to_string(bankIdx);
    std::string bankWriteName = "bank_write_" + std::to_string(bankIdx);

    // bank_read_0: Initiate read request with address (calls wrapper's bank_read_0)
    auto *bankRead0Method = entryModule->addMethod(
        bankRead0Name, {{"addr", bankAddrType}}, {});

    bankRead0Method->guard([&](mlir::OpBuilder &guardBuilder,
                               llvm::ArrayRef<mlir::BlockArgument> /*args*/) {
      auto trueConst =
          UInt::constant(1, 1, guardBuilder, entryModule->getLoc());
      guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getValue());
    });

    bankRead0Method->body([&,
                           bankIdx](mlir::OpBuilder &bodyBuilder,
                                    llvm::ArrayRef<mlir::BlockArgument> args) {
      auto addr = Signal(args[0], &bodyBuilder, entryModule->getLoc());
      auto calleeSymbol = mlir::FlatSymbolRefAttr::get(
          bodyBuilder.getContext(), wrapperInstances[bankIdx]->getName());
      auto methodSymbol =
          mlir::FlatSymbolRefAttr::get(bodyBuilder.getContext(), "bank_read_0");
      bodyBuilder.create<circt::cmt2::CallOp>(
          loc, mlir::TypeRange{}, mlir::ValueRange{addr.getValue()},
          calleeSymbol, methodSymbol, bodyBuilder.getArrayAttr({}),
          bodyBuilder.getArrayAttr({}));
      bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
    });

    bankRead0Method->finalize();

    // bank_read_1: Get read data (calls wrapper's bank_read_1 value)
    auto *bankRead1Value = entryModule->addValue(
        bankRead1Name, {bankDataType});

    bankRead1Value->guard([&](mlir::OpBuilder &guardBuilder) {
      auto trueConst =
          UInt::constant(1, 1, guardBuilder, entryModule->getLoc());
      guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getValue());
    });

    bankRead1Value->body([&, bankIdx](mlir::OpBuilder &bodyBuilder) {
      auto calleeSymbol = mlir::FlatSymbolRefAttr::get(
          bodyBuilder.getContext(), wrapperInstances[bankIdx]->getName());
      auto valueSymbol =
          mlir::FlatSymbolRefAttr::get(bodyBuilder.getContext(), "bank_read_1");
      auto callOp = bodyBuilder.create<circt::cmt2::CallOp>(
          loc, mlir::TypeRange{bankDataType}, mlir::ValueRange{},
          calleeSymbol, valueSymbol, bodyBuilder.getArrayAttr({}),
          bodyBuilder.getArrayAttr({}));
      bodyBuilder.create<circt::cmt2::ReturnOp>(loc, callOp.getResult(0));
    });

    bankRead1Value->finalize();

    auto *bankWriteMethod = entryModule->addMethod(
        bankWriteName, {{"addr", bankAddrType}, {"data", bankDataType}}, {});

    bankWriteMethod->guard([&](mlir::OpBuilder &guardBuilder,
                               llvm::ArrayRef<mlir::BlockArgument> /*args*/) {
      auto trueConst =
          UInt::constant(1, 1, guardBuilder, entryModule->getLoc());
      guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getValue());
    });

    bankWriteMethod->body([&,
                           bankIdx](mlir::OpBuilder &bodyBuilder,
                                    llvm::ArrayRef<mlir::BlockArgument> args) {
      auto addr = Signal(args[0], &bodyBuilder, entryModule->getLoc());
      auto data = Signal(args[1], &bodyBuilder, entryModule->getLoc());
      auto calleeSymbol = mlir::FlatSymbolRefAttr::get(
          bodyBuilder.getContext(), wrapperInstances[bankIdx]->getName());
      auto methodSymbol =
          mlir::FlatSymbolRefAttr::get(bodyBuilder.getContext(), "bank_write");
      bodyBuilder.create<circt::cmt2::CallOp>(
          loc, mlir::TypeRange{},
          mlir::ValueRange{addr.getValue(), data.getValue()}, calleeSymbol,
          methodSymbol, bodyBuilder.getArrayAttr({}),
          bodyBuilder.getArrayAttr({}));
      bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
    });

    bankWriteMethod->finalize();
  }

  return entryModule;
}

// ============================================================================
//
//                    MEMORY POOL (CONTAIN MEMORY ENTRY)
//
// ============================================================================

/// Generate burst access logic (address decoding and bank selection)
void APSToCMT2GenPass::generateBurstAccessLogic(
    Module *poolModule, const llvm::SmallVector<MemoryEntryInfo> &memEntryInfos,
    Circuit &circuit, Clock clk, Reset rst) {

  auto &builder = poolModule->getBuilder();
  auto loc = poolModule->getLoc();
  auto u64Type = UIntType::get(builder.getContext(), 64);
  auto u32Type = UIntType::get(builder.getContext(), 32);

  // Step 1: Create all memory entry submodules
  // Save the insertion point first, since circuit.addModule() will change it
  auto savedIP = builder.saveInsertionPoint();

  llvm::SmallVector<Module *, 4> entryModules;
  for (const auto &entryInfo : memEntryInfos) {
    // Collect bank names
    llvm::SmallVector<std::string, 4> bankNames;
    for (size_t i = 0; i < entryInfo.numBanks; ++i) {
      bankNames.push_back(entryInfo.name + "_" + std::to_string(i));
    }

    // Create submodule for this memory entry (this will change insertion point)
    auto *entryMod =
        generateMemoryEntryModule(entryInfo, circuit, clk, rst, bankNames);
    entryModules.push_back(entryMod);
  }

  // Step 2: Restore insertion point back to poolModule's body before creating
  // instances
  builder.restoreInsertionPoint(savedIP);

  llvm::SmallVector<Instance *, 4> entryInstances;
  for (size_t i = 0; i < memEntryInfos.size(); ++i) {
    const auto &entryInfo = memEntryInfos[i];
    auto *entryMod = entryModules[i];

    std::string instanceName = "inst_" + entryInfo.name;

    auto *entryInst = poolModule->addInstance(instanceName, entryMod,
                                              {clk.getValue(), rst.getValue()});
    entryInstances.push_back(entryInst);
  }

  // Create a register to store the burst read address for use in burst_read_1
  auto savedIPForReg = builder.saveInsertionPoint();
  auto *burstAddrRegMod = STLLibrary::createRegModule(32, 0, circuit);
  builder.restoreInsertionPoint(savedIPForReg);

  auto *burstAddrReg = poolModule->addInstance(
      "burst_read_addr_reg", burstAddrRegMod, {clk.getValue(), rst.getValue()});

  // Create top-level burst_read_0 method: stores addr in register, dispatches to entries
  auto *topBurstRead0 =
      poolModule->addMethod("burst_read_0", {{"addr", u32Type}}, {});

  topBurstRead0->guard([&](mlir::OpBuilder &guardBuilder,
                           llvm::ArrayRef<mlir::BlockArgument> args) {
    auto trueConst = UInt::constant(1, 1, guardBuilder, poolModule->getLoc());
    guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getValue());
  });

  topBurstRead0->body([&](mlir::OpBuilder &bodyBuilder,
                          llvm::ArrayRef<mlir::BlockArgument> args) {
    auto addr = Signal(args[0], &bodyBuilder, loc);

    // Store address in register for burst_read_1 to use
    burstAddrReg->callMethod("write", {addr.getValue()}, bodyBuilder);

    // Broadcast to all memory entries (selection happens in burst_read_1)
    for (size_t i = 0; i < memEntryInfos.size(); ++i) {
      const auto &entryInfo = memEntryInfos[i];
      auto *entryInst = entryInstances[i];

      uint32_t entryStart = entryInfo.baseAddress;

      // Calculate relative address for this entry
      auto startConst = UInt::constant(entryStart, 32, bodyBuilder, loc);
      auto relAddr = (addr - startConst).bits(31, 0);

      // Broadcast burst_read_0 to all entries
      auto calleeSymbol = mlir::FlatSymbolRefAttr::get(
          bodyBuilder.getContext(), entryInst->getName());
      auto methodSymbol = mlir::FlatSymbolRefAttr::get(
          bodyBuilder.getContext(), "burst_read_0");
      bodyBuilder.create<circt::cmt2::CallOp>(
          loc, mlir::TypeRange{}, mlir::ValueRange{relAddr.getValue()},
          calleeSymbol, methodSymbol, nullptr, nullptr);
    }

    bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
  });

  topBurstRead0->finalize();

  // Create top-level burst_read_1 method: reads addr from register, gathers from entries
  auto *topBurstRead1 =
      poolModule->addMethod("burst_read_1", {}, {u64Type});

  topBurstRead1->guard([&](mlir::OpBuilder &guardBuilder,
                           llvm::ArrayRef<mlir::BlockArgument> args) {
    auto trueConst = UInt::constant(1, 1, guardBuilder, poolModule->getLoc());
    guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getValue());
  });

  topBurstRead1->body([&](mlir::OpBuilder &bodyBuilder,
                          llvm::ArrayRef<mlir::BlockArgument> args) {
    // Read stored address from register
    auto addrValues = burstAddrReg->callValue("read", bodyBuilder);
    auto addr = Signal(addrValues[0], &bodyBuilder, loc);

    auto resultInit = UInt::constant(0, 64, bodyBuilder, loc);
    mlir::Value result = resultInit.getValue();

    for (size_t i = 0; i < memEntryInfos.size(); ++i) {
      const auto &entryInfo = memEntryInfos[i];
      auto *entryInst = entryInstances[i];

      uint32_t entryStart = entryInfo.baseAddress;
      uint32_t entryEnd = entryStart + entryInfo.bankSize;

      // Calculate relative address for this entry (32-bit)
      auto startConst = UInt::constant(entryStart, 32, bodyBuilder, loc);
      auto endConst = UInt::constant(entryEnd, 32, bodyBuilder, loc);

      auto relAddr = (addr - startConst).bits(31, 0);
      auto inRange = (addr >= startConst) & (addr < endConst);

      // Call submodule's burst_read_1 with relAddr
      auto calleeSymbol = mlir::FlatSymbolRefAttr::get(bodyBuilder.getContext(),
                                                       entryInst->getName());
      auto methodSymbol =
          mlir::FlatSymbolRefAttr::get(bodyBuilder.getContext(), "burst_read_1");
      auto callOp = bodyBuilder.create<circt::cmt2::CallOp>(
          loc, mlir::TypeRange{u64Type}, mlir::ValueRange{relAddr.getValue()},
          calleeSymbol, methodSymbol, nullptr, nullptr);
      if (callOp.getNumResults() == 0) {
        bodyBuilder.getBlock()->getParentOp()->emitError()
            << "CallOp for burst_read_1 returned no results (instance '"
            << entryInst->getName() << "')";
      }
      auto entryResult = Signal(callOp.getResult(0), &bodyBuilder, loc);

      // Use mux to conditionally include this result
      auto zeroData = UInt::constant(0, 64, bodyBuilder, loc);
      auto selectedData = inRange.mux(entryResult, zeroData);

      result = (Signal(result, &bodyBuilder, loc) | selectedData).getValue();
    }

    bodyBuilder.create<circt::cmt2::ReturnOp>(loc, result);
  });

  topBurstRead1->finalize();

  // Create top-level burst_write method that dispatches to correct memory entry
  auto *topBurstWrite = poolModule->addMethod(
      "burst_write", {{"addr", u32Type}, {"data", u64Type}}, {}
      // No return value
  );

  topBurstWrite->guard([&](mlir::OpBuilder &guardBuilder,
                           llvm::ArrayRef<mlir::BlockArgument> args) {
    auto trueConst = UInt::constant(1, 1, guardBuilder, poolModule->getLoc());
    guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getValue());
  });

  topBurstWrite->body([&](mlir::OpBuilder &bodyBuilder,
                          llvm::ArrayRef<mlir::BlockArgument> args) {
    auto addr = Signal(args[0], &bodyBuilder, poolModule->getLoc());
    auto data = Signal(args[1], &bodyBuilder, poolModule->getLoc());

    // Dispatch to all memory entries based on address range
    for (size_t i = 0; i < memEntryInfos.size(); ++i) {
      const auto &entryInfo = memEntryInfos[i];
      auto *entryInst = entryInstances[i];

      uint32_t entryStart = entryInfo.baseAddress;
      uint32_t entryEnd = entryStart + entryInfo.bankSize;

      // Calculate relative address for this entry (32-bit)
      auto startConst =
          UInt::constant(entryStart, 32, bodyBuilder, poolModule->getLoc());
      auto endConst =
          UInt::constant(entryEnd, 32, bodyBuilder, poolModule->getLoc());

      auto relAddr = (addr - startConst).bits(31, 0);
      auto inRange = (addr >= startConst) & (addr < endConst);

      // Only call submodule's burst_write if address is in range
      // Use If to conditionally execute the call
      If(
          inRange,
          [&](mlir::OpBuilder &thenBuilder) {
            auto calleeSymbol = mlir::FlatSymbolRefAttr::get(
                thenBuilder.getContext(), entryInst->getName());
            auto methodSymbol = mlir::FlatSymbolRefAttr::get(
                thenBuilder.getContext(), "burst_write");
            thenBuilder.create<circt::cmt2::CallOp>(
                loc, mlir::TypeRange{},
                mlir::ValueRange{relAddr.getValue(), data.getValue()},
                calleeSymbol, methodSymbol, nullptr, nullptr);
          },
          bodyBuilder, loc);
    }

    bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
  });

  topBurstWrite->finalize();

  // Expose direct per-bank read/write methods at the top level (no burst
  // translation). Read is split into read_0 (request) and read_1 (response)
  // for 2-cycle memory.
  for (size_t entryIdx = 0; entryIdx < memEntryInfos.size(); ++entryIdx) {
    const auto &entryInfo = memEntryInfos[entryIdx];
    auto bankAddrType =
        UIntType::get(builder.getContext(), entryInfo.addrWidth);
    auto bankDataType =
        UIntType::get(builder.getContext(), entryInfo.dataWidth);

    for (size_t bankIdx = 0; bankIdx < entryInfo.numBanks; ++bankIdx) {
      std::string topRead0Name =
          entryInfo.name + "_" + std::to_string(bankIdx) + "_read_0";
      std::string topRead1Name =
          entryInfo.name + "_" + std::to_string(bankIdx) + "_read_1";
      std::string topWriteName =
          entryInfo.name + "_" + std::to_string(bankIdx) + "_write";
      if (entryInfo.numBanks == 1) {
        // Not array_partitioned, we should not add a bankIdx on it
        topRead0Name = entryInfo.name + "_read_0";
        topRead1Name = entryInfo.name + "_read_1";
        topWriteName = entryInfo.name + "_write";
      }
      std::string entryRead0Name = "bank_read_0_" + std::to_string(bankIdx);
      std::string entryRead1Name = "bank_read_1_" + std::to_string(bankIdx);
      std::string entryWriteName = "bank_write_" + std::to_string(bankIdx);

      // topRead0: Initiate read request with address
      auto *topBankRead0 = poolModule->addMethod(
          topRead0Name, {{"addr", bankAddrType}}, {});

      topBankRead0->guard([&](mlir::OpBuilder &guardBuilder,
                              llvm::ArrayRef<mlir::BlockArgument> /*args*/) {
        auto trueConst =
            UInt::constant(1, 1, guardBuilder, poolModule->getLoc());
        guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getValue());
      });

      topBankRead0->body([&, entryIdx, entryRead0Name](
                             mlir::OpBuilder &bodyBuilder,
                             llvm::ArrayRef<mlir::BlockArgument> args) {
        auto addr = Signal(args[0], &bodyBuilder, poolModule->getLoc());
        auto calleeSymbol = mlir::FlatSymbolRefAttr::get(
            bodyBuilder.getContext(), entryInstances[entryIdx]->getName());
        auto methodSymbol = mlir::FlatSymbolRefAttr::get(
            bodyBuilder.getContext(), entryRead0Name);
        bodyBuilder.create<circt::cmt2::CallOp>(
            loc, mlir::TypeRange{}, mlir::ValueRange{addr.getValue()},
            calleeSymbol, methodSymbol, bodyBuilder.getArrayAttr({}),
            bodyBuilder.getArrayAttr({}));
        bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
      });

      topBankRead0->finalize();

      // topRead1: Get read data (value)
      auto *topBankRead1 = poolModule->addValue(topRead1Name, {bankDataType});

      topBankRead1->guard([&](mlir::OpBuilder &guardBuilder) {
        auto trueConst =
            UInt::constant(1, 1, guardBuilder, poolModule->getLoc());
        guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getValue());
      });

      topBankRead1->body([&, entryIdx, entryRead1Name](
                             mlir::OpBuilder &bodyBuilder) {
        auto calleeSymbol = mlir::FlatSymbolRefAttr::get(
            bodyBuilder.getContext(), entryInstances[entryIdx]->getName());
        auto valueSymbol = mlir::FlatSymbolRefAttr::get(bodyBuilder.getContext(),
                                                        entryRead1Name);
        auto callOp = bodyBuilder.create<circt::cmt2::CallOp>(
            loc, mlir::TypeRange{bankDataType}, mlir::ValueRange{},
            calleeSymbol, valueSymbol, bodyBuilder.getArrayAttr({}),
            bodyBuilder.getArrayAttr({}));
        bodyBuilder.create<circt::cmt2::ReturnOp>(loc, callOp.getResult(0));
      });

      topBankRead1->finalize();

      auto *topBankWrite = poolModule->addMethod(
          topWriteName, {{"addr", bankAddrType}, {"data", bankDataType}}, {});

      topBankWrite->guard([&](mlir::OpBuilder &guardBuilder,
                              llvm::ArrayRef<mlir::BlockArgument> /*args*/) {
        auto trueConst =
            UInt::constant(1, 1, guardBuilder, poolModule->getLoc());
        guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueConst.getValue());
      });

      topBankWrite->body([&, entryIdx, entryWriteName](
                             mlir::OpBuilder &bodyBuilder,
                             llvm::ArrayRef<mlir::BlockArgument> args) {
        auto addr = Signal(args[0], &bodyBuilder, poolModule->getLoc());
        auto data = Signal(args[1], &bodyBuilder, poolModule->getLoc());
        auto calleeSymbol = mlir::FlatSymbolRefAttr::get(
            bodyBuilder.getContext(), entryInstances[entryIdx]->getName());
        auto methodSymbol = mlir::FlatSymbolRefAttr::get(
            bodyBuilder.getContext(), entryWriteName);
        bodyBuilder.create<circt::cmt2::CallOp>(
            loc, mlir::TypeRange{},
            mlir::ValueRange{addr.getValue(), data.getValue()}, calleeSymbol,
            methodSymbol, bodyBuilder.getArrayAttr({}),
            bodyBuilder.getArrayAttr({}));
        bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
      });

      topBankWrite->finalize();
    }
  }
}

void APSToCMT2GenPass::addRoCCAndMemoryMethodToMainModule(
    Module *mainModule, Instance *roccInstance, Instance *hellaMemInstance) {
  auto &builder = mainModule->getBuilder();
  auto roccCmdBundleType = getRoccCmdBundleType(builder);
  auto hellaRespBundleType = getHellaRespBundleType(builder);
  auto loc = mainModule->getLoc();
  auto *roccCmdMethod = mainModule->addMethod(
      "rocc_cmd", {{{"rocc_cmd", roccCmdBundleType}}}, {});

  roccCmdMethod->guard([&](mlir::OpBuilder &guardBuilder,
                           llvm::ArrayRef<mlir::BlockArgument> args) {
    auto trueVal = UInt::constant(1, 1, guardBuilder, loc);
    guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueVal.getValue());
  });
  roccCmdMethod->body([&](mlir::OpBuilder &bodyBuilder,
                          llvm::ArrayRef<mlir::BlockArgument> args) {
    auto arg = args[0];
    auto loc = bodyBuilder.getUnknownLoc();
    roccInstance->callMethod("cmd_from_bus", arg, bodyBuilder);
    bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
  });
  roccCmdMethod->finalize();

  auto *hellaRespMethod = mainModule->addMethod(
      "hella_resp", {{"hella_resp", hellaRespBundleType}}, {});

  hellaRespMethod->guard([&](mlir::OpBuilder &guardBuilder,
                             llvm::ArrayRef<mlir::BlockArgument> args) {
    auto trueVal = UInt::constant(1, 1, guardBuilder, loc);
    guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueVal.getValue());
  });
  hellaRespMethod->body([&](mlir::OpBuilder &bodyBuilder,
                            llvm::ArrayRef<mlir::BlockArgument> args) {
    auto arg = args[0];
    auto loc = bodyBuilder.getUnknownLoc();
    hellaMemInstance->callMethod("resp_from_bus", arg, bodyBuilder);
    bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
  });
  hellaRespMethod->finalize();
}

/// Get memref.global by symbol name
memref::GlobalOp APSToCMT2GenPass::getGlobalMemRef(mlir::Operation *scope,
                                                   StringRef symbolName) {
  if (!scope) {
    llvm::dbgs() << "DEBUG getGlobalMemRef: scope is null\n";
    return nullptr;
  }
  memref::GlobalOp foundOp = nullptr;
  scope->walk([&](memref::GlobalOp globalOp) {
    if (globalOp.getSymName() == symbolName) {
      foundOp = globalOp;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });

  if (foundOp) {
    llvm::dbgs() << "DEBUG: Found symbol via manual walk\n";
  }
  
  return foundOp;
}

/// Generate the CMT2 memory pool module
MemoryPoolResult
APSToCMT2GenPass::generateMemoryPool(Circuit &circuit, ModuleOp moduleOp,
                                     aps::MemoryMapOp memoryMapOp) {
  llvm::dbgs() << "DEBUG: generateMemoryPool() started\n";
  MLIRContext *context = moduleOp.getContext();
  OpBuilder builder(context);

  // Collect all memory entries
  llvm::SmallVector<aps::MemEntryOp> memEntries;
  llvm::dbgs() << "DEBUG: About to collect memory entries\n";

  if (memoryMapOp) {
    llvm::dbgs() << "DEBUG: Memory map region has "
                 << memoryMapOp.getRegion().getBlocks().size() << " blocks\n";

    // Safely iterate through operations
    for (auto &block : memoryMapOp.getRegion()) {
      for (auto &op : block) {
        if (auto entry = dyn_cast<aps::MemEntryOp>(op)) {
          memEntries.push_back(entry);
          llvm::dbgs() << "DEBUG: Found memory entry: " << entry.getName()
                       << "\n";
        }
      }
    }

    llvm::dbgs() << "DEBUG: Collected " << memEntries.size()
                 << " memory entries\n";
  }

  if (memEntries.empty()) {
    llvm::dbgs() << "DEBUG: No memory entries found, will generate rules "
                    "without memory\n";
  }

  // Generate memory pool module
  llvm::dbgs() << "DEBUG: Creating ScratchpadMemoryPool module\n";
  auto *poolModule = circuit.addModule("ScratchpadMemoryPool");

  // Add clock and reset
  llvm::dbgs() << "DEBUG: Adding clock and reset arguments\n";
  Clock clk = poolModule->addClockArgument("clk");
  Reset rst = poolModule->addResetArgument("rst");

  // Store instances for each memory entry (for address decoding)
  llvm::SmallVector<MemoryEntryInfo> memEntryInfos;

  // Create memory entry map for fast lookup by name
  llvm::DenseMap<llvm::StringRef, MemoryEntryInfo> memEntryMap;

  // For each memory entry, create SRAM banks (only if entries exist)
  for (auto memEntry : memEntries) {
    auto bankSymbols = memEntry.getBankSymbols();
    uint32_t numBanks = memEntry.getNumBanks();
    uint32_t baseAddress = memEntry.getBaseAddress();
    uint32_t bankSize = memEntry.getBankSize();
    uint32_t cyclic = memEntry.getCyclic();

    // Get parameters from the first bank (all banks should have same type)
    if (bankSymbols.empty()) {
      llvm::errs() << "Error: Memory entry " << memEntry.getName()
                   << " has no banks\n";
      continue;
    }

    auto firstBankSymAttr = llvm::dyn_cast<FlatSymbolRefAttr>(bankSymbols[0]);
    if (!firstBankSymAttr) {
      llvm::errs() << "Error: Invalid bank symbol attribute\n";
      continue;
    }

    // Look up the memref.global for this bank
    StringRef bankSymbolName = firstBankSymAttr.getValue();
    llvm::dbgs() << "DEBUG: Looking up memref.global for bank symbol: '" << bankSymbolName << "'\n";

    auto globalOp = getGlobalMemRef(moduleOp, bankSymbolName);
    if (!globalOp) {
      llvm::errs() << "Error: Could not find memref.global for bank "
                   << bankSymbolName << "\n";
    }

    // Extract memory parameters from the memref type
    int dataWidth = 32; // defaults
    int addrWidth = 10;
    int depth = 1024;

    if (!extractMemoryParameters(globalOp, dataWidth, addrWidth, depth)) {
      llvm::errs() << "Error: Could not extract memory parameters from "
                   << firstBankSymAttr.getValue() << ", using defaults\n";
    }

    // We only check addrWidth here. If it's array partitioned to be a
    // single memory, we should convert it before to a globalload/store,
    // and should not treat it as a memory here.
    if (addrWidth == 0) {
      continue; // Only one element, not a memory, skip this entry
    }

    // Create MemoryEntryInfo for this entry
    // Note: Bank instances will be created inside submodules, not here
    MemoryEntryInfo entryInfo;
    entryInfo.name = memEntry.getName().str();
    entryInfo.baseAddress = baseAddress;
    entryInfo.bankSize = bankSize;
    entryInfo.numBanks = numBanks;
    entryInfo.isCyclic = (cyclic != 0);
    entryInfo.dataWidth = dataWidth;
    entryInfo.addrWidth = addrWidth;
    entryInfo.depth = depth;

    // Add this memory entry info to the list and map
    memEntryInfos.push_back(entryInfo);
    memEntryMap[memEntry.getName()] = std::move(entryInfo);
  }

  // Now generate address decoding logic for burst access
  // This will create memory entry submodules with bank instances
  generateBurstAccessLogic(poolModule, memEntryInfos, circuit, clk, rst);

  return MemoryPoolResult{poolModule, std::move(memEntryMap)};
}

} // namespace mlir