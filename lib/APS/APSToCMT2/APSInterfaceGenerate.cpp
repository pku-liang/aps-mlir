#include "APS/APSToCMT2.h"
#include <string>

#define DEBUG_TYPE "aps-memory-pool-gen"

namespace mlir {

using namespace mlir;
using namespace mlir::tor;
using namespace circt::cmt2::ecmt2;
using namespace circt::cmt2::ecmt2::stl;
using namespace circt::firrtl;

void APSToCMT2GenPass::addBurstMemoryInterface(Circuit &circuit) {
  auto &context = circuit.getContext();
  auto *burstMemoryInterface = circuit.addInterface("BurstDMAController");
  auto u32Type = UIntType::get(&context, 32);
  auto u8Type = UIntType::get(&context, 8);
  auto u4Type = UIntType::get(&context, 4);
  auto u1Type = UIntType::get(&context, 1);
  for (int i = 0; i < 2; i++) { // dual channel tilelink
    burstMemoryInterface->addMethod(
        "cpu_to_isax_ch" + std::to_string(i),
        {{"cpu_addr", u32Type}, {"isax_addr", u32Type}, {"length", u4Type}, 
        {"stride_x", u8Type}, {"stride_y", u8Type}}, {});
    burstMemoryInterface->addMethod(
        "isax_to_cpu_ch" + std::to_string(i),
        {{"cpu_addr", u32Type}, {"isax_addr", u32Type}, {"length", u4Type},
        {"stride_x", u8Type}, {"stride_y", u8Type}}, {});
    burstMemoryInterface->addValue(
        // This will not ready if burst engine is running
        // So the main operation can be stucked
        "poll_for_idle_ch" + std::to_string(i), {}, {TypeAttr::get(u1Type)});
  }
}

void APSToCMT2GenPass::addRoccAndHellaMemoryInterface(Circuit &circuit) {
  auto roccRespInterface = circuit.addInterface("roccRespItfc");
  auto &builder = circuit.getBuilder();
  auto roccRespBundleType = getRoccRespBundleType(builder);
  roccRespInterface->addMethod("rocc_resp_to_bus",
                               {{"result", roccRespBundleType}}, {});

  auto hellaCmdInterface = circuit.addInterface("hellaCmdItfc");
  auto hellaCmdBundleType = getHellaCmdBundleType(builder);
  hellaCmdInterface->addMethod("hella_cmd_to_bus",
                               {{"cmd", hellaCmdBundleType}}, {});
}

/// Add burst read/write methods to main module
void APSToCMT2GenPass::addBurstMethodsToMainModule(Module *mainModule,
                                                   Instance *poolInstance) {
  auto &builder = mainModule->getBuilder();
  auto *context = builder.getContext();

  // Add burst_read_0 method (initiate read request with 32-bit address)
  auto *burstRead0Method = mainModule->addMethod(
      "burst_read_0", {{"addr", circt::firrtl::UIntType::get(context, 32)}},
      {});

  burstRead0Method->guard(
      [](mlir::OpBuilder &b, llvm::ArrayRef<mlir::BlockArgument> args) {
        auto loc = b.getUnknownLoc();
        auto one = b.create<circt::firrtl::ConstantOp>(
            loc, circt::firrtl::UIntType::get(b.getContext(), 1),
            llvm::APInt(1, 1));
        b.create<circt::cmt2::ReturnOp>(loc, mlir::ValueRange{one});
      });

  burstRead0Method->body(
      [](mlir::OpBuilder &b, llvm::ArrayRef<mlir::BlockArgument> args) {
        auto loc = b.getUnknownLoc();
        auto addrArg = args[0];

        // Call the scratchpad pool burst_read_0 method
        b.create<circt::cmt2::CallOp>(
            loc, mlir::TypeRange{}, mlir::ValueRange{addrArg},
            mlir::SymbolRefAttr::get(b.getContext(), "scratchpad_pool"),
            mlir::SymbolRefAttr::get(b.getContext(), "burst_read_0"),
            mlir::ArrayAttr(), mlir::ArrayAttr());

        b.create<circt::cmt2::ReturnOp>(loc);
      });
  burstRead0Method->finalize();

  // Add burst_read_1 method (get read data, 64-bit return)
  auto *burstRead1Method = mainModule->addMethod(
      "burst_read_1", {},
      {circt::firrtl::UIntType::get(context, 64)});

  burstRead1Method->guard(
      [](mlir::OpBuilder &b, llvm::ArrayRef<mlir::BlockArgument> args) {
        auto loc = b.getUnknownLoc();
        auto one = b.create<circt::firrtl::ConstantOp>(
            loc, circt::firrtl::UIntType::get(b.getContext(), 1),
            llvm::APInt(1, 1));
        b.create<circt::cmt2::ReturnOp>(loc, mlir::ValueRange{one});
      });

  burstRead1Method->body(
      [](mlir::OpBuilder &b, llvm::ArrayRef<mlir::BlockArgument> args) {
        auto loc = b.getUnknownLoc();

        // Call the scratchpad pool burst_read_1 method
        auto result = b.create<circt::cmt2::CallOp>(
            loc, circt::firrtl::UIntType::get(b.getContext(), 64),
            mlir::ValueRange{},
            mlir::SymbolRefAttr::get(b.getContext(), "scratchpad_pool"),
            mlir::SymbolRefAttr::get(b.getContext(), "burst_read_1"),
            mlir::ArrayAttr(), mlir::ArrayAttr());

        b.create<circt::cmt2::ReturnOp>(loc, result.getResult(0));
      });
  burstRead1Method->finalize();

  // Add burst write method (32-bit address, 64-bit data)
  auto *burstWriteMethod = mainModule->addMethod(
      "burst_write",
      {{"addr", circt::firrtl::UIntType::get(context, 32)},
       {"data", circt::firrtl::UIntType::get(context, 64)}},
      {});

  burstWriteMethod->guard(
      [](mlir::OpBuilder &b, llvm::ArrayRef<mlir::BlockArgument> args) {
        auto loc = b.getUnknownLoc();
        auto one = b.create<circt::firrtl::ConstantOp>(
            loc, circt::firrtl::UIntType::get(b.getContext(), 1),
            llvm::APInt(1, 1));
        b.create<circt::cmt2::ReturnOp>(loc, mlir::ValueRange{one});
      });

  burstWriteMethod->body(
      [](mlir::OpBuilder &b, llvm::ArrayRef<mlir::BlockArgument> args) {
        auto loc = b.getUnknownLoc();
        auto addrArg = args[0];
        auto dataArg = args[1];

        // Call the scratchpad pool burst_write method
        b.create<circt::cmt2::CallOp>(
            loc, mlir::TypeRange{}, mlir::ValueRange{addrArg, dataArg},
            mlir::SymbolRefAttr::get(b.getContext(), "scratchpad_pool"),
            mlir::SymbolRefAttr::get(b.getContext(), "burst_write"),
            mlir::ArrayAttr(), mlir::ArrayAttr());

        b.create<circt::cmt2::ReturnOp>(loc);
      });
  burstWriteMethod->finalize();
}

BundleType APSToCMT2GenPass::getHellaRespBundleType(Builder &builder) {
  auto *context = builder.getContext();
  return BundleType::get(
      context, {BundleType::BundleElement{builder.getStringAttr("data"), false,
                                          UIntType::get(context, 32)},
                BundleType::BundleElement{builder.getStringAttr("tag"), false,
                                          UIntType::get(context, 8)},
                BundleType::BundleElement{builder.getStringAttr("cmd"), false,
                                          UIntType::get(context, 5)},
                BundleType::BundleElement{builder.getStringAttr("size"), false,
                                          UIntType::get(context, 2)},
                BundleType::BundleElement{builder.getStringAttr("signed"),
                                          false, UIntType::get(context, 1)}});
}

BundleType APSToCMT2GenPass::getHellaUserCmdBundleType(Builder &builder) {
  auto *context = builder.getContext();
  return BundleType::get(
      context, {BundleType::BundleElement{builder.getStringAttr("addr"), false,
                                          UIntType::get(context, 32)},
                BundleType::BundleElement{builder.getStringAttr("cmd"), false,
                                          UIntType::get(context, 1)},
                BundleType::BundleElement{builder.getStringAttr("size"), false,
                                          UIntType::get(context, 2)},
                BundleType::BundleElement{builder.getStringAttr("data"), false,
                                          UIntType::get(context, 32)},
                BundleType::BundleElement{builder.getStringAttr("mask"), false,
                                          UIntType::get(context, 4)},
                BundleType::BundleElement{builder.getStringAttr("tag"), false,
                                          UIntType::get(context, 8)}});
}

BundleType APSToCMT2GenPass::getHellaCmdBundleType(Builder &builder) {
  auto *context = builder.getContext();
  return BundleType::get(
      context, {BundleType::BundleElement{builder.getStringAttr("addr"), false,
                                          UIntType::get(context, 32)},
                BundleType::BundleElement{builder.getStringAttr("tag"), false,
                                          UIntType::get(context, 8)},
                BundleType::BundleElement{builder.getStringAttr("cmd"), false,
                                          UIntType::get(context, 5)},
                BundleType::BundleElement{builder.getStringAttr("size"), false,
                                          UIntType::get(context, 2)},
                BundleType::BundleElement{builder.getStringAttr("signed"),
                                          false, UIntType::get(context, 1)},
                BundleType::BundleElement{builder.getStringAttr("phys"), false,
                                          UIntType::get(context, 1)},
                BundleType::BundleElement{builder.getStringAttr("data"), false,
                                          UIntType::get(context, 32)},
                BundleType::BundleElement{builder.getStringAttr("mask"), false,
                                          UIntType::get(context, 4)}});
}

BundleType APSToCMT2GenPass::getHellaUserRespBundleType(Builder &builder) {
  auto *context = builder.getContext();
  return BundleType::get(
      context, {BundleType::BundleElement{builder.getStringAttr("data"), false,
                                          UIntType::get(context, 32)},
                BundleType::BundleElement{builder.getStringAttr("tag"), false,
                                          UIntType::get(context, 8)}});
}

/// Generate Memory Translator module that bridges HellaCache interface with
/// User Memory Protocol
Module *APSToCMT2GenPass::generateMemoryAdapter(Circuit &circuit) {
  auto *translatorModule = circuit.addModule("MemoryTranslator");
  auto &builder = translatorModule->getBuilder();
  auto loc = translatorModule->getLoc();

  // Add clock and reset arguments
  Clock clk = translatorModule->addClockArgument("clk");
  Reset rst = translatorModule->addResetArgument("rst");
  auto hellaCmdItfcDecl =
      translatorModule->defineInterfaceDecl("hella_cmd", "hellaCmdItfc");

  // Define types for the protocols (using inline definitions to avoid unused
  // variables)

  // Create bundle types following Rust patterns
  // UserMemoryCmd: {addr: u32, cmd: u1, size: u2, data: u32, mask: u4, tag: u8}
  auto userCmdBundleType = getHellaUserCmdBundleType(builder);

  // HellaCacheCmd: {addr: u32, tag: u8, cmd: u5, size: u2, signed: u1, phys:
  // u1, data: u32, mask: u4}
  auto hellaCmdBundleType = getHellaCmdBundleType(builder);

  // HellaCacheResp: {data: u32, tag: u8, cmd: u5, size: u2, signed: u1}
  auto hellaRespBundleType = getHellaRespBundleType(builder);
  // UserMemoryResp: {data: u32, tag: u8}
  auto userRespBundleType = getHellaUserRespBundleType(builder);

  // Create external FIFO and register modules
  auto savedIP = builder.saveInsertionPoint();

  // Command queue (FIFO1Push for HellaCache commands - 85 bits total)
  auto hellaCmdFifoMod = STLLibrary::createFIFO2IModule(85, circuit);

  // Response buffer registers (grouped by bitwidth for reuse)
  auto reg32Mod =
      STLLibrary::createRegModule(32, 0, circuit); // 32-bit registers for data
  auto reg8Mod =
      STLLibrary::createRegModule(8, 0, circuit); // 32-bit registers for tags
  auto reg1Mod =
      STLLibrary::createRegModule(1, 0, circuit); // 1-bit registers for flags

  // Wire modules for logic (grouped by bitwidth)
  auto wire1Mod =
      STLLibrary::createWireModule(1, circuit); // 1-bit wires for logic
  auto wire32Mod =
      STLLibrary::createWireModule(32, circuit); // 1-bit wires for logic

  builder.restoreInsertionPoint(savedIP);

  // Create instances
  auto *hellaCmdFifo = translatorModule->addInstance(
      "hella_cmd_fifo", hellaCmdFifoMod, {clk.getValue(), rst.getValue()});

  // Slot 0 instances (registers don't need clock/reset)
  auto *slot0DataReg = translatorModule->addInstance(
      "slot0_data_reg", reg32Mod, {clk.getValue(), rst.getValue()});
  auto *slot0TagReg = translatorModule->addInstance(
      "slot0_tag_reg", reg8Mod, {clk.getValue(), rst.getValue()});
  auto *slot0TxdReg = translatorModule->addInstance(
      "slot0_txd_reg", reg1Mod, {clk.getValue(), rst.getValue()});
  auto *slot0RxdReg = translatorModule->addInstance(
      "slot0_rxd_reg", reg1Mod, {clk.getValue(), rst.getValue()});

  // Slot 1 instances (registers don't need clock/reset)
  auto *slot1DataReg = translatorModule->addInstance(
      "slot1_data_reg", reg32Mod, {clk.getValue(), rst.getValue()});
  auto *slot1TagReg = translatorModule->addInstance(
      "slot1_tag_reg", reg8Mod, {clk.getValue(), rst.getValue()});
  auto *slot1TxdReg = translatorModule->addInstance(
      "slot1_txd_reg", reg1Mod, {clk.getValue(), rst.getValue()});
  auto *slot1RxdReg = translatorModule->addInstance(
      "slot1_rxd_reg", reg1Mod, {clk.getValue(), rst.getValue()});

  // Control flag instances (registers don't need clock/reset)
  auto *newerSlotReg = translatorModule->addInstance(
      "newer_slot_reg", reg1Mod, {clk.getValue(), rst.getValue()});

  // Wire instances for logic
  auto *slot0CanCollectWire =
      translatorModule->addInstance("slot0_can_collect_wire", wire1Mod, {});
  auto *slot1CanCollectWire =
      translatorModule->addInstance("slot1_can_collect_wire", wire1Mod, {});

  auto *recvSlot0Wire =
      translatorModule->addInstance("slot0_recv_wire", wire1Mod, {});
  auto *recvSlot1Wire =
      translatorModule->addInstance("slot1_recv_wire", wire1Mod, {});
  auto *recvSlotData =
      translatorModule->addInstance("slot_recv_data", wire32Mod, {});

  // Method: resp_from_bus - receives responses from HellaCache and buffers them
  std::vector<std::pair<std::string, mlir::Type>> respFromBusArgs = {
      {"hella_resp", hellaRespBundleType}};
  auto *respFromBus =
      translatorModule->addMethod("resp_from_bus", respFromBusArgs, {});

  respFromBus->guard(
      [&](mlir::OpBuilder &guardBuilder, llvm::ArrayRef<mlir::BlockArgument>) {
        auto trueVal = UInt::constant(1, 1, guardBuilder, loc);
        guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueVal.getValue());
      });

  respFromBus->body([&](mlir::OpBuilder &bodyBuilder,
                        llvm::ArrayRef<mlir::BlockArgument> args) {
    Bundle hellaResp(args[0], &bodyBuilder, loc);

    // Extract response fields
    Signal respData = hellaResp["data"];
    Signal respTag = hellaResp["tag"];
    Signal respCmd = hellaResp["cmd"];

    // Check if this is a read response (cmd == 0)
    Signal cmdIsRead = respCmd == UInt::constant(0, 5, bodyBuilder, loc);
    Signal tagAddr = respTag; // Use tag as address for matching

    // Read current newer_slot flag
    auto newerSlotValues = newerSlotReg->callValue("read", bodyBuilder);
    Signal newerSlot = Signal(newerSlotValues[0], &bodyBuilder, loc);
    Signal slot0Tag = Signal(slot0TagReg->callValue("read", bodyBuilder)[0],
                             &bodyBuilder, loc);
    Signal slot1Tag = Signal(slot1TagReg->callValue("read", bodyBuilder)[0],
                             &bodyBuilder, loc);

    auto b0 = UInt::constant(0, 1, bodyBuilder, loc);
    auto b1 = UInt::constant(1, 1, bodyBuilder, loc);
    // Conditional logic following Rust pattern
    recvSlot0Wire->callMethod(
        "write", {(cmdIsRead & (slot0Tag == tagAddr)).getValue()}, bodyBuilder);
    recvSlot1Wire->callMethod(
        "write", {(cmdIsRead & (slot1Tag == tagAddr)).getValue()}, bodyBuilder);
    recvSlotData->callMethod("write", respData.getValue(), bodyBuilder);
    bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
  });

  respFromBus->finalize();

  auto *clearSlot0 = translatorModule->addRule("recv_slot_0");
  clearSlot0->guard([&](mlir::OpBuilder &guardBuilder) {
    auto ret = recvSlot0Wire->callValue("read", guardBuilder);
    guardBuilder.create<circt::cmt2::ReturnOp>(loc, ret);
  });
  clearSlot0->body([&](mlir::OpBuilder &bodyBuilder) {
    slot0RxdReg->callMethod("write",
                            {UInt::constant(1, 1, bodyBuilder, loc).getValue()},
                            bodyBuilder);
    auto resp = recvSlotData->callValue("read", bodyBuilder);
    slot0DataReg->callMethod("write", {resp}, bodyBuilder);
    bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
  });
  clearSlot0->finalize();

  auto *clearSlot1 = translatorModule->addRule("recv_slot_1");
  clearSlot1->guard([&](mlir::OpBuilder &guardBuilder) {
    auto ret = recvSlot1Wire->callValue("read", guardBuilder);
    guardBuilder.create<circt::cmt2::ReturnOp>(loc, ret);
  });
  clearSlot1->body([&](mlir::OpBuilder &bodyBuilder) {
    slot1RxdReg->callMethod("write",
                            {UInt::constant(1, 1, bodyBuilder, loc).getValue()},
                            bodyBuilder);
    auto resp = recvSlotData->callValue("read", bodyBuilder);
    slot1DataReg->callMethod("write", {resp}, bodyBuilder);
    bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
  });
  clearSlot1->finalize();

  // Method: cmd_from_user - receives user memory commands and translates to
  // HellaCache format
  std::vector<std::pair<std::string, mlir::Type>> cmdFromUserArgs = {
      {"user_cmd", userCmdBundleType}};
  auto *cmdFromUser =
      translatorModule->addMethod("cmd_from_user", cmdFromUserArgs, {});

  cmdFromUser->guard(
      [&](mlir::OpBuilder &guardBuilder, llvm::ArrayRef<mlir::BlockArgument>) {
        auto trueVal = UInt::constant(1, 1, guardBuilder, loc);
        guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueVal.getValue());
      });

  cmdFromUser->body([&](mlir::OpBuilder &bodyBuilder,
                        llvm::ArrayRef<mlir::BlockArgument> args) {
    Bundle userCmd(args[0], &bodyBuilder, loc);

    // Extract user command fields
    Signal userAddr = userCmd["addr"];
    Signal userTag = userCmd["tag"];
    Signal userCmdType = userCmd["cmd"];
    Signal userSize = userCmd["size"];
    Signal userData = userCmd["data"];
    Signal userMask = userCmd["mask"];

    // Translate to HellaCache format (following Rust logic)
    // cmd = user_cmd (0=read, 1=write) -> same, but 5-bit wide
    Signal hellaCmd = userCmdType.pad(5);
    // signed = false, phys = false (following Rust)
    Signal hellaSigned = UInt::constant(0, 1, bodyBuilder, loc);
    Signal hellaPhys = UInt::constant(0, 1, bodyBuilder, loc);

    // Pack the bundle fields into a 85-bit value for FIFO
    // Layout: data(32) + mask(4) + phys(1) + signed(1) + size(2) + cmd(5) +
    // tag(8) + addr(32) = 85 bits (with padding)
    Signal packedHellaCmd = userData.cat(userMask)
                                .cat(hellaPhys)
                                .cat(hellaSigned)
                                .cat(userSize)
                                .cat(hellaCmd)
                                .cat(userTag)
                                .cat(userAddr);

    // Enqueue to HellaCache command FIFO
    hellaCmdFifo->callMethod("enq", {packedHellaCmd.getValue()}, bodyBuilder);

    bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
  });

  cmdFromUser->finalize();

  // Rule: slot_0_can_collect_logic - computes when slot 0 can provide responses
  auto *slot0CanCollectLogicRule =
      translatorModule->addRule("slot_0_can_collect_logic");

  slot0CanCollectLogicRule->guard([&](mlir::OpBuilder &guardBuilder) {
    // This rule always fires (combinational logic)
    auto trueVal = UInt::constant(1, 1, guardBuilder, loc);
    guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueVal.getValue());
  });

  slot0CanCollectLogicRule->body([&](mlir::OpBuilder &bodyBuilder) {
    // Read slot 0 flags
    auto slot0TxdValues = slot0TxdReg->callValue("read", bodyBuilder);
    auto slot0RxdValues = slot0RxdReg->callValue("read", bodyBuilder);
    Signal slot0Txd = Signal(slot0TxdValues[0], &bodyBuilder, loc);
    Signal slot0Rxd = Signal(slot0RxdValues[0], &bodyBuilder, loc);

    // Read slot 1 flags for ordering comparison
    auto slot1TxdValues = slot1TxdReg->callValue("read", bodyBuilder);
    auto newerSlotValues = newerSlotReg->callValue("read", bodyBuilder);
    Signal slot1Txd = Signal(slot1TxdValues[0], &bodyBuilder, loc);
    Signal newerSlot = Signal(newerSlotValues[0], &bodyBuilder, loc);

    // Compute slot0_ready: both transmitted and received flags set
    Signal slot0Ready = slot0Txd & slot0Rxd;

    // Compute is_earlier: slot 0 is earlier if slot 1 not transmitted OR
    // slot 1 transmitted AND newer_slot == 1 (slot 0 is the newer one)
    Signal slot1NotTx = slot1Txd == UInt::constant(0, 1, bodyBuilder, loc);
    Signal slot1TxAndNewer1 =
        slot1Txd & (newerSlot == UInt::constant(1, 1, bodyBuilder, loc));
    Signal isEarlier = slot1NotTx | slot1TxAndNewer1;

    // Compute can_collect: slot is ready AND is earlier
    Signal canCollect = slot0Ready & isEarlier;

    // Write to the wire
    slot0CanCollectWire->callMethod("write", {canCollect.getValue()},
                                    bodyBuilder);

    bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
  });

  slot0CanCollectLogicRule->finalize();

  // Rule: slot_1_can_collect_logic - computes when slot 1 can provide responses
  auto *slot1CanCollectLogicRule =
      translatorModule->addRule("slot_1_can_collect_logic");

  slot1CanCollectLogicRule->guard([&](mlir::OpBuilder &guardBuilder) {
    // This rule always fires (combinational logic)
    auto trueVal = UInt::constant(1, 1, guardBuilder, loc);
    guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueVal.getValue());
  });

  slot1CanCollectLogicRule->body([&](mlir::OpBuilder &bodyBuilder) {
    // Read slot 1 flags
    auto slot1TxdValues = slot1TxdReg->callValue("read", bodyBuilder);
    auto slot1RxdValues = slot1RxdReg->callValue("read", bodyBuilder);
    Signal slot1Txd = Signal(slot1TxdValues[0], &bodyBuilder, loc);
    Signal slot1Rxd = Signal(slot1RxdValues[0], &bodyBuilder, loc);

    // Read slot 0 flags for ordering comparison
    auto slot0TxdValues = slot0TxdReg->callValue("read", bodyBuilder);
    auto newerSlotValues = newerSlotReg->callValue("read", bodyBuilder);
    Signal slot0Txd = Signal(slot0TxdValues[0], &bodyBuilder, loc);
    Signal newerSlot = Signal(newerSlotValues[0], &bodyBuilder, loc);

    // Compute slot1_ready: both transmitted and received flags set
    Signal slot1Ready = slot1Txd & slot1Rxd;

    // Compute is_earlier: slot 1 is earlier if slot 0 not transmitted OR
    // slot 0 transmitted AND newer_slot == 0 (slot 1 is the newer one)
    Signal slot0NotTx = slot0Txd == UInt::constant(0, 1, bodyBuilder, loc);
    Signal slot0TxAndNewer0 =
        slot0Txd & (newerSlot == UInt::constant(0, 1, bodyBuilder, loc));
    Signal isEarlier = slot0NotTx | slot0TxAndNewer0;

    // Compute can_collect: slot is ready AND is earlier
    Signal canCollect = slot1Ready & isEarlier;

    // Write to the wire
    slot1CanCollectWire->callMethod("write", {canCollect.getValue()},
                                    bodyBuilder);

    bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
  });

  slot1CanCollectLogicRule->finalize();

  // Method: resp_to_user - provides user memory responses (with ordering)
  // This matches the Rust method exactly - returns user response bundle to
  // output interface
  llvm::SmallVector<std::pair<std::string, mlir::Type>, 0> respToUserArgs;
  llvm::SmallVector<mlir::Type, 1> respToUserReturns = {userRespBundleType};
  auto *respToUser = translatorModule->addMethod("resp_to_user", respToUserArgs,
                                                 respToUserReturns);

  respToUser->guard([&](mlir::OpBuilder &guardBuilder,
                        llvm::ArrayRef<mlir::BlockArgument> args) {
    // Check if any slot can collect (following Rust logic)
    auto slot0CollectValues =
        slot0CanCollectWire->callValue("read", guardBuilder);
    auto slot1CollectValues =
        slot1CanCollectWire->callValue("read", guardBuilder);
    Signal slot0CanCollect = Signal(slot0CollectValues[0], &guardBuilder, loc);
    Signal slot1CanCollect = Signal(slot1CollectValues[0], &guardBuilder, loc);

    auto hasResponse = slot0CanCollect | slot1CanCollect;
    guardBuilder.create<circt::cmt2::ReturnOp>(loc, hasResponse.getValue());
  });

  respToUser->body([&](mlir::OpBuilder &bodyBuilder,
                       llvm::ArrayRef<mlir::BlockArgument> args) {
    // Following Rust logic exactly: if_! (slot_0_can_collect.read() { ... }
    // else { ... })
    auto slot0CollectValues =
        slot0CanCollectWire->callValue("read", bodyBuilder);
    Signal slot0CanCollect = Signal(slot0CollectValues[0], &bodyBuilder, loc);

    // Rust pattern: let retval = if_! (slot_0_can_collect.read() { ... } else {
    // ... })
    auto retval = If(
        slot0CanCollect,
        [&](mlir::OpBuilder &builder) -> Signal {
          // Clear slot 0 flags BEFORE returning (matching Rust exactly)
          slot0TxdReg->callMethod(
              "write", {UInt::constant(0, 1, bodyBuilder, loc).getValue()},
              builder);
          slot0RxdReg->callMethod(
              "write", {UInt::constant(0, 1, bodyBuilder, loc).getValue()},
              builder);

          // Read slot 0 data for immediate use if needed
          auto data0Values = slot0DataReg->callValue("read", bodyBuilder);
          auto tag0Values = slot0TagReg->callValue("read", bodyBuilder);
          Signal data0 = Signal(data0Values[0], &bodyBuilder, loc);
          Signal tag0 = Signal(tag0Values[0], &bodyBuilder, loc);

          // Create UserMemoryResp bundle from unpacked fields using
          // BundleCreateOp Order must match the bundle type definition: data,
          // tag
          llvm::SmallVector<mlir::Value> respBundleFields = {data0.getValue(),
                                                             tag0.getValue()};

          auto respBundleValue = builder.create<BundleCreateOp>(
              loc, userRespBundleType, respBundleFields);
          return Signal(respBundleValue.getResult(), &builder, loc);
        },
        [&](mlir::OpBuilder &builder) -> Signal {
          // Else branch: clear slot 1 flags and check if slot 1 can collect
          slot1TxdReg->callMethod(
              "write", {UInt::constant(0, 1, builder, loc).getValue()},
              builder);
          slot1RxdReg->callMethod(
              "write", {UInt::constant(0, 1, builder, loc).getValue()},
              builder);

          // Read slot 1 data
          auto data1Values = slot1DataReg->callValue("read", builder);
          auto tag1Values = slot1TagReg->callValue("read", builder);
          Signal data1 = Signal(data1Values[0], &builder, loc);
          Signal tag1 = Signal(tag1Values[0], &builder, loc);

          // Create UserMemoryResp bundle from unpacked fields using
          // BundleCreateOp Order must match the bundle type definition: data,
          // tag
          llvm::SmallVector<mlir::Value> respBundleFields = {data1.getValue(),
                                                             tag1.getValue()};

          auto respBundleValue = builder.create<BundleCreateOp>(
              loc, userRespBundleType, respBundleFields);
          return Signal(respBundleValue.getResult(), &builder, loc);
        },
        bodyBuilder, loc);

    bodyBuilder.create<circt::cmt2::ReturnOp>(loc, retval.getValue());
  });

  respToUser->finalize();

  // Rule: commit_cmd - processes HellaCache commands and manages response slots
  // This is equivalent to the Rust always! block for command processing
  auto *commitCmdRule = translatorModule->addRule("commit_cmd");

  commitCmdRule->guard([&](mlir::OpBuilder &guardBuilder) {
    auto trueVal = UInt::constant(1, 1, guardBuilder, loc);
    guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueVal.getValue());
  });

  commitCmdRule->body([&](mlir::OpBuilder &bodyBuilder) {
    // Pop from HellaCache command queue
    auto cmdValues = hellaCmdFifo->callValue("deq", bodyBuilder);
    auto hellaCmd = Signal(cmdValues[0], &bodyBuilder, loc);

    // Extract fields from HellaCache command (assuming packed format)
    // Layout: addr(32) + tag(8) + cmd(5) + size(2) + signed(1) + phys(1) +
    // mask(4) + data(32)
    Signal addr = hellaCmd.bits(31, 0);      // bits 0-31: addr (32-bit) - LSB
    Signal tag = hellaCmd.bits(39, 32);      // bits 32-39: tag (8-bit)
    Signal cmdField = hellaCmd.bits(44, 40); // bits 40-44: cmd (5-bit)
    Signal size = hellaCmd.bits(46, 45);     // bits 45-46: size (2-bit)
    Signal signedField = hellaCmd.bits(47, 47); // bit 47: signed (1-bit)
    Signal phys = hellaCmd.bits(48, 48);        // bit 48: phys (1-bit)
    Signal mask = hellaCmd.bits(52, 49);        // bits 49-52: mask (4-bit)
    Signal data = hellaCmd.bits(84, 53); // bits 53-84: data (32-bit) - MSB

    // Check if this is a read command (cmd == 0)
    Signal cmdIsRead = cmdField == UInt::constant(0, 5, bodyBuilder, loc);

    // Read current newer_slot flag
    auto newerSlotValues = newerSlotReg->callValue("read", bodyBuilder);
    Signal newerSlot = Signal(newerSlotValues[0], &bodyBuilder, loc);

    // Command processing logic for read commands (following Rust pattern)
    If(
        cmdIsRead,
        [&](mlir::OpBuilder &builder) {
          // Slot selection logic: if newer_slot == 1, use slot 0; else use slot
          // 1
          If(
              newerSlot == UInt::constant(1, 1, builder, loc),
              [&](mlir::OpBuilder &innerBuilder) {
                // next slot is 0 - use slot 0 for tracking
                slot0TxdReg->callMethod(
                    "write",
                    {UInt::constant(1, 1, innerBuilder, loc).getValue()},
                    innerBuilder);
                slot0TagReg->callMethod("write", {tag.getValue()},
                                        innerBuilder);
                newerSlotReg->callMethod(
                    "write",
                    {UInt::constant(0, 1, innerBuilder, loc).getValue()},
                    innerBuilder);
              },
              [&](mlir::OpBuilder &innerBuilder) {
                // next slot is 1 - use slot 1 for tracking
                slot1TxdReg->callMethod(
                    "write",
                    {UInt::constant(1, 1, innerBuilder, loc).getValue()},
                    innerBuilder);
                slot1TagReg->callMethod("write", {tag.getValue()},
                                        innerBuilder);
                newerSlotReg->callMethod(
                    "write",
                    {UInt::constant(1, 1, innerBuilder, loc).getValue()},
                    innerBuilder);
              },
              builder, loc);
        },
        bodyBuilder, loc);

    // Construct the HellaCache command bundle from extracted fields
    // Bundle format: {addr: u32, tag: u8, cmd: u5, size: u2, signed: u1, phys:
    // u1, data: u32, mask: u4}
    llvm::SmallVector<mlir::Value> bundleFields = {
        addr.getValue(),        // addr: u32
        tag.getValue(),         // tag: u8
        cmdField.getValue(),    // cmd: u5
        size.getValue(),        // size: u2
        signedField.getValue(), // signed: u1
        phys.getValue(),        // phys: u1
        data.getValue(),        // data: u32
        mask.getValue()         // mask: u4
    };

    auto hellaCmdBundle = bodyBuilder.create<BundleCreateOp>(
        loc, hellaCmdBundleType, bundleFields);

    // Send the constructed bundle to the HellaCache interface
    hellaCmdItfcDecl->callMethod("hella_cmd_to_bus", {hellaCmdBundle},
                                 bodyBuilder);
    bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
  });

  commitCmdRule->finalize();

  // Add scheduling constraints (matching Rust schedule! calls)
  // Rust: schedule!(resp_from_bus, commit_cmd) means resp_from_bus must fire
  // before commit_cmd Rust: schedule!(resp_from_bus, resp_to_user) means
  // resp_from_bus must fire before resp_to_user Note: In CMT2 C++, this would
  // be handled by the scheduling system, but we document it here
  translatorModule->setPrecedence(
      {{"resp_from_bus", "commit_cmd"}, {"resp_from_bus", "resp_to_user"}});
  // Schedule relationships:
  // 1. resp_from_bus → commit_cmd (resp_from_bus must execute before
  // commit_cmd)
  // 2. resp_from_bus → resp_to_user (resp_from_bus must execute before
  // resp_to_user)

  return translatorModule;
}

BundleType APSToCMT2GenPass::getRoccCmdBundleType(Builder &builder) {
  auto *context = builder.getContext();
  return BundleType::get(
      context, {BundleType::BundleElement{builder.getStringAttr("funct"), false,
                                          UIntType::get(context, 7)},
                BundleType::BundleElement{builder.getStringAttr("rs1"), false,
                                          UIntType::get(context, 5)},
                BundleType::BundleElement{builder.getStringAttr("rs2"), false,
                                          UIntType::get(context, 5)},
                BundleType::BundleElement{builder.getStringAttr("rd"), false,
                                          UIntType::get(context, 5)},
                BundleType::BundleElement{builder.getStringAttr("xs1"), false,
                                          UIntType::get(context, 1)},
                BundleType::BundleElement{builder.getStringAttr("xs2"), false,
                                          UIntType::get(context, 1)},
                BundleType::BundleElement{builder.getStringAttr("xd"), false,
                                          UIntType::get(context, 1)},
                BundleType::BundleElement{builder.getStringAttr("opcode"),
                                          false, UIntType::get(context, 7)},
                BundleType::BundleElement{builder.getStringAttr("rs1data"),
                                          false, UIntType::get(context, 32)},
                BundleType::BundleElement{builder.getStringAttr("rs2data"),
                                          false, UIntType::get(context, 32)}});
}

BundleType APSToCMT2GenPass::getRoccRespBundleType(Builder &builder) {
  auto *context = builder.getContext();
  return BundleType::get(
      builder.getContext(),
      {BundleType::BundleElement{builder.getStringAttr("rd"), false,
                                 UIntType::get(context, 5)},
       BundleType::BundleElement{builder.getStringAttr("rddata"), false,
                                 UIntType::get(context, 32)}});
}

/// Generate RoCC Adapter module that bridges RoCC interface with accelerator
/// execution units
Module *APSToCMT2GenPass::generateRoCCAdapter(
    Circuit &circuit, const llvm::SmallVector<unsigned long, 4> &opcodes) {
  auto *roccAdapterModule = circuit.addModule("RoCCAdapter");
  auto &builder = roccAdapterModule->getBuilder();
  auto loc = roccAdapterModule->getLoc();
  auto *roccRespItfcDecl =
      roccAdapterModule->defineInterfaceDecl("rocc_resp", "roccRespItfc");

  // Add clock and reset arguments
  Clock clk = roccAdapterModule->addClockArgument("clk");
  Reset rst = roccAdapterModule->addResetArgument("rst");

  // Create bundle types following Rust patterns
  // RoccCmd: {funct: u7, rs1: u5, rs2: u5, rd: u5, xs1: u1, xs2: u1, xd: u1,
  // opcode: u7, rs1data: u32, rs2data: u32}
  auto roccCmdBundleType = getRoccCmdBundleType(builder);
  auto roccRespBundleType = getRoccRespBundleType(builder);

  // Create external FIFO modules
  auto savedIP = builder.saveInsertionPoint();

  // Calculate total bits for RoCC command bundle: 7+5+5+5+1+1+1+7+32+32 = 96
  // bits
  auto roccCmdFifoMod = STLLibrary::createFIFO1PushModule(96, circuit);

  // Calculate total bits for RoCC response bundle: 5+32 = 37 bits (padded to
  // appropriate width)
  auto roccRespFifoMod = STLLibrary::createFIFO1PushModule(37, circuit);

  builder.restoreInsertionPoint(savedIP);

  // Create FIFO instances for each opcode
  llvm::SmallVector<Instance *, 4> roccCmdFifos;
  for (uint32_t opcode : opcodes) {
    auto *fifo = roccAdapterModule->addInstance(
        "rocc_cmd_fifo_" + (std::ostringstream() << std::hex << std::setw(4) << std::setfill('0') << opcode).str(), roccCmdFifoMod,
        {clk.getValue(), rst.getValue()});
    roccCmdFifos.push_back(fifo);
  }

  // Create response FIFO instance
  auto *roccRespFifo = roccAdapterModule->addInstance(
      "rocc_resp_fifo", roccRespFifoMod, {clk.getValue(), rst.getValue()});

  // Method: cmd_from_bus - receives RoCC commands from the bus and routes to
  // appropriate opcode queues
  llvm::SmallVector<std::pair<std::string, mlir::Type>, 1> cmdFromBusArgs;
  cmdFromBusArgs.push_back({"rocc_cmd_bus", roccCmdBundleType});
  auto *cmdFromBus =
      roccAdapterModule->addMethod("cmd_from_bus", cmdFromBusArgs, {});

  cmdFromBus->guard(
      [&](mlir::OpBuilder &guardBuilder, llvm::ArrayRef<mlir::BlockArgument>) {
        auto trueVal = UInt::constant(1, 1, guardBuilder, loc);
        guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueVal.getValue());
      });

  cmdFromBus->body([&](mlir::OpBuilder &bodyBuilder,
                       llvm::ArrayRef<mlir::BlockArgument> args) {
    Bundle roccCmd(args[0], &bodyBuilder, loc);

    // Extract all fields from RoCC command bundle
    Signal funct = roccCmd["funct"];
    Signal rs1 = roccCmd["rs1"];
    Signal rs2 = roccCmd["rs2"];
    Signal rd = roccCmd["rd"];
    Signal xs1 = roccCmd["xs1"];
    Signal xs2 = roccCmd["xs2"];
    Signal xd = roccCmd["xd"];
    Signal opcode = roccCmd["opcode"];
    Signal rs1data = roccCmd["rs1data"];
    Signal rs2data = roccCmd["rs2data"];

    // Pack the bundle into a 96-bit concatenated value for FIFO
    // Layout: rs2data(32) + rs1data(32) + opcode(7) + xd(1) + xs2(1) + xs1(1) +
    // rd(5) + rs2(5) + rs1(5) + funct(7) = 96 bits
    Signal packedCmd = rs2data.cat(rs1data)
                           .cat(opcode)
                           .cat(xd)
                           .cat(xs2)
                           .cat(xs1)
                           .cat(rd)
                           .cat(rs2)
                           .cat(rs1)
                           .cat(funct);

    // Route command to appropriate queue based on opcode (following Rust
    // pattern)
    for (size_t i = 0; i < opcodes.size(); ++i) {
      uint32_t targetOpcode = opcodes[i];
      auto *fifo = roccCmdFifos[i];

      // Check if this command's opcode matches the target
      auto opcodeMatch =
          opcode.cat(funct.pad(8)) == UInt::constant(targetOpcode, 16, bodyBuilder, loc);

      // Conditional enqueue (matching Rust if_! pattern)
      If(
          opcodeMatch,
          [&](mlir::OpBuilder &innerBuilder) -> Signal {
            // Enqueue the packed 96-bit command to the appropriate FIFO
            fifo->callMethod("enq", {packedCmd.getValue()}, innerBuilder);
            return UInt::constant(0, 1, innerBuilder, loc);
          },
          [&](mlir::OpBuilder &innerBuilder) -> Signal {
            return UInt::constant(0, 1, innerBuilder, loc);
          },
          bodyBuilder, loc);
    }

    bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
  });

  cmdFromBus->finalize();

  // Method: resp_from_user - receives responses from user execution units and
  // enqueues for return
  llvm::SmallVector<std::pair<std::string, mlir::Type>, 1> respFromUserArgs;
  respFromUserArgs.push_back({"rocc_resp_user", roccRespBundleType});
  auto *respFromUser =
      roccAdapterModule->addMethod("resp_from_user", respFromUserArgs, {});

  respFromUser->guard(
      [&](mlir::OpBuilder &guardBuilder, llvm::ArrayRef<mlir::BlockArgument>) {
        auto trueVal = UInt::constant(1, 1, guardBuilder, loc);
        guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueVal.getValue());
      });

  respFromUser->body([&](mlir::OpBuilder &bodyBuilder,
                         llvm::ArrayRef<mlir::BlockArgument> args) {
    Bundle roccResp(args[0], &bodyBuilder, loc);

    // Extract fields from RoCC response bundle
    Signal rd = roccResp["rd"];
    Signal rddata = roccResp["rddata"];

    // Pack the bundle into a 37-bit concatenated value for FIFO
    // Layout: rddata(32) + rd(5) = 37 bits
    Signal packedResp = rddata.cat(rd);

    // Enqueue the packed response to the response FIFO
    roccRespFifo->callMethod("enq", {packedResp.getValue()}, bodyBuilder);

    bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
  });

  respFromUser->finalize();

  // Rule: commit - processes response queue and sends to RoCC master
  auto *commitRule = roccAdapterModule->addRule("commit");

  commitRule->guard([&](mlir::OpBuilder &guardBuilder) {
    auto trueVal = UInt::constant(1, 1, guardBuilder, loc);
    guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueVal.getValue());
  });

  commitRule->body([&](mlir::OpBuilder &bodyBuilder) {
    // Dequeue response from FIFO (37-bit concatenated value)
    auto respValues = roccRespFifo->callValue("deq", bodyBuilder);
    Signal packedResp(respValues[0], &bodyBuilder, loc);

    // Unpack the 37-bit concatenated value back into bundle fields
    // Layout: rddata(32) + rd(5) = 37 bits
    Signal rddata = packedResp.bits(36, 5); // bits 5-36: rddata (32 bits)
    Signal rd = packedResp.bits(4, 0);      // bits 0-4: rd (5 bits)

    // Create RoCC response bundle from unpacked fields using BundleCreateOp
    // Order must match the bundle type definition: rd, rddata
    llvm::SmallVector<mlir::Value> respBundleFields = {rd.getValue(),
                                                       rddata.getValue()};

    auto respBundleValue = bodyBuilder.create<BundleCreateOp>(
        loc, roccRespBundleType, respBundleFields);

    roccRespItfcDecl->callMethod("rocc_resp_to_bus", {respBundleValue},
                                 bodyBuilder);

    bodyBuilder.create<circt::cmt2::ReturnOp>(loc);
  });

  commitRule->finalize();

  // Add cmd_to_user methods for each opcode (following Rust pattern exactly)
  for (size_t i = 0; i < opcodes.size(); ++i) {
    uint32_t opcode = opcodes[i];
    std::string methodName = "cmd_to_user_" + (std::ostringstream() << std::hex << std::setw(4) << std::setfill('0') << opcode).str();

    llvm::SmallVector<std::pair<std::string, mlir::Type>, 0> cmdToUserArgs;
    llvm::SmallVector<mlir::Type, 1> cmdToUserReturns = {roccCmdBundleType};
    auto *cmdToUser = roccAdapterModule->addMethod(methodName, cmdToUserArgs,
                                                   cmdToUserReturns);

    cmdToUser->guard([&](mlir::OpBuilder &guardBuilder,
                         llvm::ArrayRef<mlir::BlockArgument> args) {
      auto trueVal = UInt::constant(1, 1, guardBuilder, loc);
      guardBuilder.create<circt::cmt2::ReturnOp>(loc, trueVal.getValue());
    });

    cmdToUser->body([&](mlir::OpBuilder &bodyBuilder,
                        llvm::ArrayRef<mlir::BlockArgument> args) {
      // Dequeue command from the appropriate FIFO (96-bit concatenated value)
      auto cmdValues = roccCmdFifos[i]->callValue("deq", bodyBuilder);
      Signal packedCmd(cmdValues[0], &bodyBuilder, loc);

      // Unpack the 96-bit concatenated value back into bundle fields
      // Layout: rs2data(32) + rs1data(32) + opcode(7) + xd(1) + xs2(1) + xs1(1)
      // + rd(5) + rs2(5) + rs1(5) + funct(7) = 96 bits
      Signal rs2data = packedCmd.bits(95, 64); // bits 64-95: rs2data (32 bits)
      Signal rs1data = packedCmd.bits(63, 32); // bits 32-63: rs1data (32 bits)
      Signal opcode = packedCmd.bits(31, 25);  // bits 25-31: opcode (7 bits)
      Signal xd = packedCmd.bits(24, 24);      // bit 24: xd (1 bit)
      Signal xs2 = packedCmd.bits(23, 23);     // bit 23: xs2 (1 bit)
      Signal xs1 = packedCmd.bits(22, 22);     // bit 22: xs1 (1 bit)
      Signal rd = packedCmd.bits(21, 17);      // bits 17-21: rd (5 bits)
      Signal rs2 = packedCmd.bits(16, 12);     // bits 12-16: rs2 (5 bits)
      Signal rs1 = packedCmd.bits(11, 7);      // bits 7-11: rs1 (5 bits)
      Signal funct = packedCmd.bits(6, 0);     // bits 0-6: funct (7 bits)

      // Create RoCC command bundle from unpacked fields using BundleCreateOp
      // Order must match the bundle type definition: funct, rs1, rs2, rd, xs1,
      // xs2, xd, opcode, rs1data, rs2data
      llvm::SmallVector<mlir::Value> bundleFields = {
          funct.getValue(),  rs1.getValue(),    rs2.getValue(),
          rd.getValue(),     xs1.getValue(),    xs2.getValue(),
          xd.getValue(),     opcode.getValue(), rs1data.getValue(),
          rs2data.getValue()};

      auto bundleValue = bodyBuilder.create<BundleCreateOp>(
          loc, roccCmdBundleType, bundleFields);

      // Return the unpacked command bundle
      bodyBuilder.create<circt::cmt2::ReturnOp>(loc, bundleValue.getResult());
    });

    cmdToUser->finalize();
  }

  return roccAdapterModule;
}

} // namespace mlir