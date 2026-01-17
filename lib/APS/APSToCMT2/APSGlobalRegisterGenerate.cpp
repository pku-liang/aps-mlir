#include "APS/APSToCMT2.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "aps-memory-pool-gen"

namespace mlir {

using namespace mlir;
using namespace mlir::tor;
using namespace circt::cmt2::ecmt2;
using namespace circt::cmt2::ecmt2::stl;
using namespace circt::firrtl;

/// Generate the CMT2 memory pool module
SmallVector<std::tuple<std::string, int8_t>, 8> APSToCMT2GenPass::generateGlobalRegisterList(Circuit &circuit,
                                              ModuleOp moduleOp,
                                              aps::MemoryMapOp memoryMapOp) {
  MLIRContext *context = moduleOp.getContext();
  OpBuilder builder(context);
  SmallVector<std::tuple<std::string, int8_t>, 8> globalRegisterList;
  if (memoryMapOp) {
    for (auto &block : memoryMapOp.getRegion()) {
      for (auto &op : block) {
        if (auto entry = dyn_cast<aps::MemEntryOp>(op)) {
          if (entry.getNumBanks() != 1) {
            continue; // this is not a global register
          }
          auto bankSymbols = entry.getBankSymbols();
          if (bankSymbols.empty()) {
            continue; // aha, no symbol?
          }
          auto firstBankSymAttr =
              llvm::dyn_cast<FlatSymbolRefAttr>(bankSymbols[0]);
          if (!firstBankSymAttr) {
            continue; // not a symbol ref?
          }
          // Look up the memref.global for this bank
          StringRef bankSymbolName = firstBankSymAttr.getValue();
          auto globalOp = getGlobalMemRef(moduleOp, bankSymbolName);
          if (!globalOp) {
            llvm::errs() << "Error: Could not find memref.global for bank "
                         << bankSymbolName << "\n";
            continue;
          }
          auto memrefType = globalOp.getType();
          auto elementType = memrefType.getElementType();
          if (memrefType.getRank() > 0) {
            continue;
          }
          if (auto intType = llvm::dyn_cast<IntegerType>(elementType)) {
            auto dataWidth = intType.getWidth();
            std::string registerName = entry.getName().str();
            globalRegisterList.push_back(
                std::make_tuple(registerName, static_cast<int8_t>(dataWidth)));
          } else {
            continue; // what the type, don't know?
          }
        }
      }
    }
  }
  return globalRegisterList;
}

} // namespace mlir