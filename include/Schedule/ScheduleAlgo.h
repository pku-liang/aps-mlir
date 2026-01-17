#ifndef SCHEDULE_ALGO_H
#define SCHEDULE_ALGO_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"

#include "Schedule/CDFG.h"
#include "Schedule/ResourceDB.h"
#include "TOR/TOR.h"
#include "TOR/TORTypes.h"
#include "APS/APSOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "nlohmann/json.hpp"

#include <fstream>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

namespace scheduling {

using std::unique_ptr;
using namespace mlir;

/// This is a template Base class of Schedule Algorithm. Every implementation of
/// schedule algorithm should inherit from this class. template parameter must
/// be a child class of OpWrapperBase. This class contains some basic
/// constraints about the scheduling problem. Can access schedule result from
/// OpAbstract Can access OpAbstract via result value from ValueMap
class ScheduleBase {
public:
  explicit ScheduleBase(Operation *op, ResourceDB &RDB) : RDB(RDB) {
    containingOp = op;
    EntryBB = nullptr;
    ExitBB = nullptr;

    auto funcOp = llvm::dyn_cast<tor::FuncOp>(op);
    assert(funcOp);

    if (auto attr = funcOp->getAttrOfType<mlir::FloatAttr>("clock"))
      ClockPeriod = attr.getValue().convertToDouble();
    else if (auto attr = funcOp->getAttrOfType<mlir::IntegerAttr>("clock"))
      ClockPeriod = (double)attr.getValue().roundToDouble();
    else
      assert(0 && "A clock period must be specified");

    isDataflowRegion = false;
  }

  virtual LogicalResult runSchedule() = 0;

  /// verify resource constraints and dependencies
  virtual LogicalResult verify();

  virtual void buildFromContainingOp();

  OpAbstract *
  createOp(Operation *op, Loop *ParentLoop, BasicBlock *ParentBB,
           ArrayRef<Value> Results, ArrayRef<Value> Operands,
           OpAbstract::OpType type = OpAbstract::OpType::DEFINED_OP) {
    int rsc = -1;
    switch (type) {
    case OpAbstract::OpType::PHI_OP:
    case OpAbstract::OpType::ASSIGN_OP:
      rsc = RDB.getResourceID("nop");
      break;
    default:
      rsc = RDB.getResourceID(op);
    }

    Operations.push_back(std::make_unique<OpConcrete>(
        OpConcrete(op, ParentLoop, ParentBB, rsc, Results, Operands, type)));

    OpAbstract *newop = Operations.back().get();

    int width = 0;

    if (Results.size() > 0) {
      // TODO: bit width analysis，不同长度位宽的如何表示？
      if (llvm::dyn_cast<arith::CmpFOp>(op) || llvm::dyn_cast<tor::CmpFOp>(op))
        width = Operands[0].getType().getIntOrFloatBitWidth();
      else if (llvm::dyn_cast<arith::SIToFPOp>(op) || llvm::dyn_cast<arith::UIToFPOp>(op) ||
                llvm::dyn_cast<arith::ExtFOp>(op) || llvm::dyn_cast<arith::TruncFOp>(op) ||
                llvm::dyn_cast<tor::MulSIOp>(op) || llvm::dyn_cast<tor::MulUIOp>(op) ||
                llvm::dyn_cast<arith::FPToSIOp>(op) || llvm::dyn_cast<arith::FPToUIOp>(op))   // 根据输入选择 TODO:应该做成array
        width = Operands[0].getType().getIntOrFloatBitWidth();
      else if (Results[0].getType().isIntOrFloat())
        width = Results[0].getType().getIntOrFloatBitWidth();
      else if (llvm::isa<mlir::MemRefType>(Results[0].getType())) {
        // For memref types (e.g., memref.get_global), use element type width if available
        auto memrefType = llvm::cast<mlir::MemRefType>(Results[0].getType());
        if (memrefType.getElementType().isIntOrFloat())
          width = memrefType.getElementType().getIntOrFloatBitWidth();
        else
          width = 32; // Default width for non-IntOrFloat memref element types
      }
      else {
        // For other types (e.g., index, etc.), use a default width
        width = 32;
      }
    }

    newop->setWidth(width);

    return newop;
  }

  OpAbstract *createMemOp(Operation *op, Loop *ParentLoop, BasicBlock *ParentBB,
                          ArrayRef<Value> Results, ArrayRef<Value> Operands,
                          OpAbstract::OpType type, ArrayRef<int> DepSigs,
                          ArrayRef<int> DepDists, const std::string& storageType) {
    // First create the operation to extract memref
    Operations.push_back(std::make_unique<MemOpConcrete>(
        MemOpConcrete(op, ParentLoop, ParentBB, 0 /* temporary */, Results, Operands, type)));

    auto newMemop = Operations.back()->getMemOp();
    Value memref = newMemop->getMemRef();

    // Determine resource based on memref and storage type
    int rsc;
    if (storageType == "RAM_1P") {
      rsc = RDB.getResourceID("memport_RAM_1P");
    } else if (storageType == "RAM_T2P") {
      rsc = RDB.getResourceID("memport_RAM_T2P");
    } else if (llvm::isa<aps::MemLoad>(op) || llvm::isa<aps::MemStore>(op)) {
      // For APS memory operations, use per-memref resources
      // All memories are treated as 1RW (one read or write per cycle)
      rsc = RDB.getOrCreateMemrefResource(memref);
    } else {
      // For other operations (tor.load, tor.store), use unified memport
      rsc = RDB.getResourceID("memport");
    }

    // Update the resource ID using the setter on the concrete operation
    auto concreteOp = llvm::dyn_cast<OpConcrete>(Operations.back().get());
    if (concreteOp) {
      concreteOp->setResourceId(rsc);
    }

    int width = 0;
    if (Results.size() > 0) {
      if (Results[0].getType().isIntOrFloat())
        width = Results[0].getType().getIntOrFloatBitWidth();
      else
        width = 32;
    }

    OpAbstract *newop = Operations.back().get();
    newop->setWidth(width);

    for (unsigned i = 0; i < DepSigs.size(); ++i)
      newMemop->Dependences[DepSigs[i]] = DepDists[i];

    return newop;
  }

  OpAbstract *createMAxiOp(Operation *op, Loop *ParentLoop, BasicBlock *ParentBB,
                           ArrayRef<Value> Results, ArrayRef<Value> Operands,
                           OpAbstract::OpType type) {
    int rsc = (type == OpAbstract::OpType::M_AXI_READ_OP)?
      RDB.getResourceID("m_axi_read") : RDB.getResourceID("m_axi_write");
    Operations.push_back(std::make_unique<MAxiOpConcrete>(
        MAxiOpConcrete(op, ParentLoop, ParentBB, rsc, Results, Operands, type)));

    int width = 0;
    if (Results.size() > 0) {
      if (Results[0].getType().isIntOrFloat())
        width = Results[0].getType().getIntOrFloatBitWidth();
      else
        width = 32;
    }

    OpAbstract *newop = Operations.back().get();
    newop->setWidth(width);

    return newop;
  }

  OpAbstract *createMAxiBurstOp(Operation *op, Loop *ParentLoop, BasicBlock *ParentBB,
                           ArrayRef<Value> Results, ArrayRef<Value> Operands,
                           OpAbstract::OpType type) {
    int rsc = RDB.getResourceID("m_axi_burst");
    Operations.push_back(std::make_unique<MAxiOpConcrete>(
        MAxiOpConcrete(op, ParentLoop, ParentBB, rsc, Results, Operands, type)));

    int width = 0;
    if (Results.size() > 0) {
      if (Results[0].getType().isIntOrFloat())
        width = Results[0].getType().getIntOrFloatBitWidth();
      else
        width = 32;
    }

    OpAbstract *newop = Operations.back().get();
    newop->setWidth(width);

    return newop;
  }

  OpAbstract *createTLOp(Operation *op, Loop *ParentLoop, BasicBlock *ParentBB,
                         ArrayRef<Value> Results, ArrayRef<Value> Operands,
                         OpAbstract::OpType type) {
    int rsc = RDB.getResourceID("tl");
    Operations.push_back(std::make_unique<MAxiOpConcrete>(
        MAxiOpConcrete(op, ParentLoop, ParentBB, rsc, Results, Operands, type)));

    int width = 32; // TileLink operations use fixed 32-bit width

    OpAbstract *newop = Operations.back().get();
    newop->setWidth(width);

    return newop;
  }

  OpAbstract *createStreamOp(Operation *op, Loop *ParentLoop, BasicBlock *ParentBB,
                          ArrayRef<Value> Results, ArrayRef<Value> Operands,
                          OpAbstract::OpType type) {
    int rsc = RDB.getResourceID("stream");
    Operations.push_back(std::make_unique<StreamOpConcrete>(
        StreamOpConcrete(op, ParentLoop, ParentBB, rsc, Results, Operands, type)));

    int width = 0;
    if (Results.size() > 0) {
      if (Results[0].getType().isIntOrFloat())
        width = Results[0].getType().getIntOrFloatBitWidth();
      else
        width = 32;
    }

    OpAbstract *newop = Operations.back().get();
    newop->setWidth(width);

    return newop;
  }
  

  void addDependency(Dependence D) {
    Dependencies.push_back(std::make_unique<Dependence>(D));
    D.SourceOp->addSucc(Dependencies.back().get());
    D.DestinationOp->addPred(Dependencies.back().get());
  }

  /// build Dependency from containingOp;
  void setClockFrequence(int Cycle) { ClockPeriod = Cycle; }

  void printCDFG();

  void printSchedule();

  /**
   * Query the scheduling information of a loop operation
   * @param op reference to the loop operation
   * @return first: pipeline flag, second: achieved II
   */
  std::pair<int, int> queryLoop(Operation *op) {
    Loop *L = LoopMap[op];
    return std::make_pair(L->PipelineFlag, L->PipelineFlag ? L->AchievedII : -1);
  }

  /**
   * Query the scheduling information of a calculate operations
   * @param op address of the operation. address must remains unchanged from
   *           that of input
   * @return first: starting cycle, seoncd: ending cycle
   */
  std::pair<int, int> queryOp(Operation *op) {
    auto getFuncOp = [&](OpAbstract *op) {
      if (auto callOp = llvm::dyn_cast<tor::CallOp>(op->getOp())) {
        auto module = callOp->getParentOfType<ModuleOp>();
        for (auto designOp : module.getOps<tor::DesignOp>()) {
          for (auto funcOp : designOp.getOps<tor::FuncOp>()) {
            if (funcOp.getName() == callOp.getCallee()) {
              return funcOp;
            }
          }
        }
      }
      assert(false);
    };
    auto getLatency = [&](OpAbstract *op) {
      assert(llvm::isa<tor::CallOp>(op->getOp()));
      if (!getFuncOp(op)->hasAttr("II")) {
        return (int64_t) 1;
      }
      SmallVector<tor::SuccTimeOp> succOps;
      for (auto timeGraphOp : getFuncOp(op).getOps<tor::TimeGraphOp>()) {
        for (auto succOp : timeGraphOp.getOps<tor::SuccTimeOp>()) {
          succOps.push_back(succOp);
        }
      }
      uint32_t maxTime = 0;
      for (auto succOp : succOps) {
        maxTime = std::max(maxTime, succOp.getTime());
      }
      auto times = SmallVector<int64_t>(maxTime + 1, 0);
      for (auto succOp : succOps) {
        for (size_t i = 0, r = succOp.getPoints().size(); i < r; ++i) {
          auto from = dyn_cast<mlir::IntegerAttr>(succOp.getPoints()[i]).getInt();
          auto to = succOp.getTime();
          for (auto attr : dyn_cast<mlir::DictionaryAttr>(succOp.getEdges()[i])) {
            auto edge_attr = attr.getValue();
            if (auto str_attr = dyn_cast<mlir::StringAttr>(edge_attr)) {
              auto str = str_attr.getValue();
              if (str.find(':') != StringRef::npos) {
                auto index = str.find(':');
                auto numberStr = str.substr(index + 1, str.size() - index - 1);
                times[to] = std::max(times[to], times[from] + std::stoi(numberStr.str()));
              } else {
                times[to] = std::max(times[to], times[from] + 1);
              }
            }
          }
        }
      }
      return times[maxTime] + 1;
    };

    // Control flow operations (tor.while, tor.for, tor.if, tor.func) are not
    // added to OperationMap as they are not data operations. Return default
    // timing (0, 0) for such operations.
    if (OperationMap.find(op) == OperationMap.end()) {
      return std::make_pair(0, 0);
    }

    OpAbstract *opA = OperationMap[op];
    int latency = std::max(1, RDB.getLatency(opA->getResource(), opA->getWidth()));
    if (llvm::isa<tor::CallOp>(opA->getOp())) {
      latency = getLatency(opA);
    }
    int start = opA->getStartTime();
    int end = start + latency;
    return std::make_pair(start, end);
  }

private:
  /**
   * @brief walk through a mlir block.
   * @return (beginning BB, exiting BB)
   */
  std::pair<BasicBlock *, BasicBlock *> buildCFG(Block &block,
                                                 Loop *ParentLoop);


  void buildScalarDFG();
  void buildTensorDFG();
  void buildStreamDFG();
  /**
   * @brief get all maxiOp that need to be analyzed in buildMAxiDFG(),
   *        and get request op to its last op in requestOpToLastOp.
   */
  void buildBurstMAxiDFG(std::vector<mlir::Operation*>& mAxiOps,
    std::unordered_map<mlir::Operation*, mlir::Operation*>& requestOpToLastOp);
  void buildMAxiDFG();
  void buildCallOpTensorDFG();
  void buildDFG();

  bool isLocalOp(mlir::Operation* op) const;

protected:
  /**
   * @brief convert OpAbstract to T
   */
  template <typename T> std::vector<std::unique_ptr<T>> initSchedule() {

    std::vector<std::unique_ptr<T>> vec;
    llvm::DenseMap<const OpAbstract *, T *> OpMap;

    for (auto &&x : Operations) {
      vec.push_back(std::make_unique<T>(x.get()));

      OpMap[x.get()] = vec.back().get();
    }

    for (auto &&BB : BasicBlocks) {
      for (auto &x : BB->getOperations())
        x = OpMap[x];
      BB->setBranchOp(OpMap[BB->getBranchOp()]);
    }

    for (auto &&D : Dependencies) {
      D->SourceOp = OpMap[D->SourceOp];
      D->DestinationOp = OpMap[D->DestinationOp];
    }

    for (auto &x : ValueMap)
      x.second = OpMap[x.second];

    for (auto &x : OperationMap)
      x.second = OpMap[x.second];

    return std::move(vec);
  }

protected:
  Operation *containingOp; /// This is the containing Op that needs to be
                           /// scheduled. i.e. a moduleOp or a while loop Op

  float ClockPeriod;

  ResourceDB& RDB;

  BasicBlock *EntryBB, *ExitBB;

  std::vector<unique_ptr<Dependence>>
      Dependencies; /// This vector contains all the dependencies.

  std::vector<unique_ptr<OpConcrete>>
      Operations; /// This vector contains all the operations that need to be
                  /// scheduled.

  std::vector<unique_ptr<BasicBlock>> BasicBlocks;

  std::vector<unique_ptr<Loop>> Loops;

  std::unordered_map<Operation *, OpAbstract *>
      OperationMap; // Map mlir operation to OpAbstract

  llvm::DenseMap<Value, OpAbstract *>
      ValueMap; /// real Operation that needs to be scheduled

  llvm::DenseMap<Operation *, Loop *> LoopMap;

  bool isDataflowRegion;
};

} // namespace scheduling

#endif
