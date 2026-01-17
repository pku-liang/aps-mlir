#ifndef CDFG_H
#define CDFG_H

#include "TOR/TOR.h"
#include "APS/APSOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <queue>
#include <string>
#include <unordered_map>
#include <memory>

namespace scheduling {

using namespace mlir;

class Loop;
class BasicBlock;
class OpAbstract;
class ControlEdge;
class Dependence;
class OpConcrete;
class MemOpConcrete;
class StreamOpConcrete;
class MAxiOpConcrete;

using ResourceType = std::string;

struct ControlEdge {
public:
  enum EdgeType { COND, FORWARD, LOOPBACK };

  ControlEdge(BasicBlock *f, BasicBlock *t, EdgeType ty)
      : fromBB(f), toBB(t), type(ty) {}

  BasicBlock *fromBB, *toBB;
  EdgeType type;
};

struct Dependence {
public:
  enum DependenceType { D_RAW, D_WAW, D_WAR, D_RAR };

  OpAbstract *SourceOp;
  OpAbstract *DestinationOp;

  int Distance;

  DependenceType type;

  Dependence(OpAbstract *SourceOp, OpAbstract *DestinationOp, int Distance,
             DependenceType type)
      : SourceOp(SourceOp), DestinationOp(DestinationOp), Distance(Distance),
        type(type) {}
};

class OpAbstract {
public:
  /// llvm rtti
  enum OpKind {
    OK_CONCRETE,
    OK_WRAPPER,
    OK_LISTWRAPPER,
    OK_SDCWRAPPER,
    OK_ENDWRAPPER
  };
  const OpKind Kind;

  OpAbstract(OpKind K) : Kind(K) {}

  OpKind getKind() const { return Kind; }

public:
  enum class OpType {
    DEFINED_OP,
    CALL_OP,
    PHI_OP,    /// x = phi(a, b) in [if, while, for]
    ASSIGN_OP, /// x = y in [yield]
    INC_OP,    /// x = x + step in [for]
    LOAD_OP,   /// load
    STORE_OP, /// store
    STREAM_READ_OP, /// stream_read
    STREAM_WRITE_OP, /// stream_write
    M_AXI_READ_OP, /// axi_read
    M_AXI_WRITE_OP, /// axi_write
    M_AXI_BURST_READ_OP, /// axi_burst_read
    M_AXI_BURST_WRITE_OP, /// axi_burst_write
    M_AXI_READ_REQUEST_OP, /// axi_read_request
    M_AXI_WRITE_REQUEST_OP, /// axi_write_request
    M_AXI_RESPONSE_OP, /// axi_write_response
    TL_READ_OP, /// TileLink read (aps.memburstload)
    TL_WRITE_OP, /// TileLink write (aps.memburststore)
  };

  virtual MemOpConcrete *getMemOp() = 0;
  virtual StreamOpConcrete *getStreamOp() = 0;
  virtual MAxiOpConcrete *getMAxiOp() = 0;
  virtual OpType getType() = 0;
  virtual Operation *getOp() = 0;
  virtual int getResource() const = 0;
  virtual int getWidth() const = 0;
  virtual void setWidth(int w) = 0;
  virtual ArrayRef<Dependence *> getPred() = 0;
  virtual ArrayRef<Dependence *> getSucc() = 0;
  virtual ArrayRef<Value> getOperands() = 0;
  virtual ArrayRef<Value> getResults() = 0;
  virtual void addPred(Dependence *D) = 0;
  virtual void addSucc(Dependence *D) = 0;
   
  virtual std::string getOpDumpId() {
    auto op = getOp();
    auto str = op->getAttrOfType<StringAttr>("dump");
    if(str != nullptr) {
      return str.str();
    }
    return "";
  }

  virtual void printName(llvm::raw_ostream &os) {
    if (getType() == OpType::DEFINED_OP || getType() == OpType::CALL_OP) {
      getOp()->print(os);
      os << "\n";
    } else if (getType() == OpType::PHI_OP)
      os << "PHI_OP:  ";
    else if (getType() == OpType::ASSIGN_OP)
      os << "ASSIGN_OP:  ";
    else if (getType() == OpType::LOAD_OP) {
      os << "LOAD_OP:  ";
      getOp()->print(os);
      os << "\n";
    } else if (getType() == OpType::STORE_OP) {
      os << "STORE_OP:  ";
      getOp()->print(os);
      os << "\n";
    } else if (getType() == OpType::STREAM_READ_OP) {
      os << "STREAM_READ_OP: ";
      getOp()->print(os);
      os << "\n";
    } else if (getType() == OpType::STREAM_WRITE_OP) {
      os << "STREAM_WRITE_OP: ";
      getOp()->print(os);
      os << "\n";
    }
  }
  /// Client can call this function to retrieve schedule results
  virtual int getStartTime() = 0;

  virtual void setStartTime(int T) = 0;
  virtual Loop *getParentLoop() = 0;
  virtual BasicBlock *getParentBB() = 0;
};

class OpConcrete : public OpAbstract {
public:
  OpConcrete(Operation *op, Loop *ParentLoop, BasicBlock *ParentBB,
             int resource, ArrayRef<Value> R, ArrayRef<Value> O, OpType type)
      : OpAbstract(OK_CONCRETE), type(type), op(op), ResourceId(resource),
        ParentLoop(ParentLoop), ParentBB(ParentBB) {
    bitwidth = 1;
    auto types = op->getOperandTypes();

    for (auto ty : types) {
      if (ty.isIntOrFloat())
        bitwidth = std::max(bitwidth, ty.getIntOrFloatBitWidth());
    }

    Operands.clear();
    Results.clear();

    Results.insert(Results.begin(), R.begin(), R.end());
    Operands.insert(Operands.begin(), O.begin(), O.end());
  }

  int getWidth() const override { return bitwidth; }

  void setWidth(int w) override { bitwidth = w; }

  virtual MemOpConcrete *getMemOp() override { return nullptr; }

  virtual StreamOpConcrete *getStreamOp() override { return nullptr; }

  virtual MAxiOpConcrete *getMAxiOp() override { return nullptr; }

  OpType getType() override { return type; }

  Operation *getOp() override { return op; }

  int getResource() const override { return ResourceId; }

  ArrayRef<Dependence *> getPred() override { return pred; }

  ArrayRef<Dependence *> getSucc() override { return succ; }

  ArrayRef<Value> getOperands() override { return Operands; }

  ArrayRef<Value> getResults() override { return Results; }

  void addPred(Dependence *D) override { pred.push_back(D); }

  void addSucc(Dependence *D) override { succ.push_back(D); }

  int getStartTime() override { return startTime; }

  void setStartTime(int T) override { startTime = T; }

  Loop *getParentLoop() override { return ParentLoop; }

  BasicBlock *getParentBB() override { return ParentBB; }

  void setResourceId(int rsc) { ResourceId = rsc; }

protected:
  OpType type;
  Operation *op;
  int ResourceId;
  uint32_t bitwidth;
  SmallVector<Dependence *, 4> succ, pred;
  SmallVector<Value, 4> Operands;
  SmallVector<Value, 4> Results;
  Loop *ParentLoop;
  BasicBlock *ParentBB;
  int startTime;
};

class StreamOpConcrete : public OpConcrete {
public:
  virtual StreamOpConcrete *getStreamOp() override { return this; }
  Value getStream() { return stream; }

  bool isWrite() { return type == OpType::STREAM_WRITE_OP; }
  bool isRead() { return type == OpType::STREAM_READ_OP; }

  StreamOpConcrete(Operation *op, Loop *ParentLoop, BasicBlock *ParentBB,
                int resource, ArrayRef<Value> R, ArrayRef<Value> O, OpType type)
      : OpConcrete(op, ParentLoop, ParentBB, resource, R, O, type) {
        if (auto streamReadOp = llvm::dyn_cast<tor::StreamReadOp>(op)) {
          stream = streamReadOp.getStream();
        } else if (auto streamWriteOp = llvm::dyn_cast<tor::StreamWriteOp>(op)) {
          stream = streamWriteOp.getStream();
        }
      }
private:
  Value stream;
};

class MAxiOpConcrete : public OpConcrete {
public:
  virtual MAxiOpConcrete *getMAxiOp() override { return this; }
  std::string getBus() const { return bus; }
  Value getAddr() const { return addr; }
  Value getMemRef() const { return memref; }
  Value getRequest() const {return request; }
  int getLength() const {return length; }

  // Override getWidth() to return a fixed valid bitwidth (32-bit, power of 2)
  // M_AXI operations don't have a meaningful bitwidth in the same sense as arithmetic ops
  int getWidth() const override { return 32; }

  bool isWrite() const {
    return type == OpType::M_AXI_WRITE_OP || type == OpType::M_AXI_WRITE_REQUEST_OP ||
           type == OpType::M_AXI_BURST_WRITE_OP || type == OpType::TL_WRITE_OP;
  }
  bool isRead() const {
    return type == OpType::M_AXI_READ_OP || type == OpType::M_AXI_READ_REQUEST_OP ||
           type == OpType::M_AXI_BURST_READ_OP || type == OpType::TL_READ_OP;
  }
  bool isBurst() const { return request != nullptr; }
  bool isRequest() const {
    return type == OpType::M_AXI_READ_REQUEST_OP || type == OpType::M_AXI_WRITE_REQUEST_OP;
  }
  bool isResponse() const { return type == OpType::M_AXI_RESPONSE_OP; }

  int getOpLength(mlir::Value length) {
    auto op = length.getDefiningOp();
    if (!op) {
      // Length is a BlockArgument or invalid - return default
      return 1;
    }
    if (auto constIndexOp = llvm::dyn_cast<arith::ConstantIndexOp>(op))
      return constIndexOp.value();
    if (auto constIntOp = llvm::dyn_cast<arith::ConstantIntOp>(op))
      return constIntOp.value();
    assert(false && "length should be constant");
    return 1;
  }

  MAxiOpConcrete(Operation *op, Loop *ParentLoop, BasicBlock *ParentBB,
                int resource, ArrayRef<Value> R, ArrayRef<Value> O, OpType type)
      : OpConcrete(op, ParentLoop, ParentBB, resource, R, O, type) {
    if (auto mAxiReadOp = llvm::dyn_cast<tor::AXIReadOp>(op)) {
      memref = mAxiReadOp.getMemref();
      addr = mAxiReadOp.getOperand(1);
    } else if (auto mAxiWriteOp = llvm::dyn_cast<tor::AXIWriteOp>(op)) {
      memref = mAxiWriteOp.getMemref();
      addr = mAxiWriteOp.getOperand(2);
    } else if (auto mAxiBurstReadOp = llvm::dyn_cast<tor::AXIBurstReadOp>(op)) {
      memref = mAxiBurstReadOp.getMemref();
      request = mAxiBurstReadOp.getRequest();
    } else if (auto mAxiBurstWriteOp = llvm::dyn_cast<tor::AXIBurstWriteOp>(op)) {
      memref = mAxiBurstWriteOp.getMemref();
      request = mAxiBurstWriteOp.getRequest();
    } else if (auto mAxiWriteResponseOp = llvm::dyn_cast<tor::AXIWriteResponseOp>(op)) {
      memref = mAxiWriteResponseOp.getMemref();
      request = mAxiWriteResponseOp.getRequest();
    } else if (auto mAxiReadRequestOp = llvm::dyn_cast<tor::AXIReadRequestOp>(op)) {
      memref = mAxiReadRequestOp.getMemref();
      request = mAxiReadRequestOp.getResult();
      length = getOpLength(mAxiReadRequestOp.getLength());
    } else if (auto mAxiWriteRequestOp = llvm::dyn_cast<tor::AXIWriteRequestOp>(op)) {
      memref = mAxiWriteRequestOp.getMemref();
      request = mAxiWriteRequestOp.getResult();
      length = getOpLength(mAxiWriteRequestOp.getLength());
    } else if (auto memBurstLoadOp = llvm::dyn_cast<aps::MemBurstLoad>(op)) {
      // APS memory burst load operation
      // For scheduling purposes, use the first memref
      // (all partitions of the same array should have the same scheduling constraints)
      auto memrefs = memBurstLoadOp.getMemrefs();
      if (!memrefs.empty()) {
        memref = *memrefs.begin();
      }
      addr = memBurstLoadOp.getStart();
      length = getOpLength(memBurstLoadOp.getLength());
    } else if (auto memBurstStoreOp = llvm::dyn_cast<aps::MemBurstStore>(op)) {
      // APS memory burst store operation
      // For scheduling purposes, use the first memref
      // (all partitions of the same array should have the same scheduling constraints)
      auto memrefs = memBurstStoreOp.getMemrefs();
      if (!memrefs.empty()) {
        memref = *memrefs.begin();
      }
      addr = memBurstStoreOp.getStart();
      length = getOpLength(memBurstStoreOp.getLength());
    }

    // Check if memref is defined by an operation (not a BlockArgument)
    if (auto defOp = memref.getDefiningOp()) {
      if (auto mAxiCreateOp = llvm::dyn_cast<tor::AXICreateOp>(defOp)) {
        if (mAxiCreateOp->hasAttr("bus")) {
          bus = dyn_cast<StringAttr>(mAxiCreateOp->getAttr("bus")).getValue().str();
        }
      }
    }
  }
private:
  std::string bus = "";
  int length = 1;
  Value addr;
  Value memref;
  Value request;
};

class MemOpConcrete : public OpConcrete {
public:
  virtual MemOpConcrete *getMemOp() override { return this; }

  Value getAddr() { return addr; }

  Value getMemRef() { return memref; }

  int getPartitionIndicies() {
    assert(hasFixedMemoryBank());

    // Check if it's a TOR MemRefType (with partition properties)
    if (auto torType = llvm::dyn_cast<tor::MemRefType>(memref.getType())) {
      auto shape = torType.getShape();
      auto property = torType.getProperty();

      int idx = 0;

      for (auto x : llvm::enumerate(partitionIndices)) {
        APInt attr;
        matchPattern(x.value(), m_ConstantInt(&attr));
        if (property[x.index()].getValue() == "complete")
          idx = idx * shape[x.index()] + attr.getLimitedValue();
      }
      return idx;
    }

    // For standard memref types (mlir::MemRefType), no partition support
    // Just return 0 as there's no partition information
    return 0;
  }

  bool hasFixedMemoryBank() {
    SmallVector<APInt, 4> idx;
    for (auto x : partitionIndices) {
      APInt attr;
      if (!matchPattern(x, m_ConstantInt(&attr)))
        return false;
    }
    return true;
  }

  SmallVector<APInt, 4> getMemoryBankIdx() {
    assert(hasFixedMemoryBank());
    SmallVector<APInt, 4> idx;
    for (auto x : partitionIndices) {
      APInt attr;
      matchPattern(x, m_ConstantInt(&attr));
      idx.push_back(attr);
    }
    return idx;
  }

  MemOpConcrete(Operation *op, Loop *ParentLoop, BasicBlock *ParentBB,
                int resource, ArrayRef<Value> R, ArrayRef<Value> O, OpType type)
      : OpConcrete(op, ParentLoop, ParentBB, resource, R, O, type) {
    assert(llvm::isa<tor::LoadOp>(op) || llvm::isa<tor::StoreOp>(op) ||
           llvm::isa<tor::GuardedStoreOp>(op) || llvm::isa<aps::MemLoad>(op) ||
           llvm::isa<aps::MemStore>(op) || llvm::isa<mlir::memref::LoadOp>(op) ||
           llvm::isa<mlir::memref::StoreOp>(op));

    if (auto loadOp = llvm::dyn_cast<tor::LoadOp>(op)) {
      memref = loadOp.getMemref();
      addr = loadOp.getOperand(
          1); // the first index is the address in the memory bank
      partitionIndices = loadOp.getIndices().drop_front(1);
    } else if (auto storeOp = llvm::dyn_cast<tor::StoreOp>(op)) {
      memref = storeOp.getMemref();
      addr = storeOp.getOperand(
          2); // the first index is the address in the memory bank
      partitionIndices = storeOp.getIndices().drop_front(1);
    } else if (auto storeOp = llvm::dyn_cast<tor::GuardedStoreOp>(op)) {
      memref = storeOp.getMemref();
      addr = storeOp.getOperand(
          3); // the first index is the address in the memory bank
      partitionIndices = storeOp.getIndices().drop_front(1);
    } else if (auto apsLoadOp = llvm::dyn_cast<aps::MemLoad>(op)) {
      memref = apsLoadOp.getMemref();
      if (apsLoadOp.getIndices().size() > 0) {
        addr = apsLoadOp.getIndices()[0]; // first index is the address
        partitionIndices = apsLoadOp.getIndices().drop_front(1);
      }
    } else if (auto apsStoreOp = llvm::dyn_cast<aps::MemStore>(op)) {
      memref = apsStoreOp.getMemref();
      if (apsStoreOp.getIndices().size() > 0) {
        addr = apsStoreOp.getIndices()[0]; // first index is the address
        partitionIndices = apsStoreOp.getIndices().drop_front(1);
      }
    } else if (auto memLoadOp = llvm::dyn_cast<mlir::memref::LoadOp>(op)) {
      memref = memLoadOp.getMemref();
      if (memLoadOp.getIndices().size() > 0) {
        addr = memLoadOp.getIndices()[0];
        partitionIndices = memLoadOp.getIndices().drop_front(1);
      }
    } else if (auto memStoreOp = llvm::dyn_cast<mlir::memref::StoreOp>(op)) {
      memref = memStoreOp.getMemref();
      if (memStoreOp.getIndices().size() > 0) {
        addr = memStoreOp.getIndices()[0];
        partitionIndices = memStoreOp.getIndices().drop_front(1);
      }
    }
    Dependences.clear();
  }

  bool isMemOpConcrete() {
    return type == OpType::LOAD_OP || type == OpType::STORE_OP;
  }

  std::unordered_map<int, int> Dependences;

private:
  Value memref;
  Value addr;
  SmallVector<Value, 4> partitionIndices;
};

/// Other scheduling algorithm may inherit this class to add extra
class OpWrapperBase : public OpAbstract {
public:
  OpWrapperBase(OpAbstract *op) : OpAbstract(OK_WRAPPER) {}

  OpWrapperBase(OpAbstract *op, OpKind K) : OpAbstract(K), op(op) {}

  virtual MemOpConcrete *getMemOp() override { return op->getMemOp(); }

  virtual StreamOpConcrete *getStreamOp() override { return op->getStreamOp(); }

  virtual MAxiOpConcrete *getMAxiOp() override { return op->getMAxiOp(); }

  virtual int getWidth() const override { return op->getWidth(); }

  virtual void setWidth(int w) override { return op->setWidth(w); }

  virtual OpType getType() override { return op->getType(); }

  virtual Operation *getOp() override { return op->getOp(); }

  virtual int getResource() const override { return op->getResource(); }

  virtual ArrayRef<Dependence *> getPred() override { return op->getPred(); }

  virtual ArrayRef<Dependence *> getSucc() override { return op->getSucc(); }

  virtual ArrayRef<Value> getOperands() override { return op->getOperands(); }

  virtual ArrayRef<Value> getResults() override { return op->getResults(); }

  virtual void addPred(Dependence *D) override { op->addPred(D); }

  virtual void addSucc(Dependence *D) override { op->addSucc(D); }

  virtual int getStartTime() override { return op->getStartTime(); }

  virtual void setStartTime(int T) override { op->setStartTime(T); }

  virtual Loop *getParentLoop() override { return op->getParentLoop(); }

  virtual BasicBlock *getParentBB() override { return op->getParentBB(); }

protected:
  OpAbstract *op;
};

class BasicBlock {
public:
  void addOperation(OpAbstract *op) { Operations.push_back(op); }

  ArrayRef<ControlEdge> getPred() { return Pred; }

  llvm::MutableArrayRef<OpAbstract *> getOperations() { return Operations; }

  ArrayRef<ControlEdge> getSucc() { return Succ; }

  static void addControlDependency(ControlEdge edge) {
    edge.fromBB->Succ.push_back(edge);
    edge.toBB->Pred.push_back(edge);
  }

  Loop *getParentLoop() const { return ParentLoop; }

  OpAbstract *getBranchOp() const { return BranchOp; }

  void setBranchOp(OpAbstract *op) { BranchOp = op; }

  void setParentLoop(Loop *P) { ParentLoop = P; }

  void setBranchValue(Value v) { BranchValue = v; }

  Value getBranchValue() const { return BranchValue; }

  BasicBlock(Loop *ParentLoop = nullptr) : ParentLoop(ParentLoop) {
    Operations.clear();
    Pred.clear();
    Succ.clear();
    BranchOp = nullptr;
  }

private:
  llvm::SmallVector<OpAbstract *> Operations;
  llvm::SmallVector<ControlEdge, 4> Pred;
  llvm::SmallVector<ControlEdge, 4> Succ;
  Loop *ParentLoop;
  OpAbstract *BranchOp;
  Value BranchValue;
};

class Loop {
public:
  struct LoopBound {
  public:
    Value lowerBound, upperBound, stride;
  };

  Loop(Loop *ParentLoop, Operation *DefiningOp, bool PipelineFlag = false,
       int TargetII = 1)
      : PipelineFlag(PipelineFlag), TargetII(TargetII), AchievedII(-1),
        ParentLoop(ParentLoop), DefiningOp(DefiningOp) {
    LoopBody.clear();
    ChildLoop.clear();
    TopLevelBlock.clear();
  }

  Operation *getDefiningOp() { return DefiningOp; }

  ArrayRef<BasicBlock *> getBody() { return LoopBody; }

  ArrayRef<Loop *> getChildLoop() { return ChildLoop; }

  ArrayRef<BasicBlock *> getTopLevelBlock() { return TopLevelBlock; }

  Loop *getParentLoop() { return ParentLoop; }

  void addBasicBlock(BasicBlock *BB) {
    TopLevelBlock.push_back(BB);
    LoopBody.push_back(BB);
  }

  void addChildLoop(Loop *L) {
    ChildLoop.push_back(L);
    LoopBody.insert(LoopBody.end(), L->getBody().begin(), L->getBody().end());
  }

public:
  bool PipelineFlag;
  int TargetII;
  int AchievedII;

private:
  llvm::SmallVector<BasicBlock *>
      LoopBody; /// all basic blocks in this loop body
  llvm::SmallVector<BasicBlock *>
      TopLevelBlock; /// top-level basic block in this loop body
  llvm::SmallVector<Loop *> ChildLoop; /// Top-level sub loops in this loop;
  Loop *ParentLoop;
  Operation *DefiningOp;
};

/// Bascic block create factory
inline std::unique_ptr<BasicBlock> createBasicBlock(Loop *parentLoop) {
  auto b = std::make_unique<BasicBlock>(parentLoop);
  if (parentLoop != nullptr) {
    parentLoop->addBasicBlock(b.get());
  }
  return b;
}

/// A BFS algorithm to determine wheter operation st and reach operation en
/// without taking loop back edge. Special case: st and en are in the same block
bool canReach(OpAbstract *st, OpAbstract *en, bool backFlag);

/// Chech whether two memory operation in DAG might have memory conflict
bool hasMemPortConflict(OpAbstract *PrevMemOp, OpAbstract *LatMemOp);

/// Check whehter two memory operation in the inner-most loop can have
/// memory port conflict with a distance Dist.
bool hasMemPortConflict(OpAbstract *PrevMemOp, OpAbstract *LatMemOp, int Dist);

} // namespace scheduling

#endif
