#include "TOR/PassDetail.h"
// #include "mlir/Analysis/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PriorityQueue.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "TOR/Passes.h"
#include "TOR/TOR.h"
#include "TOR/Utils.h"
#include "TOR/TORDialect.h"
#include "APS/APSDialect.h"
#include "APS/APSOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
// #include "mlir/Transforms/Utils.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Schedule/SDCSchedule.h"
#include "Schedule/ResourceDB.h"
#include <fstream>
#include <iostream>
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <unordered_map>
#include <vector>
#define DEBUG_TYPE "create-tor"

namespace {
using namespace mlir;
using namespace mlir::arith;

class DisjointSet {
public:
  DisjointSet(int sz) : size(sz) {
    parent.reserve(size);
    for (int i = 0; i < size; ++i)
      parent[i] = i;
  }
  int find(int x) { return parent[x] == x ? x : parent[x] = find(parent[x]); }
  /**
   * @brief merge x to y. ALERT: x and y are NOT interchangalbe
   * @param x
   * @param y
   */
  void merge(int x, int y) { parent[find(x)] = find(y); }

private:
  int size;
  std::vector<int> parent;
};

class TimeGraph {
private:
  struct Edge {
    std::string ds;
    int from, to;
    int length;
    int tripcount;
    int II;
  };

  int numNode;
  std::vector<std::vector<Edge>> edge;
  std::vector<std::vector<Edge>> redge;
  std::vector<std::pair<int, int>> intvMap;

public:
  TimeGraph() {
    numNode = 1;
    edge.clear();
    redge.clear();
    edge.push_back(std::vector<Edge>());
    redge.push_back(std::vector<Edge>());
  }
  int addNode() {
    edge.push_back(std::vector<Edge>());
    redge.push_back(std::vector<Edge>());
    numNode += 1;
    return numNode - 1;
  }
  void addEdge(int from, int to, std::string type, int length, int II = -1,
               int tripcount = -1) {
    edge[from].push_back(Edge{type, from, to, length, tripcount, II});
    redge[to].push_back(Edge{type, from, to, length, tripcount, II});
  }
  bool isEdgeType(const std::string type, const int from, const int to) {
    for (auto sub_edge : edge[from]) {
      if (sub_edge.to == to && sub_edge.ds == type) {
        return true;
      }
    }
    return false;
  }
  mlir::Attribute makeAttr(mlir::MLIRContext *ctx, Edge &edge) {
    llvm::SmallVector<mlir::NamedAttribute, 4> attrs;
    std::string retStr(edge.ds);

    if (edge.length > 0)
      retStr += std::string(":") + std::to_string(edge.length);
    attrs.push_back(mlir::NamedAttribute(mlir::StringAttr::get(ctx, "type"),
                                         mlir::StringAttr::get(ctx, retStr)));

    if (edge.II != -1) {
      attrs.push_back(mlir::NamedAttribute(
          mlir::StringAttr::get(ctx, "pipeline"),
          mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), 1)));
      attrs.push_back(mlir::NamedAttribute(
          mlir::StringAttr::get(ctx, "II"),
          mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), edge.II)));
    }

    if (edge.tripcount != -1)
      attrs.push_back(mlir::NamedAttribute(
          mlir::StringAttr::get(ctx, "times"),
          mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                 edge.tripcount)));

    mlir::DictionaryAttr dict = mlir::DictionaryAttr::get(ctx, attrs);
    return dict;
  }
  void print() {
    std::cout << "-----Here is Time Graph-----" << std::endl;
    for (int i = 0; i < numNode; i++) {
      std::cout << i << ": ";
      for (auto e : edge[i]) {
        std::cout << e.to << "(" << e.length << ") ";
      }
      std::cout << std::endl;
    }
    std::cout << "-----------------------------" << std::endl;
  }

  void rewrite(mlir::Region &region, mlir::PatternRewriter &rewriter) {
    std::vector<std::vector<mlir::Attribute>> froms(numNode);
    std::vector<std::vector<mlir::Attribute>> attrs(numNode);

    for (int i = 0; i < numNode; i++) {
      for (auto e : edge[i]) {
        froms[e.to].push_back(mlir::IntegerAttr::get(
            mlir::IntegerType::get(region.getContext(), 32,
                                   mlir::IntegerType::Signless),
            e.from));
        attrs[e.to].push_back(makeAttr(region.getContext(), e));
      }
    }

    mlir::Location loc = region.getLoc();
    rewriter.setInsertionPointToStart(&region.front());
    auto timeGraphOp =
        rewriter.create<mlir::tor::TimeGraphOp>(loc, 0, numNode - 1);
    rewriter.createBlock(&timeGraphOp.getBodyRegion());
    rewriter.setInsertionPointToStart(timeGraphOp.getBody());

    // Assume start-time is 0
    for (int i = 1; i < numNode; i++) {
      rewriter.create<mlir::tor::SuccTimeOp>(
          loc, i, mlir::ArrayAttr::get(region.getContext(), froms[i]),
          mlir::ArrayAttr::get(region.getContext(), attrs[i]));
    }
    rewriter.create<mlir::tor::FinishOp>(loc);
  }

  /**
   * @brief remove the nodes whose indegs are 1 and length are 0
   */
  void canonicalize(std::vector<int> &newId) {
    DisjointSet dset(numNode);
    for (int i = 1; i < numNode; ++i) {
      if (redge[i].size() > 1)
        continue;
      auto &e = redge[i][0];
      if (e.ds == "static" && e.length == 0) {
        llvm::dbgs() << i << " " << e.ds << " " << e.length << "\n";
        dset.merge(i, e.from);
      }
    }

    int reducedNum = 0;
    // std::vector<int> newId(numNode, 0);
    newId.resize(numNode);

    for (int i = 0; i < numNode; ++i)
      if (dset.find(i) == i)
        newId[i] = reducedNum++;

    for (int i = 0; i < numNode; ++i)
      if (dset.find(i) != i)
        newId[i] = newId[dset.find(i)];

    llvm::dbgs() << numNode << " " << reducedNum << "\n";

    std::vector<std::vector<Edge>> oldedges(std::move(edge));
    std::vector<std::vector<Edge>> oldredges(std::move(redge));

    edge.resize(reducedNum);
    redge.resize(reducedNum);

    for (int i = 0; i < numNode; ++i)
      for (auto &e : oldedges[i]) {
        int u = dset.find(e.from), v = dset.find(e.to);
        if (u == v)
          continue;

        addEdge(newId[u], newId[v], e.ds, e.length, e.II, e.tripcount);
      }

    numNode = reducedNum;
  }
};

void setIntvAttr(mlir::Operation *op, std::pair<int, int> intv) {
  op->setAttr("starttime",
              mlir::IntegerAttr::get(
                  mlir::IntegerType::get(op->getContext(), 32,
                                         mlir::IntegerType::Signless),
                  intv.first));

  op->setAttr("endtime",
              mlir::IntegerAttr::get(
                  mlir::IntegerType::get(op->getContext(), 32,
                                         mlir::IntegerType::Signless),
                  intv.second));
}

int buildTimeGraphBlock(TimeGraph &tg, std::vector<mlir::Operation *> &vec,
                        int prev, scheduling::ScheduleBase *scheduler) {
  llvm::errs() << "buildTimeGraphBlock called with " << vec.size() << " operations\n";
  for (auto op : vec) {
    llvm::errs() << "  - Op: " << op->getName() << " dump: " << llvm::dyn_cast<StringAttr>(op->getAttr("dump")).getValue() << "\n";
  }
  
  std::set<int> timeStamp;
  std::map<int, int> ts2Node;

  for (auto op : vec) {
    auto intv = scheduler->queryOp(op);
    // this op runs in [intv.fist, intv.second + 1)
    timeStamp.insert(intv.first);
    timeStamp.insert(intv.second);
  }

  int last = -1;
  for (auto ts : timeStamp) {
    int node = -1;

    if (last != -1) {
      node = tg.addNode();
      tg.addEdge(prev, node, "static", ts - last);
    } else {
      // node = tg.addNode();
      // tg.addEdge(prev, node, "static", 0);
      node = prev;
    }
    ts2Node[ts] = node;
    prev = node;
    last = ts;
  }

  for (auto op : vec) {
    auto cycle = scheduler->queryOp(op);
    auto intv = std::make_pair(ts2Node[cycle.first], ts2Node[cycle.second]);
    setIntvAttr(op, intv);
  }

  vec.clear();
  return prev;
}

long long get_attr_num(Attribute attr) {
  if (auto int_attr = dyn_cast<mlir::IntegerAttr>(attr)) {
    return int_attr.getValue().getSExtValue();
  } else if (auto bool_attr = dyn_cast<mlir::BoolAttr>(attr)) {
    return bool_attr.getValue();
  } else {
    attr.dump();
    assert(false && "Undefined attribute");
    return 0;
  }
}

bool isOpInSameIntv(mlir::Operation* op1, mlir::Operation* op2, scheduling::ScheduleBase *scheduler) {
  auto intv1 = scheduler->queryOp(op1);
  auto intv2 = scheduler->queryOp(op2);
  return intv1.first == intv2.first && intv1.second == intv2.second;
}

std::pair<int, int> getRefTimePair(mlir::Operation* op) {
  return {get_attr_num(op->getAttr("ref_starttime")), get_attr_num(op->getAttr("ref_endtime"))};
}

std::vector<mlir::Operation*> sortedOperationsInBlock(mlir::Block &block) {
  auto cmp = [&](mlir::Operation* op1, mlir::Operation* op2) {
    auto intv1 = getRefTimePair(op1);
    auto intv2 = getRefTimePair(op2);
    if (intv1.first == intv2.first && intv1.second < intv2.second) {
      if (llvm::isa<mlir::tor::CallOp>(op1) &&
          llvm::isa<mlir::tor::AXIReadOp, mlir::tor::AXIWriteOp>(op2)) {
          return true;
      } else if (llvm::isa<mlir::tor::AXIReadOp, mlir::tor::AXIWriteOp>(op1)
                 && llvm::isa<mlir::tor::CallOp>(op2)) {
        return false;
      }
    }
    return intv1.first < intv2.first
           || (intv1.first == intv2.first && intv1.second < intv2.second);
  };
  int numOfOperations = block.getOperations().size();
  std::vector<mlir::Operation*> sortedOps;
  sortedOps.reserve(numOfOperations);
  std::vector<mlir::Operation*> unSortedOps;
  unSortedOps.reserve(numOfOperations);

  for (auto &op : block) {
    if (llvm::isa<mlir::tor::YieldOp, mlir::tor::ConditionOp,
                  mlir::tor::ReturnOp, ConstantOp>(op))
      continue;

    if (llvm::isa<mlir::tor::WhileOp, mlir::tor::ForOp>(op) || !op.hasAttr("ref_starttime")) {
      std::stable_sort(unSortedOps.begin(), unSortedOps.end(), cmp);
      sortedOps.insert(sortedOps.end(), unSortedOps.begin(), unSortedOps.end());
      unSortedOps.clear();
      sortedOps.push_back(&op);
    } else {
      auto intv1 = getRefTimePair(&op);
      llvm::errs() << "Sorting op with dump id " << llvm::dyn_cast<StringAttr>((&op)->getAttr("dump")).getValue() << ": " << intv1.first << ", " << intv1.second << "\n";
      unSortedOps.push_back(&op);
    }
  }

  if (!unSortedOps.empty()) {
    std::stable_sort(unSortedOps.begin(), unSortedOps.end(), cmp);
    sortedOps.insert(sortedOps.end(), unSortedOps.begin(), unSortedOps.end());
  }
  return sortedOps;
}

bool isAxiOp(mlir::Operation* op) {
  return llvm::isa<tor::AXIReadOp, tor::AXIWriteOp, tor::AXIReadRequestOp, tor::AXIWriteRequestOp,
                   tor::AXIBurstReadOp, tor::AXIBurstWriteOp, tor::AXIWriteResponseOp>(op);
}

int buildTimeGraph(TimeGraph &tg, mlir::Block &block, int prev, bool isPipeline,
                   int isDataflow, scheduling::ScheduleBase *scheduler) {
  std::vector<mlir::Operation*> sortedOps = sortedOperationsInBlock(block);
  int currentNode = prev;

  std::vector<mlir::Operation *> vec;
  mlir::Operation *prevOp = nullptr;
  for (auto op : sortedOps) {
    if (auto ifOp = llvm::dyn_cast<mlir::tor::IfOp>(op)) {

      currentNode = buildTimeGraphBlock(tg, vec, currentNode, scheduler);

      if (!ifOp.getElseRegion().empty()) {
        int entryNodeThen = tg.addNode();
        tg.addEdge(currentNode, entryNodeThen, "static", 0);

        int thenNode = buildTimeGraph(tg, ifOp.getThenRegion().front(),
                                      entryNodeThen, isPipeline, false, scheduler);

        int entryNodeElse = tg.addNode();
        tg.addEdge(currentNode, entryNodeElse, "static", 0);
        int elseNode = buildTimeGraph(tg, ifOp.getElseRegion().front(),
                                      entryNodeElse, isPipeline, false, scheduler);

        int nxtNode = tg.addNode();
        tg.addEdge(thenNode, nxtNode, "static", 0);
        tg.addEdge(elseNode, nxtNode, "static", 0);

        setIntvAttr(op, std::make_pair(currentNode, nxtNode));
        currentNode = nxtNode;
      } else {
        int entryNode = tg.addNode();
        tg.addEdge(currentNode, entryNode, "static", 0);
        int thenNode = buildTimeGraph(tg, ifOp.getThenRegion().front(),
                                      entryNode, isPipeline, false, scheduler);
        // int nxtNode = tg.addNode(thenNode, "static", 0);
        // int nxtNode = thenNode;

        int nxtNode = tg.addNode();
        tg.addEdge(thenNode, nxtNode, "static", 0);

        setIntvAttr(op, std::make_pair(currentNode, nxtNode));
        currentNode = nxtNode;
      }

    } else if (auto whileOp = llvm::dyn_cast<mlir::tor::WhileOp>(op)) {

      currentNode = buildTimeGraphBlock(tg, vec, currentNode, scheduler);
      int beginNode = tg.addNode();
      tg.addEdge(currentNode, beginNode, "static", 0);
      auto info = scheduler->queryLoop(op);
      bool isWhilePipeline = isPipeline || info.first == true;
      int condNode = buildTimeGraph(tg, whileOp.getBefore().front(),
                                    beginNode, isWhilePipeline, false, scheduler);
      int endNode = buildTimeGraph(tg, whileOp.getAfter().front(), condNode,
                                   isWhilePipeline, false, scheduler); // body
      int nxtNode = 0;

      if (info.first == true) {
        op->setAttr("pipeline",
                   mlir::IntegerAttr::get(
                       mlir::IntegerType::get(op->getContext(), 32), 1));

        op->setAttr("II", mlir::IntegerAttr::get(
                             mlir::IntegerType::get(op->getContext(), 32),
                             info.second));
        setPragmaStructureAttrNewValueByOp(op, "pipeline", info.second);
      }

      nxtNode = tg.addNode();
      tg.addEdge(currentNode, nxtNode, "static-while", 0, info.second);

      setIntvAttr(op, std::make_pair(currentNode, endNode));
      currentNode = nxtNode;

      int extraEndNode = tg.addNode();
      tg.addEdge(nxtNode, extraEndNode, "static", 0);

      currentNode = extraEndNode;
    } else if (auto forOp = llvm::dyn_cast<mlir::tor::ForOp>(op)) {

      currentNode = buildTimeGraphBlock(tg, vec, currentNode, scheduler);
      int beginNode = tg.addNode();
      tg.addEdge(currentNode, beginNode, "static", 0);
      auto info = scheduler->queryLoop(op);
      int endNode = buildTimeGraph(tg, *forOp.getBody(), beginNode,
                                   (info.first == true || isPipeline), false, scheduler);
      int nxtNode = 0;

      if (info.first == true) {
        op->setAttr("pipeline",
                   mlir::IntegerAttr::get(
                       mlir::IntegerType::get(op->getContext(), 32), 1));

        op->setAttr("II", mlir::IntegerAttr::get(
                             mlir::IntegerType::get(op->getContext(), 32),
                             info.second));
        setPragmaStructureAttrNewValueByOp(op, "pipeline", info.second);
      }

      nxtNode = tg.addNode();
      tg.addEdge(currentNode, nxtNode, "static-for", 0, info.second);

      setIntvAttr(op, std::make_pair(currentNode, endNode));
      currentNode = nxtNode;
    } else if (auto callOp = llvm::dyn_cast<mlir::tor::CallOp>(op)) {
      if (isDataflow) {
        // Do not assign callops inside dataflow region to the timegraph
        continue;
      }
      auto getFuncOp = [&](mlir::tor::CallOp callOp) {
        auto module = callOp->getParentOfType<ModuleOp>();
        for (auto designOp : module.getOps<tor::DesignOp>()) {
          for (auto funcOp : designOp.getOps<tor::FuncOp>()) {
            if (funcOp.getName() == callOp.getCallee()) {
              return funcOp;
            }
          }
        }
        assert(false);
      };
      if (getFuncOp(callOp)->hasAttr("II")) {
        vec.push_back(op);
        continue;
      }
      int nxtNode = buildTimeGraphBlock(tg, vec, currentNode, scheduler);
      if (nxtNode == currentNode && tg.isEdgeType("static-call", prev, currentNode)
          && llvm::isa<mlir::tor::CallOp>(prevOp) && isOpInSameIntv(op, prevOp, scheduler)) {
        setIntvAttr(op, std::make_pair(prev, currentNode));
      } else {
        currentNode = nxtNode;
        nxtNode = tg.addNode();
        tg.addEdge(currentNode, nxtNode, "static-call", 0);
        setIntvAttr(op, std::make_pair(currentNode, nxtNode));
        prev = currentNode;
        currentNode = nxtNode;
      }
    } else if (!isPipeline && isAxiOp(op)) {
      // schedule op not in order, this logic does not work for pipeline,
      // rerorder op first, then add edge for pipeline
      if (prevOp != nullptr && !isAxiOp(prevOp)) {
        currentNode = buildTimeGraphBlock(tg, vec, currentNode, scheduler);
        int nxtNode = tg.addNode();
        tg.addEdge(currentNode, nxtNode, "static", 0);
        currentNode = nxtNode;
      }
      vec.push_back(op);
    } else {
      if (!isPipeline && !vec.empty() && isAxiOp(vec.back())) {
        currentNode = buildTimeGraphBlock(tg, vec, currentNode, scheduler);
      }
      if (llvm::isa<mlir::tor::YieldOp>(op))
        continue;
      if (llvm::isa<mlir::tor::ConditionOp>(op))
        continue;
      if (llvm::isa<mlir::tor::ReturnOp>(op))
        continue;
      if (llvm::isa<ConstantOp>(op))
        continue;

      vec.push_back(op);
    }
    prevOp = op;
  }

  if (!vec.empty())
    currentNode = buildTimeGraphBlock(tg, vec, currentNode, scheduler);

  return currentNode;
}

mlir::LogicalResult removeExtraEdges(mlir::tor::FuncOp funcOp, TimeGraph *tg) {
  std::vector<int> newId;
  tg->canonicalize(newId);
  if (funcOp
          .walk([&](mlir::Operation *op) {
            if (op->getDialect()->getNamespace() !=
                mlir::tor::TORDialect::getDialectNamespace())
              return mlir::WalkResult::skip();
            if (auto starttime =
                    op->getAttrOfType<mlir::IntegerAttr>("starttime")) {
              auto t = starttime.getInt();
              op->setAttr("starttime",
                          mlir::IntegerAttr::get(
                              mlir::IntegerType::get(funcOp.getContext(), 32),
                              newId[t]));
            }
            if (auto endtime =
                    op->getAttrOfType<mlir::IntegerAttr>("endttime")) {
              auto t = endtime.getInt();
              op->setAttr("endtime",
                          mlir::IntegerAttr::get(
                              mlir::IntegerType::get(funcOp.getContext(), 32),
                              newId[t]));
            }

            return mlir::WalkResult::advance();
          })
          .wasInterrupted())
    return mlir::failure();
  return mlir::success();
}

bool isMultiCycleOp(mlir::Operation* op) {
  if (llvm::isa<arith::SIToFPOp, arith::FPToSIOp, arith::ExtFOp, arith::TruncFOp,
                math::PowFOp, math::CeilOp, math::FloorOp, math::RoundOp,
                math::ExpOp, math::SqrtOp, math::LogOp, math::CosOp, math::TanOp,
                math::TanhOp, math::SinOp, tor::MulIOp, tor::MulSIOp, tor::MulUIOp,
                tor::AddFOp, tor::SubFOp, tor::MulFOp, tor::DivFOp,
                tor::CmpFOp, arith::DivSIOp, tor::MacIOp, tor::MacFOp,
                arith::RemSIOp, arith::DivUIOp, arith::RemUIOp,
                arith::UIToFPOp, arith::FPToUIOp,
                aps::CpuRfRead, aps::CpuRfWrite>(op)) {
      return true;
  }
  return false;
}

void calculateFuncOpResourceUsage(mlir::tor::FuncOp funcOp, scheduling::ResourceDB &RDB) {
  auto designOp = llvm::dyn_cast<tor::DesignOp>(funcOp->getParentOp());
  std::vector<std::unordered_map<int, int>> usageCnts(RDB.getNumResource());
  std::vector<int> usages(RDB.getNumResource(), 0);
  llvm::SmallSet<mlir::tor::FuncOp, 4> callFuncOps;
  funcOp.walk([&](mlir::Operation* op) {
    if (isMultiCycleOp(op)) {
      int startTime = op->getAttrOfType<IntegerAttr>("ref_starttime").getInt();
      int rcsId = RDB.getResourceID(op);
      ++usageCnts[rcsId][startTime];
      if (usages[rcsId] < usageCnts[rcsId][startTime]) {
        usages[rcsId] = usageCnts[rcsId][startTime];
      }
    } else if (auto callOp = llvm::dyn_cast<mlir::tor::CallOp>(op)) {
      auto callFuncOp = designOp.lookupSymbol<tor::FuncOp>(callOp.getCallee());
      callFuncOps.insert(callFuncOp);
    }
  });

  for (auto callFuncOp: callFuncOps) {
    const auto &callOpUsages = RDB.getUsage(callFuncOp);
    for (size_t i = 0; i < callOpUsages.size(); ++i) {
      usages[i] += callOpUsages[i];
    }
  }
  RDB.addUsage(funcOp, usages);
}

bool isDataflowRegionResourceConstraintSatisfied(mlir::tor::FuncOp funcOp,
                                                 scheduling::ResourceDB &RDB) {
  auto designOp = llvm::dyn_cast<tor::DesignOp>(funcOp->getParentOp());
  std::vector<int> parallelUsage;
  for (auto &op: funcOp.getRegion().front()) {
    if (auto callOp = llvm::dyn_cast<mlir::tor::CallOp>(op)) {
      auto callFuncOp = designOp.lookupSymbol<tor::FuncOp>(callOp.getCallee());
      const auto usage = RDB.getUsage(callFuncOp);
      size_t size = std::max(usage.size(), parallelUsage.size());
      parallelUsage.resize(size, 0);
      for (size_t k = 0; k < usage.size(); ++k) {
        parallelUsage[k] += usage[k];
      }
    }
  }
  return RDB.isUsageSatisfied(parallelUsage);
}

void setReferenceAttr(mlir::Operation *op, std::pair<int, int> intv) {
  op->setAttr("ref_starttime",
              mlir::IntegerAttr::get(
                  mlir::IntegerType::get(op->getContext(), 32,
                                         mlir::IntegerType::Signless),
                  intv.first));

  op->setAttr("ref_endtime",
              mlir::IntegerAttr::get(
                  mlir::IntegerType::get(op->getContext(), 32,
                                         mlir::IntegerType::Signless),
                  intv.second));
}

void queryAndSetReferenceAttr(mlir::Operation * op, scheduling::ScheduleBase *scheduler) {
  auto intv = scheduler->queryOp(op);
  setReferenceAttr(op, intv);
}

void updateIntv(std::pair<int, int>& intv, std::pair<int, int> queryIntv) {
  intv.first = std::min(queryIntv.first, intv.first);
  intv.second = std::max(queryIntv.second, intv.second);
}

std::pair<int, int> queryAllOps(mlir::Block &block, scheduling::ScheduleBase *scheduler, bool isDataflow = false) {
  std::pair<int, int> intv = {INT_MAX, INT_MIN};
  for (auto &op : block) {
    std::pair<int, int> queryIntv = {INT_MAX, INT_MIN};
    if (auto ifOp = llvm::dyn_cast<mlir::tor::IfOp>(op)) {
      if (!ifOp.getElseRegion().empty()) {
        updateIntv(queryIntv, queryAllOps(ifOp.getElseRegion().front(), scheduler, isDataflow));
      }
      updateIntv(queryIntv, queryAllOps(ifOp.getThenRegion().front(), scheduler, isDataflow));
    } else if (auto whileOp = llvm::dyn_cast<mlir::tor::WhileOp>(op)) {
      updateIntv(queryIntv, queryAllOps(whileOp.getBefore().front(), scheduler, isDataflow));
      updateIntv(queryIntv, queryAllOps(whileOp.getAfter().front(), scheduler, isDataflow));
    } else if (auto forOp = llvm::dyn_cast<mlir::tor::ForOp>(op)) {
      updateIntv(queryIntv, queryAllOps(*forOp.getBody(), scheduler, isDataflow));
    } else if (auto callOp = llvm::dyn_cast<mlir::tor::CallOp>(op)) {
      if (isDataflow) {
        // Do not assign callops inside dataflow region to the timegraph
        continue;
      }
      queryIntv = scheduler->queryOp(&op);
    } else {
      if (llvm::isa<mlir::tor::YieldOp>(op))
        continue;
      if (llvm::isa<mlir::tor::ConditionOp>(op))
        continue;
      if (llvm::isa<mlir::tor::ReturnOp>(op))
        continue;
      if (llvm::isa<ConstantOp>(op))
        continue;

      queryIntv = scheduler->queryOp(&op);
    }
    setReferenceAttr(&op, queryIntv);
    intv.first = std::min(queryIntv.first, intv.first);
    intv.second = std::max(queryIntv.second, intv.second);
  }
  return intv;
}

mlir::LogicalResult scheduleOps(mlir::tor::FuncOp funcOp,
                                mlir::PatternRewriter &rewriter,
                                scheduling::ResourceDB &RDB) {
  LLVM_DEBUG( llvm::dbgs() << "===============================\n";
    llvm::dbgs() << "Scheduling function: ";
    llvm::dbgs() << funcOp.getName() << "\n";
    llvm::dbgs() << "===============================\n";
  );
  using namespace scheduling;
  auto name =
      funcOp->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()).str();
  if (auto strategy = funcOp->getAttrOfType<StringAttr>("strategy")) {
    llvm::errs() << name << " is dynamic. No static scheduling\n";
    if (strategy.getValue().str() == "dynamic")
      return mlir::success();
  }

  std::unique_ptr<SDCSchedule> scheduler =
      std::make_unique<SDCSchedule>(SDCSchedule(funcOp.getOperation(), RDB));

  if (!mlir::succeeded(scheduler->runSchedule())) {
//    llvm::errs() << "Schedule Succeeded\n";
//  else {
    llvm::errs() << "Schedule Failed\n";
    return mlir::failure();
  }

  // Debug: print scheduled times for all operations
  llvm::dbgs() << "\n=== Scheduled times after runSchedule() ===\n";
  funcOp.walk([&](Operation *op) {
    if (op->hasAttr("dump")) {
      auto dumpId = op->getAttr("dump");
      auto interval = scheduler->queryOp(op);
      llvm::dbgs() << "Op " << dumpId << " (" << op->getName() << "): startTime = " 
                   << interval.first << ", endTime = " << interval.second << "\n";
    }
  });
  llvm::dbgs() << "=========================================\n\n";

  bool isDataflow = false;
  if (auto flag = funcOp->getAttrOfType<IntegerAttr>("dataflow")) {
    isDataflow = flag.getInt() == 1;
  }

  if (isDataflow) {
    if (!isDataflowRegionResourceConstraintSatisfied(funcOp, RDB)) {
      setPragmaStructureAttrStatusByOp(funcOp.getOperation(), "dataflow",
                                       false);
      llvm::errs() << "Schedule Failed, dataflow region resource not satisfied\n";
      return mlir::failure();
    } else {
      setPragmaStructureAttrStatusByOp(funcOp.getOperation(), "dataflow");
    }
  }

  // Set ref_* attributes only (time graph generation moved to separate pass)
  queryAllOps(funcOp.getRegion().front(), scheduler.get(), isDataflow);

  // Mark function as scheduled so time graph pass knows to process it
  funcOp->setAttr("scheduled", BoolAttr::get(funcOp.getContext(), true));

  calculateFuncOpResourceUsage(funcOp, RDB);
  return mlir::success();
}

struct FuncOpsLowering : public OpRewritePattern<mlir::tor::DesignOp> {
  using OpRewritePattern<mlir::tor::DesignOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::tor::DesignOp designOp,
                                PatternRewriter &rewriter) const override {
    if (designOp->getAttr("schedule")) {
      return failure();
    }
    llvm::SmallVector<tor::FuncOp> funcOps;
    designOp.walk([&](tor::FuncOp funcOp){
      funcOps.push_back(funcOp);
    });

    std::string filename;
    if (auto attr = funcOps.front()->getAttrOfType<mlir::StringAttr>("resource"))
      filename = attr.getValue().str();
    else
      assert(0 && "A path to the resource constraint file must be specified\n");

    std::ifstream istrm(filename, std::ios::in);
    nlohmann::json config;
    istrm >> config;
    auto RDB = scheduling::ResourceDB(config);

    for (auto funcOp: funcOps) {
      if (failed(scheduleOps(funcOp, rewriter, RDB))) {
        designOp->setAttr("schedule", BoolAttr::get(getContext(), false));
        return failure();
      }
    }
    designOp->setAttr("schedule", BoolAttr::get(getContext(), true));
    return success();
  }
};

struct TORSchedulePass : public TORScheduleBase<TORSchedulePass> {
  void runOnOperation() override {
    mlir::tor::DesignOp designOp = getOperation();
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<FuncOpsLowering>(&getContext());
    GreedyRewriteConfig config;
    config.setStrictness(GreedyRewriteStrictness::ExistingOps);
    if (failed(applyOpPatternsAndFold(designOp.getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
    auto scheduleRes = designOp->getAttrOfType<mlir::BoolAttr>("schedule");
    if (!scheduleRes.getValue()) {
      signalPassFailure();
    }
  }
};
} // namespace

namespace mlir {
  std::unique_ptr<OperationPass<mlir::tor::DesignOp>> createTORSchedulePass() {
    return std::make_unique<TORSchedulePass>();
  }
} // namespace mlir
