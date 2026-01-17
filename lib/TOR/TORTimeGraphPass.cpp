#include "TOR/Passes.h"
#include "TOR/TOR.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <map>
#include <set>
#include <vector>

#define DEBUG_TYPE "tor-time-graph"

namespace {
using namespace mlir;
using namespace mlir::arith;

class TimeGraph {
private:
  struct Edge {
    std::string ds;
    int from, to;
    int length;
    int tripcount;
    int II;
  };

  std::vector<std::vector<Edge>> edge;
  std::vector<std::vector<Edge>> redge;

public:
  int numNode;

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

    for (int i = 1; i < numNode; i++) {
      rewriter.create<mlir::tor::SuccTimeOp>(
          loc, i, mlir::ArrayAttr::get(region.getContext(), froms[i]),
          mlir::ArrayAttr::get(region.getContext(), attrs[i]));
    }
    rewriter.create<mlir::tor::FinishOp>(loc);
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

std::pair<int, int> getRefTimePair(mlir::Operation* op) {
  return {get_attr_num(op->getAttr("ref_starttime")),
          get_attr_num(op->getAttr("ref_endtime"))};
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

    if (llvm::isa<mlir::tor::WhileOp, mlir::tor::ForOp>(op) ||
        !op.hasAttr("ref_starttime")) {
      std::stable_sort(unSortedOps.begin(), unSortedOps.end(), cmp);
      sortedOps.insert(sortedOps.end(), unSortedOps.begin(), unSortedOps.end());
      unSortedOps.clear();
      sortedOps.push_back(&op);
    } else {
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
  return llvm::isa<tor::AXIReadOp, tor::AXIWriteOp, tor::AXIReadRequestOp,
                   tor::AXIWriteRequestOp, tor::AXIBurstReadOp,
                   tor::AXIBurstWriteOp, tor::AXIWriteResponseOp>(op);
}

int buildTimeGraphBlock(TimeGraph &tg, std::vector<mlir::Operation *> &vec,
                        int prev) {
  std::set<int> timeStamp;
  std::map<int, int> ts2Node;

  // Read ref_* attributes to get cycle times
  for (auto op : vec) {
    auto refTime = getRefTimePair(op);
    timeStamp.insert(refTime.first);
    timeStamp.insert(refTime.second);
  }

  int last = -1;
  for (auto ts : timeStamp) {
    int node = -1;

    if (last != -1) {
      node = tg.addNode();
      tg.addEdge(prev, node, "static", ts - last);
    } else {
      node = prev;
    }
    ts2Node[ts] = node;
    prev = node;
    last = ts;
  }

  for (auto op : vec) {
    auto refTime = getRefTimePair(op);
    auto intv = std::make_pair(ts2Node[refTime.first], ts2Node[refTime.second]);
    setIntvAttr(op, intv);
  }

  vec.clear();
  return prev;
}

int buildTimeGraph(TimeGraph &tg, mlir::Block &block, int prev, bool isPipeline,
                   bool isDataflow);

int buildTimeGraph(TimeGraph &tg, mlir::Block &block, int prev, bool isPipeline,
                   bool isDataflow) {
  std::vector<mlir::Operation*> sortedOps = sortedOperationsInBlock(block);
  int currentNode = prev;

  std::vector<mlir::Operation *> vec;
  mlir::Operation *prevOp = nullptr;

  for (auto op : sortedOps) {
    if (auto ifOp = llvm::dyn_cast<mlir::tor::IfOp>(op)) {
      currentNode = buildTimeGraphBlock(tg, vec, currentNode);

      if (!ifOp.getElseRegion().empty()) {
        int entryNodeThen = tg.addNode();
        tg.addEdge(currentNode, entryNodeThen, "static", 0);

        int thenNode = buildTimeGraph(tg, ifOp.getThenRegion().front(),
                                      entryNodeThen, isPipeline, false);

        int entryNodeElse = tg.addNode();
        tg.addEdge(currentNode, entryNodeElse, "static", 0);
        int elseNode = buildTimeGraph(tg, ifOp.getElseRegion().front(),
                                      entryNodeElse, isPipeline, false);

        int nxtNode = tg.addNode();
        tg.addEdge(thenNode, nxtNode, "static", 0);
        tg.addEdge(elseNode, nxtNode, "static", 0);

        setIntvAttr(op, std::make_pair(currentNode, nxtNode));
        currentNode = nxtNode;
      } else {
        int entryNode = tg.addNode();
        tg.addEdge(currentNode, entryNode, "static", 0);
        int thenNode = buildTimeGraph(tg, ifOp.getThenRegion().front(),
                                      entryNode, isPipeline, false);

        int nxtNode = tg.addNode();
        tg.addEdge(thenNode, nxtNode, "static", 0);

        setIntvAttr(op, std::make_pair(currentNode, nxtNode));
        currentNode = nxtNode;
      }

    } else if (auto whileOp = llvm::dyn_cast<mlir::tor::WhileOp>(op)) {
      currentNode = buildTimeGraphBlock(tg, vec, currentNode);
      int beginNode = tg.addNode();
      tg.addEdge(currentNode, beginNode, "static", 0);

      bool isWhilePipeline = isPipeline || op->hasAttr("pipeline");
      int II = op->hasAttr("II") ?
               op->getAttrOfType<IntegerAttr>("II").getInt() : -1;

      int condNode = buildTimeGraph(tg, whileOp.getBefore().front(),
                                    beginNode, isWhilePipeline, false);
      int endNode = buildTimeGraph(tg, whileOp.getAfter().front(), condNode,
                                   isWhilePipeline, false);

      int nxtNode = tg.addNode();
      tg.addEdge(currentNode, nxtNode, "static-while", 0, II);

      setIntvAttr(op, std::make_pair(currentNode, endNode));
      currentNode = nxtNode;

      int extraEndNode = tg.addNode();
      tg.addEdge(nxtNode, extraEndNode, "static", 0);
      currentNode = extraEndNode;

    } else if (auto forOp = llvm::dyn_cast<mlir::tor::ForOp>(op)) {
      currentNode = buildTimeGraphBlock(tg, vec, currentNode);
      int beginNode = tg.addNode();
      tg.addEdge(currentNode, beginNode, "static", 0);

      bool isForPipeline = isPipeline || op->hasAttr("pipeline");
      int II = op->hasAttr("II") ?
               op->getAttrOfType<IntegerAttr>("II").getInt() : -1;

      int endNode = buildTimeGraph(tg, *forOp.getBody(), beginNode,
                                   isForPipeline, false);

      int nxtNode = tg.addNode();
      tg.addEdge(currentNode, nxtNode, "static-for", 0, II);

      setIntvAttr(op, std::make_pair(currentNode, endNode));
      currentNode = nxtNode;

    } else if (auto callOp = llvm::dyn_cast<mlir::tor::CallOp>(op)) {
      if (isDataflow) {
        continue;
      }

      // Check if function is pipelined
      auto getFuncOp = [&](mlir::tor::CallOp callOp) -> mlir::tor::FuncOp {
        auto module = callOp->getParentOfType<ModuleOp>();
        for (auto designOp : module.getOps<tor::DesignOp>()) {
          for (auto funcOp : designOp.getOps<tor::FuncOp>()) {
            if (funcOp.getName() == callOp.getCallee()) {
              return funcOp;
            }
          }
        }
        return nullptr;
      };

      auto calledFunc = getFuncOp(callOp);
      if (calledFunc && calledFunc->hasAttr("II")) {
        vec.push_back(op);
        continue;
      }

      int nxtNode = buildTimeGraphBlock(tg, vec, currentNode);
      if (nxtNode == currentNode && tg.isEdgeType("static-call", prev, currentNode) &&
          prevOp && llvm::isa<mlir::tor::CallOp>(prevOp)) {
        auto prevRefTime = getRefTimePair(prevOp);
        auto curRefTime = getRefTimePair(op);
        if (prevRefTime.first == curRefTime.first &&
            prevRefTime.second == curRefTime.second) {
          setIntvAttr(op, std::make_pair(prev, currentNode));
        } else {
          currentNode = nxtNode;
          nxtNode = tg.addNode();
          tg.addEdge(currentNode, nxtNode, "static-call", 0);
          setIntvAttr(op, std::make_pair(currentNode, nxtNode));
          prev = currentNode;
          currentNode = nxtNode;
        }
      } else {
        currentNode = nxtNode;
        nxtNode = tg.addNode();
        tg.addEdge(currentNode, nxtNode, "static-call", 0);
        setIntvAttr(op, std::make_pair(currentNode, nxtNode));
        prev = currentNode;
        currentNode = nxtNode;
      }

    } else if (!isPipeline && isAxiOp(op)) {
      if (prevOp != nullptr && !isAxiOp(prevOp)) {
        currentNode = buildTimeGraphBlock(tg, vec, currentNode);
        int nxtNode = tg.addNode();
        tg.addEdge(currentNode, nxtNode, "static", 0);
        currentNode = nxtNode;
      }
      vec.push_back(op);
    } else {
      if (!isPipeline && !vec.empty() && isAxiOp(vec.back())) {
        currentNode = buildTimeGraphBlock(tg, vec, currentNode);
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
    currentNode = buildTimeGraphBlock(tg, vec, currentNode);

  return currentNode;
}

struct FuncOpsTimeGraphGeneration : public OpRewritePattern<mlir::tor::FuncOp> {
  using OpRewritePattern<mlir::tor::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::tor::FuncOp funcOp,
                                PatternRewriter &rewriter) const override {
    llvm::errs() << "[TimeGraph] Checking function: " << funcOp.getName() << "\n";

    // Check if already has time graph
    if (!funcOp.getBody().front().getOps<tor::TimeGraphOp>().empty()) {
      llvm::errs() << "[TimeGraph] Already has time graph, skipping\n";
      return failure();
    }

    // Check if scheduling was done
    if (!funcOp->hasAttr("scheduled")) {
      llvm::errs() << "[TimeGraph] Not scheduled, skipping\n";
      return failure();
    }

    // Check if dynamic strategy (skip time graph generation)
    if (auto strategy = funcOp->getAttrOfType<StringAttr>("strategy")) {
      if (strategy.getValue().str() == "dynamic") {
        llvm::errs() << "[TimeGraph] Dynamic strategy, skipping\n";
        return failure();
      }
    }

    llvm::errs() << "[TimeGraph] Building time graph...\n";

    bool isPipeline = funcOp->hasAttr("pipeline");
    bool isDataflow = false;
    if (auto flag = funcOp->getAttrOfType<IntegerAttr>("dataflow")) {
      isDataflow = flag.getInt() == 1;
    }

    TimeGraph *tg = new TimeGraph();
    buildTimeGraph(*tg, funcOp.getRegion().front(), 0, isPipeline, isDataflow);

    llvm::errs() << "[TimeGraph] Built graph with " << tg->numNode << " nodes\n";
    llvm::errs() << "[TimeGraph] Creating TimeGraphOp...\n";

    tg->rewrite(funcOp.getBody(), rewriter);

    llvm::errs() << "[TimeGraph] Successfully generated time graph for function: "
                            << funcOp.getName() << "\n";

    return success();
  }
};

struct TORTimeGraphPass
    : public PassWrapper<TORTimeGraphPass, OperationPass<mlir::tor::DesignOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TORTimeGraphPass)

  StringRef getArgument() const final { return "tor-time-graph"; }
  StringRef getDescription() const final {
    return "Generate time graph from ref_* attributes";
  }

  void runOnOperation() override {
    llvm::errs() << "[TORTimeGraphPass] Running pass\n";
    mlir::tor::DesignOp designOp = getOperation();
    OpBuilder builder(&getContext());

    // Directly process each function
    for (auto funcOp : designOp.getOps<tor::FuncOp>()) {
      llvm::errs() << "[TORTimeGraphPass] Checking function: " << funcOp.getName() << "\n";

      // Check if already has time graph
      if (!funcOp.getBody().front().getOps<tor::TimeGraphOp>().empty()) {
        llvm::errs() << "[TORTimeGraphPass] Already has time graph, skipping\n";
        continue;
      }

      // Check if scheduling was done
      if (!funcOp->hasAttr("scheduled")) {
        llvm::errs() << "[TORTimeGraphPass] Not scheduled, skipping\n";
        continue;
      }

      // Check if dynamic strategy (skip time graph generation)
      if (auto strategy = funcOp->getAttrOfType<StringAttr>("strategy")) {
        if (strategy.getValue().str() == "dynamic") {
          llvm::errs() << "[TORTimeGraphPass] Dynamic strategy, skipping\n";
          continue;
        }
      }

      llvm::errs() << "[TORTimeGraphPass] Building time graph...\n";

      bool isPipeline = funcOp->hasAttr("pipeline");
      bool isDataflow = false;
      if (auto flag = funcOp->getAttrOfType<IntegerAttr>("dataflow")) {
        isDataflow = flag.getInt() == 1;
      }

      TimeGraph *tg = new TimeGraph();
      buildTimeGraph(*tg, funcOp.getRegion().front(), 0, isPipeline, isDataflow);

      llvm::errs() << "[TORTimeGraphPass] Built graph with " << tg->numNode << " nodes\n";
      llvm::errs() << "[TORTimeGraphPass] Creating TimeGraphOp...\n";

      // Create a PatternRewriter from OpBuilder
      class SimpleRewriter : public PatternRewriter {
      public:
        SimpleRewriter(MLIRContext *ctx) : PatternRewriter(ctx) {}
      };
      SimpleRewriter rewriter(&getContext());

      tg->rewrite(funcOp.getBody(), rewriter);

      llvm::errs() << "[TORTimeGraphPass] Successfully generated time graph for function: "
                              << funcOp.getName() << "\n";
    }
  }
};

} // namespace

namespace mlir {
std::unique_ptr<OperationPass<mlir::tor::DesignOp>> createTORTimeGraphPass() {
  return std::make_unique<TORTimeGraphPass>();
}
} // namespace mlir
