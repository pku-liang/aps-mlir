#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Debug.h"

#include "TOR/GenTimegraph.h"
#include "TOR/PassDetail.h"
#include "TOR/Passes.h"
#include "TOR/TOR.h"
#include "TOR/TORDialect.h"
#include <queue>
#include <stack>
#include <map>
#include <iomanip>
#include <algorithm>

#define DEBUG_TYPE "count-cycles"

namespace {
using namespace mlir;
using namespace mlir::arith;
using namespace mlir::func;

struct TimeEdge {
  int from, to, isFor, isCall, isWhile;
  int64_t weight;
  TimeEdge() : weight(0), isFor(false), isCall(false), isWhile(false){};
};

void timeGraphClear(SmallVector<SmallVector<TimeEdge>> &timeGraph) {
  for (int i = 0, e = timeGraph.size(); i < e; i++) {
    timeGraph[i].clear();
  }
}

int64_t get_attr_num(Attribute attr) {
  if (auto int_attr = dyn_cast<mlir::IntegerAttr>(attr)) {
    return int_attr.getValue().getSExtValue();
  } else if (auto bool_attr = dyn_cast<mlir::BoolAttr>(attr)) {
    return bool_attr.getValue();
  } else {
    LLVM_DEBUG(attr.dump());
    assert(false && "Undefined attribute");
    return 0;
  }
}

arith::ConstantOp getUpperBoundConstantOp(tor::ForOp forOp) {
  if (BlockArgument arg = dyn_cast<BlockArgument>(forOp.getUpperBound())) {
    if (tor::ForOp parentForOp =
            dyn_cast<tor::ForOp>(arg.getOwner()->getParentOp())) {
      if (auto ubCstOp =
              parentForOp.getUpperBound().getDefiningOp<arith::ConstantOp>()) {
        return ubCstOp;
      } else {
        return getUpperBoundConstantOp(parentForOp);
      }
    }
  }
  if (auto subIOp = forOp.getUpperBound().getDefiningOp<tor::SubIOp>()) {
    if (BlockArgument arg = dyn_cast<BlockArgument>(subIOp.getLhs())) {
      if (tor::ForOp parentForOp =
              dyn_cast<tor::ForOp>(arg.getOwner()->getParentOp())) {
        if (auto ubCstOp = parentForOp.getUpperBound()
                               .getDefiningOp<arith::ConstantOp>()) {
          return ubCstOp;
        } else {
          return getUpperBoundConstantOp(parentForOp);
        }
      }
    } else if (auto addIOp = subIOp.getLhs().getDefiningOp<tor::AddIOp>()) {
      if (BlockArgument arg = dyn_cast<BlockArgument>(addIOp.getLhs())) {
        if (tor::ForOp parentForOp =
                dyn_cast<tor::ForOp>(arg.getOwner()->getParentOp())) {
          if (auto ubCstOp = parentForOp.getUpperBound()
                                 .getDefiningOp<arith::ConstantOp>()) {
            return ubCstOp;
          } else {
            return getUpperBoundConstantOp(parentForOp);
          }
        }
      }
    }
  }
  return NULL;
}

arith::ConstantOp getLowerBoundConstantOp(tor::ForOp forOp) {
  if (BlockArgument arg = dyn_cast<BlockArgument>(forOp.getLowerBound())) {
    if (tor::ForOp parentForOp =
            dyn_cast<tor::ForOp>(arg.getOwner()->getParentOp())) {
      if (auto ubCstOp =
              parentForOp.getLowerBound().getDefiningOp<arith::ConstantOp>()) {
        return ubCstOp;
      } else {
        return getLowerBoundConstantOp(parentForOp);
      }
    }
  }
  if (auto addIOp = forOp.getLowerBound().getDefiningOp<tor::AddIOp>()) {
    if (BlockArgument arg = dyn_cast<BlockArgument>(addIOp.getLhs())) {
      if (tor::ForOp parentForOp =
              dyn_cast<tor::ForOp>(arg.getOwner()->getParentOp())) {
        if (auto ubCstOp = parentForOp.getLowerBound()
                               .getDefiningOp<arith::ConstantOp>()) {
          return ubCstOp;
        } else {
          return getLowerBoundConstantOp(parentForOp);
        }
      }
    }
  }
  return NULL;
}

int64_t getTORConstantTripCount(tor::ForOp forOp) {
  if (forOp->getAttr("tripcount-max")) {
    return dyn_cast<mlir::IntegerAttr>(forOp->getAttr("tripcount-max")).getInt();
  }
  auto lbCstOp = forOp.getLowerBound().getDefiningOp<arith::ConstantOp>();
  auto ubCstOp = forOp.getUpperBound().getDefiningOp<arith::ConstantOp>();
  auto stepCstOp = forOp.getStep().getDefiningOp<arith::ConstantOp>();
  if (!ubCstOp) {
    ubCstOp = getUpperBoundConstantOp(forOp);
  }
  if (!lbCstOp) {
    lbCstOp = getLowerBoundConstantOp(forOp);
  }
  if (!(lbCstOp && ubCstOp && stepCstOp)) {
    return -1;
  }
  return llvm::divideCeil(dyn_cast<mlir::IntegerAttr>(ubCstOp.getValue()).getInt() -
                           dyn_cast<mlir::IntegerAttr>(lbCstOp.getValue()).getInt() + 1,
                       dyn_cast<mlir::IntegerAttr>(stepCstOp.getValue()).getInt());
}

void getTimeGraph(tor::TimeGraphOp timeGraphOp,
                  SmallVector<SmallVector<TimeEdge>> &timeGraph,
                  DenseMap<int, tor::CallOp> startTimeCallOpMap,
                  DenseMap<llvm::StringRef, long long> funcOpCycleMap) {
  for (auto &block : timeGraphOp.getRegion()) {
    for (auto &op : block) {
      if (auto succOp = dyn_cast<tor::SuccTimeOp>(op)) {
        for (unsigned i = 0; i < succOp.getPoints().size(); ++i) {
          auto from = succOp.getPoints()[i];
          TimeEdge timeEdge;
          timeEdge.to = succOp.getTime();
          timeEdge.from = get_attr_num(from);
          for (auto attr : dyn_cast<mlir::DictionaryAttr>(succOp.getEdges()[i])) {
            auto edge_attr = attr.getValue();
            if (auto str_attr = dyn_cast<mlir::StringAttr>(edge_attr)) {
              llvm::StringRef strAttrStr = str_attr.getValue();
              if (strAttrStr.find(':') != StringRef::npos) {
                int colonIndex = strAttrStr.find(':');
                llvm::StringRef numberStr = strAttrStr.substr(
                    colonIndex + 1, strAttrStr.size() - colonIndex - 1);
                numberStr.getAsInteger(10, timeEdge.weight);
              }
              if (strAttrStr.find("for") != StringRef::npos) {
                timeEdge.isFor = true;
              }
              if (strAttrStr.find("call") != StringRef::npos &&
                  startTimeCallOpMap.count(timeEdge.from)) {
                timeEdge.isCall = true;
                tor::CallOp callOp = startTimeCallOpMap[timeEdge.from];
                if (funcOpCycleMap.count(callOp.getCallee())) {
                  // call is 1 cycle + funct cycle
                  timeEdge.weight = 1 + funcOpCycleMap[callOp.getCallee()];
                } else {
                  // give an invalid value
                  timeEdge.weight = -1;
                }
              }
              if (strAttrStr.find("while") != StringRef::npos) {
                timeEdge.isWhile = true;
              }
            }
          }
          timeGraph[timeEdge.to].push_back(timeEdge);
        }
      }
    }
  }
}

template <typename T>
void getForOrIfOpMap(tor::FuncOp funcOp,
                     DenseMap<int, std::pair<T, bool>> &startTimeforOrIfOpMap) {
  startTimeforOrIfOpMap.clear();
  funcOp.walk([&](T forOrIfOp) {
    startTimeforOrIfOpMap[forOrIfOp.getStarttime()] =
        std::make_pair(forOrIfOp, false);
  });
}

void getCallOpMap(tor::FuncOp funcOp,
                  DenseMap<int, tor::CallOp> &startTimeCallOpMap) {
  startTimeCallOpMap.clear();
  funcOp.walk([&](tor::CallOp callOp) {
    startTimeCallOpMap[callOp.getStarttime()] = callOp;
  });
}

unsigned getIIAttrInterger(Operation *forOp) {
  if(!forOp){
        return 0;
  }
  auto attr = forOp->getAttr("II");
  if (!attr)  
  {
        return 0;
  }
  auto inattr = dyn_cast<mlir::IntegerAttr>(attr);
  if(!inattr){
    return 0;
  }
  auto value = inattr.getValue().getSExtValue();
  if(!value){
    return 0;
  }
  return value;
}

long long
cycleCount(SmallVector<SmallVector<TimeEdge>> &timeGraph,
           DenseMap<int, std::pair<tor::ForOp, bool>> &startTimeForOpMap,
           DenseMap<int, std::pair<tor::IfOp, bool>> &startTimeIfOpMap,
           DenseMap<int, std::pair<tor::WhileOp, bool>> &startTimeWhileOpMap,
           int startVertex) {
  int vertexNum = timeGraph.size() - 1;
  SmallVector<int> outDegreeNum(vertexNum + 1, 0);
  SmallVector<long long> distanceSum(vertexNum + 1, 0);
  for (int i = 0; i <= vertexNum; i++) {
    for (auto timeEdge : timeGraph[i]) {
      outDegreeNum[timeEdge.from] += 1;
    }
  }
  std::priority_queue<int, std::vector<int>, std::greater<int>> vertexQueue;
  for (int i = 0; i <= vertexNum; i++) {
    if (!outDegreeNum[i]) {
      vertexQueue.push(i);
    }
  }
  while (!vertexQueue.empty()) {
    int current = vertexQueue.top();
    vertexQueue.pop();
    for (auto timeEdge : timeGraph[current]) {
      if (startTimeIfOpMap.count(timeEdge.from)) {
        // if (!startTimeIfOpMap[timeEdge.from].second) {
        //   llvm::errs() << "if can't calculate cycle\n";
        //   startTimeIfOpMap[timeEdge.from].second = true;
        // }
        tor::IfOp IfOp = startTimeIfOpMap[timeEdge.from].first;
        
        if (distanceSum[timeEdge.to] > distanceSum[timeEdge.from]) {
          distanceSum[timeEdge.from] = distanceSum[timeEdge.to];
        }
        outDegreeNum[timeEdge.from] -= 1;
        if (outDegreeNum[timeEdge.from] == 0) {
          vertexQueue.push(timeEdge.from);
        }
        IfOp->setAttr("cycle",
                         mlir::IntegerAttr::get(
                             mlir::IntegerType::get(IfOp->getContext(), 64),
                             distanceSum[timeEdge.from] - distanceSum[IfOp.getEndtime()]));
        continue;
      }
      if (!timeEdge.isFor && !timeEdge.isWhile && timeEdge.weight != -1) {
        distanceSum[timeEdge.from] = distanceSum[timeEdge.to] + timeEdge.weight;
        outDegreeNum[timeEdge.from] -= 1;
        if (outDegreeNum[timeEdge.from] == 0) {
          vertexQueue.push(timeEdge.from);
        }
      } else if (timeEdge.isFor && outDegreeNum[timeEdge.from] == 1) {
        tor::ForOp forOp = startTimeForOpMap[timeEdge.from].first;
        int64_t loopTripCount = getTORConstantTripCount(forOp);
        if (loopTripCount == -1) {
          if (!startTimeForOpMap[timeEdge.from].second) {
            LLVM_DEBUG(forOp.dump());
            llvm::errs() << "Some for loop can't calculate tripcount\n";
            startTimeForOpMap[timeEdge.from].second = true;
          }
          continue;
        }
        int64_t loopInnerCycleSum = distanceSum[timeEdge.from];
        int64_t loopCycle;
        if (forOp->hasAttr("II")) {
          loopInnerCycleSum += 2; // add init + exit cycle
          loopCycle = loopInnerCycleSum +
                      getIIAttrInterger(forOp) * (loopTripCount - 1) - 1;
        } else {
          loopInnerCycleSum += 1; // add init cycle
          const int loopEffectCycle = 2;
          forOp->setAttr("cycle",
                         mlir::IntegerAttr::get(
                             mlir::IntegerType::get(forOp->getContext(), 64),
                             loopInnerCycleSum));
          loopCycle = loopInnerCycleSum * loopTripCount + loopEffectCycle;
        }
        distanceSum[timeEdge.from] = loopCycle + distanceSum[timeEdge.to];
        outDegreeNum[timeEdge.from] -= 1;
        if (outDegreeNum[timeEdge.from] == 0) {
          vertexQueue.push(timeEdge.from);
        }
      }else if(timeEdge.isWhile&&outDegreeNum[timeEdge.from] == 1){
        tor::WhileOp WhileOp = startTimeWhileOpMap[timeEdge.from].first;
        int64_t loopInnerCycleSum = distanceSum[timeEdge.from];
        int64_t loopCycle;
        loopInnerCycleSum += 1; // add init cycle
        const int loopEffectCycle = 2;
        WhileOp->setAttr("cycle",
                        mlir::IntegerAttr::get(
                            mlir::IntegerType::get(WhileOp->getContext(), 64),
                            loopInnerCycleSum));
        loopCycle = loopInnerCycleSum + loopEffectCycle;
        distanceSum[timeEdge.from] = loopCycle + distanceSum[timeEdge.to];
        outDegreeNum[timeEdge.from] -= 1;
        if (outDegreeNum[timeEdge.from] == 0) {
          vertexQueue.push(timeEdge.from);
        }
      }
    }
  }

  return distanceSum[startVertex] + 1;
}

void handleFunc(tor::FuncOp funcOp,
                DenseMap<llvm::StringRef, long long> &funcOpCycleMap) {
  tor::TimeGraphOp timeGraphOp;
  for (auto &block : funcOp) {
    for (auto &op : block) {
      if (auto timegraph = dyn_cast<tor::TimeGraphOp>(op)) {
        timeGraphOp = timegraph;
      }
    }
  }
  if (!timeGraphOp) {
    return;
  }
  int startVertex = timeGraphOp.getStarttime();
  int vertexNum = timeGraphOp.getEndtime();
  SmallVector<SmallVector<TimeEdge>> timeGraph(vertexNum + 1);
  DenseMap<int, tor::CallOp> startTimeCallOpMap;
  timeGraphClear(timeGraph);
  getCallOpMap(funcOp, startTimeCallOpMap);
  getTimeGraph(timeGraphOp, timeGraph, startTimeCallOpMap, funcOpCycleMap);
  DenseMap<int, std::pair<tor::ForOp, bool>> startTimeForOpMap;
  getForOrIfOpMap(funcOp, startTimeForOpMap);
  DenseMap<int, std::pair<tor::WhileOp, bool>> startTimeWhileOpMap;
  DenseMap<int, std::pair<tor::IfOp, bool>> startTimeIfOpMap;
  getForOrIfOpMap(funcOp, startTimeIfOpMap);
  getForOrIfOpMap(funcOp, startTimeWhileOpMap);
  auto cycle =
      cycleCount(timeGraph, startTimeForOpMap, startTimeIfOpMap, startTimeWhileOpMap, startVertex);
  if (cycle != 1) {
    funcOp->setAttr(
        "cycle", mlir::IntegerAttr::get(
                     mlir::IntegerType::get(funcOp->getContext(), 64), cycle));
    funcOpCycleMap[funcOp.getName()] = cycle;
  }
  timeGraphClear(timeGraph);
  startTimeForOpMap.clear();
}

struct node{
  std::string id;
  int64_t II = -1,cycle = -1,number = -1,pl = 0;
  bool pipe = false;
};

unsigned getcycleAttrInterger(Operation *forOp) {
  if(!forOp){
        return 0;
  }
  auto attr = forOp->getAttr("cycle");
  if (!attr)  
  {
        return 0;
  }
  auto inattr = dyn_cast<mlir::IntegerAttr>(attr);
  if(!inattr){
    return 0;
  }
  auto value = inattr.getValue().getSExtValue();
  if(!value){
    return 0;
  }
  return value;
}

unsigned getTripcountAttrInterger(Operation *forOp) {
  if(!forOp){
        return 0;
  }
  auto attr = forOp->getAttr("tripcount-max");
  if (!attr)  
  {
        return 0;
  }
  auto inattr = dyn_cast<mlir::IntegerAttr>(attr);
  if(!inattr){
    return 0;
  }
  auto value = inattr.getValue().getSExtValue();
  if(!value){
    return 0;
  }
  return value;
}
std::map<llvm::StringRef,std::vector<node> > q;
bool hasWhileOp;
void traverseOp(Operation *Op,int depth,llvm::StringRef funcName){
    if(auto FuncOp = llvm::dyn_cast<tor::FuncOp>(Op)){
      funcName = FuncOp.getName();
      node tmp;
      tmp.id = funcName.str();
      tmp.pl = depth;
      if (auto cycleAttr = FuncOp->getAttrOfType<IntegerAttr>("cycle")) {
        tmp.cycle =   cycleAttr.getInt();
      } 
      if (auto cycleAttr = FuncOp->getAttrOfType<IntegerAttr>("II")) {
        tmp.II =   cycleAttr.getInt();
      } 
      if (auto cycleAttr = FuncOp->getAttrOfType<IntegerAttr>("tripcount-max")) {
        tmp.number =   cycleAttr.getInt();
      } 
    q[funcName].push_back(tmp);
    }else if(auto forOp = llvm::dyn_cast<tor::ForOp>(Op)){
      node tmp;
      tmp.id = funcName.str() ;
      tmp.id += "-for-";
      tmp.id += std::to_string(q[funcName].size());
      tmp.pl = depth;
      auto AttrII = getIIAttrInterger(forOp);
      if(AttrII){
        tmp.II = AttrII;
        tmp.pipe = true;
        q[funcName][0].II = AttrII;
        q[funcName][0].pipe = true;
      }
      auto Attrcycle  = getcycleAttrInterger(forOp);
      if(Attrcycle){
        tmp.cycle = Attrcycle;
      }
      auto Attrnumber = getTripcountAttrInterger(forOp);
      if(Attrnumber){
        tmp.number = Attrnumber;
      }
      q[funcName].push_back(tmp);
    }else if(auto whileop = llvm::dyn_cast<tor::WhileOp>(Op)){
      hasWhileOp = true;
      node tmp;
      tmp.id = funcName.str() ;
      tmp.id += "-while-";
      tmp.id += std::to_string(q[funcName].size());
      tmp.id += "*";
      tmp.pl = depth;
      auto Attrcycle  = getcycleAttrInterger(whileop);
      if(Attrcycle){
        tmp.cycle = Attrcycle;
      }
      auto Attrnumber = getTripcountAttrInterger(whileop);
      tmp.number = -2;
      q[funcName].push_back(tmp);
    }else if(auto ifop = llvm::dyn_cast<tor::IfOp>(Op)){
      node tmp;
      tmp.id = funcName.str() ;
      tmp.id += "-if-";
      tmp.id += std::to_string(q[funcName].size());
      tmp.pl = depth;
      auto Attrcycle  = getcycleAttrInterger(ifop);
      if(Attrcycle){
        tmp.cycle = Attrcycle;
      }
      q[funcName].push_back(tmp);
    }
    for(auto &Region : Op->getRegions()){
      for(auto &Block : Region.getBlocks()){
        for(auto &nestOp : Block){
          if(llvm::isa<tor::CallOp>(nestOp)){
            auto op = llvm::dyn_cast<tor::CallOp>(nestOp);
            auto tmpfunc = op.getCallee();
            int  bk = q[funcName].size()-1;
            while(bk > 0 && q[funcName][bk].id.find(funcName) == q[funcName][bk].id.npos){
              bk--;
            }
            int pl = q[funcName][bk].pl + 1;
            for(auto nod : q[tmpfunc]){
              nod.pl = std::max(std::max(depth,3)  + nod.pl - q[tmpfunc][0].pl,3L);
              q[funcName].push_back(nod);
            }            
          }
          traverseOp(&nestOp,depth+1,funcName);
        }
          
      }
    }
}

template<typename T>
std::string formatwidth(int width,char fill,T Value){
  std::ostringstream ss;
  ss << std::setw(width) << std::setfill(fill) << Value;
  return ss.str();
}


void CountAttribute(mlir::ModuleOp &moduleOp){
    llvm::StringRef fn("module");
    traverseOp(moduleOp,0,fn);
    llvm::StringRef fm("main");
    if(hasWhileOp)llvm::dbgs()<<"WARNING: whileOp can't calculate TripCount\n";
    llvm::dbgs() << "------------------------------------------------------------------------------------------------------------------------\n";
    llvm::dbgs() <<"|"<< formatwidth(60 ,' '," ")<<"    |"<<formatwidth(10,' ',"cycle") << "     |"<<formatwidth(10,' ',"TripCount") << " |"<<formatwidth(8,' ',"II") << "      |"<<formatwidth(10,' ',"PIPELINE") << "|\n";

    for(auto nod : q[fm]){
      llvm::dbgs() << "|";
      for (int i = 0; i < nod.pl; i++){
        llvm::dbgs() << "    ";
      }
      llvm::dbgs() << nod.id;
      llvm::dbgs() << formatwidth(60 - 4 * nod.pl - nod.id.size(),' '," ")<<"    |";
      if(nod.cycle > 0&&nod.number != -2)llvm::dbgs() << formatwidth(15,' ',nod.cycle) << "|";
      else if(nod.cycle > 0)llvm::dbgs() << "(one trip)" << formatwidth(5,' ',nod.cycle) << "|";
      else llvm::dbgs() << "               |";
      if(nod.number > 0)llvm::dbgs() << formatwidth(10,' ',nod.number)  << " |";
      else if(nod.number == -2) llvm::dbgs() << formatwidth(10,' ',"inf")  << " |";
      else llvm::dbgs() << "           |";
      if(nod.II > 0)llvm::dbgs() << formatwidth(8,' ',nod.II)  << "      |";
      else llvm::dbgs() << "              |";
      if(nod.pipe)llvm::dbgs() << "       Yes|\n";
      else llvm::dbgs() << "        No|\n";
     }
    llvm::dbgs() << "------------------------------------------------------------------------------------------------------------------------\n";
}

struct CountCyclesPass : CountCyclesBase<CountCyclesPass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();
    DenseMap<llvm::StringRef, long long> funcOpCycleMap;
    tor::DesignOp designOp;
    
    moduleOp.walk([&](tor::FuncOp funcOp) {
      handleFunc(funcOp, funcOpCycleMap);
      designOp = dyn_cast<tor::DesignOp>(funcOp->getParentOp());
    });
    if (funcOpCycleMap.count("main")) {
      llvm::dbgs() << designOp.getSymbol() << " cycle is "
                   << funcOpCycleMap["main"] << "\n";
      designOp->setAttr("cycle",
                        mlir::IntegerAttr::get(
                            mlir::IntegerType::get(designOp->getContext(), 64),
                            funcOpCycleMap["main"]));
    }
    if ((this->output_dir).getValue().length() != 0) {
      llvm::dbgs() << "the approximate cycle file is located in "
                   << this->output_dir << "/timegraph.txt\n";
      GenTimgraph(getOperation(), (this->output_dir).getValue())
          .writeTimegraph();
    }
    CountAttribute(moduleOp);
  }
};
} // namespace

namespace mlir {
std::unique_ptr<OperationPass<mlir::ModuleOp>> createCountCyclesPass() {
  return std::make_unique<CountCyclesPass>();
}

} // namespace mlir
