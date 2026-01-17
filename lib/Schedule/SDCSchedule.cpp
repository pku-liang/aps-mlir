#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "TOR/TORDialect.h"
#include <map>
#include <queue>
#include <set>

// FIXME: Macro LE in lp_lib.h conflicts with variables in llvm. Changing the
// include order is an ad-hoc solution but doesn't really solve the problem.
#include "Schedule/CDFG.h"
#include "Schedule/SDCSchedule.h"
#include "Schedule/SDCSolver.h"
#include "llvm/Support/Debug.h"
#include "lpsolve/lp_lib.h"
#include <unordered_map>
#include <unordered_set>

// Fix bug in Windows
#undef min
#undef max

#define DEBUG_TYPE "sdc-schedule"

namespace scheduling {

/**
 * Not scheduled in the pipeline scheduling
 */
bool needSchedule(const BasicBlock *B) {
  // return BeginBB.find(B) != BeginBB.end();
  return B->getParentLoop() == nullptr ||
         B->getParentLoop()->PipelineFlag == false;
}


int SDCSchedule::getMAxiMII(Loop *L) {
  // todo axi pipeline + burst but this is not that important
  std::unordered_map<std::string, int> readBusTable;
  std::unordered_map<std::string, int> writeBusTable;
  for (auto BB : L->getBody()) {
    for (auto op : BB->getOperations()) {
      auto maxiOp = op->getMAxiOp();
      if (maxiOp != nullptr) {
        if (maxiOp->isBurst()) {
          if (maxiOp->isRead()) {
            readBusTable[maxiOp->getBus()] += 1;
          } else {
            writeBusTable[maxiOp->getBus()] += 1;
          }
        } else if (maxiOp->isRead()) {
          readBusTable[maxiOp->getBus()] += 2; // read latency
        } else {
          writeBusTable[maxiOp->getBus()] += 3; // write latency
        }
      }
    }
  }

  int resMII = 1;
  for (auto kv : readBusTable) {
    resMII = std::max(resMII, kv.second);
  }

  for (auto kv : writeBusTable) {
    resMII = std::max(resMII, kv.second);
  }
  return resMII;
}

int SDCSchedule::get_memportMII(Loop *L) {
  auto id = RDB.getResourceID("memport");
  assert(RDB.hasHardLimit(id) && "memport must has hard limit");

  int numport = RDB.getAmount(id);
  llvm::DenseMap<mlir::Value, int> MemTable;

  for (auto BB : L->getBody()) {
    for (auto op : BB->getOperations()) {
      if (op->getMemOp() != nullptr) {
        auto memop = op->getMemOp();
        auto memref = memop->getMemRef();
        MemTable[memref] += 1;
      }
    }
  }

  int resMII = 1;
  for (auto kv : MemTable) {
    auto defOp = kv.first.getDefiningOp();

    // Handle both tor::AllocOp and memref::GetGlobalOp
    if (auto allocOp = llvm::dyn_cast<tor::AllocOp>(defOp)) {
      if (allocOp->hasAttr("bind_storage_type")) {
        auto type = dyn_cast<StringAttr>(allocOp->getAttr("bind_storage_type")).getValue();
        if (type == "RAM_1P") {
          resMII = std::max(resMII, kv.second);
        } else if (type == "RAM_T2P") {
          resMII = std::max(resMII, (kv.second + 1) / 2);
        } else {
          assert(false && "memref type not supported");
        }
      } else {
        resMII = std::max(resMII, (kv.second + numport - 1) / numport);
      }
    } else if (llvm::isa<memref::GetGlobalOp>(defOp)) {
      // For memref.get_global, treat as regfile with no storage constraints
      // Use default port-based calculation
      resMII = std::max(resMII, (kv.second + numport - 1) / numport);
    } else {
      // Unknown memref source
      llvm::errs() << "Warning: Unknown memref defining operation in get_memportMII\n";
      resMII = std::max(resMII, (kv.second + numport - 1) / numport);
    }
  }

  return resMII;
}

int SDCSchedule::resourceMII(Loop *L) {
  int resII = 1;
  int ResourceKind = RDB.getNumResource();

  std::vector<int> resPressure(ResourceKind, 0);

  for (auto BB : L->getBody())
    for (auto op : BB->getOperations()) {
      int RId = op->getResource();
      int pressure = RDB.getII(RId);

      resPressure[RId] += pressure;
    }

  for (int i = 1; i < ResourceKind; ++i)
    if (RDB.hasHardLimit(i) && !RDB.getName(i).rfind("memport", 0) == 0
       && !RDB.getName(i).rfind("m_axi", 0) == 0) {
      LLVM_DEBUG(llvm::dbgs() << RDB.getName(i) << " " << resPressure[i] << " "
                              << RDB.getAmount(i) << "\n");
      resII = std::max(resII, resPressure[i] / (int)RDB.getAmount(i));
    }

  resII = std::max(resII, get_memportMII(L));
  resII = std::max(resII, getMAxiMII(L));
  return resII;
}

int SDCSchedule::recurrenceMII(Loop *L) {
  /// Why need this in the paper?
  int recII = 1;
  return recII;
}

bool sameLoop(Dependence *D) {
  return D->SourceOp->getParentLoop() == D->DestinationOp->getParentLoop();
}

void SDCSchedule::traverse(SDCOpWrapper *op, SDCSolver *SDC, int latency,
                           float cp, int dist, int II, SDCOpWrapper *start,
                           std::unordered_map<SDCOpWrapper *, bool> &vis,
                           std::unordered_map<SDCOpWrapper *, bool> &exceed) {
  vis[op] = true;
  for (auto Succ : op->getSucc())
    if (sameLoop(Succ)) {
      auto succOp = llvm::dyn_cast<SDCOpWrapper>(Succ->DestinationOp);
      float nxt_cp =
          cp + RDB.getDelay(succOp->getResource(), succOp->getWidth());
      int nxt_dist = dist + Succ->Distance;

      if (nxt_cp > ClockPeriod) {
        SDC->addInitialConstraint(Constraint::CreateGE(
            succOp->VarId, start->VarId, latency + 1 - nxt_dist * II)
                                  /* must schedule in different cycle */
        );
        exceed[succOp] = true;
        continue;
      }

      if (RDB.isCombLogic(succOp->getResource(), succOp->getWidth()) &&
          !vis[succOp] && exceed.find(succOp) == exceed.end())
        traverse(succOp, SDC, latency, nxt_cp, nxt_dist, II, start, vis,
                 exceed);
    }
  vis[op] = false;
}

void SDCSchedule::addChainingConstr(SDCOpWrapper *op, SDCSolver *SDC, int II) {
  std::unordered_map<SDCOpWrapper *, bool> vis;
  std::unordered_map<SDCOpWrapper *, bool> exceed;

  if (RDB.getName(op->getResource()) != "nop")
    traverse(op, SDC, RDB.getLatency(op->getResource(), op->getWidth()), 0.0, 0,
             II, op, vis, exceed);
}

static bool isCombOp(mlir::Operation* op) {
  if (llvm::isa<tor::CmpIOp, tor::AddIOp, tor::SubIOp, tor::MulIConstOp,
                arith::OrIOp, arith::XOrIOp, arith::ShLIOp, arith::ShRSIOp,
                arith::ShRUIOp, arith::NegFOp, arith::TruncIOp, arith::ExtSIOp,
                arith::ExtUIOp, arith::SelectOp>(op)) { // math::AbsFOp, math::AbsIOp, 
    return true;
  }
  return false;
}

static bool isInductionVar(mlir::Value operand) {
  auto parentOp = operand.getDefiningOp();
  if (parentOp != nullptr && llvm::isa<tor::ForOp>(parentOp)) {
    auto forOp = llvm::dyn_cast<tor::ForOp>(parentOp);
    return operand == forOp.getInductionVar();
  }
  return false;
}

static bool isConstVar(mlir::Value operand) {
  if (isa<BlockArgument>(operand)) return true;
  auto op = operand.getDefiningOp();
  if (op == nullptr) return false;
  return llvm::isa<arith::ConstantIndexOp>(op) || llvm::isa<arith::ConstantIntOp>(op);
}

static bool isOpCanMoveAround(mlir::Operation *op) {
  if (!isCombOp(op)) return false;
  return llvm::all_of(op->getOperands(),
        [](auto operand) {
          return !(isInductionVar(operand) || isConstVar(operand));
        });
}

void SDCSchedule::formulateDependency(Loop *L, int II, SDCSolver *SDC) {
  // Control Dependency
  for (auto BB : L->getBody()) {
    if (auto branchOp = BB->getBranchOp()) {
      auto branchSDCOp = llvm::dyn_cast<SDCOpWrapper>(branchOp);
      int RId = branchSDCOp->getResource();
      int Lat = RDB.getLatency(RId, branchSDCOp->getWidth());
      for (auto op : BB->getOperations()) {
          auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(op);
          SDC->addInitialConstraint(Constraint::CreateGE(
            sdcOp->VarId, branchSDCOp->VarId, Lat));
      }
    }
  }

  // formulate data dependency
  for (auto BB : L->getBody())
    for (auto op : BB->getOperations()) {
      auto destOp = llvm::dyn_cast<SDCOpWrapper>(op);
      assert(destOp);

      for (auto pred : op->getPred())
        if (sameLoop(pred)) {
          auto srcOp = llvm::dyn_cast<SDCOpWrapper>(pred->SourceOp);
          int RId = srcOp->getResource();
          int Lat = RDB.getLatency(RId, srcOp->getWidth());

          // This special case is because of codegen backend
          if (srcOp->getType() == OpAbstract::OpType::PHI_OP)
            Lat = 1;
          if (Lat == 0) {
            // if (isOpCanMoveAround(srcOp->getOp()) && pred->Distance == 0) {
            //   SDC->addInitialConstraint(Constraint::CreateGE(
            //       destOp->VarId, srcOp->VarId, Lat));
            //   SDC->addInitialConstraint(Constraint::CreateLE(
            //       destOp->VarId, srcOp->VarId, Lat));
            //   continue;
            // }

            SDC->addInitialConstraint(Constraint::CreateGE(
                destOp->VarId, srcOp->VarId, 1 - II * pred->Distance));
          } else {
            LLVM_DEBUG(if (pred->Distance != 0) {
              llvm::outs() << srcOp->getOpDumpId()
                           << "<=" << destOp->getOpDumpId() << "- "
                           << Lat - II * pred->Distance << " " << pred->Distance
                           << "\n";
            });
            SDC->addInitialConstraint(Constraint::CreateGE(
                destOp->VarId, srcOp->VarId, Lat - II * pred->Distance));
          }
          // srcOp->printName(llvm::outs());
          // destOp->printName(llvm::outs());
        }
    }

  // formulate chaining requirements
  for (auto BB : L->getBody())
    for (auto op : BB->getOperations())
      addChainingConstr(llvm::dyn_cast<SDCOpWrapper>(op), SDC, II);

  // make burst axi op under the same loop consecutive
  // key: request Op, value: vector of ops related to request op
  std::unordered_map<mlir::Operation*, std::vector<SDCOpWrapper *>> burstOps;
  for (auto BB : L->getBody()) {
    for (auto op : BB->getOperations()) {
      auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(op);
      auto maxiOp = op->getMAxiOp();
      if (maxiOp != nullptr) {
        auto request = maxiOp->getRequest();
        if (request != nullptr) {
          mlir::Operation* requestOp = request.getDefiningOp();
          burstOps[requestOp].push_back(sdcOp);
        }
      }
    }
  }
  auto cmpBurstOps = [&](SDCOpWrapper * sdcOp1, SDCOpWrapper * sdcOp2) {
    return sdcOp1->getOp()->isBeforeInBlock(sdcOp2->getOp());
  };

  for (auto iter: burstOps) {
    std::sort(iter.second.begin(), iter.second.end(), cmpBurstOps);
    for (size_t i = 1; i < iter.second.size(); ++i) {
      SDC->addInitialConstraint(Constraint::CreateLE(
        iter.second[i]->VarId, iter.second[i - 1]->VarId, 1));
    }
  }
}

void SDCSchedule::allocVariable(Loop *L, SDCSolver *SDC) {
  for (auto BB : L->getBody())
    for (auto op : BB->getOperations()) {
      auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(op);
      sdcOp->VarId = SDC->addVariable();
      llvm::errs() << "Assigning Var" << sdcOp->VarId 
                   << " to op " << sdcOp->getOp()->getName().getStringRef()
                   << " resource=" << sdcOp->getResource() << "\n";
    }
}

bool SDCSchedule::optimizeASAP(Loop *L, int II, SDCSolver *SDC) {
  int varCnt = SDC->getNumVariable();

  lprec *lp = make_lp(0, SDC->getNumVariable());
  set_verbose(lp, NEUTRAL);
  SDC->convertLP(lp);

  for (auto BB : L->getBody())
    for (auto op : BB->getOperations()) {
      auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(op);
      set_mat(lp, 0, sdcOp->VarId, 1.0);
    }
  set_minim(lp);
  int ret = solve(lp);
  if (ret == INFEASIBLE)
    return false;

  REAL *results = new REAL[varCnt];
  get_variables(lp, results);

  for (int i = 0, n = SDC->getNumVariable(); i < n; ++i) {
    int sol = std::round(results[i]);
    assert(sol >= 0);
    SDC->assignSolution(i, sol);
  }

  for (auto BB : L->getBody())
    for (auto op : BB->getOperations()) {
      auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(op);
      sdcOp->OptTime = SDC->Solution[sdcOp->VarId];
    }

  SDC->ValidFlag = true;

  assert(SDC->verify());

  delete_lp(lp);

  return true;
}

bool SDCSchedule::minimizeLifetime(Loop *L, int II, SDCSolver *SDC) {
  int varCnt = SDC->getNumVariable();

  lprec *lp = make_lp(0, SDC->getNumVariable());
  set_verbose(lp, NEUTRAL);
  SDC->convertLP(lp);

  // add lifetime variables
  for (auto BB : L->getBody()) {
    for (auto op : BB->getOperations()) {
      auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(op);
      // add lifetime variable for op
      int rowno[1] = {0};
      double col[1] = {
          (double)sdcOp->getWidth()}; // coefficient of objective function
      add_columnex(lp, 0, col, rowno);

      if(llvm::dyn_cast<tor::LoadOp>(sdcOp->getOp())){
        set_obj(lp,sdcOp->VarId,1);
      }

      varCnt++;
      int newId = get_Ncolumns(lp);

      set_add_rowmode(lp, TRUE);
      for (auto succ : op->getSucc())
        if (sameLoop(succ) && succ->type == Dependence::D_RAW) {
          auto succOp = llvm::dyn_cast<SDCOpWrapper>(succ->DestinationOp);

          int colno[3] = {succOp->VarId + 1, sdcOp->VarId + 1, newId};
          double row[3] = {1, -1, -1};
          add_constraintex(
              lp, 3, row, colno, LE,
              RDB.getLatency(sdcOp->getResource(), sdcOp->getWidth()));
        }
      set_add_rowmode(lp, FALSE);
    }
  }

  set_minim(lp);
  int ret = solve(lp);
  if (ret == INFEASIBLE)
    return false;

  REAL *results = new REAL[varCnt];
  get_variables(lp, results);

  for (int i = 0, n = SDC->getNumVariable(); i < n; ++i) {
    int sol = std::round(results[i]);
    assert(sol >= 0);
    SDC->assignSolution(i, sol);
  }

  for (auto BB : L->getBody())
    for (auto op : BB->getOperations()) {
      auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(op);
      sdcOp->OptTime = SDC->Solution[sdcOp->VarId];
    }

  SDC->ValidFlag = true;

  assert(SDC->verify());

  delete_lp(lp);

  return true;
}

bool SDCSchedule::resolveResConstraint(Loop *L, int II, SDCSolver *SDC) {
  int ResKind = RDB.getNumResource();

  // keep the resource usage
  std::vector<std::vector<int>> ResTable(ResKind, std::vector<int>(II, 0));
  std::vector<std::map<mlir::detail::ValueImpl *, int>> MemTable(II);
  std::vector<std::unordered_map<mlir::detail::ValueImpl *, int>> mAxiMemTable(II); // II, memref, resource count
  std::vector<std::unordered_map<std::string, int>> mAxiReadBusTable(II); // II, busId, resource count
  std::vector<std::unordered_map<std::string, int>> mAxiWriteBusTable(II);

  std::vector<int> ResLimit(ResKind, 0);
  std::vector<bool> HardFlag(ResKind, 0);

  for (int i = 0; i < ResKind; ++i) {
    if (RDB.hasHardLimit(i)) {
      ResLimit[i] = RDB.getAmount(i);
      HardFlag[i] = 1;
    } else {
      int NumOp = 0;
      for (auto BB : L->getBody())
        for (auto op : BB->getOperations())
          if (op->getResource() == i)
            NumOp++;
      ResLimit[i] = (NumOp + II - 1) / II;
    }
  }

  // keep the scheduled memop
  std::vector<std::vector<std::pair<SDCOpWrapper *, int>>> ScheduledMemOp(II);

  typedef std::pair<int, SDCOpWrapper *> datumT;
   // compare optTime first, if equal, then compare sdcOp varId
  auto setcmp = [](const datumT &a, const datumT &b) {
    if (a.first != b.first)
      return a.first < b.first;
    else
      return a.second->VarId > b.second->VarId;
  };

  std::set<datumT, decltype(setcmp)> S(setcmp);

  // Ensure each op is tried at most II times.
  std::map<SDCOpWrapper *, int> failedCnt;

  // key: request Op, value: vector of ops related to request op
  std::unordered_map<mlir::Operation*, std::vector<SDCOpWrapper *>> burstOps;
  // 假定同一request同一region下的burstOp是连续的, 有多少个burstOp, II就是多少
  for (auto BB : L->getBody()) {
    for (auto op : BB->getOperations()) {
      auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(op);
      auto maxiOp = op->getMAxiOp();
      if (maxiOp != nullptr) {
        auto request = maxiOp->getRequest();
        if (request != nullptr) {
          mlir::Operation* requestOp = request.getDefiningOp();
          burstOps[requestOp].push_back(sdcOp);
        }
      }
    }
  }

  auto cmpBurstOps = [&](SDCOpWrapper * a, SDCOpWrapper * b) {
                         return a->OptTime < b->OptTime;
                     };
  for (auto iter: burstOps) {
    std::sort(iter.second.begin(), iter.second.end(), cmpBurstOps);
    SDCOpWrapper* sdcOp = iter.second.front();
    S.insert(std::make_pair(sdcOp->OptTime, sdcOp));
  }

  for (auto BB : L->getBody())
    for (auto op : BB->getOperations()) {
      auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(op);
      auto maxiOp = sdcOp->getMAxiOp();
      if (RDB.hasHardLimit(sdcOp->getResource()) && (maxiOp == nullptr || maxiOp->getRequest() == nullptr))
        S.insert(std::make_pair(sdcOp->OptTime, sdcOp));
    }

  std::map<SDCOpWrapper *, int> earliestTime;
  // The perturbation caused to the old solution if the op is placed
  // in other steps as described in Zhiru Zhang's ICCAD 2013 paper
  std::map<SDCOpWrapper *, int> perturbation;
  for (auto &[step, op] : S) {
    int pert_before =
        SDC->tryAddConstr(Constraint::CreateLE(op->VarId, step - 1));
    int pert_after =
        SDC->tryAddConstr(Constraint::CreateGE(op->VarId, step + 1));
    pert_before = pert_before != -1 ? pert_before : SDC->getNumVariable();
    pert_after = pert_after != -1 ? pert_after : SDC->getNumVariable();
    // todo burstOp 的pertubation需要设置大一些吗？
    perturbation[op] = std::max(pert_before, pert_after);
  }

  while (!S.empty()) {
    int step = S.begin()->first;

    std::vector<SDCOpWrapper *> ops;
    while (!S.empty() && S.begin()->first == step) {
      ops.push_back(S.begin()->second);
      S.erase(S.begin());
    }

    // calc perturbation
    for (auto op : ops) {
      if (earliestTime.find(op) == earliestTime.end())
        earliestTime[op] = op->ASAPTime;
    }

    auto cmp = [&](SDCOpWrapper *a, SDCOpWrapper *b) {
      if (perturbation[a] != perturbation[b])
        return perturbation[a] > perturbation[b];
      return earliestTime[a] < earliestTime[b];
    };

    std::stable_sort(ops.begin(), ops.end(), cmp);

    for (auto op : ops) {
      bool scheduled = false;
      int RId = op->getResource();
      int resourceII = RDB.getII(RId);
      if (RDB.getName(RId).rfind("m_axi", 0) == 0) {
        auto request = op->getMAxiOp()->getRequest();
        if (request) {
          mlir::Operation* requestOp = request.getDefiningOp();
          resourceII = burstOps[requestOp].size();
        }
      }
      // only schedule resource constrained operations

      for (int s = step, n = earliestTime[op]; s >= n; --s) {
        // check resource availability
        bool avail = true;
        if (RDB.getName(RId).rfind("memport", 0) == 0) {
          // assume memory port has one cycle latency and can't be pipelined
          int slot = s % II;
          // for (auto &sdcOp : ScheduledMemOp[slot]) {
          // int dist = (s - sdcOp.second) / II;
          // check if op in current iteration can have resource conflict with
          // sdc op after dist iteraions.

          // if (hasMemPortConflict(op, sdcOp.first, dist)) {
          for (int i = 0; i < resourceII; ++i) {
            auto v = op->getMemOp()->getMemRef().getImpl();
            if (MemTable[(slot + i) % II][v] >= ResLimit[RId]) {
              avail = false;
              break;
            }
          }
          // }
        } else if (RDB.getName(RId).rfind("m_axi", 0) == 0) {
          // axi bus
          int slot = s % II;
          auto bus = op->getMAxiOp()->getBus();
          auto v = op->getMAxiOp()->getMemRef().getImpl();
          if (op->getMAxiOp()->isRead()) {
            for (int i = 0; i < resourceII; ++i) {
              int id = (slot + i) % II;
              if (mAxiReadBusTable[id][bus] >= ResLimit[RId] || mAxiMemTable[id][v] >= 1) {
                avail = false;
                break;
              }
            }
          } else {
            for (int i = 0; i < resourceII; ++i) {
              int id = (slot + i) % II;
              if (mAxiWriteBusTable[id][bus] >= ResLimit[RId] || mAxiMemTable[id][v] >= 1) {
                avail = false;
                break;
              }
            }
          }
          // end of axi
        } else {
          for (int i = 0; i < resourceII; ++i)
            if (ResTable[RId][(s + i) % II] >= ResLimit[RId]) {
              avail = false;
              break;
            }
        }

        if (avail == false)
          continue;

        // TODO: Resource sharing among mutual exclusive branches.
        if (SDC->addConstraint(Constraint::CreateEQ(op->VarId, s))) {
          scheduled = true;

          if (RDB.getName(RId).rfind("memport", 0) == 0) {
            auto v = op->getMemOp()->getMemRef().getImpl();
            for (int i = 0; i < resourceII; ++i) {
              MemTable[(s + i) % II][v] += 1;
              // ScheduledMemOp[s % II].push_back(std::make_pair(op, s));
            }
          } else if (RDB.getName(RId).rfind("m_axi", 0) == 0) {
            // todo burst axi II
            auto bus = op->getMAxiOp()->getBus();
            auto v = op->getMAxiOp()->getMemRef().getImpl();
            if (op->getMAxiOp()->isRead()) {
              for (int i = 0; i < resourceII; ++i) {
                int id = (s + i) % II;
                mAxiReadBusTable[id][bus] += 1;
                mAxiMemTable[id][v] += 1;
              }
            } else {
              for (int i = 0; i < resourceII; ++i) {
                int id = (s + i) % II;
                mAxiWriteBusTable[id][bus] += 1;
                mAxiMemTable[id][v] += 1;
              }
            }
          } else {
            for (int i = 0; i < resourceII; ++i)
              ResTable[RId][(s + i) % II]++;
          }
          break;
        }
      }

      if (scheduled == true)
        continue;

      if (SDC->addConstraint(Constraint::CreateGE(op->VarId, step + 1)) ==
          false)
        return false;

      if (failedCnt[op] == II)
        return false;

      failedCnt[op]++;
      earliestTime[op] = SDC->Solution[op->VarId];
      S.insert(std::make_pair(earliestTime[op], op));
    }
  }
  return true;
}

bool SDCSchedule::getASAPTime(Loop *L, int II, SDCSolver *SDC) {
  if (SDC->initSolution() == false)
    return false;

  for (auto BB : L->getBody())
    for (auto op : BB->getOperations()) {
      auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(op);
      sdcOp->ASAPTime = SDC->Solution[sdcOp->VarId];
    }

  return true;
}

bool SDCSchedule::scheduleWithII(Loop *L, int II, bool FinalFlag) {
  LLVM_DEBUG(llvm::dbgs() << "Schedule with II: " << II << "\n");
  SDCSolver *SDC = new SDCSolver();

  allocVariable(L, SDC);

  formulateDependency(L, II, SDC);

  if (getASAPTime(L, II, SDC) == false)
    return false;

  if (minimizeLifetime(L, II, SDC) == false)
    return false;

  /*
  if (optimizeASAP(L, II, SDC) == false)
    return false;
  */

  LLVM_DEBUG(llvm::dbgs() << "Schedule with II = " << II << " : Dependence Succeed\n");

  if (resolveResConstraint(L, II, SDC) == false) {
    LLVM_DEBUG(llvm::dbgs() << "Schedule with II = " << II << " : Resource Failed\n");
    return false;
  }

  LLVM_DEBUG(llvm::dbgs() << "Succceed\n");
  if (FinalFlag) {
    minimizeLifetime(L, II, SDC);

    assert(SDC->verify() == 1);

    for (auto BB : L->getBody())
      for (auto op : BB->getOperations()) {
        auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(op);
        sdcOp->SDCTime = SDC->Solution[sdcOp->VarId];
      }
  }

  return true;
}

bool SDCSchedule::isScheduleResultWithIIValid(Loop *L, int II) {
    int minTime = INT_MAX;
    int maxTime = 0;
    for (auto BB : L->getBody()) {
        for (auto op : BB->getOperations()) {
            if (llvm::isa<tor::ForOp>(op->getOp()))
                continue;
            auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(op);
            minTime = std::min(sdcOp->SDCTime, minTime);
            int Lat = RDB.getLatency(sdcOp->getResource(), sdcOp->getWidth());
            maxTime = std::max(Lat + sdcOp->SDCTime, maxTime);
            LLVM_DEBUG(llvm::dbgs() << *(op->getOp()) << " schedule sdcTime [" << sdcOp->SDCTime
                << ", " << Lat + sdcOp->SDCTime << ") \n");
        }
    }
    if (maxTime - minTime <= II) {
        llvm::errs() << "warning: pipeline II (" << II
            << ") greater than or equal to for loop cycle "
            << maxTime - minTime
            << ". There is no need to pipeline loop.\n";
        L->PipelineFlag = false;
        L->AchievedII = -1;
        LLVM_DEBUG(llvm::dbgs() << "warning: pipeline loop (" << L << ") PipelineFlag set to false.\n");
        return false;
    }
    return true;
}

bool SDCSchedule::pipelineLoop(Loop *L) {
  if (L->PipelineFlag == false)
    return false;

  int recMII = recurrenceMII(L);
  int resMII = resourceMII(L);
  LLVM_DEBUG(llvm::dbgs() << "recMII: " << recMII << " resMII: " << resMII << "\n");

  int MII = std::max(recMII, resMII);

  if (L->TargetII >= MII) {
    /// first try target II
    if (scheduleWithII(L, L->TargetII, true)) {
      LLVM_DEBUG(llvm::dbgs() << "Successfully scheduled with TargetII: " << L->TargetII
                   << "\n");
      L->AchievedII = L->TargetII;
      return isScheduleResultWithIIValid(L, L->TargetII);
    }
  } else {
    if (scheduleWithII(L, MII, true)) {
      LLVM_DEBUG(llvm::dbgs() << "Successfully scheduled with TargetII: " << MII << "\n");
      L->AchievedII = MII;
      return true;
    }
  }

  /// binary search
  int l = MII + 1, r = 128, achieved = -1;
  while (l <= r) {
    int mid = (l + r) >> 1;

    if (scheduleWithII(L, mid, false)) {
      achieved = mid;
      r = mid - 1;
    } else {
      l = mid + 1;
    }
  }

  if (achieved != -1) {
    LLVM_DEBUG(llvm::dbgs() << "Achieved II: " << achieved << "\n");
    L->AchievedII = achieved;
    scheduleWithII(L, achieved, true);
    return isScheduleResultWithIIValid(L, achieved);
  } else {
    llvm::errs() << "Failed to find a reasonable II\n";
    LLVM_DEBUG(llvm::dbgs() << "Failed to find a reasonable II\n");
    return false;
  }
}

void SDCSchedule::addTimingConstr(SDCOpWrapper *op, SDCSolver *SDC) {
  std::queue<SDCOpWrapper *> Q;
  std::unordered_map<SDCOpWrapper *, bool> inQueue;
  std::unordered_map<SDCOpWrapper *, float> cp;

  int latency = RDB.getLatency(op->getResource(), op->getWidth());

  Q.push(op);
  cp[op] = RDB.getDelay(op->getResource(), op->getWidth());
  inQueue[op] = true;

  while (!Q.empty()) {
    auto now = Q.front();
    Q.pop();
    inQueue[now] = false;

    int dist = cp[now];

    for (auto Succ : now->getSucc()) {
      if (Succ->Distance != 0)
        continue;

      auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(Succ->DestinationOp);

      if (!RDB.isCombLogic(sdcOp->getResource(), sdcOp->getWidth())) {
        // Seperate sequential logic and combinational logic
        // SDC->addInitialConstraint(
        //     Constraint::CreateGE(sdcOp->VarId, op->VarId, 1 + latency));
        continue;
      }

      if (!needSchedule(sdcOp->getParentBB()))
        continue;

      if (cp.find(sdcOp) != cp.end() && cp[sdcOp] > ClockPeriod)
        continue;

      float nxt_dist =
          dist + RDB.getDelay(sdcOp->getResource(), sdcOp->getWidth());

      if (cp.find(sdcOp) == cp.end() || cp[sdcOp] < nxt_dist) {
        cp[sdcOp] = nxt_dist;
        if (inQueue[sdcOp] == false) {
          Q.push(sdcOp);
          inQueue[sdcOp] = true;
        }
      }

      if (cp[sdcOp] > ClockPeriod) {
        SDC->addInitialConstraint(
            Constraint::CreateGE(sdcOp->VarId, op->VarId, 1 + latency));
        continue;
      }
    }
  }
}

std::vector<SDCOpWrapper *>
SDCSchedule::getFeasibleOrder(BasicBlock *BB,
                              function_ref<bool(SDCOpWrapper *)> Pred) {
  std::unordered_map<OpAbstract *, int> InDeg;
  std::unordered_map<OpAbstract *, int> ASAP;
  std::unordered_map<OpAbstract *, float> Ord;
  std::queue<OpAbstract *> Q;

  for (auto op : BB->getOperations()) {
    for (auto Succ : op->getSucc())
      if (Succ->Distance == 0 && Succ->DestinationOp->getParentBB() == BB)
        InDeg[Succ->DestinationOp]++;
  }

  for (auto op : BB->getOperations())
    if (InDeg[op] == 0) {
      Q.push(op);
      ASAP[op] = 0;
      Ord[op] = RDB.getDelay(op->getResource(), op->getWidth());
    }

  while (!Q.empty()) {
    auto now = Q.front();
    Q.pop();

    for (auto Succ : now->getSucc())
      if (Succ->Distance == 0 && Succ->DestinationOp->getParentBB() == BB) {
        auto succOp = Succ->DestinationOp;
        ASAP[succOp] = std::max(
            ASAP[succOp], ASAP[now] + (int)RDB.getLatency(now->getResource(),
                                                          now->getWidth()));
        Ord[succOp] =
            std::max(Ord[succOp], Ord[now] + RDB.getDelay(now->getResource(),
                                                          now->getWidth()));
        InDeg[succOp]--;
        if (InDeg[succOp] == 0)
          Q.push(succOp);
      }
  }

  std::vector<SDCOpWrapper *> ops;
  for (auto op : BB->getOperations()) {
    auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(op);
    if (Pred(sdcOp))
      ops.push_back(sdcOp);
  }

  auto cmp = [&](SDCOpWrapper *a, SDCOpWrapper *b) {
    if (ASAP[a] != ASAP[b])
      return ASAP[a] < ASAP[b];
    return Ord[a] < Ord[b];
  };

  stable_sort(ops.begin(), ops.end(), cmp);

  return ops;
}

std::vector<std::vector<int>>
SDCSchedule::addResourceConstrBB(BasicBlock *BB,
                                 std::vector<std::vector<int>> &&Vars, int RId,
                                 SDCSolver *SDC) {
  int Amount = RDB.getAmount(RId);

  std::vector<SDCOpWrapper *> constrainedOp = getFeasibleOrder(
      BB, [&](SDCOpWrapper *op) { return op->getResource() == RId; });

  std::vector<std::vector<int>> vec(std::move(Vars));

  for (int i = 0, n = constrainedOp.size(); i < n; ++i) {
    int slot = i % Amount;
    int Var = constrainedOp[i]->VarId;
    for (auto x : vec[slot])
      SDC->addInitialConstraint(Constraint::CreateGE(Var, x, RDB.getII(RId)));
    vec[slot] = {Var};
  }

  return vec;
}

void SDCSchedule::addResourceConstr(int RId, SDCSolver *SDC) {
  std::unordered_map<BasicBlock *, bool> Visited;
  std::unordered_map<BasicBlock *, int> InDeg;
  std::unordered_map<BasicBlock *, std::vector<std::vector<int>>> Chains;
  std::queue<BasicBlock *> Q;

  for (auto &&BB : BasicBlocks) {
    if (!needSchedule(BB.get()))
      continue;
    for (auto Succ : BB->getSucc())
      if (Succ.type != ControlEdge::LOOPBACK && needSchedule(Succ.toBB))
        InDeg[Succ.toBB]++;
  }

  int Amount = RDB.getAmount(RId);

  for (auto &&BB : BasicBlocks) {
    if (!needSchedule(BB.get()))
      continue;
    if (InDeg[BB.get()] == 0) {
      Chains[BB.get()].resize(Amount);
      Visited[BB.get()] = true;
      Q.push(BB.get());
    }
  }

  while (!Q.empty()) {
    auto now = Q.front();
    Q.pop();

    auto now_chains =
        addResourceConstrBB(now, std::move(Chains[now]), RId, SDC);

    for (auto Succ : now->getSucc())
      if (Succ.type != ControlEdge::LOOPBACK && needSchedule(Succ.toBB)) {
        auto succBB = Succ.toBB;
        if (!Visited[succBB]) {
          Chains[succBB].resize(Amount);
          Visited[succBB] = true;
        }

        auto &vec = Chains[succBB];
        for (int i = 0; i < Amount; ++i)
          vec[i].insert(vec[i].end(), now_chains[i].begin(),
                        now_chains[i].end());

        InDeg[succBB]--;
        if (InDeg[succBB] == 0)
          Q.push(succBB);
      }
  }
}

void SDCSchedule::addMemConstr(int memportRId, SDCSolver *SDC) {
  // assume no overlapping execution of basic blocks
  int amount = RDB.getAmount(memportRId);
  int II = RDB.getII(memportRId);
  llvm::errs() << "addMemConstr called for resource: " << RDB.getName(memportRId)
               << " (ID=" << memportRId << ", amount=" << amount << ", II=" << II << ")\n";
  for (auto &&BB : BasicBlocks) {
    if (!needSchedule(BB.get()))
      continue;

    // resource confliction of pipelined loop has been handled
    // memport conflict of non-pipelined loop
    std::vector<SDCOpWrapper *> constrainedOp =
        getFeasibleOrder(BB.get(),
                         [&](SDCOpWrapper *op) {
                             return op->getMemOp() != nullptr && op->getResource() == memportRId;
                         });
    llvm::errs() << "  Found " << constrainedOp.size() << " operations with this resource\n";
    llvm::DenseMap<mlir::Value, std::vector<SDCOpWrapper*>> memrefOpMap;
    for (int i = 0, n = constrainedOp.size(); i < n; ++i) {
      MemOpConcrete *memop = constrainedOp[i]->getMemOp();
      memrefOpMap[memop->getMemRef()].push_back(constrainedOp[i]);
    }

    for (auto it: memrefOpMap) {
      llvm::errs() << "    Processing memref group with " << it.second.size() << " operations\n";
      std::vector<std::vector<int>> vars(amount);
      for (size_t i = 0; i < it.second.size(); ++i) {
        int slot = i % amount;
        it.second[i]->getOp()->setAttr("slot", IntegerAttr::get(IntegerType::get(containingOp->getContext(), 32),
                               slot));
        int var = it.second[i]->VarId;
        // For each operation in the same slot, add constraint that current op starts after previous ops
        for (auto prevVar : vars[slot]){
          // current op must start at least II cycles after previous op in same slot
          llvm::errs() << "      Adding constraint: Var" << var << " >= Var" << prevVar << " + " << II << "\n";
          SDC->addInitialConstraint(Constraint::CreateGE(var, prevVar, II));
        }

        vars[slot].push_back(var);
      }
    }
  }
}

static void addMAxiConstrHelper(SDCSolver *SDC, int lat, int requestLat, int amount,
  const std::unordered_map<mlir::Operation*, std::vector<SDCOpWrapper*>>& burstOps,
  std::unordered_map<std::string, std::vector<SDCOpWrapper*>>& busOpMap)
{
  for (auto it: busOpMap) {
      std::vector<std::vector<SDCOpWrapper*>> vars(amount);
      for (size_t i = 0; i < it.second.size(); ++i) {
        int slot = i % amount;
        int var = it.second[i]->VarId;
        for (auto sdcOp : vars[slot]) {
          MAxiOpConcrete *maxiop = sdcOp->getMAxiOp();
          int x = sdcOp->VarId;
          SDC->addInitialConstraint(Constraint::CreateGE(var, x, maxiop->isBurst()? requestLat : lat));
        }
        MAxiOpConcrete *maxiop = it.second[i]->getMAxiOp();
        if (maxiop->isBurst()) {
          auto request = maxiop->getRequest();
          mlir::Operation* requestOp = request.getDefiningOp();
          vars[slot] = {burstOps.at(requestOp).back()};
        } else {
          vars[slot] = {it.second[i]};
        }
      }
    }
}

void SDCSchedule::addMAxiConstr(SDCSolver *SDC) {
  // assume no overlapping execution of basic blocks
  // or overlapping basic blocks does not have dependecy issue
  int amount = 1; // one bus channel for read and one for write
  int readLat = 1, writeLat = 1, requestLat = 1;
  for (auto &&BB : BasicBlocks) {
    if (!needSchedule(BB.get()))
      continue;

    // resource confliction of pipelined loop has been handled
    // mAxi conflict of non-pipelined loop
    // Exclude TileLink operations (they are handled by addResourceConstr with their own amount)
    std::vector<SDCOpWrapper *> constrainedOp = getFeasibleOrder(BB.get(),
                         [&](SDCOpWrapper *op) {
                            return op->getMAxiOp() != nullptr;
                         });

    std::unordered_map<mlir::Operation*, std::vector<SDCOpWrapper*>> burstOps;
    for (int i = 0, n = constrainedOp.size(); i < n; ++i) {
      MAxiOpConcrete *maxiop = constrainedOp[i]->getMAxiOp();
      if (maxiop->isBurst()) {
        auto request = maxiop->getRequest();
        mlir::Operation* requestOp = request.getDefiningOp();
        burstOps[requestOp].push_back(constrainedOp[i]);
      }
    }

    std::unordered_map<std::string, std::vector<SDCOpWrapper*>> readBusOpMap;
    std::unordered_map<std::string, std::vector<SDCOpWrapper*>> writeBusOpMap;
    std::unordered_set<mlir::Operation*> processed;
    for (int i = 0, n = constrainedOp.size(); i < n; ++i) {
      MAxiOpConcrete *maxiop = constrainedOp[i]->getMAxiOp();
      auto bus = maxiop->getBus();
      auto request = maxiop->getRequest(); // 一个request对应一次burst
      if (request != nullptr) {
        mlir::Operation* requestOp = request.getDefiningOp();
        if (processed.find(requestOp) == processed.end()) {
          if (maxiop->isRead()) {
            readBusOpMap[bus].push_back(constrainedOp[i]);
          } else {
            // write response or other writeOp
            writeBusOpMap[bus].push_back(constrainedOp[i]);
          }
        }
      } else {
        if (maxiop->isRead()) {
          readBusOpMap[bus].push_back(constrainedOp[i]);
        } else if (maxiop->isWrite()) {
          writeBusOpMap[bus].push_back(constrainedOp[i]);
        }
      }
    }

    addMAxiConstrHelper(SDC, readLat, requestLat, amount, burstOps, readBusOpMap);
    addMAxiConstrHelper(SDC, writeLat, requestLat, amount, burstOps, writeBusOpMap);
  }
}

SDCSolver *SDCSchedule::formulateSDC() {
  SDCSolver *SDC = new SDCSolver();

  std::unordered_map<BasicBlock *, int> BeginBB;
  std::unordered_map<BasicBlock *, int> EndBB;
  std::unordered_map<const Loop *, int> BeginLoop;
  std::unordered_map<const Loop *, int> EndLoop;

  // Allocate Variables
  for (auto &&BB : BasicBlocks) {
    if (!needSchedule(BB.get()))
      continue;

    BeginBB[BB.get()] = SDC->addVariable(); // super source of BB
    EndBB[BB.get()] = SDC->addVariable();   // super sink of BB

    SDC->addInitialConstraint(
      Constraint::CreateGE(EndBB[BB.get()],BeginBB[BB.get()],0)
    );

    for (auto op : BB->getOperations()) {
      auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(op);

      sdcOp->VarId = SDC->addVariable();

      // super source -> op
      SDC->addInitialConstraint(
          Constraint::CreateGE(sdcOp->VarId, BeginBB[BB.get()], 0));
      SDC->addInitialConstraint(Constraint::CreateGE(
          EndBB[BB.get()], sdcOp->VarId,
          RDB.getLatency(sdcOp->getResource(), sdcOp->getWidth())));
    }
  }

  for (auto &&L : Loops)
    if (L->PipelineFlag) {
      BeginLoop[L.get()] = SDC->addVariable();
      EndLoop[L.get()] = SDC->addVariable();

      SDC->addInitialConstraint(
          Constraint::CreateGE(EndLoop[L.get()], BeginLoop[L.get()],
                               L->AchievedII /*IS IT SUFFICIENT?*/));
    }

  // Control Dependency
  for (auto &&BB : BasicBlocks) {
    if (!needSchedule(BB.get()))
      continue;

    for (auto Succ : BB->getSucc()) {
      // We currenly don't support overlapping execution of Basicblocks.

      if (Succ.type != ControlEdge::LOOPBACK) {
        auto succBB = Succ.toBB;

        if (needSchedule(succBB))
          SDC->addInitialConstraint(
              Constraint::CreateGE(BeginBB[succBB], EndBB[BB.get()], 1));
      }
    }

    for (auto Succ : BB->getSucc()) {
      if (Succ.type == ControlEdge::LOOPBACK)
        continue;
      if (needSchedule(Succ.toBB))
        continue;

      const Loop *L = Succ.toBB->getParentLoop();
      SDC->addInitialConstraint(
          Constraint::CreateGE(BeginLoop[L], EndBB[BB.get()], 1)
          /* Pipelined Loop can't share state with non loop block */
      );
    }

    for (auto Pred : BB->getPred()) {
      if (Pred.type == ControlEdge::LOOPBACK)
        continue;
      if (needSchedule(Pred.fromBB))
        continue;

      const Loop *L = Pred.fromBB->getParentLoop();
      SDC->addInitialConstraint(
          Constraint::CreateGE(BeginBB[BB.get()], EndLoop[L], 1)
          /* Pipelined Loop can't share state with non loop block */
      );
    }
  }

  // Data Dependency
  for (auto &&BB : BasicBlocks) {
    if (BeginBB.find(BB.get()) == BeginBB.end())
      continue;

    for (auto op : BB->getOperations()) {
      auto predOp = llvm::dyn_cast<SDCOpWrapper>(op);
      for (auto Succ : predOp->getSucc()) {

        if (Succ->Distance != 0)
          continue;

        auto succOp = llvm::dyn_cast<SDCOpWrapper>(Succ->DestinationOp);

        if (!needSchedule(succOp->getParentBB())) {
          // pipeline
          SDC->addInitialConstraint(Constraint::CreateGE(
            BeginLoop[succOp->getParentBB()->getParentLoop()], predOp->VarId,
            RDB.getLatency(predOp->getResource(), predOp->getWidth())));
          continue;
        }

        // non-pipeline

        bool u = true;
        if(llvm::dyn_cast<tor::ForOp>(predOp->getOp())){
          if(predOp->getOp()->getNumOperands() > 3){
            for(auto x : succOp->getOp()->getOperands()){
              for(auto re : predOp->getOp()->getResults()){
                if(x == re){
                  u = false;
                  break;
                }
              }
              if(!u)break;
            }
          }
        }
        if(u){
          int latency = RDB.getLatency(predOp->getResource(), predOp->getWidth());
          llvm::errs() << "Adding SDC constraint: Var" << succOp->VarId
                       << " >= Var" << predOp->VarId << " + " << latency
                       << " (" << predOp->getOp()->getName().getStringRef() << " -> "
                       << succOp->getOp()->getName().getStringRef() << ")\n";
          SDC->addInitialConstraint(Constraint::CreateGE(
            succOp->VarId, predOp->VarId, latency));
        }
        
      }
    }
  }

  // Timing constraints
  for (auto &&BB : BasicBlocks)
    if (needSchedule(BB.get())) {
      for (auto op : BB->getOperations()) {
        auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(op);
        addTimingConstr(sdcOp, SDC);
      }
    }

  // Debug: Print all memory operations and their resources
  llvm::errs() << "=== Debug: All memory operations ===\n";
  for (auto &&BB : BasicBlocks) {
    for (auto op : BB->getOperations()) {
      if (op->getMemOp() != nullptr) {
        llvm::errs() << "  MemOp: " << op->getOp()->getName()
                     << " resource=" << RDB.getName(op->getResource())
                     << " (ID=" << op->getResource() << ")\n";
      }
    }
  }
  llvm::errs() << "====================================\n";

  // Resource constraints;
  int NumResource = RDB.getNumResource();

  for (int i = 0; i < NumResource; ++i) {
    if (RDB.getName(i).rfind("memport", 0) == 0) {
      // Only add constraints for memports with hard limits (SRAM, not register files)
      if (RDB.hasHardLimit(i)) {
        addMemConstr(i, SDC);
      }
    } else if (RDB.getName(i).rfind("m_axi", 0) == 0) {
      // skip
    } else if (RDB.hasHardLimit(i)) {
      addResourceConstr(i, SDC);
    }
  }
  // add m_axi constraint
  // burst maxi and normal maxi should be considered together
  addMAxiConstr(SDC);

  // Add constraints for aps.readrf and aps.writerf operations
  // readrf should be early, writerf should be at the end
  llvm::SmallVector<SDCOpWrapper *> writeRfOps;
  llvm::SmallVector<SDCOpWrapper *> readRfOps;
  llvm::SmallVector<SDCOpWrapper *> otherOps;

  for (auto &&BB : BasicBlocks) {
    if (!needSchedule(BB.get()))
      continue;
    for (auto op : BB->getOperations()) {
      auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(op);
      if (llvm::isa<aps::CpuRfWrite>(sdcOp->getOp())) {
        writeRfOps.push_back(sdcOp);
      } else if (llvm::isa<aps::CpuRfRead>(sdcOp->getOp())) {
        readRfOps.push_back(sdcOp);
      } else if (!llvm::isa<arith::ConstantOp, memref::GetGlobalOp, tor::ReturnOp>(sdcOp->getOp())) {
        // Collect non-constant, non-global, non-return operations
        otherOps.push_back(sdcOp);
      }
    }
  }

  // Add constraints: all other operations must complete before writerf
  for (auto writeRfOp : writeRfOps) {
    for (auto otherOp : otherOps) {
      int latency = RDB.getLatency(otherOp->getResource(), otherOp->getWidth());
      llvm::errs() << "Adding writerf ordering constraint: Var" << writeRfOp->VarId
                   << " >= Var" << otherOp->VarId << " + " << latency
                   << " (" << otherOp->getOp()->getName().getStringRef() << " -> "
                   << writeRfOp->getOp()->getName().getStringRef() << ")\n";
      SDC->addInitialConstraint(Constraint::CreateGE(
        writeRfOp->VarId, otherOp->VarId, latency));
    }
  }

  auto getFuncOp = [&](SDCOpWrapper *op) {
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

  auto getLatency = [&](SDCOpWrapper *op) {
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
        auto from = dyn_cast<IntegerAttr>(succOp.getPoints()[i]).getInt();
        auto to = succOp.getTime();
        for (auto attr : dyn_cast<DictionaryAttr>(succOp.getEdges()[i])) {
          auto edge_attr = attr.getValue();
          if (auto str_attr = dyn_cast<StringAttr>(edge_attr)) {
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

  for (auto &&BB : BasicBlocks) {
    llvm::DenseMap<Operation* , int > op_varid;
    for (auto op : BB->getOperations()) {
      auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(op);
      op_varid[sdcOp->getOp()] = sdcOp->VarId;
    }
    // add dependency to callOp_result -> op
    for (auto op : BB->getOperations()) { 
      auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(op);
      if (auto callOp = llvm::dyn_cast<tor::CallOp>(sdcOp->getOp())) {
        for (auto result : callOp.getResults()) {
          for (auto &use : result.getUses()) {
            auto opUser = use.getOwner();
            if (!op_varid[opUser]) continue;
            SDC->addInitialConstraint(Constraint::CreateGE(op_varid[opUser], sdcOp->VarId, getLatency(sdcOp))); 
          }
        }
      }
    }
  }
  return SDC;
}

bool SDCSchedule::minimizeLifetimeFunction(int II, SDCSolver *SDC) {
  int varCnt = SDC->getNumVariable();

  lprec *lp = make_lp(0, SDC->getNumVariable());
  set_verbose(lp, NEUTRAL);
  SDC->convertLP(lp);

  // add lifetime variables
  for (auto &&sdcOp : SDCOperations) {
    // add lifetime variable for op
    int rowno[1] = {0};
    // coefficient of objective function
    double col[1] = {(double)sdcOp->getWidth()};
    add_columnex(lp, 1, col, rowno);

    varCnt++;
    int newId = get_Ncolumns(lp);

    set_add_rowmode(lp, TRUE);
    for (auto succ : sdcOp->getSucc())
      if (succ->type == Dependence::D_RAW) {
        auto succOp = llvm::dyn_cast<SDCOpWrapper>(succ->DestinationOp);

        int colno[3] = {succOp->VarId + 1, sdcOp->VarId + 1, newId};
        double row[3] = {1, -1, -1};
        add_constraintex(
            lp, 3, row, colno, LE,
            RDB.getLatency(sdcOp->getResource(), sdcOp->getWidth()));
      }
    set_add_rowmode(lp, FALSE);
  }

  set_minim(lp);
  int ret = solve(lp);
  LLVM_DEBUG(llvm::dbgs() << ret << "\n");
  if (ret == INFEASIBLE)
    return false;

  REAL *results = new REAL[varCnt];
  get_variables(lp, results);

  for (int i = 0, n = SDC->getNumVariable(); i < n; ++i) {
    int sol = std::round(results[i]);
    assert(sol >= 0);
    SDC->assignSolution(i, sol);
  }

  for (auto &&sdcOp : SDCOperations)
    sdcOp->OptTime = SDC->Solution[sdcOp->VarId];

  SDC->ValidFlag = true;

  assert(SDC->verify() > 0);

  delete_lp(lp);

  return true;
}

bool SDCSchedule::resolveResourceConstraintFunction(int II, SDCSolver *SDC) {
  // todo support axi + axi burst
  int ResKind = RDB.getNumResource();

  // keep the resource usage
  std::vector<std::vector<int>> ResTable(ResKind, std::vector<int>(II, 0));
  std::vector<int> ResLimit(ResKind, 0);
  std::vector<bool> HardFlag(ResKind, 0);

  for (int i = 0; i < ResKind; ++i)
    if (RDB.hasHardLimit(i)) {
      if (RDB.hasHardLimit(i)) {
        ResLimit[i] = RDB.getAmount(i);
        HardFlag[i] = 1;
      } else {
        int NumOp = 0;
        for (auto &&op : SDCOperations)
          if (op->getResource() == i)
            NumOp++;
        ResLimit[i] = (NumOp + II - 1) / II;
      }
    }

  // keep the scheduled memop
  std::vector<std::vector<std::pair<SDCOpWrapper *, int>>> ScheduledMemOp(II);

  typedef std::pair<int, SDCOpWrapper *> datumT;
  auto setcmp = [](const datumT &a, const datumT &b) {
    if (a.first != b.first)
      return a.first < b.first;
    else
      return a.second->VarId < b.second->VarId;
  };

  std::set<datumT, decltype(setcmp)> S(setcmp);

  // Ensure each op is tried at most II times.
  std::map<SDCOpWrapper *, int> failedCnt;

  for (auto &&sdcOp : SDCOperations)
    if (RDB.hasHardLimit(sdcOp->getResource()))
      S.insert(std::make_pair(sdcOp->OptTime, sdcOp.get()));

    std::map<SDCOpWrapper *, int> perturbation;
    std::map<SDCOpWrapper *, int> earliestTime;

    for (auto &[step, op] : S) {
      int pert_before =
          SDC->tryAddConstr(Constraint::CreateLE(op->VarId, step - 1));
      int pert_after =
          SDC->tryAddConstr(Constraint::CreateGE(op->VarId, step + 1));
      pert_before = pert_before != -1 ? pert_before : SDC->getNumVariable();
      pert_after = pert_after != -1 ? pert_after : SDC->getNumVariable();

      perturbation[op] = std::max(pert_before, pert_after);
    }

  while (!S.empty()) {
    int step = S.begin()->first;

    std::vector<SDCOpWrapper *> ops;
    while (!S.empty() && S.begin()->first == step) {
      ops.push_back(S.begin()->second);
      S.erase(S.begin());
    }

    // calc perturbation
    for(auto op : ops){
        earliestTime[op] = op->ASAPTime;;
    }

    auto cmp = [&](SDCOpWrapper *a, SDCOpWrapper *b) {
      if (perturbation[a] != perturbation[b])
        return perturbation[a] > perturbation[b];
      return earliestTime[a] < earliestTime[b];
    };

    std::stable_sort(ops.begin(), ops.end(), cmp);

    for (auto op : ops) {
      bool scheduled = false;
      int RId = op->getResource();

      // only schedule resource constrained operations

      for (int s = step, n = earliestTime[op]; s >= n; --s) {
        // check resource availability
        bool avail = true;

        if (RDB.getName(RId).rfind("memport", 0) == 0) {
          // assume memory port has one cycle latency and can't be pipelined
          int slot = s % II;
          for (auto &sdcOp : ScheduledMemOp[slot]) {
            int dist = (s - sdcOp.second) / II;
            // check if op in current iteration can have resource conflict with
            // sdc op after dist iteraions.

            if (hasMemPortConflict(op, sdcOp.first, dist)) {
              avail = false;
              break;
            }
          }
        } else {
          for (int i = 0; i < RDB.getII(RId); ++i)
            if (ResTable[RId][(s + i) % II] >= ResLimit[RId]) {
              avail = false;
              break;
            }
        }

        if (avail == false)
          continue;

        // TODO: Resource sharing among mutual exclusive branches.
        if (SDC->addConstraint(Constraint::CreateEQ(op->VarId, s))) {
          scheduled = true;

          for (int i = 0; i < RDB.getII(RId); ++i)
            ResTable[RId][(s + i) % II]++;

          if (RDB.getName(RId).rfind("memport", 0) == 0)
            ScheduledMemOp[s % II].push_back(std::make_pair(op, s));

          break;
        }
      }

      if (scheduled == true)
        continue;

      if (SDC->addConstraint(Constraint::CreateGE(op->VarId, step + 1)) ==
          false){
            return false;
          }
        

      if (failedCnt[op] == II)
        return false;

      failedCnt[op]++;
      earliestTime[op] = SDC->Solution[op->VarId];
      S.insert(std::make_pair(earliestTime[op], op));
    }
  }

  return true;
}

void SDCSchedule::FunctionAllocVariable(std::unordered_map<BasicBlock *, int> &BeginBB,std::unordered_map<BasicBlock *, int> &EndBB,SDCSolver *SDC){
    for (auto &&BB : BasicBlocks) {
        BeginBB[BB.get()] = SDC->addVariable();
        EndBB[BB.get()] = SDC->addVariable();
        for (auto op : BB->getOperations()) {
            auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(op);
            sdcOp->VarId = SDC->addVariable();

        }
    }
}

int GetValue(mlir::Value value,llvm::DenseMap<mlir::Value,mlir::Value> & blv){
  if(blv[value] != nullptr)value = blv[value];
  auto defineOp = value.getDefiningOp();
  if(defineOp == nullptr){
    return -1e9;
  }
  if(auto constantOp = llvm::dyn_cast<arith::ConstantOp>(defineOp)){
    return dyn_cast<mlir::IntegerAttr>(constantOp.getValue()).getInt();
  }
  if(auto muliop = llvm::dyn_cast<arith::MulIOp>(defineOp)){
    return GetValue(muliop.getOperand(0),blv) * GetValue(muliop.getOperand(1),blv); 
  }else if(auto addiop = llvm::dyn_cast<arith::AddIOp>(defineOp)){
    return GetValue(addiop.getOperand(0),blv) + GetValue(addiop.getOperand(1),blv); 
  }else if(auto shliop = llvm::dyn_cast<arith::ShLIOp>(defineOp)){
    return GetValue(shliop.getOperand(0),blv) << GetValue(shliop.getOperand(1),blv);
  }else if(auto subiop = llvm::dyn_cast<arith::SubIOp>(defineOp)){
    return GetValue(subiop.getOperand(0),blv) - GetValue(subiop.getOperand(1),blv);
  }
  assert(0);
  return -1e5;
}

void SDCSchedule::FunctionformulateDependency(std::unordered_map<BasicBlock *, int> &BeginBB,std::unordered_map<BasicBlock *, int> &EndBB,SDCSolver *SDC){
    for (auto &&BB : BasicBlocks) {
        // BeginBB[BB.get()] = SDC->addVariable();
        // EndBB[BB.get()] = SDC->addVariable();

        for (auto op : BB->getOperations()) {
            auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(op);
        SDC->addInitialConstraint(
            Constraint::CreateGE(sdcOp->VarId, BeginBB[BB.get()], 0));
        SDC->addInitialConstraint(Constraint::CreateGE(
            EndBB[BB.get()], sdcOp->VarId,
            RDB.getLatency(sdcOp->getResource(), sdcOp->getWidth())));
        }
    }

    for (auto &&BB : BasicBlocks) {
        for (auto Succ : BB->getSucc()) {
        // can't have loop inside function
        assert(Succ.type != ControlEdge::LOOPBACK &&
                "Can't have loop in the pipelined function");
        auto succBB = Succ.toBB;
        SDC->addInitialConstraint(
            Constraint::CreateGE(BeginBB[succBB], EndBB[BB.get()], 1));
        }
    }

    for (auto &&op : SDCOperations) {
        auto predOp = op.get();
        for (auto Succ : predOp->getSucc()) {
        auto succOp = llvm::dyn_cast<SDCOpWrapper>(Succ->DestinationOp);

        int Lat = RDB.getLatency(predOp->getResource(), predOp->getWidth());
        // This special case is because of codegen backend
        if (predOp->getType() == OpAbstract::OpType::PHI_OP)
            Lat = 1;
        SDC->addInitialConstraint(
            Constraint::CreateGE(succOp->VarId, predOp->VarId, Lat));
        }
    }

    for (auto &&BB : BasicBlocks)
        for (auto op : BB->getOperations()) {
        auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(op);
        addTimingConstr(sdcOp, SDC);
        }

  auto getFuncOp = [&](SDCOpWrapper *op) {
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

  auto getLatency = [&](SDCOpWrapper* op,IntegerAttr end) {
    assert(llvm::isa<tor::CallOp>(op->getOp()));
    int latency = 0;
    for (auto timeGraphOp : getFuncOp(op).getOps<tor::TimeGraphOp>()) {
      for (auto succOp : timeGraphOp.getOps<tor::SuccTimeOp>()) {
        for (int i = 0, r = succOp.getPoints().size(); i < r; ++i) {
          for (auto attr : dyn_cast<DictionaryAttr>(succOp.getEdges()[i])) {
            auto strAttr = dyn_cast<StringAttr>(attr.getValue());
            if (strAttr == nullptr)
              continue;
            auto str = strAttr.getValue();
            if(succOp.getPoints()[i] == end){
              return latency + 1;
            }
            if (str.find(':') != StringRef::npos) {
              auto index = str.find(':');
              auto numberStr = str.substr(index + 1, str.size() - index - 1);
              latency += std::stoi(numberStr.str());
            }
          }
        }
      }
    }
    return latency + 1;
  };

  llvm::DenseMap<mlir::Value,mlir::Value> BlockValue;
    for (auto &&BB : BasicBlocks) {
    std::vector<SDCOpWrapper *> callOps;
    for (auto op : BB->getOperations()) {
      auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(op);
      if (llvm::isa<tor::CallOp>(sdcOp->getOp())) {
        callOps.push_back(sdcOp);
      }
    }
    for (int i = 0, r = callOps.size(); i < r; ++i) {
      auto funcOp = getFuncOp(callOps[i]);
      for (int j = i + 1; j < r; ++j) {
        auto newfuncOp = getFuncOp(callOps[j]);
        if(!funcOp->hasAttr("II") || !newfuncOp->hasAttr("II") || funcOp->hasAttr("parallel-off") || newfuncOp->hasAttr("parallel-off")){
          int sum = 0;
          for (auto timeGraphOp : getFuncOp(callOps[j]).getOps<tor::TimeGraphOp>()) {
            for (auto succOp : timeGraphOp.getOps<tor::SuccTimeOp>()) {
              for (int i = 0, r = succOp.getPoints().size(); i < r; ++i) {
                for (auto attr : dyn_cast<DictionaryAttr>(succOp.getEdges()[i])) {
                  auto strAttr = dyn_cast<StringAttr>(attr.getValue());
                  if (strAttr == nullptr)
                    continue;
                  auto str = strAttr.getValue();
                  if (str.find(':') != StringRef::npos) {
                    auto index = str.find(':');
                    auto numberStr = str.substr(index + 1, str.size() - index - 1);
                    sum += std::stoi(numberStr.str());
                  }
                }
              }
            }
          }
          SDC->addInitialConstraint(Constraint::CreateGE(callOps[i]->VarId, callOps[j]->VarId, sum + 1));
        }
      }
      if(funcOp->hasAttr("parallel-off")){
        funcOp->removeAttr("parallel-off");
      }
    }
  }

  for (auto &&BB : BasicBlocks) {
    int callnumber = 0;
    llvm::DenseMap<Operation* , int > op_varid; 
    int tmp = -1;
    for (auto op : BB->getOperations()) { //add dependency to call -> op and op -> call
      auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(op);
       if(auto callop = llvm::dyn_cast<tor::CallOp>(sdcOp->getOp())){
        callnumber++;
        for(auto res : callop.getResults()){
          for(auto &x : res.getUses()){
            auto opuser  = x.getOwner();
            auto oldattr = dyn_cast<mlir::IntegerAttr>(sdcOp->getOp()->getAttr("endtime"));
            mlir::IntegerAttr newAttr = mlir::IntegerAttr::get(oldattr.getType(), -1);
            SDC->addInitialConstraint(Constraint::CreateGE(op_varid[opuser],sdcOp->VarId , getLatency(sdcOp,newAttr))); // add dependency to callop_result -> op
          }
        }
      }
      op_varid[sdcOp->getOp()] = sdcOp->VarId;
    }
  }
}

bool SDCSchedule::FunctiongetASAPTime(SDCSolver *SDC){
  if(!SDC->initSolution()){
    return false;
  }
  for (auto &&op : SDCOperations)
    op->ASAPTime = SDC->Solution[op->VarId];
  return true;
}

bool SDCSchedule::pipelineFunctionWithII(int II, bool FinalFlag) {
  LLVM_DEBUG(llvm::dbgs() << "try II=" << II << "\n");
  
  SDCSolver *SDC = new SDCSolver();
  std::unordered_map<BasicBlock *, int> BeginBB;
  std::unordered_map<BasicBlock *, int> EndBB;

  FunctionAllocVariable(BeginBB,EndBB,SDC);
  // contol dependency
  FunctionformulateDependency(BeginBB,EndBB,SDC);

  if(FunctiongetASAPTime(SDC) == false){
    return false;
  }

  if (minimizeLifetimeFunction(II, SDC) == false){
    return false;
  }
    

  for (auto &&sdcOp : SDCOperations) {
    LLVM_DEBUG(llvm::dbgs() << sdcOp->getOp()->getName() << " " << sdcOp->ASAPTime << " "
                 << sdcOp->OptTime << "\n");
  }

  if (resolveResourceConstraintFunction(II, SDC) == false){
    return false;
  }
    
  if (FinalFlag) {
    minimizeLifetimeFunction(II, SDC);
    for (auto &&op : SDCOperations) {
      auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(op.get());
      sdcOp->SDCTime = SDC->Solution[sdcOp->VarId];
    }
  }
  return true;
}

bool SDCSchedule::pipelineFunction() {
  int targetII = containingOp->getAttrOfType<IntegerAttr>("II").getInt();
  containingOp->removeAttr("II");
  if (pipelineFunctionWithII(targetII, false)) {
    LLVM_DEBUG(llvm::dbgs() << "Achieved II: " << targetII << "\n");
    containingOp->setAttr(
        "II", IntegerAttr::get(IntegerType::get(containingOp->getContext(), 32),
                               targetII));
    pipelineFunctionWithII(targetII, true);
    return true;
  }

  int l = targetII, r = 256, achieved = -1;
  while (l <= r) {
    int mid = (l + r) >> 1;
    if (pipelineFunctionWithII(mid, 0)) {
      achieved = mid;
      r = mid - 1;
    } else {
      l = mid + 1;
    }
  }

  if (achieved != -1) {
    LLVM_DEBUG(llvm::dbgs() << "Achieved II: " << achieved << "\n");
    containingOp->setAttr(
        "II", IntegerAttr::get(IntegerType::get(containingOp->getContext(), 32),
                               achieved));
    pipelineFunctionWithII(achieved, true);
    return true;
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Failed to find a reasonable II\n");
    return false;
  }

  return false;
}

bool SDCSchedule::checkParallelCallOpResourceConstraints(SDCSolver *SDC, bool &needReschedule) {
  // check if there is parallel callOp, and check resource if satisfy,
  // if not satisfy, add constraint
  auto funcOp = llvm::dyn_cast<tor::FuncOp>(containingOp);
  auto designOp = llvm::dyn_cast<tor::DesignOp>(funcOp->getParentOp());
  for (auto &&BB: BasicBlocks) {
    std::map<int, std::vector<OpAbstract*>> callOps;
    for (auto &opA : BB->getOperations()) {
      if (opA->getType() == OpAbstract::OpType::CALL_OP) {
        callOps[opA->getStartTime()].push_back(opA);
      }
    }

    // callop will not be scheduled in parallel with other types of ops
    for (auto iter: callOps) {
      size_t n = iter.second.size();
      if (n > 1) {
        std::vector<int> parallelUsage;
        size_t startIndex = 0;
        for (size_t i = 0; i < n; ++i) {
          OpAbstract* opA = iter.second[i];
          auto callOp = llvm::dyn_cast<tor::CallOp>(opA->getOp());
          auto callFuncOp = designOp.lookupSymbol<tor::FuncOp>(callOp.getCallee());
          const auto usage = RDB.getUsage(callFuncOp);
          size_t size = std::max(usage.size(), parallelUsage.size());
          parallelUsage.resize(size, 0);
          for (size_t k = 0; k < usage.size(); ++k) {
            parallelUsage[k] += usage[k];
          }
          if (startIndex < i && !RDB.isUsageSatisfied(parallelUsage)) {
            needReschedule = true;
            for (; startIndex < i; ++startIndex) {
              auto sdcOpA = llvm::dyn_cast<SDCOpWrapper>(iter.second[startIndex]);
              for (size_t j = i; j < n; ++j) {
                auto sdcOpB = llvm::dyn_cast<SDCOpWrapper>(iter.second[j]);
                if (!SDC->addConstraint(Constraint::CreateGE(sdcOpB->VarId, sdcOpA->VarId, 1))) {
                  return false;
                }
              }
            }
            parallelUsage = usage;
          }
        }
      }
    }
  }
  return true;
}

LogicalResult SDCSchedule::runSchedule() {
  if (auto dataflow_flag = containingOp->getAttrOfType<IntegerAttr>("dataflow")) {
    isDataflowRegion = dataflow_flag.getInt() == 1;
  }

  buildFromContainingOp();

  SDCOperations = initSchedule<SDCOpWrapper>();

  if (auto pipeline_flag =
          containingOp->getAttrOfType<StringAttr>("pipeline")) {
    // pipeline this function
    if (pipeline_flag.getValue().str() == "func") {
      if (pipelineFunction())
        return success();
      LLVM_DEBUG(llvm::dbgs() << containingOp->getName()
                   << ". Function pipelining failed!\n");
      return failure();
    }
  }

  for (auto &&L : Loops) {
    if (L->PipelineFlag == true) {
      if (pipelineLoop(L.get()) == false)
        L->PipelineFlag = false;
    }
  }

  SDCSolver *SDC = formulateSDC();
  assert(SDC->initSolution());
  assert(SDC->verify());
  bool needReschedule = false;
  // todo isolate axiop
  // todo parallel callop axi conflict check
  if (!checkParallelCallOpResourceConstraints(SDC, needReschedule)) {
    LLVM_DEBUG(llvm::dbgs() << containingOp->getName()
                   << ". Parallel CallOp Resource Constraints Check failed!\n");
    return failure();
  }

  if (needReschedule) {
    assert(SDC->initSolution());
    assert(SDC->verify());
  }

  for (auto &&sdcOp : SDCOperations)
    if (sdcOp->getParentLoop() == nullptr ||
        sdcOp->getParentLoop()->PipelineFlag == false) {
      sdcOp->SDCTime = SDC->Solution[sdcOp->VarId];
      llvm::errs() << "SDC Result: Var" << sdcOp->VarId << " = " << sdcOp->SDCTime
                   << " (" << sdcOp->getOp()->getName().getStringRef() << ")\n";
    }

  assignSlotAttrAfterSchedule();
  return success();
}

void SDCSchedule::assignSlotAttrAfterSchedule() {
  // memop assign slot id
  for (auto &&BB : BasicBlocks) {
    if (needSchedule(BB.get())) {
      continue;
    }
    llvm::DenseMap<mlir::Value, llvm::DenseMap<int, std::vector<OpAbstract*>>> memrefOpMap;
    for (auto &opA : llvm::reverse(BB->getOperations())) {
      MemOpConcrete *memOp = opA->getMemOp();
      if (memOp != nullptr) {
        memrefOpMap[memOp->getMemRef()][opA->getStartTime()].push_back(opA);
      }
    }

    for (auto it: memrefOpMap) {
      for (auto timeMemOpMap: it.second) {
        int slot = 0;
        assert(timeMemOpMap.second.size() <= 2);
        for (auto op: timeMemOpMap.second) {
          op->getOp()->setAttr("slot", IntegerAttr::get(IntegerType::get(containingOp->getContext(), 32),
                               slot++));
        }
      }
    }
  }

  auto id = RDB.getResourceID("memport");
  assert(RDB.hasHardLimit(id));
  int numport = RDB.getAmount(id);
  for (auto &&L : Loops) {
    if (L->PipelineFlag == true) {
      for (auto BB : L->getBody()) {
        llvm::DenseMap<mlir::Value, int> memrefFirstUsedTimes;
        std::vector<OpAbstract *> memOpAs;
        for (auto opA : BB->getOperations()) {
          if (auto memOp = opA->getMemOp()) {
            auto memref = memOp->getMemRef();
            auto defOp = memref.getDefiningOp();

            // Handle both tor::AllocOp and memref::GetGlobalOp
            if (auto allocOp = llvm::dyn_cast<tor::AllocOp>(defOp)) {
              if (allocOp->hasAttr("bind_storage_type")) {
                auto type = dyn_cast<StringAttr>(allocOp->getAttr("bind_storage_type")).getValue();
                if (type == "RAM_1P")
                  continue;
              } else if (numport == 1) {
                continue;
              }
            } else if (llvm::isa<memref::GetGlobalOp>(defOp)) {
              // For memref.get_global, treat as multi-port regfile
              // Skip if only single port available
              if (numport == 1) {
                continue;
              }
            }
            memOpAs.push_back(opA);
            if (memrefFirstUsedTimes.find(memref) != memrefFirstUsedTimes.end()) {
              memrefFirstUsedTimes[memref] =
                std::min(memrefFirstUsedTimes[memref], opA->getStartTime());
            } else {
              memrefFirstUsedTimes[memref] = opA->getStartTime();
            }
          }
        }

        for (auto opA: memOpAs) {
          int memrefStartTime = memrefFirstUsedTimes[opA->getMemOp()->getMemRef()];
          if (opA->getStartTime() - L->AchievedII >= memrefStartTime) {
            opA->getOp()->setAttr("slot",
              IntegerAttr::get(IntegerType::get(containingOp->getContext(), 32), 1));
          }
        }
      }
    }
  }

  // TileLink channel assignment based on time interval overlap
  // Two TileLink ops need different channels if their active intervals overlap
  if (RDB.hasResource("tl")) {
    auto tlId = RDB.getResourceID("tl");
    int numChannels = RDB.getAmount(tlId);
    int tlLatency = RDB.getLatency(tlId, 32);  // TileLink latency

    // Helper: check if two intervals [s1, s1+lat) and [s2, s2+lat) overlap
    auto intervalsOverlap = [](int s1, int e1, int s2, int e2) {
      return s1 < e2 && s2 < e1;
    };

    // For non-pipelined basic blocks (needSchedule returns true for non-pipelined BBs)
    for (auto &&BB : BasicBlocks) {
      if (!needSchedule(BB.get())) {
        continue;  // Skip pipelined BBs, they are handled below
      }

      // Collect all TileLink ops with their time intervals
      std::vector<OpAbstract*> tlOps;
      for (auto &opA : BB->getOperations()) {
        if (opA->getResource() == tlId) {
          tlOps.push_back(opA);
        }
      }

      // Sort by start time
      std::sort(tlOps.begin(), tlOps.end(),
                [](OpAbstract *a, OpAbstract *b) {
                  return a->getStartTime() < b->getStartTime();
                });

      // Assign channels using interval graph coloring
      // channelEndTimes[i] = end time of the last op assigned to channel i
      std::vector<int> channelEndTimes(numChannels, -1);

      for (auto op : tlOps) {
        int startTime = op->getStartTime();
        int endTime = startTime + tlLatency;

        // Find a channel that doesn't overlap with this op
        int assignedChannel = -1;
        for (int ch = 0; ch < numChannels; ch++) {
          if (channelEndTimes[ch] <= startTime) {
            // This channel is free (its last op ended before this one starts)
            assignedChannel = ch;
            channelEndTimes[ch] = endTime;
            break;
          }
        }

        assert(assignedChannel >= 0 && "Too many overlapping TileLink ops");
        op->getOp()->setAttr("tl_channel",
          IntegerAttr::get(IntegerType::get(containingOp->getContext(), 32), assignedChannel));
      }
    }

    // For pipelined loops - use modular interval coloring
    for (auto &&L : Loops) {
      if (L->PipelineFlag == true) {
        for (auto BB : L->getBody()) {
          // Collect all TileLink ops
          std::vector<OpAbstract*> tlOps;
          for (auto opA : BB->getOperations()) {
            if (opA->getResource() == tlId) {
              tlOps.push_back(opA);
            }
          }

          // Sort by start time
          std::sort(tlOps.begin(), tlOps.end(),
                    [](OpAbstract *a, OpAbstract *b) {
                      return a->getStartTime() < b->getStartTime();
                    });

          // For pipelined loops, check overlap considering II wraparound
          // Two ops overlap if their modular intervals [s%II, (s+lat)%II) intersect
          std::vector<int> channelAssignment(tlOps.size(), -1);

          for (size_t i = 0; i < tlOps.size(); i++) {
            int si = tlOps[i]->getStartTime();
            int ei = si + tlLatency;

            // Find channels used by overlapping ops
            std::set<int> usedChannels;
            for (size_t j = 0; j < i; j++) {
              int sj = tlOps[j]->getStartTime();
              int ej = sj + tlLatency;

              // Check if intervals overlap (considering they may span multiple IIs)
              bool overlap = intervalsOverlap(si, ei, sj, ej);
              if (overlap && channelAssignment[j] >= 0) {
                usedChannels.insert(channelAssignment[j]);
              }
            }

            // Assign first available channel
            for (int ch = 0; ch < numChannels; ch++) {
              if (usedChannels.find(ch) == usedChannels.end()) {
                channelAssignment[i] = ch;
                break;
              }
            }

            assert(channelAssignment[i] >= 0 && "Too many overlapping TileLink ops in pipelined loop");
            tlOps[i]->getOp()->setAttr("tl_channel",
              IntegerAttr::get(IntegerType::get(containingOp->getContext(), 32), channelAssignment[i]));
          }
        }
      }
    }
  }
}

void SDCSchedule::printSDCSchedule(Loop *L) {
  llvm::outs() << "Loop: " << L << "\n";
  llvm::outs() << "========================================\n";

  llvm::outs() << "Target II: " << L->TargetII << "\n";
  llvm::outs() << "Achieved II: " << L->AchievedII << "\n";

  for (auto BB : L->getBody()) {
    llvm::outs() << "BasicBlock: " << BB << "\n";

    for (auto op : BB->getOperations()) {
      auto sdcOp = llvm::dyn_cast<SDCOpWrapper>(op);

      if (op->getType() == OpAbstract::OpType::DEFINED_OP) {
        llvm::outs() << op->getOp()->getName().getStringRef().str()
                     << ": operands(";
        for (auto opr : op->getOperands())
          llvm::outs() << mlir::hash_value(opr) << ", ";
        llvm::outs() << ")";
      } else if (op->getType() == OpAbstract::OpType::PHI_OP) {
        llvm::outs() << "PHI: operands(";
        for (auto opr : op->getOperands())
          llvm::outs() << mlir::hash_value(opr) << ", ";
        llvm::outs() << ")";
      } else if (op->getType() == OpAbstract::OpType::ASSIGN_OP) {
        llvm::outs() << "ASSIGN: operands(";
        for (auto opr : op->getOperands())
          llvm::outs() << mlir::hash_value(opr) << ", ";
        llvm::outs() << ")";
      } else if (op->getType() == OpAbstract::OpType::LOAD_OP) {
        auto memop = op->getMemOp();
        llvm::outs() << "LOAD: operands(";
        llvm::outs() << mlir::hash_value(memop->getAddr()) << "";
        llvm::outs() << ")";
      } else if (op->getType() == OpAbstract::OpType::STORE_OP) {
        auto memop = op->getMemOp();
        llvm::outs() << "STORE: operands(";
        llvm::outs() << mlir::hash_value(memop->getAddr()) << ", ";
        llvm::outs() << mlir::hash_value(memop->getOperands()[0]);
        llvm::outs() << ")";
      }

      llvm::outs() << " at cycle " << sdcOp->SDCTime << "\n";
    }
  }

  llvm::outs() << "===========================================\n";
}

} // namespace scheduling
