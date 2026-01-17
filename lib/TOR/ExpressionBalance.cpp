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
#include "mlir/Dialect/Affine/IR/AffineOps.h"

#include "TOR/Passes.h"
#include "TOR/TOR.h"
#include "TOR/Utils.h"
#include "TOR/TORDialect.h"
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

namespace
{
    struct node
    {
        mlir::Operation *op;
        mlir::Value opd, ops;
        bool u = false;
    };
    using namespace mlir;
    struct ExpressionBalancePass : ExpressionBalanceBase<ExpressionBalancePass>
    {

        template <typename opt>
        void Operandbalance(mlir::func::FuncOp &func)
        {
            llvm::DenseMap<Operation *, int> ct;
            int mas = 0;
            func.walk([&](Operation *op)
                      {
                          if (llvm::isa<opt>(op))
                          {
                              auto fir = op->getOperand(0);
                              auto sec = op->getOperand(1);
                              if (fir.getDefiningOp() != nullptr)
                              {
                                  auto fop = fir.getDefiningOp();
                                  if (llvm::isa<opt>(fop))
                                  {
                                      if (sec.getDefiningOp() != nullptr)
                                      {
                                          auto sop = sec.getDefiningOp();
                                          if (fop->getBlock() == sop->getBlock() && ct[sop] + 1 < ct[fop])
                                          {
                                              int cnt = 0;
                                              for(auto &x : fop->getResult(0).getUses()){
                                                cnt++;
                                              }
                                              if (cnt == 1 && fop->getOperand(0).getDefiningOp() != nullptr && fop->getOperand(1).getDefiningOp() != nullptr)
                                              {
                                                  if (fop->getOperand(0).getDefiningOp()->getBlock() == op->getBlock() && fop->getOperand(1).getDefiningOp()->getBlock() == op->getBlock())
                                                  {
                                                      fop->moveBefore(op);
                                                      if (ct[fop->getOperand(0).getDefiningOp()] > ct[fop->getOperand(1).getDefiningOp()])
                                                      {
                                                          auto tmp = fop->getOperand(0);
                                                          fop->setOperand(0, sop->getResult(0));
                                                          op->setOperand(1, tmp);
                                                      }
                                                      else
                                                      {
                                                          auto tmp = fop->getOperand(1);
                                                          fop->setOperand(1, sop->getResult(0));
                                                          op->setOperand(1, tmp);
                                                      }
                                                      ct[fop] = std::max(ct[fop->getOperand(0).getDefiningOp()], ct[fop->getOperand(1).getDefiningOp()]) + 1;
                                                  }
                                              }
                                          }
                                      }
                                      ct[op] = std::max(ct[op], ct[fop] + 1);
                                      mas = std::max(mas, ct[op]);
                                  }
                              }
                              if (sec.getDefiningOp() != nullptr)
                              {
                                  auto sop = sec.getDefiningOp();
                                  if (llvm::isa<opt>(sop))
                                  {
                                      if (fir.getDefiningOp() != nullptr)
                                      {
                                          auto fop = fir.getDefiningOp();
                                          int cnt = 0;
                                              for(auto &x : sop->getResult(0).getUses()){
                                                cnt++;
                                            }
                                          if (cnt == 1 && fop->getBlock() == sop->getBlock() && ct[fop] + 1 < ct[sop])
                                          {
                                              if (sop->getOperand(0).getDefiningOp() != nullptr && sop->getOperand(1).getDefiningOp() != nullptr)
                                              {
                                                  if (sop->getOperand(0).getDefiningOp()->getBlock() == op->getBlock() && sop->getOperand(1).getDefiningOp()->getBlock() == op->getBlock())
                                                  {
                                                      sop->moveBefore(op);
                                                      if (ct[sop->getOperand(0).getDefiningOp()] > ct[sop->getOperand(1).getDefiningOp()])
                                                      {
                                                          auto tmp = sop->getOperand(0);
                                                          sop->setOperand(0, fop->getResult(0));
                                                          op->setOperand(1, tmp);
                                                      }
                                                      else
                                                      {
                                                          auto tmp = sop->getOperand(1);
                                                          sop->setOperand(1, fop->getResult(0));
                                                          op->setOperand(1, tmp);
                                                      }
                                                      ct[sop] = std::max(ct[sop->getOperand(0).getDefiningOp()], ct[sop->getOperand(1).getDefiningOp()]) + 1;
                                                  }
                                              }
                                          }
                                      }
                                      ct[op] = std::max(ct[op], ct[sop] + 1);
                                  }
                              }
                              ct[op] = std::max(ct[op], 1);
                              mas = std::max(mas, ct[op]);
                          }
                      });
        }
        void ReduceLSchain(mlir::func::FuncOp &func){
            func.walk([&](affine::AffineForOp forOp){
            llvm::DenseMap<mlir::Value,llvm::DenseMap<mlir::Value,mlir::Operation *> > mt; //从前向后记录是否有load -> store同一块地址的情况
            llvm::DenseMap<mlir::Value,bool> opmap;
            forOp.walk([&](affine::AffineLoadOp loadop){
                if(loadop->getNumOperands() > 1){
                    auto mem = loadop.getMemref();
                    auto pl = loadop.getOperand(1);
                    mt[mem][pl] = loadop.getOperation();
                }
            });
            forOp.walk([&](affine::AffineStoreOp storeop){
                if(storeop->getNumOperands() > 2){                    
                    auto mem = storeop.getMemref();
                    auto pl = storeop.getOperand(2);
                    if(mt[mem][pl]){
                        opmap[mt[mem][pl]->getResult(0)] = true;
                    }
                }
            });
            auto iter = opmap.begin();
            for(;iter != opmap.end();){
                if(!iter->second)continue;
                int cnt = 0;
                Operation *opa,*opb;
                for(auto &use : iter->first.getUses()){ 
                    opa = use.getOwner();
                    cnt++;
                }
                if(cnt !=1){
                    iter++;
                    continue;//对此处使用>1就不进行操作，避免错误
                }
                for(auto &use : opa->getUses()){
                    opb = use.getOwner();
                    cnt++;
                }//考虑符号
                if(cnt !=2 || opb->getName() != opa->getName() || opb->getBlock() != opa->getBlock()||(!isa<arith::AddFOp>(opa)&&!isa<arith::AddIOp>(opa)&&!isa<arith::MulFOp>(opa)&&!isa<arith::MulIOp>(opa))){
                    iter++;
                    continue;//对此处使用>2就不进行操作，避免错误
                }
                int a1 = 0,b1 = 0;
                if(opa->getOperand(1) == iter->first){
                    a1 = 1;
                }
                if(opb->getOperand(1) != opa->getResult(0)){
                    b1 = 1;
                }
                opa->setOperand(a1,opb->getOperand(b1));
                opb->setOperand(b1,iter->first);
                opa->moveBefore(opb);
            }});
        }

        void runOnOperation() override
        {
            auto funcOp = getOperation();
            if (funcOp->hasAttr("balance"))
            {
                setPragmaStructureAttrStatusByOp(funcOp, "balance", true);
                ReduceLSchain(funcOp);
                llvm::DenseMap<mlir::Value, node> qmul, qadd, qsub, qdiv;
                llvm::DenseMap<mlir::Value, node> imul, iadd, isub, idiv;
                funcOp.walk([&](Operation *op)
                            {
                            OpBuilder builder(funcOp.getContext());
                            if (auto mulfOp = llvm::dyn_cast<arith::MulFOp>(op))  //单精度乘法
                            {
                                auto fir = mulfOp.getOperand(0);
                                auto sec = mulfOp.getOperand(1);
                                node nd;
                                nd.op = op;
                                if (fir.getDefiningOp() != nullptr)
                                {
                                    if (llvm::isa<arith::MulFOp>(fir.getDefiningOp()))
                                    {
                                        nd.opd = sec;
                                        nd.ops = fir;
                                        if (qmul[fir].u && qmul[fir].op->getBlock() == op->getBlock())
                                        {
                                            
                                            std::vector<std::pair<Operation *, int>> qs;
                                            int cnt = 0;
                                            for (auto &x : op->getResult(0).getUses())
                                            {
                                                auto opu = x.getOwner();
                                                qs.push_back({opu, x.getOperandNumber()});
                                                cnt++;
                                            }
                                            if(cnt == 1){
                                                builder.setInsertionPointAfter(op);
                                                op->setOperand(0, qmul[fir].opd);
                                                auto newop = builder.create<arith::MulFOp>(op->getLoc(), qmul[fir].ops, op->getResult(0));
                                                for (auto x : qs)
                                                {
                                                    if (x.first->getOperands() == newop->getOperands())
                                                        continue;
                                                    x.first->setOperand(x.second, newop->getResult(0));
                                                }
                                                qmul[fir].op->erase();
                                                qmul[fir].u = false;
                                            }
                                            
                                        }
                                        else
                                        {
                                            nd.u = true;
                                            qmul[op->getResult(0)] = nd;
                                        }
                                    }
                                }
                                if (sec.getDefiningOp() != nullptr)
                                {
                                    if (llvm::isa<arith::MulFOp>(sec.getDefiningOp()))
                                    {
                                        nd.opd = fir;
                                        nd.ops = sec;
                                        if (qmul[sec].u)
                                        {
                                            std::vector<std::pair<Operation *, int>> qs;
                                            int cnt = 0;
                                            for (auto &x : op->getResult(0).getUses())
                                            {
                                                auto opu = x.getOwner();
                                                qs.push_back({opu, x.getOperandNumber()});
                                                cnt++;
                                            }
                                            if(cnt == 1){
                                                builder.setInsertionPointAfter(op);
                                                op->setOperand(1, qmul[sec].opd);
                                                auto newop = builder.create<arith::MulFOp>(op->getLoc(), qmul[sec].ops, op->getResult(0));
                                                for (auto x : qs)
                                                {
                                                    if (x.first->getOperands() == newop->getOperands())
                                                        continue;
                                                    x.first->setOperand(x.second, newop->getResult(0));
                                                } 
                                                qmul[sec].op->erase();
                                                qmul[sec].u = false;
                                            }
                                            
                                        }
                                        else
                                        {
                                            nd.u = true;
                                            qmul[op->getResult(0)] = nd;
                                        }
                                    }
                                }
                            }
                            else if (auto addf = llvm::dyn_cast<arith::AddFOp>(op))//单精度加法
                            {
                                auto fir = addf.getOperand(0);
                                auto sec = addf.getOperand(1);
                                node nd;
                                nd.op = op;
                                if (fir.getDefiningOp() != nullptr)
                                {
                                    if (llvm::isa<arith::AddFOp>(fir.getDefiningOp()))
                                    {
                                        nd.opd = sec;
                                        nd.ops = fir;
                                        if (qadd[fir].u && qadd[fir].op->getBlock() == op->getBlock())
                                        {
                                            int cnt = 0;
                                            std::vector<std::pair<Operation *, int>> qs;
                                            for (auto &x : op->getResult(0).getUses())
                                            {
                                                auto opu = x.getOwner();
                                                qs.push_back({opu, x.getOperandNumber()});
                                                cnt++;
                                            }
                                            if(cnt == 1){
                                                builder.setInsertionPointAfter(op);
                                                op->setOperand(0, qadd[fir].opd);
                                                auto newop = builder.create<arith::AddFOp>(op->getLoc(), qadd[fir].ops, op->getResult(0));
                                                for (auto x : qs)
                                                {
                                                    if (x.first->getOperands() == newop->getOperands())
                                                        continue;
                                                    x.first->setOperand(x.second, newop->getResult(0));
                                                }
                                                qadd[fir].op->erase();
                                                qadd[fir].u = false;
                                            }
                                            
                                        }
                                        else
                                        {
                                            nd.u = true;
                                            qadd[op->getResult(0)] = nd;
                                        }
                                    }
                                }
                                if (sec.getDefiningOp() != nullptr)
                                {
                                    if (llvm::isa<arith::AddFOp>(sec.getDefiningOp()))
                                    {
                                        nd.opd = fir;
                                        nd.ops = sec;
                                        if (qadd[sec].u)
                                        {
                                            std::vector<std::pair<Operation *, int>> qs;
                                            int cnt = 0;
                                            for (auto &x : op->getResult(0).getUses())
                                            {
                                                auto opu = x.getOwner();
                                                qs.push_back({opu, x.getOperandNumber()});
                                                cnt++;
                                            }
                                            if(cnt == 1){
                                                builder.setInsertionPointAfter(op);
                                                op->setOperand(1, qadd[sec].opd);
                                                auto newop = builder.create<arith::AddFOp>(op->getLoc(), qadd[sec].ops, op->getResult(0));
                                                for (auto x : qs)
                                                {
                                                    if (x.first->getOperands() == newop->getOperands())
                                                        continue;
                                                    x.first->setOperand(x.second, newop->getResult(0));
                                                }
                                                qadd[sec].op->erase();
                                                qadd[sec].u = false;
                                            }
                                            
                                        }
                                        else
                                        {
                                            nd.u = true;
                                            qadd[op->getResult(0)] = nd;
                                        }
                                    }
                                }
                            }else if(auto subf =  llvm::dyn_cast<arith::SubFOp>(op)){
                                auto fir = subf.getOperand(0);
                                auto sec = subf.getOperand(1);
                                node nd;
                                nd.op = op;
                                if(fir.getDefiningOp() != nullptr){
                                    if(llvm::isa<arith::SubFOp>(fir.getDefiningOp())){
                                        nd.opd = sec;
                                        nd.ops = fir;
                                        if(qsub[fir].u){
                                            int cnt = 0;
                                            std::vector<std::pair<Operation *, int>> qs;
                                            for(auto &x : op->getResult(0).getUses()){
                                                auto opu = x.getOwner();
                                                qs.push_back({opu, x.getOperandNumber()});
                                                cnt++;
                                            }
                                            if(cnt == 1){
                                                builder.setInsertionPointAfter(op);
                                                op->setOperand(0, qsub[fir].opd);
                                                auto newop = builder.create<arith::AddFOp>(op->getLoc(),op->getOperands());
                                                auto newop1 = builder.create<arith::SubFOp>(op->getLoc(), qsub[fir].ops, newop->getResult(0));
                                                for (auto x : qs)
                                                {
                                                    if (x.first->getOperands() == newop1->getOperands())
                                                        continue;
                                                    x.first->setOperand(x.second, newop1->getResult(0));
                                                }
                                                op->erase();
                                                qsub[fir].op->erase();
                                                qsub[fir].u = false;
                                            }
                                        }else{
                                            nd.u = true;
                                            qsub[op->getResult(0)] = nd;
                                        }
                                    }
                                    
                                }else if(sec.getDefiningOp() != nullptr){
                                    if(llvm::isa<arith::SubFOp>(sec.getDefiningOp())){
                                        nd.opd = fir;
                                        nd.ops = sec;
                                        if(qsub[sec].u){
                                            
                                            std::vector<std::pair<Operation *, int>> qs;
                                            int cnt = 0;
                                            for(auto &x : op->getResult(0).getUses()){
                                                auto opu = x.getOwner();
                                                qs.push_back({opu, x.getOperandNumber()});
                                                cnt++;
                                            }
                                            if(cnt == 1){
                                                builder.setInsertionPointAfter(op);
                                                op->setOperand(1, qsub[sec].opd);
                                                auto newop = builder.create<arith::AddFOp>(op->getLoc(),op->getOperands());
                                                auto newop1 = builder.create<arith::SubFOp>(op->getLoc(), qsub[sec].ops, newop->getResult(0));
                                                for (auto x : qs)
                                                {
                                                    if (x.first->getOperands() == newop1->getOperands())
                                                        continue;
                                                    x.first->setOperand(x.second, newop1->getResult(0));
                                                }
                                                op->erase();
                                                qsub[sec].op->erase();
                                                qsub[sec].u = false;
                                            }
                                        }else{
                                            nd.u = true;
                                            qsub[op->getResult(0)] = nd;
                                        }
                                    }
                                }
                            }else if(auto divf = llvm::dyn_cast<arith::DivFOp>(op)){
                                auto fir = divf.getOperand(0);
                                auto sec = divf.getOperand(1);
                                node nd;
                                nd.op = op;
                                if(fir.getDefiningOp() != nullptr){
                                    if(llvm::isa<arith::DivFOp>(fir.getDefiningOp())){
                                        nd.opd = sec;
                                        nd.ops = fir;
                                        if(qsub[fir].u){
                                            int cnt = 0;
                                            std::vector<std::pair<Operation *, int>> qs;
                                            for(auto &x : op->getResult(0).getUses()){
                                                auto opu = x.getOwner();
                                                qs.push_back({opu, x.getOperandNumber()});
                                                cnt++;
                                            }
                                            if(cnt == 1){
                                                builder.setInsertionPointAfter(op);
                                                op->setOperand(0, qdiv[fir].opd);
                                                auto newop = builder.create<arith::MulFOp>(op->getLoc(),op->getOperands());
                                                auto newop1 = builder.create<arith::DivFOp>(op->getLoc(), qdiv[fir].ops, newop->getResult(0));
                                                for (auto x : qs)
                                                {
                                                    if (x.first->getOperands() == newop1->getOperands())
                                                        continue;
                                                    x.first->setOperand(x.second, newop1->getResult(0));
                                                }
                                                op->erase();
                                                qdiv[fir].op->erase();
                                                qdiv[fir].u = false;
                                            }
                                        }else{
                                            nd.u = true;
                                            qdiv[op->getResult(0)] = nd;
                                        }
                                    }
                                }else if(sec.getDefiningOp() != nullptr){
                                    if(llvm::isa<arith::DivFOp>(sec.getDefiningOp())){
                                        nd.opd = fir;
                                        nd.ops = sec;
                                        if(qdiv[sec].u){
                                            int cnt = 0;
                                            
                                            std::vector<std::pair<Operation *, int>> qs;
                                            for(auto &x : op->getResult(0).getUses()){
                                                auto opu = x.getOwner();
                                                qs.push_back({opu, x.getOperandNumber()});
                                                cnt++;
                                            }
                                            if(cnt == 1){
                                                builder.setInsertionPointAfter(op);
                                                op->setOperand(1, qdiv[sec].opd);
                                                auto newop = builder.create<arith::MulFOp>(op->getLoc(),op->getOperands());
                                                auto newop1 = builder.create<arith::DivFOp>(op->getLoc(), qdiv[sec].ops, newop->getResult(0));
                                                for (auto x : qs)
                                                {
                                                    if (x.first->getOperands() == newop1->getOperands())
                                                        continue;
                                                    x.first->setOperand(x.second, newop1->getResult(0));
                                                }
                                                op->erase();
                                                qdiv[sec].op->erase();
                                                qdiv[sec].u = false;
                                            }
                                        }else{
                                            nd.u = true;
                                            qdiv[op->getResult(0)] = nd;
                                        }
                                    }
                                }
                            }else if (auto addi = llvm::dyn_cast<arith::AddIOp>(op))//整数加法
                            {
                                auto fir = addi.getOperand(0);
                                auto sec = addi.getOperand(1);
                                node nd;
                                nd.op = op;
                                if (fir.getDefiningOp() != nullptr)
                                {
                                    if (llvm::isa<arith::AddIOp>(fir.getDefiningOp()))
                                    {
                                        nd.opd = sec;
                                        nd.ops = fir;
                                        if (iadd[fir].u && iadd[fir].op->getBlock() == op->getBlock())
                                        {
                                            int cnt = 0;
                                            std::vector<std::pair<Operation *, int>> qs;
                                            for (auto &x : op->getResult(0).getUses())
                                            {
                                                auto opu = x.getOwner();
                                                qs.push_back({opu, x.getOperandNumber()});
                                                cnt++;
                                            }
                                            if(cnt == 1){
                                                builder.setInsertionPointAfter(op);
                                                op->setOperand(0, iadd[fir].opd);
                                                auto newop = builder.create<arith::AddIOp>(op->getLoc(), iadd[fir].ops, op->getResult(0));
                                                for (auto x : qs)
                                                {
                                                    if (x.first->getOperands() == newop->getOperands())
                                                        continue;
                                                    x.first->setOperand(x.second, newop->getResult(0));
                                                }
                                                iadd[fir].op->erase();
                                                iadd[fir].u = false;
                                            }
                                            
                                        }
                                        else
                                        {
                                            nd.u = true;
                                            iadd[op->getResult(0)] = nd;
                                        }
                                    }
                                }
                                if (sec.getDefiningOp() != nullptr)
                                {
                                    if (llvm::isa<arith::AddIOp>(sec.getDefiningOp()))
                                    {
                                        nd.opd = fir;
                                        nd.ops = sec;
                                        if (iadd[sec].u)
                                        {
                                            std::vector<std::pair<Operation *, int>> qs;
                                            int cnt = 0;
                                            for (auto &x : op->getResult(0).getUses())
                                            {
                                                auto opu = x.getOwner();
                                                qs.push_back({opu, x.getOperandNumber()});
                                                cnt++;
                                            }
                                            if(cnt == 1){
                                                builder.setInsertionPointAfter(op);
                                                op->setOperand(1, iadd[sec].opd);
                                                auto newop = builder.create<arith::AddIOp>(op->getLoc(), iadd[sec].ops, op->getResult(0));
                                                for (auto x : qs)
                                                {
                                                    if (x.first->getOperands() == newop->getOperands())
                                                        continue;
                                                    x.first->setOperand(x.second, newop->getResult(0));
                                                }
                                                iadd[sec].op->erase();
                                                iadd[sec].u = false;
                                            }
                                            
                                        }
                                        else
                                        {
                                            nd.u = true;
                                            iadd[op->getResult(0)] = nd;
                                        }
                                    }
                                }
                            }else if(auto subi =  llvm::dyn_cast<arith::SubIOp>(op)){ //整数减法
                                auto fir = subi.getOperand(0);
                                auto sec = subi.getOperand(1);
                                node nd;
                                nd.op = op;
                                if(fir.getDefiningOp() != nullptr){
                                    if(llvm::isa<arith::SubIOp>(fir.getDefiningOp())){
                                        
                                        nd.opd = sec;
                                        nd.ops = fir;
                                        if(isub[fir].u){
                                            int cnt = 0;
                                            std::vector<std::pair<Operation *, int>> qs;
                                            for(auto &x : op->getResult(0).getUses()){
                                                auto opu = x.getOwner();
                                                qs.push_back({opu, x.getOperandNumber()});
                                                cnt++;
                                            }
                                            if(cnt == 1){
                                                builder.setInsertionPointAfter(op);
                                                op->setOperand(0, isub[fir].opd);
                                                auto newop = builder.create<arith::AddIOp>(op->getLoc(),op->getOperands());
                                                auto newop1 = builder.create<arith::SubIOp>(op->getLoc(), isub[fir].ops, newop->getResult(0));
                                                for (auto x : qs)
                                                {
                                                    if (x.first->getOperands() == newop1->getOperands())
                                                        continue;
                                                    x.first->setOperand(x.second, newop1->getResult(0));
                                                }
                                                op->erase();
                                                isub[fir].op->erase();
                                                isub[fir].u = false;
                                            }
                                        }else{
                                            nd.u = true;
                                            isub[op->getResult(0)] = nd;
                                        }
                                    }
                                    
                                }else if(sec.getDefiningOp() != nullptr){
                                    if(llvm::isa<arith::SubIOp>(sec.getDefiningOp())){
                                        nd.opd = fir;
                                        nd.ops = sec;
                                        if(isub[sec].u){
                                            
                                            std::vector<std::pair<Operation *, int>> qs;
                                            int cnt = 0;
                                            for(auto &x : op->getResult(0).getUses()){
                                                auto opu = x.getOwner();
                                                qs.push_back({opu, x.getOperandNumber()});
                                                cnt++;
                                            }
                                            if(cnt == 1){
                                                builder.setInsertionPointAfter(op);
                                                op->setOperand(1, isub[sec].opd);
                                                auto newop = builder.create<arith::AddIOp>(op->getLoc(),op->getOperands());
                                                auto newop1 = builder.create<arith::SubIOp>(op->getLoc(), isub[sec].ops, newop->getResult(0));
                                                for (auto x : qs)
                                                {
                                                    if (x.first->getOperands() == newop1->getOperands())
                                                        continue;
                                                    x.first->setOperand(x.second, newop1->getResult(0));
                                                }
                                                op->erase();
                                                isub[sec].op->erase();
                                                isub[sec].u = false;
                                            }
                                        }else{
                                            nd.u = true;
                                            isub[op->getResult(0)] = nd;
                                        }
                                    }
                                }
                            }else if (auto muliOp = llvm::dyn_cast<arith::MulIOp>(op))  //单精度乘法
                            {
                                auto fir = muliOp.getOperand(0);
                                auto sec = muliOp.getOperand(1);
                                node nd;
                                nd.op = op;
                                if (fir.getDefiningOp() != nullptr)
                                {
                                    if (llvm::isa<arith::MulIOp>(fir.getDefiningOp()))
                                    {
                                        nd.opd = sec;
                                        nd.ops = fir;
                                        if (imul[fir].u && imul[fir].op->getBlock() == op->getBlock())
                                        {
                                            
                                            std::vector<std::pair<Operation *, int>> qs;
                                            int cnt = 0;
                                            for (auto &x : op->getResult(0).getUses())
                                            {
                                                auto opu = x.getOwner();
                                                qs.push_back({opu, x.getOperandNumber()});
                                                cnt++;
                                            }
                                            if(cnt == 1){
                                                builder.setInsertionPointAfter(op);
                                                op->setOperand(0, imul[fir].opd);
                                                auto newop = builder.create<arith::MulIOp>(op->getLoc(), imul[fir].ops, op->getResult(0));
                                                for (auto x : qs)
                                                {
                                                    if (x.first->getOperands() == newop->getOperands())
                                                        continue;
                                                    x.first->setOperand(x.second, newop->getResult(0));
                                                }
                                                imul[fir].op->erase();
                                                imul[fir].u = false;
                                            }
                                            
                                        }
                                        else
                                        {
                                            nd.u = true;
                                            imul[op->getResult(0)] = nd;
                                        }
                                    }
                                }
                                if (sec.getDefiningOp() != nullptr)
                                {
                                    if (llvm::isa<arith::MulIOp>(sec.getDefiningOp()))
                                    {
                                        nd.opd = fir;
                                        nd.ops = sec;
                                        if (imul[sec].u)
                                        {
                                            std::vector<std::pair<Operation *, int>> qs;
                                            int cnt = 0;
                                            for (auto &x : op->getResult(0).getUses())
                                            {
                                                auto opu = x.getOwner();
                                                qs.push_back({opu, x.getOperandNumber()});
                                                cnt++;
                                            }
                                            if(cnt == 1){
                                                builder.setInsertionPointAfter(op);
                                                op->setOperand(1, imul[sec].opd);
                                                auto newop = builder.create<arith::MulIOp>(op->getLoc(), imul[sec].ops, op->getResult(0));
                                                for (auto x : qs)
                                                {
                                                    if (x.first->getOperands() == newop->getOperands())
                                                        continue;
                                                    x.first->setOperand(x.second, newop->getResult(0));
                                                } 
                                                imul[sec].op->erase();
                                                imul[sec].u = false;
                                            }
                                            
                                        }
                                        else
                                        {
                                            nd.u = true;
                                            imul[op->getResult(0)] = nd;
                                        }
                                    }
                                }
                            }else if(auto divi = llvm::dyn_cast<arith::DivUIOp>(op)){
                                auto fir = divi.getOperand(0);
                                auto sec = divi.getOperand(1);
                                node nd;
                                nd.op = op;
                                if(fir.getDefiningOp() != nullptr){
                                    if(llvm::isa<arith::DivUIOp>(fir.getDefiningOp())){
                                        nd.opd = sec;
                                        nd.ops = fir;
                                        if(qsub[fir].u){
                                            int cnt = 0;
                                            std::vector<std::pair<Operation *, int>> qs;
                                            for(auto &x : op->getResult(0).getUses()){
                                                auto opu = x.getOwner();
                                                qs.push_back({opu, x.getOperandNumber()});
                                                cnt++;
                                            }
                                            if(cnt == 1){
                                                builder.setInsertionPointAfter(op);
                                                op->setOperand(0, idiv[fir].opd);
                                                auto newop = builder.create<arith::MulIOp>(op->getLoc(),op->getOperands());
                                                auto newop1 = builder.create<arith::DivUIOp>(op->getLoc(), idiv[fir].ops, newop->getResult(0));
                                                for (auto x : qs)
                                                {
                                                    if (x.first->getOperands() == newop1->getOperands())
                                                        continue;
                                                    x.first->setOperand(x.second, newop1->getResult(0));
                                                }
                                                op->erase();
                                                idiv[fir].op->erase();
                                                idiv[fir].u = false;
                                            }
                                        }else{
                                            nd.u = true;
                                            idiv[op->getResult(0)] = nd;
                                        }
                                    }
                                }else if(sec.getDefiningOp() != nullptr){
                                    if(llvm::isa<arith::DivUIOp>(sec.getDefiningOp())){
                                        nd.opd = fir;
                                        nd.ops = sec;
                                        if(idiv[sec].u){
                                            int cnt = 0;
                                            
                                            std::vector<std::pair<Operation *, int>> qs;
                                            for(auto &x : op->getResult(0).getUses()){
                                                auto opu = x.getOwner();
                                                qs.push_back({opu, x.getOperandNumber()});
                                                cnt++;
                                            }
                                            if(cnt == 1){
                                                builder.setInsertionPointAfter(op);
                                                op->setOperand(1, idiv[sec].opd);
                                                auto newop = builder.create<arith::MulFOp>(op->getLoc(),op->getOperands());
                                                auto newop1 = builder.create<arith::DivUIOp>(op->getLoc(), idiv[sec].ops, newop->getResult(0));
                                                for (auto x : qs)
                                                {
                                                    if (x.first->getOperands() == newop1->getOperands())
                                                        continue;
                                                    x.first->setOperand(x.second, newop1->getResult(0));
                                                }
                                                op->erase();
                                                idiv[sec].op->erase();
                                                idiv[sec].u = false;
                                            }
                                        }else{
                                            nd.u = true;
                                            idiv[op->getResult(0)] = nd;
                                        }
                                    }
                                }
                            }
                            else if (llvm::isa<scf::ForOp>(op))
                            {
                                qmul.clear();
                                qadd.clear();
                                qsub.clear();
                                qdiv.clear();
                                iadd.clear();
                                isub.clear();
                                imul.clear();
                                idiv.clear();
                            } });

                for (int i = 0; i < 5; i++)
                {
                    Operandbalance<arith::MulIOp>(funcOp);
                    Operandbalance<arith::DivUIOp>(funcOp);
                    Operandbalance<arith::AddIOp>(funcOp);
                    Operandbalance<arith::SubIOp>(funcOp);
                }
                // 浮点型操作过多会更容易精度丢失
                Operandbalance<arith::MulFOp>(funcOp);
                Operandbalance<arith::AddFOp>(funcOp);
                Operandbalance<arith::SubFOp>(funcOp);
                Operandbalance<arith::DivFOp>(funcOp);
            }
            funcOp->removeAttr("balance");
        }
    };
}

namespace mlir
{
    std::unique_ptr<OperationPass<mlir::func::FuncOp>> createExpressionBalancePass()
    {
        return std::make_unique<ExpressionBalancePass>();
    }
}