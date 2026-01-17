#include "TOR/TOR.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Visitors.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/InliningUtils.h"

#include "TOR/TORDialect.h"
#include "TOR/PassDetail.h"
#include "TOR/Passes.h"
#include "TOR/Utils.h"
#include "llvm/ADT/STLExtras.h"

#include <set>
#include <algorithm>
#include <vector>
#include <bits/stdc++.h>
#include <string>
#include <iostream>
#include <fstream>
#include "nlohmann/json.hpp"

namespace
{
    using namespace mlir;
    using namespace mlir::arith;
    using namespace mlir::func;

    struct SCFIterArgsPass : SCFIterArgsBase<SCFIterArgsPass>
    {
        llvm::DenseMap<Operation *, int> arg;
        void RemoveRepeatOp(tor::DesignOp designOp){
            std::vector<arith::ConstantOp> cop;
            designOp.walk([&](scf::ForOp forOp)
                          {
                              llvm::DenseMap<mlir::OperationName, llvm::DenseMap<std::pair<mlir::Value, mlir::Value>, Operation *>> args;
                              forOp.walk([&](Operation *op)
                                         {
                                             if (op->getNumOperands() < 2 || dyn_cast<scf::ForOp>(op)){}
                                             else {
                                                 auto nam = op->getName();
                                                 auto par = std::make_pair(op->getOperand(0), op->getOperand(1));
                                                 if (args[nam][par] != nullptr && args[nam][par]->getBlock() == op->getBlock()&&!llvm::isa<tor::LoadOp>(op))
                                                 {
                                                     std::vector<std::pair<Operation *, int>> ves;
                                                     for (auto rs : op->getResults()){
                                                         for (auto &us : rs.getUses()){
                                                             auto ow = us.getOwner();
                                                             ves.push_back({ow, us.getOperandNumber()});
                                                         }
                                                     }
                                                     for (auto x : ves){
                                                         x.first->setOperand(x.second, args[nam][par]->getResult(0));
                                                     }
                                                 }
                                                 else{
                                                     args[nam][par] = op;
                                                 }
                                             }
                                         }); 
                                         }); // 消除重复计算
        }
        void AddIterargs(tor::DesignOp designOp){
            designOp.walk([&](scf::ForOp forOp){
                std::vector<mlir::Value> newiter;
                llvm::DenseMap<Operation*,unsigned> stor;
                llvm::DenseMap<mlir::Value, llvm::DenseMap<mlir::Value,Operation*> > mt;
                llvm::DenseMap<mlir::Value, llvm::DenseMap<mlir::Value,unsigned> > cnt;
                llvm::DenseMap<mlir::Value, llvm::DenseMap<mlir::Value,unsigned> > ppl;
                llvm::DenseMap<Operation*, int> pa; //pa标记lop中op的位置，减少查找时间
                unsigned opl = 0;
                std::vector<std::pair<tor::LoadOp,int> > lop;
                std::vector<Operation*> ves;
                for(auto &x : forOp.getInductionVar().getUses()){
                    ves.push_back(x.getOwner());
                }
                auto forRegion = &forOp.getRegion().front();
                forOp.walk([&](Operation* op){
                  if(op->getBlock() != forRegion || llvm::isa<scf::IfOp>(op) || llvm::isa<scf::WhileOp>(op)){
                    ves.push_back(op);
                  }
                });
                arg[forOp] = 1;
                Operation *yieop;
                for(int i = 0;i < ves.size(); i++){
                    auto &x = ves[i];
                    arg[x] = 1;
                    for(auto re : x->getResults()){
                        for(auto &k : re.getUses()){             
                            auto ops = k.getOwner();
                            if(arg[ops] == 0){
                                ves.push_back(ops);
                                arg[ops] = 1;
                            }
                        }
                    }
                }
                forOp.walk([&](Operation* op){
                    if(auto loadop = dyn_cast<tor::LoadOp>(op)){
                        auto mem = loadop.getMemref();
                        auto pl = loadop.getOperand(1);

                        if(mt[mem][pl] != nullptr){
                            stor[op] = -1;
                        }
                        mt[mem][pl] = op;
                        cnt[mem][pl] += 1;
                        ppl[mem][pl] = opl++;
                        pa[op] = -1;
                        if(!arg[op] && stor[op] != -1&&op->getBlock() == forRegion){
                            lop.push_back({loadop,ppl[mem][pl]+10000000});
                            pa[op] = lop.size() - 1;
                        }//arg为空表示和当前循环的参数无关，stor不为-1表示当前load前面没有store
                    }else if(auto storeop = dyn_cast<tor::StoreOp>(op)){
                        auto mem = storeop.getMemref();
                        auto pl = storeop.getOperand(2);
                        cnt[mem][pl] += 1;

                        if(mt[mem][pl] != nullptr&&op->getBlock() == forRegion){
                            auto ops = mt[mem][pl];
                            if(auto loadop = dyn_cast<tor::LoadOp>(ops)){
                                if(arg[ops]){
                                }else if(loadop.getOperand(1) == pl&&stor[ops] != -1){
                                    if(pa[ops] > -1){
                                        lop[pa[ops]].second -= 10000000;
                                    }
                                    stor[op] = 1;
                                }
                            }
                        }else {
                            mt[mem][pl] = op;
                        }
                    }else if(auto yieldOp = dyn_cast<scf::YieldOp>(op)){
                      if(op->getBlock() == forRegion){
                        yieop = yieldOp;
                      }
                        
                    }
                });
               auto cmp = [&](std::pair<tor::LoadOp,int> a, std::pair<tor::LoadOp,int> b) {
                    return a.second < b.second;
                };
                sort(lop.begin(),lop.end(),cmp);
                if(!lop.empty()&&!forOp.getRegionIterArgs().empty()){
                    
                    yieop->erase();
                    OpBuilder builder(designOp.getContext());
                    llvm::DenseMap<Operation*, mlir::Value> ov;
                    std::vector<std::pair<mlir::Value,int> > Yielditer;
                    int l = 0;
                    builder.setInsertionPoint(forOp);
                    llvm::DenseMap<Operation*,int> opa;
                        
                        auto itnum = forOp.getRegionIterArgs();
                        for(auto arg : itnum){
                        newiter.push_back(arg);
                        }
                        int cnt = newiter.size();
                    for(auto it = lop.begin(); it != lop.end(); ++it){
                        auto op = it->first;
                        ves.clear();
                        pa[op.getOperation()] = cnt;
                        for(auto x : op.getOperands()){
                            auto ops = x.getDefiningOp();
                            if(ops == nullptr){
                                continue;
                            }else if(dyn_cast<tor::AllocOp>(ops) || dyn_cast<scf::ForOp>(ops)){
                                continue;
                            }else if(ops->getBlock() == op->getBlock()){
                                ves.push_back(ops);
                                opa[ops] = 1;
                            }
                        }
                        
                        for(int i = 0;i < ves.size() ;i++){
                            for(auto x : ves[i]->getOperands()){
                                auto k = x.getDefiningOp();
                                if(k == nullptr){
                                    continue;
                                }
                                if(k -> getBlock() == op->getBlock()&&opa[k] == 0){
                                    ves.push_back(k);
                                }
                            }
                        }
                        for(int i = ves.size() - 1;i >= 0;i--){
                            ves[i]->moveBefore(forOp);
                        }
                            builder.setInsertionPoint(forOp);
                            auto newOp = builder.create<tor::LoadOp>(forOp.getLoc(),op.getOperands());
                            // newOp->setAttr("dump", op->getAttr("dump"));
                            newOp.setStarttime(op.getStarttime());
                            newOp.setEndtime(op.getEndtime());
                            if(it->second < 10000000){
                                newiter.push_back(newOp.getResult());
                                cnt++;
                            }else{
                            pa[op.getOperation()] = -1;
                            ov[op.getOperation()] = newOp.getResult();
                        }
                        
                    }
                    auto newforop = builder.create<scf::ForOp>(forOp.getLoc(),forOp.getLowerBound(),
                                                            forOp.getUpperBound(),forOp.getStep(),
                                                            newiter);
                    newforop.walk([&](scf::YieldOp op){
                        op.erase();
                    });                               
                    for(auto &attr : forOp.getAttributeNames()){
                        newforop->setAttr(attr,forOp->getAttr(attr));
                    }

        addHlsAttrWithNewOp(newforop, forOp);
        newforop->setAttrs(forOp->getAttrDictionary());
                std::vector<std::pair<Operation *,std::pair<int,int> > > tes;
                    for(auto val : forOp.getRegion().front().getArguments()){
                        for(auto &use : val.getUses()){
                            auto userop = use.getOwner();
                            auto opn = use.getOperandNumber();
                            tes.push_back({userop,{opn,val.getArgNumber()}});
                        }
                    }
                    cnt = 0;
                    for(auto x : tes){
                        x.first->setOperand(x.second.first,newforop.getRegion().front().getArgument(x.second.second));  
                    }
                std::vector<std::pair<tor::StoreOp,int> > store;
                forOp.walk([&](Operation* op){ 
                    if(auto loadop = dyn_cast<tor::LoadOp>(op)){
                        auto mem = loadop.getMemref();
                        auto pl = loadop.getOperand(1);
                        if(arg[op]){
                            std::vector<std::pair<Operation *,int> > temp;
                            for(auto &xs : op->getUses()){
                                temp.push_back({xs.getOwner(),xs.getOperandNumber()});
                            }
                            for(auto &x : temp){
                                x.first->setOperand(x.second,loadop.getResult());
                            }
                        }else{
                                if(pa[op] == -1&&ov[op] == nullptr){
                                }
                                else{
                                std::vector<std::pair<Operation *,unsigned>> temp;
                                for(auto &xs : loadop->getUses()){
                                    auto xo = xs.getOwner();
                                    signed num = xs.getOperandNumber();
                                    temp.push_back({xo,num});
                                }
                                for(auto x : temp){
                                    if(pa[op] > -1)x.first->setOperand(x.second,newforop.getRegionIterArg(pa[op]));
                                    else x.first->setOperand(x.second,ov[op]);
                                }
                                l++;
                                    op->erase();
                                }
                                
                        }
                    }else if(auto storeop = dyn_cast<tor::StoreOp>(op)){
                        auto mem = storeop.getMemref();
                        auto pl = storeop.getOperand(2);
                        if(stor[op] == 0){
                        }else{
                            Yielditer.push_back({storeop.getOperand(0),ppl[mem][pl]});
                            store.push_back({storeop,ppl[mem][pl]});
                            op->moveAfter(newforop);
                        }
                    }else{
                        std::vector<std::pair<Operation *,std::pair<int,int> > > temp;
                        for(auto &xs : op->getUses()){
                            auto os = xs.getOwner();
                            auto opResult = llvm::dyn_cast<OpResult>(os->getOperand(xs.getOperandNumber()));
                            temp.push_back({os ,{xs.getOperandNumber(),opResult.getResultNumber()}});
                        }
                        for(auto &x : temp){
                            x.first->setOperand(x.second.first,op->getResult(x.second.second));
                        }
                    }
                });
                auto *sourcebody = forOp.getBody();
                auto *tar = newforop.getBody();
                tar->getOperations().splice(tar->end(),sourcebody->getOperations());
                newforop.walk([&](scf::ForOp fop){
                    std::vector<std::pair<Operation *,std::pair<mlir::BlockArgument,unsigned> > > vec;
                    for(auto &arg : fop.getRegion().front().getArguments()){
                        for(auto &aus : arg.getUses()){
                            vec.push_back({aus.getOwner(),{arg,aus.getOperandNumber()}});
                        }
                    }
                    for(auto x : vec){
                        x.first->setOperand(x.second.second,x.second.first);
                    }
                });
                    tes.clear();
                    forOp.erase();
                    builder.setInsertionPointToEnd(newforop.getBody());
                    if(Yielditer.size()){
                        auto cmpy = [&](std::pair<mlir::Value,int> a, std::pair<mlir::Value,int> b) {
                            return a.second < b.second;
                        };
                        std::sort(Yielditer.begin(),Yielditer.end(),cmpy);
                        std::vector<mlir::Value> Yielditertmp;
                        for(auto x : Yielditer){
                            Yielditertmp.push_back(x.first);
                        }
                        builder.create<scf::YieldOp>(newforop.getLoc(),Yielditertmp);
                    }
                    else{
                        builder.create<scf::YieldOp>(newforop.getLoc());
                    }
                    if(store.size()){
                        auto results = newforop.getResults();
                        l = 0;            
                        auto cmps = [&](std::pair<tor::StoreOp,int> a, std::pair<tor::StoreOp,int> b) {
                            return a.second < b.second;
                        };
                        builder.setInsertionPointAfter(newforop);
                        std::sort(store.begin(),store.end(),cmps);
                        for(int i = 0;i < store.size();i++){
                            store[i].first.setOperand(0,newforop.getResult(i));
                        }
                    }
                } });
        }
        void RemoveExtraLoad(tor::DesignOp designOp){
            designOp.walk([&](scf::ForOp forOp)
                          {
                llvm::DenseMap<mlir::Value,llvm::DenseMap<mlir::Value,Operation*> >mt;
                std::vector<Operation *> vec;
                forOp.walk([&](Operation* op){
                        if(op->getBlock() != &forOp.getRegion().front()){
                            mt.clear();
                        }else if(auto loadop = dyn_cast<tor::LoadOp>(op)){
                            auto mem = loadop.getMemref();
                            auto pl = loadop.getOperand(1);
                            if(mt[mem][pl] != nullptr){
                                std::vector<std::pair<Operation*,int> >tes;
                                for(auto x = loadop.getResult().use_begin();x != loadop.getResult().use_end();x++){
                                    tes.push_back({x->getOwner(),x->getOperandNumber()});
                                }   
                                for(auto x : tes){
                                    x.first->setOperand(x.second,mt[mem][pl]->getOperand(0));
                                }
                                loadop.erase();
                            }
                        }else if(auto storeop = dyn_cast<tor::StoreOp>(op)){
                            auto mem = storeop.getMemref();
                            auto pl = storeop.getOperand(2);
                            mt[mem][pl] = op;
                        }
                    });
                 });
        }
        void runOnOperation() override{
            auto designOp = getOperation();
            for(int i = 0;i < 2;i++){
                RemoveRepeatOp(designOp);
                AddIterargs(designOp);
                RemoveExtraLoad(designOp);
            }
        }
    };
}

namespace mlir
{
    std::unique_ptr<OperationPass<tor::DesignOp>> createSCFIterArgsPass()
    {
        return std::make_unique<SCFIterArgsPass>();
    }
}
