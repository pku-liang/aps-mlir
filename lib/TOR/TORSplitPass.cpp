#include "TOR/PassDetail.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "TOR/TOR.h"
#include "TOR/TORDialect.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include <mlir/Transforms/DialectConversion.h>
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <map>
#include <set>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <functional>

#define DEBUG_TYPE "split-schedule"
using std::string;
using namespace mlir::arith;
namespace mlir {
    namespace split {
#define TIME_NODE 100000
        int attr_num = 0;

        string get_new_attr() {
            return "new_" + std::to_string(attr_num++);
        }

        string get_call_attr() {
            return "call_" + std::to_string(attr_num++);
        }

        struct TimeEdge {
            int from;
            int to;
            bool loop_exit;
            bool Static;
            bool valid;
            bool pipeline;
            Attribute attr;
            std::vector<Operation *> ops;

            TimeEdge(int _from, int _to, Attribute _attr, bool _static = true, bool _exit = false, bool _pipe = false) {
                from = _from;
                to = _to;
                attr = _attr;
                loop_exit = _exit;
                ops.clear();
                Static = _static;
                valid = true;
                pipeline = _pipe;
            }
        };

        std::vector<TimeEdge> timeGraph[TIME_NODE];
        int ifEnd[TIME_NODE], ifBegin[TIME_NODE];
        Operation *ifOp[TIME_NODE];
        int whileEnd[TIME_NODE], whileBegin[TIME_NODE];
        Operation *whileOp[TIME_NODE];
        int forEnd[TIME_NODE], forBegin[TIME_NODE];
        Operation *forOp[TIME_NODE];
        Operation *succOp[TIME_NODE];

        bool connected(int x, int end) {
            if (x == end)
                return true;
            for (auto &edge: timeGraph[x]) {
                if (edge.Static && edge.valid) {
                    if (connected(edge.to, end))
                        return true;
                }
            }
            return false;
        };

        bool strongConnected(int x, int end) {
            if (x == end)
                return true;
            for (auto &edge: timeGraph[x]) {
                if (edge.Static && edge.valid) {
                    if (!strongConnected(edge.to, end))
                        return false;
                } else return false;
            }
            return true;
        };

        void bind_operation(int src, int dest, Operation *op) {
            for (auto &edge: timeGraph[src]) {
                if (edge.to == dest || connected(edge.to, dest)) {
                    edge.ops.push_back(op);
                }
            }
        }

        struct SplitSchedule : public OpRewritePattern<tor::FuncOp> {
            SplitSchedule(MLIRContext *context) : OpRewritePattern<tor::FuncOp>(context, 1) {}

            LogicalResult
            matchAndRewrite(tor::FuncOp funcOp, PatternRewriter &rewriter) const override {
                if (funcOp->hasAttr("strategy")) {
                    if (auto str = dyn_cast<StringAttr>(funcOp->getAttr("strategy"))) {
                        if (str.getValue() == "mixed") {
                        } else {
                            return failure();
                        }
                    }
                } else {
                    funcOp->setAttr("strategy", StringAttr::get(getContext(), "mixed"));
                }
                for (int i = 0; i < TIME_NODE; i++) {
                    timeGraph[i].clear();
                }
                uint32_t MAX_INDEX = 0;
                uint32_t START, END;
                tor::TimeGraphOp timeGraphOp;
                for (auto &block: funcOp) {
                    for (auto &op: block) {
                        if (auto timegraph = dyn_cast<tor::TimeGraphOp>(op)) {
                            timeGraphOp = timegraph;
                        }
                    }
                }

                bool foundStatic = false;
                bool foundDynamic = false;
                bool foundPipeline = false;
                START = timeGraphOp.getStarttime();
                END = timeGraphOp.getEndtime();
                MAX_INDEX = std::max(START, END);
                for (auto &block: timeGraphOp.getRegion()) {
                    for (auto &op: block) {
                        // op.dump();
                        if (auto succ = dyn_cast<tor::SuccTimeOp>(op)) {
                            MAX_INDEX = std::max(MAX_INDEX, succ.getTime());
                            for (unsigned i = 0; i < succ.getPoints().size(); i++) {
                                auto from = succ.getPoints()[i];
                                auto comp_edge = dyn_cast<DictionaryAttr>(succ.getEdges()[i]);
                                bool pipeline = comp_edge.get("pipeline").operator bool();

                                foundPipeline |= pipeline;
                                auto edge_info = comp_edge.get("type");
                                int index = dyn_cast<IntegerAttr>(from).getInt();
                                auto info = dyn_cast<StringAttr>(edge_info).getValue().str();
                                if (info.find("dynamic") != StringRef::npos) {
                                    foundDynamic = true;
                                    timeGraph[index].push_back(
                                            TimeEdge(index, succ.getTime(), succ.getEdges()[i], false,
                                                     info.find("for") != StringRef::npos ||
                                                     info.find("while") != StringRef::npos, pipeline));
                                } else if (info.find("static") != StringRef::npos) {
                                    foundStatic = true;
                                    timeGraph[index].push_back(
                                            TimeEdge(index, succ.getTime(), succ.getEdges()[i], true,
                                                     info.find("for") != StringRef::npos ||
                                                     info.find("while") != StringRef::npos, pipeline));
                                } else {
                                    edge_info.dump();
                                    assert("Unexpected edge_info attribute" && false);
                                }
                                LLVM_DEBUG(llvm::dbgs() << "???" << succ.getTime() << "\n");
                                succOp[succ.getTime()] = &op;
                            }
                        }
                    }
                }
                if (funcOp->hasAttr("dataflow")) {
                    funcOp->setAttr("strategy", StringAttr::get(getContext(), "dataflow"));
                } else {
                    if (!foundStatic && !foundPipeline) {
                        funcOp->setAttr("strategy", StringAttr::get(getContext(), "dynamic"));
                        return failure();
                    }
                    if (!foundDynamic && !foundPipeline) {
                        funcOp->setAttr("strategy", StringAttr::get(getContext(), "static"));
                        return failure();
                    }
                    if (foundDynamic) {
                        funcOp->setAttr("strategy", StringAttr::get(getContext(), "dynamic"));
                    } else {
                        funcOp->setAttr("strategy", StringAttr::get(getContext(), "static"));
                    }
                }

                memset(ifEnd, -1, sizeof(ifEnd));
                memset(ifBegin, -1, sizeof(ifBegin));
                memset(whileEnd, -1, sizeof(whileEnd));
                memset(forEnd, -1, sizeof(forEnd));
                memset(whileBegin, -1, sizeof(whileBegin));
                memset(forBegin, -1, sizeof(forBegin));
                funcOp.walk([&](Operation *op) {
#define BIND(OpType)                   \
if (auto sop = dyn_cast<OpType>(op)) \
bind_operation(sop.getStarttime(), sop.getEndtime(), op);
                    BIND(tor::AddIOp)
                    BIND(tor::SubIOp)
                    BIND(tor::MulIConstOp)
                    BIND(tor::MulIOp)
                    BIND(tor::CmpIOp)
                    BIND(tor::AddFOp)
                    BIND(tor::SubFOp)
                    BIND(tor::MulFOp)
                    BIND(tor::CmpFOp)
                    BIND(tor::LoadOp)
                    BIND(tor::StoreOp)

#undef BIND
                    if (auto ifop = dyn_cast<tor::IfOp>(op)) {
                        ifEnd[ifop.getStarttime()] = ifop.getEndtime();
                        ifBegin[ifop.getEndtime()] = ifop.getStarttime();
                        ifOp[ifop.getStarttime()] = op;
                    } else if (auto whileop = dyn_cast<tor::WhileOp>(op)) {
                        whileEnd[whileop.getStarttime()] = whileop.getEndtime();
                        whileBegin[whileop.getEndtime()] = whileop.getStarttime();
                        whileOp[whileop.getStarttime()] = op;
                    } else if (auto forop = dyn_cast<tor::ForOp>(op)) {
                        forEnd[forop.getStarttime()] = forop.getEndtime();
                        forBegin[forop.getEndtime()] = forop.getStarttime();
                        forOp[forop.getStarttime()] = op;
                    }
                });
                for (uint32_t i = 0; i < MAX_INDEX; i++) {
                    LLVM_DEBUG(llvm::dbgs() << i << ": " << "\n");
                    for (auto &edge: timeGraph[i]) {
                        LLVM_DEBUG(llvm::dbgs() << "<" << edge.to << "," << edge.Static << "\n");
                        LLVM_DEBUG(llvm::dbgs() << ">; " << "\n");
                    }
                }

                int visitEnd[TIME_NODE], visitEndCount = 0;
                memset(visitEnd, 0, sizeof(visitEnd));
                std::vector<TimeEdge *> eraseEdges;

                int visitMarkCount = 0;
                std::set<std::pair<int, int>> visitMark;
                int nodeMark[TIME_NODE];
                memset(nodeMark, 0, sizeof(nodeMark));
                std::function<void(int, int)> markStatic = [&](int start, int end) {
                    LLVM_DEBUG(llvm::dbgs() << "!!!" << start << " , " << end << "\n");
                    if (visitMark.find(std::make_pair(start, end)) != visitMark.end()) {
                        return;
                    }
                    visitMark.insert(std::make_pair(start, end));
                    nodeMark[start] = visitMarkCount;
                    if (forEnd[start] != -1 && !connected(end, forEnd[start])) {
                        markStatic(start, forEnd[start]);
                        TimeEdge *edge = &(timeGraph[start][0]);
                        if (!edge->loop_exit) {
                            edge = &(timeGraph[start][1]);
                        }
                        if (edge->Static && edge->valid) {
                            eraseEdges.push_back(edge);
                            markStatic(edge->to, end);
                        }
                        return;
                    }
                    if (start != end) {
                        for (auto &edge: timeGraph[start]) {
                            if (edge.Static && edge.valid && connected(edge.to, end)) {
                                eraseEdges.push_back(&edge);
                                markStatic(edge.to, end);
                            }
                        }
                    }
                };
                std::set<Operation *> outline_ops;
                Operation *for_op;
                auto outline = [&](int start, int end, bool pipe = false) {
                    outline_ops.clear();
                    std::function<void(int, std::function<void(Operation *op)>)> visitOperation =
                            [&](int start, std::function<void(Operation *op)> insert) {
                                if (start == end) return;
                                LLVM_DEBUG(llvm::dbgs() << "~~" << start << " " << end << "\n");
                                if (ifEnd[start] != -1 && strongConnected(start, ifEnd[start])) {
                                    insert(ifOp[start]);
                                    visitOperation(ifEnd[start], insert);
                                    return;
                                }
                                if (forEnd[start] != -1 && (end == forEnd[start] || !connected(end, forEnd[start]))) {
                                    insert(forOp[start]);
                                    auto edge = timeGraph[start][0];
                                    if (!edge.loop_exit) {
                                        edge = timeGraph[start][1];
                                    }
                                    visitOperation(edge.to, insert);
                                    return;
                                }
                                for (auto &edge: timeGraph[start]) {
                                    if (edge.Static && edge.valid && connected(edge.to, end)) {
                                        for (auto &op: edge.ops) {
                                            insert(op);
                                        }
                                        visitOperation(edge.to, insert);
                                    }
                                }
                            };
                    //FIXME : Consider memory operations
                    LLVM_DEBUG(llvm::dbgs() << "OUTLINE" << start << "," << end << "\n");
                    visitOperation(start, [&](Operation *op) {
                        op->walk([&](Operation *op) {
                            outline_ops.insert(op);
                        });
                    });
                    struct Liveness {
                        Value val;

                        Liveness(const Value &_val) {
                            val = _val;
                        }

                        bool operator<(const Liveness &x) const {
                            if (val == x.val) {
                                return false;
                            }
                            return val.getImpl() < x.val.getImpl();

                        }

                        bool operator==(const Liveness &x) const {
                            return val == x.val;
                        }
                    };
                    std::set<Liveness> liveins, liveouts;
                    for (auto op: outline_ops) {
                        // op->dump();
                        for (auto val: op->getResults()) {
                            for (auto it = val.getUses().begin(); it != val.getUses().end(); ++it) {
                                auto bop = it.getUser();
                                if (outline_ops.find(bop) == outline_ops.end()) {
                                    liveouts.insert(val);
                                    LLVM_DEBUG(llvm::dbgs() << "FOUND OUT : " << "\n");
                                    // bop->dump();
                                }
                            }
                        }
                        for (unsigned idx = 0; idx != op->getNumOperands(); ++idx) {
                            auto val = op->getOperand(idx);

                            if (auto arg = dyn_cast<BlockArgument>(val)) {
                                // arg.dump();
                                if (outline_ops.find(arg.getOwner()->getParentOp()) != outline_ops.end()) {
                                    continue;
                                }
                                LLVM_DEBUG(llvm::dbgs() << "FOUND Argument IN " << "\n");
                                // val.dump();
                                liveins.insert(val);

                            } else {
                                auto bop = val.getDefiningOp();
                                if (isa<ConstantOp, tor::AllocOp, tor::StreamCreateOp,
                                    tor::AXICreateOp>(bop)) {
                                    continue;
                                }
                                if (outline_ops.find(bop) == outline_ops.end()) {
                                    LLVM_DEBUG(llvm::dbgs() << "FOUND IN ");
                                    // bop->dump();
                                    LLVM_DEBUG(llvm::dbgs() << bop->getNumResults() << "???" << "\n");
                                    for (auto bval: bop->getResults()) {
                                        // bval.dump();
                                        if (bval == val) {
                                            liveins.insert(val);
                                            LLVM_DEBUG(llvm::dbgs() << bval.getResultNumber() << ":");
                                        }
                                    }
                                    // bop->dump();
                                }
                            }
                        }
                    }
                    llvm::SmallVector<mlir::Type, 8> argTypes;
                    llvm::SmallVector<mlir::Type, 8> retTypes;
                    llvm::SmallVector<mlir::Value, 8> argValues;
                    llvm::SmallVector<mlir::Value, 8> retValues;
                    std::map<Liveness, int> argNum;
                    std::map<Liveness, int> retNum;

                    LLVM_DEBUG(llvm::dbgs() << "---------IN---------" << "\n");
                    unsigned arg_count = 0, ret_count = 0;
                    if (pipe) {
                        auto forOp = cast<tor::ForOp>(for_op);
                        auto insertArg = [&](Liveness val) {
                            // val.val.dump();
                            argTypes.push_back(val.val.getType());
                            argValues.push_back(val.val);
                            argNum[val] = arg_count++;
                        };
                        liveins.insert(forOp.getLowerBound());
                        liveins.insert(forOp.getUpperBound());
                        liveins.insert(forOp.getStep());
                        insertArg(Liveness(forOp.getLowerBound()));
                        insertArg(Liveness(forOp.getUpperBound()));
                        insertArg(Liveness(forOp.getStep()));
                        for (auto val: liveins) {
                            if (val == forOp.getLowerBound())continue;
                            if (val == forOp.getUpperBound())continue;
                            if (val == forOp.getStep())continue;
                            if (llvm::isa<tor::RequestType>(val.val.getType()))continue;
                            LLVM_DEBUG(llvm::dbgs() << "Result :" << "\n");
                            insertArg(val);
                        }
                        for (auto val: liveins) {
                            if (!llvm::isa<tor::RequestType>(val.val.getType()))
                                continue;
                            insertArg(val);
                        }
                    } else {
                        for (auto val: liveins) {
                            LLVM_DEBUG(llvm::dbgs() << "Result :" << "\n");
                            // val.val.dump();
                            argTypes.push_back(val.val.getType());
                            argValues.push_back(val.val);
                            argNum[val] = arg_count++;
                        }
                    }

                    LLVM_DEBUG(llvm::dbgs() << "---------OUT---------" << "\n");
                    for (auto val: liveouts) {
                        LLVM_DEBUG(llvm::dbgs() << "Result :" << "\n");
                        // val.val.dump();
                        retTypes.push_back(val.val.getType());
                        retNum[val] = ret_count++;
                    }
                    retValues.resize(ret_count);

                    LLVM_DEBUG(llvm::dbgs() << "---------DONE---------" << "\n");

                    rewriter.setInsertionPoint(funcOp);
                    auto func_type = rewriter.getFunctionType(argTypes, retTypes);
                    static int func_num = 0;
                    std::string func_name = "outline_" + std::to_string(func_num++);
                    auto out_func = rewriter.create<tor::FuncOp>(funcOp.getLoc(), func_name, func_type);
                    out_func->setAttr("dump", StringAttr::get(getContext(), get_new_attr().c_str()));
                    out_func->setAttr("strategy", StringAttr::get(getContext(), "static"));
                    SmallVector<Location, 8> locations(argTypes.size(), funcOp.getLoc());
                    rewriter.createBlock(&(out_func.getBody()), {}, argTypes, locations);

                    LLVM_DEBUG(llvm::dbgs() << "---------NEW FUNC---------" << "\n");

                    Operation *lastOp = NULL;

                    visitOperation(start, [&](Operation *op) {
                        if (!lastOp) {
                            lastOp = op;
                            return;
                        }
                        if (lastOp->getBlock() == op->getBlock()) {
                            if (!lastOp->isBeforeInBlock(op)) {
                                lastOp = op;
                            }
                        } else {
                            if (!lastOp->isBeforeInBlock(lastOp->getBlock()->findAncestorOpInBlock(*op))) {
                                lastOp = op;
                            }
                        }
                    });
                    rewriter.setInsertionPointAfter(lastOp);
                    auto callOp = rewriter.create<tor::CallOp>(lastOp->getLoc(), retTypes, func_name, start, end,
                                                               ValueRange(argValues));
                    callOp->setAttr("dump", StringAttr::get(getContext(), get_call_attr().c_str()));

                    LLVM_DEBUG(llvm::dbgs() << "---------NEW CALL---------" << "\n");

                    rewriter.setInsertionPointToEnd(&(out_func.getBody().front()));
                    visitOperation(start, [&](Operation *op) {
                        auto newOp = rewriter.clone(*op);

                        for (auto val: op->getResults()) {
                            if (liveouts.find(val) != liveouts.end()) {
                                retValues[retNum[val]] = newOp->getResult(val.getResultNumber());
                                int retnum = retNum[val];
                                LLVM_DEBUG(
                                        llvm::dbgs() << "---------REPLACE---------" << val.getResultNumber() << "\n");
                                // val.dump();
                                std::vector<OpOperand *> use_vec;
                                for (auto &use: val.getUses()) {
                                    use_vec.push_back(&use);
                                }
                                for (auto &use: use_vec) {
                                    auto bop = use->getOwner();
                                    val.replaceUsesWithIf(callOp.getResult(retnum), [&](OpOperand &operand) {
                                        return operand.getOwner() == bop;
                                    });
                                    // bop->dump();
                                }
                                LLVM_DEBUG(llvm::dbgs() << "-------------------------" << "\n");
                            } else {
                                val.replaceAllUsesWith(newOp->getResult(val.getResultNumber()));
                            }
                        }

                        rewriter.eraseOp(op);
                        LLVM_DEBUG(llvm::dbgs() << "------ERASE OP-------" << "\n");
                    });

                    LLVM_DEBUG(llvm::dbgs() << "------GET RET-------" << "\n");

                    out_func.walk([&](Operation *op) {
                        for (unsigned idx = 0; idx != op->getNumOperands(); ++idx) {
                            auto val = op->getOperand(idx);
                            auto bop = val.getDefiningOp();

                            if (liveins.find(val) != liveins.end()) {
                                op->setOperand(idx, out_func.getArgument(argNum[val]));
                            }
                            continue;
                            if (!bop) {
                                continue;
                            }
                            for (auto bval: bop->getResults()) {
                                if (bval == val) {
                                    if (liveins.find(bval) != liveins.end()) {
                                        op->setOperand(idx, out_func.getArgument(
                                                argNum[bval]));
                                    }
                                }
                            }
                        }
                    });
                    rewriter.create<tor::ReturnOp>(out_func.getLoc(), ValueRange(retValues));

                    LLVM_DEBUG(llvm::dbgs() << "------RETURN------" << "\n");

                    rewriter.setInsertionPointToStart(&(out_func.getBody().front()));
                    auto new_timegraph = rewriter.create<tor::TimeGraphOp>(out_func.getLoc(), start, end);
                    rewriter.createBlock(&(new_timegraph.getRegion()));
                    rewriter.setInsertionPointToStart(new_timegraph.getBody());
                    eraseEdges.clear();
                    ++visitMarkCount;
                    visitMark.clear();
                    markStatic(start, end);

                    LLVM_DEBUG(llvm::dbgs() << "------MARKED------" << "\n");

                    std::vector<std::vector<Attribute>> froms(TIME_NODE);
                    std::vector<std::vector<Attribute>> attrs(TIME_NODE);
                    auto newTimeEdge = [&](TimeEdge *edge) {
                        edge->valid = false;
                        LLVM_DEBUG(llvm::dbgs() << "newTimeEdge: " << edge->from << ";" << edge->to << ";" << "\n");
                        LLVM_DEBUG(llvm::dbgs().flush());
                        auto succ = cast<tor::SuccTimeOp>(succOp[edge->to]);
                        std::vector<Attribute> edge_array;
                        std::vector<Attribute> node_array;
                        for (size_t j = 0; j < succ.getPoints().size(); j++) {
                            if (dyn_cast<IntegerAttr>(succ.getPoints()[j]).getInt() != edge->from) {
                                edge_array.push_back(succ.getEdges()[j]);
                                node_array.push_back(succ.getPoints()[j]);
                            }
                        }
                        if (node_array.empty()) {
                            rewriter.eraseOp(succ);
                            succOp[edge->to] = NULL;
                            return;
                        }
                        succ.setEdgesAttr(ArrayAttr::get(getContext(), edge_array));
                        succ.setPointsAttr(ArrayAttr::get(getContext(), node_array));
                    };

                    for (auto &edge: eraseEdges) {
                        froms[edge->to].push_back(IntegerAttr::get(
                                mlir::IntegerType::get(getContext(), 32, mlir::IntegerType::Signless),
                                edge->from));
                        attrs[edge->to].push_back(edge->attr);
                    }
                    for (int node = 0; node != TIME_NODE; ++node) {
                        if (node != start && nodeMark[node] == visitMarkCount) {
                            rewriter.create<tor::SuccTimeOp>(out_func.getLoc(), node,
                                                             ArrayAttr::get(getContext(), froms[node]),
                                                             ArrayAttr::get(getContext(), attrs[node]));
                        }
                    }
                    rewriter.create<tor::FinishOp>(out_func.getLoc());
                    std::vector<Attribute> new_from, new_attr;
                    new_from.push_back(IntegerAttr::get(
                            mlir::IntegerType::get(getContext(), 32, mlir::IntegerType::Signless),
                            start));
                    llvm::SmallVector<NamedAttribute, 8> dict_attr;
                    dict_attr.push_back(NamedAttribute(StringAttr::get(getContext(), "type"),
                                                       StringAttr::get(getContext(), "static-call")));
                    new_attr.push_back(DictionaryAttr::get(getContext(), dict_attr));
                    rewriter.setInsertionPoint(succOp[end]);
                    auto newSuccOp = rewriter.create<tor::SuccTimeOp>(succOp[end]->getLoc(), end,
                                                                      ArrayAttr::get(getContext(), new_from),
                                                                      ArrayAttr::get(getContext(), new_attr));
                    for (auto &edge: eraseEdges) {
                        newTimeEdge(edge);
                    }
                    succOp[end] = newSuccOp;
                    if (pipe) {
                        out_func->setAttr("pipeline", StringAttr::get(getContext(), "for"));
                        Operation *forOp;
                        for (auto &sop: out_func.getBody().front()) {
                            if (isa<tor::ForOp>(sop)) {
                                forOp = &sop;
                            }
                        }
                        out_func->setAttr("II", forOp->getAttr("II"));
                    }

                    // out_func->dump();
                    // funcOp->dump();
                };
                std::function<int(int)> getEndPoint = [&](int start) {
                    ++visitEndCount;
                    int end = -1;
                    std::function<void(int)> dfs = [&](int x) {
                        if (end != -1)
                            return;
                        if (visitEnd[x] == visitEndCount)
                            return;
                        visitEnd[x] = visitEndCount;
                        bool found = false;
                        for (auto &edge: timeGraph[x]) {
                            if (edge.Static && edge.valid) {
                                dfs(edge.to);
                                found = true;
                            }
                        }
                        if (!found && x != start) {
                            end = x;
                        }
                    };
                    dfs(start);
                    return end;
                };
                int foundVisit[TIME_NODE], foundCount = 0;
                memset(foundVisit, 0, sizeof(foundVisit));
                ++foundCount;
                std::function<void(int)> foundStaticPart = [&](int start) {
                    if (foundVisit[start] == foundCount)
                        return;
                    foundVisit[start] = foundCount;
                    int count = std::count_if(timeGraph[start].begin(), timeGraph[start].end(),
                                              [&](const TimeEdge &edge) {
                                                  return !edge.loop_exit;
                                              });
                    LLVM_DEBUG(llvm::dbgs() << start << ":::" << count << "\n");
                    if (count == 1) {
                        int end = start;
                        while (1) {
                            int count = std::count_if(timeGraph[end].begin(), timeGraph[end].end(),
                                                      [&](const TimeEdge &edge) {
                                                          return !edge.loop_exit;
                                                      });
                            if (count == 0) {
                                break;
                            } else if (count == 1) {
                                TimeEdge edge = timeGraph[end][0];
                                if (edge.loop_exit) {
                                    edge = timeGraph[end][1];
                                }
                                if (edge.Static && edge.valid) {
                                    end = edge.to;
                                } else {
                                    break;
                                }
                            } else {
                                assert("FOUND BRANCH not IF" && ifEnd[end] != -1);
                                if (strongConnected(end, ifEnd[end])) {
                                    end = ifEnd[end];
                                } else break;
                            }
                        }
                        if (start != end) {
                            outline(start, end);
                            foundStaticPart(end);
                        } else {
                            foundStaticPart(timeGraph[start][0].to);
                        }
                    } else if (count) {
                        int end;
                        while (end = getEndPoint(start), end != -1) {
                            outline(start, end);
                        }
                        for (auto &edge: timeGraph[start]) {
                            foundStaticPart(edge.to);
                        }
                    } else {
                        for (auto &edge: timeGraph[start]) {
                            foundStaticPart(edge.to);
                        }
                    }
                };
                if (foundDynamic) {
                    foundStaticPart(START);
                }
                for (uint32_t start = 0; start <= MAX_INDEX; ++start) {
                    for (auto &edge: timeGraph[start]) {
                        if (edge.pipeline) {
                            for_op = forOp[start];
                            outline(start, edge.to, true);
                        }
                    }
                }
                bool timegraphErased = false;
                if (funcOp->hasAttr("dataflow") && timeGraphOp.getEndtime() != timeGraphOp.getStarttime()) {
                    rewriter.setInsertionPoint(funcOp);
                    auto out_func = rewriter.create<tor::FuncOp>(funcOp.getLoc(), "outline_dataflow",
                                                                 funcOp.getFunctionType());
                    out_func->setAttr("dump", mlir::StringAttr::get(getContext(), "outline_dataflow"));
                    out_func->setAttr("strategy", mlir::StringAttr::get(getContext(), "static"));
                    rewriter.createBlock(&out_func.getBody());
                    rewriter.cloneRegionBefore(funcOp.getBody(), &out_func.getBody().back());
                    rewriter.eraseBlock(&out_func.getBody().back());
                    for (auto &op: llvm::make_early_inc_range(out_func.getBody().front())) {
                        if (isa<tor::CallOp>(op)) {
                            rewriter.eraseOp(&op);
                        }
                    }
                    for (auto &op: llvm::make_early_inc_range(funcOp.getBody().front())) {
                        if (!isa<tor::CallOp, tor::TimeGraphOp, tor::ReturnOp>(op)) {
                            rewriter.eraseOp(&op);
                        }
                        if (isa<tor::ReturnOp>(op)) {
                            static int cnt = 0;
                            op.setAttr("dump",
                                       mlir::StringAttr::get(getContext(), "new_return_" + std::to_string(cnt++)));
                        }
                    }
                    rewriter.setInsertionPointAfter(timeGraphOp);
                    rewriter.eraseOp(timeGraphOp);
                    timegraphErased = true;
                    bool emptyOutFunc = true;
                    for (auto &op: out_func.getBody().front()) {
                        if (!isa<tor::TimeGraphOp, tor::ReturnOp>(op)) {
                            emptyOutFunc = false;
                            break;
                        }
                    }
                    if (!emptyOutFunc) {
                        auto call = rewriter.create<tor::CallOp>(funcOp.getLoc(), funcOp.getResultTypes(),
                                                                "outline_dataflow", 0, 0, funcOp.getArguments());
                        call->setAttr("dump", mlir::StringAttr::get(getContext(), "outline_dataflow_call"));
                    } else {
                        rewriter.eraseOp(out_func);
                    }
                }

                // funcOp->getParentOp()->dump();
                if (timegraphErased) {
                    mlir::Location loc = funcOp.getBody().getLoc();
                    rewriter.setInsertionPointToStart(&funcOp.getBody().front());
                    auto newTimeGraphOp =
                        rewriter.create<mlir::tor::TimeGraphOp>(loc, 0, 1);
                    rewriter.createBlock(&newTimeGraphOp.getBodyRegion());
                    rewriter.setInsertionPointToStart(newTimeGraphOp.getBody());
                    rewriter.create<mlir::tor::FinishOp>(loc);
                }
                return success();
            }
        };

#undef TIME_NODE
    }
    struct SplitPass : public TORSplitBase<SplitPass> {
        void runOnOperation() override {
            mlir::ModuleOp m = getOperation();
            GreedyRewriteConfig config;
            config.setStrictness(GreedyRewriteStrictness::ExistingOps);
            if (m.walk([&](tor::FuncOp op) {
                        mlir::RewritePatternSet patterns(&getContext());
                        patterns.insert<split::SplitSchedule>(op.getContext());

                        if (failed(applyOpPatternsAndFold(op.getOperation(), std::move(patterns), config)))
                            return WalkResult::advance();

                        return WalkResult::advance();
                    })
                    .wasInterrupted()) {
                return signalPassFailure();
            }
        }
    };

    std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
    createTORSplitPass() {
        return std::make_unique<SplitPass>();
    }

} // namespace mlir
