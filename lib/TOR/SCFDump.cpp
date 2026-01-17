#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/Support/Debug.h"

#include "circt/Dialect/Comb/CombOps.h"

#include "APS/APSDialect.h"
#include "APS/APSOps.h"
#include "TOR/TORDialect.h"
#include "TOR/PassDetail.h"
#include "TOR/Passes.h"

#include <set>
#include <string>
#include <iostream>
#include <fstream>
#include "nlohmann/json.hpp"

#define DEBUG_TYPE "dump-scf"

namespace {
    using namespace mlir;
    using namespace mlir::arith;
    using std::string;
    using nlohmann::json;
    namespace dump_scf {
        int attr_num = 0;
        int mem_num = 0;
        int maxi_num = 0;

        string get_opertion_attr() {
            return "op_" + std::to_string(attr_num++);
        }

        string get_mem_opertion_attr() {
            return "mem_op_" + std::to_string(mem_num++);
        }

        string get_maxi_opertion_attr() {
            return "m_axi_op_" + std::to_string(maxi_num++);
        }

        string get_type(Type type) {
            if (type.isInteger(1))
                return "bool";
            if (type.isIndex()) {
                return "i32";
            }
            std::string typeStr;
            llvm::raw_string_ostream stro(typeStr);
            type.print(stro);
            stro.flush();
            return typeStr;
        }

        string get_attr(Attribute attr) {
            if (auto int_attr = dyn_cast<IntegerAttr>(attr)) {
                return std::to_string(int_attr.getValue().getSExtValue());
            } else if (auto float_attr = dyn_cast<FloatAttr>(attr)) {
                double float_str = float_attr.getValue().convertToDouble();
                return std::to_string(float_str);
            } else if (auto bool_attr = dyn_cast<BoolAttr>(attr)) {
                return std::to_string(bool_attr.getValue());
            } else {
                attr.dump();
                assert(false && "Undefined attribute");
                return "";
            }
        }

        string get_dump(Operation *op) {
            return string(dyn_cast<mlir::StringAttr>(op->getAttr("dump")).getValue());
        }

        string get_value(Value val) {
            if (auto op_val = dyn_cast<OpResult>(val)) {
                auto op = val.getDefiningOp();
                if (op->getNumResults() == 1) {
                    return get_dump(op);
                } else {
                    return get_dump(op) + "_" + std::to_string(op_val.getResultNumber());
                }
            } else if (auto arg = dyn_cast<BlockArgument>(val)) {
                auto block = arg.getOwner();
                if (block->getNumArguments() == 1) {
                    return get_dump(block->getParentOp()) + "_b";
                } else {
                    return get_dump(block->getParentOp()) + "_" + std::to_string(arg.getArgNumber()) + "_b";
                }
            } else {
                assert(false);
                return "";
            }
        }

        json get_json(Operation *op);

        json get_json(scf::ForOp forOp) {
            json j;
            j["op_type"] = "for";
            j["names"] = json::array();
            j["lb"] = get_value(forOp.getLowerBound());
            j["ub"] = get_value(forOp.getUpperBound());
            j["step"] = get_value(forOp.getStep());
            j["iter_name"] = get_value(forOp.getInductionVar());
            j["iter_args"] = json::array();
            j["iter_inits"] = json::array();
            j["body"] = json::array();
            for (auto val: forOp.getRegionIterArgs()) {
                j["iter_inits"].push_back(get_value(val));
            }
            for (auto val: forOp.getRegionIterArgs()) {
                j["iter_args"].push_back(get_value(val));
            }
            for (auto &op: *(forOp.getBody())) {
                j["body"].push_back(get_json(&op));
            }
            for (auto val: forOp.getResults()) {
                j["names"].push_back(get_value(val));
            }
            return j;
        }

        json get_json(scf::WhileOp whileOp) {
            json j;
            j["op_type"] = "while";
            j["names"] = json::array();
            // todo
            return j;
        }

        json get_json(scf::ConditionOp conditionOp) {
            json j;
            j["op_type"] = "condition";
            j["names"] = json::array();
            // todo
            return j;
        }

        json get_json(scf::IfOp ifOp) {
            json j;
            j["op_type"] = "if";
            j["condition"] = get_value(ifOp.getCondition());
            j["names"] = json::array();
            j["body0"] = json::array();
            j["body1"] = json::array();
            for (auto val: ifOp.getResults()) {
                j["names"].push_back(get_value(val));
            }
            if (!ifOp.getThenRegion().empty()) {
                for (auto &op: ifOp.getThenRegion().front()) {
                    j["body0"].push_back(get_json(&op));
                }
            }
            if (!ifOp.getElseRegion().empty()) {
                for (auto &op: ifOp.getElseRegion().front()) {
                    j["body1"].push_back(get_json(&op));
                }
            }
            return j;
        }

#define OPERATION(TYPE, NAME)  if (auto sop = dyn_cast<TYPE>(op)) {\
        j["op_type"] = NAME;                                              \
        j["name"] = get_value(sop.getResult());                           \
        j["type"] = get_type(sop.getResult().getType());                  \
        j["operands"] = json::array();                                    \
        for (auto val : op->getOperands()) {                              \
            j["operands"].push_back(get_value(val));                      \
        }                                                                 \
        return j;                                                         \
    }

        json get_json(Operation *op) {
            json j;
            if (auto nop = dyn_cast<ConstantOp>(op)) {
                j["op_type"] = "constant";
                j["name"] = get_dump(nop);
                j["operands"] = {get_attr(nop.getValueAttr())};
                j["type"] = get_type(nop.getType());
            } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
                j = get_json(forOp);
            } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
                j = get_json(ifOp);
            } else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {  // todo
                j = get_json(whileOp);
            } else if (auto conditionOp = dyn_cast<scf::ConditionOp>(op)) {  //todo
                j = get_json(conditionOp);
            } else if (auto cmpIOp = dyn_cast<CmpIOp>(op)) {
                j["operands"] = json::array();
                j["op_type"] = string("cmp_") + stringifyCmpIPredicate(cmpIOp.getPredicate()).str();
                j["name"] = get_dump(cmpIOp);
                j["type"] = get_type(cmpIOp.getResult().getType());
                for (const auto &operand: cmpIOp->getOperands()) {
                    j["operands"].push_back(get_value(operand));
                }
                return j;
            } else if (auto callOp = dyn_cast<tor::CallOp>(op)) {
                j["operands"] = json::array();
                j["op_type"] = "call";
                j["names"] = json::array();
                j["function"] = callOp.getCallee();
                for (const auto &operand: callOp->getOperands()) {
                    j["operands"].push_back(get_value(operand));
                }
                for (const auto &val: callOp->getResults()) {
                    j["names"].push_back(get_value(val));
                }
                return j;
            } else if (auto cmpFOp = dyn_cast<CmpFOp>(op)) {
                j["operands"] = json::array();
                j["op_type"] = string("cmp_") + stringifyCmpFPredicate(cmpFOp.getPredicate()).str();
                j["name"] = get_dump(cmpFOp);
                j["type"] = get_type(cmpFOp.getResult().getType());
                for (const auto &operand: cmpFOp->getOperands()) {
                    j["operands"].push_back(get_value(operand));
                }
                return j;
            } else if (auto getGlobalOp = dyn_cast<memref::GetGlobalOp>(op)) {
                // Handle memref.get_global - treat it as loading a memory reference
                j["op_type"] = "get_global";
                j["name"] = get_dump(getGlobalOp);
                j["memory_name"] = getGlobalOp.getName().str();
                j["type"] = get_type(getGlobalOp.getResult().getType());
            } else if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
                // Handle memref.load operations
                j["op_type"] = "load";
                j["name"] = get_dump(loadOp);
                j["memory"] = get_value(loadOp.getMemref());
                j["indices"] = json::array();
                for (auto idx : loadOp.getIndices()) {
                    j["indices"].push_back(get_value(idx));
                }
                j["type"] = get_type(loadOp.getResult().getType());
            } else if (auto loadOp = dyn_cast<tor::LoadOp>(op)) {
                j["op_type"] = "load";
                j["name"] = get_dump(loadOp);
                assert(loadOp.getNumOperands() == 2);
                assert(loadOp->getNumResults() == 1);
                j["index"] = get_value(loadOp.getOperand(1));
                j["memory"] = get_value(loadOp.getMemref());
            } else if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
                j["op_type"] = "yield";
                j["operands"] = json::array();
                for (auto val: yieldOp->getOperands()) {
                    j["operands"].push_back(get_value(val));
                }
            } else if (auto returnOp = dyn_cast<tor::ReturnOp>(op)) {
                j["op_type"] = "return";
                j["operands"] = json::array();
                for (auto val: returnOp->getOperands()) {
                    j["operands"].push_back(get_value(val));
                }
            } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
                // Handle memref.store operations
                j["op_type"] = "store";
                j["memory"] = get_value(storeOp.getMemref());
                j["value"] = get_value(storeOp.getValueToStore());
                j["indices"] = json::array();
                for (auto idx : storeOp.getIndices()) {
                    j["indices"].push_back(get_value(idx));
                }
            } else if (auto storeOp = dyn_cast<tor::StoreOp>(op)) {
                assert(storeOp.getIndices().size() == 1);
                j["op_type"] = "store";
                j["index"] = get_value(storeOp.getIndices()[0]);
                j["memory"] = get_value(storeOp.getMemref());

                j["value"] = get_value(storeOp.getValue());
            } else if (auto storeOp = dyn_cast<tor::GuardedStoreOp>(op)) {
                j["op_type"] = "guardedstore";
                j["index"] = get_value(storeOp.getIndices()[0]);
                j["memory"] = get_value(storeOp.getMemref());
                j["value"] = get_value(storeOp.getValue());
                j["guard"] = get_value(storeOp.getGuard());
            } else if (isa<tor::StreamCreateOp, tor::StreamReadOp, tor::StreamWriteOp>(op)) {

            } else if (auto readRfOp = dyn_cast<aps::CpuRfRead>(op)) {
                j["op_type"] = "readrf";
                j["name"] = get_dump(readRfOp);
                j["type"] = get_type(readRfOp.getResult().getType());
                j["operands"] = json::array();
                j["operands"].push_back(get_value(readRfOp.getOperand()));
            } else if (auto writeRfOp = dyn_cast<aps::CpuRfWrite>(op)) {
                j["op_type"] = "writerf";
                j["operands"] = json::array();
                j["operands"].push_back(get_value(writeRfOp.getOperand(0)));
                j["operands"].push_back(get_value(writeRfOp.getOperand(1)));
            } else if (auto memDeclareOp = dyn_cast<aps::MemDeclare>(op)) {
                // Handle aps.memdeclare - similar to alloc
                j["op_type"] = "memdeclare";
                j["name"] = get_dump(memDeclareOp);
                j["type"] = get_type(memDeclareOp.getResult().getType());
            } else if (auto memLoadOp = dyn_cast<aps::MemLoad>(op)) {
                // Handle aps.memload
                j["op_type"] = "memload";
                j["name"] = get_dump(memLoadOp);
                j["memory"] = get_value(memLoadOp.getMemref());
                j["indices"] = json::array();
                for (auto idx : memLoadOp.getIndices()) {
                    j["indices"].push_back(get_value(idx));
                }
                j["type"] = get_type(memLoadOp.getResult().getType());
            } else if (auto memStoreOp = dyn_cast<aps::MemStore>(op)) {
                // Handle aps.memstore
                j["op_type"] = "memstore";
                j["memory"] = get_value(memStoreOp.getMemref());
                j["value"] = get_value(memStoreOp.getValue());
                j["indices"] = json::array();
                for (auto idx : memStoreOp.getIndices()) {
                    j["indices"].push_back(get_value(idx));
                }
            } else if (auto memBurstLoadOp = dyn_cast<aps::MemBurstLoad>(op)) {
                // Handle aps.memburstload - burst load from CPU to APS scratchpad
                j["op_type"] = "memburstload";
                j["cpu_addr"] = get_value(memBurstLoadOp.getCpuAddr());
                // Handle multiple memrefs (after array partitioning)
                for (auto memref : memBurstLoadOp.getMemrefs()) {
                    j["memory"].push_back(get_value(memref));
                }
                j["start"] = get_value(memBurstLoadOp.getStart());
                j["length"] = get_value(memBurstLoadOp.getLength());
            } else if (auto globalLoadOp = dyn_cast<aps::GlobalLoad>(op)) {
                // Handle aps.globalload - load a scalar global variable
                j["op_type"] = "globalload";
                j["memory"] = globalLoadOp.getGlobalName();
            } else if (auto globalStoreOp = dyn_cast<aps::GlobalStore>(op)) {
                // Handle aps.globalstore - store a scalar global variable
                j["op_type"] = "globalstore";
                j["memory"] = globalStoreOp.getGlobalName();
                j["value"] = get_value(globalStoreOp.getValue());
            } else if (auto memBurstStoreOp = dyn_cast<aps::MemBurstStore>(op)) {
                // Handle aps.memburststore - burst store from APS scratchpad to CPU
                j["op_type"] = "memburststore";
                // Handle multiple memrefs (after array partitioning)
                for (auto memref : memBurstStoreOp.getMemrefs()) {
                    j["memory"].push_back(get_value(memref));
                }
                j["start"] = get_value(memBurstStoreOp.getStart());
                j["cpu_addr"] = get_value(memBurstStoreOp.getCpuAddr());
                j["length"] = get_value(memBurstStoreOp.getLength());
            } else if (auto allocaOp = dyn_cast<memref::AllocaOp>(op)) {
                // Handle memref.alloca - stack allocation
                j["op_type"] = "alloca";
                j["name"] = get_dump(allocaOp);
                j["type"] = get_type(allocaOp.getResult().getType());
            } else if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
                // Handle memref.alloc - heap allocation
                j["op_type"] = "alloc";
                j["name"] = get_dump(allocOp);
                j["type"] = get_type(allocOp.getResult().getType());
            } else if (auto castOp = dyn_cast<memref::CastOp>(op)) {
                // Handle memref.cast - cast between memref types
                j["op_type"] = "cast";
                j["name"] = get_dump(castOp);
                j["operand"] = get_value(castOp.getSource());
                j["type"] = get_type(castOp.getResult().getType());
            } else {
                // int
                OPERATION(AddIOp, "add")
                OPERATION(MulIOp, "mul")
                OPERATION(SubIOp, "sub")
                OPERATION(RemSIOp, "remsi")
                OPERATION(DivSIOp, "divsi")
                OPERATION(RemUIOp, "remui")
                OPERATION(DivUIOp, "divui")

                OPERATION(AndIOp, "and")
                OPERATION(OrIOp, "or")
                OPERATION(XOrIOp, "xor")
                OPERATION(ShLIOp, "shift_left")
                OPERATION(ShRSIOp, "shrsi")
                OPERATION(ShRUIOp, "shrui")

                // float
                OPERATION(AddFOp, "add")
                OPERATION(MulFOp, "mul")
                OPERATION(SubFOp, "sub")
                OPERATION(DivFOp, "div")
                OPERATION(NegFOp, "negf")

                // convert
                OPERATION(SIToFPOp, "sitofp")
                OPERATION(UIToFPOp, "uitofp")
                OPERATION(FPToSIOp, "fptosi")
                OPERATION(FPToUIOp, "fptoui")
                OPERATION(ExtFOp, "extf")
                OPERATION(ExtSIOp, "extsi")
                OPERATION(ExtUIOp, "extui")
                OPERATION(TruncFOp, "truncf")
                OPERATION(TruncIOp, "trunci")
                OPERATION(IndexCastOp, "index_cast")

                // other
                OPERATION(SelectOp, "select")

                // arithmath
                OPERATION(math::AbsFOp, "absf")
                OPERATION(math::AbsIOp, "absi")
                OPERATION(math::CeilOp, "ceil")
                OPERATION(math::FloorOp, "floor")
                OPERATION(math::RoundOp, "round")

                OPERATION(math::SqrtOp, "sqrt")
                OPERATION(math::ExpOp, "exp")
                OPERATION(math::PowFOp, "powf")
                OPERATION(math::LogOp, "log")
                OPERATION(math::ErfOp, "erf")

                // trigonometric function
                OPERATION(math::CosOp, "cos")
                OPERATION(math::TanOp, "tan")
                OPERATION(math::TanhOp, "tanh")
                OPERATION(math::SinOp, "sin")

                // comb operations
                if (auto extractOp = dyn_cast<circt::comb::ExtractOp>(op)) {
                    j["op_type"] = "extract";
                    j["name"] = get_dump(extractOp);
                    j["type"] = get_type(extractOp.getResult().getType());
                    j["operands"] = json::array();
                    j["operands"].push_back(get_value(extractOp.getInput()));
                    j["from"] = extractOp.getLowBit();
                    return j;
                }

                std::cout << "Not support op " << std::string(op->getName().stripDialect().data(), op->getName().stripDialect().size()) << std::endl;

                op->dump();
                assert(false);
            }
            return j;
        }

        json get_json(tor::FuncOp funcOp) {
            json j;
            j["name"] = funcOp.getName();
            j["args"] = json::array();
            j["types"] = json::array();
            j["body"] = json::array();
            //TODO: types & args

            for (auto &op: funcOp.getBody().front()) {
                j["body"].push_back(get_json(&op));
            }

            for (auto val: funcOp.getArguments()) {
                j["args"].push_back(get_value(val));
                j["types"].push_back(get_type(val.getType()));
            }
            return j;
        }

        json get_json(tor::DesignOp designOp) {
            json j;
            j["level"] = "software";
            j["memory"] = json::array();
            j["modules"] = json::array();
            j["constants"] = json::array();

            for (auto &op: designOp.getBody().front()) {
                if (auto allocOp = dyn_cast<tor::AllocOp>(op)) {
                    json sj;
                    sj["name"] = get_dump(allocOp);
                    auto mem_type = dyn_cast<tor::MemRefType>(allocOp.getMemref().getType());
                    sj["size"] = mem_type.getShape()[0];
                    sj["type"] = get_type(mem_type.getElementType());
                    if (allocOp->hasAttr("value_h")) {
                        sj["value_h"] = string(dyn_cast<mlir::StringAttr>(allocOp->getAttr("value_h")).getValue());
                    }
                    j["memory"].push_back(sj);
                } else if (auto globalOp = dyn_cast<memref::GlobalOp>(op)) {
                    // Handle memref.global operations (preserved from ConvertInput pass)
                    json sj;
                    sj["name"] = get_dump(globalOp);
                    auto mem_type = globalOp.getType();
                    if (!mem_type.getShape().empty()) {
                        sj["size"] = mem_type.getShape()[0];
                    } else {
                        sj["size"] = 1;
                    }
                    sj["type"] = get_type(mem_type.getElementType());
                    // Handle initial value if present
                    if (globalOp.getInitialValue().has_value()) {
                        // Could add value_h handling here if needed
                    }
                    j["memory"].push_back(sj);
                } else if (isa<tor::StreamCreateOp>(op)) {

                } else if (auto funcOp = dyn_cast<tor::FuncOp>(op)) {
                    j["modules"].push_back(get_json(funcOp));
                } else if (auto nop = dyn_cast<ConstantOp>(op)) {
                    json sj;
                    sj["name"] = get_dump(nop);
                    sj["operands"] = get_attr(nop.getValueAttr());
                    sj["type"] = get_type(nop.getType());
                    j["constants"].push_back(sj);
                } else if (auto memMapOp = dyn_cast<aps::MemoryMapOp>(op)) {
                    // Handle aps.memorymap - skip it as it's metadata for memory layout
                    // The actual memref.global operations are already processed
                } else if (auto memEntryOp = dyn_cast<aps::MemEntryOp>(op)) {
                    // Handle aps.mem_entry - skip it as it's metadata for memory partitioning
                } else if (auto memFinishOp = dyn_cast<aps::MemFinishOp>(op)) {
                    // Handle aps.mem_finish - terminator for memorymap, skip it
                } else {
                    op.dump();
                    assert(false);
                }
            }
            return j;
        }

        struct SCFDumpPass : SCFDumpBase<SCFDumpPass> {
            void runOnOperation() override {
                auto designOp = getOperation();
                auto json_file = (this->json).getValue();
                designOp.walk([&](tor::AllocOp op) {
                    std::string mode = "";
                    if (op->hasAttr("mode"))
                        mode = dyn_cast<mlir::StringAttr>(op->getAttr("mode")).getValue().str();

                    if (mode != "m_axi") {
                        // not m_axi, is mem, if additional interface supported, need change here
                        op->setAttr("dump", StringAttr::get(&getContext(), get_mem_opertion_attr().c_str()));
                    } else {
                        op->setAttr("dump", StringAttr::get(&getContext(), get_maxi_opertion_attr().c_str()));
                    }
                });

                designOp.walk([&](Operation*op) {
                    if (!op->hasAttr("dump")) {
                        op->setAttr("dump", StringAttr::get(&getContext(), get_opertion_attr().c_str()));
                    }
                });

                designOp.walk([&](tor::DesignOp op) {
                    auto j = get_json(op);
                    
                    if (json_file != "") {
                        std::ofstream output_file( json_file );
                        output_file << std::setw(2) << j << std::endl;
                    }
                });
            }
        };
    }
} // namespace

namespace mlir {

    std::unique_ptr<OperationPass<tor::DesignOp>> createSCFDumpPass() {
        return std::make_unique<dump_scf::SCFDumpPass>();
    }

} // namespace mlir
