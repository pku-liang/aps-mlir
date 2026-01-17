#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/InliningUtils.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/Regex.h"

#include "TOR/TORDialect.h"
#include "TOR/PassDetail.h"
#include "TOR/Passes.h"

#include <set>
#include <iostream>
#include <fstream>
#include <filesystem>

#define DEBUG_TYPE "convert-input"

namespace {
    using namespace mlir;
    using namespace mlir::arith;
    using namespace mlir::func;

    template<typename SourceOp, typename TargetOp>
    void myReplaceOp(SourceOp op, TargetOp newOp, ConversionPatternRewriter &rewriter) {
        op.getResult().replaceAllUsesWith(newOp.getResult());
        rewriter.eraseOp(op);
    }

    inline void saveOldAttrWithName(mlir::Operation *newOp,
                                    mlir::Operation *op, std::string type) {
        if (auto attr = op->getAttr(type)) {
            newOp->setAttr(type, attr);
        }
    }

    inline void saveOldArgAttrWithName(mlir::Operation *newOp,
                                       mlir::Operation *op, std::string type,
                                       unsigned oldIndex) {
        if (auto attr = op->getAttr(type + "_" + llvm::Twine(oldIndex).str())) {
            newOp->setAttr(type, attr);
        }
    }

    inline void saveOldArgLineAttrWithName(mlir::Operation *newOp,
                                           mlir::Operation *op, std::string type,
                                           unsigned oldIndex) {
        if (auto attr = op->getAttr(type + "_arg_" + llvm::Twine(oldIndex).str() + "-line")) {
            newOp->setAttr(type + "-line", attr);
        }
    }

    struct FlattenArray : public OpRewritePattern<ModuleOp> {
        using OpRewritePattern<ModuleOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(ModuleOp module, PatternRewriter &rewriter) const override {
            module.walk([&](FuncOp op) {
                rewriter.eraseOp(op);
            });

            for (auto &sop: *(module.getBody())) {
                if (auto design = dyn_cast<tor::DesignOp>(sop)) {
                    if (design->getAttr("flatten-array"))
                        return failure();
                    design->setAttr("flatten-array", IntegerAttr::get(IntegerType::get(getContext(), 32), 1));
                    for (auto &op: (design.getBody().front())) {
                        if (auto alloc = dyn_cast<tor::AllocOp>(op)) {
                            auto memref = alloc.getMemref();
                            bool R = false;
                            bool W = false;
                            std::string RW;
                            for (auto &use: memref.getUses()) {
                                if (isa<tor::LoadOp, tor::StreamReadOp>(use.getOwner())) {
                                    R = true;
                                } else if (isa<tor::StoreOp, tor::StreamWriteOp>(use.getOwner())) {
                                    W = true;
                                } else {
                                    assert(false);
                                }
                            }
                            if (R) RW += "r";
                            if (W) RW += "w";
                            auto shape = memref.getType().getShape();
                            int64_t size = 1;
                            SmallVector<int64_t> one;
                            for (auto s: shape) {
                                size *= s;
                            }
                            one.push_back(size);
                            for (auto &use: memref.getUses()) {
                                if (auto load = dyn_cast<tor::LoadOp>(use.getOwner())) {
                                    auto lastInsertion = rewriter.saveInsertionPoint();
                                    rewriter.setInsertionPoint(load);
                                    if (load.getIndices().empty()) {
                                        auto index = rewriter.create<ConstantIndexOp>(load.getLoc(), 0);
                                        load->insertOperands(1, index.getResult());
                                    } else {
                                        auto last_idx = load.getIndices()[0];
                                        for (unsigned i = 1; i < shape.size(); ++i) {
                                            int length = shape[i];
                                            if (!(length & (length - 1))) {
                                                int width = log2(length);
                                                auto size = rewriter.create<ConstantIndexOp>(load.getLoc(), width);
                                                auto mul = rewriter.create<ShLIOp>(load.getLoc(), last_idx,
                                                                                   size.getResult());
                                                auto add = rewriter.create<AddIOp>(load.getLoc(), load.getIndices()[i],
                                                                                   mul.getResult());
                                                last_idx = add.getResult();
                                            } else {
                                                auto size = rewriter.create<ConstantIndexOp>(load.getLoc(), length);
                                                auto mul = rewriter.create<MulIOp>(load.getLoc(), last_idx,
                                                                                   size.getResult());
                                                auto add = rewriter.create<AddIOp>(load.getLoc(), load.getIndices()[i],
                                                                                   mul.getResult());
                                                last_idx = add.getResult();
                                            }
                                        }
                                        load->setOperand(1, last_idx);
                                        for (unsigned i = shape.size() - 1; i >= 1; --i) {
                                            load->eraseOperand(i + 1);
                                        }
                                    }
                                    rewriter.restoreInsertionPoint(lastInsertion);
                                } else if (auto store = dyn_cast<tor::StoreOp>(use.getOwner())) {
                                    auto lastInsertion = rewriter.saveInsertionPoint();
                                    rewriter.setInsertionPoint(store);
                                    if (store.getIndices().empty()) {
                                        auto index = rewriter.create<ConstantIndexOp>(store.getLoc(), 0);
                                        store->insertOperands(2, index.getResult());
                                    } else {
                                        auto last_idx = store.getIndices()[0];
                                        for (unsigned i = 1; i < shape.size(); ++i) {
                                            int length = shape[i];
                                            if (!(length & (length - 1))) {
                                                int width = log2(length);
                                                auto size = rewriter.create<ConstantIndexOp>(store.getLoc(), width);
                                                auto mul = rewriter.create<ShLIOp>(store.getLoc(), last_idx,
                                                                                   size.getResult());
                                                // auto add = rewriter.create<OrIOp>(store.getLoc(), store.getIndices()[i], mul.getResult());
                                                auto add = rewriter.create<AddIOp>(store.getLoc(),
                                                                                   store.getIndices()[i],
                                                                                   mul.getResult());
                                                last_idx = add.getResult();
                                            } else {
                                                auto size = rewriter.create<ConstantIndexOp>(store.getLoc(), length);
                                                auto mul = rewriter.create<MulIOp>(store.getLoc(), last_idx,
                                                                                   size.getResult());
                                                auto add = rewriter.create<AddIOp>(store.getLoc(),
                                                                                   store.getIndices()[i],
                                                                                   mul.getResult());
                                                last_idx = add.getResult();
                                            }
                                        }
                                        store->setOperand(2, last_idx);
                                        for (unsigned i = shape.size() - 1; i >= 1; --i) {
                                            store->eraseOperand(i + 2);
                                        }
                                    }
                                    rewriter.restoreInsertionPoint(lastInsertion);
                                }
                            }
                            auto newType = tor::MemRefType::get(one, memref.getType().getElementType(), {},
                                                                StringAttr::get(getContext(), RW));
                            alloc.getResult().setType(newType);
                        }
                    }
                }
            }
            return success();
        }
    };

    struct WriteMemrefGlobal : public OpRewritePattern<ModuleOp> {
        std::string output_path;

        WriteMemrefGlobal(MLIRContext *ctx, std::string output_path) :
                OpRewritePattern<mlir::ModuleOp>(ctx), output_path(output_path) {}

        LogicalResult matchAndRewrite(ModuleOp moduleOp, PatternRewriter &rewriter) const override {
            std::filesystem::path path(output_path);
            if (!std::filesystem::exists(path)) {
                std::filesystem::create_directory(path);
            }
            int number = 0;
            moduleOp->walk([&](tor::DesignOp designOp) {
                designOp.walk([&](tor::AllocOp op) {
                    auto value = op.getValueD();
                    if (value.has_value() && isa<DenseElementsAttr>(value.value())) {
                        std::string str;
                        llvm::raw_string_ostream os(str);
                        op.getResult().print(os);
                        std::string resultName = os.str().substr(0, os.str().find('=') - 1);

                        op.setValueHAttr(StringAttr::get(op.getContext(),
                                writeMemrefGlobalDenseH(op, resultName, rewriter)));
                        op.setValueDAttr(StringAttr::get(op.getContext(),
                                writeMemrefGlobalDenseD(op, resultName, rewriter)));
                        number++;
                    }
                });
            });
            LLVM_DEBUG(llvm::dbgs() << "memref.global number: " << number << "\n");
            return number == 0 ? failure() : success();
        }

        template<typename T>
        void writeDenseD(tor::AllocOp op, Type expectType, int64_t expectSize, int64_t &size,
                         std::ofstream &fileWrite, int64_t fixed = -1) const {
            if (op.getMemref().getType().getElementType() != expectType) {
                return;
            }
            size = expectSize;
            auto rawData = llvm::cast<DenseElementsAttr>(op.getValueDAttr()).getRawData();
            for (int i = 0, r = rawData.size(); i < r; i += size) {
                T data;
                for (int j = 0; j < size; ++j) {
                    *((char *)(&data) + j) = rawData[i + j];
                }
                if (fixed == -1) {
                    fileWrite << data << "\n";
                } else {
                    fileWrite << std::fixed << std::setprecision(fixed) << data << "\n";
                }
            }
        }

        std::string writeMemrefGlobalDenseD(tor::AllocOp op, std::string fileName, PatternRewriter &rewriter) const {
            std::filesystem::path path(output_path + "/d");
            if (!std::filesystem::exists(path)) {
                std::filesystem::create_directory(path);
            }
            std::string filePath = output_path + "/d" + "/memref_global_" + fileName;
            std::ofstream fileWrite;
            fileWrite.open(filePath);

            int64_t fixed = 9;
            int64_t size = -1;
            writeDenseD<float>(op, rewriter.getBF16Type(), 2, size, fileWrite, fixed);
            writeDenseD<float>(op, rewriter.getF16Type(), 2, size, fileWrite, fixed);
            writeDenseD<float>(op, rewriter.getF32Type(), 4, size, fileWrite, fixed);
            writeDenseD<double>(op, rewriter.getF64Type(), 8, size, fileWrite, fixed);
            writeDenseD<uint8_t>(op, rewriter.getIntegerType(8, false), 1, size, fileWrite);
            writeDenseD<int8_t>(op, rewriter.getI8Type(), 1, size, fileWrite);
            writeDenseD<int16_t>(op, rewriter.getI16Type(), 2, size, fileWrite);
            writeDenseD<int32_t>(op, rewriter.getI32Type(), 4, size, fileWrite);
            writeDenseD<int64_t>(op, rewriter.getI64Type(), 8, size, fileWrite);
            assert(size != -1 && "unsupported element type.");

            fileWrite.close();
            return filePath;
        }

        void writeDenseH(tor::AllocOp op, Type expectType, int64_t expectSize, int64_t &size,
                         std::ofstream &fileWrite) const {
            if (op.getMemref().getType().getElementType() != expectType) {
                return;
            }
            size = expectSize;
            auto rawData = llvm::cast<DenseElementsAttr>(op.getValueDAttr()).getRawData();
            auto toHex = [&](char value) -> char {
                assert(value >= 0 && value < 16);
                if (value < 10) {
                    return value + '0';
                }
                return value - 10 + 'A';
            };
            for (int i = 0, r = rawData.size(); i < r; i += size) {
                for (int j = size - 1; j >= 0; --j) {
                    auto data = rawData[i + j];
                    char high = (data >> 4) & 0x0F;
                    char low = data & 0x0F;
                    fileWrite << toHex(high) << toHex(low);
                }
                fileWrite << "\n";
            }
            auto shape = op.getType().getShape();
            int64_t mul = size;
            for (auto dim : shape) mul *= dim;
            if (rawData.size() != mul) {
                assert((int64_t) rawData.size() == size && "size not match");
                for (int i = rawData.size(); i < mul; i += size) {
                    for (int j = size - 1; j >= 0; --j) {
                        auto data = 0;
                        char high = (data >> 4) & 0x0F;
                        char low = data & 0x0F;
                        fileWrite << toHex(high) << toHex(low);
                    }
                    fileWrite << "\n";
                }
            }
        }

        std::string writeMemrefGlobalDenseH(tor::AllocOp op, std::string fileName, PatternRewriter &rewriter) const {
            std::filesystem::path path(output_path + "/h");
            if (!std::filesystem::exists(path)) {
                std::filesystem::create_directory(path);
            }
            std::string filePath = output_path + "/h" + "/memref_global_" + fileName;

            std::ofstream fileWrite;
            fileWrite.open(filePath);

            int64_t size = -1;
            writeDenseH(op, rewriter.getBF16Type(), 2, size, fileWrite);
            writeDenseH(op, rewriter.getF16Type(), 2, size, fileWrite);
            writeDenseH(op, rewriter.getF32Type(), 4, size, fileWrite);
            writeDenseH(op, rewriter.getF64Type(), 8, size, fileWrite);
            writeDenseH(op, rewriter.getI16Type(), 2, size, fileWrite);
            writeDenseH(op, rewriter.getI32Type(), 4, size, fileWrite);
            writeDenseH(op, rewriter.getI64Type(), 8, size, fileWrite);
            writeDenseH(op, rewriter.getI8Type(), 1, size, fileWrite);
            writeDenseH(op, rewriter.getIntegerType(8, false), 1, size, fileWrite);
            assert(size != -1 && "unsupported element type.");

            fileWrite.close();
            return filePath;
        }
    };
    
    struct MoveWhileOp : public OpRewritePattern<mlir::ModuleOp> {
        using OpRewritePattern<mlir::ModuleOp>::OpRewritePattern;

        LogicalResult
        matchAndRewrite(mlir::ModuleOp op, PatternRewriter &rewriter) const override {
            auto design = llvm::dyn_cast<tor::DesignOp>(op.getBody()->front());
            if (design->getAttr("move-while")) return failure();
            design->setAttr("move-while", IntegerAttr::get(IntegerType::get(getContext(), 32), 1));
            SmallVector<scf::WhileOp> ops;
            design.walk([&](scf::WhileOp op) { ops.push_back(op); });
            for (auto &op: ops) {
                if (isa<scf::ConditionOp>(op.getBefore().front().begin())) continue;
                rewriter.setInsertionPoint(op);
                auto tmp_op = cast<scf::WhileOp>(rewriter.clone(*op));
                rewriter.setInsertionPointAfter(op);
                auto tmp_op_2 = cast<scf::WhileOp>(rewriter.clone(*op));
                for (unsigned idx = 0; idx < tmp_op.getBefore().getNumArguments(); ++idx) {
                    auto arg = tmp_op.getBefore().getArgument(idx);
                    SmallVector<std::pair<Operation *, int>> pairs;
                    for (auto &use: arg.getUses()) {
                        pairs.push_back(std::make_pair(use.getOwner(), use.getOperandNumber()));
                    }
                    for (auto pair: pairs) {
                        pair.first->setOperand(pair.second, tmp_op.getInits()[idx]);
                    }
                }

                rewriter.setInsertionPoint(tmp_op);
                for (auto &sop: tmp_op.getBefore().front()) {
                    if (isa<scf::ConditionOp>(sop)) continue;
                    sop.getResults().replaceAllUsesWith(rewriter.clone(sop)->getResults());
                }

                auto cond = cast<scf::ConditionOp>(op.getBefore().front().getTerminator());
                unsigned idx = op.getNumOperands();
                op->insertOperands(idx, cond.getCondition());
                Block *new_block = new Block();
                op.getBefore().push_back(new_block);
                SmallVector<Location> locations(cond->getNumOperands(), op.getLoc());
                new_block->addArguments(cond->getOperandTypes(), locations);
                SmallVector<Value> values;
                for (unsigned idx = 1; idx < new_block->getNumArguments(); ++idx) {
                    values.push_back(new_block->getArgument(idx));
                }
                rewriter.setInsertionPointToStart(new_block);
                rewriter.create<scf::ConditionOp>(op.getLoc(), new_block->getArgument(0), values);
                op->setOperands(tmp_op.getBefore().front().getTerminator()->getOperands());
                op.getBefore().front().erase();
                rewriter.eraseOp(tmp_op);

                rewriter.setInsertionPoint(op);
                tmp_op = tmp_op_2;
                auto yield = cast<scf::YieldOp>(op.getAfter().front().getTerminator());
                for (unsigned idx = 0; idx < tmp_op.getBefore().getNumArguments(); ++idx) {
                    auto arg = tmp_op.getBefore().getArgument(idx);
                    SmallVector<std::pair<Operation *, int>> pairs;
                    for (auto &use: arg.getUses()) {
                        pairs.push_back(std::make_pair(use.getOwner(), use.getOperandNumber()));
                    }
                    for (auto pair: pairs) {
                        pair.first->setOperand(pair.second, yield.getOperand(idx));
                    }
                }
                rewriter.setInsertionPointToEnd(&op.getAfter().front());
                for (auto &sop: tmp_op.getBefore().front()) {
                    if (isa<scf::ConditionOp>(sop)) continue;
                    sop.getResults().replaceAllUsesWith(rewriter.clone(sop)->getResults());
                }
                rewriter.create<scf::YieldOp>(op.getLoc(), tmp_op.getBefore().front().getTerminator()->getOperands());
                rewriter.eraseOp(yield);
                rewriter.eraseOp(tmp_op);
            }
            return success();
        }
    };

    // REMOVED: DesignOpPattern - no longer needed, we create one design per function directly

    void solve(FuncOp op, std::string resource, double clock, PatternRewriter &rewriter) {
        rewriter.setInsertionPoint(op);
        StringRef function_name = op.getName();

        auto funcOp = rewriter.create<tor::FuncOp>(op.getLoc(), function_name,
                                                   op.getFunctionType());
        funcOp->setAttr("clock", FloatAttr::get(Float32Type::get(op.getContext()), clock));
        funcOp->setAttr("resource", StringAttr::get(op.getContext(), resource));
        if (auto IIAttr = op->getAttr("II")) {
            funcOp->setAttr("II", IIAttr);
            funcOp->setAttr("pipeline", StringAttr::get(op.getContext(), "func"));
        }
        saveOldAttrWithName(funcOp, op, "dataflow");
        saveOldAttrWithName(funcOp, op, "dataflow-line");
        // Preserve opcode and funct7 attributes for ISA extensions
        saveOldAttrWithName(funcOp, op, "opcode");
        saveOldAttrWithName(funcOp, op, "funct7");
        rewriter.inlineRegionBefore(op.getBody(), funcOp.getBody(), funcOp.end());
        rewriter.setInsertionPointToStart(op->getBlock());
        funcOp.walk([&](LLVM::UndefOp op) {
            auto lastInsertion = rewriter.saveInsertionPoint();
            rewriter.setInsertionPoint(op);
            Value undef;
            if (op.getRes().getType().isIntOrIndex()) {
                undef = rewriter.create<ConstantIntOp>(op.getLoc(), 0,
                                                       op.getRes().getType().getIntOrFloatBitWidth()).getResult();
            } else if (op.getRes().getType().isF32()) {
                undef = rewriter.create<ConstantFloatOp>(op.getLoc(), rewriter.getF32Type(),
                                                         APFloat(0.0f)).getResult();
            } else if (op.getRes().getType().isF64()) {
                undef = rewriter.create<ConstantFloatOp>(op.getLoc(), rewriter.getF64Type(),
                                                         APFloat(0.0f)).getResult();
            } else {
                assert(false);
            }
            op.getRes().replaceAllUsesWith(undef);
            rewriter.restoreInsertionPoint(lastInsertion);
            rewriter.eraseOp(op);
        });
        SmallVector<Operation *, 8> opsToErase;
        // DISABLED: ConvertInput pass no longer modifies memref dialect operations
        // funcOp.walk([&](memref::AllocaOp alloca) {
        //   if (alloca->hasAttr("dataflow")) {
        //     Type elementType;
        //     auto dataflowString = llvm::dyn_cast<StringAttr>(alloca->getAttr("dataflow")).getValue();
        //     if (dataflowString.contains("double")) {
        //         elementType = rewriter.getF64Type();
        //     }
        //     else if (dataflowString.contains("float")) {
        //         elementType = rewriter.getF32Type();
        //     }
        //     else if (dataflowString.contains("int")) {
        //         elementType = rewriter.getI32Type();
        //     }
        //     else if (dataflowString.contains("long long")) {
        //         elementType = rewriter.getI32Type();
        //     }
        //     else {
        //         llvm_unreachable("Unkown dataflow type!");
        //     }
        //     int depth = 0;
        //     llvm::Regex pattern(", .*>", llvm::Regex::RegexFlags::Newline);
        //     llvm::SmallVector<StringRef, 8> matches;
        //     if (pattern.match(dataflowString, &matches)) {
        //       std::string depthStr =
        //           matches[0].substr(2, dataflowString.size() - 3).str();
        //       depth = std::stoi(depthStr);
        //     }
        //     auto streamCreateOp = rewriter.create<tor::StreamCreateOp>(
        //         op.getLoc(),
        //         tor::StreamType::get(rewriter.getContext(),
        //                              elementType),
        //         depth);
        //     int castOpNum = 0;
        //     Operation *castOp;
        //     for (auto u : alloca->getUsers()) {
        //         castOp = u;
        //         ++castOpNum;
        //     }
        //     if (castOpNum != 1) {
        //         llvm_unreachable("Unkown dataflow alloca type!");
        //     }
        //     castOp->getResult(0).replaceAllUsesWith(streamCreateOp);
        //     opsToErase.push_back(castOp);
        //     opsToErase.push_back(alloca);
        //   }
        // });
        // for (auto *op : opsToErase)
        //     rewriter.eraseOp(op);
        funcOp.walk([&](CallOp call) {
            auto funcName = call.getCallee();
            if (funcName.find("stream") == 0) {
                if (funcName.find("read") != StringRef::npos) {
                    rewriter.setInsertionPoint(call);
                    rewriter.replaceOpWithNewOp<tor::StreamReadOp>(
                        call, call.getResultTypes()[0], call.getOperand(0));
                } else if (funcName.find("write") != StringRef::npos) {
                    rewriter.setInsertionPoint(call);
                    rewriter.replaceOpWithNewOp<tor::StreamWriteOp>(
                        call, call.getOperand(1), call.getOperand(0));
                }
            }
        });
        funcOp.walk([&](ReturnOp op) {
            rewriter.setInsertionPoint(op);
            rewriter.create<tor::ReturnOp>(op.getLoc(), op.getOperands());
            rewriter.eraseOp(op);
        });
        rewriter.setInsertionPointToStart(op->getBlock());
        // DISABLED: ConvertInput pass no longer modifies memref dialect operations
        // funcOp.walk([&](memref::AllocaOp alloca) {    // 临时变量
        //     auto memref = alloca.getMemref().getType();
        //     tor::AllocOp alloc;
        //     if (memref.getShape().empty()) {
        //         auto newType = tor::MemRefType::get({1}, memref.getElementType(), {},
        //                                             StringAttr::get(op.getContext(), ""));
        //         alloc = rewriter.create<tor::AllocOp>(op.getLoc(), newType, nullptr, nullptr, nullptr);
        //         alloca.getResult().replaceAllUsesWith(alloc.getResult());
        //         alloc.setLocalTypeAttr(StringAttr::get(alloc.getContext(), "local"));
        //     } else {
        //         auto newType = tor::MemRefType::get(memref.getShape(), memref.getElementType(), {},
        //                                             StringAttr::get(op.getContext(), ""));
        //         alloc = rewriter.create<tor::AllocOp>(op.getLoc(), newType, nullptr, nullptr, nullptr);
        //         alloca.getResult().replaceAllUsesWith(alloc.getResult());
        //         alloc.setLocalTypeAttr(StringAttr::get(alloc.getContext(), "local"));
        //     }
        //     saveOldAttrWithName(alloc, alloca, "bind_storage_type");
        //     saveOldAttrWithName(alloc, alloca, "bind_storage-line");
        //     saveOldAttrWithName(alloc, alloca, "mode");
        //     saveOldAttrWithName(alloc, alloca, "bus");
        //     saveOldAttrWithName(alloc, alloca, "offset");
        //     saveOldAttrWithName(alloc, alloca, "ARLEN");
        //     saveOldAttrWithName(alloc, alloca, "AWLEN");
        //     saveOldAttrWithName(alloc, alloca, "max_widen_bitwidth");
        //     saveOldAttrWithName(alloc, alloca, "initial_addr");
        //     saveOldAttrWithName(alloc, alloca, "interface-storage_type");
        //     saveOldAttrWithName(alloc, alloca, "num_read_outstanding");
        //     saveOldAttrWithName(alloc, alloca, "num_write_outstanding");
        //     rewriter.eraseOp(alloca);
        // });
        // DISABLED: ConvertInput pass no longer modifies memref dialect operations
        // funcOp.walk([&](memref::AllocOp alloca) {
        //     auto memref = alloca.getMemref().getType();
        //     if (memref.getShape().empty()) {
        //         auto newType = tor::MemRefType::get({1}, memref.getElementType(), {},
        //                                             StringAttr::get(op.getContext(), ""));
        //         auto alloc = rewriter.create<tor::AllocOp>(op.getLoc(), newType, nullptr, nullptr, nullptr);
        //         alloca.getResult().replaceAllUsesWith(alloc.getResult());
        //     } else {
        //         auto newType = tor::MemRefType::get(memref.getShape(), memref.getElementType(), {},
        //                                             StringAttr::get(op.getContext(), ""));
        //         auto alloc = rewriter.create<tor::AllocOp>(op.getLoc(), newType, nullptr, nullptr, nullptr);
        //         alloca.getResult().replaceAllUsesWith(alloc.getResult());
        //     }
        //     rewriter.eraseOp(alloca);
        // });
        // DISABLED: ConvertInput pass no longer modifies memref dialect operations
        // funcOp.walk([&](memref::LoadOp load) {
        //     rewriter.setInsertionPoint(load);
        //     auto new_load = rewriter.create<tor::LoadOp>(load.getLoc(), load.getResult().getType(),
        //                                                  load.getOperand(0), 0, 0,
        //                                                  load.getIndices());
        //     load.getResult().replaceAllUsesWith(new_load.getResult());
        //     if (load->getAttr("distance")) {
        //       new_load->setAttr("distance", load->getAttr("distance"));
        //     }
        //     rewriter.eraseOp(load);
        // });
        // DISABLED: ConvertInput pass no longer modifies memref dialect operations
        // funcOp.walk([&](memref::StoreOp store) {
        //     rewriter.setInsertionPoint(store);
        //     auto new_store = rewriter.create<tor::StoreOp>(store.getLoc(), store.getValue(),
        //                                                    store.getOperand(1), 0, 0,
        //                                                    store.getIndices());
        //     if (store->getAttr("distance")) {
        //       new_store->setAttr("distance", store->getAttr("distance"));
        //     }
        //     rewriter.eraseOp(store);
        // });
        // DISABLED: ConvertInput pass no longer modifies memref dialect operations
        // funcOp.walk([&](memref::CastOp cast) {
        //     cast.getResult().replaceAllUsesWith(cast.getOperand());
        //     rewriter.eraseOp(cast);
        // });
        // DISABLED: ConvertInput pass no longer modifies memref dialect operations
        // rewriter.setInsertionPointToStart(op->getBlock());
        // unsigned idx = 0, oldIdx = 0;
        // while (idx < funcOp.getNumArguments()) {
        //   auto arg = funcOp.getArgument(idx);
        //   if (auto memref = dyn_cast<MemRefType>(arg.getType())) {
        //     auto newType =
        //         tor::MemRefType::get(memref.getShape(), memref.getElementType(),
        //                              {}, StringAttr::get(op.getContext(), ""));
        //     auto alloc = rewriter.create<tor::AllocOp>(
        //         funcOp.getLoc(), newType, nullptr, nullptr, nullptr);
        //     saveOldArgAttrWithName(alloc, op, "bind_storage_type", oldIdx);
        //     saveOldArgLineAttrWithName(alloc, op, "bind_storage", oldIdx);
        //     saveOldArgLineAttrWithName(alloc, op, "interface", oldIdx);
        //     saveOldArgAttrWithName(alloc, op, "mode", oldIdx);
        //     saveOldArgAttrWithName(alloc, op, "bus", oldIdx);
        //     saveOldArgAttrWithName(alloc, op, "offset", oldIdx);
        //     saveOldArgAttrWithName(alloc, op, "ARLEN", oldIdx);
        //     saveOldArgAttrWithName(alloc, op, "AWLEN", oldIdx);
        //     saveOldArgAttrWithName(alloc, op, "max_widen_bitwidth", oldIdx);
        //     saveOldArgAttrWithName(alloc, op, "initial_addr", oldIdx);
        //     saveOldArgAttrWithName(alloc, op, "interface-storage_type", oldIdx);
        //     saveOldArgAttrWithName(alloc, op, "num_read_outstanding", oldIdx);
        //     saveOldArgAttrWithName(alloc, op, "num_write_outstanding", oldIdx);
        //     arg.replaceAllUsesWith(alloc.getResult());
        //     (void)funcOp.eraseArgument(idx);
        //   } else {
        //     idx += 1;
        //   }
        //   oldIdx += 1;
        // }
        static int func_num = 0;
        funcOp.walk([&](CallOp call) {
            // bool hasMem = false;
            // for (auto arg_type: call.getCalleeType().getInputs()) {
            //     if (isa<tor::MemRefType, MemRefType, tor::StreamType>(arg_type)) {
            //         hasMem = true;
            //         break;
            //     }
            // }
            bool hasMem = true;
            if (hasMem) {
                auto design = dyn_cast<tor::DesignOp>(funcOp->getParentOp());
                for (auto func: design.getOps<func::FuncOp>()) {
                    if (func.getSymName() == call.getCallee()) {
                        rewriter.setInsertionPoint(funcOp);
                        auto new_func = cast < func::FuncOp > (rewriter.clone(*func));
                        new_func.setSymName(new_func.getSymName().str() + "_" + std::to_string(++func_num));
                        SmallVector<Value, 8> new_operands;
                        unsigned idx = 0;
                        for (auto &operand: call->getOpOperands()) {
                            if (isa<tor::MemRefType, MemRefType, tor::StreamType>(operand.get().getType())) {
                                new_func.getArgument(idx).replaceAllUsesWith(operand.get());
                                (void)new_func.eraseArgument(idx);
                            } else {
                                new_operands.push_back(operand.get());
                                idx += 1;
                            }
                        }
                        rewriter.setInsertionPoint(call);
                        rewriter.replaceOpWithNewOp<tor::CallOp>(call, call.getResultTypes(),
                                                                 new_func.getSymName(), 0, 0, new_operands);
                        solve(new_func, resource, clock, rewriter);
                        return;
                    }
                }
            } else {
                rewriter.setInsertionPoint(call);
                auto newOp = rewriter.create<tor::CallOp>(call.getLoc(), call.getResultTypes(),
                                                          call.getCallee(), 0, 0, call->getOperands());
                rewriter.replaceOp(call, newOp.getResults());
                auto design = dyn_cast<tor::DesignOp>(funcOp->getParentOp());
                for (auto func: design.getOps<func::FuncOp>()) {
                    if (func.getSymName() == call.getCallee()) {
                        solve(func, resource, clock, rewriter);
                        return;
                    }
                }
            }
        });

        rewriter.eraseOp(op);
    }

    // REMOVED: FuncToDesignPattern - using direct manipulation instead of pattern matching


    struct MulIOpConversion : public OpRewritePattern<func::FuncOp> {
        using OpRewritePattern<func::FuncOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(func::FuncOp funcOp, PatternRewriter &rewriter) const override {
            if (funcOp->getAttr("mul-convert"))
                return failure();
            SmallVector<std::pair<MulIOp, ShLIOp>> replace;
            funcOp.walk([&](MulIOp op) {
                auto val = op.getRhs();
                APInt int_val;
                if (matchPattern(val, m_ConstantInt(&int_val))) {
                    if (int_val.isPowerOf2()) {
                        int width = int_val.logBase2();
                        rewriter.setInsertionPoint(op);
                        if (op.getRhs().getType().isIndex()) {
                            auto constant = rewriter.create<ConstantIndexOp>(op.getLoc(), width);
                            auto shift_left = rewriter.create<ShLIOp>(op.getLoc(), op.getLhs(), constant.getResult());
                            replace.push_back(std::make_pair(op, shift_left));
                        } else {
                            auto constant = rewriter.create<ConstantIntOp>(op.getLoc(), op.getRhs().getType(), 
                                                                           static_cast<int64_t>(width));
                            auto shift_left = rewriter.create<ShLIOp>(op.getLoc(), op.getLhs(), constant.getResult());
                            replace.push_back(std::make_pair(op, shift_left));
                        }
                    }
                }
            });
            for (auto &pair: replace) {
                pair.first.getResult().replaceAllUsesWith(pair.second.getResult());
            }
            funcOp->setAttr("mul-convert",
                            IntegerAttr::get(IntegerType::get(getContext(), 32), 1));
            return success();
        }
    };

    struct MulIOpErase : public OpConversionPattern<MulIOp> {
        using OpConversionPattern<MulIOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(MulIOp op, MulIOp::Adaptor adaptor,
                                      ConversionPatternRewriter &rewriter) const override {
            if (!op.use_empty())
                return failure();
            rewriter.eraseOp(op);
            return success();
        }
    };


    struct ConvertInputPass : ConvertInputBase<ConvertInputPass> {
        void runOnOperation() override {
            auto moduleOp = getOperation();
            
            // llvm::errs() << "=== ConvertInputPass::runOnOperation START ===\n";
            // llvm::errs() << "  top_function = \"" << top_function << "\"\n";
            // llvm::errs() << "  resource = \"" << resource << "\"\n";
            // llvm::errs() << "  clock = " << clock << "\n";
            // llvm::errs() << "  output_path = \"" << output_path << "\"\n";
            
            // NEW WAY: Use default config (simplest approach)
            GreedyRewriteConfig config;
            // Most settings have sensible defaults now
            
            // Huang Ruibo comments on this line of code to preserve the 
            // pragma report Attr.
            // moduleOp->setAttrs(DictionaryAttr::getWithSorted(&getContext(), {}));
            
            {
                // llvm::errs() << "  Phase 1: MulIOpConversion\n";
                moduleOp.walk([&](func::FuncOp op) {
                    RewritePatternSet Patterns(op.getContext());
                    Patterns.add<MulIOpConversion>(op.getContext());
                    if (failed(applyPatternsAndFoldGreedily(op.getOperation(), std::move(Patterns), config)))
                        signalPassFailure();
                });
            }
            {
                // llvm::errs() << "  Phase 2: MulIOpErase\n";
                ConversionTarget target(getContext());
                RewritePatternSet patterns(&getContext());
                target.addDynamicallyLegalOp<MulIOp>([](MulIOp op) {
                    if (op.use_empty())
                        return false;
                    return true;
                });
                patterns.add<MulIOpErase>(&getContext());
                if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
                    signalPassFailure();
            }
            {
                // llvm::errs() << "  Phase 3: Converting all func.func with opcode/funct7 to a single tor.design\n";

                // Collect all func.func operations at module level that have opcode and funct7 attributes
                SmallVector<FuncOp> funcsToConvert;
                // Also collect memref.global operations to move into the design
                SmallVector<memref::GlobalOp> globalMemrefs;
                
                for (Operation &op : llvm::make_early_inc_range(*moduleOp.getBody())) {
                    if (auto funcOp = dyn_cast<FuncOp>(&op)) {
                        // Only convert functions with both opcode and funct7 attributes
                        if (funcOp->hasAttr("opcode") && funcOp->hasAttr("funct7")) {
                            funcsToConvert.push_back(funcOp);
                        }
                    } else if (auto globalOp = dyn_cast<memref::GlobalOp>(&op)) {
                        // Collect all memref.global operations
                        globalMemrefs.push_back(globalOp);
                    }
                }

                // llvm::errs() << "    Found " << funcsToConvert.size() << " functions with opcode/funct7 to convert\n";
                // llvm::errs() << "    Found " << globalMemrefs.size() << " memref.global operations to move\n";

                if (!funcsToConvert.empty()) {
                    OpBuilder builder(&getContext());

                    // Create a single tor::DesignOp for all functions
                    builder.setInsertionPoint(funcsToConvert[0]);
                    auto designOp = builder.create<tor::DesignOp>(funcsToConvert[0].getLoc(), "aps_isaxes");

                    // Add entry block to designOp's body region
                    Block *designBlock = new Block();
                    designOp.getBody().push_back(designBlock);

                    // Move all memref.global operations into the design first
                    for (auto globalOp : globalMemrefs) {
                        // llvm::errs() << "    Moving memref.global: " << globalOp.getSymName() << "\n";
                        globalOp->moveBefore(designBlock, designBlock->end());
                    }

                    // Move all functions into the single design
                    for (auto funcOp : funcsToConvert) {
                        // llvm::errs() << "    Moving function: " << funcOp.getName() << "\n";
                        
                        // Keep opcode and funct7 attributes on the function itself
                        // so they can be accessed later for scheduling/routing
                        
                        // Move the function into the shared design
                        funcOp->moveBefore(designBlock, designBlock->end());
                    }

                    // llvm::errs() << "    Created single tor.design with " << funcsToConvert.size() << " functions and " << globalMemrefs.size() << " globals\n";
                }

                // llvm::errs() << "  Phase 3: Completed\n";
            }
            {
                RewritePatternSet Patterns(&getContext());
                Patterns.add<MoveWhileOp>(&getContext());
                GreedyRewriteConfig config;  // Local config for this scope
                if (failed(applyPatternsAndFoldGreedily(moduleOp.getOperation(), std::move(Patterns), config)));
            }

            {
                // llvm::errs() << "  Phase 5: Converting func.func to tor.func inside each tor.design\n";

                // Create an OpBuilder for rewriting
                OpBuilder builder(&getContext());

                // Walk through all tor.design operations and convert their functions
                SmallVector<std::pair<tor::DesignOp, SmallVector<FuncOp>>> designsWithFuncs;
                moduleOp->walk([&](tor::DesignOp designOp) {
                    SmallVector<FuncOp> funcsInDesign;
                    designOp.walk([&](FuncOp op) {
                        funcsInDesign.push_back(op);
                    });
                    if (!funcsInDesign.empty()) {
                        designsWithFuncs.push_back({designOp, funcsInDesign});
                    }
                });

                // Convert functions in each design
                for (auto &pair : designsWithFuncs) {
                    auto designOp = pair.first;
                    auto &funcsInDesign = pair.second;

                    // llvm::errs() << "    Found " << funcsInDesign.size() << " functions in design: "
                    //              << designOp.getName() << "\n";

                    for (auto funcOp : funcsInDesign) {
                        // llvm::errs() << "    Converting func: " << funcOp.getName() << "\n";

                        // Create a wrapper rewriter that inherits from PatternRewriter
                        struct SimplePatternRewriter : public PatternRewriter {
                            SimplePatternRewriter(MLIRContext *ctx) : PatternRewriter(ctx) {}
                        };

                        SimplePatternRewriter rewriter(&getContext());

                        // Call solve to convert func.func to tor.func
                        solve(funcOp, resource, clock, rewriter);
                    }
                }

                // llvm::errs() << "  Phase 5: Completed\n";
            }

            {
                RewritePatternSet Patterns(&getContext());
                Patterns.add<FlattenArray>(&getContext());

                if (failed(applyPatternsAndFoldGreedily(moduleOp.getOperation(), std::move(Patterns), config))) {
                    signalPassFailure();
                }
            }

            {
                RewritePatternSet Patterns(&getContext());
                Patterns.add<WriteMemrefGlobal>(&getContext(), output_path);

                if (failed(applyPatternsAndFoldGreedily(moduleOp.getOperation(), std::move(Patterns), config))) {
                    signalPassFailure();
                }
            }
            
            // llvm::errs() << "=== ConvertInputPass::runOnOperation END ===\n";
        }
    };

} // namespace

namespace mlir {

    std::unique_ptr<OperationPass<mlir::ModuleOp>> createConvertInputPass() {
        return std::make_unique<ConvertInputPass>();
    }

} // namespace mlir
