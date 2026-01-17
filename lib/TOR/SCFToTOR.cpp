#include "TOR/TOR.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

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
#include "APS/APSDialect.h"
#include "APS/APSOps.h"
#include "llvm/ADT/STLExtras.h"

#include <set>
#include <iostream>
#include <type_traits>

#define DEBUG_TYPE "scf-to-tor"

namespace {
    using namespace mlir;
    using namespace mlir::arith;

    std::string get_opertion_attr() {
        static int attr_num = 0;
        return "control_" + std::to_string(attr_num++);
    }

    std::string get_tmp_attr() {
        static int attr_num = 0;
        return "unknown_" + std::to_string(attr_num++);
    }

    template<typename SourceOp, typename TargetOp>
    void myReplaceOp(SourceOp op, TargetOp newOp, ConversionPatternRewriter &rewriter) {

        rewriter.replaceOp(op, newOp->getResults());

    }

    inline void saveOldAttrWithName(mlir::Operation *newOp,
                                    mlir::Operation *op, std::string type) {
        if (auto attr = op->getAttr(type)) {
            newOp->setAttr(type, attr);
        }
    }

    template<typename SourceOp>
    struct IndexTypeConversionPattern : public OpConversionPattern<SourceOp> {
        using OpConversionPattern<SourceOp>::OpConversionPattern;

        SmallVector<Value>
        prepareOperands(SourceOp op, typename SourceOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const {
            auto operands = adaptor.getOperands();
            SmallVector<Type> newOperandTypes;
            auto converter = this->getTypeConverter();
            (void) converter->convertTypes(op->getOperandTypes(), newOperandTypes);
            return llvm::to_vector(llvm::map_range(llvm::zip(operands, newOperandTypes), [&](auto it) {
                auto [operand, tpe] = it;
                // If types already match, don't insert conversion cast
                if (operand.getType() == tpe) {
                    return operand;
                }
                return converter->materializeSourceConversion(rewriter, op->getLoc(), tpe, ValueRange(operand));
            }));
        }
    };

    struct YieldOpConversion : public IndexTypeConversionPattern<scf::YieldOp> {
        using IndexTypeConversionPattern<scf::YieldOp>::IndexTypeConversionPattern;

        LogicalResult matchAndRewrite(scf::YieldOp op, typename scf::YieldOp::Adaptor adaptor,
                                      ConversionPatternRewriter &rewriter) const final {
            auto operands = this->prepareOperands(op, adaptor, rewriter);
            auto newOp = rewriter.replaceOpWithNewOp<tor::YieldOp>(op, operands);
            return success();
        }
    };

    struct CondOpConversion : public IndexTypeConversionPattern<scf::ConditionOp> {
        using IndexTypeConversionPattern<scf::ConditionOp>::IndexTypeConversionPattern;

        LogicalResult matchAndRewrite(scf::ConditionOp op, typename scf::ConditionOp::Adaptor adaptor,
                                      ConversionPatternRewriter &rewriter) const final {
            auto operands = this->prepareOperands(op, adaptor, rewriter);
            auto newOp = rewriter.replaceOpWithNewOp<tor::ConditionOp>(op, operands[0],
                                                                       ValueRange(operands).drop_front(1));
            return success();
        }
    };

    template<typename SourceOp, typename TargetOp>
    struct BinOpConversionPattern : public IndexTypeConversionPattern<SourceOp> {
        using IndexTypeConversionPattern<SourceOp>::IndexTypeConversionPattern;

        LogicalResult matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                                      ConversionPatternRewriter &rewriter) const final {
            auto operands = this->prepareOperands(op, adaptor, rewriter);
            auto resType = this->getTypeConverter()->convertType(op->getResult(0).getType());

            auto newOp = rewriter.replaceOpWithNewOp<TargetOp>(op, resType, operands[0], operands[1], 0, 0);
            if (!op->hasAttr("dump")) {
                op->setAttr("dump", StringAttr::get(rewriter.getContext(), get_tmp_attr().c_str()));
            }
            saveOldAttrWithName(newOp, op, "bind_op_impl");
            saveOldAttrWithName(newOp, op, "bind_op_latency");
            saveOldAttrWithName(newOp, op, "bind_op-line");
            newOp->setAttr("dump", op->getAttr("dump"));
            return success();
        }
    };

    using MulIOpConversion = BinOpConversionPattern<MulIOp, tor::MulIOp>;
    using AddIOpConversion = BinOpConversionPattern<AddIOp, tor::AddIOp>;
    using SubIOpConversion = BinOpConversionPattern<SubIOp, tor::SubIOp>;
    using MulFOpConversion = BinOpConversionPattern<MulFOp, tor::MulFOp>;
    using AddFOpConversion = BinOpConversionPattern<AddFOp, tor::AddFOp>;
    using SubFOpConversion = BinOpConversionPattern<SubFOp, tor::SubFOp>;
    using DivFOpConversion = BinOpConversionPattern<DivFOp, tor::DivFOp>;

    template<typename SourceOp>
    struct SimpleOpConversion : public IndexTypeConversionPattern<SourceOp> {
        using IndexTypeConversionPattern<SourceOp>::IndexTypeConversionPattern;

        LogicalResult matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                                      ConversionPatternRewriter &rewriter) const final {
            if (!llvm::any_of(op->getResultTypes(), [](auto tpe) { return isa<IndexType>(tpe); }) &&
                !llvm::any_of(op->getOperandTypes(), [](auto tpe) { return isa<IndexType>(tpe); }))
                return failure();
            auto operands = this->prepareOperands(op, adaptor, rewriter);
            SmallVector<Type> resultTypes;
            (void) this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes);
            rewriter.replaceOpWithNewOp<SourceOp>(op, resultTypes, operands, op->getAttrs());
            return success();
        }
    };

    using ShiftLeftConversionPattern = SimpleOpConversion<ShLIOp>;
    using OrIConversionPattern = SimpleOpConversion<OrIOp>;
    using SelectConversionPattern = SimpleOpConversion<SelectOp>;
    using LoadOpConversion = SimpleOpConversion<tor::LoadOp>;
    using StoreOpConversion = SimpleOpConversion<tor::StoreOp>;
    using GuardedStoreOpConversion = SimpleOpConversion<tor::GuardedStoreOp>;
    using IndexCastOpConversion = SimpleOpConversion<IndexCastOp>;
    using DivUIOpConversion = SimpleOpConversion<DivUIOp>;
    using DivSIOpConversion = SimpleOpConversion<DivSIOp>;
    using RemUIOpConversion = SimpleOpConversion<RemUIOp>;
    using RemSIOpConversion = SimpleOpConversion<RemSIOp>;
    using SIToFPOpConversion = SimpleOpConversion<SIToFPOp>;
    using ShRSIOpConversion = SimpleOpConversion<ShRSIOp>;
    using ShRUIOpConversion = SimpleOpConversion<ShRUIOp>;

    // APS dialect operations conversion
    struct CpuRfReadConversion : public OpConversionPattern<aps::CpuRfRead> {
        using OpConversionPattern<aps::CpuRfRead>::OpConversionPattern;

        LogicalResult
        matchAndRewrite(aps::CpuRfRead op, aps::CpuRfRead::Adaptor adaptor,
                        ConversionPatternRewriter &rewriter) const override {
            // Clone the operation and add timing attributes
            rewriter.setInsertionPoint(op);
            auto newOp = rewriter.create<aps::CpuRfRead>(
                op.getLoc(), op.getResult().getType(), adaptor.getOperands());
            
            // Copy existing attributes
            for (auto attr : op->getAttrs()) {
                newOp->setAttr(attr.getName(), attr.getValue());
            }
            
            // Add timing attributes if not present
            if (!op->hasAttr("starttime")) {
                newOp->setAttr("starttime",
                    mlir::IntegerAttr::get(
                        mlir::IntegerType::get(getContext(), 32, mlir::IntegerType::Signless), 0));
            }
            if (!op->hasAttr("endtime")) {
                newOp->setAttr("endtime",
                    mlir::IntegerAttr::get(
                        mlir::IntegerType::get(getContext(), 32, mlir::IntegerType::Signless), 0));
            }
            
            // Add dump attribute if not present
            if (!op->hasAttr("dump")) {
                newOp->setAttr("dump", StringAttr::get(rewriter.getContext(), get_tmp_attr().c_str()));
            }
            
            rewriter.replaceOp(op, newOp->getResults());
            return success();
        }
    };

    struct CpuRfWriteConversion : public OpConversionPattern<aps::CpuRfWrite> {
        using OpConversionPattern<aps::CpuRfWrite>::OpConversionPattern;

        LogicalResult
        matchAndRewrite(aps::CpuRfWrite op, aps::CpuRfWrite::Adaptor adaptor,
                        ConversionPatternRewriter &rewriter) const override {
            // Clone the operation and add timing attributes
            rewriter.setInsertionPoint(op);
            auto operands = adaptor.getOperands();
            // CpuRfWrite expects two operands: rd (address) and value (data)
            assert(operands.size() == 2 && "CpuRfWrite expects 2 operands");
            auto newOp = rewriter.create<aps::CpuRfWrite>(
                op.getLoc(), operands[0], operands[1]);
            
            // Copy existing attributes
            for (auto attr : op->getAttrs()) {
                newOp->setAttr(attr.getName(), attr.getValue());
            }
            
            // Add timing attributes if not present
            if (!op->hasAttr("starttime")) {
                newOp->setAttr("starttime",
                    mlir::IntegerAttr::get(
                        mlir::IntegerType::get(getContext(), 32, mlir::IntegerType::Signless), 0));
            }
            if (!op->hasAttr("endtime")) {
                newOp->setAttr("endtime",
                    mlir::IntegerAttr::get(
                        mlir::IntegerType::get(getContext(), 32, mlir::IntegerType::Signless), 0));
            }
            
            // Add dump attribute if not present
            if (!op->hasAttr("dump")) {
                newOp->setAttr("dump", StringAttr::get(rewriter.getContext(), get_tmp_attr().c_str()));
            }
            
            rewriter.replaceOp(op, newOp->getResults());
            return success();
        }
    };

    struct IndexCastOpRemoval : public OpConversionPattern<IndexCastOp> {
        using OpConversionPattern<IndexCastOp>::OpConversionPattern;

        LogicalResult
        matchAndRewrite(IndexCastOp op, IndexCastOp::Adaptor adaptor,
                        ConversionPatternRewriter &rewriter) const override {
            op.getResult().replaceAllUsesWith(op.getOperand());
            rewriter.eraseOp(op);
            return success();
        }

    };

    struct ConstIndexConversion : public OpConversionPattern<ConstantOp> {
        using OpConversionPattern<ConstantOp>::OpConversionPattern;

        LogicalResult
        matchAndRewrite(ConstantOp op, ConstantOp::Adaptor adaptor,
                        ConversionPatternRewriter &rewriter) const override {
            if (isa<IndexType>(op.getResult().getType())) {
                auto value = adaptor.getValue();
                rewriter.setInsertionPoint(op);
                auto newOp = rewriter.create<ConstantIntOp>(
                        op->getLoc(), dyn_cast<IntegerAttr>(value).getInt(), 32);
                //FIXME: ???
                if (!op->hasAttr("dump")) {
                    op->setAttr("dump", StringAttr::get(rewriter.getContext(), get_tmp_attr().c_str()));
                }
                newOp->setAttr("dump", op->getAttr("dump"));
                myReplaceOp(op, newOp, rewriter);
                return success();
            }

            return failure();
        }
    };

    struct IfOpConversion : public OpConversionPattern<scf::IfOp> {
        using OpConversionPattern<scf::IfOp>::OpConversionPattern;

        LogicalResult
        matchAndRewrite(scf::IfOp op, scf::IfOp::Adaptor adaptor,
                        ConversionPatternRewriter &rewriter) const override {
            auto operands = adaptor.getOperands();
            for (auto opr: operands)
                if (isa<IndexType>(opr.getType()))
                    return failure();

            rewriter.setInsertionPoint(op);

            SmallVector<Type, 4> resultTypes(op.getResultTypes());

            for (auto &type: resultTypes)
                if (isa<IndexType>(type))
                    type = IntegerType::get(getContext(), 32);

            auto newOp = rewriter.create<tor::IfOp>(op.getLoc(), resultTypes, operands[0], 0, 0);
            if (!op->hasAttr("dump")) {
                op->setAttr("dump", StringAttr::get(rewriter.getContext(), get_tmp_attr().c_str()));
            }
            newOp->setAttr("dump", op->getAttr("dump"));

            rewriter.createBlock(&newOp.getThenRegion());
            rewriter.inlineRegionBefore(op.getThenRegion(), &newOp.getThenRegion().back());
            rewriter.eraseBlock(&newOp.getThenRegion().back());

            if (!op.getElseRegion().empty()) {
                rewriter.createBlock(&newOp.getElseRegion());
                rewriter.inlineRegionBefore(op.getElseRegion(), &newOp.getElseRegion().back());
                rewriter.eraseBlock(&newOp.getElseRegion().back());
            }

            rewriter.replaceOp(op, newOp.getResults());

            return success();
        }
    };

    struct WhileOpConversion : public OpConversionPattern<scf::WhileOp> {
        using OpConversionPattern<scf::WhileOp>::OpConversionPattern;

        LogicalResult
        matchAndRewrite(scf::WhileOp op, scf::WhileOp::Adaptor adaptor,
                        ConversionPatternRewriter &rewriter) const override {
            auto operands = adaptor.getOperands();
            for (auto opr: operands)
                if (isa<IndexType>(opr.getType()))
                    return failure();

            SmallVector<Type, 4> resultTypes(op.getResultTypes());
            for (auto &type: resultTypes)
                if (isa<IndexType>(type))
                    type = IntegerType::get(getContext(), 32);

            rewriter.setInsertionPoint(op);
            auto newOp = rewriter.create<tor::WhileOp>(op.getLoc(), resultTypes, operands, 0, 0);

            newOp->setAttrs(op->getAttrDictionary());
            newOp->setAttr("starttime",
                           mlir::IntegerAttr::get(
                                   mlir::IntegerType::get(getContext(), 32,
                                                          mlir::IntegerType::Signless),
                                   0));
            newOp->setAttr("endtime",
                           mlir::IntegerAttr::get(
                                   mlir::IntegerType::get(getContext(), 32,
                                                          mlir::IntegerType::Signless),
                                   0));

            rewriter.inlineRegionBefore(op.getBefore(), newOp.getBefore(), newOp.getBefore().begin());
            rewriter.inlineRegionBefore(op.getAfter(), newOp.getAfter(), newOp.getAfter().begin());
            rewriter.replaceOp(op, newOp.getResults());

            return success();
        }
    };

    struct ForOpConversion : public OpConversionPattern<scf::ForOp> {
        ForOpConversion(TypeConverter &typeConverter, MLIRContext *ctx, bool pipeline)
                : OpConversionPattern<scf::ForOp>(typeConverter, ctx), pipeline(pipeline) {}

        LogicalResult
        matchAndRewrite(scf::ForOp op, scf::ForOp::Adaptor adaptor,
                        ConversionPatternRewriter &rewriter) const override {
            auto operands = adaptor.getOperands();
            for (auto opr: operands)
                if (isa<IndexType>(opr.getType()))
                    return failure();

            rewriter.setInsertionPoint(op);
            auto upperBound = rewriter.create<SubIOp>(op.getLoc(), operands[1], operands[2]);
            upperBound->setAttr("dump", StringAttr::get(rewriter.getContext(), get_opertion_attr().c_str()));

            auto newOp = rewriter.create<tor::ForOp>(
                    op.getLoc(), operands[0], upperBound.getResult(), operands[2],
                    mlir::IntegerAttr::get(
                            mlir::IntegerType::get(getContext(), 32,
                                                   mlir::IntegerType::Signless),
                            0),
                    mlir::IntegerAttr::get(
                            mlir::IntegerType::get(getContext(), 32,
                                                   mlir::IntegerType::Signless),
                            0),
                    ValueRange(operands.drop_front(3)));
            addHlsAttrWithNewOp(newOp, op);

            newOp->setAttrs(op->getAttrDictionary());
            newOp->setAttr("starttime",
                           mlir::IntegerAttr::get(
                                   mlir::IntegerType::get(getContext(), 32,
                                                          mlir::IntegerType::Signless),
                                   0));
            newOp->setAttr("endtime",
                           mlir::IntegerAttr::get(
                                   mlir::IntegerType::get(getContext(), 32,
                                                          mlir::IntegerType::Signless),
                                   0));

            rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                                        newOp.getRegion().begin());

            for (auto pair: llvm::zip(newOp.getBody()->getArguments(),
                                      newOp.getBody()->getArgumentTypes()))
                if (isa<IndexType>(std::get<1>(pair)))
                    std::get<0>(pair).setType(IntegerType::get(getContext(), 32));

            rewriter.replaceOp(op, newOp.getResults());
            if (pipeline) {
                bool findLoop = false;
                newOp.walk([&](Operation *op) {
                    if (op == newOp) return;
                    if (isa<tor::ForOp, scf::ForOp, tor::WhileOp, scf::WhileOp>(op)) {
                        findLoop = true;
                    }
                });
                if (!findLoop) {
                    newOp->setAttr("II", IntegerAttr::get(IntegerType::get(getContext(), 32), 1));
                    newOp->setAttr("pipeline", IntegerAttr::get(IntegerType::get(getContext(), 32), 1));
                }
            }

            return success();
        }

        bool pipeline;
    };

    struct CallOpConversion : public OpConversionPattern<func::CallOp> {
        using OpConversionPattern<func::CallOp>::OpConversionPattern;

        LogicalResult
        matchAndRewrite(func::CallOp op, func::CallOp::Adaptor adaptor,
                        ConversionPatternRewriter &rewriter) const override {
            auto operands = adaptor.getOperands();
            for (auto opr: operands)
                if (isa<IndexType>(opr.getType()))
                    return failure();

            rewriter.setInsertionPoint(op);
            auto newOp = rewriter.create<tor::CallOp>(op.getLoc(), op.getResultTypes(),
                                                      op.getCallee(), 0, 0, operands);
            if (!op->hasAttr("dump")) {
                op->setAttr("dump", StringAttr::get(rewriter.getContext(), get_tmp_attr().c_str()));
            }
            newOp->setAttr("dump", op->getAttr("dump"));
            rewriter.replaceOp(op, newOp.getResults());
            return success();
        }
    };

    struct CmpIOpConversion : public OpConversionPattern<CmpIOp> {
        using OpConversionPattern<CmpIOp>::OpConversionPattern;

        LogicalResult
        matchAndRewrite(CmpIOp op, CmpIOp::Adaptor adaptor,
                        ConversionPatternRewriter &rewriter) const override {
            auto operands = adaptor.getOperands();
            assert(operands.size() == 2 && "addi has two operand");

            for (auto opr: operands)
                if (isa<IndexType>(opr.getType()))
                    return failure();

            rewriter.setInsertionPoint(op);
            auto predicate = static_cast<mlir::tor::CmpIPredicate>(op.getPredicate());
            auto newOp = rewriter.create<tor::CmpIOp>(
                    op.getLoc(), op.getResult().getType(), predicate, operands[0],
                    operands[1], 0, 0);
            if (!op->hasAttr("dump")) {
                op->setAttr("dump", StringAttr::get(rewriter.getContext(), get_tmp_attr().c_str()));
            }
            newOp->setAttr("dump", op->getAttr("dump"));

            myReplaceOp(op, newOp, rewriter);

            return success();
        }
    };

    struct CmpFOpConversion : public OpConversionPattern<CmpFOp> {
        using OpConversionPattern<CmpFOp>::OpConversionPattern;

        LogicalResult
        matchAndRewrite(CmpFOp op, CmpFOp::Adaptor adaptor,
                        ConversionPatternRewriter &rewriter) const override {
            auto operands = adaptor.getOperands();
            assert(operands.size() == 2 && "cmpf has two operand");

            rewriter.setInsertionPoint(op);
            auto predicate = static_cast<mlir::tor::CmpFPredicate>(op.getPredicate());
            auto newOp = rewriter.create<tor::CmpFOp>(
                    op.getLoc(), op.getResult().getType(), predicate, operands[0],
                    operands[1], 0, 0);

            if (!op->hasAttr("dump")) {
                op->setAttr("dump", StringAttr::get(rewriter.getContext(), get_tmp_attr().c_str()));
            }
            newOp->setAttr("dump", op->getAttr("dump"));

            myReplaceOp(op, newOp, rewriter);

            return success();
        }
    };

    struct FuncOpPattern : public OpConversionPattern<tor::FuncOp> {
        using OpConversionPattern<tor::FuncOp>::OpConversionPattern;

        LogicalResult
        matchAndRewrite(tor::FuncOp op, tor::FuncOp::Adaptor adaptor,
                        ConversionPatternRewriter &rewriter) const override {
            SmallVector<Type, 4> newInputTypes;
            for (auto type: op.getFunctionType().getInputs())
                if (isa<IndexType>(type))
                    newInputTypes.push_back(IntegerType::get(getContext(), 32));
                else
                    newInputTypes.push_back(type);

            rewriter.modifyOpInPlace(op, [&] {
                for (auto arg: op.getArguments())
                    if (isa<IndexType>(arg.getType()))
                        arg.setType(IntegerType::get(getContext(), 32));
                op.setType(FunctionType::get(getContext(), newInputTypes,
                                             op.getFunctionType().getResults()));
            });

            return success();
        }
    };

    struct MoveConstantUp : OpRewritePattern<ConstantOp> {
        using OpRewritePattern<ConstantOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(ConstantOp op, PatternRewriter &rewriter) const override {
            if (llvm::isa<tor::DesignOp>(op->getParentOp()))
                return failure();
            auto topParent = op->getParentOfType<tor::DesignOp>();

            assert(topParent);

            rewriter.setInsertionPoint(&(topParent.getBody().front()), topParent.getBody().front().begin());

            auto newOp = rewriter.clone(*op.getOperation());

            rewriter.replaceOp(op, newOp->getResults());

            return success();
        }
    };

    struct GenerateGuardedStore : public OpRewritePattern<tor::DesignOp> {
        using OpRewritePattern<tor::DesignOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(tor::DesignOp design, PatternRewriter &rewriter) const override {
            design.walk([&](scf::IfOp ifOp) {
              if (ifOp->getAttr("need_guard") != rewriter.getBoolAttr(true)) {
                return;
              }
              auto ifOpPos = Block::iterator(ifOp);
              Block &thenBlock = ifOp.getThenRegion().front();
              for (auto &op : llvm::make_early_inc_range(thenBlock.without_terminator())) {
                if (!llvm::isa<tor::AXIReadOp, tor::AXIWriteOp>(op))
                    op.moveBefore(ifOp->getBlock(), ifOpPos);
                if (auto storeOp = llvm::dyn_cast<tor::StoreOp>(op)) {
                  rewriter.setInsertionPoint(storeOp);
                  auto guardedStoreOp = rewriter.create<tor::GuardedStoreOp>(storeOp->getLoc(), 
                      storeOp.getValue(), storeOp.getMemref(), ifOp.getCondition(), 0, 0, storeOp.getIndices());
                  if (storeOp->getAttr("dump")) {
                    guardedStoreOp->setAttr("dump", storeOp->getAttr("dump"));
                  }
                  rewriter.replaceOp(storeOp, guardedStoreOp->getResults());
                }
              }
            });
            return success();
        }
    };

    struct CovertAllocToAxiCreate : public OpRewritePattern<tor::DesignOp> {
        using OpRewritePattern<tor::DesignOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(tor::DesignOp design, PatternRewriter &rewriter) const override {
            SmallVector<tor::AllocOp> allocOps;
            design.walk([&](tor::AllocOp allocOp) {
                if (!allocOp->hasAttr("mode")) {
                    return;
                }
                auto mode = dyn_cast<StringAttr>(allocOp->getAttr("mode")).getValue().str();
                if (mode != "m_axi") {
                    // interface failed todo
                    return;
                }
                allocOps.push_back(allocOp);
            });
            for (auto allocOp: allocOps) {
                rewriter.setInsertionPoint(allocOp);
                auto axiCreateOp =
                    rewriter.create<tor::AXICreateOp>(allocOp->getLoc(), allocOp->getResultTypes(),
                                                      allocOp->getOperands(), allocOp->getAttrs());
                for (auto user: llvm::make_early_inc_range(allocOp->getUsers())) {
                    rewriter.setInsertionPoint(user);
                    if (auto storeOp = llvm::dyn_cast<tor::StoreOp>(user)) {
                        auto axiWriteOp =
                            rewriter.create<tor::AXIWriteOp>(storeOp->getLoc(), storeOp->getResultTypes(),
                                storeOp->getOperands(), storeOp->getAttrs());
                        user->replaceAllUsesWith(axiWriteOp->getResults());
                    } else if (auto loadOp = llvm::dyn_cast<tor::LoadOp>(user)) {
                        auto axiReadOp =
                            rewriter.create<tor::AXIReadOp>(loadOp->getLoc(), loadOp->getResultTypes(),
                                loadOp->getOperands(), loadOp->getAttrs());
                        user->replaceAllUsesWith(axiReadOp->getResults());
                    } else if (auto guardStoreOp = llvm::dyn_cast<tor::GuardedStoreOp>(user)) {
                        SmallVector<Type, 4> emptyTypes;
                        auto ifOp = rewriter.create<tor::IfOp>(guardStoreOp.getLoc(), emptyTypes, guardStoreOp.getGuard(), 0, 0);
                        rewriter.createBlock(&ifOp.getThenRegion());
                        rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
                        SmallVector<mlir::Value, 4> operands =
                            {guardStoreOp.getValue(), guardStoreOp.getMemref(), guardStoreOp.getIndices().front()};
                        auto axiWriteOp =
                            rewriter.create<tor::AXIWriteOp>(guardStoreOp->getLoc(), guardStoreOp->getResultTypes(),
                                operands, guardStoreOp->getAttrs());
                        auto yieldOp = rewriter.create<tor::YieldOp>(guardStoreOp->getLoc());
                        user->replaceAllUsesWith(axiWriteOp->getResults());
                    } else {
                        llvm::dbgs() << *user << "\n";
                        llvm::dbgs().flush();
                        assert(false && "unknown user op type of alloc op");
                    }
                    rewriter.eraseOp(user);
                }
                allocOp->replaceAllUsesWith(axiCreateOp->getResults());
                rewriter.eraseOp(allocOp);
            }
            return success();
        }
    };

    class IndexTypeConverter : public TypeConverter {
    public:
        IndexTypeConverter() {
            addConversion([](Type type) { return type; });
            addConversion(convertIndexType);
            auto addUnrealizedCast = [](OpBuilder &builder, Type type, ValueRange inputs, Location loc) {
                // Don't insert unrealized cast if types already match
                if (inputs.size() == 1 && inputs[0].getType() == type) {
                    return inputs[0];
                }
                auto cast = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
                return cast.getResult(0);
            };
            addSourceMaterialization(addUnrealizedCast);
            addTargetMaterialization(addUnrealizedCast);
        }

        static Type convertIndexType(Type type) {
            if (isa<IndexType>(type)) {
                return IntegerType::get(type.getContext(), 32);
            }
            return type;
        }
    };

    struct MoveWhileOp : public OpRewritePattern<tor::DesignOp> {
        using OpRewritePattern<tor::DesignOp>::OpRewritePattern;

        LogicalResult
        matchAndRewrite(tor::DesignOp design, PatternRewriter &rewriter) const override {
            if (design->getAttr("move-while"))
                return failure();
            design->setAttr("move-while", IntegerAttr::get(IntegerType::get(getContext(), 32), 1));
            SmallVector<tor::WhileOp> ops;
            design.walk([&](tor::WhileOp op) { ops.push_back(op); });
            for (auto &op: ops) {
                if (isa<tor::ConditionOp>(op.getBefore().front().begin())) continue;
                rewriter.setInsertionPoint(op);
                auto tmp_op = cast<tor::WhileOp>(rewriter.clone(*op));
                rewriter.setInsertionPointAfter(op);
                auto tmp_op_2 = cast<tor::WhileOp>(rewriter.clone(*op));
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
                    if (isa<tor::ConditionOp>(sop)) continue;
                    sop.getResults().replaceAllUsesWith(rewriter.clone(sop)->getResults());
                }

                auto cond = cast<tor::ConditionOp>(op.getBefore().front().getTerminator());
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
                rewriter.create<tor::ConditionOp>(op.getLoc(), new_block->getArgument(0), values);
                op->setOperands(tmp_op.getBefore().front().getTerminator()->getOperands());
                op.getBefore().front().erase();
                rewriter.eraseOp(tmp_op);

                rewriter.setInsertionPoint(op);
                tmp_op = tmp_op_2;
                auto yield = cast<tor::YieldOp>(op.getAfter().front().getTerminator());
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
                    if (isa<tor::ConditionOp>(sop)) continue;
                    sop.getResults().replaceAllUsesWith(rewriter.clone(sop)->getResults());
                }
                rewriter.create<tor::YieldOp>(op.getLoc(), tmp_op.getBefore().front().getTerminator()->getOperands());
                rewriter.eraseOp(yield);
                rewriter.eraseOp(tmp_op);
            }
            return success();
        }
    };

    void hoistIfRegion(mlir::Region &region, tor::IfOp ifOp, PatternRewriter &rewriter) {
        for (auto [operand, result] : llvm::zip(
                llvm::dyn_cast<tor::YieldOp>(region.back().getTerminator()).getOperands(),
                ifOp.getResults())) {
            rewriter.replaceAllUsesWith(result, operand); 
        }
        rewriter.eraseOp(region.back().getTerminator());
        rewriter.inlineBlockBefore(&region.front(), ifOp);
        rewriter.eraseOp(ifOp);
    }

    struct HoistConstCondIfOp : public OpRewritePattern<tor::DesignOp> {
        using OpRewritePattern<tor::DesignOp>::OpRewritePattern;       

        LogicalResult
        matchAndRewrite(tor::DesignOp design, PatternRewriter &rewriter) const override {
            if (design->getAttr("HoistConstCondIfOp"))
                return failure();
            design->setAttr("HoistConstCondIfOp", IntegerAttr::get(IntegerType::get(getContext(), 32), 1));
            SmallVector<tor::IfOp> ifOps;
            design.walk([&](tor::IfOp op) { ifOps.push_back(op); });
            for (auto &ifOp: ifOps) {
                if (ifOp.getCondition().getDefiningOp() == nullptr) {
                    // means cond is iter args, will change overtime
                    continue;
                }
                if (auto constant =
                        dyn_cast<arith::ConstantIntOp>(ifOp.getCondition().getDefiningOp())) {
                    if (!constant.value()) { // false
                        assert(!ifOp.getElseRegion().empty() && "if false no else region, then no yeild");
                        // hoist else region above if and delete if
                        hoistIfRegion(ifOp.getElseRegion(), ifOp, rewriter);
                    } else { // true
                        // hoist then region above if and delete if
                        hoistIfRegion(ifOp.getThenRegion(), ifOp, rewriter);
                    }
                }
            }

            return success();
        }
    };

    bool isConstOp(mlir::Operation* op) {
        if (op == nullptr)
            return false;

        return llvm::isa<arith::ConstantIndexOp>(op) || llvm::isa<arith::ConstantIntOp>(op);
    }

    struct ConvertMuliToMulConst : public OpRewritePattern<tor::DesignOp> {
        using OpRewritePattern<tor::DesignOp>::OpRewritePattern;

        LogicalResult
        matchAndRewrite(tor::DesignOp design, PatternRewriter &rewriter) const override {
            SmallVector<tor::MulIOp> mulIOps;
            design.walk([&](tor::MulIOp op) { mulIOps.push_back(op); });
            for (auto mulIOp: mulIOps) {
                auto lhs = mulIOp.getOperand(0);
                auto rhs = mulIOp.getOperand(1);
                mlir::Operation *lhsDefOp = lhs.getDefiningOp();
                mlir::Operation *rhsDefOp = rhs.getDefiningOp();
                rewriter.setInsertionPoint(mulIOp);
                if (isConstOp(lhsDefOp)) {
                    auto newOp = rewriter.create<tor::MulIConstOp>(mulIOp->getLoc(), rhs, lhs, mulIOp->getAttrs());
                    rewriter.replaceAllUsesWith(mulIOp->getResults(), newOp->getResults());
                    rewriter.eraseOp(mulIOp);
                } else if (isConstOp(rhsDefOp)) {
                    auto newOp = rewriter.create<tor::MulIConstOp>(mulIOp->getLoc(), lhs, rhs, mulIOp->getAttrs());
                    rewriter.replaceAllUsesWith(mulIOp->getResults(), newOp->getResults());
                    rewriter.eraseOp(mulIOp);
                }
            }

            return success();
        }
    };

    struct CleanupIndexCastChain : public OpRewritePattern<IndexCastOp> {
        using OpRewritePattern<IndexCastOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(IndexCastOp indexCast, PatternRewriter &rewriter) const override {
            // Pattern 1: i32 -> unrealized_cast -> index -> index_cast -> i32
            if (isa<IndexType>(indexCast.getIn().getType()) &&
                indexCast.getResult().getType().isInteger(32)) {

                auto unrealizedCast = indexCast.getIn().getDefiningOp<UnrealizedConversionCastOp>();
                if (unrealizedCast &&
                    unrealizedCast.getInputs().size() == 1 &&
                    unrealizedCast.getResults().size() == 1) {

                    auto originalValue = unrealizedCast.getInputs()[0];
                    if (originalValue.getType().isInteger(32)) {
                        // Found the pattern! Replace index_cast with the original i32 value
                        rewriter.replaceOp(indexCast, originalValue);

                        // Clean up the unrealized cast if it's now unused
                        if (unrealizedCast.getResult(0).use_empty()) {
                            rewriter.eraseOp(unrealizedCast);
                        }
                        return success();
                    }
                }
            }

            // Pattern 2: index -> unrealized_cast -> i32 -> index_cast -> index
            if (indexCast.getIn().getType().isInteger(32) &&
                isa<IndexType>(indexCast.getResult().getType())) {

                auto unrealizedCast = indexCast.getIn().getDefiningOp<UnrealizedConversionCastOp>();
                if (unrealizedCast &&
                    unrealizedCast.getInputs().size() == 1 &&
                    unrealizedCast.getResults().size() == 1) {

                    auto originalValue = unrealizedCast.getInputs()[0];
                    if (isa<IndexType>(originalValue.getType())) {
                        // Found the reverse pattern! Replace index_cast with original index value
                        rewriter.replaceOp(indexCast, originalValue);

                        if (unrealizedCast.getResult(0).use_empty()) {
                            rewriter.eraseOp(unrealizedCast);
                        }
                        return success();
                    }
                }
            }

            // Pattern 3: Identity cast (same input and output type)
            if (indexCast.getIn().getType() == indexCast.getResult().getType()) {
                rewriter.replaceOp(indexCast, indexCast.getIn());
                return success();
            }

            return failure();
        }
    };

    struct CleanupUnrealizedCasts : public OpRewritePattern<UnrealizedConversionCastOp> {
        using OpRewritePattern<UnrealizedConversionCastOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(UnrealizedConversionCastOp castOp, PatternRewriter &rewriter) const override {
            if (castOp.getInputs().size() != 1 || castOp.getResults().size() != 1) {
                return failure();
            }

            auto input = castOp.getInputs()[0];
            auto output = castOp.getResults()[0];

            // Pattern 1: Identity cast (i32 -> i32 or index -> index)
            if (input.getType() == output.getType()) {
                rewriter.replaceOp(castOp, input);
                return success();
            }

            // Pattern 2: Chained unrealized casts (A -> B -> A)
            if (auto prevCast = input.getDefiningOp<UnrealizedConversionCastOp>()) {
                if (prevCast.getInputs().size() == 1 && prevCast.getResults().size() == 1) {
                    auto originalValue = prevCast.getInputs()[0];
                    // If we're converting back to the original type, bypass both casts
                    if (originalValue.getType() == output.getType()) {
                        rewriter.replaceOp(castOp, originalValue);
                        if (prevCast.getResult(0).use_empty()) {
                            rewriter.eraseOp(prevCast);
                        }
                        return success();
                    }
                }
            }

            // Pattern 3: index_cast followed by unrealized_cast back to original type
            // i32 -> index_cast -> index -> unrealized_cast -> i32
            if (auto indexCast = input.getDefiningOp<IndexCastOp>()) {
                auto originalValue = indexCast.getIn();
                // If we're converting back to the original type, bypass both casts
                if (originalValue.getType() == output.getType()) {
                    rewriter.replaceOp(castOp, originalValue);
                    if (indexCast.getResult().use_empty()) {
                        rewriter.eraseOp(indexCast);
                    }
                    return success();
                }
            }

            return failure();
        }
    };

    struct SCFToTORPass : SCFToTORBase<SCFToTORPass> {
        void runOnOperation() override {
            auto designOp = getOperation();

            {
                RewritePatternSet Patterns(&getContext());
                Patterns.add<GenerateGuardedStore>(&getContext());
                GreedyRewriteConfig config;
                config.setStrictness(GreedyRewriteStrictness::ExistingOps);
                if (failed(applyOpPatternsAndFold(designOp.getOperation(), std::move(Patterns), config)));
            }
            {
                ConversionTarget target(getContext());
                RewritePatternSet patterns(&getContext());
                IndexTypeConverter converter;

                target.addLegalDialect<tor::TORDialect>();
                target.addLegalDialect<aps::APSDialect>();
                
                // APS operations need timing attributes
                target.addDynamicallyLegalOp<aps::CpuRfRead>([](aps::CpuRfRead op) {
                    return op->hasAttr("starttime") && op->hasAttr("endtime");
                });
                target.addDynamicallyLegalOp<aps::CpuRfWrite>([](aps::CpuRfWrite op) {
                    return op->hasAttr("starttime") && op->hasAttr("endtime");
                });
                
                auto hasIndexType = [](Operation *op) {
                    if (llvm::any_of(op->getOperandTypes(), [&](auto tpe) { return isa<IndexType>(tpe); })) {
                        return false;
                    }
                    if (llvm::any_of(op->getResultTypes(), [&](auto tpe) { return isa<IndexType>(tpe); })) {
                        return false;
                    }
                    return true;
                };
                
                target.addDynamicallyLegalOp<DivUIOp>(hasIndexType);
                target.addDynamicallyLegalOp<DivSIOp>(hasIndexType);
                target.addDynamicallyLegalOp<RemUIOp>(hasIndexType);
                target.addDynamicallyLegalOp<RemSIOp>(hasIndexType);
                target.addDynamicallyLegalOp<SIToFPOp>(hasIndexType);
                target.addDynamicallyLegalOp<ShRSIOp>(hasIndexType);
                target.addDynamicallyLegalOp<ShRUIOp>(hasIndexType);
                target.addDynamicallyLegalOp<ShLIOp>(hasIndexType);
                target.addDynamicallyLegalOp<OrIOp>(hasIndexType);
                target.addDynamicallyLegalOp<ConstantOp>(hasIndexType);
                target.addDynamicallyLegalOp<tor::LoadOp>(hasIndexType);
                target.addDynamicallyLegalOp<tor::StoreOp>(hasIndexType);
                target.addDynamicallyLegalOp<tor::GuardedStoreOp>(hasIndexType);
                target.addDynamicallyLegalOp<SelectOp>(hasIndexType);
                target.addDynamicallyLegalOp<tor::FuncOp>([](tor::FuncOp op) {
                    for (auto type: op.getArgumentTypes())
                        if (isa<IndexType>(type))
                            return false;
                    return true;
                });
                // Keep IndexCastOp legal - don't convert it
                target.addLegalOp<IndexCastOp>();

                patterns.add<AddIOpConversion, ConstIndexConversion, MulIOpConversion,
                        SubIOpConversion, CmpIOpConversion, MulFOpConversion,
                        AddFOpConversion, SubFOpConversion, DivFOpConversion,
                        YieldOpConversion, CondOpConversion, WhileOpConversion,
                        IfOpConversion, FuncOpPattern,
                        CmpFOpConversion,
                        DivUIOpConversion, DivSIOpConversion, RemUIOpConversion, RemSIOpConversion, SIToFPOpConversion,
                        ShRSIOpConversion, ShRUIOpConversion,
                        ShiftLeftConversionPattern, OrIConversionPattern, SelectConversionPattern,
                        LoadOpConversion, StoreOpConversion, GuardedStoreOpConversion,
                        CpuRfReadConversion, CpuRfWriteConversion
                        /*MoveConstantUp, CallOpConversion*/>(converter, &getContext());

                patterns.add<ForOpConversion>(converter, &getContext(), pipeline);

                if (failed(applyPartialConversion(designOp, target, std::move(patterns)))) {
                    signalPassFailure();
                }
            }

            {
                // change tor.alloc to tor.axi_create
                // and its user "load/store" to "axi_read/axi_write"
                RewritePatternSet Patterns(&getContext());
                Patterns.add<CovertAllocToAxiCreate>(&getContext());
                GreedyRewriteConfig config;
                config.setStrictness(GreedyRewriteStrictness::ExistingOps);
                if (failed(applyOpPatternsAndFold(designOp.getOperation(), std::move(Patterns), config)));
            }

            {
                RewritePatternSet Patterns(&getContext());
                // Patterns.add<MoveWhileOp>(&getContext());
                GreedyRewriteConfig config;
                config.setStrictness(GreedyRewriteStrictness::ExistingOps);
                if (failed(applyOpPatternsAndFold(designOp.getOperation(), std::move(Patterns), config)));
            }

            // comment out because of negative impact on unroll pipeline
            // {
            //     RewritePatternSet Patterns(&getContext());
            //     Patterns.add<ConvertMuliToMulConst>(&getContext());
            //     GreedyRewriteConfig config;
            //     config.strictMode = GreedyRewriteStrictness::ExistingOps;
            //     if (failed(applyOpPatternsAndFold(designOp.getOperation(), std::move(Patterns), config))) {
            //         llvm::errs() << "ConvertMuliToMulConst fail\n";
            //     }
            // }

            {
                RewritePatternSet Patterns(&getContext());
                Patterns.add<HoistConstCondIfOp>(&getContext());
                GreedyRewriteConfig config;
                config.setStrictness(GreedyRewriteStrictness::ExistingOps);
                if (failed(applyOpPatternsAndFold(designOp.getOperation(), std::move(Patterns), config)));
            }

            {
                ConversionTarget target(getContext());
                RewritePatternSet patterns(&getContext());

                target.addLegalDialect<tor::TORDialect>();
                target.addDynamicallyLegalOp<ConstantOp>([](ConstantOp op) {
                    if (!llvm::isa<tor::DesignOp>(op->getParentOp()))
                        return false;
                    return true;
                });

                patterns.add<MoveConstantUp>(&getContext());

                if (failed(applyPartialConversion(designOp, target, std::move(patterns))))
                    signalPassFailure();
            }

            {
                // Clean up unrealized conversion casts and redundant index_cast chains
                RewritePatternSet Patterns(&getContext());
                Patterns.add<CleanupIndexCastChain, CleanupUnrealizedCasts>(&getContext());
                GreedyRewriteConfig config;
                config.setStrictness(GreedyRewriteStrictness::ExistingOps);
                (void)applyPatternsAndFoldGreedily(designOp.getOperation(), std::move(Patterns), config);
            }
        }
    };

} // namespace

namespace mlir {
    std::unique_ptr<OperationPass<tor::DesignOp>> createSCFToTORPass() {
        return std::make_unique<SCFToTORPass>();
    }

} // namespace mlir
