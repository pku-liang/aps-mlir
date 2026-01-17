#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

#include "TOR/PassDetail.h"
#include "TOR/Passes.h"

#define DEBUG_TYPE "tor-fuse"

namespace {
    using namespace mlir;

    struct GenerateMac : public OpRewritePattern<tor::DesignOp> {
        using OpRewritePattern<tor::DesignOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(tor::DesignOp designOp, PatternRewriter &rewriter) const override {
            auto number = 0;
            designOp.walk([&](tor::AddIOp addOp) {
                if (fuse<tor::AddIOp, tor::MulIOp, tor::MacIOp>(addOp, rewriter) != nullptr) {
                    number++;
                }
            });
            designOp.walk([&](tor::AddFOp addOp) {
                if (fuse<tor::AddFOp, tor::MulFOp, tor::MacFOp>(addOp, rewriter) != nullptr) {
                    number++;
                }
            });
            LLVM_DEBUG(llvm::dbgs() << "mac fuse number: " << number << "\n");
            return success();
        }

        template<typename ADD, typename MUL, typename MAC>
        MAC fuse(ADD addOp, PatternRewriter &rewriter) const {
            rewriter.setInsertionPoint(addOp);
            auto lop = addOp.getLhs().getDefiningOp();
            auto rop = addOp.getRhs().getDefiningOp();
            if (!lop || !rop) {
                return nullptr;
            }
            auto createMacAndReplace = [&](MUL mulOp, Value value) {
                auto macOp = rewriter.create<MAC>(mulOp.getLoc(), mulOp.getLhs(), mulOp.getRhs(), 
                        value);
                macOp->setAttr("starttime", rewriter.getI32IntegerAttr(0));
                macOp->setAttr("endtime", rewriter.getI32IntegerAttr(0));
                macOp->setAttr("dump", addOp->getAttr("dump"));
                rewriter.replaceOp(addOp, macOp);
                if (mulOp.getResult().use_empty()) {
                    mulOp.erase();
                }
                return macOp;
            };
            if (auto mulOp = llvm::dyn_cast<MUL>(lop)) {
                return createMacAndReplace(mulOp, rop->getResult(0));
            }
            if (auto mulOp = llvm::dyn_cast<MUL>(rop)) {
                return createMacAndReplace(mulOp, lop->getResult(0));
            }
            return nullptr;
        }
    };

    struct GenMulInt8 : public OpRewritePattern<tor::DesignOp> {
        using OpRewritePattern<tor::DesignOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(tor::DesignOp designOp, PatternRewriter &rewriter) const override {
            auto number = 0;
            designOp.walk([&](arith::ExtSIOp extsiOp) {
                rewriter.setInsertionPoint(extsiOp);
                auto inop = extsiOp.getIn().getDefiningOp();
                auto datawidth = dyn_cast<IntegerType>(extsiOp.getIn().getType()).getWidth();
                if (datawidth != 8) return;
                if (auto mulOp = llvm::dyn_cast<tor::MulIOp>(inop)) {
                    auto muli8Op = rewriter.create<tor::MulSIOp>(mulOp.getLoc(), extsiOp.getOut().getType(), mulOp.getLhs(), mulOp.getRhs(),
                                                                rewriter.getI32IntegerAttr(0), rewriter.getI32IntegerAttr(0));
                    muli8Op->setAttr("dump", extsiOp->getAttr("dump"));
                    rewriter.replaceOp(extsiOp, muli8Op);
                    if (mulOp.getResult().use_empty()) {
                        mulOp.erase();
                    }
                }
                number++;
            });
            designOp.walk([&](arith::ExtUIOp extuiOp) {
                rewriter.setInsertionPoint(extuiOp);
                auto inop = extuiOp.getIn().getDefiningOp();
                auto datawidth = dyn_cast<IntegerType>(extuiOp.getIn().getType()).getWidth();
                if (datawidth != 8) return;
                if (auto mulOp = llvm::dyn_cast<tor::MulIOp>(inop)) {
                    auto muli8Op = rewriter.create<tor::MulUIOp>(mulOp.getLoc(), extuiOp.getOut().getType(), mulOp.getLhs(), mulOp.getRhs(),
                                                                rewriter.getI32IntegerAttr(0), rewriter.getI32IntegerAttr(0));
                    muli8Op->setAttr("dump", extuiOp->getAttr("dump"));
                    rewriter.replaceOp(extuiOp, muli8Op);
                    if (mulOp.getResult().use_empty()) {
                        mulOp.erase();
                    }
                }
                number++;
            });
            LLVM_DEBUG(llvm::dbgs() << "mul int8 fuse number: " << number << "\n");
            return success();
        }
    };

    struct TORFusePass : TORFuseBase<TORFusePass> {
        void runOnOperation() override {
            auto op = getOperation().getOperation();
            GreedyRewriteConfig config;
            config.setStrictness(GreedyRewriteStrictness::ExistingOps);
            {
                RewritePatternSet Patterns(&getContext());
                Patterns.add<GenerateMac>(&getContext());
                if (failed(applyOpPatternsAndFold(op, std::move(Patterns), config))) {
                    signalPassFailure();
                }
            }
            {
                RewritePatternSet Patterns(&getContext());
                Patterns.add<GenMulInt8>(&getContext());
                if (failed(applyOpPatternsAndFold(op, std::move(Patterns), config))) {
                    signalPassFailure();
                }
            }
        }
    };
} // namespace

namespace mlir {
    std::unique_ptr<OperationPass<tor::DesignOp>> createTORFusePass() {
        return std::make_unique<TORFusePass>();
    }
} // namespace mlir
