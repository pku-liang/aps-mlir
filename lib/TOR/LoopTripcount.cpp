#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

#include "TOR/PassDetail.h"
#include "TOR/Passes.h"

#define DEBUG_TYPE "loop-tripcount"

namespace {
    using namespace mlir;
    static std::string LOOP_NUMBER_ATTR = "tripcount-max";

    struct Loop {
        affine::AffineForOp op;
        int64_t number;

        int64_t lowerBound;
        int64_t upperBound;
        int64_t index;

        Loop(affine::AffineForOp op) : op(op), number(1) {}

        void reset(int64_t lowerBound, int64_t upperBound) {
            this->lowerBound = lowerBound;
            this->upperBound = upperBound;
            this->index = lowerBound;
        }

        void increase() {
            number++;
        }

        int64_t next() {
            index += op.getStep().getSExtValue();
            return index < upperBound ? index : -1;
        }
    };
    
    struct SetLoopNumber : public OpRewritePattern<func::FuncOp> {
        using OpRewritePattern<func::FuncOp>::OpRewritePattern;
        using Dims = SmallVector<mlir::AffineExpr>;

        LogicalResult matchAndRewrite(func::FuncOp funcOp, PatternRewriter &rewriter) const override {
            auto number = 0;
            SmallVector<affine::AffineForOp> forOps;
            funcOp.walk([&](affine::AffineForOp forOp) {
                if (!forOp->getAttr(LOOP_NUMBER_ATTR)) {
                    number += computeLoop(forOp, rewriter);
                }
                if (forOp->getAttr(LOOP_NUMBER_ATTR)) {
                    forOps.push_back(forOp);
                }
            });
            for (int i = 0, r = forOps.size(); i < r; ++i) {
                auto it = std::find(forOps.begin(), forOps.end(), forOps[i]->getParentOp());
                auto value = dyn_cast<IntegerAttr>(forOps[i]->getAttr(LOOP_NUMBER_ATTR)).getInt();
                auto number = it == forOps.end() ? 1 : 
                        dyn_cast<IntegerAttr>((*it)->getAttr(LOOP_NUMBER_ATTR)).getInt();
                auto v = (int64_t) (value * 1.0 / number + 0.5);
                forOps[i]->setAttr(LOOP_NUMBER_ATTR, rewriter.getI64IntegerAttr(v));
            }
            LLVM_DEBUG(llvm::dbgs() << "loop set number: " << number << "\n");
            return success();
        }

        int computeLoop(affine::AffineForOp forOp, PatternRewriter &rewriter) const {
            SmallVector<affine::AffineForOp> forOps;
            auto currentOp = forOp.getOperation();
            while (auto op = mlir::dyn_cast<affine::AffineForOp>(currentOp)) {
                forOps.push_back(op);
                currentOp = currentOp->getParentOp();
            }
            auto checkOperands = [&]() {
                for (int i = 0, r = forOps.size(); i < r; ++i) {
                    for (auto operand : forOps[i].getLowerBoundOperands()) {
                        if (!llvm::isa<mlir::BlockArgument>(operand)) {
                            return false;
                        }
                    }
                    for (auto operand : forOps[i].getUpperBoundOperands()) {
                        if (!llvm::isa<mlir::BlockArgument>(operand)) {
                            return false;
                        }
                    }
                }
                return true;
            };
            if (!checkOperands()) {
                return 0;
            }
            
            SmallVector<bool> visited(forOps.size(), false);
            auto markVisited = [&](Operation::operand_range operands) {
                for (auto operand : operands) {
                    if (auto arg = dyn_cast<mlir::BlockArgument>(operand)) { 
                        auto op = arg.getOwner()->getParentOp();
                        auto it = std::find(forOps.begin(), forOps.end(), op);
                        if (it != forOps.end()) {
                            visited[std::distance(forOps.begin(), it)] = true;
                        }
                    } else {
                        assert(false);
                    }
                }
                return operands.size() != 0;
            };
            for (int i = 0, r = forOps.size(); i < r; ++i) {
                if (markVisited(forOps[i].getLowerBoundOperands()) || 
                    markVisited(forOps[i].getUpperBoundOperands())) {
                    visited[i] = true;
                }
            }
            SmallVector<Loop> loops;
            for (int i = 0, r = forOps.size(); i < r; ++i) {
                if (visited[i]) {
                    loops.push_back(Loop(forOps[i]));
                }
            }
            if (!loops.empty()) {
                for (auto it = loops.rbegin(); it != loops.rend(); ++it) {
                    resetLoop(*it, loops, rewriter);
                }
                int index = 0;
                while (true) {
                    int next = loops[index].next();
                    if (index == (int) loops.size() - 1 && next == -1) {
                        break;
                    }
                    if (next == -1) {
                        index++;
                    } else {
                        loops[index].increase();
                        for (int i = index - 1; i >= 0; --i) {
                            resetLoop(loops[i], loops, rewriter);
                            loops[i].increase();
                        }
                        index = 0;
                    }
                }
                for (int i = 0, r = loops.size(); i < r; ++i) {
                    loops[i].op->setAttr(LOOP_NUMBER_ATTR, rewriter.getI64IntegerAttr(loops[i].number));
                }
            }
            int64_t accumulator = 1;
            int64_t lastNumber = 1;
            SmallVector<int64_t> values(forOps.size());
            for (int i = forOps.size() - 1; i >= 0; --i) {
                auto number = forOps[i]->getAttr(LOOP_NUMBER_ATTR);
                if (!number) {
                    auto lower = getValue(forOps[i].getLowerBoundMap().getResult(0));
                    auto upper = getValue(forOps[i].getUpperBoundMap().getResult(0));
                    auto step = forOps[i].getStep().getSExtValue();
                    accumulator *= llvm::divideCeil(upper - lower + step - 1, step);
                } else {
                    lastNumber = dyn_cast<IntegerAttr>(number).getInt();
                }
                values[i] = lastNumber * accumulator;
            }
            for (size_t i = 0, r = forOps.size(); i < r; ++i) {
                forOps[i]->setAttr(LOOP_NUMBER_ATTR, rewriter.getI64IntegerAttr(values[i]));
            }
            return (int) loops.size();
        }

        void resetLoop(Loop &loop, SmallVector<Loop> loops, PatternRewriter &rewriter) const {
            auto getDims = [&](Operation::operand_range operands) {
                Dims dims;
                for (auto operand : operands) {
                    auto arg = llvm::dyn_cast<mlir::BlockArgument>(operand);
                    auto it = std::find_if(loops.begin(), loops.end(), [&](Loop loop) {
                        return loop.op == arg.getOwner()->getParentOp();
                    });
                    dims.push_back(rewriter.getAffineConstantExpr(it->index));
                }
                return dims;
            };
            auto getLowerBound = [&](affine::AffineForOp forOp, Dims dims) {
                int64_t result = INT64_MIN;
                for (auto expr : forOp.getLowerBoundMap().getResults()) {
                    result = std::max(result, getValue(expr, dims));
                }
                return result;
            };
            auto getUpperBound = [&](affine::AffineForOp forOp, Dims dims) {
                int64_t result = INT64_MAX;
                for (auto expr : forOp.getUpperBoundMap().getResults()) {
                    result = std::min(result, getValue(expr, dims));
                }
                return result;
            };
            auto lowerBound = getLowerBound(loop.op, getDims(loop.op.getLowerBoundOperands()));
            auto upperBound = getUpperBound(loop.op, getDims(loop.op.getUpperBoundOperands()));
            loop.reset(lowerBound, upperBound);
        }

        int64_t getValue(mlir::AffineExpr expr, Dims dims = Dims()) const {
            expr = dims.size() == 0 ? expr : expr.replaceDims(dims);
            if (auto value = dyn_cast<AffineConstantExpr>(expr)) {
                return value.getValue();
            }
            assert(false);
        }
    };

    struct LoopTripcountPass : LoopTripcountBase<LoopTripcountPass> {
        void runOnOperation() override {
            auto op = getOperation().getOperation();
            GreedyRewriteConfig config;
            config.setStrictness(GreedyRewriteStrictness::ExistingOps);
            {
                RewritePatternSet Patterns(&getContext());
                Patterns.add<SetLoopNumber>(&getContext());
                if (failed(applyOpPatternsAndFold(op, std::move(Patterns), config))) {
                    signalPassFailure();
                }
            }
        }
    };
} // namespace

namespace mlir {
    std::unique_ptr<OperationPass<mlir::func::FuncOp>> createLoopTripcountPass() {
        return std::make_unique<LoopTripcountPass>();
    }
} // namespace mlir
