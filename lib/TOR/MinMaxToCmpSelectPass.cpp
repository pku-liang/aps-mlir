#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "TOR/PassDetail.h"
#include "TOR/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"


#define DEBUG_TYPE "min-max-to-cmp-select"


namespace {
    using namespace mlir;

    struct MinMaxConvPattern : public OpRewritePattern<ModuleOp> {
        using OpRewritePattern<ModuleOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(ModuleOp moduleOp,PatternRewriter &rewriter) const override {
            auto number = 0;
            moduleOp.walk([&](Operation *op) {
            Location loc = op->getLoc();
            if (isa<arith::MaximumFOp>(op) || isa<arith::MinimumFOp>(op)) {
                auto cmpFpredicate = mlir::arith::CmpFPredicate::OGT;
                if (isa<arith::MinimumFOp>(op)) {
                    cmpFpredicate = mlir::arith::CmpFPredicate::OLT;
                }
                rewriter.setInsertionPoint(op);
                auto predicate = rewriter.create<arith::CmpFOp>(loc, cmpFpredicate, op->getOperand(0), op->getOperand(1));
                auto selectOp = rewriter.create<arith::SelectOp>(loc, predicate, op->getOperand(0), op->getOperand(1));
                rewriter.replaceOp(op, selectOp);
                number++;
            }
          });
            LLVM_DEBUG(llvm::dbgs() << "arith.maxF/minF converted to cmp.select, number: " << number << "\n");
            return success();
        }
    };
    

    struct MinMaxToCmpSelectPass : public MinMaxToCmpSlectBase<MinMaxToCmpSelectPass> {
        void runOnOperation() override {
            auto op = getOperation().getOperation();
            RewritePatternSet patterns(&getContext());
            patterns.add<MinMaxConvPattern>(&getContext());
            GreedyRewriteConfig config;
            config.setStrictness(GreedyRewriteStrictness::ExistingOps);
            if (failed(applyOpPatternsAndFold(op, std::move(patterns), config))) {
                signalPassFailure();
            }
        }
    };

}; // namespace



namespace mlir {
    std::unique_ptr<OperationPass<mlir::ModuleOp>>createMinMaxToCmpSelectPass() {
        return std::make_unique<MinMaxToCmpSelectPass>();
    }

} // namespace mlir


