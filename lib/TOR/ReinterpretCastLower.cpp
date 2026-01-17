#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "TOR/PassDetail.h"
#include "TOR/Passes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include "TOR/DialectCreater.h"


#define DEBUG_TYPE "lower-reinterpret-cast"


namespace {
    using namespace mlir;


    struct ReplaceReinterpretCastOpAfeterCopyOp : public OpConversionPattern<memref::ReinterpretCastOp> {
        using OpConversionPattern::OpConversionPattern;

        LogicalResult matchAndRewrite(memref::ReinterpretCastOp reinterpretCastOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {
            auto userRange = reinterpretCastOp->getResult(0).getUsers();
            Location loc = reinterpretCastOp.getLoc();
            
            for (auto afterOp : userRange) {
              if (isa<mlir::memref::CopyOp>(afterOp)) {
                SmallVector<OpFoldResult> offsets = reinterpretCastOp.getMixedOffsets();
                SmallVector<OpFoldResult> sizes = reinterpretCastOp.getMixedSizes();
                SmallVector<OpFoldResult> strides =reinterpretCastOp.getMixedStrides();
                MemRefType memRefType = dyn_cast<MemRefType>(reinterpretCastOp.getOperandTypes().front());
                auto reinterpretCastShape =  memRefType.getShape();
                std::optional<SmallVector<int64_t>> offsetsI = getConstantIntValues(offsets);
                std::optional<SmallVector<int64_t>> sizesI = getConstantIntValues(sizes);
                std::optional<SmallVector<int64_t>> stridesI = getConstantIntValues(strides);

                Value source = afterOp->getOperand(0);
                MemRefType copyMemRefType = dyn_cast<MemRefType>(afterOp->getOperandTypes().front());
                auto copyShape =  copyMemRefType.getShape();
                auto create = OpCreater(rewriter, loc);
                SmallVector<Value, 4> loopIvs;
                SmallVector<int64_t, 4> offsetV;
                size_t start_i = 0;
                auto dimCount =reinterpretCastShape.size();

                for (size_t i = start_i; i < sizesI.value().size(); ++i) {
                    auto forOp = rewriter.create<affine::AffineForOp>(loc, 0, sizesI.value()[i]);
                    loopIvs.push_back(forOp.getInductionVar());
                    rewriter.setInsertionPointToStart(forOp.getBody());
                    offsetV.push_back((int64_t)((reinterpretCastShape[i] - copyShape[i]) / 2));
                }
                
                // 计算新旧 memref 之间的索引映射
                SmallVector<Value, 4> oldIndices, newIndices;
                for (size_t i = start_i; i < sizesI->size(); ++i) {
                    Value newIndex = rewriter.create<affine::AffineApplyOp>(loc, AffineMap::get(dimCount, 0, {rewriter.getAffineDimExpr(i) + offsetV[i]}),loopIvs);
                    newIndices.push_back(newIndex);
                }

                for (size_t i = start_i; i <  sizesI->size(); ++i) {
                    Value oldIndex = rewriter.create<affine::AffineApplyOp>(loc, AffineMap::get(dimCount, 0, {rewriter.getAffineDimExpr(i)}),loopIvs);
                    oldIndices.push_back(oldIndex);
                }
                
                SmallVector<AffineExpr, 4> loadResults,storeResults;
                for(size_t i = start_i; i < sizesI->size(); ++i) {
                  loadResults.push_back(rewriter.getAffineDimExpr(i));
                  storeResults.push_back(rewriter.getAffineDimExpr(i));
                }


                auto affineLoadMap = AffineMap::get( dimCount, 0, loadResults, rewriter.getContext());
                
                auto loadedValue = rewriter.create<affine::AffineLoadOp>(loc, source, affineLoadMap, oldIndices);
                
                auto affineStoreMap = AffineMap::get( dimCount, 0, storeResults, rewriter.getContext());
                
                auto storedValue = rewriter.create<affine::AffineStoreOp>(loc, loadedValue, reinterpretCastOp->getOperand(0), affineStoreMap, newIndices);

                rewriter.eraseOp(reinterpretCastOp);
                rewriter.eraseOp(afterOp);
                return success();
              }
            }
            return failure();
        };
    };

    struct RemoveDeallocOp : public OpConversionPattern<memref::DeallocOp> {
        using OpConversionPattern::OpConversionPattern;

        LogicalResult matchAndRewrite(memref::DeallocOp deallocOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {
            rewriter.eraseOp(deallocOp);
            return success();
        };
    };

    struct ReplaceMemrefCopy : public OpRewritePattern<memref::CopyOp> {
        using OpRewritePattern::OpRewritePattern;
        
        LogicalResult matchAndRewrite(memref::CopyOp op, PatternRewriter &rewriter) const final {
            auto sourceOp = op.getSource();
            auto targetOp = op.getTarget();
            if (isa<memref::AllocOp>(targetOp.getDefiningOp())) {
                // 直接将 sourceOp 替换为 targetOp
                if (sourceOp.getType() == targetOp.getType()) {
                    rewriter.replaceOp(targetOp.getDefiningOp(), sourceOp);
                    rewriter.eraseOp(op);
                    
                } else {
                    auto loc = op.getLoc();
                    auto sourceOp = op->getOperand(0);
                    auto targetOp = op->getOperand(1);

                    auto sourceShape = dyn_cast<MemRefType>(sourceOp.getType()).getShape();
                    auto targetShape = dyn_cast<MemRefType>(targetOp.getType()).getShape();
                    auto create = OpCreater(rewriter, loc);

                    rewriter.setInsertionPoint(op);
                    SmallVector<Value, 4> loopIvs;
                    for (size_t i = 0; i <  sourceShape.size(); ++i) {
                        auto forOp = rewriter.create<affine::AffineForOp >(loc, 0, sourceShape[i]);
                        loopIvs.push_back(forOp.getInductionVar());
                        rewriter.setInsertionPointToStart(forOp.getBody());
                    }
                    
                    SmallVector<Value, 4> oldIndices, newIndices;

                    for(size_t i = 0; i < sourceShape.size(); ++i) {
                        Value oldIndex = rewriter.create<affine::AffineApplyOp>(loc, AffineMap::get(sourceShape.size(), 0, {rewriter.getAffineDimExpr(i)}),loopIvs);
                        oldIndices.push_back(oldIndex);
                    }

                    for (size_t i = 0; i <  targetShape.size(); ++i) {
                        Value newIndex = rewriter.create<affine::AffineApplyOp>(loc, AffineMap::get(targetShape.size(), 0, {rewriter.getAffineDimExpr(i)}),loopIvs);
                        newIndices.push_back(newIndex);
                    }
                    

                    SmallVector<AffineExpr, 4> loadResults;
                    for(size_t i = 0; i < sourceShape.size(); ++i) {
                    loadResults.push_back(rewriter.getAffineDimExpr(i));
                    }
                    
                    SmallVector<AffineExpr, 4> storeResults;
                    for(size_t i = 0; i < targetShape.size(); ++i) {
                        storeResults.push_back(rewriter.getAffineDimExpr(i));
                    }

                    auto affineLoadMap = AffineMap::get( sourceShape.size(), 0, loadResults, rewriter.getContext());
                    auto loadedValue = rewriter.create<affine::AffineLoadOp>(loc, sourceOp, affineLoadMap, oldIndices);
                    
                    auto affineStoreMap = AffineMap::get( targetShape.size(), 0, storeResults, rewriter.getContext());
                    auto storedValue = rewriter.create<affine::AffineStoreOp>(loc, loadedValue, targetOp, affineStoreMap, newIndices);

                    rewriter.eraseOp(op);
                }
                return success();
            }
            return failure();

        };
    };
    struct ReinterpretCastPassLower : public RemoveNoUseReinterpretCastBase<ReinterpretCastPassLower> {
        void runOnOperation() override {
            MLIRContext &ctxt = getContext();
            ConversionTarget target(ctxt);
            target.addLegalDialect<affine::AffineDialect>();
            target.addLegalOp<memref::AllocaOp>();
            RewritePatternSet patterns(&ctxt);
            patterns.add<ReplaceReinterpretCastOpAfeterCopyOp>(&ctxt);
            patterns.add<RemoveDeallocOp>(&ctxt);
            patterns.add<ReplaceMemrefCopy>(&ctxt);

            if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
                signalPassFailure();
            }
        }
    };

}; // namespace



namespace mlir {
    std::unique_ptr<Pass> createReinterpretCastPass() {
        return std::make_unique<ReinterpretCastPassLower>();
    }
} // namespace mlir


