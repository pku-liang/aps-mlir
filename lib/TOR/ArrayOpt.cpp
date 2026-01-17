#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

#include "TOR/PassDetail.h"
#include "TOR/Passes.h"
#include "TOR/TORDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "array-opt"

namespace {
using namespace mlir;
using namespace mlir::func;
using namespace affine;

bool isPromotableUseType(Type type) {
  auto memRefType = dyn_cast<mlir::MemRefType>(type);
  if (!memRefType) {
    return false;
  }
  auto memrefDimSize = memRefType.getShape().size();
  if (memrefDimSize == 1) {
    return false;
  }
  for (auto dim : memRefType.getShape()) {
    if (dim == 1) {
      return true;
    }
  }
  return false;
}

bool isPromotable(Value AI) {
  if (!isPromotableUseType(AI.getType())) {
    return false;
  }
  for (auto *op : AI.getUsers()) {
    if (!isa<AffineReadOpInterface, AffineWriteOpInterface, memref::LoadOp,
             memref::StoreOp, func::CallOp>(op)) {
      return false;
    }
  }
  return true;
}

template <typename T>
void getNewAffineMap(T loadOrStore, AffineMap &newMap,
                     SmallVector<bool> &optDimArray,
                     PatternRewriter &rewriter) {
  rewriter.setInsertionPoint(loadOrStore);
  SmallVector<AffineExpr> newExprs;
  auto map = loadOrStore.getAffineMap();
  for (unsigned i = 0, e = map.getNumResults(); i < e; i++) {
    if (optDimArray[i]) {
      newExprs.push_back(map.getResult(i));
    }
  }
  if (newExprs.size() == 0) {
    newExprs.push_back(getAffineConstantExpr(0, rewriter.getContext()));
  }
  newMap = AffineMap::get(map.getNumDims(), map.getNumSymbols(), newExprs,
                          rewriter.getContext());
}

void repalaceLoad(AffineLoadOp load, Value handleMemref,
                  SmallVector<bool> &optDimArray, PatternRewriter &rewriter) {
  AffineMap newMap;
  getNewAffineMap(load, newMap, optDimArray, rewriter);
  rewriter.replaceOpWithNewOp<AffineLoadOp>(load, handleMemref, newMap,
                                            load.getMapOperands());
}

void repalaceStore(AffineStoreOp store, Value handleMemref,
                   SmallVector<bool> &optDimArray, PatternRewriter &rewriter) {
  AffineMap newMap;
  getNewAffineMap(store, newMap, optDimArray, rewriter);
  rewriter.replaceOpWithNewOp<AffineStoreOp>(store, store.getValueToStore(),
                                             handleMemref, newMap,
                                             store.getMapOperands());
}

template <typename T>
void getNewIndices(T loadOrStore, SmallVector<Value> &newIndices,
                   SmallVector<bool> &optDimArray, PatternRewriter &rewriter) {
  rewriter.setInsertionPoint(loadOrStore);
  auto indices = loadOrStore.getIndices();
  for (unsigned i = 0, e = indices.size(); i < e; i++) {
    if (optDimArray[i]) {
      newIndices.push_back(indices[i]);
    }
  }
  if (newIndices.size() == 0) {
    newIndices.push_back(
        rewriter.create<arith::ConstantIndexOp>(loadOrStore->getLoc(), 0));
  }
}

void repalaceMemrefLoad(memref::LoadOp load, Value handleMemref,
                        SmallVector<bool> &optDimArray,
                        PatternRewriter &rewriter) {
  SmallVector<Value> newIndices;
  getNewIndices(load, newIndices, optDimArray, rewriter);
  rewriter.replaceOpWithNewOp<memref::LoadOp>(load, handleMemref, newIndices);
}

void repalaceMemrefStore(memref::StoreOp store, Value handleMemref,
                         SmallVector<bool> &optDimArray,
                         PatternRewriter &rewriter) {
  SmallVector<Value> newIndices;
  getNewIndices(store, newIndices, optDimArray, rewriter);
  rewriter.replaceOpWithNewOp<memref::StoreOp>(store, store.getValueToStore(),
                                               handleMemref, newIndices);
}

void replaceMemrefAccess(Value oldMemref, Value newMemref,
                         SmallVector<bool> &optDimArray,
                         PatternRewriter &rewriter) {
  for (auto &u : llvm::make_early_inc_range(oldMemref.getUses())) {
    auto op = u.getOwner();
    if (auto load = dyn_cast<AffineLoadOp>(op)) {
      repalaceLoad(load, newMemref, optDimArray, rewriter);
    } else if (auto store = dyn_cast<AffineStoreOp>(op)) {
      repalaceStore(store, newMemref, optDimArray, rewriter);
    } else if (auto load = dyn_cast<memref::LoadOp>(op)) {
      repalaceMemrefLoad(load, newMemref, optDimArray, rewriter);
    } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
      repalaceMemrefStore(store, newMemref, optDimArray, rewriter);
    } else if (isa<func::CallOp>(op)) {
      op->setOperand(u.getOperandNumber(), newMemref);
    } else {
      LLVM_DEBUG(llvm::dbgs() << *op << "\n");
      llvm_unreachable("replaceMemrefAccess with unkown op");
    }
  }
}

MemRefType getNewMemref(Type type) {
  SmallVector<int64_t> newShape;
  auto memref = cast<MemRefType>(type);
  for (unsigned rank = 0; rank < memref.getRank(); ++rank) {
    if (memref.getShape()[rank] != 1) {
      newShape.push_back(memref.getShape()[rank]);
    }
  }
  if (newShape.size() == 0) {
    newShape.push_back(1);
  }
  return MemRefType::get(newShape, memref.getElementType());
}

void getOptDimArray(Type type, SmallVector<bool> &optDimArray) {
  auto memref = cast<MemRefType>(type);
  for (unsigned rank = 0; rank < memref.getRank(); ++rank) {
    optDimArray.push_back(memref.getShape()[rank] != 1);
  }
}

struct AllocaOptPattern : OpRewritePattern<memref::AllocaOp> {
  AllocaOptPattern(MLIRContext *ctx)
      : OpRewritePattern<memref::AllocaOp>(ctx) {}

  LogicalResult matchAndRewrite(memref::AllocaOp op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr("array-opt"))
      return failure();
    op->setAttr("array-opt",
                IntegerAttr::get(IntegerType::get(getContext(), 32), 1));
    auto arg = op->getResult(0);
    auto allocOp = rewriter.create<memref::AllocaOp>(
        op->getLoc(), getNewMemref(arg.getType()));
    SmallVector<bool> optDimArray;
    getOptDimArray(arg.getType(), optDimArray);
    replaceMemrefAccess(op->getResult(0), allocOp->getResult(0), optDimArray,
                        rewriter);
    return success();
  }
};

struct FuncOpPattern : OpRewritePattern<FuncOp> {
  FuncOpPattern(MLIRContext *ctx) : OpRewritePattern<FuncOp>(ctx) {}

  LogicalResult matchAndRewrite(FuncOp op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr("array-opt"))
      return failure();
    op->setAttr("array-opt",
                IntegerAttr::get(IntegerType::get(getContext(), 32), 1));
    for (int i = op.getArguments().size() - 1; i >= 0; i--) {
      auto arg = op.getArgument(i);
      if (!isa<mlir::MemRefType>(arg.getType()) || !isPromotable(arg)) {
        continue;
      }
      op.insertArgument(i + 1, getNewMemref(arg.getType()), {}, op.getLoc());
      SmallVector<bool> optDimArray;
      getOptDimArray(arg.getType(), optDimArray);
      replaceMemrefAccess(arg, op.getArgument(i + 1), optDimArray, rewriter);
      op.eraseArgument(i);
    }
    return success();
  }
};

struct GlobalOpPattern : OpRewritePattern<memref::GlobalOp> {
  DenseMap<StringRef, SmallVector<memref::GetGlobalOp>> &newGetGlobalOpMap;
  GlobalOpPattern(
      MLIRContext *ctx,
      DenseMap<StringRef, SmallVector<memref::GetGlobalOp>> &newGetGlobalOpMap)
      : OpRewritePattern<memref::GlobalOp>(ctx),
        newGetGlobalOpMap(newGetGlobalOpMap) {}

  LogicalResult matchAndRewrite(memref::GlobalOp op,
                                PatternRewriter &rewriter) const override {

    if (op->hasAttr("array-opt"))
      return failure();
    auto mr = getNewMemref(op.getType());
    Attribute initValue = rewriter.getUnitAttr();
    if (op.getInitialValue().has_value()) {
      if (auto elementsAttr =
              llvm::dyn_cast<DenseElementsAttr>(op.getInitialValue().value())) {
        auto tensorType = RankedTensorType::get(
            mr.getShape(), elementsAttr.getType().getElementType());
        initValue = DenseElementsAttr::getFromRawBuffer(
            tensorType, elementsAttr.getRawData());
      }
    }
    rewriter.create<memref::GlobalOp>(
        op.getLoc(), rewriter.getStringAttr(op.getSymName()),
        /*sym_visibility*/ mlir::StringAttr(), mlir::TypeAttr::get(mr),
        initValue, mlir::UnitAttr(),
        /*alignment*/ nullptr);
    SmallVector<bool> optDimArray;
    getOptDimArray(op.getType(), optDimArray);
    auto getGlobalOpArray = newGetGlobalOpMap[op.getSymName()];
    for (auto getGlobalOp : getGlobalOpArray) {
      rewriter.setInsertionPoint(getGlobalOp);
      auto newGetGlobalOp = rewriter.create<memref::GetGlobalOp>(getGlobalOp->getLoc(), mr, op.getSymName());
      replaceMemrefAccess(getGlobalOp, newGetGlobalOp, optDimArray, rewriter);
      getGlobalOp->erase();
    }
    op->erase();
    return success();
  }
};

struct ArrayOptPass : ArrayOptBase<ArrayOptPass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();
    SmallVector<Operation *> optArrays;
    GreedyRewriteConfig config;
    config.setStrictness(GreedyRewriteStrictness::ExistingOps);
    moduleOp.walk([&](memref::AllocaOp AI) {
      if (isPromotable(AI)) {
        optArrays.push_back(AI);
        RewritePatternSet patterns(&getContext());
        patterns.insert<AllocaOptPattern>(&getContext());
        (void)applyOpPatternsAndFold(AI.getOperation(), std::move(patterns),
                                     config);
      }
    });
    moduleOp.walk([&](FuncOp func) {
      RewritePatternSet patterns(&getContext());
      patterns.insert<FuncOpPattern>(&getContext());
      (void)applyOpPatternsAndFold(func.getOperation(), std::move(patterns),
                                   config);
    });
    DenseMap<StringRef, SmallVector<memref::GetGlobalOp>> newGetGlobalOpMap;
    moduleOp.walk([&](memref::GetGlobalOp getGlobalOp) {
      if (!newGetGlobalOpMap.count(getGlobalOp.getName())) {
        newGetGlobalOpMap[getGlobalOp.getName()] =
            SmallVector<memref::GetGlobalOp>();
      }
      newGetGlobalOpMap[getGlobalOp.getName()].push_back(getGlobalOp);
    });
    moduleOp.walk([&](memref::GlobalOp globalOp) {
      if (isPromotableUseType(globalOp.getType())) {
        RewritePatternSet patterns(&getContext());
        patterns.insert<GlobalOpPattern>(&getContext(), newGetGlobalOpMap);
        (void)applyOpPatternsAndFold(globalOp.getOperation(),
                                     std::move(patterns), config);
      }
    });
  }
};
} // namespace

namespace mlir {
std::unique_ptr<OperationPass<mlir::ModuleOp>> createArrayOptPass() {
  return std::make_unique<ArrayOptPass>();
}

} // namespace mlir
