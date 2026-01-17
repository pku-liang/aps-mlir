#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace {
    using namespace mlir;

    struct DialectCreater {
        PatternRewriter &rewriter;
        Location loc;

        DialectCreater(PatternRewriter &rewriter, Location loc) : rewriter(rewriter), loc(loc) {}
    };
    
    struct AffineCreater : public DialectCreater {
        using DialectCreater::DialectCreater;
        
        auto load(Value memref, ValueRange operands) {
            return rewriter.create<affine::AffineLoadOp>(loc, memref, operands);
        }

        auto load(Value memref, AffineMap map, ValueRange mapOperands) {
            return rewriter.create<affine::AffineLoadOp>(loc, memref, map, mapOperands);
        }

        auto store(Value value, Value memref, ValueRange operands) {
            return rewriter.create<affine::AffineStoreOp>(loc, value, memref, operands);
        }

        auto store(Value value, Value memref, AffineMap map, ValueRange mapOperands) {
            return rewriter.create<affine::AffineStoreOp>(loc, value, memref, map, mapOperands);
        }

        auto max(AffineMap affineMap, ValueRange mapOperands) {
            return rewriter.create<affine::AffineMaxOp>(loc, affineMap, mapOperands);
        }

        auto apply(AffineMap map, ValueRange mapOperands) {
            return rewriter.create<affine::AffineApplyOp>(loc, map, mapOperands);
        }

        auto loop(ValueRange lbOperands, AffineMap lbMap, ValueRange ubOperands, AffineMap ubMap,
                  int64_t step = 1) {
            return rewriter.create<affine::AffineForOp>(loc, lbOperands, lbMap, ubOperands, 
                    ubMap, step);
        }

        auto loop(int64_t lowerBound, int64_t upperBound, int64_t step = 1) {
            return rewriter.create<affine::AffineForOp>(loc, lowerBound, upperBound, step);
        }
    };

    struct ArithCreater : public DialectCreater {
        using DialectCreater::DialectCreater;

        auto constant(TypedAttr value) {
            return rewriter.create<arith::ConstantOp>(loc, value);
        }

        auto constantIndex(int64_t value) {
            return rewriter.create<arith::ConstantIndexOp>(loc, value);
        }

        auto indexCast(Type type, Value value) {
            return rewriter.create<arith::IndexCastOp>(loc, type, value);
        }

        auto add(Value lhs, Value rhs) {
            return isa<mlir::FloatType>(lhs.getType()) ? addf(lhs, rhs) : addi(lhs, rhs);
        }
        
        Value addf(Value lhs, Value rhs) {
            return rewriter.create<arith::AddFOp>(loc, lhs, rhs);
        }

        Value addi(Value lhs, Value rhs) {
            return rewriter.create<arith::AddIOp>(loc, lhs, rhs);
        }

        auto subf(Value lhs, Value rhs) {
            return rewriter.create<arith::SubFOp>(loc, lhs, rhs);
        }

        auto subi(Value lhs, Value rhs) {
            return rewriter.create<arith::SubIOp>(loc, lhs, rhs);
        }

        auto mul(Value lhs, Value rhs) {
            return isa<mlir::FloatType>(lhs.getType()) ? mulf(lhs, rhs) : muli(lhs, rhs);
        }

        Value mulf(Value lhs, Value rhs) {
            return rewriter.create<arith::MulFOp>(loc, lhs, rhs);
        }

        Value muli(Value lhs, Value rhs) {
            return rewriter.create<arith::MulIOp>(loc, lhs, rhs);
        }

        auto divf(Value lhs, Value rhs) {
            return rewriter.create<arith::DivFOp>(loc, lhs, rhs);
        }
        
        auto divui(Value lhs, Value rhs) {
            return rewriter.create<arith::DivUIOp>(loc, lhs, rhs);
        }

        auto uitofp(Type type, Value value) {
            return rewriter.create<arith::UIToFPOp>(loc, type, value);
        }

        auto sitofp(Type type, Value value) {
            return rewriter.create<arith::SIToFPOp>(loc, type, value);
        }

        auto fptoui(Type type, Value value) {
            return rewriter.create<arith::FPToUIOp>(loc, type, value);
        }

        auto fptosi(Type type, Value value) {
            return rewriter.create<arith::FPToSIOp>(loc, type, value);
        }

        auto extui(Type type, Value value) {
            return rewriter.create<arith::ExtUIOp>(loc, type, value);
        }

        auto extsi(Type type, Value value) {
            return rewriter.create<arith::ExtSIOp>(loc, type, value);
        }

        auto trunci(Type type, Value value) {
            return rewriter.create<arith::TruncIOp>(loc, type, value);
        }

        auto cmpf(arith::CmpFPredicate predicate, Value lhs, Value rhs) {
            return rewriter.create<arith::CmpFOp>(loc, predicate, lhs, rhs);
        }

        auto cmpi(arith::CmpIPredicate predicate, Value lhs, Value rhs) {
            return rewriter.create<arith::CmpIOp>(loc, predicate, lhs, rhs);
        }

        auto select(Value condition, Value lhs, Value rhs) {
            return rewriter.create<arith::SelectOp>(loc, condition, lhs, rhs);
        }
    };

    struct FuncCreater : public DialectCreater {
        using DialectCreater::DialectCreater;
        
        auto func(StringRef name, FunctionType type) {
            return rewriter.create<func::FuncOp>(loc, name, type);
        }

        auto ret() {
            return rewriter.create<func::ReturnOp>(loc);
        }

        auto call(func::FuncOp callee, ValueRange operands) {
            return rewriter.create<func::CallOp>(loc, callee, operands);
        }
    };

    struct MathCreater : public DialectCreater {
        using DialectCreater::DialectCreater;
        
        auto floor(Value value) {
            return rewriter.create<math::FloorOp>(loc, value);
        }
    };

    struct MemrefCreater : public DialectCreater {
        using DialectCreater::DialectCreater;
        
        auto alloca(MemRefType type) {
            return rewriter.create<memref::AllocaOp>(loc, type);
        }

        auto load(Value memref, ValueRange operands) {
            return rewriter.create<memref::LoadOp>(loc, memref, operands);
        }

        auto store(Value value, Value memref, ValueRange operands) {
            return rewriter.create<memref::StoreOp>(loc, value, memref, operands);
        }
    };

    struct ScfCreater : public DialectCreater {
        using DialectCreater::DialectCreater;
        
        auto loop(Value lowerBound, Value upperBound, Value step) {
            return rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
        }

        auto loop(int64_t lowerBound, int64_t upperBound, int64_t step) {
            auto lb = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(lowerBound));
            auto ub = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(upperBound));
            auto st = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(step));
            return loop(lb, ub, st);
        }
         
        auto cond(TypeRange resultTypes, Value cond, bool withElseRegion) {
          return rewriter.create<scf::IfOp>(loc, resultTypes, cond, withElseRegion);
        }

        auto yield() {
          return rewriter.create<scf::YieldOp>(loc);
        }
    };

    struct OpCreater : public DialectCreater {
        AffineCreater affine;
        ArithCreater arith;
        FuncCreater func;
        MathCreater math;
        MemrefCreater memref;
        ScfCreater scf;

        OpCreater(PatternRewriter &rewriter, Location loc) : DialectCreater(rewriter, loc),
                affine(rewriter, loc), arith(rewriter, loc), func(rewriter, loc), 
                math(rewriter, loc), memref(rewriter, loc), scf(rewriter, loc) {}
    };
}
