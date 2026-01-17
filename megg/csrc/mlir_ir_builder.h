#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include <memory>
#include <vector>
#include <string>

namespace megg {

class MLIRIRBuilder {
private:
    std::unique_ptr<mlir::OpBuilder> builder;
    mlir::MLIRContext* context;  //bu Non-owning pointer to context

public:
    explicit MLIRIRBuilder();
    ~MLIRIRBuilder()=default;

    // Function operations
    std::pair<mlir::func::FuncOp, mlir::Block*> createFunction(const std::string& name,
                                                               const std::vector<mlir::Type>& argTypes,
                                                               const std::vector<mlir::Type>& resultTypes);

    void setInsertionPointToStart(mlir::func::FuncOp func);
    void setInsertionPointToEnd(mlir::func::FuncOp func);
    void setInsertionPoint(mlir::Operation* op);
    void setInsertionPointAfter(mlir::Operation* op);

    // Save and restore insertion point
    mlir::OpBuilder::InsertPoint saveInsertionPoint();
    void restoreInsertionPoint(mlir::OpBuilder::InsertPoint ip);

    // Type creation
    mlir::Type getI1Type();
    mlir::Type getI32Type();
    mlir::Type getI64Type();
    mlir::Type getF32Type();
    mlir::Type getF64Type();
    mlir::Type getIndexType();
    mlir::Type getIntegerType(unsigned width);
    mlir::Type getMemRefType(const std::vector<int64_t>& shape, mlir::Type elementType);
    mlir::FunctionType getFunctionType(const std::vector<mlir::Type>& inputs,
                                       const std::vector<mlir::Type>& results);

    // LLVM dialect types
    mlir::Type getLLVMPtrType();

    // LLVM dialect operations
    mlir::Value createPtrToInt(const mlir::Value& ptr, mlir::Type intType);

    // Arithmetic operations
    mlir::Value createConstantI32(int32_t value);
    mlir::Value createConstantI64(int64_t value);
    mlir::Value createConstantF32(float value);
    mlir::Value createConstantF64(double value);
    mlir::Value createConstantIndex(int64_t value);

    mlir::Value createAddI(const mlir::Value& lhs, const mlir::Value& rhs);
    mlir::Value createSubI(const mlir::Value& lhs, const mlir::Value& rhs);
    mlir::Value createMulI(const mlir::Value& lhs, const mlir::Value& rhs);
    mlir::Value createDivSI(const mlir::Value& lhs, const mlir::Value& rhs);
    mlir::Value createDivUI(const mlir::Value& lhs, const mlir::Value& rhs);
    mlir::Value createRemSI(const mlir::Value& lhs, const mlir::Value& rhs);
    mlir::Value createRemUI(const mlir::Value& lhs, const mlir::Value& rhs);

    mlir::Value createAddF(const mlir::Value& lhs, const mlir::Value& rhs);
    mlir::Value createSubF(const mlir::Value& lhs, const mlir::Value& rhs);
    mlir::Value createMulF(const mlir::Value& lhs, const mlir::Value& rhs);
    mlir::Value createDivF(const mlir::Value& lhs, const mlir::Value& rhs);
    mlir::Value createRemF(const mlir::Value& lhs, const mlir::Value& rhs);

    // Logical operations (bitwise)
    mlir::Value createAndI(const mlir::Value& lhs, const mlir::Value& rhs);
    mlir::Value createOrI(const mlir::Value& lhs, const mlir::Value& rhs);
    mlir::Value createXorI(const mlir::Value& lhs, const mlir::Value& rhs);

    // Shift operations
    mlir::Value createShlI(const mlir::Value& lhs, const mlir::Value& rhs);
    mlir::Value createShrSII(const mlir::Value& lhs, const mlir::Value& rhs);
    mlir::Value createShrUII(const mlir::Value& lhs, const mlir::Value& rhs);

    // Comparison operations
    mlir::Value createCmpI(const std::string& predicate, const mlir::Value& lhs, const mlir::Value& rhs);
    mlir::Value createCmpF(const std::string& predicate, const mlir::Value& lhs, const mlir::Value& rhs);

    // Select operation (ternary conditional)
    mlir::Value createSelect(const mlir::Value& condition, const mlir::Value& trueValue, const mlir::Value& falseValue);

    mlir::Value createIndexCast(const mlir::Value& value, mlir::Type targetType);
    mlir::Value createSIToFP(const mlir::Value& value, mlir::Type targetType);
    mlir::Value createFPToSI(const mlir::Value& value, mlir::Type targetType);
    mlir::Value createFPToUI(const mlir::Value& value, mlir::Type targetType);
    mlir::Value createUIToFP(const mlir::Value& value, mlir::Type targetType);
    mlir::Value createExtSI(const mlir::Value& value, mlir::Type targetType);
    mlir::Value createExtUI(const mlir::Value& value, mlir::Type targetType);
    mlir::Value createTruncI(const mlir::Value& value, mlir::Type targetType);
    mlir::Value createBitcast(const mlir::Value& value, mlir::Type targetType);

    // Control flow
    mlir::Operation* createReturn(const std::vector<mlir::Value>& values = {});
    mlir::Operation* createFuncReturn(const std::vector<mlir::Value>& values = {});

    // Affine operations
    mlir::affine::AffineForOp createAffineFor(int64_t lowerBound, int64_t upperBound,
                                              int64_t step = 1);
    mlir::Operation* createAffineYield(const std::vector<mlir::Value>& values = {});

    // SCF operations
    mlir::scf::ForOp createSCFFor(const mlir::Value& lowerBound, const mlir::Value& upperBound,
                                   const mlir::Value& step, const std::vector<mlir::Value>& iterArgs = {});
    mlir::scf::IfOp createSCFIf(const mlir::Value& condition, const std::vector<mlir::Type>& resultTypes = {}, bool hasElse = false);
    mlir::scf::WhileOp createSCFWhile(const std::vector<mlir::Type>& resultTypes, const std::vector<mlir::Value>& initVals);
    mlir::Operation* createSCFYield(const std::vector<mlir::Value>& values = {});
    mlir::Operation* createSCFCondition(const mlir::Value& condition, const std::vector<mlir::Value>& values = {});

    // Memory operations
    mlir::Value createMemRefAlloc(mlir::MemRefType type);
    mlir::Value createMemRefAlloca(mlir::MemRefType type);
    void createMemRefStore(const mlir::Value& value, const mlir::Value& memref, const std::vector<mlir::Value>& indices);
    mlir::Value createMemRefLoad(const mlir::Value& memref, const std::vector<mlir::Value>& indices);
    mlir::Value createMemRefGetGlobal(mlir::Type memrefType, const std::string& symbolName);
    mlir::Value createMemRefExtractAlignedPointerAsIndex(const mlir::Value& memref);

    // LLVM operations
    mlir::Value createInlineAsm(const std::string& asmString,
                                 const std::string& constraints,
                                 const std::vector<mlir::Value>& operands,
                                 mlir::Type resultType,
                                 bool hasSideEffects = false);

    // Block operations
    mlir::Block* createBlock(mlir::Region* region, const std::vector<mlir::Type>& argTypes = {});
    mlir::Block* createBlockAtStart(mlir::Region* region, const std::vector<mlir::Type>& argTypes = {});
    mlir::Block* createBlockAtEnd(mlir::Region* region, const std::vector<mlir::Type>& argTypes = {});
    void setInsertionPointToStart(mlir::Block* block);
    void setInsertionPointToEnd(mlir::Block* block);

    // Utility
    mlir::Location getLoc() { return builder->getUnknownLoc(); }
    mlir::OpBuilder& getBuilder() { return *builder; }
    mlir::MLIRContext* getContext() { return context; }

    // Context management
    void setContext(mlir::MLIRContext* newContext);
};

} // namespace megg
