#include "mlir_ir_builder.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace megg {

MLIRIRBuilder::MLIRIRBuilder() {
  context = new mlir::MLIRContext();
  // Register all necessary dialects
  context->loadDialect<mlir::func::FuncDialect>();
  context->loadDialect<mlir::arith::ArithDialect>();
  context->loadDialect<mlir::affine::AffineDialect>();
  context->loadDialect<mlir::scf::SCFDialect>();
  context->loadDialect<mlir::memref::MemRefDialect>();
  context->loadDialect<mlir::LLVM::LLVMDialect>();
  builder = std::make_unique<mlir::OpBuilder>(context);
}

std::pair<mlir::func::FuncOp, mlir::Block *>
MLIRIRBuilder::createFunction(const std::string &name,
                              const std::vector<mlir::Type> &argTypes,
                              const std::vector<mlir::Type> &resultTypes) {
  auto funcType = getFunctionType(argTypes, resultTypes);
  auto func = builder->create<mlir::func::FuncOp>(getLoc(), name, funcType);
  func->setAttr("llvm.linkage",
                mlir::LLVM::LinkageAttr::get(getContext(),
                                             mlir::LLVM::Linkage::External));

  // Create entry block
  auto *entryBlock = func.addEntryBlock();
  builder->setInsertionPointToStart(entryBlock);

  return {func, entryBlock};
}

void MLIRIRBuilder::setInsertionPointToStart(mlir::func::FuncOp func) {
  builder->setInsertionPointToStart(&func.getBody().front());
}

void MLIRIRBuilder::setInsertionPointToEnd(mlir::func::FuncOp func) {
  builder->setInsertionPointToEnd(&func.getBody().back());
}

void MLIRIRBuilder::setInsertionPoint(mlir::Operation *op) {
  builder->setInsertionPoint(op);
}

void MLIRIRBuilder::setInsertionPointAfter(mlir::Operation *op) {
  builder->setInsertionPointAfter(op);
}

mlir::OpBuilder::InsertPoint MLIRIRBuilder::saveInsertionPoint() {
  return builder->saveInsertionPoint();
}

void MLIRIRBuilder::restoreInsertionPoint(mlir::OpBuilder::InsertPoint ip) {
  builder->restoreInsertionPoint(ip);
}

// Type creation
mlir::Type MLIRIRBuilder::getI1Type() { return builder->getI1Type(); }

mlir::Type MLIRIRBuilder::getI32Type() { return builder->getI32Type(); }

mlir::Type MLIRIRBuilder::getI64Type() { return builder->getI64Type(); }

mlir::Type MLIRIRBuilder::getF32Type() { return builder->getF32Type(); }

mlir::Type MLIRIRBuilder::getF64Type() { return builder->getF64Type(); }

mlir::Type MLIRIRBuilder::getIndexType() { return builder->getIndexType(); }

mlir::Type MLIRIRBuilder::getIntegerType(unsigned width) {
  return builder->getIntegerType(width);
}

mlir::Type MLIRIRBuilder::getMemRefType(const std::vector<int64_t> &shape,
                                        mlir::Type elementType) {
  return mlir::MemRefType::get(shape, elementType);
}

mlir::FunctionType
MLIRIRBuilder::getFunctionType(const std::vector<mlir::Type> &inputs,
                               const std::vector<mlir::Type> &results) {
  return builder->getFunctionType(inputs, results);
}

// LLVM dialect types
mlir::Type MLIRIRBuilder::getLLVMPtrType() {
  return mlir::LLVM::LLVMPointerType::get(getContext());
}

// LLVM dialect operations
mlir::Value MLIRIRBuilder::createPtrToInt(const mlir::Value& ptr, mlir::Type intType) {
  return builder->create<mlir::LLVM::PtrToIntOp>(getLoc(), intType, ptr);
}

// Arithmetic operations
mlir::Value MLIRIRBuilder::createConstantI32(int32_t value) {
  auto type = getI32Type();
  auto attr = builder->getIntegerAttr(type, value);
  return builder->create<mlir::arith::ConstantOp>(getLoc(), type, attr);
}

mlir::Value MLIRIRBuilder::createConstantI64(int64_t value) {
  auto type = getI64Type();
  auto attr = builder->getIntegerAttr(type, value);
  return builder->create<mlir::arith::ConstantOp>(getLoc(), type, attr);
}

mlir::Value MLIRIRBuilder::createConstantF32(float value) {
  auto type = getF32Type();
  auto attr = builder->getFloatAttr(type, value);
  return builder->create<mlir::arith::ConstantOp>(getLoc(), type, attr);
}

mlir::Value MLIRIRBuilder::createConstantF64(double value) {
  auto type = getF64Type();
  auto attr = builder->getFloatAttr(type, value);
  return builder->create<mlir::arith::ConstantOp>(getLoc(), type, attr);
}

mlir::Value MLIRIRBuilder::createConstantIndex(int64_t value) {
  return builder->create<mlir::arith::ConstantIndexOp>(getLoc(), value);
}

mlir::Value MLIRIRBuilder::createAddI(const mlir::Value &lhs,
                                      const mlir::Value &rhs) {
  return builder->create<mlir::arith::AddIOp>(getLoc(), lhs, rhs);
}

mlir::Value MLIRIRBuilder::createSubI(const mlir::Value &lhs,
                                      const mlir::Value &rhs) {
  return builder->create<mlir::arith::SubIOp>(getLoc(), lhs, rhs);
}

mlir::Value MLIRIRBuilder::createMulI(const mlir::Value &lhs,
                                      const mlir::Value &rhs) {
  return builder->create<mlir::arith::MulIOp>(getLoc(), lhs, rhs);
}

mlir::Value MLIRIRBuilder::createDivSI(const mlir::Value &lhs,
                                       const mlir::Value &rhs) {
  return builder->create<mlir::arith::DivSIOp>(getLoc(), lhs, rhs);
}

mlir::Value MLIRIRBuilder::createDivUI(const mlir::Value &lhs,
                                       const mlir::Value &rhs) {
  return builder->create<mlir::arith::DivUIOp>(getLoc(), lhs, rhs);
}

mlir::Value MLIRIRBuilder::createRemSI(const mlir::Value &lhs,
                                       const mlir::Value &rhs) {
  return builder->create<mlir::arith::RemSIOp>(getLoc(), lhs, rhs);
}

mlir::Value MLIRIRBuilder::createRemUI(const mlir::Value &lhs,
                                       const mlir::Value &rhs) {
  return builder->create<mlir::arith::RemUIOp>(getLoc(), lhs, rhs);
}

mlir::Value MLIRIRBuilder::createAddF(const mlir::Value &lhs,
                                      const mlir::Value &rhs) {
  return builder->create<mlir::arith::AddFOp>(getLoc(), lhs, rhs);
}

mlir::Value MLIRIRBuilder::createSubF(const mlir::Value &lhs,
                                      const mlir::Value &rhs) {
  return builder->create<mlir::arith::SubFOp>(getLoc(), lhs, rhs);
}

mlir::Value MLIRIRBuilder::createMulF(const mlir::Value &lhs,
                                      const mlir::Value &rhs) {
  return builder->create<mlir::arith::MulFOp>(getLoc(), lhs, rhs);
}

mlir::Value MLIRIRBuilder::createDivF(const mlir::Value &lhs,
                                      const mlir::Value &rhs) {
  return builder->create<mlir::arith::DivFOp>(getLoc(), lhs, rhs);
}

mlir::Value MLIRIRBuilder::createRemF(const mlir::Value &lhs,
                                      const mlir::Value &rhs) {
  return builder->create<mlir::arith::RemFOp>(getLoc(), lhs, rhs);
}

// Logical operations (bitwise)
mlir::Value MLIRIRBuilder::createAndI(const mlir::Value &lhs,
                                      const mlir::Value &rhs) {
  return builder->create<mlir::arith::AndIOp>(getLoc(), lhs, rhs);
}

mlir::Value MLIRIRBuilder::createOrI(const mlir::Value &lhs,
                                     const mlir::Value &rhs) {
  return builder->create<mlir::arith::OrIOp>(getLoc(), lhs, rhs);
}

mlir::Value MLIRIRBuilder::createXorI(const mlir::Value &lhs,
                                      const mlir::Value &rhs) {
  return builder->create<mlir::arith::XOrIOp>(getLoc(), lhs, rhs);
}

// Shift operations
mlir::Value MLIRIRBuilder::createShlI(const mlir::Value &lhs,
                                      const mlir::Value &rhs) {
  return builder->create<mlir::arith::ShLIOp>(getLoc(), lhs, rhs);
}

mlir::Value MLIRIRBuilder::createShrSII(const mlir::Value &lhs,
                                        const mlir::Value &rhs) {
  return builder->create<mlir::arith::ShRSIOp>(getLoc(), lhs, rhs);
}

mlir::Value MLIRIRBuilder::createShrUII(const mlir::Value &lhs,
                                        const mlir::Value &rhs) {
  return builder->create<mlir::arith::ShRUIOp>(getLoc(), lhs, rhs);
}

// Comparison operations
mlir::Value MLIRIRBuilder::createCmpI(const std::string &predicate,
                                      const mlir::Value &lhs,
                                      const mlir::Value &rhs) {
  // Convert string predicate to enum
  mlir::arith::CmpIPredicate pred;
  if (predicate == "eq")
    pred = mlir::arith::CmpIPredicate::eq;
  else if (predicate == "ne")
    pred = mlir::arith::CmpIPredicate::ne;
  else if (predicate == "slt")
    pred = mlir::arith::CmpIPredicate::slt;
  else if (predicate == "sle")
    pred = mlir::arith::CmpIPredicate::sle;
  else if (predicate == "sgt")
    pred = mlir::arith::CmpIPredicate::sgt;
  else if (predicate == "sge")
    pred = mlir::arith::CmpIPredicate::sge;
  else if (predicate == "ult")
    pred = mlir::arith::CmpIPredicate::ult;
  else if (predicate == "ule")
    pred = mlir::arith::CmpIPredicate::ule;
  else if (predicate == "ugt")
    pred = mlir::arith::CmpIPredicate::ugt;
  else if (predicate == "uge")
    pred = mlir::arith::CmpIPredicate::uge;
  else
    pred = mlir::arith::CmpIPredicate::eq; // default

  return builder->create<mlir::arith::CmpIOp>(getLoc(), pred, lhs, rhs);
}

mlir::Value MLIRIRBuilder::createCmpF(const std::string &predicate,
                                      const mlir::Value &lhs,
                                      const mlir::Value &rhs) {
  // Convert string predicate to enum
  mlir::arith::CmpFPredicate pred;
  if (predicate == "oeq")
    pred = mlir::arith::CmpFPredicate::OEQ;
  else if (predicate == "one")
    pred = mlir::arith::CmpFPredicate::ONE;
  else if (predicate == "olt")
    pred = mlir::arith::CmpFPredicate::OLT;
  else if (predicate == "ole")
    pred = mlir::arith::CmpFPredicate::OLE;
  else if (predicate == "ogt")
    pred = mlir::arith::CmpFPredicate::OGT;
  else if (predicate == "oge")
    pred = mlir::arith::CmpFPredicate::OGE;
  else if (predicate == "ord")
    pred = mlir::arith::CmpFPredicate::ORD;
  else if (predicate == "ueq")
    pred = mlir::arith::CmpFPredicate::UEQ;
  else if (predicate == "une")
    pred = mlir::arith::CmpFPredicate::UNE;
  else if (predicate == "ult")
    pred = mlir::arith::CmpFPredicate::ULT;
  else if (predicate == "ule")
    pred = mlir::arith::CmpFPredicate::ULE;
  else if (predicate == "ugt")
    pred = mlir::arith::CmpFPredicate::UGT;
  else if (predicate == "uge")
    pred = mlir::arith::CmpFPredicate::UGE;
  else if (predicate == "uno")
    pred = mlir::arith::CmpFPredicate::UNO;
  else if (predicate == "true")
    pred = mlir::arith::CmpFPredicate::AlwaysTrue;
  else if (predicate == "false")
    pred = mlir::arith::CmpFPredicate::AlwaysFalse;
  else
    pred = mlir::arith::CmpFPredicate::OEQ; // default

  return builder->create<mlir::arith::CmpFOp>(getLoc(), pred, lhs, rhs);
}

mlir::Value MLIRIRBuilder::createSelect(const mlir::Value &condition,
                                        const mlir::Value &trueValue,
                                        const mlir::Value &falseValue) {
  return builder->create<mlir::arith::SelectOp>(getLoc(), condition, trueValue, falseValue);
}

mlir::Value MLIRIRBuilder::createIndexCast(const mlir::Value &value,
                                           mlir::Type targetType) {
  return builder->create<mlir::arith::IndexCastOp>(getLoc(), targetType, value);
}

mlir::Value MLIRIRBuilder::createSIToFP(const mlir::Value &value,
                                        mlir::Type targetType) {
  return builder->create<mlir::arith::SIToFPOp>(getLoc(), targetType, value);
}

mlir::Value MLIRIRBuilder::createFPToSI(const mlir::Value &value,
                                        mlir::Type targetType) {
  return builder->create<mlir::arith::FPToSIOp>(getLoc(), targetType, value);
}

mlir::Value MLIRIRBuilder::createFPToUI(const mlir::Value &value,
                                        mlir::Type targetType) {
  return builder->create<mlir::arith::FPToUIOp>(getLoc(), targetType, value);
}

mlir::Value MLIRIRBuilder::createUIToFP(const mlir::Value &value,
                                        mlir::Type targetType) {
  return builder->create<mlir::arith::UIToFPOp>(getLoc(), targetType, value);
}

mlir::Value MLIRIRBuilder::createExtSI(const mlir::Value &value,
                                       mlir::Type targetType) {
  return builder->create<mlir::arith::ExtSIOp>(getLoc(), targetType, value);
}

mlir::Value MLIRIRBuilder::createExtUI(const mlir::Value &value,
                                       mlir::Type targetType) {
  return builder->create<mlir::arith::ExtUIOp>(getLoc(), targetType, value);
}

mlir::Value MLIRIRBuilder::createTruncI(const mlir::Value &value,
                                        mlir::Type targetType) {
  return builder->create<mlir::arith::TruncIOp>(getLoc(), targetType, value);
}

mlir::Value MLIRIRBuilder::createBitcast(const mlir::Value &value,
                                         mlir::Type targetType) {
  return builder->create<mlir::arith::BitcastOp>(getLoc(), targetType, value);
}

// Control flow
mlir::Operation *
MLIRIRBuilder::createReturn(const std::vector<mlir::Value> &values) {
  return builder->create<mlir::func::ReturnOp>(getLoc(), values);
}

mlir::Operation *
MLIRIRBuilder::createFuncReturn(const std::vector<mlir::Value> &values) {
  return builder->create<mlir::func::ReturnOp>(getLoc(), values);
}

// Affine operations
mlir::affine::AffineForOp MLIRIRBuilder::createAffineFor(int64_t lowerBound,
                                                         int64_t upperBound,
                                                         int64_t step) {
  auto forOp = builder->create<mlir::affine::AffineForOp>(getLoc(), lowerBound,
                                                          upperBound, step);
  builder->setInsertionPointToStart(forOp.getBody());
  return forOp;
}

mlir::Operation *
MLIRIRBuilder::createAffineYield(const std::vector<mlir::Value> &values) {
  return builder->create<mlir::affine::AffineYieldOp>(getLoc(), values);
}

// SCF operations
mlir::scf::ForOp MLIRIRBuilder::createSCFFor(
    const mlir::Value &lowerBound, const mlir::Value &upperBound,
    const mlir::Value &step, const std::vector<mlir::Value> &iterArgs) {
  auto forOp = builder->create<mlir::scf::ForOp>(getLoc(), lowerBound,
                                                 upperBound, step, iterArgs);
  builder->setInsertionPointToStart(forOp.getBody());
  return forOp;
}

mlir::scf::IfOp
MLIRIRBuilder::createSCFIf(const mlir::Value &condition,
                           const std::vector<mlir::Type> &resultTypes,
                           bool hasElse) {
  auto ifOp = builder->create<mlir::scf::IfOp>(getLoc(), resultTypes, condition,
                                               hasElse);

  // The then region should already have a block, but let's ensure it
  auto &thenRegion = ifOp.getThenRegion();
  if (thenRegion.empty()) {
    thenRegion.emplaceBlock();
  }

  // Initialize the else region if needed
  if (hasElse) {
    auto &elseRegion = ifOp.getElseRegion();
    if (elseRegion.empty()) {
      elseRegion.emplaceBlock();
    }
  }

  builder->setInsertionPointToStart(&thenRegion.front());
  return ifOp;
}

mlir::scf::WhileOp
MLIRIRBuilder::createSCFWhile(const std::vector<mlir::Type> &resultTypes,
                              const std::vector<mlir::Value> &initVals) {
  auto whileOp =
      builder->create<mlir::scf::WhileOp>(getLoc(), resultTypes, initVals);

  // Initialize the before region with a block and arguments
  auto &beforeRegion = whileOp.getBefore();
  if (beforeRegion.empty()) {
    auto *beforeBlock = new mlir::Block();
    for (auto type : resultTypes) {
      beforeBlock->addArgument(type, getLoc());
    }
    beforeRegion.push_back(beforeBlock);
  }

  // Initialize the after region with a block and arguments
  auto &afterRegion = whileOp.getAfter();
  if (afterRegion.empty()) {
    auto *afterBlock = new mlir::Block();
    for (auto type : resultTypes) {
      afterBlock->addArgument(type, getLoc());
    }
    afterRegion.push_back(afterBlock);
  }

  // Set insertion point to the before block (condition block)
  builder->setInsertionPointToStart(&beforeRegion.front());

  return whileOp;
}

mlir::Operation *
MLIRIRBuilder::createSCFYield(const std::vector<mlir::Value> &values) {
  return builder->create<mlir::scf::YieldOp>(getLoc(), values);
}

mlir::Operation *
MLIRIRBuilder::createSCFCondition(const mlir::Value &condition,
                                  const std::vector<mlir::Value> &values) {
  return builder->create<mlir::scf::ConditionOp>(getLoc(), condition, values);
}

// Memory operations
mlir::Value MLIRIRBuilder::createMemRefAlloc(mlir::MemRefType type) {
  return builder->create<mlir::memref::AllocOp>(getLoc(), type);
}

mlir::Value MLIRIRBuilder::createMemRefAlloca(mlir::MemRefType type) {
  return builder->create<mlir::memref::AllocaOp>(getLoc(), type);
}

void MLIRIRBuilder::createMemRefStore(const mlir::Value &value,
                                      const mlir::Value &memref,
                                      const std::vector<mlir::Value> &indices) {
  builder->create<mlir::memref::StoreOp>(getLoc(), value, memref, indices);
}

mlir::Value
MLIRIRBuilder::createMemRefLoad(const mlir::Value &memref,
                                const std::vector<mlir::Value> &indices) {
  return builder->create<mlir::memref::LoadOp>(getLoc(), memref, indices);
}

mlir::Value
MLIRIRBuilder::createMemRefGetGlobal(mlir::Type memrefType,
                                     const std::string &symbolName) {
  return builder->create<mlir::memref::GetGlobalOp>(
      getLoc(), llvm::cast<mlir::MemRefType>(memrefType), symbolName);
}

mlir::Value MLIRIRBuilder::createMemRefExtractAlignedPointerAsIndex(
    const mlir::Value &memref) {
  return builder->create<mlir::memref::ExtractAlignedPointerAsIndexOp>(
      getLoc(), memref);
}

// Block operations
mlir::Block *
MLIRIRBuilder::createBlock(mlir::Region *region,
                           const std::vector<mlir::Type> &argTypes) {
  mlir::Block &block = region->emplaceBlock();
  for (auto type : argTypes) {
    block.addArgument(type, getLoc());
  }
  return &block;
}

mlir::Block *
MLIRIRBuilder::createBlockAtStart(mlir::Region *region,
                                  const std::vector<mlir::Type> &argTypes) {
  // Create block at the beginning of the region
  mlir::Block &block = region->emplaceBlock();

  // Add arguments
  for (auto type : argTypes) {
    block.addArgument(type, getLoc());
  }

  // Set insertion point to the new block
  builder->setInsertionPointToStart(&block);
  return &block;
}

mlir::Block *
MLIRIRBuilder::createBlockAtEnd(mlir::Region *region,
                                const std::vector<mlir::Type> &argTypes) {
  // emplaceBlock always adds at the end by default
  mlir::Block &block = region->emplaceBlock();

  // Add arguments
  for (auto type : argTypes) {
    block.addArgument(type, getLoc());
  }

  // Set insertion point to the new block
  builder->setInsertionPointToStart(&block);
  return &block;
}

void MLIRIRBuilder::setInsertionPointToStart(mlir::Block *block) {
  builder->setInsertionPointToStart(block);
}

void MLIRIRBuilder::setInsertionPointToEnd(mlir::Block *block) {
  builder->setInsertionPointToEnd(block);
}

void MLIRIRBuilder::setContext(mlir::MLIRContext *newContext) {
  context = newContext;
  builder = std::make_unique<mlir::OpBuilder>(context);
}

// LLVM operations
mlir::Value
MLIRIRBuilder::createInlineAsm(const std::string &asmString,
                               const std::string &constraints,
                               const std::vector<mlir::Value> &operands,
                               mlir::Type resultType, bool hasSideEffects) {

  // Create string attributes
  auto asmStringAttr = builder->getStringAttr(asmString);
  auto constraintsAttr = builder->getStringAttr(constraints);

  // Create inline assembly operation
  // Use the simplified builder that doesn't require all attributes
  auto asmOp = builder->create<mlir::LLVM::InlineAsmOp>(
      getLoc(), resultType, operands, asmString, constraints, hasSideEffects,
      /*is_align_stack=*/false,
      mlir::LLVM::tailcallkind::TailCallKind(), // Default value
      mlir::LLVM::AsmDialectAttr{}, mlir::ArrayAttr{});

  return asmOp.getResult(0);
}

} // namespace megg
