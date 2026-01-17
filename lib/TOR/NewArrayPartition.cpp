#include "APS/APSOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

#include "TOR/PassDetail.h"
#include "TOR/Passes.h"
#include "TOR/TORDialect.h"
#include "TOR/Utils.h"
#include "TOR/AffineModSimplify.h"

#include <iostream>
#include <map>
#include <set>

#define DEBUG_TYPE "new-array-partition"

namespace {
using namespace mlir;
using namespace mlir::func;
using namespace affine;

void warningNonStandardAffineAccess(std::string type, Attribute varNameAttr) {
  llvm::errs() << "warning: " << type << " variable " << varNameAttr
               << " with applying array_partition pragma is failed, "
                  "because non standard affine access.\n";
}

std::string stringAddNumber(std::string str, int number) {
  return str = str + llvm::Twine(number).str();
}

std::string getLineStringWithRank(Value val, int rank) {
  std::string lineString = "array_partition";
  if (auto blockArg = dyn_cast<mlir::BlockArgument>(val)) {
    lineString += stringAddNumber("_arg_", blockArg.getArgNumber());
  }
  return lineString + stringAddNumber("_dim_", rank);
}

std::string getVariableTypeByValue(Value val) {
  if (auto blockArg = dyn_cast<mlir::BlockArgument>(val)) {
    auto funcOp = cast<FuncOp>(blockArg.getOwner()->getParentOp());
    return "@" + funcOp.getSymName().str() + " function with argument";
  }
  return "alloca";
}

Attribute getVarNameTypeByValue(Value val) {
  auto op = getDefiningOpByValue(val);
  if (auto blockArg = dyn_cast<mlir::BlockArgument>(val)) {
    return op->getAttr(stringAddNumber("var_name_", blockArg.getArgNumber()));
  }
  return op->getAttr("var_name");
}

void warningNonStandardAffineAccessByVal(Value val) {
  warningNonStandardAffineAccess(getVariableTypeByValue(val),
                                 getVarNameTypeByValue(val));
}

void warningBankCannotBeDivided(StringRef type, Attribute varNameAttr,
                                int dim) {
  llvm::errs() << "warning: " << type << " variable " << varNameAttr
               << " with applying array_partition pragma on dim " << dim
               << " is failed, because bank cannot be divided.\n";
}

void warningBankCannotBeDividedByVal(Value val, int dim) {
  warningBankCannotBeDivided(getVariableTypeByValue(val),
                             getVarNameTypeByValue(val), dim);
}

int getAttrInterger(Attribute attr) {
  return dyn_cast<mlir::IntegerAttr>(attr).getValue().getSExtValue();
}

std::string exprToString(AffineExpr expr) {
  std::string str;
  llvm::raw_string_ostream os(str);
  expr.print(os);
  return str;
}

// Compute all possible bank values for expression of form: d_i * coeff (+ offset)
// Only handles single dimension linear patterns - rejects complex cases
SmallVector<int64_t> getPossibleBankValues(AffineExpr expr, AffineMap map,
                                            ArrayRef<Value> operands, int factor) {
  SmallVector<int64_t> result;
  unsigned numDims = map.getNumDims();
  unsigned numSyms = map.getNumSymbols();

  // Use our custom simplifyMod: (2x + 8y + 64) % 8 -> 2x
  AffineExpr simplified = tor::simplifyMod(expr, factor, numDims, numSyms);
  if (!simplified) {
    return result; // Not linear - reject
  }

  LLVM_DEBUG(llvm::dbgs() << "  simplifyMod: " << exprToString(expr)
                          << " % " << factor << " -> " << exprToString(simplified) << "\n");

  // Reject if any symbol remains
  for (unsigned i = 0; i < numSyms; ++i) {
    if (simplified.isFunctionOfSymbol(i)) {
      return result; // empty - symbol affects bank
    }
  }

  // If constant, just return that value
  if (auto constExpr = dyn_cast<AffineConstantExpr>(simplified)) {
    int64_t val = constExpr.getValue();
    result.push_back(((val % factor) + factor) % factor);
    return result;
  }

  // Only handle single dimension case
  int relevantDim = -1;
  for (unsigned i = 0; i < numDims; ++i) {
    if (simplified.isFunctionOfDim(i)) {
      if (relevantDim >= 0) {
        return result; // multiple dims - too complex
      }
      relevantDim = i;
    }
  }

  if (relevantDim < 0 || (unsigned)relevantDim >= operands.size()) {
    return result;
  }

  // Get loop bounds for this dimension
  Value operand = operands[relevantDim];
  auto blockArg = dyn_cast<BlockArgument>(operand);
  if (!blockArg) return result;

  auto forOp = dyn_cast<AffineForOp>(blockArg.getOwner()->getParentOp());
  if (!forOp || forOp.getInductionVar() != blockArg ||
      !forOp.hasConstantLowerBound() || !forOp.hasConstantUpperBound()) {
    return result;
  }

  int64_t lb = forOp.getConstantLowerBound();
  int64_t ub = forOp.getConstantUpperBound();
  if (ub - lb > 64) return result;

  // Enumerate: substitute d_i with each value and evaluate
  std::set<int64_t> uniqueValues;
  auto *ctx = simplified.getContext();

  for (int64_t v = lb; v < ub; ++v) {
    SmallVector<AffineExpr> dimReplacements(numDims, getAffineConstantExpr(0, ctx));
    dimReplacements[relevantDim] = getAffineConstantExpr(v, ctx);
    SmallVector<AffineExpr> symReplacements(numSyms, getAffineConstantExpr(0, ctx));

    AffineExpr evaluated = simplified.replaceDimsAndSymbols(dimReplacements, symReplacements);
    evaluated = simplifyAffineExpr(evaluated, 0, 0);

    if (auto constE = dyn_cast<AffineConstantExpr>(evaluated)) {
      int64_t val = ((constE.getValue() % factor) + factor) % factor;
      uniqueValues.insert(val);
    }
  }

  for (int64_t v : uniqueValues) {
    result.push_back(v);
  }

  LLVM_DEBUG(llvm::dbgs() << "  getPossibleBankValues: dim=d" << relevantDim
                          << " [" << lb << "," << ub << ") -> banks={";
             for (auto v : result) llvm::dbgs() << v << ",";
             llvm::dbgs() << "}\n");

  return result;
}


// Check if an access can be partitioned (bank may be runtime-computed)
// Apply mod first to eliminate symbolic terms with coefficients divisible by factor
// e.g., (symbol * 64) % 8 = 0, so the symbol doesn't affect bank selection
// For loads (isLoad=true), we allow dynamic bank selection with mux if possible banks can be enumerated
bool canBankPartition(AffineMap map, int rank, MLIRContext *ctx, int factor,
                      bool cyclic, Operation *op = nullptr, bool isLoad = false,
                      SmallVector<int64_t> *possibleBanksOut = nullptr) {
  auto expr = map.getResult(rank);
  if (!expr) {
    if (op) op->emitWarning() << "array partition failed: no expression for rank " << rank;
    return false;
  }
  auto origExpr = expr;
  unsigned numDims = map.getNumDims();
  unsigned numSyms = map.getNumSymbols();

  // Use custom simplifyMod to eliminate terms with coefficients divisible by factor
  AffineExpr simplified = tor::simplifyMod(origExpr, factor, numDims, numSyms);
  if (!simplified) {
    if (op) op->emitWarning() << "array partition failed for rank " << rank << ": "
                              << "expression is not linear";
    return false;
  }

  // Check if simplified result still has symbols - reject if so
  for (unsigned i = 0; i < numSyms; ++i) {
    if (simplified.isFunctionOfSymbol(i)) {
      if (op) op->emitWarning() << "array partition failed for rank " << rank << ": "
                                << "after simplification, symbol s" << i << " still affects bank selection "
                                << "(original: " << exprToString(origExpr) << ", simplified: " << exprToString(simplified) << ")";
      return false;
    }
  }

  // Check how many dimensions affect the simplified result
  int numRelevantDims = 0;
  for (unsigned i = 0; i < numDims; ++i) {
    if (simplified.isFunctionOfDim(i)) {
      numRelevantDims++;
    }
  }

  // If more than one dimension, reject (too complex)
  if (numRelevantDims > 1) {
    if (op) op->emitWarning() << "array partition failed for rank " << rank << ": "
                              << "multiple dimensions affect bank selection "
                              << "(simplified: " << exprToString(simplified) << ")";
    return false;
  }

  // Check if bank is compile-time constant (no dimensions affect it)
  if (numRelevantDims == 0) {
    if (auto constExpr = dyn_cast<AffineConstantExpr>(simplified)) {
      if (possibleBanksOut) {
        possibleBanksOut->clear();
        int64_t val = ((constExpr.getValue() % factor) + factor) % factor;
        possibleBanksOut->push_back(val);
      }
      return true;
    }
  }

  // Bank depends on exactly one loop index - try to enumerate possible banks
  if (isLoad && op) {
    SmallVector<Value> operands;
    if (auto load = dyn_cast<AffineLoadOp>(op)) {
      operands.assign(load.getMapOperands().begin(), load.getMapOperands().end());
    } else if (auto store = dyn_cast<AffineStoreOp>(op)) {
      operands.assign(store.getMapOperands().begin(), store.getMapOperands().end());
    }
    auto possibleBanks = getPossibleBankValues(origExpr, map, operands, factor);
    if (!possibleBanks.empty()) {
      if (possibleBanksOut) {
        *possibleBanksOut = possibleBanks;
      }
      // Warn about dynamic bank selection but proceed
      bool isStore = isa<AffineStoreOp>(op);
      op->emitWarning() << "array partition: using dynamic bank "
                        << (isStore ? "read-modify-write" : "mux")
                        << " for rank " << rank << ", possible banks: {"
                        << [&]() {
                             std::string s;
                             for (size_t i = 0; i < possibleBanks.size(); ++i) {
                               if (i > 0) s += ",";
                               s += std::to_string(possibleBanks[i]);
                             }
                             return s;
                           }()
                        << "}";
      return true; // Allow with dynamic mux/read-modify-write
    }
  }

  if (op) {
    op->emitWarning() << "array partition failed for rank " << rank << ": "
                      << "bank depends on loop indices and cannot enumerate possible values";
  }
  return false;
}

int getMemBank(AffineMap map, int rank, MLIRContext *ctx, int factor,
               bool cyclic, Operation *op = nullptr) {
  auto expr = map.getResult(rank);
  if (!expr) {
    if (op) op->emitWarning() << "getMemBank failed: no expression for rank " << rank;
    return -1;
  }
  auto origExpr = expr;
  // Apply mod/div first - this simplifies (symbol * k) % factor to 0 when k % factor == 0
  if (cyclic) {
    expr = expr % factor;
  } else {
    expr = expr.floorDiv(factor);
  }
  auto compose_map = AffineMap::get(map.getNumDims(), map.getNumSymbols(), expr, ctx);
  if (compose_map.isConstant()) {
    return compose_map.getConstantResults()[0];
  }
  if (op) {
    op->emitWarning() << "getMemBank failed for rank " << rank << ": "
                      << "original expr '" << exprToString(origExpr) << "' after "
                      << (cyclic ? "mod" : "floorDiv") << " " << factor << " = '" << exprToString(expr) << "', "
                      << "result is not constant (depends on loop indices)";
  }
  return -1;
}

int isFullyPartition(ArrayAttr attr) {
  return attr.size() == 1 && getAttrInterger(attr[0]) == -1;
}

void getFactorMapAndCyclicMap(Operation *op, MemRefType memref,
                              DenseMap<int, int> &factorMap,
                              DenseMap<int, bool> &cyclicMap,
                              bool fullyPartition) {
  auto partitionDimArray =
      dyn_cast<mlir::ArrayAttr>(op->getAttr("partition_dim_array"));
  auto partitionFactorArray =
      dyn_cast<mlir::ArrayAttr>(op->getAttr("partition_factor_array"));
  auto partitionCyclicArray =
      dyn_cast<mlir::ArrayAttr>(op->getAttr("partition_cyclic_array"));
  if (fullyPartition) {
    int factor = getAttrInterger(partitionFactorArray[0]);
    for (int i = 0, e = memref.getShape().size(); i < e; i++) {
      if (factor == -1) {
        factorMap[i] = memref.getShape()[i];
      } else {
        factorMap[i] = factor;
      }
      cyclicMap[i] = getAttrInterger(partitionCyclicArray[0]);
    }
    return;
  }
  for (int i = 0, e = partitionDimArray.size(); i < e; i++) {
    int partitionDim = getAttrInterger(partitionDimArray[i]);
    factorMap[partitionDim] = getAttrInterger(partitionFactorArray[i]);
    cyclicMap[partitionDim] = getAttrInterger(partitionCyclicArray[i]);
  }
}

bool checkValueUsers(Value val, int64_t rankNum) {
  for (auto *op : val.getUsers()) {
    if (auto load = dyn_cast<AffineLoadOp>(op)) {
      if (load.getAffineMap().getNumResults() != rankNum) {
        return false;
      }
    } else if (auto store = dyn_cast<AffineStoreOp>(op)) {
      if (store.getAffineMap().getNumResults() != rankNum) {
        return false;
      }
    } else if (auto call = dyn_cast<func::CallOp>(op)) {
      // ignore recursion
      // auto getFuncOpByVal = [](Value val) {
      //   if (auto blockArg = val.dyn_cast<BlockArgument>()) {
      //     return cast<FuncOp>(blockArg.getOwner()->getParentOp());
      //   } else {
      //     return cast<FuncOp>(val.getDefiningOp()->getParentOp());
      //   }
      // };
      // if (getFuncOpByVal(val).getSymName() == call.getCallee()) {
      //   return false;
      // }
    } else if (auto burst_read = dyn_cast<aps::MemBurstLoad>(op)) {
      //
    } else if (auto burst_write = dyn_cast<aps::MemBurstStore>(op)) {
      //
    } else {
      // TODO: handle burst load/store here!
      LLVM_DEBUG(llvm::dbgs() << "Unknown memory operation: " << op << "\n");
      return false;
    }
  }
  return true;
}

// Helper to check if memref.load/store indices can form affine expressions
bool canFormAffineExpr(Value index) {
  // Check if this is a constant
  if (index.getDefiningOp<arith::ConstantOp>() ||
      index.getDefiningOp<arith::ConstantIndexOp>())
    return true;

  // Check if this is an index_cast - look through it
  if (auto cast = index.getDefiningOp<arith::IndexCastOp>())
    return canFormAffineExpr(cast.getIn());

  // Check if this is an arithmetic operation on affine-compatible values
  if (auto addOp = index.getDefiningOp<arith::AddIOp>())
    return canFormAffineExpr(addOp.getLhs()) && canFormAffineExpr(addOp.getRhs());

  if (auto mulOp = index.getDefiningOp<arith::MulIOp>()) {
    // For mul, one operand must be constant
    return (canFormAffineExpr(mulOp.getLhs()) &&
            (mulOp.getRhs().getDefiningOp<arith::ConstantOp>() ||
             mulOp.getRhs().getDefiningOp<arith::ConstantIndexOp>())) ||
           (canFormAffineExpr(mulOp.getRhs()) &&
            (mulOp.getLhs().getDefiningOp<arith::ConstantOp>() ||
             mulOp.getLhs().getDefiningOp<arith::ConstantIndexOp>()));
  }

  // Check if this is a block argument (loop induction variable)
  if (auto blockArg = llvm::dyn_cast<BlockArgument>(index)) {
    auto *parentOp = blockArg.getOwner()->getParentOp();
    return llvm::isa<affine::AffineForOp, affine::AffineParallelOp>(parentOp);
  }

  return false;
}

bool rankCanBePartition(Value arg, size_t rank, unsigned bank_factor,
                        bool cyclic, MLIRContext *ctx) {
  for (auto *op : arg.getUsers()) {
    if (auto load = dyn_cast<AffineLoadOp>(op)) {
      // For loads, allow dynamic bank selection with mux (isLoad=true)
      if (!canBankPartition(load.getAffineMap(), rank, ctx, bank_factor, cyclic, op, /*isLoad=*/true)) {
        return false;
      }
    } else if (auto store = dyn_cast<AffineStoreOp>(op)) {
      // For stores, also allow dynamic bank (will use read-modify-write)
      if (!canBankPartition(store.getAffineMap(), rank, ctx, bank_factor, cyclic, op, /*isLoad=*/true)) {
        return false;
      }
    } else if (auto load = dyn_cast<memref::LoadOp>(op)) {
      // Check if memref.load indices can form affine expressions
      if (load.getIndices().size() != 1 || !canFormAffineExpr(load.getIndices()[0])) {
        op->emitWarning() << "array partition failed: memref.load has non-affine index";
        return false;
      }
    } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
      // Check if memref.store indices can form affine expressions
      if (store.getIndices().size() != 1 || !canFormAffineExpr(store.getIndices()[0])) {
        op->emitWarning() << "array partition failed: memref.store has non-affine index";
        return false;
      }
    }
  }
  return true;
}

void getPartitionVector(SmallVector<bool> &partition, Value arg,
                        MemRefType memref, DenseMap<int, int> factorMap,
                        DenseMap<int, bool> cyclicMap, StringRef type,
                        Attribute varNameAttr, PatternRewriter &rewriter,
                        bool fullyPartition) {
  for (size_t rank = 0, e = memref.getRank(); rank < e; ++rank) {
    bool flag = false;
    if (factorMap.count(rank)) {
      unsigned bank_factor = cyclicMap[rank]
                                 ? factorMap[rank]
                                 : memref.getShape()[rank] / factorMap[rank];
      flag = rankCanBePartition(arg, rank, bank_factor, cyclicMap[rank],
                                rewriter.getContext());
      std::string lineString = "array_partition";
      if (auto blockArg = dyn_cast<mlir::BlockArgument>(arg)) {
        lineString += "_arg_" + llvm::Twine(blockArg.getArgNumber()).str();
      }
      lineString += "_dim_" + llvm::Twine(rank).str();
      if (!flag) {
        if (fullyPartition) {
          partition.clear();
          return;
        }
      } else {
        setPragmaStructureAttrStatusByValue(arg, lineString);
      }
    }
    partition.push_back(flag);
  }
  if (fullyPartition) {
    setPragmaStructureAttrStatusByValue(arg, "array_partition");
  }
}

bool needPartition(SmallVector<bool> partition) {
  for (auto p : partition) {
    if (p) {
      return true;
    }
  }
  return false;
}

int getNewBank(int bank, int index, int factor, bool cyclic, int rankShape) {
  unsigned bank_factor = cyclic ? factor : rankShape / factor;
  int addBank = cyclic ? index % bank_factor : index / bank_factor;
  if (!cyclic && addBank >= factor) {
    return bank * factor + index % factor;
  }
  return bank * factor + addBank;
}

void getInitialValue(MemRefType memref, int rank, int flattenedIndex, int bank,
                     int targetBank, DenseElementsAttr elementsAttr,
                     SmallVector<bool> partition, DenseMap<int, int> factorMap,
                     DenseMap<int, bool> cyclicMap,
                     SmallVector<char> &outDataVec) {
  if (rank == memref.getRank()) {
    if (bank == targetBank) {
      int elementByteSize =
          elementsAttr.getRawData().size() / elementsAttr.getNumElements();
      for (int i = 0; i < elementByteSize; i++) {
        outDataVec.push_back(
            elementsAttr.getRawData()[flattenedIndex * elementByteSize + i]);
      }
    }
    return;
  }
  for (int i = 0; i < memref.getShape()[rank]; i++) {
    int newBank = bank;
    if (partition[rank]) {
      newBank = getNewBank(bank, i, factorMap[rank], cyclicMap[rank],
                           memref.getShape()[rank]);
    }
    getInitialValue(
        memref, rank + 1, flattenedIndex * memref.getShape()[rank] + i, newBank,
        targetBank, elementsAttr, partition, factorMap, cyclicMap, outDataVec);
  }
}

void createNewArray(Operation *op, SmallVector<Value> &newArray,
                    SmallVector<bool> partition, MemRefType memref,
                    DenseMap<int, int> factorMap, PatternRewriter &rewriter,
                    int rank, SmallVector<int64_t> newShape,
                    DenseMap<int, bool> cyclicMap) {
  if (rank == memref.getRank()) {
    auto newMemref = MemRefType::get(newShape, memref.getElementType());
    if (auto funcOp = dyn_cast<FuncOp>(op)) {
      funcOp.insertArgument(funcOp.getNumArguments(), newMemref, {},
                            funcOp.getLoc());
      newArray.push_back(funcOp.getArgument(funcOp.getNumArguments() - 1));
    } else if (auto allocOp = dyn_cast<memref::AllocaOp>(op)) {
      auto newAllocOp =
          rewriter.create<memref::AllocaOp>(allocOp->getLoc(), newMemref);
      newArray.push_back(newAllocOp->getResult(0));
    } else if (auto globalOp = dyn_cast<memref::GlobalOp>(op)) {
      Attribute initValue = rewriter.getUnitAttr();
      if (globalOp.getInitialValue().has_value()) {
        if (auto elementsAttr = llvm::dyn_cast<DenseElementsAttr>(
                globalOp.getInitialValue().value())) {
          SmallVector<char> outDataVec;
          getInitialValue(memref, 0, 0, 0, newArray.size(), elementsAttr,
                          partition, factorMap, cyclicMap, outDataVec);
          auto tensorType = RankedTensorType::get(
              newShape, elementsAttr.getType().getElementType());
          initValue =
              DenseElementsAttr::getFromRawBuffer(tensorType, outDataVec);
        }
      }
      std::string newMemrefName =
          (globalOp.getSymName() + "_" + std::to_string(newArray.size())).str();
      auto newGlobalOp = rewriter.create<memref::GlobalOp>(
          globalOp->getLoc(), rewriter.getStringAttr(newMemrefName),
          /*sym_visibility*/ mlir::StringAttr(), mlir::TypeAttr::get(newMemref),
          initValue, mlir::UnitAttr(),
          /*alignment*/ nullptr);
      auto getGlobalOp = rewriter.create<memref::GetGlobalOp>(
          newGlobalOp->getLoc(), newMemref, newGlobalOp.getSymName());
      newArray.push_back(getGlobalOp->getResult(0));
    } else {
      llvm_unreachable("Unknown op");
    }
    return;
  }
  if (partition[rank]) {
    int smallRank = memref.getShape()[rank] / factorMap[rank];
    int bigRankNum = memref.getShape()[rank] % factorMap[rank];
    for (int i = 0; i < bigRankNum; i++) {
      SmallVector<int64_t> iterNewShape(newShape);
      iterNewShape.push_back(smallRank + 1);
      createNewArray(op, newArray, partition, memref, factorMap, rewriter,
                     rank + 1, iterNewShape, cyclicMap);
    }
    if (smallRank) {
      for (int i = bigRankNum; i < factorMap[rank]; i++) {
        SmallVector<int64_t> iterNewShape(newShape);
        iterNewShape.push_back(smallRank);
        createNewArray(op, newArray, partition, memref, factorMap, rewriter,
                       rank + 1, iterNewShape, cyclicMap);
      }
    }
  } else {
    SmallVector<int64_t> iterNewShape(newShape);
    iterNewShape.push_back(memref.getShape()[rank]);
    createNewArray(op, newArray, partition, memref, factorMap, rewriter,
                   rank + 1, iterNewShape, cyclicMap);
  }
}

struct Partition {
  Operation *op;
  unsigned bank;
};

int getDimBank(AffineMap map, int rank, PatternRewriter &rewriter, int factor,
               bool cyclic, int rankShape) {
  unsigned bank_factor = cyclic ? factor : rankShape / factor;
  auto *ctx = rewriter.getContext();
  int addBank = getMemBank(map, rank, ctx, bank_factor, cyclic);
  if (!cyclic && addBank >= factor) {
    return getMemBank(map, rank, ctx, factor, true);
  }
  return addBank;
}

AffineExpr getDimExpr(AffineMap map, int rank, PatternRewriter &rewriter,
                      int factor, bool cyclic, int rankShape) {
  unsigned bank_factor = cyclic ? factor : rankShape / factor;
  auto *ctx = rewriter.getContext();
  int addBank = getMemBank(map, rank, ctx, bank_factor, cyclic);
  auto dimExpr = getAffineDimExpr(rank, ctx);
  auto expr = cyclic ? dimExpr.floorDiv(bank_factor) : dimExpr % bank_factor;
  if (!cyclic && addBank >= factor) {
    int offset = getMemBank(map, rank, ctx, bank_factor, true);
    expr = expr + bank_factor - offset;
  }
  return expr;
}

void getExprs(SmallVector<AffineExpr> &exprs, AffineExpr expr, int rank,
              int rankNum, PatternRewriter &rewriter) {
  for (int i = 0; i < rankNum; ++i) {
    if (i != rank) {
      exprs.push_back(getAffineDimExpr(i, rewriter.getContext()));
    } else {
      exprs.push_back(expr);
    }
  }
}

template <typename T>
void changeAccessOpAttr(T loadOrStore, SmallVector<AffineExpr> exprs,
                        PatternRewriter &rewriter) {
  auto map = loadOrStore.getAffineMap();
  loadOrStore->setAttr(
      loadOrStore.getMapAttrStrName(),
      AffineMapAttr::get(
          AffineMap::get(map.getNumResults(), 0, exprs, rewriter.getContext())
              .compose(map)));
}

template <typename T>
void changeAccessOpAttrFunc(T loadOrStore, int rank, PatternRewriter &rewriter,
                            int factor, bool cyclic, int rankShape,
                            unsigned rankNum) {
  SmallVector<AffineExpr> exprs;
  auto expr = getDimExpr(loadOrStore.getAffineMap(), rank, rewriter, factor,
                         cyclic, rankShape);
  getExprs(exprs, expr, rank, rankNum, rewriter);
  changeAccessOpAttr(loadOrStore, exprs, rewriter);
}

template <typename T>
void calBankAndChangeOpAttr(T loadOrStore, unsigned &bank, int rank,
                            PatternRewriter &rewriter, int factor, bool cyclic,
                            int rankShape, unsigned rankNum) {
  auto map = loadOrStore.getAffineMap();
  bank = bank * factor +
         getDimBank(map, rank, rewriter, factor, cyclic, rankShape);
  changeAccessOpAttrFunc(loadOrStore, rank, rewriter, factor, cyclic, rankShape,
                         rankNum);
}

void changeMemrefAndOperands(Value arg, MemRefType memref,
                             DenseMap<int, int> factorMap,
                             DenseMap<int, bool> cyclicMap,
                             PatternRewriter &rewriter,
                             SmallVector<bool> partition,
                             SmallVector<Value> newArray) {
  SmallVector<Partition> new_part;
  SmallVector<AffineLoadOp> loadsToErase; // For dynamic bank loads that get replaced
  SmallVector<AffineStoreOp> storesToErase; // For dynamic bank stores that get replaced
  for (auto &use : llvm::make_early_inc_range(arg.getUses())) {
    auto op = use.getOwner();
    if (auto load = dyn_cast<AffineLoadOp>(op)) {
      unsigned bank = 0;
      bool hasDynamicBank = false;
      for (unsigned rank = 0; rank < memref.getRank(); ++rank) {
        if (partition[rank]) {
          auto map = load.getAffineMap();
          int dimBank = getDimBank(map, rank, rewriter, factorMap[rank],
                                   cyclicMap[rank], memref.getShape()[rank]);
          if (dimBank == -1) {
            hasDynamicBank = true;
            break;
          }
          calBankAndChangeOpAttr(load, bank, rank, rewriter, factorMap[rank],
                                 cyclicMap[rank], memref.getShape()[rank],
                                 memref.getRank());
        }
      }

      if (hasDynamicBank) {
        // Dynamic bank selection: generate loads from all possible banks and mux
        // Find the rank with dynamic bank and enumerate possible values
        auto *ctx = rewriter.getContext();
        auto map = load.getAffineMap();

        // Calculate possible banks for the whole access (considering all partitioned ranks)
        // For simplicity, we assume single-rank partition for dynamic case
        SmallVector<int64_t> possibleBanks;
        int dynamicRank = -1;
        int factor = 1;
        bool cyclic = true;

        for (unsigned rank = 0; rank < memref.getRank(); ++rank) {
          if (partition[rank]) {
            factor = factorMap[rank];
            cyclic = cyclicMap[rank];
            unsigned bank_factor = cyclic ? factor : memref.getShape()[rank] / factor;
            SmallVector<Value> operands(load.getMapOperands().begin(),
                                         load.getMapOperands().end());
            possibleBanks = getPossibleBankValues(map.getResult(rank), map,
                                                   operands, bank_factor);
            if (!possibleBanks.empty()) {
              dynamicRank = rank;
              break;
            }
          }
        }

        if (possibleBanks.empty() || dynamicRank < 0) {
          // Cannot enumerate possible banks, this should have been caught earlier
          load->emitError() << "Cannot enumerate possible banks for dynamic bank selection";
          continue;
        }

        LLVM_DEBUG(llvm::dbgs() << "Dynamic bank load: possible banks = {";
                   for (auto b : possibleBanks) llvm::dbgs() << b << ",";
                   llvm::dbgs() << "}\n");

        rewriter.setInsertionPoint(load);

        // Generate a load from each possible bank
        SmallVector<Value> bankLoads;
        for (int64_t bankIdx : possibleBanks) {
          // Create new affine map for this bank's access
          // The index expression becomes: (original_expr - bank) / factor (for cyclic)
          SmallVector<AffineExpr> newExprs;
          for (unsigned r = 0; r < memref.getRank(); ++r) {
            if (r == (unsigned)dynamicRank && partition[r]) {
              unsigned bank_factor = cyclic ? factor : memref.getShape()[r] / factor;
              auto expr = map.getResult(r);
              // New index = (expr - bankIdx) / bank_factor for cyclic
              // or expr % bank_factor for block
              if (cyclic) {
                newExprs.push_back(expr.floorDiv(bank_factor));
              } else {
                newExprs.push_back(expr % bank_factor);
              }
            } else {
              newExprs.push_back(getAffineDimExpr(r, ctx));
            }
          }
          auto newMap = AffineMap::get(map.getNumDims(), map.getNumSymbols(), newExprs, ctx);

          auto bankLoad = rewriter.create<AffineLoadOp>(
              load.getLoc(), newArray[bankIdx], newMap, load.getMapOperands());
          // Copy all attributes from original load except the affine map
          for (auto attr : load->getAttrs()) {
            if (attr.getName() != load.getMapAttrStrName())
              bankLoad->setAttr(attr.getName(), attr.getValue());
          }
          bankLoads.push_back(bankLoad.getResult());
        }

        // Build the runtime bank index computation using simplifyMod
        unsigned bank_factor = cyclic ? factor : memref.getShape()[dynamicRank] / factor;
        auto simplifiedBankExpr = tor::simplifyMod(map.getResult(dynamicRank), bank_factor,
                                                    map.getNumDims(), map.getNumSymbols());
        auto bankMap = AffineMap::get(map.getNumDims(), map.getNumSymbols(), simplifiedBankExpr, ctx);
        auto runtimeBank = rewriter.create<AffineApplyOp>(
            load.getLoc(), bankMap, load.getMapOperands());

        // Build mux: cascade of selects
        // select(bank == possibleBanks[n-1], loads[n-1],
        //   select(bank == possibleBanks[n-2], loads[n-2], ...))
        Value result = bankLoads.back();
        for (int i = possibleBanks.size() - 2; i >= 0; --i) {
          auto bankConst = rewriter.create<arith::ConstantIndexOp>(
              load.getLoc(), possibleBanks[i]);
          auto cmp = rewriter.create<arith::CmpIOp>(
              load.getLoc(), arith::CmpIPredicate::eq, runtimeBank, bankConst);
          result = rewriter.create<arith::SelectOp>(
              load.getLoc(), cmp, bankLoads[i], result);
        }

        // Replace original load with mux result
        load.getResult().replaceAllUsesWith(result);
        loadsToErase.push_back(load);
        continue; // Don't add to new_part
      }

      new_part.push_back(Partition{load, bank});
    } else if (auto store = dyn_cast<AffineStoreOp>(op)) {
      unsigned bank = 0;
      bool hasDynamicBank = false;
      auto storeMap = store.getAffineMap();

      for (unsigned rank = 0; rank < memref.getRank(); ++rank) {
        if (partition[rank]) {
          int dimBank = getDimBank(storeMap, rank, rewriter, factorMap[rank],
                                   cyclicMap[rank], memref.getShape()[rank]);
          if (dimBank == -1) {
            hasDynamicBank = true;
            break;
          }
          calBankAndChangeOpAttr(store, bank, rank, rewriter, factorMap[rank],
                                 cyclicMap[rank], memref.getShape()[rank],
                                 memref.getRank());
        }
      }

      if (hasDynamicBank) {
        // Dynamic bank store: read-modify-write all possible banks
        auto *ctx = rewriter.getContext();

        // Find possible banks (same logic as load)
        SmallVector<int64_t> possibleBanks;
        int dynamicRank = -1;
        int factor = 1;
        bool cyclic = true;

        for (unsigned rank = 0; rank < memref.getRank(); ++rank) {
          if (partition[rank]) {
            factor = factorMap[rank];
            cyclic = cyclicMap[rank];
            unsigned bank_factor = cyclic ? factor : memref.getShape()[rank] / factor;
            SmallVector<Value> operands(store.getMapOperands().begin(),
                                         store.getMapOperands().end());
            possibleBanks = getPossibleBankValues(storeMap.getResult(rank), storeMap,
                                                   operands, bank_factor);
            if (!possibleBanks.empty()) {
              dynamicRank = rank;
              break;
            }
          }
        }

        if (possibleBanks.empty() || dynamicRank < 0) {
          store->emitError() << "Cannot enumerate possible banks for dynamic bank store";
          continue;
        }

        LLVM_DEBUG(llvm::dbgs() << "Dynamic bank store: possible banks = {";
                   for (auto b : possibleBanks) llvm::dbgs() << b << ",";
                   llvm::dbgs() << "}\n");

        rewriter.setInsertionPoint(store);

        // Build index map for accessing partitioned banks
        SmallVector<AffineExpr> newExprs;
        for (unsigned r = 0; r < memref.getRank(); ++r) {
          if (r == (unsigned)dynamicRank && partition[r]) {
            unsigned bank_factor = cyclic ? factor : memref.getShape()[r] / factor;
            auto expr = storeMap.getResult(r);
            if (cyclic) {
              newExprs.push_back(expr.floorDiv(bank_factor));
            } else {
              newExprs.push_back(expr % bank_factor);
            }
          } else {
            newExprs.push_back(getAffineDimExpr(r, ctx));
          }
        }
        auto indexMap = AffineMap::get(storeMap.getNumDims(), storeMap.getNumSymbols(), newExprs, ctx);

        // Compute runtime bank index using simplifyMod
        unsigned bank_factor = cyclic ? factor : memref.getShape()[dynamicRank] / factor;
        auto simplifiedBankExpr = tor::simplifyMod(storeMap.getResult(dynamicRank), bank_factor,
                                                    storeMap.getNumDims(), storeMap.getNumSymbols());
        auto bankMap = AffineMap::get(storeMap.getNumDims(), storeMap.getNumSymbols(), simplifiedBankExpr, ctx);
        auto runtimeBank = rewriter.create<AffineApplyOp>(
            store.getLoc(), bankMap, store.getMapOperands());

        Value storeValue = store.getValue();

        // For each possible bank: read old value, select new/old, write back
        for (int64_t bankIdx : possibleBanks) {
          // Read old value from this bank
          auto oldValue = rewriter.create<AffineLoadOp>(
              store.getLoc(), newArray[bankIdx], indexMap, store.getMapOperands());

          // Select: if runtime_bank == bankIdx, use storeValue, else use oldValue
          auto bankConst = rewriter.create<arith::ConstantIndexOp>(store.getLoc(), bankIdx);
          auto cmp = rewriter.create<arith::CmpIOp>(
              store.getLoc(), arith::CmpIPredicate::eq, runtimeBank, bankConst);
          auto newValue = rewriter.create<arith::SelectOp>(
              store.getLoc(), cmp, storeValue, oldValue);

          // Write back to this bank
          auto newStore = rewriter.create<AffineStoreOp>(
              store.getLoc(), newValue, newArray[bankIdx], indexMap, store.getMapOperands());
          // Copy all attributes from original store except the affine map
          for (auto attr : store->getAttrs()) {
            if (attr.getName() != store.getMapAttrStrName())
              newStore->setAttr(attr.getName(), attr.getValue());
          }
        }

        // Erase original store
        storesToErase.push_back(store);
        continue;
      }

      new_part.push_back(Partition{store, bank});
    } else if (auto burstLoad = dyn_cast<aps::MemBurstLoad>(op)) {
      SmallVector<Value> newOperands;
      newOperands.push_back(burstLoad.getCpuAddr());  // cpu_addr
      for (auto memref : newArray) {
        newOperands.push_back(memref);
      }
      newOperands.push_back(burstLoad.getStart());    // start index
      newOperands.push_back(burstLoad.getLength());   // length

      rewriter.setInsertionPoint(burstLoad);
      auto newBurstLoad = rewriter.replaceOpWithNewOp<aps::MemBurstLoad>(burstLoad, TypeRange{}, newOperands);

      // Copy all attributes from original operation first
      for (auto attr : burstLoad->getAttrs()) {
        newBurstLoad->setAttr(attr.getName(), attr.getValue());
      }

      // Then update partition attributes for the new partitioned array
      SmallVector<mlir::Attribute> dimAttrs, factorAttrs, cyclicAttrs;
      auto ctx = rewriter.getContext();
      for (size_t rank = 0; rank < partition.size(); ++rank) {
        if (partition[rank]) {
          dimAttrs.push_back(mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), rank));
          factorAttrs.push_back(mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), factorMap.at(rank)));
          cyclicAttrs.push_back(mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), cyclicMap.at(rank)));
        }
      }
      if (!dimAttrs.empty()) {
        newBurstLoad->setAttr("partition_dim_array", ArrayAttr::get(ctx, dimAttrs));
        newBurstLoad->setAttr("partition_factor_array", ArrayAttr::get(ctx, factorAttrs));
        newBurstLoad->setAttr("partition_cyclic_array", ArrayAttr::get(ctx, cyclicAttrs));
      }
    } else if (auto burstStore = dyn_cast<aps::MemBurstStore>(op)) {
      SmallVector<Value> newOperands;
      for (auto memref : newArray) {
        newOperands.push_back(memref);
      }
      newOperands.push_back(burstStore.getStart());     // start index
      newOperands.push_back(burstStore.getCpuAddr());   // cpu_addr
      newOperands.push_back(burstStore.getLength());    // length

      rewriter.setInsertionPoint(burstStore);
      auto newBurstStore = rewriter.replaceOpWithNewOp<aps::MemBurstStore>(burstStore, TypeRange{}, newOperands);

      // Copy all attributes from original operation first
      for (auto attr : burstStore->getAttrs()) {
        newBurstStore->setAttr(attr.getName(), attr.getValue());
      }

      // Then update partition attributes for the new partitioned array
      SmallVector<mlir::Attribute> dimAttrs, factorAttrs, cyclicAttrs;
      auto ctx = rewriter.getContext();
      for (size_t rank = 0; rank < partition.size(); ++rank) {
        if (partition[rank]) {
          dimAttrs.push_back(mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), rank));
          factorAttrs.push_back(mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), factorMap.at(rank)));
          cyclicAttrs.push_back(mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), cyclicMap.at(rank)));
        }
      }
      if (!dimAttrs.empty()) {
        newBurstStore->setAttr("partition_dim_array", ArrayAttr::get(ctx, dimAttrs));
        newBurstStore->setAttr("partition_factor_array", ArrayAttr::get(ctx, factorAttrs));
        newBurstStore->setAttr("partition_cyclic_array", ArrayAttr::get(ctx, cyclicAttrs));
      }
    } else if (auto callOp = dyn_cast<func::CallOp>(op)) {
      auto operands = op->getOperands();
      SmallVector<Value, 4> newOperands;
      for (unsigned int i = 0, e = operands.size(); i < e; i++) {
        if (i != use.getOperandNumber()) {
          newOperands.push_back(operands[i]);
        }
      }
      for (auto val : newArray) {
        newOperands.push_back(val);
      }
      rewriter.setInsertionPoint(callOp);
      auto newCallOp = rewriter.replaceOpWithNewOp<func::CallOp>(
          callOp, callOp.getCallee(), callOp.getResultTypes(), newOperands);
      // Copy all attributes from original call operation
      for (auto attr : callOp->getAttrs()) {
        newCallOp->setAttr(attr.getName(), attr.getValue());
      }
    }
  }
  for (auto part : new_part) {
    // Determine the correct memref operand index:
    // - AffineLoadOp: memref is operand 0
    // - AffineStoreOp: memref is operand 1 (operand 0 is the value)
    unsigned memrefOperandIdx = isa<AffineStoreOp>(part.op) ? 1 : 0;

    // Check for invalid bank index (0xFFFFFFFF = -1 cast to unsigned)
    // This indicates getDimBank returned -1 due to non-constant access pattern
    if (part.bank == static_cast<unsigned>(-1)) {
      llvm::errs() << "\n=== ARRAY PARTITION ERROR ===\n";
      llvm::errs() << "Fatal: Cannot determine memory bank for operation with non-constant index\n\n";
      llvm::errs() << "Operation: " << *part.op << "\n\n";

      // Get the affine map to show the problematic access pattern
      if (auto load = dyn_cast<AffineLoadOp>(part.op)) {
        llvm::errs() << "Affine map: " << load.getAffineMap() << "\n";
        llvm::errs() << "Memref: " << load.getMemRef() << "\n";
      } else if (auto store = dyn_cast<AffineStoreOp>(part.op)) {
        llvm::errs() << "Affine map: " << store.getAffineMap() << "\n";
        llvm::errs() << "Memref: " << store.getMemRef() << "\n";
      }

      llvm::errs() << "\nReason: The memory access pattern contains symbolic expressions or\n";
      llvm::errs() << "non-constant indices that cannot be statically resolved to a specific\n";
      llvm::errs() << "memory bank during array partitioning.\n\n";
      llvm::errs() << "Possible causes:\n";
      llvm::errs() << "  1. Index depends on loop induction variable in complex ways\n";
      llvm::errs() << "  2. Index contains division/modulo operations that cannot be constant-folded\n";
      llvm::errs() << "  3. Index uses symbolic values (function arguments, loop bounds)\n\n";
      llvm::errs() << "Solutions:\n";
      llvm::errs() << "  1. Simplify the index expression to use constant offsets\n";
      llvm::errs() << "  2. Apply loop unrolling first to expose constant indices\n";
      llvm::errs() << "  3. Use simpler partitioning factors that divide evenly\n";
      llvm::errs() << "  4. Do not partition this array (remove partition attributes)\n\n";

      llvm_unreachable("Array partitioning failed: non-constant memory bank index");
    }

    // Bounds check: ensure bank index is within newArray bounds
    if (part.bank >= newArray.size()) {
      llvm::errs() << "\n=== ARRAY PARTITION ERROR ===\n";
      llvm::errs() << "Fatal: Bank index out of bounds\n\n";
      llvm::errs() << "Bank index: " << part.bank << "\n";
      llvm::errs() << "Array size: " << newArray.size() << "\n";
      llvm::errs() << "Operation: " << *part.op << "\n\n";
      llvm::errs() << "This indicates a mismatch between the calculated bank index\n";
      llvm::errs() << "and the number of partitioned arrays created.\n\n";
      llvm_unreachable("Array partitioning failed: bank index exceeds array bounds");
    }

    part.op->setOperand(memrefOperandIdx, newArray[part.bank]);
  }

  // Erase loads that were replaced with dynamic bank mux
  for (auto load : loadsToErase) {
    load->erase();
  }

  // Erase stores that were replaced with dynamic bank read-modify-write
  for (auto store : storesToErase) {
    store->erase();
  }
}

void partitionFunc(Operation *op, SmallVector<bool> partition,
                   MemRefType memref, DenseMap<int, int> factorMap,
                   DenseMap<int, bool> cyclicMap, PatternRewriter &rewriter,
                   Value arg) {
  SmallVector<Value> newArray;
  SmallVector<int64_t> newShape;
  createNewArray(op, newArray, partition, memref, factorMap, rewriter, 0,
                 newShape, cyclicMap);
  changeMemrefAndOperands(arg, memref, factorMap, cyclicMap, rewriter,
                          partition, newArray);
}

struct AllocaOpPattern : OpRewritePattern<memref::AllocaOp> {
  AllocaOpPattern(MLIRContext *ctx) : OpRewritePattern<memref::AllocaOp>(ctx) {}

  LogicalResult matchAndRewrite(memref::AllocaOp op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr("array-partition") || !op->hasAttr("partition_dim_array"))
      return failure();
    op->setAttr("array-partition",
                IntegerAttr::get(IntegerType::get(getContext(), 32), 1));
    auto arg = op->getResult(0);
    auto memref = cast<MemRefType>(arg.getType());
    if (!checkValueUsers(arg, memref.getRank())) {
      // warningNonStandardAffineAccess("alloca", op->getAttr("var_name"));
      return failure();
    }
    DenseMap<int, int> factorMap;
    DenseMap<int, bool> cyclicMap;
    bool fullyPartition = isFullyPartition(
        dyn_cast<mlir::ArrayAttr>(op->getAttr("partition_dim_array")));
    getFactorMapAndCyclicMap(op, memref, factorMap, cyclicMap, fullyPartition);
    SmallVector<bool> partition;
    getPartitionVector(partition, op, memref, factorMap, cyclicMap, "alloca",
                       op->getAttr("var_name"), rewriter, fullyPartition);
    if (!needPartition(partition)) {
      return success();
    }
    LLVM_DEBUG(llvm::dbgs() << "handle operation: " << op << "\n");
    partitionFunc(op, partition, memref, factorMap, cyclicMap, rewriter, arg);
    return success();
  }
};

void getGlobalPartitionVector(
    SmallVector<bool> &partition, Operation *op,
    SmallVector<memref::GetGlobalOp> &getGlobalOpArray, MemRefType memref,
    DenseMap<int, int> factorMap, DenseMap<int, bool> cyclicMap, StringRef type,
    Attribute varNameAttr, PatternRewriter &rewriter, bool fullyPartition) {
  for (size_t rank = 0, e = memref.getRank(); rank < e; ++rank) {
    bool flag = false;
    if (factorMap.count(rank)) {
      unsigned bank_factor = cyclicMap[rank]
                                 ? factorMap[rank]
                                 : memref.getShape()[rank] / factorMap[rank];
      std::string lineString = "array_partition";
      lineString += "_dim_" + llvm::Twine(rank).str();
      for (auto getGlobalOp : getGlobalOpArray) {
        flag = rankCanBePartition(getGlobalOp, rank, bank_factor,
                                  cyclicMap[rank], rewriter.getContext());
        if (!flag) {
          if (fullyPartition) {
            partition.clear();
            setPragmaStructureAttrStatusByOp(op, "array_partition", false);
            warningBankCannotBeDivided(type, varNameAttr, 0);
            return;
          }
          warningBankCannotBeDivided(type, varNameAttr, rank + 1);
          setPragmaStructureAttrStatusByOp(op, lineString, false);
          break;
        }
      }
      if (flag) {
        setPragmaStructureAttrStatusByOp(op, lineString);
      }
    }
    partition.push_back(flag);
  }
  if (fullyPartition) {
    setPragmaStructureAttrStatusByOp(op, "array_partition");
  }
}

void getNewGlobalArray(Operation *op, SmallVector<Value> newArray,
                       SmallVector<Value> &newPartitionArray,
                       PatternRewriter &rewriter) {
  for (auto value : newArray) {
    auto getGlobalOp = dyn_cast<memref::GetGlobalOp>(value.getDefiningOp());
    rewriter.setInsertionPoint(op);
    auto newGetGlobalOp = rewriter.create<memref::GetGlobalOp>(
        op->getLoc(), getGlobalOp.getType(), getGlobalOp.getName());
    newPartitionArray.push_back(newGetGlobalOp);
  }
}

void updateMemoryMapEntry(Operation *globalOp, SmallVector<Value> &newArray,
                         DenseMap<int, int> factorMap, DenseMap<int, bool> cyclicMap,
                         PatternRewriter &rewriter) {
  auto globalOpCast = cast<memref::GlobalOp>(globalOp);
  auto symName = globalOpCast.getSymName();

  // Find the memory map in the module
  auto moduleOp = globalOp->getParentOfType<ModuleOp>();
  aps::MemoryMapOp memoryMapOp;
  moduleOp->walk([&](aps::MemoryMapOp op) {
    memoryMapOp = op;
    return WalkResult::interrupt();
  });

  if (!memoryMapOp) {
    return; // No memory map found
  }

  // Find the mem_entry that references this global
  aps::MemEntryOp targetEntry;
  memoryMapOp.getRegion().walk([&](aps::MemEntryOp entry) {
    auto banks = entry.getBankSymbols();
    for (auto bankAttr : banks) {
      if (auto symRef = dyn_cast<FlatSymbolRefAttr>(bankAttr)) {
        if (symRef.getValue() == symName) {
          targetEntry = entry;
          return WalkResult::interrupt();
        }
      }
    }
    return WalkResult::advance();
  });

  if (!targetEntry) {
    return; // Entry not found
  }

  // Create new bank symbols array from the partitioned memrefs
  SmallVector<Attribute> newBankSymbols;
  for (auto val : newArray) {
    auto getGlobalOp = cast<memref::GetGlobalOp>(val.getDefiningOp());
    // Get the actual GlobalOp to extract its symbol
    auto globalName = getGlobalOp.getName();
    newBankSymbols.push_back(FlatSymbolRefAttr::get(rewriter.getContext(), globalName));
  }

  // Get partition info (use first partitioned dimension)
  uint32_t numBanks = newArray.size();
  uint32_t cyclicMode = 0;
  if (!factorMap.empty() && !cyclicMap.empty()) {
    // Get the first partitioned dimension
    auto firstDim = factorMap.begin()->first;
    cyclicMode = cyclicMap[firstDim] ? 1 : 0;
  }

  // Create a new mem_entry with updated banks
  rewriter.setInsertionPoint(targetEntry);
  rewriter.create<aps::MemEntryOp>(
    targetEntry.getLoc(),
    targetEntry.getNameAttr(),
    rewriter.getArrayAttr(newBankSymbols),
    targetEntry.getBaseAddressAttr(),
    targetEntry.getBankSizeAttr(),
    rewriter.getUI32IntegerAttr(numBanks),
    rewriter.getUI32IntegerAttr(cyclicMode)
  );

  // Erase the old entry
  rewriter.eraseOp(targetEntry);
}

void globalPartitionFunc(Operation *op, SmallVector<bool> partition,
                         MemRefType memref, DenseMap<int, int> factorMap,
                         DenseMap<int, bool> cyclicMap,
                         PatternRewriter &rewriter,
                         SmallVector<memref::GetGlobalOp> &getGlobalOpArray) {
  SmallVector<Value> newArray;
  SmallVector<int64_t> newShape;
  createNewArray(op, newArray, partition, memref, factorMap, rewriter, 0,
                 newShape, cyclicMap);
  for (auto getGlobalOp : getGlobalOpArray) {
    SmallVector<Value> newGlobalArray;
    getNewGlobalArray(getGlobalOp, newArray, newGlobalArray, rewriter);
    changeMemrefAndOperands(getGlobalOp, memref, factorMap, cyclicMap, rewriter,
                            partition, newGlobalArray);
  }

  // Update the memory map entry for this global
  updateMemoryMapEntry(op, newArray, factorMap, cyclicMap, rewriter);

  for (auto AI : newArray) {
    AI.getDefiningOp()->erase();
  }
}

struct GlobalOpPattern : OpRewritePattern<memref::GlobalOp> {
  DenseMap<StringRef, SmallVector<memref::GetGlobalOp>> &newGetGlobalOpMap;
  SmallVector<Operation *> &accessOpsToErase;

  GlobalOpPattern(
      MLIRContext *ctx,
      DenseMap<StringRef, SmallVector<memref::GetGlobalOp>> &newGetGlobalOpMap,
      SmallVector<Operation *> &accessOpsToErase)
      : OpRewritePattern<memref::GlobalOp>(ctx),
        newGetGlobalOpMap(newGetGlobalOpMap),
        accessOpsToErase(accessOpsToErase) {}

  LogicalResult matchAndRewrite(memref::GlobalOp op,
                                PatternRewriter &rewriter) const override {

    if (op->hasAttr("array-partition") || !op->hasAttr("partition_dim_array"))
      return failure();
    op->setAttr("array-partition",
                IntegerAttr::get(IntegerType::get(getContext(), 32), 1));
    auto memref = cast<MemRefType>(op.getType());
    auto getGlobalOpArray = newGetGlobalOpMap[op.getSymName()];
    for (auto getGlobalOp : getGlobalOpArray) {
      if (!checkValueUsers(getGlobalOp, memref.getRank())) {
        warningNonStandardAffineAccess("global", op->getAttr("var_name"));
        return failure();
      }
    }
    DenseMap<int, int> factorMap;
    DenseMap<int, bool> cyclicMap;
    bool fullyPartition = isFullyPartition(
        dyn_cast<mlir::ArrayAttr>(op->getAttr("partition_dim_array")));
    getFactorMapAndCyclicMap(op, memref, factorMap, cyclicMap, fullyPartition);
    SmallVector<bool> partition;
    getGlobalPartitionVector(partition, op, getGlobalOpArray, memref, factorMap,
                             cyclicMap, "global", op->getAttr("var_name"),
                             rewriter, fullyPartition);
    if (!needPartition(partition)) {
      return success();
    }
    LLVM_DEBUG(llvm::dbgs() << "handle operation: " << op << "\n");
    globalPartitionFunc(op, partition, memref, factorMap, cyclicMap, rewriter,
                        getGlobalOpArray);
    for (auto getGlobalOp : getGlobalOpArray) {
      accessOpsToErase.push_back(getGlobalOp);
    }
    accessOpsToErase.push_back(op);
    return success();
  }
};

void getArgFactorMapAndCyclicMap(FuncOp op, MemRefType memref, int argIndex,
                                 DenseMap<int, int> &factorMap,
                                 DenseMap<int, bool> &cyclicMap,
                                 bool fullyPartition) {
  auto partitionDimArray =
      dyn_cast<mlir::ArrayAttr>(op->getAttr(stringAddNumber("partition_dim_array_", argIndex)));
  auto partitionFactorArray =
      dyn_cast<mlir::ArrayAttr>(op->getAttr(stringAddNumber("partition_factor_array_", argIndex)));
  auto partitionCyclicArray =
      dyn_cast<mlir::ArrayAttr>(op->getAttr(stringAddNumber("partition_cyclic_array_", argIndex)));
  if (fullyPartition) {
    int factor = getAttrInterger(partitionFactorArray[0]);
    for (int i = 0, e = memref.getShape().size(); i < e; i++) {
      if (factor == -1) {
        factorMap[i] = memref.getShape()[i];
      } else {
        factorMap[i] = factor;
      }
      cyclicMap[i] = getAttrInterger(partitionCyclicArray[0]);
    }
    return;
  }
  for (int i = 0, e = partitionDimArray.size(); i < e; i++) {
    int partitionDim = getAttrInterger(partitionDimArray[i]);
    factorMap[partitionDim] = getAttrInterger(partitionFactorArray[i]);
    cyclicMap[partitionDim] = getAttrInterger(partitionCyclicArray[i]);
  }
}

struct FuncOpPattern : OpRewritePattern<FuncOp> {
  FuncOpPattern(MLIRContext *ctx) : OpRewritePattern<FuncOp>(ctx) {}

  LogicalResult matchAndRewrite(FuncOp op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr("array-partition"))
      return failure();
    op->setAttr("array-partition",
                IntegerAttr::get(IntegerType::get(getContext(), 32), 1));
    for (int i = op.getArguments().size() - 1; i >= 0; i--) {
      auto arg = op.getArgument(i);
      if (!isa<mlir::MemRefType>(arg.getType()) ||
          !op->hasAttr(stringAddNumber("partition_cyclic_array_", i))) {
        continue;
      }
      auto memref = cast<MemRefType>(arg.getType());
      if (!checkValueUsers(arg, memref.getRank())) {
        // warningNonStandardAffineAccess(
        //     "argument", op->getAttr(stringAddNumber("var_name_", i)));
        continue;
      }
      DenseMap<int, int> factorMap;
      DenseMap<int, bool> cyclicMap;
      bool fullyPartition = isFullyPartition(
              dyn_cast<mlir::ArrayAttr>(op->getAttr(stringAddNumber("partition_dim_array_", i))));
      getArgFactorMapAndCyclicMap(op, memref, i, factorMap, cyclicMap,
                                  fullyPartition);
      SmallVector<bool> partition;
      getPartitionVector(partition, arg, memref, factorMap, cyclicMap,
                         "argument",
                         op->getAttr(stringAddNumber("var_name_", i)), rewriter,
                         fullyPartition);
      if (!needPartition(partition)) {
        continue;
      }
      partitionFunc(op, partition, memref, factorMap, cyclicMap, rewriter, arg);
      op.eraseArgument(i);
      op->removeAttr(stringAddNumber("partition_dim_array_", i));
      op->removeAttr(stringAddNumber("partition_factor_array_", i));
      op->removeAttr(stringAddNumber("partition_cyclic_array_", i));
    }

    return success();
  }
};

bool checkPartitionVector(Value val, DenseMap<int, int> &factorMap,
                          DenseMap<int, bool> &cyclicMap, bool fullyPartition) {
  auto ctx = getDefiningOpByValue(val)->getContext();
  auto memref = cast<MemRefType>(val.getType());
  for (size_t rank = 0, e = memref.getRank(); rank < e; ++rank) {
    if (factorMap.count(rank)) {
      unsigned bank_factor = cyclicMap[rank]
                                 ? factorMap[rank]
                                 : memref.getShape()[rank] / factorMap[rank];
      std::string lineString = "array_partition";
      if (auto blockArg = dyn_cast<mlir::BlockArgument>(val)) {
        lineString += "_arg_" + llvm::Twine(blockArg.getArgNumber()).str();
      }
      lineString += "_dim_" + llvm::Twine(rank).str();
      if (!rankCanBePartition(val, rank, bank_factor, cyclicMap[rank], ctx)) {
        if (fullyPartition) {
          setPragmaStructureAttrStatusByValue(val, "array_partition", false);
          warningBankCannotBeDividedByVal(val, 0);
          factorMap.clear();
          return false;
        }
        factorMap.erase(rank);
        setPragmaStructureAttrStatusByValue(val, lineString, false);
        warningBankCannotBeDividedByVal(val, rank + 1);
      }
    }
  }
  return factorMap.size() != 0;
}

std::string getDimStrWithVal(Value val) {
  if (auto blockArg = dyn_cast<mlir::BlockArgument>(val)) {
    return "_" + llvm::Twine(blockArg.getArgNumber()).str();
  }
  return "";
}

void getFullyPartitionFactorMapAndCyclicMap(Value val,
                                            DenseMap<int, int> &factorMap,
                                            DenseMap<int, bool> &cyclicMap,
                                            int factor, int cyclic) {
  factorMap.clear();
  cyclicMap.clear();
  auto memref = cast<MemRefType>(val.getType());
  for (int i = 0, e = memref.getShape().size(); i < e; i++) {
    if (factor == -1) {
      factorMap[i] = memref.getShape()[i];
    } else {
      factorMap[i] = factor;
    }
    cyclicMap[i] = cyclic;
  }
}

void getFactorMapAndCyclicMapWithVal(Value val, DenseMap<int, int> &factorMap,
                                     DenseMap<int, bool> &cyclicMap) {
  auto op = getDefiningOpByValue(val);
  std::string dimStr = getDimStrWithVal(val);
  auto partitionDimArray =
      dyn_cast<mlir::ArrayAttr>(op->getAttr("partition_dim_array" + dimStr));
  auto partitionFactorArray =
      dyn_cast<mlir::ArrayAttr>(op->getAttr("partition_factor_array" + dimStr));
  auto partitionCyclicArray =
      dyn_cast<mlir::ArrayAttr>(op->getAttr("partition_cyclic_array" + dimStr));
  if (isFullyPartition(partitionDimArray)) {
    getFullyPartitionFactorMapAndCyclicMap(
        val, factorMap, cyclicMap, getAttrInterger(partitionFactorArray[0]),
        getAttrInterger(partitionCyclicArray[0]));
    return;
  }
  for (int i = 0, e = partitionDimArray.size(); i < e; i++) {
    int partitionDim = getAttrInterger(partitionDimArray[i]);
    if (factorMap.count(partitionDim) &&
        factorMap[partitionDim] != getAttrInterger(partitionFactorArray[i])) {
      auto lineStr = getLineStringWithRank(val, partitionDim);
    } else {
      factorMap[partitionDim] = getAttrInterger(partitionFactorArray[i]);
    }
    if (cyclicMap.count(partitionDim) &&
        cyclicMap[partitionDim] != getAttrInterger(partitionCyclicArray[i])) {

    } else {
      cyclicMap[partitionDim] = getAttrInterger(partitionCyclicArray[i]);
    }
  }
}

void setInvalidWithVal(Value val, ArrayAttr partitionDimArray) {
  if (isFullyPartition(partitionDimArray)) {
    setPragmaStructureAttrStatusByValue(val, "array_partition", false);
  } else {
    std::string lineString = "array_partition";
    if (auto blockArg = dyn_cast<mlir::BlockArgument>(val)) {
      lineString += "_arg_" + llvm::Twine(blockArg.getArgNumber()).str();
    }
    for (int i = 0, e = partitionDimArray.size(); i < e; i++) {
      int partitionDim = getAttrInterger(partitionDimArray[i]);
      setPragmaStructureAttrStatusByValue(
          val, lineString + "_dim_" + llvm::Twine(partitionDim).str(), false);
    }
  }
}

void removeAttrOneGroupValueSet(DenseSet<Value> &idValueSet) {
  // mark all pragma is false
  for (auto val : idValueSet) {
    auto typeStr = "partition_dim_array" + getDimStrWithVal(val);
    if (auto attr = getDefiningOpByValue(val)->getAttr(typeStr)) {
      setInvalidWithVal(val, dyn_cast<mlir::ArrayAttr>(attr));
    }
  }

  for (auto val : idValueSet) {
    auto op = getDefiningOpByValue(val);
    std::string valDimStr = getDimStrWithVal(val);
    auto removeAttrWithNameAndSuffix = [](mlir::Operation *op, std::string type,
                                          std::string suffix) {
      op->removeAttr(type + suffix);
    };
    removeAttrWithNameAndSuffix(op, "partition_dim_array", valDimStr);
    removeAttrWithNameAndSuffix(op, "partition_factor_array", valDimStr);
    removeAttrWithNameAndSuffix(op, "partition_cyclic_array", valDimStr);
  }
}

void setAttrOneGroupValueSetWithFully(DenseSet<Value> &idValueSet) {
  // mark all pragma is false exclude fully
  for (auto val : idValueSet) {
    auto typeStr = "partition_dim_array" + getDimStrWithVal(val);
    if (auto attr = getDefiningOpByValue(val)->getAttr(typeStr)) {
        if (isFullyPartition(dyn_cast<mlir::ArrayAttr>(attr))) {
        setInvalidWithVal(val, dyn_cast<mlir::ArrayAttr>(attr));
      }
    }
  }

  Value fullyVal;
  for (auto val : idValueSet) {
    auto typeStr = "partition_dim_array" + getDimStrWithVal(val);
    if (auto attr = getDefiningOpByValue(val)->getAttr(typeStr)) {
      if (isFullyPartition(dyn_cast<mlir::ArrayAttr>(attr))) {
        fullyVal = val;
        break;
      }
    }
  }
  auto fullyOp = getDefiningOpByValue(fullyVal);
  std::string fullyDimStr = getDimStrWithVal(fullyVal);
  for (auto val : idValueSet) {
    auto op = getDefiningOpByValue(val);
    std::string valDimStr = getDimStrWithVal(val);
    auto saveAttrWithNameAndSuffix =
        [](mlir::Operation *newOp, mlir::Operation *op, std::string type,
           std::string newSuffix, std::string suffix) {
          if (auto attr = op->getAttr(type + newSuffix)) {
            newOp->setAttr(type + newSuffix, attr);
          }
        };
    saveAttrWithNameAndSuffix(op, fullyOp, "partition_dim_array", valDimStr,
                              fullyDimStr);
    saveAttrWithNameAndSuffix(op, fullyOp, "partition_factor_array", valDimStr,
                              fullyDimStr);
    saveAttrWithNameAndSuffix(op, fullyOp, "partition_cyclic_array", valDimStr,
                              fullyDimStr);
  }
}

void setInvalidWithDenseMap(Value val, ArrayAttr partitionDimArray,
                            DenseMap<int, int> &factorMap) {
  std::string lineString = "array_partition";
  if (auto blockArg = dyn_cast<mlir::BlockArgument>(val)) {
    lineString += "_arg_" + llvm::Twine(blockArg.getArgNumber()).str();
  }
  for (int i = 0, e = partitionDimArray.size(); i < e; i++) {
    int partitionDim = getAttrInterger(partitionDimArray[i]);
    if (!factorMap.count(partitionDim)) {
      setPragmaStructureAttrStatusByValue(
          val, lineString + "_dim_" + llvm::Twine(partitionDim).str(), false);
    }
  }
}

void setAttrOneGroupValueSetWithFactorMap(DenseSet<Value> &idValueSet,
                                          DenseMap<int, int> &factorMap,
                                          DenseMap<int, bool> &cyclicMap) {
  // mark all pragma is false exclude valid dim
  for (auto val : idValueSet) {
    auto typeStr = "partition_dim_array" + getDimStrWithVal(val);
    if (auto attr = getDefiningOpByValue(val)->getAttr(typeStr)) {
      setInvalidWithDenseMap(val, dyn_cast<mlir::ArrayAttr>(attr), factorMap);
    }
  }

  auto ctx = getDefiningOpByValue(*idValueSet.begin())->getContext();
  SmallVector<mlir::Attribute> dimAttrs, factorAttrs, cyclicAttrs;
  for (auto dimAndFactor : factorMap) {
    auto getIntegerAttr = [](int x, MLIRContext *ctx) {
      return mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), x);
    };
    dimAttrs.push_back(getIntegerAttr(dimAndFactor.first, ctx));
    factorAttrs.push_back(getIntegerAttr(dimAndFactor.second, ctx));
    cyclicAttrs.push_back(getIntegerAttr(cyclicMap[dimAndFactor.first], ctx));
  }
  for (auto val : idValueSet) {
    auto op = getDefiningOpByValue(val);
    std::string valDimStr = getDimStrWithVal(val);
    auto setAttrWithNameAndSuffix = [](mlir::Operation *op, std::string type,
                                       std::string suffix, MLIRContext *ctx,
                                       SmallVector<mlir::Attribute> attr) {
      op->setAttr(type + suffix, ArrayAttr::get(ctx, attr));
    };
    setAttrWithNameAndSuffix(op, "partition_dim_array", valDimStr, ctx,
                             dimAttrs);
    setAttrWithNameAndSuffix(op, "partition_factor_array", valDimStr, ctx,
                             factorAttrs);
    setAttrWithNameAndSuffix(op, "partition_cyclic_array", valDimStr, ctx,
                             cyclicAttrs);
  }
}

bool hasFullyInOneGroupValueSet(DenseSet<Value> &idValueSet) {
  for (auto val : idValueSet) {
    auto typeStr = "partition_dim_array" + getDimStrWithVal(val);
    if (auto attr = getDefiningOpByValue(val)->getAttr(typeStr)) {
      if (isFullyPartition(dyn_cast<mlir::ArrayAttr>(attr))) {
        return true;
      }
    }
  }
  return false;
}

void getFactorMapInOneGroupValueSet(DenseSet<Value> &idValueSet,
                                    DenseMap<int, int> &factorMap,
                                    DenseMap<int, bool> &cyclicMap) {
  for (auto val : idValueSet) {
    auto typeStr = "partition_dim_array" + getDimStrWithVal(val);
    if (auto attr = getDefiningOpByValue(val)->getAttr(typeStr)) {
      getFactorMapAndCyclicMapWithVal(val, factorMap, cyclicMap);
      if (isFullyPartition(dyn_cast<ArrayAttr>(attr))) {
        return;
      }
    }
  }
}

void handleOneGroupValueSet(DenseSet<Value> &idValueSet) {
  auto rankNum = (cast<MemRefType>((*idValueSet.begin()).getType())).getRank();
  bool allValueUserValid = true;
  for (auto val : idValueSet) {
    if (!checkValueUsers(val, rankNum)) {
      allValueUserValid = false;
      warningNonStandardAffineAccessByVal(val);
    }
  }
  if (!allValueUserValid) {
    // clear all pragma info
    removeAttrOneGroupValueSet(idValueSet);
    return;
  }
  bool fullyPartition = hasFullyInOneGroupValueSet(idValueSet);

  DenseMap<int, int> factorMap;
  DenseMap<int, bool> cyclicMap;
  getFactorMapInOneGroupValueSet(idValueSet, factorMap, cyclicMap);

  for (auto val : idValueSet) {
    if (!checkPartitionVector(val, factorMap, cyclicMap, fullyPartition)) {
      break;
    }
  }

  if (!factorMap.size()) {
    removeAttrOneGroupValueSet(idValueSet);
  } else if (fullyPartition) {
    setAttrOneGroupValueSetWithFully(idValueSet);
  } else {
    setAttrOneGroupValueSetWithFactorMap(idValueSet, factorMap, cyclicMap);
  }
}

void getReverseGraph(ModuleOp moduleOp,
                     DenseMap<StringRef, DenseSet<StringRef>> &reverseGraph) {
  moduleOp.walk([&](FuncOp funcOp) {
    funcOp->walk([&](func::CallOp callOp) {
      // a reverse call graph excluding recursion
      if (funcOp.getSymName() != callOp.getCallee()) {
        if (!reverseGraph.count(callOp.getCallee())) {
          reverseGraph[callOp.getCallee()] = DenseSet<StringRef>();
        }
        reverseGraph[callOp.getCallee()].insert(funcOp.getSymName());
      }
    });
  });
}

// traverse callee
void traverseCalleeArg(Value arg, DenseSet<Value> &idValueSet,
                       DenseMap<StringRef, func::FuncOp> &symNameFuncOpMap) {
  for (auto &use : llvm::make_early_inc_range(arg.getUses())) {
    if (auto callOp = dyn_cast<func::CallOp>(use.getOwner())) {
      auto funcArg = symNameFuncOpMap[callOp.getCallee()].getArgument(
          use.getOperandNumber());
      if (!idValueSet.count(funcArg)) {
        idValueSet.insert(funcArg);
        traverseCalleeArg(funcArg, idValueSet, symNameFuncOpMap);
      }
    }
  }
}

// traverse caller
void traverseCallerArg(Value arg, DenseSet<Value> &idValueSet,
                       DenseMap<StringRef, func::FuncOp> &symNameFuncOpMap,
                       DenseMap<StringRef, DenseSet<StringRef>> reverseGraph) {
  if (auto blockArg = dyn_cast<mlir::BlockArgument>(arg)) {
    auto calleeFuncOp = cast<FuncOp>(blockArg.getOwner()->getParentOp());
    // traverse reverse call
    for (auto callerName : reverseGraph[calleeFuncOp.getSymName()]) {
      auto funcOp = symNameFuncOpMap[callerName];
      funcOp->walk([&](func::CallOp callOp) {
        if (calleeFuncOp.getSymName() == callOp.getCallee()) {
          auto callArg = callOp.getOperand(blockArg.getArgNumber());
          if (!idValueSet.count(callArg)) {
            idValueSet.insert(callArg);
            traverseCallerArg(callArg, idValueSet, symNameFuncOpMap,
                              reverseGraph);
            // need traverse callee arg
            traverseCalleeArg(callArg, idValueSet, symNameFuncOpMap);
          }
        }
      });
    }
  }
}

bool isInCurrentSet(Value arg, DenseMap<int, DenseSet<Value>> &idValueMap) {
  for (auto idValueSet : idValueMap) {
    if (idValueSet.second.count(arg)) {
      return true;
    }
  }
  return false;
}

void getIdValueMap(ModuleOp moduleOp,
                   DenseMap<int, DenseSet<Value>> &idValueMap) {
  DenseMap<StringRef, func::FuncOp> symNameFuncOpMap;
  moduleOp.walk(
      [&](FuncOp funcOp) { symNameFuncOpMap[funcOp.getSymName()] = funcOp; });
  // range is [0, groupCount)
  int groupCount = 0;
  moduleOp.walk([&](memref::AllocaOp AI) {
    if (AI->hasAttr("partition_dim_array")) {
      auto arg = AI->getResult(0);
      idValueMap[groupCount] = DenseSet<Value>();
      idValueMap[groupCount].insert(arg);
      traverseCalleeArg(arg, idValueMap[groupCount], symNameFuncOpMap);
      groupCount += 1;
    }
  });
  DenseMap<StringRef, DenseSet<StringRef>> reverseGraph;
  getReverseGraph(moduleOp, reverseGraph);
  moduleOp.walk([&](FuncOp op) {
    for (int i = op.getArguments().size() - 1; i >= 0; i--) {
      auto arg = op.getArgument(i);
      if (isa<mlir::MemRefType>(arg.getType()) &&
          op->hasAttr(stringAddNumber("partition_cyclic_array_", i)) &&
          !isInCurrentSet(arg, idValueMap)) {
        idValueMap[groupCount] = DenseSet<Value>();
        idValueMap[groupCount].insert(arg);
        traverseCalleeArg(arg, idValueMap[groupCount], symNameFuncOpMap);
        traverseCallerArg(arg, idValueMap[groupCount], symNameFuncOpMap,
                          reverseGraph);
        groupCount += 1;
      }
    }
  });
}

void colorAndCheckCallGraph(ModuleOp moduleOp) {
  DenseMap<int, DenseSet<Value>> idValueMap;
  getIdValueMap(moduleOp, idValueMap);
  for (int i = 0, e = idValueMap.size(); i < e; i++) {
    handleOneGroupValueSet(idValueMap[i]);
  }
}

struct NewArrayPartitionPass : NewArrayPartitionBase<NewArrayPartitionPass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();
    colorAndCheckCallGraph(moduleOp);
    GreedyRewriteConfig config;
    config.setStrictness(GreedyRewriteStrictness::ExistingOps);
    DenseMap<StringRef, SmallVector<memref::GetGlobalOp>> newGetGlobalOpMap;
    moduleOp.walk([&](memref::GetGlobalOp getGlobalOp) {
      if (!newGetGlobalOpMap.count(getGlobalOp.getName())) {
        newGetGlobalOpMap[getGlobalOp.getName()] =
            SmallVector<memref::GetGlobalOp>();
      }
      newGetGlobalOpMap[getGlobalOp.getName()].push_back(getGlobalOp);
    });
    SmallVector<Operation *> accessOpsToErase;
    moduleOp.walk([&](memref::GlobalOp globalOp) {
      RewritePatternSet patterns(&getContext());
      patterns.insert<GlobalOpPattern>(&getContext(), newGetGlobalOpMap,
                                       accessOpsToErase);
      (void)applyOpPatternsAndFold(globalOp.getOperation(), std::move(patterns),
                                   config);
    });
    for (auto op : accessOpsToErase) {
      op->erase();
    }
    moduleOp.walk([&](FuncOp func) {
      RewritePatternSet patterns(&getContext());
      patterns.insert<FuncOpPattern>(&getContext());
      (void)applyOpPatternsAndFold(func.getOperation(), std::move(patterns),
                                   config);
      func->removeAttr("array-partition");
    });
    SmallVector<Operation *> allocaOps;
    moduleOp.walk([&](memref::AllocaOp AI) { allocaOps.push_back(AI); });
    for (auto op : allocaOps) {
      RewritePatternSet patterns(&getContext());
      patterns.insert<AllocaOpPattern>(&getContext());
      (void)applyOpPatternsAndFold(op, std::move(patterns), config);
    }
  }
};
} // namespace

namespace mlir {
std::unique_ptr<OperationPass<mlir::ModuleOp>> createNewArrayPartitionPass() {
  return std::make_unique<NewArrayPartitionPass>();
}

} // namespace mlir
