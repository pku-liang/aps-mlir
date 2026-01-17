#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/Support/Casting.h"
#include <iomanip>

using namespace mlir;

static inline mlir::Operation *getDefiningOpByValue(mlir::Value val) {
  if (auto blockArg = dyn_cast<mlir::BlockArgument>(val)) {
    return blockArg.getOwner()->getParentOp();
  }
  return val.getDefiningOp();
}

static inline mlir::Operation *getAncestorOp(mlir::Operation *op) {
  mlir::Operation *iterationOp = op;
  while (iterationOp && !mlir::isa<mlir::ModuleOp>(iterationOp))
    iterationOp = iterationOp->getParentOp();
  return iterationOp;
}

static inline unsigned getLineAttrInterger(mlir::Operation *op,
                                           std::string type) {
  if (!op->hasAttr(type + "-line")) {
    auto line = llvm::dyn_cast<mlir::FileLineColLoc>(op->getLoc()).getLine();
    return line;
  }
  return 
      dyn_cast<mlir::IntegerAttr>(op->getAttr(type + "-line"))
      .getValue()
      .getSExtValue();
}

static inline void setPragmaStructureAttrInvalid(
    mlir::SmallVector<mlir::Attribute, 4> &pragmaStructureAttr,
    mlir::MLIRContext *ctx) {
  pragmaStructureAttr[1] =
      mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), 2);
}

static inline void setPragmaStructureAttrValid(
    mlir::SmallVector<mlir::Attribute, 4> &pragmaStructureAttr,
    mlir::MLIRContext *ctx) {
  pragmaStructureAttr[1] =
      mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), 1);
}

static inline void setPragmaStructureAttrNewValue(
    mlir::SmallVector<mlir::Attribute, 4> &pragmaStructureAttr,
    mlir::MLIRContext *ctx, int newValue) {
  pragmaStructureAttr[3] =
      mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), newValue);
}

static inline void
setPragmaStructureAttrStatusByModuleOp(mlir::Operation *moduleOp,
                                       int lineNumber, bool isValid = true) {
  auto ctx = moduleOp->getContext();
  if (auto lineAttr = moduleOp->getAttr("hls.pragma_line_" +
                                        llvm::Twine(lineNumber).str())) {
    auto pragmaStructureAttr =
        llvm::to_vector<4>(dyn_cast<mlir::ArrayAttr>(lineAttr));
    if (isValid) {
      setPragmaStructureAttrValid(pragmaStructureAttr, ctx);
    } else {
      setPragmaStructureAttrInvalid(pragmaStructureAttr, ctx);
    }
    moduleOp->setAttr("hls.pragma_line_" + llvm::Twine(lineNumber).str(),
                      mlir::ArrayAttr::get(ctx, pragmaStructureAttr));
  }
}

static inline void setPragmaStructureAttrStatusByOp(mlir::Operation *op,
                                                    std::string type,
                                                    bool isValid = true) {
  auto *moduleOp = getAncestorOp(op);
  if (mlir::isa<mlir::ModuleOp>(moduleOp)) {
    int lineNumber = getLineAttrInterger(op, type);
    setPragmaStructureAttrStatusByModuleOp(moduleOp, lineNumber, isValid);
  }
}

static inline void setPragmaStructureAttrNewValueByOp(mlir::Operation *op,
                                                      std::string type,
                                                      int newValue) {
  setPragmaStructureAttrStatusByOp(op, type);
  auto *moduleOp = getAncestorOp(op);
  if (mlir::isa<mlir::ModuleOp>(moduleOp)) {
    auto ctx = moduleOp->getContext();
    int lineNumber = getLineAttrInterger(op, type);
    if (auto lineAttr = moduleOp->getAttr("hls.pragma_line_" +
                                          llvm::Twine(lineNumber).str())) {
      auto pragmaStructureAttr =
          llvm::to_vector<4>(dyn_cast<mlir::ArrayAttr>(lineAttr));
      setPragmaStructureAttrNewValue(pragmaStructureAttr, ctx, newValue);
      moduleOp->setAttr("hls.pragma_line_" + llvm::Twine(lineNumber).str(),
                        mlir::ArrayAttr::get(ctx, pragmaStructureAttr));
    }
  }
}

static inline void setPragmaStructureAttrStatusByValue(mlir::Value val,
                                                       std::string type,
                                                       bool isValid = true) {
  auto *op = getDefiningOpByValue(val);
  setPragmaStructureAttrStatusByOp(op, type, isValid);
}

template <typename T>
static inline std::string getFormatWidthStr(int width, char fill, T Value) {
  std::ostringstream ss;
  ss << std::setw(width) << std::setfill(fill) << Value;
  return ss.str();
}

template <typename T>
static inline std::string getFormatWidthLeftStr(int width, char fill, T Value) {
  std::ostringstream ss;
  ss << std::left << std::setw(width) << std::setfill(fill) << Value;
  return ss.str();
}

static inline std::string getStatusString(int status) {
  if (status == 1) {
    return "Valid";
  }
  if (status == 0) {
    return "Init";
  }
  if (status == 2) {
    return "Invalid";
  }
  return "";
}

static inline void printPragmaReport(mlir::ModuleOp moduleOp) {
  if (!moduleOp->hasAttr("hls.pragma_line_list")) {
    return;
  }
  llvm::outs() << getFormatWidthStr(110, '=', "") << "\n";
  llvm::outs() << "== " << "Pragma Report\n";
  llvm::outs() << getFormatWidthStr(110, '=', "") << "\n";
  llvm::outs() << "+" << getFormatWidthStr(108, '-', "") << "+\n";
  llvm::outs() << "|" << getFormatWidthStr(22, ' ', "Type") << "     |"
               << getFormatWidthStr(46, ' ', "Options") << "     |"
               << getFormatWidthStr(9, ' ', "Status") << "  |"
               << getFormatWidthStr(12, ' ', "Line number") << "    |\n";

  mlir::SmallVector<mlir::Attribute> pragmaLineListAttr = llvm::to_vector<4>(
      dyn_cast<mlir::ArrayAttr>(moduleOp->getAttr("hls.pragma_line_list")));
  auto cmp = [&](mlir::Attribute a, mlir::Attribute b) {
    return dyn_cast<mlir::IntegerAttr>(a).getValue().getSExtValue() <
           dyn_cast<mlir::IntegerAttr>(b).getValue().getSExtValue();
  };
  std::stable_sort(pragmaLineListAttr.begin(), pragmaLineListAttr.end(), cmp);

  for (auto lineAttr : pragmaLineListAttr) {
    int lineNumber =
        dyn_cast<mlir::IntegerAttr>(lineAttr).getValue().getSExtValue();
    auto pragmaStructureAttr =
        dyn_cast<mlir::ArrayAttr>(moduleOp->getAttr("hls.pragma_line_" + llvm::Twine(lineNumber).str()));
    std::string type =
        dyn_cast<mlir::StringAttr>(pragmaStructureAttr[0]).getValue().str();
    std::string option =
        dyn_cast<mlir::StringAttr>(pragmaStructureAttr[4]).getValue().str();
    int oldValue = dyn_cast<mlir::IntegerAttr>(pragmaStructureAttr[2])
                       .getValue()
                       .getSExtValue();
    int newValue = dyn_cast<mlir::IntegerAttr>(pragmaStructureAttr[3])
                       .getValue()
                       .getSExtValue();
    auto statusStr = getStatusString(dyn_cast<mlir::IntegerAttr>(pragmaStructureAttr[1])
                                         .getValue()
                                         .getSExtValue());
    if ((type == "pipeline" || type == "unroll") && oldValue != newValue &&
        statusStr == "Valid") {
      llvm::outs() << "|" << getFormatWidthStr(22, ' ', type) << "     |"
                   << getFormatWidthStr(23, ' ', option)
                   << getFormatWidthLeftStr(23, ' ',
                                            ", effective value = " +
                                                llvm::Twine(newValue).str())
                   << "     |" << getFormatWidthStr(9, ' ', statusStr) << "  |"
                   << getFormatWidthStr(12, ' ', lineNumber) << "    |\n";
    } else {
      llvm::outs() << "|" << getFormatWidthStr(22, ' ', type) << "     |"
                   << getFormatWidthStr(46, ' ', option) << "     |"
                   << getFormatWidthStr(9, ' ', statusStr) << "  |"
                   << getFormatWidthStr(12, ' ', lineNumber) << "    |\n";
    }
  }
  llvm::outs() << "+" << getFormatWidthStr(108, '-', "") << "+\n\n";
}


// static inline mlir::scf::IfOp cloneWithResults(mlir::scf::IfOp op,
//                                                mlir::OpBuilder &rewriter,
//                                                mlir::IRMapping mapping = {}) {
//   using namespace mlir;
//   return rewriter.create<scf::IfOp>(op.getLoc(), op.getResultTypes(),
//                                     mapping.lookupOrDefault(op.getCondition()),
//                                     true);
// }
// static inline mlir::affine::AffineIfOp
// cloneWithResults(mlir::affine::AffineIfOp op, mlir::OpBuilder &rewriter,
//                  mlir::IRMapping mapping = {}) {
//   using namespace mlir;
//   SmallVector<mlir::Value> lower;
//   for (auto o : op.getOperands())
//     lower.push_back(mapping.lookupOrDefault(o));
//   return rewriter.create<affine::AffineIfOp>(op.getLoc(), op.getResultTypes(),
//                                              op.getIntegerSet(), lower, true);
// }

// static inline mlir::scf::IfOp cloneWithoutResults(mlir::scf::IfOp op,
//                                                   mlir::OpBuilder &rewriter,
//                                                   mlir::IRMapping mapping = {},
//                                                   mlir::TypeRange types = {}) {
//   using namespace mlir;
//   return rewriter.create<scf::IfOp>(
//       op.getLoc(), types, mapping.lookupOrDefault(op.getCondition()), true);
// }

// static inline mlir::affine::AffineIfOp
// cloneWithoutResults(mlir::affine::AffineIfOp op, mlir::OpBuilder &rewriter,
//                     mlir::IRMapping mapping = {}, mlir::TypeRange types = {}) {
//   using namespace mlir;
//   SmallVector<mlir::Value> lower;
//   for (auto o : op.getOperands())
//     lower.push_back(mapping.lookupOrDefault(o));
//   return rewriter.create<affine::AffineIfOp>(op.getLoc(), types,
//                                              op.getIntegerSet(), lower, true);
// }

// static inline mlir::scf::ForOp
// cloneWithoutResults(mlir::scf::ForOp op, mlir::PatternRewriter &rewriter,
//                     mlir::IRMapping mapping = {}) {
//   using namespace mlir;
//   return rewriter.create<scf::ForOp>(
//       op.getLoc(), mapping.lookupOrDefault(op.getLowerBound()),
//       mapping.lookupOrDefault(op.getUpperBound()),
//       mapping.lookupOrDefault(op.getStep()));
// }
// static inline mlir::affine::AffineForOp
// cloneWithoutResults(mlir::affine::AffineForOp op,
//                     mlir::PatternRewriter &rewriter,
//                     mlir::IRMapping mapping = {}) {
//   using namespace mlir;
//   SmallVector<Value> lower;
//   for (auto o : op.getLowerBoundOperands())
//     lower.push_back(mapping.lookupOrDefault(o));
//   SmallVector<Value> upper;
//   for (auto o : op.getUpperBoundOperands())
//     upper.push_back(mapping.lookupOrDefault(o));
//   return rewriter.create<affine::AffineForOp>(
//       op.getLoc(), lower, op.getLowerBoundMap(), upper, op.getUpperBoundMap(),
//       op.getStep());
// }

static inline void clearBlock(mlir::Block *block,
                              mlir::PatternRewriter &rewriter) {
  for (auto &op : llvm::make_early_inc_range(llvm::reverse(*block))) {
    assert(op.use_empty() && "expected 'op' to have no uses");
    rewriter.eraseOp(&op);
  }
}

static inline mlir::Block *getThenBlock(mlir::scf::IfOp op) {
  return op.thenBlock();
}
static inline mlir::Block *getThenBlock(mlir::affine::AffineIfOp op) {
  return op.getThenBlock();
}
static inline mlir::Block *getElseBlock(mlir::scf::IfOp op) {
  return op.elseBlock();
}
static inline mlir::Block *getElseBlock(mlir::affine::AffineIfOp op) {
  if (op.hasElse())
    return op.getElseBlock();
  else
    return nullptr;
}

static inline mlir::Region &getThenRegion(mlir::scf::IfOp op) {
  return op.getThenRegion();
}
static inline mlir::Region &getThenRegion(mlir::affine::AffineIfOp op) {
  return op.getThenRegion();
}
static inline mlir::Region &getElseRegion(mlir::scf::IfOp op) {
  return op.getElseRegion();
}
static inline mlir::Region &getElseRegion(mlir::affine::AffineIfOp op) {
  return op.getElseRegion();
}

static inline mlir::scf::YieldOp getThenYield(mlir::scf::IfOp op) {
  return op.thenYield();
}
static inline mlir::affine::AffineYieldOp
getThenYield(mlir::affine::AffineIfOp op) {
  return llvm::cast<mlir::affine::AffineYieldOp>(
      op.getThenBlock()->getTerminator());
}
static inline mlir::scf::YieldOp getElseYield(mlir::scf::IfOp op) {
  return op.elseYield();
}
static inline mlir::affine::AffineYieldOp
getElseYield(mlir::affine::AffineIfOp op) {
  return llvm::cast<mlir::affine::AffineYieldOp>(
      op.getElseBlock()->getTerminator());
}

static inline bool inBound(mlir::scf::IfOp op, mlir::Value v) {
  return op.getCondition() == v;
}
static inline bool inBound(mlir::affine::AffineIfOp op, mlir::Value v) {
  return llvm::any_of(op.getOperands(), [&](mlir::Value e) { return e == v; });
}
static inline bool inBound(mlir::scf::ForOp op, mlir::Value v) {
  return op.getUpperBound() == v;
}
static inline bool inBound(mlir::affine::AffineForOp op, mlir::Value v) {
  return llvm::any_of(op.getUpperBoundOperands(),
                      [&](mlir::Value e) { return e == v; });
}
static inline bool hasElse(mlir::scf::IfOp op) {
  return op.getElseRegion().getBlocks().size() > 0;
}
static inline bool hasElse(mlir::affine::AffineIfOp op) {
  return op.getElseRegion().getBlocks().size() > 0;
}

static inline bool hasHlsAttrWithNewOp(mlir::Operation *op) {
  return op->hasAttr("II") || op->hasAttr("unroll") ||
         op->hasAttr("dataflow") || op->hasAttr("flatten") ||
         op->hasAttr("merge");
}

static inline void saveAttrWithName(mlir::Operation *newOp, mlir::Operation *op,
                                    std::string type) {
  if (auto pipelineAttr = op->getAttr(type)) {
    newOp->setAttr(type, pipelineAttr);
  }
}

static inline void saveHlsAttrWithType(mlir::Operation *newOp,
                                       mlir::Operation *op, std::string type) {
  saveAttrWithName(newOp, op, type);
  saveAttrWithName(newOp, op, type + "-line");
}

static inline void addHlsAttrWithNewOp(mlir::Operation *newOp,
                                       mlir::Operation *op) {
  saveAttrWithName(newOp, op, "II");
  saveHlsAttrWithType(newOp, op, "pipeline");

  saveHlsAttrWithType(newOp, op, "unroll");

  saveHlsAttrWithType(newOp, op, "dataflow");

  saveAttrWithName(newOp, op, "tripcount-min");
  saveAttrWithName(newOp, op, "tripcount-avg");
  saveAttrWithName(newOp, op, "tripcount-max");
  saveAttrWithName(newOp, op, "tripcount-line");

  saveHlsAttrWithType(newOp, op, "flatten");

  saveHlsAttrWithType(newOp, op, "expr-balance");

  saveHlsAttrWithType(newOp, op, "merge");
}

static inline void addHlsPipelineAttrWithNewOp(mlir::Operation *newOp,
                                               mlir::Operation *op) {
  if (auto IIAttr = op->getAttr("II")) {
    newOp->setAttr("pipeline",
                   mlir::IntegerAttr::get(
                       mlir::IntegerType::get(op->getContext(), 32), 1));
  }
}