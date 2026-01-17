#include "loop_raise.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Casting.h"
#include <cassert>
#include <iostream>

namespace megg {
namespace {

static bool getConstantIndexValue(mlir::Value value, int64_t &result) {
  auto constant = value.getDefiningOp<mlir::arith::ConstantIndexOp>();
  if (constant) {
    result = constant.value();
    return true;
  }
  return false;
}

static bool getConstantStepValue(mlir::scf::ForOp forOp, int64_t &result) {
  if (auto constantAttr = forOp.getConstantStep()) {
    result = constantAttr->getSExtValue();
    return true;
  }
  if (auto constant =
          forOp.getStep().getDefiningOp<mlir::arith::ConstantIndexOp>()) {
    result = constant.value();
    return true;
  }
  return false;
}

static bool isPromotableSCFLoop(mlir::scf::ForOp forOp) {
  if (!forOp || !forOp.getOperation()->getParentRegion())
    return false;

  // CRITICAL: Do not promote loops with iter_args
  // affine.for does not support iter_args, and trying to create
  // an affine.for with iter_args causes the official lower-affine pass to fail
  if (forOp.getInitArgs().size() > 0) {
    std::cerr << "isPromotableSCFLoop: Skipping loop with "
              << forOp.getInitArgs().size() << " iter_args "
              << "(affine.for does not support iter_args)" << std::endl;
    return false;
  }

  int64_t lowerBound = 0;
  int64_t upperBound = 0;
  if (!getConstantIndexValue(forOp.getLowerBound(), lowerBound))
    return false;

  if (!getConstantIndexValue(forOp.getUpperBound(), upperBound))
    return false;

  if (upperBound <= lowerBound)
    return false;

  int64_t step = 0;
  if (!getConstantStepValue(forOp, step))
    return false;

  if (step <= 0)
    return false;

  return true;
}

static mlir::affine::AffineForOp
convertLoopToAffine(mlir::scf::ForOp forOp, mlir::Operation *&targetFor) {
  int64_t lowerBound = 0;
  int64_t upperBound = 0;
  bool flash = false;
  if (forOp.getOperation() == targetFor)
    flash = true;
  if (!getConstantIndexValue(forOp.getLowerBound(), lowerBound))
    return {};
  if (!getConstantIndexValue(forOp.getUpperBound(), upperBound))
    return {};

  int64_t step = 0;
  if (!getConstantStepValue(forOp, step) || step <= 0)
    return {};

  mlir::OpBuilder builder(forOp.getOperation());

  // Collect iter_args for affine.for creation
  llvm::SmallVector<mlir::Value, 4> iterArgs(forOp.getInitArgs().begin(),
                                              forOp.getInitArgs().end());

  // Create affine.for with iter_args support
  auto affineFor = builder.create<mlir::affine::AffineForOp>(
      forOp.getLoc(), lowerBound, upperBound, step, iterArgs);

  mlir::Region &affineRegion = affineFor.getRegion();
  mlir::Block &targetBlock = affineRegion.front();
  mlir::Block &sourceBlock = forOp.getRegion().front();

  // Remove terminator only if the block might have one
  // (blocks with iter_args may not have a terminator initially)
  if (!targetBlock.empty() &&
      targetBlock.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
    targetBlock.back().erase();
  }

  // Map induction variable
  if (sourceBlock.getNumArguments() > 0) {
    mlir::Value oldInduction = sourceBlock.getArgument(0);
    mlir::Value newInduction = targetBlock.getArgument(0);
    if (oldInduction != newInduction) {
      oldInduction.replaceAllUsesWith(newInduction);
    }
  }

  // Map iter_args (block arguments beyond the induction variable)
  for (unsigned i = 1; i < sourceBlock.getNumArguments(); ++i) {
    mlir::Value oldIterArg = sourceBlock.getArgument(i);
    mlir::Value newIterArg = targetBlock.getArgument(i);
    if (oldIterArg != newIterArg) {
      oldIterArg.replaceAllUsesWith(newIterArg);
    }
  }

  auto *sourceTerminator = sourceBlock.getTerminator();
  auto sourceEnd =
      sourceTerminator ? sourceTerminator->getIterator() : sourceBlock.end();

  targetBlock.getOperations().splice(targetBlock.end(),
                                     sourceBlock.getOperations(),
                                     sourceBlock.begin(), sourceEnd);

  // Handle yield operation - preserve operands for iter_args
  if (sourceTerminator) {
    if (auto yield = llvm::dyn_cast<mlir::scf::YieldOp>(sourceTerminator)) {
      llvm::SmallVector<mlir::Value, 4> yieldOperands;
      for (auto operand : yield.getOperands()) {
        yieldOperands.push_back(operand);
      }
      yield.erase();

      auto yieldBuilder = mlir::OpBuilder::atBlockEnd(&targetBlock);
      yieldBuilder.create<mlir::affine::AffineYieldOp>(forOp.getLoc(),
                                                       yieldOperands);
    }
  } else {
    auto yieldBuilder = mlir::OpBuilder::atBlockEnd(&targetBlock);
    yieldBuilder.create<mlir::affine::AffineYieldOp>(forOp.getLoc());
  }

  forOp.replaceAllUsesWith(affineFor.getResults());
  forOp.erase();
  if (flash)
    targetFor = affineFor.getOperation();
  return affineFor;
}

static bool convertLoopToSCF(mlir::affine::AffineForOp forOp) {
  mlir::OpBuilder builder(forOp.getOperation());
  auto loc = forOp.getLoc();

  // Use MLIR's official lower bound/upper bound lowering
  // This handles complex affine maps, not just constants
  mlir::Value lowerBound = mlir::lowerAffineLowerBound(forOp, builder);
  mlir::Value upperBound = mlir::lowerAffineUpperBound(forOp, builder);

  if (!lowerBound || !upperBound) {
    return false;
  }

  auto step =
      builder.create<mlir::arith::ConstantIndexOp>(loc, forOp.getStepAsInt());

  // Collect iter_args for scf.for creation
  llvm::SmallVector<mlir::Value, 4> iterArgs(forOp.getInits().begin(),
                                              forOp.getInits().end());

  // Create scf.for with iter_args support
  auto scfFor = builder.create<mlir::scf::ForOp>(loc, lowerBound, upperBound,
                                                 step, iterArgs);

  // Inline the region from affine.for into scf.for
  mlir::Region &scfRegion = scfFor.getRegion();
  mlir::Region &affineRegion = forOp.getRegion();

  // Erase the default scf.for body
  scfRegion.front().erase();

  // Move the affine.for body into scf.for
  scfRegion.getBlocks().splice(scfRegion.end(), affineRegion.getBlocks());

  // Replace all uses of affine.for results with scf.for results
  forOp.replaceAllUsesWith(scfFor.getResults());

  forOp.erase();
  return true;
}

} // namespace

mlir::affine::AffineForOp promoteSCFToAffine(mlir::scf::ForOp forOp,
                                             mlir::Operation *&targetFor) {
  if (!isPromotableSCFLoop(forOp))
    return {};
  return convertLoopToAffine(forOp, targetFor);
}

bool lowerAffineToSCF(mlir::affine::AffineForOp forOp) {
  if (!forOp)
    return false;

  // First convert the affine.for itself
  if (!convertLoopToSCF(forOp))
    return false;

  return true;
}

bool lowerAllAffineToSCF(mlir::ModuleOp module) {
  // Apply official affine-to-standard conversion using pass manager
  mlir::MLIRContext *context = module.getContext();

  // IMPORTANT: Enable all dialects that might be needed
  context->loadDialect<mlir::arith::ArithDialect>();
  context->loadDialect<mlir::scf::SCFDialect>();
  context->loadDialect<mlir::memref::MemRefDialect>();
  context->loadDialect<mlir::func::FuncDialect>();
  context->loadDialect<mlir::affine::AffineDialect>();

  mlir::PassManager pm(context);

  // Disable multi-threading for debugging
  context->disableMultithreading();

  // Use the standard affine-to-scf lowering pass
  pm.addPass(mlir::createLowerAffinePass());

  std::cerr << "lowerAllAffineToSCF: Running pass manager..." << std::endl;

  // Run the pass manager
  if (mlir::failed(pm.run(module))) {
    std::cerr << "lowerAllAffineToSCF: Pass manager failed" << std::endl;
    std::cerr << "Module after failed conversion:" << std::endl;
    module.print(llvm::errs());
    return false;
  }

  std::cerr << "lowerAllAffineToSCF: Pass manager succeeded" << std::endl;

  // Check if affine ops still exist
  bool hasAffineOps = false;
  module.walk([&](mlir::Operation *op) {
    if (llvm::isa<mlir::affine::AffineDialect>(op->getDialect())) {
      hasAffineOps = true;
      std::cerr << "WARNING: Found remaining affine op: " << op->getName().getStringRef().str() << std::endl;
    }
  });

  if (hasAffineOps) {
    std::cerr << "WARNING: Module still contains affine operations after lowering!" << std::endl;
  }

  return true;
}

} // namespace megg
