#include "loop_unroll.h"
#include "loop_raise.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/StringSet.h"
#include <cassert>
#include <iostream>

namespace megg {

namespace {

static bool isLoopLikeOp(mlir::Operation *op) {
  return mlir::isa<mlir::affine::AffineForOp, mlir::scf::ForOp,
                   mlir::scf::WhileOp>(op);
}

} // namespace

bool LoopUnroller::run(mlir::ModuleOp module) {
  if (unrollFactor <= 0) {
    return false;
  }

  bool modified = false;
  mlir::affine::AffineForOp target = {};
  module.walk([&](mlir::func::FuncOp funcOp) -> mlir::WalkResult {
    if (!shouldProcessFunction(funcOp)) {
      return mlir::WalkResult::advance();
    }

    // First pass: promote all SCF loops to affine
    funcOp.walk([&](mlir::scf::ForOp scfFor) -> mlir::WalkResult {
      if (auto affine_for = promoteSCFToAffine(scfFor, targetLoop_)) {
        target = affine_for;
      }
      return mlir::WalkResult::advance();
    });

    // Second pass: unroll target affine loop
    if (target && shouldUnroll(target) && applyLoopUnroll(target, unrollFactor)) {
      modified = true;
      return mlir::WalkResult::interrupt();
    }

    return mlir::WalkResult::advance();
  });

  // Lower affine back to SCF
  if (!lowerAllAffineToSCF(module)) {
    std::cerr << "LoopUnroller: Failed to lower affine to SCF" << std::endl;
    return false;
  }

  return modified;
}

bool LoopUnroller::shouldUnroll(mlir::affine::AffineForOp forOp) {
  // Check if this is the target loop operation
  if (targetLoop_ != nullptr) {
    if (forOp.getOperation() != targetLoop_) {
      return false;
    }
  }

  // Get loop bounds
  if (!forOp.hasConstantBounds()) {
    return false; // Only unroll loops with constant bounds
  }

  auto lowerBound = forOp.getConstantLowerBound();
  auto upperBound = forOp.getConstantUpperBound();
  auto step = forOp.getStepAsInt(); // Get step as integer

  // Calculate trip count
  int64_t tripCount = (upperBound - lowerBound) / step;

  // Don't unroll very large loops
  if (tripCount > maxTripCount) {
    return false;
  }

  // Don't unroll loops that are already small
  if (tripCount <= minTripCount) {
    return false;
  }

  return true;
}

bool LoopUnroller::applyLoopUnroll(mlir::affine::AffineForOp forOp,
                                   int unrollFactor) {
  // Check for valid unroll factor
  if (unrollFactor <= 0) {
    return false;
  }

  // Apply the unroll transformation
  if (mlir::succeeded(
          mlir::affine::loopUnrollByFactor(forOp, unrollFactor))) {
    // normalize
    if (!mlir::succeeded(mlir::affine::normalizeAffineFor(forOp)))
      std::cerr << "Loop Normalize Error";
    return true;
  }

  return false;
}

bool LoopUnroller::shouldProcessFunction(mlir::func::FuncOp funcOp) const {
  if (targetFunction_ == nullptr) {
    return true;
  }

  return funcOp.getOperation() == targetFunction_;
}

void LoopUnroller::setUnrollFactor(int factor) {
  if (factor > 0) {
    unrollFactor = factor;
  }
}

void LoopUnroller::setMaxTripCount(int64_t max) {
  if (max > 0) {
    maxTripCount = max;
  }
}

void LoopUnroller::setMinTripCount(int64_t min) {
  if (min >= 0) {
    minTripCount = min;
  }
}

} // namespace megg
