#include "loop_unroll_jam.h"
#include "loop_raise.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"
#include <iostream>

namespace megg {

namespace {

} // namespace

bool LoopUnrollJammer::run(mlir::ModuleOp module) {
  if (unrollJamFactor <= 0) {
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
      if (auto affine_for = promoteSCFToAffine(scfFor,targetLoopOp_)) {
        target = affine_for;
      }
      return mlir::WalkResult::advance();
    });

    // Second pass: unroll-jam target affine loop
    if (target && shouldUnrollJam(target) &&
        applyLoopUnrollJam(target, unrollJamFactor)) {
      modified = true;
      return mlir::WalkResult::interrupt();
    }

    return mlir::WalkResult::advance();
  });
  lowerAllAffineToSCF(module);
  return modified;
}

bool LoopUnrollJammer::shouldUnrollJam(mlir::affine::AffineForOp forOp) {
  // Check if this is the target loop operation
  if (targetLoopOp_ != nullptr) {
    if (forOp.getOperation() != targetLoopOp_) {
      return false;
    }
  }

  // Get loop bounds
  if (!forOp.hasConstantBounds()) {
    return false; // Only unroll-jam loops with constant bounds
  }

  auto lowerBound = forOp.getConstantLowerBound();
  auto upperBound = forOp.getConstantUpperBound();
  auto step = forOp.getStepAsInt(); // Get step as integer

  // Calculate trip count
  int64_t tripCount = (upperBound - lowerBound) / step;

  // Don't unroll-jam very large loops
  if (tripCount > maxTripCount) {
    return false;
  }

  // Don't unroll-jam loops that are already small
  if (tripCount <= minTripCount) {
    return false;
  }

  return true;
}

bool LoopUnrollJammer::applyLoopUnrollJam(mlir::affine::AffineForOp forOp,
                                          int unrollJamFactor) {
  // Check for valid unroll-jam factor
  if (unrollJamFactor <= 0) {
    return false;
  }

  // Apply the unroll-and-jam transformation
  if (mlir::succeeded(
          mlir::affine::loopUnrollJamByFactor(forOp, unrollJamFactor))) {
    // normalize
    if (!mlir::succeeded(mlir::affine::normalizeAffineFor(forOp)))
      std::cerr << "Loop Normalize Error";

    return true;
  }

  return false;
}

bool LoopUnrollJammer::shouldProcessFunction(mlir::func::FuncOp funcOp) const {
  if (targetFunction_ == nullptr) {
    return true;
  }

  return funcOp.getOperation() == targetFunction_;
}

void LoopUnrollJammer::setUnrollJamFactor(int factor) {
  if (factor > 0) {
    unrollJamFactor = factor;
  }
}

void LoopUnrollJammer::setMaxTripCount(int64_t max) {
  if (max > 0) {
    maxTripCount = max;
  }
}

void LoopUnrollJammer::setMinTripCount(int64_t min) {
  if (min >= 0) {
    minTripCount = min;
  }
}

} // namespace megg
