#pragma once
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"

namespace megg {

// Promote a single SCF for loop to Affine for loop
mlir::affine::AffineForOp promoteSCFToAffine(mlir::scf::ForOp forOp, mlir::Operation *&targetFor);

// Lower a single Affine for loop to SCF for loop
bool lowerAffineToSCF(mlir::affine::AffineForOp forOp);

// Lower all Affine operations (for, apply, min, max, etc.) to Standard/SCF
// This applies conversion patterns to the entire module
bool lowerAllAffineToSCF(mlir::ModuleOp module);

} // namespace megg
