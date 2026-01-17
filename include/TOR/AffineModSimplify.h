#ifndef TOR_AFFINE_MOD_SIMPLIFY_H
#define TOR_AFFINE_MOD_SIMPLIFY_H

#include "mlir/IR/AffineExpr.h"

namespace mlir {
namespace tor {

/// Simplify (expr % factor) by eliminating terms with coefficients divisible by factor.
/// Example: (2x + 8y + 64) % 8 -> 2x (since 8y % 8 = 0 and 64 % 8 = 0)
/// Returns the simplified expression, or nullptr if expression is not linear.
AffineExpr simplifyMod(AffineExpr expr, int64_t factor, unsigned numDims, unsigned numSyms);

} // namespace tor
} // namespace mlir

#endif // TOR_AFFINE_MOD_SIMPLIFY_H
