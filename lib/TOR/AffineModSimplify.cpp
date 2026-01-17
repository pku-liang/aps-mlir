#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

/// Extract linear coefficients: expr = sum(dimCoeffs[i] * d_i) + sum(symCoeffs[i] * s_i) + constant
/// Returns false if expression is not purely linear (contains floordiv/ceildiv/mod)
bool extractCoeffs(AffineExpr expr, unsigned numDims, unsigned numSyms,
                   SmallVector<int64_t> &dimCoeffs, SmallVector<int64_t> &symCoeffs,
                   int64_t &constant) {
  dimCoeffs.assign(numDims, 0);
  symCoeffs.assign(numSyms, 0);
  constant = 0;

  std::function<bool(AffineExpr, int64_t)> extract = [&](AffineExpr e, int64_t mul) -> bool {
    if (auto c = dyn_cast<AffineConstantExpr>(e)) {
      constant += c.getValue() * mul;
      return true;
    }
    if (auto d = dyn_cast<AffineDimExpr>(e)) {
      if (d.getPosition() < numDims) {
        dimCoeffs[d.getPosition()] += mul;
        return true;
      }
      return false;
    }
    if (auto s = dyn_cast<AffineSymbolExpr>(e)) {
      if (s.getPosition() < numSyms) {
        symCoeffs[s.getPosition()] += mul;
        return true;
      }
      return false;
    }
    if (auto bin = dyn_cast<AffineBinaryOpExpr>(e)) {
      if (bin.getKind() == AffineExprKind::Add) {
        return extract(bin.getLHS(), mul) && extract(bin.getRHS(), mul);
      }
      if (bin.getKind() == AffineExprKind::Mul) {
        if (auto c = dyn_cast<AffineConstantExpr>(bin.getRHS()))
          return extract(bin.getLHS(), mul * c.getValue());
        if (auto c = dyn_cast<AffineConstantExpr>(bin.getLHS()))
          return extract(bin.getRHS(), mul * c.getValue());
      }
    }
    return false;
  };

  return extract(expr, 1);
}

} // namespace

namespace mlir {
namespace tor {

/// Simplify (expr % factor) by eliminating terms with coefficients divisible by factor.
/// Example: (2x + 8y + 64) % 8 -> 2x % 8 (since 8y and 64 are divisible by 8)
/// Returns the simplified expression, or nullptr if expression is not linear.
AffineExpr simplifyMod(AffineExpr expr, int64_t factor, unsigned numDims, unsigned numSyms) {
  SmallVector<int64_t> dimCoeffs, symCoeffs;
  int64_t constant;

  if (!extractCoeffs(expr, numDims, numSyms, dimCoeffs, symCoeffs, constant)) {
    return nullptr; // Not linear
  }

  MLIRContext *ctx = expr.getContext();

  // Build simplified expression keeping only terms with coeff % factor != 0
  AffineExpr result = getAffineConstantExpr(constant % factor, ctx);

  for (unsigned i = 0; i < numDims; ++i) {
    int64_t coeff = dimCoeffs[i] % factor;
    if (coeff != 0) {
      result = result + getAffineDimExpr(i, ctx) * coeff;
    }
  }

  for (unsigned i = 0; i < numSyms; ++i) {
    int64_t coeff = symCoeffs[i] % factor;
    if (coeff != 0) {
      result = result + getAffineSymbolExpr(i, ctx) * coeff;
    }
  }

  return result;
}

} // namespace tor
} // namespace mlir
