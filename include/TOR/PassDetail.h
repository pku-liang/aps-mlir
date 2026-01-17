#ifndef TOR_PASS_DETAIL_H
#define TOR_PASS_DETAIL_H

#include "mlir/Pass/Pass.h"
#include "TOR/TOR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir
{
  template <typename ConcreteDialect>
  void registerDialect(DialectRegistry &registry);

  namespace polygeist
  {
    class PolygeistDialect;
  } // namespace polygeist

  namespace tor
  {
    class TORDialect;
  }
  namespace memref
  {
    class MemRefDialect;
    class ExtractStridedMetadataOp;
    class ReinterpretCastOp;
  } // end namespace memref
  namespace scf
  {
    class SCFDialect;
  } // end namespace scf
  namespace vector
  {
    class VectorDialect;
  } // namespace vector
  namespace affine
  {
    class AffineDialect;
  }
  namespace arith
  {
    class ArithDialect;
  }
  namespace LLVM
  {
    class LLVMDialect;
  }
  namespace tensor
  {
    class TensorDialect;
  }
  namespace math
  {
    class MathDialect;
  }
#define GEN_PASS_CLASSES
#include "TOR/Passes.h.inc"
}
#endif //TOR_PASS_DETAIL_H