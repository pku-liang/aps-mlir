#pragma once
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir_utils.h"
#include <string>

namespace megg {
class CustomPass;
class LoopUnroller : public CustomPass {
public:
  LoopUnroller(int factor = 4, mlir::Operation *targetFunction = nullptr,
               mlir::Operation *targetLoop = nullptr)
      : CustomPass("LoopUnroll"), unrollFactor(factor),
        maxTripCount(1000), minTripCount(2),
        targetFunction_(targetFunction),
        targetLoop_(targetLoop) {}

  // Override run method from CustomPass
  bool run(mlir::ModuleOp module) override;

  // Configuration methods
  void setUnrollFactor(int factor);
  void setMaxTripCount(int64_t max);
  void setMinTripCount(int64_t min);
  void setTargetFunction(mlir::Operation *funcOp) {
    targetFunction_ = funcOp;
  }
  void setTargetLoop(mlir::Operation *loopOp) {
    targetLoop_ = loopOp;
  }

  int getUnrollFactor() const { return unrollFactor; }
  int64_t getMaxTripCount() const { return maxTripCount; }
  int64_t getMinTripCount() const { return minTripCount; }
  mlir::Operation *getTargetFunction() const { return targetFunction_; }
  mlir::Operation *getTargetLoop() const { return targetLoop_; }

private:
  bool shouldUnroll(mlir::affine::AffineForOp forOp);
  bool shouldProcessFunction(mlir::func::FuncOp funcOp) const;
  bool applyLoopUnroll(mlir::affine::AffineForOp forOp, int unrollFactor);

  int unrollFactor;
  int64_t maxTripCount;  // Don't unroll loops larger than this
  int64_t minTripCount;  // Don't unroll loops smaller than this
  mlir::Operation *targetFunction_;
  mlir::Operation *targetLoop_;
};
} // namespace megg