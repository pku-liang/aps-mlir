#pragma once
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir_utils.h"
#include <string>

namespace megg {
class CustomPass;
class LoopUnrollJammer : public CustomPass {
public:
  LoopUnrollJammer(int factor = 4, mlir::Operation *targetFunction = nullptr,
                   mlir::Operation *targetLoop = nullptr)
      : CustomPass("LoopUnrollJam"), unrollJamFactor(factor),
        maxTripCount(1000), minTripCount(2),
        targetFunction_(targetFunction),
        targetLoopOp_(targetLoop) {}

  // Override run method from CustomPass
  bool run(mlir::ModuleOp module) override;

  // Configuration methods
  void setUnrollJamFactor(int factor);
  void setMaxTripCount(int64_t max);
  void setMinTripCount(int64_t min);
  void setTargetFunction(mlir::Operation *funcOp) {
    targetFunction_ = funcOp;
  }
  void setTargetLoop(mlir::Operation *loopOp) {
    targetLoopOp_ = loopOp;
  }

  int getUnrollJamFactor() const { return unrollJamFactor; }
  int64_t getMaxTripCount() const { return maxTripCount; }
  int64_t getMinTripCount() const { return minTripCount; }
  mlir::Operation *getTargetFunction() const { return targetFunction_; }
  mlir::Operation *getTargetLoop() const { return targetLoopOp_; }

private:
  bool shouldUnrollJam(mlir::affine::AffineForOp forOp);
  bool shouldProcessFunction(mlir::func::FuncOp funcOp) const;
  bool applyLoopUnrollJam(mlir::affine::AffineForOp forOp, int unrollJamFactor);

  int unrollJamFactor;
  int64_t maxTripCount;  // Don't unroll-jam loops larger than this
  int64_t minTripCount;  // Don't unroll-jam loops smaller than this
  mlir::Operation *targetFunction_;
  mlir::Operation *targetLoopOp_;  // Target loop operation
};
} // namespace megg
