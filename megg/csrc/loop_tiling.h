#pragma once
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir_utils.h"
#include <optional>
#include <string>
#include <vector>

namespace megg {

struct TileConfig {
  mlir::Operation *loopOp;
  uint64_t tileSize;

  TileConfig(mlir::Operation *op, uint64_t size)
      : loopOp(op), tileSize(size) {}
};

class LoopTiler : public CustomPass {
public:
  explicit LoopTiler(mlir::Operation *targetFunction, uint64_t defaultTileSize)
      : CustomPass("LoopTiling"), tileSize_(defaultTileSize),
        targetFunction_(targetFunction), tileConfigs_(nullptr, 0) {}

  bool run(mlir::ModuleOp module) override;


  /// @param loopOp: Target loop operation pointer
  /// @param tileSize: Size of each tile
  void addTileConfig(mlir::Operation *loopOp, uint64_t tileSize);

  /// Set the default tile size for loops without specific configuration
  void setDefaultTileSize(uint64_t size) {
    if (size > 0)
      tileSize_ = size;
  }

  void setTargetFunction(mlir::Operation *funcOp) {
    targetFunction_ = funcOp;
  }

  uint64_t getDefaultTileSize() const { return tileSize_; }
  mlir::Operation *getTargetFunction() const { return targetFunction_; }
  const TileConfig &getTileConfigs() const { return tileConfigs_; }

private:
  /// Check if this function should be processed
  bool shouldProcessFunction(mlir::func::FuncOp funcOp);
  /// Check if this loop should be tiled based on Operation pointer
  std::optional<uint64_t> getTileSizeForLoop(mlir::affine::AffineForOp forOp);

  /// Apply tiling transformation to a single loop
  bool applyLoopTiling(mlir::affine::AffineForOp forOp, uint64_t tileSize);

  /// Tile specific loops based on configuration
  bool tileConfiguredLoops(mlir::func::FuncOp funcOp);

  uint64_t tileSize_;
  mlir::Operation *targetFunction_;
  TileConfig tileConfigs_;
};

} // namespace megg
