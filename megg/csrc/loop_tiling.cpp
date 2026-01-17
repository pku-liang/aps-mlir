#include "loop_tiling.h"
#include "loop_raise.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Visitors.h"
#include <cassert>
#include <iostream>
#include <optional>
#include <vector>

namespace megg {

bool LoopTiler::run(mlir::ModuleOp module) {
  if (tileSize_ <= 0) {
    std::cerr << "LoopTiler: Invalid default tile size: " << tileSize_ << std::endl;
    return false;
  }

  bool modified = false;

  // Walk through all functions in the module
  module.walk([&](mlir::func::FuncOp funcOp) -> mlir::WalkResult {
    if (!shouldProcessFunction(funcOp)) {
      return mlir::WalkResult::advance();
    }

    std::string funcName = funcOp.getSymName().str();

    // First pass: promote all SCF loops to affine
    // Pass tileConfigs_.loopOp by reference so it gets updated during promotion
    funcOp.walk([&](mlir::scf::ForOp scfFor) -> mlir::WalkResult {
      promoteSCFToAffine(scfFor, tileConfigs_.loopOp);
      return mlir::WalkResult::advance();
    });

    // Second pass: tile configured affine loops
    if (tileConfigs_.loopOp != nullptr) {
      if (tileConfiguredLoops(funcOp)) {
        modified = true;
      }
    }

    return mlir::WalkResult::advance();
  });

  // Lower all affine loops back to SCF
  lowerAllAffineToSCF(module);

  return modified;
}

void LoopTiler::addTileConfig(mlir::Operation *loopOp, uint64_t tileSize) {
  if (tileSize <= 0) {
    std::cerr << "LoopTiler: Invalid tile size for loop: " << tileSize << std::endl;
    return;
  }
  tileConfigs_ = {loopOp, tileSize};
}

bool LoopTiler::shouldProcessFunction(mlir::func::FuncOp funcOp) {
  if (targetFunction_ == nullptr) {
    return true;
  }

  return funcOp.getOperation() == targetFunction_;
}

std::optional<uint64_t>
LoopTiler::getTileSizeForLoop(mlir::affine::AffineForOp forOp) {
  mlir::Operation *forOpPtr = forOp.getOperation();

  // Directly match by Operation pointer - no depth check needed
  if (tileConfigs_.loopOp != nullptr && tileConfigs_.loopOp == forOpPtr) {
    return tileConfigs_.tileSize;
  }

  return std::nullopt;
}

bool LoopTiler::applyLoopTiling(mlir::affine::AffineForOp forOp,
                                uint64_t tileSize) {
  // Validate that the loop can be tiled
  if (!forOp) {
    std::cerr << "LoopTiler: Invalid loop operation" << std::endl;
    return false;
  }

  // Check if loop has constant bounds (required for tiling)
  if (!forOp.hasConstantBounds()) {
    std::cerr << "LoopTiler: Cannot tile loop - non-constant bounds" << std::endl;
    return false;
  }

  llvm::SmallVector<mlir::affine::AffineForOp, 6> tiledLoops;
  llvm::SmallVector<mlir::affine::AffineForOp, 1> loopBand = {forOp};

  // Apply tiling with the specified tile size
  if (mlir::failed(mlir::affine::tilePerfectlyNested(
          loopBand, {static_cast<unsigned>(tileSize)}, &tiledLoops))) {
    std::cerr << "LoopTiler: Failed to tile loop" << std::endl;
    return false;
  }

  assert(tiledLoops.size() == 2);
  return true;
}

bool LoopTiler::tileConfiguredLoops(mlir::func::FuncOp funcOp) {
  bool modified = false;

  // Walk through loops and check if they should be tiled
  funcOp.walk([&](mlir::affine::AffineForOp forOp) {
    // Check if this loop should be tiled based on Operation pointer
    auto tileSizeOpt = getTileSizeForLoop(forOp);
    if (tileSizeOpt.has_value()) {
      if (applyLoopTiling(forOp, tileSizeOpt.value())) {
        modified = true;
      }
    }
  });

  return modified;
}

} // namespace megg
