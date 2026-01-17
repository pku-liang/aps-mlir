#include "aps_opt.hpp"

using namespace mlir;
using namespace circt;
using namespace cmt2;

int aps_opt_driver(int argc, char **argv) {
    mlir::registerAllPasses();

    mlir::DialectRegistry registry_aps;
    registry_aps.insert<mlir::affine::AffineDialect>();
    registry_aps.insert<mlir::LLVM::LLVMDialect>();
    registry_aps.insert<mlir::memref::MemRefDialect>();
    registry_aps.insert<mlir::arith::ArithDialect>();
    registry_aps.insert<mlir::scf::SCFDialect>();
    registry_aps.insert<mlir::tor::TORDialect>();
    registry_aps.insert<mlir::func::FuncDialect>();
    registry_aps.insert<mlir::math::MathDialect>();
    registry_aps.insert<aps::APSDialect>();
    registry_aps.insert<circt::comb::CombDialect>();
    registry_aps.insert<circt::cmt2::Cmt2Dialect>();

    mlir::registerTORPasses();
    mlir::registerAPSPasses();
    cmt2::registerPasses();

    return failed(mlir::MlirOptMain(argc, argv, "aps optimizer driver\n", registry_aps));
}
