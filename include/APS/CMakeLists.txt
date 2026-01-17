add_mlir_dialect(APS aps)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name APS)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix APS)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix APS)
add_public_tablegen_target(MLIRAPSPassIncGen)
add_dependencies(mlir-headers MLIRAPSPassIncGen)

add_mlir_doc(Passes APSPasses ./ -gen-pass-doc)