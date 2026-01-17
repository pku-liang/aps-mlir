set(LLVM_TARGET_DEFINITIONS Passes.td)
add_compile_options(-Wno-dev)

mlir_tablegen(Passes.h.inc -gen-pass-decls -name TOR)
add_public_tablegen_target(MLIRTORPassIncGen)

set(LLVM_TARGET_DEFINITIONS TOR.td)
# Registers the enum utility generator to mlir-tblgen.
mlir_tablegen(TOREnums.h.inc -gen-enum-decls)
# Registers the enum utility generator to mlir-tblgen.
mlir_tablegen(TOREnums.cpp.inc -gen-enum-defs)
mlir_tablegen(TORAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(TORAttrs.cpp.inc -gen-attrdef-defs) 
add_public_tablegen_target(MLIRTOREnumsIncGen)

# Add the TOR for the dialect operations
add_mlir_dialect(TOR tor)

# Necessary to generate documentation
add_mlir_doc(TORDialect -gen-dialect-doc TORDialect TOR/)
add_mlir_doc(TOR -gen-op-doc TOR TOR/)
