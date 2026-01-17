################################################################################
# Set up Python binding tools
################################################################################

include(AddMLIRPython)

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=circt.")

################################################################################
# Declare native Python extension
################################################################################

set(CIRCT_PYTHON_SRC_DIR ${CMAKE_SOURCE_DIR}/circt/lib/Bindings/Python)
set(CIRCT_RELATIVE_PYTHON_SRC_DIR ../../circt/lib/Bindings/Python)
set(MLIR_PYTHON_SRC_DIR ${CMAKE_SOURCE_DIR}/circt/llvm/mlir/python/mlir)
message(STATUS "CIRCT_PYTHON_SRC_DIR: ${CIRCT_PYTHON_SRC_DIR}")
message(STATUS "MLIR_PYTHON_SRC_DIR: ${MLIR_PYTHON_SRC_DIR}")


set(PYTHON_BINDINGS_SOURCES
  ${CIRCT_RELATIVE_PYTHON_SRC_DIR}/AIGModule.cpp
  CIRCTModule.cpp
  ${CIRCT_RELATIVE_PYTHON_SRC_DIR}/ESIModule.cpp
  ${CIRCT_RELATIVE_PYTHON_SRC_DIR}/HWModule.cpp
  ${CIRCT_RELATIVE_PYTHON_SRC_DIR}/MSFTModule.cpp
  ${CIRCT_RELATIVE_PYTHON_SRC_DIR}/OMModule.cpp
  ${CIRCT_RELATIVE_PYTHON_SRC_DIR}/RTGModule.cpp
  ${CIRCT_RELATIVE_PYTHON_SRC_DIR}/SeqModule.cpp
  ${CIRCT_RELATIVE_PYTHON_SRC_DIR}/SupportModule.cpp
  ${CIRCT_RELATIVE_PYTHON_SRC_DIR}/SVModule.cpp
  # Headers must be included explicitly so they are installed.
  CIRCTModules.h
  ${CIRCT_RELATIVE_PYTHON_SRC_DIR}/NanobindUtils.h
)

set(PYTHON_BINDINGS_LINK_LIBS
  APS
  APSCAPI
  CIRCTCAPIAIG
  CIRCTCAPIArc
  CIRCTCAPIComb
  CIRCTCAPIConversion
  CIRCTCAPIDC
  CIRCTCAPIDebug
  CIRCTCAPIEmit
  CIRCTCAPIESI
  CIRCTCAPIExportVerilog
  CIRCTCAPIFSM
  CIRCTCAPIHandshake
  CIRCTCAPIHW
  CIRCTCAPIHWArith
  CIRCTCAPIKanagawa
  CIRCTCAPILTL
  CIRCTCAPIMSFT
  CIRCTCAPIOM
  CIRCTCAPIPipeline
  CIRCTCAPIRTG
  CIRCTCAPISeq
  CIRCTCAPISV
  CIRCTCAPISupport
  CIRCTCAPIVerif
  CIRCTCAPITransforms
  CIRCTCAPISynth
  MLIRCAPIIndex
  MLIRCAPISMT
  MLIRCAPIExportSMTLIB
  MLIRCAPISCF
  MLIRCAPIMemRef
  # needed for mlirFrozenRewritePatternSetDestroy
  # but not the actual passes
  MLIRCAPITransforms
)

declare_mlir_python_extension(CIRCTBindingsPythonExtension
  MODULE_NAME _circt
  SOURCES
    ${PYTHON_BINDINGS_SOURCES}
  EMBED_CAPI_LINK_LIBS
    ${PYTHON_BINDINGS_LINK_LIBS}
  PRIVATE_LINK_LIBS
    LLVMSupport
  PYTHON_BINDINGS_LIBRARY
    nanobind
)

add_dependencies(CIRCTBindingsPythonExtension circt-headers)

################################################################################
# Declare Python sources
################################################################################

declare_mlir_python_sources(CIRCTBindingsPythonSources
  ROOT_DIR "${CIRCT_PYTHON_SRC_DIR}"
  SOURCES
    __init__.py
)

declare_mlir_python_sources(CIRCTBindingsPythonSources.Support
  ADD_TO_PARENT CIRCTBindingsPythonSources
  ROOT_DIR "${CIRCT_PYTHON_SRC_DIR}"
  SOURCES
    support.py
)

################################################################################
# Declare dialect-specific bindings.
################################################################################

# Ensure the build directory for generated Python files exists. Ninja is able to
# generate this, but make does not and the build fails.
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/lib/Python/circt/dialects)

declare_mlir_python_sources(CIRCTBindingsPythonSources.Dialects
  ADD_TO_PARENT CIRCTBindingsPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${CIRCT_PYTHON_SRC_DIR}"
  TD_FILE dialects/CombOps.td
  SOURCES
    dialects/comb.py
  DIALECT_NAME comb)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${CMAKE_SOURCE_DIR}/lib/Python"
  TD_FILE dialects/APSOps.td
  SOURCES
    dialects/aps.py
  DIALECT_NAME aps)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${CIRCT_PYTHON_SRC_DIR}"
  TD_FILE dialects/AIGOps.td
  SOURCES
    dialects/aig.py
  DIALECT_NAME aig)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${CIRCT_PYTHON_SRC_DIR}"
  TD_FILE dialects/DebugOps.td
  SOURCES
    dialects/debug.py
  DIALECT_NAME dbg)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${CIRCT_PYTHON_SRC_DIR}"
  TD_FILE dialects/ESIOps.td
  SOURCES
    dialects/esi.py
  DIALECT_NAME esi)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${CIRCT_PYTHON_SRC_DIR}"
  TD_FILE dialects/HandshakeOps.td
  SOURCES
    dialects/handshake.py
  DIALECT_NAME handshake)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${CIRCT_PYTHON_SRC_DIR}"
  TD_FILE dialects/HWOps.td
  SOURCES
    dialects/hw.py
  DIALECT_NAME hw)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${CIRCT_PYTHON_SRC_DIR}"
  TD_FILE dialects/MSFTOps.td
  SOURCES
    dialects/msft.py
  DIALECT_NAME msft)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${CIRCT_PYTHON_SRC_DIR}"
  TD_FILE dialects/OMOps.td
  SOURCES
    dialects/om.py
  DIALECT_NAME om)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${CIRCT_PYTHON_SRC_DIR}"
  TD_FILE dialects/RTGOps.td
  SOURCES
    dialects/rtg.py
  DIALECT_NAME rtg)

if (CIRCT_INCLUDE_TESTS)
  declare_mlir_dialect_python_bindings(
    ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
    ROOT_DIR "${CIRCT_PYTHON_SRC_DIR}"
    TD_FILE dialects/RTGTestOps.td
    SOURCES
      dialects/rtgtest.py
    DIALECT_NAME rtgtest)
endif()

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${CIRCT_PYTHON_SRC_DIR}"
  TD_FILE dialects/SeqOps.td
  SOURCES
    dialects/seq.py
  DIALECT_NAME seq)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${CIRCT_PYTHON_SRC_DIR}"
  TD_FILE dialects/SVOps.td
  SOURCES
    dialects/sv.py
  DIALECT_NAME sv)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${CIRCT_PYTHON_SRC_DIR}"
  TD_FILE dialects/FSMOps.td
  SOURCES
    dialects/fsm.py
  DIALECT_NAME fsm)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${CIRCT_PYTHON_SRC_DIR}"
  TD_FILE dialects/HWArithOps.td
  SOURCES
    dialects/hwarith.py
  DIALECT_NAME hwarith)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${CIRCT_PYTHON_SRC_DIR}"
  TD_FILE dialects/LTLOps.td
  SOURCES
    dialects/ltl.py
  DIALECT_NAME ltl)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${CIRCT_PYTHON_SRC_DIR}"
  TD_FILE dialects/VerifOps.td
  SOURCES
    dialects/verif.py
  DIALECT_NAME verif)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${CIRCT_PYTHON_SRC_DIR}"
  TD_FILE dialects/EmitOps.td
  SOURCES
    dialects/emit.py
  DIALECT_NAME emit)

  
# We need the 'arith.py' file because 'scf.py' imports from it. We are not
# calling the function that constructs arith ops and thus don't need to
# register and link against the dialect.
declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${MLIR_PYTHON_SRC_DIR}"
  TD_FILE dialects/ArithOps.td
  SOURCES
    dialects/arith.py
  DIALECT_NAME arith
  GEN_ENUM_BINDINGS)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${MLIR_PYTHON_SRC_DIR}"
  TD_FILE dialects/IndexOps.td
  SOURCES
    dialects/index.py
  DIALECT_NAME index
  GEN_ENUM_BINDINGS)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${MLIR_PYTHON_SRC_DIR}"
  TD_FILE dialects/SCFOps.td
  SOURCES
    dialects/scf.py
  DIALECT_NAME scf)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
  ROOT_DIR "${MLIR_PYTHON_SRC_DIR}"
  TD_FILE dialects/FuncOps.td
  SOURCES
    dialects/func.py
  DIALECT_NAME func)

declare_mlir_dialect_python_bindings(
    ADD_TO_PARENT CIRCTBindingsPythonSources.Dialects
    ROOT_DIR "${MLIR_PYTHON_SRC_DIR}"
    TD_FILE dialects/MemRefOps.td
    SOURCES
      dialects/memref.py
    DIALECT_NAME memref)

################################################################################
# Build composite binaries
################################################################################

# Bundle our own, self-contained CAPI library with all of our deps.
add_mlir_python_common_capi_library(CIRCTBindingsPythonCAPI
  INSTALL_COMPONENT CIRCTPythonModules
  INSTALL_DESTINATION python_packages/circt_core/circt/_mlir_libs
  OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python_packages/circt_core/circt/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    MLIRPythonSources.Core
    # MLIRPythonExtension.RegisterEverything
    CIRCTBindingsPythonExtension
)

# Bundle the CIRCT python sources into our package.
add_mlir_python_modules(CIRCTPythonModules
  ROOT_PREFIX "${CMAKE_BINARY_DIR}/python_packages/circt_core/circt"
  INSTALL_PREFIX "python_packages/circt_core/circt"
  DECLARED_SOURCES
    MLIRPythonSources.Core
    # MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Dialects.smt
    CIRCTBindingsPythonExtension
    CIRCTBindingsPythonSources
  COMMON_CAPI_LINK_LIBS
    CIRCTBindingsPythonCAPI
)
