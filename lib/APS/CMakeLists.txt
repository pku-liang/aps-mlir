add_mlir_library(APS
  aps.cpp
  SCFForIndexCast.cpp
  APSMemToMemRef.cpp
  AffineMemToAPSMem.cpp
  MemRefToAPSMem.cpp
  MemoryMapPass.cpp
  InferAffineMemAccess.cpp
  InferAffineMemAccessNew.cpp
  MemLoadDuplication.cpp
  APSSplitMemoryOpsPass.cpp
  APSHoistReadRF.cpp
  APSScalarMemToGlobal.cpp
  APSToCMT2/APSToCMT2GenPass.cpp
  APSToCMT2/APSOpRuleGenerate.cpp
  APSToCMT2/BlockHandler.cpp
  APSToCMT2/BBHandler.cpp
  APSToCMT2/LoopHandler.cpp
  APSToCMT2/ArithmeticOpGenerator.cpp
  APSToCMT2/RegisterOpGenerator.cpp
  APSToCMT2/MemoryOpGenerator.cpp
  APSToCMT2/InterfaceOpGenerator.cpp
  APSToCMT2/APSInterfaceGenerate.cpp
  APSToCMT2/APSMemoryGenerate.cpp
  APSToCMT2/APSGlobalRegisterGenerate.cpp
  Transforms/APSToStandard.cpp
  Transforms/CombExtractToArithTrunc.cpp
  Transforms/ArithSelectToSCFIf.cpp
  Transforms/ArithMulDivToShift.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/APS

  DEPENDS
  MLIRAPSIncGen
  MLIRAPSPassIncGen
  MLIRTORIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRPass
  MLIRTransforms
  CIRCTECMT2
  CIRCTCmt2
  CIRCTFIRRTL
)

add_mlir_public_c_api_library(APSCAPI
  CAPI/APS.cpp
  aps.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR

  LINK_COMPONENTS
  Support
)
