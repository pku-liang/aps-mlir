add_executable(aps-e2e
        main.cc
        aps_opt_driver.cc
)

target_link_libraries(aps-e2e PRIVATE
        MLIROptLib
        MLIRTOR
        TORSCHEDULE
        APS
        CIRCTComb
        CIRCTECMT2
        CIRCTCmt2
        CIRCTFIRRTL
        
        ${LP_SOLVE_LIB}
        nlohmann_json::nlohmann_json
      
        MLIRBytecodeReader
        MLIRBytecodeWriter
        MLIRFuncDialect

        MLIRSCFTransforms
      
        MLIRSupport
        MLIRIR
        MLIRAnalysis
        MLIRLLVMDialect
        MLIRNVVMDialect
        MLIROpenMPDialect
        MLIRGPUTransforms
        MLIRTransforms
        MLIRSCFToControlFlow
        MLIRFuncToLLVM
      
        MLIRAffineTransforms
        MLIRAffineToStandard
        MLIRMathToLLVM
        MLIRTargetLLVMIRImport
        MLIRLLVMToLLVMIRTranslation
        MLIRSCFToOpenMP
        MLIROpenMPToLLVM
        MLIROpenMPToLLVMIRTranslation
        MLIRGPUToNVVMTransforms
        MLIRGPUToGPURuntimeTransforms
        MLIRFuncAllExtensions
        MLIRLLVMIRToLLVMTranslation
        MLIRFromLLVMIRTranslationRegistration
        MLIRAffineAnalysis
        MLIRAnalysis
        MLIRCastInterfaces
        MLIRDialect
        MLIRParser
        MLIRPass
        MLIRTransforms
        MLIRTransformUtils
        MLIRSupport
        MLIRIR
        MLIRCAPIRegisterEverything
        MLIRBuiltinToLLVMIRTranslation
        )

add_executable(aps-opt
        aps_opt.cc
        aps_opt_driver.cc
)

target_link_libraries(aps-opt PRIVATE
        MLIROptLib
        MLIRTOR
        TORSCHEDULE
        APS
        CIRCTComb
        CIRCTECMT2
        CIRCTCmt2
        CIRCTFIRRTL
        
        ${LP_SOLVE_LIB}
        nlohmann_json::nlohmann_json
      
        MLIRBytecodeReader
        MLIRBytecodeWriter
        MLIRFuncDialect

        MLIRSCFTransforms
      
        MLIRSupport
        MLIRIR
        MLIRAnalysis
        MLIRLLVMDialect
        MLIRNVVMDialect
        MLIROpenMPDialect
        MLIRGPUTransforms
        MLIRTransforms
        MLIRSCFToControlFlow
        MLIRFuncToLLVM
      
        MLIRAffineTransforms
        MLIRAffineToStandard
        MLIRMathToLLVM
        MLIRTargetLLVMIRImport
        MLIRLLVMToLLVMIRTranslation
        MLIRSCFToOpenMP
        MLIROpenMPToLLVM
        MLIROpenMPToLLVMIRTranslation
        MLIRGPUToNVVMTransforms
        MLIRGPUToGPURuntimeTransforms
        MLIRFuncAllExtensions
        MLIRLLVMIRToLLVMTranslation
        MLIRFromLLVMIRTranslationRegistration
        MLIRAffineAnalysis
        MLIRAnalysis
        MLIRCastInterfaces
        MLIRDialect
        MLIRParser
        MLIRPass
        MLIRTransforms
        MLIRTransformUtils
        MLIRSupport
        MLIRIR
        MLIRCAPIRegisterEverything
        MLIRBuiltinToLLVMIRTranslation
        )

set(LIBC_DIR ${CMAKE_SOURCE_DIR}/demo/libc)
set(BIN_LIBC_DIR ${CMAKE_BINARY_DIR}/bin/demo/libc)
