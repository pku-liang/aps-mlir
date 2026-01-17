cmake_minimum_required(VERSION 3.20.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set CMP0116 to NEW to avoid deprecation warning
# This policy is about generated file dependencies
if(POLICY CMP0116)
  cmake_policy(SET CMP0116 NEW)
endif()

# Suppress warnings about runtime path conflicts
if(POLICY CMP0060)
  cmake_policy(SET CMP0060 NEW)
endif()

cmake_policy(SET CMP0174 NEW)
cmake_policy(SET CMP0177 NEW)

include(CheckCXXSourceCompiles)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED YES)

project(aps_mlir LANGUAGES C CXX)

# MLIR LLVM CIRCT should be found from ./circt/build
set(CIRCT_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/circt/build")
set(MLIR_DIR "${CIRCT_BUILD_DIR}/lib/cmake/mlir")
set(LLVM_DIR "${CIRCT_BUILD_DIR}/lib/cmake/llvm")
set(CIRCT_DIR "${CIRCT_BUILD_DIR}/lib/cmake/circt")

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

find_package(ortools CONFIG)
if (ortools_FOUND)
    message(STATUS "Found OR-Tools: ${ortools_LIBDIR}")
else()
    message(STATUS "OR-Tools not found")
endif()

find_package(CIRCT REQUIRED CONFIG)
if(CIRCT_FOUND)
    message(STATUS "Found CIRCT: ${CIRCT_PACKAGE_DIR}")
else()
    message(STATUS "CIRCT not found")
endif()


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CIRCT_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(AddCIRCT)
include(HandleLLVMOptions)

include(MLIRDetectPythonEnv)
# Note that both upstream and downstreams often call this macro. It gates
# internally on the MLIR_CONFIGURE_PYTHON_DEV_PACKAGES option.
mlir_configure_python_dev_packages()

message(STATUS "APS Python bindings are enabled.")
set(MLIR_DISABLE_CONFIGURE_PYTHON_DEV_PACKAGES 0)
mlir_detect_nanobind_install()
# Prime the search like mlir_configure_python_dev_modules
find_package(Python3 3.8 COMPONENTS Interpreter Development)
find_package(Python3 3.8 COMPONENTS Interpreter Development.Module)
find_package(nanobind 2.4.0 CONFIG REQUIRED)
find_package(LPSOLVE REQUIRED)
find_package(nlohmann_json 3.12.0 REQUIRED)

IF(CMAKE_BUILD_TYPE MATCHES Debug)
  message(STATUS "Build Debug version")
  add_compile_options(-g)
ENDIF()
IF(CMAKE_BUILD_TYPE MATCHES Release)
  message(STATUS "Build Release version, SAFE COMPILER IS ON")
  add_compile_options(-fstack-protector-all)
  add_compile_options(-s)
ENDIF()

add_compile_options(-Wno-dev)
add_compile_options(-fexceptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CIRCT_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

# ============================================================================
# Build mlir_utils Python Extension For MEGG
# ============================================================================
set(MLIR_UTILS_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/megg/csrc/mlir_utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/megg/csrc/loop_unroll.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/megg/csrc/loop_unroll_jam.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/megg/csrc/loop_tiling.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/megg/csrc/loop_raise.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/megg/csrc/mlir_ir_builder.cpp
)

nanobind_add_module(mlir_utils
  NB_STATIC
  ${MLIR_UTILS_SOURCES}
)

target_link_libraries(mlir_utils 
  PRIVATE
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRFuncDialect
  MLIRFuncInlinerExtension  # For Func dialect inliner extension
  MLIRAffineDialect
  MLIRAffineUtils
  MLIRAffineTransforms
  MLIRAffineToStandard      # For Affine to Standard/SCF conversion
  MLIRSCFDialect
  MLIRSCFUtils
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRTransforms            # For DialectConversion and patterns
  MLIRLLVMDialect
  MLIRLLVMIRTransforms      # For LLVM dialect inliner interface
  MLIRCAPIIR
)

set_target_properties(mlir_utils PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/megg/megg"
  OUTPUT_NAME "mlir_utils"
)

install(TARGETS mlir_utils LIBRARY DESTINATION megg/megg)

message(STATUS "Build configuration complete")
