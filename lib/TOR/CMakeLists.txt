add_mlir_dialect_library(MLIRTOR
    TORDialect.cpp
    TOR.cpp
    TORTypes.cpp
    SCFDump.cpp
    SCFIterArgs.cpp
    SCFToTOR.cpp
    TORSchedulePass.cpp
    TORTimeGraphPass.cpp
    TORSplitPass.cpp
    TORDump.cpp
    ConvertInput.cpp
    TORAttrs.cpp
    SCEVAnalysis.cpp
    DependenceAnalysis.cpp
    LowerAffineFor.cpp
    ReinterpretCastLower.cpp
    MinMaxToCmpSelectPass.cpp
    HlsUnroll.cpp
    NewArrayPartition.cpp
    AffineModSimplify.cpp
    NormalizeMemrefIndices.cpp
    CountCycles.cpp
    ArrayOpt.cpp
    TORFuse.cpp
    LoopTripcount.cpp
    ExpressionBalance.cpp
    RaiseToAffine.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TOR
    ${RROJECT_SOURCE_DIR}/include/Schedule
    DEPENDS
    MLIRTORIncGen
    MLIRTOREnumsIncGen
    MLIRTORPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
)


target_link_libraries(MLIRTOR PRIVATE LPSOLVE::LPSOLVE)
