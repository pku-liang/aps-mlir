#ifndef TOR_PASSES
#define TOR_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "mlir/Dialect/MemRef/IR/MemRefOps.td"

def TORSchedule : Pass<"schedule-tor", "mlir::tor::DesignOp"> {
  let summary = "Schedule operations and set ref_* timing attributes";
  let constructor = "mlir::createTORSchedulePass()";
  let dependentDialects = ["mlir::tor::TORDialect"];
}

def TORTimeGraph : Pass<"tor-time-graph", "mlir::tor::DesignOp"> {
  let summary = "Generate time graph from ref_* attributes";
  let constructor = "mlir::createTORTimeGraphPass()";
  let dependentDialects = ["mlir::tor::TORDialect"];
}

def TORSplit : Pass<"split-schedule", "mlir::ModuleOp"> {
  let summary = "Split mixed scheduling modules";
  let constructor = "mlir::createTORSplitPass()";
  let dependentDialects = ["mlir::tor::TORDialect"];
}

def SCFToTOR : Pass<"scf-to-tor", "mlir::tor::DesignOp"> {
  let summary = "Convert SCF to unscheduled TOR";
  let constructor = "mlir::createSCFToTORPass()";
  let dependentDialects = ["mlir::tor::TORDialect"];
  let options = [
    Option<"pipeline", "pipeline", "bool", "false", "Attach pipeline attribute to inner pipeline">,
  ];
}

def ConvertInput : Pass<"convert-input", "mlir::ModuleOp"> {
  let summary = "Convert to suitable input";
  let constructor = "mlir::createConvertInputPass()";
  let dependentDialects = ["mlir::tor::TORDialect"];
  let options = [
    // Option<"top_function", "top-function", "std::string", "\"main\"", "The top function name">,
    Option<"clock", "clock", "double", "6.0", "Clock cycle">,
    Option<"resource", "resource", "std::string", "\"./resource_dynamatic.json\"", "Resource file">,
    Option<"output_path", "output-path", "std::string", "\"output_path\"", "output path">,
  ];
}

// def ConvertStream : Pass<"convert-stream", "mlir::ModuleOp"> {
//   let summary = "Convert to stream representation";
//   let constructor = "mlir::createConvertStreamPass()";
//   let dependentDialects = ["mlir::tor::TORDialect"];
//   let options = [
//     Option<"top_function", "top-function", "std::string", "\"main\"", "The top function name">,
//   ];
// }

// def ArrayPartition : Pass<"array-partition", "mlir::func::FuncOp"> {
//   let summary = "Array partition";
//   let constructor = "mlir::createArrayPartitionPass()";
//   let dependentDialects = ["mlir::func::FuncDialect"];
//   let options = [
//     ListOption<"factor", "factor", "int", "Partition factor">,
//     ListOption<"cyclic", "cyclic", "int", "Cyclic or block">,
//     Option<"arg_num", "arg_num", "int", "0", "Parition argument number">
//   ];
// }

def SCFDump : Pass<"dump-scf", "mlir::tor::DesignOp"> {
  let summary = "Dump SCF information";
  let constructor = "mlir::createSCFDumpPass()";
  let dependentDialects = ["mlir::tor::TORDialect"];
  let options = [
    Option<"json", "json", "std::string", "", "SCF json output file">
  ];
}

// def FuncMemrefExtract : Pass<"func-extract","mlir::tor::DesignOp"> {
//   let summary = "extract load and store op in func";
//   let constructor = "mlir::createFuncMemrefExtractPass()";
//   let dependentDialects = ["mlir::tor::TORDialect"];
//     let options = [
//     Option<"pipeline", "pipeline", "bool", "false", "Attach pipeline attribute to inner pipeline">,
//   ];
// }

def SCFIterArgs : Pass<"scf-iterargs", "mlir::tor::DesignOp"> {
  let summary = "add scf.for iterargs";
  let constructor = "mlir::createSCFIterArgsPass()";
  let dependentDialects = ["mlir::tor::TORDialect"];
  let options = [
    Option<"pipeline", "pipeline", "bool", "false", "Attach pipeline attribute to inner pipeline">,
  ];
}

// def MAxiBurstInfer : Pass<"maxi-burst-infer", "mlir::tor::DesignOp"> {
//   let summary = "maxi burst inference";
//   let constructor = "mlir::createMAxiBurstInferPass()";
//   let dependentDialects = ["mlir::tor::TORDialect"];
// }

def TORDump : Pass<"dump-tor", "mlir::tor::DesignOp"> {
  let summary = "Dump TOR information";
  let constructor = "mlir::createTORDumpPass()";
  let dependentDialects = ["mlir::tor::TORDialect"];
  let options = [
    Option<"json", "json", "std::string", "", "TOR json output file">
  ];
}

// def RemoveRedundantAccess: Pass<"remove-access", "mlir::ModuleOp"> {
//   let summary = "Remove redundant access";
//   let constructor = "mlir::createRemoveRedundantAccessPass()";
//   let dependentDialects = ["mlir::tor::TORDialect"];
//   let options = [
//     Option<"mode", "mode", "std::string", "\"conservative\"",
//      "Opt mode, include none, conservative(default), aggressive">
//   ];
// }

def ConvertAffineFor: Pass<"lower-affine-for"> {
  let summary = "lower affine, and for save II attr";
  let constructor = "mlir::createAffineForLoweringPass()";
  let dependentDialects = [
    "memref::MemRefDialect",
    "scf::SCFDialect",
    "vector::VectorDialect"];
}


// def ExtractStridedMetadata: Pass<"lower-extract-strided-metadata"> {
//   let summary = "Lower memref.extract_strided_metadata ops";
//   let constructor = "mlir::createExtractStridedMetadataPass()";
//   let dependentDialects = [
//     "memref::MemRefDialect"];
// }

def RemoveNoUseReinterpretCast: Pass<"lower-reinterpret-cast"> {
  let summary = "Lower memref.reinterpret_cast ops";
  let constructor = "mlir::createReinterpretCastPass()";
  let dependentDialects = [
    "memref::MemRefDialect"];
}

// def EmbedWeightsAndOptimizeLoad: Pass<"embed-weights-optimize-load"> {
//   let summary = "Embed weights and optimize load";
//   let constructor = "mlir::createEmbedWeightsAndOptimizeLoadPass()";
//     let options = [
//     Option<"data_path", "data_path", "std::string", "\"data_path\"", "data path">,
//   ];
// }

// def TosaReshapeNoDataLayout: Pass<"tosa-reshape-no-data-layout","mlir::ModuleOp"> {
//   let summary = "Lower tosa.reshape without data layout";
//   let constructor = "mlir::createTosaReshapeNoDataLayoutPass()";
// }

// def RemoveFuncArg: Pass<"remove-func-arg", "mlir::ModuleOp"> {
//   let summary = "Remove function argument";
//   let constructor = "mlir::createRemoveFuncArgPass()";
// }
def MinMaxToCmpSlect: Pass<"min-max-to-cmp-select", "mlir::ModuleOp"> {
  let summary = "min max to cmp select";
  let constructor = "mlir::createMinMaxToCmpSelectPass()";
}

// def LinalgConvAttr: Pass<"linalg-conv-affine-lower", "mlir::ModuleOp"> {
//   let summary = "linalg conv attr";
//   let constructor = "mlir::createLinalgConvAttrPass()";
// }

def HlsUnroll: Pass<"hls-unroll", "mlir::ModuleOp"> {
  let summary = "HLS loop unroll";
  let constructor = "mlir::createHlsUnrollPass()";
  let dependentDialects = ["mlir::tor::TORDialect"];
}

// def KrnlLower: Pass<"krnl-lower", "mlir::ModuleOp"> {
//   let summary = "krnl lower";
//   let constructor = "mlir::createKrnlLowerPass()";
//   let options = [
//     Option<"reshape_mode", "reshape-mode", "std::string", "\"restore\"", "reshape mode">,
//   ];
// }

// def UnificationIndexCast: Pass<"unification-index-cast", "mlir::func::FuncOp"> {
//   let summary = "add trunc or extui before index cast";
//   let constructor = "mlir::createUnificationIndexCastPass()";
// }

// def FuseLoop: Pass<"fuse-loop", "mlir::func::FuncOp"> {
//   let summary = "fuse loop";
//   let constructor = "mlir::createFuseLoopPass()";
// }

// def Mem2IterArgs: Pass<"mem-to-iter-args", "mlir::ModuleOp"> {
//   let summary = "memory to scf.for iter args";
//   let constructor = "mlir::createMem2IterArgsPass()";
//   let dependentDialects = ["mlir::tor::TORDialect"];
// }

def NormalizeMemrefIndices : Pass<"normalize-memref-indices", "mlir::func::FuncOp"> {
  let summary = "Normalize memref indices for affine-raise-from-memref compatibility";
  let constructor = "mlir::createNormalizeMemrefIndicesPass()";
  let dependentDialects = ["mlir::arith::ArithDialect", "memref::MemRefDialect"];
}

def NewArrayPartition : Pass<"new-array-partition", "mlir::ModuleOp"> {
  let summary = "New Array partition";
  let constructor = "mlir::createNewArrayPartitionPass()";
  let dependentDialects = ["mlir::tor::TORDialect", "memref::MemRefDialect"];
}

def ArrayOpt: Pass<"array-opt", "mlir::ModuleOp"> {
  let summary = "array opt, dimension simplify and single data simplify";
  let constructor = "mlir::createArrayOptPass()";
  let dependentDialects = ["mlir::tor::TORDialect", "memref::MemRefDialect"];
}

def CountCycles : Pass<"count-cycles", "mlir::ModuleOp"> {
  let summary = "Count cycles";
  let constructor = "mlir::createCountCyclesPass()";
  let dependentDialects = ["mlir::tor::TORDialect"];
  let options = [
    Option<"output_dir", "output-dir", "std::string", "", "Output position">
  ];
}

// def GenPragmaReport : Pass<"generate-pragma-report", "mlir::ModuleOp"> {
//   let summary = "Generate pragma usage report";
//   let constructor = "mlir::createGenPragmaReportPass()";
// }

// def AddPragma: Pass<"add-pragma", "mlir::ModuleOp"> {
//   let summary = "add pragma";
//   let constructor = "mlir::createAddPragmaPass()";
//   let options = [
//     Option<"conv_unroll_layers", "conv-unroll-layers", "int", "1", "ONNXQLinearConv and ONNXConv op unroll layers">,
//   ];
// }

def TORFuse : Pass<"tor-fuse", "mlir::tor::DesignOp"> {
  let summary = "tor fuse";
  let constructor = "mlir::createTORFusePass()";
  let dependentDialects = ["mlir::tor::TORDialect"];
}

// def LoopFlatten: Pass<"loop-flatten", "mlir::func::FuncOp"> {
//   let summary = "loop flatten";
//   let constructor = "mlir::createLoopFlattenPass()";
// }

def ExpressionBalance : Pass<"expression-balance","mlir::func::FuncOp"> {
  let summary = "expression balance";
  let constructor = "mlir::createExpressionBalancePass()";
}

// def LoopMerge: Pass<"loop-merge", "mlir::func::FuncOp"> {
//   let summary = "loop merge";
//   let constructor = "mlir::createLoopMergePass()";
// }

def LoopTripcount: Pass<"loop-tripcount", "mlir::func::FuncOp"> {
  let summary = "loop tripcount";
  let constructor = "mlir::createLoopTripcountPass()";
}

// def MemrefReuse: Pass<"memref-reuse", "mlir::func::FuncOp"> {
//   let summary = "memref reuse";
//   let constructor = "mlir::createMemrefReusePass()";
// }

// def MemrefGlobalToConstant: Pass<"memref-global-to-constant", "mlir::ModuleOp"> {
//   let summary = "memref global to constant";
//   let constructor = "mlir::createMemrefGlobalToConstantPass()";
// }

// def StructSplit: Pass<"struct-split", "mlir::ModuleOp"> {
//   let summary = "struct split";
//   let constructor = "mlir::createStructSplitPass()";
//   let dependentDialects = ["mlir::tor::TORDialect", "LLVM::LLVMDialect",
//                            "memref::MemRefDialect"];
// }

// def ArrayUseOffset: Pass<"array-use-offset", "mlir::ModuleOp"> {
//   let summary = "array use offset";
//   let constructor = "mlir::createArrayUseOffsetPass()";
//   let dependentDialects = ["LLVM::LLVMDialect", "memref::MemRefDialect"];
// }

// def DemangleFuncName : Pass<"demangle-func-name", "mlir::ModuleOp"> {
//   let summary = "Demangle function name, solve cpp mangle";
//   let constructor = "mlir::createDemangleFuncNamePass()";
// }

// def ONNXLower: Pass<"onnx-lower", "mlir::ModuleOp"> {
//   let summary = "onnx lower";
//   let constructor = "mlir::createONNXLowerPass()";
//   let dependentDialects = ["memref::MemRefDialect", "affine::AffineDialect",
//                            "tensor::TensorDialect", "math::MathDialect"];
// }

// def ConvertMathToCall : Pass<"convert-math-to-call", "mlir::ModuleOp"> {
//   let summary = "Convert Math dialect to lib call";
//   let constructor = "mlir::createConvertMathToCallPass()";
//   let dependentDialects = ["mlir::tor::TORDialect", "affine::AffineDialect"];
//   let options = [
//     Option<"resource", "resource", "std::string", "\"./resource_dynamatic.json\"", "Resource file">
//   ];
// }

// def TopFuncArg2Global : Pass<"top-func-arg-to-global", "mlir::ModuleOp"> {
//   let summary = "Top function argument to global";
//   let constructor = "mlir::createTopFuncArg2GlobalPass()";
//   let dependentDialects = ["memref::MemRefDialect"];
//   let options = [
//     Option<"top_function", "top-function", "std::string", "\"main\"", "The top function name">,
//   ];
// }

// def IgnoreAliasingLICM : Pass<"ignore-aliasing-licm", "mlir::ModuleOp"> {
//   let summary = "LICM(Loop-Invariant Code Motion) on known loops with ignore aliasing";
//   let constructor = "mlir::createIgnoreAliasingLICMPass()";
// }

def RaiseSCFToAffine : Pass<"raise-scf-to-affine"> {
  let summary = "Raise SCF loops to Affine loops when possible";
  let constructor = "mlir::createRaiseSCFToAffinePass()";
  let dependentDialects = [
    "affine::AffineDialect",
    "scf::SCFDialect",
    "arith::ArithDialect"
  ];
}

#endif // TOR_PASSES
