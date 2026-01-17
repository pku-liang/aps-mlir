#ifndef APS_PASSES
#define APS_PASSES

include "mlir/Pass/PassBase.td"

def SCFForIndexCast : Pass<"scf-for-index-cast", "::mlir::func::FuncOp"> {
  let summary = "Convert SCF for loop bounds to index type";
  let description = [{
    This pass converts scf.for loops that use integer types (i32, i64, etc.)
    for their lower bound, upper bound, and step to use the index type instead.
    This is necessary for raising SCF loops to affine loops, which require
    index-typed bounds.
  }];
  let constructor = "mlir::createSCFForIndexCastPass()";
}

def APSMemToMemRef : Pass<"aps-mem-to-memref", "::mlir::func::FuncOp"> {
  let summary = "Convert APS memload/memstore to memref.load/memref.store";
  let description = [{
    This pass converts aps.memload and aps.memstore operations to their
    standard memref counterparts (memref.load and memref.store).
    Integer-typed indices are automatically cast to index type as required
    by memref operations.
  }];
  let constructor = "mlir::createAPSMemToMemRefPass()";
}

def MemRefToAPSMem : Pass<"affine-mem-to-aps-mem", "::mlir::func::FuncOp"> {
  let summary = "Convert affine.load/affine.store to APS memload/memstore";
  let description = [{
    This pass converts affine.load and affine.store operations to APS
    memload and memstore operations. Index-typed indices are automatically
    cast to i32 type as used by APS memory operations.
  }];
  let constructor = "mlir::createAffineMemToAPSMemPass()";
}

def MemRefMemToAPSMem : Pass<"memref-to-aps-mem", "::mlir::func::FuncOp"> {
  let summary = "Convert memref.load/memref.store to APS memload/memstore";
  let description = [{
    This pass converts memref.load and memref.store operations to APS
    memload and memstore operations. Index-typed indices are automatically
    cast to i32 type as used by APS memory operations.
  }];
  let constructor = "mlir::createMemRefToAPSMemPass()";
}

def MemoryMap : Pass<"memory-map", "::mlir::ModuleOp"> {
  let summary = "Generate memory map for global memrefs after array partition";
  let description = [{
    This pass generates a memory address map for all global memrefs in the module,
    including partitioned arrays. It creates an aps.memorymap operation containing
    aps.mem_entry operations for each memref group.

    For partitioned arrays, all partition banks are grouped under a single mem_entry
    with information about the partition mode (cyclic/block), number of banks, and
    base address assignment.
  }];
  let constructor = "mlir::createMemoryMapPass()";
  let dependentDialects = ["aps::APSDialect", "memref::MemRefDialect"];
}

def InferAffineMemAccess : Pass<"infer-affine-mem-access"> {
  let summary = "Infer affine maps for memref.load/store operations";
  let description = [{
    This pass analyzes memref.load and memref.store operations with computed
    indices and attempts to infer affine maps of the form a*x + b, where:
    - x is an affine.for induction variable
    - a is a constant multiplier (can be 1)
    - b is a constant offset (can be 0)

    When such a pattern is detected, the memref operation is replaced with
    an affine.load or affine.store operation using the inferred affine map.

    Example transformation:
      %idx = arith.index_cast %iv : index to i32
      %mul = arith.muli %idx, %c2 : i32
      %cast = arith.index_cast %mul : i32 to index
      %val = memref.load %array[%cast]
    becomes:
      %val = affine.load %array[%iv * 2]
  }];
  let constructor = "mlir::createInferAffineMemAccessPass()";
  let dependentDialects = ["affine::AffineDialect", "arith::ArithDialect", "memref::MemRefDialect"];
}

def InferAffineMemAccessNew : Pass<"infer-affine-mem-access-new", "::mlir::func::FuncOp"> {
  let summary = "Infer affine maps for memref.load/store by cloning index arithmetic";
  let description = [{
    This pass converts memref.load/store to affine.load/store by:
    1. Recursively cloning the index arithmetic chain in index type
    2. Building an AffineExpr from the clean index-type computation
    3. Creating affine.load/store with the inferred affine map

    Unlike infer-affine-mem-access, this pass handles arbitrary nested
    arithmetic including patterns like ((i*2+1)*4+j) by cloning the entire
    computation tree in index type first, then analyzing it.

    Run CSE afterward to deduplicate any redundant index computations.
  }];
  let constructor = "mlir::createInferAffineMemAccessNewPass()";
  let dependentDialects = ["affine::AffineDialect", "arith::ArithDialect", "memref::MemRefDialect"];
}

def APSMemLoadDuplication : Pass<"aps-memload-duplication", "mlir::tor::FuncOp"> {
  let summary = "Duplicate memory loads to reduce FIFO usage";
  let description = [{
    This pass duplicates aps.memload operations when they are used at different
    cycles than when they are loaded. If there is no memory conflict at the
    user's cycle, the memload is duplicated at that cycle to eliminate the need
    for a FIFO buffer to hold the value across cycles.

    For example, if a memload executes at cycle 7 but is used by an operation at
    cycle 10, and there's no other access to that memory at cycle 10, the pass
    will duplicate the memload at cycle 10, saving a FIFO.
  }];
  let constructor = "mlir::createAPSMemLoadDuplicationPass()";
  let dependentDialects = ["mlir::tor::TORDialect", "aps::APSDialect"];
}

def APSSplitMemoryOps : Pass<"aps-split-memory-ops", "mlir::tor::DesignOp"> {
  let summary = "Split APS memory operations into request-collect pairs";
  let constructor = "mlir::createAPSSplitMemoryOpsPass()";
  let dependentDialects = ["mlir::tor::TORDialect", "aps::APSDialect"];
}

def APSMemoryPoolGen : Pass<"aps-memory-pool-gen", "::mlir::ModuleOp"> {
  let summary = "Generate scratchpad memory pool and CMT2 rules from aps.memorymap";
  let description = [{
    This pass reads aps.memorymap operation and generates a CMT2-based
    scratchpad memory pool with appropriate SRAM instances and arbitration logic,
    along with rule-based main modules for TOR functions.

    For each aps.mem_entry:
    - Creates SRAM instances (using Mem1r1w from ModuleLibrary)
    - Generates one RW port for burst access (indexed by address)
    - Generates one RW port per memory entry for application access
    - Implements address decoding and bank selection logic

    For each TOR function:
    - Creates CMT2 rules based on timegraph scheduling
    - Generates burst read/write methods for memory access
    - Provides unified interface to memory pool

    This unified pass replaces the separate APSToCMT2GenPass and APSRuleGenPass.
  }];
  let constructor = "mlir::createAPSToCMT2GenPass()";
}

def APSToStandard : Pass<"aps-to-standard", "::mlir::ModuleOp"> {
  let summary = "Lower APS dialect operations to standard MLIR dialects";
  let description = [{
    This pass converts APS dialect operations to standard MLIR operations:
    - aps.readrf -> function parameters
    - aps.writerf -> function return values
    - aps.memload/memstore -> memref.load/store
    - aps.memburst operations -> removed
    - memref.get_global -> function parameters
    - Global declarations -> removed

    The transformation is designed for software-level instruction matching,
    removing hardware-specific register file and scratchpad memory semantics.
  }];
  let constructor = "mlir::createAPSToStandardPass()";
  let dependentDialects = ["::mlir::arith::ArithDialect", "::mlir::func::FuncDialect",
                          "::mlir::memref::MemRefDialect", "::mlir::scf::SCFDialect"];
}

def CombExtractToArithTrunc : Pass<"comb-extract-to-arith-trunc"> {
  let summary = "Convert comb.extract operations to arith.trunci";
  let description = [{
    This pass converts all comb.extract operations to equivalent arith.trunci
    operations, optionally preceded by arith.shrui if extracting from non-zero bit.

    Transformations:
    - comb.extract %x from 0 : (iN) -> iM
      => arith.trunci %x : iN to iM

    - comb.extract %x from K : (iN) -> iM (K > 0)
      => %shifted = arith.shrui %x, K : iN
         %result = arith.trunci %shifted : iN to iM

    This is useful for lowering to standard dialects that don't have comb dialect,
    such as for CPU simulation or instruction matching with polygeist-generated MLIR.
  }];
  let constructor = "mlir::createCombExtractToArithTruncPass()";
  let dependentDialects = ["::mlir::arith::ArithDialect"];
}

def ArithSelectToSCFIf : Pass<"arith-select-to-scf-if"> {
  let summary = "Convert arith.select operations to scf.if";
  let description = [{
    This pass converts all arith.select operations to equivalent scf.if
    operations with explicit then/else branches.

    Transformation:
    - arith.select %cond, %true_val, %false_val : T
      =>
      scf.if %cond -> T {
        scf.yield %true_val : T
      } else {
        scf.yield %false_val : T
      }

    This is useful when:
    - The backend prefers explicit control flow over conditional selection
    - Pattern matching with control-flow heavy code (e.g., C code with if/else)
    - Analyzing or optimizing control flow dependencies
    - Matching with polygeist-generated MLIR from C code

    Note: This transformation may increase code size and can make certain
    optimizations (like vectorization) more difficult. Use with caution.
  }];
  let constructor = "mlir::createArithSelectToSCFIfPass()";
  let dependentDialects = ["::mlir::arith::ArithDialect", "::mlir::scf::SCFDialect"];
}

def ArithMulDivToShift : Pass<"arith-muldiv-to-shift"> {
  let summary = "Convert arith mul/div by power of 2 to shift operations";
  let description = [{
    This pass converts arith.muli and arith.divui/divsi operations to shift
    operations when the second operand is a constant power of 2.

    Transformations:
    - arith.muli %x, 2^n  => arith.shli %x, n
    - arith.divui %x, 2^n => arith.shrui %x, n
    - arith.divsi %x, 2^n => arith.shrsi with bias correction for negative values

    For signed division (arith.divsi), the transformation adds a bias for
    negative values to ensure correct truncation toward zero (matching C
    semantics), rather than floor division which arithmetic shift alone provides.

    The bias correction is:
      sign = shrsi(x, bitwidth-1)   // -1 if negative, 0 if positive
      bias = andi(sign, 2^n - 1)    // (2^n - 1) if negative, 0 if positive
      result = shrsi(addi(x, bias), n)

    This optimization is useful for:
    - Reducing hardware complexity (shift is simpler than multiply/divide)
    - Improving performance on targets where shift is faster than multiply/divide
    - Matching patterns in instruction selection

    Example:
      %c8 = arith.constant 8 : i32
      %result = arith.muli %x, %c8 : i32
    becomes:
      %c3 = arith.constant 3 : i32
      %result = arith.shli %x, %c3 : i32
  }];
  let constructor = "mlir::createArithMulDivToShiftPass()";
  let dependentDialects = ["::mlir::arith::ArithDialect"];
}

def APSHoistReadRF : Pass<"aps-hoist-readrf", "::mlir::func::FuncOp"> {
  let summary = "Hoist all aps.readrf operations to function entry block";
  let description = [{
    This pass moves all aps.readrf operations to the entry block of their
    containing function. This ensures they can be scheduled at cycle 0,
    since readrf operations only depend on function arguments which are
    available at function entry.

    This transformation is safe because:
    - aps.readrf only reads from function arguments (register file indices)
    - Function arguments are available throughout the function
    - There are no side effects or dependencies on control flow

    This pass should run before SCF to TOR conversion and scheduling to avoid
    control-flow constraints preventing readrf from being scheduled early.

    Example transformation:
      func.func @foo(%arg0: i5, %arg1: i5) {
        // some operations
        scf.for ... {
          // loop body
        }
        %0 = aps.readrf %arg1 : i5 -> i32  // readrf after loop
        // use %0
      }
    becomes:
      func.func @foo(%arg0: i5, %arg1: i5) {
        %0 = aps.readrf %arg1 : i5 -> i32  // readrf at entry
        // some operations
        scf.for ... {
          // loop body
        }
        // use %0
      }
  }];
  let constructor = "mlir::createAPSHoistReadRFPass()";
  let dependentDialects = ["aps::APSDialect"];
}

def APSScalarMemToGlobal : Pass<"aps-scalar-mem-to-global", "::mlir::func::FuncOp"> {
  let summary = "Convert aps.memload/memstore on memref<1xT> to aps.globalload/globalstore";
  let description = [{
    This pass converts aps.memload and aps.memstore operations that access
    single-element memrefs (memref<1xT>) to aps.globalload and aps.globalstore
    operations.

    A memref<1xT> is semantically equivalent to a scalar value, and using
    the global operations enables better optimization through the existing
    canonicalizers:
    - FoldGlobalLoadAfterStore: Folds loads that follow stores to same global
    - RemoveDeadGlobalStore: Removes stores that are immediately overwritten

    This is particularly useful after loop unrolling where store-load patterns
    emerge on scalar variables stored in memref<1xT>.

    Example transformation:
      %0 = memref.get_global @vec_0 : memref<1xi32>
      aps.memstore %val, %0[%c0] : i32, memref<1xi32>, i32
      %1 = aps.memload %0[%c0] : memref<1xi32>, i32 -> i32
    becomes:
      aps.globalstore %val, @vec_0 : i32
      %1 = aps.globalload @vec_0 : i32

    The pass traces memref values back to memref.get_global operations to
    determine the global symbol name for the conversion.
  }];
  let constructor = "mlir::createAPSScalarMemToGlobalPass()";
  let dependentDialects = ["aps::APSDialect", "memref::MemRefDialect"];
}

#endif // APS_PASSES
