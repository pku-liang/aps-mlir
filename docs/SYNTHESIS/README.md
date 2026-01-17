# Hardware Synthesis

The APS synthesis flow transforms high-level ISAX descriptions in CADL to synthesizable RTL (SystemVerilog).

## Pipeline Overview

**CADL** → **Pre-Opt MLIR** → **HLS Passes** → **SDC Scheduling** → **CMT2** → **SystemVerilog**

The pipeline converts untimed software-like descriptions to timed hardware representations through optimization passes and scheduling.

## Quick Start

```bash
pixi run mlir tutorial/cadl/vgemv3d.cadl outputs/vgemv3d.mlir
pixi run opt outputs/vgemv3d.mlir outputs/vgemv3d_cmt.mlir
pixi run sv outputs/vgemv3d_cmt.mlir outputs/vgemv3d.sv
```

## Pipeline Stages

### 1. CADL Parsing

The CADL frontend parses hardware descriptions and generates MLIR using the APS dialect.

**Input:**
```cadl
#[opcode(7'b0101011)]
#[funct7(7'b0000000)]
rtype hello(rs1: u5, rs2: u5, rd: u5) {
  let a: u32 = _irf[rs1];
  let b: u32 = _irf[rs2];
  _irf[rd] = a + b;
}
```

**Output (MLIR):**
```mlir
func.func @hello(%rs1: i5, %rs2: i5, %rd: i5)
    attributes {opcode = 43 : i32, funct7 = 0 : i32} {
  %0 = aps.readrf %rs1 : i5 -> i32
  %1 = aps.readrf %rs2 : i5 -> i32
  %2 = arith.addi %0, %1 : i32
  aps.writerf %rd, %2 : i5, i32
  return
}
```

### 2. HLS Optimization Passes

| Pass | Purpose |
|------|---------|
| `memory-map` | Assign addresses to scratchpad memories |
| `raise-to-affine` | Convert SCF loops to Affine dialect |
| `hls-unroll` | Apply loop unrolling directives |
| `array-partition` | Partition arrays for parallel access |

### 3. SDC Scheduling

Uses System of Difference Constraints (SDC) to assign cycle times while respecting data dependencies, resource constraints, and timing.

### 4. CMT2 Elaboration & RTL Generation

Lowers scheduled MLIR to CMT2 dialect, then to FIRRTL and SystemVerilog via CIRCT.

## APS Dialect Operations

| Operation | Description |
|-----------|-------------|
| `aps.readrf` | Read from register file (GPR Req) |
| `aps.writerf` | Write to register file (Result Resp) |
| `aps.memload/memstore` | Scratchpad memory access |
| `aps.memburstload/memburststore` | Burst DMA transfer |

## Optimization Directives

### Loop Unrolling
```cadl
[[unroll(4)]]
with i: u32 = (0, i + 1) do { ... } while (i + 1 < 16);
```

### Array Partitioning
```cadl
#[partition_dim_array([0])]
#[partition_factor_array([4])]
#[partition_cyclic_array([1])]
static matrix: [i32; 16];
```

## File Locations

| Component | Location |
|-----------|----------|
| CADL Frontend | `cadl_frontend/` |
| APS Dialect | `lib/APS/`, `include/APS/` |
| TOR Dialect | `lib/TOR/`, `include/TOR/` |
| Scheduling | `lib/Schedule/`, `include/Schedule/` |

## Further Reading

- [cadl-language.md](cadl-language.md) - CADL language reference
- [hls-passes.md](hls-passes.md) - HLS optimization passes
- [scheduling.md](scheduling.md) - SDC scheduling algorithm
