# Processor Integration & Architecture

The APS architecture component handles integration of synthesized ISAXs into RISC-V processors, including APS-Itfc abstraction, DMA engine, memory system, and deployment flows.

## Key Components

### 1. APS-Itfc (Interface Abstraction)

A unified interface abstracting how ISAXs interact with different RISC-V platforms.

**Supported Transactions**:

| Transaction | Direction | Description |
|-------------|-----------|-------------|
| Issue Req | Core -> ISAX | Start new ISAX execution |
| GPR Req | Core -> ISAX | Receive source operands |
| Rd/WrMem Req/Resp | ISAX <-> Memory | Single memory access |
| RdBlkMem/WrBlkMem | ISAX <-> DMA | Block memory transfer |
| Result Resp | ISAX -> Core | Return result to register file |

See [aps-itfc.md](aps-itfc.md) for details.

### 2. DMA Engine

Accelerates batch data transfers between CPU memory and ISAX scratchpad:
- **Burst transfers**: Up to cache-line sized blocks per transaction
- **Dual channel**: Overlapped request/response for higher bandwidth
- **Layout transformation**: Direct tiling during transfer

See [dma-engine.md](dma-engine.md) for details.

### 3. Memory Pool

Partitioned scratchpad memory for ISAX computation:
- **Bank partitioning**: Cyclic or block partitioning
- **Multi-port access**: Parallel reads/writes to different banks
- **Address mapping**: DMA-compatible address space

### 4. RoCC Adaptor

Maps APS-Itfc to Rocket Custom Coprocessor (RoCC) interface.

## Quick Start

### Configure Integration

Edit `aps_config.json`:

```json
{
    "backend": "rocc",
    "arch": "rv32",
    "vsrc": ["wrapper_dma_trig.sv", "tl_dma_2ch_v2.sv", "v3ddist_vv.sv"],
    "maxBurstBytes": 128,
    "nXacts": 2
}
```

### Run RTL Simulation

```bash
cd $APS_CHIPYARD/sims/verilator
make CONFIG=APSRocketConfig -j4
make run-binary-debug BINARY=outputs/test.riscv LOADMEM=1
```

### FPGA Deployment

```bash
cd $APS_CHIPYARD/nextvlsi
make vivado CONFIG=APSRocketConfig
make bitstream
```

See [deployment.md](deployment.md) for full deployment guide.

## Supported Platforms

| Platform | Interface | Status |
|----------|-----------|--------|
| Rocket | RoCC | Supported |
| BOOM | RoCC | Supported |
| CVA6 | CV-X-IF | In Progress |
| CV32E | CV-X-IF | In Progress |

| FPGA Board | Device | Status |
|------------|--------|--------|
| ZC706 | XC7Z045 | Supported |
| VCU118 | XCVU9P | In Progress |

## Design Flow

### ASIC Flow

```bash
# 1. Synthesize ISAX to RTL
pixi run mlir tutorial/cadl/v3ddist_vv.cadl outputs/v3ddist_vv.mlir
pixi run opt outputs/v3ddist_vv.mlir outputs/v3ddist_vv_cmt.mlir
pixi run sv outputs/v3ddist_vv_cmt.mlir outputs/v3ddist_vv.sv

# 2. Logic synthesis with Yosys
cd $APS_CHIPYARD/nextvlsi
make yosys CONFIG=APSRocketConfig

# 3. Place and route with OpenROAD
make pnr CONFIG=APSRocketConfig
```

### FPGA Flow

```bash
cd $APS_CHIPYARD
make verilog CONFIG=APSRocketConfig
vivado -mode batch -source scripts/run_vivado.tcl
./aps_prog +uio=/dev/uio0 test.riscv
```

## Performance Example: 3D Point Cloud

Compute squared Euclidean distances between 16 point pairs:

| Implementation | Cycles |
|---------------|--------|
| Software (no ISAX) | 688 |
| Hardware (with ISAX) | 166 |
| **Speedup** | **4.14x** |

## Further Reading

- [aps-itfc.md](aps-itfc.md) - Detailed interface specification
- [dma-engine.md](dma-engine.md) - DMA engine and memory system
- [deployment.md](deployment.md) - FPGA deployment guide
