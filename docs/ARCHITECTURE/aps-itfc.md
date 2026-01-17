# APS-Itfc: Interface Abstraction

APS-Itfc is a **transactional abstraction** that models how ISAXs interact with RISC-V processors. It provides a unified interface that can be mapped to different processor platforms (Rocket/RoCC, CV-X-IF, etc.).

## Motivation

Different RISC-V platforms provide different ISAX interfaces:

| Platform | Interface | Characteristics |
|----------|-----------|-----------------
| Rocket/BOOM | RoCC | R-type instruction + L1D$ access |
| CVA6 | CV-X-IF | Flexible instruction extension interface |
| CV32E | CV-X-IF | Minimal extension interface |

**Problem**: ISAX designs are tightly coupled to platform interfaces.

**Solution**: APS-Itfc provides an abstraction layer that defines a common transaction model and maps to platform-specific interfaces via adaptors.

## Transaction Types

### Issue Req
**Direction**: RISC-V Core -> ISAX. Start new ISAX execution with instruction opcode, funct7/funct3 fields, and register specifiers.

### GPR Req
**Direction**: RISC-V Core -> ISAX. Receive source operand values from register file.

```cadl
let a: u32 = _irf[rs1];    // GPR Req for rs1
let b: u32 = _irf[rs2];    // GPR Req for rs2
```

### RdMem/WrMem Req/Resp
**Direction**: ISAX <-> Memory System. Single-element memory read/write.

```cadl
let data: u32 = _mem[addr];    // RdMem Req/Resp
_mem[addr] = data;             // WrMem Req/Resp
```

### RdBlkMem/WrBlkMem Req/Resp
**Direction**: ISAX <-> DMA Engine. Burst memory read/write (multiple elements).

```cadl
matrix[0 +: ] = _burst_read[addr +: 16];      // 16-element burst read
_burst_write[addr +: 16] = result[0 +: ];     // 16-element burst write
```

### Result Resp
**Direction**: ISAX -> RISC-V Core. Return computation result to register file.

```cadl
_irf[rd] = result;    // Result Resp
```

## Platform Mappings

### RoCC (Rocket Custom Coprocessor)

| APS-Itfc | RoCC Interface |
|----------|---------------|
| Issue Req | rocc.cmd (instruction dispatch) |
| GPR Req | rocc.cmd.rs1, rocc.cmd.rs2 |
| RdMem Req/Resp | rocc.mem.req/resp |
| RdBlkMem/WrBlkMem | TileLink (via DMA engine) |
| Result Resp | rocc.resp |

## Memory Access Modes

### Single-Element Access (via L1D$)
**Latency**: Cache hit ~2 cycles, miss varies.

### Burst Access (via DMA)
**Latency**: ~15 cycles initial, then 1 word/cycle.

## APS Dialect Operations

| APS Dialect Op | APS-Itfc Transaction |
|----------------|---------------------|
| `aps.readrf` | GPR Req |
| `aps.writerf` | Result Resp |
| `aps.memload/memstore` | Scratchpad access |
| `aps.memburstload/memburststore` | DMA transfer |
| `aps.cpumemload/cpumemstore` | CPU memory access |

## Timing Characteristics (RoCC on Rocket)

| Transaction | Typical Latency |
|-------------|-----------------|
| Issue Req | 1 cycle |
| GPR Req | 0 cycles (included in Issue) |
| RdMem/WrMem | 2+ cycles (L1D$ dependent) |
| RdBlkMem Req | ~15 cycles initial |
| RdBlkMem Resp | 1 word/cycle after initial |
| Result Resp | 1 cycle |

## Protocol Constraints

**Ordering**:
1. Issue before GPR
2. GPR before Memory (for operands used in memory access)
3. Memory ordering within same address (writes before reads for WAR hazard)
4. Result last (after all computation completes)

**Concurrency**:
- Multiple outstanding memory requests allowed (with reorder buffer)
- Single Issue Req at a time per ISAX instance
- Burst operations are atomic

**Flow Control**: Back-pressure via ready/valid handshaking.

## Extending to New Platforms

1. **Define Platform Mapping**: Map each APS-Itfc transaction to platform signals
2. **Implement Adaptor**: Create adaptor module translating between APS-Itfc and platform
3. **Register with CMT2**: Add platform adaptor to CMT2 module library

## Further Reading

- [dma-engine.md](dma-engine.md) - DMA engine details
- [deployment.md](deployment.md) - Platform deployment
