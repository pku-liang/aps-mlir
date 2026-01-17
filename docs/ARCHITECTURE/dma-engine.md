# DMA Engine

The DMA (Direct Memory Access) engine accelerates batch data transfers between CPU memory and ISAX scratchpad memory, bypassing the single-element access path through L1D cache.

## Why DMA?

### Problem: L1D$ Bottleneck

RoCC's single-element memory access is slow for vector/matrix operations: ~2 cycles per element via L1D$.

**Total for N elements**: ~2N cycles

### Solution: Burst DMA

Burst access via TileLink: ~15 cycles initial latency, then 1 word per cycle.

**Total for N elements**: ~15 + N cycles

**Speedup**: For N=64 elements: 128 cycles -> 79 cycles (~1.6x)

## Dual-Channel Optimization

Single channel has gap between requests. Dual channel overlaps preparation with transfer.

**Single Channel**: 2*(N + 15) cycles
**Dual Channel**: 2*N + 15 cycles (saves 15 cycles!)

## TileLink Protocol

The DMA engine uses **TileLink-UH** (Uncached Heavyweight) for burst transfers.

| Channel | Direction | Purpose |
|---------|-----------|---------|
| Channel A | Master -> Slave | Request messages |
| Channel D | Slave -> Master | Response messages |

**Get (Read)**: Channel A sends Get(address, size), Channel D returns AccessAckData.
**Put (Write)**: Channel A sends PutFullData, Channel D returns AccessAck.

## Timing

**Read**: ~15 cycles initial latency, then 1 word/cycle.
**Write**: 1 word/cycle, completion ~5 cycles after last data.
**Maximum Burst Size**: Limited by D-cache line size (typically 64 bytes).

## Address Space Mapping

Each ISAX's scratchpad memory is mapped to an address range:

```mlir
aps.memorymap {
    aps.mem_entry "mat" : banks([@mat_0, @mat_1, @mat_2, @mat_3]),
        base(0), size(64), count(4), cyclic(1)
    aps.mem_entry "vec" : banks([@vec]), base(64), size(16), count(1), cyclic(0)
}
```

## Multi-Bank Write

64-bit bus writes to multiple banks based on partition policy.

For cyclic partitioning with factor F:
- `bank = element_idx % F`
- `bank_offset = element_idx / F`

## Layout Transformation

The DMA engine can apply layout transformations during transfer.

**CADL Directive**:
```cadl
[[stride_x(2)]]
[[stride_y(2)]]
mat[sa +: ] = _burst_read[da +: 8];
```

## Configuration

### aps_config.json Parameters

```json
{
    "maxBurstBytes": 128,     // Maximum bytes per burst
    "nXacts": 2,              // Number of DMA channels
    "burstAlignment": 64      // Alignment requirement
}
```

### CADL Burst Directives

```cadl
array[0 +: ] = _burst_read[addr +: 16];           // Basic burst read
_burst_write[addr +: 16] = array[0 +: ];          // Burst write
```

## Integration with Memory Pool

**Bank Selection**: During DMA write, follows partition policy.

**Concurrent Access**:
- Different banks: No conflict
- Same bank: Arbiter handles priority (ISAX computation has priority)

**Double Buffering**: For streaming workloads, use double buffering to overlap DMA with computation.

## Performance Optimization

1. **Maximize Burst Length**: Larger bursts amortize initial latency
2. **Use Both Channels**: Overlap channel 0 transfer with channel 1 setup
3. **Align Addresses**: Misaligned accesses may require multiple transactions
4. **Match Partition Factor**: Ensure burst size is multiple of partition factor

## Debugging

**Key signals to monitor**:
- `dma_ch0_req_valid/ready` - Channel 0 request handshake
- `dma_ch0_resp_valid/ready` - Channel 0 response handshake
- `tl_a_valid/ready` - TileLink request
- `tl_d_valid/ready` - TileLink response

**Common Issues**:
- **Bus Contention**: High stall count, schedule DMA during low bus activity
- **Bank Conflict**: DMA stalls during write, adjust access pattern or use double buffering
- **Alignment Fault**: Transfer splits, align to burst boundary

## Further Reading

- [aps-itfc.md](aps-itfc.md) - Interface abstraction
- [deployment.md](deployment.md) - System deployment
