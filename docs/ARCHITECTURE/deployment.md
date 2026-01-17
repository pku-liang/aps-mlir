# Deployment Guide

Deploying APS-based RISC-V systems to FPGA and ASIC targets.

## Prerequisites

**Software**: Pixi, Vivado 2024.1+, Yosys + OpenROAD, RISC-V Toolchain

**FPGA Hardware**: ZC706 (XC7Z045) - 218k LUTs, 437k FFs, 19.2Mb BRAM

## RTL Simulation

### Setup Verilator Simulation

```bash
cd $APS_CHIPYARD/sims/verilator
make CONFIG=APSRocketConfig -j4
```

### Run Simulation

```bash
# Baseline (without ISAX)
make run-binary-debug BINARY=$APS/tutorial/outputs/v3ddist_vv_native.riscv LOADMEM=1

# With ISAX
make run-binary-debug BINARY=$APS/tutorial/outputs/v3ddist_vv.riscv LOADMEM=1
```

## ASIC Flow (Yosys + OpenROAD)

**Technology**: IHP130 (open-source 130nm) with sg13g2 standard cells.

### Run Logic Synthesis

```bash
cd $APS_CHIPYARD/nextvlsi
make yosys CONFIG=APSRocketConfig
```

### View Synthesis Report

```bash
cd $APS_CHIPYARD/nextvlsi/yosys
python simplify_result.py
```

### Run Place and Route

```bash
make pnr CONFIG=APSRocketConfig
```

Note: PnR is slow (~hours). For quick iteration, use synthesis only.

## FPGA Flow (Vivado)

### Step 1: Generate Verilog

```bash
cd $APS_CHIPYARD
make verilog CONFIG=APSRocketConfig
```

### Step 2: Create Vivado Project

```bash
cd $APS_CHIPYARD/fpga/vivado
vivado -mode batch -source create_project.tcl
```

### Step 3: Add ISAX Sources

Update `aps_config.json`:

```json
{
    "vsrc": ["wrapper_dma_trig.sv", "tl_dma_2ch_v2.sv", "your_isax.sv"]
}
```

### Step 4: Run Synthesis and Implementation

```bash
vivado -mode batch -source run_impl.tcl
```

### Step 5: Generate Boot Image

```bash
cd $APS_CHIPYARD/fpga
petalinux-package --boot --fsbl images/zynq_fsbl.elf --fpga impl/top.bit --u-boot -o BOOT.BIN
```

### Step 6: Prepare SD Card

Partition 1 (FAT32): BOOT.BIN, boot.scr, image.ub
Partition 2 (ext4): Root filesystem

### Step 7: Program and Run

1. Insert SD card, set boot mode switches for SD boot
2. Power on, connect via UART (115200 baud)
3. Run: `sudo ./aps_prog +uio=/dev/uio0 ~/test.riscv`

## APS-Prog Tool

```bash
./aps_prog +uio=/dev/uioX [OPTIONS] <binary>
```

**Options**:
- `+uio=/dev/uioX` - UIO device for PS-PL communication
- `+verbose` - Enable verbose output
- `+timeout=N` - Set execution timeout (ms)

## Customization

### Changing ISAX Configuration

Edit `$APS_CHIPYARD/aps_config.json`:

```json
{
    "vsrc": ["new_isax.sv"],
    "maxBurstBytes": 256,
    "nXacts": 4
}
```

### Performance Measurement

```c
static inline uint64_t read_cycles() {
    uint64_t cycles;
    asm volatile ("rdcycle %0" : "=r"(cycles));
    return cycles;
}
```

## Troubleshooting

**Simulation Hangs**: Add timeout, check for infinite loops.

**FPGA Boot Failure**: Check boot mode switches, reformat SD, check Vivado logs.

**ISAX Not Executing**: Check opcode/funct7 encoding, verify integration.

**DMA Transfer Failure**: Align addresses, verify memory map.

**Timing Violations**: Reduce frequency, pipeline ISAX logic.

## Example: Complete Deployment

```bash
# 1. Synthesize ISAX
bash a1-ex1synth.sh

# 2. Compile test program
bash a2-ex1compile.sh

# 3. Verify in simulation
bash a5-ex1nativesim.sh  # Baseline
bash a6-ex1sim.sh        # With ISAX

# 4. Run ASIC synthesis
bash a7-ex1yosys.sh

# 5. Deploy to FPGA (update aps_config.json, run Vivado, generate boot image)
sudo ./aps_prog +uio=/dev/uio0 test.riscv
```

## Further Reading

- [aps-itfc.md](aps-itfc.md) - Interface details
- [dma-engine.md](dma-engine.md) - DMA configuration
- [Chipyard Documentation](https://chipyard.readthedocs.io/) - SoC framework
