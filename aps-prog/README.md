# APS-Prog

Programmer tool for on-board FPGA evaluation of the APS project with Rocket as the base processor. APS-Prog provides host utilities for loading and running programs on Rocket-based FPGA prototypes.

## Quick Start

### Build

```bash
make            # Build everything (cross-compile for ARM)
make clean      # Clean aps-prog
make clean-all  # Clean everything including FESVR
make help       # Show all targets

# Use a different toolchain
make CROSS_COMPILE=arm-linux-gnueabihf-
```

Default toolchain: `arm-xilinx-linux-gnueabi-`

### Run a Program

```bash
# Using UIO (fast, requires device tree setup)
./src/aps-prog +uio=/dev/uio0 your_program.riscv
```

## Project Structure

```
aps-prog/
├── Makefile           # Makefile
├── README.md          # This file
├── riscv-fesvr/       # RISC-V Frontend Server library (submodule)
└── src/
    ├── main.cc        # Main entry point
    ├── uio_htif.cc    # UIO-based HTIF implementation
    ├── uio_htif.h     # UIO HTIF header
    ├── Makefile       # Source build configuration
    └── README.md      # Detailed UIO HTIF documentation
```

## Related Projects

- [Rocket Chip](https://github.com/chipsalliance/rocket-chip) - RISC-V Rocket core
- [FESVR](https://github.com/riscvarchive/riscv-fesvr) - RISC-V Frontend Server

## License

- `riscv-fesvr/` - BSD 3-Clause License (UC Regents), see [riscv-fesvr/LICENSE](riscv-fesvr/LICENSE)
- `src/` - See parent directory LICENSE file
