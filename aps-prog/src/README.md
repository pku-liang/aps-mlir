# UIO MMIO-based HTIF for Rocket Core

This is a direct memory-mapped I/O version of the HTIF (Host-Target Interface) for communicating with Rocket cores on FPGA. Unlike the UART-TSI version which uses serial communication and the TSI protocol, this version directly accesses FPGA memory regions via Linux UIO (Userspace I/O) drivers.

## Architecture

```
uio_mmio (main)
    └── uio_htif_t (UIO memory mapping + address translation)
        └── htif_t (ELF loading + syscall handling)
            ├── load_elf() - Parse and load ELF files
            └── syscall_t - Proxy syscalls from Rocket
```

## Fixed Memory Layout

The implementation uses a **single UIO device** with a fixed memory layout:

```
UIO Offset         Size      Rocket Address    Description
─────────────────────────────────────────────────────────────
0x00000000         ~1GB      0x80000000        DRAM
0x3fffc000         8KB       0x1000            Boot ROM
0x3fffe000         8KB       0x2000000         CLINT (MSIP)
```

### Address Translation

The `rocket_addr_to_uio_offset()` function (uio_htif.cc:113-132) translates Rocket addresses to UIO offsets:

```cpp
// Rocket 0x80000000 (DRAM) -> UIO offset 0x0
// Rocket 0x1000     (Boot ROM) -> UIO offset 0x3fffc000
// Rocket 0x2000000  (CLINT) -> UIO offset 0x3fffe000
```

This allows the program to write to standard Rocket addresses (like DRAM at 0x80000000) which get transparently mapped to the correct UIO offsets.

## Building

### Prerequisites

1. Build FESVR library first:
```bash
cd ../riscv-fesvr
mkdir -p build
cd build
../configure
make
```

2. Build UIO MMIO:
```bash
cd ../uio_mmio_src
make
```

This produces the `uio_mmio` executable.

## Usage

### Minimal Usage

```bash
./uio_mmio +uio=/dev/uio0 program.riscv
```

### Command Line Arguments

**Required:**
- `+uio=/dev/uioX` - UIO device to map (e.g., `/dev/uio0`)

**Optional:**
- `+uio_size=SIZE` - Total UIO region size in bytes (default: `0x40000000` = 1GB)
- `+dram_size=SIZE` - DRAM size in bytes (default: `0x3fffc000` = 1GB - 16KB)

**FESVR Options (Inherited):**
- `+signature=FILE` - Write test signature to FILE (for riscv-tests)
- `+chroot=PATH` - Use PATH as root for syscall file operations
- `+permissive` / `+permissive-off` - Ignore unknown options

### Examples

#### Run a "Hello World" program:
```bash
./uio_mmio +uio=/dev/uio0 hello.riscv
```

#### Run with custom UIO size:
```bash
./uio_mmio +uio=/dev/uio0 +uio_size=0x40000000 program.riscv
```

#### Run without loading a binary (for memory testing):
```bash
./uio_mmio +uio=/dev/uio0 none
```

#### Run with signature file (for riscv-tests):
```bash
./uio_mmio +uio=/dev/uio0 +signature=test.sig test.riscv
```

## Setting Up UIO Device

### Device Tree Configuration

Configure your FPGA's device tree to expose the entire memory region via a single UIO device:

```dts
reserved-memory {
    #address-cells = <2>;
    #size-cells = <2>;
    ranges;

    rocket_mem: rocket_mem@c0000000 {
        compatible = "shared-dma-pool";
        reg = <0x0 0xc0000000 0x0 0x40000000>;  /* 1GB */
        no-map;
    };
};

rocket_uio {
    compatible = "generic-uio";
    reg = <0x0 0xc0000000 0x0 0x40000000>;  /* 1GB */
    status = "okay";
};
```

**Note:** The physical address in the device tree (e.g., `0xc0000000`) doesn't matter - the UIO driver will map it, and the program handles Rocket's address space internally.

### Verify UIO Device

```bash
ls -l /dev/uio0
cat /sys/class/uio/uio0/name
cat /sys/class/uio/uio0/maps/map0/addr
cat /sys/class/uio/uio0/maps/map0/size
```

## How It Works

### 1. ELF Loading

**Flow:**
```
main() → htif_t::run() → htif_t::start() → htif_t::load_program()
                                             ├── load_elf() (from FESVR)
                                             └── memif->write()
                                                  └── uio_htif_t::write_chunk()
                                                       ├── rocket_addr_to_uio_offset()
                                                       └── memcpy() to mmap'd UIO region
```

**uio_htif.cc:175-189** - `write_chunk()` implementation:
```cpp
void uio_htif_t::write_chunk(addr_t taddr, size_t len, const void* src) {
  size_t uio_offset = rocket_addr_to_uio_offset(taddr);  // Translate address
  memcpy((uint8_t*)uio_base + uio_offset, src, len);     // Direct write
  __sync_synchronize();                                  // Memory barrier
}
```

**Example:** Loading ELF segment to Rocket address `0x80000000`:
1. ELF loader calls `write_chunk(0x80000000, ...)`
2. `rocket_addr_to_uio_offset(0x80000000)` returns `0x0`
3. Data is written to UIO offset `0x0` (DRAM start)

### 2. Boot Sequence

After loading the ELF, `uio_htif_t::reset()` triggers execution:

**uio_htif.cc:207-216:**
```cpp
void uio_htif_t::reset() {
  uint32_t one = 1;
  addr_t msip_addr = ROCKET_CLINT_BASE;  // 0x2000000
  write_chunk(msip_addr, sizeof(uint32_t), &one);
  // Writes to UIO offset 0x3fffe000 (CLINT/MSIP)
}
```

This triggers a machine software interrupt on hart 0, causing Rocket to jump to the ELF entry point.

### 3. Syscall Handling

The program inherits HTIF's syscall polling from FESVR:

**htif.cc:184-200** - Main polling loop:
```cpp
while (!signal_exit && exitcode == 0) {
  if (auto tohost = mem.read_uint64(tohost_addr)) {     // Poll for syscalls
    mem.write_uint64(tohost_addr, 0);                   // Clear request
    command_t cmd(mem, tohost, fromhost_callback);
    device_list.handle_command(cmd);                    // Handle syscall
  }

  if (!fromhost_queue.empty() && mem.read_uint64(fromhost_addr) == 0) {
    mem.write_uint64(fromhost_addr, fromhost_queue.front());  // Send response
    fromhost_queue.pop();
  }
}
```

- `tohost`/`fromhost` addresses are extracted from ELF symbols
- `mem.read_uint64()` → `uio_htif_t::read_chunk()` → direct memory read from UIO
- `mem.write_uint64()` → `uio_htif_t::write_chunk()` → direct memory write to UIO

### 4. Supported Syscalls

All syscalls from FESVR are supported (implemented in `syscall.cc`):

| Syscall | Number | Description |
|---------|--------|-------------|
| exit    | 93     | Terminate program |
| read    | 63     | Read from file descriptor |
| write   | 64     | Write to file descriptor |
| openat  | 56     | Open file |
| close   | 57     | Close file descriptor |
| lseek   | 62     | Seek in file |
| fstat   | 80     | Get file status |
| And more... | | unlinkat, mkdirat, renameat, etc. |

**Example:** When Rocket executes a `write(1, buf, len)` syscall:
1. Rocket writes syscall info to `tohost` memory location
2. Host polls and reads `tohost` value
3. Host reads buffer data from Rocket's memory at address `buf`
4. Host executes actual `write()` on the host system
5. Host writes return value to `fromhost`
6. Rocket reads return value from `fromhost`

## Components Reused from FESVR

✓ **ELF Loading** - `load_elf()` in elfloader.cc
✓ **Syscall Handling** - Entire `syscall_t` class
✓ **HTIF Protocol** - tohost/fromhost polling loop
✓ **Device Framework** - device_list_t
✓ **Memory Interface** - memif_t abstraction (only replaced transport)

## Performance vs UART-TSI

| Operation | UART-TSI (@115200) | UIO MMIO |
|-----------|-------------------|----------|
| Load 1MB ELF | ~90 seconds | ~0.01 seconds |
| Write syscall | ~20ms | ~1μs |
| Read syscall | ~20ms | ~1μs |

**Why faster?**
- No UART serialization overhead
- No TSI protocol encoding/decoding
- Direct memory access via `memcpy()`
- No context switching (userspace only)

## Troubleshooting

### Permission Denied on /dev/uio*
```bash
sudo chmod 666 /dev/uio0
# Or add user to uio group
sudo usermod -a -G uio $USER
newgrp uio
```

### UIO Device Not Found
```bash
# Check if UIO driver is loaded
lsmod | grep uio

# Check device tree configuration
ls -l /sys/class/uio/
cat /sys/class/uio/uio0/name
```

### Address Out of Bounds Error
- Verify UIO size is large enough: `cat /sys/class/uio/uio0/maps/map0/size`
- Default is 1GB (`0x40000000`), adjust with `+uio_size=` if needed
- Check DRAM size doesn't exceed `UIO_BOOTROM_OFFSET` (0x3fffc000)

### Program Hangs
- Verify Rocket is running and accessible
- Check `tohost`/`fromhost` symbols exist in ELF: `nm program.riscv | grep host`
- Ensure FPGA configuration matches memory layout
- Try running with `none` to test UIO mapping without loading a program

### Compilation Errors
```bash
# Make sure FESVR is built first
cd ../riscv-fesvr/build && make
cd ../../uio_mmio_src && make clean && make
```

## Memory Layout Diagram

```
Rocket Address Space              UIO Offset
─────────────────────────────────────────────────

0x1000     ┌──────────────┐      0x3fffc000 ┌──────────────┐
           │  Boot ROM    │ ◄────────────────│  Boot ROM    │
0x3000     └──────────────┘      0x3fffe000 ├──────────────┤
                                             │              │
                                             │              │
0x2000000  ┌──────────────┐                  │    CLINT     │
           │    CLINT     │ ◄────────────────│    (MSIP)    │
0x2010000  └──────────────┘      0x40000000 └──────────────┘


0x80000000 ┌──────────────┐      0x00000000 ┌──────────────┐
           │              │ ◄────────────────│              │
           │              │                  │              │
           │     DRAM     │                  │     DRAM     │
           │              │                  │              │
           │              │                  │              │
0xbfffbfff └──────────────┘      0x3fffbfff └──────────────┘
```

## Files

- **uio_htif.h** - UIO HTIF class definition with fixed memory layout
- **uio_htif.cc** - Address translation and memory access implementation
- **uio_mmio_main.cc** - Main program entry point
- **Makefile** - Build configuration
- **README.md** - This file

## License

See parent directory LICENSE file.
