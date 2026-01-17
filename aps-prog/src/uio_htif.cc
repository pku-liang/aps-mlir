#include "uio_htif.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <errno.h>

uio_htif_t::uio_htif_t(int argc, char** argv)
  : htif_t(argc, argv),
    uio_base(nullptr),
    uio_size(0),
    uio_fd(-1),
    dram_size(0)
{
  parse_uio_args(argc, argv);
}

uio_htif_t::~uio_htif_t()
{
  unmap_uio();
}

void uio_htif_t::parse_uio_args(int argc, char** argv)
{
  std::string uio_device;
  size_t total_size = 0;

  for (int i = 1; i < argc; i++) {
    std::string arg(argv[i]);

    if (arg.find("+uio=") == 0) {
      // Format: +uio=/dev/uio0
      uio_device = arg.substr(5);
    }
    else if (arg.find("+uio_size=") == 0) {
      // Format: +uio_size=0x40000000 (size in hex or decimal)
      total_size = strtoull(arg.substr(10).c_str(), 0, 0);
    }
    else if (arg.find("+dram_size=") == 0) {
      // Format: +dram_size=0x3fffc000
      dram_size = strtoull(arg.substr(11).c_str(), 0, 0);
    }
  }

  // Default sizes if not specified
  if (total_size == 0) {
    total_size = 0x40000000;  // Default 1GB
    fprintf(stderr, "Using default UIO size: 0x%lx (1GB)\n", total_size);
  }

  if (dram_size == 0) {
    dram_size = UIO_BOOTROM_OFFSET;  // DRAM goes up to boot ROM
    fprintf(stderr, "Using default DRAM size: 0x%lx\n", dram_size);
  }

  if (uio_device.empty()) {
    throw std::invalid_argument("Must specify +uio=/dev/uioX");
  }

  if (!map_uio_device(uio_device, total_size)) {
    throw std::runtime_error("Failed to map UIO device");
  }

  fprintf(stderr, "UIO Memory Map:\n");
  fprintf(stderr, "  DRAM:     UIO offset 0x%08x - 0x%08lx -> Rocket 0x%08x\n",
          UIO_DRAM_OFFSET, dram_size, ROCKET_DRAM_BASE);
  fprintf(stderr, "  Boot ROM: UIO offset 0x%08x               -> Rocket 0x%08x\n",
          UIO_BOOTROM_OFFSET, ROCKET_BOOTROM_BASE);
  fprintf(stderr, "  CLINT:    UIO offset 0x%08x               -> Rocket 0x%08x\n",
          UIO_CLINT_OFFSET, ROCKET_CLINT_BASE);
}

bool uio_htif_t::map_uio_device(const std::string& uio_device, size_t size)
{
  // Open UIO device
  uio_fd = open(uio_device.c_str(), O_RDWR | O_SYNC);
  if (uio_fd < 0) {
    fprintf(stderr, "Error opening %s: %s\n", uio_device.c_str(), strerror(errno));
    return false;
  }

  // Map the entire UIO region
  uio_base = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, uio_fd, 0);
  if (uio_base == MAP_FAILED) {
    fprintf(stderr, "Error mapping %s: %s\n", uio_device.c_str(), strerror(errno));
    close(uio_fd);
    uio_fd = -1;
    return false;
  }

  uio_size = size;
  fprintf(stderr, "Mapped UIO device %s: %p, size 0x%lx\n",
          uio_device.c_str(), uio_base, uio_size);

  return true;
}

void uio_htif_t::unmap_uio()
{
  if (uio_base != nullptr && uio_base != MAP_FAILED) {
    munmap(uio_base, uio_size);
    uio_base = nullptr;
  }
  if (uio_fd >= 0) {
    close(uio_fd);
    uio_fd = -1;
  }
}

size_t uio_htif_t::rocket_addr_to_uio_offset(addr_t rocket_addr)
{
  // Boot ROM: Rocket 0x1000 -> UIO 0x3fffc000
  if (rocket_addr >= ROCKET_BOOTROM_BASE && rocket_addr < ROCKET_BOOTROM_BASE + 0x2000) {
    return UIO_BOOTROM_OFFSET + (rocket_addr - ROCKET_BOOTROM_BASE);
  }

  // CLINT: Rocket 0x2000000 -> UIO 0x3fffe000
  if (rocket_addr >= ROCKET_CLINT_BASE && rocket_addr < ROCKET_CLINT_BASE + 0x10000) {
    return UIO_CLINT_OFFSET + (rocket_addr - ROCKET_CLINT_BASE);
  }

  // DRAM: Rocket 0x80000000+ -> UIO 0x0+
  if (rocket_addr >= ROCKET_DRAM_BASE) {
    return UIO_DRAM_OFFSET + (rocket_addr - ROCKET_DRAM_BASE);
  }

  fprintf(stderr, "Warning: Unmapped Rocket address 0x%lx\n", rocket_addr);
  return 0;
}

bool uio_htif_t::is_valid_address(addr_t rocket_addr, size_t len)
{
  size_t uio_offset = rocket_addr_to_uio_offset(rocket_addr);

  // Check if access would exceed UIO bounds
  if (uio_offset + len > uio_size) {
    return false;
  }

  // Check specific region bounds
  if (rocket_addr >= ROCKET_BOOTROM_BASE && rocket_addr < ROCKET_BOOTROM_BASE + 0x2000) {
    // Boot ROM region (8KB)
    return (rocket_addr + len) <= (ROCKET_BOOTROM_BASE + 0x2000);
  }

  if (rocket_addr >= ROCKET_CLINT_BASE && rocket_addr < ROCKET_CLINT_BASE + 0x10000) {
    // CLINT region (64KB)
    return (rocket_addr + len) <= (ROCKET_CLINT_BASE + 0x10000);
  }

  if (rocket_addr >= ROCKET_DRAM_BASE) {
    // DRAM region
    return (rocket_addr - ROCKET_DRAM_BASE + len) <= dram_size;
  }

  return false;
}

void uio_htif_t::read_chunk(addr_t taddr, size_t len, void* dst)
{
  if (!is_valid_address(taddr, len)) {
    fprintf(stderr, "Error: read_chunk address 0x%lx len 0x%lx out of bounds\n", taddr, len);
    throw std::runtime_error("Read address out of bounds");
  }

  size_t uio_offset = rocket_addr_to_uio_offset(taddr);

  // Direct memory copy from UIO mapped region
  memcpy(dst, (uint8_t*)uio_base + uio_offset, len);
}

void uio_htif_t::write_chunk(addr_t taddr, size_t len, const void* src)
{
  if (!is_valid_address(taddr, len)) {
    fprintf(stderr, "Error: write_chunk address 0x%lx len 0x%lx out of bounds\n", taddr, len);
    throw std::runtime_error("Write address out of bounds");
  }

  size_t uio_offset = rocket_addr_to_uio_offset(taddr);

  // Direct memory copy to UIO mapped region
  memcpy((uint8_t*)uio_base + uio_offset, src, len);

  // Ensure write completes (memory barrier)
  __sync_synchronize();
}

void uio_htif_t::clear_chunk(addr_t taddr, size_t len)
{
  if (!is_valid_address(taddr, len)) {
    fprintf(stderr, "Error: clear_chunk address 0x%lx len 0x%lx out of bounds\n", taddr, len);
    throw std::runtime_error("Clear address out of bounds");
  }

  size_t uio_offset = rocket_addr_to_uio_offset(taddr);

  // Clear memory region
  memset((uint8_t*)uio_base + uio_offset, 0, len);

  // Ensure write completes
  __sync_synchronize();
}

void uio_htif_t::reset()
{
  // Write 1 to MSIP register (offset 0 in CLINT) to trigger interrupt on hart 0
  uint32_t one = 1;
  addr_t msip_addr = ROCKET_CLINT_BASE;

  fprintf(stderr, "Triggering MSIP at Rocket addr 0x%lx (UIO offset 0x%lx) to start execution\n",
          msip_addr, rocket_addr_to_uio_offset(msip_addr));
  write_chunk(msip_addr, sizeof(uint32_t), &one);
}
