#ifndef __UIO_HTIF_H
#define __UIO_HTIF_H

#include <fesvr/htif.h>
#include <fesvr/memif.h>
#include <string>
#include <stdint.h>

// Fixed UIO memory layout offsets
#define UIO_DRAM_OFFSET       0x00000000
#define UIO_BOOTROM_OFFSET    0x3fffc000
#define UIO_CLINT_OFFSET      0x3fffe000

// Rocket address space mapping
#define ROCKET_BOOTROM_BASE   0x1000
#define ROCKET_CLINT_BASE     0x2000000
#define ROCKET_DRAM_BASE      0x80000000

class uio_htif_t : public htif_t
{
public:
  uio_htif_t(int argc, char** argv);
  virtual ~uio_htif_t();

  // Map the UIO device
  bool map_uio_device(const std::string& uio_device, size_t size);

  // Unmap UIO device
  void unmap_uio();

protected:
  void reset() override;
  void read_chunk(addr_t taddr, size_t len, void* dst) override;
  void write_chunk(addr_t taddr, size_t len, const void* src) override;
  void clear_chunk(addr_t taddr, size_t len) override;

  size_t chunk_align() override { return 8; }
  size_t chunk_max_size() override { return 1024 * 1024; } // 1MB chunks

private:
  // UIO mapping
  void* uio_base;       // Base virtual address of UIO mapping
  size_t uio_size;      // Size of UIO mapping
  int uio_fd;           // UIO file descriptor

  // DRAM configuration
  size_t dram_size;     // Size of DRAM region

  // Convert Rocket address to UIO offset
  size_t rocket_addr_to_uio_offset(addr_t rocket_addr);

  // Check if address is valid
  bool is_valid_address(addr_t rocket_addr, size_t len);

  // Parse command line arguments for UIO configuration
  void parse_uio_args(int argc, char** argv);
};

#endif // __UIO_HTIF_H
