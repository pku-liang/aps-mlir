#include "uio_htif.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <signal.h>

void print_usage(const char* prog_name) {
  printf("MMIO-based HTIF for FPGA Rocket Core\n");
  printf("Usage: %s +uio=/dev/uioX [OPTIONS] <binary>\n\n", prog_name);
  printf("Fixed UIO Memory Layout:\n");
  printf("  UIO Offset 0x00000000 - 0x3fffbfff: DRAM   -> Rocket 0x80000000\n");
  printf("  UIO Offset 0x3fffc000 - 0x3fffdfff: BootROM -> Rocket 0x1000\n");
  printf("  UIO Offset 0x3fffe000 - 0x3fffffff: CLINT  -> Rocket 0x2000000\n\n");
  printf("Required Options:\n");
  printf("  +uio=/dev/uioX                    UIO device to map\n");
  printf("                                    Example: +uio=/dev/uio0\n\n");
  printf("Optional Options:\n");
  printf("  +uio_size=SIZE                    Total UIO size (default: 0x40000000 = 1GB)\n");
  printf("  +dram_size=SIZE                   DRAM size (default: 0x3fffc000)\n");
  printf("  none                              Skip loading binary (for testing)\n\n");
  printf("FESVR Options:\n");
  printf("  +permissive                       Ignore unknown options until +permissive-off\n");
  printf("  +permissive-off                   Stop ignoring unknown options\n");
  printf("  +signature=FILE                   Write test signature to FILE\n");
  printf("  +chroot=PATH                      Use PATH for syscall file operations\n\n");
  printf("Examples:\n");
  printf("  # Load and run a program (minimal):\n");
  printf("  %s +uio=/dev/uio0 hello.riscv\n\n", prog_name);
  printf("  # With custom UIO size:\n");
  printf("  %s +uio=/dev/uio0 +uio_size=0x40000000 program.riscv\n\n", prog_name);
  printf("  # Memory test without loading binary:\n");
  printf("  %s +uio=/dev/uio0 none\n\n", prog_name);
}

static volatile bool signal_exit = false;
static void handle_signal(int sig) {
  signal_exit = true;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    print_usage(argv[0]);
    return 1;
  }

  // Check for help flag
  for (int i = 1; i < argc; i++) {
    std::string arg(argv[i]);
    if (arg == "-h" || arg == "--help" || arg == "-help") {
      print_usage(argv[0]);
      return 0;
    }
  }

  // Setup signal handlers
  signal(SIGINT, handle_signal);
  signal(SIGTERM, handle_signal);

  try {
    fprintf(stderr, "Initializing UIO-based HTIF...\n");

    // Add permissive flags around plusargs for FESVR compatibility
    std::vector<std::string> args;
    for (int i = 0; i < argc; i++) {
      bool is_plusarg = (argv[i][0] == '+');
      if (is_plusarg) {
        args.push_back("+permissive");
        args.push_back(std::string(argv[i]));
        args.push_back("+permissive-off");
      } else {
        args.push_back(std::string(argv[i]));
      }
    }

    // Convert to C-style argv
    std::vector<char*> c_argv;
    for (auto& arg : args) {
      c_argv.push_back(const_cast<char*>(arg.c_str()));
    }

    // Create UIO HTIF instance
    uio_htif_t htif(c_argv.size(), c_argv.data());

    fprintf(stderr, "UIO HTIF initialized successfully\n");
    fprintf(stderr, "Starting execution...\n");

    // Run the HTIF (loads program, handles syscalls, etc.)
    int exit_code = htif.run();

    fprintf(stderr, "Program exited with code: %d\n", exit_code);
    return exit_code;

  } catch (const std::exception& e) {
    fprintf(stderr, "Error: %s\n", e.what());
    return 1;
  }
}
