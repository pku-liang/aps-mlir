# APS-Prog Unified Build System

.PHONY: all fesvr aps-prog clean clean-all install help

# Cross-compiler toolchain
CROSS_COMPILE ?= arm-xilinx-linux-gnueabi-
HOST_TRIPLE := arm-xilinx-linux-gnueabi

# Directories
ROOT_DIR := $(shell pwd)
FESVR_DIR := $(ROOT_DIR)/riscv-fesvr
FESVR_BUILD := $(FESVR_DIR)/build
SRC_DIR := $(ROOT_DIR)/src

# Output
FESVR_LIB := $(FESVR_BUILD)/libfesvr.a
APS_PROG := $(SRC_DIR)/aps-prog

# Default target
all: aps-prog

# Build FESVR library (cross-compiled)
fesvr: $(FESVR_LIB)

$(FESVR_LIB):
	@echo "=== Building FESVR library ($(HOST_TRIPLE)) ==="
	@mkdir -p $(FESVR_BUILD)
	@cd $(FESVR_BUILD) && \
		CC=$(CROSS_COMPILE)gcc \
		CXX=$(CROSS_COMPILE)g++ \
		../configure --host=$(HOST_TRIPLE) && \
		$(MAKE)

# Build aps-prog (depends on fesvr)
aps-prog: $(FESVR_LIB)
	@echo "=== Building aps-prog ==="
	@$(MAKE) -C $(SRC_DIR) CROSS_COMPILE=$(CROSS_COMPILE)

# Clean only aps-prog
clean:
	@echo "=== Cleaning aps-prog ==="
	@$(MAKE) -C $(SRC_DIR) clean

# Clean everything including fesvr
clean-all: clean
	@echo "=== Cleaning FESVR ==="
	@rm -rf $(FESVR_BUILD)

# Install to /usr/local/bin (optional)
install: aps-prog
	@echo "=== Installing aps-prog ==="
	@install -m 755 $(APS_PROG) /usr/local/bin/

# Help
help:
	@echo "APS-Prog Build System (cross-compile for ARM)"
	@echo ""
	@echo "Toolchain: $(CROSS_COMPILE)"
	@echo ""
	@echo "Targets:"
	@echo "  all        - Build everything (default)"
	@echo "  fesvr      - Build FESVR library only"
	@echo "  aps-prog   - Build aps-prog executable"
	@echo "  clean      - Clean aps-prog build"
	@echo "  clean-all  - Clean everything including FESVR"
	@echo "  install    - Install aps-prog to /usr/local/bin"
	@echo "  help       - Show this help"
	@echo ""
	@echo "Override toolchain: make CROSS_COMPILE=arm-linux-gnueabihf-"
