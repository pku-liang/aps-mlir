.PHONY: default clean

default: aps-prog

# Cross-compiler toolchain
CROSS_COMPILE ?= arm-xilinx-linux-gnueabi-
CXX = $(CROSS_COMPILE)g++

# Source files
SRCS = uio_htif.cc main.cc

# Include directories
INCLUDES = -I ../riscv-fesvr

# Compiler flags
CXXFLAGS = -O3 -std=c++17 -Wall

# Libraries
FESVR_LIB = ../riscv-fesvr/build/libfesvr.a
LIBS = $(FESVR_LIB) -lpthread

# Build target
aps-prog: $(SRCS) $(FESVR_LIB)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $(SRCS) $(LIBS)

# Clean
clean:
	rm -rf aps-prog
