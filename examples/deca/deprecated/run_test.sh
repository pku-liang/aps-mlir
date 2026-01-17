#!/bin/bash
# Automatic RISC-V Test Runner
# Usage: ./run_test.sh <test_file.c>

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CC="riscv64-unknown-elf-gcc"  # Use GCC instead of clang for easier linking
SPIKE="spike"
PK="pk"  # Let spike find pk automatically
MARCH="rv64imafdc"  # Include float/double support
MABI="lp64d"  # Double-precision floating-point ABI

# Include paths
INCLUDES=(
    "-I/home/share/zyy-riscv-toolchain/root/riscv32-unknown-elf/include"
    "-I/home/share/zyy-riscv-toolchain/root/include"
    "-I/home/cloud/aps-mlir-chipyard/tests"
)

# Compiler flags
CFLAGS=(
    "-march=${MARCH}"
    "-mabi=${MABI}"
    "${INCLUDES[@]}"
    "-O2"
    "-Wall"
    "-Wextra"
    "-std=gnu11"
    "-static"  # Static linking
)

# ============================================================================
# Functions
# ============================================================================

print_usage() {
    echo "Usage: $0 <test_file.c> [options]"
    echo ""
    echo "Options:"
    echo "  -r, --run-only      Run existing binary without recompiling"
    echo "  -c, --compile-only  Compile only, don't run"
    echo "  -d, --disasm        Generate disassembly"
    echo "  -t, --trace         Generate instruction trace"
    echo "  -k, --keep          Keep intermediate files"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 test_deca_e2e.c              # Compile and run"
    echo "  $0 test_gemm.c -d               # Compile, run, and disassemble"
    echo "  $0 test_deca_decompress.c -t    # Compile and run with trace"
}

error_exit() {
    echo -e "${RED}Error: $1${NC}" >&2
    exit 1
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

# ============================================================================
# Parse Arguments
# ============================================================================

if [ $# -eq 0 ]; then
    print_usage
    exit 1
fi

INPUT_FILE=""
RUN_ONLY=false
COMPILE_ONLY=false
DISASM=false
TRACE=false
KEEP_FILES=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            print_usage
            exit 0
            ;;
        -r|--run-only)
            RUN_ONLY=true
            shift
            ;;
        -c|--compile-only)
            COMPILE_ONLY=true
            shift
            ;;
        -d|--disasm)
            DISASM=true
            shift
            ;;
        -t|--trace)
            TRACE=true
            shift
            ;;
        -k|--keep)
            KEEP_FILES=true
            shift
            ;;
        *.c)
            INPUT_FILE="$1"
            shift
            ;;
        *)
            error_exit "Unknown option: $1"
            ;;
    esac
done

# ============================================================================
# Validation
# ============================================================================

if [ -z "$INPUT_FILE" ]; then
    error_exit "No input file specified"
fi

if [ ! -f "$INPUT_FILE" ]; then
    error_exit "File not found: $INPUT_FILE"
fi

# Generate output filenames
BASENAME=$(basename "$INPUT_FILE" .c)
OUTPUT_ELF="${BASENAME}.elf"
OUTPUT_DISASM="${BASENAME}.disasm"
OUTPUT_TRACE="${BASENAME}.trace"

# ============================================================================
# Compilation
# ============================================================================

if [ "$RUN_ONLY" = false ]; then
    info "Compiling: $INPUT_FILE"

    # Build the compile command
    COMPILE_CMD=("${CC}" "${CFLAGS[@]}" "$INPUT_FILE" "-o" "$OUTPUT_ELF")

    # Show the command (truncated for readability)
    echo -e "${YELLOW}$ ${CC} -march=${MARCH} ... $INPUT_FILE -o $OUTPUT_ELF${NC}"

    # Execute compilation
    if "${COMPILE_CMD[@]}" 2>&1; then
        success "Compilation successful: $OUTPUT_ELF"
    else
        error_exit "Compilation failed"
    fi

    # Generate disassembly if requested
    if [ "$DISASM" = true ]; then
        info "Generating disassembly..."
        if riscv64-unknown-elf-objdump -d "$OUTPUT_ELF" > "$OUTPUT_DISASM" 2>&1; then
            success "Disassembly saved to: $OUTPUT_DISASM"
        else
            warning "Could not generate disassembly (objdump not found?)"
        fi
    fi
fi

# ============================================================================
# Execution
# ============================================================================

if [ "$COMPILE_ONLY" = false ]; then
    # Check if binary exists
    if [ ! -f "$OUTPUT_ELF" ]; then
        error_exit "Binary not found: $OUTPUT_ELF (compile first or use -r with existing binary)"
    fi

    echo ""
    echo "========================================"
    info "Running: $OUTPUT_ELF"
    echo "========================================"
    echo ""

    # Determine spike command
    if [ "$TRACE" = true ]; then
        SPIKE_CMD=("${SPIKE}" "-l" "--log=${OUTPUT_TRACE}" "${PK}" "$OUTPUT_ELF")
        info "Running with instruction trace (output: $OUTPUT_TRACE)"
    else
        SPIKE_CMD=("${SPIKE}" "${PK}" "$OUTPUT_ELF")
    fi

    # Execute on Spike
    if "${SPIKE_CMD[@]}"; then
        EXIT_CODE=$?
        echo ""
        echo "========================================"
        if [ $EXIT_CODE -eq 0 ]; then
            success "Test completed successfully"
        else
            warning "Test exited with code: $EXIT_CODE"
        fi
        echo "========================================"

        if [ "$TRACE" = true ]; then
            success "Instruction trace saved to: $OUTPUT_TRACE"
        fi
    else
        EXIT_CODE=$?
        echo ""
        echo "========================================"
        error_exit "Test failed with exit code: $EXIT_CODE"
    fi
fi

# ============================================================================
# Cleanup
# ============================================================================

if [ "$KEEP_FILES" = false ] && [ "$COMPILE_ONLY" = false ]; then
    info "Cleaning up intermediate files..."
    rm -f "$OUTPUT_ELF"
    success "Cleanup complete"
fi

echo ""
info "Done!"
