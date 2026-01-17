# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = ../xpack-riscv-none-elf-gcc-15.2.0-1/bin/riscv-none-elf-gcc

# bare metal version with embedded weights (no filesystem)
.PHONY: runq_bm
runq_bm: runq_bm.c model_weights.h tokenizer_data.h tinyalloc/tinyalloc.c
	$(CC) -O3 -o runq_bm runq_bm.c tinyalloc/tinyalloc.c -lm -mabi=lp64f -march=rv64imacf_zicsr

# bare metal version for Rocket core (rv32, htif)
CC_ROCKET = riscv32-unknown-elf-gcc
ARCH_ROCKET = rv32imaf_zicsr_zifencei
ABI_ROCKET = ilp32f
CMODEL_ROCKET = medany
SPECS_ROCKET = htif_nano.specs
LINKER_ROCKET = htif.ld

.PHONY: runq_bm_rocket
runq_bm_rocket: runq_bm_rocc.c runq_bm.c model_weights.h tokenizer_data.h tinyalloc/tinyalloc.c
	$(CC_ROCKET) -std=gnu99 -O3 -Wall -Wextra -fno-common -fno-builtin-printf \
		-march=$(ARCH_ROCKET) -mabi=$(ABI_ROCKET) -mcmodel=$(CMODEL_ROCKET) \
		-specs=$(SPECS_ROCKET) \
		-static -T $(LINKER_ROCKET) \
		-o runq_bm_rocket.riscv runq_bm_rocc.c tinyalloc/tinyalloc.c -lm
	$(CC_ROCKET) -std=gnu99 -O3 -Wall -Wextra -fno-common -fno-builtin-printf \
		-march=$(ARCH_ROCKET) -mabi=$(ABI_ROCKET) -mcmodel=$(CMODEL_ROCKET) \
		-specs=$(SPECS_ROCKET) \
		-static -T $(LINKER_ROCKET) \
		-o runq_bm_rocket_sw.riscv runq_bm.c tinyalloc/tinyalloc.c -lm

# the most basic way of building that is most likely to work on most systems
# We use this to compile for riscv-pk simulator
.PHONY: run
run: runq.c
# 	$(CC) -O3 -o run run.c -lm
	$(CC) -O3 -o runq runq.c -lm -mabi=lp64f -march=rv64imacf_zicsr

# generate header files from binary data
model_weights.h: stories15Mq.bin bin2header.py
	python3 bin2header.py stories15Mq.bin model_weights.h

tokenizer_data.h: tokenizer.bin tokenizer2header.py
	python3 tokenizer2header.py tokenizer.bin tokenizer_data.h
# useful for a debug build, can then e.g. analyze with valgrind, example:
# $ valgrind --leak-check=full ./run out/model.bin -n 3
rundebug: run.c
	$(CC) -g -o run run.c -lm
	$(CC) -g -o runq runq.c -lm

# https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
# https://simonbyrne.github.io/notes/fastmath/
# -Ofast enables all -O3 optimizations.
# Disregards strict standards compliance.
# It also enables optimizations that are not valid for all standard-compliant programs.
# It turns on -ffast-math, -fallow-store-data-races and the Fortran-specific
# -fstack-arrays, unless -fmax-stack-var-size is specified, and -fno-protect-parens.
# It turns off -fsemantic-interposition.
# In our specific application this is *probably* okay to use
.PHONY: runfast
runfast: run.c
	$(CC) -Ofast -o run run.c -lm
	$(CC) -Ofast -o runq runq.c -lm

# additionally compiles with OpenMP, allowing multithreaded runs
# make sure to also enable multiple threads when running, e.g.:
# OMP_NUM_THREADS=4 ./run out/model.bin
.PHONY: runomp
runomp: run.c
	$(CC) -Ofast -fopenmp -march=native run.c  -lm  -o run
	$(CC) -Ofast -fopenmp -march=native runq.c  -lm  -o runq

.PHONY: win64
win64:
	x86_64-w64-mingw32-gcc -Ofast -D_WIN32 -o run.exe -I. run.c win.c
	x86_64-w64-mingw32-gcc -Ofast -D_WIN32 -o runq.exe -I. runq.c win.c

# compiles with gnu99 standard flags for amazon linux, coreos, etc. compatibility
.PHONY: rungnu
rungnu:
	$(CC) -Ofast -std=gnu11 -o run run.c -lm
	$(CC) -Ofast -std=gnu11 -o runq runq.c -lm

.PHONY: runompgnu
runompgnu:
	$(CC) -Ofast -fopenmp -std=gnu11 run.c  -lm  -o run
	$(CC) -Ofast -fopenmp -std=gnu11 runq.c  -lm  -o runq

# run all tests
.PHONY: test
test:
	pytest

# run only tests for run.c C implementation (is a bit faster if only C code changed)
.PHONY: testc
testc:
	pytest -k runc

# run the C tests, without touching pytest / python
# to increase verbosity level run e.g. as `make testcc VERBOSITY=1`
VERBOSITY ?= 0
.PHONY: testcc
testcc:
	$(CC) -DVERBOSITY=$(VERBOSITY) -O3 -o testc test.c -lm
	./testc

.PHONY: clean
clean:
	rm -f run
	rm -f runq
	rm -f runq_bm
	rm -f runq_bm_rocket.riscv
	rm -f runq_bm_rocket_sw.riscv

.PHONY: clean-headers
clean-headers:
	rm -f model_weights.h tokenizer_data.h
