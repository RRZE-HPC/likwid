include ../config.mk

LIKWID_LIB = -L$(PREFIX)/lib -llikwid
INCLUDES = -I$(PREFIX)/include

all:  testmarker testmarkerF90 streamGCC streamICC streamC11

GCC_C11_SUPPORT_MAJOR=$(shell /bin/bash -c "g++ -v 2>&1 | grep -o -E '([0-9])\.' | head -n 1 | tr -d '[:punct:]'")
GCC_C11_SUPPORT_MINOR=$(shell /bin/bash -c "g++ -v 2>&1 | grep -o -E '\.([0-9])\.' | head -n 1 | tr -d '[:punct:]'")
ICC_AVAILABLE=$(shell /bin/bash -c "which icc | wc -l")
ICPC_AVAILABLE=$(shell /bin/bash -c "which icpc | wc -l")

testmarkerF90: chaos.F90
	ifort $(INCLUDES)  -O3  -o $@ chaos.F90 $(LIKWID_LIB) -lpthread

streamGCC: stream.c
	gcc -O3 -std=c99 $(INCLUDES) -fopenmp  -o $@  -DLIKWID_PERFMON stream.c $(LIKWID_LIB) -lm

streamICC: stream.c
	@if [ $(ICC_AVAILABLE) -ne 0 ]; then icc -O3 -std=c99 $(INCLUDES) -openmp  -o $@  -DLIKWID_PERFMON stream.c $(LIKWID_LIB) -lm; fi

streamC11: stream.cc
	@if [ $(GCC_C11_SUPPORT_MAJOR) -eq 4  -a  $(GCC_C11_SUPPORT_MINOR) -gt 8 ]; then g++ -O3 -std=c++11 -pthread -o $@ -DLIKWID_PERFMON stream.cc $(LIKWID_LIB) -lm; fi
	@if [ $(GCC_C11_SUPPORT_MAJOR) -gt 4 ]; then g++ -O3 -std=c++11 -pthread -o $@ -DLIKWID_PERFMON stream.cc $(LIKWID_LIB) -lm; fi

streamMC11: stream.cc
	@if [ $(ICPC_AVAILABLE) -ne 0 ]; then icpc -restrict -O3 -std=c++11 -pthread -o $@ -DLIKWID_PERFMON stream.cc $(LIKWID_LIB) -lm; fi

testmarker:
	gcc -O3 -std=c99  $(INCLUDES) -fopenmp -DLIKWID_PERFMON  -o $@ testmarker-cnt.c $(LIKWID_LIB) -lm

testmarker-omp:
	gcc -O3 -std=c99  $(INCLUDES) -fopenmp -DLIKWID_PERFMON  -o $@ testmarker-omp.c $(LIKWID_LIB)

test-mpi:
	mpicc -O2 -fopenmp -D_GNU_SOURCE  -o $@ MPI_pin_test.c

stream_cilk:
	@if [ $(ICC_AVAILABLE) -ne 0 ]; then icc -O3 -DLIKWID_PERFMON $(INCLUDES) -o $@ stream_cilk.c $(LIKWID_LIB); fi

.PHONY: clean stream streamM test-mpi stream_cilk streamMC11 streamC11

clean: 
	rm -f stream streamM streamC11 streamMC11 stream_cilk testmarker testmarkerF90 test-mpi testmarker-omp


