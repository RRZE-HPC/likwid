include ../config.mk

LIKWID_LIB ?= -L$(PREFIX)/lib -llikwid
LIKWID_INCLUDE ?= -I$(PREFIX)/include
LIKWID_DEFINES ?= -DLIKWID_PERFMON

all:  streamGCC jacobi-2D-5pt-gcc

GCC_C11_SUPPORT_MAJOR=$(shell /bin/bash -c "g++ -v 2>&1 | grep -o -E '([0-9])\.' | head -n 1 | tr -d '[:punct:]'")
GCC_C11_SUPPORT_MINOR=$(shell /bin/bash -c "g++ -v 2>&1 | grep -o -E '\.([0-9])\.' | head -n 1 | tr -d '[:punct:]'")
ICC_AVAILABLE=$(shell /bin/bash -c "which icc | wc -l")
ICPC_AVAILABLE=$(shell /bin/bash -c "which icpc | wc -l")
TBB_AVAILABLE=$(shell /bin/bash -c "ldconfig -v 2>/dev/null | grep libtbb.so | wc -l")

jacobi-2D-5pt-gcc: jacobi-2D-5pt.c
	gcc -O3 -std=c99 $(LIKWID_INCLUDE) $(LIKWID_DEFINES) -fopenmp  -o $@  jacobi-2D-5pt.c $(LIKWID_LIB) -lm -llikwid

streamGCC: stream.c
	gcc -O3 -std=c99 $(LIKWID_INCLUDE) $(LIKWID_DEFINES) -fopenmp  -o $@  stream.c $(LIKWID_LIB) -lm -llikwid

streamAPIGCC: stream-API.c
	gcc -O3 -std=c99 $(LIKWID_INCLUDE) -fopenmp -ftree-vectorize -ffast-math -o $@  stream-API.c $(LIKWID_LIB) -lm -llikwid

serial: serial.c
	gcc -O3 -std=c99 $(LIKWID_INCLUDE) $(LIKWID_DEFINES) -o $@  serial.c $(LIKWID_LIB) -lm -llikwid

test-likwidAPI: test-likwidAPI.c
	gcc -O3 -std=c99 $(LIKWID_INCLUDE) $(LIKWID_DEFINES) -o $@  test-likwidAPI.c $(LIKWID_LIB) -lm -llikwid

test-msr-access: test-msr-access.c
	gcc -o $@  test-msr-access.c

streamICC: stream.c
	if [ $(ICC_AVAILABLE) -ne 0 ]; then icc -O3 -xHost -std=c99 $(LIKWID_INCLUDES) -openmp  -o $@  $(LIKWID_DEFINES) stream.c $(LIKWID_LIB) -lm -llikwid; fi

jacobi-2D-5pt-icc: jacobi-2D-5pt.c
	if [ $(ICC_AVAILABLE) -ne 0 ]; then icc -O3 -xHost -std=c99 $(LIKWID_INCLUDES) -openmp  -o $@  $(LIKWID_DEFINES) jacobi-2D-5pt.c $(LIKWID_LIB) -lm -llikwid; fi

streamGCC_C11: stream.cc
	@if [ $(GCC_C11_SUPPORT_MAJOR) -eq 4  -a  $(GCC_C11_SUPPORT_MINOR) -gt 8 ]; then g++ -O3 -std=c++11 -pthread -o $@ $(LIKWID_DEFINES) stream.cc $(LIKWID_LIB) -lm -llikwid; fi
	@if [ $(GCC_C11_SUPPORT_MAJOR) -gt 4 ]; then g++ -O3 -std=c++11 -pthread -o $@ $(LIKWID_DEFINES) stream.cc $(LIKWID_LIB) -lm -llikwid; fi

streamICC_C11: stream.cc
	@if [ $(ICPC_AVAILABLE) -ne 0 ]; then icpc -restrict -O3 -std=c++11 -pthread -o $@ $(LIKWID_DEFINES) stream.cc $(LIKWID_LIB) -lm -llikwid; fi

testmarker-cnt: testmarker-cnt.c
	gcc -O3 -std=c99  $(LIKWID_INCLUDES) -fopenmp $(LIKWID_DEFINES) -o $@ testmarker-cnt.c $(LIKWID_LIB) -lm -llikwid

testmarker-omp: testmarker-omp.c
	gcc -O3 -std=c99  $(LIKWID_INCLUDES) -fopenmp $(LIKWID_DEFINES) -o $@ testmarker-omp.c $(LIKWID_LIB) -llikwid

testmarkerF90: chaos.F90
	ifort $(LIKWID_INCLUDES) $(LIKWID_DEFINES) -O3  -o $@ chaos.F90 $(LIKWID_LIB) -lpthread -llikwid

test-mpi: MPI_pin_test.c$
	mpicc -O2 -fopenmp -D_GNU_SOURCE  -o $@ MPI_pin_test.c

test-mpi-pthreads: MPI_pin_test.c$
	mpicc -O2 -pthread -DPTHREADS -D_GNU_SOURCE  -o $@ MPI_pin_test.c

stream_cilk: stream_cilk.c
	@if [ $(ICC_AVAILABLE) -ne 0 ]; then icc -O3 $(LIKWID_DEFINES) $(LIKWID_INCLUDES) -o $@ stream_cilk.c $(LIKWID_LIB) -llikwid; fi

testTBBGCC:
	@if [ $(TBB_AVAILABLE) -ne 0 ]; then g++ -O3 $(LIKWID_DEFINES) $(LIKWID_INCLUDES) -o $@ testTBB.cc -ltbb $(LIKWID_LIB) -llikwid; fi

testTBBICC:
	@if [ $(TBB_AVAILABLE) -ne 0 -a $(ICPC_AVAILABLE) -ne 0 ]; then icpc -O3 $(LIKWID_DEFINES) $(LIKWID_INCLUDES) -o $@ testTBB.cc -ltbb $(LIKWID_LIB) -llikwid; else echo "Either TBB or ICPC missing"; fi

streamCU: stream.cu
	nvcc -O3 -I. $(LIKWID_INCLUDE) -DLIKWID_NVMON stream.cu -o $@ -lm $(LIKWID_LIB) -llikwid

triadCU: triad.cu
	nvcc -O3 -I. $(LIKWID_INCLUDE) -DLIKWID_NVMON triad.cu -o $@ -lm $(LIKWID_LIB) -llikwid

.PHONY: clean streamGCC streamICC streamGCC_C11 streamICC_C11 testmarker-cnt testmarker-omp testmarkerF90 test-mpi test-mpi-pthreads stream_cilk serial test-likwidAPI streamAPIGCC test-msr-access testTBBGCC testTBBICC jacobi-2D-5pt-icc jacobi-2D-5pt-gcc matmul_marker matmul marker_overhead

clean:
	rm -f streamGCC streamICC streamGCC_C11 streamICC_C11 stream_cilk testmarker-cnt testmarkerF90 test-mpi test-mpi-pthreads testmarker-omp serial test-likwidAPI streamAPIGCC test-msr-access testTBBGCC testTBBICC jacobi-2D-5pt-icc jacobi-2D-5pt-gcc matmul_marker matmul marker_overhead
