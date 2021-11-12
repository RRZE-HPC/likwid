include ../config.mk

LIKWID_LIB ?= -L$(PREFIX)/lib
LIKWID_INC ?= -I$(PREFIX)/include
LIKWID_DEFINES ?= -DLIKWID_PERFMON

all:  targets

GCC_C11_SUPPORT_MAJOR=$(shell /bin/bash -c "g++ -v 2>&1 | grep -o -E '([0-9])\.' | head -n 1 | tr -d '[:punct:]'")
GCC_C11_SUPPORT_MINOR=$(shell /bin/bash -c "g++ -v 2>&1 | grep -o -E '\.([0-9])\.' | head -n 1 | tr -d '[:punct:]'")
ICC_AVAILABLE=$(shell /bin/bash -c "which icc | wc -l")
ICPC_AVAILABLE=$(shell /bin/bash -c "which icpc | wc -l")
TBB_AVAILABLE=$(shell /bin/bash -c "ldconfig -v 2>/dev/null | grep libtbb.so | wc -l")

targets:
	@echo "Please specify a build target:"
	@echo " - streamGCC (STREAM-like code with multiple MarkerAPI regions compiled with GCC OpenMP)"
	@echo " - streamICC (STREAM-like code with multiple MarkerAPI regions compiled with Intel C Compiler OpenMP if available)"
	@echo " - streamAPIGCC (STREAM-like code using the LikwidAPI compiled with GCC OpenMP)"
	@echo " - streamGCC_C11 (STREAM-like code with multiple MarkerAPI regions using C++11 threads compiled with GCC)"
	@echo " - streamICC_C11 (STREAM-like code with multiple MarkerAPI regions using C++11 threads compiled with Intel C Compiler)"
	@echo " - stream_cilk (STREAM-like code with Cilk+)"
	@echo " - streamCU (STREAM-like code in CUDA implementation and NvMarkerAPI)"
	@echo ""
	@echo " - jacobi-2D-5pt-gcc (2D Jacobi 5pt stencil compiled with GCC OpenMP)"
	@echo " - jacobi-2D-5pt-icc (2D Jacobi 5pt stencil compiled with Intel C Compiler OpenMP)"
	@echo ""
	@echo " - serial (Serial code computing power 2 of a vector)"
	@echo " - test-likwidAPI (LikwidAPI test suite)"
	@echo " - testmarker-cnt (Test code with code regions executed with different loop counts)"
	@echo " - testmarker-omp (Test code with code regions for OpenMP loops)"
	@echo " - testmarkerF90 (Fortran90 test code with multiple regions compiled with Intel Fortran Compiler)"
	@echo " - test-mpi (MPI pinning test code with OpenMP)"
	@echo " - test-mpi-pthreads (MPI pinning test code with Pthreads)"
	@echo " - testTBBGCC (Intel TBB test code compiled with GCC)"
	@echo " - testTBBICC (Intel TBB test code compiled with Intel C++ Compiler)"
	@echo " - triadCU (Vector triad benchmark code in CUDA implementation and NvMarkerAPI)"
	@echo ""
	@echo " - clean (Clean all compiled files)"

jacobi-2D-5pt-gcc: jacobi-2D-5pt.c
	gcc -O3 -std=c99 -fopenmp $(LIKWID_INC) $(LIKWID_DEFINES) $(LIKWID_LIB) -o $@  jacobi-2D-5pt.c -lm -llikwid

streamGCC: stream.c
	gcc -O3 -std=c99 -fopenmp $(LIKWID_INC) $(LIKWID_DEFINES) $(LIKWID_LIB) -o $@  stream.c -lm -llikwid

streamAPIGCC: stream-API.c
	gcc -O3 -std=c99 -fopenmp -ftree-vectorize -ffast-math $(LIKWID_INC) $(LIKWID_LIB) -o $@  stream-API.c -lm -llikwid

serial: serial.c
	gcc -O3 -std=c99 $(LIKWID_INC) $(LIKWID_DEFINES) $(LIKWID_LIB) -o $@  serial.c -lm -llikwid

test-likwidAPI: test-likwidAPI.c
	gcc -O3 -std=c99 $(LIKWID_INC) $(LIKWID_DEFINES) $(LIKWID_LIB) -o $@  test-likwidAPI.c -lm -llikwid

test-msr-access: test-msr-access.c
	gcc -o $@  test-msr-access.c

streamICC: stream.c
	if [ $(ICC_AVAILABLE) -ne 0 ]; then icc -O3 -xHost -std=c99 -qopenmp $(LIKWID_INC) $(LIKWID_DEFINES) $(LIKWID_LIB) -o $@ stream.c -lm -llikwid; fi

jacobi-2D-5pt-icc: jacobi-2D-5pt.c
	if [ $(ICC_AVAILABLE) -ne 0 ]; then icc -O3 -xHost -std=c99 -qopenmp $(LIKWID_INC) $(LIKWID_DEFINES) $(LIKWID_LIB) -o $@ jacobi-2D-5pt.c -lm -llikwid; fi

streamGCC_C11: stream.cc
	@if [ $(GCC_C11_SUPPORT_MAJOR) -eq 4  -a  $(GCC_C11_SUPPORT_MINOR) -gt 8 ]; then g++ -O3 -std=c++11 -pthread $(LIKWID_INC) $(LIKWID_DEFINES) $(LIKWID_LIB) -o $@ stream.cc -lm -llikwid; fi
	@if [ $(GCC_C11_SUPPORT_MAJOR) -gt 4 ]; then g++ -O3 -std=c++11 -pthread $(LIKWID_INC) $(LIKWID_DEFINES) $(LIKWID_LIB) -o $@ stream.cc -lm -llikwid; fi

streamICC_C11: stream.cc
	@if [ $(ICPC_AVAILABLE) -ne 0 ]; then icpc -restrict -O3 -std=c++11 -pthread $(LIKWID_INC) $(LIKWID_DEFINES) $(LIKWID_LIB) -o $@ stream.cc -lm -llikwid; fi

testmarker-cnt: testmarker-cnt.c
	gcc -O3 -std=c99 -fopenmp $(LIKWID_INC) $(LIKWID_DEFINES) $(LIKWID_LIB) -o $@ testmarker-cnt.c -lm -llikwid

testmarker-omp: testmarker-omp.c
	gcc -O3 -std=c99 -fopenmp $(LIKWID_INC) $(LIKWID_DEFINES) $(LIKWID_LIB) -o $@ testmarker-omp.c -llikwid

testmarkerF90: chaos.F90
	ifort -O3 $(LIKWID_INC) $(LIKWID_DEFINES) $(LIKWID_LIB) -o $@ chaos.F90 -lpthread -llikwid

test-mpi: MPI_pin_test.c
	mpicc -O2 -fopenmp -D_GNU_SOURCE  -o $@ MPI_pin_test.c

test-mpi-pthreads: MPI_pin_test.c
	mpicc -O2 -pthread -DPTHREADS -D_GNU_SOURCE  -o $@ MPI_pin_test.c

stream_cilk: stream_cilk.c
	@if [ $(ICC_AVAILABLE) -ne 0 ]; then icc -O3 $(LIKWID_INC) $(LIKWID_DEFINES) $(LIKWID_LIB) -o $@ stream_cilk.c -llikwid; fi

testTBBGCC:
	@if [ $(TBB_AVAILABLE) -ne 0 ]; then g++ -O3 $(LIKWID_INC) $(LIKWID_DEFINES) $(LIKWID_LIB) -o $@ testTBB.cc -ltbb -llikwid; fi

testTBBICC:
	@if [ $(TBB_AVAILABLE) -ne 0 -a $(ICPC_AVAILABLE) -ne 0 ]; then icpc -O3 $(LIKWID_INC) $(LIKWID_DEFINES) $(LIKWID_LIB) -o $@ testTBB.cc -ltbb -llikwid; else echo "Either TBB or ICPC missing"; fi

streamCU: stream.cu
	nvcc -O3 -I. $(LIKWID_INC) $(LIKWID_LIB) -DLIKWID_NVMON stream.cu -o $@ -lm -llikwid

triadCU: triad.cu
	nvcc -O3 -I. $(LIKWID_INC) $(LIKWID_LIB) -DLIKWID_NVMON -Xcompiler -fopenmp triad.cu -o $@ -lm -llikwid

.PHONY: clean distclean streamGCC streamICC streamGCC_C11 streamICC_C11 testmarker-cnt testmarker-omp testmarkerF90 test-mpi test-mpi-pthreads stream_cilk serial test-likwidAPI streamAPIGCC test-msr-access testTBBGCC testTBBICC jacobi-2D-5pt-icc jacobi-2D-5pt-gcc matmul_marker matmul marker_overhead

clean:
	rm -f streamGCC streamICC streamGCC_C11 streamICC_C11 stream_cilk testmarker-cnt testmarkerF90 test-mpi test-mpi-pthreads testmarker-omp serial test-likwidAPI streamAPIGCC test-msr-access testTBBGCC testTBBICC jacobi-2D-5pt-icc jacobi-2D-5pt-gcc matmul_marker matmul marker_overhead streamCU

distclean: clean
