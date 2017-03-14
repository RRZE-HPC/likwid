include ../config.mk

LIKWID_LIB ?= -L$(PREFIX)/lib -llikwid
LIKWID_INCLUDE ?= -I$(PREFIX)/include
LIKWID_DEFINES ?= -DLIKWID_PERFMON

all:  help

GCC_C11_SUPPORT_MAJOR=$(shell /bin/bash -c "g++ -v 2>&1 | grep -o -E '([0-9])\.' | head -n 1 | tr -d '[:punct:]'")
GCC_C11_SUPPORT_MINOR=$(shell /bin/bash -c "g++ -v 2>&1 | grep -o -E '\.([0-9])\.' | head -n 1 | tr -d '[:punct:]'")
ICC_AVAILABLE=$(shell /bin/bash -c "which icc | wc -l")
ICPC_AVAILABLE=$(shell /bin/bash -c "which icpc | wc -l")
IFORT_AVAILABLE=$(shell /bin/bash -c "which ifort | wc -l")
MPICC_AVAILABLE=$(shell /bin/bash -c "which mpicc | wc -l")
TBB_AVAILABLE=$(shell /bin/bash -c "/sbin/ldconfig -v 2>/dev/null | grep libtbb.so | wc -l")

help:
	@echo "Welcome to the test folder of LIKWID :)"
	@echo ""
	@echo "The following applications can be used to test the functionality of LIKWID."
	@echo ""
	@echo "Available make targets"
	@echo "streamGCC - self-written STREAM benchmark with MarkerAPI compiled with gcc"
	@echo "streamICC - self-written STREAM benchmark with MarkerAPI compiled with icc"
	@echo "streamGCC_C11 - self-written STREAM benchmark with MarkerAPI compiled with gcc with C++11 threads"
	@echo "streamICC_C11 - self-written STREAM benchmark with MarkerAPI compiled with icc with C++11 threads"
	@echo "serial - test of MarkerAPI for serial applications"
	@echo "test-likwidAPI - test internal LIKWID library functions"
	@echo "test-msr-access - test whether the msr device files are accessible"
	@echo "test-mpi - application to test pinning functionality of likwid-mpirun"
	@echo "stream_cilk - self-written STREAM benchmark with CilkPlus"
	@echo "testTBBGCC - application with Intel Threading Building Blocks and MarkerAPI compiled with gcc"
	@echo "testTBBICC - application with Intel Threading Building Blocks and MarkerAPI compiled with icc"
	@echo "testmarker-cnt - application to test MarkerAPI"
	@echo "testmarker-omp - application to test MarkerAPI"
	@echo "testmarkerF90 - application to test MarkerAPI with Fortran90"
	@echo "streamAPIGCC - Use LIKWID API directly in self-written STREAM benchmark"
	@echo ""
	@echo "Just type 'make <target>' to build the code. If some special compiler/library is required,"
	@echo "the compilation is only done when the compiler/library is available."
	@echo "The resulting executable is named like the targets."
	@echo ""
	@echo "Compilations are performed with following settings:"
	@echo "INCLUDE SETTING: $(LIKWID_INCLUDE)"
	@echo "LIBRARY SETTING: $(LIKWID_INCLUDE)"
	@echo "DEFINES SETTING: $(LIKWID_DEFINES)"
	@echo ""
	@echo "INFO: The original STREAM code by J. D. McCalpin cannot be included due to license restrictions."

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
	@if [ $(ICC_AVAILABLE) -ne 0 ]; then icc -O3 -xHost -std=c99 $(LIKWID_INCLUDE) -openmp  -o $@  $(LIKWID_DEFINES) stream.c $(LIKWID_LIB) -lm -llikwid; else echo "ICC cannot be found."; fi

streamGCC_C11: stream.cc
	@if [ $(GCC_C11_SUPPORT_MAJOR) -eq 4  -a  $(GCC_C11_SUPPORT_MINOR) -gt 8 ]; then g++ -O3 -std=c++11 -pthread -o $@ $(LIKWID_DEFINES) stream.cc $(LIKWID_LIB) -lm -llikwid; fi
	@if [ $(GCC_C11_SUPPORT_MAJOR) -gt 4 ]; then g++ -O3 -std=c++11 -pthread -o $@ $(LIKWID_DEFINES) stream.cc $(LIKWID_LIB) -lm -llikwid; else echo "GCC has no C++11 support."; fi

streamICC_C11: stream.cc
	@if [ $(ICPC_AVAILABLE) -ne 0 ]; then icpc -restrict -O3 -std=c++11 -pthread -o $@ $(LIKWID_DEFINES) stream.cc $(LIKWID_LIB) -lm -llikwid; else echo "ICPC cannot be found."; fi

testmarker-cnt: testmarker-cnt.c
	gcc -O3 -std=c99  $(LIKWID_INCLUDE) -fopenmp $(LIKWID_DEFINES) -o $@ testmarker-cnt.c $(LIKWID_LIB) -lm -llikwid

testmarker-omp: testmarker-omp.c
	gcc -O3 -std=c99  $(LIKWID_INCLUDE) -fopenmp $(LIKWID_DEFINES) -o $@ testmarker-omp.c $(LIKWID_LIB) -llikwid

testmarkerF90: chaos.F90
	@if [ $(IFORT_AVAILABLE) -ne 0 ]; then ifort $(LIKWID_INCLUDE) $(LIKWID_DEFINES) -O3  -o $@ chaos.F90 $(LIKWID_LIB) -lpthread -llikwid; else echo "IFORT cannot be found."; fi

test-mpi: MPI_pin_test.c$
	@if [ $(MPICC_AVAILABLE) -ne 0 ]; then mpicc -O2 -fopenmp -D_GNU_SOURCE  -o $@ MPI_pin_test.c; else echo "MPICC cannot be found."; fi

stream_cilk: stream_cilk.c
	if [ $(ICC_AVAILABLE) -ne 0 ]; then icc -O3 $(LIKWID_DEFINES) $(LIKWID_INCLUDE) -o $@ stream_cilk.c $(LIKWID_LIB) -llikwid; else echo "ICC cannot be found."; fi

testTBBGCC:
	if [ $(TBB_AVAILABLE) -ne 0 ]; then g++ -O3 $(LIKWID_DEFINES) $(LIKWID_INCLUDE) -o $@ testTBB.cc -ltbb $(LIKWID_LIB) -llikwid; else echo "Threading Building Blocks library cannot be found."; fi

testTBBICC:
	@if [ $(TBB_AVAILABLE) -ne 0 -a $(ICPC_AVAILABLE) -ne 0 ]; then icpc -O3 $(LIKWID_DEFINES) $(LIKWID_INCLUDE) -o $@ testTBB.cc -ltbb $(LIKWID_LIB) -llikwid; else echo "Either Threading Building Blocks library or ICPC missing"; fi

.PHONY: clean streamGCC streamICC streamGCC_C11 streamICC_C11 testmarker-cnt testmarker-omp testmarkerF90 test-mpi stream_cilk serial test-likwidAPI streamAPIGCC test-msr-access testTBBGCC testTBBICC

clean:
	rm -f streamGCC streamICC streamGCC_C11 streamICC_C11 stream_cilk testmarker-cnt testmarkerF90 test-mpi testmarker-omp serial test-likwidAPI streamAPIGCC test-msr-access testTBBGCC testTBBICC


