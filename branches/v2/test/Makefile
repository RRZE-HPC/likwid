LIKWID_LIB = -L../ -llikwid 
INCLUDES = -I../src/includes -I../ -I../GCC

all:  test-marker 

fortran: chaos.F90
	gfortran $(INCLUDES) -ffree-form -fintrinsic-modules-path ../  -O3  -o $@ chaos.F90 $(LIKWID_LIB) -lpthread

fortranfix: chaos_fixed.F90
	gfortran $(INCLUDES) -P -fintrinsic-modules-path ../  -O3  -o $@ chaos.F90 $(LIKWID_LIB) -lpthread

stream-gcc:
	gcc -O3 -fopenmp -o $@  stream.c


stream-icc:
	icc -O3 -openmp -o $@  stream.c

dumper:
	gcc -O1 -g $(INCLUDES) -pthread  -o $@ cpuid_dump.c $(LIKWID_LIB)

test:
	icc -O3 -o $@ dofp.c 

testmarker:
	icc -O3  $(INCLUDES) -pthread -DPERFMON  -o $@ testmarker-seq.c $(LIKWID_LIB)

testmarker-omp:
	icc -O3 -std=c99  $(INCLUDES) -openmp -DPERFMON  -o $@ testmarker-omp.c $(LIKWID_LIB)

test-msr:
	gcc -O2 -o $@ msrClient.c 

test-mpi:
	mpicc -O2 -openmp -I../src/includes  -I../GCC -D_GNU_SOURCE  -o $@ MPI_pin_test.c $(LIKWID_LIB)

.PHONY: clean

clean: 
	rm -f stream-* test test-marker test-msr test-mpi fortran


