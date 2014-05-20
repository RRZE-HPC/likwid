LIKWID_LIB = -L../ -llikwid 
INCLUDES = -I../src/includes -I../ -I../GCC

all:  testmarker 

fortran: chaos.F90
	gfortran $(INCLUDES) -ffree-form -fintrinsic-modules-path ../  -O3  -o $@ chaos.F90 $(LIKWID_LIB) -lpthread

fortranfix: chaos_fixed.F90
	gfortran $(INCLUDES) -P -fintrinsic-modules-path ../  -O3  -o $@ chaos.F90 $(LIKWID_LIB) -lpthread

stream-gcc:
	gcc -O3 -fopenmp -o $@  stream.c


stream-icc:
	icc -O3 -openmp -o $@  stream.c

testmarker:
	gcc -O3 -std=c99  $(INCLUDES) -fopenmp -DLIKWID_PERFMON  -o $@ testmarker-cnt.c $(LIKWID_LIB) -lm

testmarker-omp:
	gcc -O3 -mmic  -std=c99  $(INCLUDES) -openmp -DLIKWID_PERFMON  -o $@ testmarker-omp.c $(LIKWID_LIB)

test-msr:
	gcc -O2 -o $@ msrClient.c 

test-mpi:
	mpicc -O2 -openmp -I../src/includes  -I../GCC -D_GNU_SOURCE  -o $@ MPI_pin_test.c $(LIKWID_LIB)

.PHONY: clean

clean: 
	rm -f stream-* test test-marker 


