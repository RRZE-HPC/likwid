include ../config.mk

LIKWID_LIB = -L$(PREFIX)/lib -llikwid
INCLUDES = -I$(PREFIX)/include

all:  testmarker testmarkerF90 stream streamM

testmarkerF90: chaos.F90
	ifort $(INCLUDES)  -O3  -o $@ chaos.F90 $(LIKWID_LIB) -lpthread

stream: stream.c
	gcc -O3 $(INCLUDES) -fopenmp  -o $@  -DLIKWID_PERFMON stream.c $(LIKWID_LIB) -lm

streamM: stream.c
	gcc -O3 $(INCLUDES) -fopenmp  -o $@  -DMAX_NUM_THREADS=256 -DLIKWID_PERFMON stream.c $(LIKWID_LIB) -lm

testmarker:
	gcc -O3 -std=c99  $(INCLUDES) -fopenmp -DLIKWID_PERFMON  -o $@ testmarker-cnt.c $(LIKWID_LIB) -lm

testmarker-omp:
	gcc -O3 -std=c99  $(INCLUDES) -openmp -DLIKWID_PERFMON  -o $@ testmarker-omp.c $(LIKWID_LIB)

test-mpi:
	mpicc -DMAX_NUM_THREADS=128 -O2 -openmp -I../src/includes  -I../GCC -D_GNU_SOURCE  -o $@ MPI_pin_test.c $(LIKWID_LIB)

.PHONY: clean streamM

clean: 
	rm -f stream streamM  testmarker testmarkerF90


