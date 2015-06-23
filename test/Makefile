include ../config.mk

#LIKWID_LIB = -L$(PREFIX)/lib -llikwid
#INCLUDES = -I$(PREFIX)/include
LIKWID_LIB = -L.. -llikwid
INCLUDES = -I../src//includes

all:  testmarker testmarkerF90 stream streamM

testmarkerF90: chaos.F90
	ifort $(INCLUDES)  -O3  -o $@ chaos.F90 $(LIKWID_LIB) -lpthread

stream: stream.c
	gcc -O3 -std=c99 $(INCLUDES) -fopenmp  -o $@  -DLIKWID_PERFMON stream.c $(LIKWID_LIB) -lm

streamM: stream.c
	icc -O3 -std=c99 $(INCLUDES) -openmp  -o $@  -DLIKWID_PERFMON stream.c $(LIKWID_LIB) -lm

testmarker:
	gcc -O3 -std=c99  $(INCLUDES) -fopenmp -DLIKWID_PERFMON  -o $@ testmarker-cnt.c $(LIKWID_LIB) -lm

testmarker-omp:
	gcc -O3 -std=c99  $(INCLUDES) -openmp -DLIKWID_PERFMON  -o $@ testmarker-omp.c $(LIKWID_LIB)

test-mpi:
	mpicc -O2 -fopenmp -D_GNU_SOURCE  -o $@ MPI_pin_test.c

stream_cilk:
	icc -g -O0 -DLIKWID_PERFMON $(INCLUDES) -o $@ stream_cilk.c $(LIKWID_LIB)

.PHONY: clean stream streamM test-mpi stream_cilk

clean: 
	rm -f stream streamM  testmarker testmarkerF90 test-mpi testmarker-omp


