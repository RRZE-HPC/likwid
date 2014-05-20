
all: stream-gcc  stream-icc  test  test-marker



stream-gcc:
	gcc -O3 -fopenmp -o $@  stream.c


stream-icc:
	icc -O3 -openmp -o $@  stream.c


test:
	icc -O3 -o $@ dofp.c 

test-marker:
	icc -O3 -pthread -DPERFMON  -o $@ dofp.c -L../ -llikwid

.PHONY: clean

clean: 
	rm -f stream-* test test-marker


