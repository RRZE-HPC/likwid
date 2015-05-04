
include ../config.mk
PREFIX=/home/rrze/unrz/unrz139/TMP/likwid-base/trunk

all: C-markerAPI C-likwidAPI F-markerAPI Lua-likwidAPI C-markerAPI-run C-likwidAPI-run F-markerAPI-run Lua-likwidAPI-run


C-markerAPI:
	gcc -fopenmp -DLIKWID_PERFMON -I$(PREFIX)/src/includes -L$(PREFIX) C-markerAPI.c -o C-markerAPI -llikwid -lm

C-markerAPI-run:
	$(PREFIX)/bin/likwid-perfctr -C 0 -g INSTR_RETIRED_ANY:FIXC0 -m ./C-markerAPI

C-likwidAPI:
	gcc -fopenmp -DLIKWID_PERFMON -I$(PREFIX)/include -L$(PREFIX)/lib C-likwidAPI.c -o C-likwidAPI -llikwid -lm

C-likwidAPI-run:
	./C-likwidAPI

F-markerAPI:
	gfortran -fopenmp -DLIKWID_PERFMON -I$(PREFIX) -L$(PREFIX) F-markerAPI.F90 -o F-markerAPI -llikwid -lm

F-markerAPI-run:
	$(PREFIX)/bin/likwid-perfctr -C 0 -g INSTR_RETIRED_ANY:FIXC0 -m ./F-markerAPI

Lua-likwidAPI:
	sed -e "s+<PREFIX>+$(PREFIX)+g" Lua-likwidAPI.lua > Lua-likwidAPI
	chmod +x Lua-likwidAPI

Lua-likwidAPI-run:
	./Lua-likwidAPI

clean:
	rm -f C-markerAPI C-likwidAPI F-markerAPI Lua-likwidAPI

.PHONY: clean C-markerAPI C-likwidAPI F-markerAPI Lua-likwidAPI
