
include ../config.mk
include ../make/include_$(COMPILER).mk

LIKWID_INCLUDE ?= -I$(PREFIX)/include
LIKWID_LIB ?= -L$(PREFIX)/lib -llikwid

all: C-markerAPI C-likwidAPI F-markerAPI Lua-likwidAPI C-markerAPI-run C-likwidAPI-run F-markerAPI-run Lua-likwidAPI-run

help:
	@echo "Help message for examples included in LIKWID"
	@echo
	@echo "This folder contins examples how you can use the LIKWID API"
	@echo "Possible examples are:"
	@echo "- Marker API in C applications: C-markerAPI"
	@echo "- Marker API in Fortran applications: F-markerAPI"
	@echo "- Self Monitoring in C applications: C-likwidAPI"
	@echo "- Using the LIKWID API in Lua scripts: Lua-likwidAPI"
	@echo "- Monitoring a system with LIKWID: monitoring"
	@echo
	@echo "To build an example put the name behind make, e.g. make C-likwidAPI"
	@echo "To run the built example append '-run' to the name and add it to make: make C-likwidAPI-run"

C-markerAPI:
	$(CC) -fopenmp -DLIKWID_PERFMON -I$(PREFIX)/include -L$(PREFIX)/lib C-markerAPI.c -o C-markerAPI -llikwid -lm

C-markerAPI-run: C-markerAPI
	$(PREFIX)/bin/likwid-perfctr -C 0 -g INSTR_RETIRED_ANY:FIXC0 -m ./C-markerAPI

C-likwidAPI:
	$(CC) -fopenmp -I$(PREFIX)/include -L$(PREFIX)/lib C-likwidAPI.c -o C-likwidAPI -llikwid -lm

C-likwidAPI-run: C-likwidAPI
	./C-likwidAPI

monitoring:
	$(CC) -I$(PREFIX)/include -L$(PREFIX)/lib monitoring.c -o monitoring -llikwid -lm

monitoring-run: monitoring
	./monitoring

F-markerAPI:
	$(FC) -fopenmp -DLIKWID_PERFMON -I$(PREFIX) -L$(PREFIX) F-markerAPI.F90 -o F-markerAPI -llikwid -lm

F-markerAPI-run: F-markerAPI
	$(PREFIX)/bin/likwid-perfctr -C 0 -g INSTR_RETIRED_ANY:FIXC0 -m ./F-markerAPI

Lua-likwidAPI:
	sed -e "s+<PREFIX>+$(PREFIX)+g" Lua-likwidAPI.lua > Lua-likwidAPI
	chmod +x Lua-likwidAPI

Lua-likwidAPI-run: Lua-likwidAPI
	./Lua-likwidAPI

clean:
	rm -f C-markerAPI C-likwidAPI F-markerAPI Lua-likwidAPI monitoring

.PHONY: clean C-markerAPI C-likwidAPI F-markerAPI Lua-likwidAPI monitoring
