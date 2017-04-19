
include ../config.mk
include ../make/include_$(COMPILER).mk

LIKWID_BINDIR ?= $(PREFIX)/bin
LIKWID_INCLUDE ?= -I$(PREFIX)/include
LIKWID_LIBDIR ?= -L$(PREFIX)/lib
LIKWID_LIB ?= -llikwid
LIKWID_DEFINES ?= -DLIKWID_PERFMON

all: C-markerAPI C-likwidAPI F-markerAPI Lua-likwidAPI C-markerAPI-run C-likwidAPI-run Lua-likwidAPI-run F-markerAPI-run

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
	$(CC) -fopenmp $(LIKWID_DEFINES) $(LIKWID_INCLUDE) $(LIKWID_LIBDIR) C-markerAPI.c -o C-markerAPI $(LIKWID_LIB) -lm

C-markerAPI-run: C-markerAPI
	$(LIKWID_BINDIR)/likwid-perfctr -C 0 -g INSTR_RETIRED_ANY:FIXC0 -m ./C-markerAPI

C-likwidAPI:
	$(CC) -fopenmp $(LIKWID_INCLUDE) $(LIKWID_LIBDIR) C-likwidAPI.c -o C-likwidAPI $(LIKWID_LIB) -lm

C-likwidAPI-run: C-likwidAPI
	./C-likwidAPI

C-internalMarkerAPI:
	$(CC) -g -fopenmp $(LIKWID_DEFINES) $(LIKWID_INCLUDE) $(LIKWID_LIBDIR) C-internalMarkerAPI.c -o C-internalMarkerAPI $(LIKWID_LIB) -lm

C-internalMarkerAPI-run: C-internalMarkerAPI
	OMP_NUM_THREADS=3 ./C-internalMarkerAPI

monitoring:
	$(CC) $(LIKWID_INCLUDE) $(LIKWID_LIBDIR) monitoring.c -o monitoring $(LIKWID_LIB) -lm

monitoring-run: monitoring
	./monitoring

F-markerAPI: F-markerAPI.F90
	@if [ -x $(FC) ]; then $(FC) -fopenmp $(LIKWID_DEFINES) $(LIKWID_INCLUDE) $(LIKWID_LIBDIR) F-markerAPI.F90 -o F-markerAPI $(LIKWID_LIB) -lm; else echo "No Fortran compiler found"; fi
F-markerAPI-run: F-markerAPI
	@if [ -x F-markerAPI ]; then $(LIKWID_BINDIR)/likwid-perfctr -C 0 -g INSTR_RETIRED_ANY:FIXC0 -m ./F-markerAPI; else echo "No executable F-markerAPI found"; fi


Lua-likwidAPI:
	sed -e "s+<PREFIX>+$(PREFIX)+g" Lua-likwidAPI.lua > Lua-likwidAPI
	chmod +x Lua-likwidAPI

Lua-likwidAPI-run: Lua-likwidAPI
	./Lua-likwidAPI

clean:
	rm -f C-markerAPI C-likwidAPI F-markerAPI Lua-likwidAPI monitoring C-internalMarkerAPI

.PHONY: clean C-markerAPI C-likwidAPI F-markerAPI Lua-likwidAPI monitoring C-internalMarkerAPI
