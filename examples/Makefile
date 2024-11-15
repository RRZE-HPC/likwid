# =======================================================================================
#
#      Filename:  Makefile
#
#      Description:  example Makefile
#
#      Version:   5.4.0
#      Released:  15.11.2024
#
#      Author:  Jan Treibig (jt), jan.treibig@gmail.com
#               Thomas Gruber (tr), thomas.roehl@googlemail.com
#
#      Project:  likwid
#
#      Copyright (C) 2024 RRZE, University Erlangen-Nuremberg
#
#      This program is free software: you can redistribute it and/or modify it under
#      the terms of the GNU General Public License as published by the Free Software
#      Foundation, either version 3 of the License, or (at your option) any later
#      version.
#
#      This program is distributed in the hope that it will be useful, but WITHOUT ANY
#      WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
#      PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
#      You should have received a copy of the GNU General Public License along with
#      this program.  If not, see <http://www.gnu.org/licenses/>.
#
# =======================================================================================

include ../config.mk
include ../make/include_$(COMPILER).mk

LIKWID_BINDIR ?= $(PREFIX)/bin
LIKWID_INCDIR ?= $(PREFIX)/include
LIKWID_INCLUDE ?= -I$(LIKWID_INCDIR)
LIKWID_LIBDIR ?= $(PREFIX)/lib
LIKWID_LIB ?= -L$(LIKWID_LIBDIR)
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
	$(CC) -fopenmp $(LIKWID_DEFINES) $(LIKWID_INCLUDE) $(LIKWID_LIB) C-markerAPI.c -o C-markerAPI -llikwid -lm

C-markerAPI-run: C-markerAPI
	LD_LIBRARY_PATH=$(LIKWID_LIBDIR):$LD_LIBRARY_PATH $(LIKWID_BINDIR)/likwid-perfctr -C 0 -g INSTR_RETIRED_ANY:FIXC0 -m ./C-markerAPI

C-likwidAPI:
	$(CC) -fopenmp $(LIKWID_INCLUDE) $(LIKWID_LIB) C-likwidAPI.c -o C-likwidAPI -llikwid -lm

C-likwidAPI-run: C-likwidAPI
	LD_LIBRARY_PATH=$(LIKWID_LIBDIR):$LD_LIBRARY_PATH ./C-likwidAPI

C-internalMarkerAPI:
	$(CC) -g -fopenmp $(LIKWID_DEFINES) $(LIKWID_INCLUDE) $(LIKWID_LIB) C-internalMarkerAPI.c -o C-internalMarkerAPI -llikwid -lm

C-internalMarkerAPI-run: C-internalMarkerAPI
	LD_LIBRARY_PATH=$(LIKWID_LIBDIR):$LD_LIBRARY_PATH ./C-internalMarkerAPI

monitoring:
	$(CC) $(LIKWID_INCLUDE) $(LIKWID_LIB) monitoring.c -o monitoring -llikwid -lm

monitoring-run: monitoring
	./monitoring

F-markerAPI: F-markerAPI.F90
	@if [ -x $(FC) ]; then $(FC) -fopenmp $(LIKWID_DEFINES) $(LIKWID_INCLUDE) $(LIKWID_LIB) F-markerAPI.F90 -o F-markerAPI -llikwid -lm; else echo "No Fortran compiler found"; fi
F-markerAPI-run: F-markerAPI
	@if [ -x F-markerAPI ]; then LD_LIBRARY_PATH=$(LIKWID_LIBDIR):$LD_LIBRARY_PATH $(LIKWID_BINDIR)/likwid-perfctr -C 0 -g INSTR_RETIRED_ANY:FIXC0 -m ./F-markerAPI; else echo "No executable F-markerAPI found"; fi


Lua-likwidAPI:
	sed -e "s+<PREFIX>+$(PREFIX)+g" Lua-likwidAPI.lua > Lua-likwidAPI
	chmod +x Lua-likwidAPI

Lua-likwidAPI-run: Lua-likwidAPI
	LD_LIBRARY_PATH=$(LIKWID_LIBDIR):$LD_LIBRARY_PATH ./Lua-likwidAPI

clean:
	rm -f C-markerAPI C-likwidAPI F-markerAPI Lua-likwidAPI monitoring C-internalMarkerAPI

.PHONY: clean C-markerAPI C-likwidAPI F-markerAPI Lua-likwidAPI monitoring C-internalMarkerAPI
