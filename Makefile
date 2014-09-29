# w
# =======================================================================================
#
#      Filename:  Makefile
#
#      Description:  Central Makefile
#
#      Version:   <VERSION>
#      Released:  <DATE>
#
#      Author:  Jan Treibig (jt), jan.treibig@gmail.com
#      Project:  likwid
#
#      Copyright (C) 2013 Jan Treibig
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

SRC_DIR     = ./src
DOC_DIR     = ./doc
GROUP_DIR   = ./groups
FILTER_DIR  = ./filters
MAKE_DIR    = ./make
EXT_TARGETS = ./ext/lua ./ext/hwloc

#DO NOT EDIT BELOW


# Dependency chains:
# *.[ch] -> *.o -> executables
# *.ptt -> *.pas -> *.s -> *.o -> executables
# *.txt -> *.h (generated)

include ./config.mk
include $(MAKE_DIR)/include_$(COMPILER).mk
include $(MAKE_DIR)/config_checks.mk
include $(MAKE_DIR)/config_defines.mk

INCLUDES  += -I./src/includes -I./ext/lua/includes -I./ext/hwloc/include -I$(BUILD_DIR)
LIBS      +=

#CONFIGURE BUILD SYSTEM
BUILD_DIR  = ./$(COMPILER)
Q         ?= @
GENGROUPLOCK = .gengroup

ifeq ($(COMPILER),MIC)
BENCH_DIR   = ./bench/phi
else
ifeq ($(COMPILER),GCCX86)
BENCH_DIR   = ./bench/x86
else
BENCH_DIR   = ./bench/x86-64
endif
endif

ifeq ($(SHARED_LIBRARY),true)
CFLAGS += $(SHARED_CFLAGS)
LIBS += -L. -pthread -lm -lpci
DYNAMIC_TARGET_LIB := liblikwid.so
TARGET_LIB := $(DYNAMIC_TARGET_LIB)
LIBHWLOC = ext/hwloc/libhwloc.a
LIBLUA = ext/lua/liblua.a
else
STATIC_TARGET_LIB := liblikwid.a
LIBHWLOC = ext/hwloc/libhwloc.a
LIBLUA = ext/lua/liblua.a
TARGET_LIB := $(STATIC_TARGET_LIB)
endif

ifeq ($(DEBUG),true)
DEBUG_FLAGS = -g
DEFINES += -DDEBUG_LIKWID
else
DEBUG_FLAGS =
endif


VPATH     = $(SRC_DIR)
OBJ       = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o,$(wildcard $(SRC_DIR)/*.c))
OBJ      += $(patsubst $(SRC_DIR)/%.cc, $(BUILD_DIR)/%.o,$(wildcard $(SRC_DIR)/*.cc))
PERFMONHEADERS  = $(patsubst $(SRC_DIR)/includes/%.txt, $(BUILD_DIR)/%.h,$(wildcard $(SRC_DIR)/includes/*.txt))
OBJ_BENCH  =  $(patsubst $(BENCH_DIR)/%.ptt, $(BUILD_DIR)/%.o,$(wildcard $(BENCH_DIR)/*.ptt))
OBJ_LUA    =  $(wildcard ./ext/lua/$(COMPILER)/*.o)
OBJ_HWLOC  =  $(wildcard ./ext/hwloc/$(COMPILER)/*.o)

APPS      = likwid-perfctr    \
		likwid-features   \
		likwid-powermeter \
		likwid-memsweeper \
		likwid-topology   \
		likwid-genCfg     \
		likwid-pin        \
		likwid-bench


CPPFLAGS := $(CPPFLAGS) $(DEFINES) $(INCLUDES)

all: $(BUILD_DIR) $(PERFMONHEADERS) $(OBJ) $(OBJ_BENCH) $(EXT_TARGETS) $(STATIC_TARGET_LIB) $(DYNAMIC_TARGET_LIB) $(FORTRAN_INTERFACE)  $(PINLIB)  $(DAEMON_TARGET)

tags:
	@echo "===>  GENERATE  TAGS"
	$(Q)ctags -R

docs:
	@echo "===>  GENERATE DOXYGEN DOCS"
	$(Q)doxygen doc/Doxyfile

$(APPS):  $(addprefix $(SRC_DIR)/applications/,$(addsuffix  .c,$(APPS))) $(BUILD_DIR) $(GENGROUPLOCK)  $(OBJ) $(OBJ_BENCH)
	@echo "===>  LINKING  $@"
	$(Q)${CC} $(DEBUG_FLAGS) $(CFLAGS) $(ANSI_CFLAGS) $(CPPFLAGS) ${LFLAGS} -o $@  $(addprefix $(SRC_DIR)/applications/,$(addsuffix  .c,$@)) $(OBJ_BENCH) $(TARGET_LIB) $(LIBHWLOC) $(LIBS)

$(STATIC_TARGET_LIB): $(OBJ)
	@echo "===>  CREATE STATIC LIB  $(STATIC_TARGET_LIB)"
	$(Q)${AR} -crus $(STATIC_TARGET_LIB) $(OBJ) $(LIBHWLOC) $(LIBLUA)


$(DYNAMIC_TARGET_LIB): $(OBJ)
	@echo "===>  CREATE SHARED LIB  $(DYNAMIC_TARGET_LIB)"
	$(Q)${CC} $(DEBUG_FLAGS) $(SHARED_LFLAGS) $(SHARED_CFLAGS) -o $(DYNAMIC_TARGET_LIB) $(OBJ) $(LIBS) $(LIBHWLOC) $(LIBLUA)

$(DAEMON_TARGET): $(SRC_DIR)/access-daemon/accessDaemon.c
	@echo "===>  Build access daemon likwid-accessD"
	$(Q)$(MAKE) -C  $(SRC_DIR)/access-daemon

$(BUILD_DIR):
	@mkdir $(BUILD_DIR)

$(PINLIB):
	@echo "===>  CREATE LIB  $(PINLIB)"
	$(Q)$(MAKE) -s -C src/pthread-overload/ $(PINLIB)

$(GENGROUPLOCK): $(foreach directory,$(shell ls $(GROUP_DIR)), $(wildcard $(GROUP_DIR)/$(directory)/*.txt))
	@echo "===>  GENERATE GROUP HEADERS"
	$(Q)$(GEN_GROUPS) ./groups  $(BUILD_DIR) ./perl/templates
	$(Q)touch $(GENGROUPLOCK)

$(FORTRAN_INTERFACE): $(SRC_DIR)/likwid.f90
	@echo "===>  COMPILE FORTRAN INTERFACE  $@"
	$(Q)$(FC) -c  $(FCFLAGS) $<
	@rm -f likwid.o

$(EXT_TARGETS):
	@echo "===>  ENTER  $@"
	$(Q)$(MAKE) --no-print-directory -C $@ $(MAKECMDGOALS)

#PATTERN RULES
$(BUILD_DIR)/%.o:  %.c
	@echo "===>  COMPILE  $@"
	$(Q)$(CC) -c $(DEBUG_FLAGS) $(CFLAGS) $(ANSI_CFLAGS) $(CPPFLAGS) $< -o $@
	$(Q)$(CC) $(DEBUG_FLAGS) $(CPPFLAGS) -MT $(@:.d=.o) -MM  $< > $(BUILD_DIR)/$*.d

$(BUILD_DIR)/%.o:  %.cc
	@echo "===>  COMPILE  $@"
	$(Q)$(CXX) -c $(DEBUG_FLAGS) $(CXXFLAGS) $(CPPFLAGS) $< -o $@
	$(Q)$(CXX) $(DEBUG_FLAGS) $(CXXFLAGS) $(CPPFLAGS) -MT $(@:.d=.o) -MM  $< > $(BUILD_DIR)/$*.d


$(BUILD_DIR)/%.pas:  $(BENCH_DIR)/%.ptt
	@echo "===>  GENERATE BENCHMARKS"
	$(Q)$(GEN_PAS)  $(BENCH_DIR) $(BUILD_DIR) ./perl/templates

$(BUILD_DIR)/%.h:  $(SRC_DIR)/includes/%.txt
	@echo "===>  GENERATE HEADER $@"
	$(Q)$(GEN_PMHEADER) $< $@

$(BUILD_DIR)/%.o:  $(BUILD_DIR)/%.pas
	@echo "===>  ASSEMBLE  $@"
	$(Q)$(PAS) -i $(PASFLAGS) -o $(BUILD_DIR)/$*.s $<  '$(DEFINES)'
	$(Q)$(AS) $(ASFLAGS)  $(BUILD_DIR)/$*.s -o $@

ifeq ($(findstring $(MAKECMDGOALS),clean),)
-include $(OBJ:.o=.d)
endif

.PHONY: clean distclean install uninstall $(EXT_TARGETS)


.PRECIOUS: $(BUILD_DIR)/%.pas

.NOTPARALLEL:


clean: $(EXT_TARGETS)
	@echo "===>  CLEAN"
	@rm -rf $(BUILD_DIR)
	@rm -f $(GENGROUPLOCK)

distclean: clean
	@echo "===>  DIST CLEAN"
	@rm -f likwid-*
	@rm -f $(STATIC_TARGET_LIB)
	@rm -f $(DYNAMIC_TARGET_LIB)
	@rm -f $(FORTRAN_INTERFACE)
	@rm -f $(PINLIB)
	@rm -rf doc/html
	@rm -f tags

install:
	@echo "===> INSTALL applications to $(PREFIX)/bin"
	@mkdir -p $(PREFIX)/bin
	@cp -f likwid-*  $(PREFIX)/bin
	@cp -f perl/feedGnuplot  $(PREFIX)/bin
	@cp -f perl/likwid-*  $(PREFIX)/bin
	@chmod 755 $(PREFIX)/bin/likwid-*
	@echo "===> INSTALL man pages to $(MANPREFIX)/man1"
	@mkdir -p $(MANPREFIX)/man1
	@sed -e "s/<VERSION>/$(VERSION)/g" -e "s/<DATE>/$(DATE)/g" < $(DOC_DIR)/likwid-topology.1 > $(MANPREFIX)/man1/likwid-topology.1
	@sed -e "s/<VERSION>/$(VERSION)/g" -e "s/<DATE>/$(DATE)/g" < $(DOC_DIR)/likwid-features.1 > $(MANPREFIX)/man1/likwid-features.1
	@sed -e "s/<VERSION>/$(VERSION)/g" -e "s/<DATE>/$(DATE)/g" < $(DOC_DIR)/likwid-perfctr.1 > $(MANPREFIX)/man1/likwid-perfctr.1
	@sed -e "s/<VERSION>/$(VERSION)/g" -e "s/<DATE>/$(DATE)/g" < $(DOC_DIR)/likwid-powermeter.1 > $(MANPREFIX)/man1/likwid-powermeter.1
	@sed -e "s/<VERSION>/$(VERSION)/g" -e "s/<DATE>/$(DATE)/g" < $(DOC_DIR)/likwid-pin.1 > $(MANPREFIX)/man1/likwid-pin.1
	@chmod 644 $(MANPREFIX)/man1/likwid-*
	@echo "===> INSTALL headers to $(PREFIX)/include"
	@mkdir -p $(PREFIX)/include
	@cp -f src/includes/likwid*.h  $(PREFIX)/include/
	$(FORTRAN_INSTALL)
	@echo "===> INSTALL libraries to $(PREFIX)/lib"
	@mkdir -p $(PREFIX)/lib
	@cp -f liblikwid*  $(PREFIX)/lib
	@chmod 755 $(PREFIX)/lib/$(PINLIB)
	@echo "===> INSTALL filters to $(LIKWIDFILTERPATH)"
	@mkdir -p $(LIKWIDFILTERPATH)
	@cp -f filters/*  $(LIKWIDFILTERPATH)
	@chmod 755 $(LIKWIDFILTERPATH)/*

uninstall:
	@echo "===> REMOVING applications from $(PREFIX)/bin"
	@rm -f $(addprefix $(PREFIX)/bin/,$(APPS))
	@rm -f $(PREFIX)/bin/likwid-mpirun
	@rm -f $(PREFIX)/bin/likwid-perfscope
	@rm -f $(PREFIX)/bin/feedGnuplot
	@echo "===> REMOVING man pages from $(MANPREFIX)/man1"
	@rm -f $(addprefix $(MANPREFIX)/man1/,$(addsuffix  .1,$(APPS)))
	@echo "===> REMOVING libs from $(PREFIX)/lib"
	@rm -f $(PREFIX)/lib/liblikwid*
	@echo "===> REMOVING filter from $(PREFIX)/share"
	@rm -rf  $(PREFIX)/share/likwid



