#
# =======================================================================================
#
#      Filename:  Makefile
#
#      Description:  likwid-bench Makefile
#
#      Version:   <VERSION>
#      Released:  <DATE>
#
#      Author:  Jan Treibig (jt), jan.treibig@gmail.com
#      Project:  likwid
#
#      Copyright (C) 2016 Jan Treibig
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
MAKE_DIR    = ../make

#DO NOT EDIT BELOW
# Dependency chains:
# *.[ch] -> *.o -> executables
# *.ptt -> *.pas -> *.s -> *.o -> executables
# *.txt -> *.h (generated)

include ../config.mk
include $(MAKE_DIR)/include_$(COMPILER).mk
include $(MAKE_DIR)/config_checks.mk
include $(MAKE_DIR)/config_defines.mk

#INCLUDES  += -I./includes -I../src/includes -I../ext/hwloc/include -I../$(COMPILER) -I$(BUILD_DIR)
INCLUDES  += -I./includes -I$(BUILD_DIR) -I../src/includes
LIBS      +=
CFLAGS := $(filter-out -fvisibility=hidden, $(CFLAGS))

#CONFIGURE BUILD SYSTEM
BUILD_DIR  = ./$(COMPILER)
Q         ?= @

BENCH_DIR   = ./x86-64
ifeq ($(COMPILER),MIC)
BENCH_DIR   = ./phi
endif
ifeq ($(COMPILER),GCCX86)
BENCH_DIR   = ./x86
endif
ifeq ($(COMPILER),GCCARMv7)
BENCH_DIR   = ./armv7
endif
ifeq ($(COMPILER),GCCARMv8)
BENCH_DIR   = ./armv8
endif



SHARED_TARGET_LIB := -L.. -L../ext/hwloc/ -L$(LUA_LIB_DIR) -llikwid -llikwid-hwloc -l$(LUA_LIB_NAME)
STATIC_TARGET_LIB := ../liblikwid.a ../ext/hwloc/liblikwid-hwloc.a $(LUA_LIB_DIR)/lib$(LUA_LIB_NAME).a
TARGET_LIB = $(SHARED_TARGET_LIB)

BENCH_LIBS :=
ifeq ($(INSTRUMENT_BENCH),true)
	DEFINES += -DLIKWID_PERFMON
endif

ifeq ($(DEBUG),true)
DEBUG_FLAGS = -g
DEFINES += -DDEBUG_LIKWID
else
DEBUG_FLAGS =
endif

VPATH     = $(SRC_DIR)
OBJ       = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o,$(wildcard $(SRC_DIR)/*.c))
ifeq ($(SHARED_LIBRARY),false)
OBJ := $(filter-out $(BUILD_DIR)/bstrlib.o,$(OBJ))
TARGET_LIB = $(STATIC_TARGET_LIB)
endif
OBJ_BENCH  =  $(patsubst $(BENCH_DIR)/%.ptt, $(BUILD_DIR)/%.o,$(wildcard $(BENCH_DIR)/*.ptt))
BENCH = $(shell basename $(BENCH_TARGET))

CPPFLAGS := $(CPPFLAGS) $(DEFINES) $(INCLUDES)

all: $(BUILD_DIR) $(OBJ) $(OBJ_BENCH) $(BENCH_TARGET)


$(BENCH_TARGET): $(BENCH)
$(BENCH): likwid-bench.c $(BUILD_DIR) $(OBJ) $(OBJ_BENCH)
	@echo "===>  LINKING  $(BENCH)"
	$(Q)${CC} $(DEBUG_FLAGS) $(CFLAGS) $(ANSI_CFLAGS) $(CPPFLAGS) ${LFLAGS}  likwid-bench.c $(BENCH_LIBS) $(OBJ_BENCH) $(OBJ) -o $(BENCH) $(TARGET_LIB) $(LIBS) $(RPATHS)


$(BUILD_DIR):
	@mkdir $(BUILD_DIR)


#PATTERN RULES
$(BUILD_DIR)/%.o:  %.c
	@echo "===>  COMPILE C $@"
	$(Q)$(CC) -g -c $(DEBUG_FLAGS) $(CFLAGS) $(ANSI_CFLAGS) $(CPPFLAGS) $(TARGET_LIB) $< -o $@
	$(Q)$(CC) -g $(DEBUG_FLAGS) $(CPPFLAGS) -MT $(@:.d=.o) -MM  $< > $(BUILD_DIR)/$*.d


$(BUILD_DIR)/%.pas:  $(BENCH_DIR)/%.ptt
	@echo "===>  GENERATE BENCHMARKS"
	$(Q)$(GEN_PAS) $(BENCH_DIR) $(BUILD_DIR) ./perl/templates


$(BUILD_DIR)/%.o:  $(BUILD_DIR)/%.pas
	@echo "===>  ASSEMBLE  $@"
	$(Q)$(PAS) -i $(PASFLAGS) -o $(BUILD_DIR)/$*.s $<  '$(DEFINES)'
	$(Q)$(AS) $(ASFLAGS)  $(BUILD_DIR)/$*.s -o $@

ifeq ($(findstring $(MAKECMDGOALS),clean),)
-include $(OBJ:.o=.d)
endif

.PHONY: clean distclean install uninstall

.PRECIOUS: $(BUILD_DIR)/%.pas

.NOTPARALLEL:

clean:
	@rm -rf likwid-bench

distclean:
	@rm -rf $(BUILD_DIR)
	@rm -rf likwid-bench

install:
	@echo "===> INSTALL applications to $(BINPREFIX)"
	cp -f likwid-bench $(BINPREFIX)
	@echo "===> INSTALL man pages to $(MANPREFIX)/man1"
	@mkdir -p $(MANPREFIX)/man1
	@sed -e "s/<VERSION>/$(VERSION)/g" -e "s/<DATE>/$(DATE)/g" < $(DOC_DIR)/likwid-bench.1 > $(MANPREFIX)/man1/likwid-bench.1


uninstall:
	@echo "===> REMOVING applications from $(BINPREFIX)"
	rm -rf $(BINPREFIX)/likwid-bench
	@echo "===> REMOVING man pages from $(MANPREFIX)/man1"
	@rm -f $(MANPREFIX)/man1/likwid-bench.1

