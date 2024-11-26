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
ifeq ($(COMPILER),GCCARM)
BENCH_DIR   = ./armv8
endif
ifeq ($(COMPILER),ARMCLANG)
BENCH_DIR   = ./armv8
endif
ifeq ($(COMPILER),FCC)
BENCH_DIR   = ./armv8
endif
ifeq ($(COMPILER),GCCPOWER)
BENCH_DIR   = ./power
endif



SHARED_TARGET_LIB := -L.. -L$(subst bench/ext,ext,$(HWLOC_LIB_DIR)) -L$(subst bench/ext,ext,$(LUA_LIB_DIR)) -llikwid -l$(HWLOC_LIB_NAME) -l$(LUA_LIB_NAME)
STATIC_TARGET_LIB := ../liblikwid.a $(HWLOC_LIB_DIR)/lib$(HWLOC_LIB_NAME).a $(LUA_LIB_DIR)/lib$(LUA_LIB_NAME).a
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

ifeq ($(strip $(SANITIZE)),true)
DEBUG_FLAGS += -fsanitize=address -fno-omit-frame-pointer
endif

VPATH     = $(SRC_DIR)
OBJ       = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o,$(wildcard $(SRC_DIR)/*.c))
ifeq ($(SHARED_LIBRARY),false)
OBJ := $(filter-out $(BUILD_DIR)/bstrlib.o,$(OBJ))
TARGET_LIB = $(STATIC_TARGET_LIB)
LIBS += -ldl
endif
PTT_BENCH  =  $(wildcard $(BENCH_DIR)/*.ptt)
OBJ_BENCH  =  $(patsubst $(BENCH_DIR)/%.ptt, $(BUILD_DIR)/%.o,$(PTT_BENCH))
PAS_BENCH  =  $(patsubst $(BENCH_DIR)/%.ptt, $(BUILD_DIR)/%.pas,$(PTT_BENCH))
BENCH = $(shell basename $(BENCH_TARGET))

CPPFLAGS := $(CPPFLAGS) $(DEFINES) $(INCLUDES)

all: $(BENCH_TARGET)


$(BENCH_TARGET): $(BENCH)
$(BENCH): likwid-bench.c $(OBJ) $(OBJ_BENCH)
	@echo "===>  LINKING  $(BENCH)"
	$(Q)$(CC) $(DEBUG_FLAGS) $(CFLAGS) $(ANSI_CFLAGS) $(CPPFLAGS) $(LFLAGS) -Wl,-z,noexecstack $^ $(BENCH_LIBS) -o $(BENCH) $(TARGET_LIB) $(LIBS) $(RPATHS)

#PATTERN RULES
$(BUILD_DIR)/%.o:  %.c
	@echo "===>  COMPILE C $@"
	@mkdir -p $(BUILD_DIR)
	$(Q)$(CC) -g -c $(DEBUG_FLAGS) $(CFLAGS) $(ANSI_CFLAGS) $(CPPFLAGS) $< -o $@
	$(Q)$(CC) -g $(DEBUG_FLAGS) $(CPPFLAGS) -MT $(@:.d=.o) -MM  $< > $(BUILD_DIR)/$*.d


# Note: GEN_PAS can only create *.pas from *.ptt all at once, so build the rule accordingly.
# The &: operator only works with GNU make 4.3+, so keep backwards compatibility.
ifneq ($(shell [ "$(MAKE_VERSION)" = "$(echo -e "$(MAKE_VERSION)\n4.3" | sort -V | head -n1)" ] && echo TOO_OLD || echo OK),TOO_OLD)
$(PAS_BENCH) &: $(PTT_BENCH)
else
.NOTPARALLEL: $(PAS_BENCH)
$(PAS_BENCH) : $(PTT_BENCH)
endif
	@echo "===>  GENERATE BENCHMARKS"
	@mkdir -p $(BUILD_DIR)
	$(Q)$(GEN_PAS) $(BENCH_DIR) $(BUILD_DIR) ./perl/templates

$(BUILD_DIR)/%.o:  $(BUILD_DIR)/%.pas
	@echo "===>  ASSEMBLE  $@"
	@mkdir -p $(BUILD_DIR)
	$(Q)$(PAS) -i $(PASFLAGS) -o $(BUILD_DIR)/$*.s $< '$(DEFINES)'
	$(Q)$(AS) $(ASFLAGS) $(BUILD_DIR)/$*.s -o $@

ifeq ($(findstring $(MAKECMDGOALS),clean),)
-include $(OBJ:.o=.d)
endif

.PHONY: clean distclean install uninstall

.PRECIOUS: $(BUILD_DIR)/%.pas

clean:
	@echo "===>  CLEAN BENCH"
	@rm -rf likwid-bench

distclean: clean
	@echo "===>  DIST CLEAN BENCH"
	@rm -rf $(BUILD_DIR)

install:
	@echo "===> INSTALL applications to $(BINPREFIX)"
	cp -f likwid-bench $(BINPREFIX)
	@echo "===> INSTALL man pages to $(MANPREFIX)/man1"
	@mkdir -p $(MANPREFIX)/man1
	@sed -e "s#<VERSION>#$(VERSION)#g" -e "s#<DATE>#$(DATE)#g" < $(DOC_DIR)/likwid-bench.1 > $(MANPREFIX)/man1/likwid-bench.1


uninstall:
	@echo "===> REMOVING applications from $(BINPREFIX)"
	rm -rf $(BINPREFIX)/likwid-bench
	@echo "===> REMOVING man pages from $(MANPREFIX)/man1"
	@rm -f $(MANPREFIX)/man1/likwid-bench.1

