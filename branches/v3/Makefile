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
#      Copyright (C) 2012 Jan Treibig 
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
BENCH_DIR   = ./bench
GROUP_DIR   = ./groups
FILTER_DIR  = ./filters
MAKE_DIR    = ./

#DO NOT EDIT BELOW

# determine kernel Version
KERNEL_VERSION := $(shell uname -r | awk -F- '{ print $$1 }' | awk -F. '{ print $$3 }')
KERNEL_VERSION_MAJOR := $(shell uname -r | awk -F- '{ print $$1 }' | awk -F. '{ print $$1 }')

HAS_MEMPOLICY = $(shell if [ $(KERNEL_VERSION) -lt 7 -a $(KERNEL_VERSION_MAJOR) -lt 3 ]; then \
               echo 0;  else echo 1; \
			   fi; )

# determine glibc Version
GLIBC_VERSION := $(shell ldd --version | grep ldd |  awk '{ print $$NF }' | awk -F. '{ print $$2 }')

HAS_SCHEDAFFINITY = $(shell if [ $(GLIBC_VERSION) -lt 4 ]; then \
               echo 0;  else echo 1; \
			   fi; )


# Dependency chains:
# *.[ch] -> *.o -> executables
# *.ptt -> *.pas -> *.s -> *.o -> executables
# *.txt -> *.h (generated)

include $(MAKE_DIR)/config.mk
include $(MAKE_DIR)/include_$(COMPILER).mk
INCLUDES  += -I./src/includes  -I$(BUILD_DIR)
DEFINES   += -DVERSION=$(VERSION)                 \
		 -DRELEASE=$(RELEASE)                 \
		 -DMAX_NUM_THREADS=$(MAX_NUM_THREADS) \
		 -DMAX_NUM_NODES=$(MAX_NUM_NODES) \
		 -DHASH_TABLE_SIZE=$(HASH_TABLE_SIZE) \
		 -DLIBLIKWIDPIN=$(LIBLIKWIDPIN)       \
		 -DLIKWIDFILTERPATH=$(LIKWIDFILTERPATH)

#CONFIGURE BUILD SYSTEM
BUILD_DIR  = ./$(COMPILER)
Q         ?= @
GENGROUPLOCK = .gengroup

ifeq ($(SHARED_LIBRARY),true)
CFLAGS += $(SHARED_CFLAGS)
DYNAMIC_TARGET_LIB := liblikwid.so
else
STATIC_TARGET_LIB := liblikwid.a
endif

ifneq ($(COLOR),NONE)
DEFINES += -DCOLOR=$(COLOR)
endif

ifeq ($(ENABLE_SNB_UNCORE),true)
DEFINES += -DSNB_UNCORE
endif

ifeq ($(BUILDDAEMON),true)
	DAEMON_TARGET = likwid-accessD
endif

ifeq ($(INSTRUMENT_BENCH),true)
DEFINES += -DPERFMON
endif

ifeq ($(HAS_MEMPOLICY),1)
DEFINES += -DHAS_MEMPOLICY
else
$(info echo "Kernel 2.6.$(KERNEL_VERSION) has no mempolicy support!");
endif

ifeq ($(HAS_SCHEDAFFINITY),1)
DEFINES += -DHAS_SCHEDAFFINITY
PINLIB  = liblikwidpin.so
else
$(info echo "GLIBC version 2.$(GLIBC_VERSION) has no pthread_setaffinity_np support!");
PINLIB  =
endif

DEFINES += -DACCESSDAEMON=$(ACCESSDAEMON)
DEFINES += -DSYSDAEMONSOCKETPATH=$(SYSDAEMONSOCKETPATH)

ifeq ($(ACCESSMODE),sysdaemon)
DEFINES += -DACCESSMODE=2
else
ifeq ($(ACCESSMODE),accessdaemon)
DEFINES += -DACCESSMODE=1
else
DEFINES += -DACCESSMODE=0
endif
endif

VPATH     = $(SRC_DIR)
OBJ       = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o,$(wildcard $(SRC_DIR)/*.c))
PERFMONHEADERS  = $(patsubst $(SRC_DIR)/includes/%.txt, $(BUILD_DIR)/%.h,$(wildcard $(SRC_DIR)/includes/*.txt))
ifeq ($(MAKECMDGOALS),likwid-bench)
OBJ      += $(patsubst $(BENCH_DIR)/%.ptt, $(BUILD_DIR)/%.o,$(wildcard $(BENCH_DIR)/*.ptt))
endif
APPS      = likwid-perfctr    \
		likwid-features   \
		likwid-powermeter \
		likwid-memsweeper \
		likwid-topology   \
		likwid-pin        \
		likwid-bench

CPPFLAGS := $(CPPFLAGS) $(DEFINES) $(INCLUDES) 

ifneq ($(FORTRAN_INTERFACE),)
HAS_FORTRAN_COMPILER = $(shell $(FC) --version 2>/dev/null || echo 'NOFORTRAN' )
ifeq ($(HAS_FORTRAN_COMPILER),NOFORTRAN)
FORTRAN_INTERFACE=
$(info Warning: You have selected the fortran interface in config.mk, but there seems to be no fortran compiler - not compiling it!)
else
FORTRAN_INSTALL =  @cp -f likwid.mod  $(PREFIX)/include/
endif
endif

all: $(BUILD_DIR) $(GENGROUPLOCK) $(PERFMONHEADERS) $(OBJ) $(filter-out likwid-bench,$(APPS)) $(STATIC_TARGET_LIB) $(DYNAMIC_TARGET_LIB) $(FORTRAN_INTERFACE)  $(PINLIB)  $(DAEMON_TARGET)

tags:
	@echo "===>  GENERATE  TAGS"
	$(Q)ctags -R

$(APPS):  $(addprefix $(SRC_DIR)/applications/,$(addsuffix  .c,$(APPS))) $(BUILD_DIR) $(GENGROUPLOCK)  $(OBJ)
	@echo "===>  LINKING  $@"
	$(Q)${CC} $(CFLAGS) $(ANSI_CFLAGS) $(CPPFLAGS) ${LFLAGS} -o $@  $(addprefix $(SRC_DIR)/applications/,$(addsuffix  .c,$@)) $(OBJ) $(LIBS)

$(STATIC_TARGET_LIB): $(OBJ)
	@echo "===>  CREATE STATIC LIB  $(STATIC_TARGET_LIB)"
	$(Q)${AR} -cq $(STATIC_TARGET_LIB) $(OBJ)

$(DYNAMIC_TARGET_LIB): $(OBJ)
	@echo "===>  CREATE SHARED LIB  $(DYNAMIC_TARGET_LIB)"
	$(Q)${CC} $(SHARED_LFLAGS) $(SHARED_CFLAGS) -o $(DYNAMIC_TARGET_LIB) $(OBJ)

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

#PATTERN RULES
$(BUILD_DIR)/%.o:  %.c
	@echo "===>  COMPILE  $@"
	$(Q)$(CC) -c  $(CFLAGS) $(ANSI_CFLAGS) $(CPPFLAGS) $< -o $@
	$(Q)$(CC) $(CPPFLAGS) -MT $(@:.d=.o) -MM  $< > $(BUILD_DIR)/$*.d

$(BUILD_DIR)/%.pas:  $(BENCH_DIR)/%.ptt
	@echo "===>  GENERATE BENCHMARKS"
	$(Q)$(GEN_PAS) ./bench  $(BUILD_DIR) ./perl/templates

$(BUILD_DIR)/%.h:  $(SRC_DIR)/includes/%.txt
	@echo "===>  GENERATE HEADER $@"
	$(Q)$(GEN_PMHEADER) $< $@

$(BUILD_DIR)/%.o:  $(BUILD_DIR)/%.pas
	@echo "===>  ASSEMBLE  $@"
	$(Q)$(PAS) -i x86-64 -o $(BUILD_DIR)/$*.s $<  '$(DEFINES)'
	$(Q)$(AS) $(ASFLAGS)  $(BUILD_DIR)/$*.s -o $@

ifeq ($(findstring $(MAKECMDGOALS),clean),)
-include $(OBJ:.o=.d)
endif

.PHONY: clean distclean install uninstall

.PRECIOUS: $(BUILD_DIR)/%.pas

.NOTPARALLEL:


clean:
	@echo "===>  CLEAN"
	@rm -rf $(BUILD_DIR)
	@rm -f likwid.o
	@rm -f $(GENGROUPLOCK)

distclean: clean
	@echo "===>  DIST CLEAN"
	@rm -f likwid-*
	@rm -f $(STATIC_TARGET_LIB)
	@rm -f $(DYNAMIC_TARGET_LIB)
	@rm -f $(FORTRAN_INTERFACE) 
	@rm -f $(PINLIB)
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



