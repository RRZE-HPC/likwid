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
#      Copyright (C) 2014 Jan Treibig
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

#DO NOT EDIT BELOW

# determine kernel Version
KERNEL_VERSION_MAJOR := $(shell uname -r | awk '{split($$1,a,"."); print a[1]}' | cut -d '-' -f1)
KERNEL_VERSION := $(shell uname -r | awk  '{split($$1,a,"."); print a[2]}' | cut -d '-' -f1)
KERNEL_VERSION_MINOR := $(shell uname -r | awk '{split($$1,a,"."); print a[3]}' | cut -d '-' -f1)

HAS_MEMPOLICY = $(shell if [ $(KERNEL_VERSION) -lt 7 -a $(KERNEL_VERSION_MAJOR) -lt 3 -a $(KERNEL_VERSION_MINOR) -lt 7 ]; then \
               echo 0;  else echo 1; \
			   fi; )

HAS_RDTSCP = $(shell  /bin/bash -c "cat /proc/cpuinfo | grep -c rdtscp")

# determine glibc Version
GLIBC_VERSION := $(shell ldd --version | grep ldd |  awk '{ print $$NF }' | awk -F. '{ print $$2 }')

HAS_SCHEDAFFINITY = $(shell if [ $(GLIBC_VERSION) -lt 4 ]; then \
               echo 0;  else echo 1; \
			   fi; )

# Dependency chains:
# *.[ch] -> *.o -> executables
# *.ptt -> *.pas -> *.s -> *.o -> executables
# *.txt -> *.h (generated)

include ./config.mk
include $(MAKE_DIR)/include_$(COMPILER).mk
INCLUDES  += -I./src/includes  -I$(BUILD_DIR)
LIBS      +=
DEFINES   += -DVERSION=$(VERSION)         \
		 -DRELEASE=$(RELEASE)                 \
		 -DCFGFILE=$(CFG_FILE_PATH)           \
		 -DMAX_NUM_THREADS=$(MAX_NUM_THREADS) \
		 -DMAX_NUM_NODES=$(MAX_NUM_NODES)     \
		 -DHASH_TABLE_SIZE=$(HASH_TABLE_SIZE) \
		 -DLIBLIKWIDPIN=$(LIBLIKWIDPIN)       \
		 -DLIKWIDFILTERPATH=$(LIKWIDFILTERPATH)

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

LIKWID_LIB = liblikwid
ifeq ($(SHARED_LIBRARY),true)
CFLAGS += $(SHARED_CFLAGS) -ggdb
DYNAMIC_TARGET_LIB := $(LIKWID_LIB).so
TARGET_LIB := $(DYNAMIC_TARGET_LIB)
LIBS += -L. -llikwid
SHARED_LFLAGS += -lm -lpthread
else
STATIC_TARGET_LIB := $(LIKWID_LIB).a
TARGET_LIB := $(STATIC_TARGET_LIB)
endif

ifneq ($(COLOR),NONE)
DEFINES += -DCOLOR=$(COLOR)
endif

ifneq ($(COMPILER),MIC)
    DAEMON_TARGET = likwid-accessD
else
    $(info Info: Compiling for Xeon Phi. Disabling build of likwid-accessD.);
endif

ifeq ($(INSTRUMENT_BENCH),true)
DEFINES += -DPERFMON
endif

ifeq ($(HAS_MEMPOLICY),1)
DEFINES += -DHAS_MEMPOLICY
else
$(info Kernel $(KERNEL_VERSION_MAJOR).$(KERNEL_VERSION).$(KERNEL_VERSION_MINOR) has no mempolicy support! First Linux kernel with memory policies has version 2.6.7);
endif

ifeq ($(HAS_RDTSCP),0)
$(info Building without RDTSCP timing support!);
else
ifneq ($(COMPILER),MIC)
DEFINES += -DHAS_RDTSCP
else
    $(info Info: Compiling for Xeon Phi. Disabling RDTSCP support.);
endif
endif

ifeq ($(HAS_SCHEDAFFINITY),1)
DEFINES += -DHAS_SCHEDAFFINITY
PINLIB  = liblikwidpin.so
else
$(info GLIBC version 2.$(GLIBC_VERSION) has no pthread_setaffinity_np support!);
PINLIB  =
endif

DEFINES += -DACCESSDAEMON=$(ACCESSDAEMON)

ifeq ($(ACCESSMODE),accessdaemon)
ifneq ($(COMPILER),MIC)
    DEFINES += -DACCESSMODE=1
else
    $(info Info: Compiling for Xeon Phi. Set accessmode to direct.);
    DEFINES += -DACCESSMODE=0
endif
else
    DEFINES += -DACCESSMODE=0
endif

SETFREQ_TARGET = likwid-setFreq

VPATH     = $(SRC_DIR)
OBJ       = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o,$(wildcard $(SRC_DIR)/*.c))
OBJ      += $(patsubst $(SRC_DIR)/%.s, $(BUILD_DIR)/%.o,$(wildcard $(SRC_DIR)/*.s))
OBJ      += $(patsubst $(SRC_DIR)/%.cc, $(BUILD_DIR)/%.o,$(wildcard $(SRC_DIR)/*.cc))
PERFMONHEADERS  = $(patsubst $(SRC_DIR)/includes/%.txt, $(BUILD_DIR)/%.h,$(wildcard $(SRC_DIR)/includes/*.txt))
OBJ_BENCH  =  $(patsubst $(BENCH_DIR)/%.ptt, $(BUILD_DIR)/%.o,$(wildcard $(BENCH_DIR)/*.ptt))

APPS      = likwid-perfctr    \
            likwid-features   \
            likwid-powermeter \
            likwid-memsweeper \
            likwid-topology   \
            likwid-genCfg     \
            likwid-pin        \
            likwid-bench

PERL_APPS = likwid-mpirun         \
            likwid-setFrequencies \
            likwid-perfscope

DAEMON_APPS = $(SETFREQ_TARGET) \
			$(DAEMON_TARGET)

CPPFLAGS := $(CPPFLAGS) $(DEFINES) $(INCLUDES)

ifneq ($(FORTRAN_INTERFACE),false)
HAS_FORTRAN_COMPILER = $(shell $(FC) --version 2>/dev/null || echo 'NOFORTRAN' )
ifeq ($(HAS_FORTRAN_COMPILER),NOFORTRAN)
FORTRAN_INTERFACE=
$(info Warning: You have selected the fortran interface in config.mk, but there seems to be no fortran compiler - not compiling it!)
else
FORTRAN_INTERFACE = likwid.mod
FORTRAN_INSTALL =  @cp -f likwid.mod  $(PREFIX)/include/
endif
else
FORTRAN_INTERFACE =
FORTRAN_INSTALL =
endif

all: $(BUILD_DIR) $(GENGROUPLOCK) $(PERFMONHEADERS) $(OBJ) $(OBJ_BENCH) $(TARGET_LIB) $(APPS) $(FORTRAN_INTERFACE)  $(PINLIB)  $(DAEMON_TARGET) $(SETFREQ_TARGET)

tags:
	@echo "===>  GENERATE  TAGS"
	$(Q)ctags -R

$(APPS):  $(addprefix $(SRC_DIR)/applications/,$(addsuffix  .c,$(APPS))) $(BUILD_DIR) $(GENGROUPLOCK)  $(OBJ) $(OBJ_BENCH)
	@echo "===>  LINKING  $@"
	$(Q)${CC} $(CFLAGS) $(ANSI_CFLAGS) $(CPPFLAGS) ${LFLAGS} -o $@  $(addprefix $(SRC_DIR)/applications/,$(addsuffix  .c,$@)) $(OBJ_BENCH) $(STATIC_TARGET_LIB) $(LIBS)

$(STATIC_TARGET_LIB): $(OBJ)
	@echo "===>  CREATE STATIC LIB  $(STATIC_TARGET_LIB)"
	$(Q)${AR} -crus $(STATIC_TARGET_LIB) $(OBJ)

$(DYNAMIC_TARGET_LIB): $(OBJ)
	@echo "===>  CREATE SHARED LIB  $(DYNAMIC_TARGET_LIB)"
	$(Q)${CC} $(SHARED_CFLAGS) -o $(DYNAMIC_TARGET_LIB) $(OBJ) -lm $(SHARED_LFLAGS)

$(DAEMON_TARGET): $(SRC_DIR)/access-daemon/accessDaemon.c
	@echo "===>  Build access daemon $(DAEMON_TARGET)"
	$(Q)$(MAKE) -s -C  $(SRC_DIR)/access-daemon $(DAEMON_TARGET)

$(SETFREQ_TARGET): $(SRC_DIR)/access-daemon/setFreq.c
	@echo "===>  Build frequency daemon $(SETFREQ_TARGET)"
	$(Q)$(MAKE) -s -C  $(SRC_DIR)/access-daemon $(SETFREQ_TARGET)

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

#PATTERN RULES
$(BUILD_DIR)/%.o:  %.c
	@echo "===>  COMPILE  $@"
	$(Q)$(CC) -c  $(CFLAGS) $(ANSI_CFLAGS) $(CPPFLAGS) $< -o $@
	$(Q)$(CC) $(CPPFLAGS) -MT $(@:.d=.o) -MM  $< > $(BUILD_DIR)/$*.d

$(BUILD_DIR)/%.o:  %.s
	@echo "===>  ASSEMBLE  $@"
	$(Q)$(AS) $(ASFLAGS)  $< -o $@

$(BUILD_DIR)/%.o:  %.cc
	@echo "===>  COMPILE  $@"
	$(Q)$(CXX) -c  $(CXXFLAGS) $(CPPFLAGS) $< -o $@
	$(Q)$(CXX) $(CXXFLAGS) $(CPPFLAGS) -MT $(@:.d=.o) -MM  $< > $(BUILD_DIR)/$*.d


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

.PHONY: clean distclean install uninstall

.PRECIOUS: $(BUILD_DIR)/%.pas

.NOTPARALLEL:


clean:
	@echo "===>  CLEAN"
	@rm -rf $(BUILD_DIR)
	@rm -f $(GENGROUPLOCK)

distclean: clean
	@echo "===>  DIST CLEAN"
	@rm -f likwid-*
	@rm -f $(LIKWID_LIB)*
	@rm -f $(FORTRAN_INTERFACE)
	@rm -f $(PINLIB)
	@rm -f tags

install:
	@echo "===> INSTALL applications to $(PREFIX)/bin"
	@mkdir -p $(PREFIX)/bin
	@for app in $(APPS); do \
		cp -f $$app $(PREFIX)/bin; \
	done
	@cp -f perl/feedGnuplot  $(PREFIX)/bin
	@for app in $(PERL_APPS); do \
		sed -e "s+<PREFIX>+$(PREFIX)+g" perl/$$app > $(PREFIX)/bin/$$app; \
	done
	@chmod 755 $(PREFIX)/bin/likwid-*
	@echo "===> INSTALL daemon applications to $(PREFIX)/bin"
	@mkdir -p $(PREFIX)/sbin
	@for app in $(DAEMON_APPS); do \
		cp -f $$app $(PREFIX)/sbin; \
	done
	@chmod 755 $(PREFIX)/sbin/likwid-*
	@echo "===> INSTALL man pages to $(MANPREFIX)/man1"
	@mkdir -p $(MANPREFIX)/man1
	@sed -e "s/<VERSION>/$(VERSION)/g" -e "s/<DATE>/$(DATE)/g" < $(DOC_DIR)/likwid-topology.1 > $(MANPREFIX)/man1/likwid-topology.1
	@sed -e "s/<VERSION>/$(VERSION)/g" -e "s/<DATE>/$(DATE)/g" < $(DOC_DIR)/likwid-features.1 > $(MANPREFIX)/man1/likwid-features.1
	@sed -e "s/<VERSION>/$(VERSION)/g" -e "s/<DATE>/$(DATE)/g" < $(DOC_DIR)/likwid-perfctr.1 > $(MANPREFIX)/man1/likwid-perfctr.1
	@sed -e "s/<VERSION>/$(VERSION)/g" -e "s/<DATE>/$(DATE)/g" < $(DOC_DIR)/likwid-powermeter.1 > $(MANPREFIX)/man1/likwid-powermeter.1
	@sed -e "s/<VERSION>/$(VERSION)/g" -e "s/<DATE>/$(DATE)/g" < $(DOC_DIR)/likwid-pin.1 > $(MANPREFIX)/man1/likwid-pin.1
	@sed -e "s/<VERSION>/$(VERSION)/g" -e "s/<DATE>/$(DATE)/g" < $(DOC_DIR)/likwid-setFrequencies.1 > $(MANPREFIX)/man1/likwid-setFrequencies.1
	@sed -e "s/<VERSION>/$(VERSION)/g" -e "s/<DATE>/$(DATE)/g" < $(DOC_DIR)/likwid-bench.1 > $(MANPREFIX)/man1/likwid-bench.1
	@chmod 644 $(MANPREFIX)/man1/likwid-*
	@echo "===> INSTALL headers to $(PREFIX)/include"
	@mkdir -p $(PREFIX)/include/likwid
	@cp -f src/includes/likwid*.h  $(PREFIX)/include/
	@cp -f src/includes/*  $(PREFIX)/include/likwid
	@cp -f GCC/perfmon_group_types.h  $(PREFIX)/include/likwid
	$(FORTRAN_INSTALL)
	@echo "===> INSTALL libraries to $(PREFIX)/lib"
	@mkdir -p $(PREFIX)/lib
	@cp -f $(LIKWID_LIB)*  $(PREFIX)/lib
	@chmod 755 $(PREFIX)/lib/$(PINLIB)
	@echo "===> INSTALL filters to $(LIKWIDFILTERPATH)"
	@mkdir -p $(LIKWIDFILTERPATH)
	@cp -f filters/*  $(LIKWIDFILTERPATH)
	@chmod 755 $(LIKWIDFILTERPATH)/*
	@chown root $(ACCESSDAEMON)
	@chmod u+s $(ACCESSDAEMON)

uninstall:
	@echo "===> REMOVING applications from $(PREFIX)/bin"
	@rm -f $(addprefix $(PREFIX)/bin/,$(APPS))
	@rm -f $(addprefix $(PREFIX)/bin/,$(PERL_APPS))
	@rm -f $(PREFIX)/bin/feedGnuplot
	@echo "===> REMOVING daemon applications from $(PREFIX)/sbin"
	@rm -f $(addprefix $(PREFIX)/sbin/,$(DAEMON_APPS))
	@echo "===> REMOVING man pages from $(MANPREFIX)/man1"
	@rm -f $(addprefix $(MANPREFIX)/man1/,$(addsuffix  .1,$(APPS)))
	@echo "===> REMOVING headers from $(PREFIX)/include"
	@rm -f $(PREFIX)/include/likwid*.h
	@echo "===> REMOVING libs from $(PREFIX)/lib"
	@rm -f $(PREFIX)/lib/$(LIKWID_LIB)*
	@echo "===> REMOVING filter from $(PREFIX)/share"
	@rm -rf  $(PREFIX)/share/likwid



