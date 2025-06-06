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

BASE_DIR    = $(shell pwd)
SRC_DIR     = $(BASE_DIR)/src
DOC_DIR     = $(BASE_DIR)/doc
GROUP_DIR   = $(BASE_DIR)/groups
FILTER_DIR  = $(BASE_DIR)/filters
MAKE_DIR    = $(BASE_DIR)/make
EXAMPLES_DIR    = $(BASE_DIR)/examples

Q         ?= @

#DO NOT EDIT BELOW

# Dependency chains:
# *.[ch] -> *.o -> executables
# *.ptt -> *.pas -> *.s -> *.o -> executables
# *.txt -> *.h (generated)

include ./config.mk
include $(MAKE_DIR)/include_$(COMPILER).mk
include $(MAKE_DIR)/config_git.mk
include $(MAKE_DIR)/config_checks.mk
include $(MAKE_DIR)/config_defines.mk

INCLUDES  += -I./src/includes -I$(LUA_INCLUDE_DIR) -I$(HWLOC_INCLUDE_DIR) -I$(BUILD_DIR)
LIBS      += -ldl

#CONFIGURE BUILD SYSTEM
BUILD_DIR  = ./$(COMPILER)
GENGROUPLOCK = .gengroup

VPATH     = $(SRC_DIR)
OBJ       = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o,$(wildcard $(SRC_DIR)/*.c))
OBJ      += $(patsubst $(SRC_DIR)/%.cc, $(BUILD_DIR)/%.o,$(wildcard $(SRC_DIR)/*.cc))
OBJ      += $(patsubst $(SRC_DIR)/%.S, $(BUILD_DIR)/%.o,$(wildcard $(SRC_DIR)/*.S))
ifeq ($(FILTER_HWLOC_OBJ),yes)
OBJ := $(filter-out $(BUILD_DIR)/topology_hwloc.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/numa_hwloc.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/pci_hwloc.o,$(OBJ))
endif
ifneq ($(FORTRAN_INTERFACE),true)
OBJ := $(filter-out $(BUILD_DIR)/likwid_f90_interface.o,$(OBJ))
endif
ifeq ($(COMPILER), GCCARMv7)
OBJ := $(filter-out $(BUILD_DIR)/topology_cpuid.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/loadData.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_msr.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_pci.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_rdpmc.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_mmio.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_clientmem.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_translate.o,$(OBJ))
else
OBJ := $(filter-out $(BUILD_DIR)/loadDataARM.o,$(OBJ))
endif
ifeq ($(COMPILER), GCCARMv8)
OBJ := $(filter-out $(BUILD_DIR)/topology_cpuid.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/loadData.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_msr.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_pci.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_rdpmc.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_mmio.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_clientmem.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_translate.o,$(OBJ))
else
OBJ := $(filter-out $(BUILD_DIR)/loadDataARM.o,$(OBJ))
endif
ifeq ($(COMPILER), GCCARM)
OBJ := $(filter-out $(BUILD_DIR)/topology_cpuid.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/loadData.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_msr.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_pci.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_rdpmc.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_mmio.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_clientmem.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_translate.o,$(OBJ))
else
OBJ := $(filter-out $(BUILD_DIR)/loadDataARM.o,$(OBJ))
endif
ifeq ($(COMPILER), FCC)
OBJ := $(filter-out $(BUILD_DIR)/topology_cpuid.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/loadData.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_msr.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_pci.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_rdpmc.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_mmio.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_clientmem.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_translate.o,$(OBJ))
else
OBJ := $(filter-out $(BUILD_DIR)/loadDataARM.o,$(OBJ))
endif
ifeq ($(COMPILER), ARMCLANG)
OBJ := $(filter-out $(BUILD_DIR)/topology_cpuid.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/loadData.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_msr.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_pci.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_clientmem.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_rdpmc.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_translate.o,$(OBJ))
else
OBJ := $(filter-out $(BUILD_DIR)/loadDataARM.o,$(OBJ))
endif
ifeq ($(COMPILER), CLANGARMv8)
OBJ := $(filter-out $(BUILD_DIR)/topology_cpuid.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/loadData.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_msr.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_pci.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_rdpmc.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_mmio.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_clientmem.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_translate.o,$(OBJ))
else
OBJ := $(filter-out $(BUILD_DIR)/loadDataARM.o,$(OBJ))
endif
ifneq ($(NVIDIA_INTERFACE), true)
OBJ := $(filter-out $(BUILD_DIR)/nvmon.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/nvmon_nvml.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/topology_cuda.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/libnvctr.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/sysFeatures_nvml.o,$(OBJ))
endif
ifneq ($(ROCM_INTERFACE), true)
OBJ := $(filter-out $(BUILD_DIR)/rocmon.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/rocmon_marker.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/topology_rocm.o,$(OBJ))
endif
ifeq ($(COMPILER),GCCPOWER)
OBJ := $(filter-out $(BUILD_DIR)/topology_cpuid.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_msr.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_pci.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_clientmem.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_rdpmc.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_mmio.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_translate.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/loadData.o,$(OBJ))
endif
ifeq ($(COMPILER),XLC)
OBJ := $(filter-out $(BUILD_DIR)/topology_cpuid.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_msr.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_pci.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_clientmem.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_rdpmc.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_mmio.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/access_x86_translate.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/loadData.o,$(OBJ))
endif
PERFMONHEADERS  = $(patsubst $(SRC_DIR)/includes/%.txt, $(BUILD_DIR)/%.h,$(wildcard $(SRC_DIR)/includes/*.txt))
OBJ_LUA    =  $(wildcard ./ext/lua/$(COMPILER)/*.o)
OBJ_HWLOC  =  $(wildcard ./ext/hwloc/$(COMPILER)/*.o)
OBJ_GOTCHA = $(wildcard ./ext/GOTCHA/$(COMPILER)/*.o)
FILTERS := $(filter-out $(FILTER_DIR)/README,$(wildcard $(FILTER_DIR)/*))


L_APPS      =   likwid-perfctr \
				likwid-pin \
				likwid-powermeter \
				likwid-topology \
				likwid-memsweeper \
				likwid-mpirun \
				likwid-features \
				likwid-perfscope \
				likwid-genTopoCfg
C_APPS      =   bench/likwid-bench
L_HELPER    =   likwid.lua
ifeq ($(BUILDFREQ),true)
	L_APPS += likwid-setFrequencies
endif
ifeq ($(BUILD_SYSFEATURES),true)
	L_APPS += likwid-sysfeatures
	DEFINES += -DLIKWID_WITH_SYSFEATURES
else
SYSFEATURE_OBJ       = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o,$(wildcard $(SRC_DIR)/sysFeatures*.c))
OBJ := $(filter-out $(SYSFEATURE_OBJ), $(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/likwid_device.o,$(OBJ))
OBJ := $(filter-out $(BUILD_DIR)/devstring.o,$(OBJ))
endif

CPPFLAGS := $(CPPFLAGS) $(DEFINES) $(INCLUDES)

.PHONY: all
ifeq ($(BUILDDAEMON),false)
all: $(TARGET_LIB) $(FORTRAN_IF) $(PINLIB) $(L_APPS) $(L_HELPER) $(FREQ_TARGET) $(BENCH_TARGET) $(APPDAEMON_TARGET)
else
ifeq ($(BUILDFREQ),false)
all: $(TARGET_LIB) $(FORTRAN_IF) $(PINLIB) $(L_APPS) $(L_HELPER) $(DAEMON_TARGET) $(BENCH_TARGET) $(APPDAEMON_TARGET)
else
ifeq ($(CONTAINER_HELPER),false)
all: $(TARGET_LIB) $(FORTRAN_IF) $(PINLIB) $(L_APPS) $(L_HELPER) $(DAEMON_TARGET) $(FREQ_TARGET) $(BENCH_TARGET) $(APPDAEMON_TARGET)
else
all: $(TARGET_LIB) $(FORTRAN_IF) $(PINLIB) $(L_APPS) $(L_HELPER) $(DAEMON_TARGET) $(FREQ_TARGET) $(BENCH_TARGET) $(APPDAEMON_TARGET) $(CONTAINER_HELPER_TARGET)
endif
endif
endif

.PHONY: tags
tags:
	@echo "===>  GENERATE  TAGS"
	$(Q)ctags -R

.PHONY: docs
docs:
	@echo "===>  GENERATE DOXYGEN DOCS"
	@cp doc/lua-doxygen.md doc/lua-doxygen.md.safe
	@cp doc/likwid-doxygen.md doc/likwid-doxygen.md.safe
	@cp doc/Doxyfile doc/Doxyfile.safe
	@sed -i -e 's#<PREFIX>#$(PREFIX)#g' -e 's#<VERSION>#$(VERSION)#g' -e 's#<DATE>#$(DATE)#g' -e 's#<RELEASE>#$(RELEASE)#g' -e 's#<MINOR>#$(MINOR)#g' -e 's#<GITCOMMIT>#$(GITCOMMIT)#g' doc/lua-doxygen.md
	@sed -i -e 's#<PREFIX>#$(PREFIX)#g' -e 's#<VERSION>#$(VERSION)#g' -e 's#<DATE>#$(DATE)#g' -e 's#<RELEASE>#$(RELEASE)#g' -e 's#<MINOR>#$(MINOR)#g' -e 's#<GITCOMMIT>#$(GITCOMMIT)#g' doc/likwid-doxygen.md
	@sed -i -e 's#<PREFIX>#$(PREFIX)#g' -e 's#<VERSION>#$(VERSION)#g' -e 's#<DATE>#$(DATE)#g' -e 's#<RELEASE>#$(RELEASE)#g' -e 's#<MINOR>#$(MINOR)#g' -e 's#<GITCOMMIT>#$(GITCOMMIT)#g' doc/Doxyfile
	$(Q)doxygen doc/Doxyfile
	@mv doc/lua-doxygen.md.safe doc/lua-doxygen.md
	@mv doc/likwid-doxygen.md.safe doc/likwid-doxygen.md
	@mv doc/Doxyfile.safe doc/Doxyfile

$(L_APPS):  $(addprefix $(SRC_DIR)/applications/,$(addsuffix  .lua,$(L_APPS)))
	@echo "===>  ADJUSTING  $@"
	@if [ "$(ACCESSMODE)" = "direct" ]; then sed -i -e s#"access_mode = 1"#"access_mode = 0"#g $(SRC_DIR)/applications/$@.lua;fi
	@sed -e s#'<INSTALLED_BINPREFIX>'#$(subst /,\\/,$(INSTALLED_BINPREFIX))#g \
		-e s#'<INSTALLED_LIBPREFIX>'#$(subst /,\\/,$(INSTALLED_LIBPREFIX))#g \
		-e s#'<INSTALLED_PREFIX>'#$(subst /,\\/,$(INSTALLED_PREFIX))#g \
		-e s#'<VERSION>'#$(VERSION).$(RELEASE).$(MINOR)#g \
		-e s#'<DATE>'#$(DATE)#g \
		-e s#'<RELEASE>'#$(RELEASE)#g \
		-e s#'<MINOR>'#$(MINOR)#g \
		-e s#'<GITCOMMIT>'#$(GITCOMMIT)#g \
		$(addprefix $(SRC_DIR)/applications/,$(addsuffix  .lua,$@)) > $@
	@if [ "$(LUA_INTERNAL)" = "false" ]; then \
		sed -i -e s#"$(subst /,\\/,$(INSTALLED_BINPREFIX))/likwid-lua"#"$(LUA_BIN)/$(LUA_LIB_NAME)"# $@; \
	fi
	@if [ "$(ACCESSMODE)" = "direct" ]; then sed -i -e s#"access_mode = 0"#"access_mode = 1"#g $(SRC_DIR)/applications/$@.lua;fi

$(L_HELPER):
	@echo "===>  ADJUSTING  $@"
	@sed -e s#'<PREFIX>'#$(subst /,\\/,$(PREFIX))#g \
		-e s#'<INSTALLED_LIBPREFIX>'#$(subst /,\\/,$(INSTALLED_LIBPREFIX))#g \
		-e s#'<INSTALLED_PREFIX>'#$(subst /,\\/,$(INSTALLED_PREFIX))#g \
		-e s#'<LIKWIDGROUPPATH>'#$(subst /,\\/,$(LIKWIDGROUPPATH))#g \
		-e s#'<LIBLIKWIDPIN>'#$(subst /,\\/,$(LIBLIKWIDPIN))#g \
		-e s#'<VERSION>'#$(VERSION)#g \
		-e s#'<RELEASE>'#$(RELEASE)#g \
		-e s#'<MINOR>'#$(MINOR)#g \
		-e s#'<GITCOMMIT>'#$(GITCOMMIT)#g \
		$(SRC_DIR)/applications/$@ > $@

$(STATIC_TARGET_LIB): $(OBJ) $(TARGET_HWLOC_LIB) $(TARGET_LUA_LIB)
	@echo "===>  CREATE STATIC LIB  $(TARGET_LIB)"
	$(Q)$(AR) -crs $(STATIC_TARGET_LIB) $(OBJ) $(TARGET_HWLOC_LIB) $(TARGET_LUA_LIB)
	@sed -e s#'@PREFIX@'#$(INSTALLED_PREFIX)#g \
		-e s#'@NVIDIA_INTERFACE@'#$(NVIDIA_INTERFACE)#g \
		-e s#'@FORTRAN_INTERFACE@'#$(FORTRAN_INTERFACE)#g \
		-e s#'@LIBPREFIX@'#$(INSTALLED_LIBPREFIX)#g \
		-e s#'@BINPREFIX@'#$(INSTALLED_BINPREFIX)#g \
		make/likwid-config.cmake > likwid-config.cmake

$(DYNAMIC_TARGET_LIB): $(OBJ) $(TARGET_HWLOC_LIB) $(TARGET_LUA_LIB)
	@echo "===>  CREATE SHARED LIB  $(TARGET_LIB)"
	$(Q)$(CC) $(DEBUG_FLAGS) $(SHARED_LFLAGS) -Wl,-soname,$(TARGET_LIB).$(VERSION).$(RELEASE),--no-undefined $(SHARED_CFLAGS) -o $@ $^ $(LIBS) $(RPATHS)
	@ln -sf $(TARGET_LIB) $(TARGET_LIB).$(VERSION).$(RELEASE)
	@sed -e s#'@PREFIX@'#$(INSTALLED_PREFIX)#g \
		-e s#'@NVIDIA_INTERFACE@'#$(NVIDIA_INTERFACE)#g \
		-e s#'@ROCM_INTERFACE@'#$(ROCM_INTERFACE)#g \
		-e s#'@FORTRAN_INTERFACE@'#$(FORTRAN_INTERFACE)#g \
		-e s#'@LIBPREFIX@'#$(INSTALLED_LIBPREFIX)#g \
		-e s#'@BINPREFIX@'#$(INSTALLED_BINPREFIX)#g \
		make/likwid-config.cmake > likwid-config.cmake

$(DAEMON_TARGET): $(SRC_DIR)/access-daemon/accessDaemon.c
	@echo "===>  BUILD access daemon likwid-accessD"
	$(Q)$(MAKE) --no-print-directory -C  $(SRC_DIR)/access-daemon ../../likwid-accessD

$(FREQ_TARGET): $(SRC_DIR)/access-daemon/setFreqDaemon.c
	@echo "===>  BUILD frequency daemon likwid-setFreq"
	$(Q)$(MAKE) --no-print-directory -C  $(SRC_DIR)/access-daemon ../../likwid-setFreq

$(APPDAEMON_TARGET): $(SRC_DIR)/access-daemon/appDaemon.c $(TARGET_LIB) $(TARGET_GOTCHA_LIB)
	@echo "===>  BUILD application interface likwid-appDaemon.so"
	$(Q)$(MAKE) --no-print-directory -C  $(SRC_DIR)/access-daemon ../../likwid-appDaemon.so

$(CONTAINER_HELPER_TARGET): $(SRC_DIR)/bridge/bridge.c
	@echo "===>  BUILD container helper likwid-bridge"
	$(Q)$(CC) $(DEBUG_FLAGS) $(CFLAGS) $(ANSI_CFLAGS) $(CPPFLAGS) $(SRC_DIR)/bridge/bridge.c -o $@

$(PINLIB):
	@echo "===>  CREATE LIB  $(PINLIB)"
	$(Q)$(MAKE) --no-print-directory -C src/pthread-overload/ ../../$(PINLIB)

$(GENGROUPLOCK): $(foreach directory,$(shell ls $(GROUP_DIR)), $(wildcard $(GROUP_DIR)/$(directory)/*.txt))
	@echo "===>  GENERATE GROUP HEADERS"
	@mkdir -p $(BUILD_DIR)
	$(Q)$(GEN_GROUPS) ./groups  $(BUILD_DIR) ./perl/templates
	$(Q)touch $(GENGROUPLOCK)

$(FORTRAN_IF): $(SRC_DIR)/likwid.F90
	@echo "===>  COMPILE FORTRAN INTERFACE  $@"
	$(Q)$(FC) -c  $(FCFLAGS) $<
	@rm -f likwid.o

ifeq ($(LUA_INTERNAL),true)
# we usually don't change lua, so don't unconditionally rebuild it
#.PHONY: $(TARGET_LUA_LIB)
$(TARGET_LUA_LIB):
	@echo "===>  ENTER  $(LUA_FOLDER)"
	$(Q)$(MAKE) --no-print-directory -C $(LUA_FOLDER)
else
$(TARGET_LUA_LIB):
	@echo "===>  EXTERNAL LUA"
endif

# we usually don't change GOTCHA, so don't unconditionally rebuild it
#.PHONY: $(TARGET_GOTCHA_LIB)
$(TARGET_GOTCHA_LIB):
	@echo "===>  ENTER  $(GOTCHA_FOLDER)"
	$(Q)$(MAKE) --no-print-directory -C $(GOTCHA_FOLDER)

ifeq ($(USE_INTERNAL_HWLOC),true)
# we usually don't change hwloc, so don't unconditionally rebuild it
#.PHONY: $(TARGET_HWLOC_LIB)
$(TARGET_HWLOC_LIB):
	@echo "===>  ENTER  $(HWLOC_FOLDER)"
	$(Q)$(MAKE) --no-print-directory -C $(HWLOC_FOLDER)
else
$(TARGET_HWLOC_LIB):
	@echo "===>  EXTERNAL HWLOC"
endif

.PHONY: $(BENCH_TARGET)
$(BENCH_TARGET): $(TARGET_LIB)
	@echo "===>  ENTER  $(BENCH_FOLDER)"
	$(Q)$(MAKE) --no-print-directory -C $(BENCH_FOLDER)

#PATTERN RULES
$(BUILD_DIR)/%.o: %.c
	@echo "===>  COMPILE  $@"
	@mkdir -p $(BUILD_DIR)
	$(Q)$(CC) -c $(DEBUG_FLAGS) $(CFLAGS) $(ANSI_CFLAGS) $(CPPFLAGS) $< -o $@
	$(Q)$(CC) $(DEBUG_FLAGS) $(CPPFLAGS) -MT $(@:.d=.o) -MM  $< > $(BUILD_DIR)/$*.d

# At the moment the perfmon code is too complex to fix all unused variable warnings, so supress those for now
$(BUILD_DIR)/perfmon.o: perfmon.c $(PERFMONHEADERS)
	@echo "===>  COMPILE  $@"
	@mkdir -p $(BUILD_DIR)
	$(Q)$(CC) -c $(DEBUG_FLAGS) $(CFLAGS) $(ANSI_CFLAGS) -Wno-unused-variable $(CPPFLAGS) $< -o $@
	$(Q)$(CC) $(DEBUG_FLAGS) $(CPPFLAGS) -MT $(@:.d=.o) -MM  $< > $(@:.o=.d)

# same here
$(BUILD_DIR)/intel_perfmon_uncore_discovery.o: intel_perfmon_uncore_discovery.c $(PERFMONHEADERS)
	@echo "===>  COMPILE  $@"
	@mkdir -p $(BUILD_DIR)
	$(Q)$(CC) -c $(DEBUG_FLAGS) $(CFLAGS) $(ANSI_CFLAGS) -Wno-unused-variable $(CPPFLAGS) $< -o $@
	$(Q)$(CC) $(DEBUG_FLAGS) $(CPPFLAGS) -MT $(@:.d=.o) -MM  $< > $(@:.o=.d)

$(BUILD_DIR)/rocmon_marker.o:  rocmon_marker.c
	@echo "===>  COMPILE  $@"
	@mkdir -p $(BUILD_DIR)
	$(Q)$(CC) -c $(DEBUG_FLAGS) $(CFLAGS) $(ANSI_CFLAGS) $(CPPFLAGS) $< -o $@
	$(Q)objcopy --redefine-sym HSA_VEN_AMD_AQLPROFILE_LEGACY_PM4_PACKET_SIZE=HSA_VEN_AMD_AQLPROFILE_LEGACY_PM4_PACKET_SIZE2 $@

$(BUILD_DIR)/%.o:  %.cc
	@echo "===>  COMPILE  $@"
	@mkdir -p $(BUILD_DIR)
	$(Q)$(CXX) -c $(DEBUG_FLAGS) $(CXXFLAGS) $(CPPFLAGS) $< -o $@
	$(Q)$(CXX) $(DEBUG_FLAGS) $(CXXFLAGS) $(CPPFLAGS) -MT $(@:.d=.o) -MM  $< > $(BUILD_DIR)/$*.d

$(BUILD_DIR)/%.o:  %.S
	@echo "===>  COMPILE  $@"
	@mkdir -p $(BUILD_DIR)
	$(Q)$(CPP) $(CPPFLAGS) $< -o $@.tmp
	$(Q)$(AS) $(ASFLAGS) $@.tmp -o $@
	@rm $@.tmp

# Keep generated headers. Because all sources (unfortunately) depend on the
# PERFMONHEADERS, they all get rebuilt if a single source file is rebuilt.
# That is because make usually cleans up those headers, which causes them to be
# all regenerated every time, thus causing all .c files to be rebuilt.
.PRECIOUS: $(PERFMONHEADERS)
$(BUILD_DIR)/%.h:  $(SRC_DIR)/includes/%.txt
	@echo "===>  GENERATE HEADER $@"
	@mkdir -p $(BUILD_DIR)
	$(Q)$(GEN_PMHEADER) $< $@

ifeq ($(findstring $(MAKECMDGOALS),clean),)
-include $(OBJ:.o=.d)
endif

.PHONY: clean
clean:
	@echo "===>  CLEAN"
	$(Q)$(MAKE) --no-print-directory -C $(LUA_FOLDER) $(MAKECMDGOALS)
	$(Q)$(MAKE) --no-print-directory -C $(HWLOC_FOLDER) $(MAKECMDGOALS)
	$(Q)$(MAKE) --no-print-directory -C $(GOTCHA_FOLDER) $(MAKECMDGOALS)
	$(Q)$(MAKE) --no-print-directory -C $(BENCH_FOLDER) $(MAKECMDGOALS)
	@rm -f $(L_APPS) likwid-sysfeatures likwid-setFrequencies
	@rm -f likwid.lua
	@rm -f $(STATIC_TARGET_LIB)
	@rm -f $(DYNAMIC_TARGET_LIB)*
	@rm -f $(PINLIB)*
	@rm -f $(FORTRAN_IF_NAME)
	@rm -f $(FREQ_TARGET) $(DAEMON_TARGET) $(APPDAEMON_TARGET) $(CONTAINER_HELPER_TARGET)
	@rm -f likwid-config.cmake

.PHONY: distclean
distclean: clean
	@echo "===>  DIST CLEAN"
	@rm -rf $(BUILD_DIR)
	@if [ "$(LUA_INTERNAL)" = "true" ]; then rm -f $(TARGET_LUA_LIB).* $(shell basename $(TARGET_LUA_LIB)).*; fi
	@if [ "$(USE_INTERNAL_HWLOC)" = "true" ]; then rm -f $(TARGET_HWLOC_LIB).* $(shell basename $(TARGET_HWLOC_LIB)).*; fi
	@rm -f $(TARGET_GOTCHA_LIB).* $(shell basename $(TARGET_GOTCHA_LIB)).*
	@rm -f $(GENGROUPLOCK)
	@rm -rf doc/html
	@rm -f tags

.PHONY: install_daemon move_daemon uninstall_daemon uninstall_daemon_moved
ifeq ($(BUILDDAEMON),true)
ifneq ($(COMPILER),MIC)
install_daemon:
	@echo "===> INSTALL access daemon to $(ACCESSDAEMON)"
	@mkdir -p `dirname $(ACCESSDAEMON)`
	install -m 4755 $(INSTALL_CHOWN) $(DAEMON_TARGET) $(ACCESSDAEMON)
move_daemon:
	@echo "===> MOVE access daemon from $(ACCESSDAEMON) to $(INSTALLED_ACCESSDAEMON)"
	@mkdir -p `dirname $(INSTALLED_ACCESSDAEMON)`
	@install -m 4755 $(INSTALL_CHOWN) $(ACCESSDAEMON) $(INSTALLED_ACCESSDAEMON)
uninstall_daemon:
	@echo "===> REMOVING access daemon from $(ACCESSDAEMON)"
	@rm -f $(ACCESSDAEMON)
uninstall_daemon_moved:
	@echo "===> REMOVING access daemon from $(INSTALLED_ACCESSDAEMON)"
	@rm -f $(INSTALLED_ACCESSDAEMON)
else
install_daemon:
	@echo "===> No INSTALL of the access daemon"
move_daemon:
	@echo "===> No MOVE of the access daemon"
uninstall_daemon:
	@echo "===> No UNINSTALL of the access daemon"
uninstall_daemon_moved:
	@echo "===> No UNINSTALL of the access daemon"
endif
else
install_daemon:
	@echo "===> No INSTALL of the access daemon"
move_daemon:
	@echo "===> No MOVE of the access daemon"
uninstall_daemon:
	@echo "===> No UNINSTALL of the access daemon"
uninstall_daemon_moved:
	@echo "===> No UNINSTALL of the access daemon"
endif

.PHONY: install_freq move_freq uninstall_freq uninstall_freq_moved
ifeq ($(BUILDFREQ),true)
ifneq ($(COMPILER),MIC)
install_freq:
	@echo "===> INSTALL setFrequencies tool to $(SBINPREFIX)/$(FREQ_TARGET)"
	@mkdir -p $(SBINPREFIX)
	@install -m 4755 $(INSTALL_CHOWN) $(FREQ_TARGET) $(SBINPREFIX)/$(FREQ_TARGET)
move_freq:
	@echo "===> MOVE setFrequencies tool from $(SBINPREFIX)/$(FREQ_TARGET) to $(INSTALLED_SBINPREFIX)/$(FREQ_TARGET)"
	@mkdir -p $(INSTALLED_SBINPREFIX)
	@install -m 4755 $(INSTALL_CHOWN) $(SBINPREFIX)/$(FREQ_TARGET) $(INSTALLED_SBINPREFIX)/$(FREQ_TARGET)
uninstall_freq:
	@echo "===> REMOVING setFrequencies tool from $(SBINPREFIX)/$(FREQ_TARGET)"
	@rm -f $(SBINPREFIX)/$(FREQ_TARGET)
uninstall_freq_moved:
	@echo "===> REMOVING setFrequencies tool from $(INSTALLED_SBINPREFIX)/$(FREQ_TARGET)"
	@rm -f $(INSTALLED_SBINPREFIX)/$(FREQ_TARGET)
else
install_freq:
	@echo "===> No INSTALL of setFrequencies tool"
move_freq:
	@echo "===> No MOVE of setFrequencies tool"
uninstall_freq:
	@echo "===> No UNINSTALL of setFrequencies tool"
uninstall_freq_moved:
	@echo "===> No UNINSTALL of setFrequencies tool"
endif
else
install_freq:
	@echo "===> No INSTALL of setFrequencies tool"
move_freq:
	@echo "===> No MOVE of setFrequencies tool"
uninstall_freq:
	@echo "===> No UNINSTALL of setFrequencies tool"
uninstall_freq_moved:
	@echo "===> No UNINSTALL of setFrequencies tool"
endif

.PHONY: install_appdaemon move_appdaemon uninstall_appdaemon uninstall_appdaemon_moved
ifeq ($(BUILDAPPDAEMON),true)
install_appdaemon:
	@echo "===> INSTALL application interface appDaemon to $(PREFIX)/lib/$(APPDAEMON_TARGET)"
	@mkdir -p $(PREFIX)/lib
	@install -m 755 $(APPDAEMON_TARGET) $(PREFIX)/lib/$(APPDAEMON_TARGET)
move_appdaemon:
	@echo "===> MOVE application interface appDaemon from $(PREFIX)/lib/$(APPDAEMON_TARGET) to $(INSTALLED_PREFIX)/lib/$(APPDAEMON_TARGET)"
	@mkdir -p $(INSTALLED_PREFIX)/lib
	@install -m 755 $(PREFIX)/lib/$(APPDAEMON_TARGET) $(INSTALLED_PREFIX)/lib/$(APPDAEMON_TARGET)
uninstall_appdaemon:
	@echo "===> REMOVING application interface appDaemon from $(PREFIX)/lib/$(APPDAEMON_TARGET)"
	@rm -f $(PREFIX)/lib/$(APPDAEMON_TARGET)
uninstall_appdaemon_moved:
	@echo "===> REMOVING application interface appDaemon from $(INSTALLED_PREFIX)/lib/$(APPDAEMON_TARGET)"
	@rm -f $(INSTALLED_PREFIX)/lib/$(APPDAEMON_TARGET)
else
install_appdaemon:
	@echo "===> No INSTALL of the application interface appDaemon"
move_appdaemon:
	@echo "===> No MOVE of the application interface appDaemon"
uninstall_appdaemon:
	@echo "===> No UNINSTALL of the application interface appDaemon"
uninstall_appdaemon_moved:
	@echo "===> No UNINSTALL of the application interface appDaemon"
endif

.PHONY: install_container_helper move_container_helper uninstall_container_helper uninstall_container_helper_moved
ifeq ($(CONTAINER_HELPER),true)
install_container_helper: $(CONTAINER_HELPER_TARGET)
	@echo "===> INSTALL container helper likwid-bridge to $(SBINPREFIX)/likwid-bridge"
	@mkdir -p $(SBINPREFIX)
	@install -m 755 $(CONTAINER_HELPER_TARGET) $(SBINPREFIX)/likwid-bridge
move_container_helper:
	@echo "===> MOVE container helper likwid-bridge from $(SBINPREFIX)/likwid-bridge to $(INSTALLED_SBINPREFIX)/likwid-bridge"
	@mkdir -p $(INSTALLED_SBINPREFIX)
	@install -m 755 $(SBINPREFIX)/$(CONTAINER_HELPER_TARGET) $(INSTALLED_SBINPREFIX)/$(CONTAINER_HELPER_TARGET)
uninstall_container_helper:
	@echo "===> REMOVING container helper likwid-bridge from $(SBINPREFIX)/$(CONTAINER_HELPER_TARGET)"
	@rm -f $(SBINPREFIX)/$(CONTAINER_HELPER_TARGET)
uninstall_container_helper_moved:
	@echo "===> REMOVING container helper likwid-bridge from $(INSTALLED_SBINPREFIX)/$(CONTAINER_HELPER_TARGET)"
	@rm -f $(INSTALLED_SBINPREFIX)/$(CONTAINER_HELPER_TARGET)
else
install_container_helper:
	@echo "===> No INSTALL of the container helper likwid-bridge"
move_container_helper:
	@echo "===> No MOVE of the container helper likwid-bridge"
uninstall_container_helper:
	@echo "===> No UNINSTALL of the container helper likwid-bridge"
uninstall_container_helper_moved:
	@echo "===> No UNINSTALL of the container helper likwid-bridge"
endif

.PHONY: install
install: install_daemon install_freq install_appdaemon install_container_helper
	@echo "===> INSTALL applications to $(BINPREFIX)"
	@mkdir -p $(BINPREFIX)
	@chmod 755 $(BINPREFIX)
	@for APP in $(L_APPS); do \
		install -m 755 $$APP  $(BINPREFIX); \
	done
	@for APP in $(C_APPS); do \
		install -m 755 $$APP  $(BINPREFIX); \
	done
	@if [ "$(LUA_INTERNAL)" = "true" ]; then \
		install -m 755 ext/lua/lua $(BINPREFIX)/$(LUA_LIB_NAME); \
	fi
	@echo "===> INSTALL helper applications to $(BINPREFIX)"
	@install -m 755 perl/feedGnuplot $(BINPREFIX)
	@echo "===> INSTALL lua to likwid interface to $(PREFIX)/share/lua"
	@mkdir -p $(PREFIX)/share/lua
	@chmod 755 $(PREFIX)/share/lua
	@install -m 644 likwid.lua $(PREFIX)/share/lua
	@echo "===> INSTALL libraries to $(LIBPREFIX)"
	@mkdir -p $(LIBPREFIX)
	@chmod 755 $(LIBPREFIX)
	@install -m 755 $(TARGET_LIB) $(LIBPREFIX)/$(TARGET_LIB).$(VERSION).$(RELEASE)
	@install -m 755 liblikwidpin.so $(LIBPREFIX)/liblikwidpin.so.$(VERSION).$(RELEASE)
	@if [ "$(USE_INTERNAL_HWLOC)" = "true" ]; then \
		install -m 755 $(TARGET_HWLOC_LIB) $(LIBPREFIX)/$(shell basename $(TARGET_HWLOC_LIB)).$(VERSION).$(RELEASE); \
	fi
	@if [ "$(LUA_INTERNAL)" = "true" ]; then \
		install -m 755 $(TARGET_LUA_LIB) $(LIBPREFIX)/$(shell basename $(TARGET_LUA_LIB)).$(VERSION).$(RELEASE); \
	fi
	@cd $(LIBPREFIX) && ln -fs $(TARGET_LIB).$(VERSION).$(RELEASE) $(TARGET_LIB)
	@cd $(LIBPREFIX) && ln -fs $(TARGET_LIB).$(VERSION).$(RELEASE) $(TARGET_LIB).$(VERSION)
	@cd $(LIBPREFIX) && ln -fs $(PINLIB).$(VERSION).$(RELEASE) $(PINLIB)
	@cd $(LIBPREFIX) && ln -fs $(PINLIB).$(VERSION).$(RELEASE) $(PINLIB).$(VERSION)
	@if [ "$(USE_INTERNAL_HWLOC)" = "true" ]; then \
		cd $(LIBPREFIX) && ln -fs $(shell basename $(TARGET_HWLOC_LIB)).$(VERSION).$(RELEASE) $(shell basename $(TARGET_HWLOC_LIB)); \
		cd $(LIBPREFIX) && ln -fs $(shell basename $(TARGET_HWLOC_LIB)).$(VERSION).$(RELEASE) $(shell basename $(TARGET_HWLOC_LIB)).$(VERSION); \
	fi
	@if [ "$(LUA_INTERNAL)" = "true" ]; then \
		cd $(LIBPREFIX) && ln -fs $(shell basename $(TARGET_LUA_LIB)).$(VERSION).$(RELEASE) $(shell basename $(TARGET_LUA_LIB)); \
		cd $(LIBPREFIX) && ln -fs $(shell basename $(TARGET_LUA_LIB)).$(VERSION).$(RELEASE) $(shell basename $(TARGET_LUA_LIB)).$(VERSION); \
	fi
	@if [ "$(BUILDAPPDAEMON)" = "true" ]; then \
		install -m 755 $(GOTCHA_FOLDER)/$(TARGET_GOTCHA_LIB) $(LIBPREFIX)/$(TARGET_GOTCHA_LIB).$(VERSION).$(RELEASE); \
		cd $(LIBPREFIX) && ln -fs $(shell basename $(TARGET_GOTCHA_LIB)).$(VERSION).$(RELEASE) $(shell basename $(TARGET_GOTCHA_LIB)); \
		cd $(LIBPREFIX) && ln -fs $(shell basename $(TARGET_GOTCHA_LIB)).$(VERSION).$(RELEASE) $(shell basename $(TARGET_GOTCHA_LIB)).$(VERSION); \
	fi
	@echo "===> INSTALL man pages to $(MANPREFIX)/man1"
	@mkdir -p $(MANPREFIX)/man1
	@chmod 755 $(MANPREFIX)/man1
	@sed -e "s#<VERSION>#$(VERSION)#g" -e "s#<DATE>#$(DATE)#g" -e "s#<GITCOMMIT>#$(GITCOMMIT)#g" -e "s#<MINOR>#$(MINOR)#g" < $(DOC_DIR)/likwid-topology.1 > $(MANPREFIX)/man1/likwid-topology.1
	@sed -e "s#<VERSION>#$(VERSION)#g" -e "s#<DATE>#$(DATE)#g" -e "s#<GITCOMMIT>#$(GITCOMMIT)#g" -e "s#<MINOR>#$(MINOR)#g" -e "s#<PREFIX>#$(PREFIX)#g" < $(DOC_DIR)/likwid-perfctr.1 > $(MANPREFIX)/man1/likwid-perfctr.1
	@sed -e "s#<VERSION>#$(VERSION)#g" -e "s#<DATE>#$(DATE)#g" -e "s#<GITCOMMIT>#$(GITCOMMIT)#g" -e "s#<MINOR>#$(MINOR)#g" < $(DOC_DIR)/likwid-powermeter.1 > $(MANPREFIX)/man1/likwid-powermeter.1
	@sed -e "s#<VERSION>#$(VERSION)#g" -e "s#<DATE>#$(DATE)#g" -e "s#<GITCOMMIT>#$(GITCOMMIT)#g" -e "s#<MINOR>#$(MINOR)#g" < $(DOC_DIR)/likwid-pin.1 > $(MANPREFIX)/man1/likwid-pin.1
	@sed -e "s#<VERSION>#$(VERSION)#g" -e "s#<DATE>#$(DATE)#g" -e "s#<GITCOMMIT>#$(GITCOMMIT)#g" -e "s#<MINOR>#$(MINOR)#g" < $(DOC_DIR)/feedGnuplot.1 > $(MANPREFIX)/man1/feedGnuplot.1
	@sed -e "s#<VERSION>#$(VERSION)#g" -e "s#<DATE>#$(DATE)#g" -e "s#<GITCOMMIT>#$(GITCOMMIT)#g" -e "s#<MINOR>#$(MINOR)#g" < $(DOC_DIR)/likwid-accessD.1 > $(MANPREFIX)/man1/likwid-accessD.1
	@sed -e "s#<VERSION>#$(VERSION)#g" -e "s#<DATE>#$(DATE)#g" -e "s#<GITCOMMIT>#$(GITCOMMIT)#g" -e "s#<MINOR>#$(MINOR)#g" < $(DOC_DIR)/likwid-genTopoCfg.1 > $(MANPREFIX)/man1/likwid-genTopoCfg.1
	@sed -e "s#<VERSION>#$(VERSION)#g" -e "s#<DATE>#$(DATE)#g" -e "s#<GITCOMMIT>#$(GITCOMMIT)#g" -e "s#<MINOR>#$(MINOR)#g" < $(DOC_DIR)/likwid-memsweeper.1 > $(MANPREFIX)/man1/likwid-memsweeper.1
	@sed -e "s#<VERSION>#$(VERSION)#g" -e "s#<DATE>#$(DATE)#g" -e "s#<GITCOMMIT>#$(GITCOMMIT)#g" -e "s#<MINOR>#$(MINOR)#g" < $(DOC_DIR)/likwid-mpirun.1 > $(MANPREFIX)/man1/likwid-mpirun.1
	@sed -e "s#<VERSION>#$(VERSION)#g" -e "s#<DATE>#$(DATE)#g" -e "s#<GITCOMMIT>#$(GITCOMMIT)#g" -e "s#<MINOR>#$(MINOR)#g" < $(DOC_DIR)/likwid-perfscope.1 > $(MANPREFIX)/man1/likwid-perfscope.1
	@sed -e "s#<VERSION>#$(VERSION)#g" -e "s#<DATE>#$(DATE)#g" -e "s#<GITCOMMIT>#$(GITCOMMIT)#g" -e "s#<MINOR>#$(MINOR)#g" < $(DOC_DIR)/likwid-setFreq.1 > $(MANPREFIX)/man1/likwid-setFreq.1
	@sed -e "s#<VERSION>#$(VERSION)#g" -e "s#<DATE>#$(DATE)#g" -e "s#<GITCOMMIT>#$(GITCOMMIT)#g" -e "s#<MINOR>#$(MINOR)#g" < $(DOC_DIR)/likwid-features.1 > $(MANPREFIX)/man1/likwid-features.1
	@sed -e "s#<VERSION>#$(VERSION)#g" -e "s#<DATE>#$(DATE)#g" -e "s#<GITCOMMIT>#$(GITCOMMIT)#g" -e "s#<MINOR>#$(MINOR)#g" < $(DOC_DIR)/likwid-bench.1 > $(MANPREFIX)/man1/likwid-bench.1
	@sed -e "s#<VERSION>#$(VERSION)#g" -e "s#<DATE>#$(DATE)#g" -e "s#<GITCOMMIT>#$(GITCOMMIT)#g" -e "s#<MINOR>#$(MINOR)#g" < $(DOC_DIR)/likwid-setFrequencies.1 > $(MANPREFIX)/man1/likwid-setFrequencies.1
	@sed -e "s#.TH LUA#.TH LIKWID-LUA#g" -e "s#lua - Lua interpreter#likwid-lua - Lua interpreter included in LIKWID#g" -e "s#.B lua#.B likwid-lua#g" -e "s#.BR luac (1)##g" $(DOC_DIR)/likwid-lua.1 > $(MANPREFIX)/man1/likwid-lua.1
	@if [ "$(BUILD_SYSFEATURES)" = "true" ]; then \
		sed -e "s#<VERSION>#$(VERSION)#g" -e "s#<DATE>#$(DATE)#g" -e "s#<GITCOMMIT>#$(GITCOMMIT)#g" -e "s#<MINOR>#$(MINOR)#g" < $(DOC_DIR)/likwid-sysfeatures.1 > $(MANPREFIX)/man1/likwid-sysfeatures.1; \
	fi
	@chmod 644 $(MANPREFIX)/man1/likwid-*
	@echo "===> INSTALL headers to $(PREFIX)/include"
	@mkdir -p $(PREFIX)/include
	@chmod 755 $(PREFIX)/include
	@install -m 644 $(SRC_DIR)/includes/likwid.h  $(PREFIX)/include/
	@sed -i -e "s#<VERSION>#$(VERSION)#g" -e "s#<DATE>#$(DATE)#g" -e "s#<GITCOMMIT>#$(GITCOMMIT)#g" -e "s#<MINOR>#$(MINOR)#g" -e "s#VERSION.RELEASE.MINORVERSION#$(VERSION).$(RELEASE).$(MINOR)#g" -e "s#LIKWID_COMMIT GITCOMMIT#LIKWID_COMMIT \"$(GITCOMMIT)\"#g" $(PREFIX)/include/likwid.h
	@install -m 644 $(SRC_DIR)/includes/likwid-marker.h  $(PREFIX)/include/
	$(FORTRAN_INSTALL)
	@echo "===> INSTALL groups to $(PREFIX)/share/likwid/perfgroups"
	@mkdir -p $(PREFIX)/share/likwid/perfgroups
	@chmod 755 $(PREFIX)/share/likwid
	@chmod 755 $(PREFIX)/share/likwid/perfgroups
	@cp -rf $(GROUP_DIR)/* $(PREFIX)/share/likwid/perfgroups
	@chmod 755 $(PREFIX)/share/likwid/perfgroups/*
	@find $(PREFIX)/share/likwid/perfgroups -name "*.txt" -exec chmod 644 {} \;
	@echo "===> INSTALL docs and examples to $(PREFIX)/share/likwid/docs"
	@mkdir -p $(PREFIX)/share/likwid/docs
	@chmod 755 $(PREFIX)/share/likwid/docs
	@mkdir -p $(PREFIX)/share/likwid/examples
	@chmod 755 $(PREFIX)/share/likwid/examples
	@install -m 644 $(EXAMPLES_DIR)/* $(PREFIX)/share/likwid/examples
	@echo "===> INSTALL filters to $(abspath $(PREFIX)/share/likwid/filter)"
	@mkdir -p $(abspath $(PREFIX)/share/likwid/filter)
	@chmod 755 $(abspath $(PREFIX)/share/likwid/filter)
	@for F in $(FILTERS); do \
		install -m 755 $$F  $(abspath $(PREFIX)/share/likwid/filter); \
	done
	@echo "===> INSTALL cmake to $(abspath $(PREFIX)/share/likwid)"
	@install -m 644 $(PWD)/likwid-config.cmake $(PREFIX)/share/likwid

.PHONY: move
move: move_daemon move_freq move_appdaemon move_container_helper
	@echo "===> MOVE applications from $(BINPREFIX) to $(INSTALLED_BINPREFIX)"
	@mkdir -p $(INSTALLED_BINPREFIX)
	@chmod 755 $(INSTALLED_BINPREFIX)
	@for APP in $(L_APPS); do \
		install -m 755 $(BINPREFIX)/$$APP  $(INSTALLED_BINPREFIX); \
	done
	@for APP in $(C_APPS); do \
		install -m 755 $(BINPREFIX)/`basename $$APP`  $(INSTALLED_BINPREFIX); \
	done
	@install -m 755 $(BINPREFIX)/likwid-lua $(INSTALLED_BINPREFIX)/likwid-lua
	@echo "===> MOVE helper applications from $(BINPREFIX) to $(INSTALLED_BINPREFIX)"
	@install -m 755 $(BINPREFIX)/feedGnuplot $(INSTALLED_BINPREFIX)
	@echo "===> MOVE lua to likwid interface from $(PREFIX)/share/lua to $(INSTALLED_PREFIX)/share/lua"
	@mkdir -p $(INSTALLED_PREFIX)/share/lua
	@chmod 755 $(INSTALLED_PREFIX)/share/lua
	@install -m 644 $(PREFIX)/share/lua/likwid.lua $(INSTALLED_PREFIX)/share/lua
	@echo "===> MOVE libraries from $(LIBPREFIX) to $(INSTALLED_LIBPREFIX)"
	@mkdir -p $(INSTALLED_LIBPREFIX)
	@chmod 755 $(INSTALLED_LIBPREFIX)
	@install -m 755 $(LIBPREFIX)/$(TARGET_LIB).$(VERSION).$(RELEASE) $(INSTALLED_LIBPREFIX)/$(TARGET_LIB).$(VERSION).$(RELEASE)
	@install -m 755 $(LIBPREFIX)/$(PINLIB).$(VERSION).$(RELEASE) $(INSTALLED_LIBPREFIX)/$(PINLIB).$(VERSION).$(RELEASE)
	@install -m 755 $(LIBPREFIX)/$(shell basename $(TARGET_HWLOC_LIB)).$(VERSION).$(RELEASE) $(INSTALLED_LIBPREFIX)/$(shell basename $(TARGET_HWLOC_LIB)).$(VERSION).$(RELEASE)
	@install -m 755 $(LIBPREFIX)/$(shell basename $(TARGET_LUA_LIB)).$(VERSION).$(RELEASE) $(INSTALLED_LIBPREFIX)/$(shell basename $(TARGET_LUA_LIB)).$(VERSION).$(RELEASE)
	@install -m 755 $(LIBPREFIX)/$(shell basename $(TARGET_GOTCHA_LIB)).$(VERSION).$(RELEASE) $(INSTALLED_LIBPREFIX)/$(shell basename $(TARGET_GOTCHA_LIB)).$(VERSION).$(RELEASE)
	@cd $(INSTALLED_LIBPREFIX) && ln -fs $(TARGET_LIB).$(VERSION).$(RELEASE) $(TARGET_LIB)
	@cd $(INSTALLED_LIBPREFIX) && ln -fs $(TARGET_LIB).$(VERSION).$(RELEASE) $(TARGET_LIB).$(VERSION)
	@cd $(INSTALLED_LIBPREFIX) && ln -fs $(PINLIB).$(VERSION).$(RELEASE) $(PINLIB)
	@cd $(INSTALLED_LIBPREFIX) && ln -fs $(PINLIB).$(VERSION).$(RELEASE) $(PINLIB).$(VERSION)
	@cd $(INSTALLED_LIBPREFIX) && ln -fs $(shell basename $(TARGET_HWLOC_LIB)).$(VERSION).$(RELEASE) $(shell basename $(TARGET_HWLOC_LIB))
	@cd $(INSTALLED_LIBPREFIX) && ln -fs $(shell basename $(TARGET_HWLOC_LIB)).$(VERSION).$(RELEASE) $(shell basename $(TARGET_HWLOC_LIB)).$(VERSION)
	@cd $(INSTALLED_LIBPREFIX) && ln -fs $(shell basename $(TARGET_LUA_LIB)).$(VERSION).$(RELEASE) $(shell basename $(TARGET_LUA_LIB))
	@cd $(INSTALLED_LIBPREFIX) && ln -fs $(shell basename $(TARGET_LUA_LIB)).$(VERSION).$(RELEASE) $(shell basename $(TARGET_LUA_LIB)).$(VERSION)
	@cd $(INSTALLED_LIBPREFIX) && ln -fs $(shell basename $(TARGET_GOTCHA_LIB)).$(VERSION).$(RELEASE) $(shell basename $(TARGET_GOTCHA_LIB))
	@cd $(INSTALLED_LIBPREFIX) && ln -fs $(shell basename $(TARGET_GOTCHA_LIB)).$(VERSION).$(RELEASE) $(shell basename $(TARGET_GOTCHA_LIB)).$(VERSION)
	@echo "===> MOVE man pages from $(MANPREFIX)/man1 to $(INSTALLED_MANPREFIX)/man1"
	@mkdir -p $(INSTALLED_MANPREFIX)/man1
	@chmod 755 $(INSTALLED_MANPREFIX)/man1
	@install -m 644 $(MANPREFIX)/man1/*.1 $(INSTALLED_MANPREFIX)/man1
	@echo "===> MOVE headers from $(PREFIX)/include to $(INSTALLED_PREFIX)/include"
	@mkdir -p $(INSTALLED_PREFIX)/include
	@chmod 755 $(INSTALLED_PREFIX)/include
	@install -m 644 $(PREFIX)/include/likwid.h $(INSTALLED_PREFIX)/include/likwid.h
	@install -m 644 $(PREFIX)/include/likwid-marker.h $(INSTALLED_PREFIX)/include/likwid-marker.h
	@if [ -e $(PREFIX)/include/likwid.mod ]; then install $(PREFIX)/include/likwid.mod $(INSTALLED_PREFIX)/include/likwid.mod; fi
	@echo "===> MOVE groups from $(PREFIX)/share/likwid/perfgroups to $(INSTALLED_PREFIX)/share/likwid/perfgroups"
	@mkdir -p $(INSTALLED_PREFIX)/share/likwid/perfgroups
	@chmod 755 $(INSTALLED_PREFIX)/share/likwid
	@chmod 755 $(INSTALLED_PREFIX)/share/likwid/perfgroups
	@cp -rf $(PREFIX)/share/likwid/perfgroups/* $(INSTALLED_PREFIX)/share/likwid/perfgroups
	@chmod 755 $(INSTALLED_PREFIX)/share/likwid/perfgroups/*
	@find $(INSTALLED_PREFIX)/share/likwid/perfgroups -name "*.txt" -exec chmod 644 {} \;
	@mkdir -p $(INSTALLED_PREFIX)/share/likwid/docs
	@chmod 755 $(INSTALLED_PREFIX)/share/likwid/docs
	@mkdir -p $(INSTALLED_PREFIX)/share/likwid/examples
	@chmod 755 $(INSTALLED_PREFIX)/share/likwid/examples
	@install -m 644 $(EXAMPLES_DIR)/* $(INSTALLED_PREFIX)/share/likwid/examples
	@echo "===> MOVE filters from $(abspath $(PREFIX)/share/likwid/filter) to $(LIKWIDFILTERPATH)"
	@mkdir -p $(LIKWIDFILTERPATH)
	@chmod 755 $(LIKWIDFILTERPATH)
	@cp -f $(abspath $(PREFIX)/share/likwid/filter)/* $(LIKWIDFILTERPATH)
	@chmod 755 $(LIKWIDFILTERPATH)/*
	@echo "===> MOVE cmake from $(abspath $(PREFIX)/share/likwid) to $(INSTALLED_PREFIX)/share/likwid"
	@mkdir -p $(INSTALLED_PREFIX)/share/likwid
	@chmod 755 $(INSTALLED_PREFIX)/share/likwid
	@sed -e s#'\(# @MOVE_LIKWID_INSTALL@\)'#"set(_DEFAULT_PREFIXES $(INSTALLED_PREFIX) $(INSTALLED_LIBPREFIX) $(INSTALLED_BINPREFIX))\n\1"#g \
		$(PREFIX)/share/likwid/likwid-config.cmake > $(INSTALLED_PREFIX)/share/likwid/likwid-config.cmake
	@chmod 644 $(INSTALLED_PREFIX)/share/likwid/likwid-config.cmake

.PHONY: uninstall
uninstall: uninstall_daemon uninstall_freq uninstall_appdaemon uninstall_container_helper
	@echo "===> REMOVING applications from $(PREFIX)/bin"
	@rm -f $(addprefix $(BINPREFIX)/,$(addsuffix  .lua,$(L_APPS)))
	@for APP in $(L_APPS); do \
		rm -f $(BINPREFIX)/$$APP; \
	done
	@for APP in $(C_APPS); do \
		rm -f $(BINPREFIX)/$$APP; \
	done
	@rm -f $(BINPREFIX)/feedGnuplot
	@rm -f $(BINPREFIX)/likwid-lua
	@rm -f $(BINPREFIX)/likwid-bench
	@echo "===> REMOVING Lua to likwid interface from $(PREFIX)/share/lua"
	@rm -rf  $(PREFIX)/share/lua/likwid.lua
	@echo "===> REMOVING libs from $(LIBPREFIX)"
	@rm -f $(LIBPREFIX)/liblikwid*
	@rm -f $(LIBPREFIX)/$(TARGET_GOTCHA_LIB)
	@echo "===> REMOVING man pages from $(MANPREFIX)/man1"
	@rm -f $(addprefix $(MANPREFIX)/man1/,$(addsuffix  .1,$(L_APPS)))
	@rm -f $(MANPREFIX)/man1/feedGnuplot.1
	@rm -f $(MANPREFIX)/man1/likwid-setFreq.1
	@rm -f $(MANPREFIX)/man1/likwid-accessD.1
	@rm -f $(MANPREFIX)/man1/likwid-lua.1
	@rm -f $(MANPREFIX)/man1/likwid-bench.1
	@echo "===> REMOVING header from $(PREFIX)/include"
	@rm -f $(PREFIX)/include/likwid.h
	@rm -f $(PREFIX)/include/likwid-marker.h
	$(FORTRAN_REMOVE)
	@echo "===> REMOVING filter, groups and default configs from $(PREFIX)/share/likwid"
	@rm -rf $(abspath $(PREFIX)/share/likwid/filter)
	@rm -rf $(PREFIX)/share/likwid/perfgroups
	@rm -rf $(PREFIX)/share/likwid/docs
	@rm -rf $(PREFIX)/share/likwid/examples
	@rm -rf $(PREFIX)/share/likwid/likwid-config.cmake
	@rm -rf $(PREFIX)/share/likwid

.PHONY: uninstall_moved
uninstall_moved: uninstall_daemon_moved uninstall_freq_moved uninstall_appdaemon_moved uninstall_container_helper_moved
	@echo "===> REMOVING applications from $(INSTALLED_PREFIX)/bin"
	@rm -f $(addprefix $(INSTALLED_BINPREFIX)/,$(addsuffix  .lua,$(L_APPS)))
	@for APP in $(L_APPS); do \
		rm -f $(INSTALLED_BINPREFIX)/$$APP; \
	done
	@for APP in $(C_APPS); do \
		rm -f $(INSTALLED_BINPREFIX)/$$APP; \
	done
	@rm -f $(INSTALLED_BINPREFIX)/feedGnuplot
	@rm -f $(INSTALLED_BINPREFIX)/likwid-lua
	@rm -f $(INSTALLED_BINPREFIX)/likwid-bench
	@echo "===> REMOVING Lua to likwid interface from $(INSTALLED_PREFIX)/share/lua"
	@rm -rf  $(INSTALLED_PREFIX)/share/lua/likwid.lua
	@echo "===> REMOVING libs from $(INSTALLED_LIBPREFIX)"
	@rm -f $(INSTALLED_LIBPREFIX)/liblikwid*
	@rm -f $(INSTALLED_LIBPREFIX)/$(TARGET_GOTCHA_LIB)
	@echo "===> REMOVING man pages from $(INSTALLED_MANPREFIX)/man1"
	@rm -f $(addprefix $(INSTALLED_MANPREFIX)/man1/,$(addsuffix  .1,$(L_APPS)))
	@rm -f $(INSTALLED_MANPREFIX)/man1/feedGnuplot.1
	@rm -f $(INSTALLED_MANPREFIX)/man1/likwid-setFreq.1
	@rm -f $(INSTALLED_MANPREFIX)/man1/likwid-accessD.1
	@rm -f $(INSTALLED_MANPREFIX)/man1/likwid-lua.1
	@rm -f $(INSTALLED_MANPREFIX)/man1/likwid-bench.1
	@echo "===> REMOVING header from $(INSTALLED_PREFIX)/include"
	@rm -f $(INSTALLED_PREFIX)/include/likwid.h
	@rm -f $(PREFIX)/include/likwid-marker.h
	$(FORTRAN_REMOVE)
	@echo "===> REMOVING filter, groups and default configs from $(INSTALLED_PREFIX)/share/likwid"
	@rm -rf $(LIKWIDFILTERPATH)
	@rm -rf $(INSTALLED_PREFIX)/share/likwid/perfgroups
	@rm -rf $(INSTALLED_PREFIX)/share/likwid/docs
	@rm -rf $(INSTALLED_PREFIX)/share/likwid/examples
	@rm -rf $(INSTALLED_PREFIX)/share/likwid/likwid-config.cmake
	@rm -rf $(INSTALLED_PREFIX)/share/likwid

.PHONY: local
local: $(L_APPS) likwid.lua
	@echo "===> Setting Lua scripts to run from current directory"
	@for APP in $(L_APPS); do \
		sed -i -e "s#<VERSION>/#$(VERSION)#g" -e "s#<DATE>#$(DATE)#g" -e "s#<RELEASE>#$(RELEASE)#g" -e "s#<GITCOMMIT>#$(GITCOMMIT)#g" -e "s#<MINOR>#$(MINOR)#g" -e "s#$(PREFIX)/bin/likwid-lua#$(PWD)/ext/lua/lua#" -e "s#$(PREFIX)/share/lua/?.lua#$(PWD)/?.lua#" -e "s#$(PREFIX)/bin/likwid-pin#$(PWD)/likwid-pin#" -e "s#$(PREFIX)/bin/likwid-perfctr#$(PWD)/likwid-perfctr#" -e "s#$(PREFIX)/lib#$(PWD)#" $$APP; \
		chmod +x $$APP; \
	done
	@sed -i -e "s#<VERSION>#$(VERSION)#g" -e "s#<DATE>#$(DATE)#g" -e "s#<RELEASE>#$(RELEASE)#g" -e "s#$(PREFIX)/lib#$(PWD)#g" -e "s#$(PREFIX)/share/likwid/perfgroups#$(PWD)/groups#g" -e "s#<GITCOMMIT>#$(GITCOMMIT)#g" -e "s#<MINOR>#$(MINOR)#g" likwid.lua;
	@ln -sf liblikwid.so liblikwid.so.$(VERSION)
	@ln -sf liblikwid.so liblikwid.so.$(VERSION).$(RELEASE)
	@ln -sf $(HWLOC_FOLDER)/liblikwid-hwloc.so liblikwid-hwloc.so.$(VERSION)
	@ln -sf $(HWLOC_FOLDER)/liblikwid-hwloc.so liblikwid-hwloc.so.$(VERSION).$(RELEASE)
	@ln -sf $(LUA_FOLDER)/liblikwid-lua.so liblikwid-lua.so.$(VERSION)
	@ln -sf $(LUA_FOLDER)/liblikwid-lua.so liblikwid-lua.so.$(VERSION).$(RELEASE)
	@ln -sf $(GOTCHA_FOLDER)/liblikwid-gotcha.so liblikwid-gotcha.so.$(VERSION)
	@ln -sf $(GOTCHA_FOLDER)/liblikwid-gotcha.so liblikwid-gotcha.so.$(VERSION).$(RELEASE)
	@if [ "$(LUA_INTERNAL)" = "true" ]; then \
		if [ -e $(LUA_FOLDER)/liblikwid-lua.so ]; then ln -sf $(LUA_FOLDER)/liblikwid-lua.so liblikwid-lua.so.$(VERSION).$(RELEASE); fi; \
	fi
	@if [ "$(USE_INTERNAL_HWLOC)" = "true" ]; then \
		if [ -e $(HWLOC_FOLDER)/liblikwid-hwloc.so ]; then ln -sf $(HWLOC_FOLDER)/liblikwid-hwloc.so liblikwid-hwloc.so.$(VERSION).$(RELEASE); fi; \
	fi
	@if [ -e $(PINLIB) ]; then ln -sf $(PINLIB) $(PINLIB).$(VERSION).$(RELEASE); fi
	@if [ -e $(CUDA_HOME)/extras/CUPTI/lib64 ]; then echo "export LD_LIBRARY_PATH=$(PWD):$(CUDA_HOME)/extras/CUPTI/lib64:$$LD_LIBRARY_PATH"; else echo "export LD_LIBRARY_PATH=$(PWD):$$LD_LIBRARY_PATH"; fi

.PHONY: testit
testit: test/test-likwidAPI.c
	make -C test test-likwidAPI
	test/test-likwidAPI
	make -C test/executable_tests

.PHONY: help
help:
	@echo "Help for building LIKWID:"
	@echo
	@echo "Common make targets:"
	@echo "- make : build anything (integrate already compiled files)"
	@echo "- make clean : clean library and executables, keep compiled files"
	@echo "- make distclean : clean anything"
	@echo "- make docs : Build documentation (requires Doxygen)"
	@echo "- make install : Copy compiled files to $(PREFIX)"
	@echo "- make move : Copy files from $(PREFIX) to $(INSTALLED_PREFIX)"
	@echo "- make uninstall : Delete files from $(PREFIX)"
	@echo "- make uninstall_moved : Delete files from $(INSTALLED_PREFIX)"
	@echo
	@echo "Compiler selection can be done in config.mk at COMPILER:"
	@echo "- GCC : Use GCC for C code and Intel Fortran compiler for Fortran interface (default)"
	@echo "- GCCX86 : Use GCC for C code. No Fortran compiler set (only for 32 bit builds)"
	@echo "- CLANG: Use CLANG for C code and Intel Fortran compiler for Fortran interface (unsupported, may fail)"
	@echo "- ICC: Use Intel C compiler for C code and Intel Fortran compiler for Fortran interface (unsupported, may fail)"
	@echo "- MIC: Build for Intel Xeon Phi. Use Intel C compiler for C code and\n       Intel Fortran compiler for Fortran interface (unsupported)"
	@echo
	@echo "LIKWID runs only in INSTALLED_PREFIX = $(INSTALLED_PREFIX)"
	@echo "You can change it in config.mk, but it is recommended to keep INSTALLED_PREFIX = PREFIX"
	@echo "The PREFIX is used for temporary install directories (e.g. for packaging)."
	@echo "LIKWID will not run in PREFIX, it has to be in INSTALLED_PREFIX."
	@echo "The common configuration is INSTALLED_PREFIX = PREFIX, so changing PREFIX is enough."
	@echo "If PREFIX and INSTALLED_PREFIX differ, you have to move anything after 'make install' to"
	@echo "the INSTALLED_PREFIX. You can also use 'make move' which does the job for you."

.PHONY: rpm RPM
rpm: RPM
RPM: packaging/rpm/likwid.spec
	FROM_MAKEFILE=1 $(BASE_DIR)/packaging/rpm/package.sh

.PHONY: deb DEB
deb: DEB
DEB: packaging/deb/likwid.deb.control
	NAME=$(NAME) VERSION=$(VERSION) RELEASE=$(RELEASE) MINOR=$(MINOR) \
		 PREFIX=$(PREFIX) FROM_MAKEFILE=1 $(BASE_DIR)/packaging/deb/package.sh
