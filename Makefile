#
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


#DO NOT EDIT BELOW

# Dependency chains:
# *.[ch] -> *.o -> executables
# *.ptt -> *.pas -> *.s -> *.o -> executables
# *.txt -> *.h (generated)

include ./config.mk
include $(MAKE_DIR)/include_$(COMPILER).mk
include $(MAKE_DIR)/config_checks.mk
include $(MAKE_DIR)/config_defines.mk

INCLUDES  += -I./src/includes -I$(LUA_FOLDER)/includes -I$(HWLOC_FOLDER)/include -I$(BUILD_DIR)
LIBS      += -ldl

#CONFIGURE BUILD SYSTEM
BUILD_DIR  = ./$(COMPILER)
Q         ?= @
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
PERFMONHEADERS  = $(patsubst $(SRC_DIR)/includes/%.txt, $(BUILD_DIR)/%.h,$(wildcard $(SRC_DIR)/includes/*.txt))
OBJ_LUA    =  $(wildcard ./ext/lua/$(COMPILER)/*.o)
OBJ_HWLOC  =  $(wildcard ./ext/hwloc/$(COMPILER)/*.o)


L_APPS      =   likwid-perfctr \
				likwid-pin \
				likwid-powermeter \
				likwid-topology \
				likwid-memsweeper \
				likwid-agent \
				likwid-mpirun \
				likwid-features \
				likwid-perfscope \
				likwid-genTopoCfg
C_APPS      =   bench/likwid-bench
L_HELPER    =   likwid.lua
ifeq ($(BUILDFREQ),true)
	L_APPS += likwid-setFrequencies
endif

CPPFLAGS := $(CPPFLAGS) $(DEFINES) $(INCLUDES)

all: $(BUILD_DIR) $(PERFMONHEADERS) $(OBJ) $(TARGET_LIB) $(FORTRAN_IF)  $(PINLIB) $(L_APPS) $(L_HELPER) $(DAEMON_TARGET) $(FREQ_TARGET) $(BENCH_TARGET)

tags:
	@echo "===>  GENERATE  TAGS"
	$(Q)ctags -R

docs:
	@echo "===>  GENERATE DOXYGEN DOCS"
	@cp doc/lua-doxygen.md doc/lua-doxygen.md.safe
	@cp doc/likwid-doxygen.md doc/likwid-doxygen.md.safe
	@sed -i -e s+'<PREFIX>'+$(PREFIX)+g -e s+'<VERSION>'+$(VERSION)+g -e s+'<DATE>'+'$(DATE)'+g -e s+'<RELEASE>'+$(RELEASE)+g doc/lua-doxygen.md
	@sed -i -e s+'<PREFIX>'+$(PREFIX)+g -e s+'<VERSION>'+$(VERSION)+g -e s+'<DATE>'+'$(DATE)'+g -e s+'<RELEASE>'+$(RELEASE)+g doc/likwid-doxygen.md
	$(Q)doxygen doc/Doxyfile
	@mv doc/lua-doxygen.md.safe doc/lua-doxygen.md
	@mv doc/likwid-doxygen.md.safe doc/likwid-doxygen.md

$(L_APPS):  $(addprefix $(SRC_DIR)/applications/,$(addsuffix  .lua,$(L_APPS)))
	@echo "===>  ADJUSTING  $@"
	@if [ "$(ACCESSMODE)" = "direct" ]; then sed -i -e s/"access_mode = 1"/"access_mode = 0"/g $(SRC_DIR)/applications/$@.lua;fi
	@sed -e s/'<INSTALLED_BINPREFIX>'/$(subst /,\\/,$(INSTALLED_BINPREFIX))/g \
		-e s/'<INSTALLED_PREFIX>'/$(subst /,\\/,$(INSTALLED_PREFIX))/g \
		-e s/'<VERSION>'/$(VERSION).$(RELEASE)/g \
		-e s/'<DATE>'/$(DATE)/g \
		$(addprefix $(SRC_DIR)/applications/,$(addsuffix  .lua,$@)) > $@
	@if [ "$(ACCESSMODE)" = "direct" ]; then sed -i -e s/"access_mode = 0"/"access_mode = 1"/g $(SRC_DIR)/applications/$@.lua;fi

$(L_HELPER):
	@echo "===>  ADJUSTING  $@"
	@sed -e s/'<PREFIX>'/$(subst /,\\/,$(PREFIX))/g \
		-e s/'<INSTALLED_LIBPREFIX>'/$(subst /,\\/,$(INSTALLED_LIBPREFIX))/g \
		-e s/'<INSTALLED_PREFIX>'/$(subst /,\\/,$(INSTALLED_PREFIX))/g \
		-e s/'<VERSION>'/$(VERSION)/g \
		-e s/'<RELEASE>'/$(RELEASE)/g \
		$(SRC_DIR)/applications/$@ > $@

$(STATIC_TARGET_LIB): $(BUILD_DIR) $(PERFMONHEADERS) $(OBJ) $(TARGET_HWLOC_LIB) $(TARGET_LUA_LIB)
	@echo "===>  CREATE STATIC LIB  $(TARGET_LIB)"
	$(Q)${AR} -crus $(STATIC_TARGET_LIB) $(OBJ) $(TARGET_HWLOC_LIB) $(TARGET_LUA_LIB)


$(DYNAMIC_TARGET_LIB): $(BUILD_DIR) $(PERFMONHEADERS) $(OBJ) $(TARGET_HWLOC_LIB) $(TARGET_LUA_LIB)
	@echo "===>  CREATE SHARED LIB  $(TARGET_LIB)"
	$(Q)${CC} $(DEBUG_FLAGS) $(SHARED_LFLAGS) -Wl,-soname,$(TARGET_LIB).$(VERSION) $(SHARED_CFLAGS) -o $(DYNAMIC_TARGET_LIB) $(OBJ) $(LIBS) $(TARGET_HWLOC_LIB) $(TARGET_LUA_LIB) $(RPATHS)

$(DAEMON_TARGET): $(SRC_DIR)/access-daemon/accessDaemon.c
	@echo "===>  BUILD access daemon likwid-accessD"
	$(Q)$(MAKE) -s -C  $(SRC_DIR)/access-daemon likwid-accessD

$(FREQ_TARGET): $(SRC_DIR)/access-daemon/setFreq.c
	@echo "===>  BUILD frequency daemon likwid-setFreq"
	$(Q)$(MAKE) -s -C  $(SRC_DIR)/access-daemon likwid-setFreq

$(BUILD_DIR):
	@mkdir $(BUILD_DIR)

$(PINLIB):
	@echo "===>  CREATE LIB  $(PINLIB)"
	$(Q)$(MAKE) -s -C src/pthread-overload/ $(PINLIB)

$(GENGROUPLOCK): $(foreach directory,$(shell ls $(GROUP_DIR)), $(wildcard $(GROUP_DIR)/$(directory)/*.txt))
	@echo "===>  GENERATE GROUP HEADERS"
	$(Q)$(GEN_GROUPS) ./groups  $(BUILD_DIR) ./perl/templates
	$(Q)touch $(GENGROUPLOCK)

$(FORTRAN_IF): $(SRC_DIR)/likwid.f90
	@echo "===>  COMPILE FORTRAN INTERFACE  $@"
	$(Q)$(FC) -c  $(FCFLAGS) $<
	@rm -f likwid.o

$(TARGET_LUA_LIB):
	@echo "===>  ENTER  $(LUA_FOLDER)"
	$(Q)$(MAKE) -s --no-print-directory -C $(LUA_FOLDER) $(MAKECMDGOALS)

$(TARGET_HWLOC_LIB):
	@echo "===>  ENTER  $(HWLOC_FOLDER)"
	$(Q)$(MAKE) -s --no-print-directory -C $(HWLOC_FOLDER) $(MAKECMDGOALS)

$(BENCH_TARGET):
	@echo "===>  ENTER  $(BENCH_FOLDER)"
	$(Q)$(MAKE) -s --no-print-directory -C $(BENCH_FOLDER) $(MAKECMDGOALS)

#PATTERN RULES
$(BUILD_DIR)/%.o:  %.c
	@echo "===>  COMPILE  $@"
	$(Q)$(CC) -c $(DEBUG_FLAGS) $(CFLAGS) $(ANSI_CFLAGS) $(CPPFLAGS) $< -o $@
	$(Q)$(CC) $(DEBUG_FLAGS) $(CPPFLAGS) -MT $(@:.d=.o) -MM  $< > $(BUILD_DIR)/$*.d

$(BUILD_DIR)/%.o:  %.cc
	@echo "===>  COMPILE  $@"
	$(Q)$(CXX) -c $(DEBUG_FLAGS) $(CXXFLAGS) $(CPPFLAGS) $< -o $@
	$(Q)$(CXX) $(DEBUG_FLAGS) $(CXXFLAGS) $(CPPFLAGS) -MT $(@:.d=.o) -MM  $< > $(BUILD_DIR)/$*.d

$(BUILD_DIR)/%.o:  %.S
	@echo "===>  COMPILE  $@"
	$(Q)$(CPP) $(CPPFLAGS) $< -o $@.tmp
	$(Q)$(AS) $(ASFLAGS) $@.tmp -o $@
	@rm $@.tmp


$(BUILD_DIR)/%.h:  $(SRC_DIR)/includes/%.txt
	@echo "===>  GENERATE HEADER $@"
	$(Q)$(GEN_PMHEADER) $< $@


ifeq ($(findstring $(MAKECMDGOALS),clean),)
-include $(OBJ:.o=.d)
endif

.PHONY: clean distclean install uninstall $(TARGET_LUA_LIB) $(TARGET_HWLOC_LIB) $(BENCH_TARGET)


.PRECIOUS: $(BUILD_DIR)/%.pas

.NOTPARALLEL:


clean: $(TARGET_LUA_LIB) $(TARGET_HWLOC_LIB) $(BENCH_TARGET)
	@echo "===>  CLEAN"
	@for APP in $(L_APPS); do \
		rm -f $$APP; \
	done
	@rm -f likwid.lua
	@rm -f $(STATIC_TARGET_LIB)
	@rm -f $(DYNAMIC_TARGET_LIB)
	@rm -f $(PINLIB)
	@rm -f $(FORTRAN_IF_NAME)
	@rm -f $(FREQ_TARGET) $(DAEMON_TARGET)

distclean: $(TARGET_LUA_LIB) $(TARGET_HWLOC_LIB) $(BENCH_TARGET)
	@echo "===>  DIST CLEAN"
	@for APP in $(L_APPS); do \
		rm -f $$APP; \
	done
	@rm -f likwid.lua
	@rm -f $(STATIC_TARGET_LIB)
	@rm -f $(DYNAMIC_TARGET_LIB)
	@rm -f $(PINLIB)
	@rm -f $(FORTRAN_IF_NAME)
	@rm -f $(FREQ_TARGET) $(DAEMON_TARGET)
	@rm -rf $(BUILD_DIR)
	@rm -f $(GENGROUPLOCK)
	@rm -rf doc/html
	@rm -f tags

ifeq ($(BUILDDAEMON),true)
ifneq ($(COMPILER),MIC)
install_daemon:
	@echo "===> INSTALL access daemon to $(ACCESSDAEMON)"
	@mkdir -p `dirname $(ACCESSDAEMON)`
	@install -m 4775 $(INSTALL_CHOWN) $(DAEMON_TARGET) $(ACCESSDAEMON)
uninstall_daemon:
	@echo "===> REMOVING access daemon from $(ACCESSDAEMON)"
	@rm -f $(ACCESSDAEMON)
else
install_daemon:
	@echo "===> No INSTALL of the access daemon"
uninstall_daemon:
	@echo "===> No UNINSTALL of the access daemon"
endif
else
install_daemon:
	@echo "===> No INSTALL of the access daemon"
uninstall_daemon:
	@echo "===> No UNINSTALL of the access daemon"
endif

ifeq ($(BUILDFREQ),true)
ifneq ($(COMPILER),MIC)
install_freq:
	@echo "===> INSTALL setFrequencies tool to $(PREFIX)/sbin/$(FREQ_TARGET)"
	@mkdir -p $(PREFIX)/sbin
	@install -m 4775 $(INSTALL_CHOWN) $(FREQ_TARGET) $(PREFIX)/sbin/$(FREQ_TARGET)
uninstall_freq:
	@echo "===> REMOVING setFrequencies tool from $(PREFIX)/sbin/$(FREQ_TARGET)"
	@rm -f $(PREFIX)/sbin/$(FREQ_TARGET)
else
install_freq:
	@echo "===> No INSTALL of setFrequencies tool"
uninstall_freq:
	@echo "===> No UNINSTALL of setFrequencies tool"
endif
else
install_freq:
	@echo "===> No INSTALL of setFrequencies tool"
uninstall_freq:
	@echo "===> No UNINSTALL of setFrequencies tool"
endif

install: install_daemon install_freq
	@echo "===> INSTALL applications to $(BINPREFIX)"
	@mkdir -p $(BINPREFIX)
	@chmod 775 $(BINPREFIX)
	@for APP in $(L_APPS); do \
		install -m 755 $$APP  $(BINPREFIX); \
	done
	@for APP in $(C_APPS); do \
		install -m 755 $$APP  $(BINPREFIX); \
	done
	@install -m 755 ext/lua/lua $(BINPREFIX)/likwid-lua
	@echo "===> INSTALL helper applications"
	@install -m 755 perl/feedGnuplot $(BINPREFIX)
	@echo "===> INSTALL lua to likwid interface to $(PREFIX)/share/lua"
	@mkdir -p $(PREFIX)/share/lua
	@chmod 775 $(PREFIX)/share/lua
	@install -m 755 likwid.lua $(PREFIX)/share/lua
	@echo "===> INSTALL libraries to $(LIBPREFIX)"
	@mkdir -p $(LIBPREFIX)
	@chmod 775 $(LIBPREFIX)
	@install -m 755 $(TARGET_LIB) $(LIBPREFIX)/$(TARGET_LIB).$(VERSION)
	@install -m 755 liblikwidpin.so $(LIBPREFIX)/liblikwidpin.so.$(VERSION)
	@install -m 755 $(TARGET_HWLOC_LIB) $(LIBPREFIX)/$(shell basename $(TARGET_HWLOC_LIB)).$(VERSION)
	@install -m 755 $(TARGET_LUA_LIB) $(LIBPREFIX)/$(shell basename $(TARGET_LUA_LIB)).$(VERSION)
	@cd $(LIBPREFIX) && ln -fs $(TARGET_LIB).$(VERSION) $(TARGET_LIB)
	@cd $(LIBPREFIX) && ln -fs liblikwidpin.so.$(VERSION) liblikwidpin.so
	@cd $(LIBPREFIX) && ln -fs $(shell basename $(TARGET_HWLOC_LIB)).$(VERSION) $(shell basename $(TARGET_HWLOC_LIB))
	@cd $(LIBPREFIX) && ln -fs $(shell basename $(TARGET_LUA_LIB)).$(VERSION) $(shell basename $(TARGET_LUA_LIB))
	@echo "===> INSTALL man pages to $(MANPREFIX)/man1"
	@mkdir -p $(MANPREFIX)/man1
	@chmod 775 $(MANPREFIX)/man1
	@sed -e "s/<VERSION>/$(VERSION)/g" -e "s/<DATE>/$(DATE)/g" < $(DOC_DIR)/likwid-topology.1 > $(MANPREFIX)/man1/likwid-topology.1
	@sed -e "s/<VERSION>/$(VERSION)/g" -e "s/<DATE>/$(DATE)/g" -e "s+<PREFIX>+$(PREFIX)+g" < $(DOC_DIR)/likwid-perfctr.1 > $(MANPREFIX)/man1/likwid-perfctr.1
	@sed -e "s/<VERSION>/$(VERSION)/g" -e "s/<DATE>/$(DATE)/g" < $(DOC_DIR)/likwid-powermeter.1 > $(MANPREFIX)/man1/likwid-powermeter.1
	@sed -e "s/<VERSION>/$(VERSION)/g" -e "s/<DATE>/$(DATE)/g" < $(DOC_DIR)/likwid-pin.1 > $(MANPREFIX)/man1/likwid-pin.1
	@sed -e "s/<VERSION>/$(VERSION)/g" -e "s/<DATE>/$(DATE)/g" < $(DOC_DIR)/feedGnuplot.1 > $(MANPREFIX)/man1/feedGnuplot.1
	@sed -e "s/<VERSION>/$(VERSION)/g" -e "s/<DATE>/$(DATE)/g" < $(DOC_DIR)/likwid-accessD.1 > $(MANPREFIX)/man1/likwid-accessD.1
	@sed -e "s/<VERSION>/$(VERSION)/g" -e "s/<DATE>/$(DATE)/g" < $(DOC_DIR)/likwid-genTopoCfg.1 > $(MANPREFIX)/man1/likwid-genTopoCfg.1
	@sed -e "s/<VERSION>/$(VERSION)/g" -e "s/<DATE>/$(DATE)/g" < $(DOC_DIR)/likwid-memsweeper.1 > $(MANPREFIX)/man1/likwid-memsweeper.1
	@sed -e "s/<VERSION>/$(VERSION)/g" -e "s/<DATE>/$(DATE)/g" < $(DOC_DIR)/likwid-mpirun.1 > $(MANPREFIX)/man1/likwid-mpirun.1
	@sed -e "s/<VERSION>/$(VERSION)/g" -e "s/<DATE>/$(DATE)/g" < $(DOC_DIR)/likwid-perfscope.1 > $(MANPREFIX)/man1/likwid-perfscope.1
	@sed -e "s/<VERSION>/$(VERSION)/g" -e "s/<DATE>/$(DATE)/g" < $(DOC_DIR)/likwid-setFreq.1 > $(MANPREFIX)/man1/likwid-setFreq.1
	@sed -e "s/<VERSION>/$(VERSION)/g" -e "s/<DATE>/$(DATE)/g" < $(DOC_DIR)/likwid-features.1 > $(MANPREFIX)/man1/likwid-features.1
	@chmod 644 $(MANPREFIX)/man1/likwid-*
	@echo "===> INSTALL headers to $(PREFIX)/include"
	@mkdir -p $(PREFIX)/include
	@chmod 775 $(PREFIX)/include
	@install -m 644 src/includes/likwid.h  $(PREFIX)/include/
	@install -m 644 src/includes/bstrlib.h  $(PREFIX)/include/
	$(FORTRAN_INSTALL)
	@echo "===> INSTALL groups to $(PREFIX)/share/likwid/perfgroups"
	@mkdir -p $(PREFIX)/share/likwid/perfgroups
	@chmod 775 $(PREFIX)/share/likwid
	@chmod 775 $(PREFIX)/share/likwid/perfgroups
	@cp -rf groups/* $(PREFIX)/share/likwid/perfgroups
	@chmod 775 $(PREFIX)/share/likwid/perfgroups/*
	@find $(PREFIX)/share/likwid/perfgroups -name "*.txt" -exec chmod 644 {} \;
	@echo "===> INSTALL monitoring groups to $(PREFIX)/share/likwid/mongroups"
	@mkdir -p $(PREFIX)/share/likwid/mongroups
	@chmod 775 $(PREFIX)/share/likwid/mongroups
	@cp -rf monitoring/groups/* $(PREFIX)/share/likwid/mongroups
	@chmod 775 $(PREFIX)/share/likwid/mongroups/*
	@find $(PREFIX)/share/likwid/mongroups -name "*.txt" -exec chmod 644 {} \;
	@mkdir -p $(PREFIX)/share/likwid/docs
	@chmod 775 $(PREFIX)/share/likwid/docs
	@install -m 644 doc/bstrlib.txt $(PREFIX)/share/likwid/docs
	@mkdir -p $(PREFIX)/share/likwid/examples
	@chmod 775 $(PREFIX)/share/likwid/examples
	@install -m 644 examples/* $(PREFIX)/share/likwid/examples
	@echo "===> INSTALL default likwid-agent.conf to $(PREFIX)/etc"
	@sed -e "s+<PREFIX>+$(PREFIX)+g" monitoring/likwid-agent.conf > $(PREFIX)/share/likwid/mongroups/likwid-agent.conf
	@chmod 644 $(PREFIX)/share/likwid/mongroups/likwid-agent.conf
	@echo "===> INSTALL filters to $(LIKWIDFILTERPATH)"
	@mkdir -p $(LIKWIDFILTERPATH)
	@chmod 755 $(LIKWIDFILTERPATH)
	@cp -f filters/*  $(LIKWIDFILTERPATH)
	@chmod 755 $(LIKWIDFILTERPATH)/*


uninstall: uninstall_daemon uninstall_freq
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
	@echo "===> REMOVING man pages from $(MANPREFIX)/man1"
	@rm -f $(addprefix $(MANPREFIX)/man1/,$(addsuffix  .1,$(L_APPS)))
	@rm -f $(MANPREFIX)/man1/feedGnuplot.1
	@rm -f $(MANPREFIX)/man1/likwid-setFreq.1
	@rm -f $(MANPREFIX)/man1/likwid-accessD.1
	@echo "===> REMOVING header from $(PREFIX)/include"
	@rm -f $(PREFIX)/include/likwid.h
	@rm -f $(PREFIX)/include/bstrlib.h
	$(FORTRAN_REMOVE)
	@echo "===> REMOVING filter, groups and default configs from $(PREFIX)/share/likwid"
	@rm -rf $(LIKWIDFILTERPATH)
	@rm -rf $(PREFIX)/share/likwid/mongroups
	@rm -rf $(PREFIX)/share/likwid/perfgroups
	@rm -rf $(PREFIX)/share/likwid/docs
	@rm -rf $(PREFIX)/share/likwid/examples
	@rm -rf $(PREFIX)/share/likwid


local: $(L_APPS) likwid.lua
	@echo "===> Setting Lua scripts to run from current directory"
	@PWD=$(shell pwd)
	@for APP in $(L_APPS); do \
		sed -i -e "s/<VERSION>/$(VERSION)/g" -e "s/<DATE>/$(DATE)/g" -e "s/<RELEASE>/$(RELEASE)/g" -e "s+$(PREFIX)/bin/likwid-lua+$(PWD)/ext/lua/lua+" -e "s+$(PREFIX)/share/lua/?.lua+$(PWD)/?.lua+" $$APP; \
		chmod +x $$APP; \
	done
	@sed -i -e "s/<VERSION>/$(VERSION)/g" -e "s/<DATE>/$(DATE)/g" -e "s/<RELEASE>/$(RELEASE)/g" -e "s+$(PREFIX)/lib+$(PWD)+g" -e "s+$(PREFIX)/share/likwid/perfgroups+$(PWD)/groups+g" likwid.lua;
	@sed -i -e "s+$(PREFIX)/share/likwid/mongroups+$(PWD)/monitoring/groups+g" likwid-agent
	@ln -sf liblikwid.so liblikwid.so.$(VERSION)
	@ln -sf ext/hwloc/liblikwid-hwloc.so liblikwid-hwloc.so.$(VERSION)
	@ln -sf ext/lua/liblikwid-lua.so liblikwid-lua.so.$(VERSION)
	@echo "export LD_LIBRARY_PATH=$(PWD):$$LD_LIBRARY_PATH"

testit: test/test-likwidAPI.c
	make -C test test-likwidAPI
	test/test-likwidAPI
	make -C test/executable_tests
