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


DYNAMIC_TARGET_LIB := liblikwid.so
STATIC_TARGET_LIB := liblikwid.a
LIBHWLOC = ext/hwloc/libhwloc.a
LIBLUA = ext/lua/liblua.a
ifeq ($(SHARED_LIBRARY),true)
CFLAGS += $(SHARED_CFLAGS)
LIBS += -L. -pthread -lm -lpci
TARGET_LIB := $(DYNAMIC_TARGET_LIB)
LIBHWLOC = ext/hwloc/libhwloc.a
LIBLUA = ext/lua/liblua.a
else
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
OBJ_LUA    =  $(wildcard ./ext/lua/$(COMPILER)/*.o)
OBJ_HWLOC  =  $(wildcard ./ext/hwloc/$(COMPILER)/*.o)
BENCH_TARGET = likwid-bench

L_APPS      =   likwid-perfctr \
				likwid-pin \
				likwid-powermeter \
				likwid-topology \
				likwid-memsweeper \
				likwid-genTopoCfg
L_HELPER    =   likwid.lua
ifeq ($(BUILDFREQ),true)
	L_APPS += likwid-setFrequencies
endif

CPPFLAGS := $(CPPFLAGS) $(DEFINES) $(INCLUDES)

all: $(BUILD_DIR) $(EXT_TARGETS) $(PERFMONHEADERS) $(OBJ) $(OBJ_BENCH) $(STATIC_TARGET_LIB) $(DYNAMIC_TARGET_LIB) $(FORTRAN_INTERFACE)  $(PINLIB) $(L_APPS) $(L_HELPER) $(DAEMON_TARGET) $(FREQ_TARGET) $(BENCH_TARGET)

tags:
	@echo "===>  GENERATE  TAGS"
	$(Q)ctags -R

docs:
	@echo "===>  GENERATE DOXYGEN DOCS"
	$(Q)doxygen doc/Doxyfile

$(L_APPS):  $(addprefix $(SRC_DIR)/applications/,$(addsuffix  .lua,$(L_APPS)))
	@echo "===>  ADJUSTING  $@"
	@sed -e s/'<PREFIX>'/$(subst /,\\/,$(PREFIX))/g \
		-e s/'<VERSION>'/$(VERSION).$(RELEASE)/g \
		-e s/'<DATE>'/$(DATE)/g \
		$(addprefix $(SRC_DIR)/applications/,$(addsuffix  .lua,$@)) > $@

$(L_HELPER):
	@echo "===>  ADJUSTING  $@"
	@sed -e s/'<PREFIX>'/$(subst /,\\/,$(PREFIX))/g \
		-e s/'<VERSION>'/$(VERSION)/g \
		-e s/'<RELEASE>'/$(RELEASE)/g \
		$(SRC_DIR)/applications/$@ > $@

$(STATIC_TARGET_LIB): $(OBJ)
	@echo "===>  CREATE STATIC LIB  $(STATIC_TARGET_LIB)"
	$(Q)${AR} -crus $(STATIC_TARGET_LIB) $(OBJ) $(LIBHWLOC) $(LIBLUA)


$(DYNAMIC_TARGET_LIB): $(OBJ)
	@echo "===>  CREATE SHARED LIB  $(DYNAMIC_TARGET_LIB)"
	$(Q)${CC} $(DEBUG_FLAGS) $(SHARED_LFLAGS) $(SHARED_CFLAGS) -o $(DYNAMIC_TARGET_LIB) $(OBJ) -L. -pthread -lm -lpci $(LIBHWLOC) $(LIBLUA)

$(DAEMON_TARGET): $(SRC_DIR)/access-daemon/accessDaemon.c
	@echo "===>  Build access daemon likwid-accessD"
	$(Q)$(MAKE) -C  $(SRC_DIR)/access-daemon likwid-accessD

$(FREQ_TARGET): $(SRC_DIR)/access-daemon/setFreq.c
	@echo "===>  Build frequency daemon likwid-setFreq"
	$(Q)$(MAKE) -C  $(SRC_DIR)/access-daemon likwid-setFreq

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

$(BENCH_TARGET):
	@echo "===>  ENTER  $@"
	$(Q)$(MAKE) --no-print-directory -C bench $(MAKECMDGOALS)

#PATTERN RULES
$(BUILD_DIR)/%.o:  %.c
	@echo "===>  COMPILE  $@"
	$(Q)$(CC) -g -c $(DEBUG_FLAGS) $(CFLAGS) $(ANSI_CFLAGS) $(CPPFLAGS) $< -o $@
	$(Q)$(CC) -g $(DEBUG_FLAGS) $(CPPFLAGS) -MT $(@:.d=.o) -MM  $< > $(BUILD_DIR)/$*.d

$(BUILD_DIR)/%.o:  %.cc
	@echo "===>  COMPILE  $@"
	$(Q)$(CXX) -c $(DEBUG_FLAGS) $(CXXFLAGS) $(CPPFLAGS) $< -o $@
	$(Q)$(CXX) $(DEBUG_FLAGS) $(CXXFLAGS) $(CPPFLAGS) -MT $(@:.d=.o) -MM  $< > $(BUILD_DIR)/$*.d


$(BUILD_DIR)/%.h:  $(SRC_DIR)/includes/%.txt
	@echo "===>  GENERATE HEADER $@"
	$(Q)$(GEN_PMHEADER) $< $@


ifeq ($(findstring $(MAKECMDGOALS),clean),)
-include $(OBJ:.o=.d)
endif

.PHONY: clean distclean install uninstall $(EXT_TARGETS)


.PRECIOUS: $(BUILD_DIR)/%.pas

.NOTPARALLEL:


clean: $(EXT_TARGETS) $(BENCH_TARGET)
	@echo "===>  CLEAN"
	@rm -rf $(BUILD_DIR)
	@rm -f $(GENGROUPLOCK)

distclean: clean
	@echo "===>  DIST CLEAN"
	@rm -f likwid-*
	@rm -f likwid.lua
	@rm -f $(STATIC_TARGET_LIB)
	@rm -f $(DYNAMIC_TARGET_LIB)
	@rm -f $(FORTRAN_INTERFACE)
	@rm -f $(FREQ_TARGET) $(DAEMON_TARGET)
	@rm -f $(PINLIB)
	@rm -rf doc/html
	@rm -f tags

install:
	@echo "===> INSTALL applications to $(PREFIX)/bin"
	@mkdir -p $(PREFIX)/bin
	for APP in $(L_APPS); do \
		cp -f $$APP  $(PREFIX)/bin; \
	done
	@cp ext/lua/lua $(PREFIX)/bin/likwid-lua
	@chmod 755 $(PREFIX)/bin/likwid-*
	@echo "===> INSTALL lua to likwid interface to $(PREFIX)/share/lua"
	@mkdir -p $(PREFIX)/share/lua
	@cp -f likwid.lua $(PREFIX)/share/lua
	@echo "===> INSTALL libraries to $(PREFIX)/lib"
	@mkdir -p $(PREFIX)/lib
	@cp -f liblikwid*  $(PREFIX)/lib
	@cp -f ext/lua/liblua* $(PREFIX)/lib
	@cp -f ext/hwloc/libhwloc* $(PREFIX)/lib
	@chmod 755 $(PREFIX)/lib/$(PINLIB)
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
	@cp -f src/includes/likwid.h  $(PREFIX)/include/
	$(FORTRAN_INSTALL)
	@echo "===> INSTALL groups to $(PREFIX)/share/likwid"
	@mkdir -p $(PREFIX)/share/likwid
	@cp -rf groups/* $(PREFIX)/share/likwid
	@echo "===> INSTALL filters to $(LIKWIDFILTERPATH)"
	@mkdir -p $(LIKWIDFILTERPATH)
	@cp -f filters/*  $(LIKWIDFILTERPATH)
	@chmod 755 $(LIKWIDFILTERPATH)/*
	@[ -e $(DAEMON_TARGET) ] && echo "===> INSTALL access daemon to $(ACCESSDAEMON)"
	@[ -e $(DAEMON_TARGET) ] && mkdir -p `dirname $(ACCESSDAEMON)`
	@[ -e $(DAEMON_TARGET) ] && cp -f $(DAEMON_TARGET) $(ACCESSDAEMON)
	@[ -e $(ACCESSDAEMON) ] && chown root:root $(ACCESSDAEMON)
	@[ -e $(ACCESSDAEMON) ] && chmod 4755 $(ACCESSDAEMON)
	@[ -e $(FREQ_TARGET) ] && echo "===> INSTALL setFrequencies tool to $(PREFIX)/sbin/$(FREQ_TARGET)"
	@[ -e $(FREQ_TARGET) ] && mkdir -p $(PREFIX)/sbin
	@[ -e $(FREQ_TARGET) ] && cp -f $(FREQ_TARGET) $(PREFIX)/sbin/$(FREQ_TARGET)
	@[ -e $(PREFIX)/sbin/$(FREQ_TARGET) ] && chown root:root $(PREFIX)/sbin/$(FREQ_TARGET)
	@[ -e $(PREFIX)/sbin/$(FREQ_TARGET) ] && chmod 4755 $(PREFIX)/sbin/$(FREQ_TARGET)


uninstall:
	@echo "===> REMOVING applications from $(PREFIX)/bin"
	@rm -f $(addprefix $(PREFIX)/bin/,$(addsuffix  .lua,$(L_APPS)))
	@for APP in $(L_APPS); do \
		rm -f $(PREFIX)/bin/$$APP; \
	done
	@for APP in $(C_APPS); do \
		rm -f $(PREFIX)/bin/$$APP; \
	done
	rm -rf $(PREFIX)/bin/likwid-lua
	@echo "===> REMOVING Lua to likwid interface from $(PREFIX)/share/lua"
	@rm -rf  $(PREFIX)/share/lua/likwid.lua
	@echo "===> REMOVING libs from $(PREFIX)/lib"
	@rm -f $(PREFIX)/lib/liblikwid*
	@rm -f $(PREFIX)/lib/libhwloc*
	@rm -f $(PREFIX)/lib/liblua*
	@echo "===> REMOVING man pages from $(MANPREFIX)/man1"
	@rm -f $(addprefix $(MANPREFIX)/man1/,$(addsuffix  .1,$(L_APPS)))
	@echo "===> REMOVING header from $(PREFIX)/include"
	@rm -f $(PREFIX)/include/likwid.h
	$(FORTRAN_REMOVE)
	@echo "===> REMOVING filter and groups from $(PREFIX)/share/likwid"
	@rm -rf  $(PREFIX)/share/likwid
	@[ -e $(ACCESSDAEMON) ] && echo "===> REMOVING access daemon from $(ACCESSDAEMON)"
	@[ -e $(ACCESSDAEMON) ] && rm -f $(ACCESSDAEMON)
	@[ -e $(PREFIX)/sbin/$(FREQ_TARGET) ] && echo "===> REMOVING setFrequencies tool from $(PREFIX)/sbin/$(FREQ_TARGET)"
	@[ -e $(PREFIX)/sbin/$(FREQ_TARGET) ] && rm -f $(PREFIX)/sbin/$(FREQ_TARGET)



