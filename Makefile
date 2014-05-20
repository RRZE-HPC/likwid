#===========================================================================
#
#     Filename:  Makefile
#
#     Description:  Central Makefile
#
#     Version:  <VERSION>
#     Created:  <DATE>
#
#     Author:  Jan Treibig (jt), jan.treibig@gmail.com
#     Company:  RRZE Erlangen
#     Project:  likwid
#     Copyright:  Copyright (c) 2010, Jan Treibig
#
#     This program is free software; you can redistribute it and/or modify
#     it under the terms of the GNU General Public License, v2, as
#     published by the Free Software Foundation
#    
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#    
#     You should have received a copy of the GNU General Public License
#     along with this program; if not, write to the Free Software
#     Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
#
#===========================================================================

SRC_DIR    = ./src
DOC_DIR    = ./doc
BENCH_DIR  = ./bench
GROUP_DIR  = ./groups
MAKE_DIR   = ./

#DO NOT EDIT BELOW

# determine kernel Version
KERNEL_VERSION := $(shell uname -r | awk -F- '{ print $$1 }' | awk -F. '{ print $$3 }')

HAS_MEMPOLICY = $(shell if [ $(KERNEL_VERSION) -lt 7 ]; then \
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
			 -DMAX_NUM_SOCKETS=$(MAX_NUM_SOCKETS) \
			 -DLIBLIKWIDPIN=$(LIBLIKWIDPIN)

#CONFIGURE BUILD SYSTEM
BUILD_DIR  = ./$(COMPILER)
Q         ?= @
GENGROUPLOCK = .gengroup
TARGET_LIB := liblikwid.a

ifneq ($(COLOR),NONE)
DEFINES += -DCOLOR=$(COLOR)
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


VPATH     = $(SRC_DIR)
OBJ       = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o,$(wildcard $(SRC_DIR)/*.c))
OBJ      += $(patsubst $(SRC_DIR)/osdep/%.c, $(BUILD_DIR)/osdep/%.o,$(filter-out $(wildcard $(SRC_DIR)/osdep/*_win.c),$(wildcard $(SRC_DIR)/osdep/*.c)))
ifeq ($(MAKECMDGOALS),likwid-bench)
OBJ      += $(patsubst $(BENCH_DIR)/%.ptt, $(BUILD_DIR)/%.o,$(wildcard $(BENCH_DIR)/*.ptt))
endif
APPS      = likwid-perfctr  \
			likwid-features \
			likwid-topology \
			likwid-pin      \
            likwid-bench

CPPFLAGS := $(CPPFLAGS) $(DEFINES) $(INCLUDES) 

all: $(BUILD_DIR) $(GENGROUPLOCK) $(OBJ) $(filter-out likwid-bench,$(APPS)) $(TARGET_LIB)  $(PINLIB) 

tags:
	@echo "===>  GENERATE  TAGS"
	$(Q)ctags -R

$(APPS):  $(addprefix $(SRC_DIR)/applications/,$(addsuffix  .c,$(APPS))) $(BUILD_DIR) $(GENGROUPLOCK)  $(OBJ)
	@echo "===>  LINKING  $@"
	$(Q)${CC} $(CFLAGS) $(ANSI_CFLAGS) $(CPPFLAGS) ${LFLAGS} -o $@  $(addprefix $(SRC_DIR)/applications/,$(addsuffix  .c,$@)) $(OBJ) $(LIBS)

$(TARGET_LIB): $(OBJ)
	@echo "===>  CREATE LIB  $(TARGET_LIB)"
	$(Q)${AR} -cq $(TARGET_LIB) $(filter-out $(BUILD_DIR)/main.o,$(OBJ))

$(BUILD_DIR):
	@mkdir $(BUILD_DIR)

$(PINLIB): 
	@echo "===>  CREATE LIB  $(PINLIB)"
	$(Q)$(MAKE) -s -C src/pthread-overload/ $(PINLIB) 

$(GENGROUPLOCK): $(foreach directory,$(shell ls $(GROUP_DIR)), $(wildcard $(GROUP_DIR)/$(directory)/*.txt))
	@echo "===>  GENERATE GROUP HEADERS"
	$(Q)$(GEN_GROUPS) ./groups  $(BUILD_DIR) ./perl/templates
	$(Q)touch $(GENGROUPLOCK)


#PATTERN RULES
$(BUILD_DIR)/%.o:  %.c
	@echo "===>  COMPILE  $@"
	$(Q)$(CC) -c  $(CFLAGS) $(ANSI_CFLAGS) $(CPPFLAGS) $< -o $@
	$(Q)$(CC) $(CPPFLAGS) -MT $(@:.d=.o) -MM  $< > $(BUILD_DIR)/$*.d

$(BUILD_DIR)/%.pas:  $(BENCH_DIR)/%.ptt
	@echo "===>  GENERATE BENCHMARKS"
	$(Q)$(GEN_PAS) ./bench  $(BUILD_DIR) ./perl/templates

$(BUILD_DIR)/%.o:  $(BUILD_DIR)/%.pas
	@echo "===>  ASSEMBLE  $@"
	$(Q)$(PAS) -i x86-64 -o $(BUILD_DIR)/$*.s $<  '$(DEFINES)'
	$(Q)$(AS) $(ASFLAGS)  $(BUILD_DIR)/$*.s -o $@

ifeq ($(findstring $(MAKECMDGOALS),clean),)
-include $(OBJ:.o=.d)
endif

.PHONY: clean distclean install uninstall

.PRECIOUS: $(BUILD_DIR)/%.pas

clean:
	@echo "===>  CLEAN"
	@rm -rf $(BUILD_DIR)

distclean: clean
	@echo "===>  DIST CLEAN"
	@rm -f likwid-*
	@rm -f $(TARGET_LIB)
	@rm -f $(PINLIB)
	@rm -f $(GENGROUPLOCK)

install:
	@echo "===> INSTALL applications to $(PREFIX)/bin"
	@mkdir -p $(PREFIX)/bin
	@cp -f likwid-*  $(PREFIX)/bin
	@chmod 755 $(PREFIX)/bin/likwid-*
	@echo "===> INSTALL man pages to $(MANPREFIX)/man1"
	@mkdir -p $(MANPREFIX)/man1
	@sed -e "s/<VERSION>/$(VERSION)/g" -e "s/<DATE>/$(DATE)/g" < $(DOC_DIR)/likwid-topology.1 > $(MANPREFIX)/man1/likwid-topology.1
	@sed -e "s/<VERSION>/$(VERSION)/g" -e "s/<DATE>/$(DATE)/g" < $(DOC_DIR)/likwid-features.1 > $(MANPREFIX)/man1/likwid-features.1
	@sed -e "s/<VERSION>/$(VERSION)/g" -e "s/<DATE>/$(DATE)/g" < $(DOC_DIR)/likwid-perfctr.1 > $(MANPREFIX)/man1/likwid-perfctr.1
	@sed -e "s/<VERSION>/$(VERSION)/g" -e "s/<DATE>/$(DATE)/g" < $(DOC_DIR)/likwid-pin.1 > $(MANPREFIX)/man1/likwid-pin.1
	@echo "===> INSTALL header to $(PREFIX)/include"
	@mkdir -p $(PREFIX)/include
	@cp -f src/includes/likwid.h  $(PREFIX)/include
	@chmod 644 $(MANPREFIX)/man1/likwid-*
	@echo "===> INSTALL libraries to $(PREFIX)/lib"
	@mkdir -p $(PREFIX)/lib
	@cp -f liblikwid*  $(PREFIX)/lib
	@chmod 755 $(PREFIX)/lib/$(PINLIB)

	
uninstall:
	@echo "===> REMOVING applications from $(PREFIX)/bin"
	@rm -f $(addprefix $(PREFIX)/bin/,$(APPS)) 
	@echo "===> REMOVING man pages from $(MANPREFIX)/man1"
	@rm -f $(addprefix $(MANPREFIX)/man1/,$(addsuffix  .1,$(APPS))) 
	@echo "===> REMOVING libs from $(PREFIX)/lib"
	@rm -f $(PREFIX)/lib/$(TARGET_LIB) $(PREFIX)/lib/$(PINLIB)



