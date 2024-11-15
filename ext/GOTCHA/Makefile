# =======================================================================================
#
#      Filename:  Makefile
#
#      Description:  GOTCHA Makefile
#
#      Version:   <VERSION>
#      Released:  <DATE>
#
#      Author:  Thomas Gruber (tr), thomas.roehl@googlemail.com
#      Project:  likwid
#
#      Copyright (C) 2015 RRZE, University Erlangen-Nuremberg
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

SRC_DIRS    = ./src
MAKE_DIR   = ../../make

#DO NOT EDIT BELOW

include ../../config.mk
include $(MAKE_DIR)/include_$(COMPILER).mk
include $(MAKE_DIR)/config_checks.mk
include $(MAKE_DIR)/config_defines.mk

CFLAGS    = -O2 -Wall -fPIC
INCLUDES  = -I./include
LIBS      = -lm -Wl,-E -ldl

Q         ?= @
ifeq ($(strip $(DEBUG)),true)
DEBUG_FLAGS = -g
else
DEBUG_FLAGS =
endif
ifeq ($(strip $(SANITIZE)),true)
DEBUG_FLAGS += -fsanitize=address -fno-omit-frame-pointer
endif
ifeq ($(strip $(COMPILER)),MIC)
CFLAGS += -mmic
LFLAGS += -mmic
endif


#CONFIGURE BUILD SYSTEM
BUILD_DIR  = ./$(COMPILER)

VPATH     = $(SRC_DIRS)
FILES     = $(notdir $(foreach dir,$(SRC_DIRS),$(wildcard $(dir)/*.c)))
OBJ       = $(patsubst %.c, $(BUILD_DIR)/%.o, $(FILES))

CPPFLAGS := $(CPPFLAGS) $(DEFINES) $(INCLUDES)
CPPFLAGS := $(filter-out -D_GNU_SOURCE,$(CPPFLAGS))


LIBGOTCHA = $(shell basename $(TARGET_GOTCHA_LIB))
INTERPRETER = lua

.NOTPARALLEL: all

all: $(BUILD_DIR) $(OBJ)  $(LIBGOTCHA)

$(BUILD_DIR):
	@mkdir $(BUILD_DIR)

$(STATIC_LIBGOTCHA): $(OBJ)
	$(info "GOTCHA cannot be build statically in LIKWID")
	#$(Q)${AR} -cq $(LIBGOTCHA) $(OBJ)

$(SHARED_LIBGOTCHA): $(OBJ)
	$(Q)$(CC) $(DEBUG_FLAGS) $(LFLAGS) -Wl,-soname,$(LIBGOTCHA).$(VERSION).$(RELEASE) -shared -fPIC -o $(LIBGOTCHA) $(OBJ) $(LIBS)

#PATTERN RULES
$(BUILD_DIR)/%.o:  %.c
	$(Q)$(CC) -c $(DEBUG_FLAGS) $(CFLAGS) $(CPPFLAGS) $< -o $@
	$(Q)$(CC) $(DEBUG_FLAGS) $(CPPFLAGS) -MT $(@:.d=.o) -MM  $< > $(BUILD_DIR)/$*.d

ifeq ($(findstring $(MAKECMDGOALS),clean),)
-include $(OBJ:.o=.d)
endif

.PHONY: clean distclean

clean:
	@rm -f $(TARGET) $(SHARED_LIBGOTCHA) $(STATIC_LIBGOTCHA) $(LIBGOTCHA).$(VERSION).$(RELEASE) $(LIBGOTCHA).$(VERSION)

distclean: clean
	@rm -f $(TARGET) $(SHARED_LIBGOTCHA) $(STATIC_LIBGOTCHA) $(LIBGOTCHA).$(VERSION).$(RELEASE) $(LIBGOTCHA).$(VERSION)
	@rm -rf $(BUILD_DIR)
