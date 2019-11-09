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
	$(Q)$(CC) $(LFLAGS) -Wl,-soname,$(LIBGOTCHA).$(VERSION).$(RELEASE) -shared -fPIC -o $(LIBGOTCHA) $(OBJ) $(LIBS)

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
