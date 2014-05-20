SRC_DIRS    = ./src
MAKE_DIR   = ../../make

#DO NOT EDIT BELOW

include ../../config.mk
include $(MAKE_DIR)/include_$(COMPILER).mk

CFLAGS    = -O2 -Wall -fPIC
INCLUDES  = -I./include
DEFINES   =
LIBS      = -lm -Wl,-E
LFLAGS    =
Q         ?= @

#CONFIGURE BUILD SYSTEM
BUILD_DIR  = ./$(COMPILER)

VPATH     = $(SRC_DIRS)
FILES     = $(notdir $(foreach dir,$(SRC_DIRS),$(wildcard $(dir)/*.c)))
OBJ       = $(patsubst %.c, $(BUILD_DIR)/%.o, $(FILES))

LIBHWLOC = libhwloc.a

CPPFLAGS := $(CPPFLAGS) $(DEFINES) $(INCLUDES)

all: $(BUILD_DIR) $(OBJ) $(LIBHWLOC)

$(BUILD_DIR):
	@mkdir $(BUILD_DIR)

$(LIBHWLOC):
	$(Q)${AR} -cq $(LIBHWLOC) $(OBJ)

#PATTERN RULES
$(BUILD_DIR)/%.o:  %.c
	${Q}$(CC) -c  $(CFLAGS) $(CPPFLAGS) $< -o $@
	${Q}$(CC) $(CPPFLAGS) -MT $(@:.d=.o) -MM  $< > $(BUILD_DIR)/$*.d

ifeq ($(findstring $(MAKECMDGOALS),clean),)
-include $(OBJ:.o=.d)
endif

.PHONY: clean distclean

clean:
	@rm -rf $(BUILD_DIR) $(LIBHWLOC)

distclean: clean
	@rm -f $(TARGET)



