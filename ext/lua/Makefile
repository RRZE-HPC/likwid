SRC_DIRS    = ./src
MAKE_DIR   = ../../make

#DO NOT EDIT BELOW

include ../../config.mk
include $(MAKE_DIR)/include_$(COMPILER).mk
include $(MAKE_DIR)/config_checks.mk
include $(MAKE_DIR)/config_defines.mk

CFLAGS    = -O2 -Wall -fPIC
INCLUDES  = -I./includes
DEFINES   = -DLUA_COMPAT_ALL -DLUA_USE_LINUX
LIBS      = -lm -Wl,-E -ldl

Q         ?= @
ifeq ($(DEBUG),true)
DEBUG_FLAGS = -g
else
DEBUG_FLAGS =
endif
ifeq ($(COMPILER),MIC)
CFLAGS += -mmic
LFLAGS += -mmic
endif


#CONFIGURE BUILD SYSTEM
BUILD_DIR  = ./$(COMPILER)

VPATH     = $(SRC_DIRS)
FILES     = $(notdir $(foreach dir,$(SRC_DIRS),$(wildcard $(dir)/*.c)))
OBJ       = $(patsubst %.c, $(BUILD_DIR)/%.o, $(FILES))

CPPFLAGS := $(CPPFLAGS) $(DEFINES) $(INCLUDES)

LIBLUA_NAME = $(shell basename $(TARGET_LUA_LIB))
INTERPRETER = lua

all: $(BUILD_DIR) $(OBJ)  $(TARGET_LUA_LIB) $(INTERPRETER)

$(BUILD_DIR):
	@mkdir $(BUILD_DIR)

$(STATIC_LIBLUA):
	$(Q)${AR} -cq $(LIBLUA_NAME) $(OBJ)

$(SHARED_LIBLUA):
	$(Q)$(CC) $(LFLAGS) -Wl,-soname,$(LIBLUA_NAME) -shared -fPIC -o $(LIBLUA_NAME) $(OBJ) $(LIBS)

$(INTERPRETER): $(SHARED_LIBLUA) $(BUILD_DIR)/lua.o
	$(Q)$(CC) -o $@ $(LFLAGS) $(BUILD_DIR)/lua.o $(LIBLUA_NAME) -Wl,-rpath=$(PREFIX)/lib $(LIBS)

#PATTERN RULES
$(BUILD_DIR)/%.o:  %.c
	$(Q)$(CC) -c $(DEBUG_FLAGS) $(CFLAGS) $(CPPFLAGS) $< -o $@
	$(Q)$(CC) $(DEBUG_FLAGS) $(CPPFLAGS) -MT $(@:.d=.o) -MM  $< > $(BUILD_DIR)/$*.d

ifeq ($(findstring $(MAKECMDGOALS),clean),)
-include $(OBJ:.o=.d)
endif

.PHONY: clean distclean $(INTERPRETER)

clean:
	@rm -rf $(BUILD_DIR)

distclean: clean
	@rm -f $(TARGET) $(INTERPRETER) $(shell basename $(SHARED_LIBLUA)) $(shell basename $(STATIC_LIBLUA))



