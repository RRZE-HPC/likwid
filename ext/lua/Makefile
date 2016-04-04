SRC_DIRS    = ./src
MAKE_DIR   = ../../make

#DO NOT EDIT BELOW

include ../../config.mk
include $(MAKE_DIR)/include_$(COMPILER).mk
include $(MAKE_DIR)/config_checks.mk
include $(MAKE_DIR)/config_defines.mk

CFLAGS    = -O2 -Wall -fPIC
INCLUDES  = -I./includes
DEFINES   = -DLUA_COMPAT_ALL -DLUA_COMPAT_5_2 -DLUA_USE_LINUX
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

LIBLUA = $(shell basename $(TARGET_LUA_LIB))
INTERPRETER = lua

all: $(BUILD_DIR) $(OBJ)  $(LIBLUA) $(INTERPRETER)

$(BUILD_DIR):
	@mkdir $(BUILD_DIR)

$(STATIC_LIBLUA): $(OBJ)
	$(Q)${AR} -cq $(LIBLUA) $(OBJ)

$(SHARED_LIBLUA): $(OBJ)
	$(Q)$(CC) $(LFLAGS) -Wl,-soname,$(LIBLUA).$(VERSION).$(RELEASE) -shared -fPIC -o $(LIBLUA) $(OBJ) $(LIBS)

$(INTERPRETER): $(LIBLUA) $(BUILD_DIR)/lua.o
	$(Q)$(CC) -o $@ $(LFLAGS) $(BUILD_DIR)/lua.o $(RPATHS) $(LIBS) $(LIBLUA)

#PATTERN RULES
$(BUILD_DIR)/%.o:  %.c
	$(Q)$(CC) -c $(DEBUG_FLAGS) $(CFLAGS) $(CPPFLAGS) $< -o $@
	$(Q)$(CC) $(DEBUG_FLAGS) $(CPPFLAGS) -MT $(@:.d=.o) -MM  $< > $(BUILD_DIR)/$*.d

ifeq ($(findstring $(MAKECMDGOALS),clean),)
-include $(OBJ:.o=.d)
endif

.PHONY: clean distclean

clean:
	@rm -f $(TARGET) $(INTERPRETER) $(SHARED_LIBLUA) $(STATIC_LIBLUA) $(LIBLUA).$(VERSION).$(RELEASE) $(LIBLUA).$(VERSION)

distclean: clean
	@rm -f $(TARGET) $(INTERPRETER) $(SHARED_LIBLUA) $(STATIC_LIBLUA) $(LIBLUA).$(VERSION).$(RELEASE) $(LIBLUA).$(VERSION)
	@rm -rf $(BUILD_DIR)



