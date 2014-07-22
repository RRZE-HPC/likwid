SRC_DIRS    = ./src
MAKE_DIR   = ../../make

#DO NOT EDIT BELOW

include ../../config.mk
include $(MAKE_DIR)/include_$(COMPILER).mk

CFLAGS    = -O2 -Wall -fPIC
INCLUDES  = -I./includes
DEFINES   = -DLUA_COMPAT_ALL -DLUA_USE_LINUX
LIBS      = -lm -Wl,-E -ldl -lreadline
LFLAGS    =
Q         ?= @

#CONFIGURE BUILD SYSTEM
BUILD_DIR  = ./$(COMPILER)

VPATH     = $(SRC_DIRS)
FILES     = $(notdir $(foreach dir,$(SRC_DIRS),$(wildcard $(dir)/*.c)))
OBJ       = $(patsubst %.c, $(BUILD_DIR)/%.o, $(FILES))

CPPFLAGS := $(CPPFLAGS) $(DEFINES) $(INCLUDES)

STATIC_LIBLUA = liblua.a
SHARED_LIBLUA = liblua.so
INTERPRETER = lua

all: $(BUILD_DIR) $(OBJ) $(INTERPRETER)

$(BUILD_DIR):
	@mkdir $(BUILD_DIR)

$(STATIC_LIBLUA):
	$(Q)${AR} -cq $(STATIC_LIBLUA) $(OBJ)

$(SHARED_LIBLUA):
	$(Q)$(CC) $(LFLAGS) -shared -fPIC -o $(SHARED_LIBLUA) $(OBJ)

$(INTERPRETER):
	$(Q)$(CC) -o $@ $(LFLAGS) $(BUILD_DIR)/lua.o $(STATIC_LIBLUA) $(LIBS)

#PATTERN RULES
$(BUILD_DIR)/%.o:  %.c
	${Q}$(CC) -c  $(CFLAGS) $(CPPFLAGS) $< -o $@
	${Q}$(CC) $(CPPFLAGS) -MT $(@:.d=.o) -MM  $< > $(BUILD_DIR)/$*.d

ifeq ($(findstring $(MAKECMDGOALS),clean),)
-include $(OBJ:.o=.d)
endif

.PHONY: clean distclean

clean:
	@rm -rf $(BUILD_DIR)

distclean: clean
	@rm -f $(TARGET)



