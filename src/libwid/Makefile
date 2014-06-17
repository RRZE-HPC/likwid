SRC_DIRS    = ./src
MAKE_DIR   = ../../make

#DO NOT EDIT BELOW

include ../../config.mk
include $(MAKE_DIR)/include_$(COMPILER).mk

INCLUDES  = -I./includes -I../../$(COMPILER) 
ifeq ($(USE_HWLOC),true)
INCLUDES += -I../../ext/hwloc/include
LIBHWLOC = ../../ext/hwloc/libhwloc.so
endif
DEFINES   =
LIBS      = -ldl
LFLAGS    = -fPIC -Wl,-rpath=$(PREFIX)/ext/hwloc
CFLAGS	 += -fPIC 
Q         ?= @

#CONFIGURE BUILD SYSTEM
BUILD_DIR  = ./$(COMPILER)

include $(MAKE_DIR)/config_checks.mk
include $(MAKE_DIR)/config_defines.mk

STATIC_LIB = libwid.a
DYNAMIC_LIB = libwid.so
LUA_INTERFACE = luawid.so

VPATH     = $(SRC_DIRS)
FILES     = $(notdir $(foreach dir,$(SRC_DIRS),$(wildcard $(dir)/*.c)))
OBJ       = $(patsubst %.c, $(BUILD_DIR)/%.o, $(FILES))
OBJ_HWLOC = ../../ext/hwloc/$(COMPILER)/*.o
OBJ_LUA = ../../ext/lua/$(COMPILER)/*.o

CPPFLAGS := $(CPPFLAGS) $(DEFINES) $(INCLUDES)

all: $(BUILD_DIR) $(OBJ) $(DYNAMIC_LIB) $(STATIC_LIB)

$(BUILD_DIR):
	@mkdir $(BUILD_DIR)

$(DYNAMIC_LIB): $(OBJ)
	${Q}$(CC) $(SHARED_LFLAGS) $(SHARED_CFLAGS) -o $(DYNAMIC_LIB) $(OBJ) $(OBJ_HWLOC) $(OBJ_LUA) 

$(STATIC_LIB): $(OBJ)
	${Q}$(AR) -cq $(STATIC_LIB) $(OBJ) $(OBJ_HWLOC)

$(LUA_INTERFACE): $(OBJ) $(OBJ_HWLOC) $(OBJ_LUA)
	$(CC) -Wall -shared -fPIC -pthread $(INCLUDES) -I../../ext/lua/includes lua-wrapper.c $(OBJ) $(OBJ_HWLOC) $(OBJ_LUA) -o $(LUA_INTERFACE) -lpci

test: test_libwid.c
	$(CC) $(CFLAGS) -g -pthread $(INCLUDES) test_libwid.c -o test_libwid $(STATIC_LIB) $(LIBHWLOC) -lm -ldl 
#PATTERN RULES
$(BUILD_DIR)/%.o:  %.c
	${Q}$(CC) -g -c  $(CFLAGS) $(CPPFLAGS) $< -o $@
	${Q}$(CC) -g $(CPPFLAGS) -MT $(@:.d=.o) -MM  $< > $(BUILD_DIR)/$*.d

ifeq ($(findstring $(MAKECMDGOALS),clean),)
-include $(OBJ:.o=.d)
endif

.PHONY: clean distclean $(STATIC_LIB) $(DYNAMIC_LIB)

clean:
	@rm -rf $(BUILD_DIR) 

distclean: clean
	@rm -f $(TARGET) $(STATIC_LIB) $(DYNAMIC_LIB)



