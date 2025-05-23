SRC_DIRS    = ./hwloc
MAKE_DIR   = ../../make

#DO NOT EDIT BELOW

include ../../config.mk
include $(MAKE_DIR)/include_$(COMPILER).mk
include $(MAKE_DIR)/config_checks.mk
include $(MAKE_DIR)/config_defines.mk

CFLAGS    = -O2 -fPIC -fvisibility=hidden
INCLUDES  += -I./include
#DEFINES   =
LIBS      = -L. -lm
LFLAGS    = -fPIC -fvisibility=hidden
Q         ?= @
DEFINES := $(filter-out -DVERSION=$(VERSION),$(DEFINES)) -DRUNSTATEDIR=\"/var/run\"
ifeq ($(DEBUG),true)
DEBUG_FLAGS = -g
else
DEBUG_FLAGS =
endif
ifeq ($(strip $(SANITIZE)),true)
DEBUG_FLAGS += -fsanitize=address -fno-omit-frame-pointer
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
LIBHWLOC = $(shell basename $(TARGET_HWLOC_LIB))
ifeq ($(strip $(COMPILER)), GCCARMv7)
OBJ := $(filter-out $(BUILD_DIR)/topology-x86.o, $(OBJ))
endif
ifeq ($(strip $(COMPILER)), GCCARMv8)
OBJ := $(filter-out $(BUILD_DIR)/topology-x86.o, $(OBJ))
endif
ifeq ($(strip $(COMPILER)), GCCARM)
OBJ := $(filter-out $(BUILD_DIR)/topology-x86.o, $(OBJ))
endif
ifeq ($(strip $(COMPILER)), ARMCLANG)
OBJ := $(filter-out $(BUILD_DIR)/topology-x86.o, $(OBJ))
endif
ifeq ($(strip $(COMPILER)), CLANGARMv8)
OBJ := $(filter-out $(BUILD_DIR)/topology-x86.o, $(OBJ))
endif
ifeq ($(strip $(COMPILER)), FCC)
OBJ := $(filter-out $(BUILD_DIR)/topology-x86.o, $(OBJ))
endif
ifeq ($(COMPILER),GCCPOWER)
OBJ := $(filter-out $(BUILD_DIR)/topology-x86.o,$(OBJ))
endif
ifeq ($(COMPILER),XLC)
OBJ := $(filter-out $(BUILD_DIR)/topology-x86.o,$(OBJ))
endif


CPPFLAGS := $(CPPFLAGS) $(DEFINES) $(INCLUDES)

all: $(LIBHWLOC)

$(STATIC_LIBHWLOC): $(OBJ)
	@echo "===>  CREATE STATIC LIB  $@"
	$(Q)${AR} -cq $(LIBHWLOC) $(OBJ)

$(SHARED_LIBHWLOC): $(OBJ)
	@echo "===>  CREATE SHARED LIB  $@"
	$(Q)$(CC) $(DEBUG_FLAGS) $(LFLAGS) -Wl,-soname,$(LIBHWLOC).$(VERSION).$(RELEASE),--no-undefined -Wall -shared -fPIC -o $(LIBHWLOC) $^ $(LIBS) $(RPATHS)

#PATTERN RULES
$(BUILD_DIR)/%.o:  %.c
	@echo "===>  COMPILE  $@"
	@mkdir -p $(BUILD_DIR)
	$(Q)$(CC) -c $(DEBUG_FLAGS) $(CFLAGS) $(CPPFLAGS) $< -o $@
	$(Q)$(CC) $(DEBUG_FLAGS) $(CPPFLAGS) -MT $(@:.d=.o) -MM  $< > $(BUILD_DIR)/$*.d

ifeq ($(findstring $(MAKECMDGOALS),clean),)
-include $(OBJ:.o=.d)
endif

.PHONY: clean distclean

clean:
	@echo "===>  CLEAN HWLOC"
	@rm -f $(TARGET) $(SHARED_LIBHWLOC) $(STATIC_LIBHWLOC) $(LIBHWLOC).$(VERSION).$(RELEASE) $(LIBHWLOC).$(VERSION)

distclean: clean
	@echo "===>  DIST CLEAN HWLOC"
	@rm -rf $(BUILD_DIR)



