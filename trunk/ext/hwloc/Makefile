SRC_DIRS    = ./src
MAKE_DIR   = ../../make

#DO NOT EDIT BELOW

include ../../config.mk
include $(MAKE_DIR)/include_$(COMPILER).mk
LIBPCI_PATH = $(shell find /*/lib* -name 'libpci.so*' 2>/dev/null | head -n 1 )
ifneq ($(LIBPCI_PATH),"")
	LIBPCI_PATH = $(shell find /lib* -name 'libpci.so*' 2>/dev/null | head -n 1 )
endif

CFLAGS    = -O2 -Wall -fPIC
INCLUDES  = -I./include
DEFINES   =
LIBS      = -L. -lm -lpci -Wl,-E
LFLAGS    = -Wno-unused-result
Q         ?= @
PCILIB    = libpci.so

#CONFIGURE BUILD SYSTEM
BUILD_DIR  = ./$(COMPILER)

VPATH     = $(SRC_DIRS)
FILES     = $(notdir $(foreach dir,$(SRC_DIRS),$(wildcard $(dir)/*.c)))
OBJ       = $(patsubst %.c, $(BUILD_DIR)/%.o, $(FILES))



STATIC_LIBHWLOC = libhwloc.a
SHARED_LIBHWLOC = libhwloc.so

CPPFLAGS := $(CPPFLAGS) $(DEFINES) $(INCLUDES)

all: $(BUILD_DIR) $(PCILIB) $(OBJ) $(STATIC_LIBHWLOC) $(SHARED_LIBHWLOC)

$(BUILD_DIR):
	@mkdir $(BUILD_DIR)

$(PCILIB): $(LIBPCI_PATH)
	@ln -s $(LIBPCI_PATH) $@

$(STATIC_LIBHWLOC):
	$(Q)${AR} -cq $(STATIC_LIBHWLOC) $(OBJ)

$(SHARED_LIBHWLOC):
	${Q}$(CC) $(LFLAGS) -Wall -shared -fPIC -o $(SHARED_LIBHWLOC) $(OBJ) $(LIBS)

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
	@rm -f $(TARGET) $(STATIC_LIBHWLOC) $(SHARED_LIBHWLOC) $(PCILIB)



