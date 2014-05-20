include ./config.mk

#CONFIGURE BUILD SYSTEM
TAG        = $(COMPILER)
BUILD_DIR  = ./$(TAG)
SRC_DIR    = ./src
DOC_DIR    = ./doc
MAKE_DIR   = ./
Q         ?= @

#DO NOT EDIT BELOW
include $(MAKE_DIR)/include_$(TAG).mk
INCLUDES  += -I./src/includes
DEFINES   += -DVERSION=$(VERSION) \
			 -DRELEASE=$(RELEASE) \
			 -DMAX_NUM_THREADS=$(MAX_NUM_THREADS) \
			 -DMAX_NUM_SOCKETS=$(MAX_NUM_SOCKETS) \
			 -DLIBLIKWIDPIN=$(LIBLIKWIDPIN)

ifneq ($(COLOR),NONE)
DEFINES += -DCOLOR=$(COLOR)
endif


TARGET_LIB = liblikwid.a
PINLIB  = liblikwidpin.so

VPATH     = $(SRC_DIR)
OBJ       = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o,$(wildcard $(SRC_DIR)/*.c))
APPS      = likwid-perfCtr  \
			likwid-features \
			likwid-topology \
			likwid-pin

CPPFLAGS := $(CPPFLAGS) $(DEFINES) $(INCLUDES) 

all: $(BUILD_DIR) $(OBJ) $(APPS) $(TARGET_LIB)  $(PINLIB) 

tags:
	@echo "===>  GENERATE  TAGS"
	$(Q)ctags -R

$(APPS):  $(addprefix $(SRC_DIR)/applications/,$(addsuffix  .c,$(APPS))) $(OBJ)
	@echo "===>  LINKING  $@"
	$(Q)${CC} $(CFLAGS) $(CPPFLAGS) ${LFLAGS} -o $@  $(addprefix $(SRC_DIR)/applications/,$(addsuffix  .c,$@)) $(OBJ)

$(TARGET_LIB): $(OBJ)
	@echo "===>  CREATE LIB  $(TARGET_LIB)"
	$(Q)${AR} -cq $(TARGET_LIB) $(filter-out $(BUILD_DIR)/main.o,$(OBJ))

$(BUILD_DIR):
	@mkdir $(BUILD_DIR)

$(PINLIB): 
	@echo "===>  CREATE LIB  $(PINLIB)"
	$(Q)$(MAKE) -s -C src/pthread-overload/ $(PINLIB) 

#PATTERN RULES
$(BUILD_DIR)/%.o:  %.c
	@echo "===>  COMPILE  $@"
	$(Q)$(CC) -c  $(CFLAGS) $(ANSI_CFLAGS) $(CPPFLAGS) $< -o $@
	$(Q)$(CC) $(CPPFLAGS) -MT $(@:.d=.o) -MM  $< > $(BUILD_DIR)/$*.d


ifeq ($(findstring $(MAKECMDGOALS),clean),)
-include $(OBJ:.o=.d)
endif

.PHONY: clean distclean install uninstall

clean:
	@echo "===>  CLEAN"
	@rm -rf $(BUILD_DIR)

distclean: clean
	@echo "===>  DIST CLEAN"
	@rm -f likwid-*
	@rm -f $(TARGET_LIB)
	@rm -f $(PINLIB)

install:
	@echo "===> INSTALL applications to $(PREFIX)/bin"
	@mkdir -p $(PREFIX)/bin
	@cp -f likwid-*  $(PREFIX)/bin
	@chmod 755 $(PREFIX)/bin/likwid-*
	@echo "===> INSTALL man pages to $(MANPREFIX)/man1"
	@mkdir -p $(MANPREFIX)/man1
	@sed -e "s/<VERSION>/$(VERSION)/g" -e "s/<DATE>/$(DATE)/g" < $(DOC_DIR)/likwid-topology.1 > $(MANPREFIX)/man1/likwid-topology.1
	@sed -e "s/<VERSION>/$(VERSION)/g" -e "s/<DATE>/$(DATE)/g" < $(DOC_DIR)/likwid-features.1 > $(MANPREFIX)/man1/likwid-features.1
	@sed -e "s/<VERSION>/$(VERSION)/g" -e "s/<DATE>/$(DATE)/g" < $(DOC_DIR)/likwid-perfCtr.1 > $(MANPREFIX)/man1/likwid-perfCtr.1
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



