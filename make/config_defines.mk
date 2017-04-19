DEFINES   += -DVERSION=$(VERSION)         \
		 -DRELEASE=$(RELEASE)                 \
		 -DCFGFILE=$(CFG_FILE_PATH)           \
		 -DTOPOFILE=$(TOPO_FILE_PATH)           \
		 -DINSTALL_PREFIX=$(INSTALLED_PREFIX) \
		 -DMAX_NUM_THREADS=$(MAX_NUM_THREADS) \
		 -DMAX_NUM_NODES=$(MAX_NUM_NODES)     \
		 -DACCESSDAEMON=$(INSTALLED_ACCESSDAEMON) \
		 -DGROUPPATH=$(LIKWIDGROUPPATH) \
		 -DLIKWIDLOCK=$(LIKWIDLOCKPATH) \
		 -DLIKWIDSOCKETBASE=$(LIKWIDSOCKETBASE) \
		 -D_GNU_SOURCE

DYNAMIC_TARGET_LIB := liblikwid.so
STATIC_TARGET_LIB := liblikwid.a
PWD ?= $(shell pwd)
# LUA:
ifdef LUA_INCLUDE_DIR
LUA_INTERNAL := false#NO SPACE
else
LUA_FOLDER := $(PWD)/ext/lua#NO SPACE
LUA_INCLUDE_DIR := $(LUA_FOLDER)/includes#NO SPACE
LUA_LIB_DIR := $(LUA_FOLDER)#NO SPACE
LUA_LIB_NAME := likwid-lua#NO SPACE
LUA_INTERNAL := true#NO SPACE
endif
SHARED_LIBLUA := lib$(LUA_LIB_NAME).so
STATIC_LIBLUA := lib$(LUA_LIB_NAME).a
# HWLOC:
HWLOC_FOLDER := ext/hwloc
STATIC_LIBHWLOC := liblikwid-hwloc.a
SHARED_LIBHWLOC := liblikwid-hwloc.so

BENCH_FOLDER := bench
BENCH_NAME := likwid-bench
BENCH_TARGET := $(BENCH_FOLDER)/$(BENCH_NAME)

ifneq ($(COLOR),NONE)
DEFINES += -DCOLOR=$(COLOR)
endif

ifeq ($(BUILDDAEMON),true)
ifneq ($(COMPILER),MIC)
ifneq ($(COMPILER), GCCARMv7)
ifneq ($(COMPILER), GCCARMv8)
    DAEMON_TARGET = likwid-accessD
else
    $(info Info: Compiling for ARMv8. Disabling build of likwid-accessD.);
    DAEMON_TARGET =
    ACCESSMODE = direct
    BUILDDAEMON = false
endif
else
    $(info Info: Compiling for ARMv7. Disabling build of likwid-accessD.);
    DAEMON_TARGET =
    ACCESSMODE = direct
    BUILDDAEMON = false
endif
else
    $(info Info: Compiling for Xeon Phi. Disabling build of likwid-accessD.);
    DAEMON_TARGET =
    ACCESSMODE = direct
    BUILDDAEMON = false
endif
endif

ifeq ($(BUILDFREQ),true)
ifneq ($(COMPILER),MIC)
    FREQ_TARGET = likwid-setFreq
else
    $(info Info: Compiling for Xeon Phi. Disabling build of likwid-setFreq.);
endif
endif

ifeq ($(HAS_MEMPOLICY),1)
DEFINES += -DHAS_MEMPOLICY
else
$(info Kernel 2.6.$(KERNEL_VERSION) has no mempolicy support!);
endif


ifeq ($(SHARED_LIBRARY),true)
CFLAGS += $(SHARED_CFLAGS)
LIBS += -L. -pthread -lm -ldl
TARGET_LIB := $(DYNAMIC_TARGET_LIB)
TARGET_HWLOC_LIB=$(HWLOC_FOLDER)/$(SHARED_LIBHWLOC)
TARGET_LUA_LIB=$(LUA_LIB_DIR)/$(SHARED_LIBLUA)
else
TARGET_LIB := $(STATIC_TARGET_LIB)
TARGET_HWLOC_LIB=$(HWLOC_FOLDER)/$(STATIC_LIBHWLOC)
TARGET_LUA_LIB=$(LUA_LIB_DIR)/$(STATIC_LIBLUA)
endif

ifeq ($(HAS_SCHEDAFFINITY),1)
DEFINES += -DHAS_SCHEDAFFINITY
PINLIB  = liblikwidpin.so
else
$(info GLIBC version 2.$(GLIBC_VERSION) has no pthread_setaffinity_np support!);
PINLIB  =
endif

FILTER_HWLOC_OBJ = yes
LIBHWLOC =
ifeq ($(USE_HWLOC),true)
DEFINES += -DLIKWID_USE_HWLOC
LIBHWLOC_SHARED = -Lext/hwloc/ -lliblikwid-hwloc
LIBHWLOC_STATIC = ext/hwloc/liblikwid-hwloc.a
EXT_TARGETS += ./ext/hwloc
FILTER_HWLOC_OBJ =
endif

#DEFINES += -DACCESSDAEMON=$(ACCESSDAEMON)

ifeq ($(ACCESSMODE),sysdaemon)
ifneq ($(COMPILER),MIC)
DEFINES += -DACCESSMODE=2
else
$(info Info: Compiling for Xeon Phi. Changing accessmode to direct.);
ACCESSMODE = direct
DEFINES += -DACCESSMODE=0
endif
else
ifeq ($(ACCESSMODE),accessdaemon)
ifneq ($(COMPILER),MIC)
ifneq ($(BUILDDAEMON),true)
$(info Info: Compiling with accessdaemon access mode but without building the access daemon.);
$(info Info: Make sure an accessdaemon is installed and the paths ACCESSDAEMON and INSTALLED_ACCESSDAEMON point to it);
endif
DEFINES += -DACCESSMODE=1
else
$(info Info: Compiling for Xeon Phi. Changing accessmode to direct.);
DEFINES += -DACCESSMODE=0
ACCESSMODE = direct
endif
else
DEFINES += -DACCESSMODE=0
endif
endif

ifeq ($(DEBUG),true)
DEBUG_FLAGS = -g
DEBUG_CFLAGS := $(filter-out -O0, $(CFLAGS))
DEBUG_CFLAGS := $(filter-out -O1, $(DEBUG_CFLAGS))
DEBUG_CFLAGS := $(filter-out -O2, $(DEBUG_CFLAGS))
DEBUG_CFLAGS := $(filter-out -O3, $(DEBUG_CFLAGS))
CFLAGS = -O0 $(DEBUG_CFLAGS)
DEFINES += -DDEBUG_LIKWID
else
DEBUG_FLAGS =
endif

ifeq ($(USE_PERF_EVENT),true)
$(info Info: Compiling for perf_event interface. Features like power consumption or thermal stuff is disabled);
$(info Info: Currently Uncore support is experimental);
DEFINES += -DLIKWID_USE_PERFEVENT
endif

