DEFINES   += -DVERSION=$(VERSION)         \
             -DRELEASE=$(RELEASE)                 \
             -DMINORVERSION=$(MINOR)                 \
             -DCFGFILE=$(CFG_FILE_PATH)           \
             -DTOPOFILE=$(TOPO_FILE_PATH)           \
             -DINSTALL_PREFIX=$(INSTALLED_PREFIX) \
             -DMAX_NUM_THREADS=$(MAX_NUM_THREADS) \
             -DMAX_NUM_NODES=$(MAX_NUM_NODES)     \
             -DACCESSDAEMON=$(INSTALLED_ACCESSDAEMON) \
             -DFREQDAEMON=$(INSTALLED_FREQDAEMON) \
             -DGROUPPATH=$(LIKWIDGROUPPATH) \
             -DLIKWIDLOCK=$(LIKWIDLOCKPATH) \
             -DLIKWIDSOCKETBASE=$(LIKWIDSOCKETBASE) \
             -DGITCOMMIT=$(GITCOMMIT) \
             -D_GNU_SOURCE

COMPILER := $(strip $(COMPILER))


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
HWLOC_FOLDER := $(PWD)/ext/hwloc
STATIC_LIBHWLOC := liblikwid-hwloc.a
SHARED_LIBHWLOC := liblikwid-hwloc.so

GOTCHA_FOLDER := $(PWD)/ext/GOTCHA
STATIC_LIBGOTCHA := liblikwid-gotcha.a
SHARED_LIBGOTCHA := liblikwid-gotcha.so

BENCH_FOLDER := bench
BENCH_NAME := likwid-bench
BENCH_TARGET := $(BENCH_FOLDER)/$(BENCH_NAME)

ifneq ($(strip $(COLOR)),NONE)
DEFINES += -DCOLOR=$(COLOR)
endif

ifeq ($(strip $(COMPILER)),MIC)
    ifeq ($(strip $(ACCESSMODE)),sysdaemon)
        $(info Info: Compiling for Xeon Phi. Changing accessmode to direct.)
        ACCESSMODE := direct
        BUILDDAEMON := false
        BUILDFREQ := false
    endif
    ifeq ($(strip $(ACCESSMODE)),accessdaemon)
        $(info Info: Compiling for Xeon Phi. Changing accessmode to direct.)
        ACCESSMODE := direct
        BUILDDAEMON := false
        BUILDFREQ := false
    endif
    ifeq ($(strip $(ACCESSMODE)),perf_event)
        $(info Info: Compiling for Xeon Phi. Changing accessmode to direct.)
        ACCESSMODE := direct
        BUILDDAEMON := false
        BUILDFREQ := false
    endif
    ifeq ($(strip $(ACCESSMODE)),direct)
        BUILDDAEMON := false
        BUILDFREQ := false
    endif
endif

ifeq ($(strip $(COMPILER)),GCCARMv8)
    ifeq ($(strip $(ACCESSMODE)),sysdaemon)
        $(info Info: Compiling for ARMv8 architecture. Changing accessmode to perf_event.)
        ACCESSMODE := perf_event
        DEFINES += -DLIKWID_USE_PERFEVENT
        BUILDDAEMON := false
        BUILDFREQ := false
    endif
    ifeq ($(strip $(ACCESSMODE)),accessdaemon)
        $(info Info: Compiling for ARMv8 architecture. Changing accessmode to perf_event.)
        ACCESSMODE := perf_event
        DEFINES += -DLIKWID_USE_PERFEVENT
        BUILDDAEMON := false
        BUILDFREQ := false
    endif
    ifeq ($(strip $(ACCESSMODE)),direct)
        $(info Info: Compiling for ARMv8 architecture. Changing accessmode to perf_event.)
        ACCESSMODE := perf_event
        DEFINES += -DLIKWID_USE_PERFEVENT
        BUILDDAEMON := false
        BUILDFREQ := false
    endif
    ifeq ($(strip $(ACCESSMODE)),perf_event)
        DEFINES += -DLIKWID_USE_PERFEVENT
        BUILDDAEMON := false
        BUILDFREQ := false
    endif
endif

ifeq ($(strip $(COMPILER)),GCCARMv7)
    ifeq ($(strip $(ACCESSMODE)),sysdaemon)
        $(info Info: Compiling for ARMv7 architecture. Changing accessmode to perf_event.)
        ACCESSMODE := perf_event
        DEFINES += -DLIKWID_USE_PERFEVENT
        BUILDDAEMON = false
        BUILDFREQ = false
    endif
    ifeq ($(strip $(ACCESSMODE)),accessdaemon)
        $(info Info: Compiling for ARMv7 architecture. Changing accessmode to perf_event.)
        ACCESSMODE := perf_event
        DEFINES += -DLIKWID_USE_PERFEVENT
        BUILDDAEMON = false
        BUILDFREQ = false
    endif
    ifeq ($(strip $(ACCESSMODE)),direct)
        $(info Info: Compiling for ARMv7 architecture. Changing accessmode to perf_event.)
        ACCESSMODE := perf_event
        DEFINES += -DLIKWID_USE_PERFEVENT
        BUILDFREQ = false
        BUILDDAEMON = false
    endif
    ifeq ($(strip $(ACCESSMODE)),perf_event)
        DEFINES += -DLIKWID_USE_PERFEVENT
        BUILDDAEMON := false
        BUILDFREQ := false
    endif
endif

ifeq ($(strip $(COMPILER)),GCCPOWER)
    ifeq ($(strip $(ACCESSMODE)),sysdaemon)
        $(info Info: Compiling for POWER architecture. Changing accessmode to perf_event.)
        ACCESSMODE := perf_event
        DEFINES += -DLIKWID_USE_PERFEVENT
        BUILDDAEMON = false
        BUILDFREQ = false
    endif
    ifeq ($(strip $(ACCESSMODE)),accessdaemon)
        $(info Info: Compiling for POWER architecture. Changing accessmode to perf_event.)
        ACCESSMODE := perf_event
        DEFINES += -DLIKWID_USE_PERFEVENT
        BUILDDAEMON = false
        BUILDFREQ = false
    endif
    ifeq ($(strip $(ACCESSMODE)),direct)
        $(info Info: Compiling for POWER architecture. Changing accessmode to perf_event.)
        ACCESSMODE := perf_event
        DEFINES += -DLIKWID_USE_PERFEVENT
        BUILDFREQ = false
        BUILDDAEMON = false
    endif
    ifeq ($(strip $(ACCESSMODE)),perf_event)
        DEFINES += -DLIKWID_USE_PERFEVENT
        BUILDDAEMON := false
        BUILDFREQ := false
    endif
endif

ifeq ($(strip $(BUILDDAEMON)),true)
ifneq ($(strip $(COMPILER)),MIC)
    DAEMON_TARGET = likwid-accessD
else
    $(info Info: Compiling for Xeon Phi. Disabling build of likwid-accessD.);
    DAEMON_TARGET =
endif
else
    DAEMON_TARGET =
endif

ifeq ($(strip $(BUILDFREQ)),true)
    ifneq ($(strip $(COMPILER)),MIC)
        FREQ_TARGET = likwid-setFreq
    else
        $(info Info: Compiling for Xeon Phi. Disabling build of likwid-setFreq.);
        FREQ_TARGET =
    endif
else
    FREQ_TARGET =
endif
ifeq ($(strip $(BUILDAPPDAEMON)),true)
	APPDAEMON_TARGET = likwid-appDaemon.so
else
	APPDAEMON_TARGET =
endif

ifeq ($(strip $(HAS_MEMPOLICY)),1)
DEFINES += -DHAS_MEMPOLICY
else
$(info Kernel 2.6.$(KERNEL_VERSION) has no mempolicy support!);
endif


ifeq ($(strip $(SHARED_LIBRARY)),true)
CFLAGS += $(SHARED_CFLAGS)
LIBS += -L. -pthread -lm -ldl
TARGET_LIB := $(DYNAMIC_TARGET_LIB)
TARGET_HWLOC_LIB=$(HWLOC_FOLDER)/$(SHARED_LIBHWLOC)
TARGET_LUA_LIB=$(LUA_LIB_DIR)/$(SHARED_LIBLUA)
TARGET_GOTCHA_LIB=$(GOTCHA_LIB_DIR)/$(SHARED_LIBGOTCHA)
else
TARGET_LIB := $(STATIC_TARGET_LIB)
TARGET_HWLOC_LIB=$(HWLOC_FOLDER)/$(STATIC_LIBHWLOC)
TARGET_LUA_LIB=$(LUA_LIB_DIR)/$(STATIC_LIBLUA)
TARGET_GOTCHA_LIB=$(GOTCHA_LIB_DIR)/$(STATIC_LIBGOTCHA)
endif

ifeq ($(strip $(HAS_SCHEDAFFINITY)),1)
DEFINES += -DHAS_SCHEDAFFINITY
PINLIB  = liblikwidpin.so
else
$(info GLIBC version 2.$(GLIBC_VERSION) has no pthread_setaffinity_np support!);
PINLIB  =
endif

FILTER_HWLOC_OBJ = yes
LIBHWLOC =
DEFINES += -DLIKWID_USE_HWLOC
LIBHWLOC_SHARED = -Lext/hwloc/ -lliblikwid-hwloc
LIBHWLOC_STATIC = ext/hwloc/liblikwid-hwloc.a
EXT_TARGETS += ./ext/hwloc
FILTER_HWLOC_OBJ =

#DEFINES += -DACCESSDAEMON=$(ACCESSDAEMON)

ifeq ($(strip $(ACCESSMODE)),sysdaemon)
    DEFINES += -DACCESSMODE=2
else
    ifeq ($(strip $(ACCESSMODE)),accessdaemon)
        DEFINES += -DACCESSMODE=1
    else
        ifeq ($(strip $(ACCESSMODE)),direct)
            DEFINES += -DACCESSMODE=0
        else
            ifeq ($(strip $(ACCESSMODE)),perf_event)
                DEFINES += -DLIKWID_USE_PERFEVENT
                DEFINES += -DACCESSMODE=-1
                BUILDDAEMON = false
                $(info Info: Compiling for perf_event interface. Measurements of thermal information is disabled);
            else
                $(info Error: Unknown access mode $(ACCESSMODE))
            endif
        endif
    endif
endif

ifeq ($(strip $(ACCESSMODE)),accessdaemon)
    ifneq ($(strip $(BUILDDAEMON)),true)
        $(info Info: Compiling with accessdaemon access mode but without building the access daemon.);
        $(info Info: Make sure an accessdaemon is installed and the paths ACCESSDAEMON and INSTALLED_ACCESSDAEMON point to it);
    endif
endif


ifeq ($(strip $(DEBUG)),true)
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
