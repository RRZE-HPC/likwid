DEFINES   += -DVERSION=$(VERSION)         \
		 -DRELEASE=$(RELEASE)                 \
		 -DCFGFILE=$(CFG_FILE_PATH)           \
		 -DINSTALL_PREFIX=$(PREFIX)           \
		 -DMAX_NUM_THREADS=$(MAX_NUM_THREADS) \
		 -DMAX_NUM_NODES=$(MAX_NUM_NODES)     \
		 -DACCESSDAEMON=$(INSTALLED_ACCESSDAEMON) \
		 -D_GNU_SOURCE

DYNAMIC_TARGET_LIB := liblikwid.so
STATIC_TARGET_LIB := liblikwid.a

LUA_FOLDER := ext/lua
SHARED_LIBLUA := liblikwid-lua.so
STATIC_LIBLUA := liblikwid-lua.a
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
    DAEMON_TARGET = likwid-accessD
else
    $(info Info: Compiling for Xeon Phi. Disabling build of likwid-accessD.);
    DAEMON_TARGET =
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
TARGET_LUA_LIB=$(LUA_FOLDER)/$(SHARED_LIBLUA)
else
TARGET_LIB := $(STATIC_TARGET_LIB)
TARGET_HWLOC_LIB=$(HWLOC_FOLDER)/$(STATIC_LIBHWLOC)
TARGET_LUA_LIB=$(LUA_FOLDER)/$(STATIC_LIBLUA)
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
$(info Info: Compiling for Xeon Phi. Set accessmode to direct.);
ACCESSMODE = direct
DEFINES += -DACCESSMODE=0
endif
else
ifeq ($(ACCESSMODE),accessdaemon)
ifneq ($(COMPILER),MIC)
DEFINES += -DACCESSMODE=1
else
$(info Info: Compiling for Xeon Phi. Set accessmode to direct.);
DEFINES += -DACCESSMODE=0
ACCESSMODE = direct
endif
else
DEFINES += -DACCESSMODE=0
endif
endif

ifeq ($(DEBUG),true)
DEBUG_FLAGS = -g
DEFINES += -DDEBUG_LIKWID
else
DEBUG_FLAGS =
endif
