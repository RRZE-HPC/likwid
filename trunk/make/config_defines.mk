DEFINES   += -DVERSION=$(VERSION)         \
		 -DRELEASE=$(RELEASE)                 \
		 -DCFGFILE=$(CFG_FILE_PATH)           \
		 -DINSTALL_PREFIX=$(PREFIX)           \
		 -DMAX_NUM_THREADS=$(MAX_NUM_THREADS) \
		 -DMAX_NUM_NODES=$(MAX_NUM_NODES)     \
		 -DLIBLIKWIDPIN=$(LIBLIKWIDPIN)       \
		 -DLIKWIDFILTERPATH=$(LIKWIDFILTERPATH) \
		 -DLIKWIDPATH=$(PREFIX) \
		 -D_GNU_SOURCE

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

ifeq ($(INSTRUMENT_BENCH),true)
DEFINES += -DPERFMON
endif

ifeq ($(HAS_MEMPOLICY),1)
DEFINES += -DHAS_MEMPOLICY
else
$(info Kernel 2.6.$(KERNEL_VERSION) has no mempolicy support!);
endif

ifeq ($(HAS_RDTSCP),0)
$(info Buildung without RDTSCP timing support!);
else
ifneq ($(COMPILER),MIC)
DEFINES += -DHAS_RDTSCP
else
$(info Info: Compiling for Xeon Phi. Disabling RDTSCP support.);
endif
endif

ifeq ($(SHARED_LIBRARY),true)
CFLAGS += $(SHARED_CFLAGS)
LIBS += -L. -pthread -lm
TARGET_LIB := $(DYNAMIC_TARGET_LIB)
else
TARGET_LIB := $(STATIC_TARGET_LIB)
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
ifneq ($(COMPILER),MIC)
DEFINES += -DLIKWID_USE_HWLOC
LIBHWLOC = ext/hwloc/libhwloc.a
LIBS += -Lext/hwloc
EXT_TARGETS += ./ext/hwloc
FILTER_HWLOC_OBJ =
else
$(info Hwloc not usable on Xeon Phi, disabling);
endif
endif

DEFINES += -DACCESSDAEMON=$(ACCESSDAEMON)

ifeq ($(ACCESSMODE),sysdaemon)
ifneq ($(COMPILER),MIC)
DEFINES += -DACCESSMODE=2
else
$(info Info: Compiling for Xeon Phi. Set accessmode to direct.);
DEFINES += -DACCESSMODE=0
endif
else
ifeq ($(ACCESSMODE),accessdaemon)
ifneq ($(COMPILER),MIC)
DEFINES += -DACCESSMODE=1
else
$(info Info: Compiling for Xeon Phi. Set accessmode to direct.);
DEFINES += -DACCESSMODE=0
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
