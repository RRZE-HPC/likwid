DEFINES   += -DVERSION=$(VERSION)         \
		 -DRELEASE=$(RELEASE)                 \
		 -DCFGFILE=$(CFG_FILE_PATH)           \
		 -DMAX_NUM_THREADS=$(MAX_NUM_THREADS) \
		 -DMAX_NUM_NODES=$(MAX_NUM_NODES)     \
		 -DHASH_TABLE_SIZE=$(HASH_TABLE_SIZE) \
		 -DLIBLIKWIDPIN=$(LIBLIKWIDPIN)       \
		 -DLIKWIDFILTERPATH=$(LIKWIDFILTERPATH) \
		 -D_GNU_SOURCE

ifneq ($(COLOR),NONE)
DEFINES += -DCOLOR=$(COLOR)
endif

ifeq ($(BUILDDAEMON),true)
	DAEMON_TARGET = likwid-accessD
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
DEFINES += -DHAS_RDTSCP
endif

ifeq ($(HAS_SCHEDAFFINITY),1)
DEFINES += -DHAS_SCHEDAFFINITY
PINLIB  = liblikwidpin.so
else
$(info GLIBC version 2.$(GLIBC_VERSION) has no pthread_setaffinity_np support!);
PINLIB  =
endif

ifeq ($(USE_HWLOC),true)
DEFINES += -DLIKWID_USE_HWLOC
endif

DEFINES += -DACCESSDAEMON=$(ACCESSDAEMON)

ifeq ($(ACCESSMODE),sysdaemon)
DEFINES += -DACCESSMODE=2
else
ifeq ($(ACCESSMODE),accessdaemon)
DEFINES += -DACCESSMODE=1
else
DEFINES += -DACCESSMODE=0
endif
endif


