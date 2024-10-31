
#ifneq ($(strip $(MAKECMDGOALS)),docs)
# determine kernel Version
KERNEL_VERSION_MAJOR := $(shell uname -r | awk '{split($$1,a,"."); print a[1]}' | cut -d '-' -f1)
KERNEL_VERSION := $(shell uname -r | awk  '{split($$1,a,"."); print a[2]}' | cut -d '-' -f1)
KERNEL_VERSION_MINOR := $(shell uname -r | sed 's/[^[:digit:].]*//g' | awk '{split($$1,a,"."); print a[3]}' | cut -d '-' -f1)

HAS_MEMPOLICY = $(shell if [ $(KERNEL_VERSION) -lt 7 -a $(KERNEL_VERSION_MAJOR) -lt 3 -a $(KERNEL_VERSION_MINOR) -lt 8 ]; then \
               echo 0;  else echo 1; \
			   fi; )
HAS_PERFEVENT = $(shell if [ $(KERNEL_VERSION) -lt 6 -a $(KERNEL_VERSION_MAJOR) -lt 2 -a $(KERNEL_VERSION_MINOR) -lt 31 ]; then echo 0; else echo 1; fi; )

# determine glibc Version
#GLIBC_VERSION_MAJOR := $(shell echo '\#include <features.h>' | cc -dM -E - | grep -E '\#define __GLIBC__' | cut -d ' ' -f 3)
GLIBC_VERSION_MAJOR := $(shell ldd --version | grep ldd |  awk '{ print $$NF }' | awk -F. '{ print $$1 }')
#GLIBC_VERSION_MINOR := $(shell echo '\#include <features.h>' | cc -dM -E - | grep -E '\#define __GLIBC_MINOR__' | cut -d ' ' -f 3)
GLIBC_VERSION_MINOR := $(shell ldd --version | grep ldd |  awk '{ print $$NF }' | awk -F. '{ print $$2 }')

HAS_SCHEDAFFINITY = $(shell if [ $(GLIBC_VERSION_MINOR) -lt 4 ]; then echo 0; else echo 1; fi; )
ENOUGH_CPUS = $(shell [ $(shell grep processor /proc/cpuinfo | wc -l) -le $(MAX_NUM_THREADS) ] && echo True )

ifneq ($(strip $(ENOUGH_CPUS)), True)
$(info Warning: $(ENOUGH_CPUS) The MAX_NUM_THREADS variable must be larger or equal to the available CPUs. Currently, LIKWID is configured for $(MAX_NUM_THREADS) CPUs, but there are $(INSTALLED_CPUS) CPUs in the systen)
endif

ifneq ($(strip ${DESTDIR}),)
$(info Info: Destdir ${DESTDIR})
PREFIX := ${DESTDIR}
INSTALLED_PREFIX ?= ${DESTDIR}
MANPREFIX ?= $(PREFIX)/man#NO SPACE
BINPREFIX ?= $(PREFIX)/bin#NO SPACE
LIBPREFIX ?= $(PREFIX)/lib#NO SPACE
ACCESSDAEMON = $(PREFIX)/sbin/likwid-accessD#NO SPACE
INSTALLED_PREFIX := $(PREFIX)#NO SPACE
INSTALLED_BINPREFIX ?= $(INSTALLED_PREFIX)/bin#NO SPACE
INSTALLED_LIBPREFIX ?= $(INSTALLED_PREFIX)/lib#NO SPACE
INSTALLED_ACCESSDAEMON = $(INSTALLED_PREFIX)/sbin/likwid-accessD#NO SPACE
RPATHS = -Wl,-rpath=$(INSTALLED_LIBPREFIX)
LIBLIKWIDPIN = $(abspath $(INSTALLED_LIBPREFIX)/liblikwidpin.so.$(VERSION).$(RELEASE))
LIKWIDFILTERPATH = $(abspath $(INSTALLED_PREFIX)/share/likwid/filter)
LIKWIDGROUPPATH = $(abspath $(INSTALLED_PREFIX)/share/likwid/perfgroups)
endif


INST_PREFIX := $(strip $(INSTALLED_PREFIX))
ifneq "$(strip $(PREFIX))" "$(strip $(INST_PREFIX))"
$(info Info: PREFIX and INSTALLED_PREFIX differ, be aware that you have to move stuff after make install from $(strip $(PREFIX)) to $(strip $(INSTALLED_PREFIX)). You can use make move for this.)
endif

ifneq ($(strip $(SHARED_LIBRARY)),true)
$(info Warning: When building as static library, you cannot use the Lua scripts as they require a shared library. You can still link your application to the library.)
endif

FORTRAN_IF_NAME := likwid.mod
ifneq ($(strip $(FORTRAN_INTERFACE)),false)
HAS_FORTRAN_COMPILER := $(shell $(FC) --version 2>/dev/null || echo 'NOFORTRAN' )
ifeq ($(strip $(HAS_FORTRAN_COMPILER)),NOFORTRAN)
FORTRAN_IF=
$(info Warning: You have selected the fortran interface in config.mk, but there seems to be no fortran compiler $(FC) - not compiling it!)
FORTRAN_INSTALL =
FORTRAN_REMOVE =
FORTRAN_REMOVE_MOVED =
else
FORTRAN_IF := $(strip $(FORTRAN_IF_NAME))
FORTRAN_INSTALL = @echo "===> INSTALL fortran interface to $(PREFIX)/include/"; \
                  cp -f $(BASE_DIR)/likwid.mod  $(strip $(PREFIX))/include/$(strip $(FORTRAN_IF_NAME))
FORTRAN_REMOVE = @echo "===> REMOVING fortran interface from $(PREFIX)/include/"; \
                 rm -f $(strip $(PREFIX))/include/$(strip $(FORTRAN_IF_NAME))
FORTRAN_REMOVE_MOVED = @echo "===> REMOVING fortran interface from $(INSTALLED_PREFIX)/include/"; \
                 rm -f $(strip $(INSTALLED_PREFIX))/include/$(strip $(FORTRAN_IF_NAME))
endif
else
FORTRAN_IF =
FORTRAN_INSTALL =
FORTRAN_REMOVE =
FORTRAN_REMOVE_MOVED =
endif
#endif

ifeq ($(strip $(NVIDIA_INTERFACE)), true)
#LIBS+= -lcuda -ldl
INCLUDES += -I$(CUDAINCLUDE) -I$(CUPTIINCLUDE)
#CPPFLAGS += -L$(CUDALIBDIR) -L$(CUPTILIBDIR)
endif

ifeq ($(strip $(ROCM_INTERFACE)), true)
ROCM_SDK_CHECK := $(shell which rocprofv3 2>/dev/null | wc -l)
# HSA includes 'hsa/xxx.h' and rocprofiler 'xxx.h'
DEFINES += -D__HIP_PLATFORM_AMD__
INCLUDES += -I$(HIPINCLUDE) -I$(HSAINCLUDE) -I$(HSAINCLUDE)/hsa -I$(RSMIINCLUDE)
ifeq ($(strip $(ROCM_SDK_CHECK)),1)
DEFINES += -DLIKWID_ROCPROF_SDK
endif
endif
