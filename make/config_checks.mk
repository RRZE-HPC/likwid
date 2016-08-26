
ifneq ($(MAKECMDGOALS),docs)
# determine kernel Version
KERNEL_VERSION_MAJOR := $(shell uname -r | awk '{split($$1,a,"."); print a[1]}' | cut -d '-' -f1)
KERNEL_VERSION := $(shell uname -r | awk  '{split($$1,a,"."); print a[2]}' | cut -d '-' -f1)
KERNEL_VERSION_MINOR := $(shell uname -r | awk '{split($$1,a,"."); print a[3]}' | cut -d '-' -f1)

HAS_MEMPOLICY = $(shell if [ $(KERNEL_VERSION) -lt 7 -a $(KERNEL_VERSION_MAJOR) -lt 3 -a $(KERNEL_VERSION_MINOR) -lt 8 ]; then \
               echo 0;  else echo 1; \
			   fi; )
HAS_PERFEVENT = $(shell if [ $(KERNEL_VERSION) -lt 6 -a $(KERNEL_VERSION_MAJOR) -lt 2 -a $(KERNEL_VERSION_MINOR) -lt 31 ]; then echo 0; else echo 1; fi; )

# determine glibc Version
GLIBC_VERSION := $(shell ldd --version | grep ldd |  awk '{ print $$NF }' | awk -F. '{ print $$2 }')

HAS_SCHEDAFFINITY = $(shell if [ $(GLIBC_VERSION) -lt 4 ]; then \
               echo 0;  else echo 1; \
			   fi; )
ENOUGH_CPUS = $(shell [ $(shell grep processor /proc/cpuinfo | wc -l) -le $(MAX_NUM_THREADS) ] && echo True )

ifneq ($(ENOUGH_CPUS), True)
$(info Warning: $(ENOUGH_CPUS) The MAX_NUM_THREADS variable must be larger or equal to the available CPUs. Currently, LIKWID is configured for $(MAX_NUM_THREADS) CPUs, but there are $(INSTALLED_CPUS) CPUs in the systen)
endif

INST_PREFIX := $(INSTALLED_PREFIX)
ifneq "$(PREFIX)" "$(INST_PREFIX)"
$(info Info: PREFIX and INSTALLED_PREFIX differ, be aware that you have to move stuff after make install from $(PREFIX) to $(INSTALLED_PREFIX). You can use make move for this.)
endif

FORTRAN_IF_NAME := likwid.mod
ifneq ($(FORTRAN_INTERFACE),false)
HAS_FORTRAN_COMPILER = $(shell $(FC) --version 2>/dev/null || echo 'NOFORTRAN' )
ifeq ($(HAS_FORTRAN_COMPILER),NOFORTRAN)
FORTRAN_IF=
$(info Warning: You have selected the fortran interface in config.mk, but there seems to be no fortran compiler $(FC) - not compiling it!)
FORTRAN_INSTALL =
FORTRAN_REMOVE =
FORTRAN_REMOVE_MOVED =
else
FORTRAN_IF := $(FORTRAN_IF_NAME)
FORTRAN_INSTALL = @echo "===> INSTALL fortran interface to $(PREFIX)/include/"; \
                  cp -f likwid.mod  $(PREFIX)/include/$(FORTRAN_IF_NAME)
FORTRAN_REMOVE = @echo "===> REMOVING fortran interface from $(PREFIX)/include/"; \
                 rm -f $(PREFIX)/include/$(FORTRAN_IF_NAME)
FORTRAN_REMOVE_MOVED = @echo "===> REMOVING fortran interface from $(INSTALLED_PREFIX)/include/"; \
                 rm -f $(INSTALLED_PREFIX)/include/$(FORTRAN_IF_NAME)
endif
else
FORTRAN_IF =
FORTRAN_INSTALL =
FORTRAN_REMOVE =
FORTRAN_REMOVE_MOVED =
endif
endif
