
# determine kernel Version
KERNEL_VERSION_MAJOR := $(shell uname -r | awk '{split($$1,a,"."); print a[1]}' | cut -d '-' -f1)
KERNEL_VERSION := $(shell uname -r | awk  '{split($$1,a,"."); print a[2]}' | cut -d '-' -f1)
KERNEL_VERSION_MINOR := $(shell uname -r | awk '{split($$1,a,"."); print a[3]}' | cut -d '-' -f1)

HAS_MEMPOLICY = $(shell if [ $(KERNEL_VERSION) -lt 7 -a $(KERNEL_VERSION_MAJOR) -lt 3 -a $(KERNEL_VERSION_MINOR) -lt 8 ]; then \
               echo 0;  else echo 1; \
			   fi; )

HAS_RDTSCP = $(shell  /bin/bash -c "cat /proc/cpuinfo | grep -c rdtscp")

# determine glibc Version
GLIBC_VERSION := $(shell ldd --version | grep ldd |  awk '{ print $$NF }' | awk -F. '{ print $$2 }')

HAS_SCHEDAFFINITY = $(shell if [ $(GLIBC_VERSION) -lt 4 ]; then \
               echo 0;  else echo 1; \
			   fi; )


ifneq ($(FORTRAN_INTERFACE),false)
HAS_FORTRAN_COMPILER = $(shell $(FC) --version 2>/dev/null || echo 'NOFORTRAN' )
ifeq ($(HAS_FORTRAN_COMPILER),NOFORTRAN)
FORTRAN_INTERFACE=
$(info Warning: You have selected the fortran interface in config.mk, but there seems to be no fortran compiler - not compiling it!)
else
FORTRAN_INTERFACE = likwid.mod
FORTRAN_INSTALL =  @cp -f likwid.mod  $(PREFIX)/include/
endif
else
FORTRAN_INTERFACE =
FORTRAN_INSTALL =
endif

