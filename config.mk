# Please have a look in INSTALL and the WIKI for details on
# configuration options setup steps.
# supported: GCC, CLANG, ICC, MIC (ICC), GCCX86 (for 32bit systems)
COMPILER = GCC#NO SPACE

# Define the color of the likwid-pin output
# Can be NONE, BLACK, RED, GREEN, YELLOW, BLUE,
# MAGENTA, CYAN or WHITE
COLOR = BLUE#NO SPACE

# Path were to install likwid
PREFIX = /usr/local#NO SPACE

# uncomment to optionally set external lua@5.3:
# default is use internally provide lua
#LUA_INCLUDE_DIR = /usr/include/lua5.2#NO SPACE
#LUA_LIB_DIR = /usr/lib/x86_64-linux-gnu#NO SPACE
#LUA_LIB_NAME = lua5.2#NO SPACE, executable is assumed to have the same name
#LUA_BIN = /usr/bin#NO SPACE

#################################################################
# Common users do not need to change values below this comment! #
#################################################################

MANPREFIX = $(PREFIX)/man#NO SPACE
BINPREFIX = $(PREFIX)/bin#NO SPACE
LIBPREFIX = $(PREFIX)/lib#NO SPACE

# These paths are hardcoded into executables and libraries. Usually
# they'll be the same as above, but package maintainers may want to
# distinguish between the image directories and the final install
# target.
# Keep in mind that the access and setFreq daemon need enough
# privileges that may be deleted when copying the files to
# the INTSTALLED_PREFIX
INSTALLED_PREFIX = $(PREFIX)#NO SPACE
INSTALLED_BINPREFIX = $(INSTALLED_PREFIX)/bin#NO SPACE
INSTALLED_LIBPREFIX = $(INSTALLED_PREFIX)/lib#NO SPACE

# chown installed tools to this user/group
# if you change anything here, make sure that the user/group can access
# the MSR devices and (on Intel) the PCI devices.
INSTALL_CHOWN = -g root -o root

# For the daemon based secure msr/pci access configure
# the absolute path to the msr daemon executable.
# $(INSTALLED_PREFIX)/bin/likwid-accessD
ACCESSDAEMON = $(PREFIX)/sbin/likwid-accessD#NO SPACE
INSTALLED_ACCESSDAEMON = $(INSTALLED_PREFIX)/sbin/likwid-accessD#NO SPACE

# Build the accessDaemon. Have a look in the WIKI for details.
BUILDDAEMON = true#NO SPACE
#Build the setFrequencies tool
BUILDFREQ = true#NO SPACE

# Set the default mode for MSR access.
# This can usually be overriden on the commandline.
# Valid values are: direct, accessdaemon
ACCESSMODE = accessdaemon#NO SPACE

# Change to true to a build shared library instead of a static one
SHARED_LIBRARY = true#NO SPACE

# Build Fortran90 module interface for marker API. Adopt Fortran compiler
# in ./make/include_<COMPILER>.mk if necessary. Default: ifort .
FORTRAN_INTERFACE = false#NO SPACE

# Instrument likwid-bench for use with likwid-perfctr
INSTRUMENT_BENCH = false#NO SPACE

# Use recommended Portable Hardware Locality (hwloc) instead of CPUID
USE_HWLOC = true#NO SPACE

# Use Linux perf_event interface for measurements. Does not support thermal or
# energy (RAPL) readings.
USE_PERF_EVENT = false#NO SPACE

# Build LIKWID with debug flags
DEBUG = false#NO SPACE

# Basic configuration (compiled into library, can be changed by creating
# a proper config file at CFG_FILE_PATH)
MAX_NUM_THREADS = 263
MAX_NUM_NODES = 64
CFG_FILE_PATH = /etc/likwid.cfg
TOPO_FILE_PATH = /etc/likwid_topo.cfg

# Versioning Information
VERSION = 4
RELEASE = 1
DATE    = 19.05.2016

RPATHS = -Wl,-rpath=$(INSTALLED_LIBPREFIX)
LIKWIDLOCKPATH = /var/run/likwid.lock
LIKWIDSOCKETBASE = /tmp/likwid  # -%d will be added automatically to the socket name
LIBLIKWIDPIN = $(abspath $(INSTALLED_PREFIX)/lib/liblikwidpin.so.$(VERSION).$(RELEASE))
LIKWIDFILTERPATH = $(abspath $(INSTALLED_PREFIX)/share/likwid/filter)
LIKWIDGROUPPATH = $(abspath $(INSTALLED_PREFIX)/share/likwid/perfgroups)
