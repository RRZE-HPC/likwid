#################################################################
#################################################################
# Configuration options                                         #
#################################################################
#################################################################

# Please have a look in INSTALL and the WIKI for details on
# configuration options setup steps.
# supported: GCC, CLANG, ICC, MIC (ICC), GCCX86 (for 32bit systems)
COMPILER = GCC#NO SPACE

# Path were to install likwid
PREFIX ?= /usr/local#NO SPACE

# Set the default mode for MSR access.
# This can usually be overriden on the commandline.
# Valid values are: direct, accessdaemon and perf_event
ACCESSMODE = accessdaemon#NO SPACE

# Build Fortran90 module interface for Marker API. Adopt Fortran compiler
# in ./make/include_<COMPILER>.mk if necessary. Default: ifort (even for
# COMPILER=GCC)
FORTRAN_INTERFACE = false#NO SPACE

# Instrument likwid-bench with Marker API calls for use with likwid-perfctr
INSTRUMENT_BENCH = true#NO SPACE

#################################################################
#################################################################
# Advanced configuration options                                #
# Most users do not need to change values below this comment!   #
#################################################################
#################################################################

# Define the color of the likwid-pin output
# Can be NONE, BLACK, RED, GREEN, YELLOW, BLUE,
# MAGENTA, CYAN or WHITE
COLOR = BLUE#NO SPACE

# Some path definitions
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
INSTALLED_PREFIX ?= $(PREFIX)#NO SPACE
INSTALLED_BINPREFIX = $(INSTALLED_PREFIX)/bin#NO SPACE
INSTALLED_LIBPREFIX = $(INSTALLED_PREFIX)/lib#NO SPACE

# For the daemon based secure msr/pci access configure
# the absolute path to the msr daemon executable.
ACCESSDAEMON = $(PREFIX)/sbin/likwid-accessD#NO SPACE
INSTALLED_ACCESSDAEMON = $(INSTALLED_PREFIX)/sbin/likwid-accessD#NO SPACE

# Build the accessDaemon. Have a look in the WIKI for details.
BUILDDAEMON = true#NO SPACE
# Build the setFrequencies daemon to allow users setting the CPU and Uncore
# frequency
BUILDFREQ = true#NO SPACE

# chown installed tools to this user/group
# if you change anything here, make sure that the user/group can access
# the MSR devices and (on Intel) the PCI devices.
INSTALL_CHOWN = -g root -o root#NO SPACE

# uncomment to optionally set external lua@5.2 or lua@5.3:
# default is use internally provide lua
#LUA_INCLUDE_DIR = /usr/include/lua5.2#NO SPACE
#LUA_LIB_DIR = /usr/lib/x86_64-linux-gnu#NO SPACE
#LUA_LIB_NAME = lua5.2#NO SPACE, executable is assumed to have the same name
#LUA_BIN = /usr/bin#NO SPACE

# Change to true to a build shared library instead of a static one
# It is NOT recommended to switch to static libraries as some features don't
# work when compiled statically
SHARED_LIBRARY = true#NO SPACE

# Build LIKWID with debug flags
DEBUG = false#NO SPACE

# Basic configuration for some internal arrays.
# Maximal number of hardware threads
MAX_NUM_THREADS = 263
# Maximal number of sockets
MAX_NUM_NODES = 64

# Paths to some configuration files that can be used to overwrite some
# array lengths defined at compilation
CFG_FILE_PATH = /etc/likwid.cfg
# With the likwid-genTopoCfg it is possible to store the topology of a system
# in a file to avoid re-reading all topology informations again
TOPO_FILE_PATH = /etc/likwid_topo.cfg

# Versioning Information
# The libraries are named liblikwid.<VERSION>.<RELEASE>
VERSION = 4
RELEASE = 3
MINOR = 0
# Date when the release is published
DATE    = 03.08.2017

# In come cases it is important to set the rpaths for the LIKWID library. One
# example is the use of sudo because it resets environment variables like
# LD_LIBRARY_PATH
RPATHS = -Wl,-rpath=$(INSTALLED_LIBPREFIX)

# LIKWID uses a lock to avoid simultaneous usage by multiple users. The user
# owning this file has access to the LIKWID library and can use LIKWID
# simultaneously.
LIKWIDLOCKPATH = /var/run/likwid.lock

# The access daemon creates sockets under this path to communicate with the
# LIKWID library.
LIKWIDSOCKETBASE = /tmp/likwid  # -%d will be added automatically to the socket name

# The pinning library is put in LD_PRELOAD when using LIKIWD for thread/process
# pinning. The library overloads the pthread_create function to pin threads
# directly after their creation
LIBLIKWIDPIN = $(abspath $(INSTALLED_LIBPREFIX)/liblikwidpin.so.$(VERSION).$(RELEASE))

# Some tools (likwid-perfctr and likwid-topology) provide the export of their
# output to a file. LIKWID tries to format the file based on the file suffix.
# The folder contains scripts named as the file suffix, like xml, which read
# the CSV output of the tools and perform the conversation to the desired file
# format
LIKWIDFILTERPATH = $(abspath $(INSTALLED_PREFIX)/share/likwid/filter)

# LIKWID uses txt-files as input for the performance groups. The configured
# folder contains folders for each architecture with the architecture-specific
# performance group files. Despite this folder, LIKWID also checks
# $HOME/.likwid/groups
LIKWIDGROUPPATH = $(abspath $(INSTALLED_PREFIX)/share/likwid/perfgroups)
