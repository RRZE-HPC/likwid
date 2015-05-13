# Please have a look in INSTALL and the WIKI for details on
# configuration options setup steps.
# supported: GCC, MIC (ICC)
COMPILER = GCC#NO SPACE

# Define the color of the likwid-pin output
# Can be NONE, BLACK, RED, GREEN, YELLOW, BLUE,
# MAGENTA, CYAN or WHITE
COLOR = BLUE#NO SPACE

# Path were to install likwid
PREFIX = /usr/local#NO SPACE
MANPREFIX = $(PREFIX)/man#NO SPACE

# For the daemon based secure msr/pci access configure
# the absolute path to the msr daemon executable.
# $(PREFIX)/bin/likwid-accessD
ACCESSDAEMON = $(PREFIX)/sbin/likwid-accessD#NO SPACE

# Build the accessDaemon. Have a look in the WIKI for details.
BUILDDAEMON = true#NO SPACE

#Build the setFrequencies tool
BUILDFREQ = true#NO SPACE
# Set the default mode for MSR access.
# This can usually be overriden on the commandline.
# Valid values are: direct, accessdaemon
ACCESSMODE = accessdaemon$#NO SPACE

# Change to true to a build shared library instead of a static one
SHARED_LIBRARY = true#NO SPACE

# Build Fortran90 module interface for marker API. Adopt Fortran compiler
# in ./make/include_<COMPILER>.mk if necessary. Default: ifort .
FORTRAN_INTERFACE = false#NO SPACE

# Instrument likwid-bench for use with likwid-perfctr
INSTRUMENT_BENCH = false#NO SPACE

# Use Portable Hardware Locality (hwloc) instead of CPUID
USE_HWLOC = true#NO SPACE

# Usually you do not need to edit below

# Reorder PCI sockets (this is normally not needed)
#REVERSE_HASWELL_PCI_SOCKETS = 1
# Easy test whether you need this: Perform likwid-perfctr on one core with
# Uncore events where you know the outcome, e.g. memory bandwidth.
# likwid-perfctr -C 0 -g MEM ./stream
# If the results are too low, switch sockets.
# Common two socket machines have: 7f ff
# Common four socket machines have: 3f 7f bf ff
#PCI_SOCKETS = ff 7f

MAX_NUM_THREADS = 263
MAX_NUM_NODES = 64
CFG_FILE_PATH = /etc/likwid.cfg

# Versioning Information
VERSION = 4
RELEASE = 0
DATE    = 28.04.2015

LIBLIKWIDPIN = $(abspath $(PREFIX)/lib/liblikwidpin.so)
LIKWIDFILTERPATH = $(abspath $(PREFIX)/share/likwid/filter)

