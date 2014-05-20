# Please have a look in INSTALL and the WIKI for details on
#t configuration options setup steps.
# supported: GCC, MIC (ICC)
COMPILER = GCC

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
ACCESSDAEMON = $(PREFIX)/bin/likwid-accessD#NO SPACE
# Path to the msrd-socket in system daemon mode.
SYSDAEMONSOCKETPATH = /var/run/likwid-msrd.sock#NO SPACE

# Build the accessDaemon. Have a look in the WIKI for details.
BUILDDAEMON = false#NO SPACE

# Set the default mode for MSR access.
# This can usually be overriden on the commandline.
# Valid values are: direct, accessdaemon, sysdaemon
ACCESSMODE = direct#NO SPACE

# Set to true to enable SandyBridge Uncore support
ENABLE_SNB_UNCORE = false#NO SPACE

# Change to YES to a build shared library instead of a static one
SHARED_LIBRARY = false#NO SPACE

# Instrument likwid-bench for use with likwid-perfctr
INSTRUMENT_BENCH = false#NO SPACE

# Optional Fortran90 interface module
# Uncomment line below to enable
# Please refer to the WIKI documentation for details on usage
# Notice: For gfortran at least version 4.2 is required!
#FORTRAN_INTERFACE = likwid.mod

# Usually you do not need to edit below
MAX_NUM_THREADS = 256
MAX_NUM_NODES = 4
HASH_TABLE_SIZE = 20

# Versioning Information
VERSION = 3
RELEASE = 0
DATE    = 29.11.2012

LIBLIKWIDPIN = $(abspath $(PREFIX)/lib/liblikwidpin.so)
LIKWIDFILTERPATH = $(abspath $(PREFIX)/share/likwid)

