# supported: GCC, ICC
COMPILER = GCC

# Define the color of the likwid-pin output
# Can be NONE, BLACK, RED, GREEN, YELLOW, BLUE,
# MAGENTA, CYAN or WHITE
COLOR = CYAN

# Path were to install likwid
PREFIX =  /usr/local
MANPREFIX = $(PREFIX)/man

MAX_NUM_THREADS = 128
MAX_NUM_SOCKETS = 4

# Versioning Information
VERSION = 2
RELEASE = 0
DATE    = 20.08.2010

LIBLIKWIDPIN = $(PREFIX)/lib/liblikwidpin.so

