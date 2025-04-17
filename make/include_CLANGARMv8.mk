CC  = clang
FC  = flang-new
AS  = llvm-as
AR  = llvm-ar
PAS = ./perl/AsmGen.pl
GEN_PAS = ./perl/generatePas.pl
GEN_GROUPS = ./perl/generateGroups.pl
GEN_PMHEADER = ./perl/gen_events.pl

ANSI_CFLAGS   =

CFLAGS   = -march=armv8-a -mtune=cortex-a57 -O2 -std=c99 -Wno-format -fPIC
FCFLAGS  = -module ./
PASFLAGS = ARMv8
ASFLAGS  =
CPPFLAGS =
LFLAGS   =  -pthread

SHARED_CFLAGS = -fPIC -fvisibility=hidden
SHARED_LFLAGS = -shared -fvisibility=hidden

DEFINES  = -DPAGE_ALIGNMENT=4096
DEFINES  += -DLIKWID_MONITOR_LOCK
DEFINES  += -DDEBUGLEV=0
DEFINES  += -D__ARM_ARCH_8A

INCLUDES =
LIBS     = -lm -lrt
