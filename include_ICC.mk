CC  = icc
FC  = ifort
AS  = as
AR  = ar
PAS = ./perl/AsmGen.pl 
GEN_PAS = ./perl/generatePas.pl 
GEN_GROUPS = ./perl/generateGroups.pl 
GEN_PMHEADER = ./perl/gen_events.pl 

ANSI_CFLAGS  = -strict-ansi
ANSI_CFLAGS += -std=c99

CFLAGS   =  -O1 -Wno-format -vec-report=0
FCFLAGS  = -module ./ 
ASFLAGS  = -gdwarf-2
CPPFLAGS =
LFLAGS   = -pthread

SHARED_CFLAGS = -fpic
SHARED_LFLAGS = -shared

DEFINES  = -D_GNU_SOURCE
DEFINES  += -DMAX_NUM_THREADS=128
DEFINES  += -DPAGE_ALIGNMENT=4096
#enable this option to build likwid-bench with marker API for likwid-perfctr
#DEFINES  += -DPERFMON

INCLUDES =
LIBS     =


