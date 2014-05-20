CC  = icc
AS  = as
AR  = ar
PAS = ./perl/AsmGen.pl 
GEN_PAS = ./perl/generatePas.pl 
GEN_GROUPS = ./perl/generateGroups.pl 

ANSI_CFLAGS  = -strict-ansi
ANSI_CFLAGS += -std=c99

CFLAGS   =  -O0 -g -Wno-format
ASFLAGS  = -g -gstabs
CPPFLAGS =
LFLAGS   = -pthread
DEFINES  = -D_GNU_SOURCE
DEFINES  += -DMAX_NUM_THREADS=128
DEFINES  += -DPAGE_ALIGNMENT=4096
#enable this option to build likwid-bench with marker API for likwid-perfCtr
#DEFINES  += -DPERFMON

INCLUDES =
LIBS     = -lm


