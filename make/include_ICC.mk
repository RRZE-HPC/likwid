CC  = icc
FC  = ifort
AS  = as
AR  = ar
PAS = ./perl/AsmGen.pl 
GEN_PAS = ./perl/generatePas.pl 
GEN_GROUPS = ./perl/generateGroups.pl 
GEN_PMHEADER = ./perl/gen_events.pl 

ANSI_CFLAGS  = -std=c99 #-strict-ansi

CFLAGS   =  -O1 -Wno-format -vec-report=0 -fPIC -pthread
FCFLAGS  = -module ./ 
ASFLAGS  = -gdwarf-2
PASFLAGS  = x86-64
CPPFLAGS =
LFLAGS   = -pthread

SHARED_CFLAGS = -fPIC -pthread
SHARED_LFLAGS = -shared -pthread

DEFINES  = -D_GNU_SOURCE
DEFINES  += -DPAGE_ALIGNMENT=4096

INCLUDES =
LIBS     =


