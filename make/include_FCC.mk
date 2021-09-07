CC  = fcc
FC  = frt
AS  = as
AR  = ar
PAS = ./perl/AsmGen.pl
GEN_PAS = ./perl/generatePas.pl
GEN_GROUPS = ./perl/generateGroups.pl
GEN_PMHEADER = ./perl/gen_events.pl

ANSI_CFLAGS   =
#ANSI_CFLAGS += -pedantic
#ANSI_CFLAGS += -Wextra
#ANSI_CFLAGS += -Wall

CFLAGS   =  -O2 -std=c99 -fPIC
FCFLAGS  = -module ./  # ifort
#FCFLAGS  = -J ./  -fsyntax-only  #gfortran
PASFLAGS  = ARMv8
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


