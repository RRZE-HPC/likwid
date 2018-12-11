CC  = /opt/pgi/linux86-64/18.4/bin/pgcc
FC  = /opt/pgi/linux86-64/18.4/bin/pgf90
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

CFLAGS   =  -O2 -fPIC
#FCFLAGS  = -module ./  # ifort
FCFLAGS  = -J ./  -fsyntax-only  #gfortran
PASFLAGS  = x86-64
ASFLAGS  = 
CPPFLAGS =
LFLAGS   =  -pthread

SHARED_CFLAGS = -fPIC 
SHARED_LFLAGS = -shared 

DEFINES  = -DPAGE_ALIGNMENT=4096
DEFINES  += -DLIKWID_MONITOR_LOCK
DEFINES  += -DDEBUGLEV=0

INCLUDES =
LIBS     = -lm -lrt


