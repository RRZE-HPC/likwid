CC  = gcc
FC  = ifort
AS  = as
AR  = ar
PAS = ./perl/AsmGen.pl
GEN_PAS = ./perl/generatePas.pl
GEN_GROUPS = ./perl/generateGroups.pl
GEN_PMHEADER = ./perl/gen_events.pl

#ANSI_CFLAGS   = -std=c99
#ANSI_CFLAGS += -pedantic
#ANSI_CFLAGS += -Wextra
#ANSI_CFLAGS += -Wall

CFLAGS   =  -O2  -Wno-format -Wno-nonnull -std=c99
FCFLAGS  = -module ./  # ifort
#FCFLAGS  = -J ./  -fsyntax-only  #gfortran
PASFLAGS  = x86-64
ASFLAGS  =
CPPFLAGS =
LFLAGS   =  -pthread

SHARED_CFLAGS = -fpic
SHARED_LFLAGS = -shared

DEFINES  = -D_GNU_SOURCE
DEFINES  += -DPAGE_ALIGNMENT=4096
DEFINES  += -DLIKWID_MONITOR_LOCK
DEFINES  += -DDEBUGLEV=0

INCLUDES =
LIBS     = -lm


