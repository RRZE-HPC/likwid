CC  = gcc
FC  = ifort
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

CFLAGS   =  -O2 -std=c99 -Wno-format 
FCFLAGS  = -module ./  # ifort
#FCFLAGS  = -J ./  -fsyntax-only  #gfortran
PASFLAGS  = x86-64
ASFLAGS  = 
CPPFLAGS =
LFLAGS   =  -pthread 

SHARED_CFLAGS = -fpic
SHARED_LFLAGS = -shared

DEFINES  = -DPAGE_ALIGNMENT=4096
DEFINES  += -DLIKWID_MONITOR_LOCK
DEFINES  += -DDEBUGLEV=0

INCLUDES =
LIBS     = -lm


