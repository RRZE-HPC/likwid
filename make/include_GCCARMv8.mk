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

CFLAGS   = -march=armv8-a -mtune=cortex-a57 -mabi=lp64 -O2 -std=c99 -Wno-format -fPIC
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

INCLUDES =
LIBS     = -lm -lrt


