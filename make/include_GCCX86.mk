CC  = gcc
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

CFLAGS   =  -O2 -m32 -Wno-format -std=c99
FCFLAGS  = -J ./  -fsyntax-only
PASFLAGS  = x86
ASFLAGS  = --32
CPPFLAGS =
LFLAGS   = -m32 -pthread

SHARED_CFLAGS = -fpic
SHARED_LFLAGS = -shared

DEFINES  = -D_GNU_SOURCE
DEFINES  += -DPAGE_ALIGNMENT=4096
DEFINES  += -DLIKWID_MONITOR_LOCK
DEFINES  += -DDEBUGLEV=0

INCLUDES =
LIBS     = -lm


