CC  = icc
FC  = gfortran
AS  = as
AR  = ar
PAS = ./perl/AsmGen.pl 
GEN_PAS = ./perl/generatePas.pl 
GEN_GROUPS = ./perl/generateGroups.pl 
GEN_PMHEADER = ./perl/gen_events.pl 

ANSI_CFLAGS   = -std=c99
ANSI_CFLAGS += -pedantic
#ANSI_CFLAGS += -Wextra
#ANSI_CFLAGS += -Wall

CFLAGS   = -mmic -O0 -g -Wno-format
FCFLAGS  = -J ./  -fsyntax-only
#FCFLAGS  = -module ./ 
ASFLAGS  = 
CPPFLAGS =
LFLAGS   =  -pthread -g -mmic

SHARED_CFLAGS = -fpic -mmic
SHARED_LFLAGS = -shared -mmic

DEFINES  = -D_GNU_SOURCE
DEFINES  += -DPAGE_ALIGNMENT=4096
DEFINES  += -DDEBUGLEV=0

INCLUDES =
LIBS     = -lm


