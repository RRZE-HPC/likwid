CC  = icc
FC  = ifort
AS  = icc
AR  = ar
PAS = ./perl/AsmGen.pl 
GEN_PAS = ./perl/generatePas.pl 
GEN_GROUPS = ./perl/generateGroups.pl 
GEN_PMHEADER = ./perl/gen_events.pl 

ANSI_CFLAGS   = -std=c99 -fPIC
ANSI_CFLAGS += -pedantic
#ANSI_CFLAGS += -Wextra
#ANSI_CFLAGS += -Wall

CFLAGS   = -mmic -O1 -g -Wno-format -fPIC
FCFLAGS  = -J ./  -fsyntax-only
#FCFLAGS  = -module ./ 
ASFLAGS  =  -mmic -c -x assembler
PASFLAGS  = x86-64
CPPFLAGS =
LFLAGS   =  -pthread -g -mmic

SHARED_CFLAGS = -fpic -mmic -fvisibility=hidden
SHARED_LFLAGS = -shared -mmic -fvisibility=hidden

DEFINES  = -D_GNU_SOURCE
DEFINES  += -DPAGE_ALIGNMENT=4096
DEFINES  += -DDEBUGLEV=0

INCLUDES =
LIBS     = -lm -lrt

# colon seperated list of paths to search for libs at runtime on Phi file system
MIC_LIB_RPATHS = /opt/intel/compilers_and_libraries_2016.1.150/linux/compiler/lib/mic
ifneq (strip $(MIC_LIB_RPATHS),)
RPATHS += -Wl,-rpath=$(MIC_LIB_RPATHS)
endif 
