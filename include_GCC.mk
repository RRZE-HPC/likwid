CC  = gcc

ANSI_CFLAGS  = -ansi
ANSI_CFLAGS += -std=c99
ANSI_CFLAGS += -pedantic
ANSI_CFLAGS += -Wextra

CFLAGS   = -O1 -Wno-format  -Wall
CPPFLAGS =
LFLAGS   =  -pthread -lm
DEFINES  = -D_GNU_SOURCE

INCLUDES =
LIBS     = 


