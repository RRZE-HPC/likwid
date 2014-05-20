include  ../../config.mk
TARGET   = liblikwidpin.so

CC        = gcc
DEFINES  = -DMAX_NUM_THREADS=$(MAX_NUM_THREADS)
ifneq ($(COLOR),NONE)
DEFINES += -DCOLOR=$(COLOR)
endif

CFLAGS   = -O2 -Wall -std=c99 -shared -fPIC
CPPFLAGS = $(DEFINES) -D_GNU_SOURCE -I.
INCLUDES = -I../includes
LIBS     = -ldl -lpthread

all: $(TARGET)

$(TARGET): pthread-overload.c
	$(CC) $(CPPFLAGS) $(INCLUDES) $(CFLAGS) -o ../../$(TARGET) pthread-overload.c $(LIBS)

