# =======================================================================================
#  
#      Filename:  Makefile
# 
#      Description:  pthread-overload Makefile
# 
#      Version:   <VERSION>
#      Released:  <DATE>
# 
#      Author:  Jan Treibig (jt), jan.treibig@gmail.com
#      Project:  likwid
#
#      Copyright (C) 2012 Jan Treibig 
#
#      This program is free software: you can redistribute it and/or modify it under
#      the terms of the GNU General Public License as published by the Free Software
#      Foundation, either version 3 of the License, or (at your option) any later
#      version.
#
#      This program is distributed in the hope that it will be useful, but WITHOUT ANY
#      WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
#      PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
#      You should have received a copy of the GNU General Public License along with
#      this program.  If not, see <http://www.gnu.org/licenses/>.
#
# =======================================================================================

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

