# =======================================================================================
#  
#      Filename:  Makefile
# 
#      Description:  accessDaemon Makefile
# 
#      Version:   <VERSION>
#      Released:  <DATE>
# 
#      Author:  Jan Treibig (jt), jan.treibig@gmail.com
#      Project:  likwid
#
#      Copyright (C) 2013 Jan Treibig 
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

TARGET   = likwid-accessD

CC        = gcc
DEFINES   = -D_GNU_SOURCE -DMAX_NUM_THREADS=264
CFLAGS   = -O2 -Wall -ansi -pedantic -Wextra -std=c99
CPPFLAGS = $(DEFINES)
INCLUDES = -I../includes
LIBS     =

all: $(TARGET)

$(TARGET): accessDaemon.c
	$(CC) $(CPPFLAGS) $(INCLUDES) $(CFLAGS) -o ../../$(TARGET) accessDaemon.c

