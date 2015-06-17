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
#      Copyright (C) 2015 RRZE, University Erlangen-Nuremberg
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
include  ../../make/include_$(COMPILER).mk

DAEMON_TARGET = likwid-accessD
SETFREQ_TARGET = likwid-setFreq

DEFINES   = -D_GNU_SOURCE -DMAX_NUM_THREADS=$(MAX_NUM_THREADS) -DPCILIST="$(PCI_SOCKETS)"
INCLUDES  = -I../includes
ifeq ($(COMPILER),GCC)
CFLAGS    +=  -Wall -Wextra -std=c99
endif
ifeq ($(COMPILER),ICC)
CFLAGS  += -std=c99
endif
CPPFLAGS :=  $(DEFINES) $(INCLUDES)

all: $(DAEMON_TARGET) $(SETFREQ_TARGET)

$(DAEMON_TARGET): accessDaemon.c
	$(Q)$(CC) $(CFLAGS) $(CPPFLAGS) -o ../../$(DAEMON_TARGET) accessDaemon.c

$(SETFREQ_TARGET): setFreq.c
	$(Q)$(CC) $(CFLAGS) $(CPPFLAGS) -o ../../$(SETFREQ_TARGET) setFreq.c


