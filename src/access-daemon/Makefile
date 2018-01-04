# =======================================================================================
#
#      Filename:  Makefile
#
#      Description:  accessDaemon Makefile
#
#      Version:   4.3.1
#      Released:  04.01.2018
#
#      Author:  Jan Treibig (jt), jan.treibig@gmail.com
#      Project:  likwid
#
#      Copyright (C) 2018 RRZE, University Erlangen-Nuremberg
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
Q         ?= @

DEFINES   += -D_GNU_SOURCE -DMAX_NUM_THREADS=$(MAX_NUM_THREADS) -DMAX_NUM_NODES=$(MAX_NUM_NODES) -DLIKWIDLOCK=$(LIKWIDLOCKPATH) -DLIKWIDSOCKETBASE=$(LIKWIDSOCKETBASE)
INCLUDES  = -I../includes
CFLAGS    += -std=c99 -fPIC -pie -fPIE -fstack-protector
ifeq ($(COMPILER),GCCX86)
CFLAGS    +=  -m32
endif
CPPFLAGS :=  $(DEFINES) $(INCLUDES)

all: $(DAEMON_TARGET) $(SETFREQ_TARGET)

$(DAEMON_TARGET): accessDaemon.c
	$(Q)$(CC) $(CFLAGS) $(CPPFLAGS) -o ../../$(DAEMON_TARGET) accessDaemon.c

$(SETFREQ_TARGET): setFreq.c setFreq_cpufreq.c setFreq_pstate.c
	$(Q)$(CC) $(CFLAGS) $(CPPFLAGS) -I. -o ../../$(SETFREQ_TARGET) setFreq.c setFreq_cpufreq.c setFreq_pstate.c

