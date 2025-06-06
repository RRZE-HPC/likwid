# =======================================================================================
#
#      Filename:  Makefile
#
#      Description:  accessDaemon Makefile
#
#      Version:   <VERSION>
#      Released:  <DATE>
#
#      Author:   Jan Treibig (jt), jan.treibig@gmail.com
#                Thomas Gruber (tr), thomas.roehl@googlemail.com
#      Project:  likwid
#
#      Copyright (C) 2016 RRZE, University Erlangen-Nuremberg
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

DAEMON_TARGET = ../../likwid-accessD
SETFREQ_TARGET = ../../likwid-setFreq
APPDAEMON_TARGET = ../../likwid-appDaemon.so
SYSFSDAEMON_TARGET = ../../likwid-sysD
GOTCHA_FOLDER = ../../ext/GOTCHA
Q         ?= @

DEFINES   += -D_GNU_SOURCE -DMAX_NUM_THREADS=$(MAX_NUM_THREADS) -DMAX_NUM_NODES=$(MAX_NUM_NODES) -DLIKWIDLOCK=$(LIKWIDLOCKPATH) -DLIKWIDSOCKETBASE=$(LIKWIDSOCKETBASE)
ifeq ($(DEBUG),true)
DEFINES += -DDEBUG_LIKWID
endif
ifeq ($(NVIDIA_INTERFACE), true)
DEFINES += -DLIKWID_NVMON
endif
ifeq ($(ROCM_INTERFACE), true)
DEFINES += -DLIKWID_ROCMON
endif
INCLUDES  = -I../includes
CFLAGS    += -std=c99 -fPIC -pie -fPIE -fstack-protector
ifeq ($(COMPILER),GCCX86)
CFLAGS    +=  -m32
endif
CPPFLAGS :=  $(DEFINES) $(INCLUDES) -L../..

ifeq ($(COMPILER),GCCARMv8)
all:
else ifeq ($(COMPILER),CLANGARMv8)
all:
else
all: $(DAEMON_TARGET) $(SETFREQ_TARGET) $(APPDAEMON_TARGET)
endif

$(DAEMON_TARGET): accessDaemon.c ../intel_perfmon_uncore_discovery.c ../lock.c ../pci_types.c
	$(Q)$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $^

$(SETFREQ_TARGET): setFreqDaemon.c ../lock.c
	$(Q)$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $^

$(APPDAEMON_TARGET): $(GOTCHA_TARGET) appDaemon.c ../bstrlib.c ../bstrlib_helper.c
	$(Q)$(CC) -pthread -shared -fvisibility=hidden -fPIC $(CPPFLAGS) -Wl,-soname,$(notdir $(APPDAEMON_TARGET)).$(VERSION).$(RELEASE),--no-undefined -fstack-protector -I. -I$(GOTCHA_FOLDER)/include  -L$(GOTCHA_FOLDER) appDaemon.c ../bstrlib.c ../bstrlib_helper.c -o $@  -llikwid -llikwid-gotcha -ldl

$(SYSFSDAEMON_TARGET): sysfsDaemon.c
	$(Q)$(CC) -g -fsanitize=address -fsanitize=leak -fomit-frame-pointer  $(CFLAGS) $(CPPFLAGS) -o $@ sysfsDaemon.c
