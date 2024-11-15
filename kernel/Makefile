# =======================================================================================
#
#      Filename:  Makefile
#
#      Description:  Makefile for kernel module that enables RDPMC instruction in user-space
#
#      Version:   <VERSION>
#      Released:  <DATE>
#
#      Author:  Thomas Gruber (tr), thomas.roehl@googlemail.com
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

obj-m := enable_rdpmc.o
KERNELDIR ?= /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)

all:
	$(MAKE) -Wpacked -C $(KERNELDIR) M=$(PWD) modules

modules_install:
	install -m 666 enable_rdpmc.ko /lib/modules/$(shell uname -r)/extra/

clean:
	rm -f *.ko *.o modules.order Module.symvers enable_rdpmc.mod.c .enable_rdpmc*.cmd
	rm -rf .tmp_versions
