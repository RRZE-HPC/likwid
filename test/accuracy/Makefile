# =======================================================================================
#  
#      Filename:  Makefile
# 
#      Description:  Central Makefile
# 
#      Version:   3.1.1
#      Released:  12.2.2014
# 
#      Author:  Thomas Roehl (tr), thomas.roehl@googlemail.com
#      Project:  likwid
#
#      Copyright (C) 2014 Jan Treibig
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

LIKWID_ROOT=../..


all: prepare

prepare: 
	sed -i -e s/'INSTRUMENT_BENCH = .*#NO SPACE'/'INSTRUMENT_BENCH = false#NO SPACE'/g $(LIKWID_ROOT)/config.mk
	make -C $(LIKWID_ROOT) distclean
	make -C $(LIKWID_ROOT)
	@cp $(LIKWID_ROOT)/likwid-bench likwid-bench-plain
	sed -i -e s/'INSTRUMENT_BENCH = .*#NO SPACE'/'INSTRUMENT_BENCH = true#NO SPACE'/g $(LIKWID_ROOT)/config.mk
	make -C $(LIKWID_ROOT) distclean
	make -C $(LIKWID_ROOT)
	@cp $(LIKWID_ROOT)/likwid-bench likwid-bench-marker
	
run: likwid-bench-plain likwid-bench-marker likwid-tester
	exec ./likwid-tester
