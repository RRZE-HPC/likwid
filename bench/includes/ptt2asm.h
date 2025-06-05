/*
 * =======================================================================================
 *      Filename:  ptt2asm.h
 *
 *      Description:  The interface to dynamically load ptt files
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Thomas Gruber (tg), thomas.roehl@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2019 RRZE, University Erlangen-Nuremberg
 *
 *      This program is free software: you can redistribute it and/or modify it under
 *      the terms of the GNU General Public License as published by the Free Software
 *      Foundation, either version 3 of the License, or (at your option) any later
 *      version.
 *
 *      This program is distributed in the hope that it will be useful, but WITHOUT ANY
 *      WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 *      PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 *      You should have received a copy of the GNU General Public License along with
 *      this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * =======================================================================================
 */
#ifndef LIKWID_BENCH_PTT2ASM_H
#define LIKWID_BENCH_PTT2ASM_H

typedef struct {
    char* pattern;
    char* reg;
} RegisterMap;

struct bstrList* dynbench_getall();

int dynbench_test(bstring testname);
int dynbench_load(bstring testname, TestCase **testcase, char* tmpfolder, char *compilers, char* compileflags);
int dynbench_close(TestCase* testcase, char* tmpfolder);
void dynbench_asm(bstring testname, char* tmpfolder, bstring outfile);

#endif
