/*
 * =======================================================================================
 *      Filename:  ptt2asm.h
 *
 *      Description:  The interface to dynamically load ptt files
 *
 *      Version:   5.0.2
 *      Released:  31.08.2020
 *
 *      Author:  Thomas Gruber (tg), thomas.roehl@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2015 RRZE, University Erlangen-Nuremberg
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

static RegisterMap StreamPatterns[] = {
    {"STR0", "ARG2"},
    {"STR1", "ARG3"},
    {"STR2", "ARG4"},
    {"STR3", "ARG5"},
    {"STR4", "ARG6"},
    {"STR5", "[rbp+16]"},
    {"STR6", "[rbp+24]"},
    {"STR7", "[rbp+32]"},
    {"STR8", "[rbp+40]"},
    {"STR9", "[rbp+48]"},
    {"STR10", "[rbp+56]"},
    {"STR11", "[rbp+64]"},
    {"STR12", "[rbp+72]"},
    {"STR13", "[rbp+80]"},
    {"STR14", "[rbp+88]"},
    {"STR15", "[rbp+96]"},
    {"STR16", "[rbp+104]"},
    {"STR17", "[rbp+112]"},
    {"STR18", "[rbp+120]"},
    {"STR19", "[rbp+128]"},
    {"STR20", "[rbp+136]"},
    {"STR21", "[rbp+144]"},
    {"STR22", "[rbp+152]"},
    {"STR23", "[rbp+160]"},
    {"STR24", "[rbp+168]"},
    {"STR25", "[rbp+176]"},
    {"STR26", "[rbp+184]"},
    {"STR27", "[rbp+192]"},
    {"STR28", "[rbp+200]"},
    {"STR29", "[rbp+208]"},
    {"STR30", "[rbp+216]"},
    {"STR31", "[rbp+224]"},
    {"STR32", "[rbp+232]"},
    {"STR33", "[rbp+240]"},
    {"STR34", "[rbp+248]"},
    {"STR35", "[rbp+256]"},
    {"STR36", "[rbp+264]"},
    {"STR37", "[rbp+272]"},
    {"STR38", "[rbp+280]"},
    {"STR39", "[rbp+288]"},
    {"STR40", "[rbp+296]"},
    {"", ""},
};

struct bstrList* dynbench_getall();

int dynbench_test(bstring testname);
int dynbench_load(bstring testname, TestCase **testcase, char* tmpfolder, char *compilers, char* compileflags);
int dynbench_close(TestCase* testcase, char* tmpfolder);

#endif
