/*
 * =======================================================================================
 *      Filename:  isa_armv8.h
 *
 *      Description:  Definitions used for dynamically compile benchmarks for ARMv8 systems
 *
 *      Version:   5.0.2
 *      Released:  06.10.2020
 *
 *      Author:   Thomas Gruber (tg), thomas.roehl@gmail.com
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
#ifndef LIKWID_BENCH_ISA_ARMV8_H
#define LIKWID_BENCH_ISA_ARMV8_H

#include <bstrlib.h>
#include <bstrlib_helper.h>

#define ARCHNAME "armv8"
#define WORDLENGTH 4

int header(struct bstrList* code, char* funcname)
{
    bstring glline;
    bstring typeline;
    bstring label;
    if (funcname)
    {
        glline = bformat(".global %s", funcname);
        typeline = bformat(".type %s, @function", funcname);
        label = bformat("%s :", funcname);
    }
    else
    {
        glline = bformat(".global kernelfunction");
        typeline = bformat(".type kernelfunction, @function");
        label = bformat("kernelfunction :");
    }


    bstrListAddChar(code, ".cpu    generic+fp+simd");
    bstrListAddChar(code, ".data");
    bstrListAddChar(code, ".text");
    bstrListAdd(code, glline);
    bstrListAdd(code, typeline);
    bstrListAdd(code, label);

    bstrListAddChar(code, "\n");

    bdestroy(glline);
    bdestroy(typeline);
    bdestroy(label);
    return 0;
}

int footer(struct bstrList* code, char* funcname)
{
    bstring line;
    if (funcname)
    {
        line = bformat(".size %s, .-%s", funcname, funcname);
    }
    else
    {
        line = bformat(".size kernelfunction, .-kernelfunction");
    }
    bstrListAddChar(code, ".exit:");
    bstrListAddChar(code, "ret");

    bstrListAdd(code, line);

    bstrListAddChar(code, "\n");

    bstrListAddChar(code, "#if defined(__linux__) && defined(__ELF__)");
    bstrListAddChar(code, ".section .note.GNU-stack,\"\",%progbits");
    bstrListAddChar(code, "#endif");

    bdestroy(line);
    return 0;
}

int loopheader(struct bstrList* code, char* loopname, int step)
{
    bstring line;
    if (loopname)
    {
        line = bformat("%s:", loopname);
    }
    else
    {
        line = bformat("kernelfunctionloop:");
    }

    bstrListAddChar(code, "mov   GPR6, 0");
    bstrListAdd(code, line);
    bstrListAddChar(code, "\n");

    bdestroy(line);
    return 0;
}

int loopfooter(struct bstrList* code, char* loopname, int step)
{
    bstring line;
    if (loopname)
    {
        line = bformat("tblt %s", loopname);
    }
    else
    {
        line = bformat("tblt kernelfunctionloop");
    }
    bstring bstep = bformat("add GPR6, GPR6, #%d", step);
    bstrListAdd(code, bstep);
    bdestroy(bstep);
    bstrListAddChar(code, "cmp   GPR6, ARG1");
    bstrListAdd(code, line);

    bstrListAddChar(code, "\n");

    bdestroy(line);
    return 0;
}


static RegisterMap Registers[] = {
    {"GPR1", "x1"},
    {"GPR2", "x2"},
    {"GPR3", "x3"},
    {"GPR4", "x4"},
    {"GPR5", "x5"},
    {"GPR6", "x6"},
    {"GPR7", "x7"},
    {"GPR8", "x8"},
    {"GPR9", "x9"},
    {"GPR10", "x10"},
    {"GPR11", "x11"},
    {"GPR12", "x12"},
    {"GPR13", "x13"},
    {"GPR14", "x14"},
    {"GPR15", "x15"},
    {"GPR16", "x16"},
    {"GPR17", "x17"},
    {"GPR18", "x18"},
    {"GPR19", "x19"},
    {"GPR20", "x20"},
    {"GPR21", "x21"},
    {"GPR22", "x22"},
    {"FPR1", "d0"},
    {"FPR2", "d1"},
    {"FPR3", "d2"},
    {"FPR4", "d3"},
    {"FPR5", "d4"},
    {"FPR6", "d5"},
    {"FPR7", "d6"},
    {"FPR8", "d7"},
    {"FPR9", "d8"},
    {"FPR10", "d9"},
    {"FPR11", "d10"},
    {"FPR12", "d11"},
    {"FPR13", "d12"},
    {"FPR14", "d13"},
    {"FPR15", "d14"},
    {"FPR16", "d15"},
    {"", ""},
};

static RegisterMap Arguments[] = {
    {"ARG1", "x0"},
    {"ARG2", "x1"},
    {"ARG3", "x2"},
    {"ARG4", "x3"},
    {"ARG5", "x4"},
    {"ARG6", "x5"},
    {"ARG7", "x6"},
    {"ARG8", "x7"},
    {"ARG9", "[SPTR+32]"},
    {"ARG10", "[SPTR+40]"},
    {"ARG11", "[SPTR+48]"},
    {"ARG12", "[SPTR+56]"},
    {"ARG13", "[SPTR+64]"},
    {"ARG14", "[SPTR+72]"},
    {"ARG15", "[SPTR+80]"},
    {"ARG16", "[SPTR+88]"},
    {"ARG17", "[SPTR+96]"},
    {"ARG18", "[SPTR+104]"},
    {"ARG19", "[SPTR+112]"},
    {"ARG20", "[SPTR+120]"},
    {"ARG21", "[SPTR+128]"},
    {"ARG22", "[SPTR+136]"},
    {"ARG23", "[SPTR+144]"},
    {"ARG24", "[SPTR+152]"},
    {"", ""},
};

static RegisterMap Sptr = {"SPTR", "sp"};
static RegisterMap Bptr = {"BPTR", "rbp"};

#endif
