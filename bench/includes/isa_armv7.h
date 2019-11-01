/*
 * =======================================================================================
 *      Filename:  isa_armv7.h
 *
 *      Description:  Definitions used for dynamically compile benchmarks for ARMv7 systems
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Thomas Gruber (tg), thomas.roehl@gmail.com
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
#ifndef LIKWID_BENCH_ISA_ARMV7_H
#define LIKWID_BENCH_ISA_ARMV7_H

#include <bstrlib.h>
#include <bstrlib_helper.h>

#define ARCHNAME "armv7"
#define WORDLENGTH 4

int header(struct bstrList* code, char* funcname)
{
    bstring glline;
    bstring typeline;
    bstring label;
    if (funcname)
    {
        glline = bformat(".global %s", funcname);
        typeline = bformat(".type %s, \%function", funcname);
        label = bformat("%s :", funcname);
    }
    else
    {
        glline = bformat(".global kernelfunction");
        typeline = bformat(".type kernelfunction, \%function");
        label = bformat("kernelfunction :");
    }


    bstrListAddChar(code, ".cpu    cortex-a15\n.fpu    neon-vfpv4");
    bstrListAddChar(code, ".data");
    bstrListAddChar(code, ".text");
    bstrListAdd(code, glline);
    bstrListAdd(code, typeline);
    bstrListAdd(code, label);
    bstrListAddChar(code, "push     {r4-r7, lr}");
    bstrListAddChar(code, "add      r7, sp, #12");
    bstrListAddChar(code, "push     {r8, r10, r11}");
    bstrListAddChar(code, "vstmdb   sp!, {d8-d15}");



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
    bstrListAddChar(code, "vldmia   sp!, {d8-d15}");
    bstrListAddChar(code, "pop      {r8, r10, r11}");
    bstrListAddChar(code, "pop      {r4-r7, pc}");

    bstrListAdd(code, line);

    bstrListAddChar(code, "\n");

    bstrListAddChar(code, "#if defined(__linux__) && defined(__ELF__)");
    bstrListAddChar(code, ".section .note.GNU-stack,\"\",%progbits");
    bstrListAddChar(code, "#endif");

    bdestroy(line);
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

    bstrListAddChar(code, "mov   GPR4, #0");
    bstrListAddChar(code, ".align 2");
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
        line = bformat("blt %sb", loopname);
    }
    else
    {
        line = bformat("blt kernelfunctionloopb");
    }
    bstring bstep = bformat("add GPR4, #%d", step);
    bstrListAdd(code, bstep);
    bdestroy(bstep);
    bstrListAddChar(code, "cmp GPR4, GPR1");
    bstrListAdd(code, line);

    bstrListAddChar(code, "\n");

    bdestroy(line);
    return 0;
}


static RegisterMap Registers[] = {
    {"GPR1", "r0"},
    {"GPR2", "r1"},
    {"GPR3", "r2"},
    {"GPR4", "r3"},
    {"GPR5", "r4"},
    {"GPR6", "r5"},
    {"GPR7", "r6"},
    {"GPR8", "r7"},
    {"GPR9", "r8"},
    {"GPR10", "r9"},
    {"GPR11", "r10"},
    {"GPR12", "r11"},
    {"GPR13", "r12"},
    {"GPR14", "r13"},
    {"GPR15", "r14"},
    {"GPR16", "r15"},
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
    {"ARG1", "r0"},
    {"ARG2", "r1"},
    {"ARG3", "r2"},
    {"ARG4", "r3"},
    {"ARG7", "[SPTR+8]"},
    {"ARG8", "[SPTR+12]"},
    {"ARG9", "[SPTR+16]"},
    {"ARG10", "[SPTR+20]"},
    {"ARG11", "[SPTR+24]"},
    {"ARG12", "[SPTR+28]"},
    {"ARG13", "[SPTR+32]"},
    {"ARG14", "[SPTR+36]"},
    {"ARG15", "[SPTR+40]"},
    {"ARG16", "[SPTR+44]"},
    {"ARG17", "[SPTR+48]"},
    {"ARG18", "[SPTR+52]"},
    {"ARG19", "[SPTR+56]"},
    {"ARG20", "[SPTR+60]"},
    {"ARG21", "[SPTR+64]"},
    {"ARG22", "[SPTR+68]"},
    {"ARG23", "[SPTR+72]"},
    {"ARG24", "[SPTR+76]"},
    {"", ""},
};

static RegisterMap Sptr = {"SPTR", "sp"};
static RegisterMap Bptr = {"BPTR", "rbp"};

#endif
