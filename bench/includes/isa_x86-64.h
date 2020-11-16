/*
 * =======================================================================================
 *      Filename:  isa_x86-64.h
 *
 *      Description:  Definitions used for dynamically compile benchmarks for x86-64 systems
 *
 *      Version:   5.1
 *      Released:  16.11.2020
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
#ifndef LIKWID_BENCH_ISA_X8664_H
#define LIKWID_BENCH_ISA_X8664_H

#include <bstrlib.h>
#include <bstrlib_helper.h>

#define ARCHNAME "x86-64"


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


    bstrListAddChar(code, ".intel_syntax noprefix");
    bstrListAddChar(code, ".data");
    bstrListAddChar(code, ".align 64\nSCALAR:\n.double 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0");
    bstrListAddChar(code, ".align 64\nSSCALAR:\n.single 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0");
    bstrListAddChar(code, ".align 64\nISCALAR:\n.int 1, 1, 1, 1, 1, 1, 1, 1");
    bstrListAddChar(code, ".align 16\nOMM:\n.int 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15");
    bstrListAddChar(code, ".align 16\nIOMM:\n.int 0,16,32,48,64,80,96,128,144,160,176,192,208,224,240,256");
    bstrListAddChar(code, ".align 16\nTOMM:\n.int 0,2,4,6,16,18,20,22,32,34,36,38,48,50,52,54");
    bstrListAddChar(code, ".text");
    bstrListAdd(code, glline);
    bstrListAdd(code, typeline);
    bstrListAdd(code, label);
    bstrListAddChar(code, "push rbp");
    bstrListAddChar(code, "mov rbp, rsp");
    bstrListAddChar(code, "push rbx");
    bstrListAddChar(code, "push r12");
    bstrListAddChar(code, "push r13");
    bstrListAddChar(code, "push r14");
    bstrListAddChar(code, "push r15");

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
    bstrListAddChar(code, "pop r15");
    bstrListAddChar(code, "pop r14");
    bstrListAddChar(code, "pop r13");
    bstrListAddChar(code, "pop r12");
    bstrListAddChar(code, "pop rbx");
    bstrListAddChar(code, "mov  rsp, rbp");
    bstrListAddChar(code, "pop rbp");
    bstrListAddChar(code, "ret");

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

    bstrListAddChar(code, "xor   GPR1, GPR1");
    bstrListAddChar(code, ".align 16");
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
        line = bformat("jl %sb", loopname);
    }
    else
    {
        line = bformat("jl kernelfunctionloopb");
    }
    bstring bstep = bformat("add GPR1, %d", step);
    bstrListAdd(code, bstep);
    bdestroy(bstep);
    bstrListAddChar(code, "cmp   GPR1, ARG1");
    bstrListAdd(code, line);

    bstrListAddChar(code, "\n");

    bdestroy(line);
    return 0;
}


static RegisterMap Registers[] = {
    {"GPR1", "rax"},
    {"GPR2", "rbx"},
    {"GPR3", "rcx"},
    {"GPR4", "rdx"},
    {"GPR5", "rsi"},
    {"GPR6", "rdi"},
    {"GPR7", "r8"},
    {"GPR8", "r9"},
    {"GPR9", "r10"},
    {"GPR10", "r11"},
    {"GPR11", "r12"},
    {"GPR12", "r13"},
    {"GPR13", "r14"},
    {"GPR14", "r15"},
    {"FPR1", "xmm0"},
    {"FPR2", "xmm1"},
    {"FPR3", "xmm2"},
    {"FPR4", "xmm3"},
    {"FPR5", "xmm4"},
    {"FPR6", "xmm5"},
    {"FPR7", "xmm6"},
    {"FPR8", "xmm7"},
    {"FPR9", "xmm8"},
    {"FPR10", "xmm9"},
    {"FPR11", "xmm10"},
    {"FPR12", "xmm11"},
    {"FPR13", "xmm12"},
    {"FPR14", "xmm13"},
    {"FPR15", "xmm14"},
    {"FPR16", "xmm15"},
    {"", ""},
};

static RegisterMap Arguments[] = {
    {"ARG1", "rdi"},
    {"ARG2", "rsi"},
    {"ARG3", "rdx"},
    {"ARG4", "rcx"},
    {"ARG5", "r8"},
    {"ARG6", "r9"},
    {"ARG7", "[BPTR+16]"},
    {"ARG8", "[BPTR+24]"},
    {"ARG9", "[BPTR+32]"},
    {"ARG10", "[BPTR+40]"},
    {"ARG11", "[BPTR+48]"},
    {"ARG12", "[BPTR+56]"},
    {"ARG13", "[BPTR+64]"},
    {"ARG14", "[BPTR+72]"},
    {"ARG15", "[BPTR+80]"},
    {"ARG16", "[BPTR+88]"},
    {"ARG17", "[BPTR+96]"},
    {"ARG18", "[BPTR+104]"},
    {"ARG19", "[BPTR+112]"},
    {"ARG20", "[BPTR+120]"},
    {"ARG21", "[BPTR+128]"},
    {"ARG22", "[BPTR+136]"},
    {"ARG23", "[BPTR+144]"},
    {"ARG24", "[BPTR+152]"},
    {"", ""},
};

static RegisterMap Sptr = {"SPTR", "rsp"};
static RegisterMap Bptr = {"BPTR", "rbp"};

#endif
