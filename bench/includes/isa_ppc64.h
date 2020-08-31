/*
 * =======================================================================================
 *      Filename:  isa_ppc64.h
 *
 *      Description:  Definitions used for dynamically compile benchmarks for POWER systems
 *
 *      Version:   5.0.2
 *      Released:  31.08.2020
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
#ifndef LIKWID_BENCH_ISA_X86_H
#define LIKWID_BENCH_ISA_X86_H


#include <bstrlib.h>
#include <bstrlib_helper.h>

#define ARCHNAME "ppc64"
#define WORDLENGTH 4


int header(struct bstrList* code, char* funcname)
{
    bstring glline;
    bstring typeline;
    bstring label;
    bstring Llabel;
    bstring localentry;
    if (funcname)
    {
        glline = bformat(".globl %s", funcname);
        typeline = bformat(".type %s, @function", funcname);
        label = bformat("%s :", funcname);
        Llabel = bformat(".L.%s:", funcname);
        localentry = bformat(".localentry %s, .-%s", funcname, funcname);
    }
    else
    {
        glline = bformat(".globl kernelfunction");
        typeline = bformat(".type kernelfunction, @function");
        label = bformat("kernelfunction :");
        Llabel = bformat(".L.kernelfunction:");
        localentry = bformat(".localentry kernelfunction, .-kernelfunction", funcname, funcname);
    }

    bstrListAddChar(code, ".data");
    bstrListAddChar(code, ".align 64\nSCALAR:\n.double 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0");
    bstrListAddChar(code, ".align 64\nSSCALAR:\n.single 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0");
    bstrListAddChar(code, ".align 64\nISCALAR:\n.int 1, 1, 1, 1, 1, 1, 1, 1");
    bstrListAddChar(code, ".align 16\nOMM:\n.int 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15");
    bstrListAddChar(code, ".align 16\nIOMM:\n.int 0,16,32,48,64,80,96,128,144,160,176,192,208,224,240,256");
    bstrListAddChar(code, ".align 16\nTOMM:\n.int 0,2,4,6,16,18,20,22,32,34,36,38,48,50,52,54");
    bstrListAddChar(code, ".text");
    bstrListAddChar(code, ".set r0,0; .set SP,1; .set RTOC,2; .set r3,3; .set r4,4;");
    bstrListAddChar(code, ".set r5,5; .set r6,6; .set r7,7; .set r8,8; .set r9,9; .set r10,10;");
    bstrListAddChar(code, ".set r11,11; .set r12,12; .set r13,13; .set r14,14; .set r15,15; .set r16,16;");
    bstrListAddChar(code, ".set x0,0; .set x1,1; .set x2,2; .set x3,3; .set x4,4;");
    bstrListAddChar(code, ".set x5,5; .set x6,6; .set x7,7; .set x8,8; .set x9,9;");
    bstrListAddChar(code, ".set vec0,0; .set vec1,1; .set vec2,2; .set vec3,3;");
    bstrListAddChar(code, ".set vec4,4; .set vec5,5; .set vec6,6; .set vec7,7;");
    bstrListAddChar(code, ".set vec8,8; .set vec9,9; .set vec10,10; .set vec11,11;");
    bstrListAddChar(code, ".set vec12,12;");
    bstrListAddChar(code, ".abiversion 2");
    bstrListAddChar(code, ".section    \".toc\",\"aw\"");
    bstrListAddChar(code, ".section    \".text\"");
    bstrListAddChar(code, ".align 2");
    bstrListAdd(code, glline);
    bstrListAdd(code, typeline);
    bstrListAdd(code, label);
    bstrListAdd(code, Llabel);
    bstrListAdd(code, localentry);


    bstrListAddChar(code, "\n");

    bdestroy(glline);
    bdestroy(typeline);
    bdestroy(label);
    bdestroy(Llabel);
    bdestroy(localentry);
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
    bstrListAddChar(code, "blr");
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

    bstrListAddChar(code, "li r0, r12");
    bstring bstep = bformat("li r12, %d", step);
    bstrListAdd(code, bstep);
    bdestroy(bstep);
    bstrListAddChar(code, "divd r12, r3, r12");
    bstrListAddChar(code, "mtctr r12");
    bstrListAddChar(code, "li r12, r0");

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
        line = bformat("bdnz %sb", loopname);
    }
    else
    {
        line = bformat("bdnz kernelfunctionloopb");
    }
    bstring bstep = bformat("addi 4, 4, %d", step);
    bstrListAdd(code, bstep);
    bdestroy(bstep);
    bstrListAdd(code, line);

    bstrListAddChar(code, "\n");

    bdestroy(line);
    return 0;
}


static RegisterMap Registers[] = {
    {"GPR1", "r3"},
    {"GPR2", "r4"},
    {"GPR3", "r5"},
    {"GPR4", "r6"},
    {"GPR5", "r7"},
    {"GPR6", "r8"},
    {"GPR7", "r9"},
    {"GPR8", "r10"},
    {"GPR9", "r11"},
    {"GPR10", "r12"},
    {"GPR11", "r13"},
    {"GPR12", "r14"},
    {"GPR13", "r15"},
    {"GPR14", "r16"},
    {"GPR15", "17"},
    {"GPR16", "18"},
    {"GPR17", "19"},
    {"GPR18", "20"},
    {"GPR19", "21"},
    {"GPR20", "22"},
    {"GPR21", "23"},
    {"GPR22", "24"},
    {"GPR23", "25"},
    {"GPR24", "26"},
    {"GPR25", "27"},
    {"GPR26", "28"},
    {"GPR27", "29"},
    {"GPR28", "30"},
    {"GPR29", "31"},
    {"FPR1", "vec0"},
    {"FPR2", "vec1"},
    {"FPR3", "vec2"},
    {"FPR4", "vec3"},
    {"FPR5", "vec4"},
    {"FPR6", "vec5"},
    {"FPR7", "vec6"},
    {"FPR8", "vec7"},
    {"FPR9", "vec8"},
    {"FPR10", "vec9"},
    {"FPR11", "vec10"},
    {"FPR12", "vec11"},
    {"FPR13", "12"},
    {"FPR14", "13"},
    {"FPR15", "14"},
    {"FPR16", "15"},
    {"FPR17", "16"},
    {"FPR18", "17"},
    {"FPR19", "18"},
    {"FPR20", "19"},
    {"FPR21", "20"},
    {"FPR22", "21"},
    {"FPR23", "22"},
    {"FPR24", "23"},
    {"FPR25", "24"},
    {"FPR26", "25"},
    {"FPR27", "26"},
    {"FPR28", "27"},
    {"FPR29", "28"},
    {"FPR30", "29"},
    {"FPR31", "30"},
    {"FPR32", "31"},
    {"", ""},
};

static RegisterMap Arguments[] = {
    {"ARG1", "r3"},
    {"ARG2", "r4"},
    {"ARG3", "r5"},
    {"ARG4", "r6"},
    {"ARG5", "r7"},
    {"ARG6", "r8"},
    {"ARG7", "r9"},
    {"ARG8", "r10"},
    {"ARG9", "[BPTR+56]"},
    {"", ""},
};

static RegisterMap Sptr = {"SPTR", "SP"};
static RegisterMap Bptr = {"BPTR", "rbp"};

#endif
