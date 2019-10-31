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


int registerMapLength(RegisterMap* map);
int registerMapMaxPattern(RegisterMap* map);


int ptt2asm(char* pttfile, TestCase **testcase, char* asmfile);
void free_testcase(TestCase *testcase);
struct bstrList* get_benchmarks();
bstring get_user_path();
bstring get_compiler();
int compile_file(bstring compiler, bstring flags, bstring asmfile, bstring objfile);
int open_function(bstring location, TestCase *testcase);
void close_function(TestCase *testcase);

#endif
