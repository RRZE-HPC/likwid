#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <libgen.h>
#include <dirent.h>
#include <dlfcn.h>

#include <bstrlib.h>
#include <bstrlib_helper.h>

#include <test_types.h>


#include <ptt2asm.h>

#ifdef __x86_64
#include <isa_x86-64.h>
#endif
#ifdef __i386__
#include <isa_x86.h>
#endif
#ifdef __ARM_ARCH_7A__
#include <isa_armv7.h>
#endif
#ifdef __ARM_ARCH_8A
#include <isa_armv8.h>
#endif
#ifdef _ARCH_PPC
#include <isa_ppc64.h>
#endif

static struct bstrList* read_ptt(char *pttfile)
{
    int ret = 0;
    FILE* fp = NULL;
    char buf[BUFSIZ];
    struct bstrList* l = NULL;

    if (access(pttfile, R_OK))
    {
        return NULL;
    }

    bstring content = bfromcstr("");
    fp = fopen(pttfile, "r");
    if (fp == NULL) {
        fprintf(stderr, "fopen(%s): errno=%d\n", pttfile, errno);
        return NULL;
    }
    for (;;) {
        /* Read another chunk */
        ret = fread(buf, 1, sizeof(buf), fp);
        if (ret < 0) {
            fprintf(stderr, "fread(%p, 1, %lu, %p): %d, errno=%d\n", buf, sizeof(buf), fp, ret, errno);
            return NULL;
        }
        else if (ret == 0) {
            break;
        }
        bcatblk(content, buf, ret);
    }
    btrimws(content);

    l = bsplit(content, '\n');
    for (int i = 0; i < l->qty; i++)
    {
        btrimws(l->entry[i]);
    }

    bdestroy(content);
    return l;
}

static int write_asm(char* filename, struct bstrList* code)
{
    FILE* fp = NULL;
    char newline = '\n';
    size_t (*ownfwrite)(const void *ptr, size_t size, size_t nmemb, FILE *stream) = &fwrite;
    fp = fopen(filename, "w");
    if (fp)
    {
        for (int i = 0; i < code->qty; i++)
        {
            ownfwrite(bdata(code->entry[i]), 1, blength(code->entry[i]), fp);
            ownfwrite(&newline, 1, sizeof(char), fp);
        }
        fclose(fp);
        return 0;
    }
    return 1;
}

#define ANALYSE_PTT_GET_INT(line, pattern, variable) \
    bstring tmp = bmidstr((line), blength((pattern))+1, blength((line))-blength((pattern))); \
    btrimws(tmp); \
    (variable) = ownatoi(bdata(tmp)); \
    bdestroy(tmp); \

static struct bstrList* analyse_ptt(char* pttfile, TestCase** testcase)
{
    struct bstrList* ptt = NULL;
    TestCase* test = NULL;
    struct bstrList* code = NULL;
    bstring bBYTES = bformat("BYTES");
    bstring bFLOPS = bformat("FLOPS");
    bstring bSTREAMS = bformat("STREAMS");
    bstring bTYPE = bformat("TYPE");
    bstring bTYPEDOUBLE = bformat("DOUBLE");
    bstring bTYPESINGLE = bformat("SINGLE");
    bstring bTYPEINT = bformat("INT");
    bstring bDESC = bformat("DESC");
    bstring bLOADS = bformat("LOADS");
    bstring bSTORES = bformat("STORES");
    bstring bLOADSTORES = bformat("LOADSTORES");
    bstring bINSTCONST = bformat("INSTR_CONST");
    bstring bINSTLOOP = bformat("INSTR_LOOP");
    bstring bUOPS = bformat("UOPS");
    bstring bBRANCHES = bformat("BRANCHES");
    bstring bLOOP = bformat("LOOP");
    int (*ownatoi)(const char*) = &atoi;

    ptt = read_ptt(pttfile);

    if (ptt && ptt->qty > 0)
    {
        test = malloc(sizeof(TestCase));
        if (test)
        {
            test->loads = -1;
            test->stores = -1;
            test->loadstores = -1;
            test->branches = -1;
            test->instr_const = -1;
            test->instr_loop = -1;
            test->uops = -1;
            code = bstrListCreate();
            for (int i = 0; i < ptt->qty; i++)
            {
                if (bstrncmp(ptt->entry[i], bBYTES, blength(bBYTES)) == BSTR_OK)
                {
                    ANALYSE_PTT_GET_INT(ptt->entry[i], bBYTES, test->bytes);
                }
                else if (bstrncmp(ptt->entry[i], bFLOPS, blength(bFLOPS)) == BSTR_OK)
                {
                    ANALYSE_PTT_GET_INT(ptt->entry[i], bFLOPS, test->flops);
                }
                else if (bstrncmp(ptt->entry[i], bSTREAMS, blength(bSTREAMS)) == BSTR_OK)
                {
                    ANALYSE_PTT_GET_INT(ptt->entry[i], bSTREAMS, test->streams);
                }
                else if (bstrncmp(ptt->entry[i], bLOADS, blength(bLOADS)) == BSTR_OK)
                {
                    ANALYSE_PTT_GET_INT(ptt->entry[i], bLOADS, test->loads);
                }
                else if (bstrncmp(ptt->entry[i], bSTORES, blength(bSTORES)) == BSTR_OK)
                {
                    ANALYSE_PTT_GET_INT(ptt->entry[i], bSTORES, test->stores);
                }
                else if (bstrncmp(ptt->entry[i], bLOADSTORES, blength(bLOADSTORES)) == BSTR_OK)
                {
                    ANALYSE_PTT_GET_INT(ptt->entry[i], bLOADSTORES, test->loadstores);
                }
                else if (bstrncmp(ptt->entry[i], bINSTCONST, blength(bINSTCONST)) == BSTR_OK)
                {
                    ANALYSE_PTT_GET_INT(ptt->entry[i], bINSTCONST, test->instr_const);
                }
                else if (bstrncmp(ptt->entry[i], bINSTLOOP, blength(bINSTLOOP)) == BSTR_OK)
                {
                    ANALYSE_PTT_GET_INT(ptt->entry[i], bINSTLOOP, test->instr_loop);
                }
                else if (bstrncmp(ptt->entry[i], bUOPS, blength(bUOPS)) == BSTR_OK)
                {
                    ANALYSE_PTT_GET_INT(ptt->entry[i], bUOPS, test->uops);
                }
                else if (bstrncmp(ptt->entry[i], bBRANCHES, blength(bBRANCHES)) == BSTR_OK)
                {
                    ANALYSE_PTT_GET_INT(ptt->entry[i], bBRANCHES, test->branches);
                }
                else if (bstrncmp(ptt->entry[i], bLOOP, blength(bLOOP)) == BSTR_OK)
                {
                    ANALYSE_PTT_GET_INT(ptt->entry[i], bLOOP, test->stride);
                    bstrListAdd(code, ptt->entry[i]);
                }
                else if (bstrncmp(ptt->entry[i], bDESC, blength(bDESC)) == BSTR_OK)
                {
                    test->desc = malloc((blength(ptt->entry[i])+2)*sizeof(char));
                    if (test->desc)
                    {
                        int ret = snprintf(test->desc, blength(ptt->entry[i])+1, "%s", bdataofs(ptt->entry[i], blength(bDESC)+1));
                        if (ret > 0)
                        {
                            test->desc[ret] = '\0';
                        }
                    }
                }
                else if (bstrncmp(ptt->entry[i], bTYPE, blength(bTYPE)) == BSTR_OK)
                {
                    bstring btype = bmidstr(ptt->entry[i], blength(bTYPE)+1, blength(ptt->entry[i])-blength(bTYPE));
                    btrimws(btype);
                    if (bstrncmp(btype, bTYPEDOUBLE, blength(bTYPEDOUBLE)) == BSTR_OK)
                    {
                        test->type = DOUBLE;
                    }
                    else if (bstrncmp(btype, bTYPESINGLE, blength(bTYPESINGLE)) == BSTR_OK)
                    {
                        test->type = SINGLE;
                    }
                    else if (bstrncmp(btype, bTYPEINT, blength(bTYPEINT)) == BSTR_OK)
                    {
                        test->type = INT;
                    }
                    else
                    {
                        fprintf(stderr, "Failed to determine type of benchmark\n");
                        bdestroy(btype);
                        bstrListDestroy(code);
                        free(test);
                        test = NULL;
                        code = NULL;
                        break;
                    }
                    bdestroy(btype);
                }
                else
                {
                    bstrListAdd(code, ptt->entry[i]);
                }
            }
            *testcase = test;
        }
        bstrListDestroy(ptt);
    }

    bdestroy(bBYTES);
    bdestroy(bFLOPS);
    bdestroy(bSTREAMS);
    bdestroy(bTYPE);
    bdestroy(bTYPEDOUBLE);
    bdestroy(bTYPESINGLE);
    bdestroy(bTYPEINT);
    bdestroy(bDESC);
    bdestroy(bLOADS);
    bdestroy(bSTORES);
    bdestroy(bLOADSTORES);
    bdestroy(bINSTCONST);
    bdestroy(bINSTLOOP);
    bdestroy(bUOPS);
    bdestroy(bBRANCHES);
    bdestroy(bLOOP);
    return code;
}

static int set_testname(char *pttfile, TestCase* testcase)
{
    if ((!testcase)||(!pttfile))
    {
        return -EINVAL;
    }
    bstring ptt = bfromcstr(basename(pttfile));
    int dot = bstrrchrp(ptt, '.', blength(ptt)-1);
    btrunc(ptt, dot);
    testcase->name = malloc((blength(ptt)+2) * sizeof(char));
    int ret = snprintf(testcase->name, blength(ptt)+1, "%s", bdata(ptt));
    if (ret > 0)
    {
        testcase->name[ret] = '\0';
    }
    bdestroy(ptt);
    return 0;
}


static struct bstrList* parse_asm(TestCase* testcase, struct bstrList* input)
{
    struct bstrList* output = NULL;
    if (testcase && input)
    {

        struct bstrList* pre = bstrListCreate();
        struct bstrList* loop = bstrListCreate();
        int got_loop = 0;
        bstring bLOOP = bformat("LOOP");
        int step = testcase->stride;

        for (int i = 0; i < input->qty; i++)
        {
            if (bstrncmp(input->entry[i], bLOOP, blength(bLOOP)) == BSTR_OK)
            {
                got_loop = 1;
                continue;
            }
            if (!got_loop)
            {
                bstrListAdd(pre, input->entry[i]);
            }
            else
            {
                bstrListAdd(loop, input->entry[i]);
            }
        }
        bdestroy(bLOOP);

        output = bstrListCreate();

        header(output, testcase->name);

        for (int i = 0; i < pre->qty; i++)
        {
            bstrListAdd(output, pre->entry[i]);
        }
        loopheader(output, "1", step);
        for (int i = 0; i < loop->qty; i++)
        {
            bstrListAdd(output, loop->entry[i]);
        }
        loopfooter(output, "1", step);

        footer(output, testcase->name);

        bstrListDestroy(pre);
        bstrListDestroy(loop);
    }
    return output;
}

static int searchreplace(bstring line, RegisterMap* map)
{
    int maxlen = registerMapMaxPattern(map);
    int size = registerMapLength(map);
    for (int s = maxlen; s>= 1; s--)
    {
        int c = 0;
        for (int j = 0; j < size; j++)
        {
            if (strlen(map[j].pattern) == s)
            {
                bstring pat = bfromcstr(map[j].pattern);
                bstring reg = bfromcstr(map[j].reg);
                bfindreplace(line, pat, reg, 0);
                bdestroy(pat);
                bdestroy(reg);
                c++;
            }
        }
        if (c == 0)
        {
            break;
        }
    }
    return 0;
}

static int prepare_code(struct bstrList* code)
{
    if (code)
    {
        for (int i = 0; i < code->qty; i++)
        {
            searchreplace(code->entry[i], StreamPatterns);
        }
        for (int i = 0; i < code->qty; i++)
        {
            searchreplace(code->entry[i], Registers);
        }
        for (int i = 0; i < code->qty; i++)
        {
            searchreplace(code->entry[i], Arguments);
        }
        for (int i = 0; i < code->qty; i++)
        {
            bstring pat = bfromcstr(Sptr.pattern);
            bstring reg = bfromcstr(Sptr.reg);
            bfindreplace(code->entry[i], pat, reg, 0);
            bdestroy(pat);
            bdestroy(reg);
        }
        for (int i = 0; i < code->qty; i++)
        {
            bstring pat = bfromcstr(Bptr.pattern);
            bstring reg = bfromcstr(Bptr.reg);
            bfindreplace(code->entry[i], pat, reg, 0);
            bdestroy(pat);
            bdestroy(reg);
        }
    }
    return 0;
}

static void print_testcase(TestCase* test)
{
    printf("Name: %s\n", test->name);
    printf("Desc: %s\n", test->desc);
    printf("Bytes: %d\n", test->bytes);
    printf("Flops: %d\n", test->flops);
    printf("Type: %d\n", test->type);
    printf("Loads: %d\n", test->loads);
}



void free_testcase(TestCase *testcase)
{
    if (testcase)
    {
        free(testcase->name);
        free(testcase->desc);
        free(testcase);
    }
}

struct bstrList* get_benchmarks()
{
    int totalgroups = 0;
    struct bstrList* list = NULL;
    DIR *dp = NULL;
    struct dirent *ep = NULL;
    DIR * (*ownopendir)(const char* folder) = &opendir;
    int (*ownaccess)(const char*, int) = &access;

    bstring path = bformat("%s/.likwid/bench/%s", getenv("HOME"), ARCHNAME);

    if (!ownaccess(bdata(path), R_OK|X_OK))
    {
        dp = ownopendir(bdata(path));
        if (dp != NULL)
        {
            list = bstrListCreate();
            while (ep = readdir(dp))
            {
                if (strncmp(&(ep->d_name[strlen(ep->d_name)-4]), ".ptt", 4) == 0)
                {
                    totalgroups++;
                    bstring dname = bfromcstr(ep->d_name);
                    btrunc(dname, blength(dname)-4);
                    bstrListAdd(list, dname);
                    bdestroy(dname);
                }
            }
            closedir(dp);
        }
    }
    bdestroy(path);
    return list;
}

bstring get_user_path()
{
    return bformat("%s/.likwid/bench/%s", getenv("HOME"), ARCHNAME);
}

bstring get_compiler()
{
    bstring path = bfromcstr(getenv("PATH"));
    struct bstrList *plist = NULL;
    int (*ownaccess)(const char*, int) = access;

    plist = bsplit(path, ':');

    for (int i = 0; i < plist->qty; i++)
    {
        bstring tmp = bformat("%s/gcc", bdata(plist->entry[i]));
        if (!ownaccess(bdata(tmp), R_OK|X_OK))
        {
            bdestroy(path);
            bstrListDestroy(plist);
            return tmp;
        }
        bdestroy(tmp);
        tmp = bformat("%s/icc", bdata(plist->entry[i]));
        if (!ownaccess(bdata(tmp), R_OK|X_OK))
        {
            bdestroy(path);
            bstrListDestroy(plist);
            return tmp;
        }
        bdestroy(tmp);
        tmp = bformat("%s/pgcc", bdata(plist->entry[i]));
        if (!ownaccess(bdata(tmp), R_OK|X_OK))
        {
            bdestroy(path);
            bstrListDestroy(plist);
            return tmp;
        }
        bdestroy(tmp);
    }
    bdestroy(path);
    bstrListDestroy(plist);
    return bfromcstr("");
}

int compile_file(bstring compiler, bstring flags, bstring asmfile, bstring objfile)
{
    if (blength(compiler) == 0 || blength(asmfile) == 0)
        return -1;
    char buf[1024];
    bstring bstdout = bfromcstr("");


/*    bcatcstr(outfile, basename(bdata(file)));*/
/*    btrunc(outfile, blength(outfile)-1);*/
/*    bconchar(outfile, 'o');*/
/*    binsert(outfile, 0, outfolder, '\0');*/

    bstring cmd = bformat("%s %s %s -o %s", bdata(compiler), bdata(flags), bdata(asmfile), bdata(objfile));

    FILE * fp = popen(bdata(cmd), "r");
    if (fp)
    {
        for (;;) {
            /* Read another chunk */
            int ret = fread(buf, 1, sizeof(buf), fp);
            if (ret < 0) {
                fprintf(stderr, "fread(%p, 1, %lu, %p): %d, errno=%d\n", buf, sizeof(buf), fp, ret, errno);
                bdestroy(cmd);
                bdestroy(bstdout);
                return -1;
            }
            else if (ret == 0) {
                break;
            }
            bcatblk(bstdout, buf, ret);
        }
        pclose(fp);
    }
    bdestroy(cmd);
    bdestroy(bstdout);

    return 0;
}


int open_function(bstring location, TestCase *testcase)
{
    void* handle;
    char *error;
    FuncPrototype func = NULL;
    void* (*owndlsym)(void*, const char*) = dlsym;

    dlerror();
    testcase->dlhandle = dlopen(bdata(location), RTLD_LAZY);
    if (!testcase->dlhandle) {
        fprintf(stderr, "Error opening location %s: %s\n", bdata(location), dlerror());
        return -1;
    }
    dlerror();
    testcase->kernel = owndlsym(testcase->dlhandle, testcase->name);
    if ((error = dlerror()) != NULL)  {
        dlclose(testcase->dlhandle);
        fprintf(stderr, "Error opening function %s: %s\n", testcase->name, error);
        return -1;
    }
    dlerror();

    return 0;
}

void close_function(TestCase *testcase)
{
    if (testcase)
    {
        dlclose(testcase->dlhandle);
        testcase->dlhandle = NULL;
        testcase->kernel = NULL;
    }
}

int ptt2asm(char* pttfile, TestCase **testcase, char* asmfile)
{
    int err = 0;
    TestCase *test = NULL;
    struct bstrList* code = analyse_ptt(pttfile, &test);
    if (code)
    {
        set_testname(pttfile, test);
        if (asmfile)
        {
            struct bstrList* asmb = parse_asm(test, code);
            prepare_code(asmb);
            write_asm(asmfile, asmb);
            bstrListDestroy(asmb);
        }
    }
    else
    {
        err = 1;
    }
    bstrListDestroy(code);
    *testcase = test;
    return err;
}


int registerMapLength(RegisterMap* map)
{
    int i = 0;
    while (strlen(map[i].pattern) > 0)
    {
        i++;
    }
    return i;
}

int registerMapMaxPattern(RegisterMap* map)
{
    int i = 0;
    int max = 0;
    while (strlen(map[i].pattern) > 0)
    {
        if (strlen(map[i].pattern) > max)
            max = strlen(map[i].pattern);
        i++;
    }
    return max;
}

/*int main()*/
/*{*/
/*    TestCase *testcase;*/
/*    ptt2asm("x86-64/load.ptt", &testcase, "load.s");*/
/*    free_testcase(testcase);*/

/*    struct bstrList* l = get_benchmarks();*/
/*    if (l)*/
/*    {*/
/*        for (int i = 0; i < l->qty; i++)*/
/*            printf("%s\n", bdata(l->entry[i]));*/
/*    }*/
/*}*/
