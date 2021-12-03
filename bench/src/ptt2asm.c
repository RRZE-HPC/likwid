/*
 * =======================================================================================
 *      Filename:  ptt2asm.c
 *
 *      Description:  The interface to dynamically load ptt files
 *
 *      Version:   5.2.1
 *      Released:  03.12.2021
 *
 *      Author:   Thomas Gruber (tg), thomas.roehl@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2021 NHR@FAU, University Erlangen-Nuremberg
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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <libgen.h>
#include <dirent.h>
#include <dlfcn.h>
#include <sys/stat.h>
#include <sys/types.h>

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

static int registerMapLength(RegisterMap* map)
{
    int i = 0;
    while (strlen(map[i].pattern) > 0)
    {
        i++;
    }
    return i;
}

static int registerMapMaxPattern(RegisterMap* map)
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

static struct bstrList* read_ptt(bstring pttfile)
{
    int ret = 0;
    FILE* fp = NULL;
    char buf[BUFSIZ];
    struct bstrList* l = NULL;

    if (access(bdata(pttfile), R_OK))
    {
        return NULL;
    }

    bstring content = bfromcstr("");
    fp = fopen(bdata(pttfile), "r");
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

static int write_asm(bstring filename, struct bstrList* code)
{
    FILE* fp = NULL;
    char newline = '\n';
    size_t (*ownfwrite)(const void *ptr, size_t size, size_t nmemb, FILE *stream) = &fwrite;
    fp = fopen(bdata(filename), "w");
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

static struct bstrList* analyse_ptt(bstring pttfile, TestCase** testcase)
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
        bstring bloopname = bformat("%s_loop", testcase->name);
	bstring bLOOP = bfromcstr("LOOP");
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

        output = bstrListCreate();

        header(output, testcase->name);

        for (int i = 0; i < pre->qty; i++)
        {
            bstrListAdd(output, pre->entry[i]);
        }
        loopheader(output, bdata(bloopname), step);
        for (int i = 0; i < loop->qty; i++)
        {
            bstrListAdd(output, loop->entry[i]);
        }
        loopfooter(output, bdata(bloopname), step);

        footer(output, testcase->name);

        bstrListDestroy(pre);
        bstrListDestroy(loop);
        bdestroy(bLOOP);
        bdestroy(bloopname);
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

static int dynbench_getall_folder(char *path, struct bstrList** benchmarks)
{
    int files = 0;
    DIR *dp = NULL;
    struct dirent *ep = NULL;
    DIR * (*ownopendir)(const char* folder) = &opendir;
    int (*ownaccess)(const char*, int) = &access;

    if (!ownaccess(path, R_OK|X_OK))
    {
        dp = ownopendir(path);
        if (dp != NULL)
        {
            while ((ep = readdir(dp)))
            {
                if ( (strncmp(&(ep->d_name[strlen(ep->d_name)-4]), ".ptt", 4) == 0))
                {
                    files++;
                    bstring dname = bfromcstr(ep->d_name);
                    btrunc(dname, blength(dname)-4);
                    bstrListAdd(*benchmarks, dname);
                    bdestroy(dname);
                }
            }
            closedir(dp);
        }
    }
    return files;
}

struct bstrList* dynbench_getall()
{
    int totalfiles = 0;

    struct bstrList* list = bstrListCreate();

    bstring home = bformat("%s/.likwid/bench/%s", getenv("HOME"), ARCHNAME);
    totalfiles += dynbench_getall_folder(bdata(home), &list);
    bdestroy(home);

    char cwd[200];
    if (getcwd(cwd, 200) != NULL)
    {
        totalfiles += dynbench_getall_folder(cwd, &list);
    }
    return list;
}


static bstring get_compiler(bstring candidates)
{
    bstring compiler = NULL;
    bstring path = bfromcstr(getenv("PATH"));
    struct bstrList *plist = NULL;
    struct bstrList *clist = NULL;
    int (*ownaccess)(const char*, int) = access;

    plist = bsplit(path, ':');
    clist = bsplit(candidates, ',');

    for (int i = 0; i < plist->qty && (!compiler); i++)
    {
        for (int j = 0; j < clist->qty && (!compiler); j++)
        {
            bstring tmp = bformat("%s/%s", bdata(plist->entry[i]), bdata(clist->entry[j]));
            if (!ownaccess(bdata(tmp), R_OK|X_OK))
            {
                compiler = bstrcpy(tmp);
            }
            bdestroy(tmp);
        }
    }
    bdestroy(path);
    bstrListDestroy(plist);
    bstrListDestroy(clist);
    return compiler;
}

static int compile_file(bstring compiler, bstring flags, bstring asmfile, bstring objfile)
{
    if (blength(compiler) == 0 || blength(asmfile) == 0)
        return -1;
    char buf[1024];
    int exitcode = 0;
    bstring bstdout = bfromcstr("");


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
        if (blength(bstdout) > 0)
        {
            fprintf(stderr, "%s\n", bdata(bstdout));
        }
        exitcode = pclose(fp);
        fprintf(stderr, "CMD %s\n", bdata(cmd));
    }
    else
    {
        exitcode = errno;
        fprintf(stderr, "CMD %s\n", bdata(cmd));
    }
    bdestroy(cmd);
    bdestroy(bstdout);

    return exitcode;
}


static int open_function(bstring location, TestCase *testcase)
{
    void* handle;
    char *error;
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


int dynbench_test(bstring testname)
{
    int exist = 0;
    char* home = getenv("HOME");
    char pwd[100];
    bstring path;
    if (!home)
    {
        fprintf(stderr, "Failed to get $HOME from environment\n");
        return exist;
    }
    if (getcwd(pwd, 100) == NULL)
    {
        fprintf(stderr, "Failed to get current working directory\n");
        return exist;
    }
    path = bformat("%s/.likwid/bench/%s/%s.ptt", home, ARCHNAME, bdata(testname));
    if (!access(bdata(path), R_OK))
    {
        exist = 1;
    }
    bdestroy(path);
    path = bformat("%s/%s.ptt", pwd, bdata(testname));
    if (!access(bdata(path), R_OK))
    {
        exist = 1;
    }
    bdestroy(path);
    return exist;
}

int dynbench_load(bstring testname, TestCase **testcase, char* tmpfolder, char *compilers, char* compileflags)
{
    int err = -1;
    TestCase *test = NULL;
    char* home = getenv("HOME");
    char pwd[100];
    bstring folder = NULL;
    if (!home)
    {
        fprintf(stderr, "Failed to get $HOME from environment\n");
        return err;
    }
    if (getcwd(pwd, 100) == NULL)
    {
        fprintf(stderr, "Failed to get current working directory\n");
        return err;
    }

    bstring pttfile = bformat("%s/%s.ptt", pwd, bdata(testname));
    if (access(bdata(pttfile), R_OK))
    {
        bdestroy(pttfile);
        pttfile = bformat("%s/.likwid/bench/%s/%s.ptt", home, ARCHNAME, bdata(testname));
        if (access(bdata(pttfile), R_OK))
        {
            fprintf(stderr, "Cannot open ptt file %s.ptt in CWD or %s/.likwid/bench/%s\n", bdata(testname), home, ARCHNAME);
            bdestroy(pttfile);
            return err;
        }
    }

    struct bstrList* code = analyse_ptt(pttfile, &test);
    if (code && test)
    {
        test->dlhandle = NULL;
        test->kernel = NULL;
        test->name = malloc((blength(testname)+2) * sizeof(char));
        if (test->name)
        {
            int ret = snprintf(test->name, blength(testname)+1, "%s", bdata(testname));
            if (ret > 0)
            {
                test->name[ret] = '\0';
            }
            if (tmpfolder && compilers)
            {
                pid_t pid = getpid();
                bstring buildfolder = bformat("%s/%ld", tmpfolder, pid);
                if (mkdir(bdata(buildfolder), 0700) == 0)
                {
                    int asm_written = 0;
                    bstring asmfile = bformat("%s/%s.S", bdata(buildfolder), bdata(testname));

                    struct bstrList* asmb = parse_asm(test, code);
                    if (asmb)
                    {
                        prepare_code(asmb);
                        if (write_asm(asmfile, asmb) != 0)
                        {
                            fprintf(stderr, "Failed to write assembly to file %s\n", bdata(asmfile));
                        }
                        else
                        {
                            asm_written = 1;
                        }
                        bstrListDestroy(asmb);
                    }
                    else
                    {
                        fprintf(stderr, "Cannot parse assembly\n");
                    }

                    bstring candidates = bfromcstr(compilers);
                    bstring compiler = get_compiler(candidates);
                    if (asm_written && compiler)
                    {
                        int cret = 0;
                        bstring cflags;
                        if (compileflags)
                        {
                            cflags = bfromcstr(compileflags);
                        }
                        else
                        {
                            cflags = bfromcstr("");
                        }
                        bstring objfile = bformat("%s/%s.o", bdata(buildfolder), bdata(testname));
                        cret = compile_file(compiler, cflags, asmfile, objfile);
                        if (cret == 0)
                        {
                            cret = open_function(objfile, test);
                            if (cret == 0)
                            {
                                err = 0;
                                *testcase = test;
                            }
                            else
                            {
                                fprintf(stderr, "Cannot load function %s from %s\n", bdata(testname), bdata(objfile));
                                err = cret;
                            }
                        }
                        else
                        {
                            fprintf(stderr, "Cannot compile file %s to %s\n", bdata(asmfile), bdata(objfile));
                            err = cret;
                        }
                        bdestroy(cflags);
                        bdestroy(objfile);
                    }
                    else
                    {
                        fprintf(stderr, "Cannot find any compiler %s\n", bdata(buildfolder));
                        err = -1;
                    }
                    bdestroy(candidates);
                    bdestroy(compiler);
                    bdestroy(asmfile);


                }
                else
                {
                    fprintf(stderr, "Cannot create temporary directory %s\n", bdata(buildfolder));
                    err = errno;
                }
                bdestroy(buildfolder);
            }
            else
            {
                err = 0;
                *testcase = test;
            }
        }
        else
        {
            fprintf(stderr, "Failed to allocate space for the testname\n");
            err = -ENOMEM;
        }
        bstrListDestroy(code);

    }
    else
    {
        fprintf(stderr, "Cannot read ptt file %s\n", bdata(pttfile));
        err = -EPERM;
    }

    bdestroy(pttfile);

    return err;
}

int dynbench_close(TestCase* testcase, char* tmpfolder)
{
    if (testcase)
    {
        if (testcase->dlhandle)
        {
            dlclose(testcase->dlhandle);
            testcase->dlhandle = NULL;
            testcase->kernel = NULL;
        }
        if (tmpfolder)
        {
            pid_t pid = getpid();

            bstring buildfolder = bformat("%s/%ld", tmpfolder, pid);
            bstring asmfile = bformat("%s/%s.S", bdata(buildfolder), testcase->name);
            bstring objfile = bformat("%s/%s.o", bdata(buildfolder), testcase->name);

            if (!access(bdata(asmfile), R_OK)) unlink(bdata(asmfile));
            if (!access(bdata(objfile), R_OK)) unlink(bdata(objfile));
            if (!access(bdata(buildfolder), R_OK)) rmdir(bdata(buildfolder));

            bdestroy(asmfile);
            bdestroy(objfile);
            bdestroy(buildfolder);
        }
        free(testcase->name);
        testcase->name = NULL;
        free(testcase->desc);
        testcase->desc = NULL;
        free(testcase);
        testcase = NULL;
    }
    return 0;
}

int dynbench_asm(bstring testname, char* tmpfolder, bstring outfile)
{
    if (blength(testname) > 0 && tmpfolder && blength(outfile) > 0)
    {
        int ret = 0;
        bstring asmfile = bformat("%s/%ld/%s.S", tmpfolder, getpid(), bdata(testname));
        char buf[1024];
        FILE* infp = fopen(bdata(asmfile), "r");
        if (infp)
        {
            FILE* outfp = fopen(bdata(outfile), "w");
            if (outfp)
            {
                for (;;)
                {
                    ret = fread(buf, sizeof(char), sizeof(buf), infp);
                    if (ret <= 0)
                    {
                        break;
                    }
                    ret = fwrite(buf, sizeof(char), ret, outfp);
                }
                fclose(outfp);
            }
            fclose(infp);
        }
    }
}
