/*
 * =======================================================================================
 *
 *      Filename:  strUtil.c
 *
 *      Description:  Utility string routines building upon bstrlib
 *
 *      Version:   4.3.3
 *      Released:  09.11.2018
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com.
 *      Project:  likwid
 *
 *      Copyright (C) 2018 RRZE, University Erlangen-Nuremberg
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

/* #####   HEADER FILE INCLUDES   ######################################### */

#include <strUtil.h>
#include <math.h>
#include <likwid.h>
#include <allocator.h>

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE  ################## */

static int
str2int(const char* str)
{
    char* endptr;
    errno = 0;
    unsigned long val;
    val = strtoul(str, &endptr, 10);

    if ((errno == ERANGE && val == LONG_MAX)
        || (errno != 0 && val == 0))
    {
        fprintf(stderr, "Value in string %s out of range\n", str);
        return -EINVAL;
    }

    if (endptr == str)
    {
        fprintf(stderr, "No digits were found in %s\n", str);
        return -EINVAL;
    }

    return (int) val;
}

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

uint64_t
bstr_to_doubleSize(const_bstring str, DataType type)
{
    int ret;
    bstring unit = bmidstr(str, blength(str)-2, 2);
    bstring single_unit = bmidstr(str, blength(str)-1, 1);
    bstring sizeStr = bmidstr(str, 0, blength(str)-2);
    bstring single_sizeStr = bmidstr(str, 0, blength(str)-1);
    uint64_t sizeU = 0;
    uint64_t single_sizeU = 0;
    uint64_t junk = 0;
    uint64_t bytesize = 0;
    if (blength(sizeStr) == 0)
    {
        return 0;
    }
    ret = str2int(bdata(sizeStr));
    if (ret >= 0)
    {
        sizeU = str2int(bdata(sizeStr));
    }
    else
    {
        return 0;
    }
    ret = str2int(bdata(single_sizeStr));
    if (ret >= 0)
    {
        single_sizeU = str2int(bdata(single_sizeStr));
    }
    else
    {
        return 0;
    }

    bytesize = allocator_dataTypeLength(type);

    if ((biseqcstr(unit, "kB"))||(biseqcstr(unit, "KB")))
    {
        junk = (sizeU *1000)/bytesize;
    }
    else if (biseqcstr(unit, "MB"))
    {
        junk = (sizeU *1000000)/bytesize;
    }
    else if (biseqcstr(unit, "GB"))
    {
        junk = (sizeU *1000000000)/bytesize;
    }
    else if (biseqcstr(single_unit, "B"))
    {
        junk = (single_sizeU)/bytesize;
    }
    bdestroy(unit);
    bdestroy(single_unit);
    bdestroy(sizeStr);
    bdestroy(single_sizeStr);
    return junk;
}

bstring
parse_workgroup(Workgroup* group, const_bstring str, DataType type)
{
    CpuTopology_t topo;
    struct bstrList* tokens;
    bstring cpustr;
    int numThreads = 0;
    bstring domain;

    tokens = bsplit(str,':');
    if (tokens->qty == 2)
    {
        topo = get_cpuTopology();
        numThreads = topo->activeHWThreads;
        cpustr = bformat("E:%s:%d", bdata(tokens->entry[0]), numThreads );
    }
    else if (tokens->qty == 3)
    {
        cpustr = bformat("E:%s:%s", bdata(tokens->entry[0]), bdata(tokens->entry[2]));
        numThreads = str2int(bdata(tokens->entry[2]));
        if (numThreads < 0)
        {
            fprintf(stderr, "Cannot convert %s to integer\n", bdata(tokens->entry[2]));
            bstrListDestroy(tokens);
            return NULL;
        }
    }
    else if (tokens->qty == 5)
    {
        cpustr = bformat("E:%s:%s:%s:%s", bdata(tokens->entry[0]),
                                          bdata(tokens->entry[2]),
                                          bdata(tokens->entry[3]),
                                          bdata(tokens->entry[4]));
        numThreads = str2int(bdata(tokens->entry[2]));
        if (numThreads < 0)
        {
            fprintf(stderr, "Cannot convert %s to integer\n", bdata(tokens->entry[2]));
            bstrListDestroy(tokens);
            return NULL;
        }
    }
    else
    {
        fprintf(stderr, "Misformated workgroup string\n");
        bstrListDestroy(tokens);
        return NULL;
    }

    group->size = bstr_to_doubleSize(tokens->entry[1], type);
    if (group->size == 0)
    {
        fprintf(stderr, "Stream size cannot be read, should look like <domain>:<size>\n");
        bstrListDestroy(tokens);
        return NULL;
    }
    group->processorIds = (int*) malloc(numThreads * sizeof(int));
    if (group->processorIds == NULL)
    {
        fprintf(stderr, "No more memory to allocate list of processors\n");
        bstrListDestroy(tokens);
        return NULL;
    }
    group->numberOfThreads = numThreads;
    if (cpustr_to_cpulist(bdata(cpustr),group->processorIds, numThreads) < 0 )
    {
        free(group->processorIds);
        bstrListDestroy(tokens);
        return NULL;
    }
    domain = bstrcpy(tokens->entry[0]);
    bdestroy(cpustr);
    bstrListDestroy(tokens);
    return domain;
}

int
parse_streams(Workgroup* group, const_bstring str, int numberOfStreams)
{
    struct bstrList* tokens;
    struct bstrList* subtokens;
    tokens = bsplit(str,',');

    if (tokens->qty < numberOfStreams)
    {
        fprintf(stderr, "Error: Testcase requires at least %d streams\n", numberOfStreams);
        bstrListDestroy(tokens);
        return -1;
    }

    group->streams = (Stream*) malloc(numberOfStreams * sizeof(Stream));
    if (group->streams == NULL)
    {
        bstrListDestroy(tokens);
        return -1;
    }
    for (int i=0; i<numberOfStreams; i++)
    {
        subtokens = bsplit(tokens->entry[i],':');
        if (subtokens->qty >= 2)
        {
            int index = str2int(bdata(subtokens->entry[0]));
            if ((index < 0) && (index >= numberOfStreams))
            {
                free(group->streams);
                bstrListDestroy(subtokens);
                bstrListDestroy(tokens);
                return -1;
            }
            group->streams[index].domain = bstrcpy(subtokens->entry[1]);
            group->streams[index].offset = 0;
            if (subtokens->qty == 3)
            {
                group->streams[index].offset = str2int(bdata(subtokens->entry[2]));
                if (group->streams[index].offset < 0)
                {
                free(group->streams);
                bstrListDestroy(subtokens);
                bstrListDestroy(tokens);
                return -1;
                }
            }
        }
        else
        {
            fprintf(stderr, "Error in parsing stream definition %s\n", bdata(tokens->entry[i]));
            bstrListDestroy(subtokens);
            bstrListDestroy(tokens);
            free(group->streams);
            return -1;
        }
        bstrListDestroy(subtokens);
    }

    bstrListDestroy(tokens);
    return 0;
}

int
bstr_to_workgroup(Workgroup* group, const_bstring str, DataType type, int numberOfStreams)
{
    int parseStreams = 0;
    struct bstrList* tokens;
    tokens = bsplit(str,'-');
    bstring domain;
    if (tokens->qty == 2)
    {
        domain = parse_workgroup(group, tokens->entry[0], type);
        if (domain == NULL)
        {
            bstrListDestroy(tokens);
            return 1;
        }
        parse_streams(group, tokens->entry[1], numberOfStreams);
        bdestroy(domain);
    }
    else if (tokens->qty == 1)
    {
        domain = parse_workgroup(group, tokens->entry[0], type);
        if (domain == NULL)
        {
            bstrListDestroy(tokens);
            return 1;
        }
        group->streams = (Stream*) malloc(numberOfStreams * sizeof(Stream));
        if (group->streams == NULL)
        {
            bstrListDestroy(tokens);
            return 1;
        }
        for (int i = 0; i< numberOfStreams; i++)
        {
            group->streams[i].domain = bstrcpy(domain);
            group->streams[i].offset = 0;
        }
        bdestroy(domain);
    }
    else
    {
        fprintf(stderr, "Error in parsing workgroup string %s\n", bdata(str));
        bstrListDestroy(tokens);
        return 1;
    }
    bstrListDestroy(tokens);
    group->size /= numberOfStreams;
    return 0;
}

void
workgroups_destroy(Workgroup** groupList, int numberOfGroups, int numberOfStreams)
{
    int i = 0, j = 0;
    if (groupList == NULL)
        return;
    if (*groupList == NULL)
        return;
    Workgroup* list = *groupList;
    for (i = 0; i < numberOfGroups; i++)
    {
        free(list[i].processorIds);
        for (j = 0; j < numberOfStreams; j++)
        {
            bdestroy(list[i].streams[j].domain);
        }
        free(list[i].streams);
    }
    free(list);
}

