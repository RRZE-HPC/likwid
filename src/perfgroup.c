/*
 * =======================================================================================
 *
 *      Filename:  perfgroup.c
 *
 *      Description:  Handler for performance groups and event sets
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com
 *                Thomas Gruber (tr), thomas.roehl@gmail.com
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

/* #####   HEADER FILE INCLUDES   ######################################### */

#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <dirent.h>
#include <pthread.h>

#include <error.h>
#include <perfgroup.h>
#include <topology.h>
#include <likwid.h>

#include <calculator.h>
#include <bstrlib.h>
#include <bstrlib_helper.h>



static int totalgroups = 0;

/* #####   FUNCTION DEFINITIONS  -  INTERNAL FUNCTIONS   ################## */

static inline void *realloc_buffer(void *ptrmem, size_t size) {
    void *ptr = realloc(ptrmem, size);
    if (!ptr)  {
        fprintf(stderr, "realloc(%p, %lu): errno=%d\n", ptrmem, size, errno);
        free (ptrmem);
    }
    if (!ptrmem)
    {
        memset(ptr, 0, size);
    }
    return ptr;
}

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

int
isdir(char* dirname)
{
    struct stat st;
    if (NULL == dirname) {
        return 0;
    }
    if (access(dirname, R_OK) != 0)
        return 0;
    stat(dirname, &st);
    return S_ISDIR(st.st_mode) ? 1 : 0;
}

void
perfgroup_returnGroups(int groups, char** groupnames, char** groupshort, char** grouplong)
{
    int i;
    int freegroups = (totalgroups < groups ? groups : totalgroups);
    for (i = 0; i <freegroups; i++)
    {
        free(groupnames[i]);
        groupnames[i] = NULL;
        if (i < groups)
        {
            if (groupshort[i] != NULL)
            {
                free(groupshort[i]);
                groupshort[i] = NULL;
            }
            if (grouplong[i] != NULL)
            {
                free(grouplong[i]);
                grouplong[i] = NULL;
            }
        }
    }
    if (groupnames != NULL)
    {
        free(groupnames);
        groupnames = NULL;
    }
    if (groupshort != NULL)
    {
        free(groupshort);
        groupshort = NULL;
    }
    if (grouplong != NULL)
    {
        free(grouplong);
        grouplong = NULL;
    }
}


int
perfgroup_getGroups(
        const char* grouppath,
        const char* architecture,
        char*** groupnames,
        char*** groupshort,
        char*** grouplong)
{
    int i = 0, j = 0, s = 0;
    int fsize = 0, hsize = 0;
    DIR *dp = NULL;
    FILE* fp = NULL;
    char buf[256] = { [0 ... 255] = '\0' };
    struct dirent *ep = NULL;
    *groupnames = NULL;
    *groupshort = NULL;
    *grouplong = NULL;
    int search_home = 0;
    bstring SHORT = bformat("SHORT");
    bstring LONG = bformat("LONG");
    bstring REQUIRE = bformat("REQUIRE_NOHT");
    char* Home = getenv("HOME");
    if (!Home) Home = "";

    int read_long = 0;
    if ((grouppath == NULL)||(architecture == NULL)||(groupnames == NULL)||(Home == NULL))
        return -EINVAL;

    char* fullpath = malloc((strlen(grouppath)+strlen(architecture)+50) * sizeof(char));
    if (fullpath == NULL)
    {
        bdestroy(SHORT);
        bdestroy(LONG);
        bdestroy(REQUIRE);
        return -ENOMEM;
    }
    char* homepath = malloc((strlen(Home)+strlen(architecture)+50) * sizeof(char));
    if (homepath == NULL)
    {
        free(fullpath);
        bdestroy(SHORT);
        bdestroy(LONG);
        bdestroy(REQUIRE);
        return -ENOMEM;
    }
    fsize = sprintf(fullpath, "%s/%s", grouppath, architecture);
    if (isdir(fullpath))
    {
        dp = opendir(fullpath);
        if (dp == NULL)
        {
            printf("Cannot open directory %s\n", fullpath);
            free(fullpath);
            free(homepath);
            bdestroy(SHORT);
            bdestroy(LONG);
            bdestroy(REQUIRE);
            return -EACCES;
        }
    }
    else
    {
        printf("Cannot access directory %s\n", fullpath);
        free(fullpath);
        free(homepath);
        bdestroy(SHORT);
        bdestroy(LONG);
        bdestroy(REQUIRE);
        return -EACCES;
    }
    i = 0;
    s = 0;
    while ((ep = readdir(dp)))
    {
        if (strncmp(&(ep->d_name[strlen(ep->d_name)-4]), ".txt", 4) == 0)
        {
            totalgroups++;
            if (strlen(ep->d_name)-4 > s)
                s = strlen(ep->d_name)-4;
        }
    }
    closedir(dp);
    hsize = sprintf(homepath, "%s/.likwid/groups/%s", Home, architecture);
    if (isdir(homepath))
    {
        search_home = 1;
        dp = opendir(homepath);
        if (dp == NULL)
        {
            search_home = 0;
        }
        if (search_home)
        {
            while ((ep = readdir(dp)))
            {
                if (strncmp(&(ep->d_name[strlen(ep->d_name)-4]), ".txt", 4) == 0)
                {
                    totalgroups++;
                    if (strlen(ep->d_name)-4 > s)
                        s = strlen(ep->d_name)-4;
                }
            }
            closedir(dp);
        }
    }
    *groupnames = malloc(totalgroups * sizeof(char**));
    if (*groupnames == NULL)
    {
        free(fullpath);
        free(homepath);
        bdestroy(SHORT);
        bdestroy(LONG);
        bdestroy(REQUIRE);
        return -ENOMEM;
    }
    memset(*groupnames, 0, totalgroups * sizeof(char**));
    *groupshort = malloc(totalgroups * sizeof(char**));
    if (*groupshort == NULL)
    {
        free(*groupnames);
        *groupnames = NULL;
        free(fullpath);
        free(homepath);
        bdestroy(SHORT);
        bdestroy(LONG);
        bdestroy(REQUIRE);
        return -ENOMEM;
    }
    memset(*groupshort, 0, totalgroups * sizeof(char**));
    *grouplong = malloc(totalgroups * sizeof(char**));
    if (*grouplong == NULL)
    {
        free(*groupnames);
        *groupnames = NULL;
        free(*groupshort);
        *groupshort = NULL;
        free(fullpath);
        free(homepath);
        bdestroy(SHORT);
        bdestroy(LONG);
        bdestroy(REQUIRE);
        return -ENOMEM;
    }
    memset(*grouplong, 0, totalgroups * sizeof(char**));
    for (j=0; j < totalgroups; j++)
    {
        (*groupnames)[j] = malloc((s+1) * sizeof(char));
        if ((*groupnames)[j] == NULL)
        {
            for (s=0; s<j; s++)
            {
                free((*groupnames)[s]);
            }
            free(*groupnames);
            *groupnames = NULL;
            free(*groupshort);
            *groupshort = NULL;
            free(*grouplong);
            *grouplong = NULL;
            free(fullpath);
            free(homepath);
            bdestroy(SHORT);
            bdestroy(LONG);
            bdestroy(REQUIRE);
            return -ENOMEM;
        }
    }
    dp = opendir(fullpath);
    i = 0;
    int skip_group = 0;

    while ((ep = readdir(dp)))
    {
        if (strncmp(&(ep->d_name[strlen(ep->d_name)-4]), ".txt", 4) == 0)
        {
            read_long = 0;
            bstring long_info = bfromcstr("");
            sprintf(&(fullpath[fsize]), "/%s", ep->d_name);
            if (!access(fullpath, R_OK))
            {
                (*grouplong)[i] = NULL;
                s = sprintf((*groupnames)[i], "%.*s", (int)(strlen(ep->d_name)-4), ep->d_name);
                (*groupnames)[i][s] = '\0';
                fp = fopen(fullpath,"r");

                while (fgets (buf, sizeof(buf), fp)) {
                    bstring bbuf = bfromcstr(buf);
                    btrimws(bbuf);
                    if ((blength(bbuf) == 0) || (buf[0] == '#'))
                    {
                        bdestroy(bbuf);
                        continue;
                    }
                    if (bstrncmp(bbuf, SHORT, 5) == 0)
                    {
                        struct bstrList * linelist = bsplit(bbuf, ' ');
                        bstring sinfo;
                        if (linelist->qty == 1)
                        {
                            fprintf(stderr,"Cannot read SHORT section in groupfile %s",fullpath);
                            bdestroy(bbuf);
                            bstrListDestroy(linelist);
                            continue;
                        }
                        s = 1;
                        for (j=s;j<linelist->qty; j++)
                        {
                            btrimws(linelist->entry[j]);
                            if (blength(linelist->entry[j]) == 0)
                                s += 1;
                            else
                                break;
                        }
                        btrimws(linelist->entry[s]);
                        sinfo = bformat("%s", bdata(linelist->entry[s]));
                        for (j=s+1;j<linelist->qty; j++)
                        {
                            btrimws(linelist->entry[j]);
                            bstring tmp = bformat(" %s", bdata(linelist->entry[j]));
                            bconcat(sinfo, tmp);
                            bdestroy(tmp);
                        }

                        (*groupshort)[i] = malloc((blength(sinfo)+1) * sizeof(char));
                        if ((*groupshort)[i] == NULL)
                        {
                            bdestroy(SHORT);
                            bdestroy(LONG);
                            bdestroy(bbuf);
                            bdestroy(sinfo);
                            bdestroy(REQUIRE);
                            free(homepath);
                            free(fullpath);
                            bdestroy(long_info);
                            bstrListDestroy(linelist);
                            perfgroup_returnGroups(i, *groupnames, *groupshort, *grouplong);
                            return -ENOMEM;
                        }
                        s = sprintf((*groupshort)[i], "%s", bdata(sinfo));
                        (*groupshort)[i][s] = '\0';
                        bstrListDestroy(linelist);
                        bdestroy(sinfo);
                    }
                    else if (bstrncmp(bbuf, REQUIRE, blength(REQUIRE)) == 0)
                    {
                        if (cpuid_topology.numThreadsPerCore > 1)
                        {
                            skip_group = 1;
                        }
                    }
                    else if (bstrncmp(bbuf, LONG, 4) == 0)
                    {
                        read_long = 1;
                    }
                    else if ((read_long == 1) && (bstrncmp(bbuf, LONG, 4) != 0))
                    {
                        bstring tmp = bfromcstr(buf);
                        bconcat(long_info, tmp);
                        bdestroy(tmp);
                    }
                    bdestroy(bbuf);
                }
                if (read_long)
                {

                    (*grouplong)[i] = malloc((blength(long_info) + 1) * sizeof(char) );
                    if ((*grouplong)[i] != NULL)
                    {
                        j = sprintf((*grouplong)[i], "%s", bdata(long_info));
                        (*grouplong)[i][j] = '\0';
                    }
                }
                fclose(fp);
                if (skip_group)
                {
                    if ((*grouplong)[i] != NULL)
                    {
                        free((*grouplong)[i]);
                        (*grouplong)[i] = NULL;
                    }
                    if ((*groupshort)[i] != NULL)
                    {
                        free((*groupshort)[i]);
                        (*groupshort)[i] = NULL;
                    }
                    (*groupnames)[i][0] = '\0';
                    bdestroy(long_info);
                    goto skip_cur_def_group;
                }
                i++;
            }
            bdestroy(long_info);
        }
skip_cur_def_group:
        skip_group = 0;
    }
    closedir(dp);
    if (!search_home)
    {
        if (i==0)
            perfgroup_returnGroups(totalgroups, *groupnames, *groupshort, *grouplong);
        /*else if (i < totalgroups)
        {
            for (s=i;s<totalgroups;s++)
            {
                (*grouplong)[i] = NULL;
                (*groupshort)[i] = NULL;
            }
        }*/
        free(homepath);
        free(fullpath);
        bdestroy(SHORT);
        bdestroy(REQUIRE);
        bdestroy(LONG);
        return i;
    }
    else
    {
        dp = opendir(homepath);
        while ((ep = readdir(dp)))
        {
            if (strncmp(&(ep->d_name[strlen(ep->d_name)-4]), ".txt", 4) == 0)
            {
                read_long = 0;
                bstring long_info = bfromcstr("");;
                sprintf(&(homepath[hsize]), "/%s", ep->d_name);
                if (!access(homepath, R_OK))
                {
                    (*grouplong)[i] = NULL;
                    s = sprintf((*groupnames)[i], "%.*s", (int)(strlen(ep->d_name)-4), ep->d_name);
                    (*groupnames)[i][s] = '\0';
                    fp = fopen(homepath,"r");
                    while (fgets (buf, sizeof(buf), fp)) {
                        bstring bbuf = bfromcstr(buf);
                        btrimws(bbuf);
                        if ((blength(bbuf) == 0) || (buf[0] == '#'))
                        {
                            bdestroy(bbuf);
                            continue;
                        }
                        if (bstrncmp(bbuf, SHORT, 5) == 0)
                        {
                            struct bstrList * linelist = bsplit(bbuf, ' ');
                            bstring sinfo;
                            if (linelist->qty == 1)
                            {
                                fprintf(stderr,"Cannot read SHORT section in groupfile %s",fullpath);
                                bdestroy(bbuf);
                                bstrListDestroy(linelist);
                                continue;
                            }
                            s = 1;
                            for (j=s;j<linelist->qty; j++)
                            {
                                btrimws(linelist->entry[j]);
                                if (blength(linelist->entry[j]) == 0)
                                    s += 1;
                                else
                                    break;
                            }
                            btrimws(linelist->entry[s]);
                            sinfo = bformat("%s", bdata(linelist->entry[s]));
                            for (j=s+1;j<linelist->qty; j++)
                            {
                                btrimws(linelist->entry[j]);
                                bstring tmp = bformat(" %s", bdata(linelist->entry[j]));
                                bconcat(sinfo, tmp);
                                bdestroy(tmp);
                            }
                            (*groupshort)[i] = malloc((blength(sinfo)+1) * sizeof(char));
                            if ((*groupshort)[i] == NULL)
                            {
                                bdestroy(SHORT);
                                bdestroy(LONG);
                                bdestroy(REQUIRE);
                                bdestroy(bbuf);
                                bdestroy(sinfo);
                                free(homepath);
                                free(fullpath);
                                bstrListDestroy(linelist);
                                bdestroy(long_info);
                                perfgroup_returnGroups(i, *groupnames, *groupshort, *grouplong);
                                return -ENOMEM;
                            }
                            s = sprintf((*groupshort)[i], "%s", bdata(sinfo));
                            (*groupshort)[i][s] = '\0';
                            bstrListDestroy(linelist);
                            bdestroy(sinfo);
                        }
                        else if (bstrncmp(bbuf, REQUIRE, blength(REQUIRE)) == 0)
                        {
                            if (cpuid_topology.numThreadsPerCore > 1)
                            {
                                skip_group = 1;
                            }
                        }
                        else if (bstrncmp(bbuf, LONG, 4) == 0)
                        {
                            read_long = 1;
                        }
                        else if ((read_long == 1) && (bstrncmp(bbuf, LONG, 4) != 0))
                        {
                            bstring tmp = bfromcstr(buf);
                            bconcat(long_info, tmp);
                            bdestroy(tmp);
                        }
                        bdestroy(bbuf);
                    }
                    if (read_long)
                    {
                        (*grouplong)[i] = malloc((blength(long_info) + 1) * sizeof(char) );
                        if ((*grouplong)[i] != NULL)
                        {
                            j = sprintf((*grouplong)[i], "%s", bdata(long_info));
                            (*grouplong)[i][j] = '\0';
                        }
                    }
                    fclose(fp);
                    if (skip_group)
                    {
                        if ((*groupshort)[i])
                        {
                            free((*groupshort)[i]);
                            (*groupshort)[i] = NULL;
                        }
                        if ((*grouplong)[i])
                        {
                            free((*grouplong)[i]);
                            (*grouplong)[i] = NULL;
                        }
                        bdestroy(long_info);
                        goto skip_cur_home_group;
                    }
                    i++;
                }
                bdestroy(long_info);
            }
skip_cur_home_group:
        skip_group = 0;
        }
        closedir(dp);
    }
    if (i==0)
        perfgroup_returnGroups(totalgroups, *groupnames, *groupshort, *grouplong);
/*    else if (i < totalgroups)
    {
        for (s=i;s<totalgroups;s++)
        {
            printf("Setting NULL for group %d\n", s);
            (*groupnames)[i] = NULL;
            (*grouplong)[i] = NULL;
            (*groupshort)[i] = NULL;
        }
    }*/
    bdestroy(SHORT);
    bdestroy(LONG);
    bdestroy(REQUIRE);
    free(fullpath);
    free(homepath);
    return i;
}



int perfgroup_customGroup(const char* eventStr, GroupInfo* ginfo)
{
    int i, j;
    int err = 0;
    char delim = ',';
    bstring edelim = bformat(":");
    int has_fix0 = 0;
    int has_fix1 = 0;
    int has_fix2 = 0;
    int has_fix3 = 0;
    int gpu_events = 0;
    ginfo->shortinfo = NULL;
    ginfo->nevents = 0;
    ginfo->events = NULL;
    ginfo->counters = NULL;
    ginfo->nmetrics = 0;
    ginfo->metricformulas = NULL;
    ginfo->metricnames = NULL;
    ginfo->longinfo = NULL;
    bstring eventBstr;
    struct bstrList * eventList;
#if defined(__i386__) || defined(__i486__) || defined(__i586__) || defined(__i686__) || defined(__x86_64)
    bstring fix0 = bformat("FIXC0");
    bstring fix1 = bformat("FIXC1");
    bstring fix2 = bformat("FIXC2");
    bstring fix3 = bformat("FIXC3");
#endif
#ifdef _ARCH_PPC
    bstring fix0 = bformat("PMC4");
    bstring fix1 = bformat("PMC5");
#endif
#if defined(__ARM_ARCH_8A) || defined(__ARM_ARCH_7A__)
    bstring fix0 = bformat("012345");
    bstring fix1 = bformat("012345");
#endif
#ifdef LIKWID_WITH_NVMON
    bstring gpu = bformat("GPU");
#endif
    DEBUG_PRINT(DEBUGLEV_INFO, Creating custom group for event string %s, eventStr);
    ginfo->shortinfo = malloc(7 * sizeof(char));
    if (ginfo->shortinfo == NULL)
    {
        err = -ENOMEM;
        goto cleanup;
    }
    sprintf(ginfo->shortinfo, "%s", "Custom");
    ginfo->longinfo = malloc(7 * sizeof(char));
    if (ginfo->longinfo == NULL)
    {
        err = -ENOMEM;
        goto cleanup;
    }
    sprintf(ginfo->longinfo, "%s", "Custom");
    ginfo->groupname = malloc(7 * sizeof(char));
    if (ginfo->groupname == NULL)
    {
        err = -ENOMEM;
        goto cleanup;
    }
    sprintf(ginfo->groupname, "%s", "Custom");
    eventBstr = bfromcstr(eventStr);
    eventList = bsplit(eventBstr, delim);
    ginfo->nevents = eventList->qty;
    if (cpuid_info.isIntel || cpuid_info.family == PPC_FAMILY)
    {
        if (binstr(eventBstr, 0, fix0) > 0)
        {
            has_fix0 = 1;
        }
        else
        {
            ginfo->nevents++;
        }
        if (binstr(eventBstr, 0, fix1) > 0)
        {
            has_fix1 = 1;
        }
        else
        {
            ginfo->nevents++;
        }
#if defined(__i386__) || defined(__i486__) || defined(__i586__) || defined(__i686__) || defined(__x86_64)
        if (binstr(eventBstr, 0, fix2) > 0)
        {
            has_fix2 = 1;
        }
        else
        {
            ginfo->nevents++;
        }
        if (binstr(eventBstr, 0, fix3) > 0)
        {
            has_fix3 = 1;
        }
        else
        {
            ginfo->nevents++;
        }
#endif
    }
    bdestroy(eventBstr);

    ginfo->events = malloc(ginfo->nevents * sizeof(char*));
    if (ginfo->events == NULL)
    {
        err = -ENOMEM;
        bstrListDestroy(eventList);
        goto cleanup;
    }
    ginfo->counters = malloc(ginfo->nevents * sizeof(char*));
    if (ginfo->counters == NULL)
    {
        err = -ENOMEM;
        bstrListDestroy(eventList);
        goto cleanup;
    }
    for (i = 0; i< eventList->qty; i++)
    {
        int s;
        struct bstrList * elist;
        elist = bsplit(eventList->entry[i], ':');
        ginfo->events[i] = malloc((blength(elist->entry[0]) + 1) * sizeof(char));
        if (ginfo->events[i] == NULL)
        {
            bstrListDestroy(elist);
            err = -ENOMEM;
            goto cleanup;
        }
        bstring ctr = bstrcpy(elist->entry[1]);
        if (elist->qty > 2)
        {
            for (j = 2; j < elist->qty; j++)
            {
                bconcat(ctr, edelim);
                bconcat(ctr, elist->entry[j]);
            }
        }
        ginfo->counters[i] = malloc((blength(ctr) + 1) * sizeof(char));
        if (ginfo->counters[i] == NULL)
        {
            bstrListDestroy(elist);
            bdestroy(ctr);
            err = -ENOMEM;
            goto cleanup;
        }
        sprintf(ginfo->events[i], "%s", bdata(elist->entry[0]));
        snprintf(ginfo->counters[i], blength(ctr)+1, "%s", bdata(ctr));
#ifdef LIKWID_WITH_NVMON
        if (binstr(elist->entry[1], 0, gpu) != BSTR_ERR)
        {
            gpu_events++;
        }
#endif
        bdestroy(ctr);
        bstrListDestroy(elist);
    }
    i = eventList->qty;
#if defined(__i386__) || defined(__i486__) || defined(__i586__) || defined(__i686__) || defined(__x86_64)
    if (cpuid_info.isIntel && i != gpu_events)
    {
        if ((!has_fix0) && cpuid_info.perf_num_fixed_ctr > 0)
        {
            ginfo->events[i] = malloc(18 * sizeof(char));
            ginfo->counters[i] = malloc(6 * sizeof(char));
            sprintf(ginfo->events[i], "%s", "INSTR_RETIRED_ANY");
            sprintf(ginfo->counters[i], "%s", "FIXC0");
            i++;
        }
        if ((!has_fix1) && cpuid_info.perf_num_fixed_ctr > 1)
        {
            ginfo->events[i] = malloc(22 * sizeof(char));
            ginfo->counters[i] = malloc(6 * sizeof(char));
            sprintf(ginfo->events[i], "%s", "CPU_CLK_UNHALTED_CORE");
            sprintf(ginfo->counters[i], "%s", "FIXC1");
            i++;
        }
        if ((!has_fix2) && cpuid_info.perf_num_fixed_ctr > 2)
        {
            ginfo->events[i] = malloc(21 * sizeof(char));
            ginfo->counters[i] = malloc(6 * sizeof(char));
            sprintf(ginfo->events[i], "%s", "CPU_CLK_UNHALTED_REF");
            sprintf(ginfo->counters[i], "%s", "FIXC2");
            i++;
        }
        if ((!has_fix3) && cpuid_info.perf_num_fixed_ctr > 3 && 
            (cpuid_info.model == ICELAKE1 || cpuid_info.model == ICELAKE2 || cpuid_info.model == ICELAKEX1 || cpuid_info.model == ICELAKEX2 || cpuid_info.model == ROCKETLAKE || cpuid_info.model == SAPPHIRERAPIDS))
        {
            ginfo->events[i] = malloc(14 * sizeof(char));
            ginfo->counters[i] = malloc(6 * sizeof(char));
            sprintf(ginfo->events[i], "%s", "TOPDOWN_SLOTS");
            sprintf(ginfo->counters[i], "%s", "FIXC3");
            i++;
        }
    }
    ginfo->nevents = i;
#endif
#ifdef _ARCH_PPC
    if (!has_fix0)
    {
        ginfo->events[i] = malloc(18 * sizeof(char));
        ginfo->counters[i] = malloc(6 * sizeof(char));
        sprintf(ginfo->events[i], "%s", "PM_RUN_INST_CMPL");
        sprintf(ginfo->counters[i], "%s", "PMC4");
        i++;
    }
    if (!has_fix1)
    {
        ginfo->events[i] = malloc(22 * sizeof(char));
        ginfo->counters[i] = malloc(6 * sizeof(char));
        sprintf(ginfo->events[i], "%s", "PM_RUN_CYC");
        sprintf(ginfo->counters[i], "%s", "PMC5");
        i++;
    }
#endif
    bstrListDestroy(eventList);
#if defined(__i386__) || defined(__i486__) || defined(__i586__) || defined(__i686__) || defined(__x86_64)
    bdestroy(fix0);
    bdestroy(fix1);
    bdestroy(fix2);
#endif
    bdestroy(edelim);
    return 0;
cleanup:
    bstrListDestroy(eventList);
#if defined(__i386__) || defined(__i486__) || defined(__i586__) || defined(__i686__) || defined(__x86_64)
    bdestroy(fix0);
    bdestroy(fix1);
    bdestroy(fix2);
#endif
#ifdef LIKWID_WITH_NVMON
    bdestroy(gpu);
#endif
    bdestroy(edelim);
    if (ginfo->shortinfo != NULL)
        free(ginfo->shortinfo);
    if (ginfo->events != NULL)
        free(ginfo->events);
    if (ginfo->counters != NULL)
        free(ginfo->counters);
    return err;
}

int
perfgroup_readGroup(
        const char* grouppath,
        const char* architecture,
        const char* groupname,
        GroupInfo* ginfo)
{
    FILE* fp;
    int i, s, e, err = 0;
    char buf[1024];
    GroupFileSections sec = GROUP_NONE;
    bstring REQUIRE = bformat("REQUIRE_NOHT");
    char* Home = getenv("HOME");
    if (!Home) Home = "";
    if ((grouppath == NULL)||(architecture == NULL)||(groupname == NULL)||(ginfo == NULL)||(Home == NULL))
        return -EINVAL;

    bstring fullpath = bformat("%s/%s/%s.txt", grouppath,architecture, groupname);
    bstring homepath = bformat("%s/.likwid/groups/%s/%s.txt", Home,architecture, groupname);

    if (access(bdata(fullpath), R_OK))
    {
        DEBUG_PRINT(DEBUGLEV_INFO, Cannot read group file %s. Trying %s, bdata(fullpath), bdata(homepath));
        if (access(bdata(homepath), R_OK))
        {
            ERROR_PRINT(Cannot read group file %s.txt. Searched in %s and %s, groupname, bdata(fullpath), bdata(homepath));
            bdestroy(REQUIRE);
            bdestroy(fullpath);
            bdestroy(homepath);
            return -EACCES;
        }
        else
        {
            bdestroy(fullpath);
            fullpath = bstrcpy(homepath);
        }
    }

    DEBUG_PRINT(DEBUGLEV_INFO, Reading group %s from %s, groupname, bdata(fullpath));

    ginfo->shortinfo = NULL;
    ginfo->nevents = 0;
    ginfo->events = NULL;
    ginfo->counters = NULL;
    ginfo->nmetrics = 0;
    ginfo->metricformulas = NULL;
    ginfo->metricnames = NULL;
    ginfo->longinfo = NULL;
    ginfo->groupname = (char*)malloc((strlen(groupname)+10)*sizeof(char));
    if (ginfo->groupname == NULL)
    {
        err = -ENOMEM;
        goto cleanup;
    }
    //strncpy(ginfo->groupname, groupname, strlen(groupname));
    i = sprintf(ginfo->groupname, "%s", groupname);
    ginfo->groupname[i] = '\0';

    fp = fopen(bdata(fullpath), "r");
    if (fp == NULL)
    {
        free(ginfo->groupname);
        bdestroy(fullpath);
        bdestroy(homepath);
        return -EACCES;
    }
    struct bstrList * linelist;
    while (fgets (buf, 1023*sizeof(char), fp)) {
        if ((strlen(buf) == 0) || (buf[0] == '#'))
            continue;

        if (strncmp(groupFileSectionNames[GROUP_SHORT], buf, strlen(groupFileSectionNames[GROUP_SHORT])) == 0)
        {
            sec = GROUP_SHORT;
            for (i=strlen(groupFileSectionNames[GROUP_SHORT]); i < strlen(buf); i++)
            {
                if (buf[i] == ' ')
                    continue;
                break;
            }
            ginfo->shortinfo = malloc(strlen(&(buf[i])) * sizeof(char));
            sprintf(ginfo->shortinfo, "%.*s", (int)strlen(&(buf[i]))-1, &(buf[i]));
            continue;
        }
        else if (strncmp(bdata(REQUIRE), buf, blength(REQUIRE)) == 0)
        {
            if (cpuid_topology.numThreadsPerCore > 1)
            {
                err = -ENODEV;
                goto cleanup;
            }
            continue;
        }
        else if (strncmp(groupFileSectionNames[GROUP_EVENTSET], buf, strlen(groupFileSectionNames[GROUP_EVENTSET])) == 0)
        {
            sec = GROUP_EVENTSET;
            continue;
        }
        else if (strncmp(groupFileSectionNames[GROUP_METRICS], buf, strlen(groupFileSectionNames[GROUP_METRICS])) == 0)
        {
            sec = GROUP_METRICS;
            continue;
        }
        else if (strncmp(groupFileSectionNames[GROUP_LONG], buf, strlen(groupFileSectionNames[GROUP_LONG])) == 0)
        {
            sec = GROUP_LONG;
            continue;
        }
        else if (strncmp(groupFileSectionNames[GROUP_LUA], buf, strlen(groupFileSectionNames[GROUP_LUA])) == 0)
        {
            sec = GROUP_LUA;
            continue;
        }
        if (sec == GROUP_NONE)
            continue;
        if (sec == GROUP_EVENTSET)
        {
            i = 0;
            bstring bbuf = bfromcstr(buf);
            btrimws(bbuf);
            if (blength(bbuf) == 0)
            {
                bdestroy(bbuf);
                sec = GROUP_NONE;
                continue;
            }
            linelist = bsplit(bbuf, ' ');
            for (i=0; i<linelist->qty; i++)
                btrimws(linelist->entry[i]);
            bdestroy(bbuf);
            bbuf = bstrcpy(linelist->entry[0]);
            for (i=1; i<linelist->qty; i++)
            {
                if (blength(linelist->entry[i]) > 0)
                {
                    bstring tmp = bformat(" %s", bdata(linelist->entry[i]));
                    bconcat(bbuf, tmp);
                    bdestroy(tmp);
                }
            }
            if (ginfo->events == NULL)
            {
                ginfo->events = (char**)malloc(sizeof(char*));
                if (ginfo->events == NULL)
                {
                    err = -ENOMEM;
                    bdestroy(bbuf);
                    goto cleanup;
                }
            }
            else
            {
                char** tmp = NULL;
                tmp = realloc(ginfo->events, (ginfo->nevents + 1) * sizeof(char*));
                if (tmp == NULL)
                {
                    free(ginfo->events);
                    bdestroy(bbuf);
                    err = -ENOMEM;
                    goto cleanup;
                }
                else
                {
                    ginfo->events = tmp;
                    tmp = NULL;
                }
            }
            if (ginfo->counters == NULL)
            {
                ginfo->counters = (char**)malloc(sizeof(char*));
                if (ginfo->counters == NULL)
                {
                    err = -ENOMEM;
                    bdestroy(bbuf);
                    goto cleanup;
                }
            }
            else
            {
                char** tmp = NULL;
                tmp = realloc(ginfo->counters, (ginfo->nevents + 1) * sizeof(char*));
                if (tmp == NULL)
                {
                    free(ginfo->counters);
                    bdestroy(bbuf);
                    err = -ENOMEM;
                    goto cleanup;
                }
                else
                {
                    ginfo->counters = tmp;
                    tmp = NULL;
                }
            }
            bstrListDestroy(linelist);
            linelist = bsplit(bbuf, ' ');
            bdestroy(bbuf);
            for (i=0; i<linelist->qty; i++)
                btrimws(linelist->entry[i]);
            ginfo->counters[ginfo->nevents] = malloc((blength(linelist->entry[0])+1) * sizeof(char));
            if (ginfo->counters[ginfo->nevents] == NULL)
            {
                err = -ENOMEM;
                goto cleanup;
            }
            ginfo->events[ginfo->nevents] = malloc((blength(linelist->entry[1])+1) * sizeof(char));
            if (ginfo->events[ginfo->nevents] == NULL)
            {
                err = -ENOMEM;
                goto cleanup;
            }
            sprintf(ginfo->counters[ginfo->nevents], "%s", bdata(linelist->entry[0]));
            sprintf(ginfo->events[ginfo->nevents], "%s", bdata(linelist->entry[1]));
            ginfo->nevents++;
            bstrListDestroy(linelist);
            continue;
        }
        else if (sec == GROUP_METRICS)
        {
            i = 0;
            bstring bbuf = bfromcstr(buf);
            btrimws(bbuf);
            if (blength(bbuf) == 0)
            {
                bdestroy(bbuf);
                sec = GROUP_NONE;
                continue;
            }
            linelist = bsplit(bbuf, ' ');
            for (i=0; i<linelist->qty; i++)
                btrimws(linelist->entry[i]);
            bdestroy(bbuf);
            bbuf = bstrcpy(linelist->entry[0]);
            for (i=1; i<linelist->qty; i++)
            {
                if (blength(linelist->entry[i]) > 0)
                {
                    bstring tmp = bformat(" %s", bdata(linelist->entry[i]));
                    bconcat(bbuf, tmp);
                    bdestroy(tmp);
                }
            }
            char** tmp;
            tmp = realloc(ginfo->metricformulas, (ginfo->nmetrics + 1) * sizeof(char*));
            if (tmp == NULL)
            {
                free(ginfo->metricformulas);
                bdestroy(bbuf);
                bstrListDestroy(linelist);
                err = -ENOMEM;
                goto cleanup;
            }
            else
            {
                ginfo->metricformulas = tmp;
            }
            tmp = realloc(ginfo->metricnames, (ginfo->nmetrics + 1) * sizeof(char*));
            if (tmp == NULL)
            {
                free(ginfo->metricnames);
                bdestroy(bbuf);
                bstrListDestroy(linelist);
                err = -ENOMEM;
                goto cleanup;
            }
            else
            {
                ginfo->metricnames = tmp;
            }
            bstrListDestroy(linelist);
            linelist = bsplit(bbuf, ' ');
            ginfo->metricformulas[ginfo->nmetrics] = malloc((blength(linelist->entry[linelist->qty - 1])+1) * sizeof(char));
            if (ginfo->metricformulas[ginfo->nmetrics] == NULL)
            {
                err = -ENOMEM;
                bdestroy(bbuf);
                bstrListDestroy(linelist);
                goto cleanup;
            }
            ginfo->metricnames[ginfo->nmetrics] = malloc(((blength(bbuf)-blength(linelist->entry[linelist->qty - 1]))+1) * sizeof(char));
            if (ginfo->metricnames[ginfo->nmetrics] == NULL)
            {
                err = -ENOMEM;
                bdestroy(bbuf);
                bstrListDestroy(linelist);
                goto cleanup;
            }
            bdestroy(bbuf);
            sprintf(ginfo->metricformulas[ginfo->nmetrics], "%s", bdata(linelist->entry[linelist->qty - 1]));
            bbuf = bstrcpy(linelist->entry[0]);
            for (i=1; i<linelist->qty - 1; i++)
            {
                if (blength(linelist->entry[i]) > 0)
                {
                    bstring tmp = bformat(" %s", bdata(linelist->entry[i]));
                    bconcat(bbuf, tmp);
                    bdestroy(tmp);
                }
            }
            sprintf(ginfo->metricnames[ginfo->nmetrics], "%s", bdata(bbuf));
            bdestroy(bbuf);
            bstrListDestroy(linelist);
            ginfo->nmetrics++;
            continue;
        }
        else if (sec == GROUP_LONG)
        {
            s = (ginfo->longinfo == NULL ? 0 : strlen(ginfo->longinfo));
            char *tmp;
            tmp = realloc(ginfo->longinfo, (s + strlen(buf) + 3) * sizeof(char));
            if (tmp == NULL)
            {
                free(ginfo->longinfo);
                err = -ENOMEM;
                goto cleanup;
            }
            else
            {
                ginfo->longinfo = tmp;
            }
            sprintf(&(ginfo->longinfo[s]), "%.*s", (int)strlen(buf), buf);
            continue;
        }
    }
    //bstrListDestroy(linelist);
    fclose(fp);
    bdestroy(REQUIRE);
    bdestroy(homepath);
    bdestroy(fullpath);
    return 0;
cleanup:
    bdestroy(REQUIRE);
    bdestroy(homepath);
    bdestroy(fullpath);
    if (ginfo->groupname)
        free(ginfo->groupname);
    if (ginfo->shortinfo)
        free(ginfo->shortinfo);
    if (ginfo->longinfo)
        free(ginfo->longinfo);
    if (ginfo->nevents > 0)
    {
        for(i=0;i<ginfo->nevents; i++)
        {
            if (ginfo->counters[i])
                free(ginfo->counters[i]);
            if (ginfo->events[i])
                free(ginfo->events[i]);
        }
    }
    if (ginfo->nmetrics > 0)
    {
        for(i=0;i<ginfo->nmetrics; i++)
        {
            if (ginfo->metricformulas[i])
                free(ginfo->metricformulas[i]);
            if (ginfo->metricnames[i])
                free(ginfo->metricnames[i]);
        }
    }
    /*ginfo->shortinfo = NULL;
    ginfo->nevents = 0;
    ginfo->events = NULL;
    ginfo->counters = NULL;
    ginfo->nmetrics = 0;
    ginfo->metricformulas = NULL;
    ginfo->metricnames = NULL;
    ginfo->longinfo = NULL;
    ginfo->groupname = NULL;*/
    return err;
}

int
perfgroup_new(GroupInfo* ginfo)
{
    if (!ginfo)
        return -EINVAL;
    ginfo->groupname = NULL;
    ginfo->shortinfo = NULL;
    ginfo->nevents = 0;
    ginfo->events = NULL;
    ginfo->counters = NULL;
    ginfo->nmetrics = 0;
    ginfo->metricformulas = NULL;
    ginfo->metricnames = NULL;
    ginfo->longinfo = NULL;
    return 0;
}

char*
perfgroup_getEventStr(GroupInfo* ginfo)
{
    int i;
    char* string;
    int size = 0;
    if (!ginfo)
        return NULL;
    if (ginfo->nevents == 0)
        return NULL;
    for(i=0;i<ginfo->nevents-1; i++)
    {
        size += strlen(ginfo->events[i]) + strlen(ginfo->counters[i]) + 2;
    }
    size += strlen(ginfo->events[ginfo->nevents-1]) + strlen(ginfo->counters[ginfo->nevents-1]) + 1 + 1;
    size++;
    string = malloc(size * sizeof(char));
    if (string == NULL)
        return NULL;
    size = 0;
    for(i=0;i<ginfo->nevents-1; i++)
    {
        size += sprintf(&(string[size]), "%s:%s,", ginfo->events[i], ginfo->counters[i]);
    }
    size += sprintf(&(string[size]), "%s:%s", ginfo->events[ginfo->nevents-1], ginfo->counters[ginfo->nevents-1]);
    string[size] = '\0';
    return string;
}

void
perfgroup_returnEventStr(char* eventset)
{
    if (eventset != NULL)
    {
        free(eventset);
        eventset = NULL;
    }
}

int
perfgroup_addEvent(GroupInfo* ginfo, char* counter, char* event)
{
    if ((!ginfo) || (!event) || (!counter))
        return -EINVAL;
    ginfo->events = realloc(ginfo->events, (ginfo->nevents + 1) * sizeof(char*));
    if (!ginfo->events)
        return -ENOMEM;
    ginfo->counters = realloc(ginfo->counters, (ginfo->nevents + 1) * sizeof(char*));
    if (!ginfo->counters)
        return -ENOMEM;
    ginfo->events[ginfo->nevents] = malloc((strlen(event) + 1) * sizeof(char));
    if (!ginfo->events[ginfo->nevents])
        return -ENOMEM;
    ginfo->counters[ginfo->nevents] = malloc((strlen(counter) + 1) * sizeof(char));
    if (!ginfo->counters[ginfo->nevents])
        return -ENOMEM;
    sprintf(ginfo->events[ginfo->nevents], "%s", event);
    sprintf(ginfo->counters[ginfo->nevents], "%s", counter);
    ginfo->nevents++;
    return 0;
}

void perfgroup_removeEvent(GroupInfo* ginfo, char* counter)
{
    fprintf(stderr, "perfgroup_removeEvent not implemented\n");
}

int
perfgroup_addMetric(GroupInfo* ginfo, char* mname, char* mcalc)
{
    if ((!ginfo) || (!mname) || (!mcalc))
        return -EINVAL;
    ginfo->metricnames = realloc(ginfo->metricnames, (ginfo->nmetrics + 1) * sizeof(char*));
    if (!ginfo->metricnames)
    {
        ERROR_PRINT(Cannot increase space for metricnames to %d bytes, (ginfo->nmetrics + 1) * sizeof(char*));
        return -ENOMEM;
    }
    ginfo->metricformulas = realloc(ginfo->metricformulas, (ginfo->nmetrics + 1) * sizeof(char*));
    if (!ginfo->metricformulas)
    {
        ERROR_PRINT(Cannot increase space for metricformulas to %d bytes, (ginfo->nmetrics + 1) * sizeof(char*));
        return -ENOMEM;
    }
    ginfo->metricnames[ginfo->nmetrics] = malloc((strlen(mname) + 1) * sizeof(char));
    if (!ginfo->metricnames[ginfo->nmetrics])
    {
        ERROR_PRINT(Cannot increase space for metricname to %d bytes, (strlen(mname) + 1) * sizeof(char));
        return -ENOMEM;
    }
    ginfo->metricformulas[ginfo->nmetrics] = malloc((strlen(mcalc) + 1) * sizeof(char));
    if (!ginfo->metricformulas[ginfo->nmetrics])
    {
        ERROR_PRINT(Cannot increase space for metricformula to %d bytes, (strlen(mcalc) + 1) * sizeof(char));
        return -ENOMEM;
    }
    DEBUG_PRINT(DEBUGLEV_DEVELOP, Adding metric %s = %s, mname, mcalc);
    int ret = sprintf(ginfo->metricnames[ginfo->nmetrics], "%s", mname);
    if (ret > 0)
    {
        ginfo->metricnames[ginfo->nmetrics][ret] = '\0';
    }
    ret = sprintf(ginfo->metricformulas[ginfo->nmetrics], "%s", mcalc);
    if (ret > 0)
    {
        ginfo->metricformulas[ginfo->nmetrics][ret] = '\0';
    }
    ginfo->nmetrics++;
    return 0;
}

void perfgroup_removeMetric(GroupInfo* ginfo, char* mname)
{
    fprintf(stderr, "perfgroup_removeEvent not implemented\n");
}


char*
perfgroup_getGroupName(GroupInfo* ginfo)
{
    if ((ginfo != NULL) && (ginfo->groupname != NULL))
    {
        int size = strlen(ginfo->groupname)+1;
        char* gstr = malloc(size * sizeof(char));
        sprintf(gstr, "%s", ginfo->groupname);
        return gstr;
    }
    return NULL;
}

void
perfgroup_returnGroupName(char* gname)
{
    if (gname != NULL)
    {
        free(gname);
        gname = NULL;
    }
}

int
perfgroup_setGroupName(GroupInfo* ginfo, char* groupName)
{
    if ((ginfo == NULL) || (groupName == NULL))
        return -EINVAL;
    int size = strlen(groupName)+1;
    ginfo->groupname = realloc(ginfo->groupname, size * sizeof(char));
    if (ginfo->groupname == NULL)
    {
        ERROR_PRINT(Cannot increase space for groupname to %d bytes, size * sizeof(char));
        return -ENOMEM;
    }
    DEBUG_PRINT(DEBUGLEV_DEVELOP, Setting group name to %s, groupName);
    int ret = sprintf(ginfo->groupname, "%s", groupName);
    if (ret > 0)
    {
        ginfo->groupname[ret] = '\0';
    }
    return 0;
}

char*
perfgroup_getShortInfo(GroupInfo* ginfo)
{
    if ((ginfo != NULL) && (ginfo->shortinfo != NULL))
    {
        int size = strlen(ginfo->shortinfo)+1;
        char* sstr = malloc(size * sizeof(char));
        sprintf(sstr, "%s", ginfo->shortinfo);
        return sstr;
    }
    return NULL;
}

void
perfgroup_returnShortInfo(char* sinfo)
{
    if (sinfo != NULL)
    {
        free(sinfo);
        sinfo = NULL;
    }
}

int
perfgroup_setShortInfo(GroupInfo* ginfo, char* shortInfo)
{
    if ((ginfo == NULL) || (shortInfo == NULL))
        return -EINVAL;
    int size = strlen(shortInfo)+1;
    ginfo->shortinfo = realloc(ginfo->shortinfo, size * sizeof(char));
    if (ginfo->shortinfo == NULL)
        return -ENOMEM;
    sprintf(ginfo->shortinfo, "%s", shortInfo);
    return 0;
}

char*
perfgroup_getLongInfo(GroupInfo* ginfo)
{
    if ((ginfo != NULL) && (ginfo->longinfo != NULL))
    {
        int size = strlen(ginfo->longinfo)+1;
        char* lstr = malloc(size * sizeof(char));
        sprintf(lstr, "%s", ginfo->longinfo);
        return lstr;
    }
    return NULL;
}

void
perfgroup_returnLongInfo(char* linfo)
{
    if (linfo != NULL)
    {
        free(linfo);
        linfo = NULL;
    }
}

int
perfgroup_setLongInfo(GroupInfo* ginfo, char* longInfo)
{
    if ((ginfo == NULL) || (longInfo == NULL))
        return -EINVAL;
    int size = strlen(longInfo)+1;
    ginfo->longinfo = realloc(ginfo->longinfo, size * sizeof(char));
    if (ginfo->longinfo == NULL)
        return -ENOMEM;
    sprintf(ginfo->longinfo, "%s", longInfo);
    return 0;
}

void
perfgroup_returnGroup(GroupInfo* ginfo)
{
    int i;
    if (ginfo->groupname)
        free(ginfo->groupname);
    if (ginfo->shortinfo)
        free(ginfo->shortinfo);
    if (ginfo->longinfo)
        free(ginfo->longinfo);
    if (ginfo->nevents > 0)
    {
        for(i=0;i<ginfo->nevents; i++)
        {
            if (ginfo->counters[i])
                free(ginfo->counters[i]);
            if (ginfo->events[i])
                free(ginfo->events[i]);
        }
        free(ginfo->counters);
        free(ginfo->events);
    }
    if (ginfo->nmetrics > 0)
    {
        for(i=0;i<ginfo->nmetrics; i++)
        {
            if (ginfo->metricformulas[i])
                free(ginfo->metricformulas[i]);
            if (ginfo->metricnames[i])
                free(ginfo->metricnames[i]);
        }
        free(ginfo->metricformulas);
        free(ginfo->metricnames);
    }
    ginfo->groupname = NULL;
    ginfo->shortinfo = NULL;
    ginfo->longinfo = NULL;
    ginfo->counters = NULL;
    ginfo->events = NULL;
    ginfo->metricformulas = NULL;
    ginfo->metricnames = NULL;
    ginfo->nevents = 0;
    ginfo->nmetrics = 0;
}


int perfgroup_mergeGroups(GroupInfo* grp1, GroupInfo* grp2)
{
    fprintf(stderr, "perfgroup_mergeGroups not implemented\n");
    return -1;
}


void
init_clist(CounterList* clist)
{
    clist->counters = 0;
    clist->cnames = bstrListCreate();
    clist->cvalues = bstrListCreate();
}

int
add_to_clist(CounterList* clist, char* counter, double result)
{

    bstrListAddChar(clist->cnames, counter);
    bstring v = bformat("%.20f", result);
    bstrListAdd(clist->cvalues, v);
    clist->counters++;
    bdestroy(v);
    return 0;
}

int
update_clist(CounterList* clist, char* counter, double result)
{
    int i;
    int found = 0;
    if ((clist == NULL)||(counter == NULL))
        return -EINVAL;
    bstring c = bfromcstr(counter);
    for (i=0; i< clist->counters; i++)
    {
        bstring comp = bstrListGet(clist->cnames, i);
        if (bstrcmp(comp, c) == BSTR_OK)
        {
            bstring v = bformat("%.20f", result);
            bstring val = bstrListGet(clist->cvalues, i);
            btrunc(val, 0);
            bconcat(val, v);
            bdestroy(v);
            found = 1;
            break;
        }
    }
    bdestroy(c);
    if (!found)
    {
        return -ENOENT;
    }
    return 0;
}

void
destroy_clist(CounterList* clist)
{
    int i;
    if (clist != NULL)
    {
        bstrListDestroy(clist->cnames);
        bstrListDestroy(clist->cvalues);
        clist->counters = 0;
    }
}

int
calc_metric(char* formula, CounterList* clist, double *result)
{
    int i=0;
    *result = 0.0;
    int maxstrlen = 0, minstrlen = 10000;
    bstring nan;
    bstring zero;
    bstring inf;

    if ((formula == NULL) || (clist == NULL))
        return -EINVAL;

    bstring f = bfromcstr(formula);
    nan = bfromcstr("nan");
    inf = bfromcstr("inf");
    zero = bfromcstr("0.0");
    for(i=0;i<clist->counters;i++)
    {
        bstring c = bstrListGet(clist->cnames, i);
        int len = blength(c);
        maxstrlen = (maxstrlen > len ? maxstrlen : len);
        minstrlen = (minstrlen < len ? minstrlen : len);
    }

    // try to replace each counter name in clist
    while (maxstrlen >= minstrlen)
    {
        for(i=0;i<clist->counters;i++)
        {
            bstring c = bstrListGet(clist->cnames, i);
            if (blength(c) != maxstrlen)
                continue;
            bstring v = bstrListGet(clist->cvalues, i);
            if ((bstrncmp(v, nan, 3) != BSTR_OK) && (bstrncmp(v, inf, 3) != BSTR_OK))
            {
                bfindreplace(f, c, v, 0);
            }
            else
            {
                bfindreplace(f, c, zero, 0);
            }
        }
        maxstrlen--;
    }
    // now we can calculate the formula
    i = calculate_infix(bdata(f), result);
    bdestroy(f);
    bdestroy(inf);
    bdestroy(nan);
    bdestroy(zero);
    return i;
}
