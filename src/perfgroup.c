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
 *                Thomas Roehl (tr), thomas.roehl@gmail.com
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

#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <dirent.h>

#include <error.h>
#include <perfgroup.h>
#include <calculator.h>
#include <likwid.h>

int isdir(char* dirname)
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

int get_groups(char* grouppath, char* architecture, char*** groupnames, char*** groupshort, char*** grouplong)
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
    int read_long = 0;
    if ((grouppath == NULL)||(architecture == NULL)||(groupnames == NULL))
        return -EINVAL;
    char* fullpath = malloc((strlen(grouppath)+strlen(architecture)+50) * sizeof(char));
    if (fullpath == NULL)
    {
        bdestroy(SHORT);
        bdestroy(LONG);
        return -ENOMEM;
    }
    char* homepath = malloc((strlen(getenv("HOME"))+strlen(architecture)+50) * sizeof(char));
    if (homepath == NULL)
    {
        free(fullpath);
        bdestroy(SHORT);
        bdestroy(LONG);
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
        return -EACCES;
    }
    i = 0;
    s = 0;
    while (ep = readdir(dp))
    {
        if (strncmp(&(ep->d_name[strlen(ep->d_name)-4]), ".txt", 4) == 0)
        {
            i++;
            if (strlen(ep->d_name)-4 > s)
                s = strlen(ep->d_name)-4;
        }
    }
    closedir(dp);
    hsize = sprintf(homepath, "%s/.likwid/groups/%s", getenv("HOME"), architecture);
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
            while (ep = readdir(dp))
            {
                if (strncmp(&(ep->d_name[strlen(ep->d_name)-4]), ".txt", 4) == 0)
                {
                    i++;
                    if (strlen(ep->d_name)-4 > s)
                        s = strlen(ep->d_name)-4;
                }
            }
            closedir(dp);
        }
    }

    *groupnames = malloc(i * sizeof(char**));
    if (*groupnames == NULL)
    {
        free(fullpath);
        free(homepath);
        bdestroy(SHORT);
        bdestroy(LONG);
        return -ENOMEM;
    }
    *groupshort = malloc(i * sizeof(char**));
    if (*groupshort == NULL)
    {
        free(*groupnames);
        *groupnames = NULL;
        free(fullpath);
        free(homepath);
        bdestroy(SHORT);
        bdestroy(LONG);
        return -ENOMEM;
    }
    *grouplong = malloc(i * sizeof(char**));
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
        return -ENOMEM;
    }
    for (j=0; j < i; j++)
    {
        (*grouplong)[i] == NULL;
        (*groupshort)[i] == NULL;
        (*groupnames)[j] = malloc((s+1) * sizeof(char));
        if ((*groupnames)[j] == NULL)
        {
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
            return -ENOMEM;
        }
    }
    dp = opendir(fullpath);
    i = 0;
    
    while (ep = readdir(dp))
    {
        if (strncmp(&(ep->d_name[strlen(ep->d_name)-4]), ".txt", 4) == 0)
        {
            read_long = 0;
            bstring long_info = bfromcstr("");;
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
                            free(homepath);
                            free(fullpath);
                            bstrListDestroy(linelist);
                            return -ENOMEM;
                        }
                        s = sprintf((*groupshort)[i], "%s", bdata(sinfo));
                        (*groupshort)[i][s] = '\0';
                        bstrListDestroy(linelist);
                        bdestroy(sinfo);
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
                
                i++;
            }
            bdestroy(long_info);
        }
    }
    closedir(dp);
    if (!search_home)
    {
        free(homepath);
        free(fullpath);
        bdestroy(SHORT);
        bdestroy(LONG);
        return i;
    }
    else
    {
        dp = opendir(homepath);
        while (ep = readdir(dp))
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
                                bdestroy(bbuf);
                                bdestroy(sinfo);
                                free(homepath);
                                free(fullpath);
                                bstrListDestroy(linelist);
                                return -ENOMEM;
                            }
                            s = sprintf((*groupshort)[i], "%s", bdata(sinfo));
                            (*groupshort)[i][s] = '\0';
                            bstrListDestroy(linelist);
                            bdestroy(sinfo);
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
                    i++;
                }
                bdestroy(long_info);
            }
        }
        closedir(dp);
    }
    bdestroy(SHORT);
    bdestroy(LONG);
    free(fullpath);
    free(homepath);
    return i;
}

void return_groups(int groups, char** groupnames, char** groupshort, char** grouplong)
{
    int i;
    for (i = 0; i <groups; i++)
    {
        if (groupnames[i])
            free(groupnames[i]);
        if (groupshort[i])
            free(groupshort[i]);
        if (grouplong[i])
            free(grouplong[i]);
    }
    if (groupnames)
        free(groupnames);
    if (groupshort)
        free(groupshort);
    if (grouplong)
        free(grouplong);
}



int custom_group(char* eventStr, GroupInfo* ginfo)
{
    int i, j;
    int err = 0;
    char delim = ',';
    bstring edelim = bformat(":");
    int has_fix0 = 0;
    int has_fix1 = 0;
    int has_fix2 = 0;
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
    bstring fix0 = bformat("FIXC0");
    bstring fix1 = bformat("FIXC1");
    bstring fix2 = bformat("FIXC2");
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
    if (binstr(eventBstr, 0, fix2) > 0)
    {
        has_fix2 = 1;
    }
    else
    {
        ginfo->nevents++;
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
        bdestroy(ctr);
        bstrListDestroy(elist);
    }
    i = eventList->qty;
    if (!has_fix0)
    {
        ginfo->events[i] = malloc(18 * sizeof(char));
        ginfo->counters[i] = malloc(6 * sizeof(char));
        sprintf(ginfo->events[i], "%s", "INSTR_RETIRED_ANY");
        sprintf(ginfo->counters[i], "%s", "FIXC0");
        i++;
    }
    if (!has_fix1)
    {
        ginfo->events[i] = malloc(22 * sizeof(char));
        ginfo->counters[i] = malloc(6 * sizeof(char));
        sprintf(ginfo->events[i], "%s", "CPU_CLK_UNHALTED_CORE");
        sprintf(ginfo->counters[i], "%s", "FIXC1");
        i++;
    }
    if (!has_fix2)
    {
        ginfo->events[i] = malloc(21 * sizeof(char));
        ginfo->counters[i] = malloc(6 * sizeof(char));
        sprintf(ginfo->events[i], "%s", "CPU_CLK_UNHALTED_REF");
        sprintf(ginfo->counters[i], "%s", "FIXC2");
        i++;
    }

    bstrListDestroy(eventList);
    bdestroy(fix0);
    bdestroy(fix1);
    bdestroy(fix2);
    bdestroy(edelim);
    return 0;
cleanup:
    bstrListDestroy(eventList);
    bdestroy(fix0);
    bdestroy(fix1);
    bdestroy(fix2);
    bdestroy(edelim);
    if (ginfo->shortinfo != NULL)
        free(ginfo->shortinfo);
    if (ginfo->events != NULL)
        free(ginfo->events);
    if (ginfo->counters != NULL)
        free(ginfo->counters);
    return err;
}

int read_group(char* grouppath, char* architecture, char* groupname, GroupInfo* ginfo)
{
    FILE* fp;
    int i, s, e, err = 0;
    char buf[512];
    GroupFileSections sec = GROUP_NONE;
    if ((grouppath == NULL)||(architecture == NULL)||(groupname == NULL)||(ginfo == NULL))
        return -EINVAL;

    bstring fullpath = bformat("%s/%s/%s.txt", grouppath,architecture, groupname);
    bstring homepath = bformat("%s/.likwid/groups/%s/%s.txt", getenv("HOME"),architecture, groupname);

    if (access(bdata(fullpath), R_OK))
    {
        DEBUG_PRINT(DEBUGLEV_INFO, Cannot read group file %s. Trying %s, bdata(fullpath), bdata(homepath));
        if (access(bdata(homepath), R_OK))
        {
            ERROR_PRINT(Cannot read group file %s.txt. Searched in %s and %s, groupname, bdata(fullpath), bdata(homepath));
            bdestroy(fullpath);
            bdestroy(homepath);
            exit(EXIT_FAILURE);
        }
        else
        {
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
    while (fgets (buf, sizeof(buf), fp)) {
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
    bdestroy(homepath);
    bdestroy(fullpath);
    return 0;
cleanup:
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
    return err;
}

int new_group(GroupInfo* ginfo)
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

char* get_eventStr(GroupInfo* ginfo)
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

void put_eventStr(char* eventset)
{
    if (eventset != NULL)
    {
        free(eventset);
        eventset = NULL;
    }
}

int add_event(GroupInfo* ginfo, char* event, char* counter)
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

int add_metric(GroupInfo* ginfo, char* mname, char* mcalc)
{
    if ((!ginfo) || (!mname) || (!mcalc))
        return -EINVAL;
    ginfo->metricnames = realloc(ginfo->metricnames, (ginfo->nmetrics + 1) * sizeof(char*));
    if (!ginfo->metricnames)
        return -ENOMEM;
    ginfo->metricformulas = realloc(ginfo->metricformulas, (ginfo->nmetrics + 1) * sizeof(char*));
    if (!ginfo->metricformulas)
        return -ENOMEM;
    ginfo->metricnames[ginfo->nmetrics] = malloc((strlen(mname) + 1) * sizeof(char));
    if (!ginfo->metricnames[ginfo->nmetrics])
        return -ENOMEM;
    ginfo->metricformulas[ginfo->nmetrics] = malloc((strlen(mcalc) + 1) * sizeof(char));
    if (!ginfo->metricformulas[ginfo->nmetrics])
        return -ENOMEM;
    sprintf(ginfo->metricnames[ginfo->nmetrics], "%s", mname);
    sprintf(ginfo->metricformulas[ginfo->nmetrics], "%s", mcalc);
    ginfo->nmetrics++;
    return 0;
}


char* get_groupName(GroupInfo* ginfo)
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

int set_groupName(GroupInfo* ginfo, char* groupName)
{
    if ((ginfo == NULL) || (groupName == NULL))
        return -EINVAL;
    int size = strlen(groupName)+1;
    ginfo->groupname = realloc(ginfo->groupname, size * sizeof(char));
    if (ginfo->groupname == NULL)
        return -ENOMEM;
    sprintf(ginfo->groupname, "%s", groupName);
    return 0;
}

char* get_shortInfo(GroupInfo* ginfo)
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

void put_shortInfo(char* sinfo)
{
    if (sinfo != NULL)
    {
        free(sinfo);
        sinfo = NULL;
    }
}

int set_shortInfo(GroupInfo* ginfo, char* shortInfo)
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

char* get_longInfo(GroupInfo* ginfo)
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

void put_longInfo(char* linfo)
{
    if (linfo != NULL)
    {
        free(linfo);
        linfo = NULL;
    }
}

int set_longInfo(GroupInfo* ginfo, char* longInfo)
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

void return_group(GroupInfo* ginfo)
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

void init_clist(CounterList* clist)
{
    clist->counters = 0;
    clist->cnames = NULL;
    clist->cvalues = NULL;
}

int add_to_clist(CounterList* clist, char* counter, double result)
{
    char** tmpnames;
    double* tmpvalues;
    if ((clist == NULL)||(counter == NULL))
        return -EINVAL;
    tmpnames = realloc(clist->cnames, (clist->counters + 1) * sizeof(char*));
    if (tmpnames == NULL)
    {
        return -ENOMEM;
    }
    clist->cnames = tmpnames;
    tmpvalues = realloc(clist->cvalues, (clist->counters + 1) * sizeof(double));
    if (tmpvalues == NULL)
    {
        return -ENOMEM;
    }
    clist->cvalues = tmpvalues;
    clist->cnames[clist->counters] = malloc((strlen(counter)+2)*sizeof(char));
    if (clist->cnames[clist->counters] == NULL)
    {
        return -ENOMEM;
    }
    sprintf(clist->cnames[clist->counters],"%s", counter);
    clist->cvalues[clist->counters] = result;
    clist->counters++;
    return 0;
}

void destroy_clist(CounterList* clist)
{
    int i;
    if (clist != NULL)
    {
        for (i=0;i<clist->counters;i++)
        {
            free(clist->cnames[i]);
        }
        free(clist->cnames);
        free(clist->cvalues);
    }
}


int calc_metric(char* formula, CounterList* clist, double *result)
{
    int i=0;
    *result = 0.0;
    int fail = 0;
    int maxstrlen = 0, minstrlen = 10000;

    if ((formula == NULL) || (clist == NULL))
        return -EINVAL;

    bstring f = bfromcstr(formula);
    for(i=0;i<clist->counters;i++)
    {
        if (strlen(clist->cnames[i]) > maxstrlen)
            maxstrlen = strlen(clist->cnames[i]);
        if (strlen(clist->cnames[i]) < minstrlen)
            minstrlen = strlen(clist->cnames[i]);
    }

    // try to replace each counter name in clist
    while (maxstrlen >= minstrlen)
    {
        for(i=0;i<clist->counters;i++)
        {
            if (strlen(clist->cnames[i]) != maxstrlen)
                continue;
            // if we find the counter name, replace it with the value
            bstring c = bfromcstr(clist->cnames[i]);
            bstring v = bformat("%.20f", clist->cvalues[i]);
            bfindreplace(f, c, v, 0);
            bdestroy(c);
            bdestroy(v);
        }
        maxstrlen--;
    }
    bstring test = bfromcstr("aAbBcCdDfFgGhHiIjJkKlLmMnNoOpPqQrRsStTuUvVwWxXyYzZ,_:;!'§$&=?°´`#<>");
    if (binchr(f, 0, test) != BSTR_ERR)
    {
        fprintf(stderr, "Not all counter names in formula can be substituted\n");
        fprintf(stderr, "%s\n", bdata(f));
        i = -EINVAL;
        fail = 1;
    }
    bdestroy(test);
    // now we can calculate the formula
    if (!fail)
        i = calculate_infix(bdata(f), result);
    bdestroy(f);
    return i;
}
