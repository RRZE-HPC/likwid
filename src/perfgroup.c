#include <stdlib.h>
#include <sys/types.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <dirent.h>

#include <perfgroup.h>
#include <calculator.h>
#include <likwid.h>

int get_groups(char* grouppath, char* architecture, char*** groupnames, char*** groupshort, char*** grouplong)
{
    int i, j;
    int fsize;
    DIR *dp;
    FILE* fp;
    char buf[256];
    struct dirent *ep;
    *groupnames = NULL;
    *groupshort = NULL;
    *grouplong = NULL;
    int read_long = 0;
    if ((grouppath == NULL)||(architecture == NULL)||(groupnames == NULL))
        return -EINVAL;
    char* fullpath = malloc((strlen(grouppath)+strlen(architecture)+50) * sizeof(char));
    if (fullpath == NULL)
    {
        return -ENOMEM;
    }
    fsize = sprintf(fullpath, "%s/%s", grouppath, architecture);
    dp = opendir(fullpath);
    if (dp == NULL)
    {
        printf("Cannot open directory %s\n", fullpath);
        free(fullpath);
        return -EACCES;
    }
    i = 0;
    while (ep = readdir(dp))
    {
        if (strncmp(&(ep->d_name[strlen(ep->d_name)-4]), ".txt", 4) == 0)
        {
            read_long = 0;
            sprintf(&(fullpath[fsize]), "/%s", ep->d_name);
            if (!access(fullpath, R_OK))
            {
                *groupnames = realloc(*groupnames, (i+1) * sizeof(char*));
                if (*groupnames == NULL)
                {
                    free(fullpath);
                    return -ENOMEM;
                }
                (*groupnames)[i] = malloc((strlen(ep->d_name)-4) * sizeof(char));
                if ((*groupnames)[i] == NULL)
                {
                    free(fullpath);
                    return -ENOMEM;
                }
                *groupshort = realloc(*groupshort, (i+1) * sizeof(char*));
                if (*groupshort == NULL)
                {
                    free(fullpath);
                    return -ENOMEM;
                }
                *grouplong = realloc(*grouplong, (i+1) * sizeof(char*));
                if (*grouplong == NULL)
                {
                    free(fullpath);
                    return -ENOMEM;
                }
                (*grouplong)[i] = NULL;
                sprintf((*groupnames)[i], "%.*s", (int)(strlen(ep->d_name)-4), ep->d_name);
                fp = fopen(fullpath,"r");
                while (fgets (buf, sizeof(buf), fp)) {
                    if ((strlen(buf) == 0) || (buf[0] == '#'))
                        continue;
                    if (strncmp("SHORT", buf, 5) == 0)
                    {
                        for (j=5; j < strlen(buf); j++)
                        {
                            if (buf[j] == ' ')
                                continue;
                            break;
                        }
                        (*groupshort)[i] = malloc(strlen(&(buf[j])) * sizeof(char));
                        if ((*groupshort)[i] == NULL)
                        {
                            free(fullpath);
                            return -ENOMEM;
                        }
                        sprintf((*groupshort)[i], "%.*s", (int)strlen(&(buf[j]))-1, &(buf[j]));
                        continue;
                    }
                    else if (strncmp("LONG", buf, 4) == 0)
                    {
                        read_long = 1;
                        j = 0;
                        continue;
                    }
                    else if (read_long == 1)
                    {
                        if ((*grouplong)[i] == NULL)
                            (*grouplong)[i] = malloc((strlen(buf)+1) * sizeof(char));
                        else
                            (*grouplong)[i] = realloc((*grouplong)[i], (strlen((*grouplong)[i])+strlen(buf)+1) * sizeof(char));
                        if ((*grouplong)[i] == NULL)
                        {
                            free(fullpath);
                            return -ENOMEM;
                        }
                        j += sprintf(&((*grouplong)[i][j]), "%s", buf);
                    }
                }
                fclose(fp);
                i++;
            }
        }
    }
    closedir(dp);
    fsize = sprintf(fullpath, "%s/.likwid/groups/%s", getenv("HOME"), architecture);
    dp = opendir(fullpath);
    if (dp == NULL)
    {
        free(fullpath);
        if (i > 0)
            return i;
        else
            return -EACCES;
    }
    while (ep = readdir(dp))
    {
        if (strncmp(&(ep->d_name[strlen(ep->d_name)-4]), ".txt", 4) == 0)
        {
            sprintf(&(fullpath[fsize]), "/%s", ep->d_name);
            if (!access(fullpath, R_OK))
            {
                *groupnames = realloc(*groupnames, (i+1) * sizeof(char*));
                (*groupnames)[i] = malloc((strlen(ep->d_name)-4) * sizeof(char));
                *groupshort = realloc(*groupshort, (i+1) * sizeof(char*));
                *grouplong = realloc(*grouplong, (i+1) * sizeof(char*));
                (*grouplong)[i] = NULL;
                read_long = 0;
                sprintf((*groupnames)[i], "%.*s", (int)(strlen(ep->d_name)-4), ep->d_name);
                fp = fopen(fullpath,"r");
                while (fgets (buf, sizeof(buf), fp)) {
                    if ((strlen(buf) == 0) || (buf[0] == '#'))
                        continue;

                    if (strncmp("SHORT", buf, 5) == 0)
                    {
                        for (j=5; j < strlen(buf); j++)
                        {
                            if (buf[j] == ' ')
                                continue;
                            break;
                        }
                        (*groupshort)[i] = malloc(strlen(&(buf[j])) * sizeof(char));
                        sprintf((*groupshort)[i], "%.*s", (int)strlen(&(buf[j]))-1, &(buf[j]));
                        continue;
                    }
                    if (strncmp("LONG", buf, 4) == 0)
                    {
                        read_long = 1;
                        j = 0;
                    }
                    if (read_long == 1)
                    {
                        if ((*grouplong)[i] == NULL)
                            (*grouplong)[i] = malloc((strlen(buf)+1) * sizeof(char));
                        else
                            (*grouplong)[i] = realloc((*grouplong)[i], (strlen((*grouplong)[i])+strlen(buf)+1) * sizeof(char));
                        j += sprintf(&((*grouplong)[i][j]), "%s", buf);
                    }
                }
                fclose(fp);
                i++;
            }
        }
    }
    closedir(dp);
    free(fullpath);
    return i;
}


int custom_group(char* eventStr, GroupInfo* ginfo)
{
    int i;
    int err = 0;
    char *ptr, *ptr2, *split_ptr;
    char delim[] = ",";
    char *eventCopy = NULL;
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

    eventCopy = malloc((strlen(eventStr)+1)* sizeof(char));
    if (eventCopy == NULL)
    {
        err = -ENOMEM;
        goto cleanup;
    }

    strcpy(eventCopy, eventStr);
    if (strstr(eventCopy, "FIXC0"))
        has_fix0 = 1;
    if (strstr(eventCopy, "FIXC1"))
        has_fix1 = 1;
    if (strstr(eventCopy, "FIXC2"))
        has_fix2 = 1;
    
    ptr = strtok(eventCopy, delim);
    while (ptr != NULL)
    {
        ginfo->events = realloc(ginfo->events, (ginfo->nevents + 1) * sizeof(char*));
        if (ginfo->events == NULL)
        {
            err = -ENOMEM;
            goto cleanup;
        }
        ginfo->counters = realloc(ginfo->counters, (ginfo->nevents + 1) * sizeof(char*));
        if (ginfo->counters == NULL)
        {
            err = -ENOMEM;
            goto cleanup;
        }
        split_ptr = strchr(ptr, ':');
        ginfo->events[ginfo->nevents] = malloc((strlen(ptr)-strlen(split_ptr)+2)*sizeof(char));
        ginfo->counters[ginfo->nevents] = malloc((strlen(split_ptr)+2)*sizeof(char));
        snprintf(ginfo->events[ginfo->nevents], strlen(ptr)-strlen(split_ptr)+1, "%s", ptr);
        sprintf(ginfo->counters[ginfo->nevents], "%s",split_ptr+1);
        ginfo->nevents++;
        ptr = strtok(NULL, delim);
    }
    if (!has_fix0)
    {
        ginfo->events = realloc(ginfo->events, (ginfo->nevents + 1) * sizeof(char*));
        if (ginfo->events == NULL)
        {
            err = -ENOMEM;
            goto cleanup;
        }
        ginfo->counters = realloc(ginfo->counters, (ginfo->nevents + 1) * sizeof(char*));
        if (ginfo->counters == NULL)
        {
            err = -ENOMEM;
            goto cleanup;
        }
        ginfo->events[ginfo->nevents] = malloc(18 * sizeof(char));
        ginfo->counters[ginfo->nevents] = malloc(6 * sizeof(char));
        sprintf(ginfo->events[ginfo->nevents], "%s", "INSTR_RETIRED_ANY");
        sprintf(ginfo->counters[ginfo->nevents], "%s", "FIXC0");
        ginfo->nevents++;
    }
    if (!has_fix1)
    {
        ginfo->events = realloc(ginfo->events, (ginfo->nevents + 1) * sizeof(char*));
        if (ginfo->events == NULL)
        {
            err = -ENOMEM;
            goto cleanup;
        }
        ginfo->counters = realloc(ginfo->counters, (ginfo->nevents + 1) * sizeof(char*));
        if (ginfo->counters == NULL)
        {
            err = -ENOMEM;
            goto cleanup;
        }
        ginfo->events[ginfo->nevents] = malloc(22 * sizeof(char));
        ginfo->counters[ginfo->nevents] = malloc(6 * sizeof(char));
        sprintf(ginfo->events[ginfo->nevents], "%s", "CPU_CLK_UNHALTED_CORE");
        sprintf(ginfo->counters[ginfo->nevents], "%s", "FIXC1");
        ginfo->nevents++;
    }
    if (!has_fix2)
    {
        ginfo->events = realloc(ginfo->events, (ginfo->nevents + 1) * sizeof(char*));
        if (ginfo->events == NULL)
        {
            err = -ENOMEM;
            goto cleanup;
        }
        ginfo->counters = realloc(ginfo->counters, (ginfo->nevents + 1) * sizeof(char*));
        if (ginfo->counters == NULL)
        {
            err = -ENOMEM;
            goto cleanup;
        }
        ginfo->events[ginfo->nevents] = malloc(21 * sizeof(char));
        ginfo->counters[ginfo->nevents] = malloc(6 * sizeof(char));
        sprintf(ginfo->events[ginfo->nevents], "%s", "CPU_CLK_UNHALTED_REF");
        sprintf(ginfo->counters[ginfo->nevents], "%s", "FIXC2");
        ginfo->nevents++;
    }

    free(eventCopy);
    return 0;
cleanup:
    if (eventCopy != NULL)
        free(eventCopy);
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

    char* fullpath = malloc((strlen(grouppath)+strlen(architecture)+strlen(groupname)+100)*sizeof(char));
    if (fullpath == NULL)
        return -ENOMEM;
    i = sprintf(fullpath, "%s/%s/%s.txt", grouppath,architecture, groupname);
    fullpath[i] = '\0';

    if (access(fullpath, R_OK))
    {
        i = sprintf(fullpath, "%s/.likwid/groups/%s/%s.txt", getenv("HOME"), architecture, groupname);
        fullpath[i] = '\0';
        if (access(fullpath, R_OK))
        {
            printf("Cannot read group file %s\n", fullpath);
            free(fullpath);
            return -EACCES;
        }
    }

    ginfo->shortinfo = NULL;
    ginfo->nevents = 0;
    ginfo->events = NULL;
    ginfo->counters = NULL;
    ginfo->nmetrics = 0;
    ginfo->metricformulas = NULL;
    ginfo->metricnames = NULL;
    ginfo->longinfo = NULL;

    ginfo->groupname = malloc(strlen(groupname)*sizeof(char));
    if (ginfo->groupname == NULL)
    {
        err = -ENOMEM;
        goto cleanup;
    }
    strcpy(ginfo->groupname, groupname);
    
    fp = fopen(fullpath, "r");
    if (fp == NULL)
    {
        free(ginfo->groupname);
        free(fullpath);
        return -EACCES;
    }
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
            while (buf[i] == ' ') {i++;}
            if (buf[i] == '\n')
            {
                sec = GROUP_NONE;
                continue;
            }
            ginfo->events = realloc(ginfo->events, (ginfo->nevents + 1) * sizeof(char*));
            if (ginfo->events == NULL)
            {
                err = -ENOMEM;
                goto cleanup;
            }
            ginfo->counters = realloc(ginfo->counters, (ginfo->nevents + 1) * sizeof(char*));
            if (ginfo->counters == NULL)
            {
                err = -ENOMEM;
                goto cleanup;
            }
            i=0;
            while(buf[i] == ' ') {i++;}
            s = i;
            for (; i< strlen(buf); i++)
            {
                if (buf[i] == ' ')
                {
                    e = i;
                    break;
                }
            }
            ginfo->counters[ginfo->nevents] = malloc((e-s+1) * sizeof(char));
            if (ginfo->counters[ginfo->nevents] == NULL)
            {
                err = -ENOMEM;
                goto cleanup;
            }
            sprintf(ginfo->counters[ginfo->nevents], "%.*s", e-s, &(buf[s]));
            i = e;
            while(buf[i] == ' ') {i++;}
            s = i;
            e = strlen(buf)-1;
            ginfo->events[ginfo->nevents] = malloc((e-s+1) * sizeof(char));
            if (ginfo->events[ginfo->nevents] == NULL)
            {
                err = -ENOMEM;
                goto cleanup;
            }
            sprintf(ginfo->events[ginfo->nevents], "%.*s", e-s, &(buf[s]));
            
            ginfo->nevents++;
            continue;
        }
        else if (sec == GROUP_METRICS)
        {
            i = 0;
            while (buf[i] == ' ') {i++;}
            if (buf[i] == '\n')
            {
                sec = GROUP_NONE;
                continue;
            }
            ginfo->metricformulas = realloc(ginfo->metricformulas, (ginfo->nmetrics + 1) * sizeof(char*));
            if (ginfo->metricformulas == NULL)
            {
                err = -ENOMEM;
                goto cleanup;
            }
            ginfo->metricnames = realloc(ginfo->metricnames, (ginfo->nmetrics + 1) * sizeof(char*));
            if (ginfo->metricnames == NULL)
            {
                err = -ENOMEM;
                goto cleanup;
            }
            
            i = strlen(buf)-1;
            e = i;
            
            while (buf[i] != ' ') {i--;}
            while (buf[i] == ' ') {i++;}
            s = i;
            ginfo->metricformulas[ginfo->nmetrics] = malloc((e-s+1) * sizeof(char));
            if (ginfo->metricformulas[ginfo->nmetrics] == NULL)
            {
                err = -ENOMEM;
                goto cleanup;
            }
            sprintf(ginfo->metricformulas[ginfo->nmetrics], "%.*s", e-s, &(buf[s]));
            i = s;
            while(buf[i] == ' ') {i--;}
            e = i-1;
            while (buf[e] == ' ') {e--;}
            e++;
            ginfo->metricnames[ginfo->nmetrics] = malloc((e+1) * sizeof(char));
            if (ginfo->metricnames[ginfo->nmetrics] == NULL)
            {
                err = -ENOMEM;
                goto cleanup;
            }
            sprintf(ginfo->metricnames[ginfo->nmetrics], "%.*s", e, buf);
            
            ginfo->nmetrics++;
            continue;
        }
        else if (sec == GROUP_LONG)
        {
            s = (ginfo->longinfo == NULL ? 0 : strlen(ginfo->longinfo));
            ginfo->longinfo = realloc(ginfo->longinfo, (s + strlen(buf) + 3) * sizeof(char));
            if (ginfo->longinfo == NULL)
            {
                err = -ENOMEM;
                goto cleanup;
            }
            //strncpy(&(ginfo->longinfo[s]), buf, strlen(buf));
            sprintf(&(ginfo->longinfo[s]), "%.*s", (int)strlen(buf), buf);
            continue;
        }
    }
    fclose(fp);
    free(fullpath);
    return 0;
cleanup:
    free(fullpath);
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
    if ((clist == NULL)||(counter == NULL))
        return -EINVAL;
    clist->cnames = realloc(clist->cnames, (clist->counters + 1) * sizeof(char*));
    if (clist->cnames == NULL)
    {
        return -ENOMEM;
    }
    clist->cvalues = realloc(clist->cvalues, (clist->counters + 1) * sizeof(double));
    if (clist->cvalues == NULL)
    {
        return -ENOMEM;
    }
    clist->cnames[clist->counters] = malloc((strlen(counter)+2)*sizeof(char));
    if (clist->cnames[clist->counters] == NULL)
    {
        return -ENOMEM;
    }
    //strncpy(clist->cnames[clist->counters], counter, strlen(counter));
    sprintf(clist->cnames[clist->counters],"%s", counter);
    //clist->cnames[clist->counters][strlen(counter)] = '\0';
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
    int i,l_cname, l_remain, l_str, l_value;
    char* fcopy, *fcopy2;
    char vcopy[512];
    int size;
    char* ptr, *copyptr;

    if ((formula == NULL) || (clist == NULL))
        return (double)-EINVAL;
    // get current string length
    size = strlen(formula);
    // iterate over counter list and extend string length if stringified counter
    // value is longer as the counter name
    for (i=0;i<clist->counters;i++)
    {
        sprintf(vcopy, "%.20f", clist->cvalues[i]);
        size += (strlen(vcopy) > strlen(clist->cnames[i]) ? strlen(vcopy)-strlen(clist->cnames[i]) : 0);
    }
    // alloc a new string where is enough space
    fcopy = malloc(size * sizeof(char));
    if (fcopy == NULL)
        return (double)-ENOMEM;
    fcopy2 = malloc(size * sizeof(char));
    if (fcopy2 == NULL)
    {
        free(fcopy);
        return (double)-ENOMEM;
    }
    // copy orginal formula to new space
    strcpy(fcopy, formula);

    // try to replace each counter name in clist
    for(i=0;i<clist->counters;i++)
    {
        // if we find the counter name, replace it with the value
        while ((ptr = strstr(fcopy,clist->cnames[i])) != NULL)
        {
            l_cname = strlen(clist->cnames[i]);
            l_str = strlen(fcopy);
            l_remain = strlen(ptr);
            l_value = sprintf(vcopy, "%.20f", clist->cvalues[i]);
            
            // do the replacement
            sprintf(fcopy2, "%.*s%s%s", l_str-l_remain, fcopy, vcopy, ptr+l_cname);
            strcpy(fcopy, fcopy2);
        }
    }
    ptr = strpbrk(fcopy, "aAbBcCdDfFgGhHiIjJkKlLmMnNoOpPqQrRsStTuUvVwWxXyYzZ,_:;!'§$%&=?^°´`#<>");
    if (ptr != NULL)
    {
        fprintf(stderr, "Not all counter names in formula can be substituted\n");
        fprintf(stderr, "%s\n", fcopy);
        free(fcopy);
        free(fcopy2);
        return 0.0;
    }
    // now we can calculate the formula
    i = calculate_infix(fcopy, result);
    free(fcopy);
    free(fcopy2);
    return i;
}


/*int main(void)
{
    int i, size;
    groupInfo ginfo;
    counterList clist;
    char** glist;
    char** ilist;
    char** llist;
    char* estr;
    double result1 = 0, result2 = 0;
    size = get_groups("/home/rrze/unrz/unrz139/Work/likwid/groups", "haswell", &glist, &ilist, &llist);
    for (i=0; i<size; i++)
    {
        read_group("/home/rrze/unrz/unrz139/Work/likwid/groups", "haswell", glist[i], &ginfo);
        estr = get_shortInfo(&ginfo);
        printf("%s: %s\n", glist[i], estr);
        free(estr);
        return_group(&ginfo);
    }
    free(glist);
    read_group("/home/rrze/unrz/unrz139/Work/likwid/groups", "haswell", "L3", &ginfo);
    estr = get_eventStr(&ginfo);
    printf("Eventstr: %s\n", estr);
    free(estr);
    estr = get_shortInfo(&ginfo);
    printf("Short Info: %s\n", estr);
    free(estr);
    init_clist(&clist);
    for (i=0; i<ginfo.nevents; i++)
    {
        add_to_clist(&clist, ginfo.counters[i], i+100);
    }
    add_to_clist(&clist, "time", 1.004);
    add_to_clist(&clist, "inverseClock", 0.0000042);
    for (i=0;i<ginfo.nmetrics;i++)
    {
        result2 = calc_metric(ginfo.metricformulas[i], &clist);
        printf("%s: %g\n", ginfo.metricnames[i], result2);
    }
    destroy_clist(&clist);
    return_group(&ginfo);
    
    
    return 0;
}*/
