/*
 * =======================================================================================
 *
 *      Filename:  configuration.c
 *
 *      Description:  Configuration file module.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2016 RRZE, University Erlangen-Nuremberg
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
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>

#include <configuration.h>

/* #####   EXPORTED VARIABLES   ########################################### */

Likwid_Configuration config = {NULL,NULL,NULL,NULL,-1,MAX_NUM_THREADS,MAX_NUM_NODES};
int init_config = 0;

/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

//static int daemonPath_len = 0;
static int groupPath_len = 0;

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

static int
default_configuration(void)
{
    int ret = 0;
    char filename[1024] = { [0 ... 1023] = '\0' };
    char *fptr = NULL;
    size_t len = 0;
    filename[0] = '\0';

    groupPath_len = strlen(TOSTRING(GROUPPATH))+10;
    config.groupPath = malloc(groupPath_len+1);
    ret = snprintf(config.groupPath, groupPath_len, "%s", TOSTRING(GROUPPATH));
    config.groupPath[ret] = '\0';
#ifndef LIKWID_USE_PERFEVENT
    if (ACCESSMODE == 0)
    {
        config.daemonMode = ACCESSMODE_DIRECT;
        init_config = 1;
        return 0;
    }
    config.daemonMode = ACCESSMODE_DAEMON;

    FILE* fp = popen("bash --noprofile -c \"which likwid-accessD 2>/dev/null | tr -d '\n'\"","r");
    if (fp == NULL)
    {
        goto use_hardcoded;
    }
    ret = getline(&fptr, &len, fp);
    pclose(fp);
    if (ret < 0)
    {
        if (fptr)
            free(fptr);
        goto use_hardcoded;
    }
    if (!access(fptr, X_OK))
    {
        config.daemonPath = (char*)malloc((len+1) * sizeof(char));
        strncpy(config.daemonPath, fptr, len);
        config.daemonPath[len] = '\0';
        if (fptr)
            free(fptr);
    }
    else
    {
        fprintf(stderr, "Found access daemon at %s but it is not executable, using compiled in daemon path.\n", fptr);
        if (fptr)
            free(fptr);
        goto use_hardcoded;
    }
#else
    config.daemonMode = ACCESSMODE_PERF;
#endif
    init_config = 1;
    return 0;
use_hardcoded:
    ret = sprintf(filename,"%s", TOSTRING(ACCESSDAEMON));
    filename[ret] = '\0';
    if (!access(filename, X_OK))
    {
        config.daemonPath = (char*)malloc((strlen(filename)+1) * sizeof(char));
        strcpy(config.daemonPath, filename);
        init_config = 1;
    }
    else
    {
        if (getenv("LIKWID_NO_ACCESS") == NULL)
        {
            ERROR_PLAIN_PRINT(Unable to get path to access daemon. Maybe your PATH environment variable does not contain the folder where you installed it or the file was moved away / not copied to that location?);
            return -1;
        }
    }
    return 0;
}

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

int
init_configuration(void)
{
    if (init_config == 1)
    {
        return 0;
    }

    FILE* fp = NULL;
    char line[512];
    char name[128];
    char value[256];
    char filename[1024];

    filename[0] = '\0';
    char* customtopo = getenv("LIKWID_TOPO_FILE");
    if (customtopo && !access(customtopo, R_OK))
    {
        sprintf(filename, "%s", customtopo);
    }
    else if (!access(TOSTRING(INSTALL_PREFIX, TOPOFILE), R_OK))
    {
        sprintf(filename, "%s", TOSTRING(INSTALL_PREFIX, TOPOFILE));
    }
    else if (!access(TOSTRING(TOPOFILE), R_OK))
    {
        sprintf(filename, "%s", TOSTRING(TOPOFILE));
    }
    if (filename[0] != '\0')
    {
        if (1023 == strlen(filename) && access(filename, R_OK))
        {
            ERROR_PLAIN_PRINT(Topology file path too long for internal buffer);
            return -1;
        }
        config.topologyCfgFileName = (char*)malloc((strlen(filename)+1) * sizeof(char));
        stpcpy(config.topologyCfgFileName, filename);
    }

    filename[0] = '\0';
    char* customcfg = getenv("LIKWID_CFG_FILE");
    if (customcfg && !access(customcfg, R_OK))
    {
        snprintf(filename, 1024, "%s", customcfg);
    }
    else if (!access(TOSTRING(INSTALL_PREFIX, CFGFILE), R_OK))
    {
        snprintf(filename, 1024, "%s", TOSTRING(INSTALL_PREFIX, CFGFILE));
    }
    else if (!access(TOSTRING(CFGFILE), R_OK))
    {
        snprintf(filename, 1024, "%s", TOSTRING(CFGFILE));
    }
    if (filename[0] != '\0')
    {
        if (1023 == strlen(filename) && access(filename, R_OK))
        {
            ERROR_PLAIN_PRINT(Config file path too long for internal buffer);
            if (config.topologyCfgFileName) free(config.topologyCfgFileName);
            return -1;
        }
        config.configFileName = (char*)malloc((strlen(filename)+1) * sizeof(char));
        stpcpy(config.configFileName, filename);
    }
    else
    {
        return default_configuration();
    }

    DEBUG_PRINT(DEBUGLEV_INFO, Reading configuration from %s, config.configFileName)
    fp = fopen(config.configFileName, "r");
    while (fgets(line, 512, fp) != NULL) {
        if (sscanf(line,"%s = %s", name, value) != 2)
        {
            continue;
        }
        if (strncmp(name, "#", 1) == 0)
        {
            continue;
        }
        if (strcmp(name, "topology_file") == 0 && /*don't overrule user request*/!(customtopo && !access(customtopo, R_OK)))
        {
            config.topologyCfgFileName = (char*)malloc((strlen(value)+1) * sizeof(char));
            stpcpy(config.topologyCfgFileName, value);
        }
        else if (strcmp(name, "daemon_path") == 0)
        {
            config.daemonPath = (char*)malloc((strlen(value)+1) * sizeof(char));
            stpcpy(config.daemonPath, value);
            if (access(config.daemonPath, R_OK))
            {
                if (default_configuration() < 0)
                {
                    ERROR_PLAIN_PRINT(Unable to get path to access daemon);
                    fclose(fp);
                    if (config.topologyCfgFileName) free(config.topologyCfgFileName);
                    if (config.configFileName) free(config.configFileName);
                    if (config.groupPath) free(config.groupPath);
                    return -1;
                }
            }
        }
        else if (strcmp(name, "groupPath") == 0)
        {
            struct stat st;
            stat(value, &st);
            if (S_ISDIR(st.st_mode))
            {
                config.groupPath = (char*)malloc((strlen(value)+1) * sizeof(char));
                stpcpy(config.groupPath, value);
            }
            else
            {
                ERROR_PRINT(Path to group files %s is not a directory, value);
                fclose(fp);
                if (config.topologyCfgFileName) free(config.topologyCfgFileName);
                if (config.configFileName) free(config.configFileName);
                if (config.groupPath) free(config.groupPath);
                return -1;
            }
        }
        else if (strcmp(name, "daemon_mode") == 0)
        {
            if (strcmp(value, "daemon") == 0)
            {
                config.daemonMode = ACCESSMODE_DAEMON;
            }
            else if (strcmp(value, "direct") == 0)
            {
                config.daemonMode = ACCESSMODE_DIRECT;
            }
        }
        else if (strcmp(name, "max_threads") == 0)
        {
            config.maxNumThreads = atoi(value);
        }
        else if (strcmp(name, "max_nodes") == 0)
        {
            config.maxNumNodes = atoi(value);
        }
    }
    fclose(fp);

    init_config = 1;
    return 0;
}

Configuration_t
get_configuration(void)
{
    if (init_config == 1)
    {
        return &config;
    }
    return NULL;
}

int
destroy_configuration(void)
{
    if (init_config == 0)
    {
        return -EFAULT;
    }
    if (config.configFileName != NULL)
    {
        free(config.configFileName);
        config.configFileName = NULL;
    }
    if (config.groupPath != NULL)
    {
        free(config.groupPath);
        config.groupPath = NULL;
    }
    if (config.topologyCfgFileName != NULL)
    {
        free(config.topologyCfgFileName);
        config.topologyCfgFileName = NULL;
    }
    if (config.daemonMode != ACCESSMODE_DIRECT)
    {
        if (config.daemonPath != NULL)
        {
            free(config.daemonPath);
            config.daemonPath = NULL;
        }
    }
    config.daemonMode = -1;
    config.maxNumThreads = MAX_NUM_THREADS;
    config.maxNumNodes = MAX_NUM_NODES;
    init_config = 0;
    return 0;
}

int
config_setGroupPath(const char* path)
{
    int ret = 0;
    struct stat st;
    char* new;
    stat(path, &st);
    if (S_ISDIR(st.st_mode))
    {
        if ((int)(strlen(path)+1) > groupPath_len)
        {
            new = malloc(strlen(path)+1);
            if (new == NULL)
            {
                printf("Cannot allocate space for new group path\n");
                return -ENOMEM;
            }
            ret = sprintf(new, "%s", path);
            new[ret] = '\0';
            if (config.groupPath)
                free(config.groupPath);
            config.groupPath = new;
            groupPath_len = strlen(path);
        }
        else
        {
            ret = snprintf(config.groupPath, groupPath_len, "%s", path);
            config.groupPath[ret] = '\0';
        }
        return 0;
    }
    printf("Given path is no directory\n");
    return -ENOTDIR;
}
