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
 *      Author:   Thomas Roehl (tr), thomas.roehl@googlemail.com
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
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>



#include <configuration.h>

Configuration config = {NULL,NULL,NULL,NULL,-1,MAX_NUM_THREADS,MAX_NUM_NODES};
int init_config = 0;

static int daemonPath_len = 0;
static int groupPath_len = 0;

static int default_configuration(void)
{
    int ret = 0;
    char filename[1024] = { [0 ... 1023] = '\0' };
    char *fptr = NULL;
    size_t len = 0;
    filename[0] = '\0';
    if (ACCESSMODE == 0)
    {
        config.daemonMode = ACCESSMODE_DIRECT;
        init_config = 1;
        return 0;
    }
    config.daemonMode = ACCESSMODE_DAEMON;
    
    groupPath_len = strlen(TOSTRING(GROUPPATH))+10;
    config.groupPath = malloc(groupPath_len+1);
    ret = snprintf(config.groupPath, groupPath_len, "%s", TOSTRING(GROUPPATH));
    config.groupPath[ret] = '\0';
    
    
    FILE* fp = popen("which likwid-accessD 2>/dev/null | tr -d '\n'","r");
    if (fp == NULL)
    {
        goto use_hardcoded;
    }
    ret = getline(&fptr, &len, fp);
    if (ret < 0)
    {
        fclose(fp);
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
        fclose(fp);
        if (fptr)
            free(fptr);
        goto use_hardcoded;
    }
    init_config = 1;
    fclose(fp);
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
        ERROR_PLAIN_PRINT(Unable to get path to access daemon. Maybe your PATH environment variable does not contain the folder where you installed it or the file was moved away / not copied to that location?);
        exit(EXIT_FAILURE);
    }
    return 0;
}

int init_configuration(void)
{
    int i;
    FILE* fp;
    char line[512];
    char name[128];
    char value[256];
    char filename[1024];
    filename[0] = '\0';
    char preconfigured[1024];
    preconfigured[0] = '\0';
    if (init_config == 1)
    {
        return 0;
    }
    sprintf(preconfigured, "%s%s",TOSTRING(INSTALL_PREFIX),TOSTRING(CFGFILE));

    if (access(preconfigured, R_OK) != 0)
    {
        if (access(TOSTRING(CFGFILE), R_OK) != 0)
        {
            if (!access("/etc/likwid.cfg",R_OK))
            {
                sprintf(filename,"%s", "/etc/likwid.cfg");
            }
        }
        else
        {
            sprintf(filename,"%s",TOSTRING(CFGFILE));
        }
    }
    else
    {
        sprintf(filename, "%s",preconfigured);
    }
    
    if ((config.topologyCfgFileName == NULL) && (strlen(filename) == 0))
    {
        if (!access(TOSTRING(TOPOFILE), R_OK))
        {
            preconfigured[0] = '\0';
            sprintf(preconfigured,"%s", TOSTRING(TOPOFILE));
        }
        else
        {
            sprintf(preconfigured, "%s%s",TOSTRING(INSTALL_PREFIX),TOSTRING(TOPOFILE));
            if (access(preconfigured, R_OK))
            {
                preconfigured[0] = '\0';
            }
        }
        if (preconfigured[0] != '\0')
        {
            config.topologyCfgFileName = (char*)malloc((strlen(preconfigured)+1) * sizeof(char));
            strcpy(config.topologyCfgFileName, preconfigured);
            config.topologyCfgFileName[strlen(preconfigured)] = '\0';
        }
    }

    if ((strlen(filename) == 0) || (!access(filename, R_OK)))
    {
        return default_configuration();
    }
    DEBUG_PRINT(DEBUGLEV_INFO, Reading configuration from %s, filename);
    // Copy determined config filename to struct
    config.configFileName = malloc((strlen(filename)+1)*sizeof(char));
    strcpy(config.configFileName, filename);
    config.configFileName[strlen(filename)] = '\0';

    fp = fopen(config.configFileName, "r");
    if (fp == NULL)
    {
        DEBUG_PLAIN_PRINT(DEBUGLEV_INFO, Using compile-time configuration)
        return default_configuration();
    }
    DEBUG_PRINT(DEBUGLEV_INFO, Reading configuration from %s, filename)
    while (fgets(line, 512, fp) != NULL) {
        if (sscanf(line,"%s = %s", name, value) != 2)
        {
            continue;
        }
        if (strncmp(name, "#", 1) == 0)
        {
            continue;
        }
        if (strcmp(name, "topology_file") == 0)
        {
            config.topologyCfgFileName = (char*)malloc((strlen(value)+1) * sizeof(char));
            strcpy(config.topologyCfgFileName, value);
            config.topologyCfgFileName[strlen(value)] = '\0';
        }
        else if (strcmp(name, "daemon_path") == 0)
        {
            config.daemonPath = (char*)malloc((strlen(value)+1) * sizeof(char));
            strcpy(config.daemonPath, value);
            config.daemonPath[strlen(value)] = '\0';
            if (access(config.daemonPath, R_OK))
            {
                if (default_configuration() < 0)
                {
                    ERROR_PLAIN_PRINT(Unable to get path to access daemon);
                    exit(EXIT_FAILURE);
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
                strcpy(config.groupPath, value);
                config.groupPath[strlen(value)] = '\0';
            }
            else
            {
                ERROR_PRINT(Path to group files %s is not a directory, value);
                exit(EXIT_FAILURE);
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


    init_config = 1;

    fclose(fp);
    return 0;
}

Configuration_t get_configuration(void)
{
    if (init_config == 1)
    {
        return &config;
    }
    return NULL;
}

int destroy_configuration(void)
{
    if (init_config == 0)
    {
        return -EFAULT;
    }
    if (config.configFileName != NULL)
    {
        free(config.configFileName);
    }
    if (config.groupPath != NULL)
    {
        free(config.groupPath);
    }
    if (config.topologyCfgFileName != NULL)
    {
        free(config.topologyCfgFileName);
    }
    if (config.daemonMode != ACCESSMODE_DIRECT)
    {
        if (config.daemonPath != NULL)
        {
            free(config.daemonPath);
        }
    }
    init_config = 0;
    return 0;
}

int config_setGroupPath(const char* path)
{
    int ret = 0;
    struct stat st;
    char* new;
    stat(path, &st);
    if (S_ISDIR(st.st_mode))
    {
        if (strlen(path)+1 > groupPath_len)
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
