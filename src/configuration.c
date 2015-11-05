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



#include <configuration.h>

Configuration config = {NULL,NULL,NULL,-1,MAX_NUM_THREADS,MAX_NUM_NODES};
int init_config = 0;

static int default_configuration(void)
{
    int ret;
    char filename[1024];
    char *fptr;
    size_t len = 0;
    filename[0] = '\0';
    if (ACCESSMODE == 0)
    {
        config.daemonMode = ACCESSMODE_DIRECT;
        init_config = 1;
        return 0;
    }
    config.daemonMode = ACCESSMODE_DAEMON;
    FILE* fp = popen("which likwid-accessD 2>/dev/null | tr -d '\n'","r");
    if (fp == NULL)
    {
        goto use_hardcoded;
    }
    ret = getline(&fptr, &len, fp);
    if (ret < 0)
    {
        fclose(fp);
        goto use_hardcoded;
    }
    config.daemonPath = (char*)malloc((len+1) * sizeof(char));
    strncpy(config.daemonPath, fptr, len);
    config.daemonPath[len] = '\0';
    init_config = 1;
    fclose(fp);
    return 0;
use_hardcoded:
    ret = sprintf(filename,"%s", TOSTRING(ACCESSDAEMON));
    filename[ret] = '\0';
    if (!access(filename, R_OK))
    {
        config.daemonPath = (char*)malloc((strlen(filename)+1) * sizeof(char));
        strcpy(config.daemonPath, filename);
        init_config = 1;
    }
    else
    {
        ERROR_PLAIN_PRINT(Unable to get path to access daemon);
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
    sprintf(preconfigured, "%s%s",TOSTRING(INSTALL_PREFIX),"/etc/likwid.cfg");

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
        if (!access("/etc/likwid_topo.cfg", R_OK))
        {
            preconfigured[0] = '\0';
            sprintf(preconfigured,"%s", "/etc/likwid_topo.cfg");
        }
        else
        {
            sprintf(preconfigured, "%s%s",TOSTRING(INSTALL_PREFIX),"/etc/likwid_topo.cfg");
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
        return default_configuration();
    }

    while (fgets(line, 512, fp) != NULL) {
        if (sscanf(line,"%s = %s", name, value) != 2)
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
    if (config.topologyCfgFileName != NULL)
    {
        free(config.topologyCfgFileName);
    }
    if (config.daemonPath != NULL)
    {
        free(config.daemonPath);
    }
    init_config = 0;
    return 0;
}
