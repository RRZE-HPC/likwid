/*
 * =======================================================================================
 *
 *      Filename:  setFreq.c
 *
 *      Description:  Implementation of frequency daemon
 *
 *      Version:   4.1
 *      Released:  13.6.2016
 *
 *      Authors:  Jan Treibig (jt), jan.treibig@gmail.com,
 *                Thomas Roehl (tr), thomas.roehl@googlemail.com
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
#include <string.h>

char setfiles[3][100] = {"scaling_min_freq", "scaling_max_freq", "scaling_setspeed"};
char getfiles[3][100] = {"cpuinfo_min_freq", "cpuinfo_max_freq", "cpuinfo_cur_freq"};

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */
static int get_numCPUs()
{
    int cpucount = 0;
    char line[1024];
    FILE* fp = fopen("/proc/cpuinfo","r");
    if (fp != NULL)
    {
        while( fgets(line,1024,fp) )
        {
            if (strncmp(line, "processor", 9) == 0)
            {
                cpucount++;
            }
        }
    }
    return cpucount;
}

/* #####  MAIN FUNCTION DEFINITION   ################## */
int main (int argn, char** argv)
{
    int i = 0;
    int tmp;
    int cpuid;
    int freq = 0;
    int numCPUs = 0;
    char* gov;
    char* gpath = malloc(100);
    char* fpath = malloc(100);

    if (argn < 3 || argn > 4)
    {
        fprintf(stderr, "Usage: %s <processorID> <frequency> [<governor>] \n",argv[0]);
        free(gpath);
        free(fpath);
        exit(EXIT_FAILURE);
    }

    cpuid = atoi(argv[1]);
    numCPUs = get_numCPUs();
    if (cpuid < 0 || cpuid > numCPUs)
    {
        fprintf(stderr, "CPU %d not a valid CPU ID. Range from 0 to %d.\n", cpuid, numCPUs);
        free(gpath);
        free(fpath);
        exit(EXIT_FAILURE);
    }
    freq  = atoi(argv[2]);
    if (freq <= 0)
    {
        fprintf(stderr, "Frequency must be greater than 0.\n");
        free(gpath);
        free(fpath);
        exit(EXIT_FAILURE);
    }

    if (argn == 4)
    {
        FILE* f;
        gov = argv[3];

        if ((strncmp(gov,"ondemand",8) != 0) &&
            (strncmp(gov,"performance",11) != 0) &&
            (strncmp(gov,"conservative",12) != 0) &&
            (strncmp(gov,"powersave",9) != 0)) {
            fprintf(stderr, "Invalid governor %s!\n",gov);
            free(gpath);
            free(fpath);
            return (EXIT_FAILURE);
        }
        
        for (i=0; i<2; i++)
        {
            snprintf(fpath, 99, "/sys/devices/system/cpu/cpu%d/cpufreq/%s", cpuid, getfiles[i]);
            f = fopen(fpath, "r");
            if (f == NULL) {
                fprintf(stderr, "Unable to open path %s for writing\n", fpath);
                free(gpath);
                free(fpath);
                return (EXIT_FAILURE);
            }
            tmp = fread(fpath, 100, sizeof(char), f);
            freq = atoi(fpath);
            fclose(f);
            snprintf(fpath, 99, "/sys/devices/system/cpu/cpu%d/cpufreq/%s", cpuid, setfiles[i]);
            f = fopen(fpath, "w");
            if (f == NULL) {
                fprintf(stderr, "Unable to open path %s for writing\n",fpath);
                free(gpath);
                free(fpath);
                return (EXIT_FAILURE);
            }
            fprintf(f,"%d",freq);
            fclose(f);

        }
        snprintf(gpath, 99, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor", cpuid);

        f = fopen(gpath, "w");
        if (f == NULL) {
            fprintf(stderr, "Unable to open path %s for writing\n", gpath);
            free(gpath);
            free(fpath);
            return (EXIT_FAILURE);
        }
        fprintf(f,"%s",gov);
        fclose(f);
        free(gpath);
        free(fpath);
        return(EXIT_SUCCESS);
    }

    snprintf(gpath, 99, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor", cpuid);

    FILE* f = fopen(gpath, "w");
    if (f == NULL) {
        fprintf(stderr, "Unable to open path %s for writing\n", gpath);
        free(gpath);
        free(fpath);
        return (EXIT_FAILURE);
    }
    if ((argn == 4) &&
        ((strncmp(argv[3],"ondemand",8) == 0) ||
        (strncmp(argv[3],"performance",11) == 0) ||
        (strncmp(argv[3],"conservative",12) == 0) ||
        (strncmp(argv[3],"powersave",9) == 0)))
    {
        fprintf(f, "%s", argv[3]);
        tmp = 1;
    }
    else
    {
        fprintf(f, "%s", "userspace");
        tmp = 3;
    }
    fclose(f);

    for (i=0;i<tmp;i++)
    {
        snprintf(fpath, 99, "/sys/devices/system/cpu/cpu%d/cpufreq/%s", cpuid, setfiles[i]);
        f = fopen(fpath, "w");
        if (f == NULL) {
            fprintf(stderr, "Unable to open path %s for writing\n",fpath);
            free(gpath);
            free(fpath);
            return (EXIT_FAILURE);
        }
        fprintf(f,"%d",freq);
        fclose(f);
    }
    free(gpath);
    free(fpath);
    return(EXIT_SUCCESS);
}


