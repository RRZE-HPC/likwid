/*
 * =======================================================================================
 *
 *      Filename:  setFreq.c
 *
 *      Description:  Implementation of frequency daemon
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
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

enum cmds {
    SET_MIN = 0,
    SET_MAX = 1,
    SET_CURRENT = 2,
    SET_GOV
};

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

static void
help(char *execname)
{
    fprintf(stderr, "Usage: %s <processorID> <cmd> <frequency|governor> \n",execname);
    fprintf(stderr, "       Valid values for <cmd>:\n");
    fprintf(stderr, "       - cur: change current frequency\n");
    fprintf(stderr, "       - min: change minimal frequency\n");
    fprintf(stderr, "       - max: change maximal frequency\n");
    fprintf(stderr, "       - gov: change governor\n");
}

static int
get_numCPUs()
{
    int cpucount = 0;
    char line[1024];
    FILE* fp = fopen("/proc/cpuinfo","r");
    if (fp != NULL)
    {
        while( fgets(line, 1024, fp) )
        {
            if (strncmp(line, "processor", 9) == 0)
            {
                cpucount++;
            }
        }
        fclose(fp);
    }
    return cpucount;
}

static unsigned long
read_freq(char* fstr)
{
    unsigned long freq = strtoul(fstr, NULL, 10);
    if (freq <= 0)
    {
        fprintf(stderr, "Frequency must be greater than 0.\n");
        exit(EXIT_FAILURE);
    }
    return freq;
}

static int
valid_freq(unsigned long freq)
{
    FILE *f = NULL;
    const char fname[] = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies";
    char delimiter[] = " ";
    char buff[1024];
    char freqstr[25];
    char *ptr = NULL, *eptr = NULL;
    
    snprintf(freqstr, 24, "%lu", freq);
    f = fopen(fname, "r");
    if (f == NULL)
    {
        fprintf(stderr, "Cannot open file %s for reading!\n", fname);
        return 0;
    }
    eptr = fgets(buff, 1024, f);
    if (eptr == NULL)
    {
        fprintf(stderr, "Cannot read content of file %s!\n", fname);
        fclose(f);
        return 0;
    }
    ptr = strtok(buff, delimiter);
    while (ptr != NULL)
    {
        if (strncmp(ptr, freqstr, strlen(ptr)) == 0)
        {
            fclose(f);
            return 1;
        }
        ptr = strtok(NULL, delimiter);
    }
    fclose(f);
    return 0;
}

static int
valid_gov(char* gov)
{
    FILE *f = NULL;
    const char fname[] = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors";
    char delimiter[] = " ";
    char buff[1024];
    char *ptr = NULL, *eptr = NULL;
    
    f = fopen(fname, "r");
    if (f == NULL)
    {
        fprintf(stderr, "Cannot open file %s for reading!\n", fname);
        return 0;
    }
    eptr = fgets(buff, 1024, f);
    if (eptr == NULL)
    {
        fprintf(stderr, "Cannot read content of file %s!\n", fname);
        fclose(f);
        return 0;
    }
    ptr = strtok(buff, delimiter);
    while (ptr != NULL)
    {
        if (strncmp(ptr, gov, strlen(ptr)) == 0)
        {
            fclose(f);
            return 1;
        }
        ptr = strtok(NULL, delimiter);
    }
    fclose(f);
    return 0;
}

/* #####  MAIN FUNCTION DEFINITION   ################## */

int
main (int argn, char** argv)
{
    int i = 0;
    int cpuid = 0;
    int set_id = -1;
    unsigned long freq = 0;
    int numCPUs = 0;
    enum cmds cmd;
    char* gov = NULL;
    char* fpath = NULL;
    FILE* f = NULL;

    if (argn < 3 || argn > 4)
    {
        help(argv[0]);
        exit(EXIT_FAILURE);
    }

    /* Check for valid CPU */
    cpuid = atoi(argv[1]);
    numCPUs = get_numCPUs();
    if (cpuid < 0 || cpuid > numCPUs)
    {
        fprintf(stderr, "CPU %d not a valid CPU ID. Range from 0 to %d.\n", cpuid, numCPUs);
        exit(EXIT_FAILURE);
    }

    /* Read in command and argument */
    if (strncmp(argv[2], "cur", 3) == 0)
    {
        cmd = SET_CURRENT;
        freq = read_freq(argv[3]);
        if (!valid_freq(freq))
        {
            fprintf(stderr, "Invalid frequency %lu!\n\n",freq);
            help(argv[0]);
            exit(EXIT_FAILURE);
        }
    }
    else if (strncmp(argv[2], "min", 3) == 0)
    {
        cmd = SET_MIN;
        freq = read_freq(argv[3]);
        if (!valid_freq(freq))
        {
            fprintf(stderr, "Invalid frequency %lu!\n\n",freq);
            help(argv[0]);
            exit(EXIT_FAILURE);
        }
    }
    else if (strncmp(argv[2], "max", 3) == 0)
    {
        cmd = SET_MAX;
        freq = read_freq(argv[3]);
        if (!valid_freq(freq))
        {
            fprintf(stderr, "Invalid frequency %lu!\n\n",freq);
            help(argv[0]);
            exit(EXIT_FAILURE);
        }
    }
    else if (strncmp(argv[2], "gov", 3) == 0)
    {
        cmd = SET_GOV;
        gov = argv[3];
        /* Only allow specific governors */
        if (!valid_gov(gov))
        {
            fprintf(stderr, "Invalid governor %s!\n\n",gov);
            help(argv[0]);
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        fprintf(stderr, "Unknown command %s!\n\n", argv[2]);
        help(argv[0]);
        exit(EXIT_FAILURE);
    }

    fpath = malloc(100 * sizeof(char));
    if (!fpath)
    {
        fprintf(stderr, "Unable to allocate space!\n\n");
        exit(EXIT_FAILURE);
    }

    /* If the current frequency should be set we have to make sure that the governor is
     * 'userspace'. Minimal and maximal frequency are possible for other governors but
     * they dynamically adjust the current clock speed.
     */
    if (cmd == SET_CURRENT)
    {
        int tmp = 0;
        char testgov[1024];
        snprintf(fpath, 99, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor", cpuid);
        f = fopen(fpath, "r");
        if (f == NULL) {
            fprintf(stderr, "Unable to open path %s for reading\n",fpath);
            free(fpath);
            return (EXIT_FAILURE);
        }
        tmp = fread(testgov, 100, sizeof(char), f);
        if (strncmp(testgov, "userspace", 9) != 0)
        {
            fclose(f);
            snprintf(fpath, 99, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor", cpuid);
            f = fopen(fpath, "w");
            if (f == NULL) {
                fprintf(stderr, "Unable to open path %s for writing\n", fpath);
                free(fpath);
                return (EXIT_FAILURE);
            }
            fprintf(f,"userspace");
        }
        fclose(f);
    }

    switch(cmd)
    {
        case SET_CURRENT:
        case SET_MIN:
        case SET_MAX:
            /* The cmd is also used as index in the setfiles array */
            snprintf(fpath, 99, "/sys/devices/system/cpu/cpu%d/cpufreq/%s", cpuid, setfiles[cmd]);
            f = fopen(fpath, "w");
            if (f == NULL) {
                fprintf(stderr, "Unable to open path %s for writing\n",fpath);
                free(fpath);
                return (EXIT_FAILURE);
            }
            fprintf(f,"%d",freq);
            fclose(f);
            break;
        case SET_GOV:
            snprintf(fpath, 99, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor", cpuid);
            f = fopen(fpath, "w");
            if (f == NULL) {
                fprintf(stderr, "Unable to open path %s for writing\n", fpath);
                free(fpath);
                return (EXIT_FAILURE);
            }
            fprintf(f,"%s",gov);
            fclose(f);
            break;
    }
    
    free(fpath);
    return EXIT_SUCCESS;

}

