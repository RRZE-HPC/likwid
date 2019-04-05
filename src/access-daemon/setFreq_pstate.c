/*
 * =======================================================================================
 *
 *      Filename:  setFreq_pstate.c
 *
 *      Description:  Implementation of frequency daemon with Intel PState backend
 *
 *      Version:   4.3.4
 *      Released:  05.04.2019
 *
 *      Authors:  Thomas Roehl (tr), thomas.roehl@googlemail.com
 *                Amin Nabikhani, amin.nabikhani@gmail.com
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
#include <stdio.h>
#include <string.h>
#include <dirent.h>
#include <errno.h>
#include <setFreq.h>

static char setfiles[3][100] = {"min_perf_pct", "max_perf_pct","no_turbo"};
static char getfiles[3][100] = {"cpuinfo_min_freq", "cpuinfo_max_freq", "cpuinfo_cur_freq"};
static char governers[20][100];
static unsigned int freqs[100];
static unsigned int percent[100];

enum command {
    MINIMUM = 0,
    MAXIMUM = 1,
    TURBO = 2,
    GOVERNOR
};

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

static void help(char *execname)
{
    fprintf(stderr, "Usage: %s <processorID> <cmd> <frequency|governor> \n",execname);
    fprintf(stderr, "       Valid values for <cmd>:\n");
    fprintf(stderr, "       - min: change minimum ratio limit of frequency\n");
    fprintf(stderr, "       - max: change maximum ratio limit of frequency\n");
    fprintf(stderr, "       - tur: Turn turbo \"on\" or \"off\"\n");
    fprintf(stderr, "       - gov: change governor\n");
}

static int check_driver()
{
    int ret = 1;
    DIR* dir = opendir("/sys/devices/system/cpu/intel_pstate");
    if (ENOENT == errno)
    {
        fprintf(stderr, "\tEXIT WITH ERROR:  intel_pstate is not present!\n");
        ret = 0;
    }

    closedir(dir);
    return ret;
}

static unsigned int getMax()
{
    char line[1024];
    unsigned int maxFreq = 0;
    char* eptr;
    FILE* fp = fopen("/sys/devices/system/cpu/cpufreq/policy0/cpuinfo_max_freq", "r");
    if(fp != NULL)
    {
        eptr = fgets(line, 1024, fp);
        maxFreq = strtoul(line, NULL, 10);
        fclose(fp);
    }
    else
    {
        fprintf(stderr, "\tEXIT WITH ERROR:  Max Freq. could not be read\n");
        exit(EXIT_FAILURE);
    }

    return maxFreq;
}

static unsigned int getCurMax()
{
    char line[1024];
    unsigned int maxFreq = 0;
    char* eptr;
    FILE* fp = fopen("/sys/devices/system/cpu/intel_pstate/max_perf_pct", "r");
    if(fp != NULL)
    {
        eptr = fgets(line, 1024, fp);
        maxFreq = strtoul(line, NULL, 10);
        fclose(fp);
    }
    else
    {
        fprintf(stderr, "\tEXIT WITH ERROR:  Max Freq. could not be read\n");
        exit(EXIT_FAILURE);
    }

    return maxFreq;
}


static unsigned int getMin()
{
    char line[1024];
    unsigned int minFreq = 0;
    char* eptr;
    FILE* fp = fopen("/sys/devices/system/cpu/cpufreq/policy0/cpuinfo_min_freq", "r");
    if(fp != NULL)
    {
        eptr = fgets(line, 1024, fp);
        minFreq = strtoul(line, NULL, 10);
        fclose(fp);
    }
    else
    {
        fprintf(stderr, "\tEXIT WITH ERROR:  Max Freq. could not be read\n");
        exit(EXIT_FAILURE);
    }

    return minFreq;
}

static unsigned int getCurMin()
{
    char line[1024];
    unsigned int minFreq = 0;
    char* eptr;
    FILE* fp = fopen("/sys/devices/system/cpu/intel_pstate/min_perf_pct", "r");
    if(fp != NULL)
    {
        eptr = fgets(line, 1024, fp);
        minFreq = strtoul(line, NULL, 10);
        fclose(fp);
    }
    else
    {
        fprintf(stderr, "\tEXIT WITH ERROR:  Max Freq. could not be read\n");
        exit(EXIT_FAILURE);
    }

    return minFreq;
}

static unsigned int turbo_pct()
{
    char readval[4];
    unsigned int turbo_pct;
    FILE* fp = fopen("/sys/devices/system/cpu/intel_pstate/turbo_pct","r");
    if (fp != NULL)
    {
        while( fgets(readval, 4, fp) )
        {
            turbo_pct = strtoul(readval,NULL,10);
        }
        fclose(fp);
    }
    return turbo_pct;
}

static unsigned int num_pstates()
{
    char readval[4];
    unsigned int num;
    FILE* fp = fopen("/sys/devices/system/cpu/intel_pstate/num_pstates","r");
    if (fp != NULL)
    {
        while( fgets(readval, 4, fp) )
        {
            num = strtoul(readval,NULL,10);
        }
        fclose(fp);
    }
    else
    {
        exit(1);
    }
    return num;
}

static int mode()
{
    char readval[5];
    char tmode;
    FILE* fp = fopen("/sys/devices/system/cpu/intel_pstate/no_turbo","r");
    if (fp != NULL)
    {
        while( fgets(readval, 5, fp) )
        {
            tmode = atoi(readval);
        }
        fclose(fp);
    }
    return tmode;
}


static int getGov()
{
    FILE *f = NULL;
    const char fname[] = "/sys/devices/system/cpu/cpufreq/policy0/scaling_available_governors";
    char delimiter[] = " ";
    char buff[1024];
    char *ptr = NULL, *eptr = NULL;
    unsigned int count = 0;

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
        strcpy(governers[count],ptr);
        ptr = strtok(NULL, delimiter);
        ptr = strtok(ptr, "\n");
        count= count + 1;
    }
    fclose(f);
    return 0;
}

static void steps()
{
    unsigned int minFreq = getMin();
    unsigned int trb = turbo_pct();
    unsigned int maxFreq = getMax();
    unsigned int step = num_pstates();
    int range = 0;

    if(maxFreq != 0)
    {
        int t = mode();
        if (t != 0)
        {
            maxFreq = getMax()/(1+0.01*trb);
        }
    }
    else
    {
        fprintf(stderr, "\tEXIT WITH ERROR:  Max Freq. could not be read\n");
        exit(EXIT_FAILURE);
    }
    if(step != 0)
    {
        range = (maxFreq-minFreq)/step;
        freqs[0] = minFreq;
        freqs[step-1]= maxFreq;
        percent[0] = (minFreq/(float)maxFreq) * 100;
        percent[step-1] = 100;

        for(size_t i=1; i < step-1; i++)
        {
            freqs[i] = minFreq+ i* range;
            percent[i] = (freqs[i]/(float)maxFreq) * 100;
        }
    }
    else
    {
        fprintf(stderr,"\tEXIT WITH ERROR:  # of pstates could not be read");
    }
}

static void throw(char* arg)
{
    unsigned int step = num_pstates();
    unsigned int count = 0;
    help(arg);
    printf("Frequency steps: (Freqs. in kHz)\n");
    for(unsigned int i=0; i < step; i++)
    {
        //printf("\t%.1f\t%u %s\n",1E-6*((double)freqs[i]),percent[i],"%");
        unsigned int t = (freqs[i]/10000)*10000;
        
        printf("%lu ", t);
    }
    printf("\n");
    printf("Governors:\n");
    while (strcmp(governers[count],"") != 0)
    {
        printf("%s ",governers[count]);
        count+=1;
    }
    printf("\n");
}

static int valid_gov(char* gov)
{
    unsigned int count = 0;
    while (strcmp(governers[count],"") != 0)
    {
        if (strncmp(governers[count], gov, strlen(governers[count])) == 0)
        {
            return 1;
        }
        count = count + 1;
    }
    return 0;
}

static int valid_freq(char* freq)
{
    int idx = -1;
    int ret = 0;
    unsigned int step = num_pstates();
    char fstep[20];
    unsigned int f = (unsigned int)(atof(freq)*1000000);
    for (int s=0;s<step;s++)
    {
        if ((freqs[s] >= f-10000) && ((freqs[s] <= f+10000)))
        {
            idx = s;
            break;
        }
/*        memset(fstep, 0, 20*sizeof(char));*/
/*        ret = sprintf(fstep, "%.1f", 1E-6*((double)freqs[s]));*/
/*        fstep[ret] = '\0';*/
/*        if (strcmp(fstep, freq) == 0)*/
/*        {*/
/*            idx = s;*/
/*            break;*/
/*        }*/
    }
    return idx;
}


int
do_pstate (int argn, char** argv)
{
    check_driver();
    steps();
    getGov();
    unsigned int step = num_pstates();
    unsigned int minFreq = freqs[0];
    unsigned int maxFreq = freqs[step-1];
    int frq_pct = -1;
    int idx = -1;
    char* gov = NULL;
    char* freq = NULL;
    FILE *f = NULL;
    enum command key;
    char* fpath = NULL;

    if (argn != 4)
    {
        throw(argv[0]);
        exit(EXIT_FAILURE);
    }
    freq = argv[3];

    if (strncmp(argv[2], "min", 3) == 0)
    {
        key = MINIMUM;
        idx = valid_freq(freq);
        if (idx < 0)
        {
            fprintf(stderr, "Invalid frequency %s!\n\n",freq);
            throw(argv[0]);
            exit(EXIT_FAILURE);
        }
        frq_pct = percent[idx];
    }
    else if (strncmp(argv[2], "max", 3) == 0)
    {
        key = MAXIMUM;
        idx = valid_freq(freq);

        if (idx < 0)
        {
            fprintf(stderr, "Invalid frequency %s!\n\n",freq);
            throw(argv[0]);
            exit(EXIT_FAILURE);
        }
        frq_pct = percent[idx];
    }

    else if (strncmp(argv[2], "gov", 3) == 0)
    {
        key = GOVERNOR;
        gov = argv[3];
        if (!valid_gov(gov))
        {
            fprintf(stderr, "Invalid governor %s!\n\n",gov);
            throw(argv[0]);
            exit(EXIT_FAILURE);
        }
    }
    else if (strncmp(argv[2], "tur", 3) == 0)
    {
        key = TURBO;
        frq_pct = atoi(argv[3]);
        if (frq_pct != 0 && frq_pct != 1)
        {
            fprintf(stderr, "Invalid value for trubo mode: \"%u\"!, the value must be either 0 or 1 \n\n",frq_pct);
            throw(argv[0]);
            exit(EXIT_FAILURE);
        }
        frq_pct = (frq_pct == 1 ? 0 : 1);
    }
    else
    {
        fprintf(stderr, "Unknown command %s!\n\n", argv[1]);
        throw(argv[0]);
        exit(EXIT_FAILURE);
    }

    fpath = malloc(100 * sizeof(char));
    if (!fpath)
    {
        fprintf(stderr, "Unable to allocate space!\n\n");
        exit(EXIT_FAILURE);
    }

redo:
    switch(key)
    {
        case MINIMUM:
        case MAXIMUM:
            snprintf(fpath, 99, "/sys/devices/system/cpu/intel_pstate/%s", setfiles[key]);
            printf("File %s\n", fpath);
            f = fopen(fpath, "w+");
            if (f == NULL) {
                fprintf(stderr, "Unable to open path \"%s\" for writing\n",fpath);
                free(fpath);
                return (EXIT_FAILURE);
            }
            printf("Write percentage %lu\n", frq_pct);
            fprintf(f,"%u",frq_pct);
            fclose(f);
            break;

        case GOVERNOR:
            snprintf(fpath, 99, "/sys/devices/system/cpu/cpu%s/cpufreq/scaling_governor", argv[1]);
            unsigned int bturbo = mode();
            f = fopen(fpath, "w");
            if (f == NULL) {
                fprintf(stderr, "Unable to open path \"%s\" for writing\n", fpath);
                free(fpath);
                return (EXIT_FAILURE);
            }
            fprintf(f,"%s",gov);
            fclose(f);
            unsigned int aturbo = mode();
            if (bturbo != aturbo)
            {
                frq_pct = bturbo;
                key = TURBO;
                goto redo;
            }
            break;

        case TURBO:
            snprintf(fpath, 99, "/sys/devices/system/cpu/intel_pstate/%s", setfiles[key]);
            f = fopen(fpath, "w+");
            if (f == NULL) {
                fprintf(stderr, "Unable to open path \"%s\" for writing\n",fpath);
                free(fpath);
                return (EXIT_FAILURE);
            }
            fprintf(f,"%u",frq_pct);
            fclose(f);
            break;
    }

    return 0;
}

