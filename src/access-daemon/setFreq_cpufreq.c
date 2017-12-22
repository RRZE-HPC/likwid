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
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <setFreq.h>

#define AMD_TURBO_MSR 0xC0010015

static char setfiles[3][100] = {"scaling_min_freq", "scaling_max_freq", "scaling_setspeed"};
static char getfiles[3][100] = {"cpuinfo_min_freq", "cpuinfo_max_freq", "cpuinfo_cur_freq"};

static char turbo_step[20];
static char steps[30][20];
static int num_steps = 0;

static char governors[20][30];
static int num_govs = 0;

enum command {
    MINIMUM = 0,
    MAXIMUM = 1,
    TURBO = 2,
    GOVERNER
};

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

static int isTurbo(const int cpu_id)
{
    FILE *f = NULL;
    char buff[256];
    char *rptr = NULL, *sptr = NULL;

    sprintf(buff, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_max_freq", cpu_id);
    f = fopen(buff, "r");
    if (f == NULL)
    {
        fprintf(stderr, "Unable to open path %s for reading\n", buff);
        return 0;
    }
    rptr = fgets(buff, 256, f);
    if (strlen(turbo_step) > 0 && strncmp(turbo_step, rptr, strlen(turbo_step)) == 0)
    {
        return 1;
    }
    return 0;
}

static int isAMD()
{
    unsigned int eax,ebx,ecx,edx;
    eax = 0x0;
    CPUID(eax,ebx,ecx,edx);
    if (ecx == 0x444d4163)
        return 1;
    return 0;
}

static int setAMDTurbo(const int cpu_id, const int turbo)
{
    int ret = 0;
    int fd = 0;
    unsigned long int data = 0x0ULL;
    char buff[256];
    sprintf(buff, "/dev/cpu/%d/msr", cpu_id);
    fd = open(buff, O_RDWR);
    ret = pread(fd, &data, sizeof(unsigned long int), AMD_TURBO_MSR);

    if (turbo)
    {
        data &= ~(1ULL << 25);
    }
    else
    {
        data |= (1ULL << 25);
    }
    ret = pwrite(fd, &data, sizeof(unsigned long int), AMD_TURBO_MSR);
    if (ret != sizeof(unsigned long int))
        return EXIT_FAILURE;
    return EXIT_SUCCESS;
}


static int getAvailFreq(const int cpu_id )
{
    int i, j, k;
    FILE *f = NULL;
    char buff[256];
    char tmp[10];
    char *rptr = NULL, *sptr = NULL;
    unsigned int d = 0;

    sprintf(buff, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_available_frequencies", cpu_id);
    f = fopen(buff, "r");
    if (f == NULL)
    {
        fprintf(stderr, "Unable to open path %s for reading\n", buff);
        return 0;
    }
    rptr = fgets(buff, 256, f);
    if (rptr != NULL)
    {
        sptr = strtok(buff, " ");
        if (sptr != NULL)
        {
            d = strtoul(sptr, NULL, 10);
            snprintf(turbo_step, 19, "%u", d);
        }
        while (sptr != NULL)
        {
            if (sptr != NULL)
            {
                d = strtoul(sptr, NULL, 10);
                if (d == 0)
                    break;
                if (num_steps < 30)
                {
                    snprintf(steps[num_steps], 19, "%u", d);
                    num_steps++;
                }
            }
            sptr = strtok(NULL, " ");
        }
    }
    fclose(f);
    return num_steps;
}

static int getAvailGovs(const int cpu_id )
{
    int i, j, k;
    FILE *f = NULL;
    char cmd[256];
    char buff[256];
    char tmp[10];
    char* eptr = NULL, *rptr = NULL;

    sprintf(buff, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_available_governors", cpu_id);
    f = fopen(buff, "r");
    if (f == NULL)
    {
        fprintf(stderr, "Unable to open path %s for reading\n", buff);
        return 0;
    }
    rptr = fgets(buff, 256, f);
    if (rptr != NULL)
    {
        eptr = strtok(buff, " ");
        snprintf(governors[num_govs], 19, "%s", eptr);
        num_govs++;
        while (eptr != NULL)
        {
            eptr = strtok(NULL, " ");
            if (eptr != NULL && num_govs < 20 && strlen(eptr) > 1)
            {
                snprintf(governors[num_govs], 19, "%s", eptr);
                num_govs++;
            }
        }
    }
/*    if (num_govs > 0 && strlen(turbo_step) > 0 && num_govs < 20)*/
/*    {*/
/*        snprintf(governors[num_govs], 19, "turbo");*/
/*        num_govs++;*/
/*    }*/
    fclose(f);
    return num_govs;
}

static void
help(char *execname)
{
    int nsteps = num_steps, ngovs = num_govs;
    int stepstart = 0;
    fprintf(stderr, "Usage: %s <processorID> <cmd> <frequency|governor> \n",execname);
    fprintf(stderr, "       Valid values for <cmd>:\n");
    fprintf(stderr, "       - min: change minimum ratio limit of frequency\n");
    fprintf(stderr, "       - max: change maximum ratio limit of frequency\n");
    fprintf(stderr, "       - tur: Turn turbo \"on\" or \"off\"\n");
    fprintf(stderr, "       - gov: change governor\n");
    printf("Frequency steps: (Freqs. in kHz)\n");
    if (num_steps == 0)
        nsteps = getAvailFreq(0);

    if ((!isTurbo(0)) && (!isAMD()))
        stepstart = 1;
    for (int s=nsteps-1; s>=stepstart; s--)
        printf("%s ", steps[s]);
    printf("\n");
    printf("Governors:\n");
    if (num_govs == 0)
        ngovs = getAvailGovs(0);
    for (int s=0; s<ngovs; s++)
        printf("%s ", governors[s]);
    printf("\n");
    //printf("%s\n", eptr);
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

static unsigned int
read_freq(char* fstr)
{
    unsigned int freq = strtoul(fstr, NULL, 10);
    printf("%u\n", freq);
    if (freq == 0)
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
do_cpufreq (int argn, char** argv)
{
    int i = 0;
    int cpuid = 0;
    int set_id = -1;
    unsigned int freq = 0;
    int turbo = -1;
    int numCPUs = 0;
    enum command cmd;
    char* gov = NULL;
    char* fpath = NULL;
    FILE* f = NULL;
    int num_steps = 0, num_govs = 0;

    if (argn < 3 || argn > 4)
    {
        help(argv[0]);
        exit(EXIT_FAILURE);
    }

    /* Check for valid CPU */
    cpuid = atoi(argv[1]);
    num_steps = getAvailFreq(cpuid);
    num_govs = getAvailGovs(cpuid);
    numCPUs = get_numCPUs();
    if (cpuid < 0 || cpuid > numCPUs)
    {
        fprintf(stderr, "CPU %d not a valid CPU ID. Range from 0 to %d.\n", cpuid, numCPUs);
        exit(EXIT_FAILURE);
    }

    /* Read in command and argument */
    if (strncmp(argv[2], "tur", 3) == 0)
    {
        cmd = TURBO;
        if (strncmp(argv[3], "0", 1) != 0 && strncmp(argv[3], "1", 1) != 0)
        {
            fprintf(stderr, "Invalid turbo setting %s! Only 0 (off) and 1 (on) allowed\n\n",argv[3]);
            help(argv[0]);
            exit(EXIT_FAILURE);
        }
        turbo = atoi(argv[3]);
        if (turbo < 0 || turbo > 1)
        {
            fprintf(stderr, "Invalid turbo setting %d! Only 0 (off) and 1(on) allowed\n\n",turbo);
            help(argv[0]);
            exit(EXIT_FAILURE);
        }
    }
    else if (strncmp(argv[2], "min", 3) == 0)
    {
        cmd = MINIMUM;
        freq = read_freq(argv[3]);
        if (!valid_freq(freq))
        {
            fprintf(stderr, "Invalid frequency %lu!\n",freq);
            if (freq == read_freq(turbo_step))
            {
                fprintf(stderr, "In order to set the turbo frequency, use tur(bo) option\n");
            }
            fprintf(stderr, "\n\n");
            help(argv[0]);
            exit(EXIT_FAILURE);
        }
    }
    else if (strncmp(argv[2], "max", 3) == 0)
    {
        cmd = MAXIMUM;
        freq = read_freq(argv[3]);
        if (!valid_freq(freq))
        {
            fprintf(stderr, "Invalid frequency %lu!\n",freq);
            if (freq == read_freq(turbo_step))
            {
                fprintf(stderr, "In order to set the turbo frequency, use tur(bo) option\n");
            }
            fprintf(stderr, "\n\n");
            help(argv[0]);
            exit(EXIT_FAILURE);
        }
    }
    else if (strncmp(argv[2], "gov", 3) == 0)
    {
        cmd = GOVERNER;
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
    if (cmd == MINIMUM || cmd == MAXIMUM)
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
/*        if (strncmp(testgov, "userspace", 9) != 0)*/
/*        {*/
/*            fclose(f);*/
/*            snprintf(fpath, 99, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor", cpuid);*/
/*            f = fopen(fpath, "w");*/
/*            if (f == NULL) {*/
/*                fprintf(stderr, "Unable to open path %s for writing\n", fpath);*/
/*                free(fpath);*/
/*                return (EXIT_FAILURE);*/
/*            }*/
/*            fprintf(f,"userspace");*/
/*        }*/
        fclose(f);
    }

    switch(cmd)
    {
        //case SET_CURRENT:
        case MINIMUM:
        case MAXIMUM:
            /* The cmd is also used as index in the setfiles array */
            snprintf(fpath, 99, "/sys/devices/system/cpu/cpu%d/cpufreq/%s", cpuid, setfiles[cmd]);
            f = fopen(fpath, "w");
            if (f == NULL) {
                fprintf(stderr, "Unable to open path %s for writing\n",fpath);
                free(fpath);
                return (EXIT_FAILURE);
            }
            fprintf(f,"%u",freq);
            fclose(f);
            break;
        case TURBO:
            if (!isAMD())
            {
                if (turbo == 0)
                {
                    double fr = 0;
                    snprintf(fpath, 99, "/sys/devices/system/cpu/cpu%d/cpufreq/%s", cpuid, setfiles[MAXIMUM]);
                    f = fopen(fpath, "w");
                    if (f == NULL) {
                        fprintf(stderr, "Unable to open path %s for writing\n",fpath);
                        free(fpath);
                        return (EXIT_FAILURE);
                    }
                    fprintf(f,"%s",steps[1]);
                    fclose(f);
                }
                else
                {
                    snprintf(fpath, 99, "/sys/devices/system/cpu/cpu%d/cpufreq/%s", cpuid, setfiles[MAXIMUM]);
                    f = fopen(fpath, "w");
                    if (f == NULL) {
                        fprintf(stderr, "Unable to open path %s for writing\n",fpath);
                        free(fpath);
                        return (EXIT_FAILURE);
                    }
                    fprintf(f,"%s", turbo_step);
                    fclose(f);
                }
            }
            else
            {
                return setAMDTurbo(cpuid, turbo);
            }
            break;
        case GOVERNER:
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

