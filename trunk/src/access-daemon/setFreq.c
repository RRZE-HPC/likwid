#include <stdlib.h>
#include <stdio.h>
#include <string.h>

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

int main (int argn, char** argv)
{
    int cpuid;
    int freq;
    int numCPUs = 0;
    char* gov;
    char* gpath = malloc(100);
    char* fpath = malloc(100);

    if (argn < 3 || argn > 4)
    {
        fprintf(stderr, "Usage: %s <processorID> <frequency> [<governor>] \n",argv[0]);
        exit(EXIT_FAILURE);
    }

    cpuid = atoi(argv[1]);
    numCPUs = get_numCPUs();
    if (cpuid < 0 || cpuid > numCPUs)
    {
        fprintf(stderr, "CPU %d not a valid CPU ID. Range from 0 to %d.\n",cpuid,numCPUs);
        exit(EXIT_FAILURE);
    }
    freq  = atoi(argv[2]);
    if (freq <= 0)
    {
        fprintf(stderr, "Frequency must be greater than 0.\n");
        exit(EXIT_FAILURE);
    }

    if (argn == 4)
    {
        gov = argv[3];

        if ((strncmp(gov,"ondemand",8)) && (strncmp(gov,"performance",11))) {
            fprintf(stderr, "Invalid governor %s!\n",gov);
            return (EXIT_FAILURE);
        }
        snprintf(gpath, 60, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor", cpuid);

        FILE* f = fopen(gpath, "w");
        if (f == NULL) {
            fprintf(stderr, "Unable to open path for writing\n");
            return (EXIT_FAILURE);
        }
        fprintf(f,"%s",gov);
        fclose(f);
        return(EXIT_SUCCESS);
    }

    snprintf(gpath, 60, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor", cpuid);
    snprintf(fpath, 60, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_setspeed", cpuid);

    FILE* f = fopen(gpath, "w");
    if (f == NULL) {
        fprintf(stderr, "Unable to open path for writing\n");
        return (EXIT_FAILURE);
    }
    fprintf(f,"userspace");
    fclose(f);

    f = fopen(fpath, "w");
    if (f == NULL) {
        fprintf(stderr, "Unable to open path for writing\n");
        return (EXIT_FAILURE);
    }
    fprintf(f,"%d",freq);
    fclose(f);

    return(EXIT_SUCCESS);
}


