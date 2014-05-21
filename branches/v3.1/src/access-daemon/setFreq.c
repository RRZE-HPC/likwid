#include <stdlib.h>
#include <stdio.h>

int main (int argn, char** argv)
{
    int cpuid;
    int freq;
    char* gov;
    char* gpath = malloc(100);
    char* fpath = malloc(100);

    if (argn < 3)
    {
        fprintf(stderr, "Usage: %s <processorID> <frequency> [<governor>] \n",argv[0]);
    }

    cpuid = atoi(argv[1]);
    freq  = atoi(argv[2]);

    if (argn == 4)
    {
        gov = argv[3];

        if ((strncmp(gov,"ondemand",12)) && (strncmp(gov,"performance",12))) {
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


