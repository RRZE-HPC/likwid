#include <stdlib.h>
#include <stdio.h>
#include <dirent.h>
#include <errno.h>
#include <setFreq.h>


static int is_pstate()
{
    int ret = 1;
    DIR* dir = opendir("/sys/devices/system/cpu/intel_pstate");
    if (ENOENT == errno)
    {
        //fprintf(stderr, "\tEXIT WITH ERROR:  intel_pstate is not present!\n");
        ret = 0;
    }

    closedir(dir);
    return ret;
}


int main(int argc, char** argv)
{
    if (is_pstate())
    {
        printf("Pstate driver\n");
        return do_pstate(argc, argv);
    }
    else
        return do_cpufreq(argc, argv);
}
