#include <stdlib.h>
#include <stdio.h>


#include <likwid.h>





int main(int argc, char* argv[])
{
    int gpuId = 0;
    int ret = 0;
    int gid = -1;
    rocmon_setVerbosity(DEBUGLEV_DEVELOP);
    ret = rocmon_init(1, &gpuId);
    if (ret < 0)
    {
        printf("rocmon_init failed with %d\n", ret);
        return ret;
    }
    ret = rocmon_addEventSet("ROCP_SQ_WAVES:ROCM0", &gid);
    if (ret < 0)
    {
        printf("rocmon_addEventSet failed with %d\n", ret);
        rocmon_finalize();
        return ret;
    }
    printf("test_rocmon -- Event set ID %d\n", gid);
    ret = rocmon_setupCounters(gid);
    if (ret < 0)
    {
        printf("rocmon_setupCounters failed with %d\n", ret);
        rocmon_finalize();
        return ret;
    }
    ret = rocmon_startCounters();
    if (ret < 0)
    {
        printf("rocmon_startCounters failed with %d\n", ret);
        rocmon_finalize();
        return ret;
    }
    printf("test_rocmon -- Counters running\n");
    ret = rocmon_readCounters();
    if (ret < 0)
    {
        printf("rocmon_startCounters failed with %d\n", ret);
        rocmon_finalize();
        return ret;
    }
    printf("test_rocmon -- Counters running\n");
    ret = rocmon_readCounters();
    if (ret < 0)
    {
        printf("rocmon_startCounters failed with %d\n", ret);
        rocmon_finalize();
        return ret;
    }
    printf("test_rocmon -- Counters running\n");
    ret = rocmon_stopCounters();
    if (ret < 0)
    {
        printf("rocmon_stopCounters failed with %d\n", ret);
        rocmon_finalize();
        return ret;
    }
    printf("test_rocmon -- Counters stopped\n");
    rocmon_finalize();
    return 0;
}
