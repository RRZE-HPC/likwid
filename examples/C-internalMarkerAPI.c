#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <omp.h>

#include <likwid.h>


void dummy()
{
    ;;
}

int main(int argc, char* argv[])
{
    int i, k;
    char group[] = "L3";
    int gid = 0;
    char cpulist[] = "0,1,2";
    int cpus[3] =  {0,1,2};
    char filepath[] = "/tmp/test-marker.out";
    char accessmode[] = "1";
    double *A, *B;
    size_t asize = 1024*1024;
    

    setenv("LIKWID_EVENTS", group, 1);
    setenv("LIKWID_THREADS", cpulist, 1);
    setenv("LIKWID_FILEPATH", filepath, 1);
    setenv("LIKWID_MODE", accessmode, 1);
    /* If the NMI watchdog is enabled or the application does not call
     * perfmon_finalize(), e.g. because of some error, LIKWID will fail with
     * a message "Counter in use". By settings LIKWID_FORCE you can overwrite
     * the registers.
     */
    //setenv("LIKWID_FORCE", "1", 1);
    
    A = malloc(asize * sizeof(double));
    if (A==NULL)
        return 1;
    B = malloc(asize * sizeof(double));
    if (B==NULL)
    {
        free(A);
        return 1;
    }
    for (i=0; i<asize;i++)
        B[i] = ((double)i)+1.5;

    /* This is only for showcase. If your application pins them already, you
     * don't need this
     */
#pragma omp parallel
{
    likwid_pinThread(cpus[omp_get_thread_num()]);
}

    /* Calls perfmon_init() and perfmon_addEventSet */
    LIKWID_MARKER_INIT;
    /* Setup and start manually. We use group ID 0, we can switch later */
    perfmon_setupCounters(0);
    perfmon_startCounters();

    printf("Getting results during the measurements with LIKWID_MARKER_GET\n");
#pragma omp parallel private(k,i)
{
    int nr_events = 20;
    double time = 0;
    int count = 0;
    double *events = malloc(nr_events * sizeof(double));
    memset(events, 0, nr_events * sizeof(double));
    LIKWID_MARKER_START("Total");
    for (k=0; k<10; k++)
    {
        
        LIKWID_MARKER_START("Calc1");
#pragma omp for
        for (i=0; i< asize; i++)
            A[i] = B[i];
        if (A[i] < 0) dummy();
        LIKWID_MARKER_STOP("Calc1");
    }
    LIKWID_MARKER_GET("Calc1", &nr_events, events, &time, &count);
    printf("Calc1 Thread %d got %d events, runtime %f s, call count %d\n", omp_get_thread_num(), nr_events, time, count);
    nr_events = 20;
    memset(events, 0, nr_events * sizeof(double));
    for (k=0; k<10; k++)
    {
        LIKWID_MARKER_START("Calc2");
#pragma omp for
        for (i=0; i< asize; i++)
            A[i] = A[i] + B[i];
        if (A[i] < 0) dummy();
        LIKWID_MARKER_STOP("Calc2");
    }
    LIKWID_MARKER_STOP("Total");
    LIKWID_MARKER_GET("Calc2", &nr_events, events, &time, &count);
    printf("Calc2 Thread %d got %d events, runtime %f s, call count %d\n", omp_get_thread_num(), nr_events, time, count);
    nr_events = 20;
    memset(events, 0, nr_events * sizeof(double));
    LIKWID_MARKER_GET("Total", &nr_events, events, &time, &count);
    printf("Total Thread %d got %d events, runtime %f s, call count %d\n", omp_get_thread_num(), nr_events, time, count);
    free(events);
}



    perfmon_stopCounters();
    LIKWID_MARKER_CLOSE;

    

    perfmon_readMarkerFile(filepath);
    printf("\nMarker API measured %d regions\n", perfmon_getNumberOfRegions());
    for (i=0;i<perfmon_getNumberOfRegions();i++)
    {
        gid = perfmon_getGroupOfRegion(i);
        printf("Region %s with %d events and %d metrics\n",perfmon_getTagOfRegion(i),
                                                           perfmon_getEventsOfRegion(i),
                                                           perfmon_getMetricsOfRegion(i));
    }
    printf("\nExample metrics output for thread 0\n");
    
    
    for (i=0;i<perfmon_getNumberOfRegions();i++)
    {
        printf("Region %s\n", perfmon_getTagOfRegion(i));
        for (k=0;k<perfmon_getEventsOfRegion(i);k++)
            printf("Event %s:%s: %f\n", perfmon_getEventName(gid, k),
                                        perfmon_getCounterName(gid, k),
                                        perfmon_getResultOfRegionThread(i, k, 0));
        for (k=0;k<perfmon_getNumberOfMetrics(gid);k++)
            printf("Metric %s: %f\n", perfmon_getMetricName(gid, k),
                                      perfmon_getMetricOfRegionThread(i, k, 0));
        printf("\n");
    }
    remove(filepath);
    
    /* Reinitialize access to HPM registers, LIKWID_MARKER_CLOSE closed the connection */
    HPMinit();
    for (i=0;i<3; i++)
        HPMaddThread(cpus[i]);
    /* Finalize perfmon sets all used counters to zero and deletes marker results, so no
       perfmon_destroyMarkerResults() required */
    perfmon_finalize();
    HPMfinalize();
    free(A);
    free(B);
    return 0;

}
