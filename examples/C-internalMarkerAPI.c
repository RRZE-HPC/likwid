#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>

#include <omp.h>
#include <likwid.h>

#define NUM_FLOPS 100000000
#define NUM_COPIES 100000
#define NUM_THREADS 3
#define MAX_NUM_EVENTS 20
#define ARRAY_SIZE 2048

typedef unsigned long long ull;

void do_copy(double *arr, double *copy_arr, size_t n, ull num_copies) {
  for (ull iter = 0; iter < num_copies; iter++) {
    for (size_t i = 0; i < n; i++) {
      copy_arr[i] = arr[i];
    }
  }
}

double do_flops(double a, double b, double c, ull num_flops) {
  for (ull i = 0; i < num_flops; i++) {
    c = a * b + c;
  }
  return c;
}

int main(int argc, char* argv[])
{
    /* ====== Begin setting environment variables ====== 
     * These may all be set at runtime (and typically are set at runtime) with
     * a command like:
     * 
     * LIKWID_EVENTS="L2|L3" LIKWID_THREADS="0,2,3" LIKWID_FILEPATH="/tmp/likwid_marker.out" LIKWID_ACCESSMODE="1" LIKWID_FORCE="1" ./C-internalMarkerAPI
     * 
     * they are included here for completeness. If you specify environment
     * variables at runtime, be sure to pin threads with likwid-pin,
     * GOMP_CPU_AFFINITY, or similar
     */

    /* the groups to be measured. In this case, the first 3 are group names
     * specified in `/<LIKWID_PREFIX>/share/likwid/perfgroups/<ARCHITECTURE>/`
     * (default prefix is `/usr/local`). The final group is a custom group
     * specified with EVENT_NAME:COUNTER_NAME
     */
    char groups[] = "L2|L3|FLOPS_DP|INSTR_RETIRED_ANY:FIXC0";

    /* list of processors that we will use */
    char cpulist[] = "0,2,3";

    /* not used by likwid, used to pin threads */
    int cpus[NUM_THREADS] =  {0,2,3};

    /* the location the marker file will be stored */
    char filepath[] = "/tmp/likwid_marker.out";

    /* 1 is the code for access daemon */
    char accessmode[] = "1"; 

    setenv("LIKWID_EVENTS", groups, 1);
    setenv("LIKWID_THREADS", cpulist, 1);
    setenv("LIKWID_FILEPATH", filepath, 1);
    setenv("LIKWID_MODE", accessmode, 1);

    /* If the NMI watchdog is enabled or the application does not call
     * perfmon_finalize(), e.g. because of some error, LIKWID will fail with
     * a message "Counter in use". By settings LIKWID_FORCE you can overwrite
     * the registers.
     */
    setenv("LIKWID_FORCE", "1", 1);

    /* If the user desires more information about what's going on under the
     * hood, set this to get debug information. Values from 0-3 are valid, with
     * 0 being the default (none) and each level from 1-3 being an increased
     * level of verbosity
     * TODO: verify
     */
    // setenv("LIKWID_DEBUG", "3", 1);

    /* ====== End setting environment variables ====== */

    /* Calls perfmon_init() and perfmon_addEventSet */
    LIKWID_MARKER_INIT;
    
    /* Virtual threads must be pinned to a physical thread. This is
     * demonstrated below. Alternatively, threads may be pinned at runtime
     * using likwid-pin or similar. If GNU openmp is used, threads may be
     * pinned by setting the GOMP_CPU_AFFINITY environment variable to the same
     * value as LIKWID_THREADS. Be aware that in the case of openMP, threads
     * will be sequentially numbered from 0 even if that does not correspond
     * the physical thread number.
     * TODO: verify
     */
     omp_set_num_threads(NUM_THREADS);
#pragma omp parallel
{
    likwid_pinThread(cpus[omp_get_thread_num()]);

    /* each thread must be initialized with LIKWID_MARKER_THREADINIT */
    LIKWID_MARKER_THREADINIT;

    /* registering regions is optional but strongly recommended, as it reduces
     * overhead of LIKWID_MARKER_START and prevents wrong counts
     */
    LIKWID_MARKER_REGISTER("Total");
    LIKWID_MARKER_REGISTER("calc_flops");
    LIKWID_MARKER_REGISTER("copy");
}

    /* variables needed for flops/copy computations */
    double a, b, c;
    a = 1.8;
    b = 3.2;
    c = 1.0;
    double arr[ARRAY_SIZE];
    double copy_arr[ARRAY_SIZE];

#pragma omp parallel
{
    /* variables needed for LIKWID_MARKER_GET */
    int nr_events = MAX_NUM_EVENTS;
    double events[MAX_NUM_EVENTS];
    double time = 0;
    int count = 0;
    
    /* loop iterators */
    int i, k;

    /* The code that is to be measured will be run multiple times to measure
     * each group specified above. perfmon_getNumberOfGroups makes it easy to
     * run computation once for each group.
     */
    for (k=0; k<perfmon_getNumberOfGroups(); k++)
    {
        /* barriers are necessary when regions are started or stopped more than
         * once in a region. In this case, put a barrier before each call to
         * LIKWID_MARKER_START and after each call to LIKWID_MARKER_STOP.
         */
        #pragma omp barrier
        /* This region will measure everything we do */
        LIKWID_MARKER_START("Total");

        #pragma omp barrier
        /* this region will measure flop-heavy computation */
        LIKWID_MARKER_START("calc_flops");

        /* key region of code to measure */
        c = do_flops(a, b, c, NUM_FLOPS);

        /* done measuring flop-heavy code */
        LIKWID_MARKER_STOP("calc_flops");

        #pragma omp barrier

        /* this region will measure memory-heavy code */
        LIKWID_MARKER_START("copy");

        /* do memory ops */
        do_copy(arr, copy_arr, ARRAY_SIZE, NUM_COPIES);

        /* done measuring memory-heavy code */
        LIKWID_MARKER_STOP("copy");
        #pragma omp barrier

        /* stop region that measures everything */
        LIKWID_MARKER_STOP("Total");
        #pragma omp barrier

        /* LIKWID_MARKER_SWITCH should only be run by a single thread. If it is
         * called in a parallel region, it must be preceeded by a barrier and
         * run in something like a "#pragma omp single" block to ensure only
         * one thread runs it
         */
        #pragma omp single
        {
            LIKWID_MARKER_SWITCH;
        }
    }

    /* LIKWID_MARKER_GET can give us some information about what happened in
     * that region. The values of each event are stored in `events` and may be
     * inspected here if desired. We choose not to, as we will inspect
     * everything at the end.
     */
    LIKWID_MARKER_GET("calc_flops", &nr_events, events, &time, &count);
    printf("calc_flops Thread %d got %d events, runtime %f s, call count %d\n", 
        omp_get_thread_num(), nr_events, time, count);

    /* "nr_events" must be reset because LIKWID_MARKER_GET uses it as the size
     * of "events"
     */
    nr_events = MAX_NUM_EVENTS;

    /* basic info on "copy" region */
    LIKWID_MARKER_GET("copy", &nr_events, events, &time, &count);
    printf("copy Thread %d got %d events, runtime %f s, call count %d\n", 
        omp_get_thread_num(), nr_events, time, count);
    nr_events = MAX_NUM_EVENTS;

    /* basic info on "Total" region */
    LIKWID_MARKER_GET("Total", &nr_events, events, &time, &count);
    printf("Total Thread %d got %d events, runtime %f s, call count %d\n", 
        omp_get_thread_num(), nr_events, time, count);

} /* end of parallel region */

    printf("\n");

    printf("final result of flops operations: %f\n", c);
    printf("entry %d of copy_arr: %f\n", ((unsigned)c) % ARRAY_SIZE,
        copy_arr[((unsigned)c) % ARRAY_SIZE]);
    printf("\n");

    /* stops performance monitoring and writes to the file specified in the
     * LIKWID_FILEPATH environment variable. We will read this using likwid to
     * view results. 
     */
    LIKWID_MARKER_CLOSE;

    const char *regionName, *groupName, *event_name, *metric_name;
    double event_value, metric_value;
    int gid = 0;
    int t, i, k;

    /* read file output by likwid so that we can process results */
    perfmon_readMarkerFile(getenv("LIKWID_FILEPATH"));

    printf("Marker API measured %d regions\n", perfmon_getNumberOfRegions());
    for (i=0;i<perfmon_getNumberOfRegions();i++)
    {
        gid = perfmon_getGroupOfRegion(i);
        printf("Region %s with %d events and %d metrics\n",
            perfmon_getTagOfRegion(i), perfmon_getEventsOfRegion(i), 
            perfmon_getMetricsOfRegion(i));
    }
    printf("\n");

    /* Print per-thread results. */
    printf("detailed results for each thread follow:\n");

    const char * result_header = "%6s : %15s : %10s : %6s : %40s : %30s \n";
    const char * result_format = "%6d : %15s : %10s : %6s : %40s : %30f \n";
    printf(result_header, "thread", "region", "group", "type", 
        "result name", "result value");

    for (t = 0; t < NUM_THREADS; t++) 
    {
        /* perfmon_getNumberOfRegions actually returns num_regions *
         * num_groups. This is because this function considers every
         * combination of region and group its own region.
         */
        for (i = 0; i < perfmon_getNumberOfRegions(); i++) 
        {
            regionName = perfmon_getTagOfRegion(i);

            /* gid is the group ID independent of region, where as i is the ID
             * of the region/group combo
             */
            gid = perfmon_getGroupOfRegion(i);
            groupName = perfmon_getGroupName(gid);

            /* TODO: would getNumberOfEvents also work? */
            for (k = 0; k < perfmon_getEventsOfRegion(i); k++) 
            {
                event_name = perfmon_getEventName(gid, k);
                event_value = perfmon_getResultOfRegionThread(i, k, t);
                printf(result_format, t, regionName, groupName, "event", 
                    event_name, event_value);
            }

            for (k = 0; k < perfmon_getNumberOfMetrics(gid); k++) 
            {
                metric_name = perfmon_getMetricName(gid, k);
                metric_value = perfmon_getMetricOfRegionThread(i, k, t);
                printf(result_format, t, regionName, groupName, "metric", 
                    metric_name, metric_value);
            }
        }
    }

    return 0;
}
