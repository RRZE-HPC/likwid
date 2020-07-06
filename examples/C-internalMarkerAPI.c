/*
 * ==========================================================================
 *
 *      Filename:  C-internalMarkerAPI.c
 *
 *      Description:  Example how to use the C/C++ Marker API internally.
 *                    Avoids the likwid-perfctr CLI for setting environment
 *                    variables, pinning threads, and inspecting results.
 *
 *      Version:   1.0.0
 *      Released:  2020-07-06
 *
 *      Author:   Riley Weber, rileyw13@protonmail.com
 *      Project:  likwid
 *
 *
 *      This program is free software: you can redistribute it and/or modify it
 *      under the terms of the GNU General Public License as published by the
 *      Free Software Foundation, either version 3 of the License, or (at your
 *      option) any later version.
 *
 *      This program is distributed in the hope that it will be useful, but
 *      WITHOUT ANY WARRANTY; without even the implied warranty of
 *      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *      General Public License for more details.
 *
 *      You should have received a copy of the GNU General Public License along
 *      with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * ==========================================================================
 *
 *      Usage: 
 *
 *      After installing likwid, change to the `examples` directory. Then, use
 *      `make C-internalMarkerAPI` to compile and 
 *      `make C-internalMarkerAPI-run` to run. 
 *
 * ==========================================================================
 */

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

/* simple function designed to perform memory operations */
void do_copy(double *arr, double *copy_arr, size_t n, ull num_copies) {
  for (ull iter = 0; iter < num_copies; iter++) {
    for (size_t i = 0; i < n; i++) {
      copy_arr[i] = arr[i];
    }
  }
}

/* simple function designed to do floating point computations */
double do_flops(double a, double b, double c, ull num_flops) {
  for (ull i = 0; i < num_flops; i++) {
    c = a * b + c;
  }
  return c;
}

int main(int argc, char* argv[])
{
    /* ====== Begin setting environment variables ====== The envrionment
     * variables used to configure are typically set at runtime with a command
     * like:
     *
     * LIKWID_EVENTS="L2|L3" LIKWID_THREADS="0,2,3" LIKWID_FILEPATH="/tmp/likwid_marker.out" LIKWID_ACCESSMODE="1" LIKWID_FORCE="1" ./C-internalMarkerAPI
     *
     * They are set here to ensure completeness of this example. If you 
     * specify environment variables at runtime, be sure to pin threads with 
     * likwid-pin, GOMP_CPU_AFFINITY, or similar.
     */

    /* The first envrionment variable is "LIKWID_EVENTS", which indicates the
     * groups to be measured. In this case, the first 3 are group names
     * specified in `/<LIKWID_PREFIX>/share/likwid/perfgroups/<ARCHITECTURE>/`
     * (default prefix is `/usr/local`). The final group is a custom group
     * specified with EVENT_NAME:COUNTER_NAME
     *
     * Be aware that the groups chosen below are defined for most but not all
     * architectures supported by likwid. For a more compatible set of groups,
     * see the commented group set. The second, more compatible set will work
     * for all architectures supported by likwid except nvidiagpus and Xeon Phi
     * (KNC). 
     */
    setenv("LIKWID_EVENTS", "L2|FLOPS_DP|INSTR_RETIRED_ANY:FIXC0", 1);
    // setenv("LIKWID_EVENTS", "BRANCH|INSTR_RETIRED_ANY:FIXC0", 1);

    /* LIKWID_THREADS must be set to the list of hardware threads that we will
     * use 
     */
    setenv("LIKWID_THREADS", "0,2,3", 1);

    /* This array is not used by likwid but is used to pin threads below. It
     * must match the string set in LIKWID_THREADS above
     */
    int cpus[NUM_THREADS] =  {0,2,3};

    /* the location the marker file will be stored */
    setenv("LIKWID_FILEPATH", "/tmp/likwid_marker.out", 1);

    /* 1 is the code for access daemon */
    setenv("LIKWID_MODE", "1", 1);

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
     */
    // setenv("LIKWID_DEBUG", "3", 1);

    /* ====== End setting environment variables ====== */

    /* Calls perfmon_init() and perfmon_addEventSet. Uses environment variables
     * set above to configure likwid
     */
    LIKWID_MARKER_INIT;
    
    /* Virtual threads must be pinned to a physical thread. This is
     * demonstrated below. Alternatively, threads may be pinned at runtime
     * using likwid-pin or similar. If GNU openmp is used, threads may be
     * pinned by setting the GOMP_CPU_AFFINITY environment variable to the same
     * cpus specified in LIKWID_THREADS. E.g. if LIKWID_THREADS="0,1,3" then
     * GOMP_CPU_AFFINITY should be set to "0 1 3"
     *
     * Be aware that in the case of openMP, threads will be sequentially
     * numbered from 0 even if that does not correspond the physical thread
     * number. This is handy because likwid follows the same convention of
     * numbering threads sequentially despite the ID of the hardware thread
     * they are actually pinned to.
     */
    omp_set_num_threads(NUM_THREADS);
#pragma omp parallel
{
    likwid_pinThread(cpus[omp_get_thread_num()]);

    /* LIKWID_MARKER_THREADINIT was required with past versions of likwid but
     * now is now commonly not needed and is, in fact, deprecated with likwid
     * v5.0.1
     *
     * It is only required if the pinning library fails and there is a risk of
     * threads getting migrated. I am currently unaware of any runtime system
     * that doesn't work. 
     */ 
    // LIKWID_MARKER_THREADINIT;

    /* Registering regions is optional but strongly recommended, as it reduces
     * overhead of LIKWID_MARKER_START and prevents wrong counts in short
     * regions.
     *
     * There must be a barrier between registering a region and starting that
     * region. Typically these are done in separate parallel blocks, relying on
     * the implicit barrier at the end of the parallel block. Usually there is
     * a parallel block for initialization and a parallel block for execution.
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

    /* Variables for inspecting events */
    int gid;
    char * group_name, * event_name;
    
    /* Loop iterators */
    int i, k, t;

    /* The code that is to be measured will be run multiple times to measure
     * each group specified above. Using perfmon_getNumberOfGroups to get the
     * number of iterations makes it easy to run the computations once for each
     * group.
     */
    for (i=0; i<perfmon_getNumberOfGroups(); i++)
    {
        /* Barriers are necessary when regions are started or stopped more than
         * once in a region. In this case, put a barrier before each call to
         * LIKWID_MARKER_START and after each call to LIKWID_MARKER_STOP.
         */
        #pragma omp barrier

        /* Starting and stopping regions should be done in a parallel block. If
         * regions are started/stopped in a serial region, only the master
         * thread will be measured.
         */

        /* This region will measure everything we do */
        LIKWID_MARKER_START("Total");

        #pragma omp barrier
        /* This region will measure flop-heavy computation */
        LIKWID_MARKER_START("calc_flops");

        /* Key region of code to measure */
        c = do_flops(a, b, c, NUM_FLOPS);

        /* Done measuring flop-heavy code */
        LIKWID_MARKER_STOP("calc_flops");

        #pragma omp barrier

        /* LIKWID_MARKER_GET can give us some information about what happened
         * in a region. The values of each event are stored in `events` and
         * the number of events is stored in `nr_events`. We will inspect and
         * reset events here, in the middle of computation, but regions may
         * also be inspected after the marker API has been closed. This is
         * demonstrated at the end of this file.
         */
        LIKWID_MARKER_GET("calc_flops", &nr_events, events, &time, &count);

        /* Group ID will let us get group and event names */
        gid = perfmon_getIdOfActiveGroup();

        /* Get group name */
        group_name = perfmon_getGroupName(gid);

        /* Print basic info */
        printf("calc_flops iteration %d, thread %d finished measuring group "
            "%s.\n"
            "Got %d events, runtime %f s, and call count %d\n", 
            i, omp_get_thread_num(), group_name, nr_events, time, count);

        /* Only allow one thread to print so that output is less verbose. The
         * barrier prevents garbled output.
         */
        #pragma omp barrier
        if(omp_get_thread_num() == 0)
        {
            /* Uncomment the for loop if you'd like to inspect all threads. */
            t=0;
            // for (t=0; t<NUM_THREADS; t++)
            {
                printf("detailed event results: \n");
                for(k=0; k<nr_events; k++){
                    /* get event name */
                    event_name = perfmon_getEventName(gid, k);
                    /* print results */
                    printf("%40s: %30f\n", event_name, 
                        perfmon_getResult(gid, k, t)
                    );
                }
            }
            printf("\n");
        }

        /* "nr_events" must be reset because LIKWID_MARKER_GET uses it to
         * determine the capacity of "events".
         */
        nr_events = MAX_NUM_EVENTS;

        /* Regions may be reset during execution. Since we have already
         * inspected results of the calc_flops region, we will reset it here:
         */
        LIKWID_MARKER_RESET("calc_flops");

        #pragma omp barrier

        /* This region will measure memory-heavy code */
        LIKWID_MARKER_START("copy");

        /* Do memory ops */
        do_copy(arr, copy_arr, ARRAY_SIZE, NUM_COPIES);

        /* Done measuring memory-heavy code */
        LIKWID_MARKER_STOP("copy");
        #pragma omp barrier

        /* Stop region that measures everything */
        LIKWID_MARKER_STOP("Total");
        #pragma omp barrier

        /* LIKWID_MARKER_SWITCH should only be run by a single thread. If it is
         * called in a parallel region, it must be preceeded by a barrier and
         * run in something like a "#pragma omp single" block to ensure only
         * one thread runs it.
         * 
         * Regions must be switched outside of all regions (e.g. after
         * "LIKWID_MARKER_STOP" is called for each region)
         */
        #pragma omp single
        {
            LIKWID_MARKER_SWITCH;
        }
    }

    /* For demonstration, we will inspect "calc_flops" region again even though
     * it has been rest.
     */
    if(omp_get_thread_num() == 0)
        printf("notice that calc_flops has no calls, since we reset it.\n");
    LIKWID_MARKER_GET("calc_flops", &nr_events, events, &time, &count);
    printf("Region calc_flops, thread %d got %d events, runtime %f s, call "
        "count %d\n", omp_get_thread_num(), nr_events, time, count);
    nr_events = MAX_NUM_EVENTS;

    /* Basic info on "copy" region */
    #pragma omp barrier /* to prevent garbled output */
    LIKWID_MARKER_GET("copy", &nr_events, events, &time, &count);
    printf("Region copy, thread %d got %d events, runtime %f s, call count "
        "%d\n", omp_get_thread_num(), nr_events, time, count);
    nr_events = MAX_NUM_EVENTS;

    /* Basic info on "Total" region */
    #pragma omp barrier /* to prevent garbled output */
    LIKWID_MARKER_GET("Total", &nr_events, events, &time, &count);
    printf("Region Total, thread %d got %d events, runtime %f s, call count "
        "%d\n", omp_get_thread_num(), nr_events, time, count);

} /* End of parallel region */

    printf("\n");

    /* These computations are meaningless, but printing them is an easy way to
     * ensure the compiler doesn't optimize out the do_flops and do_copy
     * functions. Alternatively, those functions may be placed in another file.
     * Another solution is to declare "c" and "copy_arr" as volatile, though
     * that may negatively impact performance and is therefore not recommended.
     */
    printf("final result of flops operations: %f\n", c);
    printf("entry %d of copy_arr: %f\n", ((unsigned)c) % ARRAY_SIZE,
        copy_arr[((unsigned)c) % ARRAY_SIZE]);
    printf("\n");

    /* Stops performance monitoring and writes to the file specified in the
     * LIKWID_FILEPATH environment variable. We will read this using likwid to
     * view results. 
     */
    LIKWID_MARKER_CLOSE;

    const char *region_name, *group_name, *event_name, *metric_name;
    double event_value, metric_value;
    int gid = 0;
    int t, i, k;

    /* Read file output by likwid so that we can process results */
    perfmon_readMarkerFile(getenv("LIKWID_FILEPATH"));

    /* Get information like region name, number of events, and number of
     * metrics. Notice that number of regions printed here is actually
     * (num_regions*num_groups), because perfmon considers a region to be the
     * region/group combo. In other words, each time a region is measured with
     * a different group or event set, perfmon considers it a new region.
     *
     * Therefore, if we have two regions and 3 groups measured for each,
     * perfmon_getNumberOfRegions() will return 6.
     */
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
    printf("detailed results follow. Notice that the region \"calc_flops\"\n"
        "will not appear, as it was reset after each time it was measured.\n");
    printf("\n");

    const char * result_header = "%6s : %15s : %10s : %6s : %40s : %30s \n";
    const char * result_format = "%6d : %15s : %10s : %6s : %40s : %30f \n";
    printf(result_header, "thread", "region", "group", "type", 
        "result name", "result value");

    /* Uncomment the for loop if you'd like to inspect all threads */
    t = 0;
    // for (t = 0; t < NUM_THREADS; t++) 
    {
        for (i = 0; i < perfmon_getNumberOfRegions(); i++) 
        {
            /* Returns the user-supplied region name */
            region_name = perfmon_getTagOfRegion(i);

            /* gid is the group ID independent of region, where as i is the ID
             * of the region/group combo
             */
            gid = perfmon_getGroupOfRegion(i);
            /* Get the name of the group measured, like "FLOPS_DP" or "L2" */
            group_name = perfmon_getGroupName(gid);

            /* Get info for each event */
            for (k = 0; k < perfmon_getNumberOfEvents(gid); k++) 
            {
                /* Get the event name, like "INSTR_RETIRED_ANY" */
                event_name = perfmon_getEventName(gid, k);
                /* Get the associated value */
                event_value = perfmon_getResultOfRegionThread(i, k, t);

                printf(result_format, t, region_name, group_name, "event", 
                    event_name, event_value);
            }

            /* Get info for each metric */
            for (k = 0; k < perfmon_getNumberOfMetrics(gid); k++) 
            {
                /* Get the metric name, like "L2 bandwidth [MBytes/s]" */
                metric_name = perfmon_getMetricName(gid, k);
                /* Get the associated value */
                metric_value = perfmon_getMetricOfRegionThread(i, k, t);

                printf(result_format, t, region_name, group_name, "metric", 
                    metric_name, metric_value);
            }
        }
    }

    return 0;
}
