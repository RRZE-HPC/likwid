/*
 * =======================================================================================
 *
 *      Filename:  C-internalMarkerAPI.c
 *
 *      Description:  Example how to use the C/C++ Marker API internally.
 *                    Avoids the likwid-perfctr CLI for setting environment
 *                    variables, pinning threads, and inspecting results.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
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
 *
 *      Usage:
 *
 *      After installing likwid, change to the `examples` directory. Then, use
 *      `make C-internalMarkerAPI` to compile and
 *      `make C-internalMarkerAPI-run` to run.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>

#include <omp.h>

#include <likwid.h>         // We need the full C API
#include <likwid-marker.h>  // and the MarkerAPI macros.

#define NUM_FLOPS 100000000
#define NUM_COPIES 100000
#define NUM_THREADS 3
#define MAX_NUM_EVENTS 20
#define ARRAY_SIZE 2048

typedef long long int lli;

/* A simple function designed to perform memory operations. */
void do_copy(double *arr, double *copy_arr, size_t n, lli num_copies) {
  for (lli iter = 0; iter < num_copies; iter++) {
    for (size_t i = 0; i < n; i++) {
      copy_arr[i] = arr[i];
    }
  }
}

/* A simple function designed to do floating point computations. */
double do_flops(double a, double b, double c, lli num_flops) {
  for (lli i = 0; i < num_flops; i++) {
    c = a * b + c;
  }
  return c;
}

int main(int argc, char *argv[])
{
    /* ====== Begin setting environment variables ======
     *
     * The envrionment variables used to configure are typically set at runtime
     * with a command like:
     *
     * LIKWID_EVENTS="L2|L3" LIKWID_THREADS="0,2,3" LIKWID_FILEPATH="/tmp/likwid_marker.out" LIKWID_ACCESSMODE="1" LIKWID_FORCE="1" ./C-internalMarkerAPI
     *
     * They are set here to ensure completeness of this example. If you specify
     * environment variables at runtime, be sure to pin threads with
     * likwid-pin, GOMP_CPU_AFFINITY, or similar!
     */

    if(getenv("LIKWID_PIN"))
    {
        fprintf(stderr, "ERROR: it appears you are running this example with "
            "likwid-perfctr. This example\n"
            "is intended to be run on its own. Results will be incorrect "
            "or missing.\n");
        exit(1);
    }

    /* The first envrionment variable we define is "LIKWID_EVENTS", which
     * indicates the groups to be measured. In this case, the first 3 are group
     * names specified in
     * `/<LIKWID_PREFIX>/share/likwid/perfgroups/<ARCHITECTURE>/`
     * (default prefix is `/usr/local`). The final group is a custom group
     * specified with EVENT_NAME:COUNTER_NAME.
     *
     * Be aware that the groups chosen below are defined for most but not all
     * architectures supported by likwid. For a more compatible set of groups,
     * see the commented group set. The second, more compatible set will work
     * for all architectures supported by likwid except Nvidia GPUs and Xeon Phi
     * (KNC).
     */
    setenv("LIKWID_EVENTS", "FLOPS_DP|L2|INSTR_RETIRED_ANY:FIXC0", 1);
    // setenv("LIKWID_EVENTS", "BRANCH|INSTR_RETIRED_ANY:FIXC0", 1);

    /* LIKWID_THREADS must be set to the list of hardware threads that we will
     * use.
     */
    setenv("LIKWID_THREADS", "0,2,3", 1);

    /* This array is not utilized by likwid, but we will use it to pin threads
     * below. It must match the string set in LIKWID_THREADS above.
     */
    int cpus[NUM_THREADS] =  {0,2,3};

    /* The location the marker file will be stored at. */
    setenv("LIKWID_FILEPATH", "/home/jj/likwid_marker.out", 1);

    /* A value of 1 is the code for using the access daemon. */
    setenv("LIKWID_MODE", "1", 1);

    /* If the NMI watchdog is enabled or the application does not call
     * perfmon_finalize(), e.g., because of some error, LIKWID will fail with
     * the message "Counter in use". By setting LIKWID_FORCE, you can force
     * likwid to overwrite the used registers regardless.
     */
    setenv("LIKWID_FORCE", "1", 1);

    /* If the user desires more information about what's going on under the
     * hood, they may set this to get some debug information. Values from 0-3
     * are valid, with 0 being the default (none) and each level from 1-3
     * representing an increased level of output verbosity.
     */
    // setenv("LIKWID_DEBUG", "3", 1);

    /* ====== End setting environment variables ====== */

    /* Calls perfmon_init() and perfmon_addEventSet. Uses the environment
     * variables set above to configure likwid.
     */
    LIKWID_MARKER_INIT;

    /* Virtual threads must be pinned to a physical thread. One way to do this
     * is demonstrated below.
     *
     * Alternatively, threads may be pinned at runtime using likwid-pin or a
     * similar tool. If GNU openmp is used, threads may be pinned by setting the
     * GOMP_CPU_AFFINITY environment variable to the same cpus as specified in
     * LIKWID_THREADS. For example, if LIKWID_THREADS="0,1,3" then
     * GOMP_CPU_AFFINITY should be set to "0 1 3".
     *
     * Be aware that in the case of openMP, threads will be numbered
     * sequentially from 0, even if that does not correspond to the physical
     * thread number. This behavior comes in handy because likwid follows the
     * same convention of numbering threads sequentially independent of the ID
     * of the hardware thread they are actually pinned to.
     */
    omp_set_num_threads(NUM_THREADS);
#pragma omp parallel
{
    likwid_pinThread(cpus[omp_get_thread_num()]);

    /* LIKWID_MARKER_THREADINIT was required with past versions of likwid but
     * is now commonly not needed and was, in fact, deprecated with likwid
     * v5.0.1 for that reason.
     *
     * It is only required if the pinning library fails and there is a risk of
     * threads getting migrated. I am currently unaware of any runtime system
     * that doesn't work.
     */
    // LIKWID_MARKER_THREADINIT;

    /* Registering regions is optional but strongly recommended, as it reduces
     * the overhead of LIKWID_MARKER_START and prevents wrongly reported counts
     * in short regions.
     *
     * There must be a barrier between registering a region and starting it.
     * Typically, these are done in separate parallel blocks, relying on the
     * implicit barrier at the end of the parallel block. Usually, there is a
     * parallel block for initialization and a parallel block for execution.
     */
    LIKWID_MARKER_REGISTER("Total");
    LIKWID_MARKER_REGISTER("calc_flops");

    /* To demonstrate that registering regions is optional, we do not register
     * the "copy" region here.
     */
    // LIKWID_MARKER_REGISTER("copy");
}

    /* The variables needed for flops/copy computations. */
    double a, b, c;
    a = 1.8;
    b = 3.2;
    c = 1.0;
    double arr[ARRAY_SIZE];
    double copy_arr[ARRAY_SIZE];

#pragma omp parallel
{
    /* First, we will demonstrate measuring a single region, getting results
     * with LIKWID_MARKER_GET, and resetting the region so that these results
     * do not affect later measurements.
    */

    /* The variables needed for LIKWID_MARKER_GET. */
    int nr_events = MAX_NUM_EVENTS;
    double events[MAX_NUM_EVENTS];
    double time = 0;
    int count = 0;

    /* Variables for inspecting events. */
    int gid;
    char * group_name, * event_name;

    /* Loop iterators used later. */
    int i, t;

    /* This region will measure flop-heavy computations. Note that only the
     * first group specified, FLOPS_DP, will be measured. A way to measure
     * multiple groups is demonstrated in the next parallel block.
     */
    LIKWID_MARKER_START("calc_flops");

    /* This is the key region of code to measure. */
    a = do_flops(c, a, b, NUM_FLOPS*5);

    /* Since we are done measuring the flop-heavy code, we stop the respective
     * marker region.
     */
    LIKWID_MARKER_STOP("calc_flops");

    /* LIKWID_MARKER_GET can give us some information about what happened during
     * the execution of a measured region. The values of each event are stored
     * in `events` while the number of listed event types gets written into
     * `nr_events`.
     *
     * Alternatively, we can access the complete measurement data, including the
     * computed metrics, using the marker file likwid can create for us, as
     * demonstrated at the end of this file.
     */
    LIKWID_MARKER_GET("calc_flops", &nr_events, events, &time, &count);

    /* The group ID will let us get group and event names. */
    gid = perfmon_getIdOfActiveGroup();

    /* Get the active group's name. */
    group_name = perfmon_getGroupName(gid);

    /* Print basic information. */
    printf("calc_flops thread %d finished measuring group "
        "%s.\n"
        "Got %d events, runtime %f s, and call count %d\n",
        omp_get_thread_num(), group_name, nr_events, time, count);

    #pragma omp barrier /* OPTIONAL: prevents garbled output */
    /* Only allow one thread to print to prevent repeated output. */
    #pragma omp single
    {
        /* Uncomment the for-loop if you'd like to inspect all threads. */
        t = 0;
        // for (t = 0; t < NUM_THREADS; t++)
        {
            printf("detailed event results: \n");
            for(i = 0; i < nr_events; i++){
                /* get event name */
                event_name = perfmon_getEventName(gid, i);
                /* print results */
                printf("%40s: %30f\n", event_name,
                    perfmon_getResult(gid, i, t)
                );
            }
        }
        printf("\n");
    }

    /* "nr_events" must be reset if LIKWID_MARKER_GET will be used later. This
     * is because LIKWID_MARKER_GET uses it to determine the capacity of the
     * "events" array.
     */
    // nr_events = MAX_NUM_EVENTS;

    /* Regions may be reset during execution. This should be called on every
     * thread that should be reset. Since we have already inspected results of
     * the calc_flops region, we will reset it here:
     */
    LIKWID_MARKER_RESET("calc_flops");

} /* end of parallel region */


#pragma omp parallel
{
    /* Next, we'll demonstrate nested regions and measuring multiple groups
     * using LIKWID_MARKER_SWITCH. These will not be inspected with
     * LIKWID_MARKER_GET, but will instead use the marker file to inspect
     * regions after this parallel block is finished.
     */

    /* Loop iterators. */
    int i, k, t;

    /* The code we want to measure here will be run multiple times to measure
     * each one of the groups specified above. Using perfmon_getNumberOfGroups
     * to get the number of iterations makes it easy to run the computations
     * once for each group.
     */
    for (i = 0; i < perfmon_getNumberOfGroups(); i++)
    {

        /* Starting and stopping regions should be done in a parallel block. If
         * regions are started/stopped in a serial region, only the master
         * thread will be measured.
         */

        /* This region will measure everything we do. */
        LIKWID_MARKER_START("Total");

        /* This region will measure flop-heavy computations. */
        LIKWID_MARKER_START("calc_flops");

        /* Do flops. */
        c = do_flops(a, b, c, NUM_FLOPS);

        /* We are done measuring flop-heavy code here. */
        LIKWID_MARKER_STOP("calc_flops");

        /* Barriers between regions are typically not required, but may
         * sometimes be necessary when starting/stopping regions multiple times
         * in a parallel block. When experiencing errors like "WARN: Region
         * <region> already started", "WARN: Stopping an unknown/not-started
         * region <region>", or noting event values that are unreasonably high,
         * try placing barriers between regions.
         */
        // #pragma omp barrier

        /* This region will measure memory-heavy code. */
        LIKWID_MARKER_START("copy");

        /* Do memory ops. */
        do_copy(arr, copy_arr, ARRAY_SIZE, NUM_COPIES);

        /* Done measuring memory-heavy code. */
        LIKWID_MARKER_STOP("copy");

        /* Stop region that measured everything. */
        LIKWID_MARKER_STOP("Total");

        /* The barrier after stopping all regions but before switching groups
         * is absolutely required: Without it, some threads may not have stopped
         * the "copy" and "Total" regions before switching groups, which causes
         * erroneous results.
         */
        #pragma omp barrier

        /* LIKWID_MARKER_SWITCH should only be run by a single thread. If it is
         * called in a parallel region, it must be preceeded by a barrier and
         * run in something like a "#pragma omp single" block to ensure only
         * one thread runs it and all threads have stopped their regions before
         * switching groups.
         *
         * Regions must be switched outside of all regions (e.g., after
         * "LIKWID_MARKER_STOP" is called for each region)!
         */
        #pragma omp single
        {
            LIKWID_MARKER_SWITCH;
        }
    } /* end of for loop */

} /* end of parallel region */

    printf("\n");

    /* These computations are meaningless, but printing them is an easy way to
     * ensure the compiler doesn't optimize out the do_flops and do_copy
     * functions.
     *
     * There are also some other options to prevent optimization: Firstly, the
     * affected functions may be placed in another file. Alternatively, the user
     * may also declare output variables (e.g., "c" and "copy_arr" in the
     * example above) as volatile. This might negatively impact performance,
     * though, and is therefore not recommended.
     */
    printf("final result of flops operations: %f\n", c);
    printf("entry %d of copy_arr: %f\n", ((unsigned)c) % ARRAY_SIZE,
        copy_arr[((unsigned)c) % ARRAY_SIZE]);
    printf("\n");

    /* We can use LIKWID_MARKER_WRITE_FILE to write the current monitoring
     * results to a file at the specified location. Unlike when reading event
     * counts directly using LIKWID_MARKER_GET, this also provides access to the
     * metrics specified in the used performance groups. In this example, we use
     * the LIKWID_FILEPATH defined above to store our results. We will read the
     * contents of this file using likwid to view results next.
     */
    LIKWID_MARKER_WRITE_FILE(getenv("LIKWID_FILEPATH"));

    const char *region_name, *group_name, *event_name, *metric_name;
    double event_value, metric_value;
    int gid = 0;
    int t, i, k;

    /* Read the file output by likwid so that we can process results. Note that
     * this currently only works if the marker API was not closed using
     * LIKWID_MARKER_CLOSE since perfmon needs to still be initialized to be
     * able to retrieve group-related data.
     */
    perfmon_readMarkerFile(getenv("LIKWID_FILEPATH"));

    /* Get information like region name, number of events, and number of
     * metrics. Notice that the number of regions printed here is actually
     * (num_regions*num_groups), because perfmon considers a region to be the
     * combination of a marker region and its group. In other words, each time a
     * region is measured with a different group or event set, perfmon considers
     * it a new region. For example, if we have two regions and 3 groups
     * measured for each, perfmon_getNumberOfRegions() will return 6.
     */
    printf("Marker API measured %d regions\n", perfmon_getNumberOfRegions());
    for (i = 0; i < perfmon_getNumberOfRegions(); i++)
    {
        gid = perfmon_getGroupOfRegion(i);
        printf("Region %s with %d events and %d metrics\n",
            perfmon_getTagOfRegion(i), perfmon_getEventsOfRegion(i),
            perfmon_getMetricsOfRegion(i));
    }
    printf("\n");

    /* Print per-thread results. */
    printf("Detailed results follow. Notice that the region \"calc_flops\"\n"
        "will not appear, as it was reset after each time it was measured.\n");
    printf("\n");

    const char * result_header = "%6s : %15s : %10s : %6s : %40s : %30s \n";
    const char * result_format = "%6d : %15s : %10s : %6s : %40s : %30f \n";
    printf(result_header, "thread", "region", "group", "type",
        "result name", "result value");

    /* Uncomment the for-loop if you'd like to inspect all threads. */
    t = 0;
    // for (t = 0; t < NUM_THREADS; t++)
    {
        for (i = 0; i < perfmon_getNumberOfRegions(); i++)
        {
            /* Returns the user-supplied region name. */
            region_name = perfmon_getTagOfRegion(i);

            /* The gid is the group ID independent of region, whereas i is the
             * ID of the region/group combination.
             */
            gid = perfmon_getGroupOfRegion(i);

            /* Get the name of the group measured, e.g., "FLOPS_DP" or "L2". */
            group_name = perfmon_getGroupName(gid);

            /* Get info for each event. */
            for (k = 0; k < perfmon_getNumberOfEvents(gid); k++)
            {
                /* Get the event name, e.g., "INSTR_RETIRED_ANY". */
                event_name = perfmon_getEventName(gid, k);

                /* Get the associated value. */
                event_value = perfmon_getResultOfRegionThread(i, k, t);

                printf(result_format, t, region_name, group_name, "event",
                    event_name, event_value);
            }

            /* Get info for each metric. */
            for (k = 0; k < perfmon_getNumberOfMetrics(gid); k++)
            {
                /* Get the metric name, e.g., "L2 bandwidth [MBytes/s]". */
                metric_name = perfmon_getMetricName(gid, k);

                /* Get the associated value. */
                metric_value = perfmon_getMetricOfRegionThread(i, k, t);

                printf(result_format, t, region_name, group_name, "metric",
                    metric_name, metric_value);
            }
        }
    }

    /* Finally, close the marker API to stop performance monitoring and clean up
     * the created data structures.
     */
    LIKWID_MARKER_CLOSE;

    return 0;
}
