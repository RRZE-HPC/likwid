/*
 * =======================================================================================
 *
 *      Filename:  C-markerAPI.c
 *
 *      Description:  Example how to use the C/C++ Marker API
 *
 *      Version:   5.1.1
 *      Released:  31.03.2021
 *
 *      Authors:  Thomas Gruber (tr), thomas.roehl@googlemail.com
 *                Riley Weber, rileyw13@protonmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2021 RRZE, University Erlangen-Nuremberg
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

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <omp.h>
#include <likwid.h>

#ifdef LIKWID_PERFMON
#define MAX_NUM_EVENTS 10
#else
#define MAX_NUM_EVENTS 0
#endif

#define SLEEPTIME 2

int main(int argc, char* argv[])
{
    int i, g;
    int nevents = MAX_NUM_EVENTS;
    double events[MAX_NUM_EVENTS];
    double time;
    int count;

    // Init Marker API in serial region once in the beginning
    LIKWID_MARKER_INIT;
    #pragma omp parallel
    {
        // Each thread must add itself to the Marker API, therefore must be
        // in parallel region
        LIKWID_MARKER_THREADINIT;

        // Register region name. Optional, but highly recommended. Reduces the
        // overhead of LIKWID_MARKER_START. Furthermore, if regions are not
        // registered but the access daemon is used, short regions will report
        // lower values for the first region.
        LIKWID_MARKER_REGISTER("example");
    }

    // if the number of iterations is not greater than the number of groups you
    // are measuring, groups after the final iteration will not be measured. In
    // other words, if the loop specifies n iterations, only the first n groups
    // will be measured. Furthermore, if the number of iterations is greater
    // than the number of groups, some groups will be meausured multiple times
    // in a round-robin fashion.
    for (g=0; g<10; g++)
    {
        #pragma omp parallel
        {
            printf("Thread %d sleeps now for %d seconds\n", omp_get_thread_num(), SLEEPTIME);
            // Start measurements inside a parallel region
            LIKWID_MARKER_START("example");

            // Insert your code here.
            // Often contains an OpenMP for pragma. Regions can be nested.
            sleep(SLEEPTIME);

            // Stop measurements inside a parallel region
            LIKWID_MARKER_STOP("example");
            printf("Thread %d wakes up again\n", omp_get_thread_num());

            // If you need the performance data inside your application, use
            // LIKWID_MARKER_GET. events is an array of doubles with
            // nevents entries, time is a double* and count an int*.
            LIKWID_MARKER_GET("example", &nevents, events, &time, &count);

            // this check ensures that nothing will be printed if
            // -DLIKWID_PERFMON is not included
            if(nevents > 0){
                printf("Region example measures %d events, total measurement time is %f\n", nevents, time);
                printf("The region was called %d times\n", count);
            }
            for (i = 0; i < nevents; i++)
            {
                printf("Event %d: %f\n", i, events[i]);
            }
        }

        // If multiple groups are given, you can switch to the next group. This
        // function has no effect if one group is specified. Notice that this
        // is called outside the parallel region, as it should only be run by a
        // single thread
        LIKWID_MARKER_SWITCH;
    }

    // Close Marker API and write results to file for further evaluation done
    // by likwid-perfctr
    LIKWID_MARKER_CLOSE;
    return 0;
}
