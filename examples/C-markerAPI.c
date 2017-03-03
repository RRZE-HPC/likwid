/*
 * =======================================================================================
 *
 *      Filename:  C-markerAPI.c
 *
 *      Description:  Example how to use the C/C++ Marker API
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Thomas Roehl (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2015 RRZE, University Erlangen-Nuremberg
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

#define SLEEPTIME 2

int main(int argc, char* argv[])
{
    int i, g;
    int nevents = 10;
    double events[10];
    double time;
    int count;
    // Init Marker API in serial region once in the beginning
    LIKWID_MARKER_INIT;
    #pragma omp parallel
    {
        // Each thread must add itself to the Marker API, therefore must be
        // in parallel region
        LIKWID_MARKER_THREADINIT;
        // Optional. Register region name
        LIKWID_MARKER_REGISTER("example");
    }

    // perfmon_getNumberOfGroups is not part of the MarkerAPI,
    // it belongs to the normal LIKWID API. But the MarkerAPI
    // has no function to get the number of configured groups.
    for (g=0;g < perfmon_getNumberOfGroups(); g++)
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
            LIKWID_MARKER_GET("example", &nevents, events, &time, &count);
            // where events is an array of doubles with nevents entries,
            // time is a double* and count an int*.
            printf("Region example measures %d events, total measurement time is %f\n", nevents, time);
            printf("The region was called %d times\n", count);
            for (i = 0; i < nevents; i++)
            {
                printf("Event %d: %f\n", i, events[i]);
            }
            // If multiple groups given, you can switch to the next group
            LIKWID_MARKER_SWITCH;
        }
    }

    // Close Marker API and write results to file for further evaluation done
    // by likwid-perfctr
    LIKWID_MARKER_CLOSE;
    return 0;
}
