/*
 * =======================================================================================
 *
 *      Filename:  likwid-features.c
 *
 *      Description:  An application to read out and set the feature flag
 *                  register on Intel Core 2 processors.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2014 Jan Treibig
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
#include <string.h>
#include <sched.h>
#include <sys/types.h>
#include <unistd.h>
#include <ctype.h>

#include <types.h>
#include <error.h>
#include <strUtil.h>
#include <accessClient.h>
#include <msr.h>
#include <cpuid.h>
#include <cpuFeatures.h>

#define HELP_MSG \
    fprintf(stdout, "\nlikwid-features --  Version  %d.%d \n\n",VERSION,RELEASE); \
    fprintf(stdout, "A tool to print and toggle the feature flag msr on Intel CPUS.\n"); \
    fprintf(stdout, "Supported Features: HW_PREFETCHER, CL_PREFETCHER, DCU_PREFETCHER, IP_PREFETCHER.\n\n"); \
    fprintf(stdout, "Options:\n"); \
    fprintf(stdout, "-h\t Help message\n"); \
    fprintf(stdout, "-v\t Version information\n"); \
    fprintf(stdout, "-s <FEATURE>\t set cpu feature \n"); \
    fprintf(stdout, "-u <FEATURE>\t unset cpu feature \n"); \
    fprintf(stdout, "-c <ID>\t core id\n\n"); \
    fflush(stdout);

#define VERSION_MSG \
    fprintf(stdout, "likwid-features  %d.%d \n\n",VERSION,RELEASE); \
    fflush(stdout);

int main (int argc, char** argv)
{ 
    int socket_fd = -1;
    int optSetFeature = 0;
    int cpuId = 0;
    int c;
    bstring argString;
    CpuFeature feature = HW_PREFETCHER ;

    while ((c = getopt (argc, argv, "c:s:u:hv")) != -1)
    {
        switch (c)
        {
            case 'h':
                HELP_MSG;
                exit (EXIT_SUCCESS);
            case 'v':
                VERSION_MSG;
                exit (EXIT_SUCCESS);
            case 'u':
                optSetFeature = 2;
            case 's':
                if (! (argString = bSecureInput(40,optarg)))
                {
                    fprintf(stderr,"Failed to read argument string!\n");
                    exit(EXIT_FAILURE);
                }

                if (biseqcstr(argString,"HW_PREFETCHER"))
                {
                    feature = HW_PREFETCHER;
                }
                else if (biseqcstr(argString,"CL_PREFETCHER"))
                {
                    feature = CL_PREFETCHER;
                }
                else if (biseqcstr(argString,"DCU_PREFETCHER"))
                {
                    feature = DCU_PREFETCHER;
                }
                else if (biseqcstr(argString,"IP_PREFETCHER"))
                {
                    feature = IP_PREFETCHER;
                }
                else
                {
                    fprintf(stderr,"Feature not supported!\n");
                    exit(EXIT_FAILURE);
                }


                if (!optSetFeature)
                {
                    optSetFeature = 1;
                }
                break;
            case 'c':
                if (! (argString = bSecureInput(20,optarg)))
                {
                    fprintf(stderr,"Failed to read argument string!\n");
                    exit(EXIT_FAILURE);
                }

                cpuId = str2int((char*) argString->data);

                break;
            case '?':
                if (isprint (optopt))
                {
                    fprintf (stderr, "Unknown option `-%c'.\n", optopt);
                }
                else
                {
                    fprintf (stderr,
                            "Unknown option character `\\x%x'.\n",
                            optopt);
                }
                return EXIT_FAILURE;
            default:
                HELP_MSG;
                exit (EXIT_SUCCESS);
        }
    }

    if (cpuid_init() == EXIT_FAILURE)
    {
        ERROR_PLAIN_PRINT(Unsupported processor!);
    }

    fprintf(stdout, HLINE);
    fprintf(stdout, "CPU name:\t%s \n",cpuid_info.name);
    fprintf(stdout, "CPU core id:\t%d \n", cpuId);
    fflush(stdout);

    if (cpuid_info.family != P6_FAMILY)
    {
        fprintf (stderr, "likwid-features only supports Intel P6 based processors!\n");
        exit(EXIT_FAILURE);
    }

    if (cpuId >= (int) cpuid_topology.numHWThreads)
    {
        fprintf (stderr, "This processor has only %d HWthreads! \n",cpuid_topology.numHWThreads);
        exit(EXIT_FAILURE);
    }

    accessClient_init(&socket_fd);
    msr_init(socket_fd);
    cpuFeatures_init(cpuId);
    cpuFeatures_print(cpuId);

    if (optSetFeature == 1)
    {
        fprintf(stdout, SLINE);
        cpuFeatures_enable(cpuId, feature);
        fprintf(stdout, SLINE);
    }
    else if (optSetFeature == 2)
    {
        fprintf(stdout, SLINE);
        cpuFeatures_disable(cpuId, feature);
        fprintf(stdout, SLINE);
    }
    fflush(stdout);

    msr_finalize();
    return EXIT_SUCCESS;
}

