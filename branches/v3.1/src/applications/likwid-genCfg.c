/*
 * =======================================================================================
 *
 *      Filename:  likwid-genCfg.c
 *
 *      Description:  An application to dump the cpu topology information to
 *      a config file.
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
#include <cpuid.h>

/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */

#define HELP_MSG \
    fprintf(stdout, "\nlikwid-genCfg --  Version  %d.%d \n\n",VERSION,RELEASE); \
    fprintf(stdout, "A tool to dump node topology information into a file.\n"); \
    fprintf(stdout, "Options:\n"); \
    fprintf(stdout, "-h\t Help message\n"); \
    fprintf(stdout, "-v\t Version information\n"); \
    fprintf(stdout, "-o\t output file path (optional)\n\n"); \
    fflush(stdout);

#define VERSION_MSG \
    fprintf(stdout, "likwid-powermeter  %d.%d \n\n",VERSION,RELEASE); \
    fflush(stdout);


int main (int argc, char** argv)
{
    FILE *file;
    char *filepath = TOSTRING(CFGFILE);
    int c;

    while ((c = getopt (argc, argv, "ho:v")) != -1)
    {
        switch (c)
        {
            case 'h':
                HELP_MSG;
                exit (EXIT_SUCCESS);
            case 'o':
                filepath = optarg;
                break;
            case 'v':
                VERSION_MSG;
                exit (EXIT_SUCCESS);
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

    cpuid_init();
    fprintf(stdout, HLINE);
    fprintf(stdout, "CPU name:\t%s \n",cpuid_info.name);
    fflush(stdout);

    if ((file = fopen(filepath, "wb")) != NULL) 
    {
        (void) fwrite((void*) &cpuid_topology, sizeof(CpuTopology), 1, file);

        (void) fwrite((void*) cpuid_topology.threadPool,
                sizeof(HWThread), cpuid_topology.numHWThreads, file);

        (void) fwrite((void*) cpuid_topology.cacheLevels,
                sizeof(CacheLevel), cpuid_topology.numCacheLevels, file);

        fclose(file);
    }
    else
    {
        fprintf(stderr,"Cfg file could not be written to %s\n", filepath);
        ERROR;
    }

    return EXIT_SUCCESS;
}

