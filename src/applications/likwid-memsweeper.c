/*
 * =======================================================================================
 *
 *      Filename:  likwid-memsweeper.c
 *
 *      Description:  An application to clean up NUMA memory domains.
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
#include <strUtil.h>
#include <error.h>
#include <cpuid.h>
#include <numa.h>
#include <affinity.h>
#include <memsweep.h>

/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */

#define HELP_MSG \
    fprintf(stdout, "\nlikwid-memsweeper --  Version  %d.%d \n\n",VERSION,RELEASE); \
    fprintf(stdout, "A tool clean up NUMA memory domains and last level caches.\n"); \
    fprintf(stdout, "Options:\n"); \
    fprintf(stdout, "-h\t Help message\n"); \
    fprintf(stdout, "-v\t Version information\n"); \
    fprintf(stdout, "-q\t Silent without output\n"); \
    fprintf(stdout, "-c\t Specify NUMA domain ID to clean up\n"); \
    fprintf(stdout, "\t If no specific domain is set, all domains are swept.\n"); \
    fprintf(stdout, "Usage:\n"); \
    fprintf(stdout, "To clean specific domain: likwid-memsweeper -c 2 \n"); \
    fflush(stdout);

#define VERSION_MSG \
    fprintf(stdout, "likwid-memsweeper  %d.%d \n\n",VERSION,RELEASE); \
    fflush(stdout);


int main (int argc, char** argv)
{
    int domainId = -1;
    int c;
    int optSilent = 0;
    bstring argString;
    FILE* OUTSTREAM = stdout;

    while ((c = getopt (argc, argv, "+c:hvq")) != -1)
    {
        switch (c)
        {
            case 'h':
                HELP_MSG;
                exit (EXIT_SUCCESS);
            case 'v':
                VERSION_MSG;
                exit (EXIT_SUCCESS);
            case 'q':
                optSilent = 1;
                OUTSTREAM = NULL;
                break;
            case 'c':
                if (! (argString = bSecureInput(10,optarg)))
                {
                    fprintf(stderr,"Failed to read argument string!\n");
                    exit(EXIT_FAILURE);
                }

                domainId = str2int((char*) argString->data);

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
    numa_init();

    if (domainId < 0) 
    {
        memsweep_node(OUTSTREAM);
    }
    else if (domainId < numa_info.numberOfNodes)
    {
        memsweep_domain(OUTSTREAM, domainId);
    }
    else
    {
        fprintf(stderr, "Unknown NUMA domain %d\n", domainId);
        exit(EXIT_FAILURE);
    }

    return EXIT_SUCCESS;
}

