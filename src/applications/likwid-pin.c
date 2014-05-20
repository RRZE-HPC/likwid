/*
 * ===========================================================================
 *
 *      Filename:  likwid-pin.c
 *
 *      Description:  An application to pin a program including threads
 *
 *      Version:  <VERSION>
 *      Created:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Company:  RRZE Erlangen
 *      Project:  likwid
 *      Copyright:  Copyright (c) 2010, Jan Treibig
 *
 *      This program is free software; you can redistribute it and/or modify
 *      it under the terms of the GNU General Public License, v2, as
 *      published by the Free Software Foundation
 *     
 *      This program is distributed in the hope that it will be useful,
 *      but WITHOUT ANY WARRANTY; without even the implied warranty of
 *      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *      GNU General Public License for more details.
 *     
 *      You should have received a copy of the GNU General Public License
 *      along with this program; if not, write to the Free Software
 *      Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 *
 * ===========================================================================
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sched.h>
#include <errno.h>
#include <sys/types.h>
#include <unistd.h>
#include <ctype.h>

#include <error.h>
#include <types.h>
#include <bstrlib.h>
#include <cpuid.h>
#include <affinity.h>
#include <numa.h>
#include <strUtil.h>
#include <osdep/setenv.h>
#include <osdep/getopt.h>
#include <osdep/execute.h>

#ifdef COLOR
#include <textcolor.h>
#endif

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define HELP_MSG \
printf("likwid-pin --  Version %d.%d \n\n",VERSION,RELEASE); \
printf("\n"); \
printf("Supported Options:\n"); \
printf("Example usage: likwid-pin -c 0,4-6 ./myApp\n"); \
printf("-h\t Help message\n"); \
printf("-v\t Version information\n"); \
printf("-i\t Set numa interleave policy with all involved numa nodes\n"); \
printf("-c\t comma separated processor ids\n"); \
printf("-s\t bitmask with threads to skip\n"); \
printf("-t\t Threading implementation type (at the moment only intel)\n"); \
printf("-p\t Print thread domains\n"); \
printf("-q\t Silent without output\n"); \
printf("\t No special type necessary for gcc OpenMP\n\n")

#define VERSION_MSG \
printf("likwid-pin   %d.%d \n\n",VERSION,RELEASE)
    


void
pinPid(int cpuid, int silent)
{
    int status;
	cpu_set_t cpuset;

	CPU_ZERO(&cpuset);
	CPU_SET(cpuid, &cpuset);

	status = sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);

	if (status == -1) 
	{
		printf("sched_setaffinity failed : %s \n",strerror(errno));
	}
	else 
    {
        if(!silent)
        {
#ifdef COLOR
            color_on(BRIGHT, COLOR);
#endif
            printf("[likwid-pin] Main PID -> core %d - OK\n",  cpuid);
#ifdef COLOR
            color_reset();
#endif
        }
    }
}


int main (int argc, char** argv)
{ 
    int i;
    int c;
	int skipMask = 0;
	int optInterleaved = 0;
	int optSilent = 0;
    bstring  typeString = bformat("NoType");
    bstring  pinString;
    bstring  skipString;
    bstring  argString;
    int numThreads=0;
    int threads[MAX_NUM_THREADS];
    threads[0] = 0;

    if (argc ==  1) { 
        HELP_MSG; 
        exit (EXIT_SUCCESS);    
    }

    cpuid_init();
    affinity_init();
    numa_init();

    while ((c = getopt (argc, argv, "+c:s:t:pqihv")) != -1)
    {
        switch (c)
        {
            case 'h':
                HELP_MSG;
                exit (EXIT_SUCCESS);    
            case 'v':
                VERSION_MSG;
                exit (EXIT_SUCCESS);    
            case 'i':
                optInterleaved = 1;
                break;
            case 'c':
                if (! (argString = bSecureInput(200,optarg)))
                {
                    ERROR_MSG(Failed to read argument string!);
                }

                numThreads = bstr_to_cpuset(threads, argString);

                if(!numThreads)
                {
                    ERROR_MSG(Failed to parse cpu list.);
                }

                bdestroy(argString);
                break;
            case 't':
                if (! (argString = bSecureInput(10,optarg)))
                {
                    ERROR_MSG(Failed to read argument string!);
                }

                typeString = bstrcpy(argString);
                bdestroy(argString);
                break;
            case 'p':
                affinity_printDomains();
                exit (EXIT_SUCCESS);    
                break;
            case 'q':
                optSilent = 1;
                setenv("LIKWID_SILENT","true", 1);
                break;
            case 's':
                if (! (argString = bSecureInput(10,optarg)))
                {
                    ERROR_MSG(Failed to read argument string!);
                }

                skipMask = strtoul((char*) argString->data,NULL,16);
                bdestroy(argString);
                break;
            default:
                HELP_MSG;
                exit(EXIT_FAILURE);
        }
    }

	/* CPU List:
	 * pthread (default): pin main pid + all thread tids
	 *
	 * OpenMP: Pin OMP_NUM_THREADS
	 * intel openmp: pin main pid + all thread tids (skip thread 1)
	 * gcc openmp: pin main pid + all thread tids (one less)
	 */

    if (biseqcstr(typeString,"intel")) 
    {
		skipMask = 0x1;
        setenv("KMP_AFFINITY", "disabled", 1);
    }

    if (optInterleaved)
    {
        printf("Set mem_policy to interleaved\n");
        numa_setInterleaved(threads, numThreads);
    }

	if (numThreads > 1)
	{
		pinString = bformat("%d",threads[1]);

		for (i=2; i < numThreads;i++)
		{
			bformata(pinString,",%d",threads[i]);
		}

        skipString = bformat("%d",skipMask);

        setenv("LIKWID_PIN",(char*) pinString->data , 1);
        setenv("LIKWID_SKIP",(char*) skipString->data , 1);
        setenv("LD_PRELOAD",TOSTRING(LIBLIKWIDPIN), 1);
	}

	pinPid(threads[0], optSilent);
    fflush(stdout);

	argv +=  optind;

    EXEC_APP;

    return EXIT_SUCCESS;
}

