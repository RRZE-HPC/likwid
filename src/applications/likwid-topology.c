/*
 * =======================================================================================
 *
 *      Filename:  likwid-topology.c
 *
 *      Description:  A application to determine the thread and cache topology
 *                    on x86 processors.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2012 Jan Treibig 
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
/* #####   HEADER FILE INCLUDES   ######################################### */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sched.h>
#include <sys/types.h>
#include <unistd.h>
#include <ctype.h>

#include <types.h>
#include <cpuid.h>
#include <timer.h>
#include <affinity.h>
#include <numa.h>
#include <cpuFeatures.h>
#include <tree.h>
#include <asciiBoxes.h>
#include <strUtil.h>

/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */

#define HELP_MSG \
printf("\nlikwid-topology --  Version %d.%d \n\n",VERSION,RELEASE); \
printf("A tool to print the thread and cache topology on x86 CPUs.\n"); \
printf("Options:\n"); \
printf("-h\t Help message\n"); \
printf("-v\t Version information\n"); \
printf("-c\t list cache information\n"); \
printf("-C\t measure processor clock\n"); \
printf("-o\t Store output to file. (Optional: Apply text filter)\n"); \
printf("-g\t graphical output\n\n")

#define VERSION_MSG \
printf("likwid-topology  %d.%d \n\n",VERSION,RELEASE)

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

int main (int argc, char** argv)
{ 
    int optGraphical = 0;
    int optCaches = 0;
    int optClock = 0;
    int c;
    int tmp;
    TreeNode* socketNode;
    TreeNode* coreNode;
    TreeNode* threadNode;
    BoxContainer* container;
    bstring  argString;
    bstring  filterScript = bfromcstr("NO");
    FILE* OUTSTREAM = stdout;

    while ((c = getopt (argc, argv, "hvcCgo:")) != -1)
    {
        switch (c)
        {
            case 'h':
                HELP_MSG;
                exit (EXIT_SUCCESS);    
            case 'v':
                VERSION_MSG;
                exit (EXIT_SUCCESS);    
            case 'g':
                optGraphical = 1;
                break;
            case 'c':
                optCaches = 1;
                break;
            case 'C':
                optClock = 1;
                break;
            case 'o':
                if (! (argString = bSecureInput(200,optarg)))
                {
                    fprintf(stderr, "Failed to read argument string!\n");
                }

                OUTSTREAM = bstr_to_outstream(argString, filterScript);

                if(!OUTSTREAM)
                {
                    fprintf(stderr, "Failed to parse out file pattern.\n");
                }
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

    cpuid_init();
    affinity_init();
    numa_init();

    fprintf(OUTSTREAM, HLINE);
    fprintf(OUTSTREAM, "CPU type:\t%s \n",cpuid_info.name);

    if (optClock)
    {
        timer_init();
        fprintf(OUTSTREAM, "CPU clock:\t%3.2f GHz \n",  (float) timer_getCpuClock() * 1.E-09);
    }

    /*----------------------------------------------------------------------
     *  Thread Topology
     *----------------------------------------------------------------------*/
    fprintf(OUTSTREAM, SLINE);
    fprintf(OUTSTREAM, "Hardware Thread Topology\n");
    fprintf(OUTSTREAM, SLINE);
    fprintf(OUTSTREAM, "Sockets:\t%u \n", cpuid_topology.numSockets);
    fprintf(OUTSTREAM, "Cores per socket:\t%u \n", cpuid_topology.numCoresPerSocket);
    fprintf(OUTSTREAM, "Threads per core:\t%u \n", cpuid_topology.numThreadsPerCore);
    fprintf(OUTSTREAM, HLINE);
    fprintf(OUTSTREAM, "HWThread\tThread\t\tCore\t\tSocket\n");

    for ( uint32_t i=0; i <  cpuid_topology.numHWThreads; i++)
    {
        fprintf(OUTSTREAM, "%d\t\t%u\t\t%u\t\t%u\n",i
                ,cpuid_topology.threadPool[i].threadId
                ,cpuid_topology.threadPool[i].coreId
                ,cpuid_topology.threadPool[i].packageId);
    }
    fprintf(OUTSTREAM, HLINE);

    socketNode = tree_getChildNode(cpuid_topology.topologyTree);
    while (socketNode != NULL)
    {
        fprintf(OUTSTREAM, "Socket %d: ( ",socketNode->id);
        coreNode = tree_getChildNode(socketNode);

        while (coreNode != NULL)
        {
            threadNode = tree_getChildNode(coreNode);

            while (threadNode != NULL)
            {
                fprintf(OUTSTREAM, "%d ",threadNode->id);
                threadNode = tree_getNextNode(threadNode);
            }
            coreNode = tree_getNextNode(coreNode);
        }
        socketNode = tree_getNextNode(socketNode);
        fprintf(OUTSTREAM, ")\n");
    }
    fprintf(OUTSTREAM, HLINE"\n");

    /*----------------------------------------------------------------------
     *  Cache Topology
     *----------------------------------------------------------------------*/
    fprintf(OUTSTREAM, SLINE);
    fprintf(OUTSTREAM, "Cache Topology\n");
    fprintf(OUTSTREAM, SLINE);

    for ( uint32_t i=0; i <  cpuid_topology.numCacheLevels; i++)
    {
        if (cpuid_topology.cacheLevels[i].type != INSTRUCTIONCACHE)
        {
            fprintf(OUTSTREAM, "Level:\t%d\n",cpuid_topology.cacheLevels[i].level);
            if (cpuid_topology.cacheLevels[i].size < 1048576)
            {
                fprintf(OUTSTREAM, "Size:\t%d kB\n",
                        cpuid_topology.cacheLevels[i].size/1024);
            }
            else 
            {
                fprintf(OUTSTREAM, "Size:\t%d MB\n",
                        cpuid_topology.cacheLevels[i].size/1048576);
            }

            if( optCaches)
            {
                switch (cpuid_topology.cacheLevels[i].type) {
                    case DATACACHE:
                        fprintf(OUTSTREAM, "Type:\tData cache\n");
                        break;

                    case INSTRUCTIONCACHE:
                        fprintf(OUTSTREAM, "Type:\tInstruction cache\n");
                        break;

                    case UNIFIEDCACHE:
                        fprintf(OUTSTREAM, "Type:\tUnified cache\n");
                        break;
                    default:
                        /* make the compiler happy */
                        break;
                }
                fprintf(OUTSTREAM, "Associativity:\t%d\n",
                        cpuid_topology.cacheLevels[i].associativity);
                fprintf(OUTSTREAM, "Number of sets:\t%d\n",
                        cpuid_topology.cacheLevels[i].sets);
                fprintf(OUTSTREAM, "Cache line size:%d\n",
                        cpuid_topology.cacheLevels[i].lineSize);
                if(cpuid_topology.cacheLevels[i].inclusive)
                {
                    fprintf(OUTSTREAM, "Non Inclusive cache\n");
                }
                else
                {
                    fprintf(OUTSTREAM, "Inclusive cache\n");
                }
                fprintf(OUTSTREAM, "Shared among %d threads\n",
                        cpuid_topology.cacheLevels[i].threads);
            }
            fprintf(OUTSTREAM, "Cache groups:\t");
            tmp = cpuid_topology.cacheLevels[i].threads;
            socketNode = tree_getChildNode(cpuid_topology.topologyTree);
            fprintf(OUTSTREAM, "( ");
            while (socketNode != NULL)
            {
                coreNode = tree_getChildNode(socketNode);

                while (coreNode != NULL)
                {
                    threadNode = tree_getChildNode(coreNode);

                    while (threadNode != NULL)
                    {

                        if (tmp)
                        {
                            fprintf(OUTSTREAM, "%d ",threadNode->id);
                            tmp--;
                        }
                        else
                        {
                            fprintf(OUTSTREAM, ") ( %d ",threadNode->id);
                            tmp = cpuid_topology.cacheLevels[i].threads;
                            tmp--;
                        }

                        threadNode = tree_getNextNode(threadNode);
                    }
                    coreNode = tree_getNextNode(coreNode);
                }
                socketNode = tree_getNextNode(socketNode);
            }
            fprintf(OUTSTREAM, ")\n");

            fprintf(OUTSTREAM, HLINE);
        }
    }

    fprintf(OUTSTREAM, "\n");

    /*----------------------------------------------------------------------
     *  NUMA Topology
     *----------------------------------------------------------------------*/
    fprintf(OUTSTREAM, SLINE);
    fprintf(OUTSTREAM, "NUMA Topology\n");
    fprintf(OUTSTREAM, SLINE);

    if (numa_init() < 0)
    {
        fprintf(OUTSTREAM, "NUMA is not supported on this node!\n");
    }
    else
    {
        fprintf(OUTSTREAM, "NUMA domains: %d \n",numa_info.numberOfNodes);
        fprintf(OUTSTREAM, HLINE);

        for ( uint32_t i = 0; i < numa_info.numberOfNodes; i++)
        {
            fprintf(OUTSTREAM, "Domain %d:\n", i);
            fprintf(OUTSTREAM, "Processors: ");

            for ( int j = 0; j < numa_info.nodes[i].numberOfProcessors; j++)
            {
                fprintf(OUTSTREAM, " %d",numa_info.nodes[i].processors[j]);
            }
            fprintf(OUTSTREAM, "\n");

            fprintf(OUTSTREAM, "Relative distance to nodes: ");

            for ( int j = 0; j < numa_info.nodes[i].numberOfDistances; j++)
            {
                fprintf(OUTSTREAM, " %d",numa_info.nodes[i].distances[j]);
            }
            fprintf(OUTSTREAM, "\n");

            fprintf(OUTSTREAM, "Memory: %g MB free of total %g MB\n",
                    numa_info.nodes[i].freeMemory/1024.0, numa_info.nodes[i].totalMemory/1024.0);
            fprintf(OUTSTREAM, HLINE);
        }
    }
    fprintf(OUTSTREAM, "\n");

    /*----------------------------------------------------------------------
     *  Graphical topology
     *----------------------------------------------------------------------*/
    if(optGraphical)
    {
        int j;
        bstring  boxLabel = bfromcstr("0");

        fprintf(OUTSTREAM, SLINE);
        fprintf(OUTSTREAM, "Graphical:\n");
        fprintf(OUTSTREAM, SLINE);

        /* Allocate without instruction cache */
        if ( cpuid_info.family == P6_FAMILY || cpuid_info.family == MIC_FAMILY ) 
        {
            container = asciiBoxes_allocateContainer(
                    cpuid_topology.numCacheLevels,
                    cpuid_topology.numCoresPerSocket);
        }
        else
        {
            container = asciiBoxes_allocateContainer(
                    cpuid_topology.numCacheLevels+1,
                    cpuid_topology.numCoresPerSocket);
        }

        socketNode = tree_getChildNode(cpuid_topology.topologyTree);
        while (socketNode != NULL)
        {
            fprintf(OUTSTREAM, "Socket %d:\n",socketNode->id);
            j=0;
            coreNode = tree_getChildNode(socketNode);

            /* add threads */
            while (coreNode != NULL)
            {
                threadNode = tree_getChildNode(coreNode);
                tmp =0;

                while (threadNode != NULL)
                {
                    if (tmp > 0)
                    {
                        bformata(boxLabel,"  %d", threadNode->id);
                    }
                    else
                    {
                        boxLabel = bformat("%d",threadNode->id);
                    }
                    tmp++;
                    threadNode = tree_getNextNode(threadNode);
                }
                asciiBoxes_addBox(container, 0, j, boxLabel); 
                j++;
                coreNode = tree_getNextNode(coreNode);
            }

            /* add caches */
            {
                int columnCursor=0;
                int lineCursor=1;
                uint32_t sharedCores;
                int numCachesPerLevel;
                int cacheWidth;

                for ( uint32_t i=0; i < cpuid_topology.numCacheLevels; i++ )
                {
                    sharedCores = cpuid_topology.cacheLevels[i].threads /
                        cpuid_topology.numThreadsPerCore;

                    if (cpuid_topology.cacheLevels[i].type != INSTRUCTIONCACHE)
                    {
                        if ( sharedCores > cpuid_topology.numCoresPerSocket )
                        {
                            numCachesPerLevel = 1;
                        }
                        else
                        {
                            numCachesPerLevel =
                                cpuid_topology.numCoresPerSocket/sharedCores;
                        }

                        columnCursor=0;
                        for ( j=0; j < numCachesPerLevel; j++ )
                        {
                            if (cpuid_topology.cacheLevels[i].size < 1048576)
                            {
                                boxLabel = bformat("%dkB",
                                        cpuid_topology.cacheLevels[i].size/1024);
                            }
                            else 
                            {
                                boxLabel = bformat("%dMB",
                                        cpuid_topology.cacheLevels[i].size/1048576);
                            }

                            if (sharedCores > 1)
                            {
                                if (sharedCores > cpuid_topology.numCoresPerSocket)
                                {
                                    cacheWidth = cpuid_topology.numCoresPerSocket-1;
                                }
                                else
                                {
                                    cacheWidth = sharedCores-1;
                                }
                                asciiBoxes_addJoinedBox(
                                        container,
                                        lineCursor,
                                        columnCursor,
                                        columnCursor+cacheWidth,
                                        boxLabel); 

                                columnCursor += sharedCores;
                            }
                            else 
                            {
                                asciiBoxes_addBox(
                                        container,
                                        lineCursor,
                                        columnCursor,
                                        boxLabel); 

                                columnCursor++;
                            }

                        }
                        lineCursor++;
                    }
                }
            }

            asciiBoxes_print(container);
            socketNode = tree_getNextNode(socketNode);
        }
        bdestroy(boxLabel);
    }

    fflush(OUTSTREAM);

    /* call filterscript if specified */
    if (!biseqcstr(filterScript,"NO"))
    {
        bcatcstr(filterScript, " topology");
        if (system(bdata(filterScript)) == EOF)
        {
            fprintf(stderr, "Failed to execute filter %s!\n", bdata(filterScript));
            exit(EXIT_FAILURE);
        }
    }

    return EXIT_SUCCESS;
}

