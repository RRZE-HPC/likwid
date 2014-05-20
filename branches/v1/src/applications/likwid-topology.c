/*
 * ===========================================================================
 *
 *      Filename:  likwid-topology.c
 *
 *      Description:  A application to determine the thread and cache topology
 *                  on x86 processors.
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
#include <sys/types.h>
#include <unistd.h>
#include <ctype.h>

#include <types.h>
#include <timer.h>
#include <cpuid.h>
#include <cpuFeatures.h>
#include <tree.h>
#include <asciiBoxes.h>

#define HELP_MSG \
printf("\nlikwid-topology --  Version %d.%d \n\n",VERSION,RELEASE); \
printf("A tool to print the thread and cache topology on x86 CPUs.\n"); \
printf("Options:\n"); \
printf("-h\t Help message\n"); \
printf("-v\t Version information\n"); \
printf("-c\t list cache information\n"); \
printf("-g\t graphical output\n\n")

#define VERSION_MSG \
printf("likwid-topology  %d.%d \n\n",VERSION,RELEASE)

int main (int argc, char** argv)
{ 
    int optGraphical = 0;
    int optCaches = 0;
    int hasSMT = 0;
    int c;
    int i;
    int j;
    int tmp;
    TreeNode* socketNode;
    TreeNode* coreNode;
    TreeNode* threadNode;
    BoxContainer* container;

    while ((c = getopt (argc, argv, "hvcg")) != -1)
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

    timer_init();
    cpuid_init();

    if( cpuid_info.family == P6_FAMILY )
    {
        cpuFeatures_init(0);

        if (cpuFeatureFlags.speedstep)
        {
            fprintf (stderr, "Speedstep is enabled!\nThis produces inaccurate timing measurements.\n");
            fprintf (stderr, "For reliable clock measurements disable speedstep.\n");
        }
    }

    printf(HLINE);
    printf("CPU type:\t%s \n",cpuid_info.name);
    printf("CPU clock:\t%3.2f GHz \n\n",  (float) cpuid_info.clock * 1.E-09);

    /*----------------------------------------------------------------------
     *  Thread Topology
     *----------------------------------------------------------------------*/
    printf(SLINE);
    printf("Hardware Thread Topology\n");
    printf(SLINE);

    printf("Sockets:\t\t%u \n", cpuid_topology.numSockets);
    printf("Cores per socket:\t%u \n", cpuid_topology.numCoresPerSocket);
    printf("Threads per core:\t%u \n", cpuid_topology.numThreadsPerCore);

    if (cpuid_topology.numThreadsPerCore > 1)
    {
        hasSMT = 1;
    }

    printf(HLINE);
    printf("HWThread\tThread\t\tCore\t\tSocket\n");
    for (i=0; i< cpuid_topology.numHWThreads; i++)
    {
        printf("%d\t\t%u\t\t%u\t\t%u\n",i
                ,cpuid_topology.threadPool[i].threadId
                ,cpuid_topology.threadPool[i].coreId
                ,cpuid_topology.threadPool[i].packageId);
    }
    printf(HLINE);

    socketNode = tree_getChildNode(cpuid_topology.topologyTree);
    while (socketNode != NULL)
    {
        printf("Socket %d: ( ",socketNode->id);
        coreNode = tree_getChildNode(socketNode);

        while (coreNode != NULL)
        {
            threadNode = tree_getChildNode(coreNode);

            while (threadNode != NULL)
            {
                printf("%d ",threadNode->id);
                threadNode = tree_getNextNode(threadNode);
            }
            coreNode = tree_getNextNode(coreNode);
        }
        socketNode = tree_getNextNode(socketNode);
        printf(")\n");
    }
    printf(HLINE"\n");

    /*----------------------------------------------------------------------
     *  Cache Topology
     *----------------------------------------------------------------------*/
    printf(SLINE);
    printf("Cache Topology\n");
    printf(SLINE);

    for (i=0; i< cpuid_topology.numCacheLevels; i++)
    {
        if (cpuid_topology.cacheLevels[i].type != INSTRUCTIONCACHE)
        {
            printf("Level:\t %d\n",cpuid_topology.cacheLevels[i].level);
            if (cpuid_topology.cacheLevels[i].size < 1048576)
            {
                printf("Size:\t %d kB\n",
                        cpuid_topology.cacheLevels[i].size/1024);
            }
            else 
            {
                printf("Size:\t %d MB\n",
                        cpuid_topology.cacheLevels[i].size/1048576);
            }

            if( optCaches)
            {
                switch (cpuid_topology.cacheLevels[i].type) {
                    case DATACACHE:
                        printf("Type:\t Data cache\n");
                        break;

                    case INSTRUCTIONCACHE:
                        printf("Type:\t Instruction cache\n");
                        break;

                    case UNIFIEDCACHE:
                        printf("Type:\t Unified cache\n");
                        break;
                }
                printf("Associativity:\t %d\n",
                        cpuid_topology.cacheLevels[i].associativity);
                printf("Number of sets:\t %d\n",
                        cpuid_topology.cacheLevels[i].sets);
                printf("Cache line size: %d\n",
                        cpuid_topology.cacheLevels[i].lineSize);
                if(cpuid_topology.cacheLevels[i].inclusive)
                {
                    printf("Non Inclusive cache\n");
                }
                else
                {
                    printf("Inclusive cache\n");
                }
                printf("Shared among %d threads\n",
                        cpuid_topology.cacheLevels[i].threads);
            }
            printf("Cache groups:\t");
            tmp = cpuid_topology.cacheLevels[i].threads;
            socketNode = tree_getChildNode(cpuid_topology.topologyTree);
            printf("( ");
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
                            printf("%d ",threadNode->id);
                            tmp--;
                        }
                        else
                        {
                            printf(") ( %d ",threadNode->id);
                            tmp = cpuid_topology.cacheLevels[i].threads;
                            tmp--;
                        }

                        threadNode = tree_getNextNode(threadNode);
                    }
                    coreNode = tree_getNextNode(coreNode);
                }
                socketNode = tree_getNextNode(socketNode);
            }
            printf(")\n");

            printf(HLINE);
        }
    }

    printf("\n");

    /*----------------------------------------------------------------------
     *  Graphical topology
     *----------------------------------------------------------------------*/
    if(optGraphical)
    {
        bstring  boxLabel = bfromcstr("0");

        printf(SLINE);
        printf("Graphical:\n");
        printf(SLINE);

        /* Allocate without instruction cache */
        if ( cpuid_info.family == P6_FAMILY) 
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
            printf("Socket %d:\n",socketNode->id);
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
                int sharedCores;
                int numCachesPerLevel;
                int cacheWidth;

                for (i=0; i< cpuid_topology.numCacheLevels; i++)
                {
                    sharedCores = cpuid_topology.cacheLevels[i].threads /
                        cpuid_topology.numThreadsPerCore;

                    if (cpuid_topology.cacheLevels[i].type != INSTRUCTIONCACHE)
                    {
                        if (sharedCores > cpuid_topology.numCoresPerSocket)
                        {
                            numCachesPerLevel = 1;
                        }
                        else
                        {
                            numCachesPerLevel =
                                cpuid_topology.numCoresPerSocket/sharedCores;
                        }

                        columnCursor=0;
                        for (j=0; j< numCachesPerLevel; j++)
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

    return EXIT_SUCCESS;
}

