/*
 * =======================================================================================
 *
 *      Filename:  daemon.c
 *
 *      Description:  C Module implementing a daemon time loop
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2013 Jan Treibig 
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
#include <errno.h>
#include <sys/types.h>
#include <unistd.h>
#include <signal.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <time.h>


#include <timer.h>
#include <perfmon.h>
#include <daemon.h>

static int daemon_running = 0;
static char eventString[1024];
static TimerData timeData;
static pid_t childpid;

/* To be able to give useful error messages instead of just dieing without a
 * comment. Mainly happens because we get a SIGPIPE if the daemon drops us. */
static void Signal_Handler(int sig)
{
    fprintf(stderr, "ERROR - [%s:%d] Signal %d caught\n", __FILE__, __LINE__, sig);
}

void daemon_intr(int sig)
{
    daemon_running = 0;
}

void
daemon_init(const char* str)
{
    struct sigaction sia;
    sia.sa_handler = Signal_Handler;
    sigemptyset(&sia.sa_mask);
    sia.sa_flags = 0;
    sigaction(SIGPIPE, &sia, NULL);

    strcpy(eventString, str);
    //signal(SIGINT, daemon_stop);
    //signal(SIGUSR1, daemon_interrupt);

}

void
daemon_start(uint64_t duration, uint64_t switch_interval)
{
    int group;
    int nr_groups;
    int nr_events;
    int nr_threads;
    int i,j;
    int current_interval = 0;
    int first_round = 1;
    
    perfmon_startCounters();
    timer_start(&timeData);
    
    childpid = fork();
    daemon_running = 1;
    if (childpid == 0)
    {
        nr_groups = perfmon_getNumberOfGroups();
        nr_threads = perfmon_getNumberOfThreads();
        signal(SIGUSR1, daemon_intr);
        while (daemon_running == 1)
        {
            timer_stop(&timeData);
            perfmon_readCounters();
            group = perfmon_getIdOfActiveGroup();
            //perfmon_logCounterResults( timer_print(&timeData) );
            nr_events = perfmon_getNumberOfEvents(group);
            
            for(i=0;i<nr_events;i++)
            {
                fprintf(stderr, "%d %d %d %f %d ", group,nr_events, nr_threads, timer_print(&timeData), i);
                for(j = 0;j<nr_threads;j++)
                {
                    fprintf(stderr, "%lu ", perfmon_getResult(group, i, j));
                }
                fprintf(stderr,"\n");
            }
            if (current_interval == switch_interval && nr_groups > 1 && !first_round)
            {
                group++;
                if (group == nr_groups)
                {
                    group = 0;
                }
                current_interval = -1;
                fprintf(stderr,"Switch group to %d\n",group);
                perfmon_switchActiveGroup(group);
            }
            current_interval++;
            first_round = 0;
            timer_start(&timeData);
            usleep(duration);
        }
        signal(SIGUSR1, SIG_DFL);
        group = perfmon_getIdOfActiveGroup();
        nr_events = perfmon_getNumberOfEvents(group);
        nr_threads = perfmon_getNumberOfThreads();
        for(i=0;i<nr_events;i++)
        {
            fprintf(stderr, "%d %d %d %f %d ", group,nr_events, nr_threads, timer_print(&timeData), i);
            for(j = 0;j<nr_threads;j++)
            {
                fprintf(stderr, "%lu ", perfmon_getResult(group, i, j));
            }
            fprintf(stderr,"\n");
        }
        exit(0);
    }
}

void
daemon_stop(int sig)
{
    int status = 0;
    kill(childpid, SIGUSR1);
    waitpid(childpid, &status, 0);
    //printf("DAEMON:  EXIT on %d, status %d\n", sig, status);
    perfmon_stopCounters();
}

void
daemon_interrupt(int sig)
{
    int groupId;
    if (daemon_running)
    {
        daemon_running = 0;
        kill(childpid, SIGINT);
        perfmon_stopCounters();
        perfmon_finalize();
        //printf("DAEMON:  STOP on %d\n",sig);
        kill(getpid(), SIGTERM);
    }
    else
    {
        groupId = perfmon_addEventSet(eventString);
        perfmon_setupCounters(groupId);
        perfmon_startCounters();
        daemon_running = 1;
        //printf("DAEMON:  START\n");
    }
}


