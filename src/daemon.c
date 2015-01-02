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

#include <types.h>
#include <timer.h>
#include <perfmon.h>
#include <likwid.h>
#include <error.h>

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

int
daemon_start(uint64_t duration)
{
    int ret;
    int group;
    int nr_groups;
    int nr_events;
    int nr_threads;
    int i,j;
    int current_interval = 0;
    int first_round = 1;
    
    ret = perfmon_startCounters();
    if (ret < 0)
    {
        return -EFAULT;
    }
    timer_start(&timeData);
    
    childpid = fork();
    daemon_running = 1;
    if (childpid == 0)
    {
        nr_groups = perfmon_getNumberOfGroups();
        if (nr_groups == 0)
        {
            ERROR_PLAIN_PRINT(DAEMON: No groups configured);
            exit(1);
        }
        nr_threads = perfmon_getNumberOfThreads();
        if (nr_threads == 0)
        {
            ERROR_PLAIN_PRINT(DAEMON: No threads configured);
            exit(1);
        }
        signal(SIGUSR1, daemon_intr);
        signal(SIGINT, daemon_intr);
        while (daemon_running == 1)
        {
            timer_stop(&timeData);
            ret = perfmon_readCounters();
            if (ret < 0)
            {
                ERROR_PLAIN_PRINT(DAEMON: Failed to read counters);
                exit(1);
            }
            group = perfmon_getIdOfActiveGroup();
            if (group < 0)
            {
                ERROR_PLAIN_PRINT(DAEMON: Active group not configured);
                exit(1);
            }
            //perfmon_logCounterResults( timer_print(&timeData) );
            nr_events = perfmon_getNumberOfEvents(group);
            fprintf(stderr, "%d,%d,%d,%f,", group, nr_events, nr_threads, timer_print(&timeData));
            for(i=0;i<nr_events;i++)
            {
                fprintf(stderr, "%d,", i);
                for(j = 0;j<nr_threads;j++)
                {
                    fprintf(stderr, "%f", perfmon_getResult(group, i, j));
                    if ((i < nr_events) && (j < nr_threads))
                    {
                        fprintf(stderr, ",");
                    }
                }
            }
            fprintf(stderr,"\n");
            if (nr_groups > 1 && !first_round)
            {
                group++;
                if (group == nr_groups)
                {
                    group = 0;
                }
                current_interval = -1;
                ret = perfmon_switchActiveGroup(group);
                if (ret < 0)
                {
                    ERROR_PRINT(DAEMON: Failed to switch to group %d, group);
                    exit(1);
                }
            }
            current_interval++;
            first_round = 0;
            timer_start(&timeData);
            usleep(duration);
        }
        signal(SIGUSR1, SIG_DFL);
        signal(SIGINT, SIG_DFL);
        group = perfmon_getIdOfActiveGroup();
        if (group < 0)
        {
            ERROR_PLAIN_PRINT(DAEMON: Failed to get ID of active group);
            exit(1);
        }
        nr_events = perfmon_getNumberOfEvents(group);
        if (nr_events <= 0)
        {
            ERROR_PRINT(DAEMON: Group %d has not events configured, group);
            exit(1);
        }
        fprintf(stderr, "%d,%d,%d,%f,", group, nr_events, nr_threads, timer_print(&timeData));
        for(i=0;i<nr_events;i++)
        {
            fprintf(stderr, "%d,", i);
            for(j = 0;j<nr_threads;j++)
            {
                fprintf(stderr, "%f", perfmon_getResult(group, i, j));
                if ((i < nr_events) && (j < nr_threads))
                {
                    fprintf(stderr, ",");
                }
            }
        }
        fprintf(stderr,"\n");
        exit(0);
    }
    return 0;
}

int
daemon_stop(int sig)
{
    pid_t ret;
    int status = 0;
    status = kill(childpid, SIGUSR1);
    if (status < 0)
    {
        ERROR_PRINT(Failed to kill daemon with SIGUSR1 signal. kill returned %d, status);
        return status;
    }
    ret = waitpid(childpid, &status, 0);
    if (ret < 0)
    {
        ERROR_PRINT(Daemon process returned %d with status %d,ret, status);
        return (int)ret;
    }
    
    status = perfmon_stopCounters();
    if (status < 0)
    {
        ERROR_PRINT(Failed to stop counters. LIKWID returned %d, status);
        return status;
    }
    return 0;
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


