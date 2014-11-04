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
#include <errno.h>
#include <sys/types.h>
#include <unistd.h>
#include <signal.h>
#include <sys/time.h>
#include <time.h>

#include <timer.h>
#include <perfmon.h>
#include <daemon.h>

static volatile int daemon_run = 0;
static bstring eventString;
static TimerData timeData;
static pid_t daemonpid = 0;


void
daemon_start(bstring str, struct timespec interval)
{
    daemonpid = fork();
    if (daemonpid == 0)
    {
        eventString = bstrcpy(str);
        signal(SIGINT, daemon_interrupt);
        signal(SIGUSR1, daemon_interrupt);
        daemon_run = 1;
        perfmon_setupEventSet(eventString, NULL);
        perfmon_startCounters();
        timer_start(&timeData);

        while (1)
        {
            if (daemon_run)
            {
                timer_stop(&timeData);
                perfmon_readCounters();
                perfmon_logCounterResults( timer_print(&timeData) );
                timer_start(&timeData);
            }
            else
            {
                break;
            }
            nanosleep( &interval, NULL);
        }
        signal(SIGINT, SIG_DFL);
        signal(SIGUSR1, SIG_DFL);
        exit(EXIT_SUCCESS);
    }
}

void
daemon_stop(int sig)
{
    if (daemonpid > 0)
    {
        printf("PARENT: KILL daemon with signal %d\n", sig);
        kill(daemonpid, sig);
        //perfmon_stopCounters();
    }
}

void
daemon_interrupt(int sig)
{
    if (sig == SIGUSR1)
    {
        if (daemon_run)
        {
            perfmon_stopCounters();
            daemon_run = 0;
            printf("DAEMON: STOP on %d\n",sig);
            exit(EXIT_SUCCESS);
        }
        else
        {
            perfmon_setupEventSet(eventString, NULL);
            perfmon_startCounters();
            daemon_run = 1;
            printf("DAEMON: START with events %s\n",bdata(eventString));
        }
    } else
    {
        printf("DAEMON: EXIT on %d\n", sig);
        daemon_run = 0;
        exit(EXIT_SUCCESS);
    }
}


