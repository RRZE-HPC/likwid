/*
 * ===========================================================================
 *
 *      Filename:  daemon.c
 *
 *      Description:  C Module implementing a daemon time loop
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
#include <sys/types.h>
#include <unistd.h>
#include <signal.h>
#include <sys/time.h>

#include <timer.h>
#include <perfmon.h>
#include <daemon.h>

static int daemon_run = 0;
static bstring eventString;


void
daemon_init(bstring str)
{
    eventString = bstrcpy(str);
    printf("DAEMON:  INIT\n");
    signal(SIGINT, daemon_stop);
    signal(SIGUSR1, daemon_interrupt);

}

void
daemon_start(int interval)
{
    printf("DAEMON:  START\n");
    daemon_run = 1;
    perfmon_startCounters();

    while (1)
    {
        if (daemon_run)
        {
            printf("DAEMON:  PRINT\n");
            perfmon_readCounters();
            perfmon_logCounterResults((double) interval);
        }
        sleep(interval);
    }
}

void
daemon_stop(int sig)
{
    printf("DAEMON:  EXIT on %d\n", sig);
    perfmon_stopCounters();
    signal(SIGINT, SIG_DFL);
    kill(getpid(), SIGINT);
}

void
daemon_interrupt(int sig)
{
    if (daemon_run)
    {
        perfmon_stopCounters();
        daemon_run = 0;
        printf("DAEMON:  STOP on %d\n",sig);
    }
    else
    {
        perfmon_setupEventSet(eventString);
        perfmon_startCounters();
        daemon_run = 1;
        printf("DAEMON:  START\n");
    }
}


