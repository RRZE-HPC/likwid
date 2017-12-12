/*
 * =======================================================================================
 *
 *      Filename:  frequency.c
 *
 *      Description:  Module implementing an interface for frequency manipulation
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Thomas Roehl (tr), thomas.roehl@googlemail.com
 *                Amin Nabikhani, amin.nabikhani@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2016 RRZE, University Erlangen-Nuremberg
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
#include <stdint.h>
#include <math.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

#include <bstrlib.h>
#include <likwid.h>
#include <types.h>
#include <error.h>
#include <topology.h>
#include <access.h>
#include <registers.h>

#include <frequency_pstate.h>

static unsigned int freqs[100];
static unsigned int percent[100];
static unsigned int num_steps = 0;

static char mode()
{
    char readval[5];
    char tmode;
    FILE* fp = fopen("/sys/devices/system/cpu/intel_pstate/no_turbo","r");
    if (fp != NULL)
    {
        while( fgets(readval, 5, fp) )
        {
            tmode = readval[0];
        }
        fclose(fp);
    }
    return tmode;
}

static unsigned int turbo_pct()
{
    char readval[4];
    unsigned int turbo_pct;
    FILE* fp = fopen("/sys/devices/system/cpu/intel_pstate/turbo_pct","r");
    if (fp != NULL)
    {
        while( fgets(readval, 4, fp) )
        {
            turbo_pct = strtoul(readval,NULL,10);
        }
        fclose(fp);
    }
    return turbo_pct;
}

static unsigned int getMax()
{
    char line[1024];
    unsigned int maxFreq = 0;
    unsigned int trb = turbo_pct();
    char* eptr;
    FILE* fp = fopen("/sys/devices/system/cpu/cpufreq/policy0/cpuinfo_max_freq", "r");
    if(fp != NULL)
    {
        eptr = fgets(line, 1024, fp);
        maxFreq = strtoul(line, NULL, 10);
        fclose(fp);
    }
    else
    {
        fprintf(stderr, "\tEXIT WITH ERROR:  Max Freq. could not be read\n");
        exit(EXIT_FAILURE);
    }
    if(maxFreq != 0)
    {
        char t = mode();
        if (t != '0')
        {
            maxFreq /= (1+0.01*trb);
        }
    }
    else
    {
        fprintf(stderr, "\tEXIT WITH ERROR:  Max Freq. could not be read\n");
        exit(EXIT_FAILURE);
    }
    return maxFreq;
}


static unsigned int getMin()
{
    char line[1024];
    unsigned int minFreq = 0;
    char* eptr;
    FILE* fp = fopen("/sys/devices/system/cpu/cpufreq/policy0/cpuinfo_min_freq", "r");
    if(fp != NULL)
    {
        eptr = fgets(line, 1024, fp);
        minFreq = strtoul(line, NULL, 10);
        fclose(fp);
    }
    else
    {
        fprintf(stderr, "\tEXIT WITH ERROR:  Max Freq. could not be read\n");
        exit(EXIT_FAILURE);
    }

    return minFreq;
}



static unsigned int num_pstates()
{
    char readval[4];
    unsigned int num;
    FILE* fp = fopen("/sys/devices/system/cpu/intel_pstate/num_pstates","r");
    if (fp != NULL)
    {
        while( fgets(readval, 4, fp) )
        {
            num = strtoul(readval,NULL,10);
        }
        fclose(fp);
    }
    else
    {
        exit(1);
    }
    return num;
}



static void steps()
{
    unsigned int minFreq = getMin();
    unsigned int trb = turbo_pct();
    unsigned int maxFreq = getMax();
    unsigned int step = num_pstates();
    int range = 0;

    
    if(step != 0)
    {
        range = (maxFreq-minFreq)/step;
        freqs[0] = minFreq;
        freqs[step-1]= maxFreq;
        percent[0] = (minFreq/(float)maxFreq) * 100;
        percent[step-1] = 100;
        double t = 0;

        for(size_t i=1; i < step-1; i++)
        {
            freqs[i] = minFreq+ i* range;
            t = (((double)(freqs[i]))/((double)maxFreq)) * 100;
            percent[i] = (unsigned int)t;
        }
        num_steps = step;
    }
    else
    {
        fprintf(stderr,"\tEXIT WITH ERROR:  # of pstates could not be read");
    }
}


uint64_t freq_pstate_getCpuClockMax(const int cpu_id )
{
    char buff[256];
    unsigned int pct = 0;
    unsigned int maxFreq = getMax();
    if (num_steps == 0)
    {
        steps();
    }
    uint64_t clock = ((percent[num_steps-1]) * maxFreq) * 10;
    FILE* f = fopen("/sys/devices/system/cpu/intel_pstate/max_perf_pct","r");
    if (f != NULL)
    {
        char *eptr = fgets(buff, 256, f);
        if (eptr != NULL)
        {
            pct = strtoull(buff, NULL, 10);
            for (int i=num_steps-1; i >= 0; i--)
            {
                if (percent[i] == pct)
                {
                    //clock = freqs[i]*1000;
                    clock = ((percent[i]) * maxFreq) * 10; // *1000/100
                    break;
                }
            }
        }
        fclose(f);
    }
    return clock;
}



uint64_t freq_pstate_getCpuClockMin(const int cpu_id )
{
    char buff[256];
    unsigned int pct = 0;
    unsigned int maxFreq = getMax();
    if (num_steps == 0)
    {
        steps();
    }
    uint64_t clock = ((percent[0]) * maxFreq) * 10;
    FILE* f = fopen("/sys/devices/system/cpu/intel_pstate/min_perf_pct","r");
    if (f != NULL)
    {
        char *eptr = fgets(buff, 256, f);
        if (eptr != NULL)
        {
            pct = strtoull(buff, NULL, 10);
            for (int i=0; i < num_steps; i++)
            {
                if (percent[i] == pct)
                {
                    if (i > 0)
                        clock = ((percent[i-1]) * maxFreq) * 10;
                    else
                        clock = ((percent[i]) * maxFreq) * 10;
                    break;
                }
            }
        }
        fclose(f);
    }
    return clock;



}

