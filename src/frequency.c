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
 *                Jan Treibig (jt), jan.treibig@gmail.com
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
#include <error.h>

#include <frequency.h>

char* daemon_path = TOSTRING(INSTALL_PREFIX) "/sbin/likwid-setFreq";


enum  {
    ACPICPUFREQ,
    INTELPSTATE,
    PPCCPUFREQ,
} freq_driver;



uint64_t freq_getCpuClockCurrent(const int cpu_id )
{
    FILE *f = NULL;
    char cmd[256];
    char buff[256];
    char* eptr = NULL;
    uint64_t clock = 0x0ULL;

    sprintf(buff, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", cpu_id);
    f = fopen(buff, "r");
    if (f == NULL) {
        fprintf(stderr, "Unable to open path %s for reading\n", buff);
        return 0;
    }
    eptr = fgets(cmd, 256, f);
    if (eptr != NULL)
    {
        clock = strtoull(cmd, NULL, 10);
    }
    fclose(f);
    return clock * 1E3;
}

uint64_t freq_setCpuClockCurrent(const int cpu_id, const uint64_t freq)
{
    FILE *fpipe = NULL;
    char cmd[256];
    char buff[256];
    uint64_t cur = 0x0ULL;
    char* drv = freq_getDriver(cpu_id);
    if (strcmp(drv, "intel_pstate") == 0)
    {
        fprintf(stderr, "CPUfreq driver intel_pstate not supported\n");
        free(drv);
        return 0x0ULL;
    }
    free(drv);
    cur = freq_getCpuClockCurrent(cpu_id);
    if (cur == freq)
    {
        return cur;
    }

    sprintf(buff, "%s", daemon_path);
    if (access(buff, X_OK))
    {
        fprintf(stderr, "Daemon %s not executable", buff);
        return 0;
    }

    sprintf(cmd, "%s %d cur %lu", daemon_path, cpu_id, freq);
    if ( !(fpipe = (FILE*)popen(cmd,"r")) )
    {  // If fpipe is NULL
        fprintf(stderr, "Problems setting cpu frequency of CPU %d", cpu_id);
        return 0;
    }
    if (pclose(fpipe))
        return 0;

    return freq;
}

uint64_t freq_getCpuClockMax(const int cpu_id )
{
    FILE *f = NULL;
    char cmd[256];
    char buff[256];
    char* eptr = NULL;
    uint64_t clock = 0x0ULL;

    sprintf(buff, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_max_freq", cpu_id);
    f = fopen(buff, "r");
    if (f == NULL) {
        fprintf(stderr, "Unable to open path %s for reading\n", buff);
        return 0;
    }
    eptr = fgets(cmd, 256, f);
    if (eptr != NULL)
    {
        clock = strtoull(cmd, NULL, 10);
    }
    fclose(f);
    return clock *1E3;
}

uint64_t freq_setCpuClockMax(const int cpu_id, const uint64_t freq)
{
    FILE *fpipe = NULL;
    char cmd[256];
    char buff[256];
    uint64_t cur = 0x0ULL;
    char* drv = freq_getDriver(cpu_id);
    if (strcmp(drv, "intel_pstate") == 0)
    {
        fprintf(stderr, "CPUfreq driver intel_pstate not supported\n");
        free(drv);
        return 0x0ULL;
    }
    free(drv);
    cur = freq_getCpuClockMax(cpu_id);
    if (cur == freq)
    {
        return cur;
    }

    sprintf(buff, "%s", daemon_path);
    if (access(buff, X_OK))
    {
        fprintf(stderr, "Daemon %s not executable", buff);
        return 0;
    }

    sprintf(cmd, "%s %d max %lu", daemon_path, cpu_id, freq);
    if ( !(fpipe = (FILE*)popen(cmd,"r")) )
    {  // If fpipe is NULL
        fprintf(stderr, "Problems setting cpu frequency of CPU %d", cpu_id);
        return 0;
    }
    if (pclose(fpipe))
        return 0;

    return freq;
}

uint64_t freq_getCpuClockMin(const int cpu_id )
{

    uint64_t clock = 0x0ULL;
    FILE *f = NULL;
    char cmd[256];
    char buff[256];
    char* eptr = NULL;

    sprintf(buff, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_min_freq", cpu_id);
    f = fopen(buff, "r");
    if (f == NULL) {
        fprintf(stderr, "Unable to open path %s for reading\n", buff);
        return 0;
    }
    eptr = fgets(cmd, 256, f);
    if (eptr != NULL)
    {
        clock = strtoull(cmd, NULL, 10);
    }
    fclose(f);
    return clock *1E3;
}

uint64_t freq_setCpuClockMin(const int cpu_id, const uint64_t freq)
{
    FILE *fpipe = NULL;
    char cmd[256];
    char buff[256];
    uint64_t cur = 0x0ULL;
    char* drv = freq_getDriver(cpu_id);
    if (strcmp(drv, "intel_pstate") == 0)
    {
        fprintf(stderr, "CPUfreq driver intel_pstate not supported\n");
        free(drv);
        return 0x0ULL;
    }
    free(drv);
    cur = freq_getCpuClockMin(cpu_id);
    if (cur == freq)
    {
        return cur;
    }

    sprintf(buff, "%s", daemon_path);
    if (access(buff, X_OK))
    {
        fprintf(stderr, "Daemon %s not executable", buff);
        return 0;
    }

    sprintf(cmd, "%s %d min %lu", daemon_path, cpu_id, freq);
    if ( !(fpipe = (FILE*)popen(cmd,"r")) )
    {  // If fpipe is NULL
        fprintf(stderr, "Problems setting cpu frequency of CPU %d", cpu_id);
        return 0;
    }
    if (pclose(fpipe))
        return 0;

    return freq;
}

char * freq_getGovernor(const int cpu_id )
{
    FILE *f = NULL;
    char cmd[256];
    char buff[256];
    char* eptr = NULL, *sptr = NULL;

    sprintf(buff, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor", cpu_id);
    f = fopen(buff, "r");
    if (f == NULL) {
        fprintf(stderr, "Unable to open path %s for reading\n", buff);
        return NULL;
    }
    eptr = fgets(cmd, 256, f);
    if (eptr != NULL)
    {
        bstring bbuff = bfromcstr(cmd);
        btrimws(bbuff);
        eptr = NULL;
        eptr = malloc((blength(bbuff)+1) * sizeof(char));
        if (eptr == NULL)
        {
            return NULL;
        }
        sptr = bdata(bbuff);
        strcpy(eptr, sptr);
        return eptr;
    }
    return NULL;
}

int freq_setGovernor(const int cpu_id, const char* gov)
{
    FILE *fpipe = NULL;
    char cmd[256];
    char buff[256];
    char* drv = freq_getDriver(cpu_id);
    if (strcmp(drv, "intel_pstate") == 0)
    {
        fprintf(stderr, "CPUfreq driver intel_pstate not supported\n");
        free(drv);
        return 0;
    }
    free(drv);
    sprintf(buff, "%s", daemon_path);
    if (access(buff, X_OK))
    {
        fprintf(stderr, "Daemon %s not executable", buff);
        return 0;
    }

    sprintf(cmd, "%s %d gov %s", daemon_path, cpu_id, gov);
    if ( !(fpipe = (FILE*)popen(cmd,"r")) )
    {  // If fpipe is NULL
        fprintf(stderr, "Problems setting cpu frequency of CPU %d", cpu_id);
        return 0;
    }
    if (pclose(fpipe))
        return 0;
    return 1;
}

char * freq_getAvailFreq(const int cpu_id )
{
    int i, j, k;
    FILE *f = NULL;
    char cmd[256];
    char buff[256];
    char tmp[10];
    char *eptr = NULL, *rptr = NULL, *sptr = NULL;
    double d = 0.0;
    bstring bbuff;

    sprintf(buff, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_available_frequencies", cpu_id);
    f = fopen(buff, "r");
    if (f == NULL)
    {
        fprintf(stderr, "Unable to open path %s for reading\n", buff);
        return NULL;
    }
    rptr = fgets(buff, 256, f);
    if (rptr != NULL)
    {
        struct bstrList * freq_list;
        bbuff = bfromcstr(buff);
        btrimws(bbuff);
        DEBUG_PRINT(DEBUGLEV_DETAIL, Result: %s, bdata(bbuff));

        freq_list = bsplit(bbuff, ' ');
        eptr = malloc(freq_list->qty * 10 * sizeof(char));
        if (eptr == NULL)
        {
            fclose(f);
            return NULL;
        }
        sptr = bdata(freq_list->entry[0]);
        d = strtod(sptr, NULL);
        j = sprintf(eptr, "%.3f", d * 1E-6);
        for (i=1; i< freq_list->qty; i++)
        {
            sptr = bdata(freq_list->entry[i]);
            d = strtod(sptr, NULL);
            sprintf(tmp, " %.3f", d * 1E-6);
            for (k= strlen(tmp)-1; k >= 0; k--)
            {
                if (tmp[k] != '0') break;
                if (tmp[k] == '0' && k > 0 && tmp[k-1] != '.') tmp[k] = '\0';
            }
            j+= sprintf(&(eptr[j]), "%s", tmp);
        }
        bstrListDestroy(freq_list);
    }
    fclose(f);
    return eptr;
}

char * freq_getAvailGovs(const int cpu_id )
{
    int i, j, k;
    FILE *f = NULL;
    char cmd[256];
    char buff[256];
    char tmp[10];
    char* eptr = NULL, *rptr = NULL;
    bstring bbuff;

    sprintf(buff, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_available_governors", cpu_id);
    f = fopen(buff, "r");
    if (f == NULL)
    {
        fprintf(stderr, "Unable to open path %s for reading\n", buff);
        return NULL;
    }
    rptr = fgets(buff, 256, f);
    if (rptr != NULL)
    {
        struct bstrList * freq_list;
        bbuff = bfromcstr(buff);
        btrimws(bbuff);
        freq_list = bsplit(bbuff, ' ');
        k = 0;
        for (i=0;i < freq_list->qty; i++)
        {
            k += blength(freq_list->entry[i]);
        }
        eptr = malloc((k+1) * sizeof(char));
        if (eptr == NULL)
        {
            fclose(f);
            return NULL;
        }
        j = sprintf(eptr, "%s", bdata(freq_list->entry[0]));

        for (i=1; i< freq_list->qty; i++)
        {
            j += sprintf(&(eptr[j]), " %s", bdata(freq_list->entry[i]));
        }
        bstrListDestroy(freq_list);
    }
    fclose(f);
    return eptr;
}

char * freq_getDriver(const int cpu_id )
{
    FILE *f = NULL;
    char cmd[256];
    char buff[256];
    char* eptr = NULL, *rptr = NULL;
    bstring bbuff;

    sprintf(buff, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_driver", cpu_id);
    f = fopen(buff, "r");
    if (f == NULL)
    {
        fprintf(stderr, "Unable to open path %s for reading\n", buff);
        return NULL;
    }
    rptr = fgets(buff, 256, f);
    if (rptr != NULL)
    {
        bbuff = bfromcstr(buff);
        btrimws(bbuff);
        eptr = malloc((strlen(buff)+1) * sizeof(char));
        if (eptr == NULL)
        {
            fclose(f);
            return NULL;
        }
        sprintf(eptr, "%s", bdata(bbuff));
    }
    fclose(f);
    return eptr;
}
