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
#include <types.h>
#include <error.h>
#include <topology.h>
#include <access.h>
#include <registers.h>
#if !defined(__ARM_ARCH_7A__) && !defined(__ARM_ARCH_8A)
#include <cpuid.h>
#endif

#include <frequency.h>
#include <frequency_acpi.h>
#include <frequency_pstate.h>

char* daemon_path = TOSTRING(INSTALL_PREFIX) "/sbin/likwid-setFreq";


typedef enum  {
    NOT_DETECTED = 0,
    ACPICPUFREQ,
    INTELPSTATE,
    PPCCPUFREQ,
} likwid_freq_driver;

likwid_freq_driver drv = NOT_DETECTED;

static int freq_getDriver(const int cpu_id )
{
    FILE *f = NULL;
    char buff[256];
    char* rptr = NULL;
    bstring bbuff;

    sprintf(buff, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_driver", cpu_id);
    f = fopen(buff, "r");
    if (f == NULL)
    {
        fprintf(stderr, "Unable to open path %s for reading\n", buff);
        return -errno;
    }
    rptr = fgets(buff, 256, f);
    if (rptr != NULL)
    {
        bbuff = bfromcstr(buff);
        btrimws(bbuff);
        if (strncmp(bdata(bbuff), "acpi-cpufreq", blength(bbuff)) == 0)
        {
            drv = ACPICPUFREQ;
        }
        else if (strncmp(bdata(bbuff), "intel_pstate", blength(bbuff)) == 0)
        {
            drv = INTELPSTATE;
        }
        bdestroy(bbuff);
    }
    fclose(f);
    return 0;
}

static int _freq_getUncoreMinMax(const int socket_id, int *cpuId, double* min, double* max)
{
    int cpu = -1;
    *cpuId = -1;
    *min = 0;
    *max = 0;
    for (int i=0; i<cpuid_topology.numHWThreads; i++)
    {
        if (cpuid_topology.threadPool[i].packageId == socket_id)
        {
            cpu = cpuid_topology.threadPool[i].apicId;
            break;
        }
    }
    if (cpu < 0)
    {
        fprintf(stderr, "Unknown socket ID %d\n", socket_id);
        return -ENODEV;
    }

    char* avail = freq_getAvailFreq(cpu);
    if (!avail)
    {
        fprintf(stderr, "Failed to get available frequencies\n");
        return -EINVAL;
    }

    double dmin = 0.0;
    double dmax = 0.0;
    bstring bavail = bfromcstr(avail);
    free(avail);
    struct bstrList* bavail_list;
    bavail_list = bsplit(bavail, ' ');
    bdestroy(bavail);
    if (bavail_list->qty < 2)
    {
        fprintf(stderr, "Failed to read minimal and maximal frequencies\n");
        bstrListDestroy(bavail_list);
        return -EINVAL;
    }
    if (blength(bavail_list->entry[0]) > 0)
    {
        char* tptr = NULL;
        dmin = strtod(bdata(bavail_list->entry[0]), &tptr);
        if (bdata(bavail_list->entry[0]) != tptr)
        {
            dmin *= 1000;
        }
        else
        {
            fprintf(stderr, "Problem converting %s to double for comparison with given freq.\n", bdata(bavail_list->entry[0]));
            return -EINVAL;
        }
    }
    if (blength(bavail_list->entry[bavail_list->qty-1]) > 0)
    {
        char* tptr = NULL;
        dmax = strtod(bdata(bavail_list->entry[bavail_list->qty-1]), &tptr);
        if (bdata(bavail_list->entry[bavail_list->qty-1]) != tptr)
        {
            dmax *= 1000;
        }
        else
        {
            fprintf(stderr, "Problem converting %s to double for comparison with given freq.\n", bdata(bavail_list->entry[bavail_list->qty-1]));
            return -EINVAL;
        }
    }
    bstrListDestroy(bavail_list);

    *cpuId = cpu;
    if (dmin < dmax)
    {
        *min = dmin;
        *max = dmax;
    }
    else
    {
        *max = dmin;
        *min = dmax;
    }

    power_init(cpu);
    if (power_info.turbo.numSteps > 0)
    {
        if (power_info.turbo.steps[0] > *max)
        {
            *max = power_info.turbo.steps[0];
        }
    }

    return 0;
}

uint64_t freq_getCpuClockMax(const int cpu_id )
{
    if (drv == NOT_DETECTED)
    {
        freq_getDriver(cpu_id);
    }
    if (drv == ACPICPUFREQ)
    {
        return freq_acpi_getCpuClockMax(cpu_id);
    }
    else if (drv == INTELPSTATE)
    {
        return freq_pstate_getCpuClockMax(cpu_id);
    }
    return 0;
}

uint64_t freq_getCpuClockMin(const int cpu_id )
{
    if (drv == NOT_DETECTED)
    {
        freq_getDriver(cpu_id);
    }
    if (drv == ACPICPUFREQ)
    {
        return freq_acpi_getCpuClockMin(cpu_id);
    }
    else if (drv == INTELPSTATE)
    {
        return freq_pstate_getCpuClockMin(cpu_id);
    }
    return 0;
}


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


uint64_t freq_setCpuClockMax(const int cpu_id, const uint64_t freq)
{
    FILE *fpipe = NULL;
    char cmd[256];
    char buff[256];
    uint64_t cur = 0x0ULL;

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

    if (drv == ACPICPUFREQ)
    {
        sprintf(cmd, "%s %d max %lu", daemon_path, cpu_id, freq);
    }
    else if (drv == INTELPSTATE)
    {
        double f = (double)freq;
        sprintf(cmd, "%s %d max %g", daemon_path, cpu_id, f/1000000);
    }
    if ( !(fpipe = (FILE*)popen(cmd,"r")) )
    {  // If fpipe is NULL
        fprintf(stderr, "Problems setting cpu frequency of CPU %d", cpu_id);
        return 0;
    }
    if (pclose(fpipe))
        return 0;

    return freq;
}

/*uint64_t freq_getCpuClockMin(const int cpu_id )*/
/*{*/

/*    uint64_t clock = 0x0ULL;*/
/*    FILE *f = NULL;*/
/*    char cmd[256];*/
/*    char buff[256];*/
/*    char* eptr = NULL;*/

/*    sprintf(buff, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_min_freq", cpu_id);*/
/*    f = fopen(buff, "r");*/
/*    if (f == NULL) {*/
/*        fprintf(stderr, "Unable to open path %s for reading\n", buff);*/
/*        return 0;*/
/*    }*/
/*    eptr = fgets(cmd, 256, f);*/
/*    if (eptr != NULL)*/
/*    {*/
/*        clock = strtoull(cmd, NULL, 10);*/
/*    }*/
/*    fclose(f);*/
/*    return clock *1E3;*/
/*}*/

uint64_t freq_setCpuClockMin(const int cpu_id, const uint64_t freq)
{
    FILE *fpipe = NULL;
    char cmd[256];
    char buff[256];
    uint64_t cur = 0x0ULL;

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

    //sprintf(cmd, "%s %d min %lu", daemon_path, cpu_id, freq);
    if (drv == ACPICPUFREQ)
    {
        sprintf(cmd, "%s %d min %lu", daemon_path, cpu_id, freq);
    }
    else if (drv == INTELPSTATE)
    {
        double f = (double)freq;
        sprintf(cmd, "%s %d min %g", daemon_path, cpu_id, f/1000000);
    }
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

/*int freq_setTurbo(const int cpu_id, int turbo)
{
    FILE *fpipe = NULL;
    char cmd[256];

    sprintf(cmd, "%s %d tur %d", daemon_path, cpu_id, turbo);
    if ( !(fpipe = (FILE*)popen(cmd,"r")) )
    {  // If fpipe is NULL
        fprintf(stderr, "Problems setting turbo mode of CPU %d", cpu_id);
        return 0;
    }
    if (pclose(fpipe))
        return 0;
    return 1;
}*/

static int getAMDTurbo(const int cpu_id)
{
    int err = 0;
    int own_hpm = 0;

    if (!HPMinitialized())
    {
        HPMinit();
        own_hpm = 1;
        
        err = HPMaddThread(cpu_id);
        if (err != 0)
        {
            ERROR_PLAIN_PRINT(Cannot get access to MSRs)
            return err;
        }
    }

    uint64_t tmp = 0x0ULL;
    err = HPMread(cpu_id, MSR_DEV, 0xC0010015, &tmp);
    if (err)
    {
        ERROR_PLAIN_PRINT(Cannot read register 0xC0010015);
        return err;
    }
    if (own_hpm)
        HPMfinalize();
    err = ((tmp >> 25) & 0x1);
    return err == 0;
}

static int setAMDTurbo(const int cpu_id, const int turbo)
{
    int err = 0;
    int own_hpm = 0;

    if (!HPMinitialized())
    {
        HPMinit();
        own_hpm = 1;
        
        err = HPMaddThread(cpu_id);
        if (err != 0)
        {
            ERROR_PLAIN_PRINT(Cannot get access to MSRs)
            return err;
        }
    }

    uint64_t tmp = 0x0ULL;
    err = HPMread(cpu_id, MSR_DEV, 0xC0010015, &tmp);
    if (err)
    {
        ERROR_PLAIN_PRINT(Cannot read register 0xC0010015);
        return err;
    }
    
    if (turbo)
    {
        tmp &= ~(1ULL<<25);
    }
    else
    {
        tmp |= (1ULL << 25);
    }
    err = HPMwrite(cpu_id, MSR_DEV, 0xC0010015, tmp);
    if (err)
    {
        ERROR_PLAIN_PRINT(Cannot write register 0xC0010015);
        return err;
    }

    if (own_hpm)
        HPMfinalize();
    return err == 0;
}

static int getIntelTurbo(const int cpu_id)
{
    int err = 0;
    int own_hpm = 0;

    if (!HPMinitialized())
    {
        HPMinit();
        own_hpm = 1;
        err = HPMaddThread(cpu_id);
        if (err != 0)
        {
            ERROR_PLAIN_PRINT(Cannot get access to MSRs)
            return err;
        }
    }

    uint64_t tmp = 0x0ULL;
    err = HPMread(cpu_id, MSR_DEV, MSR_IA32_MISC_ENABLE, &tmp);
    if (err)
    {
        ERROR_PRINT(Cannot read register 0x%x, MSR_IA32_MISC_ENABLE);
        return err;
    }
    if (own_hpm)
        HPMfinalize();
    err = ((tmp >> 38) & 0x1);
    return err == 0;
}

static int setIntelTurbo(const int cpu_id, const int turbo)
{
    int err = 0;
    int own_hpm = 0;

    if (!HPMinitialized())
    {
        HPMinit();
        own_hpm = 1;
        
        err = HPMaddThread(cpu_id);
        if (err != 0)
        {
            ERROR_PLAIN_PRINT(Cannot get access to MSRs)
            return err;
        }
    }

    uint64_t tmp = 0x0ULL;
    err = HPMread(cpu_id, MSR_DEV, MSR_IA32_MISC_ENABLE, &tmp);
    if (err)
    {
        ERROR_PRINT(Cannot read register 0x%x, MSR_IA32_MISC_ENABLE);
        return err;
    }
    if (turbo)
    {
        tmp &= ~(1ULL << 38);
    }
    else
    {
        tmp |= (1ULL << 38);
    }
    err = HPMwrite(cpu_id, MSR_DEV, MSR_IA32_MISC_ENABLE, tmp);
    if (err)
    {
        ERROR_PRINT(Cannot write register 0x%x, MSR_IA32_MISC_ENABLE);
        return err;
    }

    if (own_hpm)
        HPMfinalize();
    return err == 0;
}
#if !defined(__ARM_ARCH_7A__) && !defined(__ARM_ARCH_8A)
static int isAMD()
{
    unsigned int eax,ebx,ecx,edx;
    eax = 0x0;
    CPUID(eax,ebx,ecx,edx);
    if (ecx == 0x444d4163)
        return 1;
    return 0;
}
#else
static int isAMD()
{
    return 0;
}
#endif

int freq_getTurbo(const int cpu_id)
{
    if (drv == ACPICPUFREQ)
    {
        if (isAMD())
            return getAMDTurbo(cpu_id);
        return getIntelTurbo(cpu_id);
    }
    else if (drv == INTELPSTATE)
    {
        return freq_pstate_getTurbo(cpu_id);
    }
    return -1;
}

int freq_setTurbo(const int cpu_id, const int turbo)
{
    FILE *fpipe = NULL;
    char cmd[256];

    sprintf(cmd, "%s %d tur %d", daemon_path, cpu_id, turbo);
    if ( !(fpipe = (FILE*)popen(cmd,"r")) )
    {  // If fpipe is NULL
        fprintf(stderr, "Problems setting turbo mode of CPU %d", cpu_id);
        return 0;
    }
    pclose(fpipe);
    if (isAMD())
        return setAMDTurbo(cpu_id, turbo);
    else
        return setIntelTurbo(cpu_id, turbo);
    return 1;
}

int freq_setGovernor(const int cpu_id, const char* gov)
{
    FILE *fpipe = NULL;
    char cmd[256];
    char buff[256];

    sprintf(buff, "%s", daemon_path);
    if (access(buff, X_OK))
    {
        ERROR_PRINT(Daemon %s not executable, buff);
        return 0;
    }

    sprintf(cmd, "%s %d gov %s", daemon_path, cpu_id, gov);
    if ( !(fpipe = (FILE*)popen(cmd,"r")) )
    {  // If fpipe is NULL
        ERROR_PRINT(Problems setting cpu frequency of CPU %d, cpu_id);
        return 0;
    }
    if (pclose(fpipe))
        return 0;
    return 1;
}

char * freq_getAvailFreq(const int cpu_id )
{
    int i, j, k;
    FILE *fpipe = NULL;
    char cmd[256];
    char buff[2048];
    char tmp[10];
    char *eptr = NULL, *rptr = NULL, *sptr = NULL;
    double d = 0;
    int take_next = 0;
    bstring bbuff;

    sprintf(cmd, "%s 2>&1", daemon_path);
    if ( !(fpipe = (FILE*)popen(cmd,"r")) )
    {  // If fpipe is NULL
        ERROR_PRINT(Problem executing %s, daemon_path);
        return NULL;
    }
    while (fgets(buff, 2048, fpipe))
    {
        if (strncmp(buff, "Frequency steps:", 16) == 0)
        {
            //printf("Take next\n");
            take_next = 1;
            continue;
        }
        if (take_next)
        {
            int eidx = 0;
            //printf("Take %s\n", buff);
            eptr = malloc(strlen(buff) * sizeof(char));
            sptr = strtok(buff, " ");
            while (sptr != NULL)
            {
                d = atof(sptr);
                if (d > 0)
                {
                    eidx += snprintf(&(eptr[eidx]), 19, "%g ", d*1E-6);
                }
                sptr = strtok(NULL, " ");
            }
            break;
        }
    }
    if (pclose(fpipe) == -1)
    {
        return NULL;
    }
    for (int i=strlen(eptr)-1; i>= 0; i--)
    {
        if (eptr[i] == ' ')
        {
            eptr[i] = '\0';
        }
        else
        {
            break;
        }
    }
    return eptr;
}

char * freq_getAvailGovs(const int cpu_id )
{
    int i, j, k;
    FILE *fpipe = NULL;
    char cmd[256];
    char buff[2048];
    char tmp[10];
    char *eptr = NULL, *rptr = NULL, *sptr = NULL;
    double d = 0;
    int take_next = 0;
    bstring bbuff;

    sprintf(cmd, "%s 2>&1", daemon_path);
    if ( !(fpipe = (FILE*)popen(cmd,"r")) )
    {  // If fpipe is NULL
        ERROR_PRINT(Problem executing %s, daemon_path);
        return NULL;
    }
    while (fgets(buff, 2048, fpipe))
    {
        if (strncmp(buff, "Governors:", 10) == 0)
        {
            take_next = 1;
            continue;
        }
        if (take_next)
        {
            int eidx = 0;
            eptr = malloc((strlen(buff)+1) * sizeof(char));
            memset(eptr, 0, (strlen(buff)+1) * sizeof(char));
            strncpy(eptr, buff, strlen(buff));
            break;
        }
    }
    if (pclose(fpipe) == -1)
    {
        return NULL;
    }
    for (int i=strlen(eptr)-1; i>= 0; i--)
    {
        if (eptr[i] == ' ')
        {
            eptr[i] = '\0';
        }
        else
        {
            break;
        }
    }
    return eptr;
}

int freq_setUncoreFreqMin(const int socket_id, const uint64_t freq)
{
    int err = 0;
    int own_hpm = 0;
    int cpuId = -1;
    uint64_t f = freq / 100;
    double fmin, fmax;
    if (isAMD())
    {
        return 0;
    }
    err = _freq_getUncoreMinMax(socket_id, &cpuId, &fmin, &fmax);
    if (err < 0)
    {
        return err;
    }
    if (freq < (uint64_t)fmin)
    {
        ERROR_PRINT(Given frequency %llu MHz lower than system limit of %.0f MHz, freq, fmin);
        return -EINVAL;
    }
    if (freq > (uint64_t)fmax)
    {
        ERROR_PRINT(Given frequency %llu MHz higher than system limit of %.0f MHz, freq, fmax);
        return -EINVAL;
    }
    
    if (!HPMinitialized())
    {
        HPMinit();
        own_hpm = 1;
        err = HPMaddThread(cpuId);
        if (err != 0)
        {
            ERROR_PLAIN_PRINT(Cannot get access to MSRs)
            return err;
        }
    }

    uint64_t tmp = 0x0ULL;
    err = HPMread(cpuId, MSR_DEV, MSR_UNCORE_FREQ, &tmp);
    if (err)
    {
        //ERROR_PRINT(Cannot read register 0x%X on CPU %d, MSR_UNCORE_FREQ, cpuId);
        return err;
    }
    tmp &= ~(0xFF00);
    tmp |= (f<<8);
    err = HPMwrite(cpuId, MSR_DEV, MSR_UNCORE_FREQ, tmp);
    if (err)
    {
        ERROR_PRINT(Cannot write register 0x%X on CPU %d, MSR_UNCORE_FREQ, cpuId);
        return err;
    }

    if (own_hpm)
        HPMfinalize();
    return 0;
}




uint64_t freq_getUncoreFreqMin(const int socket_id)
{
    int err = 0;
    int own_hpm = 0;
    int cpuId = -1;
    if (isAMD())
    {
        return 0;
    }
    for (int i=0; i<cpuid_topology.numHWThreads; i++)
    {
        if (cpuid_topology.threadPool[i].packageId == socket_id)
        {
            cpuId = cpuid_topology.threadPool[i].apicId;
            break;
        }
    }
    if (cpuId < 0)
    {
        ERROR_PRINT(Unknown socket ID %d, socket_id);
        return 0;
    }
    if (!HPMinitialized())
    {
        HPMinit();
        own_hpm = 1;
        err = HPMaddThread(cpuId);
        if (err != 0)
        {
            ERROR_PLAIN_PRINT(Cannot get access to MSRs)
            return 0;
        }
    }

    uint64_t tmp = 0x0ULL;
    err = HPMread(cpuId, MSR_DEV, MSR_UNCORE_FREQ, &tmp);
    if (err)
    {
        //ERROR_PRINT(Cannot read register 0x%X on CPU %d, MSR_UNCORE_FREQ, cpuId);
        return 0;
    }
    tmp = ((tmp>>8) & 0xFFULL) * 100;

    if (own_hpm)
        HPMfinalize();
    return tmp;
}

int freq_setUncoreFreqMax(const int socket_id, const uint64_t freq)
{
    int err = 0;
    int own_hpm = 0;
    int cpuId = -1;
    uint64_t f = freq / 100;
    double fmin, fmax;
    if (isAMD())
    {
        return 0;
    }
    err = _freq_getUncoreMinMax(socket_id, &cpuId, &fmin, &fmax);
    if (err < 0)
    {
        return err;
    }
    if (freq < (uint64_t)fmin)
    {
        ERROR_PRINT(Given frequency %llu MHz lower than system limit of %.0f MHz, freq, fmin);
        return -EINVAL;
    }
    if (freq > (uint64_t)fmax)
    {
        ERROR_PRINT(Given frequency %llu MHz higher than system limit of %.0f MHz, freq, fmax);
        return -EINVAL;
    }

    if (!HPMinitialized())
    {
        HPMinit();
        own_hpm = 1;
        err = HPMaddThread(cpuId);
        if (err != 0)
        {
            ERROR_PLAIN_PRINT(Cannot get access to MSRs)
            return err;
        }
    }

    uint64_t tmp = 0x0ULL;
    err = HPMread(cpuId, MSR_DEV, MSR_UNCORE_FREQ, &tmp);
    if (err)
    {
        //ERROR_PRINT(Cannot read register 0x%X on CPU %d, MSR_UNCORE_FREQ, cpuId);
        return err;
    }
    tmp &= ~(0xFFULL);
    tmp |= (f & 0xFFULL);
    err = HPMwrite(cpuId, MSR_DEV, MSR_UNCORE_FREQ, tmp);
    if (err)
    {
        ERROR_PRINT(Cannot write register 0x%X on CPU %d, MSR_UNCORE_FREQ, cpuId);
        return err;
    }

    if (own_hpm)
        HPMfinalize();
    return 0;
}

uint64_t freq_getUncoreFreqMax(const int socket_id)
{
    int err = 0;
    int own_hpm = 0;
    int cpuId = -1;
    if (isAMD())
    {
        return 0;
    }
    for (int i=0; i<cpuid_topology.numHWThreads; i++)
    {
        if (cpuid_topology.threadPool[i].packageId == socket_id)
        {
            cpuId = cpuid_topology.threadPool[i].apicId;
            break;
        }
    }
    if (cpuId < 0)
    {
        ERROR_PRINT(Unknown socket ID %d, socket_id);
        return 0;
    }
    if (!HPMinitialized())
    {
        HPMinit();
        own_hpm = 1;
        err = HPMaddThread(cpuId);
        if (err != 0)
        {
            ERROR_PLAIN_PRINT(Cannot get access to MSRs)
            return 0;
        }
    }

    uint64_t tmp = 0x0ULL;
    err = HPMread(cpuId, MSR_DEV, MSR_UNCORE_FREQ, &tmp);
    if (err)
    {
        //ERROR_PRINT(Cannot read register 0x%X on CPU %d, MSR_UNCORE_FREQ, cpuId);
        return 0;
    }
    tmp = (tmp & 0xFFULL) * 100;

    if (own_hpm)
        HPMfinalize();
    return tmp;
}
