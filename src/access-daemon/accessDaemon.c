/*
 * =======================================================================================
 *
 *      Filename:  accessDaemon.c
 *
 *      Description:  Implementation of access daemon.
 *
 *      Version:   4.3.1
 *      Released:  04.01.2018
 *
 *      Authors:  Michael Meier, michael.meier@rrze.fau.de
 *                Jan Treibig (jt), jan.treibig@gmail.com,
 *                Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2018 RRZE, University Erlangen-Nuremberg
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
#include <stdint.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <fcntl.h>
#include <syslog.h>
#include <signal.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <sys/fsuid.h>
#include <getopt.h>
#include <dirent.h>
#include <sys/mman.h>
#include <fnmatch.h>

#include <types.h>
#include <registers.h>
#include <perfmon_haswellEP_counters.h>
#include <perfmon_ivybridgeEP_counters.h>
#include <perfmon_sandybridgeEP_counters.h>
#include <perfmon_broadwelld_counters.h>
#include <perfmon_broadwellEP_counters.h>
#include <perfmon_knl_counters.h>
#include <perfmon_skylakeX_counters.h>
#include <topology.h>
#include <cpuid.h>
#include <lock.h>

/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */

#define SA struct sockaddr
#define str(x) #x

#define CHECK_FILE_ERROR(func, msg)  \
    if ((func) == 0) { syslog(LOG_ERR, "ERROR - [%s:%d] " str(msg) " - %s \n", __FILE__, __LINE__, strerror(errno)); }

#define LOG_AND_EXIT_IF_ERROR(func, msg)  \
    if ((func) < 0) {  \
        syslog(LOG_ERR, "ERROR - [%s:%d] " str(msg) " - %s \n", __FILE__, __LINE__, strerror(errno)); \
        exit(EXIT_FAILURE); \
    }

#define PCI_ROOT_PATH    "/proc/bus/pci/"
#define MAX_PATH_LENGTH   80
//#define MAX_NUM_NODES    4

#define PCM_CLIENT_IMC_BAR_OFFSET       (0x0048)
#define PCM_CLIENT_IMC_DRAM_IO_REQUESTS  (0x5048)
#define PCM_CLIENT_IMC_DRAM_DATA_READS  (0x5050)
#define PCM_CLIENT_IMC_DRAM_DATA_WRITES (0x5054)
#define PCM_CLIENT_IMC_MMAP_SIZE (0x6000)

/* Lock file controlled from outside which prevents likwid to start.
 * Can be used to synchronize access to the hardware counters
 * with an external monitoring system. */

/* #####   TYPE DEFINITIONS   ########### */

typedef int (*AllowedPrototype)(uint32_t);
typedef int (*AllowedPciPrototype)(PciDeviceType, uint32_t);

/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

static int sockfd = -1;
static int connfd = -1; /* temporary in to make it compile */
static char* filepath;
static const char* ident = "accessD";
static AllowedPrototype allowed = NULL;
static AllowedPciPrototype allowedPci = NULL;
/*static int FD_MSR[MAX_NUM_THREADS];*/
/*static int FD_PCI[MAX_NUM_NODES][MAX_NUM_PCI_DEVICES];*/
static int* FD_MSR = NULL;
static int** FD_PCI = NULL;
static int isPCIUncore = 0;
static int isPCI64 = 0;
static PciDevice* pci_devices_daemon = NULL;
static char pci_filepath[MAX_PATH_LENGTH];
static int num_pmc_counters = 0;

static int clientmem_handle = -1;
static char *clientmem_addr = NULL;
static int isClientMem = 0;

/* Socket to bus mapping -- will be determined at runtime;
 * typical mappings are:
 * Socket  Bus (2S)  Bus (4s)  Bus (8s)
 *   0        0xff      0x3f     0x1f
 *   1        0x7f      0x7f     0x3f
 *   2                  0xbf     0x5f
 *   3                  0xff     0x7f
 *                               0x9f
 *                               0xbf
 *                               0xdf
 *                               0xff
 */
static int avail_sockets = 0;
static int avail_cpus = 0;
static int avail_nodes = 0;

static int getNumberOfCPUs()
{
    FILE* fpipe = NULL;
    char cmd[1024] = "cat /proc/cpuinfo | grep \"processor\" | sort -u | wc -l";
    char buff[1024];
    if ( !(fpipe = popen(cmd,"r")) )
    {
        return 0;
    }
    char* ptr = fgets(buff, 1024, fpipe);
    if (ptr)
    {
        return atoi(buff);
    }
    return 0;
}

static int getNumberOfSockets()
{
    FILE* fpipe = NULL;
    char cmd[1024] = "cat /proc/cpuinfo | grep \"physical id\" | sort -u | wc -l";
    char buff[1024];
    if ( !(fpipe = popen(cmd,"r")) )
    {
        return 0;
    }
    char* ptr = fgets(buff, 1024, fpipe);
    if (ptr)
    {
        return atoi(buff);
    }
    return 0;
}

static int getNumberOfNumaNodes()
{
    DIR *dp = NULL;
    struct dirent *ep = NULL;
    int count = 0;
    dp = opendir ("/sys/devices/system/node/");
    if (dp != NULL)
    {
        while (ep = readdir (dp))
        {
            if (fnmatch("node?*", ep->d_name, FNM_PATHNAME) == 0)
            {
                count++;
            }
        }
        closedir(dp);
    }
    return count;
}

void __attribute__((constructor (101))) init_accessdaemon(void)
{

    FILE *rdpmc_file = NULL;
    FILE *nmi_watchdog_file = NULL;
    int retries = 10;

    do {
        avail_cpus = getNumberOfCPUs();
        retries--;
    } while (avail_cpus == 0 && retries > 0);
    if (retries <= 0) return;
    retries = 10;
    do {
        avail_sockets = getNumberOfSockets();
        retries--;
    } while (avail_sockets == 0 && retries > 0);
    if (retries <= 0) return;
    retries = 10;
    do {
        avail_nodes = getNumberOfNumaNodes();
        retries--;
    } while (avail_nodes == 0 && retries > 0);
    if (retries <= 0) return;

    FD_MSR = malloc(avail_cpus * sizeof(int));
    if (!FD_MSR)
        return;

    FD_PCI = malloc(avail_sockets * sizeof(int*));
    if (!FD_PCI)
        return;
    for (int i = 0; i < avail_sockets; i++)
    {
        FD_PCI[i] = malloc(MAX_NUM_PCI_DEVICES * sizeof(int));
        if (!FD_PCI[i])
        {
            for (int j = i-1; j >= 0; j--)
            {
                free(FD_PCI[j]);
            }
            free(FD_PCI);
            return;
        }
    }
    // Explicitly allow RDPMC instruction
    rdpmc_file = fopen("/sys/bus/event_source/devices/cpu/rdpmc", "wb");
    if (rdpmc_file)
    {
        fputc('2', rdpmc_file);
        fclose(rdpmc_file);
    }
    // Explicitly disable NMI watchdog (Uses the first fixed-purpose counter INSTR_RETIRED_ANY)
    nmi_watchdog_file = fopen("/proc/sys/kernel/nmi_watchdog", "wb");
    if (nmi_watchdog_file)
    {
        fputc('0', nmi_watchdog_file);
        fclose(nmi_watchdog_file);
    }
}

void __attribute__((destructor (101))) close_accessdaemon(void)
{
    if (FD_PCI)
    {
        for (int i = 0; i < avail_sockets; i++)
        {
            if (FD_PCI[i])
            {
                free(FD_PCI[i]);
                FD_PCI[i] = NULL;
            }
        }
        free(FD_PCI);
        FD_PCI = NULL;
    }
    if (FD_MSR)
    {
        free(FD_MSR);
        FD_MSR = NULL;
    }
}

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

static int
allowed_intel(uint32_t reg)
{
    if ( ((reg & 0x0F0U) == 0x0C0U) ||
            ((reg & 0x190U) == 0x180U) ||
            ((reg & 0x190U) == 0x190U && num_pmc_counters > 4) ||
            ((reg & 0xF00U) == 0x300U) ||
            ((reg & 0xF00U) == 0xC00U) ||
            ((reg & 0xF00U) == 0xD00U) ||
            ((reg & 0xF00U) == 0xE00U) ||
            ((reg & 0xF00U) == 0xF00U) ||
            (reg == 0x1A0)  ||
            (reg == 0x1A4)  ||
            (reg == 0x0CE)  ||
            (reg == 0x19C)  ||
            (reg == 0x1A2)  ||
            (reg == 0x1AD)  ||
            (reg == 0x1AE)  ||
            (reg == 0x1AF)  ||
            (reg == 0x1AC)  ||
            (reg == 0x1A6)  ||
            (reg == 0x1A7)  ||
            (reg == 0x620)  ||
            (reg == 0xCD)   ||
            (reg == 0x1B0)  ||
            (reg == 0x1B1))
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

static int
allowed_sandybridge(uint32_t reg)
{
    if ((allowed_intel(reg)) ||
        (((reg & 0xF00U) == 0x600U)) ||
        (((reg & 0xF00U) == 0x700U)) ||
        (reg == MSR_ALT_PEBS))
    {
        return 1;
    }
    return 0;
}

static int
allowed_pci_sandybridge(PciDeviceType type, uint32_t reg)
{
    switch (type)
    {
        case NODEVTYPE:
            return 1;
            break;
        case R3QPI:
            if ((reg == PCI_UNC_R3QPI_PMON_BOX_CTL) ||
                (reg == PCI_UNC_R3QPI_PMON_BOX_STATUS) ||
                (reg == PCI_UNC_R3QPI_PMON_CTL_0) ||
                (reg == PCI_UNC_R3QPI_PMON_CTL_1) ||
                (reg == PCI_UNC_R3QPI_PMON_CTL_2) ||
                (reg == PCI_UNC_R3QPI_PMON_CTR_0_A) ||
                (reg == PCI_UNC_R3QPI_PMON_CTR_1_A) ||
                (reg == PCI_UNC_R3QPI_PMON_CTR_2_A) ||
                (reg == PCI_UNC_R3QPI_PMON_CTR_0_B) ||
                (reg == PCI_UNC_R3QPI_PMON_CTR_1_B) ||
                (reg == PCI_UNC_R3QPI_PMON_CTR_2_B))
            {
                return 1;
            }
            return 0;
            break;
        case R2PCIE:
            if ((reg == PCI_UNC_R2PCIE_PMON_BOX_CTL) ||
                (reg == PCI_UNC_R2PCIE_PMON_BOX_STATUS) ||
                (reg == PCI_UNC_R2PCIE_PMON_CTL_0) ||
                (reg == PCI_UNC_R2PCIE_PMON_CTL_1) ||
                (reg == PCI_UNC_R2PCIE_PMON_CTL_2) ||
                (reg == PCI_UNC_R2PCIE_PMON_CTL_3) ||
                (reg == PCI_UNC_R2PCIE_PMON_CTR_0_A) ||
                (reg == PCI_UNC_R2PCIE_PMON_CTR_1_A) ||
                (reg == PCI_UNC_R2PCIE_PMON_CTR_2_A) ||
                (reg == PCI_UNC_R2PCIE_PMON_CTR_3_A) ||
                (reg == PCI_UNC_R2PCIE_PMON_CTR_0_B) ||
                (reg == PCI_UNC_R2PCIE_PMON_CTR_1_B) ||
                (reg == PCI_UNC_R2PCIE_PMON_CTR_2_B) ||
                (reg == PCI_UNC_R2PCIE_PMON_CTR_3_B))
            {
                return 1;
            }
            return 0;
            break;
        case IMC:
            if ((reg == PCI_UNC_MC_PMON_BOX_CTL) ||
                (reg == PCI_UNC_MC_PMON_BOX_STATUS) ||
                (reg == PCI_UNC_MC_PMON_CTL_0) ||
                (reg == PCI_UNC_MC_PMON_CTL_1) ||
                (reg == PCI_UNC_MC_PMON_CTL_2) ||
                (reg == PCI_UNC_MC_PMON_CTL_3) ||
                (reg == PCI_UNC_MC_PMON_CTR_0_A) ||
                (reg == PCI_UNC_MC_PMON_CTR_1_A) ||
                (reg == PCI_UNC_MC_PMON_CTR_2_A) ||
                (reg == PCI_UNC_MC_PMON_CTR_3_A) ||
                (reg == PCI_UNC_MC_PMON_CTR_0_B) ||
                (reg == PCI_UNC_MC_PMON_CTR_1_B) ||
                (reg == PCI_UNC_MC_PMON_CTR_2_B) ||
                (reg == PCI_UNC_MC_PMON_CTR_3_B) ||
                (reg == PCI_UNC_MC_PMON_FIXED_CTL) ||
                (reg == PCI_UNC_MC_PMON_FIXED_CTR_A) ||
                (reg == PCI_UNC_MC_PMON_FIXED_CTR_B))
            {
                return 1;
            }
            return 0;
            break;
        case HA:
            if ((reg == PCI_UNC_HA_PMON_BOX_CTL) ||
                (reg == PCI_UNC_HA_PMON_BOX_STATUS) ||
                (reg == PCI_UNC_HA_PMON_CTL_0) ||
                (reg == PCI_UNC_HA_PMON_CTL_1) ||
                (reg == PCI_UNC_HA_PMON_CTL_2) ||
                (reg == PCI_UNC_HA_PMON_CTL_3) ||
                (reg == PCI_UNC_HA_PMON_CTR_0_A) ||
                (reg == PCI_UNC_HA_PMON_CTR_1_A) ||
                (reg == PCI_UNC_HA_PMON_CTR_2_A) ||
                (reg == PCI_UNC_HA_PMON_CTR_3_A) ||
                (reg == PCI_UNC_HA_PMON_CTR_0_B) ||
                (reg == PCI_UNC_HA_PMON_CTR_1_B) ||
                (reg == PCI_UNC_HA_PMON_CTR_2_B) ||
                (reg == PCI_UNC_HA_PMON_CTR_3_B) ||
                (reg == PCI_UNC_HA_PMON_OPCODEMATCH) ||
                (reg == PCI_UNC_HA_PMON_ADDRMATCH0) ||
                (reg == PCI_UNC_HA_PMON_ADDRMATCH1))
            {
                return 1;
            }
            return 0;
            break;
        case QPI:
            if ((reg == PCI_UNC_QPI_PMON_BOX_CTL) ||
                (reg == PCI_UNC_QPI_PMON_BOX_STATUS) ||
                (reg == PCI_UNC_QPI_PMON_CTL_0) ||
                (reg == PCI_UNC_QPI_PMON_CTL_1) ||
                (reg == PCI_UNC_QPI_PMON_CTL_2) ||
                (reg == PCI_UNC_QPI_PMON_CTL_3) ||
                (reg == PCI_UNC_QPI_PMON_CTR_0_A) ||
                (reg == PCI_UNC_QPI_PMON_CTR_1_A) ||
                (reg == PCI_UNC_QPI_PMON_CTR_2_A) ||
                (reg == PCI_UNC_QPI_PMON_CTR_3_A) ||
                (reg == PCI_UNC_QPI_PMON_CTR_0_B) ||
                (reg == PCI_UNC_QPI_PMON_CTR_1_B) ||
                (reg == PCI_UNC_QPI_PMON_CTR_2_B) ||
                (reg == PCI_UNC_QPI_PMON_CTR_3_B) ||
                (reg == PCI_UNC_QPI_PMON_MASK_0) ||
                (reg == PCI_UNC_QPI_PMON_MASK_1) ||
                (reg == PCI_UNC_QPI_PMON_MATCH_0) ||
                (reg == PCI_UNC_QPI_PMON_MATCH_1) ||
                (reg == PCI_UNC_QPI_RATE_STATUS))
            {
                return 1;
            }
            return 0;
            break;
        case IRP:
            if ((reg == PCI_UNC_IRP_PMON_BOX_STATUS) ||
                (reg == PCI_UNC_IRP_PMON_BOX_CTL) ||
                (reg == PCI_UNC_IRP0_PMON_CTL_0) ||
                (reg == PCI_UNC_IRP0_PMON_CTL_1) ||
                (reg == PCI_UNC_IRP0_PMON_CTR_0) ||
                (reg == PCI_UNC_IRP0_PMON_CTR_1) ||
                (reg == PCI_UNC_IRP1_PMON_CTL_0) ||
                (reg == PCI_UNC_IRP1_PMON_CTL_1) ||
                (reg == PCI_UNC_IRP1_PMON_CTR_0) ||
                (reg == PCI_UNC_IRP1_PMON_CTR_1))
            {
                return 1;
            }
            return 0;
            break;
        default:
            return 0;
            break;
    }
    return 0;
}

static int
allowed_haswell(uint32_t reg)
{
    if ((allowed_intel(reg)) ||
        (allowed_sandybridge(reg)) ||
        (((reg & 0xF00U) == 0x700U)))
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

static int
allowed_pci_haswell(PciDeviceType type, uint32_t reg)
{
    switch (type)
    {
        case NODEVTYPE:
            return 1;
            break;
        case R3QPI:
            if ((reg == PCI_UNC_R3QPI_PMON_BOX_CTL) ||
                (reg == PCI_UNC_R3QPI_PMON_BOX_STATUS) ||
                (reg == PCI_UNC_R3QPI_PMON_CTL_0) ||
                (reg == PCI_UNC_R3QPI_PMON_CTL_1) ||
                (reg == PCI_UNC_R3QPI_PMON_CTL_2) ||
                (reg == PCI_UNC_R3QPI_PMON_CTR_0_A) ||
                (reg == PCI_UNC_R3QPI_PMON_CTR_1_A) ||
                (reg == PCI_UNC_R3QPI_PMON_CTR_2_A) ||
                (reg == PCI_UNC_R3QPI_PMON_CTR_0_B) ||
                (reg == PCI_UNC_R3QPI_PMON_CTR_1_B) ||
                (reg == PCI_UNC_R3QPI_PMON_CTR_2_B))
            {
                return 1;
            }
            return 0;
            break;
        case R2PCIE:
            if ((reg == PCI_UNC_R2PCIE_PMON_BOX_CTL) ||
                (reg == PCI_UNC_R2PCIE_PMON_BOX_STATUS) ||
                (reg == PCI_UNC_R2PCIE_PMON_CTL_0) ||
                (reg == PCI_UNC_R2PCIE_PMON_CTL_1) ||
                (reg == PCI_UNC_R2PCIE_PMON_CTL_2) ||
                (reg == PCI_UNC_R2PCIE_PMON_CTL_3) ||
                (reg == PCI_UNC_R2PCIE_PMON_CTR_0_A) ||
                (reg == PCI_UNC_R2PCIE_PMON_CTR_1_A) ||
                (reg == PCI_UNC_R2PCIE_PMON_CTR_2_A) ||
                (reg == PCI_UNC_R2PCIE_PMON_CTR_3_A) ||
                (reg == PCI_UNC_R2PCIE_PMON_CTR_0_B) ||
                (reg == PCI_UNC_R2PCIE_PMON_CTR_1_B) ||
                (reg == PCI_UNC_R2PCIE_PMON_CTR_2_B) ||
                (reg == PCI_UNC_R2PCIE_PMON_CTR_3_B))
            {
                return 1;
            }
            return 0;
            break;
        case IMC:
            if ((reg == PCI_UNC_MC_PMON_BOX_CTL) ||
                (reg == PCI_UNC_MC_PMON_BOX_STATUS) ||
                (reg == PCI_UNC_MC_PMON_CTL_0) ||
                (reg == PCI_UNC_MC_PMON_CTL_1) ||
                (reg == PCI_UNC_MC_PMON_CTL_2) ||
                (reg == PCI_UNC_MC_PMON_CTL_3) ||
                (reg == PCI_UNC_MC_PMON_CTR_0_A) ||
                (reg == PCI_UNC_MC_PMON_CTR_1_A) ||
                (reg == PCI_UNC_MC_PMON_CTR_2_A) ||
                (reg == PCI_UNC_MC_PMON_CTR_3_A) ||
                (reg == PCI_UNC_MC_PMON_CTR_0_B) ||
                (reg == PCI_UNC_MC_PMON_CTR_1_B) ||
                (reg == PCI_UNC_MC_PMON_CTR_2_B) ||
                (reg == PCI_UNC_MC_PMON_CTR_3_B) ||
                (reg == PCI_UNC_MC_PMON_FIXED_CTL) ||
                (reg == PCI_UNC_MC_PMON_FIXED_CTR_A) ||
                (reg == PCI_UNC_MC_PMON_FIXED_CTR_B))
            {
                return 1;
            }
            return 0;
            break;
        case HA:
            if ((reg == PCI_UNC_HA_PMON_BOX_CTL) ||
                (reg == PCI_UNC_HA_PMON_BOX_STATUS) ||
                (reg == PCI_UNC_HA_PMON_CTL_0) ||
                (reg == PCI_UNC_HA_PMON_CTL_1) ||
                (reg == PCI_UNC_HA_PMON_CTL_2) ||
                (reg == PCI_UNC_HA_PMON_CTL_3) ||
                (reg == PCI_UNC_HA_PMON_CTR_0_A) ||
                (reg == PCI_UNC_HA_PMON_CTR_1_A) ||
                (reg == PCI_UNC_HA_PMON_CTR_2_A) ||
                (reg == PCI_UNC_HA_PMON_CTR_3_A) ||
                (reg == PCI_UNC_HA_PMON_CTR_0_B) ||
                (reg == PCI_UNC_HA_PMON_CTR_1_B) ||
                (reg == PCI_UNC_HA_PMON_CTR_2_B) ||
                (reg == PCI_UNC_HA_PMON_CTR_3_B) ||
                (reg == PCI_UNC_HA_PMON_OPCODEMATCH) ||
                (reg == PCI_UNC_HA_PMON_ADDRMATCH0) ||
                (reg == PCI_UNC_HA_PMON_ADDRMATCH1))
            {
                return 1;
            }
            return 0;
            break;
        case QPI:
            if ((reg == PCI_UNC_V3_QPI_PMON_BOX_CTL) ||
                (reg == PCI_UNC_V3_QPI_PMON_BOX_STATUS) ||
                (reg == PCI_UNC_V3_QPI_PMON_CTL_0) ||
                (reg == PCI_UNC_V3_QPI_PMON_CTL_1) ||
                (reg == PCI_UNC_V3_QPI_PMON_CTL_2) ||
                (reg == PCI_UNC_V3_QPI_PMON_CTL_3) ||
                (reg == PCI_UNC_V3_QPI_PMON_CTR_0_A) ||
                (reg == PCI_UNC_V3_QPI_PMON_CTR_1_A) ||
                (reg == PCI_UNC_V3_QPI_PMON_CTR_2_A) ||
                (reg == PCI_UNC_V3_QPI_PMON_CTR_3_A) ||
                (reg == PCI_UNC_V3_QPI_PMON_CTR_0_B) ||
                (reg == PCI_UNC_V3_QPI_PMON_CTR_1_B) ||
                (reg == PCI_UNC_V3_QPI_PMON_CTR_2_B) ||
                (reg == PCI_UNC_V3_QPI_PMON_CTR_3_B) ||
                (reg == PCI_UNC_V3_QPI_PMON_RX_MASK_0) ||
                (reg == PCI_UNC_V3_QPI_PMON_RX_MASK_1) ||
                (reg == PCI_UNC_V3_QPI_PMON_RX_MATCH_0) ||
                (reg == PCI_UNC_V3_QPI_PMON_RX_MATCH_1) ||
                (reg == PCI_UNC_V3_QPI_PMON_TX_MASK_0) ||
                (reg == PCI_UNC_V3_QPI_PMON_TX_MASK_1) ||
                (reg == PCI_UNC_V3_QPI_PMON_TX_MATCH_0) ||
                (reg == PCI_UNC_V3_QPI_PMON_TX_MATCH_1) ||
                (reg == PCI_UNC_V3_QPI_RATE_STATUS) ||
                (reg == PCI_UNC_V3_QPI_LINK_LLR) ||
                (reg == PCI_UNC_V3_QPI_LINK_IDLE))
            {
                return 1;
            }
            return 0;
            break;
        default:
            return 0;
            break;
    }
    return 0;
}

static int
allowed_silvermont(uint32_t reg)
{

    if ( ((reg & 0x0F8U) == 0x0C0U) ||
            ((reg & 0xFF0U) == 0x180U) ||
            ((reg & 0xF00U) == 0x300U) ||
            ((reg & 0xF00U) == 0x600U) ||
            ((reg & 0xF00U) == 0xC00U) ||
            ((reg & 0xF00U) == 0xD00U) ||
            (reg == 0x1A0)  ||
            (reg == 0x0CE)  ||
            (reg == 0x1AD)  ||
            (reg == 0x19C)  ||
            (reg == 0x1A2)  ||
            (reg == 0x1A6) ||
            (reg == 0x1A7))
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

static int allowed_knl(uint32_t reg)
{
    if (allowed_silvermont(reg))
        return 1;
    else
    {
        if (((reg & 0xF00U) == 0x700U) ||
            ((reg & 0xF00U) == 0xE00U) ||
            ((reg & 0xF00U) == 0xF00U) ||
            (reg == MSR_PREFETCH_ENABLE))
            return 1;
    }
    return 0;
}

static int allowed_pci_knl(PciDeviceType type, uint32_t reg)
{
    switch(type)
    {
	case EDC:
	    if ((reg == PCI_MIC2_EDC_U_CTR0_A) ||
		(reg == PCI_MIC2_EDC_U_CTR0_B) ||
	        (reg == PCI_MIC2_EDC_U_CTR1_A) ||
		(reg == PCI_MIC2_EDC_U_CTR1_B) ||
	        (reg == PCI_MIC2_EDC_U_CTR2_A) ||
		(reg == PCI_MIC2_EDC_U_CTR2_B) ||
	        (reg == PCI_MIC2_EDC_U_CTR3_A) ||
		(reg == PCI_MIC2_EDC_U_CTR3_B) ||
		(reg == PCI_MIC2_EDC_U_CTRL0) ||
		(reg == PCI_MIC2_EDC_U_CTRL1) ||
		(reg == PCI_MIC2_EDC_U_CTRL2) ||
		(reg == PCI_MIC2_EDC_U_CTRL3) ||
		(reg == PCI_MIC2_EDC_U_BOX_CTRL) ||
		(reg == PCI_MIC2_EDC_U_BOX_STATUS) ||
		(reg == PCI_MIC2_EDC_U_FIXED_CTR_A) ||
		(reg == PCI_MIC2_EDC_U_FIXED_CTR_B) ||
		(reg == PCI_MIC2_EDC_U_FIXED_CTRL) ||
	        (reg == PCI_MIC2_EDC_D_CTR0_A) ||
		(reg == PCI_MIC2_EDC_D_CTR0_B) ||
	        (reg == PCI_MIC2_EDC_D_CTR1_A) ||
		(reg == PCI_MIC2_EDC_D_CTR1_B) ||
	        (reg == PCI_MIC2_EDC_D_CTR2_A) ||
		(reg == PCI_MIC2_EDC_D_CTR2_B) ||
	        (reg == PCI_MIC2_EDC_D_CTR3_A) ||
		(reg == PCI_MIC2_EDC_D_CTR3_B) ||
		(reg == PCI_MIC2_EDC_D_CTRL0) ||
		(reg == PCI_MIC2_EDC_D_CTRL1) ||
		(reg == PCI_MIC2_EDC_D_CTRL2) ||
		(reg == PCI_MIC2_EDC_D_CTRL3) ||
		(reg == PCI_MIC2_EDC_D_BOX_CTRL) ||
		(reg == PCI_MIC2_EDC_D_BOX_STATUS) ||
		(reg == PCI_MIC2_EDC_D_FIXED_CTR_A) ||
		(reg == PCI_MIC2_EDC_D_FIXED_CTR_B) ||
		(reg == PCI_MIC2_EDC_D_FIXED_CTRL))
	    {
		return 1;
	    }
	    break;
	case IMC:
	    if ((reg == PCI_MIC2_MC_U_CTR0_A) ||
		(reg == PCI_MIC2_MC_U_CTR0_B) ||
	        (reg == PCI_MIC2_MC_U_CTR1_A) ||
		(reg == PCI_MIC2_MC_U_CTR1_B) ||
	        (reg == PCI_MIC2_MC_U_CTR2_A) ||
		(reg == PCI_MIC2_MC_U_CTR2_B) ||
	        (reg == PCI_MIC2_MC_U_CTR3_A) ||
		(reg == PCI_MIC2_MC_U_CTR3_B) ||
		(reg == PCI_MIC2_MC_U_CTRL0) ||
		(reg == PCI_MIC2_MC_U_CTRL1) ||
		(reg == PCI_MIC2_MC_U_CTRL2) ||
		(reg == PCI_MIC2_MC_U_CTRL3) ||
		(reg == PCI_MIC2_MC_U_BOX_CTRL) ||
		(reg == PCI_MIC2_MC_U_BOX_STATUS) ||
		(reg == PCI_MIC2_MC_U_FIXED_CTR_A) ||
		(reg == PCI_MIC2_MC_U_FIXED_CTR_B) ||
		(reg == PCI_MIC2_MC_U_FIXED_CTRL) ||
	        (reg == PCI_MIC2_MC_D_CTR0_A) ||
		(reg == PCI_MIC2_MC_D_CTR0_B) ||
	        (reg == PCI_MIC2_MC_D_CTR1_A) ||
		(reg == PCI_MIC2_MC_D_CTR1_B) ||
	        (reg == PCI_MIC2_MC_D_CTR2_A) ||
		(reg == PCI_MIC2_MC_D_CTR2_B) ||
	        (reg == PCI_MIC2_MC_D_CTR3_A) ||
		(reg == PCI_MIC2_MC_D_CTR3_B) ||
		(reg == PCI_MIC2_MC_D_CTRL0) ||
		(reg == PCI_MIC2_MC_D_CTRL1) ||
		(reg == PCI_MIC2_MC_D_CTRL2) ||
		(reg == PCI_MIC2_MC_D_CTRL3) ||
		(reg == PCI_MIC2_MC_D_BOX_CTRL) ||
		(reg == PCI_MIC2_MC_D_BOX_STATUS) ||
		(reg == PCI_MIC2_MC_D_FIXED_CTR_A) ||
		(reg == PCI_MIC2_MC_D_FIXED_CTR_B) ||
		(reg == PCI_MIC2_MC_D_FIXED_CTRL))
	    {
		return 1;
	    }
	    break;
	case R2PCIE:
	    if ((reg == PCI_MIC2_M2PCIE_CTR0_A) ||
		(reg == PCI_MIC2_M2PCIE_CTR0_B) ||
		(reg == PCI_MIC2_M2PCIE_CTR1_A) ||
                (reg == PCI_MIC2_M2PCIE_CTR1_B) ||
		(reg == PCI_MIC2_M2PCIE_CTR2_A) ||
                (reg == PCI_MIC2_M2PCIE_CTR2_B) ||
		(reg == PCI_MIC2_M2PCIE_CTR3_A) ||
                (reg == PCI_MIC2_M2PCIE_CTR3_B) ||
		(reg == PCI_MIC2_M2PCIE_CTRL0) ||
		(reg == PCI_MIC2_M2PCIE_CTRL1) ||
		(reg == PCI_MIC2_M2PCIE_CTRL2) ||
		(reg == PCI_MIC2_M2PCIE_CTRL3) ||
		(reg == PCI_MIC2_M2PCIE_BOX_CTRL) ||
		(reg == PCI_MIC2_M2PCIE_BOX_STATUS))
	    {
		return 1;
	    }
	    break;
	case IRP:
	    if ((reg == PCI_MIC2_IRP_CTR0) ||
		(reg == PCI_MIC2_IRP_CTR1) ||
		(reg == PCI_MIC2_IRP_CTRL0) ||
		(reg == PCI_MIC2_IRP_CTRL1) ||
		(reg == PCI_MIC2_IRP_BOX_CTRL) ||
		(reg == PCI_MIC2_IRP_BOX_STATUS))
	    {
		return 1;
	    }
	    break;
	default:
	    break;

    }
    return 0;
}

static int allowed_skx(uint32_t reg)
{
    if (allowed_sandybridge(reg))
        return 1;
    else
    {
        if (((reg & 0xF00U) == 0x700U) ||
            ((reg & 0xF00U) == 0xE00U) ||
            ((reg & 0xF00U) == 0xF00U) ||
            (reg == MSR_PREFETCH_ENABLE) ||
            (reg == TSX_FORCE_ABORT) ||
            ((reg & 0xA00U) == 0xA00U))
            return 1;
    }
    return 0;
}


static int allowed_pci_skx(PciDeviceType type, uint32_t reg)
{
    switch(type)
    {
        case HA:
            if ((reg == MSR_UNC_SKX_M2M_PMON_CTL0)||
                (reg == MSR_UNC_SKX_M2M_PMON_CTL1)||
                (reg == MSR_UNC_SKX_M2M_PMON_CTL2)||
                (reg == MSR_UNC_SKX_M2M_PMON_CTL3)||
                (reg == MSR_UNC_SKX_M2M_PMON_CTR0) ||
                (reg == MSR_UNC_SKX_M2M_PMON_CTR1) ||
                (reg == MSR_UNC_SKX_M2M_PMON_CTR2) ||
                (reg == MSR_UNC_SKX_M2M_PMON_CTR3) ||
                (reg == MSR_UNC_SKX_M2M_PMON_BOX_CTL) ||
                (reg == MSR_UNC_SKX_M2M_PMON_BOX_STATUS))
                return 1;
            break;
        case IMC:
            if ((reg == PCI_UNC_SKX_MC_PMON_CTL0) ||
                (reg == PCI_UNC_SKX_MC_PMON_CTL1) ||
                (reg == PCI_UNC_SKX_MC_PMON_CTL2) ||
                (reg == PCI_UNC_SKX_MC_PMON_CTL3) ||
                (reg == PCI_UNC_SKX_MC_PMON_CTR0) ||
                (reg == PCI_UNC_SKX_MC_PMON_CTR1) ||
                (reg == PCI_UNC_SKX_MC_PMON_CTR2) ||
                (reg == PCI_UNC_SKX_MC_PMON_CTR3) ||
                (reg == PCI_UNC_SKX_MC_PMON_FIXED_CTL) ||
                (reg == PCI_UNC_SKX_MC_PMON_FIXED_CTR) ||
                (reg == PCI_UNC_SKX_MC_PMON_BOX_CTL) ||
                (reg == PCI_UNC_SKX_MC_PMON_BOX_STATUS))
                return 1;
            break;
        case QPI:
            if ((reg == MSR_UNC_SKX_UPI_PMON_CTL0) ||
                (reg == MSR_UNC_SKX_UPI_PMON_CTL1) ||
                (reg == MSR_UNC_SKX_UPI_PMON_CTL2) ||
                (reg == MSR_UNC_SKX_UPI_PMON_CTL3) ||
                (reg == MSR_UNC_SKX_UPI_PMON_CTR0) ||
                (reg == MSR_UNC_SKX_UPI_PMON_CTR1) ||
                (reg == MSR_UNC_SKX_UPI_PMON_CTR2) ||
                (reg == MSR_UNC_SKX_UPI_PMON_CTR3) ||
                (reg == MSR_UNC_SKX_UPI_PMON_BOX_CTL) ||
                (reg == MSR_UNC_SKX_UPI_PMON_BOX_STATUS))
                return 1;
            break;
        case R3QPI:
            if ((reg == MSR_UNC_SKX_M3UPI_PMON_CTL0) ||
                (reg == MSR_UNC_SKX_M3UPI_PMON_CTL1) ||
                (reg == MSR_UNC_SKX_M3UPI_PMON_CTL2) ||
                (reg == MSR_UNC_SKX_M3UPI_PMON_CTR0) ||
                (reg == MSR_UNC_SKX_M3UPI_PMON_CTR1) ||
                (reg == MSR_UNC_SKX_M3UPI_PMON_CTR2) ||
                (reg == MSR_UNC_SKX_M3UPI_PMON_BOX_CTL) ||
                (reg == MSR_UNC_SKX_M3UPI_PMON_BOX_STATUS))
                return 1;
            break;
    }
    return 0;
}

static int allowed_amd(uint32_t reg)
{
    if ( (reg & 0xFFFFFFF0U) == 0xC0010000U)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

static int
allowed_amd15(uint32_t reg)
{
    if ( ((reg & 0xFFFFFFF0U) == 0xC0010000U) ||
            ((reg & 0xFFFFFFF0U) == 0xC0010200U) ||
            ((reg & 0xFFFFFFF8U) == 0xC0010240U))
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

static int
allowed_amd16(uint32_t reg)
{
    if ( ((reg & 0xFFFFFFF0U) == 0xC0010000U) ||
            ((reg & 0xFFFFFFF8U) == 0xC0010240U))
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

static int
allowed_amd17(uint32_t reg)
{
    if ((reg >= 0xC0010200 && reg <= 0xC0010207) ||
        (reg >= 0xC0010230 && reg <= 0xC001023B) ||
        (reg >= 0xC0010299 && reg <= 0xC001029B) ||
        (reg >= 0xC00000E7 && reg <= 0xC00000E9) ||
        (reg >= 0xC0010240 && reg <= 0xC0010247) ||
        (reg == 0xC0010015) ||
        (reg == 0xC0010010) ||
        (reg == 0xC0000080))
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

static int
clientmem_getStartAddr(uint64_t* startAddr)
{
    uint64_t imcbar = 0;

    int pcihandle = open("/proc/bus/pci/00/00.0", O_RDONLY);
    if (pcihandle < 0)
    {
        syslog(LOG_ERR, "ClientMem: Failed to open /proc/bus/pci/00/00.0\n");
        return -1;
    }

    ssize_t ret = pread(pcihandle, &imcbar, sizeof(uint64_t), PCM_CLIENT_IMC_BAR_OFFSET);
    if (ret < 0)
    {
        syslog(LOG_ERR, "ClientMem: mmap failed: %s\n", strerror(errno));
        close(pcihandle);
        return -1;
    }
    if (!imcbar)
    {
        syslog(LOG_ERR, "ClientMem: imcbar is zero.\n");
        close(pcihandle);
        return -1;
    }

    close(pcihandle);
    if (startAddr)
        *startAddr = imcbar & (~(4096 - 1));
    return 1;
}

static int
clientmem_init()
{
    uint64_t startAddr = 0;

    int ret = clientmem_getStartAddr(&startAddr);
    if (ret < 0)
    {
        syslog(LOG_ERR, "ClientMem: Failed to get startAddr\n");
        return -1;
    }

    clientmem_handle = open("/dev/mem", O_RDONLY);
    if (clientmem_handle < 0)
    {
        syslog(LOG_ERR, "ClientMem: Cannot open /dev/mem\n");
        return -1;
    }

    clientmem_addr = (char *)mmap(NULL, PCM_CLIENT_IMC_MMAP_SIZE, PROT_READ, MAP_SHARED, clientmem_handle, startAddr);
    if (clientmem_addr == MAP_FAILED)
    {
        close(clientmem_handle);
        syslog(LOG_ERR, "ClientMem: mmap failed: %s\n", strerror(errno));
        return -1;
    }
    return 0;
}

static void
clientmem_finalize()
{
    if (isClientMem)
    {
        if (clientmem_handle >= 0)
        {
            if (clientmem_addr)
            {
                munmap(clientmem_addr, PCM_CLIENT_IMC_MMAP_SIZE);
            }
            close(clientmem_handle);
        }
    }
}

static void
clientmem_read(AccessDataRecord *dRecord)
{
    uint64_t data = 0;
    uint32_t reg = dRecord->reg;

    dRecord->errorcode = ERR_NOERROR;
    dRecord->data = 0x0ULL;


    if (!lock_check())
    {
        syslog(LOG_ERR,"Access to performance counters is locked.\n");
        dRecord->errorcode = ERR_LOCKED;
        return;
    }

    if (!isClientMem)
    {
        syslog(LOG_ERR, "ClientMem: Not available for this architecture\n");
        dRecord->errorcode = ERR_RESTREG;
        return;
    }

    if (clientmem_handle < 0 || !clientmem_addr)
    {
        syslog(LOG_ERR, "ClientMem: Handle %d Addr %p\n", clientmem_handle, clientmem_addr);
        dRecord->errorcode = ERR_NODEV;
        return;
    }
    switch (reg)
    {
        case 0x00:
            data = (uint64_t)*((uint32_t *)(clientmem_addr + PCM_CLIENT_IMC_DRAM_IO_REQUESTS));
            break;
        case 0x01:
            data = (uint64_t)*((uint32_t *)(clientmem_addr + PCM_CLIENT_IMC_DRAM_DATA_READS));
            break;
        case 0x02:
            data = (uint64_t)*((uint32_t *)(clientmem_addr + PCM_CLIENT_IMC_DRAM_DATA_WRITES));
            break;
        default:
            syslog(LOG_ERR, "Access to register 0x%X not allowed\n", reg);
#ifdef DEBUG_LIKWID
            syslog(LOG_ERR, "%s", strerror(errno));
#endif
            dRecord->errorcode = ERR_RESTREG;
            return;
    }
    dRecord->data = data;
}

static void
clientmem_check(AccessDataRecord *dRecord)
{
    dRecord->errorcode = ERR_NOERROR;
    if (!isClientMem)
    {
        syslog(LOG_ERR, "ClientMem: Not available for this architecture\n");
        dRecord->errorcode = ERR_RESTREG;
        return;
    }

    if (clientmem_handle < 0 || !clientmem_addr)
    {
        syslog(LOG_ERR, "ClientMem: Handle %d Addr %p\n", clientmem_handle, clientmem_addr);
        dRecord->errorcode = ERR_NODEV;
        return;
    }
    return;
}

static void
msr_read(AccessDataRecord * dRecord)
{
    uint64_t data;
    uint32_t cpu = dRecord->cpu;
    uint32_t reg = dRecord->reg;

    dRecord->errorcode = ERR_NOERROR;
    dRecord->data = 0;

    if (!lock_check())
    {
        syslog(LOG_ERR,"Access to performance counters is locked.\n");
        dRecord->errorcode = ERR_LOCKED;
        return;
    }

    if (FD_MSR[cpu] <= 0)
    {
        dRecord->errorcode = ERR_NODEV;
        return;
    }

    if (!allowed(reg))
    {
        syslog(LOG_ERR, "Access to register 0x%X not allowed\n", reg);
        dRecord->errorcode = ERR_RESTREG;
        return;
    }

    if (pread(FD_MSR[cpu], &data, sizeof(data), reg) != sizeof(data))
    {
#ifdef DEBUG_LIKWID
        syslog(LOG_ERR, "Failed to read data from register 0x%x on core %u", reg, cpu);
        syslog(LOG_ERR, "%s", strerror(errno));
#endif
        dRecord->errorcode = ERR_RWFAIL;
        return;
    }
    dRecord->data = data;
}

static void
msr_write(AccessDataRecord * dRecord)
{
    uint32_t cpu = dRecord->cpu;
    uint32_t reg = dRecord->reg;
    uint64_t data = dRecord->data;

    dRecord->errorcode = ERR_NOERROR;

    if (!lock_check())
    {
        syslog(LOG_ERR,"Access to performance counters is locked.\n");
        dRecord->errorcode = ERR_LOCKED;
        return;
    }

    if (FD_MSR[cpu] <= 0)
    {
        dRecord->errorcode = ERR_NODEV;
        return;
    }

    if (!allowed(reg))
    {
        syslog(LOG_ERR, "Attempt to write to restricted register 0x%x on core %u", reg, cpu);
        dRecord->errorcode = ERR_RESTREG;
        return;
    }

    if (pwrite(FD_MSR[cpu], &data, sizeof(data), reg) != sizeof(data))
    {
#ifdef DEBUG_LIKWID
        syslog(LOG_ERR, "Failed to write data to register 0x%x on core %u", reg, cpu);
        syslog(LOG_ERR, "%s", strerror(errno));
#endif
        dRecord->errorcode = ERR_RWFAIL;
        return;
    }
}

static void
msr_check(AccessDataRecord * dRecord)
{
    uint32_t cpu = dRecord->cpu;
    dRecord->errorcode = ERR_NOERROR;

    if (FD_MSR[cpu] < 0)
    {
        dRecord->errorcode = ERR_NODEV;
        return;
    }
    return;
}


#define PCI_SLOT(devfn)         (((devfn) >> 3) & 0x1f)
#define PCI_FUNC(devfn)         ((devfn) & 0x07)
/* This code uses the filename and the devid specified in the PciDevice list.
 * It traverses sysfs because the sysfs contains (commonly) the node ID the PCI
 * device is attached to. If the numa_node == -1 it uses a counter to get the
 * node ID
 */
static int getBusFromSocketByNameDevid(const uint32_t socket, PciDevice* pcidev, int pathlen, char** filepath)
{
    struct dirent *pDirent, *pDirentInner;
    DIR *pDir, *pDirInner;
    FILE* fp = NULL;
    int tmplen = 200;
    char tmpPath[200], buff[200];
    size_t ret = 0;
    int bus_id = -1;
    int numa_ctr = 0;

    pDir = opendir ("/sys/devices");
    if (pDir == NULL)
    {
        syslog(LOG_ERR, "Failed open directory /sys/devices\n");
        return -1;
    }

    while ((pDirent = readdir(pDir)) != NULL)
    {
        if (strncmp(pDirent->d_name, "pci0", 4) == 0)
        {
            memset(tmpPath, '\0', tmplen*sizeof(char));
            sprintf(tmpPath, "/sys/devices/%s", pDirent->d_name);

            char bus[4];
            strncpy(bus, &(pDirent->d_name[strlen(pDirent->d_name)-2]), 2);
            bus[2] = '\0';

            pDirInner = opendir (tmpPath);
            if (pDir == NULL)
            {
                syslog(LOG_ERR, "Failed read file %s\n", tmpPath);
                return -1;
            }
            while ((pDirentInner = readdir(pDirInner)) != NULL)
            {
                if (strncmp(pDirentInner->d_name, "0000", 4) == 0)
                {
                    uint32_t dev_id = 0x0;
                    int numa_node = 0;
                    memset(tmpPath, '\0', tmplen*sizeof(char));
                    sprintf(tmpPath, "/sys/devices/%s/%s/device", pDirent->d_name, pDirentInner->d_name);
                    if (pcidev->path && strcmp(&(pDirentInner->d_name[strlen(pDirentInner->d_name)-4]), pcidev->path) != 0)
                    {
                        continue;
                    }
                    fp = fopen(tmpPath,"r");
                    if( fp != NULL )
                    {
                        memset(buff, '\0', tmplen*sizeof(char));
                        ret = fread(buff, sizeof(char), tmplen-1, fp);
                        fclose(fp);
                        if (ret > 0)
                        {
                            dev_id = strtoul(buff, NULL, 16);

                            if (dev_id == pcidev->devid)
                            {
                                memset(tmpPath, '\0', tmplen*sizeof(char));
                                sprintf(tmpPath, "/sys/devices/%s/%s/numa_node", pDirent->d_name, pDirentInner->d_name);
                                fp = fopen(tmpPath,"r");
                                if( fp != NULL )
                                {
                                    memset(buff, '\0', tmplen*sizeof(char));
                                    ret = fread(buff, sizeof(char), tmplen-1, fp);
                                    numa_node = atoi(buff);
                                    if (numa_node < 0)
                                    {
                                        numa_node = numa_ctr;
                                        numa_ctr++;
                                    }
                                    if (numa_node == socket)
                                    {
                                        bus_id = strtoul(bus, NULL, 16);
                                        if (filepath && *filepath && pathlen > 0)
                                            snprintf(*filepath, pathlen-1, "/proc/bus/pci/%02x/%s", bus_id, pcidev->path);
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            closedir (pDirInner);
            if (bus_id != -1)
                break;
        }
    }
    closedir (pDir);
#ifdef DEBUG_LIKWID
    syslog(LOG_INFO, "PCI device %s (%s) for socket %d is on PCI bus 0x%02x", pcidev->likwid_name, pcidev->path, socket, bus_id);
#endif
    return bus_id;
}



/* This code gets the PCI device using the given devid in pcidev. It assumes that the
 * PCI busses are sorted like: if sock_id1 < sock_id2 then bus1 < bus2 end
 * This code is only the fallback if a device is not found using the combination of
 * filename and devid
 */
typedef struct {
    uint32_t bus;
    uint32_t devfn;
} PciCandidate;

static int getBusFromSocketByDevid(const uint32_t socket, PciDevice* pcidev, int pathlen, char** filepath)
{
    int ret = 0;
    int cur_socket = (int)socket;
    int out_bus_id = -1;
    uint32_t out_devfn = 0x0;
    int bufflen = 1024;
    char buff[1024];
    FILE* fp = NULL;
    uint32_t bus, devfn, vendor, devid;
    PciCandidate candidates[10];
    int candidate = -1;
    int cand_idx = 0;

    fp = fopen("/proc/bus/pci/devices", "r");
    if (fp)
    {
        while (fgets(buff, bufflen, fp) != NULL)
        {
            ret = sscanf((char*)buff, "%02x%02x\t%04x%04x", &bus, &devfn, &vendor, &devid);
            if (ret == 4 && devid == pcidev->devid)
            {
                candidates[cand_idx].bus = bus;
                candidates[cand_idx].devfn = devfn;
                cand_idx++;
            }
        }
        fclose(fp);
    }
    else
    {
        syslog(LOG_ERR, "Failed read file /proc/bus/pci/devices\n");
    }

    while (cur_socket >= 0)
    {
        int min_idx = 0;
        uint32_t min = 0xFFF;
        for (ret = 0; ret < cand_idx; ret++)
        {
            if (candidates[ret].bus < min)
            {
                min = candidates[ret].bus;
                min_idx = ret;
            }
        }
        if (cur_socket > 0)
        {
            candidates[min_idx].bus = 0xFFF;
            cur_socket--;
        }
        else
        {
            if (candidates[min_idx].bus <= 0xff)
            {
                candidate = min_idx;
            }
            cur_socket = -1;
            break;
        }
    }

    if (filepath && *filepath && pathlen > 0 && candidate >= 0 && candidates[candidate].bus >= 0 && candidates[candidate].bus <= 0xff)
    {
        snprintf(*filepath, pathlen-1, "/proc/bus/pci/%02x/%02x.%01x", candidates[candidate].bus, PCI_SLOT(candidates[candidate].devfn), PCI_FUNC(candidates[candidate].devfn));
    }

    if (candidate >= 0 && candidates[candidate].bus > 0 && candidates[candidate].devfn > 0)
        return candidates[candidate].bus;
    return -1;
}


static int get_devid(int pathlen, char* path)
{
    uint32_t devid = 0x0;
    int ret = 0;
    char devidpath[1024];
    char buff[1024];

    ret = snprintf(devidpath, 1023, "%.*s/device", pathlen, path);
    if (ret)
    {
        devidpath[ret] = '\0';
        FILE *fp = fopen(devidpath, "r");
        if (fp)
        {
            ret = fread(buff, sizeof(char), 1023, fp);
            if (ret)
            {
                buff[ret] = '\0';
                ret = sscanf(buff, "%x", &devid);
                if (ret == 1)
                {
                    devid = -1;
                }
            }
            fclose(fp);
        }
    }
    return 0x0;
}

static int get_nodeid(int pathlen, char* path)
{
    int nodeid = -1;
    int ret = 0;
    char devidpath[1024];
    char buff[1024];

    ret = snprintf(devidpath, 1023, "%.*s/numa_node", pathlen, path);
    if (ret)
    {
        devidpath[ret] = '\0';
        FILE *fp = fopen(devidpath, "r");
        if (fp)
        {
            ret = fread(buff, sizeof(char), 1023, fp);
            if (ret)
            {
                buff[ret] = '\0';
                ret = sscanf(buff, "%d", &nodeid);
                if (ret != 1)
                {
                    nodeid = -1;
                }
            }
            fclose(fp);
        }
    }
    return nodeid;
}

static int getBusFromSocketByNameDevidNode(const uint32_t socket, PciDevice* pcidev, int pathlen, char** filepath)
{
    FILE *fpipe = NULL;
    int isSNC = 0;
    int SNCscale = 1;
    if (avail_sockets != avail_nodes)
    {
        isSNC = 1;
        SNCscale = avail_nodes/avail_sockets;
    }
    uint32_t devid = 0x0;
    int nodeid = -1;
    char cmd[1024];
    char buff[1024];
    int ret = 0;

    snprintf(cmd, 1023, "ls -d /sys/bus/pci/devices/*%s* 2>/dev/null", pcidev->path);
    if ( !(fpipe = popen(cmd,"r")) )
    {
        return -1;
    }
    while ((fgets(buff, 1024, fpipe)) != NULL)
    {
        if (strncmp(buff, ".", 1) == 0) continue;
        devid = get_devid((int)(strlen(buff)-1), buff);
        if (devid == 0) continue;
        nodeid = get_nodeid((int)(strlen(buff)-1), buff);
        if (nodeid < 0) continue;
        nodeid /= SNCscale;
        if (devid == pcidev->devid && nodeid == socket)
        {
            int mid = 0;
            int start = 0;
            for (int i= strlen(buff)-1; i>= 0; i--)
            {
                if (mid == 0 && buff[i] == ':')
                {
                    mid = i+1;
                    continue;
                }
                if (mid > 0 && start == 0 && buff[i] == ':')
                {
                    start = i+1;
                }
            }
            if (filepath && *filepath && pathlen > 0 && mid > 0 && start > 0)
            {
                ret = snprintf(*filepath, pathlen-1, "/proc/bus/pci/%.*s/%.*s", (int)(mid-start-1), &(buff[start]), (int)(strlen(buff)-mid-1), &(buff[mid]));
                if (ret > 0)
                    (*filepath)[ret] = '\0';
            }
            fclose(fpipe);
            return socket;
        }
    }
    fclose(fpipe);
    return -1;
}


static int getBusFromSocket(const uint32_t socket, PciDevice* pcidev, int pathlen, char** filepath)
{
    int ret_sock = -1;
    int ret = getBusFromSocketByNameDevidNode(socket, pcidev, pathlen, filepath);
    if (ret < 0)
    {
        ret = getBusFromSocketByNameDevid(socket, pcidev, pathlen, filepath);
        if (ret < 0)
        {
            ret = getBusFromSocketByDevid(socket, pcidev, pathlen, filepath);
            if (ret >= 0)
            {
                ret_sock = socket;
            }
        }
        else
        {
            ret_sock = socket;
        }
    }
    else
    {
        ret_sock = socket;
    }
    return ret_sock;
}


static void
pci_read(AccessDataRecord* dRecord)
{
    uint32_t socketId = dRecord->cpu;
    uint32_t reg = dRecord->reg;
    uint32_t device = dRecord->device;
    uint32_t data;
    uint64_t data64;
    char* pcipath = NULL;

    dRecord->errorcode = ERR_NOERROR;
    dRecord->data = 0;

    if (!lock_check())
    {
        syslog(LOG_ERR,"Access to performance counters is locked.\n");
        dRecord->errorcode = ERR_LOCKED;
        return;
    }

    if (FD_PCI[socketId][device] == -2)
    {
        dRecord->errorcode = ERR_NODEV;
        return;
    }

    if (allowedPci)
    {
        if (!allowedPci(pci_devices_daemon[device].type, reg))
        {
            dRecord->errorcode = ERR_RESTREG;
            return;
        }
    }
    if ( !FD_PCI[socketId][device] )
    {
        pcipath = malloc(1024 * sizeof(char));
        memset(pcipath, '\0', 1024 * sizeof(char));
        int sid = getBusFromSocket(socketId, &(pci_devices_daemon[device]), 1024, &pcipath);
        if (sid == socketId)
        {
#ifdef DEBUG_LIKWID
            syslog(LOG_ERR, "Open device file %s for device %s (%s) on socket %u", pcipath,
                        pci_types[pci_devices_daemon[device].type].name, pci_devices_daemon[device].name, socketId);
#endif
            FD_PCI[socketId][device] = open( pcipath, O_RDWR);

            if ( FD_PCI[socketId][device] < 0)
            {
                syslog(LOG_ERR, "Failed to open device file %s for device %s (%s) on socket %u", pcipath, pci_types[pci_devices_daemon[device].type].name, pci_devices_daemon[device].name, socketId);
                dRecord->errorcode = ERR_OPENFAIL;
                if (pcipath)
                    free(pcipath);
                return;
            }
        }
    }

    if (!isPCI64)
    {
        if (FD_PCI[socketId][device] > 0 && pread(FD_PCI[socketId][device], &data, sizeof(data), reg) != sizeof(data))
        {
            syslog(LOG_ERR, "Failed to read data from pci device file %s for device %s (%s) on socket %u",
                    pcipath,pci_types[pci_devices_daemon[device].type].name, pci_devices_daemon[device].name, socketId);
            dRecord->errorcode = ERR_RWFAIL;
            if (pcipath)
                free(pcipath);
            return;
        }

        dRecord->data = (uint64_t) data;
    }
    else
    {
        if (FD_PCI[socketId][device] > 0 && pread(FD_PCI[socketId][device], &data64, sizeof(data64), reg) != sizeof(data64))
        {
            syslog(LOG_ERR, "Failed to read data from pci device file %s for device %s (%s) on socket %u",
                    pcipath,pci_types[pci_devices_daemon[device].type].name, pci_devices_daemon[device].name, socketId);
            dRecord->errorcode = ERR_RWFAIL;
            if (pcipath)
                free(pcipath);
            return;
        }

        dRecord->data = data64;
    }
    if (pcipath)
        free(pcipath);
}

static void
pci_write(AccessDataRecord* dRecord)
{
    uint32_t socketId = dRecord->cpu;
    uint32_t reg = dRecord->reg;
    uint32_t device = dRecord->device;
    uint32_t data = (uint32_t)dRecord->data;
    uint64_t data64 = dRecord->data;
    char* pcipath = NULL;

    dRecord->errorcode = ERR_NOERROR;

    if (!lock_check())
    {
        syslog(LOG_ERR,"Access to performance counters is locked.\n");
        dRecord->errorcode = ERR_LOCKED;
        return;
    }

    if (FD_PCI[socketId][device] == -2)
    {
        dRecord->errorcode = ERR_NODEV;
        return;
    }

    if (allowedPci)
    {
        if (!allowedPci(pci_devices_daemon[device].type, reg))
        {
        dRecord->errorcode = ERR_RESTREG;
        return;
        }
    }

    if ( !FD_PCI[socketId][device] )
    {
        pcipath = malloc(1024 * sizeof(char));
        memset(pcipath, '\0', 1024 * sizeof(char));
        int sid = getBusFromSocket(socketId, &(pci_devices_daemon[device]), 1024, &pcipath);
        if (sid == socketId)
        {
#ifdef DEBUG_LIKWID
            syslog(LOG_ERR, "Open device file %s for device %s (%s) on socket %u", pcipath,
                        pci_types[pci_devices_daemon[device].type].name, pci_devices_daemon[device].name, socketId);
#endif
            FD_PCI[socketId][device] = open( pcipath, O_RDWR);

            if ( FD_PCI[socketId][device] < 0)
            {
                syslog(LOG_ERR, "Failed to open device file %s for device %s (%s) on socket %u", pcipath, pci_types[pci_devices_daemon[device].type].name, pci_devices_daemon[device].name, socketId);
                dRecord->errorcode = ERR_OPENFAIL;
                if (pcipath)
                    free(pcipath);
                return;
            }
        }

    }

    if (!isPCI64)
    {
        if (FD_PCI[socketId][device] > 0 && pwrite(FD_PCI[socketId][device], &data, sizeof data, reg) != sizeof data)
        {
            syslog(LOG_ERR, "Failed to write data to pci device file %s for device %s (%s) on socket %u",pci_filepath,
                    pci_types[pci_devices_daemon[device].type].name, pci_devices_daemon[device].name, socketId);
            dRecord->errorcode = ERR_RWFAIL;
            if (pcipath)
                free(pcipath);
            return;
        }
    }
    else
    {
        if (FD_PCI[socketId][device] > 0 && pwrite(FD_PCI[socketId][device], &data64, sizeof data64, reg) != sizeof data64)
        {
            syslog(LOG_ERR, "Failed to write data to pci device file %s for device %s (%s) on socket %u",pci_filepath,
                    pci_types[pci_devices_daemon[device].type].name, pci_devices_daemon[device].name, socketId);
            dRecord->errorcode = ERR_RWFAIL;
            if (pcipath)
                free(pcipath);
            return;
        }
    }
    if (pcipath)
        free(pcipath);
}

static void
pci_check(AccessDataRecord* dRecord)
{
    uint32_t socketId = dRecord->cpu;
    uint32_t device = dRecord->device;
    dRecord->errorcode = ERR_NOERROR;

    if (FD_PCI[socketId][device] == -2)
    {
        dRecord->errorcode = ERR_NODEV;
        return;
    }
    return;
}

static void
kill_client(void)
{
    if (connfd != -1)
    {
        CHECK_ERROR(close(connfd), socket close failed);
    }

    connfd = -1;
}

static void
stop_daemon(void)
{
    kill_client();

    if (sockfd != -1)
    {
        CHECK_ERROR(close(sockfd), socket close sockfd failed);
    }

    free(filepath);
    closelog();
    exit(EXIT_SUCCESS);
}



static void
Signal_Handler(int sig)
{
    if (sig == SIGPIPE)
    {
        syslog(LOG_NOTICE, "SIGPIPE? client crashed?!");
        stop_daemon();
    }

    /* For SIGALRM we just return - we're just here to create a EINTR */
    if (sig == SIGTERM)
    {
        stop_daemon();
    }
}

static void
daemonize(int* parentPid)
{
    pid_t pid, sid;

    *parentPid = getpid();

    /* already a daemon */
    if ( getppid() == 1 ) return;

    /* Fork off the parent process */
    pid = fork();

    if (pid < 0)
    {
        syslog(LOG_ERR, "fork failed: %s", strerror(errno));
        exit(EXIT_FAILURE);
    }

    /* If we got a good PID, then we can exit the parent process. */
    if (pid > 0)
    {
        exit(EXIT_SUCCESS);
    }

    /* At this point we are executing as the child process */

    /* Create a new SID for the child process */
    sid = setsid();

    if (sid < 0)
    {
        syslog(LOG_ERR, "setsid failed: %s", strerror(errno));
        exit(EXIT_FAILURE);
    }

    /* Change the current working directory.  This prevents the current
       directory from being locked; hence not being able to remove it. */
    if ((chdir("/")) < 0)
    {
        syslog(LOG_ERR, "chdir failed:  %s", strerror(errno));
        exit(EXIT_FAILURE);
    }

    /* Redirect standard files to /dev/null */
    {
        CHECK_FILE_ERROR(freopen( "/dev/null", "r", stdin), freopen stdin failed);
        CHECK_FILE_ERROR(freopen( "/dev/null", "w", stdout), freopen stdout failed);
        CHECK_FILE_ERROR(freopen( "/dev/null", "w", stderr), freopen stderr failed);
    }
}

/* #####  MAIN FUNCTION DEFINITION   ################## */

int main(void)
{
    int ret;
    pid_t pid;
    struct sockaddr_un  addr1;
    socklen_t socklen;
    AccessDataRecord dRecord;
    mode_t oldumask;
    uint32_t numHWThreads = sysconf(_SC_NPROCESSORS_CONF);
    uint32_t model;
    for (int i=0;i<avail_cpus;i++)
    {
        FD_MSR[i] = -1;
    }

    openlog(ident, 0, LOG_USER);

    if (!lock_check())
    {
        syslog(LOG_ERR,"Access to performance counters is locked.\n");
        stop_daemon();
    }
    if (!FD_MSR || !FD_PCI)
    {
        syslog(LOG_ERR,"Failed to allocate FD space.\n");
        stop_daemon();
    }

    daemonize(&pid);
#ifdef DEBUG_LIKWID
    syslog(LOG_INFO, "AccessDaemon runs with UID %d, eUID %d\n", getuid(), geteuid());
#endif

    {
        uint32_t  eax = 0x00;
        uint32_t  ebx = 0x00;
        uint32_t  ecx = 0x00;
        uint32_t  edx = 0x00;
        /*int isIntel = 1;
        CPUID(eax, ebx, ecx, edx);
        if (ebx == 0x68747541U)
        {
            isIntel = 0;
        }*/

        eax = 0x01;
        CPUID(eax, ebx, ecx, edx);
        uint32_t family = ((eax >> 8) & 0xFU) + ((eax >> 20) & 0xFFU);
        model  = (((eax >> 16) & 0xFU) << 4) + ((eax >> 4) & 0xFU);
        eax = 0x0A;
        CPUID(eax, ebx, ecx, edx);
        num_pmc_counters = (int)((eax>>8)&0xFFU);

        switch (family)
        {
            case P6_FAMILY:
                allowed = allowed_intel;

                if ((model == SANDYBRIDGE) || (model == IVYBRIDGE))
                {
                    isClientMem = 1;
                    allowed = allowed_sandybridge;
                }
                else if ((model == SANDYBRIDGE_EP) || (model == IVYBRIDGE_EP))
                {
                    allowed = allowed_sandybridge;
                    allowedPci = allowed_pci_sandybridge;
                    isPCIUncore = 1;
                }
                else if ((model == HASWELL) ||
                         (model == HASWELL_M1) ||
                         (model == HASWELL_M2) ||
                         (model == BROADWELL) ||
                         (model == BROADWELL_E3) ||
                         (model == SKYLAKE1) ||
                         (model == SKYLAKE2) ||
                         (model == KABYLAKE1) ||
                         (model == KABYLAKE2))
                {
                    allowed = allowed_sandybridge;
                    isClientMem = 1;
                }
                else if (model == BROADWELL_D)
                {
                    allowed = allowed_sandybridge;
                    isPCIUncore = 1;
                    allowedPci = allowed_pci_haswell;
                }
                else if (model == HASWELL_EP)
                {
                    isPCIUncore = 1;
                    allowed = allowed_sandybridge;
                    allowedPci = allowed_pci_haswell;
                }
                else if (model == BROADWELL_E)
                {
                    isPCIUncore = 1;
                    allowed = allowed_sandybridge;
                    allowedPci = allowed_pci_haswell;
                }
                else if (model == SKYLAKEX)
                {
                    isPCIUncore = 1;
                    allowed = allowed_skx;
                    allowedPci = allowed_pci_skx;
                    isPCI64 = 1;
                }
                else if ((model == ATOM_SILVERMONT_C) ||
                         (model == ATOM_SILVERMONT_E) ||
                         (model == ATOM_SILVERMONT_Z1) ||
                         (model == ATOM_SILVERMONT_Z2) ||
                         (model == ATOM_SILVERMONT_F) ||
                         (model == ATOM_SILVERMONT_AIR))
                {
                    allowed = allowed_silvermont;
                }
                else if ((model == XEON_PHI_KNL) ||
                         (model == XEON_PHI_KML))
                {
                    allowed = allowed_knl;
                    isPCIUncore = 1;
                    allowedPci = allowed_pci_knl;
                }
                break;
            case K8_FAMILY:
            case K10_FAMILY:
                allowed = allowed_amd;
                break;
            case K15_FAMILY:
                allowed = allowed_amd15;
                break;
            case K16_FAMILY:
                allowed = allowed_amd16;
                break;
            case ZEN_FAMILY:
                allowed = allowed_amd17;
                break;
            default:
                syslog(LOG_ERR, "ERROR - [%s:%d] - Unsupported processor. Exiting!  \n",
                        __FILE__, __LINE__);
                exit(EXIT_FAILURE);
        }
    }

    /* setup filename for socket */
    filepath = (char*) calloc(sizeof(addr1.sun_path), 1);
    snprintf(filepath, sizeof(addr1.sun_path), TOSTRING(LIKWIDSOCKETBASE) "-%d", pid);

    /* get a socket */
    LOG_AND_EXIT_IF_ERROR(sockfd = socket(AF_LOCAL, SOCK_STREAM, 0), socket failed);

    /* initialize socket data structure */
    bzero(&addr1, sizeof(addr1));
    addr1.sun_family = AF_LOCAL;
    strncpy(addr1.sun_path, filepath, (sizeof(addr1.sun_path) - 1)); /* null terminated by the bzero() above! */

    /* Change the file mode mask so only the calling user has access
     * and switch the user/gid with which the following socket creation runs. */
    oldumask = umask(077);
    CHECK_ERROR(setfsuid(getuid()), setfsuid failed);

    /* bind and listen on socket */
    LOG_AND_EXIT_IF_ERROR(bind(sockfd, (SA*) &addr1, sizeof(addr1)), bind failed);
    LOG_AND_EXIT_IF_ERROR(listen(sockfd, 1), listen failed);
    LOG_AND_EXIT_IF_ERROR(chmod(filepath, S_IRUSR|S_IWUSR), chmod failed);

    socklen = sizeof(addr1);

    { /* Init signal handler */
        struct sigaction sia;
        sia.sa_handler = Signal_Handler;
        sigemptyset(&sia.sa_mask);
        sia.sa_flags = 0;
        sigaction(SIGALRM, &sia, NULL);
        sigaction(SIGPIPE, &sia, NULL);
        sigaction(SIGTERM, &sia, NULL);
    }

    /* setup an alarm to stop the daemon if there is no connect.*/
    alarm(15U);

    if ((connfd = accept(sockfd, (SA*) &addr1, &socklen)) < 0)
    {
        if (errno == EINTR)
        {
            syslog(LOG_ERR, "exiting due to timeout - no client connected after 15 seconds.");
        }
        else
        {
            syslog(LOG_ERR, "accept() failed:  %s", strerror(errno));
        }
        CHECK_ERROR(unlink(filepath), unlink of socket failed);
        exit(EXIT_FAILURE);
    }

    alarm(0);
    CHECK_ERROR(unlink(filepath), unlink of socket failed);

    /* Restore the old umask and fs ids. */
    (void) umask(oldumask);
    CHECK_ERROR(setfsuid(geteuid()), setfsuid failed);

    {
        char* msr_file_name = (char*) malloc(MAX_PATH_LENGTH * sizeof(char));

        /* Open MSR device files for less overhead.
         * NOTICE: This assumes consecutive processor Ids! */
        for ( uint32_t i=0; i < avail_cpus; i++ )
        {
            sprintf(msr_file_name,"/dev/cpu/%d/msr",i);
            FD_MSR[i] = open(msr_file_name, O_RDWR);

            if ( FD_MSR[i] < 0 )
            {
                syslog(LOG_ERR, "Failed to open device file %s: %s, trying /dev/msr%d", msr_file_name, strerror(errno), i);
                sprintf(msr_file_name,"/dev/msr%d",i);
                FD_MSR[i] = open(msr_file_name, O_RDWR);
                if ( FD_MSR[i] < 0 )
                {
                    syslog(LOG_ERR, "Failed to open device file %s: %s.", msr_file_name, strerror(errno));
                }
            }
        }

        free(msr_file_name);
        if (isClientMem)
        {
            ret = clientmem_init();
            if (ret)
            {
                syslog(LOG_ERR, "Failed to initialize Intel desktop memory support");
            }
        }
        if (isPCIUncore)
        {
            int cntr = 0;
            int socket_count = 0;
            if (model == SANDYBRIDGE_EP)
            {
                //testDevice = 0x80863c44;
                pci_devices_daemon = sandybridgeEP_pci_devices;
            }
            else if (model == IVYBRIDGE_EP)
            {
                //testDevice = 0x80860e36;
                pci_devices_daemon = ivybridgeEP_pci_devices;
            }
            else if (model == HASWELL_EP)
            {
                //testDevice = 0x80862f30;
                pci_devices_daemon = haswellEP_pci_devices;
            }
            else if (model == BROADWELL_D)
            {
                //testDevice = 0x80862f30;
                pci_devices_daemon = broadwelld_pci_devices;
            }
            else if (model == BROADWELL_E)
            {
                //testDevice = 0x80862f30;
                pci_devices_daemon = broadwellEP_pci_devices;
            }
            else if (model == SKYLAKEX)
            {
                //testDevice = 0x80862f30;
                pci_devices_daemon = skylakeX_pci_devices;
            }
            else if ((model == XEON_PHI_KNL) ||
                     (model == XEON_PHI_KML))
            {
                pci_devices_daemon = knl_pci_devices;
            }
            else
            {
                //testDevice = 0;
                syslog(LOG_NOTICE, "PCI Uncore not supported on this system");
                goto LOOP;
            }

            for (int j=0; j<avail_sockets; j++)
            {
                for (int i=0; i<MAX_NUM_PCI_DEVICES; i++)
                {
                    FD_PCI[j][i] = -2;
                }
            }

            for (int i=1; i<MAX_NUM_PCI_DEVICES; i++)
            {
                if (pci_devices_daemon[i].path && strlen(pci_devices_daemon[i].path) > 0)
                {
                    int socket_id = getBusFromSocket(0, &(pci_devices_daemon[i]), 0, NULL);
                    if (socket_id == 0)
                    {
                        for (int j=0; j<avail_sockets; j++)
                        {
                            FD_PCI[j][i] = 0;
                        }
                        pci_devices_daemon[i].online = 1;
                    }
#ifdef DEBUG_LIKWID
                    else
                    {
                        syslog(LOG_ERR, "Device %s not found, excluded it from device list\n",pci_devices_daemon[i].name);
                    }
#endif
                }
            }
        }
    }
LOOP:
    while (1)
    {
        ret = read(connfd, (void*) &dRecord, sizeof(AccessDataRecord));

        if (ret < 0)
        {
            stop_daemon();
        }
        else if ((ret == 0) && (dRecord.type != DAEMON_EXIT))
        {
            syslog(LOG_ERR, "ERROR - [%s:%d] zero read, remote socket closed before reading", __FILE__, __LINE__);
            stop_daemon();
        }
        else if (ret != sizeof(AccessDataRecord))
        {
            syslog(LOG_ERR, "ERROR - [%s:%d] unaligned read", __FILE__, __LINE__);
            stop_daemon();
        }


        if (dRecord.type == DAEMON_READ)
        {
            if (dRecord.device == MSR_DEV)
            {
                msr_read(&dRecord);
            }
            else if (isClientMem)
            {
                clientmem_read(&dRecord);
            }
            else
            {
                pci_read(&dRecord);
            }
        }
        else if (dRecord.type == DAEMON_WRITE)
        {
            if (dRecord.device == MSR_DEV)
            {
                msr_write(&dRecord);
                dRecord.data = 0x0ULL;
            }
            else
            {
                pci_write(&dRecord);
                dRecord.data = 0x0ULL;
            }
        }
        else if (dRecord.type == DAEMON_CHECK)
        {
            if (dRecord.device == MSR_DEV)
            {
                msr_check(&dRecord);
            }
            else if (isClientMem)
            {
                clientmem_check(&dRecord);
            }
            else
            {
                pci_check(&dRecord);
            }
        }
        else if (dRecord.type == DAEMON_EXIT)
        {
            stop_daemon();
        }
        else
        {
            syslog(LOG_ERR, "unknown daemon access type  %d", dRecord.type);
            dRecord.errorcode = ERR_UNKNOWN;
        }

        LOG_AND_EXIT_IF_ERROR(write(connfd, (void*) &dRecord, sizeof(AccessDataRecord)), write failed);
    }

    /* never reached */
    return EXIT_SUCCESS;
}

