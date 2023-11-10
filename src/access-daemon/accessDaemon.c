/*
 * =======================================================================================
 *
 *      Filename:  accessDaemon.c
 *
 *      Description:  Implementation of access daemon.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
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
#include <perfmon_icelakeX_counters.h>
#include <intel_perfmon_uncore_discovery.h>
#include <perfmon_sapphirerapids_counters.h>
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
#define PCM_CLIENT_IMC_PP0_TEMP (0x597C)
#define PCM_CLIENT_IMC_PP1_TEMP (0x5980)
#define PCM_CLIENT_IMC_MMAP_SIZE (0x6000)


/* MMIO_BASE found at Bus U0, Device 0, Function 1, offset D0h. */
#define ICX_IMC_MMIO_BASE_OFFSET 0xD0
#define ICX_IMC_MMIO_BASE_MASK 0x1FFFFFFF
#define ICX_IMC_MMIO_BASE_SHIFT 23
/* MEM0_BAR found at Bus U0, Device 0, Function 1, offset D8h. */
#define ICX_IMC_MMIO_MEM0_OFFSET 0xD8
#define ICX_IMC_MMIO_MEM_STRIDE 0x4
#define ICX_IMC_MMIO_MEM_MASK 0x7FF
#define ICX_IMC_MMIO_MEM_SHIFT 12
/* MEM1_BAR found at Bus U0, Device 0, Function 1, offset DCh. */
#define ICX_IMC_MMIO_MEM1_OFFSET 0xDC
/* MEM2_BAR found at Bus U0, Device 0, Function 1, offset AE0h. */
#define ICX_IMC_MMIO_MEM2_OFFSET 0xE0
/* MEM3_BAR found at Bus U0, Device 0, Function 1, offset E4h. */
#define ICX_IMC_MMIO_MEM3_OFFSET 0xE4
/*
* Each IMC has two channels.
* The offset starts from 0x22800 with stride 0x4000
*/
#define ICX_IMC_MMIO_CHN_OFFSET 0x22800
#define ICX_IMC_MMIO_CHN_STRIDE 0x4000
/* IMC MMIO size*/
#define ICX_IMC_MMIO_SIZE 0x4000

#define ICX_IMC_MMIO_FREERUN_OFFSET 0x2290
#define ICX_IMC_MMIO_FREERUN_SIZE 0x4000

/*
 * I'm following the Linux kernel here but documentation tells us that
 * there are three channels out of which 2 are active.
*/
#define ICX_NUMBER_IMC_CHN          2
#define ICX_IMC_MEM_STRIDE          0x4
#define ICX_NUMBER_IMC_DEVS         4

typedef struct {
    int device_id;
    int socket;
    void* channel0;
    void* channel1;
    void* channel2;
    void* freerun;
} IntelMemoryDevice;

/* Lock file controlled from outside which prevents likwid to start.
 * Can be used to synchronize access to the hardware counters
 * with an external monitoring system. */

/* #####   TYPE DEFINITIONS   ########### */

typedef int (*AllowedPrototype)(uint32_t);
typedef int (*AllowedPciPrototype)(PciDeviceType, uint32_t);
static int getBusFromSocket(const uint32_t socket, PciDevice* pcidev, int pathlen, char** filepath);

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
static int* socket_map = NULL;
static int isPCIUncore = 0;
static int isPCI64 = 0;
static int isIntelUncoreDiscovery = 0;
static PciDevice* pci_devices_daemon = NULL;
static char pci_filepath[MAX_PATH_LENGTH];
static int num_pmc_counters = 0;

static int clientmem_handle = -1;
static char *clientmem_addr = NULL;
static int isClientMem = 0;

static int isServerMem = 0;
static void *** servermem_addrs = NULL;
static void *** servermem_freerun_addrs = NULL;

int perfmon_verbosity = 0;
static PerfmonDiscovery* perfmon_discovery = NULL;

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
 *
 * With Intel Icelake, the mapping changed for 2S to 0x7e and 0xfe.
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
    pclose(fpipe);
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
    pclose(fpipe);
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
        while ((ep = readdir (dp)) != NULL)
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
            clientmem_handle = -1;
        }
    }
}

static void
servermem_finalize()
{
    if (isServerMem)
    {
        int i = 0;
        int j = 0;
        if (servermem_addrs)
        {
            for (i = 0; i < avail_sockets; i++)
            {
                for (j = 0; j < ICX_NUMBER_IMC_DEVS*ICX_NUMBER_IMC_CHN; j++)
                {
                    if (servermem_addrs[i][j])
                    {
                        munmap(servermem_addrs[i][j], ICX_IMC_MMIO_SIZE);
                    }
                }
                free(servermem_addrs[i]);
                servermem_addrs[i] = NULL;
            }
            free(servermem_addrs);
            servermem_addrs = NULL;
        }
        if (servermem_freerun_addrs)
        {
            for (i = 0; i < avail_sockets; i++)
            {
                for (j = 0; j < ICX_NUMBER_IMC_DEVS; j++)
                {
                    if (servermem_freerun_addrs[i][j])
                    {
                        munmap(servermem_freerun_addrs[i][j], ICX_IMC_MMIO_FREERUN_SIZE);
                    }
                }
                free(servermem_freerun_addrs[i]);
                servermem_freerun_addrs[i] = NULL;
            }
            free(servermem_freerun_addrs);
            servermem_freerun_addrs = NULL;
        }
    }
}

void __attribute__((constructor (101))) init_accessdaemon(void)
{

    FILE *rdpmc_file = NULL;
    FILE *nmi_watchdog_file = NULL;
    int retries = 10;
    char fname[1024];
    char buf[256];

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
    {
        return;
    }

    socket_map = malloc(avail_sockets * sizeof(int));
    if (!socket_map)
    {
        free(FD_MSR);
        FD_MSR = NULL;
        return;
    }
    memset(socket_map, -1, avail_sockets * sizeof(int));

    FD_PCI = malloc(avail_sockets * sizeof(int*));
    if (!FD_PCI)
    {
        free(socket_map);
        socket_map = NULL;
        free(FD_MSR);
        FD_MSR = NULL;
        return;
    }

    for (int i = 0; i < avail_sockets; i++)
    {
        for (int j = 0; j < avail_cpus; j++)
        {
            int ret = snprintf(fname, 1023, "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", j);
            if (ret < 0)
            {
                continue;
            }
            fname[ret] = '\0';
            if (!access(fname, R_OK))
            {
                FILE* fp = fopen(fname, "r");
                if (fp)
                {
                    ret = fread(buf, sizeof(char), 255, fp);
                    if (ret >= 0)
                    {
                        buf[ret] = '\0';
                        int tmp = atoi(buf);
                        if (tmp >= 0 && tmp == i && socket_map[tmp] < 0)
                        {
                            socket_map[tmp] = j;
                            break;
                        }
                    }
                    fclose(fp);
                    buf[0] = '\0';
                }
            }
            fname[0] = '\0';
        }
        FD_PCI[i] = malloc(MAX_NUM_PCI_DEVICES * sizeof(int));
        if (!FD_PCI[i])
        {
            for (int j = i-1; j >= 0; j--)
            {
                free(FD_PCI[j]);
            }
            free(FD_PCI);
            FD_PCI = NULL;
            free(FD_MSR);
            FD_MSR = NULL;
            free(socket_map);
            socket_map = NULL;
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
    if (socket_map)
    {
        free(socket_map);
        socket_map = NULL;
    }
    if (FD_PCI)
    {
        for (int i = 0; i < avail_sockets; i++)
        {
            if (FD_PCI[i])
            {
                for (int j = 0; j < MAX_NUM_PCI_DEVICES; j++)
                {
                    if (FD_PCI[i][j] > 0)
                    {
                        close(FD_PCI[i][j]);
                        FD_PCI[i][j] = 0;
                    }
                }
                free(FD_PCI[i]);
                FD_PCI[i] = NULL;
            }
        }
        free(FD_PCI);
        FD_PCI = NULL;
    }
    if (FD_MSR)
    {
        for (int i=0; i < avail_cpus; i++)
        {
            if (FD_MSR[i] >= 0)
            {
                close(FD_MSR[i]);
                FD_MSR[i] = -1;
            }
        }
        free(FD_MSR);
        FD_MSR = NULL;
    }
    if (isClientMem && clientmem_handle >= 0 && clientmem_addr)
    {
        clientmem_finalize();
    }
    if (isServerMem)
    {
        servermem_finalize();
    }
    if (perfmon_discovery)
    {
        perfmon_uncore_discovery_free(perfmon_discovery);
        perfmon_discovery = NULL;

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
            (reg == 0x48)  ||
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
        (reg == MSR_MPERF)   ||
        (reg == MSR_APERF)   ||
        (reg == MSR_PERF_STATUS)  ||
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
        default:
            break;
    }
    return 0;
}


static int allowed_pci_icx(PciDeviceType type, uint32_t reg)
{
    switch(type)
    {
        case IMC:
            if ((reg == MMIO_ICX_IMC_BOX_CTRL) ||
                (reg == MMIO_ICX_IMC_BOX_STATUS) ||
                (reg == MMIO_ICX_IMC_BOX_CTL0) ||
                (reg == MMIO_ICX_IMC_BOX_CTL1) ||
                (reg == MMIO_ICX_IMC_BOX_CTL2) ||
                (reg == MMIO_ICX_IMC_BOX_CTL3) ||
                (reg == MMIO_ICX_IMC_BOX_CTR0) ||
                (reg == MMIO_ICX_IMC_BOX_CTR1) ||
                (reg == MMIO_ICX_IMC_BOX_CTR2) ||
                (reg == MMIO_ICX_IMC_BOX_CTR3) ||
                (reg == MMIO_ICX_IMC_BOX_CLK_CTL) ||
                (reg == MMIO_ICX_IMC_BOX_CLK_CTR)||
                (reg == MMIO_ICX_IMC_FREERUN_DDR_RD) ||
                (reg == MMIO_ICX_IMC_FREERUN_DDR_WR) ||
                (reg == MMIO_ICX_IMC_FREERUN_PMM_RD) ||
                (reg == MMIO_ICX_IMC_FREERUN_PMM_RD) ||
                (reg == MMIO_ICX_IMC_FREERUN_DCLK))
                return 1;
        case HA:
            if ((reg == PCI_UNC_ICX_M2M_PMON_CTRL) ||
                (reg == PCI_UNC_ICX_M2M_PMON_STATUS) ||
                (reg == PCI_UNC_ICX_M2M_PMON_CTL0) ||
                (reg == PCI_UNC_ICX_M2M_PMON_CTL1) ||
                (reg == PCI_UNC_ICX_M2M_PMON_CTL2) ||
                (reg == PCI_UNC_ICX_M2M_PMON_CTL3) ||
                (reg == PCI_UNC_ICX_M2M_PMON_CTR0) ||
                (reg == PCI_UNC_ICX_M2M_PMON_CTR1) ||
                (reg == PCI_UNC_ICX_M2M_PMON_CTR2) ||
                (reg == PCI_UNC_ICX_M2M_PMON_CTR3))
                return 1;
            break;
        case R3QPI:
            if ((reg == PCI_UNC_ICX_M3UPI_PMON_CTRL) ||
                (reg == PCI_UNC_ICX_M3UPI_PMON_STATUS) ||
                (reg == PCI_UNC_ICX_M3UPI_PMON_CTL0) ||
                (reg == PCI_UNC_ICX_M3UPI_PMON_CTL1) ||
                (reg == PCI_UNC_ICX_M3UPI_PMON_CTL2) ||
                (reg == PCI_UNC_ICX_M3UPI_PMON_CTL3) ||
                (reg == PCI_UNC_ICX_M3UPI_PMON_CTR0) ||
                (reg == PCI_UNC_ICX_M3UPI_PMON_CTR1) ||
                (reg == PCI_UNC_ICX_M3UPI_PMON_CTR2) ||
                (reg == PCI_UNC_ICX_M3UPI_PMON_CTR3))
                return 1;
            break;
        case QPI:
            if ((reg == PCI_UNC_ICX_UPI_PMON_CTRL) ||
                (reg == PCI_UNC_ICX_UPI_PMON_STATUS) ||
                (reg == PCI_UNC_ICX_UPI_PMON_CTL0) ||
                (reg == PCI_UNC_ICX_UPI_PMON_CTL1) ||
                (reg == PCI_UNC_ICX_UPI_PMON_CTL2) ||
                (reg == PCI_UNC_ICX_UPI_PMON_CTL3) ||
                (reg == PCI_UNC_ICX_UPI_PMON_CTR0) ||
                (reg == PCI_UNC_ICX_UPI_PMON_CTR1) ||
                (reg == PCI_UNC_ICX_UPI_PMON_CTR2) ||
                (reg == PCI_UNC_ICX_UPI_PMON_CTR3))
                return 1;
            break;
        default:
            break;
    }
    return 0;
}

static int allowed_icx(uint32_t reg)
{
    if (allowed_sandybridge(reg))
        return 1;
    else
    {
        if (((reg & 0xF00U) == 0x700U) ||
            ((reg & 0xF00U) == 0xA00U) ||
            ((reg & 0xF00U) == 0xB00U) ||
            ((reg & 0xF00U) == 0xE00U) ||
            ((reg & 0xF00U) == 0xF00U) ||
            (reg == MSR_PREFETCH_ENABLE) ||
            (reg == TSX_FORCE_ABORT) ||
            (reg == MSR_PERF_METRICS))
            return 1;
    }
    return 0;
}

static int allowed_icl(uint32_t reg)
{
    if (allowed_sandybridge(reg))
        return 1;
    else
    {
        if ((reg == MSR_PREFETCH_ENABLE) ||
            (reg == MSR_PERF_METRICS))
            return 1;
    }
    return 0;
}

static int allowed_spr(uint32_t reg)
{
    return allowed_icx(reg);
}

static int allowed_pci_spr(PciDeviceType type, uint32_t reg)
{
    return 1;
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
allowed_amd17_zen2(uint32_t reg)
{
    if (allowed_amd17(reg))
    {
        return 1;
    }
    else if ((reg == 0xC0010208) ||
             (reg == 0xC0010209) ||
             (reg == 0xC001020A) ||
             (reg == 0xC001020B))
    {
        return 1;
    }
    return 0;
}

static int
allowed_amd19_zen4(uint32_t reg)
{
    if (allowed_amd17_zen2(reg))
    {
        return 1;
    }
    else if ((reg == 0xC0000108) ||
             (reg == 0x00000048) ||
	     (reg == 0x0000010B))
    {
        return 1;
    }
    return 0;
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
        case 0x03:
            data = (uint64_t)*((uint32_t *)(clientmem_addr + PCM_CLIENT_IMC_PP0_TEMP));
            break;
        case 0x04:
            data = (uint64_t)*((uint32_t *)(clientmem_addr + PCM_CLIENT_IMC_PP1_TEMP));
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


static int servermem_getStartAddr(int socketId, int pmc_idx, void **mmap_addr)
{
    off_t addr;
    off_t addr2;
    uint32_t tmp = 0x0U;
    int pagesize = sysconf(_SC_PAGE_SIZE);
    char sysfile[1000];
    uint32_t pci_bus[2] = {0x7e, 0xfe};
    // pcipath = malloc(1024 * sizeof(char));
    // memset(pcipath, '\0', 1024 * sizeof(char));
    //int sid = getBusFromSocket(socketId, &(pci_devices_daemon[UBOX]), 999, &sysfile);
    //syslog(LOG_ERR, "Sysfs file %s", sysfile);

    int ret = snprintf(sysfile, 999, "/sys/bus/pci/devices/0000:%.2x:00.1/config", pci_bus[socketId]);
    if (ret >= 0)
    {
        sysfile[ret] = '\0';
    }
    else
    {
        return -1;
    }
    int pcihandle = open(sysfile, O_RDONLY);
    if (pcihandle < 0)
    {
        return -1;
    }

    ret = pread(pcihandle, &tmp, sizeof(uint32_t), ICX_IMC_MMIO_BASE_OFFSET);
    if (ret < 0)
    {
        close(pcihandle);
        return -1;
    }
    if (!tmp)
    {
        close(pcihandle);
        return -1;
    }
    addr = ((tmp & ICX_IMC_MMIO_BASE_MASK)) << ICX_IMC_MMIO_BASE_SHIFT;
    int mem_offset = ICX_IMC_MMIO_MEM0_OFFSET + (pmc_idx / ICX_NUMBER_IMC_CHN) * ICX_IMC_MEM_STRIDE;
    ret = pread(pcihandle, &tmp, sizeof(uint32_t), mem_offset);
    if (ret < 0)
    {
        close(pcihandle);
        return -1;
    }
    addr2 = ((tmp & ICX_IMC_MMIO_MEM_MASK)) << ICX_IMC_MMIO_MEM_SHIFT;
    addr |= addr2;
    addr += ICX_IMC_MMIO_CHN_OFFSET + ICX_IMC_MMIO_CHN_STRIDE * (pmc_idx % ICX_NUMBER_IMC_CHN);

    close(pcihandle);

    pcihandle = open("/dev/mem", O_RDWR | O_SYNC);
    uint64_t page_mask = ~(pagesize - 1);
    void* maddr = mmap(0, ICX_IMC_MMIO_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, pcihandle, addr & page_mask);
    if (maddr == MAP_FAILED)
    {
        syslog(LOG_ERR, "ServerMem: MMAP of addr 0x%lx failed\n", addr & page_mask);
        close(pcihandle);
        return -1;
    }
    close(pcihandle);
    //*reg_offset = addr - (addr & page_mask);
    //*raw_offset = addr;
    *mmap_addr = maddr + (addr - (addr & page_mask));
    
    return 0;
}

static int servermem_freerun_getStartAddr(int socketId, int pmc_idx, void **mmap_addr)
{
    off_t addr;
    off_t addr2;
    uint32_t tmp = 0x0U;
    int pagesize = sysconf(_SC_PAGE_SIZE);
    char sysfile[1000];
    uint32_t pci_bus[2] = {0x7e, 0xfe};
    // pcipath = malloc(1024 * sizeof(char));
    // memset(pcipath, '\0', 1024 * sizeof(char));
    //int sid = getBusFromSocket(socketId, &(pci_devices_daemon[UBOX]), 999, &sysfile);
    //syslog(LOG_ERR, "Sysfs file %s", sysfile);

    int ret = snprintf(sysfile, 999, "/sys/bus/pci/devices/0000:%.2x:00.1/config", pci_bus[socketId]);
    if (ret >= 0)
    {
        sysfile[ret] = '\0';
    }
    else
    {
        return -1;
    }
    int pcihandle = open(sysfile, O_RDONLY);
    if (pcihandle < 0)
    {
        return -1;
    }

    ret = pread(pcihandle, &tmp, sizeof(uint32_t), ICX_IMC_MMIO_BASE_OFFSET);
    if (ret < 0)
    {
        close(pcihandle);
        return -1;
    }
    if (!tmp)
    {
        close(pcihandle);
        return -1;
    }
    addr = ((tmp & ICX_IMC_MMIO_BASE_MASK)) << ICX_IMC_MMIO_BASE_SHIFT;
    int mem_offset = ICX_IMC_MMIO_MEM0_OFFSET + pmc_idx * ICX_IMC_MEM_STRIDE;
    ret = pread(pcihandle, &tmp, sizeof(uint32_t), mem_offset);
    if (ret < 0)
    {
        close(pcihandle);
        return -1;
    }
    addr2 = ((tmp & ICX_IMC_MMIO_MEM_MASK)) << ICX_IMC_MMIO_MEM_SHIFT;
    addr |= addr2;
    addr += ICX_IMC_MMIO_FREERUN_OFFSET;

    close(pcihandle);

    pcihandle = open("/dev/mem", O_RDWR | O_SYNC);
    uint64_t page_mask = ~(pagesize - 1);
    void* maddr = mmap(0, ICX_IMC_MMIO_FREERUN_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, pcihandle, addr & page_mask);
    if (maddr == MAP_FAILED)
    {
        syslog(LOG_ERR, "ServerMem: MMAP of addr 0x%lx failed\n", addr & page_mask);
        close(pcihandle);
        return -1;
    }
    close(pcihandle);
    //*reg_offset = addr - (addr & page_mask);
    //*raw_offset = addr;
    *mmap_addr = maddr + (addr - (addr & page_mask));
    
    return 0;
}

static int
servermem_init()
{
    int i = 0;
    int j = 0;
    int ret = 0;
    if (!isServerMem)
        return -1;
    if (avail_sockets == 0)
        avail_sockets = getNumberOfSockets();

    servermem_addrs = malloc(avail_sockets * sizeof(void **));
    if (!servermem_addrs)
    {
        return -ENOMEM;
    }
    servermem_freerun_addrs = malloc(avail_sockets * sizeof(void **));
    if (!servermem_freerun_addrs)
    {
        free(servermem_addrs);
        return -ENOMEM;
    }
    for (i = 0; i < avail_sockets; i++)
    {
        servermem_addrs[i] = malloc(ICX_NUMBER_IMC_DEVS * ICX_NUMBER_IMC_CHN * sizeof(void *));
        if (!servermem_addrs[i])
        {
            for (j = 0; j < i; j++)
            {
                free(servermem_addrs[j]);
            }
            free(servermem_addrs);
            servermem_addrs = NULL;
            return -ENOMEM;
        }
        servermem_freerun_addrs[i] = malloc(ICX_NUMBER_IMC_DEVS * sizeof(void *));
        if (!servermem_freerun_addrs[i])
        {
            for (j = 0; j < i; j++)
            {
                free(servermem_freerun_addrs[j]);
            }
            for (j = 0; j < avail_sockets; j++)
            {
                free(servermem_addrs[j]);
            }
            free(servermem_freerun_addrs);
            servermem_freerun_addrs = NULL;
            free(servermem_addrs);
            servermem_addrs = NULL;
            return -ENOMEM;
        }
        for (j = 0; j < ICX_NUMBER_IMC_DEVS * ICX_NUMBER_IMC_CHN; j++)
        {
            servermem_addrs[i][j] = NULL;
        }
        for (j = 0; j < ICX_NUMBER_IMC_DEVS ; j++)
        {
            servermem_freerun_addrs[i][j] = NULL;
        }
    }
    for (i = 0; i < avail_sockets; i++)
    {
        for (j = 0; j < ICX_NUMBER_IMC_DEVS * ICX_NUMBER_IMC_CHN; j++)
        {
            
            ret = servermem_getStartAddr(i, j, &servermem_addrs[i][j]);
            if (ret < 0)
            {
                syslog(LOG_ERR, "Failed to open servermem socket %d offset %d\n", i, j);
            }
        }
        for (j = 0; j < ICX_NUMBER_IMC_DEVS; j++)
        {
            ret = servermem_freerun_getStartAddr(i, j, &servermem_freerun_addrs[i][j]);
            if (ret < 0)
            {
                syslog(LOG_ERR, "Failed to open servermem socket %d device %d\n", i, j);
            }
        }
    }
    return 0;
}


static void
servermem_read(AccessDataRecord *dRecord)
{
    uint64_t data = 0;
    uint32_t reg = dRecord->reg;
    uint32_t socketId = dRecord->cpu;
    uint32_t device = dRecord->device;
    int offset = device - MMIO_IMC_DEVICE_0_CH_0;

    dRecord->errorcode = ERR_NOERROR;
    dRecord->data = 0x0ULL;


    if (!lock_check())
    {
        syslog(LOG_ERR,"Access to performance counters is locked.\n");
        dRecord->errorcode = ERR_LOCKED;
        return;
    }

    if (!isServerMem)
    {
        syslog(LOG_ERR, "ServerMem: Not available for this architecture\n");
        dRecord->errorcode = ERR_RESTREG;
        return;
    }
    if (socketId < 0 || socketId >= 2)
    {
        syslog(LOG_ERR, "ServerMem: Socket %d out of range\n", socketId);
        dRecord->errorcode = ERR_NODEV;
        return;
    }
    if (offset < 0 || offset >= ICX_NUMBER_IMC_DEVS*ICX_NUMBER_IMC_CHN)
    {
        syslog(LOG_ERR, "ServerMem: Device offset %d out of range\n", offset);
        dRecord->errorcode = ERR_NODEV;
        return;
    }
    if (allowedPci && (!allowedPci(pci_devices_daemon[device].type, reg)))
    {
        dRecord->errorcode = ERR_RESTREG;
        return;
    }
    if (servermem_addrs && servermem_addrs[socketId] && servermem_addrs[socketId][offset])
    {
#ifdef DEBUG_LIKWID
        syslog(LOG_INFO, "ServerMem: Read S %d Box %d Reg 0x%x\n", socketId, offset, reg);
#endif
        switch(reg)
        {
            case 0x08:
            case 0x10:
            case 0x18:
            case 0x20:
            case 0x38:
                dRecord->data = *((uint64_t*)(servermem_addrs[socketId][offset] + reg));
                break;
            case 0x40:
            case 0x44:
            case 0x48:
            case 0x4C:
            case 0x00:
            case 0x5C:
            case 0x54:
               dRecord->data = (uint64_t) *((uint32_t*)(servermem_addrs[socketId][offset] + reg));
               break;
            default:
                break;
        }
    }
    else
    {
        dRecord->errorcode = ERR_NODEV;
    }
}

static void
servermem_freerun_read(AccessDataRecord *dRecord)
{
    uint64_t data = 0;
    uint32_t reg = dRecord->reg;
    uint32_t socketId = dRecord->cpu;
    uint32_t device = dRecord->device;
    int offset = device - MMIO_IMC_DEVICE_0_FREERUN;

    dRecord->errorcode = ERR_NOERROR;
    dRecord->data = 0x0ULL;


    if (!lock_check())
    {
        syslog(LOG_ERR,"Access to performance counters is locked.\n");
        dRecord->errorcode = ERR_LOCKED;
        return;
    }

    if (!isServerMem)
    {
        syslog(LOG_ERR, "ServerMem: Not available for this architecture\n");
        dRecord->errorcode = ERR_RESTREG;
        return;
    }
    if (socketId < 0 || socketId >= 2)
    {
        syslog(LOG_ERR, "ServerMem: Socket %d out of range\n", socketId);
        dRecord->errorcode = ERR_NODEV;
        return;
    }
    if (offset < 0 || offset >= ICX_NUMBER_IMC_DEVS)
    {
        syslog(LOG_ERR, "ServerMem: Device offset %d out of range\n", offset);
        dRecord->errorcode = ERR_NODEV;
        return;
    }
    if (allowedPci && (!allowedPci(pci_devices_daemon[device].type, reg)))
    {
        dRecord->errorcode = ERR_RESTREG;
        return;
    }
    if (servermem_freerun_addrs && servermem_freerun_addrs[socketId] && servermem_freerun_addrs[socketId][offset])
    {
#ifdef DEBUG_LIKWID
        syslog(LOG_INFO, "ServerMem: Read S %d Box %d Reg 0x%x\n", socketId, offset, reg);
#endif
        switch(reg)
        {
            case 0x00:
            case 0x08:
            case 0x10:
            case 0x18:
            case 0x20:
                dRecord->data = *((uint64_t*)(servermem_freerun_addrs[socketId][offset] + reg));
                break;
            default:
                break;
        }
    }
    else
    {
        dRecord->errorcode = ERR_NODEV;
    }
}

static void
servermem_write(AccessDataRecord *dRecord)
{
    uint32_t reg = dRecord->reg;
    uint32_t socketId = dRecord->cpu;
    uint32_t device = dRecord->device;
    int offset = device - MMIO_IMC_DEVICE_0_CH_0;

    dRecord->errorcode = ERR_NOERROR;


    if (!lock_check())
    {
        syslog(LOG_ERR,"Access to performance counters is locked.\n");
        dRecord->errorcode = ERR_LOCKED;
        return;
    }

    if (!isServerMem)
    {
        syslog(LOG_ERR, "ServerMem: Not available for this architecture\n");
        dRecord->errorcode = ERR_RESTREG;
        return;
    }
    if (socketId < 0 || socketId >= 2)
    {
        syslog(LOG_ERR, "ServerMem: Socket %d out of range\n", socketId);
        dRecord->errorcode = ERR_NODEV;
        return;
    }
    if (offset < 0 || offset >= ICX_NUMBER_IMC_DEVS*ICX_NUMBER_IMC_CHN)
    {
        syslog(LOG_ERR, "ServerMem: Device offset %d out of range\n", offset);
        dRecord->errorcode = ERR_NODEV;
        return;
    }
    if (allowedPci && (!allowedPci(pci_devices_daemon[device].type, reg)))
    {
        dRecord->errorcode = ERR_RESTREG;
        return;
    }
    if (servermem_addrs && servermem_addrs[socketId] && servermem_addrs[socketId][offset])
    {
#ifdef DEBUG_LIKWID
        syslog(LOG_INFO, "ServerMem: Write S %d Box %d Reg 0x%x Data 0x%x\n", socketId, offset, reg, dRecord->data);
#endif
        switch(reg)
        {
            case 0x08:
            case 0x10:
            case 0x18:
            case 0x20:
            case 0x38:
                *((uint64_t*)(servermem_addrs[socketId][offset] + reg)) = dRecord->data;
                break;
            case 0x40:
            case 0x44:
            case 0x48:
            case 0x4C:
            case 0x00:
            case 0x5C:
            case 0x54:
               *((uint32_t*)(servermem_addrs[socketId][offset] + reg)) = (uint32_t) dRecord->data;
               break;
            default:
                break;
        }
    }
    else
    {
        dRecord->errorcode = ERR_NODEV;
    }
}

static void
servermem_freerun_write(AccessDataRecord *dRecord)
{
    uint32_t reg = dRecord->reg;
    uint32_t socketId = dRecord->cpu;
    uint32_t device = dRecord->device;
    int offset = device - MMIO_IMC_DEVICE_0_FREERUN;

    dRecord->errorcode = ERR_NOERROR;


    if (!lock_check())
    {
        syslog(LOG_ERR,"Access to performance counters is locked.\n");
        dRecord->errorcode = ERR_LOCKED;
        return;
    }

    if (!isServerMem)
    {
        syslog(LOG_ERR, "ServerMem: Not available for this architecture\n");
        dRecord->errorcode = ERR_RESTREG;
        return;
    }
    if (socketId < 0 || socketId >= 2)
    {
        syslog(LOG_ERR, "ServerMem: Socket %d out of range\n", socketId);
        dRecord->errorcode = ERR_NODEV;
        return;
    }
    if (offset < 0 || offset >= ICX_NUMBER_IMC_DEVS)
    {
        syslog(LOG_ERR, "ServerMem: Device offset %d out of range\n", offset);
        dRecord->errorcode = ERR_NODEV;
        return;
    }
    if (allowedPci && (!allowedPci(pci_devices_daemon[device].type, reg)))
    {
        dRecord->errorcode = ERR_RESTREG;
        return;
    }
    if (servermem_freerun_addrs && servermem_freerun_addrs[socketId] && servermem_freerun_addrs[socketId][offset])
    {
#ifdef DEBUG_LIKWID
        syslog(LOG_INFO, "ServerMem: Write S %d Box %d Reg 0x%x Data 0x%x\n", socketId, offset, reg, dRecord->data);
#endif
        switch(reg)
        {
            case 0x00:
            case 0x08:
            case 0x10:
            case 0x18:
            case 0x20:
                *((uint64_t*)(servermem_freerun_addrs[socketId][offset] + reg)) = dRecord->data;
                break;
            default:
                break;
        }
    }
    else
    {
        dRecord->errorcode = ERR_NODEV;
    }
}

static void
servermem_check(AccessDataRecord *dRecord)
{
    dRecord->errorcode = ERR_NOERROR;
    if (!isServerMem)
    {
        syslog(LOG_ERR, "ServerMem: Not available for this architecture\n");
        dRecord->errorcode = ERR_RESTREG;
        return;
    }
    int socketId = dRecord->cpu;
    int offset = dRecord->device - MMIO_IMC_DEVICE_0_CH_0;
#ifdef DEBUG_LIKWID
    syslog(LOG_ERR, "ServerMem: Check Socket %d Box %d\n", socketId, offset);
#endif
    if (socketId < 0 || socketId >= 2)
    {
        syslog(LOG_ERR, "ServerMem: Socket %d out of range\n", socketId);
        dRecord->errorcode = ERR_NODEV;
        return;
    }
    if (offset < 0 || offset >= ICX_NUMBER_IMC_DEVS*ICX_NUMBER_IMC_CHN)
    {
        syslog(LOG_ERR, "ServerMem: Device offset %d out of range\n", offset);
        dRecord->errorcode = ERR_NODEV;
        return;
    }
    if (!(servermem_addrs && servermem_addrs[socketId] && servermem_addrs[socketId][offset]))
    {
        syslog(LOG_ERR, "ServerMem: Socket %d Device %d addr NULL\n", socketId, offset);
        dRecord->errorcode = ERR_NODEV;
        return;
    }
}

static void
servermem_freerun_check(AccessDataRecord *dRecord)
{
    dRecord->errorcode = ERR_NOERROR;
    if (!isServerMem)
    {
        syslog(LOG_ERR, "ServerMem: Not available for this architecture\n");
        dRecord->errorcode = ERR_RESTREG;
        return;
    }
    int socketId = dRecord->cpu;
    int offset = dRecord->device - MMIO_IMC_DEVICE_0_FREERUN;
#ifdef DEBUG_LIKWID
    syslog(LOG_ERR, "ServerMem: Check Socket %d Box %d\n", socketId, offset);
#endif
    if (socketId < 0 || socketId >= 2)
    {
        syslog(LOG_ERR, "ServerMem: Socket %d out of range\n", socketId);
        dRecord->errorcode = ERR_NODEV;
        return;
    }
    if (offset < 0 || offset >= ICX_NUMBER_IMC_DEVS)
    {
        syslog(LOG_ERR, "ServerMem: Device offset %d out of range\n", offset);
        dRecord->errorcode = ERR_NODEV;
        return;
    }
    if (!(servermem_freerun_addrs && servermem_freerun_addrs[socketId] && servermem_freerun_addrs[socketId][offset]))
    {
        syslog(LOG_ERR, "ServerMem: Socket %d Device %d addr NULL\n", socketId, offset);
        dRecord->errorcode = ERR_NODEV;
        return;
    }
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
                                    fclose(fp);
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
                if (ret != 1)
                {
                    devid = -1;
                }
            }
            fclose(fp);
        }
    }
    return devid;
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
            pclose(fpipe);
            return socket;
        }
    }
    pclose(fpipe);
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
    close_accessdaemon();
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
    signal(SIGCHLD, SIG_IGN);

    /* If we got a good PID, then we can exit the parent process. */
    if (pid > 0)
    {
        syslog(LOG_ERR, "Closing parent process %d", *parentPid);
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

static void handle_record_default(AccessDataRecord* record)
{
    if (record->type == DAEMON_READ)
    {
        if (record->device == MSR_DEV)
        {
            msr_read(record);
        }
        else if (isClientMem)
        {
            clientmem_read(record);
        }
        else
        {
            if (record->device >= MMIO_IMC_DEVICE_0_CH_0 && record->device <= MMIO_IMC_DEVICE_3_CH_1)
            {
                servermem_read(record);
            }
            else if (record->device >= MMIO_IMC_DEVICE_0_FREERUN && record->device <= MMIO_IMC_DEVICE_3_FREERUN)
            {
                servermem_freerun_read(record);
            }
            else if (pci_devices_daemon != NULL)
            {
                pci_read(record);
            }
        }
    }
    else if (record->type == DAEMON_WRITE)
    {
        if (record->device == MSR_DEV)
        {
            msr_write(record);
            record->data = 0x0ULL;
        }
        else
        {
            if (record->device >= MMIO_IMC_DEVICE_0_CH_0 && record->device <= MMIO_IMC_DEVICE_3_CH_1)
            {
                servermem_write(record);
                record->data = 0x0ULL;
            }
            else if (record->device >= MMIO_IMC_DEVICE_0_FREERUN && record->device <= MMIO_IMC_DEVICE_3_FREERUN)
            {
                servermem_freerun_write(record);
                record->data = 0x0ULL;
            }
            else if (pci_devices_daemon != NULL)
            {
                pci_write(record);
                record->data = 0x0ULL;
            }
        }
    }
    else if (record->type == DAEMON_CHECK)
    {
        if (record->device == MSR_DEV)
        {
            msr_check(record);
        }
        else if (isClientMem)
        {
            clientmem_check(record);
        }
        else
        {
            if (record->device >= MMIO_IMC_DEVICE_0_CH_0 && record->device <= MMIO_IMC_DEVICE_3_CH_1)
            {
                servermem_check(record);
            }
            else if (record->device >= MMIO_IMC_DEVICE_0_FREERUN && record->device <= MMIO_IMC_DEVICE_3_FREERUN)
            {
                servermem_freerun_check(record);
            }
            else if (pci_devices_daemon != NULL)
            {
                pci_check(record);
            }
        }
    }
    else if (record->type == DAEMON_EXIT)
    {
        stop_daemon();
    }
    else
    {
        syslog(LOG_ERR, "unknown daemon access type  %d", record->type);
        record->errorcode = ERR_UNKNOWN;
    }
}

static void
msr_read_spr(AccessDataRecord * dRecord)
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
msr_write_spr(AccessDataRecord * dRecord)
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


static void spr_check(AccessDataRecord *record)
{
    record->errorcode = ERR_UNKNOWN;
    if (perfmon_discovery && record->cpu >= 0 && record->cpu < perfmon_discovery->num_sockets)
    {
        PerfmonDiscoverySocket* cur = &perfmon_discovery->sockets[record->cpu];
        if (cur->units)
        {
            if (cur->units[record->device].num_regs > 0)
            {
                record->errorcode = ERR_NOERROR;
                return;
            }
            else
            {
                record->errorcode = ERR_NODEV;
                return;
            }
        }
        else
        {
            record->errorcode = ERR_NODEV;
            return;
        }
    }
}

static int spr_get_register_offset(PerfmonDiscoveryUnit* unit)
{
    switch (unit->access_type)
    {
        case ACCESS_TYPE_MSR:
            return 1;
            break;
        case ACCESS_TYPE_MMIO:
        case ACCESS_TYPE_PCI:
            if (unit->bit_width <= 8)
            {
                return 1;
            }
            else if (unit->bit_width <= 16)
            {
                return 2;
            }
            else if (unit->bit_width <= 32)
            {
                return 4;
            }
            else if (unit->bit_width <= 64)
            {
                return 8;
            }
            break;
    }
    return 0;
}

static int spr_open_unit(PerfmonDiscoveryUnit* unit)
{
    int err = 0;
    int PAGE_SIZE = sysconf (_SC_PAGESIZE);
    if (!unit)
    {
        return -EINVAL;
    }
    int pcihandle = open("/dev/mem", O_RDWR);
    if (pcihandle < 0)
    {
        err = errno;
        return -err;
    }
    if (unit->access_type == ACCESS_TYPE_MMIO)
    {
        void* io_addr = mmap(NULL, unit->mmap_size, PROT_READ|PROT_WRITE, MAP_SHARED, pcihandle, unit->mmap_addr);
        if (io_addr == MAP_FAILED)
        {
            err = errno;
            close(pcihandle);
            return -err;
        }
        unit->io_addr = io_addr;
    }
    else if (unit->access_type == ACCESS_TYPE_PCI)
    {
        void* io_addr = mmap(NULL, unit->mmap_size, PROT_READ|PROT_WRITE, MAP_SHARED, pcihandle, unit->mmap_addr );
        if (io_addr == MAP_FAILED)
        {
            err = errno;
            close(pcihandle);
            return -err;
        }
        unit->io_addr = io_addr;
    }
    close(pcihandle);
    return 0;
}


static void spr_read_global(AccessDataRecord *record)
{
    record->errorcode = ERR_OPENFAIL;
    if (perfmon_discovery && record->cpu >= 0 && record->cpu < perfmon_discovery->num_sockets && record->device == MSR_UBOX_DEVICE)
    {
        PerfmonDiscoverySocket* cur = &perfmon_discovery->sockets[record->cpu];
        if (cur && cur->socket_id == record->cpu && cur->global.global_ctl && cur->global.access_type == ACCESS_TYPE_MSR && socket_map[record->cpu] >= 0)
        {
            AccessDataRecord msr_record = {
                .cpu = socket_map[record->cpu],
                .data = record->data,
                .device = MSR_DEV,
                .type = DAEMON_READ,
                .errorcode = ERR_NOERROR,
            };
            if (record->reg == FAKE_UNC_GLOBAL_CTRL)
            {
                msr_record.reg = cur->global.global_ctl;
            }
            else if ((record->reg >= FAKE_UNC_GLOBAL_STATUS0) && (record->reg <= FAKE_UNC_GLOBAL_STATUS3) && (cur->global.num_status > (record->reg-FAKE_UNC_GLOBAL_STATUS0)))
            {
                msr_record.reg = cur->global.global_ctl + cur->global.status_offset + ((record->reg - FAKE_UNC_GLOBAL_STATUS0));
            }
            msr_read_spr(&msr_record);
            record->errorcode = msr_record.errorcode;
        }
    }
}

static void spr_read_unit(AccessDataRecord *record)
{
    record->errorcode = ERR_OPENFAIL;
    if (perfmon_discovery && record->cpu >= 0 && record->cpu < perfmon_discovery->num_sockets)
    {
        PerfmonDiscoverySocket* cur = &perfmon_discovery->sockets[record->cpu];
        if (cur->units && cur->socket_id == record->cpu && cur->units[record->device].num_regs > 0)
        {
            PerfmonDiscoveryUnit* unit = &cur->units[record->device];
            int offset = 0;
            int reg_offset = 0;
            AccessDataRecord msr_record = {
                .cpu = socket_map[record->cpu],
                .data = 0x00,
                .device = MSR_DEV,
                .type = DAEMON_READ,
                .errorcode = ERR_NOERROR,
            };
            if ((!unit->io_addr) && (unit->mmap_addr))
            {
                int err = spr_open_unit(unit);
                if (err < 0)
                {
                    record->errorcode = ERR_OPENFAIL;
                    return;
                }
            }
            else if (!unit->mmap_addr)
            {
                record->errorcode = ERR_NODEV;
                return;
            }
            switch(record->reg)
            {
                case FAKE_UNC_UNIT_CTRL:
                    switch (unit->access_type)
                    {
                        case ACCESS_TYPE_MMIO:
                            record->data = (uint64_t)*((uint64_t *)(unit->io_addr + unit->mmap_offset));
                            record->errorcode = ERR_NOERROR;
                            break;
                        case ACCESS_TYPE_PCI:
                            if ((record->device >= PCI_HA_DEVICE_0 && record->device <= PCI_HA_DEVICE_31) ||
                                (record->device >= PCI_R3QPI_DEVICE_LINK_0 && record->device <= PCI_R3QPI_DEVICE_LINK_3) ||
                                (record->device >= PCI_QPI_DEVICE_PORT_0 && record->device <= PCI_QPI_DEVICE_PORT_3))
                            {
                                uint32_t lo = 0ULL, hi = 0ULL;
                                lo = (uint32_t)*((uint32_t *)(unit->io_addr + unit->mmap_offset));
                                hi = (uint32_t)*((uint32_t *)(unit->io_addr + unit->mmap_offset + sizeof(uint32_t)));
                                record->data = (((uint64_t)hi)<<32)|((uint64_t)lo);
                            }
                            else
                            {
                                record->data = (uint64_t)*((uint64_t *)(unit->io_addr + unit->mmap_offset));
                            }
                            record->errorcode = ERR_NOERROR;
                            break;
                        case ACCESS_TYPE_MSR:
                            msr_record.reg = unit->box_ctl;
                            msr_read_spr(&msr_record);
                            record->errorcode = msr_record.errorcode;
                            record->data = msr_record.data;
                            break;
                    }
                    break;
                case FAKE_UNC_UNIT_STATUS:
                    switch (unit->access_type)
                    {
                        case ACCESS_TYPE_MMIO:
                            record->data = (uint64_t)*((uint64_t *)(unit->io_addr + unit->mmap_offset + unit->status_offset));
                            record->errorcode = ERR_NOERROR;
                            break;
                        case ACCESS_TYPE_PCI:
                            if ((record->device >= PCI_HA_DEVICE_0 && record->device <= PCI_HA_DEVICE_31) ||
                                (record->device >= PCI_R3QPI_DEVICE_LINK_0 && record->device <= PCI_R3QPI_DEVICE_LINK_3) ||
                                (record->device >= PCI_QPI_DEVICE_PORT_0 && record->device <= PCI_QPI_DEVICE_PORT_3))
                            {
                                uint32_t lo = 0ULL, hi = 0ULL;
                                lo = (uint32_t)*((uint32_t *)(unit->io_addr + unit->mmap_offset + unit->status_offset));
                                hi = (uint32_t)*((uint32_t *)(unit->io_addr + unit->mmap_offset + unit->status_offset + sizeof(uint32_t)));
                                record->data = (((uint64_t)hi)<<32)|((uint64_t)lo);
                            }
                            else
                            {
                                record->data = (uint64_t)*((uint64_t *)(unit->io_addr + unit->mmap_offset + unit->status_offset));
                            }
                            record->errorcode = ERR_NOERROR;
                            break;
                        case ACCESS_TYPE_MSR:
                            msr_record.reg = unit->box_ctl + unit->status_offset;
                            msr_read_spr(&msr_record);
                            record->data = msr_record.data;
                            record->errorcode = msr_record.errorcode;
                            break;
                    }
                    break;
                case FAKE_UNC_CTRL0:
                case FAKE_UNC_CTRL1:
                case FAKE_UNC_CTRL2:
                case FAKE_UNC_CTRL3:
                    offset = (record->reg - FAKE_UNC_CTRL0);
                    reg_offset = spr_get_register_offset(unit);
                    switch (unit->access_type)
                    {
                        case ACCESS_TYPE_MMIO:
                            if ((record->device >= MMIO_IMC_DEVICE_0_CH_0 && record->device <= MMIO_IMC_DEVICE_1_CH_7) ||
                                (record->device >= MMIO_HBM_DEVICE_0 && record->device <= MMIO_HBM_DEVICE_31))
                            {
                                record->data = (uint32_t)*((uint32_t *)(unit->io_addr + unit->mmap_offset + unit->ctrl_offset + (sizeof(uint32_t) * offset)));
                            }
                            else
                            {
                                record->data = (uint64_t)*((uint64_t *)(unit->io_addr + unit->mmap_offset + unit->ctrl_offset + (reg_offset * offset)));
                            }
                            record->errorcode = ERR_NOERROR;
                            break;
                        case ACCESS_TYPE_PCI:
                            if ((record->device >= PCI_HA_DEVICE_0 && record->device <= PCI_HA_DEVICE_31) ||
                                (record->device >= PCI_R3QPI_DEVICE_LINK_0 && record->device <= PCI_R3QPI_DEVICE_LINK_3) ||
                                (record->device >= PCI_QPI_DEVICE_PORT_0 && record->device <= PCI_QPI_DEVICE_PORT_3))
                            {
                                uint32_t lo = 0ULL, hi = 0ULL;
                                lo = (uint32_t)*((uint32_t *)(unit->io_addr + unit->mmap_offset + unit->ctrl_offset + (sizeof(uint32_t) * offset)));
                                hi = (uint32_t)*((uint32_t *)(unit->io_addr + unit->mmap_offset + unit->ctrl_offset + (sizeof(uint32_t) * offset) + sizeof(uint32_t)));
                                record->data = (((uint64_t)hi)<<32)|((uint64_t)lo);
                            }
                            else
                            {
                                record->data = (uint64_t)*((uint64_t *)(unit->io_addr + unit->mmap_offset + unit->ctrl_offset + (reg_offset * offset)));
                            }
                            
                            record->errorcode = ERR_NOERROR;
                            break;
                        case ACCESS_TYPE_MSR:
                            msr_record.reg = unit->box_ctl + unit->ctrl_offset + (reg_offset * offset);
                            msr_read_spr(&msr_record);
                            record->data = msr_record.data;
                            record->errorcode = msr_record.errorcode;
                            break;
                    }
                    break;
                case FAKE_UNC_CTR0:
                case FAKE_UNC_CTR1:
                case FAKE_UNC_CTR2:
                case FAKE_UNC_CTR3:
                    offset = (record->reg - FAKE_UNC_CTR0);
                    reg_offset = spr_get_register_offset(unit);
                    switch (unit->access_type)
                    {
                        case ACCESS_TYPE_MMIO:
                            record->data = (uint64_t)*((uint64_t *)(unit->io_addr + unit->mmap_offset + unit->ctr_offset + (reg_offset * offset)));
                            record->errorcode = ERR_NOERROR;
                            break;
                        case ACCESS_TYPE_PCI:
                            if ((record->device >= PCI_HA_DEVICE_0 && record->device <= PCI_HA_DEVICE_31) ||
                                (record->device >= PCI_R3QPI_DEVICE_LINK_0 && record->device <= PCI_R3QPI_DEVICE_LINK_3) ||
                                (record->device >= PCI_QPI_DEVICE_PORT_0 && record->device <= PCI_QPI_DEVICE_PORT_3))
                            {
                                uint32_t lo = 0ULL, hi = 0ULL;
                                lo = (uint32_t)*((uint32_t *)(unit->io_addr + unit->mmap_offset + unit->ctr_offset + (sizeof(uint32_t) * offset)));
                                hi = (uint32_t)*((uint32_t *)(unit->io_addr + unit->mmap_offset + unit->ctr_offset + (sizeof(uint32_t) * offset) + sizeof(uint32_t)));
                                record->data = (((uint64_t)hi)<<32)|((uint64_t)lo);
                            }
                            else
                            {
                                record->data = (uint64_t)*((uint64_t *)(unit->io_addr + unit->mmap_offset + unit->ctr_offset + (reg_offset * offset)));
                            }
                            record->errorcode = ERR_NOERROR;
                            break;
                        case ACCESS_TYPE_MSR:
                            msr_record.reg = unit->box_ctl + unit->ctr_offset + (reg_offset * offset);
                            msr_read_spr(&msr_record);
                            record->data = msr_record.data;
                            record->errorcode = msr_record.errorcode;
                            break;
                    }
                    break;
                case FAKE_UNC_FILTER0:
                    offset = (record->reg - FAKE_UNC_FILTER0);
                    reg_offset = unit->filter_offset;
                    if (reg_offset != 0x0 && unit->access_type == ACCESS_TYPE_MSR)
                    {
                        msr_record.reg = unit->box_ctl + reg_offset;
                        msr_read_spr(&msr_record);
                        record->data = msr_record.data;
                        record->errorcode = msr_record.errorcode;
                    }
                    break;
                case FAKE_UNC_FIXED_CTRL:
                    if (unit->fixed_ctrl_offset != 0)
                    {
                        if (unit->access_type == ACCESS_TYPE_MSR)
                        {
                            msr_record.reg = unit->box_ctl + unit->fixed_ctrl_offset;
                            msr_read_spr(&msr_record);
                            record->data = msr_record.data;
                            record->errorcode = msr_record.errorcode;
                        }
                        else if (unit->access_type == ACCESS_TYPE_MMIO)
                        {
                            uint32_t lo = 0ULL, hi = 0ULL;
                            lo = (uint32_t)*((uint32_t *)(unit->io_addr + unit->mmap_offset + unit->fixed_ctrl_offset));
                            hi = (uint32_t)*((uint32_t *)(unit->io_addr + unit->mmap_offset + unit->fixed_ctrl_offset + sizeof(uint32_t)));
                            record->data = (((uint64_t)hi)<<32)|((uint64_t)lo);
                            record->errorcode = ERR_NOERROR;
                        }
                    }
                    break;
               case FAKE_UNC_FIXED_CTR:
                    if (unit->fixed_ctr_offset != 0)
                    {
                        if (unit->access_type == ACCESS_TYPE_MSR)
                        {
                            msr_record.reg = unit->box_ctl + unit->fixed_ctr_offset;
                            msr_read_spr(&msr_record);
                            record->data = msr_record.data;
                            record->errorcode = msr_record.errorcode;
                        }
                        else if (unit->access_type == ACCESS_TYPE_MMIO)
                        {
                            record->data = (uint64_t)*((uint64_t *)(unit->io_addr + unit->mmap_offset + unit->fixed_ctr_offset));
                            record->errorcode = ERR_NOERROR;
                        }
                    }
                    break;
            }
        }
    }
}

static void spr_write_global(AccessDataRecord *record)
{
    record->errorcode = ERR_OPENFAIL;
    if (perfmon_discovery && record->cpu >= 0 && record->cpu < perfmon_discovery->num_sockets && record->device == MSR_UBOX_DEVICE)
    {
        PerfmonDiscoverySocket* cur = &perfmon_discovery->sockets[record->cpu];
        if (cur && cur->socket_id == record->cpu && cur->global.global_ctl && cur->global.access_type == ACCESS_TYPE_MSR && socket_map[record->cpu] >= 0)
        {
            AccessDataRecord msr_record = {
                .cpu = socket_map[record->cpu],
                .data = record->data,
                .reg = 0x0,
                .device = MSR_DEV,
                .type = DAEMON_WRITE,
                .errorcode = ERR_NOERROR,
            };
            if (record->reg == FAKE_UNC_GLOBAL_CTRL)
            {
                msr_record.reg = cur->global.global_ctl;
            }
            else if ((record->reg >= FAKE_UNC_GLOBAL_STATUS0) && (record->reg <= FAKE_UNC_GLOBAL_STATUS3) && (cur->global.num_status > (record->reg-FAKE_UNC_GLOBAL_STATUS0)))
            {
                msr_record.reg = cur->global.global_ctl + cur->global.status_offset + ((record->reg - FAKE_UNC_GLOBAL_STATUS0));
            }
            if (msr_record.reg != 0x0)
            {
                msr_write_spr(&msr_record);
                record->errorcode = msr_record.errorcode;
            }
        }
    }
}

static void spr_write_unit(AccessDataRecord *record)
{
    record->errorcode = ERR_OPENFAIL;
    if (perfmon_discovery && record->cpu >= 0 && record->cpu < perfmon_discovery->num_sockets)
    {
        PerfmonDiscoverySocket* cur = &perfmon_discovery->sockets[record->cpu];
        if (cur->units && cur->socket_id == record->cpu && cur->units[record->device].num_regs > 0)
        {
            PerfmonDiscoveryUnit* unit = &cur->units[record->device];
            int offset = 0;
            int reg_offset = 0;
            AccessDataRecord msr_record = {
                .cpu = socket_map[record->cpu],
                .data = record->data,
                .device = MSR_DEV,
                .type = DAEMON_WRITE,
                .errorcode = ERR_NOERROR,
            };
            if ((!unit->io_addr) && (unit->mmap_addr))
            {
                int err = spr_open_unit(unit);
                if (err < 0)
                {
                    record->errorcode = ERR_OPENFAIL;
                    return;
                }
            }
            else if (!unit->mmap_addr)
            {
                record->errorcode = ERR_NODEV;
                return;
            }
            switch(record->reg)
            {
                case FAKE_UNC_UNIT_CTRL:
                    
                    switch (unit->access_type)
                    {
                        case ACCESS_TYPE_MMIO:
                            *((uint64_t *)(unit->io_addr + unit->mmap_offset)) = record->data;
                            record->errorcode = ERR_NOERROR;
                            break;
                        case ACCESS_TYPE_PCI:
                            if ((record->device >= PCI_HA_DEVICE_0 && record->device <= PCI_HA_DEVICE_31) ||
                                (record->device >= PCI_R3QPI_DEVICE_LINK_0 && record->device <= PCI_R3QPI_DEVICE_LINK_3) ||
                                (record->device >= PCI_QPI_DEVICE_PORT_0 && record->device <= PCI_QPI_DEVICE_PORT_3))
                            {
                                *((uint32_t *)(unit->io_addr + unit->mmap_offset)) = (uint32_t)record->data;
                                *((uint32_t *)(unit->io_addr + unit->mmap_offset + sizeof(uint32_t))) = (uint32_t)(record->data>>32);
                            }
                            else
                            {
                                *((uint64_t *)(unit->io_addr + unit->mmap_offset)) = (uint64_t)record->data;
                            }
                            record->errorcode = ERR_NOERROR;
                            break;
                        case ACCESS_TYPE_MSR:
                            msr_record.reg = unit->box_ctl;
                            msr_write_spr(&msr_record);
                            record->errorcode = msr_record.errorcode;
                            break;
                    }
                    break;
                case FAKE_UNC_UNIT_STATUS:
                    record->data = (uint64_t)*((uint64_t *)(unit->io_addr + unit->mmap_offset + unit->status_offset));
                    switch (unit->access_type)
                    {
                        case ACCESS_TYPE_MMIO:
                            *((uint64_t *)(unit->io_addr + unit->mmap_offset + unit->status_offset)) = record->data;
                            record->errorcode = ERR_NOERROR;
                            break;
                        case ACCESS_TYPE_PCI:
                            if ((record->device >= PCI_HA_DEVICE_0 && record->device <= PCI_HA_DEVICE_31) ||
                                (record->device >= PCI_R3QPI_DEVICE_LINK_0 && record->device <= PCI_R3QPI_DEVICE_LINK_3) ||
                                (record->device >= PCI_QPI_DEVICE_PORT_0 && record->device <= PCI_QPI_DEVICE_PORT_3))
                            {
                                *((uint32_t *)(unit->io_addr + unit->mmap_offset + unit->status_offset)) = (uint32_t)record->data;
                                *((uint32_t *)(unit->io_addr + unit->mmap_offset + unit->status_offset + sizeof(uint32_t))) = (uint32_t)(record->data>>32);
                            }
                            else
                            {
                                *((uint64_t *)(unit->io_addr + unit->mmap_offset + unit->status_offset)) = (uint64_t)record->data;
                            }
                            record->errorcode = ERR_NOERROR;
                            break;
                        case ACCESS_TYPE_MSR:
                            msr_record.reg = unit->box_ctl + unit->status_offset;
                            msr_write_spr(&msr_record);
                            record->errorcode = msr_record.errorcode;
                            break;
                    }
                    break;
                case FAKE_UNC_CTRL0:
                case FAKE_UNC_CTRL1:
                case FAKE_UNC_CTRL2:
                case FAKE_UNC_CTRL3:
                    offset = (record->reg - FAKE_UNC_CTRL0);
                    reg_offset = spr_get_register_offset(unit);
                    
                    switch (unit->access_type)
                    {
                        case ACCESS_TYPE_MMIO:
                            if ((record->device >= MMIO_IMC_DEVICE_0_CH_0 && record->device <= MMIO_IMC_DEVICE_1_CH_7) ||
                                (record->device >= MMIO_HBM_DEVICE_0 && record->device <= MMIO_HBM_DEVICE_31))
                            {
                                *((uint32_t *)(unit->io_addr + unit->mmap_offset + unit->ctrl_offset + (sizeof(uint32_t) * offset))) = (uint32_t)record->data;
                            }
                            else
                            {
                                *((uint64_t *)(unit->io_addr + unit->mmap_offset + unit->ctrl_offset + (reg_offset * offset))) = (uint64_t)record->data;
                            }
                            record->errorcode = ERR_NOERROR;
                            break;
                        case ACCESS_TYPE_PCI:
                            if ((record->device >= PCI_HA_DEVICE_0 && record->device <= PCI_HA_DEVICE_31) ||
                                (record->device >= PCI_R3QPI_DEVICE_LINK_0 && record->device <= PCI_R3QPI_DEVICE_LINK_3) ||
                                (record->device >= PCI_QPI_DEVICE_PORT_0 && record->device <= PCI_QPI_DEVICE_PORT_3))
                            {
                                *((uint32_t *)(unit->io_addr + unit->mmap_offset + unit->ctrl_offset + (sizeof(uint32_t) * offset))) = (uint32_t)record->data;
                                *((uint32_t *)(unit->io_addr + unit->mmap_offset + unit->ctrl_offset + (sizeof(uint32_t) * offset) + sizeof(uint32_t))) = (uint32_t)(record->data>>32);
                            }
                            else
                            {
                                *((uint64_t *)(unit->io_addr + unit->mmap_offset + unit->ctrl_offset + (reg_offset * offset))) = (uint64_t)record->data;
                            }
                            record->errorcode = ERR_NOERROR;
                            break;
                        case ACCESS_TYPE_MSR:
                            msr_record.reg = unit->box_ctl + unit->ctrl_offset + (reg_offset * offset);
                            msr_write_spr(&msr_record);
                            record->errorcode = msr_record.errorcode;
                            break;
                    }
                    break;
                case FAKE_UNC_CTR0:
                case FAKE_UNC_CTR1:
                case FAKE_UNC_CTR2:
                case FAKE_UNC_CTR3:
                    offset = (record->reg - FAKE_UNC_CTR0);
                    reg_offset = spr_get_register_offset(unit);
                    
                    switch (unit->access_type)
                    {
                        case ACCESS_TYPE_MMIO:
                            *((uint64_t *)(unit->io_addr + unit->mmap_offset + unit->ctr_offset + (reg_offset * offset))) = record->data;
                            record->errorcode = ERR_NOERROR;
                            break;
                        case ACCESS_TYPE_PCI:
                            if ((record->device >= PCI_HA_DEVICE_0 && record->device <= PCI_HA_DEVICE_31) ||
                                (record->device >= PCI_R3QPI_DEVICE_LINK_0 && record->device <= PCI_R3QPI_DEVICE_LINK_3) ||
                                (record->device >= PCI_QPI_DEVICE_PORT_0 && record->device <= PCI_QPI_DEVICE_PORT_3))
                            {
                                *((uint32_t *)(unit->io_addr + unit->mmap_offset + unit->ctr_offset + (sizeof(uint32_t) * offset))) = (uint32_t)record->data;
                                *((uint32_t *)(unit->io_addr + unit->mmap_offset + unit->ctr_offset + (sizeof(uint32_t) * offset) + sizeof(uint32_t))) = (uint32_t)(record->data>>32);
                            }
                            else
                            {
                                *((uint64_t *)(unit->io_addr + unit->mmap_offset + unit->ctr_offset + (reg_offset * offset))) = (uint64_t)record->data;
                            }
                            record->errorcode = ERR_NOERROR;
                            break;
                        case ACCESS_TYPE_MSR:
                            msr_record.reg = unit->box_ctl + unit->ctr_offset + (reg_offset * offset);
                            msr_write_spr(&msr_record);
                            record->errorcode = msr_record.errorcode;
                            break;
                    }
                    break;
                case FAKE_UNC_FILTER0:
                    offset = (record->reg - FAKE_UNC_FILTER0);
                    if (unit->filter_offset != 0x0 && unit->access_type == ACCESS_TYPE_MSR)
                    {
                            msr_record.reg = unit->box_ctl + unit->filter_offset;
                            msr_write_spr(&msr_record);
                            record->errorcode = msr_record.errorcode;
                    }
                    break;
                case FAKE_UNC_FIXED_CTRL:
                    if (unit->fixed_ctrl_offset != 0)
                    {
                        if (unit->access_type == ACCESS_TYPE_MSR)
                        {
                            msr_record.reg = unit->box_ctl + unit->fixed_ctrl_offset;
                            msr_write_spr(&msr_record);
                            record->errorcode = msr_record.errorcode;
                        }
                        else if (unit->access_type == ACCESS_TYPE_MMIO)
                        {
                            *((uint32_t *)(unit->io_addr + unit->mmap_offset + unit->fixed_ctrl_offset)) = (uint32_t)record->data;
                            *((uint32_t *)(unit->io_addr + unit->mmap_offset + unit->fixed_ctrl_offset + sizeof(uint32_t))) = (uint32_t)(record->data>>32);
                            record->errorcode = ERR_NOERROR;
                        }
                    }
                    break;
               case FAKE_UNC_FIXED_CTR:
                    if (unit->fixed_ctr_offset != 0)
                    {
                        if (unit->access_type == ACCESS_TYPE_MSR)
                        {
                            msr_record.reg = unit->box_ctl + unit->fixed_ctr_offset;
                            msr_write_spr(&msr_record);
                            record->errorcode = msr_record.errorcode;
                        }
                        else if (unit->access_type == ACCESS_TYPE_MMIO)
                        {
                            *((uint64_t *)(unit->io_addr + unit->mmap_offset + unit->fixed_ctr_offset)) = record->data;
                            record->errorcode = ERR_NOERROR;
                        }
                    }
                    break;
            }
        }
    }
}

static void spr_read(AccessDataRecord *record)
{
    if (record->device == MSR_UBOX_DEVICE && ((record->reg == FAKE_UNC_GLOBAL_CTRL) || ((record->reg >= FAKE_UNC_GLOBAL_STATUS0) && (record->reg <= FAKE_UNC_GLOBAL_STATUS3))))
    {
        spr_read_global(record);
    }
    else
    {
        spr_read_unit(record);
    }
}

static void spr_write(AccessDataRecord *record)
{
    if (record->device == MSR_UBOX_DEVICE && ((record->reg == FAKE_UNC_GLOBAL_CTRL) || ((record->reg >= FAKE_UNC_GLOBAL_STATUS0) && (record->reg <= FAKE_UNC_GLOBAL_STATUS3))))
    {
        spr_write_global(record);
    }
    else
    {
        spr_write_unit(record);
    }
}



static void handle_record_spr(AccessDataRecord *record)
{
    if (record->type == DAEMON_READ)
    {
        if (record->device == MSR_DEV)
        {
            msr_read(record);
        }
        else
        {
            spr_read(record);
        }
    }
    else if (record->type == DAEMON_WRITE)
    {
        if (record->device == MSR_DEV)
        {
            msr_write(record);
            record->data = 0x0ULL;
        }
        else
        {
            spr_write(record);
        }
    }
    else if (record->type == DAEMON_CHECK)
    {
        if (record->device == MSR_DEV)
        {
            msr_check(record);
        }
        else
        {
            spr_check(record);
        }
    }
    else if (record->type == DAEMON_EXIT)
    {
        stop_daemon();
    }
    else
    {
        syslog(LOG_ERR, "unknown daemon access type  %d", record->type);
        record->errorcode = ERR_UNKNOWN;
    }
}

/* #####  MAIN FUNCTION DEFINITION   ################## */

int main(void)
{
    int ret;
    pid_t pid = getpid();
    struct sockaddr_un  addr1;
    socklen_t socklen;
    AccessDataRecord dRecord;
    mode_t oldumask;
    uint32_t numHWThreads = sysconf(_SC_NPROCESSORS_CONF);
    uint32_t model;
    struct stat stats;
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

    stat("/run/systemd/system", &stats);
    if (!S_ISDIR(stats.st_mode)) {
        daemonize(&pid);
    }
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
                         (model == KABYLAKE2) ||
                         (model == COMETLAKE1) ||
                         (model == COMETLAKE2) ||
                         (model == CANNONLAKE))
                {
                    allowed = allowed_sandybridge;
                    isClientMem = 1;
                }
                else if (model == ICELAKE1 || model == ICELAKE2 || model == ROCKETLAKE)
                {
                    allowed = allowed_icl;
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
                else if (model == ICELAKEX1 || model == ICELAKEX2)
                {
                    isPCIUncore = 1;
                    isServerMem = 1;
                    allowed = allowed_icx;
                    allowedPci = allowed_pci_icx;
                    isPCI64 = 1;
                }
                else if (model == SAPPHIRERAPIDS)
                {
                    isPCIUncore = 1;
                    allowed = allowed_spr;
                    isPCI64 = 1;
                    allowedPci = allowed_pci_spr;
                    isIntelUncoreDiscovery = 1;
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
                else if (model == SAPPHIRERAPIDS)
                {
                    allowed = allowed_icx;
                    isPCI64 = 1;
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
                switch (model)
                {
                    case ZEN2_RYZEN:
                    case ZEN2_RYZEN2:
                    case ZEN2_RYZEN3:
                        allowed = allowed_amd17_zen2;
                        break;
                    default:
                        allowed = allowed_amd17;
                        break;
                }
                break;
            case ZEN3_FAMILY:
                switch (model)
                {
                    case ZEN3_RYZEN:
                    case ZEN3_RYZEN2:
                    case ZEN3_RYZEN3:
                    case ZEN3_EPYC_TRENTO:
                        allowed = allowed_amd17_zen2;
                        break;
                    case ZEN4_RYZEN:
                    case ZEN4_EPYC:
                        allowed = allowed_amd19_zen4;
                        break;
                    default:
                        allowed = allowed_amd17;
                        break;
                }
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
            snprintf(msr_file_name, MAX_PATH_LENGTH-1, "/dev/cpu/%d/msr", i);
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
            else if (model == ICELAKEX1 || model == ICELAKEX2)
            {
                pci_devices_daemon = icelakeX_pci_devices;
            }
            else if ((model == XEON_PHI_KNL) ||
                     (model == XEON_PHI_KML))
            {
                pci_devices_daemon = knl_pci_devices;
            }
            else if (isIntelUncoreDiscovery)
            {
                pci_devices_daemon = NULL;
                int err = perfmon_uncore_discovery(&perfmon_discovery);
                if (err < 0)
                {
                    syslog(LOG_ERR, "Failed to run uncore discovery");
                }
            }
            else
            {
                //testDevice = 0;
                syslog(LOG_NOTICE, "PCI Uncore not supported on this system");
                goto LOOP;
            }
            if ((!pci_devices_daemon) && (!perfmon_discovery))
            {
                syslog(LOG_NOTICE, "PCI Uncore not supported on this system");
                goto LOOP;
            }
            if (isServerMem)
            {
                ret = servermem_init();
                if (ret < 0)
                {
                    syslog(LOG_ERR, "Failed to initialize Intel server memory support");
                }
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
                if (pci_devices_daemon && pci_devices_daemon[i].path && strlen(pci_devices_daemon[i].path) > 0)
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

        if (!isIntelUncoreDiscovery)
        {
            handle_record_default(&dRecord);
        }
        else
        {
            handle_record_spr(&dRecord);
        }

        LOG_AND_EXIT_IF_ERROR(write(connfd, (void*) &dRecord, sizeof(AccessDataRecord)), write failed);
    }

    /* never reached */
    return EXIT_SUCCESS;
}
