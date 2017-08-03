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
 *                Thomas Roehl (tr), thomas.roehl@googlemail.com
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
static int FD_MSR[MAX_NUM_THREADS];
static int FD_PCI[MAX_NUM_NODES][MAX_NUM_PCI_DEVICES];
static int bus_map[MAX_NUM_NODES];
static int isPCIUncore = 0;
static PciDevice* pci_devices_daemon = NULL;
static char pci_filepath[MAX_PATH_LENGTH];
static int num_pmc_counters = 0;

/* Socket to bus mapping -- will be determined at runtime;
 * typical mappings are:
 * Socket  Bus (2S)  Bus (4s)
 *   0        0xff      0x3f
 *   1        0x7f      0x7f
 *   2                  0xbf
 *   3                  0xff
 */
static char* socket_bus[MAX_NUM_NODES] = { [0 ... (MAX_NUM_NODES-1)] = NULL};

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
        syslog(LOG_ERR, "Testing 0x%X %d %d\n", reg, (reg & 0xF00U), (reg & 0xA00U));
        if (((reg & 0xF00U) == 0x700U) ||
            ((reg & 0xF00U) == 0xE00U) ||
            ((reg & 0xF00U) == 0xF00U) ||
            (reg == MSR_PREFETCH_ENABLE) ||
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
        syslog(LOG_ERR, "Failed to read data from register 0x%x on core %u", reg, cpu);
#ifdef DEBUG_LIKWID
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
        syslog(LOG_ERR, "Failed to write data to register 0x%x on core %u", reg, cpu);
#ifdef DEBUG_LIKWID
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

static int
getBusFromSocket(PciDevice* dev, int socket, char** filepath)
{
    int cur_bus = 0;
    uint32_t cur_socket = socket;
    char pci_filepath[1024];
    int fp;
    int ret = 0;
    while(socket >= 0)
    {
        while (cur_bus <= 255)
        {
            snprintf(pci_filepath, 1023, "/proc/bus/pci/%02x/%s", cur_bus, dev->path);
            fp = open(pci_filepath, O_RDONLY);
            if (fp < 0)
            {
                cur_bus++;
                continue;
            }
            if (fp > 0)
            {
                uint32_t indev = 0;
                ret = pread(fp, &indev, sizeof(uint32_t), 0x0);
                indev = (indev >> 16);
                if (indev != dev->devid)
                {
                    cur_bus++;
                    continue;
                }
            }
            if (fp > 0 && socket > 0)
            {
                cur_bus++;
                socket--;
                close(fp);
                continue;
            }
            
            break;
        }
        socket--;
    }
    if (fp > 0)
    {
        if (filepath)
            strncpy(*filepath, pci_filepath, strlen(pci_filepath));
        close(fp);
        return cur_socket;
    }
    return -1;
}


static void
pci_read(AccessDataRecord* dRecord)
{
    uint32_t socketId = dRecord->cpu;
    uint32_t reg = dRecord->reg;
    uint32_t device = dRecord->device;
    uint32_t data;
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
        //snprintf(pci_filepath, MAX_PATH_LENGTH-1, "%s%s%s", PCI_ROOT_PATH, socket_bus[socketId], pci_devices_daemon[device].path);
        pcipath = malloc(1024 * sizeof(char));
        memset(pcipath, '\0', 1024 * sizeof(char));
        int sid = getBusFromSocket(&(pci_devices_daemon[device]), socketId, &pcipath);
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
    if (pcipath)
        free(pcipath);
}

static void
pci_write(AccessDataRecord* dRecord)
{
    uint32_t socketId = dRecord->cpu;
    uint32_t reg = dRecord->reg;
    uint32_t device = dRecord->device;
    uint32_t data = (uint32_t) dRecord->data;
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
        //snprintf(pci_filepath, MAX_PATH_LENGTH-1, "%s%s%s", PCI_ROOT_PATH, socket_bus[socketId], pci_devices_daemon[device].path);
        pcipath = malloc(1024 * sizeof(char));
        memset(pcipath, '\0', 1024 * sizeof(char));
        int sid = getBusFromSocket(&(pci_devices_daemon[device]), socketId, &pcipath);
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

    if (FD_PCI[socketId][device] > 0 && pwrite(FD_PCI[socketId][device], &data, sizeof data, reg) != sizeof data)
    {
        syslog(LOG_ERR, "Failed to write data to pci device file %s for device %s (%s) on socket %u",pci_filepath,
                pci_types[pci_devices_daemon[device].type].name, pci_devices_daemon[device].name, socketId);
        dRecord->errorcode = ERR_RWFAIL;
        if (pcipath)
            free(pcipath);
        return;
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
    for (int i=0;i<MAX_NUM_NODES;i++)
    {
        if (socket_bus[i] != NULL)
        {
            free(socket_bus[i]);
        }
    }

    if (sockfd != -1)
    {
        CHECK_ERROR(close(sockfd), socket close sockfd failed);
    }

    free(filepath);
    closelog();
    exit(EXIT_SUCCESS);
}

static int
getBusFromSocketOld(const uint32_t socket, const char* name, const uint32_t in_dev_id)
{
    int cur_bus = 0;
    uint32_t cur_socket = 0;
    char pci_filepath[1024];
    int fp;
    int ret = 0;
    while(cur_socket <= socket)
    {
        snprintf(pci_filepath, MAX_PATH_LENGTH-1, "%s%02x/%s", PCI_ROOT_PATH, cur_bus, name);
        fp = open(pci_filepath, O_RDONLY);
        if (fp < 0)
        {
            return -1;
        }
        uint32_t cpubusno = 0;
        ret = pread(fp, &cpubusno, sizeof(uint32_t), 0x108);
        if (ret != sizeof(uint32_t))
        {
            close(fp);
            return -1;
        }
        cur_bus = (cpubusno >> 8) & 0x0ff;
        close(fp);
        if(socket == cur_socket)
            return cur_bus;
        ++cur_socket;
        ++cur_bus;
        if(cur_bus > 0x0ff)
           return -1;
    }

    return -1;
}

#define SKYLAKE_SERVER_SOCKETID_MBOX_DID 0x2042
static int getBusFromSocketSysFS(const uint32_t socket, const char* name, const uint32_t in_dev_id)
{
    struct dirent *pDirent, *pDirentInner;
    DIR *pDir, *pDirInner;
    pDir = opendir ("/sys/devices");
    FILE* fp = NULL;
    char iPath[200], iiPath[200], buff[100];
    char testDev[50];
    size_t ret = 0;
    int bus_id = -1;
    if (pDir == NULL)
    {
        syslog(LOG_ERR, "Failed open directory /sys/devices");
        return -1;
    }
    
    while ((pDirent = readdir(pDir)) != NULL)
    {
        if (strncmp(pDirent->d_name, "pci0", 4) == 0)
        {
            sprintf(iPath, "/sys/devices/%s", pDirent->d_name);
            char bus[4];
            strncpy(bus, &(pDirent->d_name[strlen(pDirent->d_name)-2]), 2);
            bus[2] = '\0';
            
            pDirInner = opendir (iPath);
            if (pDir == NULL)
            {
                syslog(LOG_ERR, "Failed read file %s", iPath);
                return -1;
            }
            while ((pDirentInner = readdir(pDirInner)) != NULL)
            {
                if (strncmp(pDirentInner->d_name, "0000", 4) == 0)
                {
                    uint32_t dev_id = 0x0;
                    int numa_node = 0;
                    sprintf(iiPath, "/sys/devices/%s/%s/device", pDirent->d_name, pDirentInner->d_name);
                    fp = fopen(iiPath,"r");
                    if( fp == NULL )
                    {
                        continue;
                    }
                    ret = fread(buff, sizeof(char), 99, fp);
                    dev_id = strtoul(buff, NULL, 16);
                    if (dev_id == in_dev_id)
                    {
                        fclose(fp);
                        iiPath[0] = '\0';
                        sprintf(iiPath, "/sys/devices/%s/%s/numa_node", pDirent->d_name, pDirentInner->d_name);
                        fp = fopen(iiPath,"r");
                        if( fp == NULL )
                        {
                            continue;
                        }
                        ret = fread(buff, sizeof(char), 99, fp);
                        numa_node = atoi(buff);
                        if (numa_node == socket)
                        {
                            bus_id = strtoul(bus, NULL, 16);
                        }
                    }
                    fclose(fp);
                    iiPath[0] = '\0';
                    buff[0] = '\0';
                    if (bus_id != -1)
                        break;
                }
            }
            closedir (pDirInner);
            iPath[0] = '\0';
            if (bus_id != -1)
                break;
        }
    }
    closedir (pDir);
    return bus_id;
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
    for (int i=0;i<MAX_NUM_THREADS;i++)
    {
        FD_MSR[i] = -1;
    }

    openlog(ident, 0, LOG_USER);

    if (!lock_check())
    {
        syslog(LOG_ERR,"Access to performance counters is locked.\n");
        stop_daemon();
    }

    daemonize(&pid);
    syslog(LOG_INFO, "AccessDaemon runs with UID %d, eUID %d\n", getuid(), geteuid());

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
                else if (model == XEON_PHI_KNL)
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
        for ( uint32_t i=0; i < numHWThreads; i++ )
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
            else if (model == XEON_PHI_KNL)
            {
                pci_devices_daemon = knl_pci_devices;
            }
            else
            {
                //testDevice = 0;
                syslog(LOG_NOTICE, "PCI Uncore not supported on this system");
                goto LOOP;
            }

            for (int j=0; j<MAX_NUM_NODES; j++)
            {
                //socket_bus[j] = (char*)malloc(4);
                //sprintf(socket_bus[j], "N-A");
                for (int i=0; i<MAX_NUM_PCI_DEVICES; i++)
                {
                    FD_PCI[j][i] = -2;
                }
            }
            
            for (int i=1; i<MAX_NUM_PCI_DEVICES; i++)
            {
                if (pci_devices_daemon[i].path && strlen(pci_devices_daemon[i].path) > 0)
                {
                    int socket_id = getBusFromSocket(&(pci_devices_daemon[i]), 0, NULL);
                    if (socket_id == 0)
                    {
                        for (int j=0; j<MAX_NUM_NODES; j++)
                        {
                            FD_PCI[j][i] = 0;
                        }
                        pci_devices_daemon[i].online = 1;
                    }
                    else
                    {
                        syslog(LOG_ERR, "Device %s for socket %d, excluded it from device list: %s\n",pci_devices_daemon[i].name, 0, strerror(errno));
                    }
                }
            }

            /* determine PCI-BUSID mapping ... */
            /*syslog(LOG_ERR, "Determine PCI-BUSID mapping");
            int sbus = -1;
            cntr = 0;
            sbus = getSocketBus(cntr);
            syslog(LOG_ERR, "Socket %d Bus %02x", cntr, sbus);
            while (sbus != -1 && sbus < 0xFF)
            {
                sprintf(socket_bus[cntr], "%02x/", sbus);
                cntr++;
                sbus = getSocketBus(cntr);
                syslog(LOG_ERR, "Socket %d Bus %02x", cntr, sbus);
            }

            if ( cntr == 0 )
            {
                syslog(LOG_NOTICE, "Uncore not supported on this system");
            }
            else
            {
                socket_count = cntr;
                int fd;
                for (int j=0; j<socket_count; j++)
                {
                    for (int i=1; i<MAX_NUM_PCI_DEVICES; i++)
                    {
                        if (pci_devices_daemon[i].path)
                        {
                            sprintf(pci_filepath, "%s%s%s", PCI_ROOT_PATH, socket_bus[j], pci_devices_daemon[i].path);
                            fd = open(pci_filepath, O_RDWR);
                            if (fd > 0)
                            {
                                FD_PCI[j][i] = 0;
                                pci_devices_daemon[i].online = 1;
                                close(fd);
                            }
#ifdef DEBUG_LIKWID
                            else if (j==0)
                            {
                                syslog(LOG_ERR, "Device %s for socket %d not found at path %s, excluded it from device list: %s\n",pci_devices_daemon[i].name,j, pci_filepath, strerror(errno));
                            }
#endif
                        }
                    }
                }
            }*/
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
            syslog(LOG_ERR, "ERROR - [%s:%d] zero read", __FILE__, __LINE__);
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

