/*
 * =======================================================================================
 *
 *      Filename:  intel_perfmon_uncore_discovery.h
 *
 *      Description:  Code to look up Uncore perfmon units on Intel SPR and later.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
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

#ifndef INTEL_PERFMON_UNCORE_DISCOVERY_H
#define INTEL_PERFMON_UNCORE_DISCOVERY_H

#include <error.h>
#include <registers.h>
#include <topology.h>
#include <pci_types.h>
#include <perfmon_types.h>

// Data structures provided by Intel
typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16;

#define PCI_EXT_CAP_BASE_OFFSET                 0x100
#define PCI_EXT_CAP_ID_MASK                     0xffff
#define PCI_EXT_CAP_NEXT_OFFSET                 0x2
#define PCI_EXT_CAP_NEXT_SHIFT                  4
#define PCI_EXT_CAP_NEXT_MASK                   0xfff
#define  PCI_BASE_ADDRESS_MEM_MASK              (~0x0fUL)
#define  PCI_BASE_ADDRESS_MEM_TYPE_64           0x04    /* 64 bit address */
#define  PCI_BASE_ADDRESS_MEM_TYPE_MASK         0x06


#define UNCORE_DISCOVERY_TABLE_DEVICE           0x09a7

/* Capability ID for discovery table device */
#define UNCORE_EXT_CAP_ID_DISCOVERY             0x23
/* First DVSEC offset */
#define UNCORE_DISCOVERY_DVSEC_OFFSET           0x8
/* Mask of the supported discovery entry type */
#define UNCORE_DISCOVERY_DVSEC_ID_MASK          0xffff
/* PMON discovery entry type ID */
#define UNCORE_DISCOVERY_DVSEC_ID_PMON          0x1
/* Second DVSEC offset */
#define UNCORE_DISCOVERY_DVSEC2_OFFSET          0xC
/* Mask of the discovery table BAR offset */
#define UNCORE_DISCOVERY_DVSEC2_BIR_MASK        0x7
/* Discovery table BAR base offset */
#define UNCORE_DISCOVERY_BIR_BASE               0x10
/* Discovery table BAR step */
#define UNCORE_DISCOVERY_BIR_STEP               0x4
/* Global discovery table size */
#define UNCORE_DISCOVERY_GLOBAL_MAP_SIZE        0x20
#define UNCORE_DISCOVERY_MAP_SIZE               0x80000

#define UNCORE_DISCOVERY_PCI_DOMAIN_OFFSET      28
#define UNCORE_DISCOVERY_PCI_DOMAIN(data)       ((data >> UNCORE_DISCOVERY_PCI_DOMAIN_OFFSET) & 0x7)
#define UNCORE_DISCOVERY_PCI_BUS_OFFSET         20
#define UNCORE_DISCOVERY_PCI_BUS(data)          ((data >> UNCORE_DISCOVERY_PCI_BUS_OFFSET) & 0xff)
#define UNCORE_DISCOVERY_PCI_DEVFN_OFFSET       12
#define UNCORE_DISCOVERY_PCI_DEVFN(data)        ((data >> UNCORE_DISCOVERY_PCI_DEVFN_OFFSET) & 0xff)

#define UNCORE_DISCOVERY_PCI_BOX_CTRL(data)     (data & 0xfff)

#define PCI_ANY_ID (-1)
#define PCI_VENDOR_ID_INTEL 0x8086
#define PCI_SLOT(devfn) (((devfn) >> 3) & 0x1f)
#define PCI_DEV_TO_DEVFN(dev) ((dev != NULL) ? (((dev)->device << 3) | (dev)->func) : 0)

#define uncore_discovery_invalid_unit(unit)                    \
       (!unit.table1 || \
        unit.table1 == -1ULL ||        \
        unit.table3 == -1ULL)


struct uncore_global_discovery {
    union {
        u64 table1;
        struct {
            u64 type : 8;
            u64 stride : 8;
            u64 max_units : 10;
            u64 __reserved_1 : 36;
            u64 access_type : 2;
        };
    };

    union {
        u64 table2;
        u64 global_ctl;
    };

    union {
        u64 table3;
        struct {
            u64 status_offset : 8;
            u64 num_status : 16;
            u64 __reserved_2 : 40;
        };
    };
};


struct uncore_unit_discovery {
    union {
        u64 table1;
        struct {
            u64 num_regs : 8;
            u64 ctl_offset : 8;
            u64 bit_width : 8;
            u64 ctr_offset : 8;
            u64 status_offset : 8;
            u64 __reserved_1 : 22;
            u64 access_type : 2;
        };
    };
    union {
        u64 table2;
        u64 box_ctl;
    };
    union {
        u64 table3;
        struct {
            u64 box_type : 16;
            u64 box_id : 16;
            u64 global_status_position : 16;
            u64 __reserved_2 : 16;
        };
    };
};

typedef enum {
    ACCESS_TYPE_MSR = 0,
    ACCESS_TYPE_MMIO = 1,
    ACCESS_TYPE_PCI = 2,
    ACCESS_TYPE_MAX
} AccessTypes;

static const char* AccessTypeNames[ACCESS_TYPE_MAX] = {
    [ACCESS_TYPE_MSR] = "MSR",
    [ACCESS_TYPE_PCI] = "PCI",
    [ACCESS_TYPE_MMIO] = "MMIO",
};

// Data structures used by LIKWID

typedef enum {
    FAKE_UNC_UNIT_CTRL = (1ULL<<31),
    FAKE_UNC_UNIT_STATUS,
    FAKE_UNC_UNIT_RESET,
    FAKE_UNC_CTRL0,
    FAKE_UNC_CTRL1,
    FAKE_UNC_CTRL2,
    FAKE_UNC_CTRL3,
    FAKE_UNC_CTRL4,
    FAKE_UNC_CTRL5,
    FAKE_UNC_CTR0,
    FAKE_UNC_CTR1,
    FAKE_UNC_CTR2,
    FAKE_UNC_CTR3,
    FAKE_UNC_CTR4,
    FAKE_UNC_CTR5,
    FAKE_UNC_CTR6,
    FAKE_UNC_CTR7,
    FAKE_UNC_CTR8,
    FAKE_UNC_CTR9,
    FAKE_UNC_CTR10,
    FAKE_UNC_CTR11,
    FAKE_UNC_CTR12,
    FAKE_UNC_CTR13,
    FAKE_UNC_CTR14,
    FAKE_UNC_CTR15,
    FAKE_UNC_FIXED_CTRL,
    FAKE_UNC_FIXED_CTR,
    FAKE_UNC_GLOBAL_CTRL,
    FAKE_UNC_GLOBAL_STATUS0,
    FAKE_UNC_GLOBAL_STATUS1,
    FAKE_UNC_GLOBAL_STATUS2,
    FAKE_UNC_GLOBAL_STATUS3,
    FAKE_UNC_GLOBAL_STATUS4,
    FAKE_UNC_GLOBAL_STATUS5,
    FAKE_UNC_GLOBAL_STATUS6,
    FAKE_UNC_GLOBAL_STATUS7,
    FAKE_UNC_GLOBAL_STATUS8,
    FAKE_UNC_FILTER0,
    FAKE_UNC_CTR_FREERUN0,
    FAKE_UNC_CTR_FREERUN1,
    FAKE_UNC_CTR_FREERUN2,
    FAKE_UNC_CTR_FREERUN3,
} LikwidFakePerfmonCounters;


typedef struct {
    int num_regs;
    uint64_t ctrl_offset;
    int bit_width;
    uint64_t ctr_offset;
    uint64_t status_offset;
    AccessTypes access_type;
    int box_type;
    uint64_t box_ctl;
    uint32_t box_id;
    uint64_t filter_offset;
    uint64_t fixed_ctrl_offset;
    uint64_t fixed_ctr_offset;
    uint64_t mmap_addr;
    uint64_t mmap_offset;
    size_t mmap_size;
    void volatile* io_addr;
} PerfmonDiscoveryUnit;

typedef struct {
    uint64_t global_ctl;
    uint64_t status_offset;
    int num_status;
    AccessTypes access_type;
    uint64_t mmap_addr;
    uint64_t mmap_offset;
    size_t mmap_size;
    void* io_addr;
} PerfmonDiscoveryGlobal;

typedef struct {
    int socket_id;
    PerfmonDiscoveryGlobal global;
    PerfmonDiscoveryUnit units[MAX_NUM_PCI_DEVICES];
} PerfmonDiscoverySocket;

typedef struct {
    int num_sockets;
    PerfmonDiscoverySocket* sockets;
} PerfmonDiscovery;

#define ACCESS_TYPE_ERROR(access_type) \
    do { \
        ERROR_PRINT("%s has invalid value: %d", (#access_type), (access_type)); \
        exit(EXIT_FAILURE); \
    } while (0)

int perfmon_uncore_discovery(int model, PerfmonDiscovery **perfmon);
void perfmon_uncore_discovery_free(PerfmonDiscovery* perfmon);


#endif /* INTEL_PERFMON_UNCORE_DISCOVERY_H */
