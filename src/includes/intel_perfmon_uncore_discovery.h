#ifndef INTEL_PERFMON_UNCORE_DISCOVERY_H
#define INTEL_PERFMON_UNCORE_DISCOVERY_H

// Data structures provided by Intel

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


#define UNCORE_DISCOVERY_PCI_BOX_CTRL(data)	(data & 0xfff)

#define PCI_ANY_ID (-1)

#define uncore_discovery_invalid_unit(unit)                    \
       (!unit.table1 || \
        unit.table1 == -1ULL ||        \
        unit.table3 == -1ULL)


struct uncore_global_discovery {
    union {
        uint64_t table1;
        struct {
            uint64_t type : 8;
            uint64_t stride : 8;
            uint64_t max_units : 10;
            uint64_t __reserved_1 : 36;
            uint64_t access_type : 2;
        };
    };

    union {
        uint64_t table2;
        uint64_t global_ctl;
    };

    union {
        uint64_t table3;
        struct {
            uint64_t status_offset : 8;
            uint64_t num_status : 16;
            uint64_t __reserved_2 : 40;
        };
    };
};


struct uncore_unit_discovery {
    union {
        uint64_t table1;
        struct {
            uint64_t num_regs : 8;
            uint64_t ctl_offset : 8;
            uint64_t bit_width : 8;
            uint64_t ctr_offset : 8;
            uint64_t status_offset : 8;
            uint64_t __reserved_1 : 22;
            uint64_t access_type : 2;
        };
    };
    union {
        uint64_t table2;
        uint64_t box_ctl;
    };
    union {
        uint64_t table3;
        struct {
            uint64_t box_type : 16;
            uint64_t box_id : 16;
            uint64_t __reserved_2 : 32;
        };
    };
};

typedef enum {
    DEVICE_ID_CHA = 0,
    DEVICE_ID_IIO = 1,
    DEVICE_ID_IRP = 2,
    DEVICE_ID_M2PCIe = 3,
    DEVICE_ID_PCU = 4,
    DEVICE_ID_5_UNKNOWN = 5,
    DEVICE_ID_iMC = 6,
    DEVICE_ID_M2M = 7,
    DEVICE_ID_UPI = 8,
    DEVICE_ID_M3UPI = 9,
    DEVICE_ID_10_UNKNOWN = 10,
    DEVICE_ID_MDF = 11,
    DEVICE_ID_12_UNKNOWN = 12,
    DEVICE_ID_13_UNKNOWN = 13,
    DEVICE_ID_HBM = 14,
    DEVICE_ID_MAX,
} uncore_discovery_box_types;

static char* uncore_discovery_box_type_names[DEVICE_ID_MAX] = {
    [DEVICE_ID_CHA] = "CBOX",
    [DEVICE_ID_IIO] = "IIO",
    [DEVICE_ID_IRP] = "IRP",
    [DEVICE_ID_M2PCIe] = "PBOX",
    [DEVICE_ID_PCU] = "WBOX",
    [DEVICE_ID_5_UNKNOWN] = "UNKNOWN",
    [DEVICE_ID_iMC] = "MBOX",
    [DEVICE_ID_M2M] = "M2M",
    [DEVICE_ID_UPI] = "QBOX",
    [DEVICE_ID_M3UPI] = "RBOX",
    [DEVICE_ID_10_UNKNOWN] = "UNKNOWN",
    [DEVICE_ID_MDF] = "MDF",
    [DEVICE_ID_12_UNKNOWN] = "UNKNOWN",
    [DEVICE_ID_13_UNKNOWN] = "UNKNOWN",
    [DEVICE_ID_HBM] = "HBM",
};

typedef enum {
    ACCESS_TYPE_MSR = 0,
    ACCESS_TYPE_MMIO = 1,
    ACCESS_TYPE_PCI = 2,
    ACCESS_TYPE_MAX
} AccessTypes;

static char* AccessTypeNames[ACCESS_TYPE_MAX] = {
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
    uncore_discovery_box_types box_type;
    uint64_t box_ctl;
    uint32_t box_id;
    uint64_t filter_offset;
    uint64_t fixed_ctrl_offset;
    uint64_t fixed_ctr_offset;
    uint64_t mmap_addr;
    uint64_t mmap_offset;
    size_t mmap_size;
    void* io_addr;
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


int perfmon_uncore_discovery(PerfmonDiscovery **perfmon);
void perfmon_uncore_discovery_free(PerfmonDiscovery* perfmon);


#endif /* INTEL_PERFMON_UNCORE_DISCOVERY_H */
