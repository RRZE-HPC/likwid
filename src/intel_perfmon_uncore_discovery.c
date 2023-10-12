#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <string.h>


#include <error.h>
#include <pci_types.h>
#include <intel_perfmon_uncore_discovery.h>


/* Functions for the same handling of PCI devices and their memory in user-space as in kernel-space */

struct pci_dev {
    uint16_t domain;
    uint16_t bus;
    uint16_t device;
    uint16_t func;
    int numa_node;
};

static int pci_test_bus(int domain, int bus)
{
    char fname[1025];
    int ret = snprintf(fname, 1024, "/sys/class/pci_bus/%.04x:%.02x", domain, bus);
    if (ret > 0)
    {
        fname[ret] = '\0';
    }
    else
    {
        ERROR_PRINT(Cannot format path to PCI bus directory for domain %d and bus %d, domain, bus);
    }
    return !access(fname, R_OK);
}

static int pci_read_config_dword(const struct pci_dev *dev, int where, uint32_t *val)
{
    int ret = -1;
    char path[1025];
    ret = snprintf(path, 1024, "/sys/bus/pci/devices/%.04x:%.02x:%.02x.%.01x/config", dev->domain, dev->bus, dev->device, dev->func);
    if (ret > 0) {
        path[ret] = '\0';
        ret = -1;
        if (!access(path, R_OK))
        {
            //printf("%s\n", path);
            int fp = open(path, O_RDONLY);
            if (fp > 0)
            {
                ret = pread(fp, val, sizeof(uint32_t), where);
                close(fp);
            }
        }
    }
    return ret;
}

static int pci_find_next_ext_capability(struct pci_dev *dev, int start, int cap) {
    int ret = 0;
    int offset = PCI_EXT_CAP_BASE_OFFSET + start;
    uint32_t cap_data = 0x0;
    do
    {
        ret = pci_read_config_dword(dev, offset, &cap_data);
        if (ret > 0)
        {
            uint16_t cap_id = (cap_data & PCI_EXT_CAP_ID_MASK);
            if (cap_id == cap) break;
            ret = pci_read_config_dword(dev, offset + PCI_EXT_CAP_NEXT_OFFSET, &cap_data);
            offset = (ret > 0 ? (cap_data >> PCI_EXT_CAP_NEXT_SHIFT & PCI_EXT_CAP_NEXT_MASK) : 0);
        } else {
            offset = 0;
        }
    } while (offset > 0);
    return offset;
}

/* Helper function to read files from sysfs */
static int file_to_uint(char* path, unsigned int *data) {
    int ret = -1;
    char content[1025];
    int fp = open(path, O_RDONLY);
    if (fp >= 0)
    {
        ret = read(fp, content, 1024 * sizeof(char));
        if (ret >= 0) {
            content[ret] = '\0';
            unsigned int new = 0x0;
            ret = sscanf(content, "0x%x", &new);
            if (ret == 1)
            {
                *data = new;
                ret = 0;
            }
            else
            {
                ret = sscanf(content, "%d", &new);
                if (ret == 1)
                {
                    *data = new;
                    ret = 0;
                }
            }
        }
        close(fp);
    }
    
    return ret;
}


struct pci_dev * pci_get_device(unsigned int vendor, unsigned int device, struct pci_dev * from)
{
    int domStart = (from ? from->domain : 0);
    int busStart = (from ? from->bus : 0);
    int devStart = (from ? from->device : 0);
    int funcStart = (from ? from->func : 0);
    char busbase[1025];
    char devbase[1025];
    char fname[1025];
    if (from)
    {
        domStart = from->domain;
        busStart = from->bus;
        devStart = from->device;
        funcStart = from->func;
    }
    for (int dom = domStart; dom < 0xffff; dom++) {
        for (int bus = busStart; bus < 0xff; bus++) {
            int ret = 0;
            // Early skip if PCI bus does not exist
            if (!pci_test_bus(dom, bus)) {
                continue;
            }
            for (int devid = devStart; devid < 0xff; devid++) {
                for (int func = funcStart; func < 0xf; func++) {
                    // Skip the 'from' device
                    if (from && dom == domStart && bus == busStart && devid == devStart && func == funcStart)
                    {
                        continue;
                    }
                    // Create directory path to PCI device
                    int ret = snprintf(devbase, 1024, "/sys/bus/pci/devices/%.04x:%.02x:%.02x.%.01x", dom, bus, devid, func);
                    if (ret > 0)
                    {
                        devbase[ret] = '\0';
                    }
                    else
                    {
                        printf("Cannot create PCI device path\n");
                        continue;
                    }
                    // If the device path does not exist, go to next one
                    if (access(devbase, R_OK))
                    {
                        continue;
                    }
                    if (vendor != PCI_ANY_ID)
                    {
                        unsigned int hVend = 0x0;
                        int ret = snprintf(fname, 1024, "%s/vendor", devbase);
                        if (ret > 0)
                        {
                            fname[ret] = '\0';
                            ret = file_to_uint(fname, &hVend);
                            if (ret == 0 && hVend != vendor)
                            {
                                continue;
                            }
                        }
                        else
                        {
                            continue;
                        }
                    }
                    if (device != PCI_ANY_ID)
                    {
                        unsigned int hDev = 0x0;
                        int ret = snprintf(fname, 1024, "%s/device", devbase);
                        if (ret > 0)
                        {
                            fname[ret] = '\0';
                            ret = file_to_uint(fname, &hDev);
                            if (ret == 0 && hDev != device)
                            {
                                continue;
                            }
                        }
                        else
                        {
                            continue;
                        }
                    }
                    // We directly read the NUMA node ID if present
                    int node = -1;
                    ret = snprintf(fname, 1024, "%s/numa_node", devbase);
                    if (ret > 0)
                    {
                        fname[ret] = '\0';
                        ret = file_to_uint(fname, &node);
                    }
                    else
                    {
                        continue;
                    }
                    // Reuse space
                    if (from)
                    {
                       DEBUG_PRINT(DEBUGLEV_DEVELOP, Using discovery entry device %s, devbase);
                        from->domain = dom;
                        from->bus = bus;
                        from->device = devid;
                        from->func = func;
                        from->numa_node = node;
                        return from;
                    }
                    else
                    {
                        // First find, allocate a new PCI device
                        struct pci_dev* next = malloc(sizeof(struct pci_dev));
                        if (next) {
                           DEBUG_PRINT(DEBUGLEV_DEVELOP, Using discovery entry device %s, devbase);
                            next->domain = dom;
                            next->bus = bus;
                            next->device = devid;
                            next->func = func;
                            next->numa_node = node;
                            return next;
                        }
                       else
                       {
                           ERROR_PRINT(Failed to allocate space for PCI device info);
                       }
                    }
                }
            }
        }
    }
    return NULL;
}

PciDeviceIndex get_likwid_device(int type, int id)
{
    //printf("get_likwid_device(%d, %d) = ", type, id);
    switch (type)
    {
        case DEVICE_ID_CHA:
            return MSR_CBOX_DEVICE_C0 + id;
            break;
        case DEVICE_ID_iMC:
            if (id < 0 || id > 15)
            {
                ERROR_PRINT(Cannot transform IMC device with ID %d, id);
            }
           else
           {
                return MMIO_IMC_DEVICE_0_CH_0 + id;
           }
            break;
        case DEVICE_ID_M2PCIe:
            if (id < 0 || id > 11)
            {
                ERROR_PRINT(Cannot transform M2PCIe device with ID %d, id);
            }
           else
            {
                return PCI_R2PCIE_DEVICE0 + id;
           }
            break;
        case DEVICE_ID_PCU:
            if (id < 0 || id > 1)
            {
                ERROR_PRINT(Cannot transform PCU device with ID %d, id);
            }
           else
            {
                return MSR_PCU_DEVICE;
           }
            break;
        case DEVICE_ID_IRP:
            if (id >= 0 && id <= 12)
            {
                return MSR_IRP_DEVICE_0 + id;
            }
            else
            {
                ERROR_PRINT(Cannot transform IRP device with ID %d, id);
            }
            break;
        case DEVICE_ID_IIO:
            if (id >= 0 && id <= 12)
            {
                return MSR_IIO_DEVICE_0 + id;
            }
            else
            {
                ERROR_PRINT(Cannot transform IIO device with ID %d, id);
            }
            break;
        case DEVICE_ID_UPI:
            if (id >= 0 && id <= 3)
            {
                return PCI_QPI_DEVICE_PORT_0 + id;
            }
            else
            {
                ERROR_PRINT(Cannot transform UPI device with ID %d, id);
            }
            break;
        case DEVICE_ID_MDF:
            if (id < 0 || id > 49)
            {
                ERROR_PRINT(Cannot transform MDF device with ID %d, id);
            }
           else
            {
                return MSR_MDF_DEVICE_0 + id;
           }
            break;
        case DEVICE_ID_M2M:
            if (id < 0 || id > 31)
            {
                ERROR_PRINT(Cannot transform M2M device with ID %d, id);
                return MAX_NUM_PCI_DEVICES;
            }
           else
            {
                return PCI_HA_DEVICE_0 + id;
           }
            break;
        case DEVICE_ID_M3UPI:
            if (id < 0 || id > 3)
            {
                ERROR_PRINT(Cannot transform M3UPI device with ID %d, id);
            }
           else
            {
                return PCI_R3QPI_DEVICE_LINK_0 + id;
           }
            break;
        case DEVICE_ID_HBM:
            if (id < 0 || id > 31)
            {
                ERROR_PRINT(Cannot transform HBM device with ID %d, id);
            }
           else
            {
                return MMIO_HBM_DEVICE_0 + id;
           }
            break;
        default:
            return MAX_NUM_PCI_DEVICES;
            break;
    }
}


void print_unit(PciDeviceIndex idx, PerfmonDiscoveryUnit* unit)
{
    DEBUG_PRINT(DEBUGLEV_DEVELOP, PCIIDX %d Access %s NumRegs %d ID %d Type %s(%d) box_ctl 0x%X ctrl_offset 0x%X ctr_offset 0x%X mmap_addr 0x%X mmap_offset 0x%X, idx, AccessTypeNames[unit->access_type], unit->num_regs, unit->box_id, uncore_discovery_box_type_names[unit->box_type], unit->box_type, unit->box_ctl, unit->ctrl_offset, unit->ctr_offset, unit->mmap_addr, unit->mmap_offset);
}

int max_socket_id(int* max_socket)
{
    int num_procs = sysconf (_SC_NPROCESSORS_CONF);
    int cur_procs = 0;
    int ret = -1;
    int cur_max = 0;
    char path[1025];
    for (int i = 0; i < 100000 && cur_procs < num_procs; i++)
    {
        ret = snprintf(path, 1024, "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", i);
        if (ret >= 0)
        {
            path[ret] = '\0';
            int id = 0;
            ret = file_to_uint(path, &id);
            if (ret == 0 && id > cur_max)
            {
                cur_max = id;
            }
        }
    }
    DEBUG_PRINT(DEBUGLEV_DEVELOP, Found max socket ID %d, cur_max);
    *max_socket = cur_max;
    return 0;
}

int perfmon_uncore_discovery(PerfmonDiscovery** perfmon)
{
    int num = 0;
    PerfmonDiscoverySocket* socks = NULL;
    int socket_id = -1;
    struct pci_dev* dev = NULL;
    struct uncore_global_discovery global;
    int dvsec = 0;
    int PAGE_SIZE = sysconf (_SC_PAGESIZE);
    void* io_addr = NULL;
    int max_sockets = 0;

    if (max_socket_id(&max_sockets) < 0)
    {
       ERROR_PRINT(Failed to determine number of sockets);
        return -1;
    }

    /* Open memory (requires root permissions) */
    int pcihandle = open("/dev/mem", O_RDWR);
    if (pcihandle < 0)
    {
       ERROR_PRINT(Cannot open /dev/mem);
        return -errno;
    }
    PerfmonDiscovery* perf = malloc(sizeof(PerfmonDiscovery));
    if (!perf)
    {
        close(pcihandle);
       ERROR_PRINT(Cannot allocate space for device tables);
        return -ENOMEM;
    }
    perf->sockets = NULL;
    perf->num_sockets = 0;

    while ((dev = pci_get_device(0x8086, UNCORE_DISCOVERY_TABLE_DEVICE, dev)) != NULL) {
        socket_id++;
        if (socket_id > max_sockets) {
           ERROR_PRINT(Socket_id too large);
            break;
        }
        PerfmonDiscoverySocket* tmp = realloc(perf->sockets, (num + 1) * sizeof(PerfmonDiscoverySocket));
        if (!tmp)
        {
           ERROR_PRINT(Cannot enlarge socket device table to %d, num);
            if (perf->sockets) free(perf->sockets);
            free(perf);
            close(pcihandle);
            return -ENOMEM;
        }
        perf->sockets = tmp;
        PerfmonDiscoverySocket* cur = &perf->sockets[num];

        cur->socket_id = socket_id;
        memset(cur->units, 0, MAX_NUM_PCI_DEVICES*sizeof(PerfmonDiscoveryUnit));

        while ((dvsec = pci_find_next_ext_capability(dev, dvsec, UNCORE_EXT_CAP_ID_DISCOVERY))) {
            /* read the DVSEC_ID (15:0) */
            uint32_t val = 0;
            pci_read_config_dword(dev, dvsec + UNCORE_DISCOVERY_DVSEC_OFFSET, &val);
            uint32_t entry_id = val & UNCORE_DISCOVERY_DVSEC_ID_MASK;
            if (entry_id == UNCORE_DISCOVERY_DVSEC_ID_PMON) {
                uint32_t bir = 0;
                pci_read_config_dword(dev, dvsec + UNCORE_DISCOVERY_DVSEC2_OFFSET, &bir);
                /* read BIR value (2:0) */
                bir = bir & UNCORE_DISCOVERY_DVSEC2_BIR_MASK;
                /* calculate the BAR offset of global discovery table */
                uint32_t bar_offset = 0x10 + (bir * 4);
                /* read the BAR address of global discovery table */
                uint32_t pci_dword = 0;
                pci_read_config_dword(dev, bar_offset, &pci_dword);
                /* get page boundary address of pci_dword */
                uint64_t addr = pci_dword & ~(PAGE_SIZE - 1);
                /* Map whole discovery table */
                /* User-space version of ioremap */
                io_addr = mmap(NULL, UNCORE_DISCOVERY_MAP_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, pcihandle, addr);
                if (io_addr == MAP_FAILED)
                {
                    continue;
                }
                memcpy(&global, io_addr, sizeof(struct uncore_global_discovery));
                
                // record stuff from global struct in cur->global
                cur->global.global_ctl = global.global_ctl;
                cur->global.access_type = global.access_type;
                cur->global.status_offset = global.status_offset;
                cur->global.num_status = global.num_status;
                
                for (int i = 0; i < global.max_units; i++)
                {
                    struct uncore_unit_discovery unit;
                    if ((i + 1) * (global.stride * 8) > UNCORE_DISCOVERY_MAP_SIZE) continue;
                    memcpy(&unit, io_addr + (i + 1) * (global.stride * 8), sizeof(struct uncore_unit_discovery));
                    // record stuff from unit struct in cur->units[likwid-device-id]
                    if (unit.num_regs == 0) continue;
                    PciDeviceIndex idx = get_likwid_device(unit.box_type, unit.box_id);
                    if (idx >= 0 && idx < MAX_NUM_PCI_DEVICES)
                    {
                        cur->units[idx].box_type = (uncore_discovery_box_types) unit.box_type;
                        cur->units[idx].box_id = unit.box_id;
                        cur->units[idx].num_regs = unit.num_regs;
                        cur->units[idx].ctrl_offset = unit.ctl_offset;
                        cur->units[idx].bit_width = unit.bit_width;
                        cur->units[idx].ctr_offset = unit.ctr_offset;
                        cur->units[idx].status_offset = unit.status_offset;
                        cur->units[idx].access_type = (AccessTypes)unit.access_type;
/*                        if (cur->units[idx].access_type == ACCESS_TYPE_PCI)*/
/*                        {*/
/*                            // Workaround: When running the perfmon discovery in kernel-space, the*/
/*                            // mapped PCI devices differ to the user-space discovery by bit 27 only*/
/*                            // for the PCI based Uncore devices.*/
/*                            unit.box_ctl |= (1<<27);*/
/*                        }*/
                        cur->units[idx].box_ctl = unit.box_ctl;
                        cur->units[idx].filter_offset = 0x0;
                        cur->units[idx].fixed_ctrl_offset = 0x0;
                        cur->units[idx].fixed_ctr_offset = 0x0;
                        if (unit.box_type == DEVICE_ID_CHA)
                        {
                            cur->units[idx].filter_offset = 0xE;
                        }
                        else if (unit.box_type == DEVICE_ID_iMC || unit.box_type == DEVICE_ID_HBM)
                        {
                            cur->units[idx].fixed_ctrl_offset = 0x54;
                            cur->units[idx].fixed_ctr_offset = 0x38;
                        }

                        cur->units[idx].mmap_addr = cur->units[idx].box_ctl & ~(PAGE_SIZE - 1);
                        cur->units[idx].mmap_offset = cur->units[idx].box_ctl - cur->units[idx].mmap_addr;
                        cur->units[idx].mmap_size = PAGE_SIZE;
                        cur->units[idx].io_addr = NULL;
                        print_unit(idx, &cur->units[idx]);
                    }
                }
                /* Unmap PCI config space */
                munmap(io_addr, UNCORE_DISCOVERY_MAP_SIZE);
            }
           else
            {
                   DEBUG_PRINT(DEBUGLEV_DEVELOP, not the right dvsec);
           }
        }
/*        printf("Add M2IOSF devices \n");*/
/*        for (int i = 0; i < 8; i++)*/
/*        {*/
/*            PciDeviceIndex idx = MSR_M2IOSF_DEVICE_0 + i;*/
/*            printf("Device %d -> %d\n", i, idx);*/
/*            cur->units[idx].box_ctl = 0x3800 + (0x2 * i);*/
/*            printf("Device %d -> %d (0x%X)\n", i, idx, cur->units[idx].box_ctl);*/
/*            cur->units[idx].box_type = DEVICE_ID_IIO;*/
/*            cur->units[idx].box_id = i;*/
/*            cur->units[idx].num_regs = 8;*/
/*            cur->units[idx].ctr_offset = 0x0;*/
/*            cur->units[idx].ctrl_offset = 0x0;*/
/*            cur->units[idx].access_type = ACCESS_TYPE_MSR;*/
/*            cur->units[idx].bit_width = 48;*/
/*            cur->units[idx].filter_offset = 0x0;*/
/*            cur->units[idx].fixed_ctrl_offset = 0x0;*/
/*            cur->units[idx].fixed_ctr_offset = 0x0;*/
/*            print_unit(idx, &cur->units[idx]);*/
/*        }*/
        num++;
    }
    if (num == 0)
    {
         ERROR_PRINT(Failed to get any socket device tables);
        return -EFAULT;
    }
    close(pcihandle);
    perf->num_sockets = num;
    *perfmon = perf;
    return 0;
}


void perfmon_uncore_discovery_free(PerfmonDiscovery* perfmon)
{
    if (perfmon->sockets)
    {
        for (int i = 0; i < perfmon->num_sockets; i++)
        {
            for (int j = 0; j < MAX_NUM_PCI_DEVICES; j++)
            {
                if (perfmon->sockets[i].units[j].io_addr)
                {
                    munmap(perfmon->sockets[i].units[j].io_addr, perfmon->sockets[i].units[j].mmap_size);
                    perfmon->sockets[i].units[j].io_addr = NULL;
                }
            }
        }
        free(perfmon->sockets);
        perfmon->sockets = NULL;
        perfmon->num_sockets = 0;
    }
    free(perfmon);
}
