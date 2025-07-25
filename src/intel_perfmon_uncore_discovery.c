/*
 * =======================================================================================
 *
 *      Filename:  intel_perfmon_uncore_discovery.c
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


#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <string.h>
#include <dirent.h>

#include <likwid.h>
#include <intel_perfmon_uncore_discovery.h>
#include <perfmon_sapphirerapids_counters.h>
#include <perfmon_graniterapids_counters.h>
#include <perfmon_sierraforrest_counters.h>
#include <perfmon_emeraldrapids_counters.h>

static PerfmonUncoreDiscovery* uncore_discovery_map = NULL;


/* Functions for the same handling of PCI devices and their memory in user-space as in kernel-space */
#define MAX_FILENAME_LENGTH 1024

struct pci_dev {
    uint16_t domain;
    uint16_t bus;
    uint16_t device;
    uint16_t func;
    int numa_node;
    int socket;
    int vendor;
    int device_id;
    int index;
    char path[MAX_FILENAME_LENGTH];
};

static struct pci_dev* sysfs_pci_devices = NULL;
static int num_sysfs_pci_devices = 0;

// TODO remove all this ugly file handling here:
// - fix missing error handling
// - fix hardcoded constants at e.g. snprintf
// - do not check if a file exists with 'access'. Just open the file and handle the error accordingly
// - fix more unsigned/signed variables


// glibc memcpy uses optimized versions but copies have to be bytewise
// otherwise the data is incorrect. 
void mymemcpy(void* dest, void* src, size_t size)
{
    uint8_t volatile* udest = dest;
    uint8_t volatile* usrc = src;
    for (unsigned i = 0; i < size/sizeof(uint8_t); i++)
    {
        *(udest + i) = *(usrc + i);
    }
}

// Read 32bit from PCI device config space
static int pci_read_config_dword(const struct pci_dev *dev, int where, u32 *val)
{
    int ret = -1;
    char path[1025];
    ret = snprintf(path, 1024, "/sys/bus/pci/devices/%.04x:%.02x:%.02x.%.01x/config", dev->domain, dev->bus, dev->device, dev->func);
    if (ret > 0) {
        path[ret] = '\0';
        if (!access(path, R_OK))
        {
            int fp = open(path, O_RDONLY);
            if (fp > 0)
            {
                ret = pread(fp, val, sizeof(u32), where);
                close(fp);
            }
        }
    }
    return ret;
}

static int pci_find_next_ext_capability(struct pci_dev *dev, int start, int cap) {
    int ret = 0;
    int offset = PCI_EXT_CAP_BASE_OFFSET + start;
    u32 cap_data = 0x0;
    do
    {
        ret = pci_read_config_dword(dev, offset, &cap_data);
        if (ret > 0)
        {
            u16 cap_id = (cap_data & PCI_EXT_CAP_ID_MASK);
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
    // TODO use fopen/fgets here
    int ret = -1;
    char content[1025];
    int fp = open(path, O_RDONLY);
    if (fp >= 0)
    {
        ssize_t read_count = read(fp, content, 1024 * sizeof(char));
        if (read_count >= 0) {
            content[read_count] = '\0';
            u64 new = 0x0;
            int scan_count = sscanf(content, "0x%lx", &new);
            if (scan_count == 1)
            {
                *data = new;
                ret = 0;
            }
            else
            {
                scan_count = sscanf(content, "%ld", &new);
                if (scan_count == 1)
                {
                    *data = new;
                    ret = 0;
                }
            }
        }
        close(fp);
    }
    else
    {
        ret = -errno;
    }
    
    return ret;
}

static int file_to_nonneg_int(char *path, int *data) {
    unsigned uint_data = 0;
    int retval = file_to_uint(path, &uint_data);
    if (retval < 0)
        return retval;
    *data = (int)uint_data;
    return 0;
}

static int read_sysfs_pci_devs()
{
    DIR *dp = NULL;
    struct dirent *ep = NULL;

    if (sysfs_pci_devices == NULL)
    {
        dp = opendir("/sys/bus/pci/devices");
        if (!dp)
        {
            return -1;
        }
        while ((ep = readdir(dp)) != NULL)
        {
            int ret = 0;
            char devbase[MAX_FILENAME_LENGTH] = { [0 ... (MAX_FILENAME_LENGTH-1)] = '\0' };
            char filename[MAX_FILENAME_LENGTH] = { [0 ... (MAX_FILENAME_LENGTH-1)] = '\0' };
            uint32_t vendor = 0;
            uint32_t device_id = 0;
            int numa_node = -1;
            int socket = -1;
            int dom = 0;
            int bus = 0;
            int devid = 0;
            int func = 0;
            ret = sscanf(ep->d_name, "%04x:%02x:%02x.%01x", &dom, &bus, &devid, &func);
            if (ret != 4)
            {
                continue;
            }
            ret = snprintf(devbase, 1023, "/sys/bus/pci/devices/%s", ep->d_name);
            devbase[ret] = '\0';

            ret = snprintf(filename, 1023, "%s/vendor", devbase);
            filename[ret] = '\0';
            ret = file_to_uint(filename, &vendor);
            if (vendor != PCI_VENDOR_ID_INTEL)
            {
                continue;
            }
            ret = snprintf(filename, 1023, "%s/device", devbase);
            filename[ret] = '\0';
            ret = file_to_uint(filename, &device_id);
            ret = snprintf(filename, 1023, "%s/numa_node", devbase);
            filename[ret] = '\0';
            ret = file_to_nonneg_int(filename, &numa_node);

            if (numa_node >= 0)
            {
                int cpuid = -1;
                ret = snprintf(filename, 1023, "/sys/devices/system/node/node%d/cpulist", numa_node);
                ret = file_to_nonneg_int(filename, &cpuid);
                if (cpuid >= 0)
                {
                    ret = snprintf(filename, 1023, "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpuid);
                    ret = file_to_nonneg_int(filename, &socket);
                }
            }

            struct pci_dev* tmp = realloc(sysfs_pci_devices, (num_sysfs_pci_devices+1) * sizeof(struct pci_dev));
            if (!tmp)
            {
                break;
            }
            sysfs_pci_devices = tmp;

            struct pci_dev* p = &sysfs_pci_devices[num_sysfs_pci_devices];
            p->domain = dom;
            p->bus = bus;
            p->device = devid;
            p->func = func;
            p->numa_node = numa_node;
            p->socket = socket;
            p->vendor = vendor;
            p->device_id = device_id;
            p->index = num_sysfs_pci_devices;
            ret = snprintf(p->path, 1023, "%s", devbase);
            p->path[ret] = '\0';
            num_sysfs_pci_devices++;
        }

        closedir(dp);
    }
    return 0;
}


PciDeviceIndex get_likwid_device(int type, int id)
{
    int i = 0;
    while (uncore_discovery_map[i].discovery_type >= 0)
    {
        if (uncore_discovery_map[i].discovery_type == type)
        {
            if (id >= 0 && id < uncore_discovery_map[i].max_devices)
            {
                return uncore_discovery_map[i].base_device + id;
            }
        }
        i++;
    }
    return MAX_NUM_PCI_DEVICES;
}


static void print_unit(PciDeviceIndex idx, PerfmonDiscoveryUnit* unit)
{
    char* name = NULL;
    int i = 0;
    if (!uncore_discovery_map) return;

    while (uncore_discovery_map[i].discovery_type >= 0)
    {
        if (unit->box_type == uncore_discovery_map[i].discovery_type)
        {
            name = uncore_discovery_map[i].name;
            break;
        }
        i++;
    }
    if (name != NULL)
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, "PCIIDX %d Access %s NumRegs %d ID %d Type %s(%d) box_ctl 0x%lX ctrl_offset 0x%lX ctr_offset 0x%lX mmap_addr 0x%lX mmap_offset 0x%lX",
                idx, AccessTypeNames[unit->access_type], unit->num_regs, unit->box_id, name, unit->box_type,
                unit->box_ctl, unit->ctrl_offset, unit->ctr_offset, unit->mmap_addr, unit->mmap_offset);
    }
}

static int perfmon_uncore_discovery_update_dev_location(PerfmonDiscoveryUnit* unit)
{
    struct pci_dev* dev = NULL;
    uint32_t check_device = 0;
    uint32_t check_devfn = 0;
    uint16_t mydevfn = 0;
    uint16_t myid = 0;

    switch (unit->box_type)
    {
        case SPR_DEVICE_ID_UPI:
            check_device = 0x3241;
            check_devfn = 0x9;
            break;
        case SPR_DEVICE_ID_M3UPI:
            check_device = 0x3246;
            check_devfn = 0x29;
            break;
        default:
            return 0;
    }

    // TODO why do we need those variables?
    (void)check_device;

    for (int i = 0; i < num_sysfs_pci_devices; i++)
    {
        struct pci_dev* dev = &sysfs_pci_devices[i];
        if (dev->numa_node < 0)
        {
            continue;
        }
        mydevfn = PCI_DEV_TO_DEVFN(dev);
        myid = PCI_SLOT(mydevfn) - PCI_SLOT(check_devfn);
        if (myid == unit->box_id)
        {
            unit->box_ctl = dev->domain  << UNCORE_DISCOVERY_PCI_DOMAIN_OFFSET |
                        dev->bus << UNCORE_DISCOVERY_PCI_BUS_OFFSET |
                        mydevfn << UNCORE_DISCOVERY_PCI_DEVFN_OFFSET |
                        unit->box_ctl;
        }
    }
    if (dev) free(dev);
    return 0;
}

int perfmon_uncore_discovery(int model, PerfmonDiscovery** perfmon)
{
    int ret = 0;
    int num_sockets = 0;
    struct uncore_global_discovery global;
    int dvsec = 0;
    int PAGE_SIZE = sysconf (_SC_PAGESIZE);
    void * io_addr = NULL;
    u32 val = 0;
    u32 entry_id = 0;
    u32 bir;
    u32 bar_offset = 0, pci_dword = 0;
    u64 addr = 0;

    switch (model)
    {
        case SAPPHIRERAPIDS:
            uncore_discovery_map = sapphirerapids_uncore_discovery_map;
            break;
        case GRANITERAPIDS:
            uncore_discovery_map = graniterapids_uncore_discovery_map;
            break;
        case SIERRAFORREST:
            uncore_discovery_map = sierraforrest_uncore_discovery_map;
            break;
        case EMERALDRAPIDS:
            uncore_discovery_map = emeraldrapids_uncore_discovery_map;
            break;
        default:
            ERROR_PRINT("Uncore discovery not supported for model 0x%X", model);
            return -1;
    }

    ret = read_sysfs_pci_devs();
    if (ret < 0)
    {
        ERROR_PRINT("Failed to read PCI devices from sysfs");
        return -1;
        uncore_discovery_map = NULL;
    }

    int num_tmp = 0;
    int* tmp = malloc(num_sysfs_pci_devices * sizeof(int));
    if (!tmp)
    {
        free(sysfs_pci_devices);
        sysfs_pci_devices = NULL;
        num_sysfs_pci_devices = 0;
        uncore_discovery_map = NULL;
        return -1;
    }
    memset(tmp, -1, num_sysfs_pci_devices * sizeof(int));

    // Determine number of sockets based on the returned devices
    for (int i = 0; i < num_sysfs_pci_devices; i++)
    {
        struct pci_dev* dev = &sysfs_pci_devices[i];
        int found = 0;
        for (int j = 0; j < num_tmp; j++)
        {
            if (tmp[j] == dev->socket)
            {
                found = 1;
                break;
            }
        }
        if (!found)
        {
            tmp[num_tmp++] = dev->socket;
        }
    }
    num_sockets = num_tmp;
    free(tmp);

    /* Open memory (requires root permissions) */
    int pcihandle = open("/dev/mem", O_RDWR);
    if (pcihandle < 0)
    {
        ERROR_PRINT("Cannot open /dev/mem");
        free(sysfs_pci_devices);
        sysfs_pci_devices = NULL;
        num_sysfs_pci_devices = 0;
        uncore_discovery_map = NULL;
        return -errno;
    }
    PerfmonDiscovery* perf = malloc(sizeof(PerfmonDiscovery));
    if (!perf)
    {
        close(pcihandle);
        ERROR_PRINT("Cannot allocate space for device tables");
        free(sysfs_pci_devices);
        sysfs_pci_devices = NULL;
        num_sysfs_pci_devices = 0;
        uncore_discovery_map = NULL;
        return -ENOMEM;
    }
    perf->sockets = malloc(num_sockets * sizeof(PerfmonDiscoverySocket));
    if (!perf->sockets)
    {
        free(perf);
        close(pcihandle);
        ERROR_PRINT("Cannot allocate space for socket device tables");
        free(sysfs_pci_devices);
        sysfs_pci_devices = NULL;
        num_sysfs_pci_devices = 0;
        uncore_discovery_map = NULL;
        return -ENOMEM;
    }
    perf->num_sockets = 0;

    for (int i = 0; i < num_sysfs_pci_devices; i++)
    {
        struct pci_dev* dev = &sysfs_pci_devices[i];
        while ((dvsec = pci_find_next_ext_capability(dev, dvsec, UNCORE_EXT_CAP_ID_DISCOVERY))) {
            /* read the DVSEC_ID (15:0) */
            val = 0;
            ret = pci_read_config_dword(dev, dvsec + UNCORE_DISCOVERY_DVSEC_OFFSET, &val);
            if (ret < 0)
            {
                ERROR_PRINT("Failed to read DVSEC offset from device %.04x:%.02x:%.02x.%.01x", dev->domain, dev->bus, dev->device, dev->func);
                continue;
            }
            entry_id = val & UNCORE_DISCOVERY_DVSEC_ID_MASK;
            if (entry_id == UNCORE_DISCOVERY_DVSEC_ID_PMON)
            {
                //printf("Socket %d Dev %.04x:%.02x:%.02x.%.01x Node %d\n", socket_id, dev->domain, dev->bus, dev->device, dev->func, dev->numa_node);
                bir = 0;
                ret = pci_read_config_dword(dev, dvsec + UNCORE_DISCOVERY_DVSEC2_OFFSET, &bir);
                if (ret < 0)
                {
                    ERROR_PRINT("Failed to read DIR from device %.04x:%.02x:%.02x.%.01x", dev->domain, dev->bus, dev->device, dev->func);
                    continue;
                }
               /* read BIR value (2:0) */
                bir = bir & UNCORE_DISCOVERY_DVSEC2_BIR_MASK;
                /* calculate the BAR offset of global discovery table */
                bar_offset = 0x10 + (bir * 4);
                /* read the BAR address of global discovery table */
                ret = pci_read_config_dword(dev, bar_offset, &pci_dword);
                if (ret < 0)
                {
                    ERROR_PRINT("Failed to read BAR offset from device %.04x:%.02x:%.02x.%.01x", dev->domain, dev->bus, dev->device, dev->func);
                    continue;
                }
                /* get page boundary address of pci_dword */
                addr = pci_dword & ~(PAGE_SIZE - 1);
                if ((pci_dword & PCI_BASE_ADDRESS_MEM_TYPE_MASK) == PCI_BASE_ADDRESS_MEM_TYPE_64)
                {
                    DEBUG_PRINT(DEBUGLEV_DEVELOP, "Mem type 64");
                    uint32_t pci_dword2;
                    ret = pci_read_config_dword(dev, bar_offset + sizeof(uint32_t), &pci_dword2);
                    if (ret > 0) addr |= ((uint64_t)pci_dword2) << 32;
                }
                /* Map whole discovery table */
                /* User-space version of ioremap */
                //printf("MMap address 0x%lX\n", addr);
                io_addr = mmap(NULL, UNCORE_DISCOVERY_MAP_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, pcihandle, addr);
                if (io_addr == MAP_FAILED)
                {
                    ERROR_PRINT("Failed to mmap device %.04x:%.02x:%.02x.%.01x", dev->domain, dev->bus, dev->device, dev->func);
                    continue;
                }
                memset(&global, 0, sizeof(struct uncore_global_discovery));
                mymemcpy(&global, io_addr, sizeof(struct uncore_global_discovery));
                if (uncore_discovery_invalid_unit(global))
                {
                    munmap(io_addr, UNCORE_DISCOVERY_MAP_SIZE);
                    continue;
                }
                //printf("Global 1=0x%lX 2=0x%lX 3=0x%lX\n", global.table1, global.table2, global.table3);
                DEBUG_PRINT(DEBUGLEV_DEVELOP, "Device %.04x:%.02x:%.02x.%.01x usable with %d units", dev->domain, dev->bus, dev->device, dev->func, global.max_units);

                PerfmonDiscoverySocket* cur = &perf->sockets[dev->socket];

                cur->socket_id = dev->socket;
                memset(cur->units, 0, MAX_NUM_PCI_DEVICES*sizeof(PerfmonDiscoveryUnit));
                // record stuff from global struct in cur->global
                cur->global.global_ctl = global.global_ctl;
                cur->global.access_type = global.access_type;
                cur->global.status_offset = global.status_offset;
                cur->global.num_status = global.num_status;
                for (int i = 0; i < global.max_units; i++)
                {
                    struct uncore_unit_discovery unit;
                    if ((i + 1) * (global.stride * 8) > UNCORE_DISCOVERY_MAP_SIZE)
                    {
                        //ERROR_PRINT(Access to 0x%X outside of mapped memory, (i + 1) * (global.stride * 8));
                        continue;
                    }
                    mymemcpy(&unit, io_addr + (i + 1) * (global.stride * 8), sizeof(struct uncore_unit_discovery));
                    if ((uncore_discovery_invalid_unit(unit)) || (unit.num_regs == 0))
                    {
                        continue;
                    }
                    // record stuff from unit struct in cur->units[likwid-device-id]
                    PciDeviceIndex idx = get_likwid_device(unit.box_type, unit.box_id);
                    if (idx >= 0 && idx < MAX_NUM_PCI_DEVICES)
                    {
                        cur->units[idx].box_type = unit.box_type;
                        cur->units[idx].box_id = unit.box_id;
                        cur->units[idx].num_regs = unit.num_regs;
                        cur->units[idx].ctrl_offset = unit.ctl_offset;
                        cur->units[idx].bit_width = unit.bit_width;
                        cur->units[idx].ctr_offset = unit.ctr_offset;
                        cur->units[idx].status_offset = unit.status_offset;
                        cur->units[idx].access_type = (AccessTypes)unit.access_type;
                        cur->units[idx].box_ctl = unit.box_ctl;
                        cur->units[idx].filter_offset = 0x0;
                        cur->units[idx].fixed_ctrl_offset = 0x0;
                        cur->units[idx].fixed_ctr_offset = 0x0;
                        switch (model)
                        {
                            case SAPPHIRERAPIDS:
                            case EMERALDRAPIDS:
                            case GRANITERAPIDS:
                                if (unit.box_type == SPR_DEVICE_ID_CHA)
                                {
                                    cur->units[idx].filter_offset = 0xE;
                                }
                                else if (unit.box_type == SPR_DEVICE_ID_iMC || (unit.box_type == SPR_DEVICE_ID_HBM && model == SAPPHIRERAPIDS))
                                {
                                    cur->units[idx].fixed_ctrl_offset = 0x54;
                                    cur->units[idx].fixed_ctr_offset = 0x38;
                                }
                                break;
                            default:
                                break;
                        }

                        switch (model)
                        {
                            case SAPPHIRERAPIDS:
                            case EMERALDRAPIDS:
                            case GRANITERAPIDS:
                                perfmon_uncore_discovery_update_dev_location(&cur->units[idx]);
                                break;
                            default:
                                break;
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
                io_addr = NULL;
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
    }

    close(pcihandle);
    perf->num_sockets = num_sockets;
    free(sysfs_pci_devices);
    sysfs_pci_devices = NULL;
    num_sysfs_pci_devices = 0;
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
                    munmap((void*)perfmon->sockets[i].units[j].io_addr, perfmon->sockets[i].units[j].mmap_size);
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

