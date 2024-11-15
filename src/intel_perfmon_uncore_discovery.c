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

static PerfmonUncoreDiscovery* uncore_discovery_map = NULL;


/* Functions for the same handling of PCI devices and their memory in user-space as in kernel-space */

struct pci_dev {
    uint16_t domain;
    uint16_t bus;
    uint16_t device;
    uint16_t func;
    int numa_node;
};

// glibc memcpy uses optimized versions but copies have to be bytewise
// otherwise the data is incorrect. 
void mymemcpy(void* dest, void* src, int size)
{
    uint8_t volatile* udest = (uint8_t*)dest;
    uint8_t volatile* usrc = (uint8_t*)src;
    for (int i = 0; i < size/sizeof(uint8_t); i++)
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
    int ret = -1;
    char content[1025];
    int fp = open(path, O_RDONLY);
    if (fp >= 0)
    {
        ret = read(fp, content, 1024 * sizeof(char));
        if (ret >= 0) {
            content[ret] = '\0';
            u64 new = 0x0;
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

// Test whether PCI domain and bus exist.
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

// commonly only a few PCI domains are populated, so we get the max domain
// to return from PCI device parsing early. Otherwise, we have to search
// 65535 domains.
static int get_max_pci_domain()
{
    int max_dom = 0;
    DIR* dir = NULL;
    struct dirent* dent = NULL;

    dir = opendir("/sys/class/pci_bus");
    if (!dir)
    {
        return -errno;
    }
    while((dent = readdir(dir)))
    {
        int dom = -1, bus = -1;
        int c = sscanf(dent->d_name, "%X:%X", &dom, &bus);
        if ((c == 2) && (dom > max_dom))
        {
            max_dom = dom;
        }
    }

    closedir(dir);
    return max_dom+1;
}

// Get the maximal CPU socket ID to stop parsing. This does not work if
// sockets are not numbered consecutively. 
static int max_socket_id(int* max_socket)
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




static int _max_pci_domain_id = -1;
// This functions should mimic the behavior of pci_get_device in the kernel
// It searches the whole PCI space for a specific device and defined by vendor
// and device ID. Instead of vendor and device ID, the special PCI_ANY_ID can
// be used to get any device. In order to not start from zero each time, the
// function takes a device as input (from) to start from there.
struct pci_dev * pci_get_device (unsigned int vendor, unsigned int device, struct pci_dev * from)
{
    //struct pci_dev next = {from->domain, from->bus, from->device, from->func};
    int domStart = (from ? from->domain : 0);
    int busStart = (from ? from->bus : 0);
    int devStart = (from ? from->device : 0);
    int funcStart = (from ? from->func : 0);
    char busbase[1025];
    char devbase[1025];
    char fname[1025];
    
    if (_max_pci_domain_id < 0)
    {
        int ret = get_max_pci_domain();
        if (ret < 0)
        {
            return NULL;
        }
        _max_pci_domain_id = ret;
    }

    for (int dom = domStart; dom < _max_pci_domain_id; dom++) {
        // if we are beyond the domain given in from, we start with bus=0 again
        // instead of the bus given in from
        if (from && dom > from->domain) busStart = 0;
        for (int bus = busStart; bus < 0xff; bus++) {
            int ret = 0;
            // Early skip if PCI bus does not exist
            if (!pci_test_bus(dom, bus)) {
            
                continue;
            }
            // if we are beyond the bus given in from, we start with devid=0 again
            // instead of the device given in from
            if (from && bus > from->bus) devStart = 0;
            for (int devid = devStart; devid < 0xff; devid++) {
                // if we are beyond the device given in from, we start with func=0 again
                // instead of the func given in from
                if (from && devid > from->device) funcStart = 0;
                for (int func = funcStart; func < 0xf; func++) {
                    // Skip the 'from' device
                    if (from && dom == from->domain && bus == from->bus && devid == from->device && func ==  from->func)
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
                            next->domain = dom;
                            next->bus = bus;
                            next->device = devid;
                            next->func = func;
                            next->numa_node = node;
                            return next;
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
        DEBUG_PRINT(DEBUGLEV_DEVELOP, PCIIDX %d Access %s NumRegs %d ID %d Type %s(%d) box_ctl 0x%X ctrl_offset 0x%X ctr_offset 0x%X mmap_addr 0x%X mmap_offset 0x%X,
                idx, AccessTypeNames[unit->access_type], unit->num_regs, unit->box_id, name, unit->box_type,
                unit->box_ctl, unit->ctrl_offset, unit->ctr_offset, unit->mmap_addr, unit->mmap_offset);
    }
}

static int perfmon_uncore_discovery_update_dev_location(PerfmonDiscoveryUnit* unit)
{
    struct pci_dev* dev = NULL;
    uint32_t device = 0;
    uint32_t devfn = 0;
    switch (unit->box_type)
    {
        case SPR_DEVICE_ID_UPI:
            device = 0x3241;
            devfn = 0x9;
            break;
        case SPR_DEVICE_ID_M3UPI:
            device = 0x3246;
            devfn = 0x29;
            break;
        default:
            return 0;
    }

    while ((dev = pci_get_device(PCI_VENDOR_ID_INTEL, device, dev)) != NULL)
    {
        if (dev->numa_node < 0)
        {
            continue;
        }
        unit->box_ctl = dev->domain  << UNCORE_DISCOVERY_PCI_DOMAIN_OFFSET |
                        dev->bus << UNCORE_DISCOVERY_PCI_BUS_OFFSET |
                        devfn << UNCORE_DISCOVERY_PCI_DEVFN_OFFSET |
                        unit->box_ctl;
    }
    return 0;
}

int perfmon_uncore_discovery(int model, PerfmonDiscovery** perfmon)
{
    int ret = 0;
    PerfmonDiscoverySocket* socks = NULL;
    int socket_id = 0;
    struct pci_dev* dev = NULL;
    struct uncore_global_discovery global;
    int dvsec = 0;
    int PAGE_SIZE = sysconf (_SC_PAGESIZE);
    void * io_addr = NULL;
    int max_sockets = 0;
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
        default:
            ERROR_PRINT(Uncore discovery not supported for model 0x%X, model);
            return -1;
    }

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

    while (((dev = pci_get_device(PCI_VENDOR_ID_INTEL, PCI_ANY_ID, dev)) != NULL)  && (socket_id < max_sockets+1)){
        while ((dvsec = pci_find_next_ext_capability(dev, dvsec, UNCORE_EXT_CAP_ID_DISCOVERY))) {
            /* read the DVSEC_ID (15:0) */
            val = 0;
            ret = pci_read_config_dword(dev, dvsec + UNCORE_DISCOVERY_DVSEC_OFFSET, &val);
            if (ret < 0)
            {
                ERROR_PRINT(Failed to read DVSEC offset from device %.04x:%.02x:%.02x.%.01x, dev->domain, dev->bus, dev->device, dev->func);
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
                    ERROR_PRINT(Failed to read DIR from device %.04x:%.02x:%.02x.%.01x, dev->domain, dev->bus, dev->device, dev->func);
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
                    ERROR_PRINT(Failed to read BAR offset from device %.04x:%.02x:%.02x.%.01x, dev->domain, dev->bus, dev->device, dev->func);
                    continue;
                }
                /* get page boundary address of pci_dword */
                addr = pci_dword & ~(PAGE_SIZE - 1);
                if ((pci_dword & PCI_BASE_ADDRESS_MEM_TYPE_MASK) == PCI_BASE_ADDRESS_MEM_TYPE_64)
                {
                    DEBUG_PRINT(DEBUGLEV_DEVELOP, Mem type 64);
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
                    ERROR_PRINT(Failed to mmap device %.04x:%.02x:%.02x.%.01x, dev->domain, dev->bus, dev->device, dev->func);
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
                DEBUG_PRINT(DEBUGLEV_DEVELOP, Device %.04x:%.02x:%.02x.%.01x usable with %d units, dev->domain, dev->bus, dev->device, dev->func, global.max_units);

                PerfmonDiscoverySocket* tmp = realloc(perf->sockets, (socket_id + 1) * sizeof(PerfmonDiscoverySocket));
                if (!tmp)
                {
                    ERROR_PRINT(Cannot enlarge socket device table to %d, socket_id);
                    if (perf->sockets) free(perf->sockets);
                    free(perf);
                    close(pcihandle);
                    munmap(io_addr, UNCORE_DISCOVERY_MAP_SIZE);
                    return -ENOMEM;
                }
                perf->sockets = tmp;
                PerfmonDiscoverySocket* cur = &perf->sockets[socket_id];

                cur->socket_id = socket_id;
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
                socket_id++;
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
    if (dev) free(dev);
    close(pcihandle);
    perf->num_sockets = socket_id+1;
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
