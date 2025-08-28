/*
 * =======================================================================================
 *
 *      Filename:  access_x86_translate.c
 *
 *      Description:  Interface to x86 related functions for the access module.
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

/* #####   HEADER FILE INCLUDES   ######################################### */

#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include <access.h>
#include <affinity.h>
#include <error.h>
#include <topology.h>
#include <types.h>

#include <access_x86_msr.h>
#include <access_x86_translate.h>
#include <affinity.h>
#include <intel_perfmon_uncore_discovery.h>

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

static PerfmonDiscovery *perfmon_discovery = NULL;
static int perfmon_discovery_inits         = 0;

static int access_x86_translate_open_unit(PerfmonDiscoveryUnit *unit)
{
    int err       = 0;
    int pcihandle = open("/dev/mem", O_RDWR);
    if (pcihandle < 0) {
        err = errno;
        ERROR_PRINT("Failed to open /dev/mem");
        return -err;
    }
    if (unit->access_type == ACCESS_TYPE_MMIO) {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, "Opening /dev/mem at 0x%lX", unit->mmap_addr);
        void *io_addr = mmap(
            NULL, unit->mmap_size, PROT_READ | PROT_WRITE, MAP_SHARED, pcihandle, unit->mmap_addr);
        if (io_addr == MAP_FAILED) {
            err = errno;
            close(pcihandle);
            ERROR_PRINT("Failed to mmap offset 0x%lX (MMIO)", unit->box_ctl);
            return -err;
        }
        DEBUG_PRINT(
            DEBUGLEV_DEVELOP, "Opening /dev/mem at 0x%lX -> 0x%p", unit->mmap_addr, io_addr);
        unit->io_addr = io_addr;
    } else if (unit->access_type == ACCESS_TYPE_PCI) {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, "Opening /dev/mem at 0x%lX", unit->mmap_addr);
        void *io_addr = mmap(
            NULL, unit->mmap_size, PROT_READ | PROT_WRITE, MAP_SHARED, pcihandle, unit->mmap_addr);
        if (io_addr == MAP_FAILED) {
            err = errno;
            close(pcihandle);
            ERROR_PRINT("Failed to mmap offset 0x%lX (PCI)", unit->box_ctl);
            return -err;
        }
        unit->io_addr = io_addr;
    }
    close(pcihandle);

    return 0;
}

static int access_x86_translate_get_ctl_register_offset(PerfmonDiscoveryUnit *unit)
{
    switch (unit->access_type) {
    case ACCESS_TYPE_MSR:
        return 1;
        break;
    case ACCESS_TYPE_MMIO:
    case ACCESS_TYPE_PCI:
        if (unit->bit_width <= 8) {
            return 1;
        } else if (unit->bit_width <= 16) {
            return 2;
        } else if (unit->bit_width <= 32) {
            return 4;
        } else if (unit->bit_width <= 64) {
            return 8;
        }
        break;
    default:
        ACCESS_TYPE_ERROR(unit->access_type);
    }
    return 0;
}

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

int access_x86_translate_init(uint32_t cpu_id)
{
    (void)cpu_id;

    if (!perfmon_discovery) {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, "Running Perfmon Discovery to populate counter lists");
        int ret = perfmon_uncore_discovery(cpuid_info.model, &perfmon_discovery);
        if (ret != 0) {
            ERROR_PRINT("Failed to run Perfmon Discovery");
            return ret;
        }
    }
    perfmon_discovery_inits++;
    return 0;
}

int access_x86_translate_read(PciDeviceIndex dev, uint32_t cpu_id, uint32_t reg, uint64_t *data)
{
    int err             = 0;
    uint64_t offset     = 0;
    uint64_t reg_offset = 0;
    if ((!perfmon_discovery) || (!data)) {
        return -EINVAL;
    }
    int socket_id = affinity_thread2socket_lookup[cpu_id];
    if (dev == MSR_UBOX_DEVICE) {
        PerfmonDiscoverySocket *cur = &perfmon_discovery->sockets[socket_id];
        if (cur->socket_id == socket_id) {
            uint64_t newreg = 0x0;
            if (reg == FAKE_UNC_GLOBAL_CTRL) {
                newreg = cur->global.global_ctl;
            } else if ((reg >= FAKE_UNC_GLOBAL_STATUS0) && (reg <= FAKE_UNC_GLOBAL_STATUS8)) {
                newreg = cur->global.global_ctl + cur->global.status_offset +
                         (reg - FAKE_UNC_GLOBAL_STATUS0);
            }
            uint64_t tmp = 0x0;
            DEBUG_PRINT(DEBUGLEV_DEVELOP,
                "Read Uncore counter 0x%lX (%s) on CPU %d (socket %d)",
                newreg,
                pci_device_names[dev],
                cpu_id,
                socket_id);
            err = access_x86_msr_read(cpu_id, newreg, &tmp);
            if (err == 0) {
                *data = tmp;
            }
        }
    } else {
        PerfmonDiscoverySocket *cur = &perfmon_discovery->sockets[socket_id];
        if (cur->socket_id == socket_id && cur->units[dev].num_regs > 0) {
            PerfmonDiscoveryUnit *unit = &cur->units[dev];
            if ((!unit->io_addr) && (unit->mmap_addr)) {
                int err = access_x86_translate_open_unit(unit);
                if (err < 0) {
                    ERROR_PRINT("Failed to open unit %s", pci_device_names[dev]);
                    return -ENODEV;
                }
            } else if (!unit->mmap_addr) {
                ERROR_PRINT("Failed to find unit %s", pci_device_names[dev]);
                return -ENODEV;
            }
            DEBUG_PRINT(DEBUGLEV_DEVELOP,
                "Read Uncore counter 0x%X (%s) on CPU %d (socket %d, access %s)",
                reg,
                pci_device_names[dev],
                cpu_id,
                socket_id,
                AccessTypeNames[unit->access_type]);
            switch (reg) {
            case FAKE_UNC_UNIT_CTRL:

                switch (unit->access_type) {
                case ACCESS_TYPE_MMIO:
                    *data = (uint64_t)*((uint64_t *)(unit->io_addr + unit->mmap_offset));
                    break;
                case ACCESS_TYPE_PCI:
                    if ((dev >= PCI_HA_DEVICE_0 && dev <= PCI_HA_DEVICE_31) ||
                        (dev >= PCI_R3QPI_DEVICE_LINK_0 && dev <= PCI_R3QPI_DEVICE_LINK_5) ||
                        (dev >= PCI_QPI_DEVICE_PORT_0 && dev <= PCI_QPI_DEVICE_PORT_5)) {
                        uint32_t lo = 0ULL, hi = 0ULL;
                        lo = (uint32_t)*((uint32_t *)(unit->io_addr + unit->mmap_offset));
                        hi = (uint32_t)*(
                            (uint32_t *)(unit->io_addr + unit->mmap_offset + sizeof(uint32_t)));
                        *data = (((uint64_t)hi) << 32) | ((uint64_t)lo);
                    } else {
                        *data = (uint64_t)*((uint64_t *)(unit->io_addr + unit->mmap_offset));
                    }
                    break;
                case ACCESS_TYPE_MSR:
                    err = access_x86_msr_read(cpu_id, unit->box_ctl, data);
                    break;
                default:
                    ACCESS_TYPE_ERROR(unit->access_type);
                }
                break;
            case FAKE_UNC_UNIT_STATUS:
                switch (unit->access_type) {
                case ACCESS_TYPE_MMIO:
                    *data = (uint64_t)*(
                        (uint64_t *)(unit->io_addr + unit->mmap_offset + unit->status_offset));
                    break;
                case ACCESS_TYPE_PCI:
                    if ((dev >= PCI_HA_DEVICE_0 && dev <= PCI_HA_DEVICE_31) ||
                        (dev >= PCI_R3QPI_DEVICE_LINK_0 && dev <= PCI_R3QPI_DEVICE_LINK_5) ||
                        (dev >= PCI_QPI_DEVICE_PORT_0 && dev <= PCI_QPI_DEVICE_PORT_5)) {
                        uint32_t lo = 0ULL, hi = 0ULL;
                        lo = (uint32_t)*(
                            (uint32_t *)(unit->io_addr + unit->mmap_offset + unit->status_offset));
                        hi    = (uint32_t)*((uint32_t *)(unit->io_addr + unit->mmap_offset +
                                                      unit->status_offset + sizeof(uint32_t)));
                        *data = (((uint64_t)hi) << 32) | ((uint64_t)lo);
                    } else {
                        *data = (uint64_t)*(
                            (uint64_t *)(unit->io_addr + unit->mmap_offset + unit->status_offset));
                    }
                    break;
                case ACCESS_TYPE_MSR:
                    err = access_x86_msr_read(cpu_id, unit->box_ctl + unit->status_offset, data);
                    break;
                default:
                    ACCESS_TYPE_ERROR(unit->access_type);
                }
                break;
            case FAKE_UNC_CTRL0:
            case FAKE_UNC_CTRL1:
            case FAKE_UNC_CTRL2:
            case FAKE_UNC_CTRL3:
                offset     = (reg - FAKE_UNC_CTRL0);
                reg_offset = access_x86_translate_get_ctl_register_offset(unit);

                switch (unit->access_type) {
                case ACCESS_TYPE_MMIO:
                    if ((dev >= MMIO_IMC_DEVICE_0_CH_0 && dev <= MMIO_IMC_DEVICE_1_CH_7) ||
                        (dev >= MMIO_HBM_DEVICE_0 && dev <= MMIO_HBM_DEVICE_31)) {
                        DEBUG_PRINT(DEBUGLEV_DEVELOP,
                            "0x%lX + 0x%lX + 0x%lX + (%zu * %lu)",
                            (uint64_t)unit->io_addr,
                            unit->mmap_offset,
                            unit->ctrl_offset,
                            sizeof(uint32_t),
                            offset);
                        *data = (uint32_t)*(
                            (uint32_t *)(unit->io_addr + unit->mmap_offset + unit->ctrl_offset +
                                         (sizeof(uint32_t) * offset)));
                    } else {
                        DEBUG_PRINT(DEBUGLEV_DEVELOP,
                            "0x%lX + 0x%lX + 0x%lX + (%lu * %lu)",
                            (uint64_t)unit->io_addr,
                            unit->mmap_offset,
                            unit->ctrl_offset,
                            reg_offset,
                            offset);
                        *data =
                            (uint64_t)*((uint64_t *)(unit->io_addr + unit->mmap_offset +
                                                     unit->ctrl_offset + (reg_offset * offset)));
                    }
                    break;
                case ACCESS_TYPE_PCI:
                    if ((dev >= PCI_HA_DEVICE_0 && dev <= PCI_HA_DEVICE_31) ||
                        (dev >= PCI_R3QPI_DEVICE_LINK_0 && dev <= PCI_R3QPI_DEVICE_LINK_5) ||
                        (dev >= PCI_QPI_DEVICE_PORT_0 && dev <= PCI_QPI_DEVICE_PORT_5)) {
                        uint32_t lo = 0ULL, hi = 0ULL;
                        lo = (uint32_t)*(
                            (uint32_t *)(unit->io_addr + unit->mmap_offset + unit->ctrl_offset +
                                         (sizeof(uint32_t) * offset)));
                        hi = (uint32_t)*(
                            (uint32_t *)(unit->io_addr + unit->mmap_offset + unit->ctrl_offset +
                                         (sizeof(uint32_t) * offset) + sizeof(uint32_t)));
                        *data = (((uint64_t)hi) << 32) | ((uint64_t)lo);
                    } else {
                        *data =
                            (uint64_t)*((uint64_t *)(unit->io_addr + unit->mmap_offset +
                                                     unit->ctrl_offset + (reg_offset * offset)));
                    }
                    break;
                case ACCESS_TYPE_MSR:
                    err = access_x86_msr_read(
                        cpu_id, unit->box_ctl + unit->ctrl_offset + (reg_offset * offset), data);
                    break;
                default:
                    ACCESS_TYPE_ERROR(unit->access_type);
                }
                break;
            case FAKE_UNC_CTR0:
            case FAKE_UNC_CTR1:
            case FAKE_UNC_CTR2:
            case FAKE_UNC_CTR3:
                offset     = (reg - FAKE_UNC_CTR0);
                reg_offset = access_x86_translate_get_ctl_register_offset(unit);

                switch (unit->access_type) {
                case ACCESS_TYPE_MMIO:
                    *data = (uint64_t)*((uint64_t *)(unit->io_addr + unit->mmap_offset +
                                                     unit->ctr_offset + (reg_offset * offset)));
                    break;
                case ACCESS_TYPE_PCI:
                    if ((dev >= PCI_HA_DEVICE_0 && dev <= PCI_HA_DEVICE_31) ||
                        (dev >= PCI_R3QPI_DEVICE_LINK_0 && dev <= PCI_R3QPI_DEVICE_LINK_5) ||
                        (dev >= PCI_QPI_DEVICE_PORT_0 && dev <= PCI_QPI_DEVICE_PORT_5)) {
                        uint32_t lo = 0ULL, hi = 0ULL;
                        lo = (uint32_t)*(
                            (uint32_t *)(unit->io_addr + unit->mmap_offset + unit->ctr_offset +
                                         (sizeof(uint32_t) * offset)));
                        hi = (uint32_t)*(
                            (uint32_t *)(unit->io_addr + unit->mmap_offset + unit->ctr_offset +
                                         (sizeof(uint32_t) * offset) + sizeof(uint32_t)));
                        *data = (((uint64_t)hi) << 32) | ((uint64_t)lo);
                    } else {
                        *data = (uint64_t)*((uint64_t *)(unit->io_addr + unit->mmap_offset +
                                                         unit->ctr_offset + (reg_offset * offset)));
                    }
                    break;
                case ACCESS_TYPE_MSR:
                    err = access_x86_msr_read(
                        cpu_id, unit->box_ctl + unit->ctr_offset + (reg_offset * offset), data);
                    break;
                default:
                    ACCESS_TYPE_ERROR(unit->access_type);
                }
                break;
            case FAKE_UNC_FILTER0:
                offset     = (reg - FAKE_UNC_FILTER0);
                reg_offset = unit->filter_offset;
                if (reg_offset != 0x0 && unit->access_type == ACCESS_TYPE_MSR) {
                    err = access_x86_msr_read(cpu_id, unit->box_ctl + unit->filter_offset, data);
                }
                break;
            case FAKE_UNC_FIXED_CTRL:
                if (unit->fixed_ctrl_offset != 0) {
                    if (unit->access_type == ACCESS_TYPE_MSR) {
                        err = access_x86_msr_read(
                            cpu_id, unit->box_ctl + unit->fixed_ctrl_offset, data);
                    } else if (unit->access_type == ACCESS_TYPE_MMIO) {
                        uint32_t lo = 0ULL, hi = 0ULL;
                        lo    = (uint32_t)*((uint32_t *)(unit->io_addr + unit->mmap_offset +
                                                      unit->fixed_ctrl_offset));
                        hi    = (uint32_t)*((uint32_t *)(unit->io_addr + unit->mmap_offset +
                                                      unit->fixed_ctrl_offset + sizeof(uint32_t)));
                        *data = (((uint64_t)hi) << 32) | ((uint64_t)lo);
                    }
                }
                break;
            case FAKE_UNC_FIXED_CTR:
                if (unit->fixed_ctr_offset != 0) {
                    if (unit->access_type == ACCESS_TYPE_MSR) {
                        err = access_x86_msr_read(
                            cpu_id, unit->box_ctl + unit->fixed_ctr_offset, data);
                    } else if (unit->access_type == ACCESS_TYPE_MMIO) {
                        *data = (uint64_t)*((uint64_t *)(unit->io_addr + unit->mmap_offset +
                                                         unit->fixed_ctr_offset));
                    }
                }
                break;
            }
        }
    }
    return 0;
}

int access_x86_translate_write(PciDeviceIndex dev, uint32_t cpu_id, uint32_t reg, uint64_t data)
{
    uint64_t offset     = 0;
    uint64_t reg_offset = 0;
    if (!perfmon_discovery) {
        return -EINVAL;
    }
    int socket_id = affinity_thread2socket_lookup[cpu_id];
    if (dev == MSR_UBOX_DEVICE) {
        PerfmonDiscoverySocket *cur = &perfmon_discovery->sockets[socket_id];
        if (cur->socket_id == socket_id && cur->global.global_ctl &&
            cur->global.access_type == ACCESS_TYPE_MSR) {
            uint64_t newreg = 0x0;
            if (reg == FAKE_UNC_GLOBAL_CTRL) {
                newreg = cur->global.global_ctl;
            } else if ((reg >= FAKE_UNC_GLOBAL_STATUS0) && (reg <= FAKE_UNC_GLOBAL_STATUS8)) {
                newreg = cur->global.global_ctl + cur->global.status_offset +
                         (reg - FAKE_UNC_GLOBAL_STATUS0);
            }
            if (newreg != 0x0) {
                DEBUG_PRINT(DEBUGLEV_DEVELOP,
                    "Write Uncore counter 0x%lX (%s) on CPU %d (socket %d): 0x%lX",
                    newreg,
                    pci_device_names[dev],
                    cpu_id,
                    socket_id,
                    data);
                int err = access_x86_msr_write(cpu_id, newreg, data);
                if (err < 0)
                    return err;
            }
        }

    } else {
        PerfmonDiscoverySocket *cur = &perfmon_discovery->sockets[socket_id];
        if (cur->socket_id == socket_id && cur->units[dev].num_regs > 0) {
            int err;
            PerfmonDiscoveryUnit *unit = &cur->units[dev];
            if ((!unit->io_addr) && (unit->mmap_addr)) {
                err = access_x86_translate_open_unit(unit);
                if (err < 0) {
                    ERROR_PRINT("Failed to open unit %s", pci_device_names[dev]);
                    return -ENODEV;
                }
            } else if (!unit->mmap_addr) {
                ERROR_PRINT("Failed to find unit %s", pci_device_names[dev]);
                return -ENODEV;
            }
            DEBUG_PRINT(DEBUGLEV_DEVELOP,
                "Write Uncore counter 0x%X (%s) on CPU %d (socket %d, access %s): 0x%lX",
                reg,
                pci_device_names[dev],
                cpu_id,
                socket_id,
                AccessTypeNames[unit->access_type],
                data);
            switch (reg) {
            case FAKE_UNC_UNIT_CTRL:

                switch (unit->access_type) {
                case ACCESS_TYPE_MMIO:
                    *((uint64_t *)(unit->io_addr + unit->mmap_offset)) = data;
                    break;
                case ACCESS_TYPE_PCI:
                    if ((dev >= PCI_HA_DEVICE_0 && dev <= PCI_HA_DEVICE_31) ||
                        (dev >= PCI_R3QPI_DEVICE_LINK_0 && dev <= PCI_R3QPI_DEVICE_LINK_5) ||
                        (dev >= PCI_QPI_DEVICE_PORT_0 && dev <= PCI_QPI_DEVICE_PORT_5)) {
                        *((uint32_t *)(unit->io_addr + unit->mmap_offset)) = (uint32_t)data;
                        *((uint32_t *)(unit->io_addr + unit->mmap_offset + sizeof(uint32_t))) =
                            (uint32_t)(data >> 32);
                    } else {
                        *((uint64_t *)(unit->io_addr + unit->mmap_offset)) = (uint64_t)data;
                    }
                    break;
                case ACCESS_TYPE_MSR:
                    err = access_x86_msr_write(cpu_id, unit->box_ctl, data);
                    if (err < 0)
                        return err;
                    break;
                default:
                    ACCESS_TYPE_ERROR(unit->access_type);
                }
                break;
            case FAKE_UNC_UNIT_STATUS:
                data = (uint64_t)*(
                    (uint64_t *)(unit->io_addr + unit->mmap_offset + unit->status_offset));
                switch (unit->access_type) {
                case ACCESS_TYPE_MMIO:
                    *((uint64_t *)(unit->io_addr + unit->mmap_offset + unit->status_offset)) = data;
                    break;
                case ACCESS_TYPE_PCI:
                    if ((dev >= PCI_HA_DEVICE_0 && dev <= PCI_HA_DEVICE_31) ||
                        (dev >= PCI_R3QPI_DEVICE_LINK_0 && dev <= PCI_R3QPI_DEVICE_LINK_5) ||
                        (dev >= PCI_QPI_DEVICE_PORT_0 && dev <= PCI_QPI_DEVICE_PORT_5)) {
                        *((uint32_t *)(unit->io_addr + unit->mmap_offset + unit->status_offset)) =
                            (uint32_t)data;
                        *((uint32_t *)(unit->io_addr + unit->mmap_offset + unit->status_offset +
                                       sizeof(uint32_t))) = (uint32_t)(data >> 32);
                    } else {
                        *((uint64_t *)(unit->io_addr + unit->mmap_offset + unit->status_offset)) =
                            (uint64_t)data;
                    }
                    break;
                case ACCESS_TYPE_MSR:
                    err = access_x86_msr_write(cpu_id, unit->box_ctl + unit->status_offset, data);
                    break;
                default:
                    ACCESS_TYPE_ERROR(unit->access_type);
                }
                break;
            case FAKE_UNC_CTRL0:
            case FAKE_UNC_CTRL1:
            case FAKE_UNC_CTRL2:
            case FAKE_UNC_CTRL3:
                offset     = (reg - FAKE_UNC_CTRL0);
                reg_offset = access_x86_translate_get_ctl_register_offset(unit);

                switch (unit->access_type) {
                case ACCESS_TYPE_MMIO:
                    if ((dev >= MMIO_IMC_DEVICE_0_CH_0 && dev <= MMIO_IMC_DEVICE_1_CH_7) ||
                        (dev >= MMIO_HBM_DEVICE_0 && dev <= MMIO_HBM_DEVICE_31)) {
                        DEBUG_PRINT(DEBUGLEV_DEVELOP,
                            "0x%lX + 0x%lX + 0x%lX + (%zu * %lu) = 0x%X",
                            (uint64_t)unit->io_addr,
                            unit->mmap_offset,
                            unit->ctrl_offset,
                            sizeof(uint32_t),
                            offset,
                            (uint32_t)data);
                        *((uint32_t *)(unit->io_addr + unit->mmap_offset + unit->ctrl_offset +
                                       (sizeof(uint32_t) * offset))) = (uint32_t)data;
                    } else {
                        DEBUG_PRINT(DEBUGLEV_DEVELOP,
                            "0x%lX + 0x%lX + 0x%lX + (%lu * %lu) = 0x%lX",
                            (uint64_t)unit->io_addr,
                            unit->mmap_offset,
                            unit->ctrl_offset,
                            reg_offset,
                            offset,
                            (uint64_t)data);
                        *((uint64_t *)(unit->io_addr + unit->mmap_offset + unit->ctrl_offset +
                                       (reg_offset * offset))) = (uint64_t)data;
                    }
                    break;
                case ACCESS_TYPE_PCI:
                    if ((dev >= PCI_HA_DEVICE_0 && dev <= PCI_HA_DEVICE_31) ||
                        (dev >= PCI_R3QPI_DEVICE_LINK_0 && dev <= PCI_R3QPI_DEVICE_LINK_5) ||
                        (dev >= PCI_QPI_DEVICE_PORT_0 && dev <= PCI_QPI_DEVICE_PORT_5)) {
                        *((uint32_t *)(unit->io_addr + unit->mmap_offset + unit->ctrl_offset +
                                       (sizeof(uint32_t) * offset))) = (uint32_t)data;
                        *((uint32_t *)(unit->io_addr + unit->mmap_offset + unit->ctrl_offset +
                                       (sizeof(uint32_t) * offset) + sizeof(uint32_t))) =
                            (uint32_t)(data >> 32);
                    } else {
                        *((uint64_t *)(unit->io_addr + unit->mmap_offset + unit->ctrl_offset +
                                       (reg_offset * offset))) = (uint64_t)data;
                    }
                    break;
                case ACCESS_TYPE_MSR:
                    err = access_x86_msr_write(
                        cpu_id, unit->box_ctl + unit->ctrl_offset + (reg_offset * offset), data);
                    break;
                default:
                    ACCESS_TYPE_ERROR(unit->access_type);
                }
                break;
            case FAKE_UNC_CTR0:
            case FAKE_UNC_CTR1:
            case FAKE_UNC_CTR2:
            case FAKE_UNC_CTR3:
                offset     = (reg - FAKE_UNC_CTR0);
                reg_offset = access_x86_translate_get_ctl_register_offset(unit);
                switch (unit->access_type) {
                case ACCESS_TYPE_MMIO:
                    *((uint64_t *)(unit->io_addr + unit->mmap_offset + unit->ctr_offset +
                                   (reg_offset * offset))) = data;
                    break;
                case ACCESS_TYPE_PCI:
                    if ((dev >= PCI_HA_DEVICE_0 && dev <= PCI_HA_DEVICE_31) ||
                        (dev >= PCI_R3QPI_DEVICE_LINK_0 && dev <= PCI_R3QPI_DEVICE_LINK_5) ||
                        (dev >= PCI_QPI_DEVICE_PORT_0 && dev <= PCI_QPI_DEVICE_PORT_5)) {
                        *((uint32_t *)(unit->io_addr + unit->mmap_offset + unit->ctr_offset +
                                       (sizeof(uint32_t) * offset))) = (uint32_t)data;
                        *((uint32_t *)(unit->io_addr + unit->mmap_offset + unit->ctr_offset +
                                       (sizeof(uint32_t) * offset) + sizeof(uint32_t))) =
                            (uint32_t)(data >> 32);
                    } else {
                        *((uint64_t *)(unit->io_addr + unit->mmap_offset + unit->ctr_offset +
                                       (reg_offset * offset))) = (uint64_t)data;
                    }
                    break;
                case ACCESS_TYPE_MSR:
                    err = access_x86_msr_write(
                        cpu_id, unit->box_ctl + unit->ctr_offset + (reg_offset * offset), data);
                    break;
                default:
                    ACCESS_TYPE_ERROR(unit->access_type);
                }
                break;
            case FAKE_UNC_FILTER0:
                offset = (reg - FAKE_UNC_FILTER0);
                if (unit->filter_offset != 0x0 && unit->access_type == ACCESS_TYPE_MSR) {
                    err = access_x86_msr_write(cpu_id, unit->box_ctl + unit->filter_offset, data);
                }
                break;
            case FAKE_UNC_FIXED_CTRL:
                if (unit->fixed_ctrl_offset != 0) {
                    if (unit->access_type == ACCESS_TYPE_MSR) {
                        err = access_x86_msr_write(
                            cpu_id, unit->box_ctl + unit->fixed_ctrl_offset, data);
                    } else if (unit->access_type == ACCESS_TYPE_MMIO) {
                        *((uint32_t *)(unit->io_addr + unit->mmap_offset +
                                       unit->fixed_ctrl_offset)) = (uint32_t)data;
                        *((uint32_t *)(unit->io_addr + unit->mmap_offset + unit->fixed_ctrl_offset +
                                       sizeof(uint32_t)))        = (uint32_t)(data >> 32);
                    }
                }
                break;
            case FAKE_UNC_FIXED_CTR:
                if (unit->fixed_ctr_offset != 0) {
                    if (unit->access_type == ACCESS_TYPE_MSR) {
                        err = access_x86_msr_write(
                            cpu_id, unit->box_ctl + unit->fixed_ctr_offset, data);
                    } else if (unit->access_type == ACCESS_TYPE_MMIO) {
                        *((uint64_t *)(unit->io_addr + unit->mmap_offset +
                                       unit->fixed_ctr_offset)) = data;
                    }
                }
                break;
            }
        }
    }
    return 0;
}

int access_x86_translate_finalize(uint32_t cpu_id)
{
    (void)cpu_id;

    perfmon_discovery_inits--;
    if (perfmon_discovery_inits == 0 && perfmon_discovery != NULL) {
        perfmon_uncore_discovery_free(perfmon_discovery);
        perfmon_discovery = NULL;
    }
    return 0;
}

int access_x86_translate_check(PciDeviceIndex dev, uint32_t cpu_id)
{
    if (!perfmon_discovery) {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, "CPU < 0 or no perfmon_initialization");
        return 0;
    }
    int socket_id = affinity_thread2socket_lookup[cpu_id];
    if (socket_id < 0) {
        return 0;
    }
    PerfmonDiscoverySocket *cur = &perfmon_discovery->sockets[socket_id];
    if (cur) {
        if (cur->socket_id == socket_id && cur->units[dev].num_regs > 0) {
            return 1;
        }
        if (dev == MSR_UBOX_DEVICE) {
            return 1;
        }
    }
    return 0;
}
