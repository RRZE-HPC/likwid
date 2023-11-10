/*
 * =======================================================================================
 *
 *      Filename:  access_x86_mmio.c
 *
 *      Description:  Implementation of pci module.
 *                   Provides API to read and write values to the hardware
 *                   performance monitoring registers in PCI Cfg space
 *                   for Intel Sandy Bridge Processors.
 *
 *      Version:   4.3.1
 *      Released:  04.01.2018
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com,
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
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/mman.h>


#include <types.h>
#include <bstrlib.h>
#include <error.h>
#include <topology.h>

#include <access_x86_mmio.h>


/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */


#define PCM_CLIENT_IMC_BAR_OFFSET       (0x0048)
#define PCM_CLIENT_IMC_DRAM_IO_REQUESTS  (0x5048)
#define PCM_CLIENT_IMC_DRAM_DATA_READS  (0x5050)
#define PCM_CLIENT_IMC_DRAM_DATA_WRITES (0x5054)
#define PCM_CLIENT_IMC_PP0_TEMP (0x597C)
#define PCM_CLIENT_IMC_PP1_TEMP (0x5980)
#define PCM_CLIENT_IMC_MMAP_SIZE (0x6000)

#define ICX_IMC_NUM_DEVICES 4
#define ICX_IMC_NUM_CHANNELS 2
#define ICX_IMC_DEVICE 0x3451
#define ICX_IMC_MMIO_BASE_OFFSET 0xd0
#define ICX_IMC_MMIO_BASE_MASK 0x1FFFFFFF
#define ICX_IMC_MMIO_BASE_SHIFT 23
/* MEM0_BAR found at Bus U0, Device 0, Function 1, offset D8h. */
#define ICX_IMC_MMIO_MEM0_OFFSET 0xd8
#define ICX_IMC_MMIO_MEM_STRIDE 0x4
#define ICX_IMC_MMIO_MEM_MASK 0x7FF
#define ICX_IMC_MMIO_MEM_SHIFT 12
/*
* Each IMC has two channels.
* The offset starts from 0x22800 with stride 0x8000
*/
#define ICX_IMC_MMIO_CHN_OFFSET 0x22800
#define ICX_IMC_MMIO_CHN_STRIDE 0x4000//0x8000

#define ICX_IMC_MMIO_FREERUN_OFFSET 0x2290

/* IMC MMIO size*/
#define ICX_IMC_MMIO_SIZE 0x4000

#define PCI_ENABLE                         0x80000000
#define FORM_PCI_ADDR(bus,dev,fun,off)     (((PCI_ENABLE))          |   \
                                            ((bus & 0xFF) << 16)    |   \
                                            ((dev & 0x1F) << 11)    |   \
                                            ((fun & 0x07) <<  8)    |   \
                                            ((off & 0xFF) <<  0))

typedef struct {
    uint32_t deviceId;
    off_t base_offset;
    off_t base_mask;
    off_t base_shift;
    int device_count;
    off_t device_offset;
    off_t device_stride;
    off_t device_mask;
    off_t device_shift;
    int channel_count;
    off_t channel_offset;
    off_t channel_stride;
    off_t mmap_size;
    off_t freerun_offset;
} MMIOConfig;

static MMIOConfig mmio_icelakeX = {
    .deviceId = ICX_IMC_DEVICE,
    .base_offset = ICX_IMC_MMIO_BASE_OFFSET,
    .base_mask = ICX_IMC_MMIO_BASE_MASK,
    .base_shift = ICX_IMC_MMIO_BASE_SHIFT,
    .device_count = ICX_IMC_NUM_DEVICES,
    .device_offset = ICX_IMC_MMIO_MEM0_OFFSET,
    .device_stride = ICX_IMC_MMIO_MEM_STRIDE,
    .device_mask = ICX_IMC_MMIO_MEM_MASK,
    .device_shift = ICX_IMC_MMIO_MEM_SHIFT,
    .channel_count = ICX_IMC_NUM_CHANNELS,
    .channel_offset = ICX_IMC_MMIO_CHN_OFFSET,
    .channel_stride = ICX_IMC_MMIO_CHN_STRIDE,
    .mmap_size = ICX_IMC_MMIO_SIZE,
    .freerun_offset = ICX_IMC_MMIO_FREERUN_OFFSET,
};



typedef struct {
    int fd;
    uint64_t addr;
    void* mmap_addr;
    uint32_t reg_offset;
} MMIOBoxHandle;

typedef struct {
    uint32_t pci_bus;
    uint64_t base_addr;
    int num_boxes;
    MMIOBoxHandle* boxes;
    int num_freerun;
    MMIOBoxHandle* freerun;
} MMIOSocketBoxes;

typedef struct {
    int num_sockets;
    MMIOSocketBoxes sockets;
} MMIOSockets;


/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

static int access_mmio_initialized[MAX_NUM_NODES] = {0};

static MMIOConfig* mmio_config = NULL;
static int num_mmio_sockets = 0;
static MMIOSocketBoxes* mmio_sockets = NULL;

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

static int mmio_validDevice(uint32_t pci_bus, uint32_t deviceId)
{
    uint32_t pci_dev = 0;
    char buf[20];
    bstring bdevfile = bformat("/sys/bus/pci/devices/0000:%.2x:00.1/device", pci_bus);
    FILE* fd = fopen(bdata(bdevfile), "r");
    if (fd < 0)
    {
        ERROR_PRINT(Cannot get device id: failed to open %s, bdata(bdevfile));
        bdestroy(bdevfile);
        return 0;
    }
    
    int ret = fread(buf, sizeof(char), 20, fd);
    if (ret < 0)
    {
        ERROR_PRINT(Cannot get device id: failed to read %s, bdata(bdevfile));
        fclose(fd);
        bdestroy(bdevfile);
        return 0;
    }
    fclose(fd);
    pci_dev = strtoul(buf, NULL, 16);
    if (pci_dev != deviceId)
    {
        ERROR_PRINT(Cannot get device id: device ids do not match 0x%X and 0x%X, pci_dev, deviceId);
        fclose(fd);
        bdestroy(bdevfile);
        return 0;
    }
    bdestroy(bdevfile);
    return 1;
}

static int
mmio_fillBox(MMIOConfig* config, uint32_t pci_bus, int imc_idx, MMIOBoxHandle* handle)
{
    //uint32_t pci_bus = get_pci_bus_of_socket(pkg_id);
    uint32_t pci_dev = 0;
    
    uint32_t tmp = 0;
    off_t addr = 0;
    off_t mem_offset = 0;

    if (!mmio_validDevice(pci_bus, config->deviceId))
    {
        return -1;
    }
    
    bstring bdevmem = bformat("/sys/bus/pci/devices/0000:%.2x:00.1/config", pci_bus);

    int pcihandle = open(bdata(bdevmem), O_RDONLY);
    if (pcihandle < 0)
    {
        ERROR_PRINT(Cannot get start address: failed to open %s, bdata(bdevmem));
        bdestroy(bdevmem);
        return -1;
    }
    int ret = pread(pcihandle, &tmp, sizeof(uint32_t), config->base_offset);
    if (ret < 0 || ret != sizeof(uint32_t))
    {
        ERROR_PRINT(Cannot get start address: read failed);
        close(pcihandle);
        bdestroy(bdevmem);
        return -1;
    }
    if (!tmp)
    {
        ERROR_PRINT(Cannot get address: MMIO base is zero);
        close(pcihandle);
        bdestroy(bdevmem);
        return -1;
    }
    addr = (tmp & config->base_mask) << config->base_shift;
    //DEBUG_PRINT(DEBUGLEV_DEVELOP, IMC %d BASE 0x%lX = (0x%lX & 0x%lX) << %d, imc_idx, addr, tmp, config->base_mask, config->base_shift);
    tmp = 0;
    mem_offset = config->device_offset + (imc_idx / config->channel_count) * config->device_stride;
    //DEBUG_PRINT(DEBUGLEV_DEVELOP, IMC %d offset 0x%X, imc_idx, mem_offset);
    ret = pread(pcihandle, &tmp, sizeof(uint32_t), mem_offset);
    if (ret < 0)
    {
        ERROR_PRINT(Cannot get start address of device: read failed);
        close(pcihandle);
        bdestroy(bdevmem);
        return -1;
    }
    addr |= (tmp & config->device_mask) << config->device_shift;
    addr += config->channel_offset + config->channel_stride * (imc_idx % config->channel_count);

    //DEBUG_PRINT(DEBUGLEV_DEVELOP, IMC %d IMC_OFF 0x%lX (0x%lX & 0x%lX) << %d, imc_idx, addr, tmp, config->device_mask, config->device_shift);
    close(pcihandle);

    pcihandle = open("/dev/mem", O_RDWR);
    if (pcihandle < 0)
    {
        ERROR_PRINT(Cannot get mmap address: failed to open /dev/mem);
        bdestroy(bdevmem);
        return -1;
    }
    //DEBUG_PRINT(DEBUGLEV_DEVELOP, IMC %d MMAP 0x%llX, imc_idx, addr);

    //DEBUG_PRINT(DEBUGLEV_DEVELOP, MMap size 0x%x addr %lld (0x%llX), ICX_IMC_MMIO_SIZE, addr & (~(4096 - 1)), addr & (~(4096 - 1)));
    void* maddr = mmap(NULL, config->channel_count*ICX_IMC_MMIO_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, pcihandle, addr & (~(4096 - 1)));
    if (maddr == MAP_FAILED)
    {
        ERROR_PRINT(Cannot get start address of device: mmap failed);
        bdestroy(bdevmem);
        close(pcihandle);
        return -1;
    }
    DEBUG_PRINT(DEBUGLEV_DEVELOP, IMC %d MMAP %p ADDR %lX, imc_idx, maddr, addr);
    handle->mmap_addr = maddr;
    handle->addr = addr;
    handle->fd = pcihandle;
    handle->reg_offset = addr - (addr & (~(4096 - 1)));

    bdestroy(bdevmem);

    return 0;
}

static int
mmio_fillFreerunBox(MMIOConfig* config, uint32_t pci_bus, int imc_idx, MMIOBoxHandle* handle)
{
    //uint32_t pci_bus = get_pci_bus_of_socket(pkg_id);
    uint32_t pci_dev = 0;
    
    uint32_t tmp = 0;
    off_t addr = 0;
    off_t mem_offset = 0;

    if (!mmio_validDevice(pci_bus, config->deviceId))
    {
        return -1;
    }
    
    bstring bdevmem = bformat("/sys/bus/pci/devices/0000:%.2x:00.1/config", pci_bus);

    int pcihandle = open(bdata(bdevmem), O_RDONLY);
    if (pcihandle < 0)
    {
        ERROR_PRINT(Cannot get start address: failed to open %s, bdata(bdevmem));
        bdestroy(bdevmem);
        return -1;
    }
    int ret = pread(pcihandle, &tmp, sizeof(uint32_t), config->base_offset);
    if (ret < 0 || ret != sizeof(uint32_t))
    {
        ERROR_PRINT(Cannot get start address: read failed);
        close(pcihandle);
        bdestroy(bdevmem);
        return -1;
    }
    if (!tmp)
    {
        ERROR_PRINT(Cannot get address: MMIO base is zero);
        close(pcihandle);
        bdestroy(bdevmem);
        return -1;
    }
    addr = (tmp & config->base_mask) << config->base_shift;
    //DEBUG_PRINT(DEBUGLEV_DEVELOP, IMC %d BASE 0x%lX = (0x%lX & 0x%lX) << %d, imc_idx, addr, tmp, config->base_mask, config->base_shift);
    tmp = 0;
    mem_offset = config->device_offset + imc_idx * config->device_stride;
    //DEBUG_PRINT(DEBUGLEV_DEVELOP, IMC %d offset 0x%X, imc_idx, mem_offset);
    ret = pread(pcihandle, &tmp, sizeof(uint32_t), mem_offset);
    if (ret < 0)
    {
        ERROR_PRINT(Cannot get start address of device: read failed);
        close(pcihandle);
        bdestroy(bdevmem);
        return -1;
    }
    addr |= (tmp & config->device_mask) << config->device_shift;
    addr += config->freerun_offset;

    //DEBUG_PRINT(DEBUGLEV_DEVELOP, IMC %d IMC_OFF 0x%lX (0x%lX & 0x%lX) << %d, imc_idx, addr, tmp, config->device_mask, config->device_shift);

    close(pcihandle);

    pcihandle = open("/dev/mem", O_RDWR);
    if (pcihandle < 0)
    {
        ERROR_PRINT(Cannot get mmap address: failed to open /dev/mem);
        bdestroy(bdevmem);
        return -1;
    }
    //DEBUG_PRINT(DEBUGLEV_DEVELOP, IMC %d MMAP 0x%llX, imc_idx, addr);

    //DEBUG_PRINT(DEBUGLEV_DEVELOP, MMap size 0x%x addr %lld (0x%llX), ICX_IMC_MMIO_SIZE, addr & (~(4096 - 1)), addr & (~(4096 - 1)));
    void* maddr = mmap(NULL, config->channel_count*ICX_IMC_MMIO_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, pcihandle, addr & (~(4096 - 1)));
    if (maddr == MAP_FAILED)
    {
        ERROR_PRINT(Cannot get start address of device: mmap failed);
        bdestroy(bdevmem);
        close(pcihandle);
        return -1;
    }
    DEBUG_PRINT(DEBUGLEV_DEVELOP, IMC %d MMAP %p ADDR %lX, imc_idx, maddr, addr);
    handle->mmap_addr = maddr;
    handle->addr = addr;
    handle->fd = pcihandle;
    handle->reg_offset = addr - (addr & (~(4096 - 1)));

    bdestroy(bdevmem);

    return 0;
}

static uint32_t get_pci_bus_of_socket(int socket)
{
    switch(socket)
    {
        case 0:
            return 0x7e;
            break;
        case 1:
            return 0xfe;
            break;
        default:
            return 0xff;
            break;
    }
    return 0xff;
}

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

int
access_x86_mmio_init(const int socket)
{
    int i = 0;
    uint64_t startAddr = 0;
    if (access_mmio_initialized[socket])
    {
        return 0;
    }

    if (!access_mmio_initialized[socket])
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, access_x86_mmio_init for socket %d, socket);
        topology_init();
        if (cpuid_info.family != P6_FAMILY)
        {
            ERROR_PRINT(MMIO only supported for Intel platforms);
            return -1;
        }
        switch(cpuid_info.model)
        {
            case ICELAKEX1:
            case ICELAKEX2:
                mmio_config = &mmio_icelakeX;
                break;
            default:
                return -1;
                break;
        }
        
        if (!mmio_sockets)
        {
            num_mmio_sockets = cpuid_topology.numSockets;
            mmio_sockets = malloc(num_mmio_sockets * sizeof(MMIOSocketBoxes));
            if (!mmio_sockets)
            {
                ERROR_PRINT(Failed to malloc space for socket);
                num_mmio_sockets = 0;
                mmio_config = NULL;
                return -1;
            }
            memset(mmio_sockets, 0, num_mmio_sockets * sizeof(MMIOSocketBoxes));
        }
    }
    if (mmio_sockets && socket >= 0 && socket < num_mmio_sockets)
    {
        MMIOSocketBoxes* sbox = &mmio_sockets[socket];
        if (sbox->pci_bus == 0)
        {
            sbox->pci_bus = get_pci_bus_of_socket(socket);
        }

        /* General units, one per iMC channel */
        if (!sbox->boxes)
        {
            int num_devs = mmio_config->device_count * mmio_config->channel_count;
            sbox->boxes = malloc(num_devs * sizeof(MMIOBoxHandle));
            if (!sbox->boxes)
            {
                ERROR_PRINT(Failed to malloc space for socket boxes);
                num_mmio_sockets = 0;
                free(mmio_sockets);
                mmio_sockets = NULL;
                mmio_config = NULL;
                return -1;
            }
            sbox->num_boxes = num_devs;
        }

        /* Free-running counter units, one per iMC device */
        if (!sbox->freerun)
        {
            int num_devs = mmio_config->device_count;
            sbox->freerun = malloc(num_devs * sizeof(MMIOBoxHandle));
            if (!sbox->freerun)
            {
                ERROR_PRINT(Failed to malloc space for freerun boxes);
                free(sbox->boxes);
                sbox->boxes = 0;
                sbox->num_boxes = 0;
                sbox->pci_bus = 0;
                num_mmio_sockets = 0;
                free(mmio_sockets);
                mmio_sockets = NULL;
                mmio_config = NULL;
                return -1;
            }
            sbox->num_freerun = num_devs;
        }

        for (i = 0; i < sbox->num_boxes; i++)
        {
            MMIOBoxHandle* handle = &sbox->boxes[i];

            int ret = mmio_fillBox(mmio_config, sbox->pci_bus, i, handle);
            if (ret < 0)
                return ret;
        }

        for (i = 0; i < sbox->num_freerun; i++)
        {
            MMIOBoxHandle* handle = &sbox->freerun[i];

            int ret = mmio_fillFreerunBox(mmio_config, sbox->pci_bus, i, handle);
            if (ret < 0)
            {
                return ret;
            }
        }

        access_mmio_initialized[socket] = 1;
    }
    return 0;
}

void
access_x86_mmio_finalize(const int socket)
{
    int i = 0, j = 0;
    if (access_mmio_initialized[socket])
    {
        MMIOSocketBoxes* sbox = &mmio_sockets[socket];
        for (i = 0; i < mmio_config->device_count*mmio_config->channel_count; i++)
        {
            MMIOBoxHandle* handle = &sbox->boxes[i];
            if (handle->fd >= 0)
            {
                if (handle->mmap_addr)
                {
                    munmap(handle->mmap_addr, mmio_config->mmap_size);
                    handle->mmap_addr = NULL;
                }
                close(handle->fd);
                handle->fd = -1;
                handle->addr = 0;
            }
        }
        for (i = 0; i < mmio_config->device_count; i++)
        {
            MMIOBoxHandle* handle = &sbox->freerun[i];
            if (handle->fd >= 0)
            {
                if (handle->mmap_addr)
                {
                    munmap(handle->mmap_addr, mmio_config->mmap_size);
                    handle->mmap_addr = NULL;
                }
                close(handle->fd);
                handle->fd = -1;
                handle->addr = 0;
            }
        }
        access_mmio_initialized[socket] = 0;
        int not_done = 0;
        for (i = 0; i < num_mmio_sockets; i++)
        {
            MMIOSocketBoxes* sbox = &mmio_sockets[i];
            for (j = 0; j < mmio_config->device_count * mmio_config->channel_count; j++)
            {
                MMIOBoxHandle* handle = &sbox->boxes[j];
                if (handle->fd >= 0)
                {
                    not_done = 1;
                    break;
                }
            }
            for (j = 0; j < mmio_config->device_count; j++)
            {
                MMIOBoxHandle* handle = &sbox->freerun[j];
                if (handle->fd >= 0)
                {
                    not_done = 1;
                    break;
                }
            }
        }
        if (!not_done)
        {
            for (i = 0; i < num_mmio_sockets; i++)
            {
                MMIOSocketBoxes* sbox = &mmio_sockets[i];
                if (sbox)
                {
                    free(sbox->freerun);
                    sbox->freerun = NULL;
                    sbox->num_freerun = 0;
                    free(sbox->boxes);
                    sbox->boxes = NULL;
                    sbox->num_boxes = 0;
                    sbox->pci_bus = 0;
                }
            }
            free(mmio_sockets);
            mmio_sockets = NULL;
            num_mmio_sockets = 0;
            mmio_config = NULL;
        }
    }
}

int
access_x86_mmio_read(PciDeviceIndex dev, const int socket, uint32_t reg, uint64_t *data)
{
    int imc_idx = 0;
    int width = 64;
    uint64_t d = 0;
    *data = d;
    if (!access_mmio_initialized[socket])
    {
        int ret = access_x86_mmio_init(socket);
        if (ret < 0)
            return ret;
    }

    if (socket < 0 || socket >= num_mmio_sockets)
    {
        return -ENODEV;
    }
    MMIOSocketBoxes* sbox = &mmio_sockets[socket];
    MMIOBoxHandle* box = NULL;
    if (dev >= MMIO_IMC_DEVICE_0_CH_0 && dev <= MMIO_IMC_DEVICE_0_CH_7)
    {
        imc_idx = (dev - MMIO_IMC_DEVICE_0_CH_0);
        box = &sbox->boxes[imc_idx];
        switch(reg)
        {
            case 0x08:
            case 0x10:
            case 0x18:
            case 0x20:
            case 0x38:
                width = 64;
                break;
            case 0x40:
            case 0x44:
            case 0x48:
            case 0x4C:
            case 0x00:
            case 0x5C:
            case 0x54:
                width = 32;
                break;
        }
    }
    else if (dev >= MMIO_IMC_DEVICE_0_FREERUN && dev <= MMIO_IMC_DEVICE_3_FREERUN)
    {
        imc_idx = dev - MMIO_IMC_DEVICE_0_FREERUN;
        box = &sbox->freerun[imc_idx];
        width = 64;
    }
    if (box)
    {
        uint64_t d = 0;
        if (width == 64)
        {
            d = (uint64_t)*((uint64_t *)(box->mmap_addr + box->reg_offset + reg));
        }
        else if (width == 32)
        {
            d = (uint64_t)(*((uint32_t *)(box->mmap_addr + box->reg_offset + reg)));
        }
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Read MMIO counter 0x%X Dev %d on socket %d: 0x%lX, reg, imc_idx, socket, d);
        *data = d;
        return 0;
    }
    return -ENODEV;

}


int
access_x86_mmio_write(PciDeviceIndex dev, const int socket, uint32_t reg, uint64_t data)
{
    int width = 64;
    if (!access_mmio_initialized[socket])
    {
        int ret = access_x86_mmio_init(socket);
        if (ret < 0)
            return ret;
    }
    if (socket < 0 || socket >= num_mmio_sockets)
    {
        return -ENODEV;
    }
    /*if (dev >= MMIO_IMC_DEVICE_0_FREERUN && dev <= MMIO_IMC_DEVICE_3_FREERUN)
    {
        return -EPERM;
    }*/
    if (dev < MMIO_IMC_DEVICE_0_CH_0 || dev > MMIO_IMC_DEVICE_0_CH_7)
    {
        return -ENODEV;
    }

    int imc_idx = (dev - MMIO_IMC_DEVICE_0_CH_0);
    switch(reg)
    {
        case 0x08:
        case 0x10:
        case 0x18:
        case 0x20:
        case 0x38:
            width = 64;
            break;
        case 0x40:
        case 0x44:
        case 0x48:
        case 0x4C:
        case 0x00:
        case 0x5C:
        case 0x54:
            width = 32;
            break;
    }

    MMIOSocketBoxes* sbox = &mmio_sockets[socket];
    if (sbox)
    {
        MMIOBoxHandle* box = &sbox->boxes[imc_idx];
        if (box)
        {
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Write MMIO counter 0x%X Dev %d on socket %d: 0x%lX, reg, imc_idx, socket, data);
            if (width == 64)
            {
                *((uint64_t *)(box->mmap_addr + box->reg_offset + reg)) = data;
            }
            else if (width == 32)
            {
                *((uint32_t*)(box->mmap_addr + box->reg_offset + reg)) = (uint32_t)data;
            }
            return 0;
        }
    }
    return -ENODEV;
}

int
access_x86_mmio_check(PciDeviceIndex dev, int socket)
{
    int imc_idx = 0;
    if (!access_mmio_initialized[socket])
    {
        int ret = access_x86_mmio_init(socket);
        if (ret < 0)
            return 0;
    }
    if (socket < 0 || socket >= num_mmio_sockets)
    {
        return 0;
    }
    MMIOSocketBoxes* sbox = &mmio_sockets[socket];
    if (!sbox)
    {
        return 0;
    }
    MMIOBoxHandle* box = NULL;
    if (dev >= MMIO_IMC_DEVICE_0_CH_0 && dev <= MMIO_IMC_DEVICE_0_CH_7)
    {
        imc_idx = (dev - MMIO_IMC_DEVICE_0_CH_0);
        box = &sbox->boxes[imc_idx];
    }
    else if (dev >= MMIO_IMC_DEVICE_0_FREERUN && dev <= MMIO_IMC_DEVICE_3_FREERUN)
    {
        imc_idx = (dev - MMIO_IMC_DEVICE_0_FREERUN);
        box = &sbox->freerun[imc_idx];
    }
    //DEBUG_PRINT(DEBUGLEV_DEVELOP, MMIO device check dev %d box %d socket %d, dev, imc_idx, socket);

    if (box && box->mmap_addr)
    {
        return 1;
    }
    return 0;
}

