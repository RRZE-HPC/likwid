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


#include <types.h>
#include <error.h>
#include <topology.h>
#include <msr.h>
#include <pci.h>
#include <accessClient.h>
#include <perfmon.h>
#include <access.h>


static int globalSocket = -1;
static int cpuSockets[MAX_NUM_THREADS] = { [0 ... MAX_NUM_THREADS-1] = -1};
static int registeredCpus = 0;

int _HPMinit(int cpu_id)
{
    int ret = 0;
    if (accessClient_mode == ACCESSMODE_DIRECT)
    {
        ret = msr_init(0);
        if (ret == 0)
        {
            if (cpuid_info.supportUncore)
            {
                ret = pci_init(0);
            }
        }
    }
    else if (accessClient_mode == ACCESSMODE_DAEMON)
    {
        accessClient_init(&cpuSockets[cpu_id]);
        if (globalSocket == -1)
        {
            globalSocket = cpuSockets[cpu_id];
            ret = msr_init(globalSocket);
            if (ret == 0)
            {
                if (cpuid_info.supportUncore)
                {
                    ret = pci_init(globalSocket);
                }
            }
        }
    }
    if (ret == 0)
    {
        registeredCpus++;
    }
    return 0;
}

int HPMinit(void)
{
    return _HPMinit(0);
}

int HPMinitialized(void)
{
    return registeredCpus;
}

int HPMaddThread(int cpu_id)
{
    if (((cpuSockets[cpu_id] == -1) && (accessClient_mode == ACCESSMODE_DAEMON)) ||
         (accessClient_mode == ACCESSMODE_DIRECT))
    {
        return _HPMinit(cpu_id);
    }
    return 0;
}

void HPMfinalize(void)
{
    msr_finalize();
    pci_finalize();
    if (accessClient_mode == ACCESSMODE_DAEMON)
    {
        for (int i=0;i<cpuid_topology.numHWThreads; i++)
        {
            if (cpuSockets[i] != -1)
            {
                close(cpuSockets[i]);
                cpuSockets[i] = -1;
                registeredCpus--;
            }
        }
    }
    globalSocket = -1;
    return;
}

int HPMread(int cpu_id, PciDeviceIndex dev, uint32_t reg, uint64_t* data)
{
    int socket = globalSocket;
    uint64_t tmp = 0x0ULL;
    int err = 0;
    if ((dev >= MAX_NUM_PCI_DEVICES) || (data == NULL))
    {
        return -EFAULT;
    }
    if ((cpu_id < 0) || (cpu_id >= cpuid_topology.numHWThreads))
    {
        return -ERANGE;
    }
    if (accessClient_mode == ACCESSMODE_DAEMON)
    {
        if ((cpuSockets[cpu_id] >= 0) && (cpuSockets[cpu_id] != socket))
        {
            socket = cpuSockets[cpu_id];
        }
        else if (socket < 0)
        {
            return -ENOENT;
        }
    }
    *data = 0x0ULL;
    DEBUG_PRINT(DEBUGLEV_DEVELOP, READ S[%d] C[%d] DEV[%d] R 0x%X, socket, cpu_id, dev, reg);
    if (dev == MSR_DEV)
    {
        err = msr_tread(socket, cpu_id, reg, &tmp);
        *data = tmp;
    }
    else if (pci_checkDevice(dev, cpu_id))
    {
        err = pci_tread(socket, cpu_id, dev, reg, (uint32_t*)&tmp);
        *data = tmp;
    }
    DEBUG_PRINT(DEBUGLEV_DEVELOP, READ S[%d] C[%d] DEV[%d] R 0x%X = 0x%llX ERR[%d], socket, cpu_id, dev, reg, LLU_CAST tmp, err);
    return err;
}

int HPMwrite(int cpu_id, PciDeviceIndex dev, uint32_t reg, uint64_t data)
{
    int socket = globalSocket;
    int err = 0;
    uint64_t tmp;
    if (dev >= MAX_NUM_PCI_DEVICES)
    {
        ERROR_PRINT(MSR WRITE D %d NOT VALID, dev);
        return -EFAULT;
    }
    if ((cpu_id < 0) || (cpu_id >= cpuid_topology.numHWThreads))
    {
        ERROR_PRINT(MSR WRITE C %d OUT OF RANGE, cpu_id);
        return -ERANGE;
    }
    if (accessClient_mode == ACCESSMODE_DAEMON)
    {
        if ((cpuSockets[cpu_id] >= 0) && (cpuSockets[cpu_id] != socket))
        {
            socket = cpuSockets[cpu_id];
        }
        if (socket < 0)
        {
            ERROR_PRINT(MSR WRITE S %d INVALID, socket);
            return -ENOENT;
        }
    }
    DEBUG_PRINT(DEBUGLEV_DEVELOP, WRITE S[%d] C[%d] DEV[%d] R 0x%X D 0x%llX, socket, cpu_id, dev, reg, LLU_CAST data);
    if (dev == MSR_DEV)
    {
        err = msr_twrite(socket, cpu_id, reg, data);
        DEBUG_PRINT(DEBUGLEV_DEVELOP, WRITE S[%d] C[%d] DEV[%d] R 0x%X D 0x%llX ERR[%d], socket, cpu_id, dev, reg, LLU_CAST data, err);
        if (perfmon_verbosity == DEBUGLEV_DEVELOP)
        {
            int err2 = msr_tread(socket, cpu_id, reg, &tmp);
            DEBUG_PRINT(DEBUGLEV_DEVELOP, VERIFY S[%d] C[%d] DEV[%d] R 0x%X D 0x%llX ERR[%d] CMP %d, socket, cpu_id, dev, reg, LLU_CAST tmp, err2, (data == tmp));
        }
    }
    else if (pci_checkDevice(dev, cpu_id))
    {
        err = pci_twrite(socket, cpu_id, dev, reg, data);
        DEBUG_PRINT(DEBUGLEV_DEVELOP, WRITE S[%d] C[%d] DEV[%d] R 0x%X D 0x%llX ERR[%d], socket, cpu_id, dev, reg, LLU_CAST data, err);
        if (perfmon_verbosity == DEBUGLEV_DEVELOP)
        {
            int err2 = pci_tread(socket, cpu_id, dev, reg, (uint32_t*)&tmp);
            DEBUG_PRINT(DEBUGLEV_DEVELOP, VERIFY S[%d] C[%d] DEV[%d] R 0x%X D 0x%llX ERR[%d] CMP %d, socket, cpu_id, dev, reg, LLU_CAST tmp, err2, (data == tmp));
        }
    }
    return err;
}
