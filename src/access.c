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
#include <cpuid.h>
#include <accessClient.h>
#include <perfmon.h>
#include <access.h>


static int globalSocket = -1;
static int cpuSockets[MAX_NUM_THREADS] = { [0 ... MAX_NUM_THREADS-1] = -1};
static int registeredCpus = 0;
static int init = 0;

int _HPMinit(int cpu_id)
{
    if (accessClient_mode == ACCESSMODE_DIRECT)
    {
        msr_init(0);
        if (cpuid_info.supportUncore)
        {
            pci_init(0);
        }
    }
    else if (accessClient_mode == ACCESSMODE_DAEMON)
    {
        accessClient_init(&cpuSockets[cpu_id]);
        if (globalSocket == -1)
        {
            globalSocket = cpuSockets[cpu_id];
            msr_init(globalSocket);
            if (cpuid_info.supportUncore)
            {
                pci_init(globalSocket);
            }
        }
    }
    init = 1;
    return 0;
}

int HPMinit(void)
{
    return _HPMinit(0);
}

int HPMinitialized(void)
{
    return init;
}

int HPMaddThread(int cpu_id)
{
    if ((cpuSockets[cpu_id] == -1) && (accessClient_mode == ACCESSMODE_DAEMON))
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
            }
        }
    }
    globalSocket = -1;
    return;
}

int HPMread(int cpu_id, PciDeviceIndex dev, uint32_t reg, uint64_t* data)
{
    int socket = globalSocket;
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
    if (dev == MSR_DEV)
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, MSR READ S[%d] C[%d] R 0x%X, socket, cpu_id, reg);
        return msr_tread(socket, cpu_id, reg, data);
    }
    else if (pci_checkDevice(dev, cpu_id))
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, PCI READ S[%d] C[%d] D[%d] R 0x%X, socket, cpu_id, dev, reg);
        return pci_tread(socket, cpu_id, dev, reg, (uint32_t*)data);
    }
    return 0;
}

int HPMwrite(int cpu_id, PciDeviceIndex dev, uint32_t reg, uint64_t data)
{
    int socket = globalSocket;
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

    if (dev == MSR_DEV)
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, MSR WRITE S[%d] C[%d] R 0x%X D 0x%llX, socket, cpu_id, reg, LLU_CAST data);
        return msr_twrite(socket, cpu_id, reg, data);
    }
    else if (pci_checkDevice(dev, cpu_id))
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, PCI WRITE S[%d] C[%d] D[%d] R 0x%X D 0x%llX, socket, cpu_id, dev, reg, LLU_CAST data);
        return pci_twrite(socket, cpu_id, dev, reg, data);
    }
    return 0;
}
