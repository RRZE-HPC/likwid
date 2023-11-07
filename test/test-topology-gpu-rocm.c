#include <stdlib.h>
#include <stdio.h>
#include <likwid.h>

int main()
{
    printf("Init\n");

    // Init
    int ret = topology_rocm_init();
    if (ret != 0)
    {
        printf("Oops! Failed to initialize ROCm GPU topology.");
        return -1;
    }

    // Use
    RocmTopology_t topo = get_rocmTopology();
    printf("Number of devices: %d\n\n", topo->numDevices);

    for (int i = 0; i < topo->numDevices; i++)
    {
        RocmDevice *device = &topo->devices[i];

        printf("---\n");
        printf("devid: %d\n", device->devid);
        printf("numaNode: %d\n", device->numaNode);
        printf("name: %s\n", device->name);
        printf("short_name: %s\n", device->short_name);
        printf("mem: %u\n", device->mem);
        printf("ccapMajor: %d\n", device->ccapMajor);
        printf("ccapMinor: %d\n", device->ccapMinor);
        printf("maxThreadsPerBlock: %d\n", device->maxThreadsPerBlock);
        printf("maxThreadsPerDim: %d / %d / %d\n", device->maxThreadsDim[0], device->maxThreadsDim[1], device->maxThreadsDim[2]);
        printf("maxGridSize: %d / %d / %d\n", device->maxGridSize[0], device->maxGridSize[1], device->maxGridSize[2]);
        printf("sharedMemPerBlock: %d\n", device->sharedMemPerBlock);
        printf("totalConstantMemory: %u\n", device->totalConstantMemory);
        printf("simdWidth: %d\n", device->simdWidth);
        printf("memPitch: %u\n", device->memPitch);
        printf("regsPerBlock: %d\n", device->regsPerBlock);
        printf("clockRatekHz: %d\n", device->clockRatekHz);
        printf("textureAlign: %u\n", device->textureAlign);
        printf("l2Size: %d\n", device->l2Size);
        printf("memClockRatekHz: %d\n", device->memClockRatekHz);
        printf("pciBus: %d\n", device->pciBus);
        printf("pciDev: %d\n", device->pciDev);
        printf("pciDom: %d\n", device->pciDom);
        printf("numMultiProcs: %d\n", device->numMultiProcs);
        printf("maxThreadPerMultiProc: %d\n", device->maxThreadPerMultiProc);
        printf("memBusWidth: %d\n", device->memBusWidth);
        printf("ecc: %d\n", device->ecc);
        printf("mapHostMem: %d\n", device->mapHostMem);
        printf("integrated: %d\n", device->integrated);
        printf("---\n\n");
    }

    // Finalize
    topology_rocm_finalize();

    printf("Finalized\n");
    return 0;
}
