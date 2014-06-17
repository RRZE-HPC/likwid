#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>


#include <types.h>
#include <accessClient.h>
#include <bstrlib.h>
#include <affinity.h>
#include <topology.h>
#include <error.h>


int 
proc_pci_init(uint16_t testDevice, char** socket_bus, int* nrSockets)
{
    FILE *fptr;
    char buf[1024];
    int cntr = 0;
    uint16_t testVendor = 0x8086;
    uint32_t sbus, sdevfn, svend, sdev;
    
    

    if ( (fptr = fopen( "/proc/bus/pci/devices", "r")) == NULL )
    {
        fprintf(stderr, "Unable to open /proc/bus/pci/devices. \
                Thus, no support for PCI based Uncore counters.\n");
        return -ENODEV;
    }

    while( fgets(buf, sizeof(buf)-1, fptr) )
    {
        if ( sscanf(buf, "%2x%2x %4x%4x", &sbus, &sdevfn, &svend, &sdev) == 4 &&
             svend == testVendor && sdev == testDevice )
        {
            socket_bus[cntr] = (char*)malloc(4);
            sprintf(socket_bus[cntr++], "%02x/", sbus);
        }
    }
    fclose(fptr);
    
    *nrSockets = cntr;
    
    if ( cntr == 0 )
    {
        //fprintf(stderr, "Uncore not supported on this system\n");
        return -ENODEV;
    }
    
    return 0;
}
