#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <endian.h>

#include <types.h>
#include <error.h>
#include <topology.h>
#include <access_power.h>
#include <registers.h>


#define POWER8_MAX_STRING_LENGTH 40
#define MAX_POWER8_REGISTERS 14
#define POWER8_SYSFS_BASE "/sys/devices/system/cpu"

typedef struct {
    uint8_t width;
    char*  filename;
} Power8RegisterInfo;

static Power8RegisterInfo Power8Registers[MAX_POWER8_REGISTERS] = {
    [IBM_MMCR0] = {64, "mmcr0"},
    [IBM_MMCR1] = {64, "mmcr1"},
    [IBM_MMCRA] = {64, "mmcra"},
    [IBM_MMCRC] = {64, "mmcrc"},
    [IBM_PMC0] = {32, "pmc1"},
    [IBM_PMC1] = {32, "pmc2"},
    [IBM_PMC2] = {32, "pmc3"},
    [IBM_PMC3] = {32, "pmc4"},
    [IBM_PMC4] = {32, "pmc5"},
    [IBM_PMC5] = {32, "pmc6"},
    [IBM_PIR] = {32, "pir"},
    [IBM_PURR] = {64, "purr"},
    [IBM_SPURR] = {64, "spurr"},
    [IBM_DSCR] = {25, "dscr"},
};

static uint64_t swap_endian(uint64_t num, uint8_t width)
{
    if (width == 32)
    {
	return ((num>>24)&0xff) | 
		((num<<8)&0xff0000) | 
		((num>>8)&0xff00) | 
		((num<<24)&0xff000000);
    }
    else if (width == 64)
    {
	uint64_t val = 0x0ULL;
	val = ((num << 8) & 0xFF00FF00FF00FF00ULL ) | ((num >> 8) & 0x00FF00FF00FF00FFULL );
	val = ((num << 16) & 0xFFFF0000FFFF0000ULL ) | ((num >> 16) & 0x0000FFFF0000FFFFULL );
	return (val << 32) | (val >> 32);
    }
    return 0x0ULL;
}


/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */
static int FD[MAX_NUM_THREADS][MAX_POWER8_REGISTERS] = { [0 ... MAX_NUM_THREADS-1] = { [0 ... MAX_POWER8_REGISTERS-1] = -1 } };
static char cpustrings[MAX_NUM_THREADS][POWER8_MAX_STRING_LENGTH];

int
access_power_init(const int cpu_id)
{
    int i = 0;
    int fd = 0;
    char* pmc_file_name = NULL;
    if (!pmc_file_name)
    {
	pmc_file_name = (char*) malloc(1024 * sizeof(char));
	if (!pmc_file_name)
	{
	    return -ENOMEM;
	}
    }

    for (i = 0; i<MAX_POWER8_REGISTERS; i++)
    {
	sprintf(pmc_file_name,"%s/cpu%d/%s", POWER8_SYSFS_BASE, cpu_id, Power8Registers[i].filename);
	if (!access(pmc_file_name, F_OK))
	{
	    fd = open(pmc_file_name, O_RDWR);
	    if (fd < 0)
	    {
		continue;
	    }
	    FD[cpu_id][i] = 1;
	    DEBUG_PRINT(DEBUGLEV_DEVELOP, Opened PMC device %s for CPU %d,pmc_file_name, cpu_id);
	    close(fd);
	}
    }
    free(pmc_file_name);
    return 0;
}


void
access_power_finalize(const int cpu_id)
{
    int i = 0;

    for (i = 0; i<MAX_POWER8_REGISTERS; i++)
    {
	if (FD[cpu_id][i] > 0)
	{
	    //close(FD[cpu_id][i]);
	    FD[cpu_id][i] = 0;
	}
    }
}

int
access_power_read(PciDeviceIndex dev, const int cpu_id, uint32_t reg, uint64_t *data)
{
    uint64_t tmpdata = 0x0;
    char command[1024];
    char buff[100];
    char* ptr; 
    FILE* fpipe;
    if (reg < 0 || reg >= MAX_POWER8_REGISTERS)
	return -EINVAL;
    if (FD[cpu_id][reg] > 0)
    {
	sprintf(command, "cat %s/cpu%d/%s", POWER8_SYSFS_BASE, cpu_id, Power8Registers[reg].filename);
	if ( !(fpipe = (FILE*)popen(command,"r")) )
	{
	    return -EIO;
	}
	ptr = fgets(buff, 100, fpipe);
	sscanf(buff, "%lx", &tmpdata);
	//printf("Read %s = 0x%llX\n", command, tmpdata);
	//*data = swap_endian(tmpdata, Power8Registers[reg].width);	
	*data = tmpdata;
    }
    else
    {
	return -ENODEV;
    }
    return 0;
}


int
access_power_write(PciDeviceIndex dev, const int cpu_id, uint32_t reg, uint64_t data)
{
    int ret = 0;
    int len = 0;
    char command[1024];
    if (reg < 0 || reg >= MAX_POWER8_REGISTERS)
	return -EINVAL;
    if (FD[cpu_id][reg] > 0)
    {
	/*len = snprintf(cpustrings[cpu_id], POWER8_MAX_STRING_LENGTH, "%016lx", data); 
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Write MSR counter 0x%X with WRMSR instruction on CPU %d data %s, reg, cpu_id, cpustrings[cpu_id]);
        ret = pwrite(FD[cpu_id][reg], cpustrings[cpu_id], len*sizeof(char), 0);
	printf("Write: String %s uint 0x%016lx\n", cpustrings[cpu_id], data); 
        if (ret != len)
        {
            return -EIO;
        }
	memset(cpustrings[cpu_id], '\0', POWER8_MAX_STRING_LENGTH*sizeof(char));*/
	//sprintf(command, "echo %lx >  %s/cpu%d/%s", swap_endian(data, Power8Registers[reg].width), POWER8_SYSFS_BASE, cpu_id, Power8Registers[reg].filename);
	sprintf(command, "echo %lx >  %s/cpu%d/%s", data, POWER8_SYSFS_BASE, cpu_id, Power8Registers[reg].filename);
	//printf("Write %s\n", command);
	ret = system(command);	
	if (ret != 0)
	    return -EIO;
    }
    else
    {
	return -ENODEV;
    }
    return 0;
}


int access_power_check(PciDeviceIndex dev, int cpu_id)
{
    if (FD[cpu_id][0] > 0)
    {
        return 1;
    }
    return 0;
}

