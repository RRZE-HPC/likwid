
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

static char msr_name[] = "/dev/cpu/0/msr";
static int msr_fd;

int check_msr()
{
    if (access(msr_name, R_OK|W_OK))
    {
        fprintf(stderr,"Unable to access MSR device %s: %s\n", msr_name, strerror(errno));
        return 1;
    }
    return 0;
}

int open_msr()
{
    msr_fd = open(msr_name, O_RDWR);
    if (msr_fd < 0)
    {
        fprintf(stderr,"Cannot open MSR device %s: %s\n", msr_name, strerror(errno));
        return 1;
    }
    return 0;
}

int close_msr()
{
    if (msr_fd > 0)
    {
        close(msr_fd);
    }
    return 0;
}

int read_msr()
{
    ssize_t ret;
    uint64_t data = 0;
    uint32_t reg = 0x38D;
    if (msr_fd > 0)
    {
        ret = pread(msr_fd, &data, sizeof(uint64_t), reg);
        if (ret < 0)
        {
            fprintf(stderr, "Cannot read register 0x%x at MSR %s: %s\n", reg, msr_name, strerror(errno));
            return 1;
        }
        else if (ret != sizeof(uint64_t))
        {
            fprintf(stderr, "Incomplete read on register 0x%x at MSR %s: Only %lu bytes\n", reg, msr_name, ret);
            return 1;
        }
        return 0;
    }
    return 1;
}

int write_msr()
{
    ssize_t ret;
    uint64_t data = 0;
    uint32_t reg = 0x38D;
    if (msr_fd > 0)
    {
        ret = pwrite(msr_fd, &data, sizeof(uint64_t), reg);
        if (ret < 0)
        {
            fprintf(stderr, "Cannot write register 0x%x at MSR %s: %s\n", reg, msr_name, strerror(errno));
            return 1;
        }
        else if (ret != sizeof(uint64_t))
        {
            fprintf(stderr, "Incomplete read on register 0x%x at MSR %s: Only %lu bytes\n", reg, msr_name, ret);
            return 1;
        }
        return 0;
    }
    return 1;
}

int main()
{
    int ret = 0;
    if (check_msr()) return 1;
    if (open_msr()) return 1;
    if (read_msr()) return 1;
    if (write_msr()) return 1;
    if (close_msr()) return 1;
    printf("All OK!\n");
    return 0;
}
