/*
 * =======================================================================================
 *
 *      Filename:  stream.cu
 *
 *      Description:  STREAM benchmark with NvMarkerAPI
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2019 RRZE, University Erlangen-Nuremberg
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
#include <sys/types.h>
#include <sys/syscall.h>

#include <stdint.h>
#include <sys/time.h>
#include <unistd.h>
#include <errno.h>
#include <cuda.h>

#define ITER 100
#define SIZE 400000000U
#define DATATYPE float

#define gettid() syscall(SYS_gettid)
#include <likwid.h>
#include <likwid-marker.h>
#define HLINE "-------------------------------------------------------------\n"

#ifndef MIN
#define MIN(x,y) ((x)<(y)?(x):(y))
#endif

typedef struct {
    struct timeval before;
    struct timeval after;
} TimeData;


void time_start(TimeData* time)
{
    gettimeofday(&(time->before),NULL);
    time->after.tv_sec = time->before.tv_sec;
    time->after.tv_usec = time->before.tv_usec;
}


void time_stop(TimeData* time)
{
    gettimeofday(&(time->after),NULL);
}

double time_print(TimeData* time)
{
    long int sec;
    double timeDuration;

    sec = time->after.tv_sec - time->before.tv_sec;
    timeDuration = ((double)((sec*1000000)+time->after.tv_usec) - (double) time->before.tv_usec);

    return (timeDuration/1000000);
}

__global__ void initKernel(DATATYPE * devPtr, const DATATYPE val, int len)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len)
    {
        devPtr[idx] = val;
    }
}



__global__ void Copykernel(DATATYPE const * __restrict__ const a, DATATYPE * __restrict__ const b, int len)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len)
    {
        b[idx] = a[idx];
    }
}

__global__ void Scalekernel(DATATYPE const * __restrict__ const a, DATATYPE * __restrict__ const b, DATATYPE scale, int len)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len)
    {
        b[idx] = scale * a[idx];
    }
}

__global__ void Addkernel(DATATYPE const * __restrict__ const a, DATATYPE * __restrict__ const b, DATATYPE * __restrict__ const c, int len)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len)
    {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void Triadkernel(DATATYPE const * __restrict__ const a, DATATYPE * __restrict__ const b, DATATYPE * __restrict__ const c, DATATYPE scale, int len)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len)
    {
        c[idx] = a[idx] + scale * b[idx];
    }
}

int main(int argn, char** argc)
{
    DATATYPE *a,*b,*c,*d;
    DATATYPE scalar = 3.0;
    int blockSize = 192;
    TimeData timer;
    double triad_time, copy_time, scale_time, add_time;

    cudaMalloc(&a, SIZE*sizeof(DATATYPE));
    cudaMalloc(&b, SIZE*sizeof(DATATYPE));
    cudaMalloc(&c, SIZE*sizeof(DATATYPE));
    cudaMalloc(&d, SIZE*sizeof(DATATYPE));

#ifdef LIKWID_NVMON
    printf("Using likwid\n");
#endif

    NVMON_MARKER_INIT;

    dim3 dimBlock(blockSize);
    dim3 dimGrid(SIZE/dimBlock.x );
    if( SIZE % dimBlock.x != 0 ) dimGrid.x+=1;

    initKernel<<<dimGrid,dimBlock>>>(a, 1.0, SIZE);
    initKernel<<<dimGrid,dimBlock>>>(b, 2.0, SIZE);
    initKernel<<<dimGrid,dimBlock>>>(c, 0.0, SIZE);
    initKernel<<<dimGrid,dimBlock>>>(d, 1.0, SIZE);
    cudaDeviceSynchronize();

    NVMON_MARKER_REGISTER("copy");
    NVMON_MARKER_REGISTER("scale");
    NVMON_MARKER_REGISTER("add");
    NVMON_MARKER_REGISTER("triad");


    time_start(&timer);
    for (int k=0; k<ITER; k++)
    {
        NVMON_MARKER_START("copy");
        Copykernel<<<dimGrid,dimBlock>>>(a, c, SIZE);
        cudaDeviceSynchronize();
        NVMON_MARKER_STOP("copy");
    }
    time_stop(&timer);
    copy_time = time_print(&timer)/(double)ITER;

    time_start(&timer);
    for (int k=0; k<ITER; k++)
    {
        NVMON_MARKER_START("scale");
        Scalekernel<<<dimGrid,dimBlock>>>(b, c, scalar, SIZE);
        cudaDeviceSynchronize();
        NVMON_MARKER_STOP("scale");
    }
    time_stop(&timer);
    scale_time = time_print(&timer)/(double)ITER;

    time_start(&timer);
    for (int k=0; k<ITER; k++)
    {
        NVMON_MARKER_START("add");
        Addkernel<<<dimGrid,dimBlock>>>(a, b, c, SIZE);
        cudaDeviceSynchronize();
        NVMON_MARKER_STOP("add");
    }
    time_stop(&timer);
    add_time = time_print(&timer)/(double)ITER;

    time_start(&timer);
    for (int k=0; k<ITER; k++)
    {
        NVMON_MARKER_START("triad");
        Triadkernel<<<dimGrid,dimBlock>>>(a, b, c, scalar, SIZE);
        cudaDeviceSynchronize();
        NVMON_MARKER_STOP("triad");
    }
    time_stop(&timer);
    triad_time = time_print(&timer)/(double)ITER;

    printf("Processed %.1f Mbyte at copy benchmark in %.4f seconds: %.2f MByte/s\n",
                        1E-6*(2*SIZE*sizeof(DATATYPE)),
                        copy_time,
                        1E-6*((2*SIZE*sizeof(DATATYPE))/copy_time));
    printf("Processed %.1f Mbyte at scale benchmark in %.4f seconds: %.2f MByte/s %.2f MFLOP/s\n",
                        1E-6*(2*SIZE*sizeof(DATATYPE)),
                        scale_time,
                        1E-6*((2*SIZE*sizeof(DATATYPE))/scale_time),
                        1E-6*(SIZE/scale_time));
    printf("Processed %.1f Mbyte at add benchmark in %.4f seconds: %.2f MByte/s %.2f MFLOP/s\n",
                        1E-6*(3*SIZE*sizeof(DATATYPE)),
                        add_time,
                        1E-6*((3*SIZE*sizeof(DATATYPE))/add_time),
                        1E-6*(SIZE/add_time));
    printf("Processed %.1f Mbyte at triad benchmark in %.4f seconds: %.2f MByte/s %.2f MFLOP/s\n",
                        1E-6*(3*SIZE*sizeof(DATATYPE)),
                        triad_time,
                        1E-6*((3*SIZE*sizeof(DATATYPE))/triad_time),
                        1E-6*((2*SIZE)/triad_time));


    NVMON_MARKER_CLOSE;
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(d);
    return 0;
}
