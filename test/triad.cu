/*
 * =======================================================================================
 *
 *      Filename:  triad.cu
 *
 *      Description:  Triad kernel in CUDA to test NvMarkerAPI
 *
 *      Version:   5.0
 *      Released:  10.11.2019
 *
 *      Author:   Dominik Ernst (de)
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


#include <iomanip>
#include <iostream>
#include <sys/time.h>

#include <likwid-marker.h>

double dtime() {
  double tseconds = 0;
  struct timeval t;
  gettimeofday(&t, NULL);
  tseconds = (double)t.tv_sec + (double)t.tv_usec * 1.0e-6;
  return tseconds;
}

#define GPU_ERROR(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    std::cerr << "GPUassert: \"" << cudaGetErrorString(code) << "\"  in "
              << file << ": " << line << "\n";
    if (abort)
      exit(code);
  }
}

using namespace std;

template <typename T>
__global__ void init_kernel(T *A, const T *__restrict__ B,
                            const T *__restrict__ C, const T *__restrict__ D,
                            const size_t N) {
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  for (size_t i = tidx; i < N; i += blockDim.x * gridDim.x) {
    A[i] = 0.1;
  }
}

template <typename T>
__global__ void sch_triad_kernel(T *A, const T *__restrict__ B,
                                 const T *__restrict__ C,
                                 const T *__restrict__ D, const int64_t N) {
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  for (int64_t i = tidx; i < N; i += blockDim.x * gridDim.x) {
    A[i] = B[i] + C[i] * D[i];
  }
}

int main(int argc, char **argv) {
  const size_t buffer_size = 128 * 1024 * 1024;

  double *dA, *dB, *dC, *dD;

  GPU_ERROR(cudaMalloc(&dA, buffer_size * sizeof(double)));
  GPU_ERROR(cudaMalloc(&dB, buffer_size * sizeof(double)));
  GPU_ERROR(cudaMalloc(&dC, buffer_size * sizeof(double)));
  GPU_ERROR(cudaMalloc(&dD, buffer_size * sizeof(double)));

  init_kernel<<<256, 400>>>(dA, dA, dA, dA, buffer_size);
  init_kernel<<<256, 400>>>(dB, dB, dB, dB, buffer_size);
  init_kernel<<<256, 400>>>(dC, dC, dC, dC, buffer_size);
  init_kernel<<<256, 400>>>(dD, dD, dD, dD, buffer_size);
  GPU_ERROR(cudaDeviceSynchronize());
  LIKWID_NVMARKER_INIT;
  LIKWID_NVMARKER_REGISTER("triad");
  const int iters = 10;

  const int block_size = 512;
  cudaDeviceProp prop;
  int deviceId;
  GPU_ERROR(cudaGetDevice(&deviceId));
  GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
  int smCount = prop.multiProcessorCount;
  int maxActiveBlocks = 0;
  GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxActiveBlocks, sch_triad_kernel<double>, block_size, 0));

  int max_blocks = maxActiveBlocks * smCount;


  sch_triad_kernel<double>
      <<<max_blocks, block_size>>>(dA, dB, dC, dD, buffer_size);

  GPU_ERROR(cudaDeviceSynchronize());
  double t1 = dtime();
  for (int i = 0; i < iters; i++) {
    LIKWID_NVMARKER_START("triad");
    sch_triad_kernel<double>
        <<<max_blocks, block_size>>>(dA, dB, dC, dD, buffer_size);
    LIKWID_NVMARKER_STOP("triad");
  }
  GPU_ERROR(cudaGetLastError());
  GPU_ERROR(cudaDeviceSynchronize());
  double t2 = dtime();

  double dt = (t2 - t1) / iters;

  cout << fixed << setprecision(2) << setw(6) << dt * 1000 << "ms " << setw(5)
       << 4 * buffer_size * sizeof(double) / dt * 1e-9 << "GB/s \n";

  GPU_ERROR(cudaFree(dA));
  GPU_ERROR(cudaFree(dB));
  GPU_ERROR(cudaFree(dC));
  GPU_ERROR(cudaFree(dD));
  LIKWID_NVMARKER_CLOSE;
  return 0;
}
