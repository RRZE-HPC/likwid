## Build & Install

```bash
export ROCM_HOME=/opt/rocm
make
make install
```

## Test

Build

```bash
cd test
# make clean
make test-topology-gpu-rocm
make test-rocmon-triad
make test-rocmon-triad-marker
```

Run

```bash
export LD_LIBRARY_PATH=/home/users/kraljic/likwid-rocmon/install/lib:/opt/rocm/hip/lib:/opt/rocm/hsa/lib:/opt/rocm/rocprofiler/lib:$LD_LIBRARY_PATH
export ROCP_METRICS=/opt/rocm/rocprofiler/lib/metrics.xml # for rocmon test
export HSA_TOOLS_LIB=librocprofiler64.so.1 # allows rocmon to intercept hsa commands
./gpu-test-topology-gpu-rocm
```
