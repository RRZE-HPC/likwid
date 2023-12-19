--------------------------------------------------------------------------------
Introduction
--------------------------------------------------------------------------------

Likwid is a simple to install and use toolsuite of command line applications and a library
for performance oriented programmers. It works for Intel, AMD, ARMv8 and POWER9
processors on the Linux operating system. There is additional support for Nvidia and AMD GPUs.
There is support for ARMv7 and POWER8/9 but there is currently no test machine in
our hands to test them properly.

[LIKWID Playlist (YouTube)](https://www.youtube.com/playlist?list=PLxVedhmuwLq2CqJpAABDMbZG8Whi7pKsk)

[![Build Status](https://gitos.rrze.fau.de/ub55yzis/likwid/badges/master/pipeline.svg)](https://gitos.rrze.fau.de/ub55yzis/likwid/-/commits/master) [![General LIKWID DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4275676.svg)](https://doi.org/10.5281/zenodo.4275676)

It consists of:

- likwid-topology: print thread, cache and NUMA topology
- likwid-perfctr: configure and read out hardware performance counters on Intel, AMD, ARM and POWER processors and Nvidia GPUs
- likwid-powermeter: read out RAPL Energy information and get info about Turbo mode steps
- likwid-pin: pin your threaded application (pthread, Intel and gcc OpenMP to dedicated processors)
- likwid-bench: Micro benchmarking platform for CPU architectures
- likwid-features: Print and manipulate cpu features like hardware prefetchers (x86 only)
- likwid-genTopoCfg: Dumps topology information to a file
- likwid-mpirun: Wrapper to start MPI and Hybrid MPI/OpenMP applications (Supports Intel MPI, OpenMPI, MPICH and SLURM)
- likwid-perfscope: Frontend to the timeline mode of likwid-perfctr, plots live graphs of performance metrics using gnuplot
- likwid-memsweeper: Sweep memory of NUMA domains and evict cachelines from the last level cache
- likwid-setFrequencies: Tool to control the CPU and Uncore frequencies (x86 only)
- likwid-sysFeatures: Tool to system settings like frequencies, powercaps and prefetchers (experimental)

For further information please take a look at the [Wiki](https://github.com/RRZE-HPC/likwid/wiki) or contact us via Matrix chat [LIKWID General](https://matrix.to/#/#likwid:matrix.org?via=matrix.org).


--------------------------------------------------------------------------------
Supported architectures
--------------------------------------------------------------------------------
Intel
- Intel Atom
- Intel Pentium M
- Intel Core2
- Intel Nehalem
- Intel NehalemEX
- Intel Westmere
- Intel WestmereEX
- Intel Xeon Phi (KNC)
- Intel Silvermont & Airmont
- Intel Goldmont
- Intel SandyBridge
- Intel SandyBridge EP/EN
- Intel IvyBridge
- Intel IvyBridge EP/EN/EX
- Intel Xeon Phi (KNL, KNM)
- Intel Haswell
- Intel Haswell EP/EN/EX
- Intel Broadwell
- Intel Broadwell D
- Intel Broadwell EP
- Intel Skylake
- Intel Kabylake
- Intel Coffeelake
- Intel Skylake SP
- Intel Cascadelake SP
- Intel Icelake
- Intel Icelake SP
- Intel Tigerlake (experimental)
- Intel SapphireRapids

AMD
- AMD K8
- AMD K10
- AMD Interlagos
- AMD Kabini
- AMD Zen
- AMD Zen2
- AMD Zen3
- AMD Zen4

ARM
- ARMv7
- ARMv8
- Special support for Marvell Thunder X2
- Fujitsu A64FX
- ARM Neoverse N1 (AWS Graviton 2)
- ARM Neoverse V1
- HiSilicon TSV110
- Apple M1 (only with Linux)

POWER (experimental)
- IBM POWER8
- IBM POWER9

Nvidia GPUs

AMD GPUs

--------------------------------------------------------------------------------
Download, Build and Install
--------------------------------------------------------------------------------
You can get the releases of LIKWID at:
http://ftp.fau.de/pub/likwid/

For build and installation hints see INSTALL file or check the build instructions
page in the wiki https://github.com/RRZE-HPC/likwid/wiki/Build

For quick install:
```bash
VERSION=stable
wget http://ftp.fau.de/pub/likwid/likwid-$VERSION.tar.gz
tar -xaf likwid-$VERSION.tar.gz
cd likwid-*
vi config.mk # configure build, e.g. change installation prefix and architecture flags
make
sudo make install # sudo required to install the access daemon with proper permissions
```

For ARM builds, the `COMPILER` flag in `config.mk` needs to changed to `GCCARMv8` or `ARMCLANG` (experimental).
For POWER builds, the `COMPILER` flag in `config.mk` needs to changed to `GCCPOWER` or `XLC` (experimental).
For Nvidia GPU support, set `NVIDIA_INTERFACE` in `config.mk` to `true` and adjust build-time variables if needed
For AMD GPU support, set `ROCM_INTERFACE` in `config.mk` to `true` and adjust build-time variables if needed

--------------------------------------------------------------------------------
Usage examples
--------------------------------------------------------------------------------
<details>
<summary><code>likwid-topology</code></summary>
<pre>
--------------------------------------------------------------------------------
CPU name:	Intel(R) Core(TM) i7-6700K CPU @ 4.00GHz
CPU type:	Intel Skylake processor
CPU stepping:	3
********************************************************************************
Hardware Thread Topology
********************************************************************************
Sockets:		1
Cores per socket:	4
Threads per core:	2
--------------------------------------------------------------------------------
HWThread        Thread        Core        Die        Socket        Available
0               0             0           0          0             *                
1               0             1           0          0             *                
2               0             2           0          0             *                
3               0             3           0          0             *                
4               1             0           0          0             *                
5               1             1           0          0             *                
6               1             2           0          0             *                
7               1             3           0          0             *                
--------------------------------------------------------------------------------
Socket 0:		( 0 4 1 5 2 6 3 7 )
--------------------------------------------------------------------------------
********************************************************************************
Cache Topology
********************************************************************************
Level:			1
Size:			32 kB
Cache groups:		( 0 4 ) ( 1 5 ) ( 2 6 ) ( 3 7 )
--------------------------------------------------------------------------------
Level:			2
Size:			256 kB
Cache groups:		( 0 4 ) ( 1 5 ) ( 2 6 ) ( 3 7 )
--------------------------------------------------------------------------------
Level:			3
Size:			8 MB
Cache groups:		( 0 4 1 5 2 6 3 7 )
--------------------------------------------------------------------------------
********************************************************************************
NUMA Topology
********************************************************************************
NUMA domains:		1
--------------------------------------------------------------------------------
Domain:			0
Processors:		( 0 4 1 5 2 6 3 7 )
Distances:		10
Free memory:		318.203 MB
Total memory:		7626.23 MB
--------------------------------------------------------------------------------
</pre>
</details>

<details>
<summary><code>likwid-perfctr</code></summary>
<pre>
$ likwid-perfctr -C 0 -g L2 hostname
--------------------------------------------------------------------------------
CPU name:	Intel(R) Core(TM) i7-6700K CPU @ 4.00GHz
CPU type:	Intel Skylake processor
CPU clock:	4.01 GHz
--------------------------------------------------------------------------------
mytesthost
--------------------------------------------------------------------------------
Group 1: L2
+-----------------------+---------+------------+
|         Event         | Counter | HWThread 0 |
+-----------------------+---------+------------+
|   INSTR_RETIRED_ANY   |  FIXC0  |     321342 |
| CPU_CLK_UNHALTED_CORE |  FIXC1  |     450498 |
|  CPU_CLK_UNHALTED_REF |  FIXC2  |    1118900 |
|    L1D_REPLACEMENT    |   PMC0  |       6670 |
|      L1D_M_EVICT      |   PMC1  |       1840 |
| ICACHE_64B_IFTAG_MISS |   PMC2  |       9293 |
+-----------------------+---------+------------+

+--------------------------------+------------+
|             Metric             | HWThread 0 |
+--------------------------------+------------+
|       Runtime (RDTSC) [s]      |     0.0022 |
|      Runtime unhalted [s]      |     0.0001 |
|           Clock [MHz]          |  1613.6392 |
|               CPI              |     1.4019 |
|  L2D load bandwidth [MBytes/s] |   197.8326 |
|  L2D load data volume [GBytes] |     0.0004 |
| L2D evict bandwidth [MBytes/s] |    54.5745 |
| L2D evict data volume [GBytes] |     0.0001 |
|     L2 bandwidth [MBytes/s]    |   528.0381 |
|     L2 data volume [GBytes]    |     0.0011 |
+--------------------------------+------------+
</pre>
</details>

<details>
<summary><code>likwid-pin</code></summary>
<pre>
$ likwid-pin -c 0,1,2 ./a.out
[pthread wrapper] 
[pthread wrapper] MAIN -> 0
[pthread wrapper] PIN_MASK: 0->1  1->2  
[pthread wrapper] SKIP MASK: 0x0
	threadid 140566548539136 -> hwthread 1 - OK
	threadid 140566540146432 -> hwthread 2 - OK
Number of Threads requested = 3
Thread 0 running on processor 0 ....
Thread 1 running on processor 1 ....
Thread 2 running on processor 2 ....
[...]
</pre>
</details>

<details>
<summary><code>likwid-bench</code></summary>
<pre>
$ likwid-bench -t triad_avx -W N:2GB:3
Warning: Sanitizing vector length to a multiple of the loop stride 16 and thread count 3 from 62500000 elements (500000000 bytes) to 62499984 elements (499999872 bytes)
Allocate: Process running on hwthread 0 (Domain N) - Vector length 62499984/499999872 Offset 0 Alignment 512
Allocate: Process running on hwthread 0 (Domain N) - Vector length 62499984/499999872 Offset 0 Alignment 512
Allocate: Process running on hwthread 0 (Domain N) - Vector length 62499984/499999872 Offset 0 Alignment 512
Allocate: Process running on hwthread 0 (Domain N) - Vector length 62499984/499999872 Offset 0 Alignment 512
Initialization: Each thread in domain initializes its own stream chunks
--------------------------------------------------------------------------------
LIKWID MICRO BENCHMARK
Test: triad_avx
--------------------------------------------------------------------------------
Using 1 work groups
Using 3 threads
--------------------------------------------------------------------------------
Running without Marker API. Activate Marker API with -m on commandline.
--------------------------------------------------------------------------------
Group: 0 Thread 1 Global Thread 1 running on hwthread 4 - Vector length 20833328 Offset 20833328
Group: 0 Thread 0 Global Thread 0 running on hwthread 0 - Vector length 20833328 Offset 0
Group: 0 Thread 2 Global Thread 2 running on hwthread 1 - Vector length 20833328 Offset 41666656
--------------------------------------------------------------------------------
Cycles:			22977763263
CPU Clock:		4007946861
Cycle Clock:		4007946861
Time:			5.733051e+00 sec
Iterations:		96
Iterations per thread:	32
Inner loop executions:	1302083
Size (Byte):		1999999488
Size per thread:	666666496
Number of Flops:	3999998976
MFlops/s:		697.71
Data volume (Byte):	63999983616
MByte/s:		11163.34
Cycles per update:	11.488885
Cycles per cacheline:	91.911077
Loads per update:	3
Stores per update:	1
Load bytes per element:	24
Store bytes per elem.:	8
Load/store ratio:	3.00
Instructions:		2374999408
UOPs:			3749999040
--------------------------------------------------------------------------------
</pre>
</details>

<details>
<summary><code>likwid-mpirun</code></summary>
<pre>
$ likwid-mpirun -mpi slurm -np 4 -t 2 ./a.out
MPI started
Process with rank 0 running on Node f0846.nhr.fau.de core 0
Process with rank 2 running on Node f0859.nhr.fau.de core 0
Process with rank 3 running on Node f0859.nhr.fau.de core 36
Process with rank 1 running on Node f0846.nhr.fau.de core 36
Enter OpenMP parallel region
Start OpenMP threads
Rank 0 Thread 0 running on Node f0846.nhr.fau.de core 0
Rank 0 Thread 1 running on Node f0846.nhr.fau.de core 1
Rank 1 Thread 0 running on Node f0846.nhr.fau.de core 36
Rank 1 Thread 1 running on Node f0846.nhr.fau.de core 37
Rank 2 Thread 0 running on Node f0859.nhr.fau.de core 0
Rank 2 Thread 1 running on Node f0859.nhr.fau.de core 1
Rank 3 Thread 0 running on Node f0859.nhr.fau.de core 36
Rank 3 Thread 1 running on Node f0859.nhr.fau.de core 37
</pre>
</details>

<details>
<summary><code>likwid-powermeter</code></summary>
<pre>
$ likwid-powermeter 
--------------------------------------------------------------------------------
CPU name:	Intel(R) Core(TM) i7-6700K CPU @ 4.00GHz
CPU type:	Intel Skylake processor
CPU clock:	4.01 GHz
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
Runtime: 2.00019 s
Measure for socket 0 on CPU 0
Domain PKG:
Energy consumed: 7.47705 Joules
Power consumed: 3.73817 Watt
Domain PP0:
Energy consumed: 5.42047 Joules
Power consumed: 2.70998 Watt
Domain PP1:
Energy consumed: 0.0872803 Joules
Power consumed: 0.043636 Watt
Domain DRAM:
Energy consumed: 1.02612 Joules
Power consumed: 0.513013 Watt
Domain PLATFORM:
Energy consumed: 0 Joules
Power consumed: 0 Watt
--------------------------------------------------------------------------------
</pre>
</details>

<details>
<summary><code>likwid-features</code></summary>
<pre>
$ likwid-features -c 0 -l
Feature               HWThread 0	
HW_PREFETCHER         on	
CL_PREFETCHER         on	
DCU_PREFETCHER        on	
IP_PREFETCHER         on	
FAST_STRINGS          on	
THERMAL_CONTROL       on	
PERF_MON              on	
FERR_MULTIPLEX        off	
BRANCH_TRACE_STORAGE  on	
XTPR_MESSAGE          off	
PEBS                  on	
SPEEDSTEP             on	
MONITOR               on	
SPEEDSTEP_LOCK        off	
CPUID_MAX_VAL         off	
XD_BIT                on	
DYN_ACCEL             off	
TURBO_MODE            on	
TM2                   off
</pre>
</details>


--------------------------------------------------------------------------------
Documentation
--------------------------------------------------------------------------------
For a detailed  documentation on the usage of the tools have a look at the
html documentation build with doxygen. Call

`make docs`

or after installation, look at the man pages.

There is also a wiki at the github page:
https://github.com/rrze-likwid/likwid/wiki

If you have problems or suggestions please let me know on the likwid mailing list:
http://groups.google.com/group/likwid-users

or if it is bug, add an issue at:
https://github.com/rrze-likwid/likwid/issues

You can also chat with us through Matrix:
- General chat: https://matrix.to/#/#likwid:matrix.org?via=matrix.org
- Development chat: https://matrix.to/#/#likwid-dev:matrix.org?via=matrix.org

--------------------------------------------------------------------------------
Extras
--------------------------------------------------------------------------------
- If you want to use the Marker API with Java, you can find the Java module here:
https://github.com/jacek-lewandowski/likwid-java-api
- For Python you can find an interface to the LIKWID API here:
https://github.com/RRZE-HPC/pylikwid or `pip install pylikwid`
- A Julia interface to LIKWID is provided by the [Paderborn Center for Parallel Computing (PC²)](https://pc2.uni-paderborn.de) and the [MIT JuliaLab](https://julia.mit.edu/):
https://github.com/JuliaPerf/LIKWID.jl or `] add LIKWID`

--------------------------------------------------------------------------------
Survey
--------------------------------------------------------------------------------
We opened a survey at the user mailing list to get a feeling who uses LIKWID and how.
Moreover we would be interested if you are missing a feature or what annoys you when using LIKWID.
Link to the survey:
https://groups.google.com/forum/#!topic/likwid-users/F7TDho3k7ps

--------------------------------------------------------------------------------
Funding
--------------------------------------------------------------------------------

LIKWID development was funded by BMBF Germany under the [FEPA project](https://gauss-allianz.de/en/project/title/FEPA), grant 01IH13009. Since 2017 the development is further funded by BMBF Germany under the [SeASiTe project](https://gauss-allianz.de/en/project/title/SeASiTe), grant 01IH16012A. In 2022, the [EE-HPC project](https://gauss-allianz.de/en/project/title/EE-HPC) is funded by BMBF Germany in the GreenHPC grant.

<div align=center><img src="https://raw.githubusercontent.com/wiki/RRZE-HPC/likwid/images/BMBF.png" alt="BMBF logo" width="150"/></div>
