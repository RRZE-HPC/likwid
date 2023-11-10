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
- Apple M1

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

LIKWID development was funded by BMBF Germany under the FEPA project, grant 01IH13009. Since 2017 the development is further funded by BMBF Germany under the SeASiTe project, grant 01IH16012A.

<div align=center><img src="https://raw.githubusercontent.com/wiki/RRZE-HPC/likwid/images/BMBF.png" alt="BMBF logo"/></div>
