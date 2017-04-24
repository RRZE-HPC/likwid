--------------------------------------------------------------------------------
Introduction
--------------------------------------------------------------------------------

Likwid is a simple to install and use toolsuite of command line applications
for performance oriented programmers. It works for Intel and AMD processors
on the Linux operating system.

[![Build Status](https://travis-ci.org/RRZE-HPC/likwid.svg?branch=master)](https://travis-ci.org/RRZE-HPC/likwid)

It consists of:

- likwid-topology: print thread, cache and NUMA topology
- likwid-perfctr: configure and read out hardware performance counters on Intel and AMD processors
- likwid-powermeter: read out RAPL Energy information and get info about Turbo mode steps
- likwid-pin: pin your threaded application (pthread, Intel and gcc OpenMP to dedicated processors)
- likwid-bench: Micro benchmarking platform
- likwid-features: Print and manipulate cpu features like hardware prefetchers
- likwid-genTopoCfg: Dumps topology information to a file
- likwid-mpirun: Wrapper to start MPI and Hybrid MPI/OpenMP applications (Supports Intel MPI, OpenMPI and MPICH)
- likwid-perfscope: Frontend to the timeline mode of likwid-perfctr, plots live graphs of performance metrics using gnuplot
- likwid-agent: Monitoring agent for hardware performance counters
- likwid-memsweeper: Sweep memory of NUMA domains and evict cachelines from the last level cache
- likwid-setFrequencies: Tool to control the CPU frequency

--------------------------------------------------------------------------------
Download, Build and Install
--------------------------------------------------------------------------------
You can get the releases of LIKWID at:
http://ftp.fau.de/pub/likwid/

For build and installation hints see INSTALL file or check the build instructions
page in the wiki https://github.com/RRZE-HPC/likwid/wiki/Build

For quick install:
```
$ tar -xjf likwid-<VERSION>.tar.bz2
$ cd likwid-<VERSION>
$ vi config.mk (configure build, e.g. change installation prefix)
$ make
$ sudo make install
```
--------------------------------------------------------------------------------
Documentation
--------------------------------------------------------------------------------
For a detailed  documentation on the usage of the tools have a look at the
html documentation build with doxygen. Call

make docs

or after installation, look at the man pages.

There is also a wiki at the github page:
https://github.com/rrze-likwid/likwid/wiki

If you have problems or suggestions please let me know on the likwid mailing list:
http://groups.google.com/group/likwid-users

or if it is bug, add an issue at:
https://github.com/rrze-likwid/likwid/issues

--------------------------------------------------------------------------------
Extras
--------------------------------------------------------------------------------
- If you want to use the Marker API with Java, you can find the Java module here:
https://github.com/jlewandowski/likwid-java-api
- For Python you can find an interface to the LIKWID API here:
https://github.com/RRZE-HPC/pylikwid

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

LIKWID development was funded by BMBF Germany under the FEPA project, grant 01IH13009.

<div align=center><img src="https://raw.githubusercontent.com/wiki/RRZE-HPC/likwid/images/BMBF.png" alt="BMBF logo"/></div>
