/*! \mainpage LIKWID - Like I Knew What I Am Doing

\section Introduction
This is an effort to develop easy to use but yet powerful performance tools for the GNU Linux operating system. While the focus of LIKWID is on x86 processors some of the tools are portable and not limited to any specific architecture. LIKWID follows the philosophy:
- Simple
- Efficient
- Portable
- Extensible

\ref build instructions

\section Tools LIKWID Tools
- \ref likwid-topology : A tool to display the thread and cache topology on multicore/multisocket computers.
- \ref likwid-pin : A tool to pin your threaded application without changing your code. Works for pthreads and OpenMP.
- \ref likwid-perfctr : A tool to measure hardware performance counters on recent Intel and AMD processors. It can be used as wrapper application without modifying the profiled code or with a marker API to measure only parts of the code.
- \ref likwid-powermeter : A tool for accessing RAPL counters and query Turbo mode steps on Intel processor. RAPL counters are also available in \ref likwid-perfctr.
- \ref likwid-setFrequencies : A tool to print and manage the clock frequency of CPU cores.
- \ref likwid-agent : A monitoring agent for LIKWID with multiple output backends.
- \ref likwid-memsweeper : A tool to cleanup ccNUMA domains and LLC caches to get a clean environment for benchmarks.
- \ref likwid-bench : A benchmarking framework for streaming benchmark kernels written in assembly.
- \ref likwid-genTopoCfg : A config file writer that gets system topology and writes them to file for faster LIKWID startup.
<!-- - \ref likwid-features : A tool to toggle the prefetchers on Core 2 processors.-->

Wrapper scripts using the basic likwid tools:
- \ref likwid-mpirun : A wrapper script enabling simple and flexible pinning of MPI and MPI/threaded hybrid applications. With integrated \ref likwid-perfctr support.
- \ref likwid-perfscope : A frontend application for the timeline mode of \ref likwid-perfctr that performs live plotting using gnuplot.

LIKWID requires in most environments some daemon application to perform its operations with higher priviledges:
- \ref likwid-accessD : Daemon to perform MSR and PCI read/write operations with higher priviledges.
- \ref likwid-setFreq : Daemon to set the CPU frequencies with higher priviledges.

Optionally, a global configuration file \ref likwid.cfg can be given to modify some basic run time parameters of LIKWID.

\section Library LIKWID Library
\subsection C_Interface C/C++ Interface
- \ref MarkerAPI
- \ref AccessClient
- \ref Config
- \ref CPUTopology
- \ref NumaTopology
- \ref AffinityDomains
- \ref PerfMon
- \ref PowerMon
- \ref ThermalMon
- \ref TimerMon
- \ref Daemon
- \ref MemSweep

\subsection Lua_Interface Lua Interface
- \ref lua_Info
- \ref lua_InputOutput
- \ref lua_Config
- \ref lua_Access
- \ref lua_CPUTopology
- \ref lua_NumaInfo
- \ref lua_AffinityInfo
- \ref lua_Perfmon
- \ref lua_PowerInfo
- \ref lua_ThermalInfo
- \ref lua_Timer
- \ref lua_MemSweep
- \ref lua_Misc (Some functionality not provided by Lua natively)

\subsection Fortran90_Interface Fortran90 Interface
- \ref Fortran_Interface

\section Architectures Supported Architectures
\subsection Architectures_Intel Intel&reg;
- \subpage pentiumm
- \subpage core2
- \subpage atom
- \subpage nehalem
- \subpage nehalemex
- \subpage westmere
- \subpage westmereex
- \subpage phi
- \subpage silvermont
- \subpage sandybridge
- \subpage sandybridgeep
- \subpage ivybridge
- \subpage ivybridgeep
- \subpage haswell
- \subpage haswellep
- \subpage broadwell

\subsection Architectures_AMD AMD&reg;
- \subpage k8
- \subpage k10
- \subpage interlagos
- \subpage kabini
*/


/*! \page build Build and Install
\section allg Introduction
Likwid is build using GNU make and has no external dependencies apart from the Linux kernel and the standard C library.
It should build on any Linux distribution with a recent GCC compiler or CLANG compiler and 2.6 or newer kernel without any changes.

There is one generic top level Makefile and one .mk configuration file for each compiler (at the moment GCC, CLANG and ICC). Please note that I can only test LIKWID with GCC and CLANG. ICC is only tested for basic functionality and is not supported by us.

There is one exception: If you want to use LIKWID on a Intel Xeon Phi card you have to choose the MIC as Compiler in config.mk, which is based on Intel ICC compiler.

\subsection directory Directory structure
All source files are in the src/ directory. All header files are located in src/includes/. Each application main source files are in src/applications/. All external tools, namely HWLOC and Lua, are located in ext/. The bench/ folder contains all files of the benchmarking suite of LIKWID.

All build products are generated in the directory ./TAG, where TAG is the compiler configuration, default ./GCC.

\subsection config Configuration
Usually the only thing you have to configure is the PREFIX install path in the config file config.mk in the top directory.
\subsubsection color Changing color of likwid-pin output
Depending on the background of your terminal window you can choose a color for likwid-pin output.
\subsubsection accessD Usage of the access daemon likwid-accessD
Usually you can use LIKWID in the standard setup with direct access to the MSR files. If you install LIKWID on a shared system as a HPC compute cluster you may consider to use the accessDaemon. This is a proxy application which was implemented with security in mind and performs address checks for allowed access. Using the accessDaemon the measurements involve more overhead, especially if you use likwid-perfctr in timeline mode or with the marker API.

To enable using the accessDaemon:
    - Set BUILDDAEMON to true
    - Configure the path to the accessDaemon binary (usually keep default)
    - Set the default ACCESSMODE

ACCESSMODE can be direct, accessdaemon and sysdaemon (not yet officially supported). You can overwrite the default setting on the command line using the -M switch.

If you want to access Uncore performance counters that are located in the PCI memory range, like they are implemented in Intel SandyBridge EP and IvyBridge EP, you have to use the access daemon or have root priviledges because access to the PCI space is only permitted for highly priviledged users.

\subsubsection sharedlib Build Likwid as shared library
Per default the LIKWID library is build as a shared library. You need the library is necessary if you want to use the Marker API. You can also use the LIKWID modules directly. This is still not officially supported at the moment. In some settings it is necessary to build LIKWID as a shared library. To do so set SHARED_LIBRARY to true.

\subsubsection instr_bench Instrument likwid-bench for usage with likwid-perfctr
likwid-bench is instrumented for use with likwid-perfctr. This allows you to measure various metrics of your likwid-bench kernels. Enable instrumentation by setting INSTRUMENT_BENCH to true.

\subsubsection fortran Enabling Fortran interface for marker API
If you want to use the Marker API in Fortran programs LIKWID offers a native Fortran90 interface. To enable it uncomment he FORTRAN_INTERFACE line.

\subsection accessD Setting up access for hardware performance monitoring
Hardware performance monitoring on x86 is configured using model-specific registers (MSR). MSR registers are special registers not part of the instruction set architecture. To read and write to these registers the x86 ISA provides special instructions. These instructions can only be executed in protected mode or in other work only kernel code can execute these instructions. Fortunately any Linux kernel 2.6 or newer provides access to these registers via a set of device files. This allows to implement all of the functionality in user space. Still it does not allow to use those more advanced features of hardware performance monitoring which require to setup interrupt service routines.

Per default only root has read/write access to these msr device files. In order to use the LIKWID tools, which need access to these files (likwid-perfctr and likwid-powermeter) as standard user you need to setup access rights to these files.

likwid-perfctr, likwid-powermeter and likwid-features require the Linux msr kernel module. This module is part of most standard distro kernels. You have to be root to do the initial setup.

    - Check if the msr module is loaded with lsmod | grep msr. There should be an output.
    - It the module is not loaded load it with modprobe msr . For automatic loading at startup consult your distros documentation how to do so.
    - Adopt access rights on the msr device files for normal user. To allow everybody access you can use chmod o+rw /dev/cpu/*/msr . This is only recommended on single user desktop systems.

As a general access to the msr registers is not desired on security sensitive systems you can either implement a more sophisticated access rights settings with e.g. setgid. A common solution used on many other device files, e.g. for audio, is to introduce a group and make a chown on the msr device files to that group. Now if you execute likwid-perfctr with setgid on that group the executing user can use the tool but cannot directly write or read the msr device files.

Some distributions backported the capabilities check for the msr device to older kernels. If there are problems with accessing the msr device for older kernels with file system permissions set to read&write, please check your kernel code (arch/x86/kernel/msr.c) for the backport and set the MSR capabilities in case.

A secure solution is to use the accessDaemon, which encapsulates the access to the msr device files and performs a address check for allowed registers. For more information how to setup and use this solution have a look at the WIKI page (http://code.google.com/p/likwid/w/list).

Some newer kernels implement the so-called capabilities, a fine-grained permission system that can allow access to the MSR files for common users. On the downside it may be not enough anymore to set the suid-root flag for the access daemon, the executable must be registerd at the libcap.

sudo setcap cap_sys_rawio+ep EXECUTABLE

This is only possible on local file systems. A feasible way is to use the likwid-accessD for all accesses and just enable the capabilities for this one binary. This will enable the usage for all LIKWID tools and also for all instrumented binaries. If the likwid-perfctr utility should only be used in wrapper mode, it is suitable to set the capabilities for likwid-perfctr only. Please remember to set the file permission of the MSR device file to read/write for all users, even if capabilites are configured correctly.

\subsubsection depends Dependencies
Although we tried to minimize the external dependencies of LIKWID, some advanced tools or only specific tool options require external packages.<BR>
\ref likwid-perfscope uses the Perl script \a feedGnuplot to forward the real-time data to gnuplot. \a feedGnuplot is included into LIKWID, but gnuplot itself is not.<BR>
\ref likwid-agent provided multiple backends to output the periodically measured data. The syslog backend requires the shell tool \a logger to be installed. The <A HREF="https://oss.oetiker.ch/rrdtool/">RRD</A> backend requires \a rrdtool and the GMetric backend the \gmetric tool, part of the <A HREF="http://ganglia.sourceforge.net/">Ganglia Monitoring System</A>.<BR>
In order to create the HTML documentation of LIKWID, the toold \a doxygen is required.
*/
