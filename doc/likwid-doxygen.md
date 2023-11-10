/*! \mainpage LIKWID - Like I Knew What I Am Doing

\section Introduction
This is an effort to develop easy to use but yet powerful performance tools for the GNU Linux operating system. While the focus of LIKWID was on x86 processors, it is now ported to ARM and POWER processors. A backend for Nvidia GPUs is part of LIKWID with version 5.0. With the Rocmon backend, AMD GPUs can be monitored.<BR>

LIKWID follows the philosophy:
- Simple
- Efficient
- Portable
- Extensible

\ref build

\ref faq

\section Tools LIKWID Tools
- \ref likwid-topology : A tool to display the thread and cache topology on multicore/multisocket computers.
- \ref likwid-pin : A tool to pin your threaded application without changing your code. Works for pthreads and OpenMP.
- \ref likwid-perfctr : A tool to measure hardware performance counters on x86, ARM and POWER processors as well as Nvidia/AMD GPUs. It can be used as wrapper application without modifying the profiled code or with a marker API to measure only parts of the code.
- \ref likwid-powermeter : A tool for accessing RAPL counters and query Turbo mode steps on Intel processor. RAPL counters are also available in \ref likwid-perfctr.
- \ref likwid-setFrequencies : A tool to print and manage the clock frequency of CPU hardware threads and the Uncore (Intel only).
- \ref likwid-memsweeper : A tool to cleanup ccNUMA domains and LLC caches to get a clean environment for benchmarks.
- \ref likwid-bench : A benchmarking framework for streaming benchmark kernels written in assembly.
- \ref likwid-genTopoCfg : A config file writer that gets system topology and writes them to file for faster LIKWID startup.
- \ref likwid-features : A tool to toggle the prefetchers and print available CPU features.

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
- \ref Access
- \ref Config
- \ref CPUTopology
- \ref NumaTopology
- \ref AffinityDomains
- \ref CPUParse
- \ref PerfMon
- \ref PowerMon
- \ref ThermalMon
- \ref TimerMon
- \ref MemSweep
- \ref CpuFeatures
- \ref CpuFreq
- \ref NvMarkerAPI
- \ref Nvmon

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
- \ref lua_cpuFeatures
- \ref lua_CpuFreq
- \ref lua_GPUTopology
- \ref lua_Misc (Some functionality not provided by Lua natively)

\subsection Fortran90_Interface Fortran90 Interface
- \ref Fortran_Interface
- \ref Fortran_RocmGPU_Interface
- \ref Fortran_NvGPU_Interface

\section Architectures Supported Architectures
\subsection Architectures_X86 X86 Architectures
\subsubsection Architectures_Intel Intel&reg;
- \subpage pentiumm
- \subpage core2
- \subpage atom
- \subpage nehalem
- \subpage nehalemex
- \subpage westmere
- \subpage westmereex
- \subpage phi
- \subpage silvermont
- \subpage goldmont
- \subpage sandybridge
- \subpage sandybridgeep
- \subpage ivybridge
- \subpage ivybridgeep
- \subpage haswell
- \subpage haswellep
- \subpage broadwell
- \subpage broadwelld
- \subpage broadwellep
- \subpage skylake
- \subpage skylakesp
- \subpage phi_knl
- \subpage phi_knm
- Kaby Lake (use \subpage skylake)
- Coffee Lake (use \subpage skylake)
- Cascade Lake SP/AP (use \subpage skylakesp)
- \subpage icelake
- \subpage icelakesp
- \subpage tigerlake
- \subpage sapphirerapids

\subsubsection Architectures_AMD AMD&reg;
- \subpage k8
- \subpage k10
- \subpage interlagos
- \subpage kabini
- \subpage zen
- \subpage zen2
- \subpage zen3
- \subpage zen4

\subsection Architectures_ARM ARM architectures
- \subpage armA15
- \subpage armA57
- \subpage armThunderX2
- \subpage a64fx
- \subpage neoverseN1
- \subpage neoverseV1
- \subpage applem1
- \subpage hisilicontsv110


\subsection Architectures_POWER POWER architectures
- \subpage power8
- \subpage power9

\subsection Architectures_NVIDIA Nvidia GPU architectures
- For compute capability < 7.0: support based on CUPTI Events API
- For compute capability >= 7.0: support based on CUpti Profiling API

\subsection Architectures_AMD AMD GPU architectures
- ROCm 5.0 and higher capable GPUs

\section Examples Example Codes
Using the Likwid API:
- \ref C-likwidAPI-code
- \ref Lua-likwidAPI-code

Using the Marker API:
- \ref C-markerAPI-code
- \ref F-markerAPI-code

Using the NvMarker API (for Nvidia GPUs):
- \ref C-nvMarkerAPI-code

If you have problems with LIKWID:<BR>
GitHub: <A HREF="https://github.com/RRZE-HPC/likwid">https://github.com/RRZE-HPC/likwid</A><BR>
Bugs: <A HREF="https://github.com/RRZE-HPC/likwid/issues">https://github.com/RRZE-HPC/likwid/issues</A><BR>
Mailinglist: <A HREF="http://groups.google.com/group/likwid-users">http://groups.google.com/group/likwid-users</A><BR>


\page build Build and install instructions
\section allg Introduction
Likwid is build using GNU make and Perl. Besides the Linux kernel and the standard C library, all required dependencies are shipped with the archive (<A HREF="http://www.lua.org/">Lua</A>, <A HREF="http://www.open-mpi.org/projects/hwloc/">hwloc</A> and <A HREF="https://github.com/LLNL/GOTCHA">GOTCHA</A>).
It should build on any Linux distribution with a recent GCC compiler or CLANG compiler and 2.6 or newer kernel without any changes.

There is one generic top level Makefile and one .mk configuration file for each
compiler (at the moment GCC, CLANG and ICC). Please note that we test LIKWID only with GCC. CLANG and ICC is only tested for basic functionality.

There is one exception: If you want to use LIKWID on a Intel Xeon Phi card you have to choose the MIC as compiler in config.mk, which is based on Intel ICC compiler.

\subsection directory Directory structure
All source files are in the src/ directory. All header files are located in
src/includes/ . Lua application source files are in src/applications/. All external tools, namely HWLOC and Lua, are located in ext/. The bench/ folder contains all files of the benchmarking suite of LIKWID.

All build products are generated in the directory ./TAG, where TAG is the compiler configuration, default ./GCC.

\subsection config Configuration
Usually the only thing you have to configure is the <CODE>PREFIX</CODE> install path in the build config file config.mk in the top directory.

\subsubsection color Changing color of likwid-pin output
Depending on the background of your terminal window you can choose a color for <CODE>likwid-pin</CODE> output.

\subsubsection accessD Usage of the access daemon likwid-accessD
Usually on your own system, you can use LIKWID with direct access to the MSR files. If you install LIKWID on a shared system as a HPC compute cluster you may consider to use the access daemon. This is a proxy application which was implemented with security in mind and performs address checks for allowed access. Using the access daemon, the measurements involve more overhead, especially if you use \ref likwid-perfctr in timeline mode or with the marker API.

To enable using the access daemon, configure in config.mk:
    - Set BUILDDAEMON to true
    - Configure the path to the accessDaemon binary at <CODE>ACCESSDAEMON</CODE>
    - Set the ACCESSMODE to accessdaemon

<CODE>ACCESSMODE</CODE> can be direct, accessdaemon and perf_event.

If you want to access Uncore performance counters that are located in the PCI memory range, like they are implemented in Intel SandyBridge EP and IvyBridge EP, you have to use the access daemon or have root privileges because access to the PCI space is only permitted for highly privileged users.

\subsubsection perf_event Usage of Linux Kernel's perf_event interface

LIKWID can be built to run on top of perf_event (config.mk: USE_PERF_EVENT in 4.2 and <CODE>ACCESSMODE=perf_event</CODE> in 4.3), an interface of the Linux kernel to the hardware performance monitors. The events and counters are the same but not all might be supported. LIKWID supports the Uncore and RAPL (energy) units provided by perf. The thermal module is currently not supported with perf_event as perf_event does not provide thermal information. For HWthread-local measurements the paranoid level of 1 is enough to measure applications. To get the same behavior as native access, the paranoid level must be 0 or less. 0 or less is also required for Uncore measurements.

Be aware that LIKWID reads information out of registers that is not provided by any other source like procfs and sysfs. When switching to perf_event backend, these registers cannot be accessed and less information is printed. An example for this are the different CPU hardware thread frequencies in turbo mode (<CODE>likwid-powermeter -i</CODE>). If you want to use perf_event for measurements and the access daemon for other operations, install LIKWID first with <CODE>ACCESSMODE=accessdaemon</CODE> and followed by make distclean, change <CODE>ACCESSMODE=perf_event</CODE> in config.mk and then build and install LIKWID again.

\subsubsection setfreqinstall Usage of frequency daemon likwid-setFreq
The application \ref likwid-setFrequencies uses another daemon to modify the frequency of CPUs. The daemon is build and later installed if <CODE>BUILDFREQ</CODE> is set to true in config.mk.

\subsubsection sharedlib Build Likwid as shared library
Per default the LIKWID library is build as a shared library. You need the library if you want to use the Marker API. You can also use the LIKWID modules like <I>perfmon</I> directly. This is still not officially supported at the moment. In some settings it is necessary to build LIKWID as a shared library. To do so set <CODE>SHARED_LIBRARY</CODE> to true.

\subsubsection instr_bench Instrument likwid-bench for usage with likwid-perfctr
\ref likwid-bench is instrumented for use with \ref likwid-perfctr. This allows you to measure various metrics of your \ref likwid-bench kernels. Enable instrumentation by setting <CODE>INSTRUMENT_BENCH</CODE> to true in config.mk.

\subsubsection fortran Enabling Fortran interface for marker API
If you want to use the Marker API in Fortran programs LIKWID offers a native Fortran90 interface. To enable it set <CODE>FORTRAN_INTERFACE</CODE> to true in config.mk.

\subsubsection system_lua Use system Lua instead of internal Lua

By setting the variables <CODE>LUA_INCLUDE_DIR</CODE>, <CODE>LUA_LIB_DIR</CODE>, <CODE>LUA_LIB_NAME</CODE> (Lua interpreter has to use the same name, e.g. lua5.1) and <CODE>LUA_BIN</CODE> (directory of the Lua interpreter) in config.mk, the internal Lua library and interpreter are deactivated. The LIKWID library is linked against the provided Lua library and the Shebang of all Lua scripts is changed to the system Lua interpreter. LIKWID's Lua interface was patched to support Lua 5.1, 5.2 and 5.3. It is still recommended to use the internal Lua and only basic tests are done before a release.

\subsection targets Build targets
You have to edit config.mk to configure your build and install path.

The following make targets are available:

- <B>make</B> - Build everything
- <B>make likwid-bench</B> - Build likwid-bench
- <B>make likwid-accessD</B> - Build likwid-accessD
- <B>make likwid-setFreq</B> - Build likwid-setFreq
- <B>make docs</B> - Create HTML documentation using doxygen
- <B>make clean</B> - Remove the object file directory *./GCC*, keep the executables
- <B>make distclean</B> - Remove all generated files
- <B>make local</B> - Adjust paths in Lua scripts to work from the build directory. Requires the daemons and the pinning library to be already installed. Mainly used for testing.

The build system has a working dependency tracking, therefore <B>make clean</B> is only needed if you change the Makefile configuration.

\subsection installtargets Installing

NOTE: The pinning functionality and the daemons only work if configured in config.mk and
installed with <B>make install</B>. If you do not use the pinning functionality the tools
can be used without installation.

 - <B>make install</B> - Installs the executables, libraries, man pages and headers to the path you configured in config.mk (<CODE>PREFIX</CODE>).
 - <B>make uninstall</B> - Delete all installed files under <CODE>PREFIX</CODE>.
  - <B>make move</B> - Copy the executables, libraries, man pages and headers from <CODE>PREFIX</CODE> in config.mk to <CODE>INSTALLED_PREFIX</CODE>.
 - <B>make uninstall_moved</B> - Delete all installed files under <CODE>INSTALLED_PREFIX</CODE>.

\subsection accessD Setting up access for hardware performance monitoring on x86 processors
Hardware performance monitoring on x86 is enabled using model-specific registers (MSR). MSR registers are special registers not part of the instruction set architecture. To read and write to these registers the x86 ISA provides special instructions. These instructions can only be executed in protected mode or in other words only kernel code can execute these instructions. Fortunately, any Linux kernel 2.6 or newer provides access to these registers via a set of device files. This allows to implement all of the functionality in user space. Still it does not allow to use those more advanced features of hardware performance monitoring which require to setup interrupt service routines  or kernel located memory.

Per default only root has read/write access to these msr device files. In order to use the LIKWID tools, which need access to these files (likwid-perfctr, likwid-powermeter, ...) as standard user, you need to setup access rights to these files.

likwid-perfctr, likwid-powermeter and likwid-features require the Linux <CODE>msr</CODE> kernel module. This module is part of most standard distro kernels. You have to be root to do the initial setup.

    - Check if the <CODE>msr</CODE> module is loaded with <CODE>lsmod | grep msr</CODE>. There should be an output.
    - It the module is not loaded, load it with <CODE>modprobe msr</CODE>. For automatic loading at startup consult your distros documentation how to do so.
    - Adopt access rights on the MSR device files for normal user. To grant access to anyone, you can use <CODE>chmod o+rw /dev/cpu/*/msr</CODE>. This is only recommended on single user desktop systems.

As in general access to MSRs is not desired on security sensitive systems, you can either implement a more sophisticated access rights settings with e.g. setgid. A common solution used on many other device files, e.g. for audio, is to introduce a group and make a <CODE>chown</CODE> on the msr device files to that group. Now if you execute likwid-perfctr with setgid on that group, the executing user can use the tool but cannot directly write or read the MSR device files.

Some distributions backported the capabilities check for the msr device to older kernels. If there are problems with accessing the msr device for older kernels with file system permissions set to read&write, please check your kernel code (<CODE>arch/x86/kernel/msr.c</CODE>) for the backport and set the MSR capabilities in case.

A secure solution is to use the access daemon \ref likwid-accessD, which encapsulates the access to the MSR device files and performs a address check for allowed registers.

Some newer kernels implement the so-called capabilities, a fine-grained permission system that can allow access to the MSR files for common users. On the downside it may be not enough anymore to set the suid-root flag for the access daemon, the executable must be registerd at the <CODE>libcap</CODE>.

<CODE>sudo setcap cap_sys_rawio+ep EXECUTABLE</CODE>

This is only possible on local file systems. A feasible way is to use the \ref likwid-accessD for all accesses and just enable the capabilities for this one binary. This will enable the usage for all LIKWID tools and also for all instrumented binaries. If \ref likwid-perfctr utility should only be used in wrapper mode, it is suitable to set the capabilities for \ref likwid-perfctr only. Please remember to set the file permission of the MSR device files to read/write for all users, even if capabilites are configured correctly.

\subsection accessD Installation on ARM- and POWER-based systems
ARM support was added in January. The main switch is the <CODE>COMPILER</CODE> setting in config.mk. There are two possibilities: <CODE>GCCARMv7</CODE> and <CODE>GCCARMv8</CODE>. For build flags changes, please use the appropriate file <CODE>make/include_&lt;COMPILER&gt;.mk</CODE>. The backend for ARM is <CODE>perf_event</CODE>. There is a native backend as well but it is currently not usable as the user would need to measure multiple times per second to catch all register overflows. As soon as LIKWID starts a management thread to read the registers in the background, I will publish this backend as well.<BR>
For POWER systems, the main switch is the <CODE>COMPILER</CODE> setting in config.mk. The change to <CODE>GCCPOWER</CODE> configures the build to work on POWER and switch to the proper <CODE>ACCESSMODE</CODE>.

\subsubsection secureboot LIKWID and SecureBoot
LIKWID in direct or accessdaemon mode uses the common MSR and PCI interfaces from the Linux kernel. Unfortunately, some recent operating systems (e.g. Ubuntu 18.04) patches the Linux kernel so that these interfaces (at least MSR) are not allowed to use anymore. You probably just get read/write errors because LIKWID currently has no check of the SecureBoot state. <B>If you want to use LIKWID, you have to disable SecureBoot</B>.

\subsubsection depends Dependencies
Although we tried to minimize the external dependencies of LIKWID, some advanced tools or only specific tool options require external packages.<BR>
\ref likwid-perfscope uses the Perl script <A HREF="https://github.com/dkogan/feedgnuplot">feedGnuplot</A> to forward the real-time data to gnuplot. <A HREF="https://github.com/dkogan/feedgnuplot">feedGnuplot</A> is included into LIKWID, but <A HREF="http://www.gnuplot.info/">gnuplot</A> itself is not.<BR>
In order to create the HTML documentation of LIKWID, the tool <A HREF="www.doxygen.org">Doxygen</A> is required.
*/

/*! \page C-markerAPI-code Marker API in a C/C++ application
\include C-markerAPI.c
*/

/*! \page F-markerAPI-code Marker API in a Fortran90 application
\include F-markerAPI.F90
*/

/*! \page C-likwidAPI-code LIKWID API in a C/C++ application
\include C-likwidAPI.c
*/
/*! \page Lua-likwidAPI-code LIKWID API in a Lua application
\include Lua-likwidAPI.lua
*/

/*! \page C-nvMarkerAPI-code NvMarker API in a C/C++ application (Nvidia GPU monitoring)
\include C-nvMarkerAPI.c
*/


/*! \page faq FAQ
\section faq1 Which architectures are supported?
LIKWID supports a range of x86 CPU architectures but likely not all. We concentrated the development effort on Intel and AMD machines. Almost all architecture code is tested. With LIKWID 5.0 the ARM and POWER architectures are supported as well. For a list of architectures see section \ref Architectures or call <CODE>likwid-perfctr -i</CODE>.

\section faq2 Are all hardware events supported?
LIKWID offers almost all events that are defined in the <A HREF="http://www.Intel.com/content/www/us/en/processors/architectures-software-developer-manuals.html">Intel&reg; Software Developer System Programming Manual</A> and the <A HREF="http://developer.amd.com/resources/documentation-articles/developer-guides-manuals/">AMD&reg; BIOS and Kernel Developer’s Guides</A>. Some may be missing caused by special handling likely with additional registers. But, LIKWID also provides some events that are not documented but we do not guarantee that they count the right stuff.

\section faq3 Does LIKWID support Intel's PEBS?
No, PEBS is an interface that must be initialized at kernel level. Since LIKWID is a user-space tool, there is no possibility to maintain PEBS.

\section faq4 Which unit does LIKWID use internally for B, kB, MB, GB?
As the units imply, you get from one unit to the other by multiplying or dividing it by 1000. E.g. 1kB = 1000B. There is no kiB or MiB possible by now.

\section faq5 Does LIKWID support power capping (Intel only)?
No, by now LIKWID does not support limiting the power consumption of your machine using the RAPL interface. We added some functions but they are not exported because they need to be rechecked.

\section faq6 Is LIKWID case-sensitive?
Yes, all strings are case-sensitive. The only exception are the event options, they are case-insensitive. For upcomming versions we change to case-insensitive for all string parsing where possible.

\section faq7 I have given multiple eventsets on the command line but the values are too low? Are they multiplexed?
LIKWID does not support multiplexing of eventsets. It rotates through its eventset list and measures each for a specific amount of time. The output contains the results of all measurements of that eventset, no interpolation to the complete runtime is done. Since most other tools that support multiplexing use linear interpolation, you can scale the results yourself with <CODE>(1.0 - (measurement_time/all_time)) * result</CODE>. As you can see, the calculation is pretty simple, but it introduces a high degree of inaccuracy. You cannot know if the operations that were performed during the measurement of the eventset are similar to the operations that were done when measuring another eventset. As a recommendation, perform the interpolation only with Marker API measurements where you know what the application does.

\section faq8 Are there plans to port LIKWID to other operating systems?
We do not really plan to port LIKWID to other operating systems. We come from the HPC world and there the main operating systems base on the Linux kernel. The latest Top500 list contains 13 systems using Unix and 1 system with Microsoft&reg; Windows.

\section faq9 Are there plans to port LIKWID to other CPU architectures?
There is currently no need, LIKWID supports already the three major HPC-relevant CPU architectures: x86, ARM and POWER.

\section faq10 Do you plan to introduce a graphical frontend for LIKWID?
No, we do not!

\section faq11 Why does the startup of likwid-perfctr take so long?
In order to get reliable time measurements, LIKWID must determine the base clock frequency of your CPU. This is done by a measurement loop that takes about 1 second. You can avoid the measurement loop by creating a topology configuration file with \ref likwid-genTopoCfg.

\section faq12 What about the security issue found with the MSR device files (CVE-2013-0268)? Can someone use the access daemon to exploit this
No it is not possible. At the current state, the access daemon only allows accesses to performance counter MSRs and not to MSRs like SYSENTER_EIP_MSR that are used in the exploit. Consequently, the access daemon cannot be used to exploit the security issue CVE-2013-0268.

\section faq13 I get messages like "Counter register FOO not supported or PCI device not available", what does it mean?
Every time an event set is added to LIKWID, it checks whether the registers are accessible. If not, such a message is printed. In most cases, this is not a failure, it is just informing you that the counter will be skipped later at the measurements. It happens more often with the predefined performance groups because they are created for a maximally equipped machine. If your system has e.g. less memory channels than the maximal possible, not all MBOX registers will work but the counts will be valid because the counts of the non-existant register are commonly handled by another register.

\section faq14 Likwid reports something like 'ERROR: The selected register XYZ is in use'. What causes this and is the -f/--force option safe?
Some time ago, Intel release a document Intel PMU sharing guide which lists some checks that should be performed before programming the hardware performance counters. Starting with commit 8efa0ec46a30438e1b85eac3ba31ebe0b7a03303 LIKWID now checks the counters and exits if one of the selected counters is in use. When you set on command line <CODE>-f</CODE> or <CODE>--force</CODE> or set the environment variable <CODE>LIKWID_FORCE</CODE>, LIKWID ignores the already running counter and clears it. On Linux system this commonly occurs for the FIXC0 which is used by the Linux kernel in the NMI watchdog. Until now we havn't seen problems clearing the FIXC0 counter although used by the NMI watchdog. If you are not sure about that, you can deactivate the NMI watchdog with <CODE>sysctl -w kernel.nmi_watchdog=0</CODE> until reboot or instantly by writing <CODE>kernel.nmi_watchdog = 0</CODE> into <CODE>/etc/sysctl.conf</CODE>.

\section faq15 I want to help, were do I start?
The best way is to talk to us at the <A HREF="http://groups.google.com/group/likwid-users">mailing list</A>. There are a bunch of small work packages on our ToDo list that can be used as a good starting point for learning how LIKWID works. If you are not a programmer but you have a good idea, let us know and we will discuss it.
*/
