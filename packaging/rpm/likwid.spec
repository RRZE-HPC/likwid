Name: likwid
Version: 5.0.0
Release: 1
Source: likwid-5.0.0.tar.gz
License: GPL-3.0+
Group: Development/Tools
Packager: Holger Obermaier <holger.obermaier@kit.edu>
Summary: Performance tools for the Linux console
URL: https://github.com/RRZE-HPC/likwid/
%if 0%{?fedora}
BuildRequires: gcc-gfortran
%if %{fedora} > 0
BuildRequires: lua-devel
%endif
%if %{fedora} >= 23
BuildRequires: perl-Text-Balanced perl-File-Copy
%endif
%endif

%if 0%{?centos}
BuildRequires: gcc-gfortran
%if %{centos} == 7
BuildRequires: lua-devel
%endif
%endif

%if 0%{?rhel}
BuildRequires: gcc-gfortran
%endif

%if 0%{?opensuse_bs}
BuildRequires: gcc-fortran lua-devel
Patch0: remove_avx512_kernels.diff
%endif

%if 0%{?suse}
BuildRequires: gcc-fortran
%if %{sles_version} = 11
BuildRequires: lua lua-devel
%endif
%if %{sles_version} = 12
BuildRequires: liblua5_1 lua5_1-devel
%endif
%endif
BuildRequires: perl
BuildRequires: perl-Data-Dumper

Requires(post): /sbin/ldconfig
Requires(postun): /sbin/ldconfig
Requires: lua

# Turn off creation of debuginfo packages on RHEL
%if 0%{?fedora} || 0%{?rhel} || 0%{?centos}
%global debug_package %{nil}
%endif
%if 0%{?opensuse_bs}
BuildRequires:  -post-build-checks -rpmlint-Factory -brp-check-suse
%endif

%description
Likwid is a simple to install and use toolsuite of command line applications
for performance oriented programmers. It works for Intel and AMD processors
on the Linux operating system.

It consists of:

 * likwid-topology:
     print thread, cache and NUMA topology
 * likwid-perfctr:
     configure and read out hardware performance counters on x86, ARM and POWER
     processors
 * likwid-powermeter:
     read out RAPL Energy information and get info about Turbo mode steps
 * likwid-pin:
     pin your threaded application (pthread, Intel and gcc OpenMP to dedicated
     processors)
 * likwid-genTopoCfg:
     Dumps topology information to a file
 * likwid-memsweeper:
     Sweep memory of NUMA domains and evict cachelines from the last level
     cache

#%package bench
#Summary: The micro benchmarking tool of LIKWID for assembly benchmarks.
#Group: Applications/Tools
#%description bench
#The likwid-bench micro benchmarking tool enabled easy determination of system
#capabilities like maximal bandwidths or maximal FLOP rate. It takes care about
#the data and thread placement and performs time measurements and the evaluation
#of the benchmark including the read data volume, number of executed assembly
#instructions, executed micro-ops and many more.

#%package perfscope
#Summary: Frontend to the timeline mode of likwid-perfctr, plots live graphs of performance metrics using gnuplot.
#Group: Applications/Tools
#Requires: gnuplot
#%description perfscope
#likwid-perfscope is a command line application written in Lua that uses the timeline mode of likwid-perfctr to create on-the-fly pictures with the current measurements. It uses the feedGnuplot Perl script to send the current data to gnuplot. In order to make it more convenient for users, preconfigured plots of interesting metrics are embedded into likwid-perfscope. Since the plot windows are normally closed directly after the execution of the monitored applications, likwid-perfscope waits until Ctrl+c is pressed.

#%package setFrequencies
#Summary: Tool to control the CPU frequency.
#Group: Applications/Tools
#%description setFrequencies
#Often systems are configured to use as little power as needed and therefore reduce the clock frequency of single cores. For benchmarking purpose, it is important to have a defined environment where all CPU cores work at the same speed. The operation is commonly only allowed to privileged users since it may interfere with the needs of other users.

#%package mpirun
#Summary: Wrapper to start MPI and Hybrid MPI/OpenMP applications.
#Group: Applications/Tools
#%description mpirun
#Pinning to dedicated compute resources is important for pure MPI and even more for hybrid MPI/threaded applications. While all major MPI implementations include their mechanism for pinning, likwid-mpirun provides a simple and portable solution based on the powerful capabilities of likwid-pin. This is still experimental at the moment. Still it can be adopted to any MPI and OpenMP combination with the help of a tuning application in the test directory of LIKWID. likwid-mpirun works in conjunction with PBS, LoadLeveler and SLURM. The tested MPI and compilers are Intel C/C++ compiler, GCC, Intel MPI and OpenMPI. The support for mvapich is untested.

#%package devel
#Summary: Header files for LIKWID library
#Group: Applications/Tools
#%description devel
#The header files for embedding LIKWID into own applications with either the Marker API or the full API. The library is part of the main release as it is needed by the LIKWID applications.

#%package examples
#Summary: Example applications using the LIKWID library
#Group: Applications/Tools
#%description examples
#Some examples how to use the LIKWID library for hardware performance measurements.

%prep
%setup
%if 0%{?opensuse_bs}
%patch0 -p1
%endif


%build
%ifarch i386 i486 i586 i686
COMPILER="GCCX86"
%else
%ifarch %{arm}
COMPILER="GCCARMv8"
%else
%ifarch %{power64}
COMPILER="GCCPOWER"
%else
COMPILER="GCC"
%endif
%endif
%endif
# Parallel build fails
%{__make} \
    PREFIX="%{_prefix}" \
    MANPREFIX="%{_mandir}" \
    BINPREFIX="%{_bindir}" \
    LIBPREFIX="%{_libdir}" \
    INSTALLED_PREFIX="%{_prefix}" \
    INSTALLED_BINPREFIX="%{_bindir}" \
    INSTALLED_LIBPREFIX="%{_libdir}" \
    COMPILER="${COMPILER}" \
    INSTRUMENT_BENCH="true" \
    FC="gfortran" \
    FCFLAGS="-J ./  -fsyntax-only" \
    FORTRAN_INTERFACE="true"

%install
%ifarch i386 i486 i586 i686
COMPILER="GCCX86"
%else
%ifarch %{arm}
COMPILER="GCCARMv8"
%else
%ifarch %{power64}
COMPILER="GCCPOWER"
%else
COMPILER="GCC"
%endif
%endif
%endif
%{__make} install \
    PREFIX="$RPM_BUILD_ROOT/%{_prefix}" \
    MANPREFIX="$RPM_BUILD_ROOT/%{_mandir}" \
    BINPREFIX="$RPM_BUILD_ROOT/%{_bindir}" \
    LIBPREFIX="$RPM_BUILD_ROOT/%{_libdir}" \
    INSTALLED_PREFIX="%{_prefix}" \
    INSTALLED_BINPREFIX="%{_bindir}" \
    INSTALLED_LIBPREFIX="%{_libdir}" \
    INSTALLED_MANPREFIX="%{_mandir}" \
    COMPILER="${COMPILER}" \
    INSTRUMENT_BENCH="true" \
    FC="gfortran" \
    FCFLAGS="-J ./  -fsyntax-only" \
    FORTRAN_INTERFACE="true"

chmod 755 $RPM_BUILD_ROOT/%{_prefix}/sbin/likwid-accessD
chmod 755 $RPM_BUILD_ROOT/%{_prefix}/sbin/likwid-setFreq

%post
/sbin/ldconfig
chown root:root $RPM_BUILD_ROOT/%{_prefix}/sbin/likwid-accessD
chmod u+s $RPM_BUILD_ROOT/%{_prefix}/sbin/likwid-accessD
chown root:root $RPM_BUILD_ROOT/%{_prefix}/sbin/likwid-setFreq
chmod u+s $RPM_BUILD_ROOT/%{_prefix}/sbin/likwid-setFreq

%postun
/sbin/ldconfig

%clean
rm -rf $RPM_BUILD_ROOT

%files
%defattr(-,root,root)
%{_bindir}/likwid-features
%{_bindir}/likwid-genTopoCfg
%{_bindir}/likwid-lua
%{_bindir}/likwid-memsweeper
%{_bindir}/likwid-perfctr
%{_bindir}/likwid-pin
%{_bindir}/likwid-powermeter
%{_bindir}/likwid-topology
%{_bindir}/likwid-bench
%{_bindir}/likwid-mpirun
%{_bindir}/feedGnuplot
%{_bindir}/likwid-perfscope
%{_bindir}/likwid-setFrequencies
%{_sbindir}/likwid-setFreq
%{_sbindir}/likwid-accessD
%{_libdir}/*
%{_datadir}/likwid/docs
%{_datadir}/likwid/perfgroups
%{_datadir}/likwid/filter
%{_datadir}/likwid/*.cmake
%{_datadir}/likwid/examples
%{_datadir}/lua/likwid.lua
%{_includedir}/*
%doc COPYING README.md INSTALL
%doc %{_mandir}/man*/likwid-features*
%doc %{_mandir}/man*/likwid-genTopoCfg*
%doc %{_mandir}/man*/likwid-lua*
%doc %{_mandir}/man*/likwid-memsweeper*
%doc %{_mandir}/man*/likwid-perfctr*
%doc %{_mandir}/man*/likwid-pin*
%doc %{_mandir}/man*/likwid-powermeter*
%doc %{_mandir}/man*/likwid-topology*
%doc %{_mandir}/man*/likwid-accessD*
%doc %{_mandir}/man*/likwid-bench*
%doc %{_mandir}/man*/likwid-perfscope*
%doc %{_mandir}/man*/feedGnuplot*
%doc %{_mandir}/man*/likwid-setFrequencies*
%doc %{_mandir}/man*/likwid-setFreq*
%doc %{_mandir}/man*/likwid-mpirun*



