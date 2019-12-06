#!/bin/bash -l

# * =======================================================================================
# *
# *      Filename:  likwid-benchmark.sh
# *
# *      Description:  Script that collects current system settings
# *
# *      Version:   <VERSION>
# *      Released:  <DATE>
# *
# *      Author:   Thomas Gruber (tr), thomas.roehl@gmail.com
# *      Project:  likwid
# *
# *      Copyright (C) 2018 RRZE, University Erlangen-Nuremberg
# *
# *      This program is free software: you can redistribute it and/or modify it under
# *      the terms of the GNU General Public License as published by the Free Software
# *      Foundation, either version 3 of the License, or (at your option) any later
# *      version.
# *
# *      This program is distributed in the hope that it will be useful, but WITHOUT ANY
# *      WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# *      PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# *
# *      You should have received a copy of the GNU General Public License along with
# *      this program.  If not, see <http://www.gnu.org/licenses/>.
# *
# * =======================================================================================

DO_LIKWID=1

function header {
    echo
    echo "################################################################################"
    echo "# $1"
    echo "################################################################################"
}

function print_powercap_folder {
    FOLDER=$1
    if [ -e $FOLDER/name ]; then
        NAME=$(cat $FOLDER/name)
        echo "RAPL domain ${NAME}"
    else
        return
    fi
    if [ -e $FOLDER/constraint_0_name ]; then
        LIMIT0_NAME=$(cat $FOLDER/constraint_0_name)
        if [ -e $FOLDER/constraint_0_power_limit_uw ]; then
            LIMIT0_LIMIT=$(cat $FOLDER/constraint_0_power_limit_uw)
        else
            LIMIT0_LIMIT="NA"
        fi
        if [ -e $FOLDER/constraint_0_max_power_uw ]; then
            LIMIT0_MAXPOWER=$(cat $FOLDER/constraint_0_max_power_uw)
        else
            LIMIT0_MAXPOWER="NA"
        fi
        if [ -e $FOLDER/constraint_0_time_window_us ]; then
            LIMIT0_TIMEWIN=$(cat $FOLDER/constraint_0_time_window_us)
        else
            LIMIT0_TIMEWIN="NA"
        fi
        echo "- Limit0 ${LIMIT0_NAME} MaxPower ${LIMIT0_MAXPOWER}uW Limit ${LIMIT0_LIMIT}uW TimeWindow ${LIMIT0_TIMEWIN}us"
    fi
    if [ -e $FOLDER/constraint_1_name ]; then
        LIMIT1_NAME=$(cat $FOLDER/constraint_1_name)
        if [ -e $FOLDER/constraint_1_power_limit_uw ]; then
            LIMIT1_LIMIT=$(cat $FOLDER/constraint_1_power_limit_uw)
        else
            LIMIT1_LIMIT="NA"
        fi
        if [ -e $FOLDER/constraint_0_max_power_uw ]; then
            LIMIT1_MAXPOWER=$(cat $FOLDER/constraint_1_max_power_uw)
        else
            LIMIT1_MAXPOWER="NA"
        fi
        if [ -e $FOLDER/constraint_0_time_window_us ]; then
            LIMIT1_TIMEWIN=$(cat $FOLDER/constraint_1_time_window_us)
        else
            LIMIT1_TIMEWIN="NA"
        fi
        echo "- Limit1 ${LIMIT1_NAME} MaxPower ${LIMIT1_MAXPOWER}uW Limit ${LIMIT1_LIMIT}uW TimeWindow ${LIMIT1_TIMEWIN}us"
    fi
}


if [ $(which likwid-topology 2>/dev/null | wc -l) -ne 1 ]; then
    if [ -e /opt/modules/init/bash ]; then
        source /opt/modules/init/bash
    fi
    module load likwid 2>/dev/null
    if [ $(which likwid-topology 2>/dev/null | wc -l) -ne 1 ]; then
        DO_LIKWID=0
    fi
fi


##### Users
header "Logged in users"
ps -ef | grep "/bin/bash" | grep -v -E "^root" | cut -d ' ' -f 1 | sort -u

if [ ${DO_LIKWID} -a -x likwid-pin ]; then
    header "CPUset"
    likwid-pin -p
fi

##### OS execution environment
header "CGroups"
echo -n "Allowed CPUs: "
cat /sys/fs/cgroup/cpuset/cpuset.effective_cpus
echo -n "Allowed Memory controllers: "
cat /sys/fs/cgroup/cpuset/cpuset.effective_mems

##### System topology
header "Topology"
if [ ${DO_LIKWID} -a -x likwid-topology ]; then
    likwid-topology
else
    lscpu
fi
if [ $(which numactl 2>/dev/null | wc -l ) == 1 ]; then
    numactl -H
fi

##### Hyper Threading
header "Hyper Threading"
ALL_CPUS=$(grep 'processor' /proc/cpuinfo | wc -l)
SIBLINGS=$(grep 'siblings' /proc/cpuinfo  | cut -d ':' -f 2 | head -n 1 | xargs)
if [ $(expr ${ALL_CPUS} / ${SIBLINGS}) > 1 ]; then
    echo "HyperThreading on"
else
    echo "HyperThreading off"
fi

##### CPU frequencies
if [ ${DO_LIKWID} -a -x likwid-setFrequencies ]; then
    header "Frequencies"
    likwid-setFrequencies -p
fi


##### Hardware prefetchers
if [ ${DO_LIKWID} -a -x likwid-features ]; then
    header "Prefetchers"
    likwid-features -l -c N
fi

##### System load
header "Load"
cat /proc/loadavg

##### System load
if [ ${DO_LIKWID} -a -x likwid-powermeter ]; then
    header "Performance energy bias"
    likwid-powermeter -i | grep -i bias
fi

##### NUMA balancing
header "NUMA balancing"
echo -n "Enabled: "
cat /proc/sys/kernel/numa_balancing

##### Current memory information
header "General memory info"
cat /proc/meminfo

##### Transparent huge pages
header "Transparent huge pages";
if [ $(cat /sys/kernel/mm/transparent_hugepage/enabled) == "1" ]; then
    echo "State: on"
else
    echo "State: off"
fi
if [ $(cat /sys/kernel/mm/transparent_hugepage/use_zero_page) == "1" ]; then
    echo  "Use zero page: on"
else
    echo  "Use zero page: off"
fi

##### Hardware power limits
if [ -e /sys/devices/virtual/powercap ]; then
    header "Hardware power limits"
    RAPL_FOLDERS=$(find /sys/devices/virtual/powercap -name "intel-rapl\:*")
    for F in ${RAPL_FOLDERS}; do print_powercap_folder $F; done
fi

##### Currently loaded modules
if [ $(module 2>/dev/null || echo $?) -eq 0 ]; then
    header "Modules"
    module list
fi

##### Compilers
CC=""
if [ $(which icc 2>/dev/null | wc -l ) == 1 ]; then
    CC=$(which icc)
elif [ $(which gcc 2>/dev/null | wc -l ) == 1 ]; then
    CC=$(which gcc)
elif [ $(which clang 2>/dev/null | wc -l ) == 1 ]; then
    CC=$(which clang)
elif [ $(which pgcc 2>/dev/null | wc -l ) == 1 ]; then
    CC=$(which pgcc)

fi
if [ ! -z $CC ]; then
    header "C Compiler"
    $CC --version
fi

CXX=""
if [ $(which g++ 2>/dev/null | wc -l ) == 1 ]; then
    CXX=$(which g++)
elif [ $(which pgc++ 2>/dev/null | wc -l ) == 1 ]; then
    CC=$(which pgc++)
fi
if [ ! -z $CXX ]; then
    header "C++ Compiler"
    $CXX --version
fi

FORTRAN=""
if [ $(which ifort 2>/dev/null | wc -l ) == 1 ]; then
    FORTRAN=$(which ifort)
elif [ $(which gfortran 2>/dev/null | wc -l ) == 1 ]; then
    FORTRAN=$(which gfortran)
elif [ $(which pgf90 2>/dev/null | wc -l ) == 1 ]; then
    FORTRAN=$(which pgf90)
fi
if [ ! -z $FORTRAN ]; then
    header "Fortran Compiler"
    $FORTRAN --version
fi



##### MPI
header "MPI"
if [ $(which mpiexec 2>/dev/null | wc -l ) == 1 ]; then
    mpiexec --version
elif [ $(which mpiexec.hydra 2>/dev/null | wc -l ) == 1 ]; then
    mpiexec.hydra --version
elif [ $(which mpirun 2>/dev/null | wc -l ) == 1 ]; then
    mpirun --version
else
    echo "No MPI found"
fi

##### OS distribution
header "Operating System"
cat /etc/*release*

##### OS kernel
header "Operating System Kernel"
uname -a

##### Hostname
header "Hostname"
hostname -f

##### If NVIDIA tool is available, also print NVIDIA information
if [ $(which nvidia-smi 2>/dev/null | wc -l ) == 1 ]; then
    header "Nvidia GPUs"
    nvidia-smi
fi

##### If NEC Tsubasa tool is available, also print NEC Tsubasa information
if [ $(which veosinfo 2>/dev/null | wc -l ) == 1 ]; then
    header "NEC Tsubasa"
    veosinfo
fi

##### If InfiniBand is used
if [ $(which ibstat 2>/dev/null | wc -l ) == 1 ]; then
    header "InfiniBand"
    ibstat
fi

##### If Intel OmniPath is used
if [ $(which opafabricinfo 2>/dev/null | wc -l ) == 1 ]; then
    header "Intel OmniPath"
    opafabricinfo
fi

##### If an application is given, try to get version and linked libraries
if [ $# -ge 1 ]; then
    header "Executable"
    echo "Name: $1"
    if [ $($1 --version 2>/dev/null | wc -l) -gt 0 ]; then
        echo -n "Version: "
        $1 --version
    fi
    if [ $(which $1 2>/dev/null | wc -l ) == 1 ]; then
        echo "Libraries:"
        ldd $(which $1)
    fi
fi


