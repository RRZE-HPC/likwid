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
# *      Author:   Thomas Roehl (tr), thomas.roehl@gmail.com
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


if [ $(which likwid-topology 2>/dev/null | wc -l) != "1" ]; then
    module load likwid
    if [ $(which likwid-topology 2>/dev/null | wc -l) != "1" ]; then
        DO_LIKWID=0
    fi
fi

header "Logged in users"
users
w

if [ ${DO_LIKWID} -a -x likwid-pin ]; then
    header "CPUset"
    likwid-pin -p
fi

header "CGroups"
echo -n "Allowed CPUs: "
cat /sys/fs/cgroup/cpuset/cpuset.effective_cpus
echo -n "Allowed Memory controllers: "
cat /sys/fs/cgroup/cpuset/cpuset.effective_mems

header "Topology"
if [ ${DO_LIKWID} -a -x likwid-topology ]; then
    likwid-topology
else
    lscpu
fi
numactl -H

if [ ${DO_LIKWID} -a -x likwid-setFrequencies ]; then
    header "Frequencies"
    likwid-setFrequencies -p
fi

if [ ${DO_LIKWID} -a -x likwid-features ]; then
    header "Prefetchers"
    likwid-features -l -c N
fi

header "Load"
cat /proc/loadavg

if [ ${DO_LIKWID} ]; then
header "Performance energy bias"
likwid-powermeter -i | grep -i bias
fi

header "NUMA balancing"
echo -n "Enabled: "
cat /proc/sys/kernel/numa_balancing

header "General memory info"
cat /proc/meminfo

header "Transparent huge pages"
echo -n "Enabled: "
cat /sys/kernel/mm/transparent_hugepage/enabled
echo -n "Use zero page: "
cat /sys/kernel/mm/transparent_hugepage/use_zero_page

header "Hardware power limits"
RAPL_FOLDERS=$(find /sys/devices/virtual/powercap -name "intel-rapl\:*")
for F in ${RAPL_FOLDERS}; do print_powercap_folder $F; done


header "Modules"
module list

header "Compiler"
CC=""
if [ $(which icc 2>/dev/null | wc -l ) == 1 ]; then
    CC=$(which icc)
elif [ $(which gcc 2>/dev/null | wc -l ) == 1 ]; then
    CC=$(which gcc)
fi
$CC --version

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

header "Operating System"
cat /etc/*release*
header "Operating System Kernel"
uname -a
header "Hostname"
hostname -f

if [ $(which nvidia-smi 2>/dev/null | wc -l ) == 1 ]; then
    header "Nvidia GPUs"
    nvidia-smi
fi

if [ $(which veosinfo 2>/dev/null | wc -l ) == 1 ]; then
    header "NEC Tsubasa"
    veosinfo
fi

if [ $# -ge 1 ]; then
    header "Executable"
    echo -n "Name: "
    echo $1
    if [ $($1 --version 2>/dev/null | wc -l) -gt 0 ]; then
        echo -n "Version: "
        $1 --version
    fi
    if [ $(which $1 2>/dev/null | wc -l ) == 1 ]; then
        echo "Libraries:"
        ldd $(which $1)
    fi
fi


