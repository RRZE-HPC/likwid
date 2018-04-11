/*
 * =======================================================================================
 *
 *      Filename:  setFreq.c
 *
 *      Description:  Entry point of frequency daemon
 *
 *      Version:   4.3.2
 *      Released:  12.04.2018
 *
 *      Authors:  Thomas Roehl (tr), thomas.roehl@googlemail.com
 *
 *      Project:  likwid
 *
 *      Copyright (C) 2018 RRZE, University Erlangen-Nuremberg
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

#include <stdlib.h>
#include <stdio.h>
#include <dirent.h>
#include <errno.h>
#include <setFreq.h>


static int is_pstate()
{
    int ret = 1;
    DIR* dir = opendir("/sys/devices/system/cpu/intel_pstate");
    if (ENOENT == errno)
    {
        //fprintf(stderr, "\tEXIT WITH ERROR:  intel_pstate is not present!\n");
        ret = 0;
    }

    closedir(dir);
    return ret;
}


int main(int argc, char** argv)
{
    if (is_pstate())
    {
        printf("Pstate driver\n");
        return do_pstate(argc, argv);
    }
    else
        return do_cpufreq(argc, argv);
}
