/*
 * =======================================================================================
 *
 *      Filename:  C-nvMarkerAPI.c
 *
 *      Description:  Example how to use the NvMarker API for Nvidia GPUs
 *                    in C/C++ applications
 *
 *      Version:   5.2.2
 *      Released:  26.07.2022
 *
 *      Author:  Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2022 NHR@FAU, University Erlangen-Nuremberg
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
#include <unistd.h>
#include <stdint.h>

#include <likwid-gpumarker.h>

extern int cuda_function(int gpu, size_t size);


int main(int argc, char* argv[])
{
    int i = 0;
    int numDevices = 1;

    LIKWID_GPUMARKER_INIT;

    LIKWID_GPUMARKER_START("matmul");
    // You can read the environment variable LIKWID_GPUS to determine list of GPUs
    for (i = 0; i < numDevices; i++)
        int err = cuda_function(0, 3200);

    LIKWID_GPUMARKER_STOP("matmul");

    LIKWID_GPUMARKER_CLOSE;

    return 0;
}
