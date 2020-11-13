/*
 * =======================================================================================
 *
 *      Filename:  likwid_f90_interface.c
 *
 *      Description: F90 interface for marker API
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com,
 *               Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2016 RRZE, University Erlangen-Nuremberg
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

/* #####   HEADER FILE INCLUDES   ######################################### */

#include <stdlib.h>
#include <string.h>

#include <likwid.h>

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

void __attribute__ ((visibility ("default") ))
likwid_markerinit_(void)
{
    likwid_markerInit();
}

void __attribute__ ((visibility ("default") ))
likwid_markerthreadinit_(void)
{
    likwid_markerThreadInit();
}

void __attribute__ ((visibility ("default") ))
likwid_markerclose_(void)
{
    likwid_markerClose();
}

void __attribute__ ((visibility ("default") ))
likwid_markernextgroup_(void)
{
    likwid_markerNextGroup();
}

void __attribute__ ((visibility ("default") ))
likwid_markerregisterregion_(char* regionTag, int len)
{
    char* tmp = (char*) malloc((len+1) * sizeof(char) );
    strncpy(tmp, regionTag, len * sizeof(char) );

    for (int i=(len-1); len > 0; len--)
    {
        if (tmp[i] != ' ') {
            tmp[i+1] = 0;
            break;
        }
    }

    likwid_markerRegisterRegion( tmp );
    free(tmp);
}

void __attribute__ ((visibility ("default") ))
likwid_markerstartregion_(char* regionTag, int len)
{
    char* tmp = (char*) malloc((len+1) * sizeof(char) );
    strncpy(tmp, regionTag, len * sizeof(char) );

    for (int i=(len-1); len > 0; len--)
    {
        if (tmp[i] != ' ') {
            tmp[i+1] = 0;
            break;
        }
    }

    likwid_markerStartRegion( tmp );
    free(tmp);
}

void __attribute__ ((visibility ("default") ))
likwid_markerstopregion_(char* regionTag, int len)
{
    char* tmp = (char*) malloc((len+1) * sizeof(char));
    strncpy(tmp, regionTag, len * sizeof(char) );

    for (int i=(len-1); len > 0; len--)
    {
        if (tmp[i] != ' ') {
            tmp[i+1] = 0;
            break;
        }
    }

    likwid_markerStopRegion( tmp );
    free(tmp);
}

void __attribute__ ((visibility ("default") ))
likwid_markergetregion_(
        char* regionTag,
        int* nr_events,
        double* events,
        double *time,
        int *count,
        int len)
{
    char* tmp = (char*) malloc((len+1) * sizeof(char));
    strncpy(tmp, regionTag, len * sizeof(char) );

    for (int i=(len-1); len > 0; len--)
    {
        if (tmp[i] != ' ') {
            tmp[i+1] = 0;
            break;
        }
    }
    likwid_markerGetRegion( tmp, nr_events,  events, time, count);
    free(tmp);
}

void __attribute__ ((visibility ("default") ))
likwid_markerresetregion_(char* regionTag, int len)
{
    char* tmp = (char*) malloc((len+1) * sizeof(char));
    strncpy(tmp, regionTag, len * sizeof(char) );

    for (int i=(len-1); len > 0; len--)
    {
        if (tmp[i] != ' ') {
            tmp[i+1] = 0;
            break;
        }
    }
    likwid_markerResetRegion( tmp);
    free(tmp);
}




void __attribute__ ((visibility ("default") ))
likwid_nvmarkerinit_(void)
{
    likwid_gpuMarkerInit();
}


void __attribute__ ((visibility ("default") ))
likwid_nvmarkerclose_(void)
{
    likwid_gpuMarkerClose();
}

void __attribute__ ((visibility ("default") ))
likwid_nvmarkernextgroup_(void)
{
    likwid_markerNextGroup();
}

void __attribute__ ((visibility ("default") ))
likwid_nvmarkerregisterregion_(char* regionTag, int len)
{
    char* tmp = (char*) malloc((len+1) * sizeof(char) );
    strncpy(tmp, regionTag, len * sizeof(char) );

    for (int i=(len-1); len > 0; len--)
    {
        if (tmp[i] != ' ') {
            tmp[i+1] = 0;
            break;
        }
    }

    likwid_gpuMarkerRegisterRegion( tmp );
    free(tmp);
}

void __attribute__ ((visibility ("default") ))
likwid_nvmarkerstartregion_(char* regionTag, int len)
{
    char* tmp = (char*) malloc((len+1) * sizeof(char) );
    strncpy(tmp, regionTag, len * sizeof(char) );

    for (int i=(len-1); len > 0; len--)
    {
        if (tmp[i] != ' ') {
            tmp[i+1] = 0;
            break;
        }
    }

    likwid_gpuMarkerStartRegion( tmp );
    free(tmp);
}

void __attribute__ ((visibility ("default") ))
likwid_nvmarkerstopregion_(char* regionTag, int len)
{
    char* tmp = (char*) malloc((len+1) * sizeof(char));
    strncpy(tmp, regionTag, len * sizeof(char) );

    for (int i=(len-1); len > 0; len--)
    {
        if (tmp[i] != ' ') {
            tmp[i+1] = 0;
            break;
        }
    }

    likwid_gpuMarkerStopRegion( tmp );
    free(tmp);
}

void __attribute__ ((visibility ("default") ))
likwid_nvmarkerresetregion_(char* regionTag, int len)
{
    char* tmp = (char*) malloc((len+1) * sizeof(char));
    strncpy(tmp, regionTag, len * sizeof(char) );

    for (int i=(len-1); len > 0; len--)
    {
        if (tmp[i] != ' ') {
            tmp[i+1] = 0;
            break;
        }
    }
    likwid_gpuMarkerResetRegion( tmp);
    free(tmp);
}
