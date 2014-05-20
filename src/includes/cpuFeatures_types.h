/*
 * ===========================================================================
 *
 *      Filename:  CpuFeature_types.h
 *
 *      Description:  Types file for CpuFeature module.
 *
 *      Version:  <VERSION>
 *      Created:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Company:  RRZE Erlangen
 *      Project:  likwid
 *      Copyright:  Copyright (c) 2010, Jan Treibig
 *
 *      This program is free software; you can redistribute it and/or modify
 *      it under the terms of the GNU General Public License, v2, as
 *      published by the Free Software Foundation
 *     
 *      This program is distributed in the hope that it will be useful,
 *      but WITHOUT ANY WARRANTY; without even the implied warranty of
 *      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *      GNU General Public License for more details.
 *     
 *      You should have received a copy of the GNU General Public License
 *      along with this program; if not, write to the Free Software
 *      Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 *
 * ===========================================================================
 */


#ifndef CPUFEATURES_TYPES_H
#define CPUFEATURES_TYPES_H

typedef enum {
    HW_PREFETCHER=0,
    CL_PREFETCHER,
    DCU_PREFETCHER,
    IP_PREFETCHER} CpuFeature;

typedef struct {
	unsigned int fastStrings:1;
	unsigned int thermalControl:1;
	unsigned int perfMonitoring:1;
	unsigned int hardwarePrefetcher:1;
	unsigned int ferrMultiplex:1;
	unsigned int branchTraceStorage:1;
	unsigned int pebs:1;
	unsigned int speedstep:1;
	unsigned int monitor:1;
	unsigned int clPrefetcher:1;
	unsigned int speedstepLock:1;
	unsigned int cpuidMaxVal:1;
	unsigned int xdBit:1;
	unsigned int dcuPrefetcher:1;
	unsigned int dynamicAcceleration:1;
	unsigned int ipPrefetcher:1;
    } CpuFeatureFlags;



#endif /*CPUFEATURES_TYPES_H*/
