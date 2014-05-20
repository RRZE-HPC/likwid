/*
 * ===========================================================================
 *
 *      Filename:  barrier.h
 *
 *      Description:  Header File barrier Module
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

#ifndef BARRIER_H
#define BARRIER_H

#include <types.h>

/**
 * @brief  Initialize the barrier module
 * @param  numberOfThreads The total number of threads in the barrier
 */
extern void barrier_init(int numberOfGroups);

/**
 * @brief  Register a thread for a barrier
 * @param  threadId The id of the thread to register
 */
extern int barrier_registerGroup(int numThreads);
extern void barrier_registerThread(BarrierData* barr, int groupsId, int threadId);

/**
 * @brief  Synchronize threads
 * @param  threadId The id of the calling thread
 * @param  numberOfThreads Total number of threads in the barrier
 */
extern void  barrier_synchronize(BarrierData* barr);


#endif /*BARRIER_H*/
