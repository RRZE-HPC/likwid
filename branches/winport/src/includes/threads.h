/*
 * ===========================================================================
 *
 *      Filename:  threads.h
 *
 *      Description:  Header file of pthread interface module
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

#ifndef THREADS_H
#define THREADS_H

#include <types.h>
#include <pthread.h>
#include <threads_types.h>

#define THREADS_BARRIER pthread_barrier_wait(&threads_barrier)

extern pthread_barrier_t threads_barrier;
extern ThreadData* threads_data;
extern ThreadGroup* threads_groups;


/**
 * @brief  Initialization of the thread module
 * @param  numberOfThreads  The total number of threads
 */
extern void threads_init(int numberOfThreads);

/**
 * @brief  Create all threads
 * @param  startRoutine thread entry function pointer
 */
extern void threads_create(void *(*startRoutine)(void*));

/**
 * @brief  Register User thread data for all threads
 * @param  data  Reference to the user data structo
 * @param  func  Optional function pointer to copy data
 */
extern void threads_registerDataAll(
        ThreadUserData* data,
        threads_copyDataFunc func);

/**
 * @brief  Register User thread data for one thread
 * @param  threadId thread Id 
 * @param  data  Reference to the user data structo
 * @param  func  Optional function pointer to copy data
 */
extern void threads_registerDataThread(
        int threadId,
        ThreadUserData* data,
        threads_copyDataFunc func);

/**
 * @brief  Register User thread data for a thread group
 * @param  groupId  group Id
 * @param  data  Reference to the user data structo
 * @param  func  Optional function pointer to copy data
 */
extern void threads_registerDataGroup(
        int groupId,
        ThreadUserData* data,
        threads_copyDataFunc func);

/**
 * @brief  Join the threads and free memory of data structures
 * @param
 */
extern void threads_destroy(void);

/**
 * @brief  Create Thread groups
 * @param  numberOfGroups The number of groups to create
 */
extern void threads_createGroups(int numberOfGroups);

#endif /* THREADS_H */
