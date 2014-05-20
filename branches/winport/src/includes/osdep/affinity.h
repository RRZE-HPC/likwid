#ifndef OSDEP_AFFINITY_H
#define OSDEP_AFFINITY_H

#include <stdint.h>

#ifdef WIN32
#include <Windows.h>
typedef HANDLE ThreadId;

// in windows the affinity mask is a DWORD_PTR which means that we can take a size_t as well
#include "stddef.h"
typedef size_t AffinityMask;

#define MASK_ZERO(mask)  *mask = 0
#define MASK_ISSET(processorId, mask)  ((*mask & (1 << processorId)) != 0)
#define MASK_SET(processorId, mask) *mask |= (1 << processorId)

#else

#include <sys/types.h>
typedef pthread_t ThreadId;
#include <sched.h>
typedef cpu_set_t AffinityMask;

#define MASK_ZERO(mask)  CPU_ZERO(mask)
#define MASK_ISSET(processorId, mask) CPU_ISSET(processorId,(mask))
#define MASK_SET(processorId, mask) CPU_SET((mask),processorId)

#define gettid() syscall(SYS_gettid)
#endif

static int
getProcessorID(AffinityMask* cpu_set)
{
    int processorId;

    for (processorId=0;processorId<MAX_NUM_THREADS;processorId++)
    {
        if (MASK_ISSET(processorId,cpu_set))
        {
            break;
        }
    }
    return processorId;
}


static int 
processGetProcessorId()
{
#ifdef WIN32
	DWORD_PTR processAffinityMask;
	DWORD_PTR systemAffinityMask;

	BOOL res = GetProcessAffinityMask(
	  GetCurrentProcess(),
	  &processAffinityMask,
	  &systemAffinityMask);

	if (res == 0)
    {
		perror("GetCurrentProcess");
		exit(1);
	}
	return getProcessorID(&processAffinityMask);
#else
    int ret;
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    ret = sched_getaffinity(getpid(),sizeof(cpu_set_t), &cpu_set);

    if (ret < 0)
    {
        ERROR;
    }

    return getProcessorID(&cpu_set);
#endif
}



static int 
threadGetProcessorId()
{
#ifdef WIN32
    DWORD_PTR
        tmpThreadAffinityMask = (1 << 0),
                              tmpThreadAffinityMask1 = -1;

    // temporarily set the thread affinity mask and retrieve old affinity mask
    DWORD_PTR origThreadAffinityMask = SetThreadAffinityMask(
            threadId,
            tmpThreadAffinityMask
            );
    if (origThreadAffinityMask == 0) {
        perror("SetThreadAffinityMask, used to get the thread affinity mask (1)");
        exit(1);
    }
    // reset the thread affinity mask to its original value
    tmpThreadAffinityMask1 = SetThreadAffinityMask(
            threadId,
            origThreadAffinityMask
            );
    if (tmpThreadAffinityMask1 == 0) {
        perror("SetThreadAffinityMask, used to get the thread affinity mask (2)");
        exit(1);
    }
    // check whether the resetting succeeded
    if (tmpThreadAffinityMask1 != tmpThreadAffinityMask) {
        perror("SetThreadAffinityMask, failed to reset thread affinity mask");
        exit(1);
    }
    return getProcessorID(&origThreadAffinityMask);
#else
    cpu_set_t  cpu_set;
    CPU_ZERO(&cpu_set);
    sched_getaffinity(gettid(),sizeof(cpu_set_t), &cpu_set);

    return getProcessorID(&cpu_set);
#endif
}


static int
pinProcess(int processorId)
{
	AffinityMask mask;
	MASK_ZERO(&mask);
	MASK_SET(&mask,processorId);

#ifdef WIN32
	BOOL r = SetProcessAffinityMask(
		GetCurrentProcess(),
		affinityMask
	);
	if (r == 0) {
		setErrnoFromLastWindowsError();
		return FALSE;
	}
	return TRUE;
#else
	if (sched_setaffinity(0, sizeof(AffinityMask), &mask) == -1)
	{
		//perror("sched_setaffinity failed");
		/*
		TODO: take this error treatment or the one above?
		if (errno == EFAULT)
        {
            fprintf(stderr, "A supplied memory address was invalid\n");
            exit(EXIT_FAILURE);
        }
        else if (errno == EINVAL)
        {
            fprintf(stderr, "Processor is not available\n");
            exit(EXIT_FAILURE);
        }
        else
        {
            fprintf(stderr, "Unknown error\n");
            exit(EXIT_FAILURE);
        }
		*/
        return FALSE;
	}

    return TRUE;
#endif
}


#ifdef HAS_SCHEDAFFINITY
static int
pinThread(int processorId)
{
    int ret;
    ThreadId thread;
	AffinityMask mask;
	MASK_ZERO(&mask);
	MASK_SET(&mask, processorId);

#ifdef WIN32
    thread = GetCurrentThread();
	DWORD_PTR origThreadAffinityMask = SetThreadAffinityMask(
		threadId,
		affinityMask
	);
	if (origThreadAffinityMask == 0) {
		setErrnoFromLastWindowsError();
		return FALSE;
	}
	return TRUE;
#else
    thread = pthread_self();
    ret = pthread_setaffinity_np(thread, sizeof(AffinityMask), &mask);

    if (ret != 0) 
    {
        ERROR;
    }

    return TRUE;
#endif

}
#endif /* HAS_SCHEDAFFINITY */

#endif /* OSDEP_AFFINITY_H */
