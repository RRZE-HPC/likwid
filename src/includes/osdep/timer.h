#ifndef OSDEPS_TIMER_H
#define OSDEPS_TIMER_H

#ifdef WIN32

#include <time.h>
#include <windows.h>
#include <assert.h>

struct timezone 
{
  int  tz_minuteswest; /* minutes W of Greenwich */
  int  tz_dsttime;     /* type of dst correction */
};
 

struct timespec {
	time_t  tv_sec;  //seconds
	long    tv_nsec; //nanoseconds
};


#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
  #define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#else
  #define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#endif

int gettimeofday(struct timeval *tv, struct timezone *tz)
{
  FILETIME ft;
  unsigned __int64 tmpres = 0;
  static int tzflag;

  if (NULL != tv)
  {
    GetSystemTimeAsFileTime(&ft);

    tmpres |= ft.dwHighDateTime;
    tmpres <<= 32;
    tmpres |= ft.dwLowDateTime;

    /*converting file time to unix epoch*/
    tmpres /= 10;  /*convert into microseconds*/
    tmpres -= DELTA_EPOCH_IN_MICROSECS;
    tv->tv_sec = (long)(tmpres / 1000000UL);
    tv->tv_usec = (long)(tmpres % 1000000UL);
  }

  if (NULL != tz)
  {
    if (!tzflag)
    {
      _tzset();
      tzflag++;
    }
	{
		long timezone;
		int daylight;
		if (_get_timezone(&timezone) != 0) {
			perror("_get_timezone");
			exit(1);
		}
		if (_get_daylight(&daylight) != 0) {
			perror("_get_daylight");
			exit(1);
		}
		tz->tz_minuteswest = timezone / 60;
		tz->tz_dsttime = daylight;
	}
  }

  return 0;
}

int nanosleep(const struct timespec *rqtp, struct timespec *rmtp) {
	DWORD milliSeconds_byN = (DWORD)(rqtp->tv_nsec / (1000*1000));
	DWORD milliSeconds_byS = (DWORD)(rqtp->tv_sec * 1000);

	assert(rmtp == NULL);

	// make shure that no nano part was used
	assert(rqtp->tv_nsec == milliSeconds_byN * (1000*1000));

	Sleep(milliSeconds_byN + milliSeconds_byS);

	return 0;
}


#else /* WIN32 */

#include <sys/time.h>

#endif /* WIN32 */

#endif /* OSDEPS_TIMER_H */
