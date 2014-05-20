#include <osdep/setenv.h>

#include <stdio.h>
#include <Windows.h>
#include <osdep/windowsError_win.h>

int setenv(const char *envname, const char *envval, int overwrite) {
	LPCTSTR lpName = envname;
	LPCTSTR lpValue = envval;

	// The operating system creates the environment variable if it does not exist and lpValue is not NULL
	BOOL success = SetEnvironmentVariable( lpName, lpValue );

	// Upon successful completion, zero shall be returned.
	// Otherwise, -1 shall be returned, errno set to indicate the error, and the environment shall be unchanged
	if (success) {
		return 0;
	} else {
		fprintf(stderr, "SetEnvironmentVariable failed (%d)\n", GetLastError());
		setErrnoFromLastWindowsError();
		return -1;
	}
}
