#include <Windows.h>
#include <errno.h>
#include <stdio.h>

// errno list taken from http://sources.redhat.com/cgi-bin/cvsweb.cgi/src/winsup/cygwin/errno.cc?rev=1.23&content-type=text/x-cvsweb-markup&cvsroot=src

// windows error to errno
static const struct
{
	DWORD w;             /* windows version of error */
	int e;               /* errno version of error */
}
errmap[] =
{
	/* FIXME: Some of these choices are arbitrary! */
	//(ERROR_INVALID_FUNCTION,          EBADRQC),
	{ERROR_FILE_NOT_FOUND,            ENOENT},
	{ERROR_PATH_NOT_FOUND,            ENOENT},
	{ERROR_TOO_MANY_OPEN_FILES,       EMFILE},
	{ERROR_ACCESS_DENIED,             EACCES},
	{ERROR_INVALID_HANDLE,            EBADF},
	{ERROR_NOT_ENOUGH_MEMORY,         ENOMEM},
	{ERROR_INVALID_DATA,              EINVAL},
	{ERROR_OUTOFMEMORY,               ENOMEM},
	{ERROR_INVALID_DRIVE,             ENODEV},
	{ERROR_NOT_SAME_DEVICE,           EXDEV},
	//{ERROR_NO_MORE_FILES,             ENMFILE},
	{ERROR_WRITE_PROTECT,             EROFS},
	{ERROR_BAD_UNIT,                  ENODEV},
	{ERROR_SHARING_VIOLATION,         EACCES},
	{ERROR_LOCK_VIOLATION,            EACCES},
	{ERROR_SHARING_BUFFER_EXCEEDED,   ENOLCK},
	{ERROR_HANDLE_EOF,                ENODATA},
	{ERROR_HANDLE_DISK_FULL,          ENOSPC},
	{ERROR_NOT_SUPPORTED,             ENOSYS},
	//{ERROR_REM_NOT_LIST,              ENONET},
	//{ERROR_DUP_NAME,                  ENOTUNIQ},
	//{ERROR_BAD_NETPATH,               ENOSHARE},
	//{ERROR_BAD_NET_NAME,              ENOSHARE},
	{ERROR_FILE_EXISTS,               EEXIST},
	{ERROR_CANNOT_MAKE,               EPERM},
	{ERROR_INVALID_PARAMETER,         EINVAL},
	{ERROR_NO_PROC_SLOTS,             EAGAIN},
	{ERROR_BROKEN_PIPE,               EPIPE},
	{ERROR_OPEN_FAILED,               EIO},
	{ERROR_NO_MORE_SEARCH_HANDLES,    ENFILE},
	{ERROR_CALL_NOT_IMPLEMENTED,      ENOSYS},
	{ERROR_INVALID_NAME,              ENOENT},
	{ERROR_WAIT_NO_CHILDREN,          ECHILD},
	{ERROR_CHILD_NOT_COMPLETE,        EBUSY},
	{ERROR_DIR_NOT_EMPTY,             ENOTEMPTY},
	{ERROR_SIGNAL_REFUSED,            EIO},
	{ERROR_BAD_PATHNAME,              ENOENT},
	{ERROR_SIGNAL_PENDING,            EBUSY},
	{ERROR_MAX_THRDS_REACHED,         EAGAIN},
	{ERROR_BUSY,                      EBUSY},
	{ERROR_ALREADY_EXISTS,            EEXIST},
	{ERROR_NO_SIGNAL_SENT,            EIO},
	{ERROR_FILENAME_EXCED_RANGE,      EINVAL},
	{ERROR_META_EXPANSION_TOO_LONG,   EINVAL},
	{ERROR_INVALID_SIGNAL_NUMBER,     EINVAL},
	{ERROR_THREAD_1_INACTIVE,         EINVAL},
	{ERROR_BAD_PIPE,                  EINVAL},
	{ERROR_PIPE_BUSY,                 EBUSY},
	{ERROR_NO_DATA,                   EPIPE},
	//{ERROR_PIPE_NOT_CONNECTED,        ECOMM},
	{ERROR_MORE_DATA,                 EAGAIN},
	{ERROR_DIRECTORY,                 ENOTDIR},
	{ERROR_PIPE_CONNECTED,            EBUSY},
	//{ERROR_PIPE_LISTENING,            ECOMM},
	{ERROR_NO_TOKEN,                  EINVAL},
	{ERROR_PROCESS_ABORTED,           EFAULT},
	{ERROR_BAD_DEVICE,                ENODEV},
	{ERROR_BAD_USERNAME,              EINVAL},
	{ERROR_NOT_CONNECTED,             ENOLINK},
	{ERROR_OPEN_FILES,                EAGAIN},
	{ERROR_ACTIVE_CONNECTIONS,        EAGAIN},
	{ERROR_DEVICE_IN_USE,             EAGAIN},
	{ERROR_INVALID_AT_INTERRUPT_TIME, EINTR},
	{ERROR_IO_DEVICE,                 EIO},
	{ERROR_NOT_OWNER,                 EPERM},
	{ERROR_END_OF_MEDIA,              ENOSPC},
	{ERROR_EOM_OVERFLOW,              ENOSPC},
	{ERROR_BEGINNING_OF_MEDIA,        ESPIPE},
	{ERROR_SETMARK_DETECTED,          ESPIPE},
	{ERROR_NO_DATA_DETECTED,          ENOSPC},
	{ERROR_POSSIBLE_DEADLOCK,         EDEADLOCK},
	{ERROR_CRC,                       EIO},
	{ERROR_NEGATIVE_SEEK,             EINVAL},
	//{ERROR_NOT_READY,                 ENOMEDIUM},
	{ERROR_DISK_FULL,                 ENOSPC},
	{ERROR_NOACCESS,                  EFAULT},
	{ERROR_FILE_INVALID,              ENXIO},
	{0, 0}
};

int errnoFromWindows(DWORD code)
{
	int deferrno = 0;
	int i;

	for (i = 0; errmap[i].w != 0; ++i) {
		if (code == errmap[i].w) {
			return errmap[i].e;
		}
	}

	fprintf(stderr, "unknown windows error %u, setting errno to %d", code, deferrno);
	return deferrno;     
}

void setErrnoFromLastWindowsError()
{
	errno = errnoFromWindows(GetLastError());
}

// read native windows error messages
char *getLastErrorString()
{
	LPTSTR errStr;
	FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER |
		FORMAT_MESSAGE_FROM_SYSTEM,
		NULL,
		GetLastError(),
		0,
		(LPTSTR)&errStr,
		0,
		NULL);

	return errStr;
}

void freeLastErrorString(char *s) {
	LocalFree(s);
}

void printLastWinError(const char *msg) {
	char *errMsg = getLastErrorString();
	fprintf(stderr, "\n%s: %s\n", msg, errMsg);
	freeLastErrorString(errMsg);
}
