#include <lock.h>

int lock_check(void)
{
    struct stat buf;
    int lock_handle = -1;
    int result      = 0;
    char *filepath  = TOSTRING(LIKWIDLOCK);

    if ((lock_handle = open(filepath, O_RDONLY)) == -1) {
        if (errno == ENOENT) {
            /* There is no lock file. Proceed. */
            result = 1;
        } else if (errno == EACCES) {
            /* There is a lock file. We cannot open it. */
            result = 0;
        } else {
            /* Another error occured. Proceed. */
            result = 1;
        }
    } else {
        /* There is a lock file and we can open it. Check if we own it. */
        stat(filepath, &buf);

        if (buf.st_uid == getuid()) /* Succeed, we own the lock */
        {
            result = 1;
        } else /* we are not the owner */
        {

            result = 0;
        }
    }

    if (lock_handle > 0) {
        close(lock_handle);
    }

    return result;
}
