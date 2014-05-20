#ifdef WIN32

int setenv(const char *envname, const char *envval, int overwrite);

#else

#include <stdlib.h>

#endif
