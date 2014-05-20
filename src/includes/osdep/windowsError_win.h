#include <WinDef.h>

// windows error to errno
int errnoFromWindows(DWORD code);

void setErrnoFromLastWindowsError();

// read native windows error messages
char *getLastErrorString();
void freeLastErrorString(char *s);
void printLastWinError(const char *msg);
