#ifndef BSTRLIB_HELPER_INCLUDE
#define BSTRLIB_HELPER_INCLUDE

#include <bstrlib.h>

#ifdef __cplusplus
extern "C" {
#endif


int bstrListAdd(struct bstrList * sl, bstring str);
int bstrListAddChar(struct bstrList * sl, char* str);

int bstrListDel(struct bstrList * sl, int idx);

bstring bstrListGet(struct bstrList * sl, int idx);

void bstrListPrint(struct bstrList * sl);

int btrimbrackets (bstring b);
int bisnumber(bstring b);

bstring read_file(char *filename);

#ifdef __cplusplus
}
#endif

#endif /* BSTRLIB_HELPER_INCLUDE */
