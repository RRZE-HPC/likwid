/*
 * =======================================================================================
 *
 *      Filename:  bstrlib_helper.h
 *
 *      Description:  Additional functions to the bstrlib library (header file)
 *
 *      Version:   5.1.1
 *      Released:  31.03.2021
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2021 RRZE, University Erlangen-Nuremberg
 *
 *      This program is free software: you can redistribute it and/or modify it under
 *      the terms of the GNU General Public License as published by the Free Software
 *      Foundation, either version 3 of the License, or (at your option) any later
 *      version.
 *
 *      This program is distributed in the hope that it will be useful, but WITHOUT ANY
 *      WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 *      PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 *      You should have received a copy of the GNU General Public License along with
 *      this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * =======================================================================================
 */

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
int bstrListToCharList(struct bstrList* sl, char*** list);

int btrimbrackets (bstring b);
int bisnumber(bstring b);

bstring read_file(char *filename);

#ifdef __cplusplus
}
#endif

#endif /* BSTRLIB_HELPER_INCLUDE */
