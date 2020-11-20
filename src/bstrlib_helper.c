/*
 * =======================================================================================
 *
 *      Filename:  bstrlib_helper.c
 *
 *      Description:  Additional functions to the bstrlib library
 *
 *      Version:   5.1.0
 *      Released:  20.11.2020
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2020 RRZE, University Erlangen-Nuremberg
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
#include <stdio.h>
#include <stdlib.h>


#include <bstrlib.h>
#include <errno.h>


int bstrListAdd(struct bstrList * sl, bstring str)
{
    if (sl->qty >= sl->mlen) {
        int mlen = sl->mlen * 2;
        bstring * tbl;
        while (sl->qty >= mlen) {
            if (mlen < sl->mlen) return BSTR_ERR;
            mlen += mlen;
        }

        tbl = (bstring *) realloc (sl->entry, sizeof (bstring) * mlen);
        if (tbl == NULL) return BSTR_ERR;

        sl->entry = tbl;
        sl->mlen = mlen;
    }
    sl->entry[sl->qty] = bstrcpy(str);
    sl->qty++;
    return BSTR_OK;
}

int bstrListAddChar(struct bstrList * sl, char* str)
{
    if (!sl || !str) return BSTR_ERR;
    bstring tmp = bformat("%s", str);
    int err = bstrListAdd(sl, tmp);
    bdestroy(tmp);
    return err;
}

void bstrListPrint(struct bstrList * sl)
{
    int i = 0;
    if (!sl) return;
    if (sl->qty > 0)
    {
        printf("[%s", bdata(sl->entry[0]));
        for (i = 1; i < sl->qty; i++)
        {
            printf(", %s", bdata(sl->entry[i]));
        }
        printf("]\n");
    }
    else if (sl->qty == 0)
    {
        printf("[]\n");
    }
}

int bstrListDel(struct bstrList * sl, int idx)
{
    int i;

    if (!sl || idx < 0 || idx >= sl->qty) return BSTR_ERR;

    bdestroy(sl->entry[idx]);

    for (i = idx+1; i < sl->qty; i++)
    {
        sl->entry[i-1] = bstrcpy(sl->entry[i]);
    }
    sl->qty--;

    return BSTR_OK;
}

bstring bstrListGet(struct bstrList * sl, int idx)
{
    if (!sl || idx < 0 || idx >= sl->qty) return NULL;
    return sl->entry[idx];
}

int bstrListToCharList(struct bstrList* sl, char*** list)
{
    int i, j, err = 0;
    if (!sl || !list) return BSTR_ERR;
    char** l = malloc(sizeof(char*) * sl->qty);
    if (l)
    {
        for (i = 0; i < sl->qty; i++)
        {
            l[i] = malloc(sizeof(char) * (blength(sl->entry[i]) + 2));
            if (l[i])
            {
                int ret = snprintf(l[i], blength(sl->entry[i])+1, "%s", bdata(sl->entry[i]));
                if (ret > 0)
                {
                    l[i][ret] = '\0';
                }
            }
            else
            {
                for (j = 0; j < i; j++)
                {
                    if (l[j]) free(l[j]);
                }
                free(l);
                return -ENOMEM;
            }
        }
        *list = l;
        return sl->qty;
    }
    return -ENOMEM;
}

/*
 * int btrimbrackets (bstring b)
 *
 * Delete opening and closing brackets contiguous from both ends of the string.
 */
 #define   bracket(c) ((((unsigned char) c) == '(' || ((unsigned char) c) == ')'))
int btrimbrackets (bstring b) {
int i, j;

    if (b == NULL || b->data == NULL || b->mlen < b->slen ||
        b->slen < 0 || b->mlen <= 0) return BSTR_ERR;

    for (i = b->slen - 1; i >= 0; i--) {
        if (!bracket (b->data[i])) {
            if (b->mlen > i) b->data[i+1] = (unsigned char) '\0';
            b->slen = i + 1;
            for (j = 0; bracket (b->data[j]); j++) {}
            return bdelete (b, 0, j);
        }
    }

    b->data[0] = (unsigned char) '\0';
    b->slen = 0;
    return BSTR_OK;
}

int bisnumber(bstring b)
{
    int count = 0;
    for (int i = 0; i < blength(b); i++)
    {
        if (!isdigit(bchar(b, i)))
            break;
        count++;
    }
    return count == blength(b);
}

bstring read_file(char *filename)
{
    int ret = 0;
    FILE* fp = NULL;
    char buf[BUFSIZ];
    bstring content = bfromcstr("");
    fp = fopen(filename, "r");
    if (fp == NULL) {
        fprintf(stderr, "fopen(%s): errno=%d\n", filename, errno);
        return content;
    }
    for (;;) {
        /* Read another chunk */
        ret = fread(buf, 1, sizeof(buf), fp);
        if (ret < 0) {
            fprintf(stderr, "fread(%p, 1, %lu, %p): %d, errno=%d\n", buf, sizeof(buf), fp, ret, errno);
            fclose(fp);
            return content;
        }
        else if (ret == 0) {
            break;
        }
        bcatblk(content, buf, ret);
    }
    fclose(fp);
    return content;
}
