/*
 * ===========================================================================
 *
 *      Filename:  strUtil.c
 *
 *      Description:  Utility routines for strings. Depends on bstring lib.
 *
 *      Version:  <VERSION>
 *      Created:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Company:  RRZE Erlangen
 *      Project:  likwid
 *      Copyright:  Copyright (c) 2010, Jan Treibig
 *
 *      This program is free software; you can redistribute it and/or modify
 *      it under the terms of the GNU General Public License, v2, as
 *      published by the Free Software Foundation
 *     
 *      This program is distributed in the hope that it will be useful,
 *      but WITHOUT ANY WARRANTY; without even the implied warranty of
 *      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *      GNU General Public License for more details.
 *     
 *      You should have received a copy of the GNU General Public License
 *      along with this program; if not, write to the Free Software
 *      Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 *
 * ===========================================================================
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <sys/types.h>

#include <multiplex.h>
#include <strUtil.h>

int
str2int(const char* str)
{
    char* endptr;
    errno = 0;
    unsigned long val;
    val = strtoul(str, &endptr, 10);

    if ((errno == ERANGE && val == LONG_MAX )
            || (errno != 0 && val == 0)) 
    {
        perror("strtol");
        exit(EXIT_FAILURE);
    }

    if (endptr == str) 
    {
        fprintf(stderr, "No digits were found\n");
        exit(EXIT_FAILURE);
    }

    return (int) val;
}

int
bstr_to_cpuset(int* threads,  bstring q)
{
    int i;
    unsigned int rangeBegin;
    unsigned int rangeEnd;
    int numThreads=0;
    struct bstrList* tokens;
    struct bstrList* subtokens;
//    bstring q = bfromcstr(str);

    tokens = bsplit(q,',');

    for (i=0;i<tokens->qty;i++)
    {
        subtokens = bsplit(tokens->entry[i],'-');

        if (numThreads > MAX_NUM_THREADS)
        {
            fprintf(stderr, "Number Of threads too large!\n");
            exit(EXIT_FAILURE);
        }

        if( subtokens->qty == 1 )
        {
            threads[numThreads] = str2int((char *) subtokens->entry[0]->data);
            numThreads++;
        }
        else if ( subtokens->qty == 2 )
        {
            rangeBegin = str2int((char*) subtokens->entry[0]->data);
            rangeEnd = str2int((char*) subtokens->entry[1]->data);

            if (!(rangeBegin <= rangeEnd))
            {
                fprintf(stderr, "Range End bigger than begin\n");
                exit(EXIT_FAILURE);
            }

            while (rangeBegin <= rangeEnd) {
                threads[numThreads] = rangeBegin;
                numThreads++;
                rangeBegin++;
            }
        }
        else
        {
            fprintf(stderr, "Parse Error\n");
            exit(EXIT_FAILURE);
        }
        bstrListDestroy(subtokens);
    }

    bstrListDestroy(tokens);
    bdestroy(q);

    return numThreads;
}


void
bstr_to_eventset(StrUtilEventSet* set, bstring q)
{
    int i;
    struct bstrList* tokens;
    struct bstrList* subtokens;

    tokens = bsplit(q,',');
    set->numberOfEvents = tokens->qty;
    set->events = (StrUtilEvent*)
        malloc(set->numberOfEvents * sizeof(StrUtilEvent));

    for (i=0;i<tokens->qty;i++)
    {
        subtokens = bsplit(tokens->entry[i],':');

        if ( subtokens->qty != 2 )
        {
            fprintf(stderr, "Error in parsing event string\n");
            exit(EXIT_FAILURE);
        }
        else
        {
            set->events[i].eventName = bstrcpy(subtokens->entry[0]);
            set->events[i].counterName = bstrcpy(subtokens->entry[1]);
        }

        bstrListDestroy(subtokens);
    }

    bstrListDestroy(tokens);
    bdestroy(q);
}

#define INIT_SECURE_INPUT_LENGTH 256

bstring
bSecureInput (int maxlen, char* vgcCtx) {
    int i, m, c = 1;
    bstring b, t;
    int termchar = 0;

    if (!vgcCtx) return NULL;

    b = bfromcstralloc (INIT_SECURE_INPUT_LENGTH, "");

    for (i=0; ; i++)
    {
        if (termchar == c)
        {
            break;
        }
        else if ((maxlen > 0) && (i >= maxlen))
        {
            b = NULL;
            return b;
        }
        else
        {
            c = *(vgcCtx++);
        }

        if (EOF == c)
        {
            break;
        }

        if (i+1 >= b->mlen) {

            /* Double size, but deal with unusual case of numeric
               overflows */

            if ((m = b->mlen << 1)   <= b->mlen     &&
                    (m = b->mlen + 1024) <= b->mlen &&
                    (m = b->mlen + 16)   <= b->mlen &&
                    (m = b->mlen + 1)    <= b->mlen)
            {
                t = NULL;
            }
            else
            {
                t = bfromcstralloc (m, "");
            }

            if (t)
            {
                memcpy (t->data, b->data, i);
            }

            bdestroy (b); /* Clean previous buffer */
            b = t;
            if (!b)
            {
                return b;
            }
        }

        b->data[i] = (unsigned char) c;
    }

    i--;
    b->slen = i;
    b->data[i] = (unsigned char) '\0';
    return b;
}


int bJustifyCenter (bstring b, int width) 
{
    unsigned char space  = ' ';
    int alignSpace = (width - b->slen) / 2;
    int restSpace = (width - b->slen) % 2;
    if (width <= 0) return -__LINE__;

    if (b->slen <= width)
    {
        binsertch (b, 0, alignSpace, space);
    }

    binsertch (b, b->slen , alignSpace+restSpace, space);

    return BSTR_OK;
}


