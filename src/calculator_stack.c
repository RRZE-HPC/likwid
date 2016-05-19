/*
 * =======================================================================================
 *
 *      Filename:  calculator_stack.c
 *
 *      Description:  Stack implementation for infix calculator
 *
 *      Version:   4.1
 *      Released:  19.5.2016
 *
 *      Author:   Brandon Mills (bm), mills.brandont@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2016 RRZE, University Erlangen-Nuremberg
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
#include <calculator_stack.h>

void stackInit(Stack *s, int size)
{
    s->content = malloc(size * sizeof(void*));
    s->size = size;
    s->top = -1;
}

void stackPush(Stack *s, void* val)
{
    (s->top)++;
    s->content[s->top] = val;
}

void* stackTop(Stack *s)
{
    void *ret = NULL;
    if(s->top >= 0 && s->content != NULL)
        ret = s->content[s->top];
    return ret;
}

void* stackPop(Stack *s)
{
    void *ret = NULL;
    if(s->top >= 0 && s->content != NULL)
        ret = s->content[(s->top)--];
    return ret;
}

int stackSize(Stack *s)
{
    return s->top + 1;
}

void stackFree(Stack *s)
{
    if (s->content)
        free(s->content);
    s->content = NULL;
    s->size = 0;
    s->top = -1;
}

