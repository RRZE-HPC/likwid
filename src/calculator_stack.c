/*
 * =======================================================================================
 *
 *      Filename:  calculator_stack.c
 *
 *      Description:  Stack implementation for infix calculator
 *
 *      Version:   4.1
 *      Released:  8.8.2016
 *
 *      Author:   Brandon Mills (bm), mills.brandont@gmail.com
 *
 *      Copyright (C) 2016 Brandon Mills
 *
 *      Permission is hereby granted, free of charge, to any person obtaining a copy of this
 *      software and associated documentation files (the "Software"), to deal in the
 *      Softwarewithout restriction, including without limitation the rights to use, copy,
 *      modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
 *      and to permit persons to whom the Software is furnished to do so, subject to the
 *      following conditions:
 *
 *      The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 *      THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 *      INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 *      PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 *      HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 *      OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 *      SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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

