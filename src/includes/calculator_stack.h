/*
 * =======================================================================================
 *
 *      Filename:  calculator_stack.h
 *
 *      Description:  Stack implementation for infix calculator
 *
 *      Version:   4.2
 *      Released:  22.12.2016
 *
 *      Author:   Brandon Mills (bm), mills.brandont@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) Brandon Mills
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

#ifndef CALCULATOR_STACK_H
#define CALCULATOR_STACK_H

typedef struct {
    void **content;
    int size;
    int top;
} Stack;

#define FRESH_STACK { NULL, 0, -1 }

void stackInit(Stack *s, int size);
void stackPush(Stack *s, void *val);
void *stackTop(Stack *s);
void *stackPop(Stack *s);
int stackSize(Stack *s);
void stackFree(Stack *s);

#endif /* CALCULATOR_STACK_H */
