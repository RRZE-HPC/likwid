/*
 * =======================================================================================
 *
 *      Filename:  calculator_stack.h
 *
 *      Description:  Stack implementation for infix calculator
 *
 *      Version:   4.1
 *      Released:  19.5.2016
 *
 *      Author:   Brandon Mills (bm), mills.brandont@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) Brandon Mills
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

#ifndef CALCULATOR_STACK_H
#define CALCULATOR_STACK_H

typedef struct
{
    void **content;
    int size;
    int top;
} Stack;

void stackInit(Stack *s, int size);
void stackPush(Stack *s, void* val);
void* stackTop(Stack *s);
void* stackPop(Stack *s);
int stackSize(Stack *s);
void stackFree(Stack *s);

#endif /* CALCULATOR_STACK_H */
