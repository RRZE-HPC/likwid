#include <stdlib.h>
#include <sys/types.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdint.h>

#include <calculator.h>

#define MAXVAL 100

int dsptr = 0;
double dstack[MAXVAL];
int opsptr = 0;
int opstack[MAXVAL];

void dpush(double f)
{
    if( dsptr < MAXVAL)
    {
        dstack[dsptr++] = f;
    }
    else
    {
        printf("error: value stack full, cant push %g\n",f);
    }
}

double dpop(void)
{
    if(dsptr > 0)
    {
        return dstack[--dsptr];
    }
    else
    {
        printf("error: value stack empty\n");
        return 0;
    }
}



void oppush(int f)
{
    if( opsptr < MAXVAL)
    {
        opstack[opsptr++] = f;
    }
    else
    {
        printf("error: op stack full, cant push %d\n",f);
    }
}

int oppop(void)
{
    if (opsptr > 0)
    {
        return opstack[--opsptr];
    }
    else
    {
        return -1;
    }
}


void _calc_infix(int op)
{
    double num1 = dpop();
    double num2 = dpop();
    
    switch(op){
        case '+' :
            dpush(num1 + num2);
            break;
        case '-' :
            dpush(num1 - num2);
            break;
        case '*' :
            dpush(num1 * num2);
            break;
        case '/' :
            dpush(num2 / num1);
            break;
        case -1 :
            printf("opstack empty\n");
            break;
    }
}

int calculate_infix(char* finfix, double *result)
{
    int i = 0;
    char* ptr;
    double num1, num2;
    char op;
    if (result == NULL)
        return -1;
    *result = 0;
    if (finfix[0] == '\0')
        return -1;

    // evaluate
    while ( finfix[i] != '\0')
    {
        switch (finfix[i])
        {
            case '(':
                break;
            case '+':
            case '-':
            case '*':
            case '/':
                oppush(finfix[i]);
                break;
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
                num1 = strtod(&(finfix[i]), &ptr);
                dpush(num1);
                while (finfix[i] != *ptr) {i++;}
                i--;
                break;
            case ')':
            case '\n':
            case ' ':
            case '\0':
                op = oppop();
                _calc_infix(op);
                break;
                
            default:
                break;
        }
        i++;
    }
    while((op = oppop()) > 0)
    {
        _calc_infix(op);
    }
    *result = dpop();
    opsptr = 0;
    dsptr = 0;
    return 0;
}



/**/
