#include <stdlib.h>
#include <sys/types.h>
#include <stdio.h>
#include <error.h>
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
        ERROR_PRINT(Error: value stack full cant push %g, f);
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
        ERROR_PLAIN_PRINT(Error: value stack empty);
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
        ERROR_PRINT(Error: op stack full cant push %d, f);
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


double _calc_infix(int op, double num1, double num2)
{
    double res = 0.0;

    switch(op){
        case '+' :
            res = num1 + num2;
            break;
        case '-' :
            res = num1 - num2;
            break;
        case '*' :
            res = num1 * num2;
            break;
        case '/' :
            res = num2 / num1;
            break;
        case -1 :
            printf("opstack empty\n");
            break;
    }
    return res;
}

int calculate_infix(char* finfix, double *result)
{
    int i = 0;
    char* ptr;
    double num1, num2;
    char op;
    
    if (result == NULL)
        return -EINVAL;
    *result = 0;
    if (finfix[0] == '\0')
        return -EINVAL;

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
                num1 = dpop();
                num2 = dpop();
                num1 = _calc_infix(op, num1, num2);
                dpush(num1);
                break;
                
            default:
                break;
        }
        i++;
    }
    while((op = oppop()) > 0)
    {
        num1 = dpop();
        num2 = dpop();
        num1 = _calc_infix(op, num1, num2);
        dpush(num1);
    }
    *result = dpop();
    opsptr = 0;
    dsptr = 0;
    return 0;
}



/**/
