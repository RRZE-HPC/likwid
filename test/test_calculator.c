#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h> // Temporary
#include <getopt.h>
#include <calculator.h>
#include <calculator_stack.h>


#ifndef NAN
#define NAN (0.0/0.0)
#endif

#ifndef INFINITY
#define INFINITY (1.0/0.0)
#endif

typedef struct {
    char* formula;
    double result;
    int err;
} CalcTest;

static CalcTest calc_tests[] = {
    {"10", 10},
    {"10.00000000000", 10},
    {"0.00000000000", 0},
    {"00000001.1", 1.1},
    {"1+1", 2},
    {"1    +    1", 2},
    {"(7+8)", 15},
    {"((7+8))", 15},
    {"(((7+8)))", 15},
    {"(((7)+(8)))", 15},
    {"(7+8)+(6+9)", 30},
    {"2*2", 4},
    {"2^2", 4},
    {"8/2", 4},
    {"8%2", 0},
    {"8%3", 2},
    {"22-22", 0},
    {"1E+6*2", 2E6},
    {"1E6*2", 2E6},
    {"1.0E6*2", 2E6},
    {"1E+06*2", 2E6},
    {"1.0E+06*2", 2E6},
    {"1E+006*2", 2E6},
    {"1.0E+006*2", 2E6},
    {"1/0", INFINITY},
    {"0.0/0.0", NAN},
    {"1E-06*(((256*512.0)/128.0)+14)/0.000006", 172.99999999999997},
    {"1E-06*(((0*512.0)/128.0)+14)/0.000006", 2.333333333333333},
    {"1E-06*(((256*512.0)/128.0)+0)/0.000006", 170.66666666666666},
    {"1.0E-06*(19)*256.0/2.000000e-06", 2432.00000},
    {"1.0E-06*(19+22+54+12+1+856745+2134)*64.0/2.000000e-06", 27487584.0},
    {"1.0E-09*(19+22+54+12+1+856745+2134)*64.0", 0.054975168000000005},
    {"1.0E-6*nan*64.0/0.111", NAN},
    {"1.0E-6*NAN*64.0/0.111", NAN},
    {"sum(1,2,3,4,5,6,7,8,9,10)", 55},
    {"2.0*sum(1,2,3,4,5,6,7,8,9,10)", 110},
    {"(sum(1,2,3,4,5,6,7,8,9,10))", 55},
    {"sum(1.1+2.2,3.3+4.4)", 11},
    {"min(1,2,3,4,5,6,7,8,9,10)", 1},
    {"max(1,2,3,4,5,6,7,8,9,10)", 10},
    {"avg(1,2,3,4,5,6,7,8,9,10)", 5.5},
    {"mean(1,2,3,4,5,6,7,8,9,10)", 5.5},
    {"median(1,2,3,4,5,6,7,8,9,10)", 5},
    {"floor(2.2)", 2},
    {"ceil(2.2)", 3},
    {"abs(-2.2)", 2.2},
    {"abs(2.2)", 2.2},
    {"abs(2.2,1.1)", 2.2, -1}, // should fail, abs function is taking only a single argument
    {"abs(2.2+1.1)", 3.3, 0},
    {"exp(2.0)", 7.38905609893065040694},
    {"2+", NAN, -1},
    {"2-", NAN, -1},
    {"2*", NAN, -1},
    {"2/", NAN, -1},
    {"2%", NAN, -1},
    {"2^", NAN, -1},
    {"()", NAN, -1},
    {"+2", NAN, -1},
    {"-2", -2, 0},
    {"*2", NAN, -1},
    {"/2", NAN, -1},
    {"%2", NAN, -1},
    {"(2", NAN, -1},
    {"+", NAN, -1},
    {"-", NAN, -1},
    {"*", NAN, -1},
    {"/", NAN, -1},
    {"%", NAN, -1},
    {"^", NAN, -1},
    {"2)", 2, 0}, // shouldn't it return an error? {"2)", NAN, -1}
    {"sumi(1,2)", 3}, // shouldn't it return an error? {"sumi(1,2)", NAN, -1}
    {"2--2", 4, 0},
    {"2*-2", -4, 0},
    //{"2++2", NAN, -1}, // segfault, maybe we should catch it
    //{"2**2", NAN, -1}, // segfault, maybe we should catch it
    //{"2//2", NAN, -1}, // segfault, maybe we should catch it
    //{"2%%2", NAN, -1}, // segfault, maybe we should catch it
    //{"2^^2", NAN, -1}, // segfault, maybe we should catch it
    //{"(2+", NAN, -1}, // segfault, maybe we should catch it
    {NULL, 0.0}, // do not remove
};


int main(int argc, char* argv[])
{
    int i = 0;
    int all = 0;
    int success = 0;
    int shouldfail = 0;
    double res = 0.0;
    char ref[100];
    char test[100];
    CalcTest *cur = &calc_tests[0];

    while (cur->formula)
    {
        res = 0.0;
        int ret = calculate_infix(cur->formula, &res);
        if (ret < 0)
        {
            if (ret == cur->err)
            {
                shouldfail++;
            }
            else
            {
                printf("Failed calculating '%s'. Reference %f. CalcResult %f. Error %d\n", cur->formula, cur->result, res, ret);
            }
            all++;
            cur++;
            continue;
        }
        else
        {
            if (cur->err < 0)
            {
                printf("Oops calculating '%s'. Reference %f. CalcResult %f. Error %d\n", cur->formula, cur->result, res, ret);
                all++;
                cur++;
                continue;
            }
        }
        ret = snprintf(ref, 99, "%.*f", CALCULATOR_MAXPRECISION, cur->result);
        if (ret > 0)
            ref[ret] = '\0';
        ret = snprintf(test, 99, "%.*f", CALCULATOR_MAXPRECISION, res);
        if (ret > 0)
            test[ret] = '\0';
        if ((strlen(ref) != strlen(test)) || strncmp(ref, test, strlen(ref) != 0))
        {
            printf("Wrong calculating '%s'. Reference %f. CalcResult %f. Error %d\n", cur->formula, cur->result, res, ret);
        }
        else
        {
            success++;
        }
        all++;
        cur++;
    }
    printf("All\tSuccess\tShouldFail\tFail\n");
    printf("%d\t%d\t%d\t\t%d\n", all, success, shouldfail, all-success-shouldfail);

    return (all-success-shouldfail != 0);
}
