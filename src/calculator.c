/*
 * =======================================================================================
 *
 *      Filename:  calculator.c
 *
 *      Description:  Infix calculator
 *
 *      Version:  <VERSION>
 *      Released: <DATE>
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
/*
 * =======================================================================================
 *
 *      Some changes done for the integration in LIKWID, see inline comments
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com
 *                Thomas Gruber (tr), thomas.roehl@gmail.com
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

/* #####   HEADER FILE INCLUDES   ######################################### */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h> // Temporary
#include <getopt.h>
#include <error.h>
#include <calculator_stack.h>

/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */

#define bool char
#define true 1
#define false 0
#define PI 3.141592653589793

#ifndef NAN
#define NAN (0.0/0.0)
#endif

#ifndef INFINITY
#define INFINITY (1.0/0.0)
#endif

/* Added by Thomas Gruber (Thomas.Roehl@fau.de) to reduce reallocs by allocating a temporary
 * token for parsing as well as for transforming a number to a string.
 */
#define MAXTOKENLENGTH 512
#define MAXPRECISION 20
#define DEFAULTPRECISION 5
#define AUTOPRECISION -1
#define FUNCTIONSEPARATOR "|"

/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

typedef enum
{
    addop,
    multop,
    expop,
    lparen,
    rparen,
    digit,
    value,
    decimal,
    space,
    text,
    function,
    identifier,
    argsep,
    invalid
} Symbol;

struct Preferences
{
    struct Display
    {
        bool tokens;
        bool postfix;
        bool errors;
    } display;
    struct Mode
    {
        bool degrees;
    } mode;
    int precision;
    int maxtokenlength;
} prefs;

typedef enum
{
    divZero,
    overflow,
    parenMismatch,
    inputMissing,
    faultyExpression,
} Error;

typedef char* token;

typedef double number;

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

void raise(Error err)
{
    const char *msg = "";
    switch(err)
    {
        case divZero:
            msg = "Divide by zero";
            break;
        case overflow:
            msg = "Overflow";
            break;
        case parenMismatch:
            msg = "Mismatched parentheses";
            break;
        case inputMissing:
            msg = "Function input missing";
            break;
        case faultyExpression:
            msg = "Expression invalid";
            break;
    }
    DEBUG_PRINT(DEBUGLEV_DEVELOP, "\tError: %s\n", msg);
}

inline unsigned int
toDigit(char ch)
{
    return ch - '0';
}

number buildNumber(token str)
{
    number result = 0;
    /*while(*str && *str != '.')
    {
        result = result * 10 + toDigit(*str++);
    }*/
    result = strtod(str, NULL);
    return result;
}

token num2Str(number num)
{
    int len = 0;
    int precision = MAXPRECISION;
    if (prefs.precision >= 0 && prefs.precision < precision)
        precision = prefs.precision;
    token str = (token)malloc(prefs.maxtokenlength*sizeof(char));
    len = snprintf(str, prefs.maxtokenlength-1, "%.*f", precision, num);
    if (prefs.precision == AUTOPRECISION)
    {
        while (str[len-1] == '0')
        {
            len = snprintf(str, prefs.maxtokenlength-1, "%.*f", --precision, num);
        }
    }

    return str;
}

number
toRadians(number degrees)
{
    return degrees * PI / 180.0;
}

number
toDegrees(number radians)
{
    return radians * 180.0 / PI;
}

int doFunc(Stack *s, token function)
{
    if (stackSize(s) == 0)
    {
        raise(inputMissing);
        stackPush(s, num2Str(NAN));
        return -EFAULT;
    }
    else if (stackSize(s) == 1 && strcmp(stackTop(s), FUNCTIONSEPARATOR) == 0)
    {
        stackPop(s);
        raise(inputMissing);
        stackPush(s, num2Str(NAN));
        return -EFAULT;
    }
    else if (stackSize(s) > 2)
    {
        if  (!(  (strncmp(function, "min", 3) == 0)
              || (strncmp(function, "max", 3) == 0)
              || (strncmp(function, "sum", 3) == 0)
              || (strncmp(function, "avg", 3) == 0)
              || (strncmp(function, "mean", 4) == 0)
              || (strncmp(function, "median", 6) == 0)
              || (strncmp(function, "var", 3) == 0)))
        {
            while (stackSize(s) > 0) stackPop(s);
            raise(inputMissing);
            stackPush(s, num2Str(NAN));
            return -EFAULT;
        }
    }
    token input = (token)stackPop(s);
    number num = buildNumber(input);
    number result = num;
    number counter = 0;

    if(strncmp(function, "abs", 3) == 0)
        result = fabs(num);
    else if(strncmp(function, "floor", 5) == 0)
        result = floor(num);
    else if(strncmp(function, "ceil", 4) == 0)
        result = ceil(num);
    else if(strncmp(function, "sin", 3) == 0)
        result = !prefs.mode.degrees ? sin(num) : sin(toRadians(num));
    else if(strncmp(function, "cos", 3) == 0)
        result = !prefs.mode.degrees ? cos(num) : cos(toRadians(num));
    else if(strncmp(function, "tan", 3) == 0)
        result = !prefs.mode.degrees ? tan(num) : tan(toRadians(num));
    else if(strncmp(function, "arcsin", 6) == 0
         || strncmp(function, "asin", 4) == 0)
        result = !prefs.mode.degrees ? asin(num) : toDegrees(asin(num));
    else if(strncmp(function, "arccos", 6) == 0
         || strncmp(function, "acos", 4) == 0)
        result = !prefs.mode.degrees ? acos(num) : toDegrees(acos(num));
    else if(strncmp(function, "arctan", 6) == 0
         || strncmp(function, "atan", 4) == 0)
        result = !prefs.mode.degrees ? atan(num) : toDegrees(atan(num));
    else if(strncmp(function, "sqrt", 4) == 0)
        result = sqrt(num);
    else if(strncmp(function, "cbrt", 4) == 0)
        result = cbrt(num);
    else if(strncmp(function, "log", 3) == 0)
        result = log(num);
    else if(strncmp(function, "exp", 3) == 0)
        result = exp(num);
    else if(strncmp(function, "min", 3) == 0)
    {
        while (stackSize(s) > 0 && strcmp(stackTop(s), FUNCTIONSEPARATOR) != 0)
        {
            input = (token)stackPop(s);
            num = buildNumber(input);
            if (num < result)
                result = num;
        }
    }
    else if(strncmp(function, "max", 3) == 0)
    {
        while (stackSize(s) > 0 && strcmp(stackTop(s), FUNCTIONSEPARATOR) != 0)
        {
            input = (token)stackPop(s);
            num = buildNumber(input);
            if (num > result)
                result = num;
        }
    }
    else if(strncmp(function, "sum", 3) == 0)
    {
        while (stackSize(s) > 0  && strcmp(stackTop(s), FUNCTIONSEPARATOR) != 0)
        {
            input = (token)stackPop(s);
            num = buildNumber(input);
            result += num;
        }
    }
    else if(strncmp(function, "avg", 3) == 0 ||
            strncmp(function, "mean", 4) == 0)
    {
        // Result already initialized with first number
        counter = 1;
        while (stackSize(s) > 0  && strcmp(stackTop(s), FUNCTIONSEPARATOR) != 0)
        {
            input = (token)stackPop(s);
            num = buildNumber(input);
            result += num;
            counter++;
        }
        result /= counter;
    }
    else if(strncmp(function, "median", 6) == 0)
    {
        // needed for sorting
        Stack tmp = FRESH_STACK, safe = FRESH_STACK;
        // Result already initialized with first number
        counter = 1;
        stackInit(&tmp, (stackSize(s) > 0 ? stackSize(s) : 1));
        stackInit(&safe, (stackSize(s) > 0 ? stackSize(s) : 1));
        // add first value to the later sorted stack
        stackPush(&tmp, input);
        while (stackSize(s) > 0  && strcmp(stackTop(s), FUNCTIONSEPARATOR) != 0)
        {
            input = (token)stackPop(s);
            num = buildNumber(input);
            // save all numbers larger as the stack value
            while (stackSize(&tmp) > 0 && buildNumber(stackTop(&tmp)) < num)
            {
                stackPush(&safe, stackPop(&tmp));
            }
            // push value on the sorted stack
            stackPush(&tmp, input);
            // push all saved numbers back on the sorted stack
            while (stackSize(&safe) > 0)
            {
                stackPush(&tmp, stackPop(&safe));
            }
            counter++;
        }
        stackFree(&safe);
        // calculate the median index
        counter = (number)(((int)counter+1)/2);
        // pop all numbers until median index
        while (counter > 1)
        {
            stackPop(&tmp);
            counter--;
        }
        result = buildNumber(stackPop(&tmp));
        // pop the remaining sorted stack
        while (stackSize(&tmp) > 0)
        {
            stackPop(&tmp);
        }
        stackFree(&tmp);
    }
    else if(strncmp(function, "var", 3) == 0)
    {
        Stack tmp = FRESH_STACK;
        counter = 1;
        // second stack to store values during calculation of mean
        stackInit(&tmp, (stackSize(s) > 0 ? stackSize(s) : 1));
        // push first value to temporary stack
        stackPush(&tmp, input);
        number mean = result;
        while (stackSize(s) > 0  && strcmp(stackTop(s), FUNCTIONSEPARATOR) != 0)
        {
            input = (token)stackPop(s);
            // push value to temporary stack
            stackPush(&tmp, input);
            num = buildNumber(input);
            mean += num;
            counter++;
        }
        // calculate mean
        mean /= counter;
        result = 0;
        // calculate sum of squared differences
        while (stackSize(&tmp) > 0)
        {
            input = (token)stackPop(&tmp);
            num = buildNumber(input)-mean;
            result += pow(num,2);
        }
        // determine variance
        result /= counter;
        stackFree(&tmp);
    }
    if (strcmp(stackTop(s), FUNCTIONSEPARATOR) == 0)
        stackPop(s);
    stackPush(s, num2Str(result));
    return 0;
}

int doOp(Stack *s, token op)
{
    int err = 0;
    token roperand = (token)stackPop(s);
    token loperand = (token)stackPop(s);
    number lside = buildNumber(loperand);
    number rside = buildNumber(roperand);
    number ret = NAN;
    switch(*op)
    {
        case '^':
            {
                ret = pow(lside, rside);
            }
            break;
        case '*':
            {
                ret = lside * rside;
            }
            break;
        case '/':
            {
                if(rside == 0)
                {
                    raise(divZero);
                    if (lside == 0)
                        ret = NAN;
                    else
                        ret = INFINITY;
                    err = -EFAULT;
                }
                else
                    ret = lside / rside;
            }
            break;
        case '%':
            {
                if(rside == 0)
                {
                    raise(divZero);
                    if (lside == 0)
                        ret = NAN;
                    else
                        ret = INFINITY;
                    err = -EFAULT;
                }
                else
                {
                    ret = (int)(lside / rside);
                    ret = lside - (ret * rside);
                }
            }
            break;
        case '+':
            {
                ret = lside + rside;
            }
            break;
        case '-':
            {
                ret = lside - rside;
            }
            break;
    }
    stackPush(s, num2Str(ret));
    return err;
}

Symbol type(char ch)
{
    Symbol result;
    switch(ch)
    {
        case '+':
        case '-':
            result = addop;
            break;
        case '*':
        case '/':
        case '%':
            result = multop;
            break;
        case '^':
            result = expop;
            break;
        case '(':
            result = lparen;
            break;
        case ')':
            result = rparen;
            break;
        case '.':
            result = decimal;
            break;
        case ' ':
            result = space;
            break;
        case ',':
            result = argsep;
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
            result = digit;
            break;
        case 'A':
        case 'B':
        case 'C':
        case 'D':
        case 'E':
        case 'F':
        case 'G':
        case 'H':
        case 'I':
        case 'J':
        case 'K':
        case 'L':
        case 'M':
        case 'N':
        case 'O':
        case 'P':
        case 'Q':
        case 'R':
        case 'S':
        case 'T':
        case 'U':
        case 'V':
        case 'W':
        case 'X':
        case 'Y':
        case 'Z':
        case 'a':
        case 'b':
        case 'c':
        case 'd':
        case 'e':
        case 'f':
        case 'g':
        case 'h':
        case 'i':
        case 'j':
        case 'k':
        case 'l':
        case 'm':
        case 'n':
        case 'o':
        case 'p':
        case 'q':
        case 'r':
        case 's':
        case 't':
        case 'u':
        case 'v':
        case 'w':
        case 'x':
        case 'y':
        case 'z':
            result = text;
            break;
        default:
            result = invalid;
            break;
    }
    return result;
}

bool isFunction(token tk)
{
    return ((strncmp(tk, "abs", 3) == 0 && strlen(tk) == 3)
        || (strncmp(tk, "floor", 5) == 0 && strlen(tk) == 5)
        || (strncmp(tk, "ceil", 4) == 0 && strlen(tk) == 4)
        || (strncmp(tk, "sin", 3) == 0 && strlen(tk) == 3)
        || (strncmp(tk, "cos", 3) == 0 && strlen(tk) == 3)
        || (strncmp(tk, "tan", 3) == 0 && strlen(tk) == 3)
        || (strncmp(tk, "arcsin", 6) == 0 && strlen(tk) == 6)
        || (strncmp(tk, "arccos", 6) == 0 && strlen(tk) == 6)
        || (strncmp(tk, "arctan", 6) == 0 && strlen(tk) == 6)
        || (strncmp(tk, "asin", 4) == 0 && strlen(tk) == 4)
        || (strncmp(tk, "acos", 4) == 0 && strlen(tk) == 4)
        || (strncmp(tk, "atan", 4) == 0 && strlen(tk) == 4)
        || (strncmp(tk, "sqrt", 4) == 0 && strlen(tk) == 4)
        || (strncmp(tk, "cbrt", 4) == 0 && strlen(tk) == 4)
        || (strncmp(tk, "log", 3) == 0 && strlen(tk) == 3)
        || (strncmp(tk, "min", 3) == 0 && strlen(tk) == 3)
        || (strncmp(tk, "max", 3) == 0 && strlen(tk) == 3)
        || (strncmp(tk, "sum", 3) == 0 && strlen(tk) == 3)
        || (strncmp(tk, "avg", 3) == 0 && strlen(tk) == 3)
        || (strncmp(tk, "mean", 4) == 0 && strlen(tk) == 4)
        || (strncmp(tk, "median", 6) == 0 && strlen(tk) == 6)
        || (strncmp(tk, "var", 3) == 0 && strlen(tk) == 3)
        || (strncmp(tk, "exp", 3) == 0 && strlen(tk) == 3));
}

bool isSpecialValue(token tk)
{
    return ((strlen(tk) == 3) && ((strncmp(tk, "nan", 3) == 0) || (strncmp(tk, "inf", 3))));
}

Symbol tokenType(token tk)
{
    if (!tk)
        return invalid;
    Symbol ret = type(*tk);
    switch(ret)
    {
        case text:
            if(isFunction(tk))
                ret = function;
            else if(isSpecialValue(tk))
                ret = value;
            else
                ret = identifier;
            break;
        case addop:
            if(*tk == '-' && strlen(tk) > 1)
                ret = tokenType(tk+1);
            break;
        case decimal:
        case digit:
            ret = value;
            break;
        default:
            break;
    }
    return ret;
}

int tokenize(char *str, char *(**tokensRef))
{
    int i = 0;
    char** tokens = NULL;
    char** tmp = NULL;
    char* ptr = str;
    char ch = '\0';
    int numTokens = 0;
    int lparen_count = 0;
    int rparen_count = 0;
    int ops_count = 0;
    if ((!str) || (!tokensRef))
    {
        return -EINVAL;
    }
    if (strlen(str) == 0)
    {
        raise(faultyExpression);
        return -EFAULT;
    }
    char* tmpToken = malloc((prefs.maxtokenlength+1) * sizeof(char));
    if (!tmpToken)
    {
        fprintf(stderr, "Malloc of temporary buffer failed\n");
        return -ENOMEM;
    }
    while((ch = *ptr++))
    {
        if(type(ch) == invalid) // Stop tokenizing when we encounter an invalid character
            break;

        token newToken = NULL;
        tmpToken[0] = '\0';
        Symbol sym = type(ch);
        switch(sym)
        {
            case addop:
                {
                    // Check if this is a negative
                    if(ch == '-'
                        && (numTokens == 0
                            || (tokenType(tokens[numTokens-1]) == addop
                                || tokenType(tokens[numTokens-1]) == multop
                                || tokenType(tokens[numTokens-1]) == expop
                                || tokenType(tokens[numTokens-1]) == lparen
                                || tokenType(tokens[numTokens-1]) == argsep)))
                    {
                        // Assemble an n-character (plus null-terminator) number token
                        {
                            int len = 1;
                            bool hasDecimal = false;
                            bool hasExponent = false;

                            if(type(ch) == decimal) // Allow numbers to start with decimal
                            {
                                //printf("Decimal\n");
                                hasDecimal = true;
                                len++;
                                tmpToken[0] = '0';
                                tmpToken[1] = '.';
                            }
                            else // Numbers that do not start with decimal
                            {
                                tmpToken[len-1] = ch;
                            }

                            // Assemble rest of number
                            for(; // Don't change len
                                *ptr // There is a next character and it is not null
                                && len <= prefs.maxtokenlength
                                && (type(*ptr) == digit // The next character is a digit
                                     || ((type(*ptr) == decimal // Or the next character is a decimal
                                         && hasDecimal == 0)) // But we have not added a decimal
                                     || ((*ptr == 'E' || *ptr == 'e') // Or the next character is an exponent
                                         && hasExponent == false) // But we have not added an exponent yet
                                || ((*ptr == '+' || *ptr == '-') && hasExponent == true)); // Exponent with sign
                                ++len)
                            {
                                if(type(*ptr) == decimal)
                                    hasDecimal = true;
                                else if(*ptr == 'E' || *ptr == 'e')
                                    hasExponent = true;
                                tmpToken[len] = *ptr++;
                            }

                            // Append null-terminator
                            tmpToken[len] = '\0';
                            if (len > 1)
                            {
                                break;
                            }
                        }
                    }
                    // If it's not part of a number, it's an op - fall through
                }
                // The comment below supresses the fallthrough warning.
                // fall through
            case multop:
            case expop:
            case lparen:
            case rparen:
            case argsep:
                // Assemble a single-character (plus null-terminator) operation token
                {
                    tmpToken[0] = ch;
                    tmpToken[1] = '\0';
                    if (sym == lparen) lparen_count++;
                    else if (sym == rparen) rparen_count++;
                    else if (sym == multop || sym == expop || sym == addop) ops_count++;
                }
                break;
            case digit:
            case decimal:
                // Assemble an n-character (plus null-terminator) number token
                {
                    int len = 1;
                    bool hasDecimal = false;
                    bool hasExponent = false;

                    if(type(ch) == decimal) // Allow numbers to start with decimal
                    {
                        //printf("Decimal\n");
                        hasDecimal = true;
                        len++;
                        tmpToken[0] = '0';
                        tmpToken[1] = '.';
                    }
                    else // Numbers that do not start with decimal
                    {
                        tmpToken[len-1] = ch;
                    }

                    // Assemble rest of number
                    for(; // Don't change len
                        *ptr // There is a next character and it is not null
                        && len <= prefs.maxtokenlength
                        && (type(*ptr) == digit // The next character is a digit
                             || ((type(*ptr) == decimal // Or the next character is a decimal
                                 && hasDecimal == 0)) // But we have not added a decimal
                             || ((*ptr == 'E' || *ptr == 'e') // Or the next character is an exponent
                                 && hasExponent == false) // But we have not added an exponent yet
                             || ((*ptr == '+' || *ptr == '-') && hasExponent == true)); // Exponent with sign
                        ++len)
                    {
                        if(type(*ptr) == decimal)
                            hasDecimal = true;
                        else if(*ptr == 'E' || *ptr == 'e')
                            hasExponent = true;
                        tmpToken[len] = *ptr++;
                    }

                    // Append null-terminator
                    tmpToken[len] = '\0';
                }
                break;
            case text:
                // Assemble an n-character (plus null-terminator) text token
                {
                    int len = 1;
                    tmpToken[0] = ch;
                    for(len = 1; *ptr && type(*ptr) == text && len <= prefs.maxtokenlength; ++len)
                    {
                        tmpToken[len] = *ptr++;
                    }
                    tmpToken[len] = '\0';
                }
                break;
            default:
                break;
        }
        // Add to list of tokens
        if(tmpToken[0] != '\0' && strlen(tmpToken) > 0)
        {
            numTokens++;
            /*if(tokens == NULL) // First allocation
                tokens = (char**)malloc(numTokens * sizeof(char*));
            else*/

            newToken = malloc((strlen(tmpToken)+1) * sizeof(char));
            if (!newToken)
            {
                numTokens--;
                break;
            }
            strcpy(newToken, tmpToken);
            newToken[strlen(tmpToken)] = '\0';
            tmp = (char**)realloc(tokens, numTokens * sizeof(char*));
            if (tmp == NULL)
            {
                if (tokens != NULL)
                {
                    for(i=0;i<numTokens-1;i++)
                    {
                        if (tokens[i] != NULL)
                        {
                            free(tokens[i]);
                            tokens[i] = NULL;
                        }
                    }
                    free(tokens);
                    tokens = NULL;
                }
                *tokensRef = NULL;
                free(newToken);
                free(tmpToken);
                return -ENOMEM;
            }
            tokens = tmp;
            tmp = NULL;
            tokens[numTokens - 1] = newToken;
        }
    }
    free(tmpToken);
    tmpToken = NULL;
    if (lparen_count != rparen_count)
    {
        for (int i = 0; i < numTokens; i++) free(tokens[i]);
        free(tokens);
        raise(parenMismatch);
        return -EFAULT;
    }
    if (numTokens == lparen_count + rparen_count + ops_count)
    {
        for (int i = 0; i < numTokens; i++) free(tokens[i]);
        free(tokens);
        raise(faultyExpression);
        return -EFAULT;
    }
    *tokensRef = tokens; // Send back out
    return numTokens;
}

bool leftAssoc(token op)
{
    bool ret = false;
    switch(tokenType(op))
    {
        case addop:
        case multop:

            ret = true;
            break;
        case function:
        case expop:
            ret = false;
            break;
        default:
            break;
    }
    return ret;
}

int precedence(token op1, token op2)
{
    int ret = 0;

    if (op2 == NULL)
        ret = 1;
    else if(tokenType(op1) == tokenType(op2)) // Equal precedence
        ret = 0;
    else if(tokenType(op1) == addop
            && (tokenType(op2) == multop || tokenType(op2) == expop)) // op1 has lower precedence
        ret = -1;
    else if(tokenType(op2) == addop
            && (tokenType(op1) == multop || tokenType(op1) == expop)) // op1 has higher precedence
        ret = 1;
    else if(tokenType(op1) == multop
            && tokenType(op2) == expop) // op1 has lower precedence
        ret = -1;
    else if(tokenType(op1) == expop
            && tokenType(op2) == multop) // op1 has higher precedence
        ret = 1;
    else if (tokenType(op1) == function
            && (tokenType(op2) == addop || tokenType(op2) == multop || tokenType(op2) == expop || tokenType(op2) == lparen))
        ret = 1;
    else if ((tokenType(op1) == addop || tokenType(op1) == multop || tokenType(op1) == expop)
            && tokenType(op2) == function)
        ret = -1;
    return ret;
}

int evalStackPush(Stack *s, token val)
{
    int err = 0;
    if(prefs.display.postfix)
        printf("\t%s\n", val);

    switch(tokenType(val))
    {
        case function:
            {
                err = doFunc(s, val);
                if (err < 0)
                {
                    while (stackSize(s) > 0)
                    {
                        char* t = stackPop(s);
                        free(t);
                    }
                    return err;
                }
            }
            break;
        case expop:
        case multop:
        case addop:
            {
                if(stackSize(s) >= 2)
                {
                    err = doOp(s, val);
                    if (err < 0)
                    {
                        return err;
                    }
                }
                else
                {
                    stackPush(s, val);
                }
            }
            break;
        case value:
            {
                stackPush(s, val);
            }
            break;
        default:
            break;
    }
    return 0;
}

int postfix(token *tokens, int numTokens, Stack *output)
{
    Stack operators = FRESH_STACK, intermediate = FRESH_STACK;
    int i;
    int err = 0;
    stackInit(&operators, numTokens);
    stackInit(&intermediate, numTokens);
    for(i = 0; i < numTokens; i++)
    {
        // From Wikipedia/Shunting-yard_algorithm:
        switch(tokenType(tokens[i]))
        {
            case value:
                {
                    // If the token is a number, then add it to the output queue.
                    //printf("Adding number %s to output stack\n", tokens[i]);
                    err = evalStackPush(output, tokens[i]);
                    if (err < 0)
                    {
                        stackFree(&operators);
                        stackFree(&intermediate);
                        return err;
                    }
                }
                break;
            case function:
                {
                    while(stackSize(&operators) > 0
                        && (tokenType(tokens[i]) != lparen)
                        && ((precedence(tokens[i], (char*)stackTop(&operators)) <= 0)))
                    {
                        //printf("Moving operator %s from operator stack to output stack\n", (char*)stackTop(&operators));
                        err = evalStackPush(output, stackPop(&operators));
                        if (err < 0)
                        {
                            stackFree(&operators);
                            stackFree(&intermediate);
                            return err;
                        }
                        stackPush(&intermediate, stackTop(output));
                    }

                    // If the token is a function token, then push it onto the stack.
                    //printf("Adding operator %s to operator stack\n", tokens[i]);
                    stackPush(&operators, tokens[i]);
                }
                break;
            case argsep:
                {
                    /*
                     * If the token is a function argument separator (e.g., a comma):
                     *     Until the token at the top of the stack is a left
                     *     paren, pop operators off the stack onto the output
                     *     queue. If no left paren encountered, either separator
                     *     was misplaced or parens mismatched.
                     */
                    while(stackSize(&operators) > 0
                        && tokenType((token)stackTop(&operators)) != lparen
                        && stackSize(&operators) > 1)
                    {
                        //printf("Moving operator from operator stack to output stack\n");
                        err = evalStackPush(output, stackPop(&operators));
                        if (err < 0)
                        {
                            stackFree(&operators);
                            stackFree(&intermediate);
                            return err;
                        }
                        stackPush(&intermediate, stackTop(output));
                    }
                }
                break;
            case addop:
            case multop:
            case expop:
                {
                    /*
                     * If the token is an operator, op1, then:
                     *     while there is an operator token, op2, at the top of the stack, and
                     *             either op1 is left-associative and its precedence is less than or equal to that of op2,
                     *             or op1 is right-associative and its precedence is less than that of op2,
                     *         pop op2 off the stack, onto the output queue
                     *     push op1 onto the stack
                     */
                    while(stackSize(&operators) > 0
                        && (tokenType((char*)stackTop(&operators)) == addop || tokenType((char*)stackTop(&operators)) == multop || tokenType((char*)stackTop(&operators)) == expop)
                        && ((leftAssoc(tokens[i]) && precedence(tokens[i], (char*)stackTop(&operators)) <= 0)
                            || (!leftAssoc(tokens[i]) && precedence(tokens[i], (char*)stackTop(&operators)) < 0)))
                    {
                        //printf("Moving operator %s from operator stack to output stack\n", (char*)stackTop(&operators));
                        err = evalStackPush(output, stackPop(&operators));
                        if (err < 0)
                        {
                            stackFree(&operators);
                            stackFree(&intermediate);
                            return err;
                        }
                        stackPush(&intermediate, stackTop(output));
                    }
                    //printf("Adding operator %s to operator stack\n", tokens[i]);
                    stackPush(&operators, tokens[i]);
                }
                break;
            case lparen:
                {
                    // If the token is a left paren, then push it onto the stack
                    //printf("Adding left paren to operator stack\n");
                    if (tokenType(stackTop(&operators)) == function)
                        stackPush(output, FUNCTIONSEPARATOR);
                    stackPush(&operators, tokens[i]);
                }
                break;
            case rparen:
                {
                    /*
                     * If the token is a right paren:
                     *     Until the token at the top of the stack is a left paren, pop operators off the stack onto the output queue
                     *     Pop the left paren from the stack, but not onto the output queue
                     *     If the stack runs out without finding a left paren, then there are mismatched parens
                     */
                    while(stackSize(&operators) > 0
                        && tokenType((token)stackTop(&operators)) != lparen
                        && stackSize(&operators) > 1)
                    {
                        //printf("Moving operator %s from operator stack to output stack\n", (char*)stackTop(&operators));
                        err = evalStackPush(output, stackPop(&operators));
                        if (err < 0)
                        {
                            stackFree(&operators);
                            stackFree(&intermediate);
                            return err;
                        }
                        stackPush(&intermediate, stackTop(output));
                    }
                    if(stackSize(&operators) > 0
                        && tokenType((token)stackTop(&operators)) != lparen)
                    {
                        err = -EFAULT;
                        raise(parenMismatch);
                    }
                    //printf("Removing left paren from operator stack\n");
                    stackPop(&operators); // Discard lparen
                    while (stackSize(&operators) > 0 && tokenType((token)stackTop(&operators)) == function)
                    {
                        //printf("Removing function from operator stack to output stack\n");
                        err = evalStackPush(output, stackPop(&operators));
                        if (err < 0)
                        {
                            stackFree(&operators);
                            stackFree(&intermediate);
                            return err;
                        }
                        stackPush(&intermediate, stackTop(output));
                    }
                }
                break;
            default:
                break;
        }
    }
    /*
     * When there are no more tokens to read:
     *     While there are still operator tokens on the stack:
     *         If the operator token on the top of the stack is a paren, then there are mismatched parens
     *         Pop the operator onto the output queue
     */
    while(stackSize(&operators) > 0)
    {
        if(tokenType((token)stackTop(&operators)) == lparen)
        {
            raise(parenMismatch);
            err = true;
        }
        //printf("Moving operator from operator stack to output stack\n");
        err = evalStackPush(output, stackPop(&operators));
        if (err < 0)
        {
            stackFree(&operators);
            stackFree(&intermediate);
            return err;
        }
        stackPush(&intermediate, stackTop(output));
    }
    // pop result from intermediate stack
    stackPop(&intermediate);
    // free remaining intermediate results
    while (stackSize(&intermediate) > 0)
    {
        token s = stackPop(&intermediate);
        free(s);
    }
    if (err != 0)
    {
        while (stackSize(&operators) > 0)
        {
            token s = stackPop(&operators);
            //printf("Freeing %s from operators stack\n", s);
            free(s);
        }
    }
    stackFree(&intermediate);
    stackFree(&operators);
    return err;
}
/* Added by Thomas Gruber (Thomas.Roehl@fau.de) as interface for LIKWID */
int
calculate_infix(char* finfix, double *result)
{
    int i;
    int ret = 0;
    *result = 0;
    double res = 0;
    token* tokens = NULL;
    Stack expr;
    prefs.maxtokenlength = MAXTOKENLENGTH;
    prefs.precision = MAXPRECISION;
    int numTokens = tokenize(finfix, &tokens);
    if (numTokens < 0)
    {
        return numTokens;
    }
    if (numTokens == 1)
    {
        if (tokenType(tokens[0]) == value)
        {
            *result = strtod((char*)tokens[0], NULL);
        }
        else
        {
            *result = NAN;
        }
        goto freeTokens;
    }
    stackInit(&expr, numTokens);
    ret = postfix(tokens, numTokens, &expr);
    if (ret != 0)
    {
        if (stackSize(&expr) > 0)
        {
            char* t = stackTop(&expr);
            res = atof(t);
            free(t);
        }
        if (prefs.display.errors) printf("\tError evaluating expression\n");;
        goto calcerror;
    }
    else if (stackSize(&expr) != 1)
    {
        raise(faultyExpression);
        if (prefs.display.errors) printf("\tError evaluating expression\n");
        ret = -EFAULT;
        goto calcerror;
    }
    else
    {
        res = atof((char*)stackTop(&expr));
        for (i=0; i< numTokens; i++)
        {
            if (tokens[i] == stackTop(&expr))
                tokens[i] = NULL;
        }
        free(stackPop(&expr));
    }
    *result = res;
    ret = 0;
calcerror:
    stackFree(&expr);
freeTokens:
    for (i=0;i<numTokens;i++)
    {
        if (tokens[i])
        {
            free(tokens[i]);
            tokens[i] = NULL;
        }
    }
    if (tokens)
    {
        free(tokens);
        tokens = NULL;
        numTokens = 0;
    }

    return ret;
}
