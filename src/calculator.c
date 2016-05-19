/*
 * =======================================================================================
 *
 *      Filename:  calculator.c
 *
 *      Description:  Infix calculator
 *
 *      Author:   Brandon Mills (bm), mills.brandont@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2016 Brandon Mills
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
/*
 * =======================================================================================
 *
 *      Some changes done for the integration in LIKWID, see inline comments
 *
 *      Version:   4.1
 *      Released:  19.5.2016
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com
 *                Thomas Roehl (tr), thomas.roehl@gmail.com
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
#include <string.h>
#include <math.h> // Temporary
#include <getopt.h>
#include <calculator_stack.h>

#define bool char
#define true 1
#define false 0

#define PI 3.141592653589793

/* Added by Thomas Roehl (Thomas.Roehl@fau.de) to reduce reallocs by allocating a temporary
 * token for parsing as well as for transforming a number to a string.
 */
#define MAXTOKENLENGTH 512

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
    } display;
    struct Mode
    {
        bool degrees;
    } mode;
} prefs;

typedef enum
{
    divZero,
    overflow,
    parenMismatch
} Error;

typedef char* token;
/* Added by Thomas Roehl (Thomas.Roehl@fau.de) to keep track of the
 * intermediate calculation results to free them in the end
 */
token* calcTokens = NULL;
int nrCalcTokens = 0;

typedef double number;

void raise(Error err)
{
    char* msg;
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
    }
    printf("\tError: %s\n", msg);
}

inline unsigned int toDigit(char ch)
{
    return ch - '0';
}

number buildNumber(token str)
{
    number result = 0;
    result = strtod(str, NULL);
    return result;
}

token num2Str(number num)
{
    /* Increased precision by Thomas Roehl (Thomas.Roehl@fau.de) as required for LIKWID */
    token str = (token)malloc((MAXTOKENLENGTH+1)*sizeof(char));
    snprintf(str, 39, "%.20f", num);
    return str;
}



inline number toRadians(number degrees)
{
    return degrees * PI / 180.0;
}

inline number toDegrees(number radians)
{
    return radians * 180.0 / PI;
}

token doFunc(token input, token function)
{
    number num = buildNumber(input);
    number result = num;

    if(strcmp(function, "abs") == 0)
        result = fabs(num);
    else if(strcmp(function, "floor") == 0)
        result = floor(num);
    else if(strcmp(function, "ceil") == 0)
        result = ceil(num);
    else if(strcmp(function, "sin") == 0)
        result = !prefs.mode.degrees ? sin(num) : sin(toRadians(num));
    else if(strcmp(function, "cos") == 0)
        result = !prefs.mode.degrees ? cos(num) : cos(toRadians(num));
    else if(strcmp(function, "tan") == 0)
        result = !prefs.mode.degrees ? tan(num) : tan(toRadians(num));
    else if(strcmp(function, "arcsin") == 0
         || strcmp(function, "asin") == 0)
        result = !prefs.mode.degrees ? asin(num) : toDegrees(asin(num));
    else if(strcmp(function, "arccos") == 0
         || strcmp(function, "acos") == 0)
        result = !prefs.mode.degrees ? acos(num) : toDegrees(acos(num));
    else if(strcmp(function, "arctan") == 0
         || strcmp(function, "atan") == 0)
        result = !prefs.mode.degrees ? atan(num) : toDegrees(atan(num));
    else if(strcmp(function, "sqrt") == 0)
        result = sqrt(num);
    else if(strcmp(function, "cbrt") == 0)
        result = cbrt(num);
    else if(strcmp(function, "log") == 0)
        result = log(num);
    else if(strcmp(function, "exp") == 0)
        result = exp(num);
    printf("Free %s\n", function);
    free(function);
    return num2Str(result);
}

int doOp(token loperand, token op, token roperand, token *result)
{
    /* Added by Thomas Roehl (Thomas.Roehl@fau.de) to return
     * errors from calculation like devide-by-zero, ... */
    int err = 0;
    number lside = buildNumber(loperand);
    number rside = buildNumber(roperand);
    number ret;
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
                    /* Changed by Thomas Roehl */
                    //raise(divZero);
                    err = -1;
                }
                else
                    ret = lside / rside;
            }
            break;
        case '%':
            {
                if(rside == 0)
                {
                    /* Changed by Thomas Roehl */
                    //raise(divZero);
                    err = -1;
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
    *result = num2Str(ret);
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
    return (strcmp(tk, "abs") == 0
        || strcmp(tk, "floor") == 0
        || strcmp(tk, "ceil") == 0
        || strcmp(tk, "sin") == 0
        || strcmp(tk, "cos") == 0
        || strcmp(tk, "tan") == 0
        || strcmp(tk, "arcsin") == 0
        || strcmp(tk, "arccos") == 0
        || strcmp(tk, "arctan") == 0
        || strcmp(tk, "asin") == 0
        || strcmp(tk, "acos") == 0
        || strcmp(tk, "atan") == 0
        || strcmp(tk, "sqrt") == 0
        || strcmp(tk, "cbrt") == 0
        || strcmp(tk, "log") == 0
        || strcmp(tk, "exp") == 0);
}

Symbol tokenType(token tk)
{
    Symbol ret = type(*tk);
    switch(ret)
    {
        case text:
            if(isFunction(tk))
                ret = function;
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
    }
    return ret;
}

int tokenize(char *str, char *(**tokensRef))
{
    char** tokens = NULL;
    char** tmp = NULL;
    char* ptr = str;
    char ch = '\0';
    int numTokens = 0;
    /* Added by Thomas Roehl (Thomas.Roehl@fau.de) to parse string
     * in a temporary token to reduce frequent reallocs. newToken
     * is replaced by tmpToken during parsing. Removed all reallocs
     * and not required mallocs from the original code.
     */
    char* tmpToken = malloc((MAXTOKENLENGTH+1) * sizeof(char));
    if (!tmpToken)
    {
        fprintf(stderr, "Malloc of temporary buffer failed\n");
        return 0;
    }
    while(ch = *ptr++)
    {
        if(type(ch) == invalid) // Stop tokenizing when we encounter an invalid character
            break;

        token newToken = NULL;
        /* Added by Thomas Roehl (Thomas.Roehl@fau.de)
         * Prepare temporary token for next parsing step */
        memset(tmpToken, '\0', MAXTOKENLENGTH+1);
        switch(type(ch))
        {
            case addop:
                {
                    // Check if this is a negative
                    if(ch == '-'
                        && (numTokens == 0
                            || (tokenType(tokens[numTokens-1]) == addop
                                || tokenType(tokens[numTokens-1]) == multop
                                || tokenType(tokens[numTokens-1]) == expop
                                || tokenType(tokens[numTokens-1]) == lparen)))
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
                                //newToken = (char*)malloc((len + 1) * sizeof(char));
                                tmpToken[0] = '0';
                                tmpToken[1] = '.';
                            }
                            else // Numbers that do not start with decimal
                            {
                                //newToken = (char*)malloc((len + 1) * sizeof(char)); // Leave room for '\0'
                                tmpToken[len-1] = ch;
                            }

                            // Assemble rest of number
                            for(; // Don't change len
                                *ptr // There is a next character and it is not null
                                && len <= MAXTOKENLENGTH 
                                && (type(*ptr) == digit // The next character is a digit
                                     || ((type(*ptr) == decimal // Or the next character is a decimal
                                         && hasDecimal == 0)) // But we have not added a decimal
                                     || ((*ptr == 'E' || *ptr == 'e') // Or the next character is an exponent
                                         && hasExponent == false) // But we have not added an exponent yet
                                     /* Added by Thomas Roehl (Thomas.Roehl@fau.de) to parse scientific notation
                                      * with signed exponent correctly
                                      */
                                     || ((*ptr == '+' || *ptr == '-') && hasExponent == true)); // Exponent with sign
                                ++len)
                            {
                                if(type(*ptr) == decimal)
                                    hasDecimal = true;
                                else if(*ptr == 'E' || *ptr == 'e')
                                    hasExponent = true;
                                //newToken = (char*)realloc(newToken, (len + 1) * sizeof(char)); // Leave room for '\0'
                                tmpToken[len] = *ptr++;
                            }

                            // Append null-terminator
                            tmpToken[len] = '\0';
                        }
                        break;
                    }
                    // If it's not part of a number, it's an op - fall through
                }
            case multop:
            case expop:
            case lparen:
            case rparen:
            case argsep:
                // Assemble a single-character (plus null-terminator) operation token
                {
                    //newToken = (char*)malloc(2 * sizeof(char)); // Leave room for '\0'
                    tmpToken[0] = ch;
                    tmpToken[1] = '\0';
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
                        //newToken = (char*)malloc((len + 1) * sizeof(char));
                        tmpToken[0] = '0';
                        tmpToken[1] = '.';
                    }
                    else // Numbers that do not start with decimal
                    {
                        //newToken = (char*)malloc((len + 1) * sizeof(char)); // Leave room for '\0'
                        tmpToken[len-1] = ch;
                    }

                    // Assemble rest of number
                    /* Added support for signed exponents in scientific notation 
                     * by Thomas Roehl (Thomas.Roehl@fau.de) as required for LIKWID */
                    for(; // Don't change len
                        *ptr // There is a next character and it is not null
                        && len <= MAXTOKENLENGTH 
                        && (type(*ptr) == digit // The next character is a digit
                             || ((type(*ptr) == decimal // Or the next character is a decimal
                                 && hasDecimal == false)) // But we have not added a decimal
                             || ((*ptr == 'E' || *ptr == 'e') // Or the next character is an exponent
                                 && hasExponent == false) // But we have not added an exponent yet
                             /* Added by Thomas Roehl (Thomas.Roehl@fau.de) to parse scientific notation
                              * with signed exponent correctly
                              */
                             || ((*ptr == '+' || *ptr == '-') && hasExponent == true)); // Exponent with sign
                        ++len)
                    {
                        if(type(*ptr) == decimal)
                        {
                            hasDecimal = true;
                        }
                        else if(*ptr == 'E' || *ptr == 'e')
                        {
                            hasExponent = true;
                        }
                        //newToken = (char*)realloc(newToken, (len + 1) * sizeof(char)); // Leave room for '\0'
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
                    //newToken = (char*)malloc((len + 1) * sizeof(char)); // Leave room for '\0'
                    tmpToken[0] = ch;
                    for(len = 1; *ptr && type(*ptr) == text && len <= MAXTOKENLENGTH; ++len)
                    {
                        //newToken = (char*)realloc(newToken, (len + 1) * sizeof(char)); // Leave room for '\0'
                        tmpToken[len] = *ptr++;
                    }
                    tmpToken[len] = '\0';
                }
                break;
        }
        // Add to list of tokens
        if(tmpToken[0] != '\0')
        {
            numTokens++;
            /*if(tokens == NULL) // First allocation
                tokens = (char**)malloc(numTokens * sizeof(char*));
            else*/
            /* Added by Thomas Roehl (Thomas.Roehl@fau.de)
             * Allocate new output token and copy temporary token
             */
            newToken = malloc((strlen(tmpToken)+1) * sizeof(char));
            strcpy(newToken, tmpToken);
            newToken[strlen(tmpToken)] = '\0';
            tmp = (char**)realloc(tokens, numTokens * sizeof(char*));
            if (tmp == NULL)
            {
                *tokensRef = NULL;
                free(tmpToken);
                return 0;
            }
            tokens = tmp;
            tmp = NULL;
            tokens[numTokens - 1] = newToken;
        }
    }
    *tokensRef = tokens; // Send back out
    /* Added by Thomas Roehl (Thomas.Roehl@fau.de) */
    free(tmpToken);
    return numTokens;
}

bool leftAssoc(token op)
{
    bool ret;
    switch(tokenType(op))
    {
        case addop:
        case multop:
            ret = true;
            break;
        case expop:
            ret = false;
            break;
    }
    return ret;
}

int precedence(token op1, token op2)
{
    int ret;

    if(tokenType(op1) == tokenType(op2)) // Equal precedence
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

    return ret;
}

int evalStackPush(Stack *s, token val)
{
    /* Added by Thomas Roehl (Thomas.Roehl@fau.de) to return
     * calculation errors. Function now returns an int.
     */
    int ret = 0;
    if(prefs.display.postfix)
        printf("\t%s\n", val);

    switch(tokenType(val))
    {
        case function:
            {
                token operand, res;
                operand = (token)stackPop(s);
                res = doFunc(operand, val);
                //free(operand);
                stackPush(s, res);
            }
            break;
        case expop:
        case multop:
        case addop:
            {
                if(stackSize(s) >= 2)
                {
                    // Pop two operands
                    token l, r, res;
                    r = (token)stackPop(s);
                    l = (token)stackPop(s);

                    // Evaluate
                    /* Added return value by Thomas Roehl (Thomas.Roehl@fau.de) */
                    ret = doOp(l, val, r, &res);
                    // Push result
                    stackPush(s, res);
                    /* Added by Thomas Roehl (Thomas.Roehl@fau.de)
                     * Keeping track of the intermediate results
                     */
                    calcTokens[nrCalcTokens] = res;
                    nrCalcTokens++;
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
    }
    /* Return value by Thomas Roehl (Thomas.Roehl@fau.de) */
    return ret;
}

int postfix(token *tokens, int numTokens, Stack *output)
{
    Stack operators;
    int i;
    int err = 0;
    stackInit(&operators, 2*numTokens);
    for(i = 0; i < numTokens; i++)
    {
        // From Wikipedia/Shunting-yard_algorithm:
        switch(tokenType(tokens[i]))
        {
            case value:
                {
                    // If the token is a number, then add it to the output queue.
                    //printf("Adding number to output stack\n");
                    err = evalStackPush(output, tokens[i]);
                }
                break;
            case function:
                {
                    // If the token is a function token, then push it onto the stack.
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
                        && stackSize(&operators) > 1
                        && err == 0)
                    {
                        //printf("Moving operator from operator stack to output stack\n");
                        token t = (token)stackPop(&operators);
                        err = evalStackPush(output, t);
                        //free(t);
                    }
                    if(stackSize(&operators) > 0
                        && tokenType((token)stackTop(&operators)) != lparen)
                    {
                        err = -1;
                        /* Changed by Thomas Roehl */
                        //raise(parenMismatch);
                    }
                    //printf("Removing left paren from operator stack\n");
                    token t = stackPop(&operators); // Discard lparen
                    //free(t);
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
                            || (!leftAssoc(tokens[i]) && precedence(tokens[i], (char*)stackTop(&operators)) < 0))
                        && err == 0)
                    {
                        //printf("Moving operator from operator stack to output stack\n");
                        token t = (token)stackPop(&operators);
                        err = evalStackPush(output, t);
                        //free(t);
                    }
                    //printf("Adding operator to operator stack\n");
                    stackPush(&operators, tokens[i]);
                }
                break;
            case lparen:
                {
                    // If the token is a left paren, then push it onto the stack
                    //printf("Adding left paren to operator stack\n");
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
                        && stackSize(&operators) > 1
                        && err == 0)
                    {
                        //printf("Moving operator from operator stack to output stack\n");
                        token t = (token)stackPop(&operators);
                        err = evalStackPush(output, t);
                        //free(t);
                    }
                    if(stackSize(&operators) > 0
                        && tokenType((token)stackTop(&operators)) != lparen)
                    {
                        err = -1;
                        /* Changed by Thomas Roehl */
                        //raise(parenMismatch);
                    }
                    //printf("Removing left paren from operator stack\n");
                    token t = (token)stackPop(&operators);
                    //stackPop(&operators); // Discard lparen
                    //free(t);
                }
                break;
        }
        if (err)
            break;
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
            /* Changed by Thomas Roehl */
            //raise(parenMismatch);
            err = -1;
        }
        //printf("Moving operator from operator stack to output stack\n");
        token t = (token)stackPop(&operators);
        err = evalStackPush(output, t);
        //free(t);
    }
    stackFree(&operators);
    return err;
}



/* Added by Thomas Roehl (Thomas.Roehl@fau.de) as interface for LIKWID */
int calculate_infix(char* finfix, double *result)
{
    int i;
    int ret = 0;
    *result = 0;
    token* tokens = NULL;
    Stack expr;
    nrCalcTokens = 0;
    int numTokens = tokenize(finfix, &tokens);
    calcTokens = (token*)malloc(2 * numTokens * sizeof(token));
    if (calcTokens == NULL)
    {
        ret = -1;
        *result = NAN;
    }
    memset(calcTokens, 0, 2 * numTokens * sizeof(token));
    stackInit(&expr, 2*numTokens);
    ret = postfix(tokens, numTokens, &expr);
    if ((stackSize(&expr) != 1) || (ret < 0))
    {
        *result = NAN;
        goto calcerror;
    }
    else
    {
        *result = strtod((char*)stackTop(&expr), NULL);
    }
    ret = 0;
calcerror:
    for (i=0;i<nrCalcTokens; i++)
    {
        if (calcTokens[i] != NULL)
            free(calcTokens[i]);
    }
    if (calcTokens)
        free(calcTokens);
    calcTokens = NULL;
    nrCalcTokens = 0;
    for (i=0;i<numTokens;i++)
    {
        if (tokens[i])
        {
            free(tokens[i]);
        }
    }
    if (tokens)
    {
        free(tokens);
        tokens = NULL;
        numTokens = 0;
    }
    stackFree(&expr);
    return ret;
}


