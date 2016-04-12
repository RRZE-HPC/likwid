/*
 * =======================================================================================
 *
 *      Filename:  calculator.c
 *
 *      Description:  Infix calculator
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
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
    token str = (token)malloc(40*sizeof(char));
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

    return num2Str(result);
}

token doOp(token loperand, token op, token roperand)
{
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
                    raise(divZero);
                else
                    ret = lside / rside;
            }
            break;
        case '%':
            {
                if(rside == 0)
                    raise(divZero);
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
    return num2Str(ret);
}

/*
 * Similar to fgets(), but handles automatic reallocation of the buffer.
 * Only parameter is the input stream.
 * Return value is a string. Don't forget to free it.
 */
char* ufgets(FILE* stream)
{
    unsigned int maxlen = 128, size = 128;
    char* buffer = (char*)malloc(maxlen);

    if(buffer != NULL) /* NULL if malloc() fails */
    {
        char ch = EOF;
        int pos = 0;

        /* Read input one character at a time, resizing the buffer as necessary */
        while((ch = getchar()) != EOF && ch != '\n')
        {
            buffer[pos++] = ch;
            if(pos == size) /* Next character to be inserted needs more memory */
            {
                size = pos + maxlen;
                buffer = (char*)realloc(buffer, size);
            }
        }
        buffer[pos] = '\0'; /* Null-terminate the completed string */
    }
    return buffer;
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

token tokenizeNumber(char *str)
{
    ;
}

int tokenize(char *str, char *(**tokensRef))
{
    char** tokens = NULL;
    char* ptr = str;
    char ch = '\0';
    int numTokens = 0;
    while(ch = *ptr++)
    {
        if(type(ch) == invalid) // Stop tokenizing when we encounter an invalid character
            break;

        token newToken = NULL;
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
                                newToken = (char*)malloc((len + 1) * sizeof(char));
                                newToken[0] = '0';
                                newToken[1] = '.';
                            }
                            else // Numbers that do not start with decimal
                            {
                                newToken = (char*)malloc((len + 1) * sizeof(char)); // Leave room for '\0'
                                newToken[len-1] = ch;
                            }

                            // Assemble rest of number
                            for(; // Don't change len
                                *ptr // There is a next character and it is not null
                                && (type(*ptr) == digit // The next character is a digit
                                     || ((type(*ptr) == decimal // Or the next character is a decimal
                                         && hasDecimal == 0)) // But we have not added a decimal
                                     || ((*ptr == 'E' || *ptr == 'e') // Or the next character is an exponent
                                         && hasExponent == false)); // But we have not added an exponent yet
                                ++len)
                            {
                                if(type(*ptr) == decimal)
                                    hasDecimal = true;
                                else if(*ptr == 'E' || *ptr == 'e')
                                    hasExponent = true;
                                newToken = (char*)realloc(newToken, (len + 1) * sizeof(char)); // Leave room for '\0'
                                newToken[len] = *ptr++;
                            }

                            // Append null-terminator
                            newToken[len] = '\0';
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
                    newToken = (char*)malloc(2 * sizeof(char)); // Leave room for '\0'
                    newToken[0] = ch;
                    newToken[1] = '\0';
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
                        newToken = (char*)malloc((len + 1) * sizeof(char));
                        newToken[0] = '0';
                        newToken[1] = '.';
                    }
                    else // Numbers that do not start with decimal
                    {
                        newToken = (char*)malloc((len + 1) * sizeof(char)); // Leave room for '\0'
                        newToken[len-1] = ch;
                    }

                    // Assemble rest of number
                    /* Added support for signed exponents in scientific notation 
                     * by Thomas Roehl (Thomas.Roehl@fau.de) as required for LIKWID */
                    for(; // Don't change len
                        *ptr // There is a next character and it is not null
                        && (type(*ptr) == digit // The next character is a digit
                             || ((type(*ptr) == decimal // Or the next character is a decimal
                                 && hasDecimal == false)) // But we have not added a decimal
                             || ((*ptr == 'E' || *ptr == 'e') // Or the next character is an exponent
                                 && hasExponent == false) // But we have not added an exponent yet
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
                        newToken = (char*)realloc(newToken, (len + 1) * sizeof(char)); // Leave room for '\0'
                        newToken[len] = *ptr++;
                    }

                    // Append null-terminator
                    newToken[len] = '\0';
                }
                break;
            case text:
                // Assemble an n-character (plus null-terminator) text token
                {
                    int len = 1;
                    newToken = (char*)malloc((len + 1) * sizeof(char)); // Leave room for '\0'
                    newToken[0] = ch;
                    for(len = 1; *ptr && type(*ptr) == text; ++len)
                    {
                        newToken = (char*)realloc(newToken, (len + 1) * sizeof(char)); // Leave room for '\0'
                        newToken[len] = *ptr++;
                    }
                    newToken[len] = '\0';
                }
                break;
        }
        // Add to list of tokens
        if(newToken != NULL)
        {
            numTokens++;
            /*if(tokens == NULL) // First allocation
                tokens = (char**)malloc(numTokens * sizeof(char*));
            else*/
                tokens = (char**)realloc(tokens, numTokens * sizeof(char*));
            tokens[numTokens - 1] = newToken;
        }
    }
    *tokensRef = tokens; // Send back out
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

void evalStackPush(Stack *s, token val)
{
    if(prefs.display.postfix)
        printf("\t%s\n", val);

    switch(tokenType(val))
    {
        case function:
            {
                token operand, res;
                operand = (token)stackPop(s);
                res = doFunc(operand, val);
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
                    res = doOp(l, val, r);

                    // Push result
                    stackPush(s, res);
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
}

bool postfix(token *tokens, int numTokens, Stack *output)
{
    Stack operators;
    int i;
    bool err = false;
    stackInit(&operators);
    for(i = 0; i < numTokens; i++)
    {
        // From Wikipedia/Shunting-yard_algorithm:
        switch(tokenType(tokens[i]))
        {
            case value:
                {
                    // If the token is a number, then add it to the output queue.
                    //printf("Adding number to output stack\n");
                    evalStackPush(output, tokens[i]);
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
                        && stackSize(&operators) > 1)
                    {
                        //printf("Moving operator from operator stack to output stack\n");
                        evalStackPush(output, stackPop(&operators));
                    }
                    if(stackSize(&operators) > 0
                        && tokenType((token)stackTop(&operators)) != lparen)
                    {
                        err = true;
                        raise(parenMismatch);
                    }
                    //printf("Removing left paren from operator stack\n");
                    stackPop(&operators); // Discard lparen
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
                        //printf("Moving operator from operator stack to output stack\n");
                        evalStackPush(output, stackPop(&operators));
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
                        && stackSize(&operators) > 1)
                    {
                        //printf("Moving operator from operator stack to output stack\n");
                        evalStackPush(output, stackPop(&operators));
                    }
                    if(stackSize(&operators) > 0
                        && tokenType((token)stackTop(&operators)) != lparen)
                    {
                        err = true;
                        raise(parenMismatch);
                    }
                    //printf("Removing left paren from operator stack\n");
                    stackPop(&operators); // Discard lparen
                }
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
        evalStackPush(output, stackPop(&operators));
    }
    //stackFree(&operators);
    return err;
}

char* substr(char *str, size_t begin, size_t len)
{
    if(str == NULL
        || strlen(str) == 0
        || strlen(str) < (begin+len))
        return NULL;

    char *result = (char*)malloc((len + 1) * sizeof(char));
    int i;
    for(i = 0; i < len; i++)
        result[i] = str[begin+i];
    result[i] = '\0';
    return result;
}

bool strBeginsWith(char *haystack, char *needle)
{
    bool result;
    if(strlen(haystack) < strlen(needle))
    {
        return false;
    }
    else
    {
        char *sub = substr(haystack, 0, strlen(needle));
        result = (strcmp(sub, needle) == 0);
        free(sub);
        sub = NULL;
    }
    return result;
}

int strSplit(char *str, const char split, char *(**partsRef))
{
    char **parts = NULL;
    char *ptr = str;
    char *part = NULL;
    int numParts = 0;
    char ch;
    int len = 0;
    while(1)
    {
        ch = *ptr++;

        if((ch == '\0' || ch == split) && part != NULL) // End of part
        {
            // Add null terminator
            part = (char*)realloc(part, (len+1) * sizeof(char));
            part[len] = '\0';

            // Add to parts
            numParts++;
            if(parts == NULL)
                parts = (char**)malloc(sizeof(char**));
            else
                parts = (char**)realloc(parts, numParts * sizeof(char*));
            parts[numParts - 1] = part;
            part = NULL;
            len = 0;
        }
        else // Add to part
        {
            len++;
            if(part == NULL)
                part = (char*)malloc(sizeof(char));
            else
                part = (char*)realloc(part, len * sizeof(char));
            part[len - 1] = ch;
        }

        if(ch == '\0')
            break;
    }
    *partsRef = parts;
    return numParts;
}

/*bool isCommand(char *str)
{
    bool result = false;
    char **words = NULL;
    int len = strSplit(str, ' ', &words);
    if(len >= 1 && strcmp(words[0], "set") == 0)
    {
        result = true;
    }
    return result;
}*/

bool execCommand(char *str)
{
    bool recognized = false;
    char **words = NULL;
    int len = strSplit(str, ' ', &words);
    if(len >= 1 && strcmp(words[0], "get") == 0)
    {
        if(len >= 2 && strcmp(words[1], "display") == 0)
        {
            if(len >= 3 && strcmp(words[2], "tokens") == 0)
            {
                recognized = true;
                printf("\t%s\n", (prefs.display.tokens ? "on" : "off"));
            }
            else if(len >= 3 && strcmp(words[2], "postfix") == 0)
            {
                recognized = true;
                printf("\t%s\n", (prefs.display.postfix ? "on" : "off"));
            }
        }
        else if(len >= 2 && strcmp(words[1], "mode") == 0)
        {
            recognized = true;
            printf("\t%s\n", (prefs.mode.degrees ? "degrees" : "radians"));
        }
    }
    else if(len >= 1 && strcmp(words[0], "set") == 0)
    {
        if(len >= 2 && strcmp(words[1], "display") == 0)
        {
            if(len >= 3 && strcmp(words[2], "tokens") == 0)
            {
                if(len >= 4 && strcmp(words[3], "on") == 0)
                {
                    recognized = true;
                    prefs.display.tokens = true;
                }
                else if(len >= 4 && strcmp(words[3], "off") == 0)
                {
                    recognized = true;
                    prefs.display.tokens = false;
                }
            }
            else if(len >= 3 && strcmp(words[2], "postfix") == 0)
            {
                if(len >= 4 && strcmp(words[3], "on") == 0)
                {
                    recognized = true;
                    prefs.display.postfix = true;
                }
                else if(len >= 4 && strcmp(words[3], "off") == 0)
                {
                    recognized = true;
                    prefs.display.postfix = false;
                }
            }
        }
        else if(len >= 2 && strcmp(words[1], "mode") == 0)
        {
            if(len >= 3 && strcmp(words[2], "radians") == 0)
            {
                recognized = true;
                prefs.mode.degrees = false;
            }
            else if(len >= 3 && strcmp(words[2], "degrees") == 0)
            {
                recognized = true;
                prefs.mode.degrees = true;
            }
        }
    }

    return recognized;
}

/* Added by Thomas Roehl (Thomas.Roehl@fau.de) as interface for LIKWID */
int calculate_infix(char* finfix, double *result)
{
    int i;
    int ret = 0;
    *result = 0;
    token* tokens = NULL;
    Stack expr;
    int numTokens = tokenize(finfix, &tokens);
    stackInit(&expr);
    postfix(tokens, numTokens, &expr);
    if(stackSize(&expr) != 1)
    {
        ret = -1;
        goto calcerror;
    }
    else
    {
        *result = strtod((char*)stackTop(&expr), NULL);
    }
    ret = 0;
calcerror:
    for(i = 0; i < numTokens; i++)
    {
        free(tokens[i]);
    }
    free(tokens);
    tokens = NULL;
    numTokens = 0;
    stackFree(&expr);
    return ret;
}


