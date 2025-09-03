# LIKWID Coding Guidelines

## Overview

In LIKWID we would like the code to follow basic guidelines.
The goal being to make the code easy to read, easy to develop, and easy to debug.

Of course LIKWID has to deal with things, which you do not encounter in normal programs.
These include weird bitfield decoding, inline assembly, and generally a lot of non-portable code.
While good abstraction of such code is encouraged, we understand that it is not always possible.
Please make sure to separate architecture/hardware depenend code appropriately.

One of the main reasons for introducing these guidelines is LIKWID's old code base.
Especially older code used to (and still does) follow conventions, which could be considered bad practice in C.
Often these were presumably a consequence of poor understanding of C and its standard library.

The former paragraph may suggest that at least some conventions were followed.
This is also a not entirely true statement, as older code often does the same thing in very different ways.

Going forward, we would like the code to be more consistent.
The code should use the C language, as it is intended to use and in a way to avoid programming errors.
At the time being, a lot of code still does not follow these rules.
That is because changing all the code at once would be a lot of work.
Also because some of the code is subject to major refactoring, rendering guideline fixes useless.

## Guidelines

### Keep your code compact

Most of the time, less code is simpler code.
Simpler code is easier to understand and subject to less errors.
Avoid making the code unnecessarily complicated.
Do not perform work, which doesn't need to be done.

For example, here is a piece of code, which you will find in a lot of older parts of LIKWID:
```c
int len = 257;
char *newstring = (char *)malloc(len * sizeof(char));
strncpy(newstring, oldstring, len);
newstring[strlen(oldstring) - 1] = '\0';
return newstring;
```

The code above is not only unnecessarily complicated.
It is also unsafe.
So what's all wrong about it?

- Strings with more than 256 characters are omitted.
- `malloc` returns `void *`. Casting `void *` to other pointer types in C is perfectly legal. No explicit cast is necessary (even with `-Wconversion`).
- `strncpy` correctly limits the number of bytes copied, but it misses the terminating nullbyte.
- The manual copy (which `strncpy` doesn't perform), is to the wrong index. If the old string is longer than 257 characters, the nullbyte is written outside the malloc'd area.

Now, what you should do instead is to be aware of the standard library (POSIX is ok) and use it instead:
```c
return strdup(oldstring);
```

The code above doesn't limit the string length.
If you'd actually wanted to limit the length, do this:

```c
return strndup(oldstring, 256);
```

You can apply this rule to your own code too.
If you perform a certain operation repeatedly, put it into its own function.

Another very common pattern in old LIKWID code is the following:
```c
if (strncmp(inputstringA, inputstringB, strlen(inputstringA)) == 0)
    // ...
```

The `strlen` there doesn't serve any purpose if the compared length is exactly the same as the shorter string of the two.
Just use `strcmp` instead:
```c
if (strcmp(inputstringA, inputstringB) == 0)
    // ...
```

There is of course a semantic difference between the two versions.
The first variant ignores a trailing suffix, while the second does not.
In most cases the second one is the one you probably want to use.

Your takeaway message should be:
Do not overthink your code, but avoid doing things, which are just unnecessary bloated.

### Use C99 style

Always declare variables in the innermost possible scope.
This prevents variables retaining values from unexpected scopes and makes catching uninitialized variables much easier.

```c
int i;

i = count_things();
do_things(i);

for (i = 0; i < 10; i++) {
    foo(i);
}

for (i = 0; i < 10; i++) {
    foo(i);
}
```

In the example above we use the variable i for counting both loops, as well as temporary between `count_things` and `do_things`.
You may have a slight argument for the `i` being used in both for loops, but using `i` for an entirely different purpose should be fixed.
We do not live in a time where the compiler run into issues, because we are using too many variables.

Instead do:

```c
int count = count_things();
do_things(count);

for (int i = 0; i < 10; i++) {
    foo(i);
}

for (int i = 0; i < 10; i++) {
    foo(i);
}
```

Now, if you actually need the value `i` outside the for-loop, then it is of course fine to do so.
But reusing variables for different purposes should be avoided under all circumstances!

### Declare constant data with const

If data (global or static variables) is not to be modified during the lifetime of the program, mark it as `const`.
Doing so will ensure that the compiler warns you about erroneously writing there.

For example, a very common thing in LIKWID is to have tables with data definitions.
These should be all marked as `const`.
If you need to make modiciations to these tables, create a copy and then modify the data.
This avoids problems where at one part of the program you may need to e.g. reinitialize LIKWID and the original table values are all/partially deleted/overwritten.

### Declare char * as const char * if possible

Another more evil case are strings.
If a `char *` is known to always point at a string literal, make the pointer `const char *`.
That way there is no confusion whether the string has to be free'd or not after you're done.

Should there be the possibility that the string is dynamically created, use `char *`, but always make sure to copy the strings.
Otherwise the program will segfault when trying to free a string from a string literal.

The other evil thing about C is that string literals are of type `char *` by default.
However, writing to them will usually trigger a segmentation fault to read-only memory.
In the future we intend to turn on `-Wwrite-strings`, which makes string literals of type `const char *`.
One way to avoid accidentally writing or free'ing those string literals is to convert them to `const char *` during their first usage (either as function argument or variable).

### Make pointers in function parameters const if possible

When calling a function it is often unclear, whether the function will modify the passed data or not.
Therefore you should always use const pointers.
This also avoids issues with string literals being passed to a function, since they cannot be modified.

### Prefer using a single error handler for the entire function

Error handling in C is unfortunately annoying either way.
However, there are ways of making it less annoying.
One thing, which you may find in older LIKWID code customized cleanup code for every single error case.
This is error prone, especially with many allocations and error cases.

Don't:

```c
// ...
int *array1 = xmalloc(123);
int err = dothings();
if (err < 0) {
    free(array1);
    return err;
}
int *array2 = xmalloc(456);
err = dothings();
if (err < 0) {
    free(array1);
    free(array2);
    return err;
}
int *array3 = xmalloc(789);
err = dothings();
if (err < 0) {
    free(array1);
    free(array2);
    free(array3);
    return err;
}
// ...
```

Do:

```c
  int *array1 = xmalloc(123);
  int *array2 = NULL;
  int *array3 = NULL;
  
  int err = dothings();
  if (err < 0)
      goto error;
  
  array2 = xmalloc(123);
  
  err = dothings();
  if (err < 0)
      goto error;
  
  array3 = xmalloc(123);
  
  err = dothings();
  if (err < 0)
      goto error;
  
  // ...
  
  return 0;
  
  // ...
error:
  free(array1);
  free(array2);
  free(array3);
  return err;
}
```

In the latter example you can see there is only a single error handler for all the errors.
This requires significantly less thoughts being spent on which buffer to free in which error case.

While the gain in the shown example is questionable, the first case becomes way worse if more allocations are performed in a function.
Remember that calling `free` on a NULL pointer is perfectly safe.
Take care of other functions though (e.g. `fclose`).

Care should be taken that all pointers, which are free'd in the error case, are initialized (i.e. not undefined).
This may happen because you skip over a pointer initialization via `goto`.
Make sure to initialize all pointers/handles before the first `goto`.
With warnings enabled, your compiler should warn you if you get this wrong.

### Format your code

LIKWID provides a `.clang-format` file with all necessary code formatting rules.
Use `make format` to apply these rules.

Currently, perfmon headers are treated specially, since they contain large tables, where we need an increased column limit.

### Using somewhat clear variable names

Do not use short or unclear variable names.
For example, avoid the following:

```c
sscanf(line,"%s %s %d %s = %d", structure, field, &tmp, value, &tmp1);
```

Variables do not have to be entirely self explanatory as long as the context provides enough information.
Still, names like `tmp` or `tmp1` should be avoided.

For single letter variables or amigous names like tmp, the context *must* immediately provide information what the variable does.
Do not use something like tmp if you don't use this value immediately in the next couple of lines.
Single letter variables are okay (e.g. i, j) in for-loops or if the letter abbreviates a very obvious thing in the current context.

We are not mathematicians who try to write out a formula.
The person after you will probably not understand the code you wrote on their first read.
So please make their life easier by making variable names meaningful.
Though, `myVariableToHostCpuMappingTableWithPrefix` might be a bit excessive as well.

### Do not use unsafe string functions

- `sprintf` -> `snprintf`
- `strcpy`  -> `snprintf`
- `strncpy`

While `strncpy` performs bounds checking, it does not write a terminating null byte.
Because `strlcpy` is not part of the standard C library, you can fall back to use `snprintf`, which always writes a terminating null byte.

### Do not hard code values

If a value can be derived otherwise, do not hard code it.

Don't:

```c
char buffer[256];
// ...
snprintf(buffer, 256, "hello");
```

Do:

```c
char buffer[256];
// ...
snprintf(buffer, sizeof(buffer), "hello");
```

When allocating data of a certain pointer type, derive the element size using the actual variable.
See the example below.

Don't:

```c
int *foobar = malloc(mysize * sizeof(int));
```

Do:

```c
int *foobar = malloc(mysize * sizeof(*foobar));
```

This makes it easier to change the type of that memory allocation and reduces programming errors, where one would specify two different types.

### Correctly opening files

Files should be opened with fopen, unless you need to set flags manually for open.
Do not check previously if the file exists with access, stat, etc.
This can cause race conditions and is not a reliable way to check whether a file can be opened or not.
Simply open the file and check if the file was opened correctly.

### Avoid bulky branches

If a code branch contains a lot of code, do not ident the code (if possible).

Don't:

```c
void myfunc(const char *filepath) {
    FILE *f = fopen(filepath, "r");
    if (f) {
        //
        // many lines of code...
        //
    }
    return;
}
```

Do:

```c
void myfunc(const char *filepath) {
    FILE *f = fopen(filepath, "r");
    if (!f)
        return;

    //
    // many lines of code...
    //
}
```

This code layout makes it easier to follow its intended flow.
All code, which is not indented (or intented to a lesser degree) is considered the "expected" flow.
If however code is indented one, two, or even three times it suggestes that the code is only executed only under specific circumstances, which should be consistent with what is actually happening.

This guideline should be fairly straight forward to follow for things like return statements or exit calls (function scope).
It is equally valid for continue and break:

Don't:

```c
for (size_t i = 0; i < 100; i++) {
    if (myarr[i]) {
        //
        // many lines of code
        //
        if (myarr[i].data) {
            //
            // more lines of code
            //
        }
    }
}
```

```c
for (size_t i = 0; i < 100; i++) {
    if (!myarr[i])
        continue;

    //
    // many lines of code
    //

    if (!myarr[i].data)
        continue;

    //
    // more lines of code
    //
}
```

The less the code is indented, the easier it is to read, and it allows for more headroom if you actually need indentation.
In the first of the two examples, you may argue that it's still reasonably easy to read.
Though, if these are four or five nested if statements, things get really messy.

### Do not use int type for binary logic

Technically C doesn't have a native `bool` type, but for readability you should use `bool` from the C99 stdbool header.
Otherwise it is not clear if the variable also takes values other than 0 or 1 and how they should be interpreted.

Don't:

```c
int myfunc(const char *str) {
    if (strcmp(str, "hellothere") != 0)
        return 0;

    // do more things
    return 1;
}
```

Do:

```c
#include <stdbool.h>

bool myfunc(const char *str) {
    if (strcmp(str, "hellothere") != 0)
        return false;

    // do more things
    return true;
}
```
