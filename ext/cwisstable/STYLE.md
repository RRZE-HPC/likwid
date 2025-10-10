# Style Guide

As a rule, we follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
where it happens to be compatible with C.

We make the following deviations and additions.
- Documentation comments start with `///`.
- All symbols visible from public headers must be prefixed with `CWISS_`, except
  for the include guards, which start with `CWISSTABLE_`.
  - Other Google C++ conventions apply: constants are named `CWISS_kMyConst`,
    types and functions are named `CWISS_MyThing`.
- Functions can be associated with a struct by including the struct's name
  in function's name: `CWISS_MyType_MyFunction()`. The same applies for types
  and constants (e.g. `CWISS_MyType_kMyConst`).
- Function-like macros that are intended to be used as functions should be named
  as though they were real functions; variable-like macros intended to be used
  as constants should be named as though they are real constants. All other
  macros (e.g. those that expand to definitions) are named `CWISS_SHOUTY_CASE`.
  - Arguments of macros must end with an underscore.
- Functions which are emulating a C++ standard API (namely, the
  `std::unordered_set` API) should use `CWISS_MyType_snake_case()`. This
  includes functions which we had to make up names for, such as `_new()`
  (default constructor), `_dup()` (copy constructor), and `_next()` (iteration).
- Data members of a struct that are nominally private (i.e., functions that
  aren't associated with the struct shouldn't be touching them) should have
  trailing underscores, as if they were data members of a C++ class.
- Braces on `if/for/while` are mandatory.