# `cwisstable.h` - SwissTables for All

`cwisstable` is a single-header C11 port of the Abseil project's
[SwissTables](https://abseil.io/about/design/swisstables).
This project is intended to bring the proven performance and flexibility
of SwissTables to C projects which cannot require a C++ compiler or for projects
which otherwise struggle with dependency management.

The public API is currently in flux, as is test coverage, but the C programs in
the `examples` directory illustrate portions of the API.

Code in this project follows the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
where applicable; variations due to C are described in [STYLE.md](./STYLE.md).

## Getting It

> TL;DR: run `./unified.py cwisstable/*.h`, grab `cwisstable.h`, and
> get cwissing.

There are two options for using this library: Bazel, or single-header.

Bazel support exists by virtue of that being how the tests and benchmarks are
build and run. All development on `cwisstable` itself requires Bazel. Downstream
projects may use `git_repository` to depend on this workspace; two build
targets are available:
- `//:split` is the "native" split form of the headers (i.e., as they are
  checked in).
- `//:unified` is the generated unified header, which is functionality
  identical.

Most C projects use other build systems: Make, CMake, Autoconf, Meson, and
artisanal shell scripts; expecting projects to fuss with Bazel is an
accessibility barrier. This is the raison d-être: you generate the unified
header, vendor it into your project, and never think about it again. Even
generating the file doesn't require installing Bazel; all you need is Python:

```sh
git clone https://github.com/google/cwisstable.git && cd cwisstable
./unify.py
```

This will output a `cwisstable.h` file that you can vendor in; the checkout
and generation step is only necessary for upgrading the header.

That said, if you're writing C++, this library is very much not for you.
Please use https://github.com/abseil/abseil-cpp instead!

## Using It

To use `cwisstable`, include `cwisstable.h` and use the code-generating macros
to create a new map type:

```c
#include "cwisstable.h"

CWISS_DECLARE_FLAT_HASHSET(MyIntSet, int);

int main(void) {
  MyIntSet set = MyIntSet_new(8);

  for (int i = 0; i < 8; ++i) {
    int val = i * i + 1;
    MyIntSet_insert(&set, &val);
  }

  int k = 4;
  assert(!MyIntSet_contains(&set, &k));
}
```

[`cwisstable/declare.h`](cwisstable/declare.h) has a detailed description of
these macros and their generated API.

[`cwisstable/policy.h`](cwisstable/policy.h) describes how to use hash table
policies to define more complex sets and maps.
[`examples/stringmap.c`](examples/stringmap.c) shows this in action!

## Compatibility Warnings

We don't version `cwisstable.h`; instead, users are expected to vendor the
unified file into their projects and manually update it; `cwisstable`'s
distribution method (or, lack of one) is intended to eliminate all friction to
its use in traditional C projects.

That said, we do not provide any kind of ABI or API stability from one revision
to another that would allow `cwisstable.h` to appear in another library's public
headers. You must not, under any circumstances, expose this header in your
public headers. Doing so may clash with _another_ project making the same
mistake, and potentially destroy compile times due to the enormous number of
inline functions generated in every translation unit.

Including `cwisstable.h` into a public header will also cause it to become part
of _your_ API, due to [Hyrum's Law](https://www.hyrumslaw.com/), even if our
types do not appear in your interfaces. Such a mistake may be difficult to
unwind, and we will not provide support for it.

The *intended* use of `cwisstable.h` is that it will either be included directly
into your `.c` files, or a private `.h` file that declares common table types
which is then includes in your `.c` files.

---

This is not an officially supported Google product.
