# Design Notes for `cwisstable`

`cwisstable` is a pure C reimplementation of SwissTable, the hash table data
structure provided by Abseil. This document is not about that data structure,
which is described at https://abseil.io/about/design/swisstables. `cwisstable`
provides no new algorithmic insight not already provided by the countless hours
of work of SwissTable's original designers and authors.

The goal of `cwisstable` is to make this work accessible to *all*, not just C++
users. Because so much code that already exists is written in C, it seems
valuable to be able to drop a SwissTable implementation into a random C project
with minimal effort, and benchmark any improvements. It is actually *easier* to
pull `cwisstable` into a C++ project than to pull in all of Abseil, although we
don't recommend doing that if at all possible (Abseil supports Bazel and CMake
already).

`cwisstable`'s API does not need to be pretty or ergonomic. Ergonomics in C are
unachievable without sacrificing safety or encouraging dialectization, so we
provide a simple API that makes the common tasks easy and everything else
reasonably possible.

This document summarizes how this implementation is *different* from the one in 
Abseil. Abseil's implementation is written in C++ and makes heavy use of
template metaprogramming to achieve specialized optimization for each type. We
also provide a sketch of the SwissTable data structure for reference, and to
show what we get "for free" from the resounding successes of SwissTable.

## Policies

The Abseil implementation implements SwissTable via the `raw_hash_set` type,
which provides sufficient extension points that all of the various flavors of
hash table it offers (sets, maps, flat, node) can be implemented by tuning a
specific template parameter, the *policy*. The policy describes how elements are
manipulated in the table, as well as providing a notion of "key": set elements'
keys are themselves, while map elements' keys are, well, the key in the
key-value pair.

Policies, concretely, are types that are never instantiated and exists to hold
static member functions that get called at various key points, as well as
some relevant types (such as the underlying type of the backing array).

In C, we (fortunately? unfortunately?) don't have templates, so the obvious
solution is to replace these statically-dispatched functions with a vtable, the
assumption being that most applications will want to be inlining everything
anyways, at which point the compiler can aggressively devirtualize the vtable.
Microbenchmarks seem to indicate that this gets us performance comparable to
that of the C++ version, with at worst a 2x slowdown penalty (which is, all
things considered, a great result.)

Policies are defined in `cwisstable/policy.h`, and comprise half of
`ciwsstable`'s public API; unlike C++, where we can detect operations via
templates and ADL, in C, we need to expect callers to define policies
themselves. This API provides helpers to make the simple case easy and
everything else possible. Policy vtables are also use to plumb through the
layout, hash, and equality information of the type, which would normally come
from other type parameters on `raw_hash_set`.

## `CWISS_RawHashSet`

`CWISS_RawHashSet` is the `cwisstable` answer to
`absl::container_internal::raw_hash_set` and serves the same purpose: all
hash-table-ey containers share a bunch of code, so it makes sense for them to
all use the same core data structure implementation.

All of `CWISS_RawHashSet`'s functions take a pointer to a policy; it gets passed
as an argument, rather than being stored in the set itself, to maximize the
chance for it to get inlined away. The compiler is far more likely to see that
a call like `CWISS_RawHashSet_DoSomething(&kMyPolicy, my_set)` is a target for
constant propagation, than an operation like `my_set->policy->...`, where the
construction of `my_set` is likely far removed and not a devirtualization
candidate.

Because `CWISS_RawHashSet` is fully-typed erased, all of its functions are
*violently* type-unsafe. Failing to use the same policy pointer for every
single operation on a `CWISS_RawHashSet` will lead to erratic, incorrect probing
behavior at best, and full-on Undefined Behavior (and its associated fiends and
demons) at worst.

Users are not expected to interact with `CWISS_RawHashSet` in this way directly,
since it is not part of the public API. Instead, `cwisstable/declare.h` exports
a collection of macros for generating new map/set types, which provide a
type-safe wrapper over `CWISS_RawHashSet` that ensure the same policy is always
applied. This also ensures that `CWISS_RawHashSet`'s functions always receive
a pointer to a statically-initialized global variable, maximizing the
probability that devirtualization will succeed.

## Locally-Defined Hash Tables

One massive pitfall with faux-templates in C is trying to make syntax like
OpenSSL's `STACK_OF(T)` macro work, with the desire being that these should
be vocabulary types in peoples' interfaces.

The `cwisstable` observation is that this problem is far too difficult for the
small benefit it gives; you might as well switch to C++. Instead, `cwisstable`
requires you to give a new, personalized name to every specialization it
produces for you. `CWISS_DECLARE_FLAT_HASHSET(MyIntSet, int);` produces a type
`MyIntSet` with a bunch of functions starting with `MyIntSet_`. This type is
suitable to share between files in your project.

Note that, in spite of this design choice, you *must not* include `cwisstable`'s
headers in your public headers if you are vendoring it in, lest someone else
do that too and you wind up with duplicate symbol definitions that don't match
(best case, a linker error; worst case, a mysterious runtime loader error).
`MyIntSet` is not intended to be part of your callers' vocabulary, and your
use of `cwisstable` must remain an implementation detail. If you leak any
`CWISS_` names into your headers, you're asking for all kinds of Hyrum's Law
mayhem.

## Single-Header: Build System Begone!

The landscape of C build systems is vast and unforgiving. Attempting to support
anything more than Bazel and CMake would exhaust any potential value this
project could offer; to say nothing of the nightmare that is dependencies for C.

To sidestep all of that, we leverage the fact that you want to be inlining this
whole library anyways to provide it in single-header form. One
many-thousand-line file that includes nothing but ISO C standard headers,
guaranteed to work with any build system: just vendor the file and forget
about it.

Of course, as noted above, this "bring your own build system" strategy leads to
risks when symbols escape into public headers (or anything they might include
transitively). As a result, we reiterate the warning: *do not* leak `CWISS_`
names into any header your callers could include. Declare your own maps in a
private header and include that in your `.c` files, but don't put it in your
`include` directory (or, wherever else your project keeps its public API).

------

## Appendix: The SwissTable Data Structure

A SwissTable, at its core, is best viewed as a hash table specialized for some
type `T`; this type is equipped with a hash function `h(x)` that produces 64-bit
values and an equality function `x == y` that is compatible with the hash.

A table exports three core operations: insert, find, erase. Insert consumes
a `T` and inserts it into the table. Find takes a `T*` (or something like it,
which supports the same hash function and equality comparison)
and returns a pointer to the corresponding value in the table; if no element is
found an error is returned. Erase takes the result of a find operation and
removes it from the set.

Additionally, a table can be iterated over, which produces elements in an
unspecified order.

(Maps and sets are a special case of this general setup.)

SwissTable implements this interface via the "open-addressing with tombstones"
design. A table is represented by an array of `T`s in memory, plus some bits
of metadata. Each slot in the array can be empty, full, or deleted
(a "tombstone", indicating that an element used to be there and we need to
pretend it still is during find operations).

Insert and erase are implemented in terms of find, so we describe that one
first. To find a value `x`, we compute `h(x)`. From `h(x)` and the current size
of the backing array we construct a *probing sequence* (how this is done is a
parameter of the data structure) that visits indices of the backing array in
some interesting order. We now walk through these indices, checking each
corresponding slot. If the slot is empty, we stop and return an error. If the
slot contains a value `y`, then if `x == y` we are done and return `&y`;
otherwise we continue to the next index. If the slot is deleted, skip it and
continue to the next index.

Insert is implemented in terms of an "unsafe insert", which inserts a value
presumed to not be in the table (violating this requirement will cause the table
to behave erratically). Given `x` and its hash `h(x)`, to insert it, we
construct a probing sequence once again, and use it to find the first non-full
(empty *or* deleted) slot. We place `x` into that slot and mark it as full.

To insert `x`, we compute `h(x)` and first perform a find to see if it's already
present; if it is, we're done. If it's not, we may decide the table is getting
overcrowded (how we decide this is a parameter called the "load factor"); in
this case, we allocate a bigger array, unsafe-insert each element of the table
into the new array, and discard the old backing array. At this point, we may
unsafe-insert `x`.

To erase a value, if we have a pointer into the table (the result of find), we
simply mark that slot as deleted and destroy its contents. If we have a value
`x`, we can perform a find, followed by an erase if successful, to remove it
from the set if it's present.

To iterate, we simply traverse the array, returning only those slots which are
full.

SwissTable takes this relatively common algorithm and improves it on several
points:
1.  The metadata is stored in a side table for the benefit of the cache.
2.  Rather than just storing whether the slot is empty, full, or deleted, each
    slot's metadata (a single octet) also stores the low 7 bits of the hash,
    with the high bit being used to denote the special empty/deleted states.
    This allows for efficient elimination of candidates, cutting down on
    comparisons.
3.  Slots are collected into "groups", and the probe sequence selects groups
    rather than individual slots. Groups are sized in such a way that, given
    the hash value and the metadata for all slots in a group, it is efficient
    to compute a bitset indicating potential match candidates. SIMD instructions
    are especially well-suited to this task.
4.  Various subtle optimizations around minimizing tombstones. Tombstones
    can artificially extend probe sequences, so minimizing them is ideal.
5.  Optimizations for the empty and single-group cases.

The probe sequence, load factor, hash function, and group size are all tuneable
parameters that Abseil's C++ implementation (and, `cwisstable`, following it)
fixes; in particular, group size wants to be the size of the largest register
you can manage: on x86 with extremely modest requirements, this is 16 slots
(i.e. the size of an MMX register).