# verify-contracts examples

Examples for the `verify-contracts` pass (`Mojo/lib/Transforms/VerifyContracts.cpp`),
which checks `where` contracts right after lifetime checking, before
elaboration and inlining. See `Mojo/proposals/modular-verification.md`.

## Running

`kgen --verify-contracts` parses the file (with the functions it calls),
runs the check pipeline (up to lifetime checking), runs the pass, and stops
before elaboration. Results are diagnostics with source locations. It needs
`z3` on `PATH` (or `z3-path=`).

```bash
./bazelw build --config=build-mojo //Mojo/tools/kgen:kgen //Mojo/stdlib/std:std
bazel-bin/Mojo/tools/kgen/kgen -I bazel-bin/Mojo/stdlib/std \
    Mojo/test/kgen/verify-contracts/straight_line.mojo \
    -elaborate --verify-contracts="verbose=true"
```

With `-lsp=no-dump` instead of `-elaborate`, it runs on the module the
language server checks. That parse is lazy: a stdlib function whose body
was not needed has no body there, and so no contracts, and calls to it are
not checked.

Options: `verbose=true` reports proven obligations as remarks,
`include-stdlib=true` also checks `std`, `rlimit=` sets the solver's
deterministic resource limit (default 100000000; z3 counts it across a
script, so it only stops runaway queries), `wall-seconds=` caps each z3
process (default 60), `dump-dir=` writes the SMT-LIB scripts, and
`cache-dir=` caches the solver's answers by a hash of each script.

Functions are verified in parallel, and their scripts are the same from run
to run, so with `cache-dir=` a function whose encoding did not change costs
no solver time. Changing a callee's contract changes its callers' scripts,
so they are verified again.

An unproven precondition is a warning at the call, with a note at the clause:

```text
straight_line.mojo:78:14: warning: cannot prove the precondition of 'List.__getitem__'
    return xs[i]  # nothing is known about `i`
             ^
list.mojo:1572:34: note: precondition declared here
        ref self, idx: Int where 0 <= idx and idx < len(self), /
                                 ^
```

To compare a file against its expected results (every call in a `bad_*`
function warned about, none in an `ok_*` one):

```bash
f=Mojo/test/kgen/verify-contracts/straight_line.mojo
bazel-bin/Mojo/tools/kgen/kgen -I bazel-bin/Mojo/stdlib/std $f -elaborate \
    --verify-contracts 2>&1 | grep "warning:" |
  grep -oE "^[^ ]*$(basename $f):[0-9]+" |
  cut -d: -f2 | while read l; do
    awk -v L=$l 'NR<=L && /^def /{n=$2} NR==L{print n}' $f | cut -d'(' -f1
  done | sort -u
```

The output must be exactly the `bad_*` functions.

## What is checked

- A function's own preconditions are assumed at its entry.
- At every call to a function with preconditions, they are obligations,
  instantiated with the call's arguments. `xs[i]` is such a call: `List`'s
  `__getitem__(idx: Int)` states `0 <= idx and idx < len(self)`.
- After `assert_true(c)`, `assert_false(c)`, `assert_equal(a, b)` and
  `assert_not_equal(a, b)` from `std.testing`, what they checked holds
  where they did not raise: for `Bool` conditions, and for `Int`, `Bool`
  and integer scalar operands (a type's own `__eq__` is not equality of
  its value, so other types give no fact).
- Integer conversions (`Int(n)` of a `UInt8`, `UInt8(i)`) extend by the
  source's signedness or truncate.
- Integer and Boolean operators are bit-vector and Boolean operations, and
  `len(x)` is an uninterpreted function of `x`'s value (assumed
  non-negative).
- Local variables are followed through `var` declarations, stores and loads,
  `ref` locals to what they refer to, and fields of known structs.
- A call changes only what it can write: memory reachable through its `mut`
  references is unknown after it (`bad_after_mut_call`), and memory passed by
  immutable reference is kept (`ok_after_imm_call`). Its results are unknown.
- Control flow: `if` (with `elif`), `return`, `try`, and loops (`for` over
  `reversed(range(n))`, `reversed(range(start, end))`, `range(n)` and
  `range(start, end)`, `while`, `break`). `range` iteration follows the
  stdlib's definition. At each loop head, what the loop may write is unknown,
  bound by invariants found with Houdini over small templates: bounds
  against 0, against the values before the loop and lengths, and between the
  loop's variables. Only variables the loop's conditions depend on are
  considered. The invariants hold only where the loop is reached.

- Postconditions: a function's own clauses on `out` and `mut` arguments are
  proven where it returns, with `old(...)` evaluated at its entry, and a
  callee's are assumed after the call (only where it did not raise). An
  unproven one is reported at the return, with a note at the clause.
- Quantifiers (`all([... for i in range(lo, hi)])`) are proven for an
  arbitrary index and assumed as real quantifiers.
- List elements: `xs[i]` reads `elem(xs, i)`; writing it keeps the length
  and the other elements. `_same_elements(a._data, b._data, n)` says the
  first `n` elements of two lists are equal, which is how `List.append`,
  `pop` and `_realloc` state what they keep.
- Loop invariants also bound the lengths of lists a loop appends to.
- Generic code is verified once, for every value of its parameters.
  `comptime for` is a loop over an arbitrary iteration, `comptime if` joins
  its arms, and parameter expressions are evaluated where they are
  integer or Boolean operators (`n >= 0` on a parameter `n` is `n >= 0`);
  anything else about a parameter is unknown.

- `List` states its lengths: its constructors (empty, `capacity=`,
  `length=`, literals, `copy=`) and `append`, `pop`, `insert`, `clear`,
  `extend`, `reverse` and `resize`; element access (`xs[i]`, `xs[0]`, and
  the unchecked `unsafe_get` and `unsafe_set`), `pop(i)` and
  `insert(i, ...)` state their index bounds.

- `Span` and contiguous slices: `Span(list=)` is as long as the list,
  `span[i]` requires `i` in range, and `xs[a:b]` (on a list or a span)
  requires `0 <= a <= b <= len(xs)` and has length `b - a`, with absent
  bounds meaning 0 and `len(xs)`. The pass models `Optional` (its
  constructors, `or_else`, `__bool__`) to read the bounds. `List(span)`
  copies a span and is as long as it.
- Strided slices (`xs[a:b:c]`, `xs[::-1]`) copy into a new list whose
  length is that of `range(*slice.indices(len(xs)))`: bounds are
  normalized as `indices` does (negative ones count from the end, out of
  range ones are clamped) and a zero step selects nothing. The stdlib
  states this through two small helpers, `_normalize_bound` and
  `_strided_count`, whose bodies the pass verifies against their
  contracts (`include-stdlib=true`). Integer `//` and `%` round towards
  negative infinity, as Mojo defines them.

- `Array`: `len(a)` is the size parameter in `a`'s type (a number, or a
  parameter such as `n` in generic code), `a[i]` requires `i` in range, and
  `Span(array=)` is as long as the array. A callee's contract that names its
  parameters (`array.length`, `Self.size`) sees the values the call binds.

- `BitSet`: `set`, `clear`, `toggle` and `test` require `0 <= idx < size`.
- `Deque`: `d[i]` requires `i` in range; constructors, `append`,
  `appendleft`, `pop`, `popleft`, `insert`, `remove`, `extend`,
  `extendleft`, `clear`, `reverse` and `rotate` state lengths. A bounded
  deque (`maxlen=`) evicts when full, so these clauses go through its
  `_maxlen`, which constructors state and mutators keep.
- `LinkedList`: `get_nth(i)`, `pop(i)` and `insert(i, ...)` require their
  index in range; constructors, `append`, `prepend`, `pop`, `insert`,
  `extend`, `clear` and `reverse` state lengths. Indices are generic
  (`I: Indexer`): `index(i)` is `i` for an `Int` and the literal's value
  for an `IntLiteral`.
- `Dict` and `Set` state only lengths (nothing about them is indexed):
  empty constructors are empty, literals at most as long as their
  elements (repeats collapse), copies as long as the original, `d[k] = v`,
  `setdefault` and `add` at least 1 and at most one longer, `pop` and
  `remove` one shorter (where they did not raise), `pop(k, default)` and
  `discard` at most one shorter, and `clear` empty.
- Strings: `s.byte_length()` of a `String` or `StringSlice` is `len` of its
  value, and of a literal the literal's length. `s[byte=i]` requires `i`
  in range, `s[byte=a:b]` requires `0 <= a <= b <= s.byte_length()` and has
  `b - a` bytes, `as_bytes()` is as long as the string, `s += t` adds
  `t.byte_length()`, and `String()`, `String(literal)` and the
  `StringSlice` constructors state their lengths. That `s[byte=i]` must
  also fall on a codepoint boundary is not stated.
- Nested collections (`xs[i][j]`, `xs[i].append(v)`,
  `l.get_nth(i).get_nth(j)`): a reference to an element is a place whose
  value is the element; writing it (directly, through a field, or by a
  call that takes it `mut`) writes the element back into the collection,
  which keeps its length. A list literal's elements are the values it is
  given. A call given an interior origin (`xs["element"]`) may change the
  collection's elements but not its length.
- `x.copy()` (the `Copyable` default, `Self(copy=self)`) states what the
  struct's copy constructor states: a copied list, deque, linked list or
  dictionary is as long as the original.
- Element access through `ref self` (`List`, `Deque` and `Array`
  `__getitem__`, `LinkedList.get_nth`) keeps the collection: it reads an
  element and does not write the collection, although its origin may be
  mutable.

Not analyzed yet: loops with loop-carried values, and ranges with a step
(`range(a, b, c)`) or over other integer types. Lists built from other
iterables (the generic constructor states no length) have no length
contract. The obligations inside
unsupported control flow are reported as not analyzed. `append` does not state the value it adds (a
generic `T` has no `==` to state it with).

## Expected results

| File                 | Must stay unproven | Should be proven |
|----------------------|--------------------|------------------|
| `straight_line.mojo` | all `bad_*`        | all `ok_*`       |
| `loops.mojo`         | all `bad_*`        | all `ok_*`       |
| `postconditions.mojo`| all `bad_*`        | all `ok_*`       |
| `comptime.mojo`      | all `bad_*`        | all `ok_*`       |
| `lists.mojo`         | all `bad_*`        | all `ok_*`       |
| `spans.mojo`         | all `bad_*`        | all `ok_*`       |
| `arrays.mojo`        | all `bad_*`        | all `ok_*`       |
| `collections.mojo`   | all `bad_*`        | all `ok_*`       |
| `strings.mojo`       | all `bad_*`        | all `ok_*`       |
| `assertions.mojo`    | all `bad_*`        | all `ok_*`       |
