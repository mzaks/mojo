# verify-contracts examples

Examples for the `verify-contracts` pass (`Mojo/lib/Transforms/VerifyContracts.cpp`),
which checks `where` contracts right after lifetime checking, before
elaboration and inlining. See `Mojo/proposals/modular-verification.md`.

## Running

The pass runs on the module `kgen -lsp` checks, and reports through the same
diagnostics, with source locations. It needs `z3` on `PATH` (or `z3-path=`).

```bash
./bazelw build --config=build-mojo //Mojo/tools/kgen:kgen //Mojo/stdlib/std:std
bazel-bin/Mojo/tools/kgen/kgen -I bazel-bin/Mojo/stdlib/std \
    Mojo/test/kgen/verify-contracts/straight_line.mojo \
    -lsp=no-dump --verify-contracts="verbose=true"
```

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
bazel-bin/Mojo/tools/kgen/kgen -I bazel-bin/Mojo/stdlib/std $f -lsp=no-dump \
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
- Integer and Boolean operators are bit-vector and Boolean operations, and
  `len(x)` is an uninterpreted function of `x`'s value (assumed
  non-negative).
- Local variables are followed through `var` declarations, stores and loads,
  `ref` locals to what they refer to, and fields of known structs.
- A call changes only what it can write: memory reachable through its `mut`
  references is unknown after it (`bad_after_mut_call`), and memory passed by
  immutable reference is kept (`ok_after_imm_call`). Its results are unknown.
- Control flow: `if`, `return`, `try`, and loops (`for` over `range(n)` and
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

Not analyzed yet: loops with loop-carried values, and
`reversed(range(...))` (a strided range). The obligations inside
unsupported control flow are reported as not analyzed. Constructors and
literals do not state their lengths yet (`List()`, `[1, 2]`), nor does
`append` state the value it adds.

## Expected results

| File                 | Must stay unproven | Should be proven |
|----------------------|--------------------|------------------|
| `straight_line.mojo` | all `bad_*`        | all `ok_*`       |
| `loops.mojo`         | all `bad_*`        | all `ok_*`       |
| `postconditions.mojo`| all `bad_*`        | all `ok_*`       |
| `comptime.mojo`      | all `bad_*`        | all `ok_*`       |
