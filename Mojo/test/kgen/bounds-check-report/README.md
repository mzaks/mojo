# bounds-check-report examples

Example programs for the `bounds-check-report` pass, which tries to prove the
`kgen.obligation`s that `check_bounds` emits for every `List` index, and the
`_ensures` contracts of `List` methods.

These are not lit tests: the pass needs a `z3` executable, which CI does not
provide. The lit rule in this directory only collects `.mlir` and `.ll` files,
so these `.mojo` files are not run automatically.

## Running

Build the tools and the stdlib with the in-tree compiler (the prebuilt
toolchain does not know `kgen.obligation`):

```bash
./bazelw build --config=build-mojo //Mojo/tools/kgen //Mojo/tools/kgen-opt //Mojo/stdlib/std
```

Run the pass in-process right after elaboration:

```bash
bazel-bin/Mojo/tools/kgen/kgen -I bazel-bin/Mojo/stdlib/std \
    Mojo/test/kgen/bounds-check-report/cases.mojo -elaborate \
    --bounds-check-report="verbose=true" -o /dev/null
```

Prefer this over dumping the IR and running `kgen-opt --bounds-check-report`
on it: the textual form of some ops (e.g. `pop.call_llvm_intrinsic
side_effecting`) does not parse back.

Options: `verbose`, `include-stdlib`, `z3-path`, `timeout-ms`, and `dump-dir`
(keeps the generated SMT-LIB scripts).

## Expected results

Each file groups its functions into cases that must stay unproven and cases
that should be proven. Every `List` access produces two obligations, one from
`__getitem__` and one from `unsafe_get`. The second is reported as "implied" by
the first, so each access counts once.

| File               | Must stay unproven                          | Should be proven             |
|--------------------|---------------------------------------------|------------------------------|
| `basic.mojo`       | `get`                                       | `sum_all`, `get_or_zero`     |
| `cases.mojo`       | all `bad_*`, `maybe_mutating` (known limit) | all `ok_*`, `maybe_reversed` |
| `adversarial.mojo` | all `bad_*`                                 | all `ok_*`                   |
| `contracts.mojo`   | all `bad_*` (see below)                     | all `ok_*`                   |
| `slicing.mojo`     | all `bad_*`, `limit_strided` (known limit)  | all `ok_*`                   |

`adversarial.mojo` targets the SMT encoding itself. For example,
`bad_overflow` must stay unproven because `i + 1` wraps for `Int.MAX`, which a
model with unbounded integers would miss.

`contracts.mojo` exercises the `_ensures` contracts of `List.append`,
`List.pop` and `List._realloc`, the `_requires(len(self) > 0)` precondition of
`List.pop()`, and the `_assume(0 <= len)` type invariant.
`bad_pop_first_unguarded` guards against using an assumption made after a check
to justify that check. `bad_pop_unguarded` is not inlined, so the precondition
of `pop()` is checked at its call and reported as an unproven `requires` there;
`ok_pop_guarded` establishes it. `ok_first_requires` has its own `_requires`
and proves its body from it; `bad_call_without_precondition` and
`ok_call_with_precondition` call it without and with the guard, and
`bad_check_before_pop` checks that a later precondition does not justify an
earlier access. `main` calls the examples on one shared list.

`slicing.mojo` covers the length contracts of `List.extend(Span)` (used by
`List.copy()`), of the `List` constructor from an iterable (for iterators with
exact bounds, such as a `Span`'s), and of `List.__getitem__(StridedSlice)`, as
well as the slice bounds that `check_slice_bounds` now records as obligations.
For strided slices the contract is proven inside `__getitem__`, but call sites
cannot yet evaluate `len(range(slice.indices(len)))`, so `limit_strided` stays
unproven.
