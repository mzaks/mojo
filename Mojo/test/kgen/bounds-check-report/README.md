# bounds-check-report examples

Example programs for the `bounds-check-report` pass, which tries to prove the
`kgen.obligation`s that `check_bounds` emits for every `List` index.

These are not lit tests: the pass needs a `z3` executable, which CI does not
provide. The lit rule in this directory only collects `.mlir` and `.ll` files,
so these `.mojo` files are not run automatically.

## Running

Build the tools and the stdlib with the in-tree compiler (the prebuilt
toolchain does not know `kgen.obligation`):

```bash
./bazelw build --config=build-mojo //Mojo/tools/kgen //Mojo/tools/kgen-opt //Mojo/stdlib/std
```

Elaborate an example, then run the pass on the result:

```bash
bazel-bin/Mojo/tools/kgen/kgen -I bazel-bin/Mojo/stdlib/std \
    Mojo/test/kgen/bounds-check-report/cases.mojo -elaborate -S -o /tmp/cases.mlir
bazel-bin/Mojo/tools/kgen-opt/kgen-opt --bounds-check-report="verbose=true" \
    /tmp/cases.mlir -o /dev/null
```

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

`adversarial.mojo` targets the SMT encoding itself. For example,
`bad_overflow` must stay unproven because `i + 1` wraps for `Int.MAX`, which a
model with unbounded integers would miss.
