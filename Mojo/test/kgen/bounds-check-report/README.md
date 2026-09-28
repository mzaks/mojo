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

### Building on Linux or a fresh checkout

Two things are not part of the branch and have to be set up locally.

**`DEFINE_OR_RETURN_ERROR`.** The open-source export uses this macro in
`Mojo/lib/Compiler/KGENCompiler.cpp` (three uses, around lines 524, 546 and
812) without shipping the header that defines it, so the build stops in that
file with an undeclared `DEFINE_OR_RETURN_ERROR`. Add this definition right
after `using namespace KGEN;` near the top of the file, and keep it out of
commits (it is only a stand-in for the missing header):

```cpp
// LOCAL WORKAROUND (do not commit): the open-source export references this
// macro without shipping its definition.
#ifndef DEFINE_OR_RETURN_ERROR
#define DEFINE_OR_RETURN_ERROR(TYPE, NAME, EXPR)                               \
  auto NAME##OrErr = (EXPR);                                                   \
  if (NAME##OrErr.isError())                                                   \
    return NAME##OrErr.takeError();                                            \
  TYPE NAME = NAME##OrErr.takeValue();
#endif
```

The `#ifndef` makes it harmless where the header does exist.

**z3.** The pass runs `z3` as a subprocess and finds it on `PATH`; pass
`z3-path=/path/to/z3` in the report options otherwise. On Debian or Ubuntu,
`sudo apt-get install z3`; on Fedora, `sudo dnf install z3`; or take a release
binary from <https://github.com/Z3Prover/z3/releases>. The results here were
produced with z3 4.16 on macOS.

Then build as above, always with `--config=build-mojo`: without it Bazel
compiles the stdlib with the prebuilt toolchain, which does not know
`kgen.obligation`, `kgen.assume` or `kgen.copy_marker`. A cold build takes
about 15 minutes, a rebuild of the pass alone under a minute. If
`./bazelw run //:format` was run in between, rebuild `kgen` and `std` again
before running the report: formatting can leave `bazel-bin` without the stdlib
(the report then fails with "unable to locate module 'std'").

To check the setup, run the report on `cases.mojo` (as below): every `bad_*`
function should stay unproven and every `ok_*` function be proven.
`Mojo/stdlib/test/collections/test_list.mojo` (add
`-I Mojo/stdlib/test`) should report 963/963 obligations proven. The runtime
tests should still pass:

```bash
./bazelw test --config=build-mojo //Mojo/stdlib/test/collections/... //Mojo/stdlib/test/memory/...
```

On macOS this build registers no GPU target (`--target-triple` rejects both
NVPTX and the Metal `air64` triples), which is why `tensor.mojo` uses a
stand-in for the GPU index functions. A Linux build ships the NVPTX backend
separately, so elaborating a real kernel with
`--target-triple=nvptx64-nvidia-cuda --target-cpu=sm_80` may work there;
this has not been tried.

Run the pass in-process right after elaboration:

```bash
bazel-bin/Mojo/tools/kgen/kgen -I bazel-bin/Mojo/stdlib/std \
    Mojo/test/kgen/bounds-check-report/cases.mojo -elaborate \
    --bounds-check-report="verbose=true" -o /dev/null
```

Prefer this over dumping the IR and running `kgen-opt --bounds-check-report`
on it: the textual form of some ops (e.g. `pop.call_llvm_intrinsic
side_effecting`) does not parse back.

Options: `verbose`, `include-stdlib`, `z3-path`, `timeout-ms`, `dump-dir`
(keeps the generated SMT-LIB scripts, annotated with the IR value each term
stands for), and `explain`, which lists under each unproven obligation the
unknown values its condition depends on, for example:

```text
  UNPROVEN  bounds  test_list.mojo:180:33
      depends on unknown argument #1 of hlcf.loop at test_list.mojo:179:5
```

Each unknown shows the value it takes in a counterexample. An unproven
obligation without unknowns has a condition that is false in some execution
the analysis considers, rather than one it cannot see into. Uninitialized
reads, typically from arms that cannot be taken, are listed last.

Two options switch off speedups, for comparison: `lazy-heap=false` encodes
every heap load eagerly instead of on demand for obligations not proven
without it, and `all-loop-invariants=true` infers invariants for every loop,
not only for loops some obligation's condition or reach depends on.

`timeout-ms` (default 2000) is wall-clock time per query, so results near the
limit depend on the machine and its load: on a busy machine an obligation can
come out unproven that is proven on an idle one. Raise it before concluding
that an obligation cannot be proven.

## Expected results

Each file groups its functions into cases that must stay unproven and cases
that should be proven. Every `List` access produces two obligations, one from
`__getitem__` and one from `unsafe_get`. The second is reported as "implied" by
the first, so each access counts once.

| File                 | Must stay unproven        | Should be proven                               |
|----------------------|---------------------------|------------------------------------------------|
| `basic.mojo`         | `get`                     | `sum_all`, `get_or_zero`                       |
| `cases.mojo`         | all `bad_*`               | all `ok_*`, `maybe_reversed`, `maybe_mutating` |
| `adversarial.mojo`   | all `bad_*`               | all `ok_*`                                     |
| `contracts.mojo`     | all `bad_*` (see below)   | all `ok_*`                                     |
| `slicing.mojo`       | all `bad_*`               | all `ok_*`                                     |
| `loops.mojo`         | all `bad_*`               | all `ok_*`                                     |
| `memory.mojo`        | all `bad_*` and `limit_*` | all `ok_*`, `Bag.ok_get`                       |
| `unrolling.mojo`     | all `bad_*` and `limit_*` | all `ok_*`                                     |
| `tensor.mojo`        | all `bad_*` and `limit_*` | all `ok_*`, `Tensor2D.load`/`store`            |
| `where_clauses.mojo` | all `bad_*`               | all `ok_*`                                     |

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

`where_clauses.mojo` states preconditions as `where` clauses on arguments
(see `Mojo/proposals/argument-contracts.md`), as in
`def ok_get(xs: List[Int], i: Int where 0 <= i and i < len(xs))`. The parser
lowers each clause to a `kgen.requires` op whose region computes the condition
from block arguments that stand for the function's arguments, so optimizing the
body cannot fold it away. The analysis
assumes it in the function (`ok_get` and `ok_window` prove their indices from
it), checks it at every call to a function that is not inlined (reported as a
`requires` at the call: `bad_get_unchecked`, `bad_get_past_end`,
`bad_window_swapped`), and checks an inlined function's clause where it was
inlined (`ok_at`, `bad_at_unchecked`). Clauses are only supported on `imm`,
`var` and owned arguments so far.

`slicing.mojo` covers the length contracts of `List.extend(Span)` (used by
`List.copy()`), of the `List` constructor from an iterable (for iterators with
exact bounds, such as a `Span`'s), and of `List.__getitem__(StridedSlice)`, as
well as the slice bounds that `check_slice_bounds` now records as obligations.
Strided slices (`vs[::-1]`, `vs[::2]`, `vs[1:0:-1]`) are proven.

`loops.mojo` needs loop invariants that relate two loop-carried values: the
index and the length of a list that shrinks by one per iteration keep their
sum (`i + len == i0 + len0`), and one that grows keeps their difference. The
analysis tries both templates for every pair of values of the same type.

`memory.mojo` covers lists in memory that calls may or may not write. A call
that takes a list `read` receives its address as an `imm_mem` argument, which
the callee may read but not mutate, so it does not stop the analysis from
tracking the list's fields past the call (`ok_strided_twice`,
`ok_read_calls_keep_length`). For the same reason a `read self` stays unchanged
throughout a method, even across calls in between (`Bag.ok_get`). This is a
trusted language guarantee: unsafe code that casts the origin away breaks it.
`mut` calls (`bad_after_mut_call`, `bad_read_then_mut_call`,
`Bag.bad_get_after_clear`) and a mutable `Pointer` taken to the list before a
read-only call (`bad_escaped_pointer`) still count as writes.

Once a list's address has escaped, a call may change it, so its fields are no
longer known after one. Two loads of the same field with nothing that may write
it in between still read the same value, even if that value is unknown
(`ok_eq_after_closure`, after a closure appended to the list): `List.__eq__`
checks the lengths and then indexes the other list while iterating, which needs
its two loads of the length to agree. A write in between breaks the link
(`bad_len_then_write_through_pointer`, `bad_len_twice_write_between`).

The `*_nested_*` examples in `memory.mojo` index lists of lists, where the inner
list's length is a load from the outer list's heap buffer. The analysis finds
the value by searching back from the load for the stores that may have written
it, including the ones inside the non-inlined `List.append` that put the inner
list there. Each store contributes "if it wrote this address, the stored value",
and memory from different allocations never overlaps
(`ok_nested_other_list_written`). Clearing the inner list
(`bad_nested_cleared`) or overwriting it through an unsafe pointer
(`bad_nested_overwritten_through_pointer`) still counts as a write.
When `append` reallocates, it moves the elements with a byte-wise `memcpy`
(`ok_nested_after_realloc`): the stdlib brackets it with `kgen.copy_marker
"begin"` / `"end"` (typed pointers and element count, erased when lowering to
LLVM), and the analysis reads a copied element from the source as it was
before the copy instead of following the bytes. The markers are trusted, like
`_assume`.

Heap values also have to get past the code between the write and the read.
`ok_nested_int_after_asserts` and `ok_nested_string_field_after_asserts` read
after `assert_equal` calls: a failed assert raises through an arm that formats
the message, and a read after the assert is only reached when it passed, so
that arm is not searched. Failed `debug_assert`s in the same position abort and
are skipped the same way. String literals and other compile-time constants
live outside every allocation, so writes through them never touch a list's
buffer. `limit_nested_after_string_dropped` stays unproven: `s += "def"` takes
the String `mut`, which hides its buffer pointer from the analysis, so the
reference count update when the String is dropped might hit the list's buffer.

`unrolling.mojo` fills lists of lists in loops, like `test_2d_dynamic_list` in
`test_list.mojo`. When the heap search meets a loop that writes memory, it
unrolls it: iteration `k` is evaluated like a callee whose arguments are the
previous iteration's loop-carried values, and the value after the loop is the
one the exiting iteration leaves. It unrolls at most 8 iterations and skips
iterations whose exit condition is a known constant, so `range(2)` costs two
iterations; a loop that may run longer (`limit_loop_many_iterations`) or an
unknown number of times (`bad_loop_unknown_count`) leaves the value unknown.
Lists that start empty reallocate on the way (`ok_2d_realloc`), which the copy
markers described above cover.

`tensor.mojo` verifies indexing into a small tensor from GPU-style kernels.
`Tensor2D[rows, cols]` has static dimensions, like a `LayoutTensor` with a
static layout, and its `load`/`store` state their bounds as `_requires`: each
call site must establish them, and the flat index inside is proven from them.
`ThreadCtx` stands in for `thread_idx`, `block_idx`, `block_dim` and
`grid_dim`, which only compile for a GPU target (this build has none): each
accessor `_assume`s what the hardware guarantees, including the bounds of the
dimensions it relies on (`1 <= block_dim <= 1024`, `grid_dim <= 2^31 - 1`), so
products of indices cannot wrap. With that, a bounds guard (`ok_kernel_guarded`,
`ok_kernel_2d`) or a launch precondition `grid_dim * block_dim == N`
(`ok_kernel_exact_launch`) proves every access, and a host loop that launches
the kernel for each block and thread is checked against that precondition
(`ok_launch_exact`, `bad_launch_too_many_blocks`). Unguarded kernels, a launch
that only covers "at least N" threads, and swapped coordinates stay unproven.
`limit_global_index` is a known limit: its postcondition multiplies two unknown
dimensions, which the solver does not bound in time.
