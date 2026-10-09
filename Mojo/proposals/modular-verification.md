# Modular verification before elaboration

**September 29, 2026**
Status: Draft. Stages 1 to 5 and the `List`, `Span` and slicing parts of
stage 6 are implemented; see
[Implementation status](#implementation-status).

This document proposes moving static verification of contracts and bounds
from after elaboration to right after lifetime checking, where functions are
still generic and nothing is inlined. Each function is verified once, against
its own contracts, and every call is checked against the callee's
preconditions and uses only its postconditions. Most of the resulting
obligations are discharged by a cheap dataflow analysis; only the rest reach
the SMT solver. Diagnostics are reported on source locations, like the other
checks at that stage.

It builds on [function contracts](function-contracts.md), which already
states preconditions and postconditions as `requires` and `ensures` clauses
and keeps them as `kgen.requires` and `kgen.ensures` ops.

Contracts were first written as `where` clauses on arguments, and moved to
`requires` and `ensures` clauses on October 9, 2026. The design below uses
the new spelling. The entries of
[Implementation status](#implementation-status) are a log: the older ones
describe clauses as they were written at the time (`i: Int where 0 <= i` for
`requires 0 <= i`, `out result: Int where result >= 0` for
`ensures result >= 0`).

## Background

The `bounds-check-report` pass (on the `mojo-bounds-verifier` branch) runs
after elaboration and after the first optimization pipelines. It proves all
977 obligations of `Mojo/stdlib/test/collections/test_list.mojo` in about 41 s
(278 s of z3 CPU time spread over workers). A measurement of where that time
goes:

| Where the solver time goes (default mode) | z3 CPU | Share |
|-------------------------------------------|--------|-------|
| `_test_copyinit_trivial_types`, 7 dtypes  | 249 s  | 90 %  |
| `test_list_insert`                        | 16 s   | 6 %   |
| everything else (47 functions)            | 13 s   | 4 %   |

| Kind of query                      | z3 CPU | Share | Queries |
|------------------------------------|--------|-------|---------|
| Loop invariant inference (Houdini) | 206 s  | 74 %  | 12,702  |
| Obligations                        | 72 s   | 26 %  | 2,997   |

`_test_copyinit_trivial_types[dt]` is one generic function. Elaboration
makes seven copies of it, one per dtype, and in each copy a `comptime for`
over ten sizes stamps out the same two loops ten times. The verifier then
infers invariants for about 140 copies of two loops, with up to 40 Houdini
rounds per function. The source contains two loops.

Post-elaboration checking has other costs too:

- **Instantiation.** Generic code is proven once per instantiation, and
  code that is never instantiated is never checked.
- **Inlining.** Callee bodies are inlined into every caller, so a callee's
  postcondition is re-proven at every call site. Adding element facts to
  `List.append` (`_same_elements`) made modular runs 30 % slower for this
  reason alone. A `kgen.obligation` inside `check_bounds` is only reported at
  a user's `xs[i]` because inlining resolves its `kgen.source_loc`.
- **Optimizations.** What can be proven depends on what SROA, mem2reg,
  inlining and SCCP did first. Contract ops needed snapshots in SROA and
  mem2reg and an exception in the inliner's cost model to survive them.
- **Diagnostics.** Messages refer to low-level IR, and to the location a
  `source_loc` resolves to, not to the call and the clause involved.
- **Solver behaviour.** Timeouts are wall-clock (2 s), so results change with
  machine load. About 4 in 30 runs of `test_list.mojo` took 150 s to over an
  hour, in the old build too.

## Goals and non-goals

Goals:

- Verify each function once, generically, before elaboration and inlining.
- State bounds as preconditions (`requires` clauses) on the declarations
  (`List.__getitem__`, `Span.__getitem__`, ...). A call to such a function
  creates an assertion at the call, with the call's source location.
- Prove most assertions with dataflow analysis, without the solver.
- Make the solver part deterministic, incremental and cached.
- Report results as compiler diagnostics that point at source.
- On `test_list.mojo`: prove at least what the current pass proves, at least
  ten times faster.

Non-goals:

- Replacing the post-elaboration pass right away. It stays, as a fallback for
  functions the generic checker cannot prove, until the new one covers them.
- Runtime checks. `check_bounds` keeps its `debug_assert`s.
- Full functional verification. The target is bounds and the contracts
  written for them.

## Where the checker runs

A new pass, `verify-contracts`, is added to `buildCheckLITPipeline`, after
`check-lifetimes`. It works on `lit.fn` ops. At that point:

- Generic functions are single `lit.fn`s with symbolic parameters
  (`<dt: !DType>`); `comptime for` is one `hlcf.comptime.for`, not ten copies.
- Nothing is inlined. Every call is a `lit.call` with its parameter bindings
  and origins; `xs[i]` is a call to `List.__getitem__`, and `i < n` a call to
  `SIMD.__lt__` with `dtype index`.
- Fields are visible (`lit.ref.struct.gep %self[_len]`), and references carry
  their origins and mutability (`!lit.ref<..., mut *"x">`).
- The contract ops (`kgen.requires`, `kgen.ensures`, `kgen.old`,
  `kgen.contract.entry`, `kgen.forall`, `kgen.contract.same_elements`) are
  present, with source locations from the parser.

Because the check-LIT pipeline also runs for the language server (`kgen
-lsp`), the same results can appear in the editor.

## Verification model

### Functions and calls

Each `lit.fn` with a body is verified on its own:

1. Its `kgen.requires` clauses are assumed at entry.
2. Every obligation in its body is proven:
   - at each call, the callee's `kgen.requires` clauses, instantiated with the
     call's arguments and parameters (the caller-side assertion);
   - at each return, the function's own `kgen.ensures` clauses;
   - explicit `_assert`-style obligations.
3. After a call, the callee's `kgen.ensures` clauses are assumed. Nothing else
   about the callee's body is used.

So a callee's postcondition is proven once, in the callee, and each call only
checks the callee's preconditions.

What a call may change is given by origins, which the IR already carries: a
`mut` reference argument may be written, an `imm` one may not, and values
passed by `var` are the callee's. This is the frame, and it replaces the heap
search the current pass does through callee bodies.

### Built-in semantics

Some calls are modelled directly, not through contracts, because they are the
arithmetic the contracts are written in:

- `SIMD` operators with `dtype index` or another concrete integer dtype and
  width 1 (`__add__`, `__sub__`, `__mul__`, `__lt__`, `__eq__`, `__and__`,
  ...), `Bool` operators, and `Int` conversions: bit-vector operations, with
  wrapping as today.
- `len(x)` for a type with a known `__len__`: the value of that call, one
  uninterpreted function per type, so `len(xs)` in a contract and in the body
  are the same term.
- Field reads and writes through `lit.ref.struct.gep`, `lit.ref.load` and
  `lit.ref.store` on local and argument references.

A call with neither contract nor built-in semantics returns an unknown value
and havocs its `mut` arguments.

### Stdlib contracts

Bounds move from `kgen.obligation`s inside `check_bounds` to preconditions on
the declarations:

```mojo
def __getitem__(ref self, idx: Int, /) -> ref[...] Self.T
    requires 0 <= idx and idx < len(self):
```

`check_bounds` stays for its runtime assertion. Since negative indexing has
been removed, the precondition is the whole requirement.

Loops over ranges need the iterators to say what they yield. `range(n)`'s
`__next__` gets a postcondition (`0 <= result and result < self.end`), and the
iterator's fields a type invariant (`curr <= end`), which the parser can
already express as clauses about `self`.

Collections that the post-elaboration pass handles through their bodies
(`List`, `Span`, `String`, `Dict`, `InlineArray`) need contracts on the
methods tests use. This is the bulk of the work, and it is staged below.

### Generic parameters

Parameters are symbolic. A `comptime if` on a parameter is verified along
both branches, with the condition as a path condition. A `comptime for` is
verified as a loop over a symbolic index. A dtype parameter makes values of
that dtype opaque, which is fine for bounds: an index is an `Int`.

Where a proof needs a parameter's value (for example `size_of[T]()`), the
function is reported as not verifiable generically, and the post-elaboration
pass checks its instances.

## Proving obligations

### Dataflow first

Before any solver query, an abstract interpreter runs over each function. It
tracks integer intervals and differences between pairs of variables (`i <
len(xs)`, `curr <= end`), which is what bounds need. Loops are handled with
widening. It gives:

- **Loop invariants** without Houdini. Houdini is 74 % of today's solver time.
- **Discharged obligations.** `xs[i]` inside `for i in range(len(xs))` is
  proven by `0 <= i` and `i - len(xs) < 0`, without a query.

What the domain proves is sound on its own. What it cannot prove goes to the
solver, with the facts it found as assumptions.

### The solver, Dafny-style

For what reaches the solver:

- **One solver per function, incremental.** One z3 process is kept per
  function; each obligation is one `push`/`check-sat`/`pop` over a shared
  encoding.
- **Small queries.** Each obligation is checked on its own and sees only what
  it depends on (its backward slice).
- **Deterministic limits.** `rlimit` instead of wall-clock timeouts, so a
  result does not depend on machine load. Each z3 process also gets a hard
  wall-clock cap, so a stuck process cannot stall a run.
- **Caching.** A result is cached under a hash of the function's encoding and
  of the contracts of the callees it uses, as Boogie's verification snapshots
  do. Changing a function body re-verifies that function only; changing a
  contract re-verifies its callers.
- **Parallelism** across functions, as today.

### Loops the domain cannot handle

Houdini stays for obligations that depend on a loop invariant the domain does
not find. It runs in the function's incremental solver, and only for the
loops in the obligation's slice (as `all-loop-invariants=false` does today).

## Diagnostics

Results are MLIR diagnostics:

- An unproven precondition is reported at the call, with a note at the clause
  in the callee's declaration:

  ```text
  test.mojo:12:14: warning: cannot prove this call's precondition
      return xs[i + 1]
               ^
  list.mojo:1623:18: note: precondition of 'List.__getitem__'
      requires 0 <= idx and idx < len(self):
               ^
  ```

- An unproven postcondition is reported at the `return`, with a note at the
  clause.
- A counterexample can be attached as a note (`i = 5, len(xs) = 5`), from the
  model the solver returns.

The default is warnings behind an option. Errors, for code that opts in, are
later work.

## Relationship to the current pass

The `bounds-check-report` pass stays as it is, as a fallback tier. For a
function the generic checker reports as not verifiable generically, the
post-elaboration pass checks its instances. Its results are then reported on
the instance, as today.

## Stages

Each stage is measured on the `bounds-check-report` examples and on
`test_list.mojo`, against the post-elaboration pass.

1. **Skeleton.** The pass on `lit.fn`, straight-line code, built-in integer
   semantics, calls through contracts and origin frames, `kgen.requires` as
   caller-side assertions, and diagnostics. `List.__getitem__` and `len` get
   contracts. Target: the straight-line examples in `where_clauses.mojo` and
   `basic.mojo`.
2. **Loops.** The interval and difference domain, `hlcf.loop`, `lit.try`
   (range iteration raises `StopIteration`), and `range` iterator contracts.
   Target: `sum_all`-style loops proven without the solver.
3. **Postconditions.** `kgen.ensures`, `kgen.old`, `kgen.forall` and
   `kgen.contract.same_elements` at the LIT level. Target:
   `where_clauses.mojo` and `modular.mojo`.
4. **Solver engineering.** Incremental solving, `rlimit`, the wall-clock
   cap, and the cache.
5. **Generic parameters.** `comptime if`, `comptime for`, and the fallback
   to the post-elaboration pass. Target: `_test_copyinit_trivial_types`
   verified once.
6. **Collections.** Contracts for `List` (then `Span`, `String`, `InlineArray`,
   `Dict`) until `test_list.mojo` proves what it proves today. Target: at
   least ten times faster.

## Implementation status

Stage 1, on the `mojo-bounds-verifier` branch:

- The `verify-contracts` pass (`Mojo/lib/Transforms/VerifyContracts.cpp`)
  runs through `kgen -lsp --verify-contracts[=options]`, on the module the
  check pipeline produced. It is not part of the compiler's pipelines yet.
- `List.__getitem__(idx: Int)` states its bound as a `where` precondition.
  The post-elaboration pass checks the inlined clause too, and still proves
  every obligation of `test_list.mojo` (1219 of 1219, 242 of them the new
  clause).
- `Mojo/test/kgen/verify-contracts/straight_line.mojo` proves all its `ok_*`
  functions and flags all its `bad_*` ones.

Differences from the design above, found while implementing it:

- Frames come from origins in two places: a call's mutable reference
  operands, and the implicit origins of the call (`lit.call @f[mut *"xs"]`),
  which also cover memory reached through structs passed by value. A store
  through a reference the pass cannot trace makes unknown the roots its
  type's origin names. When an origin names no root of the function, all of
  its memory is unknown after the call.
- Literal indices (`xs[0]`) call `__getitem__(idx: IntLiteral)`, which has
  no precondition yet; `test_list.mojo` indexes mostly that way, so only 24
  of its calls are checked so far.
- `len(x)` is assumed non-negative, for every `Sized` type.

Stage 2:

- Loops (`hlcf.loop` with `continue` and `break`) and `lit.try` are
  analyzed; `Mojo/test/kgen/verify-contracts/loops.mojo` proves all its
  `ok_*` functions (including nested loops) and flags all its `bad_*` ones.
- Loop invariants come from Houdini, not yet from an abstract domain: before
  inlining, a loop body is a handful of calls, so the candidates are few
  (only variables the loop's conditions depend on) and each round is one
  small z3 script. `test_list.mojo` walks its 40 loops in 2.6 s in all. The
  interval and difference domain stays the plan for when bodies or
  candidate sets grow.
- `range` iteration is built in (as the stdlib defines it) rather than
  stated as contracts on the iterators, until the pass uses postconditions
  (stage 3). `reversed(range(...))` is not modelled yet.
- z3's `rlimit` counts resources across a script rather than per query, so
  it cannot be a tight per-query budget. It stays as a deterministic cap on
  runaway queries (100M by default), next to the wall-clock cap.

Stage 3:

- Postconditions are proven at returns and assumed after calls, with
  `kgen.old`, quantifiers, register `out` results and raising callees;
  `Mojo/test/kgen/verify-contracts/postconditions.mojo` proves all its
  `ok_*` functions and flags all its `bad_*` ones.
- Elements need a model before inlining: `xs[i]` is `elem(xs, i)`, and a
  write to it is a new list value with the same length and the other
  elements unchanged. Without it, `xs[0] = 7` would make `xs`, and its
  length, unknown.
- `_same_elements(a._data, b._data, n)` works through the lists' `_data`
  fields, so `List`'s contracts need no change for it.
- Houdini also bounds the lengths of lists a loop appends to.
- `test_list.mojo` proves 14 of the 24 calls the pass checks, in 4.5 s. The
  rest need contracts the stdlib does not state yet: the lengths of `List()`
  and list literals, and the value `append` adds (stage 6).

Stage 4:

- Functions are verified on a thread pool and reported in order afterwards.
- Scripts are deterministic: places are ordered by their position in the
  function rather than by address. Until then, 99 of test_list.mojo's 123
  scripts changed from run to run.
- `cache-dir=` caches each script's answers under a hash of the script and
  of the z3 binary; runs stopped at the wall-clock cap are not cached.
- test_list.mojo (the check pipeline alone takes 1.1 s):

  | Stage end | Verification's share |
  |-----------|----------------------|
  | 3         | 3.4 s                |
  | 4, cold   | 0.7 s                |
  | 4, warm   | 0.3 s                |

- One incremental z3 process per function (instead of one per Houdini round)
  is not done: in parallel, the remaining solver time is about 0.4 s, and it
  needs a two-way pipe to z3 that LLVM's process API does not provide.
- Each query still sees the whole function's encoding; before inlining the
  encodings are small enough that slicing per query has not been needed.

Stage 5:

- `comptime for` is a loop over an arbitrary iteration and `comptime if` a
  join of its arms; `Mojo/test/kgen/verify-contracts/comptime.mojo` proves
  all its `ok_*` functions and flags all its `bad_*` ones.
- The parser folds comparisons on comptime values into parameter
  expressions, so the pass evaluates those: operators, literals, `apply` of
  `SIMD` integer operators, and `param.identical` of scalars. A parameter
  itself, or a comptime computation such as indexing a comptime tuple, is
  an unknown (one per expression).
- `_test_copyinit_trivial_types` in test_list.mojo, 90% of the
  post-elaboration pass's solver time over 7 instantiations and 10 unrolled
  sizes, is analyzed once. Proving its accesses needs `List()`'s length
  (stage 6).
- Reporting a function as not verifiable generically, for the
  post-elaboration pass to check its instances, is not done yet: nothing is
  unanalyzed in the examples and test_list.mojo.

Stage 6, `List`:

- `kgen --verify-contracts` parses the file as a compilation does, runs the
  check pipeline and the pass, and stops before elaboration. The `-lsp`
  parse is lazy, so stdlib functions whose bodies nothing needed there had
  no contracts for the pass.
- `List` states its lengths in its constructors (including literals, whose
  element count the pass reads from the variadic pack) and mutating
  methods, and its index bounds in `__getitem__` (`Int` and `IntLiteral`),
  `pop(i)` and `insert`; `Mojo/test/kgen/verify-contracts/lists.mojo`
  proves all its `ok_*` functions and flags all its `bad_*` ones.
- The post-elaboration pass proves the new clauses where it inlines them
  (1822 of test_list.mojo's 1833 obligations; the rest are on lists of
  lists, through heap memory), which checks the contracts against their
  bodies.
- test_list.mojo, one run each at load 9 to 14:

  | Pass                     | Time   | Proven                |
  |--------------------------|--------|-----------------------|
  | post-elaboration         | 47 s   | 1822 of 1833          |
  | `verify-contracts`       | 1.7 s  | 150 of 222            |

  The obligations differ: the new pass checks each call once, generically.
  Its 72 unproven calls need what `List`'s contracts cannot say yet
  (the value `append` adds, the length of a list that is an element of
  another), `Span` and slicing contracts (`test_list_span`, 23), and
  `reversed(range(...))`. Other collections come next.

Stage 6, `Span` and contiguous slices:

- `Span`'s index and slice access, and `List`'s slice access, state their
  bounds and result lengths; `Span(list=)` states its length and that the
  list keeps its own. `Mojo/test/kgen/verify-contracts/spans.mojo` proves
  all its `ok_*` functions and flags all its `bad_*` ones.
- The bounds are `Optional[Int]`s, so the pass models `Optional` and the
  slice constructor as records of their fields. Postconditions on results
  need named `out` results; callers do not change.
- A reference argument whose origin may be mutable is unknown after the
  call, even when the callee does not write it, unless its contract says
  what it keeps: `Span(list=)` states `len(list) == old(len(list))`.
- test_span.mojo: 57 of 101 calls proven (44 before), in 0.9 s. Not
  covered yet: spans over `Array` (`len(array)` is not allowed in the
  contract for arrays in other address spaces, and `array.length` is a
  parameter the pass cannot relate to the caller's), strided slices, and
  `List(span)` through the generic iterable constructor.
- Two bugs found on the way: literals wider than their dtype's signed range
  (`255` as `UInt8`) asserted, and places rooted at a callee's contract
  temporaries were still ordered by address, so scripts, and answers near
  the resource limit, could change between runs.

Stage 6, `Array` and `List(span)`:

- `Array.__getitem__(Int)` requires its index in range, and `Span(array=)`
  states that it is as long as the array. `len(array)` is the size
  parameter in the array's type, evaluated as a parameter expression, so in
  generic code it is the parameter (`i < n` proves `a[i]` for an
  `Array[Int, n]`). Literal indices on an array are checked at compile time
  (`__getitem_param__`) and are not obligations.
- A callee's contract can name the callee's parameters (`array.length`,
  inferred from `Array[Self.T, _]`). At a call they are the values the call
  binds: its struct's parameters, then its own, in order. Unknown parameter
  expressions of a callee are kept apart from the caller's, which could
  otherwise share a name.
- `List(span)` has an overload of its own that states its length; the
  generic iterable constructor cannot, before elaboration. Slicing a list
  states that the list keeps its length, like `Span(list=)`: the list is a
  reference that may be mutable, so without it `List(vs[1:])` of a local
  list lost `len(vs)`. `Mojo/test/kgen/verify-contracts/arrays.mojo` and
  `spans.mojo` prove all their `ok_*` functions and flag all their `bad_*`
  ones.
- One run each, at load about 4:

  | File            | Before       | Now          | Time  |
  |-----------------|--------------|--------------|-------|
  | test_list.mojo  | 153 of 225   | 160 of 225   | 1.6 s |
  | test_span.mojo  | 57 of 101    | 84 of 110    | 0.8 s |
  | test_array.mojo | 0 of 7       | 30 of 38     | 0.8 s |

  test_span's two `Array` tests are fully proven. What stays unproven there
  needs `enumerate` and `reversed` loops, element writes through a span
  (they make the list's length unknown, since the span's origin names the
  whole list), and span iterators. In test_list, `test_list_span`'s rest is
  strided slices.
- The post-elaboration pass proves the new clauses where it inlines them
  (test_span.mojo 864 of 866, test_list.mojo 3010 of 3024, both short only
  where they were before).

Stage 6, `BitSet`, `Deque` and `LinkedList`:

- `BitSet`'s bit operations require `0 <= idx < size` (a struct
  parameter). `Deque` and `LinkedList` state their index bounds and the
  lengths their constructors and mutators produce;
  `Mojo/test/kgen/verify-contracts/collections.mojo` proves all its `ok_*`
  functions and flags all its `bad_*` ones.
- A bounded deque evicts when full, so its length clauses name `_maxlen`,
  a private field, which its constructors state and its mutators keep. A
  public `maxlen` accessor would read better; the contracts can switch to
  it.
- Binding callee parameters missed callees with implicit origin
  parameters: those are listed with the function's parameters but bound by
  the call's origin list. They are skipped now.
- `LinkedList`'s index methods take `I: Indexer`; `index(i)` is modelled
  for `Int` and `IntLiteral`, looking through the `upcast` a trait-bounded
  parameter is passed as.
- Element reads through `ref self` (`Deque`, `Array`, `LinkedList`, like
  `List` before) no longer make the collection unknown. Without it, every
  `d[0]` lost `len(d)` for the next access.
- One run each:

  | File                  | Before     | Now          |
  |-----------------------|------------|--------------|
  | test_bitset.mojo      | 0 of 0     | 266 of 271   |
  | test_linked_list.mojo | 3 of 10    | 124 of 173   |
  | test_deque.mojo       | 3 of 10    | 82 of 105    |

  The rest need `with assert_raises()` blocks (not analyzed), `.copy()`
  through the `Copyable` trait, loops over literals, and iterators.
- The post-elaboration pass proves the new clauses where it inlines them,
  except two it cannot follow through loops over heap memory:
  `Deque.insert`'s length (proven in 7 of 12 places) and the length of a
  `LinkedList` literal, built through a closure (5 of 40).

Stage 6, strings:

- `String` has no `__len__`, so its contracts are stated in
  `byte_length()`, which the pass models as `len` of the string's value
  (for `String` and `StringSlice`) and as the literal's length for a
  `StringLiteral`, whose bytes are in its type. Byte indexing and
  slicing state their bounds and result lengths, `as_bytes()` its length,
  and `+=` the new length; `Mojo/test/kgen/verify-contracts/strings.mojo`
  proves all its `ok_*` functions and flags all its `bad_*` ones.
- `String(literal)` cannot state its length as a contract: naming
  `String.byte_length()` in its signature makes the declaration depend on
  itself (through `SIMD.cast`, which has a `String` default argument), and
  the language server's lazy parse fails. The pass models that constructor
  instead.
- A callee's postcondition is one `kgen.ensures` per clause. The pass
  assumed only the first, which dropped the rest of a contract with
  clauses on several arguments (`unsafe_as_bytes_mut`'s `self` and
  result); it now assumes all of them.
- One run each:

  | File                  | Before     | Now        |
  |-----------------------|------------|------------|
  | test_string.mojo      | 1 of 3     | 17 of 20   |
  | test_string_span.mojo | 12 of 28   | 50 of 57   |

  The rest are a comptime span (`comptime slc = "Hello".as_bytes()`, a
  parameter value), the results of `split`, and `String(i)` of an `Int`.
- The post-elaboration pass cannot check most of the string clauses
  against their bodies: `String`'s length is read from an inline or a heap
  representation, and it loses those fields across the pointer calls in
  `as_bytes()` (its length is proven in 5 of 461 places), `+=` and
  `StringSlice(String)`: test_string.mojo 1173 of 1661 obligations (580 of
  606 before), test_string_span.mojo 1299 of 1469 (603 of 629). The string
  contracts are stated from the bodies, not checked by a tool.

Stage 6, `Dict` and `Set`:

- Nothing in them is indexed, so their contracts are lengths, most of them
  bounds rather than values: an insertion may or may not add an entry.
  `collections.mojo` gains cases for both.
- test_dict.mojo: 1 of 32 calls proven (0 of 31 before). Its calls index
  lists built from a dictionary's contents (keys, values, items), which
  lengths of the dictionary do not reach. test_set.mojo checks no calls.
- The post-elaboration pass checks some of the clauses against the bodies: the
  empty constructors everywhere, `setdefault` and `Set.discard` in most places,
  `clear` in half. It cannot follow the swiss table through `d[k] = v` (unproven
  in 85 of 144 places) or the literal constructor (62 of 62), and reports
  nothing for `Set.add`. test_dict.mojo: 1519 of 1880 obligations (1394 of 1611
  before); test_set.mojo: 513 of 557 (567 of 607), both short of before only on
  the new clauses.

Stage 6, strided slices:

- `List.__getitem__(StridedSlice)` states its result's length as
  `slice._length(len(self))`, whose contract composes two helpers of
  `Slice.indices`: `_normalize_bound` (a bound as `indices` normalizes it)
  and `_strided_count` (the length of `range(start, end, step)`: 0 for a
  zero step or one pointing away from `end`, else a ceiling division).
  `indices` itself now calls the helpers, so there is one definition. The
  pass verifies all three helpers against their bodies
  (`include-stdlib=true`); the post-elaboration pass creates no obligation
  for the new clause.
- The pass records `StridedSlice(start, end, stride)` like
  `ContiguousSlice`, and models integer `//` and `%` with Mojo's rounding
  towards negative infinity (SMT-LIB's signed division truncates).
- Four pass fixes the helpers needed, all general:
  - A call inside a contract assumes its callee's postcondition, but did so
    in a copy of the state that was then dropped, so nested helper results
    were unknown. The facts now belong to the clause (conjoined where it
    is assumed, premises where it is proven).
  - `if` with `elif` arms was not analyzed at all.
  - `Bool`'s operators (`not` is `__invert__`) were unknown.
  - Encoding: equal expressions are one term (hash-consed definitions);
    values after an `if` are chosen by the arms' own conditions rather
    than their whole paths; and an `if` whose arms add no facts keeps the
    path condition it started with. Without these, `_length`'s body and
    contract evaluated the same nested helper calls into different terms
    and the proof did not finish in 120 s; with them it is proven within
    the default limits. test_list.mojo still takes 1.7 s.
- test_list.mojo: 176 of 225 calls proven (160 before); `test_list_span`
  is fully proven. The post-elaboration pass proves 3021 of 3036
  obligations, short only where it was before and on `String.as_bytes`.

Stage 6, `reversed(range(...))`:

- `reversed(range(n))` and `reversed(range(start, end))` over `Int` are
  built in, like forward ranges. The stdlib walks from `end - 1` down to
  `start` inclusive and flags exhaustion; the pass models the same
  sequence as the forward range mirrored: a cursor starting at `end`,
  exclusive, and the lower bound, so `__next__` raises when they meet and
  otherwise steps down and yields. That keeps the loop invariants the
  shape Houdini already finds for forward loops (the cursor between its
  bounds). `loops.mojo` proves `ok_reversed*` and flags indexing one past
  either end.
- `reversed(...)` returns through an `out` slot, and calls with no single
  result bypassed the built-in table; it is dispatched before it.
- The test files do not index inside reversed loops, so their numbers do
  not change. What test_range.mojo still misses needs facts from test
  assertions (`assert_equal(len(a), len(b))`), which generic `assert_*`
  functions cannot state before elaboration.

Test assertions and integer conversions:

- Much of what the tests index is set up by assertions:
  `assert_equal(len(backward), len(forward))` before a loop over both.
  The generic `assert_*` functions cannot state before elaboration what
  they check (`T.__eq__` is a trait method), so the pass builds in their
  definitions: after `assert_true(c)`, `assert_false(c)`,
  `assert_equal(a, b)` or `assert_not_equal(a, b)`, the check holds where
  the call did not raise. Only for `Bool` conditions and for `Int`, `Bool`
  and integer scalar operands, whose `==` is equality of the terms; for
  other types a user-defined `__eq__` need not be.
- Integer conversions between `SIMD` integer types (`Int(n)` of a
  `UInt8`) extend by the source's signedness or truncate, as integer
  `cast` does. Terms of different widths meeting in one fact make the
  script ill-sorted, which z3 rejects and the pass reports as unproven;
  both additions check widths, and no script from the examples and test
  files has a sort error.
- One run each:

  | File                  | Before     | Now        |
  |-----------------------|------------|------------|
  | test_range.mojo       | 4 of 15    | 15 of 15   |
  | test_dict.mojo        | 1 of 32    | 15 of 32   |
  | test_linked_list.mojo | 124 of 173 | 139 of 173 |
  | test_list.mojo        | 176 of 225 | 185 of 225 |
  | test_span.mojo        | 84 of 110  | 93 of 110  |
  | test_array.mojo       | 30 of 38   | 37 of 38   |
  | test_deque.mojo       | 82 of 105  | 89 of 105  |
  | test_string_span.mojo | 50 of 57   | 55 of 57   |

Nested collections:

- Almost every call the pass reported as not analyzed was an access to a
  nested collection (`list[0][1]`, `list.get_nth(0).get_nth(1)`): the
  inner access's `self` is the reference the outer one returned, which was
  not a place. A reference to an element is now a place of its own whose
  value is `elem(collection, index)`; any write to it (a store, a field
  store, the havoc of a call that takes it `mut`) writes the element's new
  value back into the collection, which keeps its length, and writing a
  collection makes its element places read from it again. So `xs[0].pop()`
  shortens `xs[0]` as seen through `xs`, and the pass does not keep the
  old, longer length.
- A call on an element carries the collection's interior origin
  (`#lit.interior.origin<xs, "element">`). That origin reaches the
  elements, not the collection's own fields, so it keeps the collection's
  length instead of making the whole collection unknown. When it is the
  origin of a mutable reference argument whose place is known, the
  argument's own havoc covers it: a reference to one element does not
  reach its neighbours (without unsafe code). Origins are applied before
  arguments, so the element's write-back lands on the collection the
  origin left.
- A list literal's elements are the values it is given (read before the
  call, which moves them in); its contract can only state its length. So
  `[[1, 2, 3], [4, 5]]` has rows of known lengths, and `[1, 2, 3][0]` is 1.
  References passed through `lit.ref.upcast` (as the variadic pack's are)
  are the same places.
- test_list.mojo: 198 of 225 calls proven (185); test_linked_list.mojo:
  152 of 173 (139); test_array.mojo: 38 of 38 (37). A property of every
  row (each row grew in a loop over them) needs a quantified invariant,
  which Houdini's templates do not include.

`copy()`:

- `x.copy()` goes through `Copyable`'s default, `Self(copy=self)`, a call
  through the trait that the pass cannot follow. After a call to such a
  default (a `copy` whose `defaultFnRef` is `Copyable.copy`), the pass
  assumes the postcondition of the struct's own `__init__(copy:)`, whose
  arguments are laid out as the wrapper's. test_deque.mojo: 95 of 105
  calls proven (89), test_linked_list.mojo 153 of 173 (152),
  test_list.mojo 199 of 225 (198).

`unsafe_get` and `unsafe_set`:

- `List.unsafe_get(i)` and `unsafe_set(i, v)` skip the run-time bound
  check (out of range is undefined behaviour). They now state it as a
  precondition, so the unchecked accesses are the ones checked
  statically, at no run-time cost; `unsafe_set` keeps the length. The pass
  models them as an element read and an element write. test_list.mojo:
  217 of 237 calls proven (199 of 225). `List.__getitem__(IntLiteral)`
  calls `unsafe_get`, so the post-elaboration pass sees the new clause in
  336 places; it cannot prove it in the same 10 list-of-lists places where
  it cannot prove the outer bound (3438 of 3463 obligations).

Iterating collections and `enumerate`:

- `for x in xs` and `for i, x in enumerate(xs)` over a `List`, `Span`,
  `Array` or `Deque` are built in, as their iterators define them: an
  iterator holds a cursor from 0 and the collection's length when it was
  made; `__next__` raises at the length and otherwise advances.
  `enumerate` wraps one with a count from `start` and yields a tuple
  whose first field is the count; a tuple's `__getitem_param__[k]` is its
  field `k`. The yielded elements are unknown, and reversed iteration over
  collections is not modelled.
- A counter the loop increments next to the iterator is related to the
  cursor by the invariants Houdini already tries (`i` between the
  cursor's bounds), so `for x in xs: ... xs[i]; i += 1` proves too.
- test_span.mojo: 101 of 110 calls proven (93), test_deque.mojo 97 of 105
  (95). Iterator loops now have invariants to find: test_span.mojo takes
  1.4 s instead of 0.9 s (254 invariants instead of 143; both measured
  back to back at load 14), test_list.mojo is unchanged at 1.8 s.

Trait contracts:

- The parser accepted `where` clauses on a required trait method (body
  `...`) and dropped them: only a default kept its contract ops. A
  required method with clauses now has them as its body, followed by
  `hlcf.unreachable`, so the trait declaration carries the contract.
  Required methods are declarations; the pass does not verify them.
- A generic call (`#kgen.get_witness<T, @Counter, "get($0,...)">`)
  resolves to the method's declaration in the trait, or in a trait it
  refines. Its preconditions are obligations at the call and its
  postconditions are assumed after it, with the trait's `Self` bound to
  the call's type.
- A struct gets a wrapper for each default it inherits, which only
  forwards to the default. A call to the wrapper has the default's
  contract, its parameters bound by the forwarding call. The wrapper
  itself is not verified: its only obligation is its own precondition.
  That removes a false warning at `struct X(Trait):`, which the
  default's precondition caused at the wrapper's forwarding call.
- A trait method that only reads its arguments and returns an integer
  or a Boolean (`t.count()`) is an uninterpreted function of their
  values, one per trait, method and type. So calls on unchanged
  arguments agree, and `count() == old(count()) + 1` after `bump()`
  relates the two. This assumes an implementation's result depends only
  on the values it is given, as `len(x)` does. `old(e)` now keeps the
  facts of the calls in `e` (`count() >= 0` from `count`'s
  postcondition); it dropped them before.
- Assuming a trait's contract is only sound if every implementation meets
  it. Each implementation of a trait method with clauses is verified a
  second time against the trait: the trait's precondition is assumed at
  entry and must imply the implementation's own, and the trait's
  postcondition must hold at every return. Inside the trait's clauses,
  its methods on `Self` are the struct's implementations with their own
  clauses, and a direct call to an implementation is the same function
  as a generic call on that type. A failure is a warning at the
  implementation ("cannot prove that 'BadCount.count' establishes the
  postcondition of 'Counter.count'"). It found a real bug in the first
  version of the example: `bump` as `n += 1` breaks `count() ==
  old(count()) + 1` at `Int.MAX`, where `n` wraps. The trait now also
  requires `old(self.count()) < Int.MAX`, a second clause on `self`.
  `Int.MAX` and the other integer bounds (`max_or_inf`, `min_or_neg_inf`)
  are now evaluated.
- Limits: only structs without parameters link their implementations
  (instantiations would otherwise share one function); conditional conformances
  are checked as if unconditional; conformances in the stdlib are only checked
  with `include-stdlib=true`. No stdlib trait had clauses at this step, so the
  stdlib numbers are unchanged (test_list.mojo 217 of 237, test_span.mojo 101 of
  110), and the built-in models of `index(x)`, `copy()` and the assertions stay
  until their traits state them. traits.mojo: 12 of 20 obligations proven, the
  other 8 in `bad_*` functions and in `BadCount`.

`Sized`:

- The first stdlib trait with a contract: `__len__(self, out result: Int
  where result >= 0)`. Implementations written `-> Int` still conform, and
  the refinement check reads their returned value.
- The trait's clauses never reached the pass for a stdlib trait: the
  parser's symbol DCE after importing drops every trait method that no
  symbol refers to, and generic calls name methods only through
  witnesses. A live trait's methods that carry contract ops are now kept.
- A generic `x.__len__()` is the same uninterpreted function as `len(x)`,
  so the two agree, and `len`'s non-negativity, until now an assumption
  of the pass, is the trait's contract: checked for every `Sized` struct
  outside the stdlib (a `__len__` that returns an `Int` field as it is
  does not prove), for the stdlib's own with `include-stdlib=true`.
- Numbers unchanged: test_list.mojo 217 of 237, test_span.mojo 101 of 110
  (their `len` was already non-negative by assumption); the
  post-elaboration pass is unaffected (test_list.mojo 3438 of 3463). No
  stdlib test defines a `Sized` struct. traits.mojo: 16 of 26, the other
  10 in `bad_*` functions and in `BadCount` and `BadLen`.

Performance after `Sized` (2026-09-30):

- Measured with `--mlir-timing`, where verify-contracts appears as "Rest" (it
  runs after the pass manager). Median of 5 runs, one run per process, 16 cores
  at load average about 4 (an unrelated compiler at full load on one core).
  "Cold" solves every query; "warm" answers them from a primed `cache-dir=`, so
  it is the encoding alone and the difference is solver time. CPU counts kgen
  and its z3 processes.
- "Verify" is the pass's own wall-clock time, and nothing else: timing the pass
  directly (a temporary build) gave the same values as "Rest" to the hundredth
  of a second (test_list.mojo 0.88 s cold and 0.13 s warm, test_deque.mojo 0.91
  s, test_dict.mojo 0.99 s, test_bitset.mojo 0.18 s, at load about 3). It covers
  encoding every function, the loop-invariant search and waiting for z3, with
  functions verified in parallel. It does not cover importing the file with the
  stdlib ("Import", 0.4 to 1.1 s) or the check pipeline before the pass (0.03
  s); printing the warnings and exiting add 0.02 to 0.03 s to the process. The
  same holds for the later tables, which were measured the same way.

  | File                  | Proven  | Import | Verify, cold | Verify, warm | CPU, cold | Total, cold |
  |-----------------------|---------|--------|--------------|--------------|-----------|-------------|
  | test_list.mojo        | 217/237 | 0.93 s | 0.83 s       | 0.13 s       | 7.4 s     | 1.82 s      |
  | test_span.mojo        | 101/110 | 0.63 s | 0.61 s       | 0.03 s       | 3.0 s     | 1.29 s      |
  | test_deque.mojo       | 97/105  | 0.80 s | 0.51 s       | 0.04 s       | 3.5 s     | 1.37 s      |
  | test_linked_list.mojo | 153/173 | 0.54 s | 0.32 s       | 0.03 s       | 2.6 s     | 0.92 s      |
  | test_array.mojo       | 38/38   | 0.64 s | 0.31 s       | 0.02 s       | 2.7 s     | 1.00 s      |
  | test_dict.mojo        | 15/32   | 1.16 s | 0.89 s       | 0.06 s       | 8.9 s     | 2.11 s      |
  | test_bitset.mojo      | 266/271 | 0.45 s | 0.19 s       | 0.03 s       | 1.7 s     | 0.70 s      |
  | test_string_span.mojo | 55/57   | 0.95 s | 0.42 s       | 0.05 s       | 2.1 s     | 1.43 s      |

  Spreads were small (test_list cold 0.82 to 0.85 s). Verification now costs
  about as much as importing the file, and with a warm cache at most 0.13 s.
- The post-elaboration pass on the same files: test_list.mojo 84 and 88 s
  wall, about 470 s CPU (3438 of 3463 obligations); test_span.mojo 1.6 s
  (884 of 886). The earlier 70 s for test_list.mojo was measured at a
  different load; these runs raised the load average to about 12
  themselves, and the difference is not explained. Its obligations are
  not the pass's: it checks every inlined call, stdlib bodies included,
  and verify-contracts each call once against its contract.
- The trait contract steps did not change these times measurably:
  test_list.mojo was 1.8 s before them, test_span.mojo 1.4 s (at load
  14).

`Iterator`:

- `bounds()` now states what its docstring asked of implementations,
  plus that a lower bound on a length is not negative: `result[0] >= 0
  and result[1].or_else(result[0]) >= result[0]`. `nth(n)` requires
  `n >= 0`, which its docstring states and a `debug_assert` checks.
  Contracts are erased when lowering to LLVM, so nothing changes at run
  time.
- Nothing is stated about `__next__`: the docstring calls `bounds()` a
  hint that iterators may not comply with, so the trait has no vocabulary
  for when `__next__` raises.
- Proving an implementation's `bounds()` needed the tuple it returns:
  `Tuple(*pack)` (a tuple literal) is now modelled field by field from
  the references in its `VariadicPack`, and a value a modelled
  constructor built keeps its fields wherever it is moved, including the
  returned value that the postcondition reads (`result[1].or_else(...)`
  goes through the tuple's field `/1` to the `Optional`'s `/has` and
  `/val`). Following nested fields only through such built values
  matters: following them through any field read first made test_span.mojo
  drop from 101 to 94 and two loops examples fail, because the same field
  then got two unrelated names depending on the order of reads.
- With `include-stdlib=true`, the trait's default `bounds()` and
  `_SpanIter.bounds` prove; the `List`, `Array`, `Dict` and `Optional`
  iterators do not. They return `length - index`, which is not negative
  only because the iterator keeps `index <= length`: an invariant of the
  struct that the language cannot state. A test iterator in
  test_zip.mojo (`TestIter`, whose `bounds()` returns its fields) is now
  flagged; its 10 other obligations were unproven before.
- Numbers unchanged (test_list.mojo 217 of 237, test_span.mojo 101 of 110,
  test_nth.mojo 13 of 13); the post-elaboration pass is unaffected
  (test_list.mojo 3438 of 3463, test_span.mojo 884 of 886).
  traits.mojo: 20 of 33, the other 13 in `bad_*` functions and `Bad*`
  structs.

Loop lengths and yielded values:

- Houdini has a new template: for a list the loop changes and an integer
  place its conditions depend on, `len(xs) + v` and `len(xs) - v` keeping
  their values on entry. `for _ in range(3, 6): xs.pop(3)` then keeps at
  least 4 elements, and `for i in range(n): xs.append(i)` leaves
  `len(xs) == n`. An iterator's `/end` and `/length` never change in the
  loop, so they are left out: against them the template only restates a
  bound on the length.
- Iteration yields element `cursor` of the collection, as a reference to
  the element's place when borrowing and as the element's value when
  consuming (`for x in xs^`, `for x in [1, 2, 3]`, now modelled for `List`
  and `Array`). An array literal's elements are known, as a list
  literal's were. A borrowed element's place is named by the `__next__`
  call, whose result is the raised flag, so the place carries the
  element's type explicitly: without it, test_deque.mojo's scripts
  applied `len` and `elem` to a Boolean, which z3 rejects, silently
  reading as unproven (and never cached). A scan of every stdlib test's
  and example's scripts with z3 now finds no errors.
- Proven: test_list.mojo 220 of 237 (217), test_deque.mojo 100 of 105
  (97), test_linked_list.mojo 155 of 173 (153), test_bitset.mojo 271 of
  271 (266); the others unchanged. loops.mojo: 19 of 31, the rest in
  `bad_*` functions.
- Timings after this step, measured as after `Sized` (median of 5, load average
  3.2 falling to 2.7), with the earlier verification times in parentheses:

  | File                  | Proven  | Verify, cold    | Verify, warm | Total, cold |
  |-----------------------|---------|-----------------|--------------|-------------|
  | test_list.mojo        | 220/237 | 0.88 s (0.83 s) | 0.13 s       | 1.82 s      |
  | test_span.mojo        | 101/110 | 0.59 s (0.61 s) | 0.03 s       | 1.27 s      |
  | test_deque.mojo       | 100/105 | 1.01 s (0.51 s) | 0.04 s       | 1.83 s      |
  | test_linked_list.mojo | 155/173 | 0.40 s (0.32 s) | 0.03 s       | 0.97 s      |
  | test_array.mojo       | 38/38   | 0.31 s (0.31 s) | 0.02 s       | 0.97 s      |
  | test_dict.mojo        | 15/32   | 1.12 s (0.89 s) | 0.06 s       | 2.31 s      |
  | test_bitset.mojo      | 271/271 | 0.17 s (0.19 s) | 0.03 s       | 0.65 s      |
  | test_string_span.mojo | 55/57   | 0.67 s (0.42 s) | 0.04 s       | 1.65 s      |

  Warm-cache times are unchanged, so the increase is solver time.
- Which change costs what, from builds interleaved file by file (5 runs
  each, medians, load average 1.5 rising to 4.7): the iteration model
  alone, then with the template.

  | File                  | Iteration only    | With the template |
  |-----------------------|-------------------|-------------------|
  | test_list.mojo        | 0.84 s (217/237)  | 0.93 s (220/237)  |
  | test_span.mojo        | 0.61 s (101/110)  | 0.62 s (101/110)  |
  | test_deque.mojo       | 0.85 s (100/105)  | 1.07 s (100/105)  |
  | test_linked_list.mojo | 0.32 s (153/173)  | 0.41 s (155/173)  |
  | test_array.mojo       | 0.33 s (38/38)    | 0.36 s (38/38)    |
  | test_dict.mojo        | 0.89 s (15/32)    | 1.16 s (15/32)    |
  | test_bitset.mojo      | 0.18 s (271/271)  | 0.18 s (271/271)  |
  | test_string_span.mojo | 0.43 s (55/57)    | 0.69 s (55/57)    |

  The iteration model costs test_deque.mojo 0.34 s and brings its 3 new
  proofs and test_bitset.mojo's 5; elsewhere it is within noise. The
  template brings test_list.mojo's 3 and test_linked_list.mojo's 2 for
  0.09 s each, and costs test_deque.mojo, test_dict.mojo and
  test_string_span.mojo 0.22 to 0.27 s for nothing. Restricting it as
  above (interleaved against the unrestricted template, same results):
  test_list.mojo 0.93 s instead of 1.05 s, test_linked_list.mojo 0.41 s
  instead of 0.47 s, test_dict.mojo 1.17 s instead of 1.25 s,
  test_deque.mojo 1.07 s and 1.08 s; test_string_span.mojo 0.69 s instead
  of 0.56 s, the exception: one Houdini script there (in `test_upper`)
  takes z3 0.35 s instead of 0.20 s although it has fewer candidates, and
  why is not explained. Candidates that only cost time are the next
  thing to cut: the template could be limited to lists the loop changes
  by a known step.

Cutting the template's cost:

- The time did not go into the template's own queries. Split into single
  queries, the slowest Houdini script of test_string_span.mojo took z3 at
  most 0.02 s per query either way. Together they took 0.47 s instead of
  0.12 s, because every inductiveness query assumed all kept candidates,
  the template's included.
- The template's candidates now come last in each Houdini script and are
  assumed only in their own queries; the others assume only each other,
  as before the template. It is still one z3 process per round. Running
  the template as a second Houdini phase instead, after the others,
  helped less: it costs a z3 process more per loop.
- Candidates are also generated only for a list and a variable that the
  iteration both changes: some end of the body holds another term for
  them than the loop head. This replaces excluding `/end` and `/length`
  by name.
- Results are the same on every file. Verification time, interleaved (5 runs
  each, medians); the load rose to 11.5 during the run, so the times are high,
  but the order matches an earlier run that was disturbed the same way:

  | File                  | No template | Committed template | Now    |
  |-----------------------|-------------|--------------------|--------|
  | test_list.mojo        | 0.92 s      | 1.02 s             | 0.98 s |
  | test_span.mojo        | 0.62 s      | 0.63 s             | 0.63 s |
  | test_deque.mojo       | 0.88 s      | 1.09 s             | 0.94 s |
  | test_linked_list.mojo | 0.39 s      | 0.48 s             | 0.43 s |
  | test_array.mojo       | 0.40 s      | 0.41 s             | 0.40 s |
  | test_dict.mojo        | 1.15 s      | 1.51 s             | 1.27 s |
  | test_bitset.mojo      | 0.27 s      | 0.27 s             | 0.28 s |
  | test_string_span.mojo | 0.49 s      | 0.77 s             | 0.48 s |

  The template now costs 0.04 to 0.12 s where it used to cost up to 0.36 s, and
  nothing on test_string_span.mojo. test_dict.mojo still pays 0.12 s for no new
  proof; which of its 94 loops that goes to is not measured.

Dictionary iteration:

- `for k in d`, `d.keys()`, `d.values()` and `d.items()` yield exactly
  `len(d)` times: the dictionary's iterators count the entries they have
  seen (`seen`), skipping removed ones, and raise when that reaches its
  length, in either direction. `reversed` of a dictionary or of its
  values or items starts a new count over the same dictionary. The keys
  and values are unknown: an indexed element model would make key `i`
  and value `i` the same unknown. Consuming a dictionary (`for k in d^`)
  counts over its order array instead and is not modelled.
- `List(it)` of an iterator the pass models is as long as the iterator
  has entries left (`length - index`), so `List(d.keys())` is as long as
  `d`.
- test_dict.mojo: 22 of 32 calls proven (15). The other 10 index lists
  built from dictionaries whose length the contracts only bound
  (`d[k] = v` adds one entry or none, depending on whether `k` was
  there).
- Cost: test_dict.mojo's verification doubles, interleaved against the
  previous build (1.06 s and 2.10 s, and 1.23 s and 2.46 s in another
  run; the load rose to over 14 in both, so only the ratio holds), while
  test_list.mojo and test_deque.mojo are unchanged. Almost all of it is
  `test_reversed_items`, whose three `reversed(d.items())` loops each
  append to two lists and were skipped before: each now runs a Houdini
  search of about 388 queries in its first round and six or seven
  rounds in all, one after the other (1.76 s of z3 time).

Lists as Houdini variables:

- A loop that appends to a list reads the list's length, and the list
  itself counted as one of the loop's integer variables: its value is a
  handle, so Houdini tried bounds between it, the lengths and the counters
  (`keys <= len(vals)`), none of which mean anything. Only one fact about
  a handle does: that the list still holds its value on entry, which
  the elements a loop reads depend on (`ok_list_elements` failed without
  it). A whole variable that is not an integer or a Boolean now only gets
  that equality candidate; fields are kept as before, as their types are
  not tracked.
- In `test_reversed_items` of test_dict.mojo this takes a loop from 194
  candidates to 104. Results are the same on every file. Verification
  time, interleaved against the previous build (5 runs each, medians;
  the load was between 4 and 12, so only the ratios hold): test_dict.mojo
  1.60 s against 2.05 s (and 1.55 s against 1.99 s in a second run),
  test_deque.mojo 0.83 s against 1.02 s (0.72 s against 0.91 s),
  test_list.mojo 0.82 s against 0.90 s (0.85 s against 0.88 s),
  test_linked_list.mojo 0.38 s against 0.40 s (0.33 s against 0.38 s);
  the other files within noise. Dictionary iteration still makes
  test_dict.mojo slower than before it, by about half instead of double.

Fewer Houdini candidates:

- Two templates can produce the same candidate (`len(xs) >= 0` from the
  bound against 0 and from 0 among the unchanged terms); each is now
  asked once.
- A list the loop changes is no longer bounded against other lists'
  lengths (`len(keys) <= len(other)`): a list the loop appends to fails
  those at once, and no example or stdlib test proves anything with
  them. It keeps its bounds against its own length on entry, 0, the
  other unchanged terms and the loop's variables.
- Results are the same on every file. Verification time, interleaved (5 runs
  each, medians, load 3 to 6):

  | File                  | Before dict iteration | Handle cut | Deduplicated | Now    |
  |-----------------------|-----------------------|------------|--------------|--------|
  | test_dict.mojo        | 0.98 s (15/32)        | 1.47 s     | 1.42 s       | 1.17 s |
  | test_list.mojo        | 0.86 s                | 0.73 s     | 0.69 s       | 0.65 s |
  | test_deque.mojo       | 0.88 s                | 0.71 s     | 0.68 s       | 0.67 s |
  | test_linked_list.mojo | 0.35 s                | 0.32 s     | 0.33 s       | 0.31 s |

  (The first column was measured against "Now" in a separate interleaved run,
  the other three together.) Dictionary iteration now costs test_dict.mojo about
  a fifth for its 7 new proofs, and the other files are faster than before it,
  as the cuts apply to every loop.
- What is left in test_dict.mojo is mostly `test_reversed_items`: three
  loops of about 100 candidates, four rounds each, one after the other.
  Each query takes z3 about 0.6 ms; the time is their number.

Linked list iteration:

- `for x in l`, `l.__reversed__()`, `enumerate(l)` and consuming
  iteration (`for x in l^`) yield `len(l)` times, element `cursor` (from
  the end when reversed), the element `l.get_nth(i)` reads. The consuming
  iterator pops while the size is positive, a real count; the borrowing
  one follows the nodes until there are none, so the model assumes the
  nodes are as many as the size, as `LinkedList`'s methods keep them.
- test_linked_list.mojo: 157 of 173 calls proven (155): a forward loop
  with a counter beside it, and `enumerate`. Cost, interleaved against
  the previous build: 0.38 s against 0.30 s (load 6 to 10; test_list.mojo
  and test_deque.mojo unchanged).
- Two loop shapes stay unproven, and not because of the linked list: a
  reversed loop with a counter counting down (`i = len - 1`, `i -= 1`)
  needs `i + cursor` to stay constant, a template between two integer
  variables that Houdini does not have; and a counter used only after
  the loop (`for _ in l: i += 1`, then `xs[i]`) is not one of the
  variables the loop's conditions depend on, so nothing relates it to
  the cursor. Both fail for `List` and `range` too.

Tensors and GPU kernels:

- A survey of `max/kernels/src` (675 functions that read GPU ids; the
  split is a regex classifier's, checked by hand on about 40 kernels):
  a guarded flat index (`if tid < n: t[tid]`) is about 5% of kernels and
  3% of indexing sites, mostly naive and reference kernels. Tiled views
  are 24% of sites, vectorized access 9%, raw pointers 29%, shared memory
  16%, and 29% of kernels only call tensor-core or copy intrinsics. Guards
  compare against scalar arguments more often than against `dim`.
- The post-elaboration pass cannot see kernels: they are compiled through
  `kgen.compile_offload`, a separate pipeline per target (a temporary
  hook running the pass there worked on a CPU offload), the open-source
  build has no GPU target at all, and tensor indexing recorded no
  obligations. So kernels are checked here, before elaboration, on the
  host, where they are ordinary functions.
- `TileTensor` (MAX's `layout` package) now states indexing bounds as
  contracts: `t[i]` (rank 1) and `t[i, j]` require each scalar
  coordinate in `[0, dim[k])` through `_coord_in_bounds`, and writes
  require every flat index in range through `_indices_in_bounds(*items)`.
  Contract regions are compiled for every instantiation, and `value()`
  of a tuple coordinate is a compile-time error, hence the helpers:
  tuple coordinates and nested layouts are not checked. The `layout`,
  `nn` and `linalg` kernel tests pass with them.
- The pass reads `dim[k]()` as the layout's static size where its shape
  says `ComptimeInt[n]`, and otherwise as `tdim(tensor, k)`, and models
  the helpers. GPU ids are one value per axis in a function with the
  launch limits every supported GPU has, as assumptions:
  `0 <= thread_idx < block_dim <= 1024`, `0 <= block_idx < grid_dim <
  2^31`; `global_idx` is `block_idx * block_dim + thread_idx`.
- Functions from every imported package are now skipped like the
  stdlib's: a file importing `layout` verified its 27 loops before.
- max/examples/gpu-intro's `vector_addition` kernel verifies unchanged
  (3 of 3). tensors.mojo: 11 of 16, the other 5 in `bad_*` functions.
- Tiled and vectorized views: `tile[*sizes](*coords)` requires
  `0 <= c < dim // size` per dimension through `_tile_in_bounds` (a tile
  has exactly `sizes` elements, and `(c + 1) * size <= dim` would wrap
  for a large `c`); the tile's own dimensions are static in its type, so
  indexing it needed nothing new. `vectorize[*sizes]()` requires each
  dimension a multiple of its size through `_vectorize_in_bounds`, a
  clause on `self`: the view's shape is `ceildiv(dim, size)`, so a
  ragged dimension's last vector reads past the tensor. The pass models
  the view's dimensions at the call as `ceildiv` of the parent's (its
  layout type holds them unevaluated, `ComptimeInt[apply(floordiv, 9,
  4)]`; the static parse now refuses that explicitly, where before it
  only failed because the `->` of the generator's type broke its bracket
  count, and would otherwise have read the operand 9 as the extent).
  tensors.mojo: 46 of 58, the other 12 in `bad_*` functions,
  among them a guard `j * 4 < dim` that does not prove `j < dim / 4`
  because the product wraps. The `layout`, `nn` and `linalg` kernel
  tests still compile and pass with both clauses.
- Vectorized loads and stores: `load[width](coord)` and
  `store[width](coord, v)` require, through `_access_in_bounds`, every
  scalar coordinate in `[0, dim)` and, above width one, the last at most
  `dim - width` with stride 1, since the access reads `width`
  consecutive elements of storage. Only a flat, unvectorized layout
  indexed by one value per dimension is checked; elsewhere (nested
  layouts, other ranks, vectorized views, whose elements `vectorize` and
  `__getitem__` already cover) the helper is true. The pass models
  `Coord(*values)` and `Coord(tuple)` as fields per element, reads a
  `ComptimeInt` element's value from its type, widens narrower integer
  coordinates by their signedness, and treats a runtime last stride as
  unknown. The idiomatic guard `j + width <= dim` does not prove the
  access: the sum wraps for `j` near `Int.MAX`. `j <= dim - width` does.
  tensors.mojo: 51 of 68, the other 17 in `bad_*` functions.
- Launch configuration: a kernel that tiles at `block_idx` without a
  guard relies on the grid its host chose. It now states that as `where`
  clauses on its arguments (`grid_dim.y <= Int(c.dim[0]()) // BM`,
  `block_dim.x == BM * BN`), assumed in its body like any precondition,
  and each launch, `enqueue_function[kernel](args..., grid_dim=,
  block_dim=)`, must establish them: the pass finds the kernel among the
  launch's parameters (behind the thunk that converts it to the declared
  `def` type), binds its arguments to the launch's, and its `grid_dim` and
  `block_dim` to the launch's `Dim` values. `Dim` is modelled from
  integers, literals and tuples. The custom-ops example's tiled matmul
  kernel, copied out with such a clause added (the example's host code
  needs the `extensibility` package, which was not built here), verifies
  10 of 10 in about 11 s. Its host launches with `grid_dim=(ceildiv(N,
  BN), ceildiv(M, BM))`; that pattern does not establish the clause
  (kernels.mojo, `bad_launch_rounded_up`) unless `M` and `N` are multiples
  of the tile, which the kernel assumes without stating it.
- What that kernel needed besides: `udivmod` and `divmod` (with the facts
  `q * b + r == a` and `r < b` for an unsigned nonzero divisor; without
  them, `tid udiv BN < BM` from `tid < BM * BN` timed out), the bounds of
  `comptime for k in range(n)`, an `Int` seen through the parameter type
  of a `comptime for` element, and tile sizes and static extents that are
  parameters of a generic function (`tile[BM, BN]`, `row_major[BM,
  BK]()`), evaluated in the scope they come from: a caller's `BM` passed
  to `tile` was a fresh unknown in the callee's scope before.
- Launched instantiations: of the tiled matmul's 10 s, z3 took 10.1 on
  three obligations needing `tid udiv BN < BM` from `tid < BM * BN` for
  every 64-bit `BM` and `BN` (2.3 to 3.8 s each, 57 to 65 million
  resource units; with one of them fixed, 0.1 s; with both, 0.01 s;
  bounding both to `[1, 1024]` only halved it). A generic kernel that is
  launched was then first proven generically with
  a smaller budget per query (`generic-rlimit`, default 10 million: its
  other obligations needed at most about 6 million), and what that leaves
  open is proven for each launched instantiation, the kernel encoded with
  its parameters bound to the launch's; such an obligation is reported as
  proven for the launched instantiations, or warned with a note at each
  launch where it is not. The matmul with a launch: 4.8 s at a 20M
  budget, 3.9 s at 10M, 2.9 s at 5M (load about 2); 1.2 s of it is the
  generic Houdini search, which the budget does not limit. z3's `rlimit`
  turned out to be per query, not cumulative over a script as the README
  said: three queries at 20M stopped at 60M together. Finding the launches
  first cost 1.6 s by itself while it named every call with
  `calleeName`, which prints all of a call's parameters; it now looks at
  the callee's symbol only (3 ms). A launch from a generic function is
  checked too. A kernel parameter the launch passes from its launcher's
  parameters is followed to the calls of the launcher that give it, up to
  three calls out: an instantiation is then a chain of parameter frames
  (the kernel's from the launch, the launcher's from its call, ...),
  the mechanism calls already use. Where no call gives a parameter, it is
  left unbound, an unknown in the kernel's scope (binding it to a
  parameter of another function could let a launcher's `BM` be read as
  the kernel's own `BM`), so the check holds for every value of it. The
  matmul launched from a function generic over `dtype` and layouts with
  fixed sizes (the custom-ops `execute`'s shape): 3.9 s; with `BM` and
  `BN` its parameters too, given one call out: 4.2 s (10.9 s before the
  chains, when the sizes stayed unknown).
- Of those 4 s, the instantiation took 0.12 s; the rest was the generic
  attempt (its Houdini search 1.1 s, which the budget did not limit, the
  three exhausted budgets 1.5 s, the obligations it did prove 0.6 s) and
  the compiler itself (about 0.45 s). So a launched generic kernel is now
  verified only for its launched instantiations, and the generic attempt
  first is the option `generic-launched=true`. The matmul from each of
  the four launchers: 0.62 s (3.96 s with the option). The claim is
  weaker, safe for the sizes launched in the module rather than for every
  size; kernels not launched in the module are still proven for every
  value of their parameters. kernels.mojo: 14 of 22, three of them only
  for their launched instantiations, the other 8 in `bad_*` functions,
  the same with the option.
- `tile` with `Coord` coordinates: `tile[*sizes](coords)` and the two
  `tile_with_offset` overloads require `_tile_coords_in_bounds[*sizes](
  coords)`, `tile(shape, coords)` requires `_tile_shape_in_bounds(shape,
  coords)`, both `0 <= c < dim // size` per dimension like the `Int` form.
  The first helper takes the sizes as integers rather than as the `Coord`
  `coord[*sizes]`: inside the generic `tile`, that `Coord`'s element types
  are an unevaluated `param_list.tabulate` over `tile_sizes`, which the
  pass cannot read, while an integer list it resolves through the frames.
  The model reads each `Coord` element as `load` and `store` do (a literal
  `ComptimeInt`, or the integer it was built with, widened), now one
  function, `coordElement`; a shape given at run time (`Coord(s, 8)`)
  works the same way. The `layout`, `nn` and `linalg` kernel tests pass
  with the clauses. tensors.mojo: 62 of 83, the other 21 in `bad_*`
  functions.
- `distribute[thread_layout](tid)` has no contract: a thread's
  coordinate is `(tid // stride) % threads`, in `[0, threads)` with
  floored division, and its view has `dim // threads` elements (rounded
  down) `threads` apart, the last at `threads - 1 + (dim // threads - 1)
  - threads <= dim - 1`, so the view is within the tensor for every
  `tid`; a dimension that `threads` does not divide leaves elements
  uncovered, not out of range. A swizzle remaps the offset and is assumed
  to keep it inside. What indexing the view needs is its extents: the
  pass models the call, `tdim(view, k) = dim[k] // threads[k]` for a
  positive count, reading the thread layout's sizes as terms (a generic
  kernel's `row_major[TM, TN]()` included) through `comptimeIntTerm`,
  now shared with `comptimeShape`. A static view's extents were already
  read from its type. `distribute_with_offset` (five uses in
  `max/kernels/src`, after `vectorize`) returns the same view as the
  first of a tuple: the pass models it as a tuple whose field `/0` is
  that view, so destructuring (`var v, coords, offset = ...`) and `r[0]`
  give it; the thread's coordinates and offset stay unknown.
  tensors.mojo: 75 of 99, the other 24 in `bad_*` functions.
- A function's own `where` constraints on its parameters (`def f[N:
  Int](...) where N <= 8`) are now facts in its body: the compiler rejects
  any instantiation that breaks them. They live in the function's
  signature (`getBodyConstraints` of its parameter list) and name its
  parameters by position (`#kgen.param.index.ref<0, i>`) where the body
  names them (`#kgen.param.decl.ref<"N">`); read as two unknowns the fact
  said nothing about the body's `N`, so a position is now read as the
  function's own `i`th parameter declaration. A constraint the parameter
  model cannot read is an unknown, which is harmless to assume. With
  `comptime for k in range(N)`, `t[0, k]` on 8 columns is now proven under
  `where N <= 8`. comptime.mojo: 11 of 17, the other 6 in `bad_*`
  functions, one of them a constraint on another parameter than the one
  indexed.
- The other `enqueue_function` overloads: a kernel compiled first,
  `ctx.enqueue_function(f, args..., grid_dim=, block_dim=)` with `f =
  ctx.compile_function[kernel]()` (133 `enqueue_function(` calls and 119
  `compile_function`s in the tree), is now checked like the thin one:
  `launchedKernel` already found the kernel among the call's parameters,
  but the pack and the dimensions were taken as operands 1 to 3, where
  `f` comes first. They are now found by value: the pack is the first
  operand the pass knows as one, the dimensions the first two `Dim`s
  after it. A generic kernel compiled per size is checked per launched
  instantiation too. A closure kernel (`enqueue_function(closure,
  grid_dim=, block_dim=)`) has no arguments and so no clauses; the
  overloads with `host_arg` (4 uses) and external functions are not
  checked. kernels.mojo: 16 of 26, the other 10 in `bad_*` functions.
- `ceildiv` (1213 calls in max/kernels/src, launch grids and loop bounds
  among them) is modelled as `SIMD.__ceildiv__` defines it: `-(a // -b)`
  for a signed type, the floored quotient plus one for a nonzero
  remainder for an unsigned one, so it agrees with the stdlib for every
  input, a zero divisor and wraparound included. The free function is
  generic over its type and returns through an out slot, where the model
  stores it; the method is a `SIMD` operator. A launch with
  `grid_dim=(ceildiv(n, 16), ...)` now establishes the tiled kernel's
  `grid_dim.x <= n // 16` under `n % 16 == 0`, and without that guard
  still does not. straight_line.mojo: 16 of 27, kernels.mojo: 17 of 27,
  the others in `bad_*` functions.
- Launches with `host_arg=` (four in max/kernels/src: a closure passed
  by host layout as the kernel's last argument) were found, but their
  obligations were not analyzed: the kernel's arguments were only the
  pack's, so its last formal had no actual. The operands between the pack
  and the first `Dim` are now appended as the kernel's last arguments
  (none in the other overloads); a launch with a grid that fits is
  proven and one with a block too many is not. The overload only takes a
  kernel whose host argument is a closure, so a clause on that argument
  is rarely one the pass could read anyway.
- With `generic-launched=true`, the loop-invariant (Houdini) queries of a
  launched kernel's generic proof now get a twentieth of `generic-rlimit`
  each, like its obligations get `generic-rlimit`, instead of a twentieth
  of `rlimit`: a candidate that runs out is dropped, which is sound. The
  matmul launched from a generic launcher: 3.57 and 3.56 s against 3.94
  and 4.14 s (interleaved, load about 3), keeping 4 invariants instead of
  6 with the same results. kernels.mojo: 19 of 30, the other 11 in
  `bad_*` functions.
- Raw pointers (29% of the kernel survey's indexing sites) have no size,
  so their bounds have to be stated. `Pointer._extent()` is a
  contract-only method (it returns `Int.MAX` at run time, and nothing
  computes it) that the pass reads as an uninterpreted `pext(p)`, not
  negative, of the pointer's value. Accesses at an offset require it:
  `p[unsafe_offset=i]` and the deprecated `p[i]` (235 uses in
  max/kernels/src) `0 <= i < _extent()`, the `load`, `store`,
  `unsafe_load` and `unsafe_store` forms with an offset `0 <= i <=
  _extent() - width`, through `_offset_in_bounds[width](i)`. Postconditions
  state it where pointers are made: `alloc`/`unsafe_alloc` and the
  `stack_allocation`s give their count, `unsafe_offset`, `+`, `-`, `+=` and
  `-=` move it (forward only: a pointer moved backwards has no known
  extent), `DeviceBuffer.unsafe_ptr()` gives `old(len(self))` (`old`,
  since the call takes a mutable reference and the pass forgets what it
  knew of the buffer across it), and `enqueue_create_buffer(n)` states the
  buffer's length. A kernel states it for its pointer arguments
  (`n: Int where dst._extent() >= n`); at a launch, a `DeviceBuffer` passed
  for a pointer has its length as the extent, the pointer it becomes on
  the device. A pointer whose extent nothing states is reported, as the
  user chose over leaving such accesses unchecked.
- `_extent()` returns its value as a literal rather than `Int.MAX`: with
  `Int.MAX`, the language server's lazy parse looped (String uses
  `unsafe_offset`, whose clause names `_extent`, whose body reached
  `max_or_inf` and `SIMD`'s conversions back to String) and three
  `kgen_lsp_check` tests failed. The stdlib tests that index a
  collection's private pointer (`q._data[unsafe_offset=0]` in test_deque,
  test_array) now report those accesses: test_deque 100/224 (was
  100/105), test_array 38/48 (was 38/38).
- On the way: the conversions in a callee's clause (`Int(offset)` of an
  `offset: Scalar` of any dtype) now resolve their dtypes through the
  parameter frames, and `paramTerm` reads `cast_from_builtin` (`Int(width)`
  of a `SIMDLength`).
- Not stated then (see below): `List.unsafe_ptr()` (its `ref self` may
  be in another address space, which `len(self)` in a clause cannot copy
  from) and
  `Span.unsafe_ptr()` (`@always_inline("builtin")`, which cannot carry
  clauses); `Pointer(to=x)`; and the accesses without an offset, `p[]`,
  `load()` and `store(v)`, which would need every such pointer's extent
  first. pointers.mojo: 6 of 11, kernels.mojo: 23 of 36, the others in
  `bad_*` functions.
- The remaining pointer extents. `p[]`, `load[width]()` and `store(v)`
  need an extent of at least `width` like any other access: `load()` and
  `store(v)` through `_offset_in_bounds[width](0)` (on `self`, or on `val`
  for the stores), `p[]` through an obligation the pass makes (`pext(p) >=
  1`). What no clause can state the pass models: `Pointer(to=x)` gives a
  fresh pointer with `pext >= 1` (at least: `x` may be an element of an
  array), `List.unsafe_ptr()` and `Span.unsafe_ptr()` give a fresh pointer with `pext >= len(self)` (a
  list's capacity and a span's memory can be longer), `Array.unsafe_ptr()`
  `pext >=` its size parameter, and the casts that keep the element type
  and the address (the implicit mutable-to-immutable conversion, which is
  an ordinary call, `as_imm`, `as_unsafe_any_origin`, `unsafe_as_noalias`,
  `unsafe_mut_cast`, `unsafe_origin_cast` and the address space casts)
  return the same pointer value, so its extent carries over.
  `unsafe_bitcast` changes the element size and keeps none.
- `Pointer(to=x)` first had the clause `out self where self._extent() >=
  1`. That compiled, but compile-time evaluation through it failed: the
  interpreter read a null `to` in `Pointer(to=self._mlir_value)` (in
  `Tuple.__getitem__`, reached from float formatting), so 257 Mojo tests
  and 84 of 110 kernel tests no longer built. It did so with `out self`
  first or last; without the clause they build. The pass models it
  instead. The same clause on `p[]`'s `self` (the method returns a
  reference) made code generation assert in `StackReuse` ("was supposed
  to be elidable") in 18 Mojo tests and 93 kernel tests, so that
  obligation is the pass's too.
- Extents describe bounds, not lifetimes: a pointer from
  `xs.unsafe_ptr()` keeps its extent after `xs.append(...)` reallocates,
  and one from `alloc` after it is freed.
- Stdlib tests: test_array 51/51 (was 38/48: the pointers from
  `Array.unsafe_ptr()` now have extents), test_list 225/237 (was 220),
  test_span 102/111 (was 101/110; `Pointer(to=a.unsafe_ptr()[])` adds one),
  test_dict 24/34 (was 22/32; `Pointer(to=dict["a"])[]` adds two), the
  others unchanged. pointers.mojo: 15 of 24, the others in `bad_*`
  functions.
- MAX's own kernels. Imported code is in the module only as far as the
  file uses it, so `packages=nn,linalg` (verify the named top-level
  packages' functions) only helps for what is referenced; to verify the
  kernels themselves, each source file of max/kernels/src ran as the main
  file (relative imports made absolute in a copy). 430 files, about 35
  minutes on 4 workers, median 1 s a file: 5951 obligations, 580 proven;
  145 files have none (their accesses go through `raw_load`/`raw_store`
  and intrinsics, which have no contracts). The unproven ones are mostly
  missing preconditions: kernels bound indices by scalar arguments (`m`,
  `k`, `num_tokens`) that nothing relates to the extents of their
  pointers or the dimensions of their tensors. About 20 files do not
  parse in this setup (the `extensibility` package is not built here).
- What the sweep found in the pass: three files crashed it in the IR
  printer, which asserts on a member-alias sugar whose sugared value is
  not a type (`--mlir-print-ir-after-all` asserts on them too); types and
  attributes it cannot print are now opaque. Integer `//` and `%` by zero
  were SMT-LIB's values (all ones for an unsigned quotient); they are 0,
  as `SIMD` defines them, with the zero case only where the divisor is
  not a nonzero literal. `lane_id()` is in `[0, 64)`. `ufloordiv`,
  `udiv_unchecked` and `uceildiv` are bounded (`0 <= q <= a` for `a >=
  0`): an exact quotient, multiplied by an unknown, made one Houdini
  script of a MAX kernel take 909 s instead of 4 s (11.7 s as shift and
  mask by the literal 32). `warp.broadcast(x)` stays an unknown (sound;
  the user chose it over assuming warp-uniform values).
- `gemv_kernel` with stated extents (`a._extent() >= m * k`, `b._extent()
  >= k`, `c._extent() >= m`, in a scratch copy): `b.load(idx)` proven;
  `c[global_warp_id]` and `a.load(global_warp_id * k + idx)` only without
  its `warp.broadcast`, the latter only over the integers.
- The integer retry. z3 does not decide `g*k + i < m*k` from `0 <= g <
  m < 2^31`, `0 <= i < k < 2^31` over 64-bit bit-vectors (`unknown` after
  13 s, also with the monotonicity fact given), and proves it over the
  integers in 0.15 s (0.12 s with exact carries, 0.00 s with `mod`). So a
  query answered `unknown` is translated and asked again: each `w`-bit
  value is its signed value; `+`, `-`, `*` wrap by `mod 2^w`; unsigned
  comparisons and divisions use the unsigned value; division keeps
  SMT-LIB's results for a zero divisor; shifts, masks, extracts,
  extensions and concatenations by constants are arithmetic; applications
  of declared functions are kept in range; other bit operations become
  unknowns in range (refused under a quantifier). Exact or weaker, so
  only `unsat` is taken. A differential fuzzer (random scripts in the
  pass's format, 8- and 64-bit, every translated operator, through
  `int-translate-file=`) found no query `sat` over bit-vectors and `unsat`
  over the integers in about 6,300 pairs both decided. Before results of
  declared functions were kept in range, gemv's query was `sat` over the
  integers (`pext` = 2^64): incomplete, not unsound. `int-retry=false`
  turns it off. The examples and stdlib
  tests have no query it changes; pointers.mojo's `ok_row_major` is proven
  only with it. On the kernel sweep it changed no count: those
  obligations fail for missing preconditions (`sat`), not `unknown`.
- Measuring: wall-clock A/B on this machine was dominated by other load
  (the same file 9 s and 920 s with the same binary, as queries crossed
  the 60 s cap); the comparison that held was z3's deterministic
  `rlimit-count` summed over the dumped scripts: old pass against this
  one 81.7M / 50.4M, 313.9M / 201.1M and 1028.8M / 1111.3M on the three
  files that looked slowest.
- Division by zero. Integer `//` and `%` return 0 for a zero divisor
  (what `SIMD` computes; its docstrings say neither that nor an error),
  and the encoding now says so; but dividing by zero is almost always a
  bug, so `check-division=true` makes each integer division whose
  divisor is not a nonzero literal an obligation ("that the divisor is
  not 0"): `//`, `%`, `__ceildiv__`, in-place `//=` and `%=` (otherwise
  not modelled), `divmod`, `udivmod`, `ceildiv`, `ufloordiv` and
  `uceildiv`. `udiv_unchecked` and `udivmod_unchecked` are undefined for
  `b == 0` (their docstrings), so they state `b > 0` as a precondition,
  checked at every call; the bounded model of `udiv_unchecked` had
  assumed a result for 0. Off by default (the user's choice) because on
  MAX's kernels it adds 2317 obligations, 190 proven: 2127 warnings in 237
  of 409 files, about 96% of them divisions by runtime or parameter values
  (`num_splits`, `inner_dim`) that callers are trusted with, 82 by
  `WARP_SIZE`, which is 0 on a host without an accelerator.
- Hardware as contracts. `WARP_SIZE` is `_resolve_warp_size()`: 32 or 64
  on a GPU, but 0 on a host without an accelerator, so a kernel cannot
  divide by it, or index with it, without saying where it runs. The user's
  direction: kernels state the hardware and capabilities they are built
  for as contracts (`where is_gpu()`, `is_nvidia_gpu()`, `WARP_SIZE ==
  32`), checked where they are launched, rather than leaving that to the
  launcher. Step one, in the pass: the target as `std.sys.info` describes
  it (the triple predicates, mutually exclusive, RDNA within AMD; the
  build's accelerator of at most one vendor; `is_gpu()`,
  `has_*_accelerator()`, `has_accelerator()` and `_resolve_warp_size()`
  following their bodies, the warp size of an accelerator the host only
  names unknown), and `comptime assert c` (`kgen.param.assert`) assumed
  from there on, which is sound because the compiler checks it wherever
  the code is compiled. A function's `where is_gpu()` was already assumed
  (its constraints), and the compiler enforces it at calls ("lacking
  evidence"), accepting `comptime if is_gpu():` or the same clause on the
  caller as evidence. `enqueue_function` does not accept such a kernel
  yet: the clause is part of the function's type, which none of its
  overloads take.
- Hardware contracts at launches. The user chose to change MAX's
  launch API so kernels can say `where is_gpu()` in their signature. The
  thin `enqueue_function`, `compile_function` and `DeviceGraph`'s
  `add_function` were given `func: def(*args: *declared_arg_types) thin
  -> None where is_gpu()`; unconstrained kernels still converted, but a
  kernel with the very clause did not. The compiler does not convert a
  function with a `where` clause to a variadic function type, even one
  with the same clause:

  ```mojo
  def run[
      declared: TypeList[Trait=AnyType, ...], //,
      f: def(* args: * declared) thin -> None where is_gpu(),
  ](): pass

  def k_gpu(x: Int) where is_gpu(): pass
  def k_plain(x: Int): pass

  run[k_plain]()  # accepted
  run[k_gpu]()    # error: 'run' parameter 'f' has 'def(*args:
                  # *TypeList[Int]()) thin -> None where is_gpu()' type,
                  # but value has type 'def k_gpu(x: Int) thin -> None
                  # where is_gpu()'
  ```

  (A non-variadic `def() thin -> None where is_gpu()` parameter accepts
  both; a generic clause `c: Bool, f: def() ... where c` accepts any
  clause but `c` is not inferred.) The change was reverted; the compiler
  would have to convert constrained functions to variadic types, and to
  let a launch discharge any clause against the device target (infer the
  clause, or have offload compilation check it), for signature clauses
  to work. Meanwhile a kernel's top-level `comptime assert`s that name
  the target are its hardware contract: the compiler checks them where
  the kernel is
  compiled, the pass assumes them in its body, and at each launch they
  are obligations for the build's accelerator (target predicates read as
  the accelerator's vendor, `WARP_SIZE` as its warp size; assumptions: a
  kernel is compiled for that accelerator, and the launcher runs on the
  host). A mismatch is reported at the launch ("cannot prove the
  assertion of 'k' for the accelerator it is launched on", with a note at
  the assert) instead of when the kernel is compiled for a GPU.
  `comptime if has_nvidia_gpu_accelerator():` establishes
  `is_nvidia_gpu()` and `WARP_SIZE == 32`; `has_accelerator()` does not
  establish `is_gpu()`, because an accelerator name the stdlib does not
  recognize has no known triple.
- Two regressions found by sweeping MAX's kernels with the launch checks.
  (1) The solver's wall-clock cap was not enforced: `ExecuteAndWait`'s
  timeout is the process-wide `alarm()`, which parallel calls overwrite,
  so z3 processes ran for 15 minutes to over an hour past a 60 s cap;
  z3 now runs under `waitpid` polling and is killed at the cap. (2)
  Assuming every `comptime assert` put products of kernel parameters
  into every query's path condition: block_scaled_matmul_small_bn.mojo
  took 18 s instead of 3.1 s; nonlinear asserts are no longer assumed,
  and only asserts that name the target are checked at launches. With
  both fixed, the files that had looked slowest take, one at a time at
  load 3-4: 3.4 s (block_scaled_matmul, 672 s in the broken sweep), 4.1
  s, 3.6 s, 10.2 s (gemv), 14.3 s, 18.7 s, 20.5 s and 24.4 s. A sweep
  that runs several `kgen`s at once oversubscribes the machine: each
  already runs its solvers on every core.
- Where solver time goes. Per query, with z3's cumulative `rlimit`
  after each check, over 30 random kernel files and 6 stdlib tests: in
  the kernels, 8% of the work proves obligations, 37% fails to (31% `sat`,
  6% at the limit), and 56% is loop-invariant candidates (16% that hold,
  30% that fail, 10% undecided at their limit); the stdlib tests need
  5% of the kernels' work and no proof there costs more than 2M. A lower
  first limit with a retry of what it leaves open saves nothing: nothing
  predicts which queries will be proofs. Measured on the same files
  (proofs, summed `rlimit-count`): the loop-invariant limit at 5M (a
  twentieth of 100M, the old default) 720/1395 and 3783M; at 1M 711/1395
  and 2784M (-26%; all 9 lost proofs in pipeline/schedulers.mojo,
  compile-time code whose 550 invariant candidates are 45% of the
  sample's work); at 0.5M 701/1395 and 2511M (also 3 in test_list). The
  obligation limit at 50M changed nothing (no obligation needed more).
  The user chose 1M as the default (`houdini-rlimit=`).
- A launch of a real kernel, proven. linalg/gemv.mojo's `gemv_kernel`
  takes raw pointers and `m`, `n`, `k`; with its extents stated
  (`a._extent() >= m * k`, ...), its launch in `gemv_gpu_dispatch`
  (`a.to_device_buffer(ctx)`, `Int32(m)`, ...) needs: the buffer's length
  (`to_device_buffer` now states `len(result) == self.num_elements()`),
  `num_elements()` as `dim[0] * dim[1]` (the pass models it; the layout
  is generic, so by the rank `comptime assert a.rank == 2` gives), `M`,
  `N`, `K` from `GemmShape.get` (now stated), `c` converted to immutable
  for `GemmShape.get` being the same tensor (modelled), and what nothing
  stated: that the shapes agree, fit in `Int32`, and that `n >= 1` (the
  kernel writes `c[row]` for every row below `m`, which `c` with `m * n`
  elements only has room for when `n >= 1`; `gemv_gpu` dispatches
  `GEMV_KERNEL` only when `n == 1`, so no caller breaks it, but the
  dispatcher does not say so). With those as preconditions of
  `gemv_gpu_dispatch` and `gemv_gpu` (in a copy; MAX's sources are not
  changed), the launch's two preconditions and the dispatcher's are
  proven (the products by the integer retry); the file takes 20 s. The
  cost of pushing contracts down to kernels is this chain: every
  function between the user and the launch states what its kernel
  needs.
- `warp.broadcast` (the user asked to address it after the gemv write-up
  had to remove it). Lane 0 is some thread of the same block, so
  `broadcast(x)` is `x` as some thread of the block computes it: the
  pass copies `x`'s term, renaming what varies by thread (thread ids,
  `lane_id`, arguments, since a device helper may be called with
  lane-dependent ones, and bounded quotients) to fresh copies and keeping
  what does not (parameters, block ids and dimensions, the target), and
  copies every fact about the renamed names. Which lane is never needed,
  so no warp-size arithmetic. The unmasked `shuffle_idx`, `shuffle_up`,
  `shuffle_down` and `shuffle_xor` get the same model (another lane's
  value or this lane's own); the masked forms do not (a source lane
  outside the mask is undefined). A term that depends on anything else (a
  load, an unknown result) leaves the value unknown, and a name that once
  failed to copy is never shared. Checked: bounds every thread has hold
  for a broadcast value; claims that it equals this lane's value, this
  thread's id, a load or a helper's argument are not proven. gemv_kernel
  as MAX has it (with `warp.broadcast`) now has all three accesses and
  its launch proven (18.7 s).
- Enum-like wrappers, so a contract can name the algorithm it is about.
  The gemv contracts stated the vector side of C as `>= 1` (memory
  safety); a correct result needs `== 1`, but only for `GEMV_KERNEL`,
  since the dispatcher also runs `GEMV_SPLIT_K` with `M` up to 16. Saying
  `kernel_func is not GEMVAlgorithm.GEMV_KERNEL or ...` needs the
  comparison: a struct with one integer field and no parameters is now
  represented by that integer (field extraction is the identity, a
  constructor that only stores its argument gives it, also in a
  `comptime` constant's `apply`), and its own methods whose bodies only
  extract, rebind, call and return are evaluated in place (`__is__` calls
  `__eq__`, which compares fields). A method from a trait default (taking
  references, like `Equatable.__ne__`) is not evaluated. `Bool.__bool__`
  (inserted for `not (...)`) was an unknown; it is the identity.
- With the per-kernel clause (non-transposed `GEMV_KERNEL` needs `N ==
  1`, transposed `M == 1`) on the dispatcher, in a copy of gemv.mojo,
  `gemv_gpu`'s call to it is reported: its `n == 1` branch picks
  `GEMV_KERNEL` without checking `transpose_b`, the transposed bug the
  GPU test confirmed. With that branch fixed (`n == 1 and not
  transpose_b`), the call is still reported, for the MiniMax branch
  (`M <= 16`, f32, static `K = 6144`), which takes `GEMV_KERNEL` only if
  `k % simd_width != 0`: that never happens, but proving it needs the
  pass to relate a generic tensor's `static_shape` to `dim()` and to know
  the target's SIMD width (not modelled; a false positive). Without that
  branch the call is proven.
- The MiniMax branch, proven. Four pieces, each general:
  - `simd_width_of[dtype]()` is 0 or a power of two, at most 256 (an
    assumption read off the targets: `simd_bit_width` is 128 on every GPU
    target, at most 512 on CPUs, 0 for no target; a dtype's bit width is 8
    times its size, a power of two). `6144 % w == 0` follows.
  - `static_shape[k]` of a generic layout and `dim[k]()` of its tensors
    are related, but only where the layout is flat: `static_shape` counts
    flattened leaves, `dim` outer modes, so for a nested mode 0 of shape
    `(p, q)`, `static_shape[1]` is `q`. The fact is `flat_rank == rank and
    static_shape[k] >= 0 => dim[k]() == static_shape[k]`, with `rank`
    and `flat_rank` one term per layout (keyed by the frame the layout
    resolves to, so a callee's `Self.LayoutType` is its caller's tensor's
    and two callees' are different). `flat_rank` is known only as the
    layout's witness (`a.LayoutType.flat_rank`): `TileTensor.flat_rank`
    is `_Flattened[...].length`, which the frontend unrolls into a 35 KB
    expression that no longer names it, and recognizing it by shape could
    take something else for it.
  - `dtype in (DType.float32, ...)` (`Tuple.__contains__` as a parameter
    expression) is a disjunction of code comparisons, only for tuples of
    dtype literals.
  - A dtype's code (`#pop.dtype_to_ui8`, what `DType.__eq__` compares) is
    one term per dtype after resolving through call frames; a callee's
    clause (`a_type == .float32`) now speaks of its caller's `a.dtype`
    instead of a fresh unknown per callee.
  In the gemv copy, with `is_minimax_router_gemm` given its result as a
  clause and `gemv_gpu` stating `comptime assert a.LayoutType.flat_rank
  == 2`, the dispatcher's precondition is proven; with the `n == 1` bug
  restored it is still reported. That assertion is new: the MiniMax test
  reads K from `a.static_shape[1]`, which is not K for a nested `a`, so
  gemv relies on flat layouts without saying so.
- What the MiniMax models cost. Measured against the pass before them on
  43 files (the examples, 8 stdlib tests, three gemv copies, and 18
  linalg kernel files that use `simd_width_of`, `static_shape` or `in
  (...)`): summed z3 work (`:rlimit-count` over the dumped scripts,
  deterministic), and the pass's time (median of 3 interleaved runs).
  As first committed, gemv's work tripled (755M -> 2262M) and its pass
  went from 24 s to 40 s: with the width a power of two (`w & (w - 1) ==
  0`), `(WARP_SIZE * w) % w == 0` in the vector kernels became provable
  but not within the bit-vector limit, so four queries per kernel
  instance ran to 100M and six `vectorize` preconditions were left
  undecided. Two changes:
  - The width as cases (`w` is 0, 1, 2, ... or 256): all eight are
    proven, still at the limit, by the integer retry.
  - `nonlinear-rlimit`: a goal that multiplies or divides two unknowns
    is first asked with 10M; what that leaves open over the integers
    with that limit and a 1 s timeout (nonlinear integer queries can run
    for long within their limit: one of matmul_mma's ran 30 s under
    10M); what is still open once more with the full limit, as before.
    No proof is lost; the timeout decides only which step proves.
  Result: work 3384M -> 2658M, proven 1410 -> 1439, every obligation
  decided in both; summed pass time 131.8 s -> 97.4 s. gemv 23.9 ->
  14.0 s, pointers.mojo 7.5 -> 2.0 s. Slower: amd_ping_pong_matmul 5.6
  -> 7.7 s, matmul_mma 4.9 -> 6.3 s (integer timeouts, with less work),
  matmul_output 0.8 -> 1.1 s.
- gemv's clauses are in max/kernels/src/linalg/gemv.mojo (`flat_rank`
  as a clause of `gemv_gpu`, not an assertion: clauses do not change
  what compiles). The in-tree file reports the transposed bug.
- gemv_kernel_vector, proven in all three launched instantiations, with
  its clauses (`m`, `k` within the tensors, `k % simd_width == 0`, which
  its last iteration needs: it checks where a vector starts, not where it
  ends) and the dispatcher's for `GEMV_KERNEL_VECTOR` (`k % simd_width ==
  0`; `N == 1`, or transposed with `M == 1`: its `m == 1` launch reads B
  as N x K). What it took, each general:
  - `x if comptime (c) else y` (`num_iters`): an unsupported region that
    forgot everything known; now the taken arm's value.
  - An empty origin union (`#lit<origin.union >`) wrote everything at a
    loop head; it names no memory (`MutAnyOrigin` is another attribute).
  - `lane_id() < WARP_SIZE` (was `< 64`); an unnamed accelerator's warp
    size 32 or 64; a launched kernel runs on a GPU; `ufloordiv(a, 0) == 0`.
  - Partial tiles: the kernel's last iteration takes a tile past the row's
    end and guards each lane. `_valid_dim` (contract-only, in
    tile_tensor.mojo) is what `__getitem__` and the other access clauses
    check; `tile` requires only coordinates not negative; the pass tracks
    `clamp(valid - c * size, 0, size)` through `tile` and `valid // size`
    through `vectorize`, with a whole tile stated as `size` (else a tile of
    parametric size became undecided). A partial tile passed elsewhere is
    an obligation that it is whole. In a kernel without clauses the
    failure moves from taking the tile to reading it (the in-tree gemv: 33
    -> 30 proven before the other changes, the same check).
  - `reshape(row_major(Coord(n, k)))`: the view's dimensions, and an
    obligation that it fits (the `n == 1` launch reshapes B).
  - Case splitting: `(lane + last * WARP_SIZE) * simd_width` and
    `k // (WARP_SIZE * simd_width)` are products and quotients of two
    unknowns, which bit-blasting does not decide. Both have few values (32
    or 64; 0 or a power of two up to 256), so a query left open is asked
    per combination with them defined as constants (at most 64 cases; the
    first case alone, then batches of 8), with unsigned quotients exact
    only there (exact everywhere cost 8x). A refuted query is asked again
    only if it depends on what the cases change. A `simd_width_of` call
    in a clause and the same `comptime` in a body are one term. Measured
    with `case-split=false`: the cases cost nothing overall (125.0 s
    against 124.4 s on the 43 files) and prove 4 more.
  - Local closures called directly (`_one_row_per_warp`) are walked at
    their calls with the caller's state; before, the closure's definition
    forgot everything and its launch was checked without context. One
    passed on (an epilogue, `vectorize[f]`) is still verified alone; the
    first version also skipped those (lora's epilogue went unchecked), a
    callee's parameter list now counts as passing it on.
  A third latent issue: gemv's MiniMax branch can pick
  `GEMV_KERNEL_VECTOR` for `1 < M <= 16` (when `ceildiv(N, 2)` exceeds
  the device's `MAX_GRID_DIM_Y`), and the dispatcher's vector branch then
  launches nothing. Unreachable today (MiniMax's static N is 128), but the
  code states neither that the runtime N is the static one nor the grid
  limit; the in-tree file reports the call for it and for the transposed
  bug. Cost on the 43 files against the committed pass: proven 1439 ->
  1500, obligations analyzed 2034 -> 2154 (closures, comptime-if arms),
  solver work 2658M -> 3801M, summed pass time 97.5 -> 156.5 s (load
  3.5-4); gemv 14 -> 27.5 s with 18 more proven.
- `gemv_kernel_vector_multirow` (the vector kernel with `rows_per_warp`
  rows per warp), with the vector kernel's clauses: proven for its launched
  instantiation (`rows_per_warp = 4`; not for every value, where
  `global_warp_id * rows_per_warp` may wrap), and both of its launches.
  What it needed:
  - Closures with parameters of their own. The kernel is launched from
    `_rows_per_warp[rows]`, called as `_rows_per_warp[4]()`: the callee is
    `bind_params(closure, 4)`, which counted as passing the closure on, so
    it was verified alone. Such a call is now walked like a plain
    closure's, with a frame that binds only the closure's parameters and
    passes every other name on to the caller's (a caller's `transpose_b`
    read inside the closure, or given to a callee there, is the caller's
    term). Parameter constants in the body are evaluated again at each
    call: the first version kept the first call's values, and `f[4]()`
    followed by `f[8]()` proved `rows == 4` twice
    (`bad_closure_parameter`).
  - Launches in such closures: the kernel's instantiation takes the
    closure's values from its calls (`kernel[rows]` is `kernel[4]`), and
    the enclosing function's callers give the rest as before. This also
    gives `gemv_split_k`, launched from
    `_gemv_split_k_dispatch[num_threads, tile_n, ...]`, five
    instantiations instead of two.
  - `min` and `max` of integers (`min(row_base + r, last_row)` clamps the
    row): the smaller and larger as the dtype compares.
  - The same script is run once per run of the pass. Instantiations that
    differ in a parameter the proof does not read have identical scripts
    (the ping-pong matmul's three `KernelConfig`s: 421M of work without
    this, 140M with it).
  Removing the clamp or the `row < _m` guard of the stores in a copy is
  reported. Cost on 44 inputs against the committed pass: proven 1660 ->
  1685, obligations 2453 -> 2459, solver work 6063M -> 6920M (+14%): the
  gemv copies 920M -> 1065M each (the three more `gemv_split_k`
  instantiations), the ping-pong matmul 118M -> 140M with nothing more
  proven (the closures in its kernel are now walked at each call: its
  script grew from 1285 to 1729 lines) and blockwise fp8 307M -> 332M,
  not looked into. Times were not comparable (load 7-60).
- `dump-dir=` naming a directory that does not exist left every script
  unwritten and every obligation "not decided within the solver limits";
  the directory is now created, and it is an error if it cannot be.
- `gemv_split_k`, first step (its clauses came with the second step,
  below). The
  kernel accumulates over K in a closure, `_k_iter_body`, that counts its
  calls in a captured `iteration`; the launcher's loops call it
  `unroll_factor` times per step (`comptime for`), then once per
  remaining iteration. What it needed:
  - A fix: a loop did not count what a closure it calls writes among the
    variables it changes, so `iteration` stayed 0 at every loop head, and
    six tile preconditions were proven for iteration 0 only. With a
    counter incremented in a closure called five times, `xs[counter]` was
    proven for a list of length 1 (`bad_closure_counter`).
  - `comptime for` indices in written elements and `Coord`s, one unknown
    per iteration, `load`/`store` with a parameter width,
    `size_of[dtype]`, `warp_id()`, strided ranges with the whole-steps
    invariant, `align_down`.
  - Unrolling `comptime for _u in range(unroll_factor): _k_iter_body()`,
    so that `iteration` is the strided loop's cursor exactly.
  - `simd_width_of[dtype, target=get_gpu_target()]` is 16 over the
    dtype's size in bytes: 0 only for `int256`/`uint256`, which nothing
    rules out for the dispatcher's dtypes.
  With clauses on `m`, `n` and `k` (extents, `k % simd_width == 0`,
  `block_dim.x == num_threads`, and the grid covering whole tiles where
  the bounds guards are off), the two router-gate instantiations prove
  everything but the AMD `weight_tile.load[simd_width]` (the `load`
  clause's model needs a literal layout). The remainder loop needs
  `invariant-retry=true`: that the counter ends where the unrolled loop's
  cursor did takes 1.2M units on entry against the candidates' 1M. Open:
  the dispatcher's instantiations (one with an unknown config tuple),
  both launches, the AMD load.
  Cost, 41 inputs on a quiet machine, against the committed pass: proven
  1549 -> 1573, solver work 4.30B -> 4.66B (+8%), summed time 220 ->
  310 s. The first version cost +60%: the invariant search rediscovered
  each strided range's end and step every round (gemv: 195M -> 1356M
  units), `warp_id()` was a quotient of two unknowns, and asking
  undecided candidates again, then by default, made kernels without
  clauses fail slowly. What is left: gemv's longest chain of solver runs
  (functions are verified in parallel) is now a `gemv_split_k`
  instantiation whose config is unknown (29 s against 16 s), and three
  obligations of blockwise_fp8 that could not be evaluated before are
  now nonlinear queries that stay open (332M -> 632M).
- `gemv_split_k`, second step: its clauses are in the tree, with the
  launchers' (`router_gate_mixed_gemv`, and `gemv_gpu_dispatch` for
  `GEMV_SPLIT_K`). What it needed:
  - `size_of` called in a clause; a tile of a generic layout in `load`'s
    clause, with a compile-time requirement on the kernel that `act` and
    `weight` are flat with a last stride of 1 (stated only by the AMD
    `load`, relied on by `vectorize` too).
  - The K loop with `unroll_factor == 1`, whose variable is unused beside
    the counter: a candidate that the counter is as many steps from its
    value on entry as the cursor is from its start.
  - `static_N` is `dim[1]` only for a flat layout: the dispatcher asserts
    `type_of(c).LayoutType.flat_rank == 2`.
  - A report fix: an instantiation that unrolls a loop fewer times was
    reported as not analyzing the iterations it lacks (22 warnings).
  - `invariant-retry` for any undecided candidate, at four times the
    limit: this kernel's invariants are valid and at the limit (0.7M to
    2.0M units against 1M), and which instantiation lost them changed
    with any change to the script.
  A dead end: the dispatcher takes `(num_threads, tile_n,
  unroll_factor)` from a function the verifier does not evaluate.
  Stating the values it can return as facts changed nothing, since
  unrolling is decided from constants while the body is walked. Such an
  instantiation is now skipped with a warning (it proved nothing and
  cost the most), and `gemv_split_k.mojo` launches the kernel with eight
  configurations written out, which a compile of the file checks
  against the NVIDIA function (the AMD one depends on the accelerator
  it is built for and is not covered; the dispatcher's own launch with
  these values is not either).
  Result with `invariant-retry=true`: both router-gate launches and
  their instantiations proven; the dispatcher's MiniMax launch proven
  and its instantiation except two queries undecided at 5.1M to 6.8M
  units against 5M (the dtype is unknown there); the harness 139/139
  in 18 s. On the gemv file that is 153 of 179 obligations; without the
  option 150, one router-gate instantiation losing its loop invariants.
  Cost, 43 inputs, against the pass before this step: proven 1694 ->
  1845, solver work 6.54B -> 5.42B, summed time 443 -> 291 s; only the
  gemv copies and the examples changed.
- `gevm_kernel` and `naive_gemv`: their clauses are in the tree, with the
  dispatcher's for `GEVM_KERNEL`. What they needed:
  - `b[row * n + col]` with `row = i * warps + warp_id`: valid, and
    unknown after 60 s over the bit-vectors and over the integers. The
    integers' `mod 2^64` is in the way: products that are first proven
    in range are now taken exactly (the exact retry, see the README),
    and the bound is decided in under a second.
  - `t.ptr` as one value per tensor, so that a clause can state its
    extent.
  - A write through `MutAnyOrigin` (every tensor's) no longer makes the
    local variables unknown that are never borrowed mutably: the
    pointers and sizes read before a loop, its range, the index of the
    loop around it. Assumed: an immutable borrow is not turned into a
    pointer that is written through.
  - The launch clauses of a skipped instantiation are not asked.
  `naive_gemv`'s first clause had no bound on M and K: the verifier's
  counterexample was `M * K` wrapping.
  Result with `invariant-retry=true` on the gemv file: 159 of 175, both
  kernels and the gevm launch proven. Open there: the two skipped
  launches, the dispatcher's clause at `gemv_gpu`'s call for
  `GEMV_KERNEL` (transposed `b`, `n == 1`, more than one row), and the
  `gemm_mma_cpasync` code.
  Cost, 43 inputs, against the pass before this step, on a loaded
  machine (so work, not time): proven 1862 -> 1874, solver work 5.92B ->
  4.67B; the gemv file 1854M -> 1222M (the skipped launches' clauses).
  Blockwise fp8 proves 9 of 22 where it proved 3, for less work (632M ->
  474M). Two files cost more with nothing more proven: the AMD ping-pong
  matmul 130M -> 174M and matmul_mma 196M -> 219M, not looked into. The
  exact retry's steps are limited by time, so its answers can depend on
  the machine's load.
- `gemm_mma_cpasync`, first part (the kernel is not done). Its memory
  accesses are intrinsics and shared-memory types that had no contracts,
  in loader and computer structs whose methods advance a stage counter.
  What the pass needed so far, none of it specific to the kernel: a
  struct's other fields kept when one is written (also at loop heads and
  joins), fields of struct values, writes into shared or global memory
  confined to those address spaces, `rebind`, `Int` of an index, the
  pointer of an array field, a loop's parameters as invariant bounds,
  and clauses kept when a reference argument is not tracked. See the
  README. `dump-fn=` prints a function as the pass reads it.
- `gemm_mma_cpasync`, second part: the shared-memory side. Copies into
  a tile go to swizzled offsets that one method (`prepare`) stores in an
  array field and another reads in its loop; the tensor-core loads read
  at swizzled offsets computed in place. `Swizzle.__call__` now states
  what it keeps (every bit outside its `zzz_mask`, when a right shift
  moves the `yyy` bits onto it), proven from its body; the kernel pins
  the masks of its swizzle with `comptime assert`s, which the compiler
  checks per instantiation; and `prepare` states its array with a
  quantified clause. In the pass: shifts and `~`, clause-less helpers
  with one `return` evaluated at their calls, the fields of a comptime
  struct value, a `comptime for` that fills an array (a quantified fact
  for any count, not an unrolling), comptime-indexed array elements,
  `Optional` comptime parameters, and two fixes to what a call is taken
  to write (clauses, `muttoimm` origins). The first version of the
  array rule proved a false example (an iteration reading an element an
  earlier one stored, which the pass had folded to the stored value);
  reads are now tracked where they happen. Cost on the 44 corpus inputs:
  as many obligations proven as before (1892), 5% more solver work
  (5.83B to 6.14B units); a first version cost 25% more, from loops
  tried as array-filling ones that could not be, and from helpers
  evaluated that only load through pointers. Still open in this kernel:
  the reads from global memory through `TileTensor._linear_offset`.
- No answer depends on the clock any more. The gemv file gave between
  137 and 155 proven obligations and took 80 to 950 s from run to run,
  with the same solver work. CPU load did not reproduce it (12 runs with
  every core busy agreed); freezing the solver processes 80% of the time
  did: three obligations lost and two solver runs killed at the 60 s
  cap. The pass had three z3 `:timeout`s (0.5, 1 and 5 s, on the integer
  retry's queries) and the cap. Measured on the corpus' 916 time-limited
  queries: those decided need little work (range checks at most 67k
  units, exact goals 567k, integer retries under 1M for 47 of 49), those
  not decided burn their whole limit slowly (10M units in a minute), and
  one ran 314 s under a 60 s `:timeout`. So the timeouts are now work
  limits (`range-rlimit` 300k, `int-rlimit` 1M, for every integer
  query), and the cap is a safety net (600 s) whose use is reported.
  With the solver frozen as before, the gemv file now gives the same
  diagnostics as on a quiet machine (in 556 s instead of 85). Corpus:
  1910 obligations proven before and after, pass time 471 s to 440 s.
  What stalled the original runs was the machine: it slept, every
  quarter of an hour (one 16 s solver run took 917 s, of which the
  power log shows 903 s asleep), and the cap, on a clock that runs on
  through sleep, then killed solver runs that had hardly started. The
  cap now counts only the time a process could run. Run times here are
  CPU seconds since.
- `gemm_mma_cpasync`, third part: the reads from global memory, and the
  kernel's clauses in the source. `TileTensor._linear_offset` is modelled
  on the assumption `t[i, j]` already rests on (a tensor's elements are
  backed): in bounds and with contiguous rows, the row's rest follows
  the offset. The K column a vector reads was the hard part: chunk index
  times chunk size plus iteration times `tile_k // 4`, for any `tile_k`.
  In the loop this did not prove at any cost tried. It is now a helper
  (`_gmem_k`, replacing an offset the loop carried) with a contract, and
  the kernel asserts `tile_k` is 64, 128, 256 or 512, which the pass
  checks one by one: the helper's contract proves in 2 s, and the loop
  only establishes its precondition. Tried and dropped: an invariant
  for a variable incremented by a parameter amount (needs the exact
  integer retry for every obligation about it), and divisibility facts
  for `%` by a symbolic divisor (cost proofs elsewhere). A launch of
  the kernel with `tile_k` 128 and two stages is fully proven
  (`gemm_mma_cpasync.mojo`); with the linalg package, 54 of 55 in 172
  CPU seconds.
- `gemm_mma_cpasync`, the launcher: clauses on its sizes and, by the
  operands' rank, on the batch, the output's extent and the operands'
  dimensions; `_to_batched_3d` states what its view is. With these all
  four launches establish the kernel's clause and the kernel is verified
  for each: 72 of 72 obligations with the linalg and structured-kernels
  packages. Found on the way: the launcher was never verified before (a
  package's function is loaded only if the file verified uses it; the
  example now calls it); a local named by debug info counted as
  borrowed, so package code forgot locals that the same code as a main
  file kept; the launch's shared-memory size was taken from the wrong
  optional argument; and a kernel's `stage_cnt <= 65536` was not assumed
  where a launcher computes `stage_cnt` by a division.
- Not covered yet: a runtime last stride, tensors
  whose runtime size comes from a scalar (`row_major(n)`: `dim` is not related
  to `n`), and kernels in MAX's own packages, which are now
  skipped as imported code unless `include-stdlib=true`.

## Risks and open questions

- **Stdlib coverage.** Before inlining, every call the proof goes through
  needs a contract or built-in semantics. Until a function has them, facts
  about its result are lost, and proofs that the post-elaboration pass finds
  by looking into bodies fail. Stage 6 is where this shows. The fallback tier
  keeps those functions checked meanwhile.
- **Trait methods.** Calls through a witness (`T.__init__`, `T.__eq__`) have no
  body to look at before elaboration. A trait method's `where` clauses are now
  its contract, checked against every implementation, but a trait's clauses can
  only use its own methods, and of the stdlib traits only `Sized` and `Iterator`
  state any yet. Contracts that are too strong break legitimate implementations
  (a `copy` that counts copies is not equal to its source), so they need to stay
  narrow.
- **Unsafe code.** Pointer arithmetic inside `List` and `Span` is what their
  contracts abstract. Verifying those implementations against their contracts
  needs the heap model of the current pass at the LIT level, or stays with the
  post-elaboration pass.
- **Soundness of built-ins.** The built-in table must match the stdlib's
  definitions of the operators. A test compares each entry with the result of
  elaborating the operator.
- **Integer semantics.** `Int` wraps, and the proofs keep that (bit-vectors),
  so `i + 1` can still overflow. The domain must be sound for wrapping too.
- **Closures and `raises`.** `lit.try` must be modelled for range loops.
  Closures are opaque at first.
- **Where exactly to run.** Right after `check-lifetimes` gives the most
  stable input. One round of cheap cleanups (the ones in the check pipeline)
  could make the IR easier to analyze without making results depend on
  optimization levels.
