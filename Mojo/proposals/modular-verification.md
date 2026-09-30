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

It builds on [argument contracts](argument-contracts.md), which already
states preconditions and postconditions as `where` clauses and keeps them as
`kgen.requires` and `kgen.ensures` ops.

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
- State bounds as preconditions (`where` clauses) on the declarations
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
def __getitem__(ref self, idx: Int where 0 <= idx and idx < len(self), /)
    -> ref[...] Self.T:
```

`check_bounds` stays for its runtime assertion. Since negative indexing has
been removed, the precondition is the whole requirement.

Loops over ranges need the iterators to say what they yield. `range(n)`'s
`__next__` gets a postcondition (`0 <= result and result < self.end`), and the
iterator's fields a type invariant (`curr <= end`), which the parser can
already express as `where` clauses on `mut self`.

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
  list.mojo:1572:20: note: precondition of 'List.__getitem__'
      ref self, idx: Int where 0 <= idx and idx < len(self), /
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
