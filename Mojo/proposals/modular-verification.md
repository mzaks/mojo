# Modular verification before elaboration

**September 29, 2026**
Status: Draft. Stages 1 to 3 are implemented; see
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

| Where the solver time goes (default mode) | z3 CPU  | Share |
|-------------------------------------------|---------|-------|
| `_test_copyinit_trivial_types`, 7 dtypes   | 249 s   | 90 %  |
| `test_list_insert`                        | 16 s    | 6 %   |
| everything else (47 functions)            | 13 s    | 4 %   |

| Kind of query                  | z3 CPU | Share | Queries |
|--------------------------------|--------|-------|---------|
| Loop invariant inference (Houdini) | 206 s | 74 % | 12,702 |
| Obligations                    | 72 s   | 26 %  | 2,997   |

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

## Risks and open questions

- **Stdlib coverage.** Before inlining, every call the proof goes through
  needs a contract or built-in semantics. Until a function has them, facts
  about its result are lost, and proofs that the post-elaboration pass finds
  by looking into bodies fail. Stage 6 is where this shows. The fallback tier
  keeps those functions checked meanwhile.
- **Trait methods.** Calls through a witness (`T.__init__`, `T.__eq__`) have
  no body to look at before elaboration. They need contracts on the trait
  method, which the language cannot express yet. Until then they are opaque.
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
