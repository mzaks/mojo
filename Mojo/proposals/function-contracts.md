# Function contracts: `requires` and `ensures` clauses

**October 9, 2026** (replaces the argument-clause design of September 28, 2026)
Status: Draft. Preconditions and postconditions are implemented; see
[Implementation status](#implementation-status).

This document proposes contracts on functions, written as `requires` and
`ensures` clauses at the end of the signature:

```mojo
def get(xs: List[Int], i: Int) -> Int
    requires 0 <= i and i < len(xs):
    return xs[i]

struct List[T: Copyable & Movable]:
    def pop(mut self) -> Self.T
        requires len(self) > 0
        ensures len(self) == old(len(self)) - 1: ...
```

A `requires` clause is a precondition: the caller must establish it, and the
body may rely on it. An `ensures` clause is a postcondition: the body must
establish it at every return, and the caller may rely on it after the call.
This is Dafny's spelling, and a reader sees at once which is which. The
contracts are kept through elaboration as operations of their own, so a
static verifier can check each function body against its contract once, and
check each call against the callee's contract without looking into the
callee's body. That is what makes tools like Dafny fast, and what our current
contract helpers cannot provide.

The first version of this proposal wrote contracts as `where` clauses on the
arguments, and the argument's convention decided what a clause meant. That
turned out to be hard to read; see
[Why not clauses on arguments](#why-not-clauses-on-arguments).

## Background

The `bounds-check-report` pass (`Mojo/lib/Transforms/BoundsCheckReport.cpp`,
on the `mojo-bounds-verifier` branch) proves `List`, `Span`, `String` and
similar indices in bounds with an SMT solver, after elaboration. Contracts are
written with stdlib helpers in the function body today:

```mojo
def _realloc(mut self, new_capacity: Int):
    var old_len = len(self)
    ...
    _ensures(len(self) == old_len)
    _ensures(self._capacity == new_capacity)
```

This works when the verifier reads the callee's body anyway, which it does:
for every call it evaluates the callee in the caller's context. That is the
main cost of verification. Before the pass learned to encode heap values
lazily, functions in `Mojo/stdlib/test/collections/test_dict.mojo` reached
around 226,000 SMT definitions each, 70% of them from unrolling loops inside
callee bodies. Lazy encoding cut that down, but every call still opens its
callee.

The obvious fix is Dafny's model: a call is checked against the callee's
contract, and the callee's body is checked against the same contract, once.
We tried that with the existing helpers, in an experimental `modular` mode of
the pass that does not open callees with contracts, and it fails for a
structural reason. The contracts are ordinary code in the body,
and elaboration optimizes them together with it. `mut self` is passed as a
struct and returned as a struct, so inside `_realloc` the clause
`len(self) == old_len` compares a value with itself, and both clauses fold to
the constant `true`:

```mlir
kgen.obligation "ensures"(%true) ...         // len(self) == old_len
kgen.obligation "ensures"(%true) ...         // _capacity == new_capacity
%46 = kgen.struct.create(%5, %1, %arg1) ...  // the facts now live only here
hlcf.return %46
```

With bodies closed, `test_list.mojo` drops from 963 of 963 obligations proven to
483 of 693 (the totals differ because obligations implied by earlier ones are
not counted): the contracts either say too little or were folded away. A
contract has to be a specification separate from the body, over the arguments
and results, that the optimizer cannot fold into the body. A place in the
signature gives it that, and shows it to the reader of the signature too.

## Syntax

A function declaration may end with contract clauses, after its result type
and its trailing `where` clauses, and before the `:`:

```mojo
def name[params](args) effects -> Result
    where ...        # parameters: checked by the parser
    requires ...     # preconditions: proven at each call
    ensures ...:     # postconditions: proven at each return
```

The order is fixed: `where` clauses, then `requires` clauses, then `ensures`
clauses. It follows the order in which the clauses matter: when the call is
type checked, when it happens, and when it returns. A clause out of order is
an error.

`requires` and `ensures` are soft keywords: they are only recognized at this
place, and stay ordinary names everywhere else.

The grammar of a clause is the one `where` clauses already have:

- A clause takes a boolean expression. Several clauses of one kind are all
  required, like one clause joining them with `and`:

  ```mojo
  def f(a: Int)
      requires a >= 0
      requires a < 100: ...
  ```

  One fact per clause gives the most precise reports: the verifier names the
  clause it could not prove.

- A clause may carry a message, in either spelling `where` clauses accept:

  ```mojo
  def f(a: Int) requires a >= 0 else "a must not be negative": ...
  def g(a: Int) requires (a >= 0, "a must not be negative"): ...
  ```

- The expression may use every argument of the function and its parameters.
  It may call functions (`len(xs)`), which must not have side effects (see
  [Contract expressions](#contract-expressions)).

- A clause may start a new line, indented under the `def`. The formatter
  writes a contract that does not fit on the line of the signature one clause
  per line, with the function's `where` clauses lined up with them.

Trait methods may have contracts too, including required methods:

```mojo
trait Sized:
    def __len__(self) -> Int
        ensures result >= 0: ...
```

A trait method's contract is what callers of the trait rely on. An
implementation may require less and ensure more, and the verifier checks
that it does (see [modular verification](modular-verification.md)). Trailing
`where` clauses stay unsupported on trait methods.

Function types have no contracts: a contract belongs to a declaration.

## Semantics

| Clause     | States                        | Checked         | Assumed        |
|------------|-------------------------------|-----------------|----------------|
| `requires` | the values on entry           | at the call     | in the body    |
| `ensures`  | the values on a normal return | at every return | after the call |

**In a `requires` clause** every name means the value on entry. A `mut`
argument is the value the caller passes in. An `out` argument has no value
yet, so a `requires` clause cannot use it, and it cannot use the function's
result.

**In an `ensures` clause** every name means the value on exit:

- A `mut` argument is the value the caller gets back. `old(e)` is the value
  of `e` on entry, for any expression `e`. `old` is only recognized inside
  `ensures` clauses; in a `requires` clause it is an error, since there is
  nothing older than the entry.
- An `out` argument is the value the function stored in it.
- `result` is the function's result. It names the value of a `-> T`
  function, and is another name for a named `out` result (`out self` in a
  constructor, for example). `result` is a soft keyword: an argument called
  `result` keeps its meaning, and outside `ensures` clauses the name is free.
- Read-only arguments have one value, the same on entry and exit.

```mojo
struct List[T: Copyable & Movable]:
    def __init__(out self, *, capacity: Int)
        requires capacity >= 0
        ensures len(self) == 0
        ensures self.capacity() >= capacity: ...

    def append(mut self, var value: Self.T)
        ensures len(self) == old(len(self)) + 1: ...

    # Before and after: stated twice.
    def reserve(mut self, capacity: Int)
        requires len(self) <= self.capacity()
        ensures len(self) <= self.capacity(): ...

def make_zeros(n: Int) -> List[Int]
    requires n >= 0
    ensures len(result) == n:
    return List[Int](length=n, fill=0)
```

A fact that holds before and after a call, like the one on `reserve`, is
written in both places. That is a struct invariant in disguise, and
[struct invariants](#future-work) are the better home for it.

A clause is not tied to an argument. Facts about several arguments, or about
none of them, have an obvious place:

```mojo
def gevm_kernel[..., tile_size: Int](
    c: UnsafePointer[...], a: UnsafePointer[...], n: Int32, k: Int32
)
    requires n >= 0 and c._extent() >= Int(n)
    requires k >= 0 and a._extent() >= Int(k)
    requires block_dim.x == tile_size: ...
```

For a function that raises, postconditions hold on normal returns only. A
raise establishes nothing, and preconditions still apply.

A function whose result is a reference (`-> ref T`) cannot name it in an
`ensures` clause yet.

### What is not a contract

A contract clause never takes part in overload resolution, and it is never
evaluated by the parser or the comptime interpreter. Trailing `where` clauses
keep doing both: they constrain parameters and select overloads at parse time
(see [`where_clauses.md`](where_clauses.md)). The two are different tools
with different keywords:

| Clause     | About              | Decided by   | A failure is                  |
|------------|--------------------|--------------|-------------------------------|
| `where`    | parameters         | the parser   | an error, or another overload |
| `requires` | arguments on entry | the verifier | a warning at the call         |
| `ensures`  | arguments on exit  | the verifier | a warning in the function     |

A `where` clause on an argument is an error, with a note that points to
`requires` and `ensures`.

### Contract expressions

Contract expressions are not executed by default (see
[Runtime checking](#runtime-checking)). The verifier reads them as
specifications, so they must be pure: calls may only take their arguments as
`imm` or by value, must not raise, and must not write memory other than
their own locals. The parser can enforce the first two; the third is part of
what the verifier assumes, like the other stdlib trust assumptions.

Calls inside a contract are evaluated transparently: the verifier reads the
body of `len`, `capacity` or a user predicate like `is_sorted(xs)`. Such
functions are small, and most of them are `@always_inline` already.

Inside a contract, `all([cond for i in range(lo, hi)])` is a quantifier: the
condition holds for every index in the range.

## Representation after parsing

Clauses lower into verification-only ops, placed in the function body, whose
regions are isolated from it:

```mlir
kgen.func @get(%xs: !List, %i: index) -> index {
  kgen.requires at(loc) (%xs, %i) {
  ^bb0(%a: !List, %b: index):
    ...                                  // 0 <= b and b < len(a)
    kgen.contract.yield %cond
  }
  ...
}
```

- **`kgen.requires`** sits at the start of the body. Its operands are the
  function arguments the clause uses; its region's block arguments stand for
  them. One op per clause, so a failure points at its clause.
- **`kgen.ensures`** sits right before every `hlcf.return`. Its operands are
  the exit values: the values the function returns (after elaboration these
  include `out` arguments and `mut` arguments passed as values), and the
  `mut` arguments passed as pointers, whose memory the region reads at that
  point.
- **`kgen.old`** sits at the start of the body when a postcondition uses
  `old(e)`: its region computes `e` from the arguments, and its result is an
  operand of the `kgen.ensures` ops. For a `mut` argument passed as a value,
  `old(len(self))` needs no snapshot: the region reads the argument itself.

The ops are the same whatever the surface syntax is. Moving from argument
clauses to `requires` and `ensures` changed the parser and no analysis.

The regions are `IsolatedFromAbove`, and the ops do not implement any region
interface, so no pass forwards body values into them or folds them against
the body. Within a region, folding is fine: it only simplifies the contract
itself. This is what keeps `len(self) == old(len(self))` meaningful after
elaboration: the two sides are different block arguments of the region, not
one SSA value.

Like `kgen.obligation` today, the ops have a write effect on a
verification-only memory resource, so dead code elimination keeps them, and
`LowerKGENToLLVM` erases them. Elaboration specializes them with the body
they live in, so every specialization carries its contract.

Inlining copies the ops with the body. An inlined `kgen.requires` becomes an
obligation of the caller, and an inlined `kgen.ensures` becomes one too,
checked on the inlined code. That is how `_requires` and `_ensures` behave
today, and it suits the always-inlined accessors, whose bodies are the
cheapest possible proof.

## Verification

The pass handles a function with contracts as follows.

**Checking the body, once per specialization:**

1. Assume every `kgen.requires`, with the region's block arguments bound to
   the function's arguments.
2. Encode the body as today.
3. At every `kgen.ensures`, add an obligation: the region's condition with
   its block arguments bound to the exit values and the `kgen.old` results.

**Checking a call to a callee with contracts,** without opening the callee:

1. Every callee `kgen.requires`, with its block arguments bound to the call's
   operands, is an obligation of the caller, reported at the call.
2. The call's results, and the memory of `mut` arguments passed as pointers,
   become fresh values after the call.
3. Every callee `kgen.ensures` is assumed, with the returned values bound to
   the call's results, `kgen.old` evaluated on the call's operands before the
   call, and memory reads in its region reading the memory after the call.

The experimental `modular` mode of the pass (an option, off by default) already
implements step 2 and the memory side of step 3: it names each place's value
after an opaque call once, and shares that name between the callee's contract
and the caller's later reads. With contract regions it would evaluate a
one-expression region instead of a body.

A function without contracts is handled as today: its body is evaluated at
each call. This keeps existing code working and lets contracts be adopted one
function at a time, starting with the hot ones in the standard library.

### Frames

A `mut` argument's value after a call is unknown except for what the callee's
clauses say. Dafny asks for a `modifies` clause here; Mojo's argument
conventions already provide most of it, because only `mut` arguments can
change. What is left is which fields of a `mut` argument change. The verifier
can derive this cheaply and safely from the callee's body: a field the body
returns unchanged from its input is unchanged. For `pop`, that keeps the
caller's knowledge of `self.capacity()` and the data pointer without a clause
saying so. Such a derived frame is still a summary of the body. It depends
only on how the body passes values through, not on what it computes.

### Reporting

The verifier reports as today: each obligation is proven, refuted with a
counterexample, or undecided. An unproven precondition is reported at the
call, with the clause's message and location:

```text
test.mojo:12:15: warning: cannot prove 'i < len(xs)'
test.mojo:3:5: note: required by this 'requires' clause of 'get'
```

These are warnings: verification is opt-in (see
[Adoption](#adoption)).

## Runtime checking

The verification ops are erased and cost nothing. Besides them, the parser
emits each clause as an ordinary `debug_assert` with the clause's message: at
the start of the body for preconditions, and before every return for
postconditions, with `old(e)` computed into a local on entry. These follow
the existing assert modes, so they are off by default and on under
`-D ASSERT=all`, which is how the stdlib tests run. That makes contracts
useful before any proof: tests exercise them. The verifier does not treat
these asserts as obligations; the contract ops are what it checks.

A proven obligation needs no runtime check, which is the eventual payoff for
bounds checks: `check_bounds` could be dropped where the verifier proved it.

## Adoption

- Contracts are opt-in per function, and verification is opt-in per build
  (`--bounds-check-report` today).
- The `_requires`, `_ensures` and `_assume` helpers stay for facts inside a
  body. The contracts they state on `List` today (`_realloc`, `append`, `pop`,
  `extend`, `__len__`) move to signatures.
- `_requires` currently finds out whether it is the function's own
  precondition by whether its call location resolved during elaboration. That
  heuristic goes away: a precondition is a `kgen.requires` op.
- Contracts become part of a function's documented interface, and the doc
  generator can show them in signatures.

## Why this helps interactive verification

With contracts, verifying a function depends on its body and its callees'
contracts, not their bodies. Editing a callee's body re-verifies only that
callee, as long as its contract stays the same. Each function's formula stays
small, and a cache keyed by the function's IR and its callees' contracts
turns most re-verification into a lookup. That is the property that makes
Dafny usable while typing.

## Implementation status

Implemented, on the `mojo-bounds-verifier` branch:

- `kgen.requires`, `kgen.ensures` and `kgen.old`, with the terminator
  `kgen.contract.yield`, kept through the LIT lowering, elaboration and SCCP,
  and erased when lowering to LLVM. `arg-promotion` treats their uses of an
  argument as reads, so contracts do not keep `mut` arguments in memory.
- The parser accepts `requires` and `ensures` clauses on function
  declarations, trait methods included, with the semantics above:
  preconditions become `kgen.requires` at the start of the body,
  postconditions a `kgen.ensures` before every return, and the `old(e)`
  values they use a `kgen.old` at the start. A `where` clause on an argument
  is an error that points to the new clauses.
- `result` in an `ensures` clause. A named `out` result and a result in
  memory are what the clause reads. An unnamed result in a register goes
  through a local of the function at each return, which the clause reads, so
  the analyses see the same shape as for a named `out` result.
- The formatter (`mblack`) knows the clauses: it reads them from
  continuation lines, and writes a contract that does not fit on the
  signature's line one clause per line.
- `bounds-check-report` assumes a function's own preconditions and proves its
  postconditions at every return; it checks a callee's preconditions at each
  call (inlined or not) and assumes its postconditions after the call. The
  examples are in `Mojo/test/kgen/bounds-check-report/where_clauses.mojo`
  (the file keeps its name from the first design).
- `verify-contracts` checks the same contracts before elaboration, one
  function at a time (see [modular verification](modular-verification.md)).
- The contract ops cost nothing at runtime: SROA and mem2reg give a contract
  op that reads a stack slot a snapshot of it, so the slot stays promotable,
  and inlining leaves contract ops (and snapshots) out of its size estimates.
- The contracts of the standard library, of `max.gpu` and of the kernels are
  written as `requires` and `ensures` clauses. Those of `List.append`, `pop`,
  `pop(i)` and `_realloc` also state which elements they keep or move, with
  `_same_elements(dst, src, count)` (a `kgen.contract.same_elements`): the
  `count` elements at `dst` on exit are those at `src` on entry. It compares
  bits, so it states nothing for element types whose move is not trivial.
- The experimental `modular=contracts` mode of `bounds-check-report` checks a
  call to a callee with contracts against them alone, without opening its
  body.

Differences from the design above, found while implementing it:

- The regions end in their own terminator, `kgen.contract.yield`, not a reuse
  of `hlcf.yield`: HLCF analyses take every `hlcf.yield` to end a region of an
  HLCF op, and crash on anything else.
- The regions are not `IsolatedFromAbove`. Canonicalization hoists constants
  out of them into the function body, and that is harmless: only the
  function's arguments must not be used directly, and the parser binds their
  names to the regions' block arguments. Each op takes all the function's
  arguments as operands rather than only the ones its clause uses.
- To tell a function's own contract from an inlined callee's, the ops carry
  the location of the call to the function (`kgen.source_loc[0]`), which only
  resolves once the function is inlined. That is the same mechanism
  `_requires` uses; the [Adoption](#adoption) section expected it to go away,
  but inlining leaves no other trace.
- `old` is not a keyword: a call `old(e)` is recognized while an `ensures`
  clause is emitted, and rejected while a `requires` clause is. The parser
  emits an `ensures` clause at the start of the body once to record the
  `old(e)` values, and then before every return.
- The comptime interpreter evaluates `kgen.old`, since it runs functions such
  as `List.append` at compile time. The parser therefore keeps only the
  computation of the `old(e)` values in its region: it emits each one at the
  top of the region, where the region can yield it even if the call sits in a
  short-circuit arm, and drops the rest of the condition.
- Postconditions are only assumed after calls to callees with a single
  return, and the analysis still reads the callee's body as well. A callee
  that breaks its postcondition then makes the rest of its caller unreachable;
  the broken postcondition is reported in the callee.
- The parser does not emit runtime `debug_assert`s for the clauses yet.
- The doc generator does not show contracts in signatures yet.

Findings from using the contracts:

- Closing callee bodies (`modular=contracts`) on `test_list.mojo` proves 927
  of 950 obligations instead of 977 of 977, in about the same time: `List`'s
  methods are mostly inlined, and `_realloc` is small. Stating that
  `_realloc` keeps the elements (`_same_elements`) did not recover the lost
  proofs: they are lengths of lists built from slices and spans, which the
  closed callees' contracts do not state. Contracts only replace a body when
  they say everything callers rely on.
- Proving `_same_elements` for every inlined `append` made the analysis
  treat an unknown memory state (at the function's entry, at the start of a
  loop iteration, after a write it cannot follow) as one uninterpreted
  function of the address, so that two reads of one address through
  different pointer terms get the same value. Its first form, pairwise
  equalities between such reads, grew quadratically and produced a 16 GB
  solver script.

## Why not clauses on arguments

The first version of this proposal, implemented and used on this branch,
attached contracts to arguments as `where` clauses:

```mojo
def get(xs: List[Int], i: Int where 0 <= i and i < len(xs)) -> Int: ...

def pop(mut self where old(len(self)) > 0
                 where len(self) == old(len(self)) - 1) -> Self.T: ...
```

The argument's convention decided what a clause meant:

| Convention                           | Clause was                         |
|--------------------------------------|------------------------------------|
| `imm`, `var`, owned                  | a precondition                     |
| `out`                                | a postcondition                    |
| `mut`, without `old`                 | a precondition and a postcondition |
| `mut`, with `old` and current values | a postcondition                    |
| `mut`, only `old`                    | a precondition                     |

It reused `where`, needed no new keyword, and got a named result from `out`
for free. In use it had these problems:

- **A reader could not tell preconditions from postconditions** without
  knowing the table. The three `mut` rows were the worst: the meaning of a
  clause changed with where `old` appeared in it.
- **Postconditions needed workarounds.** `List.clear` ensures
  `len(self) == 0`. On `mut self` that was also a precondition, so the clause
  had to be written `len(self) == 0 and old(len(self)) >= 0`: the second half
  is only there to mention `old`. `_realloc` had the same problem with
  `self._capacity == new_capacity`.
- **Preconditions on `mut` arguments read backwards.** "The list is not
  empty" was `old(len(self)) > 0`.
- **The place of a clause meant nothing.** A clause could use every
  argument, so the argument it was written on was arbitrary for a fact about
  two of them, and wrong for a fact about none: the kernels' clauses about
  `block_dim` and `grid_dim` sat on whichever argument came last.
- **One keyword had two meanings.** A trailing `where` is decided by the
  parser and selects overloads; an argument `where` was neither.
- **A postcondition needed an `out` argument.** `-> Int` had to become
  `out result: Int`, which changes how the body is written.

`requires` and `ensures` cost two soft keywords and the soft keyword
`result`. A fact about one argument is no longer written next to it; since a
clause could always mention any argument, that locality was weaker than it
looked.

Moving the code over was mechanical: 324 functions in the standard library,
`max.gpu`, the kernels and the tests. The contract ops they lower to did not
change.

## Alternatives considered

### Clauses on arguments

The first design; see
[Why not clauses on arguments](#why-not-clauses-on-arguments).

### Other names for the result

C++26 contracts name the result in the clause (`post(r: r >= 0)`), and
Dafny in the signature (`returns (r: int)`). Mojo already has the second
form as `out r: Int`, and it keeps working: an `ensures` clause may use the
name of an `out` argument. For `-> T` functions a fixed name is the smallest
addition. `result` is what Eiffel and Ada (`'Result`) use.

### Any order of clauses

Dafny lets `requires` and `ensures` clauses mix. A fixed order costs nothing
and makes every signature read the same way.

### Keeping contracts as stdlib helpers

This is today's design. As shown in [Background](#background), elaboration
folds the helpers into the body, so a call cannot be checked against the
contract alone.

### Decorators

`@requires(i < len(xs))` does not work: decorators are evaluated at compile
time, and cannot refer to runtime arguments.

## Future work

- **Refinement types.** A type that carries a fact, like a `Nat` that is an
  `Int` with `self >= 0`: an argument `n: Nat` would behave like `n: Int`
  with `requires n >= 0`. This needs its own syntax decision.
- **Struct invariants,** such as
  `0 <= self._len and self._len <= self._capacity` for `List`, assumed on
  entry to every method and checked where constructors and `mut` methods
  return. They would replace the `_assume` in `List.__len__`, and the clauses
  that today are written once as `requires` and once as `ensures`.
- **Loop invariants,** stated in the source rather than inferred.
- **Frame clauses.** A `ref` argument with a mutable origin may change, so
  its clauses must state what it keeps (`len(xs) == old(len(xs))`). Dafny's
  `modifies` and `reads` would say that once.
- **Results that are references.** `result` for `-> ref T` functions.
- **Verifier-visible `assert`.** Today `assert` lowers to a `debug_assert`
  call that the verifier cannot tell from other code. As an obligation that is
  assumed afterwards, it would let users give the verifier hints, which is how
  Dafny users fix slow proofs.

## Open questions

- Contract expressions call functions. How much purity should the parser
  enforce, and should predicates used in contracts be marked (like Dafny's
  `function`, as opposed to `method`)?
- Can elaboration keep the contract ops of a function it specializes but
  never calls? Today the verifier only sees specializations that some caller
  reaches, which is fine for callers, but a library would want its functions
  verified on their own.
- Should the formatter always write one clause per line, also when the whole
  signature fits on one line?
