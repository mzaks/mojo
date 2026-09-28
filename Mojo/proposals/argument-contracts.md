# Argument contracts: runtime `where` clauses for verification

**September 28, 2026**
Status: Draft. Preconditions and postconditions are implemented; see
[Implementation status](#implementation-status).

This document proposes contracts on function arguments, written as `where`
clauses on the arguments themselves:

```mojo
def get(xs: List[Int], i: Int where 0 <= i and i < len(xs)) -> Int:
    return xs[i]
```

A clause on an input argument is a precondition, and a clause on an `out`
argument is a postcondition. A `mut` argument's clause can talk about the
value on entry through `old(...)`. The contracts are kept through
elaboration as operations of their own, so a static verifier can check each
function body against its contract once, and check each call against the
callee's contract without looking into the callee's body. That is what makes
tools like Dafny fast, and what our current contract helpers cannot provide.

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
and results, that the optimizer cannot fold into the body.

## Syntax

A `where` clause may follow the type of any function argument, including
`self`:

```mojo
def f(a: Int where 0 <= a and a <= 100): ...

struct List[T: Copyable & Movable]:
    def pop(mut self where old(len(self)) > 0) -> Self.T: ...
```

The grammar is the one trailing `where` clauses already use:

- A clause takes a boolean expression. Several clauses on one argument chain
  with an implicit `and`:

  ```mojo
  def f(a: Int where a >= 0 where a < 100): ...
  ```

- A clause may carry a message, in either spelling `where` clauses accept:

  ```mojo
  def f(a: Int where a >= 0 else "a must not be negative"): ...
  def g(a: Int where (a >= 0, "a must not be negative")): ...
  ```

- The expression may use every argument of the function, in any order, and
  the function's parameters. It may call functions (`len(xs)`), which must not
  have side effects (see [Contract expressions](#contract-expressions)).

- Inside a clause on a `mut` argument, or on an `out` argument, `old(e)` is
  the value of `e` on entry to the function. `old` is a soft keyword that is
  only recognized inside argument `where` clauses.

The parser already parsed a `where` after an argument's type
(`Mojo/lib/MojoParser/Signatures.cpp`, `ParsedArgument::parse`) and rejected
it with "'where' clauses must be used with parameters and cannot be used with
arguments". This proposal gives that syntax a meaning for arguments. The
deprecated `where` inside parameter lists stays an error: parameters have
trailing `where` clauses.

## Semantics

A clause states a fact about the argument's value at a point of the call. The
argument convention decides which point:

| Convention                           | Clause                           | Checked                         | Assumed                        |
|--------------------------------------|----------------------------------|---------------------------------|--------------------------------|
| `imm`, `var`, owned, `deinit`        | on entry                         | at the call                     | in the body                    |
| `out`                                | on exit                          | at every return                 | after the call                 |
| `mut`, without `old`                 | on entry and exit                | at the call and at every return | in the body and after the call |
| `mut`, with `old` and current values | on exit, relating entry and exit | at every return                 | after the call                 |
| `mut`, only `old`                    | on entry                         | at the call                     | in the body                    |

A mutable `ref` argument follows `mut`, and an immutable one follows `imm`.

The three `mut` rows cover the facts the standard library needs:

```mojo
struct List[T: Copyable & Movable]:
    # Before and after: holds on entry and on exit.
    def reserve(mut self where len(self) <= self.capacity(),
                capacity: Int): ...

    # A relation between entry and exit.
    def append(mut self where len(self) == old(len(self)) + 1,
               var value: Self.T): ...

    # Only before: after `pop`, the list may be empty.
    def pop(mut self where old(len(self)) > 0
                     where len(self) == old(len(self)) - 1) -> Self.T: ...
```

An `out` argument names the result, so postconditions need no special name
for it:

```mojo
def make_zeros(n: Int where n >= 0,
               out result: List[Int] where len(result) == n):
    result = List[Int](length=n, fill=0)

struct List[T: Copyable & Movable]:
    def __init__(out self where len(self) == 0
                          where self.capacity() >= capacity,
                 *, capacity: Int where capacity >= 0):
        ...
```

A function declared with `-> T` has no name for its result. It has to use an
`out` argument to state a postcondition about it. This keeps the first version
small; a named `->` result can be added later.

A clause on an `out` argument may use the other arguments: read-only ones have
one value, and a `mut` argument means its value on exit unless wrapped in
`old`.

For a function that raises, postconditions hold on normal returns only. A
raise establishes nothing, and preconditions still apply.

### What is not a contract

An argument clause never takes part in overload resolution, and it is never
evaluated by the parser or the comptime interpreter. Trailing `where` clauses
keep doing both: they constrain parameters and select overloads at parse time
(see [`where_clauses.md`](where_clauses.md)). A clause on an argument that
mentions only parameters is almost certainly a misplaced trailing clause and
is diagnosed like the parameter-list form today:

```text
error: this 'where' clause only uses parameters
note: use a trailing 'where' clause after the signature instead
```

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

## Representation after parsing

Clauses lower into verification-only ops, placed in the function body, whose
regions are isolated from it:

```mlir
kgen.func @get(%xs: !List, %i: index) -> index {
  kgen.requires "where" at(loc) (%xs, %i) {
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
test.mojo:3:26: note: required by this 'where' clause of 'get'
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
- The parser accepts clauses on every argument but `ref` and `deinit` ones,
  with the semantics of the table above: preconditions become `kgen.requires`
  at the start of the body, postconditions a `kgen.ensures` before every
  return, and the `old(e)` values they use a `kgen.old` at the start.
- `bounds-check-report` assumes a function's own preconditions and proves its
  postconditions at every return; it checks a callee's preconditions at each
  call (inlined or not) and assumes its postconditions after the call. The
  examples are in `Mojo/test/kgen/bounds-check-report/where_clauses.mojo`.

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
- `old` is not a keyword: a call `old(e)` is recognized while a clause on a
  `mut` or `out` argument is emitted, and nowhere else. The parser emits such
  a clause at the start of the body once to record the `old(e)` values, and
  once more to see whether it uses any `mut` or `out` argument's value on
  exit, which decides between precondition and postcondition.
- The comptime interpreter cannot evaluate `kgen.old`: a function whose
  postcondition uses `old` cannot run at compile time.
- Postconditions are only assumed after calls to callees with a single
  return, and the analysis still reads the callee's body as well. A callee
  that breaks its postcondition then makes the rest of its caller unreachable;
  the broken postcondition is reported in the callee.
- The parser does not emit runtime `debug_assert`s for the clauses yet.
- Not implemented yet: checking a call without opening the callee's body.

## Alternatives considered

### Trailing `requires` and `ensures` clauses

```mojo
def get(xs: List[Int], i: Int) -> Int
    requires 0 <= i and i < len(xs)
    ensures result >= 0:
```

This is Dafny's spelling, and close to C++26 contracts (`pre(...)`,
`post(r: ...)`) and Ada's `Pre` and `Post` aspects. It needs two new keywords
and a reserved name for the result, and it separates a fact about `i` from
`i`. Argument clauses reuse `where`, put the fact next to the argument, and
get a named result from `out` for free. The cost is the `old`-only form for
preconditions on `mut` arguments, which reads less directly than a separate
`requires`.

### A function-level clause for `mut` preconditions

Instead of `mut self where old(len(self)) > 0`, preconditions on `mut`
arguments could go into a trailing clause. That clause would look like a
trailing `where`, but mean something else: it would be a runtime contract
rather than a parse-time constraint. Two meanings for one trailing form seemed
worse than the `old`-only rule.

### Keeping contracts as stdlib helpers

This is today's design. As shown in [Background](#background), elaboration
folds the helpers into the body, so a call cannot be checked against the
contract alone.

### Decorators

`@requires(i < len(xs))` does not work: decorators are evaluated at compile
time, and cannot refer to runtime arguments.

## Future work

- **Refinement types.** The same clause on a type alias would give types like
  `comptime Nat = Int where self >= 0`, with `self` meaning the value. An
  argument `n: Nat` would behave like `n: Int where n >= 0`. A trailing
  `where` on an alias already constrains the alias's parameters, so this needs
  its own syntax decision.
- **Struct invariants,** such as
  `0 <= self._len and self._len <= self._capacity` for `List`, assumed on
  entry to every method and checked where constructors and `mut` methods
  return. They would replace the `_assume` in `List.__len__`.
- **Loop invariants,** stated in the source rather than inferred.
- **Named `->` results,** as in `-> (r: Int where r >= 0)`.
- **Verifier-visible `assert`.** Today `assert` lowers to a `debug_assert`
  call that the verifier cannot tell from other code. As an obligation that is
  assumed afterwards, it would let users give the verifier hints, which is how
  Dafny users fix slow proofs.

## Open questions

- Should a `mut` clause without `old` really hold on entry too? It does for
  invariant-like facts (`len(self) <= capacity`), but a user who meant only
  the exit state gets a precondition they did not intend. A warning when a
  `mut` clause without `old` is never needed on entry would catch this, but
  needs the verifier.
- Contract expressions call functions. How much purity should the parser
  enforce, and should predicates used in contracts be marked (like Dafny's
  `function`, as opposed to `method`)?
- Can elaboration keep the contract ops of a function it specializes but
  never calls? Today the verifier only sees specializations that some caller
  reaches, which is fine for callers, but a library would want its functions
  verified on their own.
