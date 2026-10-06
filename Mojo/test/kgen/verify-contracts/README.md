# verify-contracts examples

Examples for the `verify-contracts` pass
(`Mojo/lib/Transforms/VerifyContracts.cpp`), which checks `where` contracts
right after lifetime checking, before elaboration and inlining. See
`Mojo/proposals/modular-verification.md`.

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

`tensors.mojo` and `kernels.mojo` import the `max` and `layout` packages:
build them and add them to the import path. `gemv_split_k.mojo` launches a
kernel of `linalg`: it needs every kernel package on the import path and
`packages=linalg`, and is compiled too (`-elaborate` without
`--verify-contracts`), which checks its `comptime assert`s.

```bash
./bazelw build --config=build-mojo //max/kernels/src/layout:layout
bazel-bin/Mojo/tools/kgen/kgen -I bazel-bin/Mojo/stdlib/std \
    -I bazel-bin/max/mojo/max -I bazel-bin/max/kernels/src/layout \
    Mojo/test/kgen/verify-contracts/tensors.mojo \
    -elaborate --verify-contracts="verbose=true"
```

Functions from imported packages (the stdlib, `layout`, ...) are not
verified unless `include-stdlib=true`, or `packages=nn,linalg` for the
named top-level packages. Only what the file uses is in the module, so
to verify a package's own kernels, run each of its source files as the
main file (relative imports made absolute).

With `-lsp=no-dump` instead of `-elaborate`, it runs on the module the
language server checks. That parse is lazy: a stdlib function whose body
was not needed has no body there, and so no contracts, and calls to it are
not checked.

Options: `verbose=true` reports proven obligations as remarks,
`include-stdlib=true` also checks `std`, `packages=` the named imported
packages, `rlimit=` sets the solver's
deterministic resource limit of each query (default 100000000),
`generic-launched=true` also verifies launched generic kernels for every
value of their parameters, `generic-rlimit=` the limit of each such query
(default 10000000; see below), `houdini-rlimit=` the limit of each
loop-invariant candidate's query (default 1000000, at most a twentieth
of the query limit; a candidate not decided within it is dropped),
`wall-seconds=` caps each z3
process (default 60), `check-division=true` reports integer divisions
by a value that may be 0, `int-retry=false` turns off the integer retry
(below), `nonlinear-rlimit=` the limit a query whose goal multiplies or
divides two unknowns is first asked with (default 10000000; then over the
integers, then with the full limit), `case-split=false` turns off case
splitting (below), `invariant-retry=true` asks a loop-invariant candidate
left undecided once more with four times its limit (a loop after
a large body can need it; off by default, since on kernels without
clauses it costs about half again as much), `dump-dir=` writes the
SMT-LIB scripts (the directory
is created; a script the same as one already run is neither run nor
written again), and `cache-dir=` caches
the solver's answers by a hash of each script.

Values are bit-vectors, as in the program, so wrap-around is modelled.
z3 decides few nonlinear bit-vector facts (`row * k + col < m * k` from
`row < m` and `col < k` is `unknown` after 13 s), so a query answered
`unknown` is asked again over the integers: each `w`-bit value is its
signed value, arithmetic wraps by `mod 2^w`, unsigned operations use the
unsigned value, division keeps SMT-LIB's results for a zero divisor, and
shifts, masks, extracts, extensions and concatenations by constants are
arithmetic; other bit operations become unknowns in range. The
translation is exact or weaker, so only its proofs are taken (the same
query: `unsat` in 0.15 s). A query with a product in its goal that is
still open is asked once more with exact products: the wrapping is what
keeps the solver from `(i * 16 + w) * n + col < k * n`, which takes a
bound on the row first. For each product of two unknowns, and each sum
over one, it is first asked whether its exact value is in range under
the query's assumptions (within 0.5 s each, at most 16 terms: those the
goal depends on, and the products of plain terms, an extent's `k * n`);
then the query is asked with the terms that are in range defined without
`mod 2^w` (within 5 s), in each case of the finite-domain unknowns. That
is the same query: where the assumptions hold, those terms have their
exact values. Both steps have time limits, so a loaded machine can leave
such a query open. `int-translate-file=F` writes the translation
of the SMT-LIB file `F` to `F.int.smt2` and does nothing else (for
testing).

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
function warned about, none in an `ok_*` one; in `traits.mojo`, also every
`Bad*` struct):

```bash
f=Mojo/test/kgen/verify-contracts/straight_line.mojo
bazel-bin/Mojo/tools/kgen/kgen -I bazel-bin/Mojo/stdlib/std $f -elaborate \
    --verify-contracts 2>&1 | grep "warning:" |
  grep -oE "^[^ ]*$(basename $f):[0-9]+" |
  cut -d: -f2 | while read l; do
    awk -v L=$l 'NR<=L && /^(def|struct) /{n=$2} NR==L{print n}' $f |
      cut -d'(' -f1 | cut -d'[' -f1
  done | sort -u
```

The output must be exactly the `bad_*` functions (and `Bad*` structs).

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
- Iterating a `List`, `Span`, `Array` or `Deque` (`for x in xs`, and
  `for i, x in enumerate(xs)`), or consuming a `List` or `Array`
  (`for x in xs^`, `for x in [1, 2, 3]`), yields exactly `len(xs)` times:
  the iterator holds a cursor and the length it started with, and
  `enumerate`'s count is the cursor (plus `start`). It yields element
  `cursor` of the collection: a reference to its place when borrowing, its
  value when consuming. The elements `enumerate` yields are unknown, and
  reversed iteration over collections is not modelled.
- Iterating a `LinkedList` (`for x in l`, `l.__reversed__()`,
  `enumerate(l)`, and consuming it with `for x in l^`) yields exactly
  `len(l)` times, element `cursor` (from the end when reversed), as
  `l.get_nth(i)` reads it. The borrowing iterator follows the nodes until
  there are none, so this assumes the nodes are as many as the list's
  size, which its methods keep.
- Iterating a `Dict` (`for k in d`, `d.keys()`, `d.values()`,
  `d.items()`, and `reversed` of `d` or of its values or items) yields
  exactly `len(d)` times: its iterators count the entries they have seen,
  skipping removed ones, in either direction. The keys and values are
  unknown. `List(it)` of such an iterator (or of a list's) is as long as
  the iterator has entries left. Consuming a `Dict` (`for k in d^`) is
  not modelled.
- Control flow: `if` (with `elif`), `return`, `try`, and loops (`for` over
  `reversed(range(n))`, `reversed(range(start, end))`, `range(n)` and
  `range(start, end)`, `while`, `break`). `range` iteration follows the
  stdlib's definition. At each loop head, what the loop may write is unknown,
  bound by invariants found with Houdini over small templates: bounds
  against 0, against the values before the loop and lengths, and between the
  loop's variables, and for a list the loop changes, that its length plus or
  minus a loop variable keeps its value on entry (`len(xs) + i` when each
  step pops once). Only variables the loop's conditions depend on are
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
  anything else about a parameter is unknown. A function's own `where`
  constraints on its parameters (`def f[n: Int](...) where n < 8`) hold
  in its body, since the compiler rejects any instantiation that breaks
  them; a `comptime for k in range(n)` has `0 <= k < n`.

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
  negative infinity, as Mojo defines them, and are 0 for a zero divisor
  (what `SIMD` computes; undocumented). With `check-division=true` each
  `//`, `%`, `divmod`, `ceildiv` and `uutils` helper whose divisor is not
  a nonzero literal is an obligation ("that the divisor is not 0").

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
- Nested collections (`xs[i][j]`, `xs[i].append(v)`, `l.get_nth(i).get_nth(j)`):
  a reference to an element is a place whose value is the element; writing it
  (directly, through a field, or by a call that takes it `mut`) writes the
  element back into the collection, which keeps its length. A list or array
  literal's elements are the values it is given. A call given an interior origin
  (`xs["element"]`) may change the collection's elements but not its length.
- `x.copy()` (the `Copyable` default, `Self(copy=self)`) states what the
  struct's copy constructor states: a copied list, deque, linked list or
  dictionary is as long as the original.
- Element access through `ref self` (`List`, `Deque` and `Array`
  `__getitem__`, `LinkedList.get_nth`) keeps the collection: it reads an
  element and does not write the collection, although its origin may be
  mutable.

- Traits: a trait method's `where` clauses (on a required method, `...`,
  as on a default) are its contract. A generic call (`t.get(i)` on a
  `T: Counter`) is held to the trait's precondition and assumes its
  postcondition, and so is a call of an inherited default through a
  struct (`f.pick(i)`). Every implementation is checked against the trait:
  the trait's precondition must imply the implementation's own, and the
  implementation must establish the trait's postcondition where it
  returns (with the trait's other methods on `Self` being the struct's
  own, and their own clauses). A trait method that only reads its
  arguments and returns an integer or a Boolean (`count()`) is a function
  of their values, one per method and type: calls on unchanged arguments
  agree, the same assumption as for `len(x)`. Implementations in the
  stdlib, and of structs with parameters, are not linked this way.
- `Sized.__len__` states `result >= 0`, so a generic `x.__len__()` is not
  negative, and it is `len(x)`: the same uninterpreted function. Every
  `Sized` struct outside the stdlib is checked against it (a `__len__`
  that returns an `Int` field as it is does not prove); the stdlib's own
  implementations only with `include-stdlib=true`.
- `Iterator.bounds()` states `lower >= 0` and, with an upper bound,
  `lower <= upper`; `Iterator.nth(n)` requires `n >= 0`. A tuple literal
  (`(n, None)`) is modelled field by field, and a value a modelled
  constructor built keeps its fields when it is moved (returned, stored).
  Nothing is stated about when `__next__` raises: `bounds()` is only a
  hint.
- `TileTensor` (from MAX's `layout` package): `t[i]` (rank 1), `t[i, j]`
  and writes `t[i, ...] = v` require each index in `[0, dim[k])`.
  `dim[k]()` is the layout's static size where its shape says
  `ComptimeInt[n]`, and otherwise one non-negative unknown per tensor and
  dimension. Tuple coordinates and nested layouts are not checked.
- `t.tile[*sizes](*coords)` requires the whole tile inside the tensor
  (tiles are not clipped): `0 <= c < dim // size` per dimension. The
  tile's dimensions are its static sizes. `t.vectorize[*sizes]()`
  requires every dimension a multiple of its size (the view has
  `ceildiv(dim, size)` vectors, so the last one would reach past the
  tensor), and the view's `dim[k]()` is `ceildiv(dim[k], size)` of its
  parent. `t.load[width](coord)` and `t.store[width](coord, v)` require
  every integer coordinate in `[0, dim)` and, for a width above one, the
  last at most `dim - width` with a static stride of 1 (`j + 4 <= dim`
  does not prove it: the sum wraps for a large `j`; `j <= dim - 4` does).
  `Coord(i, j)`, `Coord(Idx[3], j)` and a tuple `(i, j)` are modelled
  element by element. A nested layout, a coordinate of another rank and a
  vectorized view are not checked. `tile` with `Coord` coordinates
  (`tile[2, 4](Coord(i, j))`, `tile(coord[2, 4], Coord(i, j))`, also
  with a runtime shape) and `tile_with_offset` have the same contract as
  `tile[2, 4](i, j)`. `t.distribute[thread_layout](tid)` needs no
  contract: each thread starts at `(tid // stride) % threads`, in
  `[0, threads)`, and gets `dim // threads` elements `threads` apart, so
  its view stays within the tensor for any `tid` (a swizzle, which remaps
  the offset, is assumed to as well). The view's `dim[k]()` is
  `dim[k] // threads[k]` of its parent, also when the thread layout's
  sizes are parameters (`row_major[TM, TN]()`), and so is that of the
  view `distribute_with_offset` returns first in its tuple.
  `t.num_elements()` is the product of `t`'s dimensions, for a generic
  layout by the rank a `comptime assert t.rank == r` states (r up to 4);
  `t.to_device_buffer(ctx)` states that the buffer holds
  `t.num_elements()` elements, and a mutable tensor converted to an
  immutable one is the same tensor. `GemmShape.get(c, a, b)` states
  `M`, `N` and `K` as `c`'s and `a`'s dimensions.
- GPU kernels: `thread_idx`, `block_idx`, `block_dim` and `grid_dim` are
  one value per axis in a function, with the launch limits every
  supported GPU has (assumptions): `0 <= thread_idx < block_dim <= 1024`
  and `0 <= block_idx < grid_dim < 2^31`; `global_idx` is
  `block_idx * block_dim + thread_idx`; `lane_id()` is in `[0, 64)`
  (warps have 32 or 64 lanes). `ufloordiv`, `udiv_unchecked` and
  `uceildiv` are bounded unsigned divisions (`udiv_unchecked` and
  `udivmod_unchecked` require `b > 0`: by 0 they are undefined);
  `warp.broadcast(x)` and the
  unmasked `shuffle_idx/up/down/xor` are `x` as some thread of the same
  block computes it: `x`'s term with thread ids, `lane_id`, arguments and
  bounded quotients renamed to fresh copies (with their facts), so a
  bound every thread has holds for it, but it is not this thread's `x`;
  if `x` depends on a load or an unknown result, it is an unknown. The compilation target is modelled as `std.sys.info`
  describes it: at most one of NVIDIA's, AMD's (RDNA among them) and
  Apple's GPU triples, and the build's accelerator, if any, of one
  vendor; `is_gpu()`, `has_*_accelerator()` and `WARP_SIZE` (32 or 64 on
  a GPU, 0 on a host without an accelerator, the accelerator's otherwise)
  follow. `comptime assert c` is assumed from there on (the compiler
  checks it wherever the code is compiled), unless `c` is nonlinear
  (products of parameters; assuming those made every query slow), as
  are a function's `where` constraints, so a function that states
  `is_gpu()` knows `WARP_SIZE`. A kernel's top-level `comptime assert`s
  that name the target are its hardware contract: at each launch they
  must hold for the build's accelerator
  (assumptions: a kernel is compiled for the accelerator the build names,
  `--target-accelerator`, and the launcher runs on the host), which the
  launcher states with `comptime if has_nvidia_gpu_accelerator():` and the
  like; `has_accelerator()` is not enough (an accelerator name the stdlib
  does not recognize has no known GPU triple). Kernels are verified like any
  function, on the host, before they are compiled for a GPU.
- A kernel states the launch it relies on as `where` clauses on its
  arguments (`grid_dim.y <= Int(c.dim[0]()) // 16`, `block_dim.x ==
  256`), which are assumed in its body. At a launch,
  `ctx.enqueue_function[kernel](args..., grid_dim=g, block_dim=b)`, they
  are obligations, with the kernel's arguments the launch's and its
  `grid_dim` and `block_dim` the launch's `Dim`s (from `Int`s, literals
  or tuples; an omitted axis is 1). Other GPU ids in a clause are unknown
  at the launch. A kernel compiled first and launched as
  `ctx.enqueue_function(ctx.compile_function[kernel](), args..., ...)`
  is checked the same way, and so is one launched with `host_arg=` (the
  host arguments are the kernel's last arguments). Kernels launched as
  closures (no arguments, so no clauses) or as precompiled external
  functions are not checked.
- A generic kernel that is launched (`enqueue_function[kernel[8]](...)`) is
  verified for each distinct launch, its parameters bound to the launch's, not
  for every value of them: a proof for every `BM` and `BN` is nonlinear and slow
  (the custom-ops tiled matmul: 10 s, against 0.12 s for its launched sizes). A
  parameter that the launch passes from its own function's parameters
  (`kernel[dtype, N]` in a launcher generic over them) is followed to the calls
  of that function that give it (`launch[DType.float32, 8](...)`), up to three
  calls out; where no call gives it, it stays unknown, so the check holds for
  every value of it. An obligation proven in all of them is reported as proven
  for the launched instantiations (and counted apart in the summary); otherwise
  the warning has a note at each launch where it is not proven (and at the call
  that gave the parameters). An instantiation that unrolls a loop fewer times
  than another has nothing to prove for the iterations it lacks. An
  instantiation with an integer parameter computed by a function outside the
  standard library (`config[1]` of `comptime config = _gemv_config[...]()`) is
  not verified, with a warning at its launch: the verifier does not evaluate
  the function, and with loop counts and tile sizes unknown next to nothing is
  proven; its launch's clauses, over the same unknown values, are not asked
  either. Launch the kernel with the values written out to verify it for them
  (`gemv_split_k.mojo`). `generic-launched=true` also verifies such a kernel
  for every value of its parameters first, each query within `generic-rlimit`
  (and each loop-invariant candidate within a twentieth of it), and checks only
  what that leaves open per launch. Kernels that are not launched in the module
  are verified for every value of their parameters.
- A write through an origin that names no variable (`MutAnyOrigin`, which
  every kernel tensor has: `c.fill(0)`, `c_ptr[m] += v`) may reach any memory
  a reference or pointer leads to, so what is known about memory is lost
  after it, and at the head of a loop that does it. Not lost: a local
  variable that is only read, assigned, borrowed immutably or passed to a
  range's own methods (a loop's index, a pointer or a size read before the
  loop), and the end and step of such a range. That an immutable borrow is
  not turned into a pointer that is written through is assumed.
- `t.ptr` of a tensor is the same pointer wherever it is read from the same
  tensor value, so a clause can state its extent (`where t.ptr._extent() >=
  n`); nothing else does.
- Raw pointers: `Pointer._extent()` is how many elements are valid from a
  pointer, for contracts only (it is not computed: `Int.MAX` at run time);
  the pass reads it as an unknown, not negative, per pointer value.
  `p[unsafe_offset=i]` (and the deprecated `p[i]`) require `0 <= i <
  _extent()`; `load[width](i)`, `store[width](i, v)`, `unsafe_load` and
  `unsafe_store` require `0 <= i <= _extent() - width`; `p[]`,
  `load[width]()` and `store(v)` the same at offset 0. It is stated by
  `alloc[T](n)` and `unsafe_alloc` (`n`), `stack_allocation[n, ...]()`
  (`n`), `Pointer(to=x)` (at least 1), `p.unsafe_offset(k)` and `p + k`
  (`k` fewer, for `k >= 0`; a pointer moved backwards has no known
  extent), `List.unsafe_ptr()`, `Span.unsafe_ptr()` and
  `Array.unsafe_ptr()` (at least the length), `DeviceBuffer.unsafe_ptr()`
  (the buffer's length, which `enqueue_create_buffer(n)` states), and by a
  function's own clauses (`n: Int where p._extent() >= n`). Casts of a
  pointer's origin or address space (the implicit mutable-to-immutable
  conversion, `as_imm()`, `unsafe_origin_cast`, ...) keep its extent;
  `unsafe_bitcast` does not. A `DeviceBuffer` passed for a kernel's
  pointer argument has its length as that pointer's extent. A pointer
  whose extent nothing states is reported. Extents are about bounds only:
  a pointer used after its memory is freed or reallocated (a list
  appended to after `unsafe_ptr()`) keeps the extent it had.
- A struct with a single integer field and no parameters (an enum-like
  wrapper: `GEMVAlgorithm`, a mode) is represented by that integer: its
  field is the value, `Wrapper(n)` (a constructor that only stores its
  argument, also as a `comptime` constant) is `n`, and its own methods
  whose bodies only read fields, call and return (`__eq__`, `__ne__`,
  `__is__`, `__isnot__`) are evaluated in place. So a clause can say
  `mode is not Mode.FAST or len(xs) > 0`, and a caller that picks the
  mode on a branch establishes it. `Bool.__bool__` is the identity.
- `simd_width_of[dtype]()` is 0 or a power of two up to 256 (assumed from
  the targets' SIMD widths). `dtype in (DType.float32, ...)` of dtype
  literals is one of the comparisons `dtype == ...`.
- A generic layout's `static_shape[k]`, when not -1, is `dim[k]()` of its
  tensors only if the layout is flat: say `comptime assert
  t.LayoutType.flat_rank == t.rank` (the layout's own `flat_rank`;
  `t.flat_rank` is expanded beyond recognition).
- A tile may extend past its tensor's edge (a partial tile, as a kernel
  takes for the last block of a row): `tile[...](*coords)` requires only
  coordinates not negative, and the tile's `_valid_dim[k]()` (contract-only)
  counts the elements memory backs, `clamp(valid - c * size, 0, size)`;
  indexing, `load`, `store` and nested tiles are checked against it, and a
  vectorized view of it holds `valid // size` vectors. A partial tile passed
  to any other function is an obligation that it is whole: that function
  was verified for a tensor every element of which is backed.
- `t.reshape(row_major(Coord(...)))`: the view's dimensions are the
  layout's shape, and it must cover no more elements than `t` validly holds
  (assumed: `t` has no zero stride). Other layouts leave them unknown.
- `x if comptime (c) else y` is the value of the taken arm.
- A local closure (`@__parameter def` in a function) that is only ever
  called is verified where it is called, with its caller's state; one
  passed on (`vectorize[f]`, a launch, an epilogue parameter) on its own.
  A closure with parameters of its own (`f[4]()`) is walked with that
  call's values; a kernel it launches is verified for them (`launch[4]()`
  launching `kernel[rows]` gives the instantiation `kernel[4]`).
- `min(a, b)` and `max(a, b)` of two integers are the smaller and the
  larger as their dtype compares (signed or unsigned); `align_down(a, b)`
  and `align_up(a, b)` are `a // b` and `ceildiv(a, b)` times `b`.
- A loop that calls a local closure changes what the closure writes.
- The index of a `comptime for` is one unknown per loop, in `[0, end)`
  over `range(end)`, also in a written element (`t[1, j] = v`) and in a
  `Coord`. A loop over a constant range of at most 8 iterations that
  changes an integer variable from outside it (a counter, also in a
  closure it calls) is walked once per iteration instead, so that what
  the iterations do adds up. A `comptime if` arm that is false by
  constants alone is unreachable.
- `range(start, end, step)`: the loop variable is the cursor, a whole
  number of steps from `start`. A loop that only advances its range
  keeps the range's end and step. A counter the body changes beside the
  range is as many steps from its value on entry as the cursor is from
  `start` (a candidate invariant).
- `load[width]` and `store[width]` with a parameter width; of a tile of a
  generic layout they need the parent's rows contiguous: say `comptime
  assert L.rank == 2 and L.flat_rank == 2 and L.static_stride[1] == 1`;
  `size_of[dtype]()` (also called in a clause) is the dtype's size in
  bytes (a power of two up to 32 where the dtype is not a literal);
  `simd_width_of[dtype, target=get_gpu_target()]()` is 16 over that size
  (every GPU target has 128-bit vectors: 0 only for a 256-bit dtype);
  `warp_id()` is `thread_idx.x` over a warp size of 32 or 64.
- `lane_id()` is below `WARP_SIZE`; the warp size of an accelerator the
  stdlib does not hard-code is 32 or 64 (every `GPUInfo` says so); a
  launched kernel's own target is a GPU; `ufloordiv(a, 0)` is 0.
- Queries left open over the warp size and SIMD widths (unknowns with few
  values) are asked once per combination of their values, as constants
  (at most 64 combinations; `case-split=false` turns this off), with
  unsigned quotients exact in those cases.
- `ceildiv(a, b)` (and `a.__ceildiv__(b)`) of integers is `-(a // -b)`
  when signed and the quotient plus one for a nonzero remainder when
  unsigned, as `SIMD` defines it; so a launch with `grid_dim=ceildiv(n,
  16)` establishes `grid_dim <= n // 16` where `n % 16 == 0` is known.
- `divmod(a, b)` of `Int`s and `udivmod(a, b)` are the quotient and
  remainder (floored, or unsigned for `udivmod`); a `comptime for k in
  range(n)` has `0 <= k < n`. In a function generic over tile sizes
  (`tile[BM, BN]`, `row_major[BM, BK]()`), the sizes are the parameters,
  so what is proven holds for every value of them.
- `Int.MAX`, `Int.MIN` and the bounds of the other integer dtypes
  (`max_or_inf`, `min_or_neg_inf`) are their values.

Not analyzed yet: loops with loop-carried values, and ranges with a step
(`range(a, b, c)`) or over other integer types. Lists built from other iterables
(the generic constructor states no length) have no length contract. The
obligations inside unsupported control flow are reported as not analyzed.
`append` does not state the value it adds (a generic `T` has no `==` to state it
with).

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
| `traits.mojo`        | all `bad_*`, `Bad*`| all `ok_*`       |
| `tensors.mojo`       | all `bad_*`        | all `ok_*`       |
| `kernels.mojo`       | all `bad_*`        | all `ok_*`       |
| `pointers.mojo`      | all `bad_*`        | all `ok_*`       |
| `gemv_split_k.mojo`  | (none)             | all `ok_*`       |
