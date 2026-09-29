# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
# Contracts written as `where` clauses on arguments, for
# `bounds-check-report`; see README.md and
# Mojo/proposals/argument-contracts.md.


# A function whose precondition proves its own index. Not inlined, so every
# call is checked against the clause.
@inline(.never)
def ok_get(xs: List[Int], i: Int where 0 <= i and i < len(xs)) -> Int:
    return xs[i]  # proven from the precondition


# A precondition on one argument may use the others, and an argument may have
# several clauses.
@inline(.never)
def ok_window(
    xs: List[Int],
    lo: Int where 0 <= lo,
    hi: Int where lo <= hi where hi < len(xs) else "hi must be an index",
) -> Int:
    return xs[hi] - xs[lo]


# Inlined into its callers: its clause becomes their obligation.
@always_inline
def ok_at(xs: List[Int], i: Int where 0 <= i and i < len(xs)) -> Int:
    return xs[i]


# --- must stay UNPROVEN ---
@inline(.never)
def bad_get_unchecked(xs: List[Int], i: Int) -> Int:
    return ok_get(xs, i)  # i may be out of range


@inline(.never)
def bad_get_past_end(xs: List[Int]) -> Int:
    return ok_get(xs, len(xs))  # one past the end


@inline(.never)
def bad_window_swapped(xs: List[Int]) -> Int:
    if len(xs) > 2:
        return ok_window(xs, 2, 1)  # lo > hi
    return 0


@inline(.never)
def bad_at_unchecked(xs: List[Int]) -> Int:
    return ok_at(xs, 2)  # xs may be shorter


# --- must be PROVEN ---
@inline(.never)
def ok_get_checked(xs: List[Int], i: Int) -> Int:
    if 0 <= i and i < len(xs):
        return ok_get(xs, i)
    return 0


@inline(.never)
def ok_get_last(xs: List[Int]) -> Int:
    if len(xs) > 0:
        return ok_get(xs, len(xs) - 1)
    return 0


@inline(.never)
def ok_window_all(xs: List[Int]) -> Int:
    if len(xs) > 0:
        return ok_window(xs, 0, len(xs) - 1)
    return 0


@inline(.never)
def ok_at_checked(xs: List[Int]) -> Int:
    if len(xs) > 2:
        return ok_at(xs, 2)
    return 0


# --- Postconditions: clauses on `out` and `mut` arguments ---
# Proven from the body, and assumed after calls.
@inline(.never)
def ok_make_two(out result: List[Int] where len(result) == 2):
    result = List[Int]()
    result.append(1)
    result.append(2)


@inline(.never)
def ok_next(a: Int where a < 1000, out r: Int where r > a):
    r = a + 1


@inline(.never)
def ok_push(mut xs: List[Int] where len(xs) == old(len(xs)) + 1, v: Int):
    xs.append(v)


# A clause only in terms of `old` is a precondition: `pop` needs it.
@inline(.never)
def ok_shrink(
    mut xs: List[Int] where old(len(xs)) > 0 where len(xs) == old(len(xs)) - 1,
):
    _ = xs.pop()


# Without `old`, a `mut` clause holds on entry and on exit.
@inline(.never)
def ok_keep(mut xs: List[Int] where len(xs) >= 1):
    xs[0] = 5


# A loop fills the result; the length follows from a loop invariant.
@inline(.never)
def ok_make_loop(
    n: Int where n >= 0, out result: List[Int] where len(result) == n
):
    result = List[Int](capacity=n)
    for i in range(n):
        result.append(i)


# Quantifiers: `all([cond for i in range(lo, hi)])`.
@inline(.never)
def ok_swap_front(
    mut xs: List[Int] where old(len(xs)) >= 2 where len(xs) == old(
        len(xs)
    ) and all([xs[i] == old(xs[i]) for i in range(2, len(xs))])
):
    var t = xs[0]
    xs[0] = xs[1]
    xs[1] = t


@inline(.never)
def ok_first_nonneg(
    xs: List[Int] where len(xs) > 0 and all(
        [xs[i] >= 0 for i in range(len(xs))]
    ),
) -> Int:
    return xs[0]


# --- must stay UNPROVEN ---
@inline(.never)
def bad_swap_touches_rest(
    mut xs: List[Int] where old(len(xs)) >= 4 where len(xs) == old(
        len(xs)
    ) and all([xs[i] == old(xs[i]) for i in range(2, len(xs))])
):
    var t = xs[0]
    xs[0] = xs[1]
    xs[1] = t
    xs[3] = 7  # not unchanged


@inline(.never)
def bad_first_negative() -> Int:
    var xs = List[Int]()
    xs.append(1)
    xs.append(-2)
    return ok_first_nonneg(xs)  # xs[1] < 0


@inline(.never)
def bad_push_twice(mut xs: List[Int] where len(xs) == old(len(xs)) + 2, v: Int):
    xs.append(v)  # grows by one


@inline(.never)
def bad_keep_cleared(mut xs: List[Int] where len(xs) >= 1):
    xs.clear()  # empty on exit


@inline(.never)
def bad_shrink_unchecked(mut xs: List[Int]):
    ok_shrink(xs)  # xs may be empty


@inline(.never)
def bad_after_make() -> Int:
    var xs = ok_make_loop(3)
    return xs[3]  # one past the end


# --- must be PROVEN ---
@inline(.never)
def ok_first_of_appended() -> Int:
    var xs = List[Int]()
    xs.append(1)
    xs.append(2)
    return ok_first_nonneg(xs)


@inline(.never)
def ok_after_make_two() -> Int:
    var xs = ok_make_two()
    return xs[1]


@inline(.never)
def ok_after_make() -> Int:
    var xs = ok_make_loop(3)
    return xs[2]  # from the contract


@inline(.never)
def ok_after_push(mut xs: List[Int]) -> Int:
    ok_push(xs, 7)
    return xs[len(xs) - 1]


@inline(.never)
def ok_after_shrink(mut xs: List[Int]) -> Int:
    if len(xs) > 1:
        ok_shrink(xs)
        return xs[0]
    return 0


def main():
    var xs: List[Int] = [1, 2, 3]
    print(
        bad_get_unchecked(xs, 1),
        bad_get_past_end(xs),
        bad_window_swapped(xs),
        bad_at_unchecked(xs),
        ok_get_checked(xs, 1),
        ok_get_last(xs),
        ok_window_all(xs),
        ok_at_checked(xs),
        ok_next(1),
        bad_after_make(),
        ok_after_make_two(),
        ok_after_make(),
        ok_after_push(xs),
        ok_after_shrink(xs),
    )
    bad_push_twice(xs, 1)
    ok_swap_front(xs)
    bad_swap_touches_rest(xs)
    print(bad_first_negative(), ok_first_of_appended())
    ok_keep(xs)
    bad_keep_cleared(xs)
    bad_shrink_unchecked(xs)
