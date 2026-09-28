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
# Preconditions written as `where` clauses on arguments, for
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
    )
