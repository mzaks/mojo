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
# Postconditions for `verify-contracts`: proven at returns, assumed after
# calls; see README.md.

from std.builtin._verification import _same_elements


# --- must be PROVEN (their postconditions, and their callers' calls) ---
def ok_next(a: Int, out r: Int) requires a < 1000 ensures r > a:
    r = a + 1


def ok_push(mut xs: List[Int], v: Int) ensures len(xs) == old(len(xs)) + 1:
    xs.append(v)  # from `List.append`'s postcondition


def ok_reset_first(mut xs: List[Int])
    requires len(xs) > 0
    ensures len(xs) == old(len(xs)) and all(
        [xs[i] == old(xs[i]) for i in range(1, len(xs))]
    ):
    xs[0] = 7  # an element write keeps the length and the other elements


def ok_swap_front(mut xs: List[Int])
    requires len(xs) >= 2
    ensures len(xs) == old(len(xs)) and all(
        [xs[i] == old(xs[i]) for i in range(2, len(xs))]
    ):
    var t = xs[0]
    xs[0] = xs[1]
    xs[1] = t


def ok_push_keep(mut xs: List[Int])
    ensures len(xs) == old(len(xs)) + 1 and _same_elements(
        xs._data, old(xs._data), old(len(xs))
    ):
    xs.append(9)  # `List.append` keeps the elements it had


def ok_after_next(xs: List[Int]) -> Int:
    var i = ok_next(-1)  # `i > -1`
    if len(xs) > i:
        return xs[i]
    return 0


def ok_after_push(mut xs: List[Int], v: Int) -> Int:
    ok_push(xs, v)
    return xs[len(xs) - 1]  # `len(xs) >= 1`


def ok_after_reset(mut xs: List[Int], k: Int) -> Int:
    if len(xs) > 2:
        ok_reset_first(xs)
        return xs[xs[1] * 0 + 2]  # the length is kept
    return 0


def ok_index_through_kept_element(mut xs: List[Int], ys: List[Int]) -> Int:
    if len(xs) > 1 and xs[1] >= 0 and xs[1] < len(ys):
        ok_reset_first(xs)
        return ys[xs[1]]  # `xs[1]` is unchanged
    return 0


def ok_append_in_loop(mut xs: List[Int]) -> Int:
    var s = 0
    for i in range(len(xs)):
        xs.append(i)
        s += xs[i]  # `len(xs)` only grows
    return s


# --- must stay UNPROVEN ---
def bad_push_twice(mut xs: List[Int], v: Int)
    ensures len(xs) == old(len(xs)) + 2:
    xs.append(v)


def bad_swap_touches_rest(mut xs: List[Int])
    requires len(xs) >= 3
    ensures all([xs[i] == old(xs[i]) for i in range(2, len(xs))]):
    xs[2] = xs[0]


def bad_next_equal(a: Int, out r: Int) ensures r > a:
    r = a


def bad_index_through_reset_element(mut xs: List[Int], ys: List[Int]) -> Int:
    if len(xs) > 1 and xs[0] >= 0 and xs[0] < len(ys):
        ok_reset_first(xs)
        return ys[xs[0]]  # `xs[0]` was reset
    return 0


def bad_after_push_past_end(mut xs: List[Int], v: Int) -> Int:
    ok_push(xs, v)
    return xs[len(xs)]
