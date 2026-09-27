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
# Basic `bounds-check-report` examples; see README.md.
# Contracts (`_ensures` / `_assume`) of `List.append` / `List.pop` for
# `bounds-check-report`; see README.md.


# --- must stay UNPROVEN ---
def bad_append_past_end(n: Int) -> Int:
    var x = List[Int]()
    for i in range(n):
        x.append(i)
    return x[n]  # one past the end


def bad_pop_may_empty(mut xs: List[Int]) -> Int:
    if len(xs) > 0:
        _ = xs.pop()
        return xs[0]  # empty if xs had one element
    return 0


def bad_stale_index_after_pop(mut xs: List[Int]) -> Int:
    var n = len(xs)
    if n > 0:
        _ = xs.pop()
        return xs[n - 1]  # the popped slot
    return 0


def bad_pop_unguarded(mut xs: List[Int]) -> Int:
    return xs.pop()  # xs may be empty


def bad_pop_first_unguarded(mut xs: List[Int]) -> Int:
    # `pop(i)` is inlined: its bounds check comes before the length update, so
    # the `0 <= len` invariant of the new length must not justify the check.
    return xs.pop(0)


# --- should be PROVEN ---
def ok_fill_then_read(n: Int) -> Int:
    var x = List[Int]()
    for i in range(n):
        x.append(i)
    var s = 0
    for i in range(n):
        s += x[i]
    return s


def ok_append_then_last(mut xs: List[Int]) -> Int:
    xs.append(7)
    return xs[len(xs) - 1]


def ok_pop_keeps_rest(mut xs: List[Int]) -> Int:
    if len(xs) >= 2:
        _ = xs.pop()
        return xs[0]
    return 0


def ok_pop_index(mut xs: List[Int], i: Int) -> Int:
    if i >= 0 and i < len(xs):
        return xs.pop(i)
    return 0


def ok_append_in_loop(mut xs: List[Int]) -> Int:
    var s = 0
    for i in range(len(xs)):
        xs.append(i)
        s += xs[i]
    return s


def main():
    var xs: List[Int] = [1, 2, 3]
    print(
        bad_append_past_end(3),
        bad_pop_may_empty(xs),
        bad_stale_index_after_pop(xs),
        bad_pop_unguarded(xs),
        bad_pop_first_unguarded(xs),
        ok_fill_then_read(3),
        ok_append_then_last(xs),
        ok_pop_keeps_rest(xs),
        ok_pop_index(xs, 0),
        ok_append_in_loop(xs),
    )
