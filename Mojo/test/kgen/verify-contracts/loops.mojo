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
# Loops for `verify-contracts`: invariants found at the loop head; see
# README.md.


def change(mut xs: List[Int]):
    xs.clear()


# --- must be PROVEN ---
def ok_sum(xs: List[Int]) -> Int:
    var s = 0
    for i in range(len(xs)):
        s += xs[i]
    return s


def ok_sum_from(xs: List[Int], start: Int) -> Int:
    var s = 0
    if start >= 0:
        for i in range(start, len(xs)):
            s += xs[i]
    return s


def ok_while(xs: List[Int]) -> Int:
    var s = 0
    var i = 0
    while i < len(xs):
        s += xs[i]
        i += 1
    return s


def ok_pairs(xs: List[Int]) -> Int:
    var s = 0
    for i in range(len(xs)):
        for j in range(i):
            s += xs[i] * xs[j]
    return s


def ok_prefix(xs: List[Int], ys: List[Int]) -> Int:
    var s = 0
    if len(ys) >= len(xs):
        for i in range(len(xs)):
            s += xs[i] + ys[i]
    return s


def ok_early_break(xs: List[Int], stop: Int) -> Int:
    var i = 0
    while i < len(xs):
        if xs[i] == stop:
            break
        i += 1
    return i


def ok_reversed(xs: List[Int]) -> Int:
    var t = 0
    for i in reversed(range(len(xs))):
        t += xs[i]  # from `len(xs) - 1` down to 0
    return t


def ok_reversed_from(xs: List[Int], start: Int) -> Int:
    var t = 0
    if 0 <= start:
        for i in reversed(range(start, len(xs))):
            t += xs[i]  # down to `start`, inclusive
    return t


def ok_reversed_pairs(xs: List[Int]) -> Int:
    var t = 0
    for i in reversed(range(1, len(xs))):
        t += xs[i] - xs[i - 1]  # `i` never reaches 0
    return t


# --- must stay UNPROVEN ---
def bad_one_past(xs: List[Int]) -> Int:
    var s = 0
    for i in range(len(xs) + 1):
        s += xs[i]  # `i == len(xs)` in the last iteration
    return s


def bad_other_list(xs: List[Int], ys: List[Int]) -> Int:
    var s = 0
    for i in range(len(xs)):
        s += ys[i]  # `ys` may be shorter
    return s


def bad_while_le(xs: List[Int]) -> Int:
    var s = 0
    var i = 0
    while i <= len(xs):
        s += xs[i]
        i += 1
    return s


def bad_changed_in_loop(mut xs: List[Int]) -> Int:
    var s = 0
    for i in range(len(xs)):
        s += xs[i]  # the first iteration is fine, a later one may not be
        change(xs)
    return s


def bad_after_loop(xs: List[Int]) -> Int:
    var i = 0
    while i < len(xs):
        i += 1
    return xs[i]  # `i == len(xs)` after the loop


def bad_reversed_one_past(xs: List[Int]) -> Int:
    var t = 0
    for i in reversed(range(len(xs))):
        t += xs[i + 1]  # `i + 1` is `len(xs)` on the first step
    return t


def bad_reversed_below(xs: List[Int]) -> Int:
    var t = 0
    for i in reversed(range(len(xs))):
        t += xs[i - 1]  # `i` reaches 0
    return t
