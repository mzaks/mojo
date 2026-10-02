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
# Straight-line code for `verify-contracts`: preconditions checked at calls,
# before elaboration; see README.md.

from std.math import ceildiv
from std.math.uutils import udiv_unchecked


def ok_get(xs: List[Int], i: Int where 0 <= i and i < len(xs)) -> Int:
    return xs[i]  # from the precondition


def ok_window(
    xs: List[Int],
    lo: Int where 0 <= lo,
    hi: Int where lo <= hi where hi < len(xs),
) -> Int:
    return xs[hi] - xs[lo]


def keep(xs: List[Int]):
    pass


def change(mut xs: List[Int]):
    xs.clear()


def bump(mut n: Int):
    n += 5


struct Mode(Equatable, TrivialRegisterPassable):
    """An enum-like wrapper of one integer, compared by value."""

    var _value: Int

    comptime FAST = Self(0)
    comptime SAFE = Self(1)

    @always_inline
    def __init__(out self, value: Int):
        self._value = value

    @always_inline
    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    @always_inline
    def __ne__(self, other: Self) -> Bool:
        return self._value != other._value

    @always_inline
    def __is__(self, other: Self) -> Bool:
        return self == other

    @always_inline
    def __isnot__(self, other: Self) -> Bool:
        return self != other


def run(
    mode: Mode, xs: List[Int] where mode is not Mode.FAST or len(xs) > 0
) -> Int:
    if mode is Mode.FAST:
        return xs[0]  # FAST needs a non-empty list
    return 0


# --- must be PROVEN ---
def ok_get_checked(xs: List[Int], i: Int) -> Int:
    if 0 <= i and i < len(xs):
        return ok_get(xs, i)
    return 0


def ok_get_last(xs: List[Int]) -> Int:
    if len(xs) > 0:
        return ok_get(xs, len(xs) - 1)
    return 0


def ok_early_return(xs: List[Int], i: Int) -> Int:
    if i < 0 or i >= len(xs):
        return 0
    return xs[i]


def ok_through_local(xs: List[Int], i: Int) -> Int:
    var j = i + 1
    if j > 0 and j <= len(xs):
        return xs[j - 1]
    return 0


def ok_window_checked(xs: List[Int]) -> Int:
    if len(xs) > 3:
        return ok_window(xs, 1, 3)
    return 0


def ok_after_imm_call(xs: List[Int], i: Int) -> Int:
    if 0 <= i and i < len(xs):
        keep(xs)  # an immutable reference: `xs` is unchanged
        return xs[i]
    return 0


def ok_elif(xs: List[Int], i: Int) -> Int:
    var j: Int
    if i < 0:
        j = 0
    elif i >= len(xs):
        j = len(xs) - 1
    else:
        j = i  # clamped into range
    if len(xs) > 0:
        return xs[j]
    return 0


def ok_unsigned_index(xs: List[Int], n: UInt8) -> Int:
    if Int(n) < len(xs):
        return xs[Int(n)]  # a zero-extended `UInt8` is not negative
    return 0


def ok_ceildiv(xs: List[Int], n: Int) -> Int:
    # `ceildiv(n, 4)` blocks of 4 cover `n`: the last starts below `n`.
    if n > 0 and len(xs) >= n:
        return xs[(ceildiv(n, 4) - 1) * 4]
    return 0


def ok_udiv_unchecked(a: Int, b: Int) -> Int:
    if b > 0:
        return udiv_unchecked(a, b)  # by 0 it is undefined
    return 0


def ok_mode_chosen(xs: List[Int]) -> Int:
    var mode = Mode.SAFE
    if len(xs) > 0:
        mode = Mode.FAST
    return run(mode, xs)  # FAST only when the list is not empty


# --- must stay UNPROVEN ---
def bad_get(xs: List[Int], i: Int) -> Int:
    return xs[i]  # nothing is known about `i`


def bad_get_unchecked(xs: List[Int], i: Int) -> Int:
    return ok_get(xs, i)  # `i` may be out of range


def bad_get_past_end(xs: List[Int]) -> Int:
    return ok_get(xs, len(xs))  # one past the end


def bad_off_by_one(xs: List[Int], i: Int) -> Int:
    if 0 <= i and i <= len(xs):
        return xs[i]  # `i == len(xs)` passes the check
    return 0


def bad_window_swapped(xs: List[Int]) -> Int:
    if len(xs) > 2:
        return ok_window(xs, 2, 1)  # `lo > hi`
    return 0


def bad_after_mut_call(mut xs: List[Int], i: Int) -> Int:
    if 0 <= i and i < len(xs):
        change(xs)  # may shorten `xs`
        return xs[i]
    return 0


def bad_after_local_mutation(xs: List[Int]) -> Int:
    var i = 0
    if len(xs) > 0:
        bump(i)  # `i` may have changed
        return xs[i]
    return 0


def bad_through_ref_variable(xs: List[Int]) -> Int:
    var i = 0
    ref r = i
    if len(xs) > 0:
        bump(r)  # changes `i` through `r`
        return xs[i]
    return 0


def bad_elif_gap(xs: List[Int], i: Int) -> Int:
    var j: Int
    if i < 0:
        j = 0
    elif i > len(xs):
        j = len(xs) - 1
    else:
        j = i  # `i` may be `len(xs)`
    if len(xs) > 0:
        return xs[j]
    return 0


def bad_signed_index(xs: List[Int], n: Int8) -> Int:
    if Int(n) < len(xs):
        return xs[Int(n)]  # a sign-extended `Int8` may be negative
    return 0


def bad_ceildiv(xs: List[Int], n: Int) -> Int:
    if n > 0 and len(xs) >= n:
        return xs[ceildiv(n, 4) * 4 - 1]  # past `n` when 4 does not divide it
    return 0


def bad_udiv_unchecked(a: Int, b: Int) -> Int:
    if b >= 0:
        return udiv_unchecked(a, b)  # `b` may be 0
    return 0


def bad_mode_chosen(xs: List[Int], fast: Bool) -> Int:
    var mode = Mode.SAFE
    if fast:
        mode = Mode.FAST
    return run(mode, xs)  # FAST may come with an empty list
