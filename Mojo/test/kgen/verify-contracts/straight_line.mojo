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
