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
# Facts from `std.testing` assertions for `verify-contracts`; see README.md.

from std.testing import (
    assert_equal,
    assert_false,
    assert_not_equal,
    assert_true,
)


# --- must be PROVEN ---
def ok_after_assert_equal(xs: List[Int]) raises -> Int:
    assert_equal(len(xs), 3)  # past it, `len(xs)` is 3
    return xs[2]


def ok_after_assert_true(xs: List[Int], i: Int) raises -> Int:
    assert_true(0 <= i and i < len(xs))
    return xs[i]


def ok_after_assert_false(xs: List[Int], i: Int) raises -> Int:
    assert_false(i < 0 or i >= len(xs))
    return xs[i]


def ok_after_assert_not_equal(xs: List[Int]) raises -> Int:
    assert_not_equal(len(xs), 0)
    return xs[len(xs) - 1]


def ok_scalar(xs: List[Int], n: UInt8) raises -> Int:
    assert_equal(n, UInt8(2))
    assert_equal(len(xs), Int(n) + 1)
    return xs[2]


# --- must stay UNPROVEN ---
def bad_before_assert(xs: List[Int]) raises -> Int:
    var x = xs[2]  # the assertion comes after
    assert_equal(len(xs), 3)
    return x


def bad_weaker_assert(xs: List[Int], i: Int) raises -> Int:
    assert_true(i < len(xs))  # `i` may be negative
    return xs[i]


def bad_assert_other(xs: List[Int], ys: List[Int]) raises -> Int:
    assert_equal(len(ys), 3)  # about another list
    return xs[2]
