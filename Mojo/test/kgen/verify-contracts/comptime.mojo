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
# Generic code for `verify-contracts`: each function is verified once, for
# every value of its parameters; see README.md.


# --- must be PROVEN ---
def ok_param_index[n: Int](xs: List[Int]) -> Int:
    if n >= 0 and n < len(xs):  # folded into a parameter expression
        return xs[n]
    return 0


def ok_comptime_if[n: Int](xs: List[Int]) -> Int:
    comptime if n >= 0:
        if n < len(xs):
            return xs[n]
    return 0


def ok_comptime_elif[n: Int](xs: List[Int]) -> Int:
    comptime if n < 0:
        return 0
    elif n < 4:
        if len(xs) >= 4:
            return xs[n]
    return 0


def ok_comptime_for(xs: List[Int]) -> Int:
    comptime sizes = (1, 2, 4)
    var s = 0
    comptime for k in range(len(sizes)):
        comptime n = rebind[Int](sizes[k])
        if n >= 0 and n < len(xs):
            s += xs[n]
    return s


def ok_loop_in_comptime_for[dt: DType](xs: List[Int]) -> Int:
    var s = 0
    comptime for _ in range(3):
        for i in range(len(xs)):
            s += xs[i]
    return s


def ok_dtype_branches[dt: DType](xs: List[Int], i: Int) -> Int:
    comptime if dt.is_integral():
        if 0 <= i and i < len(xs):
            return xs[i]
    else:
        if 0 <= i and i < len(xs) - 1:  # `i + 1 < len(xs)` could wrap
            return xs[i + 1]
    return 0


# --- must stay UNPROVEN ---
def bad_param_no_lower[n: Int](xs: List[Int]) -> Int:
    if n < len(xs):
        return xs[n]  # `n` may be negative
    return 0


def bad_comptime_else[n: Int](xs: List[Int]) -> Int:
    comptime if n >= 0:
        return 0
    else:
        return xs[n]  # `n < 0` here


def bad_comptime_for_unguarded(xs: List[Int]) -> Int:
    comptime sizes = (1, 2, 4)
    var s = 0
    comptime for k in range(len(sizes)):
        comptime n = rebind[Int](sizes[k])
        s += xs[n]  # `xs` may be short
    return s


def bad_dtype_branch[dt: DType](xs: List[Int], i: Int) -> Int:
    comptime if dt.is_integral():
        if 0 <= i and i < len(xs):
            return xs[i]
    else:
        if 0 <= i and i < len(xs):
            return xs[i + 1]  # one past the end
    return 0
