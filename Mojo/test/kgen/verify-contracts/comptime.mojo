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

from std.sys import simd_width_of, size_of


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


def ok_where_constraint[n: Int](xs: List[Int]) -> Int where 0 <= n and n < 8:
    if len(xs) >= 8:
        return xs[n]  # the function's `where` holds wherever it is instantiated
    return 0


def ok_where_second[
    m: Int, n: Int
](xs: List[Int]) -> Int where 0 <= n and n < 8:
    if len(xs) >= 8:
        return xs[n]
    return 0


def ok_where_range[n: Int](xs: List[Int]) -> Int where n <= 8:
    var sum = 0
    if len(xs) >= 8:
        comptime for k in range(n):
            sum += xs[k]
    return sum


def ok_simd_width[dt: DType](xs: List[Int]) -> Int:
    comptime width = simd_width_of[dt]()  # 0 or a power of two up to 256
    if len(xs) == 1:
        return xs[6144 % width]
    return 0


def ok_dtype_in[dt: DType](xs: List[Int]) -> Int:
    comptime if dt == DType.float32:
        comptime if dt in (DType.float32, DType.bfloat16):
            return 0
        else:
            return xs[0]  # not reached: `float32` is in the tuple
    return 0


def ok_comptime_expression[
    flag: Bool
](xs: List[Int], n: Int where n >= 0 and n < 1000) -> Int:
    var k = (n + 1) if comptime (flag) else (n + 2)
    if len(xs) > n + 2:
        return xs[k]  # either arm's value is below n + 3
    return 0


def ok_closure_parameter(xs: List[Int]) -> Int:
    var total = 0

    @__parameter
    def add[k: Int]():
        total += xs[k]  # checked at each call, for that call's `k`

    if len(xs) >= 4:
        add[0]()
        add[3]()
    return total


def at_most[limit: Int](x: Int where x <= limit) -> Int:
    return x


def ok_closure_passes_parameter[limit: Int](x: Int) -> Int:
    var total = 0

    @__parameter
    def add[k: Int]():
        # `limit` is the enclosing function's, in the callee's clause too.
        total += at_most[limit](x - k)

    if 0 <= x and x <= limit:
        add[1]()
    return total


def ok_dtype_size(xs: List[Int]) -> Int:
    comptime width = 16 // size_of[DType.float32]()
    if len(xs) == 5:
        return xs[width]  # 4: a `float32` is 4 bytes
    return 0


def ok_dtype_size_any[dt: DType](xs: List[Int]) -> Int:
    comptime bytes = size_of[dt]()
    if len(xs) == 33:
        return xs[bytes]  # at most 32, a `uint256`
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


def bad_where_bound[n: Int](xs: List[Int]) -> Int where 0 <= n and n <= 8:
    if len(xs) >= 8:
        return xs[n]  # `n` may be 8
    return 0


def bad_where_other[
    m: Int, n: Int
](xs: List[Int]) -> Int where 0 <= m and m < 8:
    if len(xs) >= 8:
        return xs[n]  # the constraint is on `m`
    return 0


def bad_simd_width[dt: DType](xs: List[Int]) -> Int:
    comptime width = simd_width_of[dt]()
    if len(xs) == 1:
        return xs[6 % width]  # 2 for a width of 4
    return 0


def bad_dtype_in[dt: DType](xs: List[Int]) -> Int:
    comptime if dt in (DType.float32, DType.bfloat16):
        comptime if dt != DType.float32:
            return xs[0]  # `bfloat16` gets here
    return 0


def bad_comptime_expression[
    flag: Bool
](xs: List[Int], n: Int where n >= 0 and n < 1000) -> Int:
    var k = n if comptime (flag) else (n + 3)
    if len(xs) > n + 2:
        return xs[k]  # n + 3 when `flag` is False
    return 0


def bad_closure_parameter(xs: List[Int]) -> Int:
    var total = 0

    @__parameter
    def add[k: Int]():
        total += xs[k]

    if len(xs) >= 4:
        add[0]()
        add[4]()  # this call's `k` is past the end
    return total


def bad_dtype_size[dt: DType](xs: List[Int]) -> Int:
    comptime width = 16 // size_of[dt]()
    if len(xs) == 16:
        return xs[width]  # 16 for a one-byte dtype
    return 0
