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
# `List`'s contracts for `verify-contracts`: lengths from constructors,
# literals and mutating methods; see README.md.


# --- must be PROVEN ---
def ok_literal() -> Int:
    var xs: List[Int] = [1, 2, 3]
    return xs[2]


def ok_built_by_append() -> Int:
    var xs = List[Int]()
    xs.append(1)
    xs.append(2)
    return xs[0] + xs[1]


def ok_filled(n: Int) -> Int:
    if n > 0:
        var xs = List[Int](length=n, fill=7)
        return xs[n - 1]
    return 0


def ok_filled_in_loop(n: Int) -> Int:
    var s = 0
    if n >= 0:
        var xs = List[Int](capacity=n)
        for i in range(n):
            xs.append(i)
        for i in range(n):
            s += xs[i]
    return s


def ok_insert(mut xs: List[Int]) -> Int:
    xs.insert(0, 5)
    return xs[0]


def ok_clear_then_append(mut xs: List[Int]) -> Int:
    xs.clear()
    xs.append(3)
    return xs[0]


def ok_reverse_keeps(mut xs: List[Int]) -> Int:
    if len(xs) > 0:
        xs.reverse()
        return xs[len(xs) - 1]
    return 0


def ok_extend(mut xs: List[Int], var ys: List[Int]) -> Int:
    if len(ys) > 0:
        xs.extend(ys^)
        return xs[0]
    return 0


def ok_resize(mut xs: List[Int]) -> Int:
    xs.resize(4, 0)
    return xs[3]


def ok_pop_index(mut xs: List[Int]) -> Int:
    if len(xs) > 1:
        _ = xs.pop(1)
        return xs[0]
    return 0


def ok_literal_values() -> Int:
    var xs: List[Int] = [1, 2, 3]
    return xs[xs[0]]  # a literal's elements are its values


def ok_nested_literal() -> Int:
    var xs: List[List[Int]] = [[1, 2, 3], [4, 5]]
    return xs[0][2] + xs[1][1]


def ok_nested_append() -> Int:
    var xs: List[List[Int]] = [[1], [2]]
    xs[0].append(7)  # the inner list grows, the outer keeps its length
    return xs[0][1] + xs[1][0]


def ok_unsafe_access() -> Int:
    var xs: List[Int] = [1, 2, 3]
    xs.unsafe_set(0, 5)  # checked like `xs[0] = 5`, and keeps the length
    return xs.unsafe_get(2)


def ok_copy(xs: List[Int]) -> Int:
    var ys = xs.copy()  # as long as the original
    if len(xs) > 0:
        return ys[len(xs) - 1]
    return 0


def ok_grow_rows(mut xs: List[List[Int]]):
    for i in range(len(xs)):
        xs[i].append(0)  # growing a row keeps the number of rows


def ok_nested_param(xs: List[List[Int]], i: Int, j: Int) -> Int:
    if 0 <= i and i < len(xs):
        if 0 <= j and j < len(xs[i]):
            return xs[i][j]
    return 0


# --- must stay UNPROVEN ---
def bad_literal_past_end() -> Int:
    var xs: List[Int] = [1, 2, 3]
    return xs[3]


def bad_after_clear(mut xs: List[Int]) -> Int:
    xs.clear()
    return xs[0]


def bad_insert_past_end(mut xs: List[Int]) -> Int:
    xs.insert(len(xs) + 1, 5)
    return 0


def bad_negative_length() -> Int:
    var xs = List[Int](length=-1, fill=0)
    return len(xs)


def bad_pop_index_past_end(mut xs: List[Int]) -> Int:
    if len(xs) > 1:
        return xs.pop(len(xs))
    return 0


def bad_nested_after_pop() -> Int:
    var xs: List[List[Int]] = [[1, 2], [3]]
    _ = xs[0].pop()
    return xs[0][1]  # the inner list shrank


def bad_nested_other_row(xs: List[List[Int]], j: Int) -> Int:
    if len(xs) > 1 and 0 <= j and j < len(xs[0]):
        return xs[1][j]  # bounded by the first row, not the second
    return 0


def bad_copy_then_grow(xs: List[Int]) -> Int:
    var ys = xs.copy()
    var zs = xs.copy()
    zs.append(1)
    return ys[len(zs) - 1]  # `ys` did not grow


def bad_unsafe_get_past() -> Int:
    var xs: List[Int] = [1, 2, 3]
    return xs.unsafe_get(3)  # unchecked at run time, not statically
