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
