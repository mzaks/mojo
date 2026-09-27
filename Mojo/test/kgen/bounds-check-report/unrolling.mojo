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
# Lists of lists filled in loops, for `bounds-check-report`; see README.md.


# --- must stay UNPROVEN ---
@inline(.never)
def bad_loop_straight_child() -> Int:
    var list = List[List[Int]](capacity=2)
    for i in range(2):
        var v = List[Int](capacity=3)
        v.append(i)
        v.append(i + 1)
        list.append(v^)
    return list[1][2]


@inline(.never)
def bad_loop_growing_children() -> Int:
    var list = List[List[Int]](capacity=3)
    for i in range(3):
        var v = List[Int](capacity=3)
        v.append(i)
        if i > 0:
            v.append(i)
        if i > 1:
            v.append(i)
        list.append(v^)
    return list[1][2]


@inline(.never)
def bad_loop_unknown_count(n: Int) -> Int:
    var list = List[List[Int]](capacity=2)
    for i in range(n):
        var v = List[Int](capacity=1)
        v.append(i)
        list.append(v^)
    return list[0][0]


@inline(.never)
def bad_2d_realloc_short_first() -> Int:
    var list = List[List[Int]]()
    for i in range(2):
        var v = List[Int]()
        for j in range(i + 2):
            v.append(j)
        list.append(v^)
    return list[0][2]


@inline(.never)
def bad_2d_realloc_index() -> Int:
    var list = List[List[Int]]()
    for i in range(3):
        var v = List[Int]()
        for j in range(3):
            v.append(j)
        list.append(v^)
    return list[2][3]


# Known limit: the loop runs 20 times, more than the analysis unrolls.
@inline(.never)
def limit_loop_many_iterations() -> Int:
    var list = List[List[Int]](capacity=20)
    for i in range(20):
        var v = List[Int](capacity=1)
        v.append(i)
        list.append(v^)
    return list[0][0]


# --- must be PROVEN ---
@inline(.never)
def ok_loop_straight_child() -> Int:
    var list = List[List[Int]](capacity=2)
    for i in range(2):
        var v = List[Int](capacity=3)
        v.append(i)
        v.append(i + 1)
        v.append(i + 2)
        list.append(v^)
    return list[1][2]


@inline(.never)
def ok_loop_growing_children() -> Int:
    var list = List[List[Int]](capacity=3)
    for i in range(3):
        var v = List[Int](capacity=3)
        v.append(i)
        if i > 0:
            v.append(i)
        if i > 1:
            v.append(i)
        list.append(v^)
    return list[2][2] + list[1][1]


@inline(.never)
def ok_loop_nested() -> Int:
    var list = List[List[Int]](capacity=2)
    for i in range(2):
        var v = List[Int](capacity=3)
        for j in range(3):
            v.append(i + j)
        list.append(v^)
    return list[1][2]


@inline(.never)
def ok_2d_realloc() -> Int:
    var list = List[List[Int]]()
    for i in range(2):
        var v = List[Int]()
        for j in range(3):
            v.append(i + j)
        list.append(v^)
    return list[0][2] + list[1][2]


@inline(.never)
def ok_2d_realloc_short_first() -> Int:
    var list = List[List[Int]]()
    for i in range(2):
        var v = List[Int]()
        for j in range(i + 2):
            v.append(j)
        list.append(v^)
    return list[0][1] + list[1][2]


@inline(.never)
def ok_nested_copy() -> Int:
    var list = List[List[Int]](capacity=1)
    var child = List[Int](capacity=2)
    child.append(1)
    child.append(2)
    list.append(child^)
    var copy = list.copy()
    return copy[0][1]


def main():
    print(
        bad_loop_straight_child(),
        bad_loop_growing_children(),
        bad_loop_unknown_count(1),
        bad_2d_realloc_short_first(),
        bad_2d_realloc_index(),
        limit_loop_many_iterations(),
        ok_loop_straight_child(),
        ok_loop_growing_children(),
        ok_loop_nested(),
        ok_2d_realloc(),
        ok_2d_realloc_short_first(),
        ok_nested_copy(),
    )
