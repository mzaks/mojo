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
# Calls checked against the callee's contracts alone, for
# `bounds-check-report="modular=contracts"`; see README.md.

from std.builtin._verification import _same_elements


# Resets the first element and keeps the others: the quantifier says so.
@inline(.never)
def ok_reset_first(
    mut xs: List[Int] where old(len(xs)) > 0 where len(xs) == old(
        len(xs)
    ) and all([xs[i] == old(xs[i]) for i in range(1, len(xs))])
):
    xs[0] = 7


# Resets the first element; its contract only states the length.
@inline(.never)
def ok_reset_first_length_only(
    mut xs: List[Int] where old(len(xs)) > 0 where len(xs) == old(len(xs)),
):
    xs[0] = 7


# Appends and keeps the elements it had: `_same_elements` says so.
@inline(.never)
def ok_push_keep(
    mut xs: List[Int] where len(xs) == old(len(xs)) + 1 and _same_elements(
        xs._data, old(xs._data), old(len(xs))
    )
):
    xs.append(9)


# --- must be PROVEN ---
@inline(.never)
def ok_index_after_reset(ys: List[Int]) -> Int:
    var xs = List[Int]()
    xs.append(5)
    xs.append(1)
    ok_reset_first(xs)
    if len(ys) == 2:
        return ys[xs[1]]  # xs[1] is still 1: from the quantifier
    return 0


@inline(.never)
def ok_index_after_push(ys: List[Int]) -> Int:
    var xs = List[Int]()
    xs.append(1)
    ok_push_keep(xs)
    if len(ys) == 2:
        return ys[xs[0]]  # xs[0] is still 1: the callee keeps it
    return 0


# --- must stay UNPROVEN ---
@inline(.never)
def bad_index_reset_element(ys: List[Int]) -> Int:
    var xs = List[Int]()
    xs.append(0)
    xs.append(1)
    ok_reset_first(xs)
    if len(ys) == 2:
        return ys[xs[0]]  # the contract says nothing about xs[0]
    return 0


@inline(.never)
def limit_index_after_length_only(ys: List[Int]) -> Int:
    var xs = List[Int]()
    xs.append(5)
    xs.append(1)
    ok_reset_first_length_only(xs)
    if len(ys) == 2:
        return ys[xs[1]]  # 1 in fact, but the contract does not say so
    return 0


@inline(.never)
def bad_index_pushed_element(ys: List[Int]) -> Int:
    var xs = List[Int]()
    xs.append(1)
    ok_push_keep(xs)
    if len(ys) == 2:
        return ys[xs[1]]  # the contract says nothing about the new element
    return 0


def main():
    var ys: List[Int] = [1, 2]
    print(
        ok_index_after_reset(ys),
        ok_index_after_push(ys),
        bad_index_pushed_element(ys),
        bad_index_reset_element(ys),
        limit_index_after_length_only(ys),
    )
