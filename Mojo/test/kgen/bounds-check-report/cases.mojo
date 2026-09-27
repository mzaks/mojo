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
# Loop and guard shapes for `bounds-check-report`; see README.md.


# --- must stay UNPROVEN ---
def bad_off_by_one(xs: List[Int]) -> Int:
    var s = 0
    for i in range(len(xs) + 1):
        s += xs[i]
    return s


def bad_le(xs: List[Int], i: Int) -> Int:
    if i >= 0 and i <= len(xs):
        return xs[i]
    return 0


def bad_no_lower(xs: List[Int], i: Int) -> Int:
    if i < len(xs):
        return xs[i]
    return 0


def bad_plus_one(xs: List[Int]) -> Int:
    var s = 0
    for i in range(len(xs)):
        s += xs[i + 1]
    return s


def bad_or(xs: List[Int], i: Int) -> Int:
    if i >= 0 or i < len(xs):
        return xs[i]
    return 0


def bad_other_list(xs: List[Int], ys: List[Int]) -> Int:
    var s = 0
    for i in range(len(xs)):
        s += ys[i]
    return s


# --- should be PROVEN ---
def ok_shifted(xs: List[Int]) -> Int:
    var s = 0
    for i in range(1, len(xs)):
        s += xs[i - 1]
    return s


def ok_pairs(xs: List[Int]) -> Int:
    var s = 0
    for i in range(len(xs) - 1):
        s += xs[i] + xs[i + 1]
    return s


def ok_while(xs: List[Int]) -> Int:
    var s = 0
    var i = 0
    while i < len(xs):
        s += xs[i]
        i += 1
    return s


def ok_early_return(xs: List[Int], i: Int) -> Int:
    if i < 0 or i >= len(xs):
        return 0
    return xs[i]


# --- should be PROVEN (the index flows through the iterator's Optional) ---
def maybe_reversed(xs: List[Int]) -> Int:
    var s = 0
    for i in reversed(range(len(xs))):
        s += xs[i]
    return s


# --- UNPROVEN, known limit: the list length after `append` is not modeled ---
def maybe_mutating(mut xs: List[Int]) -> Int:
    var s = 0
    for i in range(len(xs)):
        xs.append(i)
        s += xs[i]
    return s


def main():
    var xs: List[Int] = [1, 2, 3]
    var ys: List[Int] = [1]
    print(
        bad_off_by_one(xs),
        bad_le(xs, 1),
        bad_no_lower(xs, 1),
        bad_plus_one(xs),
        bad_or(xs, 1),
        bad_other_list(xs, ys),
        ok_shifted(xs),
        ok_pairs(xs),
        ok_while(xs),
        ok_early_return(xs, 1),
        maybe_reversed(xs),
        maybe_mutating(xs),
    )
