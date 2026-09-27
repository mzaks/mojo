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
# Adversarial cases for the SMT encoding of `bounds-check-report`; see README.md.


# --- must stay UNPROVEN ---
def bad_overflow(xs: List[Int], i: Int) -> Int:
    # i + 1 wraps for i == Int.MAX, so k < len(xs) says nothing about i.
    if i >= 0:
        var k = i + 1
        if k < len(xs):
            return xs[i]
    return 0


def bad_nested(xs: List[Int]) -> Int:
    var s = 0
    for i in range(len(xs)):
        for j in range(i + 2):
            s += xs[j]
    return s


def bad_elif(xs: List[Int], i: Int) -> Int:
    if i < 0:
        return -1
    elif i > len(xs):
        return -2
    else:
        return xs[i]


def bad_try(xs: List[Int], i: Int) -> Int:
    try:
        if i < 0:
            raise Error("negative")
        return xs[i]
    except:
        return 0


def bad_countdown(xs: List[Int]) -> Int:
    var s = 0
    var i = len(xs)
    while i >= 0:
        s += xs[i]
        i -= 1
    return s


# --- should be PROVEN ---
def ok_overflow_safe(xs: List[Int], i: Int) -> Int:
    if i >= 0 and i < len(xs) - 1:
        return xs[i + 1]
    return 0


def ok_nested(xs: List[Int]) -> Int:
    var s = 0
    for i in range(len(xs)):
        for j in range(i):
            s += xs[j]
    return s


def ok_elif(xs: List[Int], i: Int) -> Int:
    if i < 0:
        return -1
    elif i >= len(xs):
        return -2
    else:
        return xs[i]


def ok_try(xs: List[Int], i: Int) -> Int:
    try:
        if i < 0 or i >= len(xs):
            raise Error("out of range")
        return xs[i]
    except:
        return 0


def ok_countdown(xs: List[Int]) -> Int:
    var s = 0
    var i = len(xs)
    while i > 0:
        i -= 1
        s += xs[i]
    return s


def ok_reversed(xs: List[Int]) -> Int:
    var s = 0
    for i in reversed(range(len(xs))):
        s += xs[i]
    return s


def ok_mid(xs: List[Int], lo: Int, hi: Int) -> Int:
    if lo >= 0 and lo <= hi and hi < len(xs):
        return xs[lo + (hi - lo) // 2]
    return 0


def main():
    var xs: List[Int] = [1, 2, 3]
    print(
        bad_overflow(xs, 1),
        bad_nested(xs),
        bad_elif(xs, 1),
        bad_try(xs, 1),
        bad_countdown(xs),
        ok_overflow_safe(xs, 0),
        ok_nested(xs),
        ok_elif(xs, 1),
        ok_try(xs, 1),
        ok_countdown(xs),
        ok_reversed(xs),
        ok_mid(xs, 0, 2),
    )
