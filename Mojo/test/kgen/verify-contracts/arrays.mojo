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
# `Array` and spans over arrays for `verify-contracts`; see README.md.


# --- must be PROVEN ---
def ok_index(a: Array[Int, 5], i: Int) -> Int:
    if 0 <= i and i < 5:
        return a[i]  # the length is the type's size parameter
    return 0


def ok_loop(a: Array[Int, 5]) -> Int:
    var t = 0
    for i in range(len(a)):
        t += a[i]
    return t


def ok_generic[n: Int](a: Array[Int, n], i: Int) -> Int:
    if 0 <= i and i < n:
        return a[i]  # `len(a)` is `n`
    return 0


def ok_span_of_array() -> Int:
    var a: Array[Int, 7] = [1, 2, 3, 4, 5, 6, 7]
    var s = Span(array=a)
    return s[6]  # the span is as long as the array


def ok_span_of_array_loop(a: Array[Int, 4]) -> Int:
    var s = Span(array=a)
    var t = 0
    for i in range(len(s)):
        t += s[i]
    return t


def ok_slice_of_array_span() -> Int:
    var a: Array[Int, 7] = [1, 2, 3, 4, 5, 6, 7]
    var s = Span(array=a)[2:]
    return s[4]


# --- must stay UNPROVEN ---
def bad_index_past(a: Array[Int, 5], i: Int) -> Int:
    if 0 <= i and i <= 5:
        return a[i]  # `i` may be 5
    return 0


def bad_generic_past[n: Int](a: Array[Int, n], i: Int) -> Int:
    if 0 <= i and i <= n:
        return a[i]  # `i` may be `n`
    return 0


def bad_span_of_array_past() -> Int:
    var a: Array[Int, 3] = [1, 2, 3]
    var s = Span(array=a)
    return s[3]


def bad_slice_of_array_span() -> Int:
    var a: Array[Int, 3] = [1, 2, 3]
    return len(Span(array=a)[4:])  # the array is shorter than 4
