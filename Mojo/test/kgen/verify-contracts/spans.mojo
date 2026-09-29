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
# `Span` and contiguous slicing for `verify-contracts`; see README.md.


# --- must be PROVEN ---
def ok_span_of_list(l: List[Int]) -> Int:
    var s = Span(list=l)
    var t = 0
    for i in range(len(l)):
        t += s[i] + l[i]  # the span is as long as the list, which it keeps
    return t


def ok_slice_tail(vs: List[Int]) -> Int:
    if len(vs) > 2:
        var s = vs[1:]
        return s[len(vs) - 2]
    return 0


def ok_slice_prefix(vs: List[Int], n: Int) -> Int:
    if n > 0 and n <= len(vs):
        var s = vs[:n]
        return s[n - 1]
    return 0


def ok_slice_of_span() -> Int:
    var l: List[Int] = [1, 2, 3, 4, 5, 6, 7]
    var s = Span(list=l)
    var s2 = s[2:]
    return s2[0] + s2[4]


def ok_slice_middle(vs: List[Int], a: Int, b: Int) -> Int:
    if 0 <= a and a < b and b <= len(vs):
        var s = vs[a:b]
        return s[b - a - 1]
    return 0


# --- must stay UNPROVEN ---
def bad_slice_tail_past(vs: List[Int]) -> Int:
    if len(vs) > 0:
        var s = vs[1:]
        return s[len(vs) - 1]  # the slice is one shorter
    return 0


def bad_slice_start(vs: List[Int]) -> Int:
    return len(vs[2:])  # `vs` may be shorter than 2


def bad_slice_swapped(vs: List[Int], a: Int, b: Int) -> Int:
    if 0 <= a and a <= len(vs) and 0 <= b and b <= len(vs):
        return len(vs[a:b])  # `a` may be after `b`
    return 0


def bad_span_past_end() -> Int:
    var l: List[Int] = [1, 2, 3]
    var s = Span(list=l)
    return s[3]
