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
# `match` and `if case` for `verify-contracts`: what is known before a match
# holds after it, and a value set in its cases is one of those; see
# README.md.


# --- must be PROVEN ---
def ok_after_match(xs: List[Int], o: Optional[Int]) -> Int:
    var t = 0
    match o:
        case .None:
            pass
        case .Some(v):
            t = v
    if len(xs) > 3:
        return xs[3] + t  # the length is known through the match
    return t


def ok_set_in_cases(xs: List[Int], k: Int) -> Int:
    var i = 0
    match k:
        case 0:
            i = 1
        case 1:
            i = 2
        case _:
            pass
    if len(xs) > 2:
        return xs[i]  # 0, 1 or 2
    return 0


def ok_if_case(xs: List[Int], o: Optional[Int]) -> Int:
    if len(xs) > 0:
        if case .Some(v) = o:
            return xs[0] + v
        return xs[0]
    return 0


def ok_index_in_case(xs: List[Int], k: Int) -> Int:
    match k:
        case 0:
            if len(xs) > 0:
                return xs[0]
        case _:
            pass
    return 0


# --- must stay UNPROVEN ---
def bad_set_in_case(xs: List[Int], k: Int) -> Int:
    var i = 0
    match k:
        case 0:
            i = 5
        case _:
            pass
    if len(xs) > 2:
        return xs[i]  # 5 when `k` is 0
    return 0


def bad_index_in_case(xs: List[Int], k: Int) -> Int:
    match k:
        case 0:
            return xs[0]  # the list may be empty
        case _:
            pass
    return 0


def main():
    var xs: List[Int] = [1, 2, 3, 4]
    print(ok_after_match(xs, Optional[Int](1)), ok_set_in_cases(xs, 1))
    print(ok_if_case(xs, None), ok_index_in_case(xs, 0))
    print(bad_set_in_case(xs, 1), bad_index_in_case(xs, 0))
