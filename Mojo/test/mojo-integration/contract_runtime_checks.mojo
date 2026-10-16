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

# `requires` and `ensures` clauses are checked at run time when assertions
# are enabled (see Mojo/proposals/function-contracts.md), and not evaluated
# at all otherwise. A clause that uses `old` is not checked at run time.

# Without assertions every case runs to its end.
# RUN: %mojo-no-debug-no-assert -D CASE=1 %s | FileCheck %s --check-prefix=OFF
# RUN: %mojo-no-debug-no-assert -D CASE=2 %s | FileCheck %s --check-prefix=OFF

# RUN: not %mojo -D CASE=1 %s 2>&1 | FileCheck %s --check-prefix=PRE
# RUN: not %mojo -D CASE=2 %s 2>&1 | FileCheck %s --check-prefix=POST
# RUN: not %mojo -D CASE=3 %s 2>&1 | FileCheck %s --check-prefix=MESSAGE
# RUN: not %mojo -D CASE=4 %s 2>&1 | FileCheck %s --check-prefix=MEMORY
# RUN: %mojo -D CASE=5 %s 2>&1 | FileCheck %s --check-prefix=PASS

from std.sys.defines import get_defined_int


@inline(.never)
def get(xs: List[Int], i: Int) -> Int requires 0 <= i and i < len(xs):
    return xs[i] if 0 <= i and i < len(xs) else -1


@inline(.never)
def next_up(a: Int) -> Int ensures result > a:
    return a  # breaks its postcondition


@inline(.never)
def positive(a: Int) -> Int ensures result > 0 else "result must be positive":
    return a


@inline(.never)
def make(n: Int) -> List[Int] ensures len(result) == n:
    return List[Int](length=n + 1, fill=0)  # one too many


@inline(.never)
def grow(mut xs: List[Int]) ensures len(xs) == old(len(xs)) + 2:
    xs.append(1)  # one short, but `old` clauses are not checked at run time


def main():
    comptime test_case = get_defined_int["CASE", 0]()
    var xs: List[Int] = [1, 2, 3]
    comptime if test_case == 1:
        # PRE: precondition of 'get' does not hold: 0 <= i and i < len(xs)
        print(get(xs, 5))
    elif test_case == 2:
        # POST: postcondition of 'next_up' does not hold: result > a
        print(next_up(3))
    elif test_case == 3:
        # MESSAGE: result must be positive
        print(positive(-1))
    elif test_case == 4:
        # MEMORY: postcondition of 'make' does not hold: len(result) == n
        print(len(make(2)))
    elif test_case == 5:
        grow(xs)
        print(len(xs))
    # OFF: done
    # PASS: done
    print("done")
