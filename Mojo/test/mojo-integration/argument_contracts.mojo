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

# `where` clauses on runtime arguments are contracts for static verification
# (see Mojo/proposals/argument-contracts.md). They have no runtime effect: the
# program compiles and runs as without them, even when a call violates one.

# RUN: %mojo %s | FileCheck %s


@inline(.never)
def get(xs: List[Int], i: Int where 0 <= i and i < len(xs)) -> Int:
    return xs[i]


@always_inline
def doubled(a: Int where a > 0 else "a must be positive") -> Int:
    return a * 2


@inline(.never)
def first[T: Copyable](xs: List[T] where len(xs) > 0) -> T:
    return xs[0].copy()


struct Counter:
    var count: Int

    def __init__(out self, count: Int):
        self.count = count

    @inline(.never)
    def minus(self where self.count >= n, n: Int where n >= 0) -> Int:
        return self.count - n


def main():
    var xs: List[Int] = [10, 20, 30]
    # CHECK: 20
    print(get(xs, 1))
    # CHECK: 8
    print(doubled(4))
    # A violated contract changes nothing at runtime.
    # CHECK: -6
    print(doubled(-3))
    # CHECK: 10
    print(first(xs))
    # CHECK: 3
    print(Counter(5).minus(2))
