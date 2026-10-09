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

# `requires` and `ensures` clauses are contracts for static verification
# (see Mojo/proposals/function-contracts.md). They have no runtime effect: the
# program compiles and runs as without them, even when a call violates one.

# RUN: %mojo %s | FileCheck %s


@inline(.never)
def get(xs: List[Int], i: Int) -> Int requires 0 <= i and i < len(xs):
    return xs[i]


@always_inline
def doubled(a: Int) -> Int requires a > 0 else "a must be positive":
    return a * 2


@inline(.never)
def first[T: Copyable](xs: List[T]) -> T requires len(xs) > 0:
    return xs[0].copy()


struct Counter:
    var count: Int

    def __init__(out self, count: Int):
        self.count = count

    @inline(.never)
    def minus(self, n: Int) -> Int requires self.count >= n requires n >= 0:
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
    postconditions()


# Postconditions, including `old(e)`, have no runtime effect either.
@inline(.never)
def make(n: Int, out result: List[Int])
    requires n >= 0
    ensures len(result) == n:
    result = List[Int](capacity=n)
    for i in range(n):
        result.append(i)


@inline(.never)
def push(mut xs: List[Int], v: Int) ensures len(xs) == old(len(xs)) + 1:
    xs.append(v)


@inline(.never)
def shrink(mut xs: List[Int])
    requires len(xs) > 0
    ensures len(xs) == old(len(xs)) - 1:
    _ = xs.pop()


@always_inline
def bump(mut a: Int) ensures a == old(a) + 1:
    a += 1


# Quantifiers too.
@inline(.never)
def swap_front(mut xs: List[Int])
    requires len(xs) >= 2
    ensures len(xs) == old(len(xs)) and all(
        [xs[i] == old(xs[i]) for i in range(2, len(xs))]
    ):
    var t = xs[0]
    xs[0] = xs[1]
    xs[1] = t


def postconditions():
    var xs = make(3)
    push(xs, 7)
    shrink(xs)
    var a = 41
    bump(a)
    swap_front(xs)
    # CHECK: 3 1 42
    print(len(xs), xs[0], a)
