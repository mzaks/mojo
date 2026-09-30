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
# Contracts on trait methods for `verify-contracts`; see README.md.
#
# Generic calls (`t.get(i)` on a `T: Counter`) are held to the trait's
# clauses, every implementation is checked against them, and read-only
# trait methods (`count`) are functions of their arguments' values.


trait Counter:
    def count(self, out result: Int where result >= 0):
        ...

    def get(self, i: Int where 0 <= i and i < self.count()) -> Int:
        ...

    # Two clauses: a precondition (it only reads `self` on entry) and a
    # postcondition. Without the first, `n += 1` could wrap at `Int.MAX`.
    def bump(
        mut self where old(self.count()) < Int.MAX where self.count() == old(
            self.count()
        ) + 1
    ):
        ...

    def reset(mut self):
        ...

    # A default: its clauses hold for every conforming struct, which calls
    # it through a wrapper of its own.
    def pick(self, i: Int where i >= 0) -> Int:
        return i


# Implementations: the trait's precondition must imply theirs, and they must
# establish its postcondition. `Tally`'s own clauses say what its `count()`
# is, so its `bump` provably increments it.
struct Tally(Counter):
    var n: Int

    def __init__(out self, n: Int):
        self.n = n

    def count(
        self,
        out result: Int where (self.n >= 0 and result == self.n) or (
            self.n < 0 and result == 0
        ),
    ):
        if self.n >= 0:
            result = self.n
        else:
            result = 0

    def get(self, i: Int where i >= 0) -> Int:  # weaker than the trait's
        return i

    def bump(mut self):
        if self.n < 0:
            self.n = 1
        else:
            self.n += 1

    def reset(mut self):
        self.n = 0


struct BadCount(Counter):
    var n: Int

    def __init__(out self, n: Int):
        self.n = n

    def count(self, out result: Int):
        result = -1  # breaks `result >= 0`

    def get(self, i: Int where i < 3) -> Int:  # not implied by the trait's
        return i

    def bump(mut self):  # `count()` says nothing about `n`
        self.n += 1

    def reset(mut self):
        self.n = 0


# --- must be PROVEN ---
def ok_bump_get[T: Counter](mut t: T) -> Int:
    if t.count() < 10:
        t.bump()  # `count()` is now at least 1
        return t.get(0)
    return t.get(1)


def ok_checked_get[T: Counter](t: T) -> Int:
    if t.count() > 2:  # the same `count()` the precondition names
        return t.get(1)
    return 0


def ok_last[T: Counter](t: T) -> Int:
    var n = t.count()
    if n == 0:
        return 0
    return t.get(n - 1)


def ok_default[T: Counter](t: T) -> Int:
    return t.pick(t.count())


def ok_default_direct(f: Tally) -> Int:
    return f.pick(2)  # the default's precondition, through `Tally`'s wrapper


# --- must stay UNPROVEN ---
def bad_bump[T: Counter](mut t: T):
    t.bump()  # `count()` may be `Int.MAX`


def bad_get[T: Counter](t: T) -> Int:
    return t.get(0)  # `count()` may be 0


def bad_get_after_reset[T: Counter](mut t: T) -> Int:
    if t.count() > 2:
        t.reset()  # `reset` states nothing: `count()` is unknown again
        return t.get(1)
    return 0


def bad_past_end[T: Counter](t: T) -> Int:
    return t.get(t.count())


def bad_default_direct(f: Tally) -> Int:
    return f.pick(-1)


def main():
    var f = Tally(1)
    bad_bump(f)
    print(
        ok_bump_get(f),
        ok_checked_get(f),
        ok_last(f),
        ok_default(f),
        ok_default_direct(f),
        bad_get(f),
        bad_get_after_reset(f),
        bad_past_end(f),
        bad_default_direct(f),
    )
    var b = BadCount(1)
    print(b.get(0))
