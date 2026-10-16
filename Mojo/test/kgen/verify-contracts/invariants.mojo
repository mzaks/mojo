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
# Struct invariants for `verify-contracts`: assumed where a method starts,
# proven where a method that initializes or changes the value returns, and
# assumed after a call to such a method; see README.md.

from std.memory.alloc import unsafe_alloc


struct Counter(Movable)
    invariant 0 <= self.count
    invariant self.count <= self.limit:
    var count: Int
    var limit: Int

    def __init__(out self, limit: Int) requires limit >= 0:
        self.count = 0
        self.limit = limit

    def __init__(out self, *, copy: Self):
        # From the invariant of `copy`, an argument of the struct's type.
        self.count = copy.count
        self.limit = copy.limit

    # --- must be PROVEN ---
    def ok_bump(mut self):
        if self.count < self.limit:
            self.count += 1

    def ok_room(self) -> Int ensures result >= 0:
        return self.limit - self.count  # from the invariant on entry

    def ok_merge(mut self, mut other: Self):
        # Both keep it: `other` too, a `mut` argument of the struct's type.
        # (Not `self.count + other.count <= self.limit`, which can wrap.)
        if other.count <= self.limit - self.count:
            self.count += other.count
            other.count = 0

    # --- must be WARNED (where they return) ---
    def bad_bump(mut self):
        self.count += 1  # may pass `limit`

    def bad_merge(mut self, mut other: Self):
        self.count += other.count  # may pass `limit`, or wrap
        other.count = 0


# A cursor into a buffer of `size` elements.
struct Ring
    invariant self.size > 0 and self.data._extent() >= self.size
    invariant 0 <= self.at and self.at < self.size:
    var data: Pointer[Int, MutUntrackedOrigin]
    var size: Int
    var at: Int

    def __init__(out self, size: Int) requires size > 0:
        self.data = unsafe_alloc[Int](size)
        self.size = size
        self.at = 0

    # --- must be PROVEN ---
    def ok_current(self) -> Int:
        return self.data[unsafe_offset=self.at]  # from the invariant

    def ok_advance(mut self):
        if self.at + 1 < self.size:
            self.at += 1
        else:
            self.at = 0

    # --- must be WARNED ---
    def bad_seek(mut self, at: Int):
        self.at = at


# --- must be PROVEN ---
def ok_after_mut_call(xs: List[Int], var c: Counter) -> Int:
    c.ok_bump()
    if c.limit < len(xs):
        return xs[c.count]  # the invariant holds again after a `mut` method
    return 0


def ok_new_ring() -> Int:
    var r = Ring(4)
    return r.data[unsafe_offset=r.at]  # from the constructor


# --- must be WARNED ---
def bad_argument(xs: List[Int], c: Counter) -> Int:
    # A function that is not a method of the struct does not assume the
    # invariant of its arguments.
    if c.limit < len(xs):
        return xs[c.count]
    return 0


def main():
    var xs: List[Int] = [1, 2, 3]
    print(ok_after_mut_call(xs, Counter(1)), bad_argument(xs, Counter(1)))
    print(ok_new_ring())
