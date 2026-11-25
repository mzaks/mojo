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
# Updates of a struct with an invariant by code outside it, with
# `type-invariants=true`: a write into the value opens it, and its
# invariant must hold again before anything can see the value (a call that
# is given it or may read it, a copy, leaving the function), so an update
# in several steps is checked once, also where the steps are in different
# branches; see README.md.


struct Nat(ImplicitlyCopyable, Movable) invariant self.value >= 0:
    var value: Int

    def __init__(out self, value: Int) requires value >= 0:
        self.value = value

    def set(mut self, value: Int)
        requires value >= 0
        ensures self.value == value:
        self.value = value


# `lo <= hi`
struct Bounds(ImplicitlyCopyable, Movable) invariant self.lo <= self.hi:
    var lo: Int
    var hi: Int

    def __init__(out self, lo: Int, hi: Int) requires lo <= hi:
        self.lo = lo
        self.hi = hi


# `lo.value <= hi.value`
struct NatBounds(ImplicitlyCopyable, Movable)
    invariant self.lo.value <= self.hi.value:
    var lo: Nat
    var hi: Nat

    def __init__(out self, lo: Nat, hi: Nat) requires lo.value <= hi.value:
        self.lo = lo
        self.hi = hi


def use(b: Bounds) -> Int:
    return b.hi - b.lo


def reads(p: Pointer[Bounds, MutUntrackedOrigin]) -> Int
    requires p._extent() >= 1:
    return p[].lo


def use_nat(n: Nat) -> Int:
    return n.value


def address_of(ref b: Bounds) -> Pointer[Bounds, MutUntrackedOrigin]:
    return (
        Pointer(to=b)
        .unsafe_mut_cast[True]()
        .unsafe_origin_cast[MutUntrackedOrigin]()
    )


# --- must be PROVEN ---
def ok_two_steps_local() -> Int:
    var b = Bounds(0, 5)
    b.lo = 10
    b.hi = 20
    return b.hi


def ok_two_steps_argument(mut b: Bounds):
    b.lo = 10
    b.hi = 20


def ok_two_steps_computed(mut b: Bounds, a: Int):
    if -1000 < a and a < 1000:
        b.lo = a
        b.hi = a + 1


def ok_two_steps_pointer(p: Pointer[Bounds, MutUntrackedOrigin])
    requires p._extent() >= 1:
    p[].lo = 10
    p[].hi = 20


def ok_two_steps_element(mut bs: List[Bounds]):
    if len(bs) > 0:
        bs[0].lo = 10
        bs[0].hi = 20


def ok_then_used(mut b: Bounds) -> Int:
    b.lo = 10
    b.hi = 20
    return use(b)


def ok_field_replaced(mut nb: NatBounds, n: Nat):
    if n.value <= nb.hi.value:
        nb.lo = n


def ok_method_on_field(mut nb: NatBounds):
    nb.lo.set(1000)
    nb.hi.set(1000)


def ok_part_used(mut nb: NatBounds) -> Int:
    nb.lo.set(1000)
    var t = use_nat(nb.lo)  # the part is a value of its own
    nb.hi.set(1000)
    return t


def ok_both_arms(mut b: Bounds, c: Bool):
    b.lo = 10
    if c:
        b.hi = 20
    else:
        b.hi = 30


def ok_in_one_arm(mut b: Bounds, c: Bool):
    if c:
        b.lo = 10
        b.hi = 20


def ok_already_enough(mut b: Bounds, c: Bool):
    if b.hi >= 10:
        b.lo = 10
        if c:
            b.hi = 20


def ok_across_loop(mut b: Bounds, n: Int) -> Int:
    var t = 0
    b.lo = 10
    for i in range(n):
        t = t + i
    b.hi = 20
    return t


def ok_each_iteration(mut b: Bounds, n: Int):
    if n < 1000:
        for i in range(n):
            b.lo = i
            b.hi = i + 1


def ok_pointer_one_arm(p: Pointer[Bounds, MutUntrackedOrigin], c: Bool)
    requires p._extent() >= 1:
    if c:
        p[].lo = 10
        p[].hi = 20


def ok_pointer_both_arms(p: Pointer[Bounds, MutUntrackedOrigin], c: Bool)
    requires p._extent() >= 1:
    p[].lo = 10
    if c:
        p[].hi = 20
    else:
        p[].hi = 30


# --- must be WARNED (at the last write before the value is seen) ---
def bad_left_broken(mut b: Bounds):
    b.lo = 10  # `hi` may be less


def bad_local_left_broken() -> Bounds:
    var b = Bounds(0, 5)
    b.lo = 10
    return b


def bad_used_between(mut b: Bounds) -> Int:
    b.lo = 10
    var t = use(b)  # sees it broken
    b.hi = 20
    return t


def bad_copied_between(mut b: Bounds) -> Bounds:
    b.lo = 10
    var c = b  # a copy of it broken
    b.hi = 20
    return c


def bad_branch_between(mut b: Bounds, c: Bool):
    b.lo = 10
    if c:
        b.hi = 20


def bad_read_between(
    p: Pointer[Bounds, MutUntrackedOrigin],
    q: Pointer[Bounds, MutUntrackedOrigin],
) -> Int
    requires p._extent() >= 1 and q._extent() >= 1:
    p[].lo = 10
    var t = q[].hi - q[].lo  # may be the same element, broken
    p[].hi = 20
    return t


def bad_call_reads_between(p: Pointer[Bounds, MutUntrackedOrigin]) -> Int
    requires p._extent() >= 1:
    p[].lo = 10
    var t = reads(p)
    p[].hi = 20
    return t


def bad_other_element_between(mut bs: List[Bounds], i: Int) -> Int:
    var t = 0
    if len(bs) > 0 and 0 <= i and i < len(bs):
        bs[0].lo = 10
        t = bs[i].hi - bs[i].lo  # may be element 0
        bs[0].hi = 20
    return t


def bad_field_replaced(mut nb: NatBounds, n: Nat):
    nb.lo = n  # may pass `hi`


def bad_field_replaced_local(n: Nat) -> Int:
    var nb = NatBounds(Nat(0), Nat(5))
    nb.lo = n
    return nb.hi.value - nb.lo.value


def bad_method_on_field(mut nb: NatBounds):
    nb.lo.set(1000)  # may pass `hi`


def bad_in_loop(mut b: Bounds, n: Int):
    for i in range(n):
        b.lo = i


def bad_part_used(mut nb: NatBounds) -> Int:
    nb.lo.value = -5
    var t = use_nat(nb.lo)  # the part is broken itself
    nb.lo.value = 0
    return t


def bad_pointer_left_broken(p: Pointer[Bounds, MutUntrackedOrigin])
    requires p._extent() >= 1:
    p[].lo = 10


def bad_seen_through_pointer() -> Int:
    var b = Bounds(0, 5)
    var p = address_of(b)
    b.lo = 10
    var t = reads(p)  # reads `b` through the pointer
    b.hi = 20
    return t


def bad_one_arm_only(mut b: Bounds, c: Bool):
    b.lo = 10
    if c:
        b.hi = 20  # not when `c` is false


def bad_arm_leaves_broken(mut b: Bounds, c: Bool):
    if c:
        b.lo = 10
    else:
        b.lo = 10
        b.hi = 20


def bad_iteration_leaves_broken(mut b: Bounds, n: Int):
    for i in range(n):
        b.lo = i
    b.hi = n


def bad_used_in_loop(mut b: Bounds, n: Int) -> Int:
    var t = 0
    b.lo = 10
    for _ in range(n):
        t = t + use(b)
    b.hi = 20
    return t


def bad_pointer_arm(p: Pointer[Bounds, MutUntrackedOrigin], c: Bool)
    requires p._extent() >= 1:
    if c:
        p[].lo = 10


def bad_pointer_one_arm_only(p: Pointer[Bounds, MutUntrackedOrigin], c: Bool)
    requires p._extent() >= 1:
    p[].lo = 10
    if c:
        p[].hi = 20  # not when `c` is false


def bad_pointer_other_in_arm(
    p: Pointer[Bounds, MutUntrackedOrigin],
    q: Pointer[Bounds, MutUntrackedOrigin],
    c: Bool,
)
    requires p._extent() >= 1 and q._extent() >= 1:
    p[].lo = 10
    if c:
        q[].hi = 5  # may be the same element
    p[].hi = 20


def main():
    pass
