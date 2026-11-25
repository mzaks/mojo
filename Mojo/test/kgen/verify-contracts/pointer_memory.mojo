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
# Memory behind raw pointers for `verify-contracts`: what is read at an
# address is the same value until something may have written it; a write
# through a pointer keeps only what it wrote (and the other fields of the
# element, and the other elements of the same pointer); see README.md.


struct R2(ImplicitlyCopyable, Movable):
    var lo: Int
    var hi: Int

    def __init__(out self, lo: Int, hi: Int):
        self.lo = lo
        self.hi = hi


struct View(ImplicitlyCopyable, Movable):
    var data: Pointer[Int, MutUntrackedOrigin]

    def __init__(out self, data: Pointer[Int, MutUntrackedOrigin])
        ensures self.data._extent() == data._extent():
        self.data = data

    def poke(self) requires self.data._extent() >= 2:
        self.data[] = 99999


def writes(p: Pointer[Int, MutUntrackedOrigin]) requires p._extent() >= 2:
    p[] = 99999


def bumps(mut x: Int):
    x = 99999


def reads(p: Pointer[Int, MutUntrackedOrigin]) -> Int requires p._extent() >= 2:
    return p[]


trait Poker:
    def poke(self):
        ...


def address_of(ref x: Int) -> Pointer[Int, MutUntrackedOrigin]:
    return (
        Pointer(to=x)
        .unsafe_mut_cast[True]()
        .unsafe_origin_cast[MutUntrackedOrigin]()
    )


struct Holder(Movable):
    var p: Pointer[Int, MutUntrackedOrigin]

    def __init__(out self, mut x: Int):
        self.p = address_of(x)


def set_to(mut x: Int, v: Int):
    x = v


# --- must be PROVEN ---
def ok_two_reads(p: Pointer[Int, MutUntrackedOrigin], xs: List[Int]) -> Int
    requires p._extent() >= 2:
    if 0 <= p[] and p[] < len(xs):
        return xs[p[]]
    return 0


def ok_pure_call(p: Pointer[Int, MutUntrackedOrigin], xs: List[Int]) -> Int
    requires p._extent() >= 2:
    if 0 <= p[] and p[] < len(xs):
        var t = reads(p) + len(xs)
        return xs[p[]] + t
    return 0


def ok_written(p: Pointer[Int, MutUntrackedOrigin], xs: List[Int]) -> Int
    requires p._extent() >= 2:
    if len(xs) > 3:
        p[] = 3
        return xs[p[]]
    return 0


def ok_other_field(p: Pointer[R2, MutUntrackedOrigin], xs: List[Int]) -> Int
    requires p._extent() >= 2:
    if 0 <= p[].hi and p[].hi < len(xs):
        p[].lo = 7
        return xs[p[].hi]
    return 0


def ok_other_element(p: Pointer[Int, MutUntrackedOrigin], xs: List[Int]) -> Int
    requires p._extent() >= 2:
    if 0 <= p[unsafe_offset=0] and p[unsafe_offset=0] < len(xs):
        p[unsafe_offset=1] = 99999
        return xs[p[unsafe_offset=0]]
    return 0


def ok_loop_reads(p: Pointer[Int, MutUntrackedOrigin], xs: List[Int]) -> Int
    requires p._extent() >= 2:
    var t = 0
    if 0 <= p[] and p[] < len(xs):
        for _ in range(4):
            t = t + xs[p[]]
    return t


def ok_counter(p: Pointer[Int, MutUntrackedOrigin], xs: List[Int]) -> Int
    requires p._extent() >= 2:
    var t = 0
    var i = 0
    if 0 <= p[] and p[] < len(xs):
        while i < 4:
            t += xs[p[]]
            i += 1
    return t


def ok_local_call(p: Pointer[Int, MutUntrackedOrigin], xs: List[Int]) -> Int
    requires p._extent() >= 2:
    var x = 0
    if 0 <= p[] and p[] < len(xs):
        set_to(x, 5)
        return xs[p[]] + x
    return 0


# --- must stay UNPROVEN ---
def bad_other_pointer(
    p: Pointer[Int, MutUntrackedOrigin],
    q: Pointer[Int, MutUntrackedOrigin],
    xs: List[Int],
) -> Int
    requires p._extent() >= 2 and q._extent() >= 2:
    if 0 <= p[] and p[] < len(xs):
        q[] = 99999
        return xs[p[]]
    return 0


def bad_other_type(
    p: Pointer[Int, MutUntrackedOrigin],
    q: Pointer[R2, MutUntrackedOrigin],
    xs: List[Int],
) -> Int
    requires p._extent() >= 2 and q._extent() >= 2:
    if 0 <= p[] and p[] < len(xs):
        q[].hi = 99999
        return xs[p[]]
    return 0


def bad_callee_writes(p: Pointer[Int, MutUntrackedOrigin], xs: List[Int]) -> Int
    requires p._extent() >= 2:
    if 0 <= p[] and p[] < len(xs):
        writes(p)
        return xs[p[]]
    return 0


def bad_mut_argument(p: Pointer[Int, MutUntrackedOrigin], xs: List[Int]) -> Int
    requires p._extent() >= 2:
    if 0 <= p[] and p[] < len(xs):
        bumps(p[])
        return xs[p[]]
    return 0


def bad_through_view(p: Pointer[Int, MutUntrackedOrigin], xs: List[Int]) -> Int
    requires p._extent() >= 2:
    var v = View(p)
    if 0 <= p[] and p[] < len(xs):
        v.poke()
        return xs[p[]]
    return 0


def bad_generic[
    T: Poker
](p: Pointer[Int, MutUntrackedOrigin], xs: List[Int], t: T) -> Int
    requires p._extent() >= 2:
    if 0 <= p[] and p[] < len(xs):
        t.poke()
        return xs[p[]]
    return 0


def bad_loop_writes(
    p: Pointer[Int, MutUntrackedOrigin], xs: List[Int], n: Int
) -> Int
    requires p._extent() >= 2:
    if 0 <= p[] and p[] < len(xs):
        for i in range(n):
            p[] = i
        return xs[p[]]
    return 0


def bad_in_loop(
    p: Pointer[Int, MutUntrackedOrigin],
    q: Pointer[Int, MutUntrackedOrigin],
    xs: List[Int],
) -> Int
    requires p._extent() >= 2 and q._extent() >= 2:
    var t = 0
    if 0 <= p[] and p[] < len(xs):
        for _ in range(4):
            t += xs[p[]]
            q[] = 99999
    return t


def bad_pointer_changed(
    p: Pointer[Int, MutUntrackedOrigin],
    r: Pointer[Int, MutUntrackedOrigin],
    xs: List[Int],
) -> Int
    requires p._extent() >= 2 and r._extent() >= 2:
    var q = p
    if 0 <= q[] and q[] < len(xs):
        q = r
        return xs[q[]]
    return 0


def bad_index_changed(
    p: Pointer[Int, MutUntrackedOrigin], xs: List[Int], k: Int
) -> Int
    requires p._extent() >= 2:
    var i = k
    if 0 <= p[unsafe_offset=i] and p[unsafe_offset=i] < len(xs):
        i += 1
        return xs[p[unsafe_offset=i]]
    return 0


def bad_unknown_index(
    p: Pointer[Int, MutUntrackedOrigin], xs: List[Int], k: Int
) -> Int
    requires p._extent() >= 2:
    if 0 <= p[unsafe_offset=0] and p[unsafe_offset=0] < len(xs):
        p[unsafe_offset=k] = 99999
        return xs[p[unsafe_offset=0]]
    return 0


def bad_branch(
    p: Pointer[Int, MutUntrackedOrigin], xs: List[Int], c: Bool
) -> Int
    requires p._extent() >= 2:
    if 0 <= p[] and p[] < len(xs):
        if c:
            p[] = 99999
        return xs[p[]]
    return 0


def bad_mut_alias(
    mut x: Int, p: Pointer[Int, MutUntrackedOrigin], xs: List[Int]
) -> Int
    requires p._extent() >= 2:
    if 0 <= p[] and p[] < len(xs):
        x = 99999
        return xs[p[]]
    return 0


def bad_local_address(xs: List[Int]) -> Int:
    var x = 0
    var p = Pointer(to=x)
    if 0 <= p[] and p[] < len(xs):
        x = 99999
        return xs[p[]]
    return 0


def bad_list_storage(mut ys: List[Int], xs: List[Int]) -> Int:
    if len(ys) > 0:
        var p = ys.unsafe_ptr()
        if 0 <= p[] and p[] < len(xs):
            ys[0] = 99999
            return xs[p[]]
    return 0


def bad_field_rewritten(
    p: Pointer[R2, MutUntrackedOrigin], xs: List[Int], a: Int
) -> Int
    requires p._extent() >= 2:
    if 0 <= p[].hi and p[].hi < len(xs):
        p[].hi = a
        return xs[p[].hi]
    return 0


def bad_whole_rewritten(
    p: Pointer[R2, MutUntrackedOrigin], xs: List[Int], r: R2
) -> Int
    requires p._extent() >= 2:
    if 0 <= p[].hi and p[].hi < len(xs):
        p[] = r
        return xs[p[].hi]
    return 0


def bad_overlap(
    p: Pointer[R2, MutUntrackedOrigin],
    q: Pointer[R2, MutUntrackedOrigin],
    xs: List[Int],
) -> Int
    requires p._extent() >= 2 and q._extent() >= 2:
    if 0 <= p[].hi and p[].hi < len(xs):
        q[].lo = 99999
        return xs[p[].hi]
    return 0


def bad_method_writes(p: Pointer[Int, MutUntrackedOrigin], xs: List[Int]) -> Int
    requires p._extent() >= 2:
    if 0 <= p[] and p[] < len(xs):
        p.unsafe_write(99999)
        return xs[p[]]
    return 0


# the address taken by a helper
def bad_helper_address(xs: List[Int]) -> Int:
    var x = 0
    var p = address_of(x)
    if 0 <= p[] and p[] < len(xs):
        x = 99999
        return xs[p[]]
    return 0


# ... through an `out` argument
def bad_out_address(xs: List[Int]) -> Int:
    var x = 0
    var h = Holder(x)
    var p = h.p
    if 0 <= p[] and p[] < len(xs):
        x = 99999
        return xs[p[]]
    return 0


# ... and written by a call
def bad_helper_write(xs: List[Int]) -> Int:
    var x = 0
    var p = address_of(x)
    if 0 <= p[] and p[] < len(xs):
        set_to(x, 99999)
        return xs[p[]]
    return 0


# ... and incremented
def bad_increment(xs: List[Int]) -> Int:
    var x = 0
    var p = address_of(x)
    if 0 <= p[] and p[] < len(xs):
        x += 99999
        return xs[p[]]
    return 0


def main():
    pass
