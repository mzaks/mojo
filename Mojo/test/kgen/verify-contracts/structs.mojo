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
# Structs that hold a pointer and its bookkeeping, for `verify-contracts`:
# what is known about one field while another is written; see README.md.

from std.memory import AddressSpace


struct Ring(TrivialRegisterPassable):
    var data: Pointer[Int, MutUntrackedOrigin]
    var size: Int
    var at: Int


struct Slots[size: Int](TrivialRegisterPassable):
    comptime ptr_type = Pointer[Int, MutUntrackedOrigin]
    comptime Storage = Array[Int, Self.size]
    var ptr: Self.ptr_type

    def __init__(
        out self where self.ptr._extent() >= unsafe_ptr._extent(),
        unsafe_ptr: Self.ptr_type,
    ):
        self.ptr = unsafe_ptr

    def __init__(
        ref storage: Self.Storage,
        out result: Self where result.ptr._extent() >= Self.size,
    ):
        # `rebind` is the same pointer; the array's has `size` elements.
        result = Self(rebind[Self.ptr_type](storage.unsafe_ptr()))

    def get[
        T: Intable
    ](
        self,
        index: T where (
            0 <= Int(index) < Self.size and self.ptr._extent() >= Self.size
        ),
    ) -> Int:
        # `Int(index)` is the clause's, whatever `T` is.
        return self.ptr[unsafe_offset=Int(index)]


struct Holder[size: Int]:
    var items: Array[Int, Self.size]


struct Buffer:
    var data: Pointer[Int, MutUntrackedOrigin]
    var size: Int
    var other: Int

    def rewind(mut self):
        self.other = 0

    def shrink(mut self):
        self.size = self.size - 1


def fill_shared(
    dst: Pointer[Int, MutUntrackedOrigin, address_space=AddressSpace.SHARED]
):
    pass


@always_inline
def shift_by(a: Int, s: Int) -> Int:
    # No clauses and one `return`: evaluated where it is called.
    return a >> s if s > 0 else a << -s


struct Mixer(TrivialRegisterPassable):
    """Moves the `high` bits onto the `low` ones: permutes within blocks."""

    var high: Int
    var low: Int
    var shift: Int

    def __init__(out self, shift: Int):
        self.high = 56 << shift
        self.low = 56
        self.shift = shift

    def ok_mix(
        self,
        offset: Int,
        out result: Int where (
            not (0 < self.shift < 64 and (self.high >> self.shift) == self.low)
            or (result & ~self.low) == (offset & ~self.low)
        ),
    ):
        result = offset ^ shift_by(offset & self.high, self.shift)

    def bad_mix(
        self,
        offset: Int,
        out result: Int where (result & ~self.low) == (offset & ~self.low),
    ):
        # `high >> shift` may reach bits outside `low`.
        result = offset ^ shift_by(offset & self.high, self.shift)


def make_mixer[shift: Int]() -> Mixer:
    return Mixer(shift)


def _below(x: Int, n: Int where 0 <= x < n) -> Int:
    return x


def _take[
    fill: Optional[Int] = None
](
    p: Pointer[Int, MutUntrackedOrigin],
    n: Int where p._extent() >= (0 if fill else n),
):
    pass


struct Filler[n: Int]:
    """Offsets computed once (`prepare`) and read in a loop later."""

    comptime mixer = make_mixer[4]()
    var base: Int
    var offsets: Array[Int, Self.n]

    def __init__(out self where self.base == base, base: Int):
        self.base = base
        self.offsets = Array[Int, Self.n](uninitialized=True)

    def ok_prepare(
        mut self where old(0 <= self.base < 64) where (
            self.base == old(self.base)
            and all([0 <= self.offsets[v] < 64 * Self.n for v in range(Self.n)])
        )
    ):
        comptime assert 1 <= Self.n <= 65536
        # What the compiler checks of the comptime value, the clause of
        # `ok_mix` reads as the fields of the same value.
        comptime assert Self.mixer.low == 56 and 0 < Self.mixer.shift < 64
        comptime assert (Self.mixer.high >> Self.mixer.shift) == 56
        # Each iteration stores its own element, from `v` alone.
        comptime for v in range(Self.n):
            self.offsets[v] = Self.mixer.ok_mix(self.base + 64 * v)

    def bad_prepare_any_base(
        mut self where (
            self.base == old(self.base)
            and all([0 <= self.offsets[v] < 64 * Self.n for v in range(Self.n)])
        )
    ):
        comptime assert 1 <= Self.n <= 65536
        comptime assert Self.mixer.low == 56 and 0 < Self.mixer.shift < 64
        comptime assert (Self.mixer.high >> Self.mixer.shift) == 56
        comptime for v in range(Self.n):
            self.offsets[v] = Self.mixer.ok_mix(self.base + 64 * v)

    def bad_prepare_unchecked_mixer(
        mut self where old(0 <= self.base < 64) where (
            self.base == old(self.base)
            and all([0 <= self.offsets[v] < 64 * Self.n for v in range(Self.n)])
        )
    ):
        comptime assert 1 <= Self.n <= 65536
        comptime for v in range(Self.n):
            self.offsets[v] = Self.mixer.ok_mix(self.base + 64 * v)

    def bad_prepare_half(
        mut self where (
            self.base == old(self.base)
            and all([self.offsets[v] == v for v in range(Self.n)])
        )
    ):
        comptime for v in range(Self.n // 2):
            self.offsets[v] = v

    def bad_prepare_chained(
        mut self where (
            self.base == old(self.base)
            and all([self.offsets[v] == 1 for v in range(Self.n)])
        )
    ):
        # From an element an earlier iteration stored: 1, then 2s.
        self.offsets[0] = 0
        comptime for v in range(Self.n):
            self.offsets[v] = self.offsets[0] + 1

    def ok_sum(
        self,
        p: Pointer[Int, MutUntrackedOrigin] where (
            p._extent() >= 64 * Self.n
            and all([0 <= self.offsets[v] < 64 * Self.n for v in range(Self.n)])
        ),
    ) -> Int:
        var t = 0
        comptime for v in range(Self.n):
            # The offset is read through a reference lent immutably.
            t += p.unsafe_offset(self.offsets[v])[]
        return t

    def bad_sum(
        self, p: Pointer[Int, MutUntrackedOrigin] where p._extent() >= 64 * Self.n
    ) -> Int:
        var t = 0
        comptime for v in range(Self.n):
            t += p.unsafe_offset(self.offsets[v])[]  # nothing bounds the offsets
        return t


# --- must be PROVEN ---
def ok_field_store(
    mut r: Ring,
    i: Int where 0 <= i < r.size and r.data._extent() >= r.size,
) -> Int:
    r.at = i  # `r.data` and `r.size` are what they were
    return r.data[unsafe_offset=r.at]


def ok_field_loop(
    mut r: Ring,
    n: Int where (
        1 <= r.size <= 65536
        and 0 <= r.at < r.size
        and r.data._extent() >= r.size
    ),
) -> Int:
    var sum = 0
    for _ in range(n):
        sum += r.data[unsafe_offset=r.at]  # only `r.at` changes in the loop
        var next = r.at + 1
        r.at = 0 if next == r.size else next
    return sum


def ok_after_branch(
    mut r: Ring,
    i: Int,
    reset: Bool where 0 <= i < r.size and r.data._extent() >= r.size,
) -> Int:
    if reset:
        r.at = 0
    return r.data[unsafe_offset=i]  # the same `r.data` on both paths


def _get(
    r: Ring, i: Int where 0 <= i < r.size and r.data._extent() >= r.size
) -> Int:
    return r.data[unsafe_offset=i]


def ok_by_value(
    mut r: Ring, i: Int where 0 <= i < r.size and r.data._extent() >= r.size
) -> Int:
    return _get(r, i)  # the callee reads the fields of the value


def ok_slots(ref storage: Array[Int, 4]) -> Int:
    var slots = Slots[4](storage)
    return slots.get(3)


def ok_array_field[size: Int](h: Holder[size]) -> Int:
    comptime assert size >= 1
    # An array that is a field: its pointer has the array's length.
    return h.items.unsafe_ptr()[unsafe_offset=size - 1]


def ok_shared_write(
    mut r: Ring,
    dst: Pointer[Int, MutUntrackedOrigin, address_space=AddressSpace.SHARED],
    i: Int where 0 <= i < r.size and r.data._extent() >= r.size,
) -> Int:
    r.at = i
    fill_shared(dst)  # writes shared memory, not `r`
    return r.data[unsafe_offset=r.at]


def ok_after_method(
    mut b: Buffer, i: Int where 0 <= i < b.size and b.data._extent() >= b.size
) -> Int:
    b.rewind()  # writes `b.other` only
    return b.data[unsafe_offset=i]


def ok_borrowed(
    b: Buffer,
    q: Pointer[Int, MutAnyOrigin],
    i: Int where (
        0 <= i < b.size and b.data._extent() >= b.size and q._extent() >= 1
    ),
) -> Int:
    q[unsafe_offset=0] = 7  # may write anything but what is borrowed
    return b.data[unsafe_offset=i]


def ok_chosen[
    wide: Bool
](
    b: Buffer,
    p: Pointer[Int, MutAnyOrigin] where (
        1 <= b.size <= 1000
        and 1 <= b.other <= 1000
        and p._extent() >= b.size + b.other
    ),
) -> Int:
    var n = b.size if wide else b.other  # a field chosen at compile time
    return p[unsafe_offset=n - 1]


def ok_bitcast(
    p: Pointer[Int64, MutAnyOrigin], i: Int where 0 <= i < 8 and p._extent() >= 4
) -> Int32:
    var q = p.unsafe_bitcast[Int32]()  # four 8-byte elements: eight of 4
    return q[unsafe_offset=i]


def ok_prepared(base: Int where 0 <= base < 64) -> Int:
    var f = Filler[8](base)
    f.ok_prepare()  # writes `offsets` only: `base` is as constructed
    return _below(f.offsets[3], 512) + _below(f.base, 64)


def ok_filled(p: Pointer[Int, MutUntrackedOrigin]):
    _take[fill=0](p, 5)  # a fill value: nothing is read


# --- must stay UNPROVEN ---
def bad_after_method(
    mut b: Buffer, i: Int where 0 <= i < b.size and b.data._extent() >= b.size
) -> Int:
    b.shrink()
    return b.data[unsafe_offset=b.size]  # `size` changed: may be -1 + 1...


def bad_mutable(
    mut b: Buffer,
    q: Pointer[Int, MutAnyOrigin],
    i: Int where (
        0 <= i < b.size and b.data._extent() >= b.size and q._extent() >= 1
    ),
) -> Int:
    q[unsafe_offset=0] = 7  # `b` is not borrowed immutably: may be `b`
    return b.data[unsafe_offset=i]


def bad_chosen[
    wide: Bool
](
    b: Buffer,
    p: Pointer[Int, MutAnyOrigin] where (
        1 <= b.size <= 1000 and 1 <= b.other <= 1000 and p._extent() >= b.size
    ),
) -> Int:
    var n = b.size if wide else b.other
    return p[unsafe_offset=n - 1]  # `other` may be past the extent


def bad_bitcast(
    p: Pointer[Int64, MutAnyOrigin], i: Int where 0 <= i <= 8 and p._extent() >= 4
) -> Int32:
    var q = p.unsafe_bitcast[Int32]()
    return q[unsafe_offset=i]  # the ninth


def bad_field_store(
    mut r: Ring,
    i: Int where 0 <= i < r.size and r.data._extent() >= r.size,
) -> Int:
    r.at = i + 1  # may be `r.size`
    return r.data[unsafe_offset=r.at]


def bad_field_loop(
    mut r: Ring,
    n: Int where (
        1 <= r.size <= 65536
        and 0 <= r.at < r.size
        and r.data._extent() >= r.size
    ),
) -> Int:
    var sum = 0
    for _ in range(n):
        sum += r.data[unsafe_offset=r.at]
        r.at = r.at + 1  # never wraps
    return sum


def bad_after_branch(
    mut r: Ring,
    i: Int,
    reset: Bool where 0 <= i < r.size and r.data._extent() >= r.size,
) -> Int:
    if reset:
        r.size = 0
    return r.data[unsafe_offset=r.size - 1]  # -1 on one path


def bad_slots(ref storage: Array[Int, 4]) -> Int:
    var slots = Slots[4](storage)
    return slots.get(4)


def bad_array_field[size: Int](h: Holder[size]) -> Int:
    return h.items.unsafe_ptr()[unsafe_offset=size]  # one past the end


def bad_unprepared(base: Int where 0 <= base < 64) -> Int:
    var f = Filler[8](base)
    return _below(f.offsets[3], 512)  # never stored


def bad_unfilled(p: Pointer[Int, MutUntrackedOrigin]):
    _take(p, 5)  # no fill value, and nothing about `p`


def main():
    pass
