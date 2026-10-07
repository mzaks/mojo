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


def fill_shared(
    dst: Pointer[Int, MutUntrackedOrigin, address_space=AddressSpace.SHARED]
):
    pass


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


# --- must stay UNPROVEN ---
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


def main():
    pass
