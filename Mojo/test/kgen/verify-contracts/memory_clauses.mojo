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
# Memory clauses for `verify-contracts`: `_allocated(p)`, `_initialized(p,
# lo, hi)` and `_uninitialized(p, lo, hi)` in the `requires` and `ensures`
# of a function that takes a pointer. Inside it the memory behind the
# argument is tracked from an unknown state the preconditions constrain,
# and the postconditions are proven; at a call, the preconditions are
# proven of memory the caller allocated and the postconditions assumed of
# it. See README.md.


from std.builtin._verification import (
    _allocated,
    _initialized,
    _same_elements,
    _uninitialized,
)
from std.memory import (
    Allocation,
    Layout,
    alloc,
    dealloc,
    unsafe_uninit_move_n,
)
from std.memory.alloc import unsafe_alloc


def fill(p: Pointer[Int, MutUntrackedOrigin], n: Int)
    requires 0 <= n < 1000 and _allocated(p) and p._extent() >= n
    requires _uninitialized(p, 0, n)
    ensures _initialized(p, 0, n):
    for i in range(n):
        p.unsafe_offset(i).unsafe_write(i)


def total(p: Pointer[Int, MutUntrackedOrigin], n: Int) -> Int
    requires 0 <= n < 1000 and _allocated(p) and p._extent() >= n
    requires _initialized(p, 0, n):
    var t = 0
    for i in range(n):
        t += p[unsafe_offset=i]
    return t


def release(p: Pointer[Int, MutUntrackedOrigin])
    requires _allocated(p)
    ensures not _allocated(p):
    p.unsafe_free()


def destroy_lists(p: Pointer[List[Int], MutUntrackedOrigin], n: Int)
    requires 0 <= n < 1000 and _allocated(p) and p._extent() >= n
    requires _initialized(p, 0, n) and _uninitialized(p, n, p._extent())
    ensures _uninitialized(p, 0, n):
    for i in range(n):
        p.unsafe_offset(i).unsafe_deinit_pointee()


# --- must be PROVEN ---
def ok_filled_by_contract(n: Int) -> Int requires 0 <= n < 1000:
    var p = unsafe_alloc[Int](n)
    fill(p, n)
    var t = total(p, n)
    p.unsafe_free()
    return t


def ok_released_by_callee():
    var p = unsafe_alloc[Int](1)
    p.unsafe_write(1)
    release(p)


def ok_lists_destroyed_by_contract(n: Int) requires 0 <= n < 1000:
    var p = unsafe_alloc[List[Int]](n)
    for i in range(n):
        p.unsafe_offset(i).unsafe_write(List[Int]())
    destroy_lists(p, n)
    p.unsafe_free()


# --- must stay UNPROVEN ---
def bad_read_by_contract(n: Int) -> Int requires 0 <= n < 1000:
    var p = unsafe_alloc[Int](n)
    var t = total(p, n)  # nothing written yet
    p.unsafe_free()
    return t


def bad_use_after_release() -> Int:
    var p = unsafe_alloc[Int](1)
    p.unsafe_write(1)
    release(p)
    return p[]  # freed by `release`


def bad_fill_short(p: Pointer[Int, MutUntrackedOrigin], n: Int)
    requires 1 <= n < 1000 and _allocated(p) and p._extent() >= n
    requires _uninitialized(p, 0, n)
    ensures _initialized(p, 0, n):
    for i in range(n - 1):
        p.unsafe_offset(i).unsafe_write(i)  # one short


def bad_lists_leaked_by_contract(n: Int) requires 1 <= n < 1000:
    var p = unsafe_alloc[List[Int]](n)
    for i in range(n):
        p.unsafe_offset(i).unsafe_write(List[Int]())
    destroy_lists(p, n - 1)
    p.unsafe_free()  # the last one is leaked


def bad_fill_on_initialized(n: Int) requires 1 <= n < 1000:
    var p = unsafe_alloc[Int](n)
    p.unsafe_write(7)
    fill(p, n)  # element 0 is not uninitialized
    p.unsafe_free()


# --- the frame of a call: what a callee's preconditions name is all it
# touches; the rest is as the caller left it ---
def fill3(p: Pointer[List[Int], MutUntrackedOrigin])
    requires _allocated(p) and p._extent() >= 3 and _uninitialized(p, 0, 3)
    ensures _initialized(p, 0, 3):
    p.unsafe_offset(0).unsafe_write(List[Int]())
    p.unsafe_offset(1).unsafe_write(List[Int]())
    p.unsafe_offset(2).unsafe_write(List[Int]())


def destroy_tail(p: Pointer[List[Int], MutUntrackedOrigin])
    requires _allocated(p) and p._extent() >= 3
    requires _initialized(p, 1, 3) and _uninitialized(p, 3, p._extent())
    ensures _uninitialized(p, 1, p._extent()):
    p.unsafe_offset(2).unsafe_deinit_pointee()  # the last each time
    p.unsafe_offset(1).unsafe_deinit_pointee()


def ok_shift_down(p: Pointer[List[Int], MutUntrackedOrigin])
    requires _allocated(p) and _initialized(p, 0, 3)
    requires _uninitialized(p, 3, p._extent())
    ensures _initialized(p, 0, 2) and _uninitialized(p, 2, p._extent())
    ensures _same_elements(p, old(p).unsafe_offset(1), 2):
    p.unsafe_deinit_pointee()
    # The two above the gap move down over it; the last is moved out.
    unsafe_uninit_move_n[overlapping=True](
        dest=p, src=p.unsafe_offset(1), count=2
    )


def bad_shift_claims_kept(p: Pointer[List[Int], MutUntrackedOrigin])
    requires _allocated(p) and _initialized(p, 0, 3)
    requires _uninitialized(p, 3, p._extent())
    ensures _same_elements(p, old(p), 2):  # they moved down by one
    p.unsafe_deinit_pointee()
    unsafe_uninit_move_n[overlapping=True](
        dest=p, src=p.unsafe_offset(1), count=2
    )


def ok_cut_last(p: Pointer[List[Int], MutUntrackedOrigin])
    requires _allocated(p) and p._extent() >= 3 and _uninitialized(p, 0, 3)
    ensures _uninitialized(p, 2, 3):
    fill3(p)
    p.unsafe_offset(2).unsafe_deinit_pointee()


def ok_head_kept_by_callee():
    var p = unsafe_alloc[List[Int]](3)
    fill3(p)
    destroy_tail(p)  # names elements 1 to 3: element 0 is as it was
    p.unsafe_deinit_pointee()
    p.unsafe_free()


def bad_cut_claims_tail(p: Pointer[List[Int], MutUntrackedOrigin])
    requires _allocated(p) and p._extent() >= 3 and _uninitialized(p, 0, 3)
    ensures _uninitialized(p, 1, 3):
    fill3(p)
    p.unsafe_offset(1).unsafe_deinit_pointee()  # element 2 stays initialized


def bad_head_leaked_after_callee():
    var p = unsafe_alloc[List[Int]](3)
    fill3(p)
    destroy_tail(p)
    p.unsafe_free()  # element 0 is still initialized


# --- a struct whose invariant states its memory: the elements below its
# length are initialized, the rest of its storage is not ---
struct Strs
    invariant self._len >= 0
    invariant self._len <= len(self._storage)
    invariant self._storage.unsafe_ptr()._extent() >= len(self._storage)
    invariant _allocated(self._storage.unsafe_ptr())
    invariant _initialized(self._storage.unsafe_ptr(), 0, self._len)
    invariant _uninitialized(
        self._storage.unsafe_ptr(),
        self._len,
        self._storage.unsafe_ptr()._extent(),
    ):
    var _storage: Allocation[String]
    var _len: Int

    def __init__(out self, capacity: Int)
        requires capacity > 0
        ensures self._len == 0 and len(self._storage) == capacity:
        self._storage = alloc(Layout[String](length=capacity))
        self._len = 0

    def ok_push(mut self, var s: String)
        requires self._len < len(self._storage)
        ensures self._len == old(self._len) + 1:
        self._storage.storage()._unchecked_get(self._len).unsafe_write(s^)
        self._len += 1

    def ok_pop(mut self) -> String
        requires self._len > 0
        ensures self._len == old(self._len) - 1:
        self._len -= 1
        return (
            self._storage.unsafe_ptr()
            .unsafe_offset(self._len)
            .unsafe_take_pointee()
        )

    def ok_clear(mut self):
        Span(
            unsafe_ptr=self._storage.unsafe_ptr(), length=self._len
        ).unsafe_deinit_elements()
        self._len = 0

    def ok_remove(mut self, i: Int) -> String
        requires 0 <= i < self._len
        ensures self._len == old(self._len) - 1:
        # Untracked: `dest` and `src` alias.
        var data = self._storage.unsafe_ptr().unsafe_origin_cast[
            MutUntrackedOrigin
        ]()
        var s = data.unsafe_offset(i).unsafe_take_pointee()
        # The elements above the gap move down over it.
        unsafe_uninit_move_n[overlapping=True](
            dest=data.unsafe_offset(i),
            src=data.unsafe_offset(i + 1),
            count=self._len - i - 1,
        )
        self._len -= 1
        return s^

    def ok_insert(mut self, i: Int, var s: String)
        requires 0 <= i <= self._len and self._len < len(self._storage)
        ensures self._len == old(self._len) + 1
        ensures len(self._storage) == old(len(self._storage)):
        var data = self._storage.unsafe_ptr().unsafe_origin_cast[
            MutUntrackedOrigin
        ]()
        # The elements from `i` move up, leaving a gap at `i`.
        unsafe_uninit_move_n[overlapping=True](
            dest=data.unsafe_offset(i + 1),
            src=data.unsafe_offset(i),
            count=self._len - i,
        )
        data.unsafe_offset(i).unsafe_write(s^)
        self._len += 1

    def __deinit__(deinit self):
        Span(
            unsafe_ptr=self._storage.unsafe_ptr(), length=self._len
        ).unsafe_deinit_elements()
        dealloc(self._storage^)  # every element destroyed

    def bad_pop_copy(mut self) -> String requires self._len > 0:
        self._len -= 1
        # A copy of the element: it stays initialized above `_len`.
        return (
            self._storage.storage()
            ._unchecked_get(self._len)
            .unsafe_assume_init()
        )


    def bad_remove_leaves_gap(mut self, i: Int) -> String
        requires 0 <= i < self._len:
        var data = self._storage.unsafe_ptr()
        var s = data.unsafe_offset(i).unsafe_take_pointee()
        self._len -= 1  # the elements above `i` were not moved down
        return s^

    def bad_insert_no_gap(mut self, i: Int, var s: String)
        requires 0 <= i < self._len and self._len < len(self._storage):
        var data = self._storage.unsafe_ptr()
        data.unsafe_offset(i).unsafe_write(s^)  # nothing moved up
        self._len += 1  # the last element is not initialized


def ok_strs() -> Int:
    var v = Strs(capacity=3)
    v.ok_insert(0, "b")
    v.ok_push("a")
    var t = v.ok_remove(1)
    var s = v.ok_pop()
    v.ok_clear()
    return s.byte_length() + t.byte_length()
