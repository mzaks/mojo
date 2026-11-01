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


from std.builtin._verification import _allocated, _initialized, _uninitialized
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
