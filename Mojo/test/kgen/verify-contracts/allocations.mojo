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
# Allocations for `verify-contracts`: memory this function allocates must
# still be allocated when it is read, written or freed, and an element
# must be written before it is read; a pointer taken from a list must not
# outlive a reallocation of the list. See README.md.


from std.memory.alloc import unsafe_alloc


def fill(p: Pointer[Int, MutUntrackedOrigin], n: Int)
    requires p._extent() >= n:
    for i in range(n):
        p[unsafe_offset=i] = i


# --- must be PROVEN ---
def ok_read_after_write() -> Int:
    var p = unsafe_alloc[Int](1)
    p[] = 4
    var t = p[]
    p.unsafe_free()
    return t


def ok_list_pointee() -> Int:
    var p = unsafe_alloc[List[Int]](1)
    p.unsafe_write([1, 2, 3])
    var n = len(p[])
    p.unsafe_free()
    return n


def ok_any_order() -> Int:
    var p = unsafe_alloc[Int](3)
    p[unsafe_offset=2] = 2
    p[unsafe_offset=0] = 0
    p[unsafe_offset=1] = 1
    var t = p[unsafe_offset=2] + p[unsafe_offset=1]
    p.unsafe_free()
    return t


def ok_filled_in_loop(n: Int) -> Int:
    var p = unsafe_alloc[Int](n)
    for i in range(n):
        p[unsafe_offset=i] = i
    var t = 0
    for i in range(n):
        t += p[unsafe_offset=i]
    p.unsafe_free()
    return t


def ok_filled_by_callee(n: Int) -> Int:
    var p = unsafe_alloc[Int](n)
    fill(p, n)
    var t = 0
    if n > 0:
        t = p[]
    p.unsafe_free()
    return t


def ok_offset_pointer() -> Int:
    var p = unsafe_alloc[Int](4)
    var q = p.unsafe_offset(2)
    q[] = 7
    var t = p[unsafe_offset=2]
    p.unsafe_free()
    return t


def ok_store_load() -> Float32:
    var p = unsafe_alloc[Float32](8)
    p.unsafe_store(0, SIMD[DType.float32, 4](1))
    var v = p.unsafe_load[width=4](0)
    p.unsafe_free()
    return v[0]


def ok_take() -> Int:
    var p = unsafe_alloc[Int](1)
    p.unsafe_write(3)
    var x = p.unsafe_take_pointee()
    p.unsafe_write(4)
    var y = p[]
    p.unsafe_free()
    return x + y


def ok_list_pointer(mut xs: List[Int]) -> Int:
    if len(xs) > 0:
        var p = xs.unsafe_ptr()
        xs[0] = 5
        return p[]  # writing an element does not move the storage
    return 0


def ok_freed_in_branch(c: Bool) -> Int:
    var p = unsafe_alloc[Int](1)
    p[] = 1
    var t = p[]
    if c:
        p.unsafe_free()
    else:
        p.unsafe_free()
    return t


# --- must stay UNPROVEN ---
def bad_read_before_write() -> Int:
    var p = unsafe_alloc[Int](1)
    var t = p[]
    p.unsafe_free()
    return t


def bad_list_pointee() -> Int:
    var p = unsafe_alloc[List[Int]](1)
    var n = len(p[])  # uninitialized
    p.unsafe_write([1, 2, 3])
    p.unsafe_free()
    return n


def bad_other_element() -> Int:
    var p = unsafe_alloc[Int](3)
    p[unsafe_offset=0] = 0
    var t = p[unsafe_offset=1]
    p.unsafe_free()
    return t


def bad_after_free() -> Int:
    var p = unsafe_alloc[Int](1)
    p[] = 4
    p.unsafe_free()
    return p[]


def bad_double_free():
    var p = unsafe_alloc[Int](1)
    p.unsafe_free()
    p.unsafe_free()


def bad_after_take() -> Int:
    var p = unsafe_alloc[Int](1)
    p.unsafe_write(3)
    var x = p.unsafe_take_pointee()
    var y = p[]  # moved out
    p.unsafe_free()
    return x + y


def bad_partial_loop(n: Int) -> Int:
    var p = unsafe_alloc[Int](n)
    for i in range(n - 1):
        p[unsafe_offset=i] = i
    var t = 0
    if n > 0:
        t = p[unsafe_offset=n - 1]  # not written
    p.unsafe_free()
    return t


def bad_list_appended(mut xs: List[Int]) -> Int:
    if len(xs) > 0:
        var p = xs.unsafe_ptr()
        xs.append(1)  # may reallocate
        return p[]
    return 0


def bad_freed_in_branch(c: Bool) -> Int:
    var p = unsafe_alloc[Int](1)
    p[] = 1
    if c:
        p.unsafe_free()
    var t = p[]
    if not c:
        p.unsafe_free()
    return t


def bad_load_uninit() -> Float32:
    var p = unsafe_alloc[Float32](8)
    p.unsafe_store(0, SIMD[DType.float32, 2](1))
    var v = p.unsafe_load[width=4](0)  # elements 2 and 3
    p.unsafe_free()
    return v[0]
