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
# Raw pointers for `verify-contracts`: a pointer's extent (how many elements
# are valid from it) is stated by `_extent()` clauses; see README.md.

from std.memory import stack_allocation
from std.memory.alloc import unsafe_alloc


# --- must be PROVEN ---
def ok_stated(
    p: Pointer[Int, MutAnyOrigin],
    n: Int where p._extent() >= n,
    i: Int,
) -> Int:
    if 0 <= i and i < n:
        return p[unsafe_offset=i]  # the caller states the extent
    return 0


def ok_alloc(n: Int, i: Int) -> Int:
    var p = unsafe_alloc[Int](n)
    if 0 <= i and i < n:
        return p[unsafe_offset=i]  # `unsafe_alloc` states it
    return 0


def ok_offset(n: Int, i: Int) -> Int:
    var p = unsafe_alloc[Int](n)
    if n >= 2 and 0 <= i and i < n - 2:
        return p.unsafe_offset(2)[unsafe_offset=i]  # two fewer from there
    return 0


def ok_load(
    p: Pointer[Float32, MutAnyOrigin],
    n: Int where n >= 0 and p._extent() >= n,
    i: Int,
) -> SIMD[DType.float32, 4]:
    if 0 <= i and i <= n - 4:
        return p.unsafe_load[width=4](i)
    return 0


def ok_store(
    p: Pointer[Float32, MutAnyOrigin],
    n: Int where n >= 0 and p._extent() >= n,
    i: Int,
):
    if 0 <= i and i <= n - 4:
        p.unsafe_store(i, SIMD[DType.float32, 4](0))


def ok_stack(i: Int) -> Float32:
    var p = stack_allocation[16, DType.float32]()
    if 0 <= i and i < 4:
        return p.unsafe_load[width=4](i * 4).reduce_add()
    return 0


def ok_deref(n: Int) -> Int:
    var p = unsafe_alloc[Int](n)
    if n >= 1:
        return p[]  # no offset: one element
    return 0


def ok_load_first(
    p: Pointer[Float32, MutAnyOrigin],
    n: Int where p._extent() >= n,
) -> Float32:
    if n >= 4:
        p.unsafe_store(SIMD[DType.float32, 4](1))
        return p.unsafe_load[width=4]().reduce_add()
    return 0


def ok_to(x: Int) -> Int:
    var y = x
    return Pointer(to=y)[]  # a pointer to one value


def first(
    p: ImmPointer[Int, _], n: Int where p._extent() >= n and n >= 1
) -> Int:
    return p[]


def ok_immutable(
    p: Pointer[Int, MutAnyOrigin],
    n: Int where p._extent() >= n,
) -> Int:
    if n >= 1:
        return first(p, n)  # converted to immutable: the same pointer
    return 0


def ok_list(xs: List[Int], i: Int) -> Int:
    var p = xs.unsafe_ptr()
    if 0 <= i and i < len(xs):
        return p[unsafe_offset=i]  # at least `len(xs)` elements
    return 0


def ok_span(xs: Span[Int, _], i: Int) -> Int:
    var p = xs.unsafe_ptr()
    if 0 <= i and i < len(xs):
        return p[unsafe_offset=i]
    return 0


def ok_array(i: Int) -> Int:
    var a: Array[Int, 3] = [1, 2, 3]
    var p = a.unsafe_ptr()
    if 0 <= i and i < 3:
        return p[unsafe_offset=i]
    return 0


# --- must stay UNPROVEN ---
def bad_unstated(p: Pointer[Int, MutAnyOrigin], i: Int) -> Int:
    if 0 <= i and i < 4:
        return p[unsafe_offset=i]  # nothing states how many
    return 0


def bad_alloc(n: Int, i: Int) -> Int:
    var p = unsafe_alloc[Int](n)
    if 0 <= i and i <= n:
        return p[unsafe_offset=i]  # `i` may be `n`
    return 0


def bad_offset(n: Int, i: Int) -> Int:
    var p = unsafe_alloc[Int](n)
    if 0 <= i and i < n:
        return p.unsafe_offset(2)[unsafe_offset=i]  # `2 + i` may pass `n`
    return 0


def bad_backwards(n: Int) -> Int:
    var p = unsafe_alloc[Int](n)
    if n > 2:
        return p.unsafe_offset(-1)[unsafe_offset=0]  # before the allocation
    return 0


def bad_load(
    p: Pointer[Float32, MutAnyOrigin],
    n: Int where n >= 0 and p._extent() >= n,
    i: Int,
) -> SIMD[DType.float32, 4]:
    if 0 <= i and i < n:
        return p.unsafe_load[width=4](i)  # the last 3 lanes may pass `n`
    return 0


def bad_deref(p: Pointer[Int, MutAnyOrigin]) -> Int:
    return p[]  # may point to nothing


def bad_load_first(
    p: Pointer[Float32, MutAnyOrigin],
    n: Int where p._extent() >= n,
) -> Float32:
    if n >= 3:
        return p.unsafe_load[width=4]().reduce_add()  # three may be all
    return 0


def bad_to(x: Int) -> Int:
    var y = x
    return Pointer(to=y)[unsafe_offset=1]  # past the one value


def bad_list(xs: List[Int], i: Int) -> Int:
    var p = xs.unsafe_ptr()
    if 0 <= i and i <= len(xs):
        return p[unsafe_offset=i]  # `i` may be `len(xs)`
    return 0
