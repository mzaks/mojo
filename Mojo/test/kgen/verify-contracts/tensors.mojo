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
# `TileTensor` indexing and GPU kernels for `verify-contracts`; see README.md.
# Needs the `max` and `layout` packages on the import path (see README.md).

from layout import Idx, TileTensor, row_major, stack_allocation
from layout.swizzle import make_swizzle
from max.gpu import block_dim, block_idx, global_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from std.utils.coord import Coord, coord

comptime L8 = type_of(row_major[8]())
comptime L24 = type_of(row_major[2, 4]())
comptime L512 = type_of(row_major[512]())
comptime L1000 = type_of(row_major[1000]())
comptime L1024 = type_of(row_major[1024]())
comptime L88 = type_of(row_major[8, 8]())
comptime L86 = type_of(row_major[8, 6]())
comptime LD = type_of(row_major(1, 1))  # dimensions known at run time


# --- must be PROVEN ---
def ok_static(t: TileTensor[DType.float32, L8, MutAnyOrigin]) -> Float32:
    return t[7]  # the layout's size is 8


def ok_guard_dim(
    t: TileTensor[DType.float32, L8, MutAnyOrigin], i: Int
) -> Float32:
    if 0 <= i and i < Int(t.dim[0]()):
        return t[i]
    return 0


def ok_2d(
    t: TileTensor[DType.float32, L24, MutAnyOrigin], i: Int, j: Int
) -> Float32:
    if 0 <= i and i < 2 and 0 <= j and j < 4:
        return t[i, j]
    return 0


def ok_tuple(t: TileTensor[DType.float32, L8, MutAnyOrigin], i: Int) -> Float32:
    return t[Coord(i)]  # tuple coordinates are not checked


def ok_loop(t: TileTensor[DType.float32, L8, MutAnyOrigin]) -> Float32:
    var s: Float32 = 0
    for i in range(Int(t.dim[0]())):
        s += t[i]
    return s


def ok_kernel(
    a: TileTensor[DType.float32, L1000, MutAnyOrigin],
    dst: TileTensor[DType.float32, L1000, MutAnyOrigin],
):
    var tid = block_idx.x * block_dim.x + thread_idx.x
    if tid < 1000:
        dst[tid] = a[tid]  # a read and a write


def ok_kernel_global(a: TileTensor[DType.float32, L1000, MutAnyOrigin]) -> Float32:
    if global_idx.x < Int(a.dim[0]()):
        return a[global_idx.x]
    return 0


def ok_thread_only(a: TileTensor[DType.float32, L1024, MutAnyOrigin]) -> Float32:
    return a[thread_idx.x]  # a block has at most 1024 threads


def ok_tile(t: TileTensor[DType.float32, L88, MutAnyOrigin]) -> Float32:
    var x = t.tile[2, 4](3, 1)  # rows 6..7, columns 4..7
    return x[1, 3]  # the tile's dimensions are 2 and 4


def ok_tile_guard(
    t: TileTensor[DType.float32, LD, MutAnyOrigin], i: Int, j: Int
) -> Float32:
    if 0 <= i and i < Int(t.dim[0]()) // 16:
        if 0 <= j and j < Int(t.dim[1]()) // 16:
            return t.tile[16, 16](i, j)[15, 15]
    return 0


def ok_tile_of_tile(t: TileTensor[DType.float32, L88, MutAnyOrigin]) -> Float32:
    var x = t.tile[4, 4](1, 1)
    return x.tile[2, 2](1, 0)[1, 1]


def ok_block_tile(t: TileTensor[DType.float32, LD, MutAnyOrigin]):
    if block_idx.y < Int(t.dim[0]()) // 16:
        if block_idx.x < Int(t.dim[1]()) // 16:
            var x = t.tile[16, 16](block_idx.y, block_idx.x)
            if thread_idx.y < 16 and thread_idx.x < 16:
                x[thread_idx.y, thread_idx.x] = 0


def ok_vectorize(
    t: TileTensor[DType.float32, L88, MutAnyOrigin]
) -> SIMD[DType.float32, 4]:
    var v = t.vectorize[1, 4]()  # 8 x 2 vectors of 4
    return v[7, 1]


def ok_vectorize_dynamic(
    t: TileTensor[DType.float32, LD, MutAnyOrigin], j: Int
) -> SIMD[DType.float32, 4]:
    if Int(t.dim[1]()) % 4 == 0 and 0 < Int(t.dim[0]()):
        var v = t.vectorize[1, 4]()
        if 0 <= j and j < Int(t.dim[1]()) // 4:
            return v[0, j]  # `v` has `dim / 4` vectors per row
    return 0


def ok_tile_vectorize(
    t: TileTensor[DType.float32, LD, MutAnyOrigin], i: Int
) -> SIMD[DType.float32, 4]:
    if 0 <= i and i < Int(t.dim[0]()) // 4 and Int(t.dim[1]()) >= 8:
        return t.tile[4, 8](i, 0).vectorize[1, 4]()[3, 1]
    return 0


def ok_load(
    t: TileTensor[DType.float32, L88, MutAnyOrigin]
) -> SIMD[DType.float32, 4]:
    return t.load[4](Coord(7, 4))  # columns 4..7 of the last row


def ok_load_guard(
    t: TileTensor[DType.float32, LD, MutAnyOrigin], i: Int, j: Int
) -> SIMD[DType.float32, 4]:
    if 0 <= i and i < Int(t.dim[0]()):
        if 0 <= j and j <= Int(t.dim[1]()) - 4:
            return t.load[4](Coord(i, j))
    return 0


def ok_load_tuple(
    t: TileTensor[DType.float32, L88, MutAnyOrigin], i: Int
) -> SIMD[DType.float32, 4]:
    if 0 <= i and i < 8:
        return t.load[4]((i, 4))  # a tuple converts to a `Coord`
    return 0


def ok_store(
    t: TileTensor[DType.float32, L88, MutAnyOrigin], v: SIMD[DType.float32, 8]
):
    t.store(Coord(Idx[3], Idx[0]), v)  # a whole row


def ok_load_narrow(
    t: TileTensor[DType.float32, L88, MutAnyOrigin], i: Int32
) -> Float32:
    if Int32(0) <= i and i < Int32(8):
        return t.load[1](Coord(i, Idx[2]))
    return 0


def ok_tile_coord(t: TileTensor[DType.float32, L88, MutAnyOrigin]) -> Float32:
    return t.tile[2, 4](Coord(3, 1))[1, 3]  # coordinates as a `Coord`


def ok_tile_shape(
    t: TileTensor[DType.float32, LD, MutAnyOrigin], i: Int, j: Int
) -> Float32:
    if 0 <= i and i < Int(t.dim[0]()) // 2:
        if 0 <= j and j < Int(t.dim[1]()) // 4:
            return t.tile(coord[2, 4], Coord(i, j))[1, 3]  # shape as a `Coord`
    return 0


def ok_tile_tuple(
    t: TileTensor[DType.float32, L88, MutAnyOrigin], i: Int
) -> Float32:
    if 0 <= i and i < 4:
        return t.tile[2, 4]((i, 1))[1, 3]
    return 0


def ok_distribute(
    t: TileTensor[DType.float32, L88, MutAnyOrigin]
) -> Float32:
    # 2 x 4 threads: each gets 4 x 2 elements, for any thread id.
    var v = t.distribute[row_major[2, 4]()](thread_idx.x)
    return v[3, 1]


def ok_distribute_dynamic(
    t: TileTensor[DType.float32, LD, MutAnyOrigin], i: Int, j: Int
) -> Float32:
    var v = t.distribute[row_major[2, 4]()](thread_idx.x)
    if 0 <= i and i < Int(t.dim[0]()) // 2:
        if 0 <= j and j < Int(t.dim[1]()) // 4:
            return v[i, j]  # `v` has `dim // threads` elements
    return 0


def ok_vectorize_distribute(
    t: TileTensor[DType.float32, LD, MutAnyOrigin], i: Int
) -> SIMD[DType.float32, 4]:
    if Int(t.dim[1]()) % 4 == 0 and Int(t.dim[1]()) >= 32:
        var v = t.vectorize[1, 4]().distribute[row_major[1, 8]()](
            thread_idx.x
        )
        if 0 <= i and i < Int(t.dim[0]()):
            return v[i, 0]
    return 0


def ok_distribute_with_offset(
    t: TileTensor[DType.float32, LD, MutAnyOrigin], i: Int
) -> SIMD[DType.float32, 2]:
    if Int(t.dim[1]()) % 2 == 0 and Int(t.dim[1]()) >= 8:
        # As kernels write it: a vectorized view, distributed, destructured.
        var v, coords, offset = t.vectorize[
            1, 2
        ]().distribute_with_offset[row_major[1, 4]()](thread_idx.x)
        if 0 <= i and i < Int(t.dim[0]()):
            return v[i, 0]
    return 0


def buffer_of(
    buf: DeviceBuffer[DType.float32], n: Int where len(buf) == n
) -> Int:
    return n


def rows_of(
    t: TileTensor[mut=False, DType.float32, ...],
    n: Int where Int(t.dim[0]()) == n,
) -> Int:
    return n


def ok_device_buffer(
    t: TileTensor[mut=True, DType.float32, ...], ctx: DeviceContext
) raises -> Int:
    comptime assert t.rank == 2
    var buf = t.to_device_buffer(ctx)  # `num_elements()` elements
    var rows = Int(t.dim[0]())
    _ = rows_of(t, rows)  # converted to immutable: the same tensor
    return buffer_of(buf, rows * Int(t.dim[1]()))


def ok_comptime_for_write[N: Int]() -> Float32:
    var t = stack_allocation[dtype=DType.float32](row_major[4, N]())
    comptime for j in range(N):
        t[1, j] = 1  # the index of a `comptime for` in a written element
    return t[0, 0]


def ok_store_width[
    N: Int, W: Int
](v: SIMD[DType.float32, W]) -> Float32:
    comptime assert W > 0
    var t = stack_allocation[dtype=DType.float32](row_major[N, W]())
    comptime for i in range(N):
        t.store(Coord(i, Idx[0]), v)  # a row of `W`, whatever `W` is
    return 0


def ok_comptime_for_guard[
    N: Int
](t: TileTensor[DType.float32, LD, MutAnyOrigin]):
    comptime for i in range(N):
        if i >= Int(t.dim[0]()):
            continue
        if Int(t.dim[1]()) > 0:
            t[i, 0] = 0  # the same `i` as in the guard


def ok_tensor_pointer(
    t: TileTensor[mut=False, ...], i: Int where 0 <= i < t.ptr._extent()
) -> Float32:
    var p = t.ptr  # the pointer the clause is about
    return Float32(p[unsafe_offset=i].cast[DType.float32]())


def ok_after_fill(
    c: TileTensor[mut=True, ...],
    t: TileTensor[mut=False, ...],
    n: Int where t.ptr._extent() >= n and c.ptr._extent() >= n,
) -> Float32:
    var p = t.ptr
    var q = c.ptr
    var sum: Float32 = 0
    # Writes through the tensor, here and in the loop, reach whatever a
    # pointer leads to: not `p`, `q` and the loop's range, which are never
    # borrowed.
    _ = c.fill(0)
    for i in range(n):
        sum += Float32(p[unsafe_offset=i].cast[DType.float32]())
        q[unsafe_offset=i] += 1
    return sum


def ok_matrix_vector(
    c: TileTensor[mut=True, ...],
    a: TileTensor[mut=False, ...],
    b: TileTensor[mut=False, ...] where (
        Int(a.dim[0]()) < 2147483648
        and Int(a.dim[1]()) < 2147483648
        and c.ptr._extent() >= Int(a.dim[0]())
        and a.ptr._extent() >= Int(a.dim[0]()) * Int(a.dim[1]())
        and b.ptr._extent() >= Int(a.dim[1]())
    ),
):
    var M = Int(a.dim[0]())
    var K = Int(a.dim[1]())
    var c_ptr = c.ptr
    var a_ptr = a.ptr
    var b_ptr = b.ptr
    _ = c.fill(0)
    for k in range(K):
        var b_val = b_ptr[unsafe_offset=k].cast[c.dtype]()
        for m in range(M):
            # `k` is the outer loop's across the inner loop's writes
            var a_val = a_ptr[unsafe_offset=m * K + k].cast[c.dtype]()
            c_ptr[unsafe_offset=m] += a_val * b_val


def _at_most(x: Int, lim: Int where 0 <= x <= lim) -> Int:
    return x


def ok_tensor_of_pointer(
    p: Pointer[Float32, MutAnyOrigin] where p._extent() >= 12
) -> Float32:
    var t = TileTensor(p, row_major[3, 4]())
    return t.ptr[unsafe_offset=11]  # the pointer it was built from


def ok_swizzled[
    tile_k: Int
](lin: Int where 0 <= lin < 16 * tile_k and lin % 8 == 0) -> Int:
    comptime sw = make_swizzle[8, tile_k, 8]()
    comptime assert 64 <= tile_k <= 65536 and tile_k % 64 == 0
    # Checked by the compiler for each `tile_k`: the swizzle moves the bits
    # of its `yyy_mask` onto bits 3 to 5, so within blocks of 64.
    comptime assert sw.zzz_mask == 56 and 0 < sw.shift < 64
    comptime assert (sw.yyy_mask >> sw.shift) == 56
    return _at_most(sw(lin), 16 * tile_k - 8)


# --- must stay UNPROVEN ---
def bad_static(t: TileTensor[DType.float32, L8, MutAnyOrigin]) -> Float32:
    return t[8]  # one past the end


def bad_unguarded(
    t: TileTensor[DType.float32, L8, MutAnyOrigin], i: Int
) -> Float32:
    return t[i]


def bad_2d(
    t: TileTensor[DType.float32, L24, MutAnyOrigin], i: Int, j: Int
) -> Float32:
    if 0 <= i and i < 2 and 0 <= j and j < 5:
        return t[i, j]  # 4 columns
    return 0


def bad_kernel_write(dst: TileTensor[DType.float32, L1000, MutAnyOrigin]):
    var tid = block_idx.x * block_dim.x + thread_idx.x
    if tid <= 1000:
        dst[tid] = 1.0  # `tid` may be 1000


def bad_thread_only(a: TileTensor[DType.float32, L512, MutAnyOrigin]) -> Float32:
    return a[thread_idx.x]  # a block may have more than 512 threads


def bad_tile_coord(t: TileTensor[DType.float32, L88, MutAnyOrigin]) -> Float32:
    return t.tile[2, 4](4, 0)[0, 0]  # 4 tiles per column


def bad_tile_element(
    t: TileTensor[DType.float32, L88, MutAnyOrigin]
) -> Float32:
    return t.tile[2, 4](3, 1)[2, 0]  # the tile has 2 rows


def bad_partial_tile(
    t: TileTensor[DType.float32, LD, MutAnyOrigin], i: Int
) -> Float32:
    if 0 <= i and i * 16 < Int(t.dim[0]()) and Int(t.dim[1]()) >= 16:
        return t.tile[16, 16](i, 0)[0, 0]  # the last tile may be partial
    return 0


def bad_block_tile(t: TileTensor[DType.float32, LD, MutAnyOrigin]):
    var x = t.tile[16, 16](block_idx.y, block_idx.x)  # the grid may be larger
    if thread_idx.y < 16 and thread_idx.x < 16:
        x[thread_idx.y, thread_idx.x] = 0


def bad_vectorize_ragged(
    t: TileTensor[DType.float32, L86, MutAnyOrigin]
) -> SIMD[DType.float32, 4]:
    return t.vectorize[1, 4]()[0, 0]  # 6 is not a multiple of 4


def bad_vectorize_index(
    t: TileTensor[DType.float32, L88, MutAnyOrigin]
) -> SIMD[DType.float32, 4]:
    return t.vectorize[1, 4]()[7, 2]  # 2 vectors per row


def bad_vectorize_overflow(
    t: TileTensor[DType.float32, LD, MutAnyOrigin], j: Int
) -> SIMD[DType.float32, 4]:
    if Int(t.dim[1]()) % 4 == 0 and 0 < Int(t.dim[0]()):
        var v = t.vectorize[1, 4]()
        if 0 <= j and j * 4 < Int(t.dim[1]()):
            return v[0, j]  # `j * 4` wraps for a large `j`
    return 0


def bad_load_width(
    t: TileTensor[DType.float32, L88, MutAnyOrigin]
) -> SIMD[DType.float32, 4]:
    return t.load[4](Coord(7, 5))  # 5 + 4 > 8


def bad_load_row(
    t: TileTensor[DType.float32, LD, MutAnyOrigin], i: Int, j: Int
) -> SIMD[DType.float32, 4]:
    if 0 <= i and i < Int(t.dim[0]()) and 0 <= j and j < Int(t.dim[1]()):
        return t.load[4](Coord(i, j))  # may run past the row
    return 0


def bad_load_wrap(
    t: TileTensor[DType.float32, LD, MutAnyOrigin], i: Int, j: Int
) -> SIMD[DType.float32, 4]:
    if 0 <= i and i < Int(t.dim[0]()) and 0 <= j and j + 4 <= Int(t.dim[1]()):
        return t.load[4](Coord(i, j))  # `j + 4` wraps for a large `j`
    return 0


def bad_store(
    t: TileTensor[DType.float32, L88, MutAnyOrigin], v: SIMD[DType.float32, 8]
):
    t.store(Coord(Idx[3], Idx[1]), v)  # one past the row


def bad_load_narrow(
    t: TileTensor[DType.float32, L88, MutAnyOrigin], i: UInt8
) -> Float32:
    if i < UInt8(9):
        return t.load[1](Coord(i, Idx[2]))
    return 0


def bad_tile_coords(
    t: TileTensor[DType.float32, L88, MutAnyOrigin]
) -> Float32:
    return t.tile[2, 4](Coord(4, 0))[0, 0]  # 4 tiles per column


def bad_tile_shape(
    t: TileTensor[DType.float32, L88, MutAnyOrigin], s: Int
) -> Float32:
    if 0 < s and s <= 8:
        return t.tile(Coord(s, 8), Coord(1, 0))[0, 0]  # `2 * s` may pass 8
    return 0


def bad_distribute(
    t: TileTensor[DType.float32, L86, MutAnyOrigin]
) -> Float32:
    var v = t.distribute[row_major[1, 4]()](thread_idx.x)
    return v[7, 1]  # 6 // 4 = 1 column per thread


def bad_distribute_dynamic(
    t: TileTensor[DType.float32, LD, MutAnyOrigin], i: Int
) -> Float32:
    var v = t.distribute[row_major[2, 4]()](thread_idx.x)
    if 0 <= i and i * 2 < Int(t.dim[0]()) and Int(t.dim[1]()) >= 4:
        return v[i, 0]  # a row past the last full one
    return 0


def bad_distribute_with_offset(
    t: TileTensor[DType.float32, L88, MutAnyOrigin]
) -> Float32:
    var r = t.distribute_with_offset[row_major[2, 4]()](thread_idx.x)
    return r[0][4, 0]  # 8 // 2 = 4 rows per thread


def bad_device_buffer(
    t: TileTensor[mut=True, DType.float32, ...], ctx: DeviceContext
) raises -> Int:
    var buf = t.to_device_buffer(ctx)  # the rank is not known here
    return buffer_of(buf, Int(t.dim[0]()) * Int(t.dim[1]()))


def bad_comptime_for_write[N: Int]() -> Float32:
    var t = stack_allocation[dtype=DType.float32](row_major[4, N]())
    comptime for j in range(N):
        t[1, j + 1] = 1  # column N
    return t[0, 0]


def bad_store_width[
    N: Int, W: Int
](v: SIMD[DType.float32, W]) -> Float32:
    comptime assert W > 0
    var t = stack_allocation[dtype=DType.float32](row_major[N, W]())
    comptime for i in range(N):
        t.store(Coord(i, Idx[1]), v)  # one past the row for any width
    return 0


def bad_comptime_for_guard[
    N: Int
](t: TileTensor[DType.float32, LD, MutAnyOrigin]):
    comptime for i in range(N):
        if i > Int(t.dim[0]()):
            continue
        if Int(t.dim[1]()) > 0:
            t[i, 0] = 0  # row `dim[0]` gets through


def bad_tensor_pointer(t: TileTensor[mut=False, ...], i: Int) -> Float32:
    var p = t.ptr  # nothing states its extent
    return Float32(p[unsafe_offset=i].cast[DType.float32]())


def bad_after_fill(
    c: TileTensor[mut=True, ...],
    t: TileTensor[mut=False, ...],
    n: Int where t.ptr._extent() >= n and c.ptr._extent() >= n,
) -> Float32:
    var p = t.ptr
    var sum: Float32 = 0
    _ = c.fill(0)
    for i in range(n):
        sum += Float32(p[unsafe_offset=i + 1].cast[DType.float32]())  # past
    return sum


def bad_matrix_vector(
    c: TileTensor[mut=True, ...],
    a: TileTensor[mut=False, ...],
    b: TileTensor[mut=False, ...] where (
        c.ptr._extent() >= Int(a.dim[0]())
        and a.ptr._extent() >= Int(a.dim[0]()) * Int(a.dim[1]())
        and b.ptr._extent() >= Int(a.dim[1]())
    ),
):
    var M = Int(a.dim[0]())
    var K = Int(a.dim[1]())
    var c_ptr = c.ptr
    var a_ptr = a.ptr
    var b_ptr = b.ptr
    _ = c.fill(0)
    for k in range(K):
        var b_val = b_ptr[unsafe_offset=k].cast[c.dtype]()
        for m in range(M):
            # nothing bounds the dimensions: `M * K` may wrap
            var a_val = a_ptr[unsafe_offset=m * K + k].cast[c.dtype]()
            c_ptr[unsafe_offset=m] += a_val * b_val


def bad_tensor_of_pointer(
    p: Pointer[Float32, MutAnyOrigin] where p._extent() >= 12
) -> Float32:
    var t = TileTensor(p, row_major[3, 4]())
    return t.ptr[unsafe_offset=12]


def bad_swizzled[
    tile_k: Int
](lin: Int where 0 <= lin < 16 * tile_k and lin % 8 == 0) -> Int:
    comptime sw = make_swizzle[8, tile_k, 8]()
    comptime assert 64 <= tile_k <= 65536 and tile_k % 64 == 0
    return _at_most(sw(lin), 16 * tile_k - 8)  # nothing about its masks
