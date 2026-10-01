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

from layout import Idx, TileTensor, row_major
from max.gpu import block_dim, block_idx, global_idx, thread_idx
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
