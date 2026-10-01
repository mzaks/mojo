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

from layout import TileTensor, row_major
from max.gpu import block_dim, block_idx, global_idx, thread_idx
from std.utils.coord import Coord

comptime L8 = type_of(row_major[8]())
comptime L24 = type_of(row_major[2, 4]())
comptime L512 = type_of(row_major[512]())
comptime L1000 = type_of(row_major[1000]())
comptime L1024 = type_of(row_major[1024]())


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
