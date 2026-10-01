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
# GPU kernels that state their launch configuration as `where` clauses, and
# the launches that must establish it, for `verify-contracts`; see README.md.
# Needs the `max` and `layout` packages on the import path (see README.md).

from layout import TensorLayout, TileTensor, row_major, stack_allocation
from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceContext
from std.math import ceildiv
from std.math.uutils import udivmod

comptime LD = type_of(row_major(1, 1))  # dimensions known at run time
comptime L88 = type_of(row_major[8, 8]())


# --- must be PROVEN ---
def ok_tiled_kernel(
    c: TileTensor[DType.float32, LD, MutAnyOrigin] where (
        grid_dim.y <= Int(c.dim[0]()) // 16
        and grid_dim.x <= Int(c.dim[1]()) // 16
        and block_dim.x == 256
    ),
):
    # One 16 x 16 tile per block, one element per thread.
    var dst = c.tile[16, 16](block_idx.y, block_idx.x)
    var row, col = udivmod(thread_idx.x, 16)
    dst[row, col] = 0


def ok_generic_kernel[
    L: TensorLayout, BM: Int, BN: Int
](
    c: TileTensor[DType.float32, L, MutAnyOrigin] where (
        grid_dim.y <= Int(c.dim[0]()) // BM
        and grid_dim.x <= Int(c.dim[1]()) // BN
        and block_dim.x == BM * BN
    ),
):
    # The tile sizes are parameters: proven for every `BM` and `BN`.
    var dst = c.tile[BM, BN](block_idx.y, block_idx.x)
    var row, col = udivmod(thread_idx.x, BN)
    dst[row, col] = 0


def ok_shared_tile[BK: Int](i: Int) -> Float32:
    var s = stack_allocation[dtype=DType.float32](row_major[4, BK]())
    var sum: Float32 = 0
    if 0 <= i and i < 4:
        comptime for k in range(BK):
            sum += s[i, k]  # `k` is in `[0, BK)`
    return sum


def ok_launch(
    ctx: DeviceContext, c: TileTensor[DType.float32, LD, MutAnyOrigin]
) raises:
    var m = Int(c.dim[0]())
    var n = Int(c.dim[1]())
    ctx.enqueue_function[ok_tiled_kernel](
        c, grid_dim=(n // 16, m // 16), block_dim=256
    )


def ok_comptime_for(t: TileTensor[DType.float32, L88, MutAnyOrigin]) -> Float32:
    var sum: Float32 = 0
    comptime for k in range(8):
        sum += t[7, k]
    return sum


def ok_sized_kernel[N: Int](t: TileTensor[DType.float32, L88, MutAnyOrigin]):
    # Not for every `N`, but for the one launched: proven for that size.
    t[7, N - 1] = 0


def ok_launch_sized(
    ctx: DeviceContext, t: TileTensor[DType.float32, L88, MutAnyOrigin]
) raises:
    ctx.enqueue_function[ok_sized_kernel[8]](t, grid_dim=1, block_dim=1)


def ok_partial_kernel[
    dtype: DType, N: Int
](t: TileTensor[dtype, L88, MutAnyOrigin]):
    t[7, N - 1] = 0


def ok_launch_partial[
    dtype: DType
](ctx: DeviceContext, t: TileTensor[dtype, L88, MutAnyOrigin]) raises:
    # The launcher's `dtype` stays unknown; the size is fixed.
    ctx.enqueue_function[ok_partial_kernel[dtype, 8]](
        t, grid_dim=1, block_dim=1
    )


def ok_through_kernel[N: Int](t: TileTensor[DType.float32, L88, MutAnyOrigin]):
    t[7, N - 1] = 0


def ok_launch_through[
    N: Int
](ctx: DeviceContext, t: TileTensor[DType.float32, L88, MutAnyOrigin]) raises:
    ctx.enqueue_function[ok_through_kernel[N]](t, grid_dim=1, block_dim=1)


def ok_specialize(
    ctx: DeviceContext, t: TileTensor[DType.float32, L88, MutAnyOrigin]
) raises:
    # `N` is fixed here, one call out from the launch.
    ok_launch_through[8](ctx, t)


def ok_launch_compiled(
    ctx: DeviceContext, c: TileTensor[DType.float32, LD, MutAnyOrigin]
) raises:
    # Compiled first, then launched: the same check.
    var m = Int(c.dim[0]())
    var n = Int(c.dim[1]())
    var f = ctx.compile_function[ok_tiled_kernel]()
    ctx.enqueue_function(f, c, grid_dim=(n // 16, m // 16), block_dim=256)


def ok_compiled_kernel[N: Int](t: TileTensor[DType.float32, L88, MutAnyOrigin]):
    t[7, N - 1] = 0


def ok_launch_compiled_sized(
    ctx: DeviceContext, t: TileTensor[DType.float32, L88, MutAnyOrigin]
) raises:
    var f = ctx.compile_function[ok_compiled_kernel[8]]()
    ctx.enqueue_function(f, t, grid_dim=1, block_dim=1)


# --- must stay UNPROVEN ---
def bad_unguarded_kernel(c: TileTensor[DType.float32, LD, MutAnyOrigin]):
    # Nothing relates the grid to the tensor.
    var dst = c.tile[16, 16](block_idx.y, block_idx.x)
    dst[0, 0] = 0


def bad_launch_rounded_up(
    ctx: DeviceContext, c: TileTensor[DType.float32, LD, MutAnyOrigin]
) raises:
    var m = Int(c.dim[0]())
    var n = Int(c.dim[1]())
    # The last tile of a dimension that is not a multiple of 16 is partial.
    ctx.enqueue_function[ok_tiled_kernel](
        c, grid_dim=(ceildiv(n, 16), ceildiv(m, 16)), block_dim=256
    )


def bad_launch_block(
    ctx: DeviceContext, c: TileTensor[DType.float32, LD, MutAnyOrigin]
) raises:
    var m = Int(c.dim[0]())
    var n = Int(c.dim[1]())
    ctx.enqueue_function[ok_tiled_kernel](
        c, grid_dim=(n // 16, m // 16), block_dim=128
    )


def bad_launch_swapped(
    ctx: DeviceContext, c: TileTensor[DType.float32, LD, MutAnyOrigin]
) raises:
    var m = Int(c.dim[0]())
    var n = Int(c.dim[1]())
    ctx.enqueue_function[ok_tiled_kernel](
        c, grid_dim=(m // 16, n // 16), block_dim=256
    )


def bad_comptime_for(t: TileTensor[DType.float32, L88, MutAnyOrigin]) -> Float32:
    var sum: Float32 = 0
    comptime for k in range(9):
        sum += t[7, k]
    return sum


def bad_sized_kernel[N: Int](t: TileTensor[DType.float32, L88, MutAnyOrigin]):
    t[7, N - 1] = 0  # launched with 9 below, past the row


def bad_launch_sized(
    ctx: DeviceContext, t: TileTensor[DType.float32, L88, MutAnyOrigin]
) raises:
    ctx.enqueue_function[bad_sized_kernel[8]](t, grid_dim=1, block_dim=1)
    ctx.enqueue_function[bad_sized_kernel[9]](t, grid_dim=1, block_dim=1)


def bad_partial_kernel[
    dtype: DType, N: Int
](t: TileTensor[dtype, L88, MutAnyOrigin]):
    t[7, N - 1] = 0  # the launcher below passes any `N`


def bad_launch_unsized[
    N: Int
](ctx: DeviceContext, t: TileTensor[DType.float32, L88, MutAnyOrigin]) raises:
    ctx.enqueue_function[bad_partial_kernel[DType.float32, N]](
        t, grid_dim=1, block_dim=1
    )


def bad_through_kernel[N: Int](t: TileTensor[DType.float32, L88, MutAnyOrigin]):
    t[7, N - 1] = 0  # specialized with 9 two calls out


def bad_launch_through[
    N: Int
](ctx: DeviceContext, t: TileTensor[DType.float32, L88, MutAnyOrigin]) raises:
    ctx.enqueue_function[bad_through_kernel[N]](t, grid_dim=1, block_dim=1)


def bad_specialize(
    ctx: DeviceContext, t: TileTensor[DType.float32, L88, MutAnyOrigin]
) raises:
    bad_launch_through[8](ctx, t)
    bad_launch_through[9](ctx, t)


def bad_launch_compiled(
    ctx: DeviceContext, c: TileTensor[DType.float32, LD, MutAnyOrigin]
) raises:
    var m = Int(c.dim[0]())
    var n = Int(c.dim[1]())
    var f = ctx.compile_function[ok_tiled_kernel]()
    ctx.enqueue_function(f, c, grid_dim=(n // 16 + 1, m // 16), block_dim=256)


def bad_compiled_kernel[N: Int](
    t: TileTensor[DType.float32, L88, MutAnyOrigin]
):
    t[7, N - 1] = 0  # compiled with 9 below


def bad_launch_compiled_sized(
    ctx: DeviceContext, t: TileTensor[DType.float32, L88, MutAnyOrigin]
) raises:
    var f = ctx.compile_function[bad_compiled_kernel[9]]()
    ctx.enqueue_function(f, t, grid_dim=1, block_dim=1)
