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
# Launches of `linalg.gemv.gemv_split_k` with the launch configurations its
# dispatcher computes, for `verify-contracts`; see README.md. The dispatcher
# takes `(num_threads, tile_n, unroll_factor)` from a function the verifier
# does not evaluate, so it does not verify the kernel for those launches;
# here the same values are written out, and `check_configs` (compiled, not
# verified) keeps them equal to what the function returns.
# Needs the kernel packages on the import path and `packages=linalg`.

from layout import TensorLayout, TileTensor
from layout.tensor_engine import TensorEngine
from linalg.gemv import _nvidia_gemv_config, gemv_split_k
from max.gpu.host import DeviceContext, get_gpu_target
from std.math import ceildiv
from std.sys.info import simd_width_of


def check_configs():
    """The configurations launched below are the dispatcher's for these
    dtypes and static shapes (`simd_width` is 16 bytes of the dtype)."""
    comptime fp32 = _nvidia_gemv_config[DType.float32, 4, 6144, True, 128]()
    comptime assert fp32[0] == 256 and fp32[1] == 1 and fp32[2] == 2
    comptime bf16 = _nvidia_gemv_config[DType.bfloat16, 8, 4096, True, 4096]()
    comptime assert bf16[0] == 128 and bf16[1] == 4 and bf16[2] == 1
    comptime bf16_small = _nvidia_gemv_config[
        DType.bfloat16, 8, 2048, True, 128
    ]()
    comptime assert (
        bf16_small[0] == 128 and bf16_small[1] == 1 and bf16_small[2] == 1
    )
    comptime bf16_deep = _nvidia_gemv_config[
        DType.bfloat16, 8, 16384, True, 1024
    ]()
    comptime assert (
        bf16_deep[0] == 128 and bf16_deep[1] == 2 and bf16_deep[2] == 1
    )
    comptime fp8 = _nvidia_gemv_config[
        DType.float8_e4m3fn, 16, 4096, True, 8192
    ]()
    comptime assert fp8[0] == 64 and fp8[1] == 4 and fp8[2] == 4
    comptime fp8_deep = _nvidia_gemv_config[
        DType.float8_e4m3fn, 16, 16384, True, 1024
    ]()
    comptime assert (
        fp8_deep[0] == 256 and fp8_deep[1] == 2 and fp8_deep[2] == 4
    )


# --- must be PROVEN ---
def ok_fp32_router[
    c_layout: TensorLayout,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    c_engine: TensorEngine,
    a_engine: TensorEngine,
    b_engine: TensorEngine,
](
    c: TileTensor[DType.float32, c_layout, MutAnyOrigin, Engine=c_engine],
    a: TileTensor[DType.float32, a_layout, ImmutAnyOrigin, Engine=a_engine],
    b: TileTensor[DType.float32, b_layout, ImmutAnyOrigin, Engine=b_engine],
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises:
    comptime simd_width = simd_width_of[DType.float32, target=get_gpu_target()]()
    # What the dispatcher's callers establish.
    if (
        m < 0
        or m >= 2147483648
        or n < 0
        or n >= 2147483648
        or k < 0
        or k >= 2147483648
        or Int(a.dim[0]()) < m
        or Int(c.dim[0]()) < m
        or Int(b.dim[0]()) < n
        or Int(c.dim[1]()) < n
        or Int(a.dim[1]()) < k
        or Int(b.dim[1]()) < k
        or k % simd_width != 0
    ):
        return

    # The launch of `gemv_gpu_dispatch`, for a static N.
    @__parameter
    def launch[
        static_N: Int,
        num_threads: Int,
        tile_n: Int,
        unroll_factor: Int,
        tile_m: Int,
    ]() raises:
        if n != static_N:
            return
        comptime kernel = gemv_split_k[
            DType.float32,
            DType.float32,
            DType.float32,
            c_layout,
            a_layout,
            b_layout,
            c_engine,
            a_engine,
            b_engine,
            simd_width=simd_width,
            tile_m=tile_m,
            tile_n=tile_n,
            num_threads=num_threads,
            unroll_factor=unroll_factor,
            check_bounds_m=tile_m > 1,
            check_bounds_n=static_N % tile_n != 0,
        ]
        ctx.enqueue_function[kernel](
            c,
            a,
            b,
            Int32(m),
            Int32(n),
            Int32(k),
            grid_dim=(ceildiv(m, tile_m), ceildiv(n, tile_n)),
            block_dim=num_threads,
        )

    # The dispatcher's row tiles: 1 up to m = 6, 2 up to 12, then 4.
    launch[128, 256, 1, 2, 1]()
    launch[128, 256, 1, 2, 2]()
    launch[128, 256, 1, 2, 4]()


def ok_bf16[
    c_layout: TensorLayout,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    c_engine: TensorEngine,
    a_engine: TensorEngine,
    b_engine: TensorEngine,
](
    c: TileTensor[DType.bfloat16, c_layout, MutAnyOrigin, Engine=c_engine],
    a: TileTensor[DType.bfloat16, a_layout, ImmutAnyOrigin, Engine=a_engine],
    b: TileTensor[DType.bfloat16, b_layout, ImmutAnyOrigin, Engine=b_engine],
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises:
    comptime simd_width = simd_width_of[DType.bfloat16, target=get_gpu_target()]()
    # What the dispatcher's callers establish.
    if (
        m < 0
        or m >= 2147483648
        or n < 0
        or n >= 2147483648
        or k < 0
        or k >= 2147483648
        or Int(a.dim[0]()) < m
        or Int(c.dim[0]()) < m
        or Int(b.dim[0]()) < n
        or Int(c.dim[1]()) < n
        or Int(a.dim[1]()) < k
        or Int(b.dim[1]()) < k
        or k % simd_width != 0
    ):
        return

    # The launch of `gemv_gpu_dispatch`, for a static N.
    @__parameter
    def launch[
        static_N: Int,
        num_threads: Int,
        tile_n: Int,
        unroll_factor: Int,
        tile_m: Int,
    ]() raises:
        if n != static_N:
            return
        comptime kernel = gemv_split_k[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            c_layout,
            a_layout,
            b_layout,
            c_engine,
            a_engine,
            b_engine,
            simd_width=simd_width,
            tile_m=tile_m,
            tile_n=tile_n,
            num_threads=num_threads,
            unroll_factor=unroll_factor,
            check_bounds_m=tile_m > 1,
            check_bounds_n=static_N % tile_n != 0,
        ]
        ctx.enqueue_function[kernel](
            c,
            a,
            b,
            Int32(m),
            Int32(n),
            Int32(k),
            grid_dim=(ceildiv(m, tile_m), ceildiv(n, tile_n)),
            block_dim=num_threads,
        )

    launch[4096, 128, 4, 1, 1]()
    launch[128, 128, 1, 1, 1]()
    launch[1024, 128, 2, 1, 1]()


def ok_fp8[
    c_layout: TensorLayout,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    c_engine: TensorEngine,
    a_engine: TensorEngine,
    b_engine: TensorEngine,
](
    c: TileTensor[DType.float8_e4m3fn, c_layout, MutAnyOrigin, Engine=c_engine],
    a: TileTensor[DType.float8_e4m3fn, a_layout, ImmutAnyOrigin, Engine=a_engine],
    b: TileTensor[DType.float8_e4m3fn, b_layout, ImmutAnyOrigin, Engine=b_engine],
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises:
    comptime simd_width = simd_width_of[DType.float8_e4m3fn, target=get_gpu_target()]()
    # What the dispatcher's callers establish.
    if (
        m < 0
        or m >= 2147483648
        or n < 0
        or n >= 2147483648
        or k < 0
        or k >= 2147483648
        or Int(a.dim[0]()) < m
        or Int(c.dim[0]()) < m
        or Int(b.dim[0]()) < n
        or Int(c.dim[1]()) < n
        or Int(a.dim[1]()) < k
        or Int(b.dim[1]()) < k
        or k % simd_width != 0
    ):
        return

    # The launch of `gemv_gpu_dispatch`, for a static N.
    @__parameter
    def launch[
        static_N: Int,
        num_threads: Int,
        tile_n: Int,
        unroll_factor: Int,
        tile_m: Int,
    ]() raises:
        if n != static_N:
            return
        comptime kernel = gemv_split_k[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            c_layout,
            a_layout,
            b_layout,
            c_engine,
            a_engine,
            b_engine,
            simd_width=simd_width,
            tile_m=tile_m,
            tile_n=tile_n,
            num_threads=num_threads,
            unroll_factor=unroll_factor,
            check_bounds_m=tile_m > 1,
            check_bounds_n=static_N % tile_n != 0,
        ]
        ctx.enqueue_function[kernel](
            c,
            a,
            b,
            Int32(m),
            Int32(n),
            Int32(k),
            grid_dim=(ceildiv(m, tile_m), ceildiv(n, tile_n)),
            block_dim=num_threads,
        )

    launch[8192, 64, 4, 4, 1]()
    launch[1024, 256, 2, 4, 1]()


def main():
    check_configs()
