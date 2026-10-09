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
# `linalg.gemv.gemm_mma_cpasync_kernel` and its launcher, for
# `verify-contracts`; see README.md. One launch writes a configuration out
# (`tile_k` 128, two stages) and guards what the kernel requires. Two calls
# of the launcher `gemm_mma_cpasync`, with batched and with 2-D operands,
# guard what it requires: the launcher takes its stage count from the
# device's shared memory, and is verified only because this file uses it.
# Needs the kernel packages on the import path,
# `packages=linalg,structured_kernels` and `invariant-retry=true`.

from layout import TensorLayout, TileTensor
from layout.tensor_engine import TensorEngine
from linalg.gemv import (
    _MmaCpAsyncSmem,
    gemm_mma_cpasync,
    gemm_mma_cpasync_kernel,
)
from max.gpu.host import DeviceContext
from std.math import ceildiv
from std.sys.info import has_nvidia_gpu_accelerator, size_of


def ok_launch_cpasync[
    c_layout: TensorLayout,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    c_engine: TensorEngine,
    a_engine: TensorEngine,
    b_engine: TensorEngine,
](
    c: TileTensor[DType.bfloat16, c_layout, MutAnyOrigin, Engine=c_engine],
    act: TileTensor[DType.bfloat16, a_layout, ImmutAnyOrigin, Engine=a_engine],
    weight: TileTensor[
        DType.bfloat16, b_layout, ImmutAnyOrigin, Engine=b_engine
    ],
    m: Int,
    k: Int,
    n: Int,
    batch: Int,
    ctx: DeviceContext,
) raises:
    if (
        m < 0
        or m > 2147483648 - 16  # the grid's rows, in whole tiles, fit Int32
        or k < 0
        or k >= 2147483648
        or n < 0
        or n > 2147483648 - 8
        or batch < 1
        or batch >= 2147483648
        or batch * m >= 2147483648
        or c.ptr._extent() < batch * m * n
        or Int(act.dim[0]()) < batch
        or Int(act.dim[1]()) < m
        or Int(act.dim[2]()) < k
        or Int(weight.dim[0]()) < batch
        or Int(weight.dim[1]()) < n
        or Int(weight.dim[2]()) < k
    ):
        return
    comptime Smem = _MmaCpAsyncSmem[DType.bfloat16, 16, 8, 128, 2]
    comptime kernel = gemm_mma_cpasync_kernel[
        DType.bfloat16,
        DType.bfloat16,
        DType.bfloat16,
        c_layout,
        a_layout,
        b_layout,
        c_engine,
        a_engine,
        b_engine,
        tile_m=16,
        tile_n=8,
        tile_k=128,
        stage_cnt=2,
        swapAB=False,
    ]
    comptime if has_nvidia_gpu_accelerator():
        ctx.enqueue_function[kernel](
            c,
            act,
            weight,
            Int32(m),
            Int32(k),
            Int32(n),
            Int32(batch),
            grid_dim=(ceildiv(m, 16), ceildiv(n, 8), batch),
            block_dim=256,
            shared_mem_bytes=size_of[Smem](),
        )


def ok_launcher_batched[
    c_layout: TensorLayout,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    c_engine: TensorEngine,
    a_engine: TensorEngine,
    b_engine: TensorEngine,
](
    c: TileTensor[DType.bfloat16, c_layout, MutAnyOrigin, Engine=c_engine],
    act: TileTensor[DType.bfloat16, a_layout, ImmutAnyOrigin, Engine=a_engine],
    weight: TileTensor[
        DType.bfloat16, b_layout, ImmutAnyOrigin, Engine=b_engine
    ],
    m: Int,
    k: Int,
    n: Int,
    batch: Int,
    ctx: DeviceContext,
) raises:
    """A call of the kernel's launcher with (batch, rows, K) operands."""
    comptime assert c_layout.rank == 3 and a_layout.rank == 3
    comptime assert b_layout.rank == 3
    if (
        m < 0
        or m > 2147483648 - 16
        or n < 0
        or n > 2147483648 - 16
        or k < 0
        or k >= 2147483648
        or batch < 1
        or batch >= 2147483648
        or batch * m >= 2147483648
        or batch * n >= 2147483648
        or c.ptr._extent() < batch * m * n
        or Int(act.dim[0]()) < batch
        or Int(act.dim[1]()) < m
        or Int(act.dim[2]()) < k
        or Int(weight.dim[0]()) < batch
        or Int(weight.dim[1]()) < n
        or Int(weight.dim[2]()) < k
    ):
        return
    gemm_mma_cpasync(c, act, weight, m, k, n, batch, ctx)


def ok_launcher_2d[
    c_layout: TensorLayout,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    c_engine: TensorEngine,
    a_engine: TensorEngine,
    b_engine: TensorEngine,
](
    c: TileTensor[DType.bfloat16, c_layout, MutAnyOrigin, Engine=c_engine],
    act: TileTensor[DType.bfloat16, a_layout, ImmutAnyOrigin, Engine=a_engine],
    weight: TileTensor[
        DType.bfloat16, b_layout, ImmutAnyOrigin, Engine=b_engine
    ],
    m: Int,
    k: Int,
    n: Int,
    ctx: DeviceContext,
) raises:
    """A call of the kernel's launcher with (rows, K) operands."""
    comptime assert c_layout.rank == 2 and a_layout.rank == 2
    comptime assert b_layout.rank == 2
    if (
        m < 0
        or m > 2147483648 - 16
        or n < 0
        or n > 2147483648 - 16
        or k < 0
        or k >= 2147483648
        or c.ptr._extent() < m * n
        or Int(act.dim[0]()) < m
        or Int(act.dim[1]()) < k
        or Int(weight.dim[0]()) < n
        or Int(weight.dim[1]()) < k
    ):
        return
    gemm_mma_cpasync(c, act, weight, m, k, n, 1, ctx)


def main():
    pass
