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
# Calls of the GPU matmul dispatcher with static N and K and a runtime M,
# for `verify-contracts` with `packages=linalg`: every launcher and kernel
# the dispatcher reaches for these shapes is verified for the launched
# instantiation (run as main files, the kernel files are verified only
# generically, with no launch facts). See README.md.
# Needs the kernel packages on the import path.

from layout import Coord, Idx, TileTensor, row_major
from linalg.matmul.gpu import _matmul_gpu
from max.gpu.host import DeviceContext


def ok_bf16_transposed(ctx: DeviceContext, m: Int) raises:
    """A decode-sized bf16 GEMM, B transposed (N x K), as served."""
    comptime N = 4096
    comptime K = 4096
    if m <= 0 or m >= 65536:
        return
    var a_dev = ctx.enqueue_create_buffer[DType.bfloat16](m * K)
    var b_dev = ctx.enqueue_create_buffer[DType.bfloat16](N * K)
    var c_dev = ctx.enqueue_create_buffer[DType.bfloat16](m * N)
    var a = TileTensor(a_dev, row_major(Coord(m, Idx[K])))
    var b = TileTensor(b_dev, row_major(Coord(Idx[N], Idx[K])))
    var c = TileTensor(c_dev, row_major(Coord(m, Idx[N])))
    _matmul_gpu[use_tensor_core=True, transpose_b=True](
        c, a.as_imm(), b.as_imm(), ctx
    )


def main():
    pass
