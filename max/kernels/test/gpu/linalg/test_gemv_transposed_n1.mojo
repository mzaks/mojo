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
# GEMV with N == 1 and transpose_b=True, M > 1.
#
# `gemv_gpu` picks `GEMV_KERNEL` for N == 1 (float32 always) without checking
# `transpose_b`; `gemv_gpu_dispatch` then takes its transposed launch, which
# runs the kernel over N rows (`c[j] = dot(B[j], A[0])`). With N == 1 that is
# one row, so for M > 1 rows 1.. of C would never be written. With N == 1, B
# is the same K contiguous elements whether it is K x 1 or 1 x K, so the
# expected result is `c[i] = dot(A[i], B)` for every i < M in both layouts.
#
# Cases: the non-transposed launch and the transposed one with M == 1 are
# controls; the transposed one with M == 4 is the case in question, called
# directly and through `_matmul_gpu` (which calls `gemv_gpu` for n == 1).

from max.gpu.host import DeviceContext
from layout import Coord, TileTensor, row_major
from linalg.gemv import gemv_gpu
from linalg.matmul.gpu import _matmul_gpu


def run_case[
    transpose_b: Bool, through_matmul: Bool
](ctx: DeviceContext, m: Int, k: Int) raises -> Bool:
    comptime dtype = DType.float32
    var n = 1
    print(
        "M =",
        m,
        "N = 1, K =",
        k,
        "transpose_b =",
        transpose_b,
        "through _matmul_gpu =" if through_matmul else "through gemv_gpu =",
        True,
    )

    var a_host = ctx.enqueue_create_host_buffer[dtype](m * k)
    var b_host = ctx.enqueue_create_host_buffer[dtype](k)
    var c_host = ctx.enqueue_create_host_buffer[dtype](m)
    # Small integers: every sum is exact in float32.
    for i in range(m * k):
        a_host[i] = Float32((i * 7) % 11 - 5)
    for i in range(k):
        b_host[i] = Float32((i * 3) % 7 - 3)
    # A sentinel no dot product of these values reaches: a row the kernel
    # never writes keeps it.
    comptime sentinel = Float32(-123456)
    for i in range(m):
        c_host[i] = sentinel

    var a_dev = ctx.enqueue_create_buffer[dtype](m * k)
    var b_dev = ctx.enqueue_create_buffer[dtype](k)
    var c_dev = ctx.enqueue_create_buffer[dtype](m)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.enqueue_copy(c_dev, c_host)

    var a = TileTensor(a_dev, row_major(Coord(m, k)))
    # B is N x K transposed, K x N otherwise; with N == 1 the same memory.
    var b = TileTensor(
        b_dev,
        row_major(Coord(n if transpose_b else k, k if transpose_b else n)),
    )
    var c = TileTensor(c_dev, row_major(Coord(m, n)))

    comptime if through_matmul:
        _matmul_gpu[transpose_b=transpose_b](c, a.as_imm(), b.as_imm(), ctx)
    else:
        gemv_gpu[transpose_b=transpose_b](c, a.as_imm(), b.as_imm(), ctx)

    ctx.enqueue_copy(c_host, c_dev)
    ctx.synchronize()

    var errors = 0
    for i in range(m):
        var expected = Float32(0)
        for j in range(k):
            expected += a_host[i * k + j] * b_host[j]
        if c_host[i] != expected:
            if errors < 8:
                print(
                    "  row",
                    i,
                    ": got",
                    c_host[i],
                    "(never written)" if c_host[i] == sentinel else "",
                    "expected",
                    expected,
                )
            errors += 1
    print("  errors:", errors, "of", m, "rows")
    return errors == 0


def main() raises:
    with DeviceContext() as ctx:
        var failed = List[String]()
        # Controls.
        if not run_case[transpose_b=False, through_matmul=False](ctx, 4, 64):
            failed.append("gemv_gpu, not transposed, M = 4")
        if not run_case[transpose_b=True, through_matmul=False](ctx, 1, 64):
            failed.append("gemv_gpu, transposed, M = 1")
        # The case in question.
        if not run_case[transpose_b=True, through_matmul=False](ctx, 4, 64):
            failed.append("gemv_gpu, transposed, M = 4")
        if not run_case[transpose_b=True, through_matmul=True](ctx, 4, 64):
            failed.append("_matmul_gpu, transposed, M = 4")
        if failed:
            for f in failed:
                print("FAILED:", f)
            raise Error("GEMV with transpose_b=True and N == 1 failed")
        print("PASS")
