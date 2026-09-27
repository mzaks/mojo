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
# A small tensor and GPU-style kernels for `bounds-check-report`; see README.md.

from std.builtin._verification import _requires, _ensures, _assume


# A small 2D tensor with static dimensions, like a LayoutTensor with a static
# layout. Indexing states its bounds as a precondition: callers must establish
# it, and the flat index inside is proven from it.
struct Tensor2D[rows: Int, cols: Int](Movable):
    var data: Array[Float32, Self.rows * Self.cols]

    def __init__(out self):
        self.data = Array[Float32, Self.rows * Self.cols](fill=0)

    @inline(.never)
    def load(self, i: Int, j: Int) -> Float32:
        _requires(i >= 0 and i < Self.rows and j >= 0 and j < Self.cols)
        return self.data[i * Self.cols + j]

    @inline(.never)
    def store(mut self, i: Int, j: Int, value: Float32):
        _requires(i >= 0 and i < Self.rows and j >= 0 and j < Self.cols)
        self.data[i * Self.cols + j] = value


# Stands in for thread_idx / block_idx / block_dim / grid_dim, which need a GPU
# target. The accessors assume what the hardware guarantees.
@fieldwise_init
struct ThreadCtx(Copyable):
    var thread_x: Int
    var block_x: Int
    var block_dim_x: Int
    var grid_dim_x: Int
    var thread_y: Int
    var block_y: Int
    var block_dim_y: Int
    var grid_dim_y: Int

    @inline(.always)
    def block_dim(self) -> Int:
        _assume(self.block_dim_x >= 1 and self.block_dim_x <= 1024)
        return self.block_dim_x

    @inline(.always)
    def grid_dim(self) -> Int:
        _assume(self.grid_dim_x >= 1 and self.grid_dim_x <= 2147483647)
        return self.grid_dim_x

    @inline(.always)
    def thread_idx(self) -> Int:
        _assume(self.block_dim_x >= 1 and self.block_dim_x <= 1024)
        _assume(self.thread_x >= 0 and self.thread_x < self.block_dim_x)
        return self.thread_x

    @inline(.always)
    def block_idx(self) -> Int:
        _assume(self.grid_dim_x >= 1 and self.grid_dim_x <= 2147483647)
        _assume(self.block_x >= 0 and self.block_x < self.grid_dim_x)
        return self.block_x

    @inline(.always)
    def block_dim_2(self) -> Int:
        _assume(self.block_dim_y >= 1 and self.block_dim_y <= 1024)
        return self.block_dim_y

    @inline(.always)
    def grid_dim_2(self) -> Int:
        _assume(self.grid_dim_y >= 1 and self.grid_dim_y <= 65535)
        return self.grid_dim_y

    @inline(.always)
    def thread_idx_2(self) -> Int:
        _assume(self.block_dim_y >= 1 and self.block_dim_y <= 1024)
        _assume(self.thread_y >= 0 and self.thread_y < self.block_dim_y)
        return self.thread_y

    @inline(.always)
    def block_idx_2(self) -> Int:
        _assume(self.grid_dim_y >= 1 and self.grid_dim_y <= 65535)
        _assume(self.block_y >= 0 and self.block_y < self.grid_dim_y)
        return self.block_y


comptime N = 1024


# --- must stay UNPROVEN ---
@inline(.never)
def bad_kernel_unguarded(
    ctx: ThreadCtx, a: Tensor2D[1, N], mut out: Tensor2D[1, N]
):
    # The grid may have more threads than elements.
    var i = ctx.block_idx() * ctx.block_dim() + ctx.thread_idx()
    out.store(0, i, a.load(0, i))


@inline(.never)
def bad_kernel_launch_covers(
    ctx: ThreadCtx, a: Tensor2D[1, N], mut out: Tensor2D[1, N]
):
    # "At least N threads" does not bound the index.
    _requires(ctx.grid_dim() * ctx.block_dim() >= N)
    var i = ctx.block_idx() * ctx.block_dim() + ctx.thread_idx()
    out.store(0, i, a.load(0, i))


@inline(.never)
def bad_kernel_2d_swapped(ctx: ThreadCtx, mut t: Tensor2D[64, 32]):
    var row = ctx.block_idx_2() * ctx.block_dim_2() + ctx.thread_idx_2()
    var col = ctx.block_idx() * ctx.block_dim() + ctx.thread_idx()
    if row < 64 and col < 32:
        t.store(col, row, 1.0)  # rows and columns swapped


@inline(.never)
def bad_launch_too_many_blocks(a: Tensor2D[1, N], mut out: Tensor2D[1, N]):
    # 5 blocks of 256 threads: 1280 threads for 1024 elements.
    for b in range(5):
        for t in range(256):
            var ctx = ThreadCtx(t, b, 256, 5, 0, 0, 1, 1)
            ok_kernel_exact_launch(ctx, a, out)


# --- must be PROVEN ---
@inline(.never)
def ok_kernel_guarded(
    ctx: ThreadCtx, a: Tensor2D[1, N], mut out: Tensor2D[1, N]
):
    var i = ctx.block_idx() * ctx.block_dim() + ctx.thread_idx()
    if i < N:
        out.store(0, i, a.load(0, i))


@inline(.never)
def ok_kernel_exact_launch(
    ctx: ThreadCtx, a: Tensor2D[1, N], mut out: Tensor2D[1, N]
):
    # No guard: the launch configuration covers the tensor exactly.
    _requires(ctx.grid_dim() * ctx.block_dim() == N)
    var i = ctx.block_idx() * ctx.block_dim() + ctx.thread_idx()
    out.store(0, i, a.load(0, i))


@inline(.never)
def ok_kernel_2d(ctx: ThreadCtx, mut t: Tensor2D[64, 32]):
    var row = ctx.block_idx_2() * ctx.block_dim_2() + ctx.thread_idx_2()
    var col = ctx.block_idx() * ctx.block_dim() + ctx.thread_idx()
    if row < 64 and col < 32:
        t.store(row, col, 1.0)


# Known limit: the postcondition multiplies two unknown dimensions, which the
# solver cannot bound in time (proven when one side is a constant, as in
# ok_kernel_exact_launch). Callers still use it.
@inline(.never)
def limit_global_index(ctx: ThreadCtx) -> Int:
    var i = ctx.block_idx() * ctx.block_dim() + ctx.thread_idx()
    _ensures(i >= 0 and i < ctx.grid_dim() * ctx.block_dim())
    return i


@inline(.never)
def ok_kernel_via_postcondition(
    ctx: ThreadCtx, a: Tensor2D[1, N], mut out: Tensor2D[1, N]
):
    # The index comes from a helper whose postcondition bounds it.
    _requires(ctx.grid_dim() * ctx.block_dim() == N)
    var i = limit_global_index(ctx)
    out.store(0, i, a.load(0, i))


@inline(.never)
def ok_launch_exact(a: Tensor2D[1, N], mut out: Tensor2D[1, N]):
    # 4 blocks of 256 threads: the kernel's precondition holds at each call.
    for b in range(4):
        for t in range(256):
            var ctx = ThreadCtx(t, b, 256, 4, 0, 0, 1, 1)
            ok_kernel_exact_launch(ctx, a, out)


@inline(.never)
def ok_host_loop(mut t: Tensor2D[64, 32]):
    for i in range(64):
        for j in range(32):
            t.store(i, j, t.load(i, j) + 1.0)


def main():
    var a = Tensor2D[1, N]()
    var out = Tensor2D[1, N]()
    var t = Tensor2D[64, 32]()
    var ctx = ThreadCtx(0, 0, 256, 4, 0, 0, 8, 8)
    bad_kernel_unguarded(ctx, a, out)
    bad_kernel_launch_covers(ctx, a, out)
    bad_kernel_2d_swapped(ctx, t)
    bad_launch_too_many_blocks(a, out)
    ok_kernel_guarded(ctx, a, out)
    ok_kernel_exact_launch(ctx, a, out)
    ok_kernel_2d(ctx, t)
    ok_kernel_via_postcondition(ctx, a, out)
    ok_launch_exact(a, out)
    ok_host_loop(t)
