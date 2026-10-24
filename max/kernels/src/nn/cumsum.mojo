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
"""Implements the ONNX CumSum operator, computing prefix sums along a specified tensor axis."""

from std.algorithm import vectorize
from std.math import align_up, ceildiv
from std.memory import stack_allocation
from std.sys import align_of, bit_width_of
from std.sys.info import simd_width_of
from std.utils.static_tuple import StaticTuple

from layout import (
    Coord,
    ImmTileTensor,
    MutTileTensor,
    TensorLayout,
    TileTensor,
    coord_to_index_list,
    row_major,
)
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
    global_idx,
    thread_idx,
)
from max.gpu.host import DeviceContext
from max.gpu.host.info import is_cpu
from max.gpu.memory import AddressSpace
from max.gpu.primitives import block
from max.gpu.sync import barrier
from max.runtime.tracing import Trace, TraceLevel, get_safe_task_id

from std.utils.numerics import get_accum_type

# Row-scan geometries. Long integral rows are scanned in one trip per block:
# 512 threads folding 16 elements each, one wide access per thread, covers an
# 8192-element row; past 12288 elements 1024 threads cover up to 16384 in one
# trip. Other rows stay on the small tile, where a wide geometry's mostly idle
# block costs more than it saves, and floating rows keep that tile so their
# rounding sequence is the one the previous revision produced.
comptime _CUMSUM_SCAN_THREADS = 512
comptime _CUMSUM_SCAN_ITEMS = 16
comptime _CUMSUM_LONG_THREADS = 1024
comptime _CUMSUM_LONG_ITEMS = 16
comptime _CUMSUM_NARROW_THREADS = 256
comptime _CUMSUM_NARROW_ITEMS = 4
# Integral rows at least this long take the wide geometry, and the longer one;
# the second bound is the 512-thread tile plus one, so no wide row takes two
# trips (int32[10240]: 10.30 us at 512 threads, 4.17 us here). Rows past one
# long trip keep the long tile too: capping it there fixes int32[102400]
# (20.53 us behind the long tile, 10.31 us capped and split) but costs
# int32[65536] (12.37 us behind the long tile, 14.40 us capped), and which
# side wins turns on whether the row splits. Open, in the shapes campaign.
comptime _CUMSUM_WIDE_MIN_LEN = 4096
comptime _CUMSUM_LONG_MIN_LEN = 8193
# The long geometry keeps 16 elements per thread, which is 64 bytes at 4-byte
# elements. At 8 bytes it is 128 bytes per thread, and 1024 threads per block
# leaves 64 registers each, so 8-byte integrals spill there: int64[65536] reads
# 12.37 us on the 512-thread geometry and 38.97 us on the long one. They stay.
comptime _CUMSUM_LONG_MAX_WIDTH = 32
# Threads per block of the chunk-sum kernel, and the independent loads each
# thread keeps in flight per trip.
comptime _CUMSUM_BLOCK_SIZE = 256
comptime _CUMSUM_TOTAL_ITEMS = 16

# Contiguous rows at least this long always take the row-scan kernel. Shorter
# rows take it only when they are too few to occupy the GPU one thread each
# (see `_use_row_scan`): with many short rows, a block of
# `_CUMSUM_NARROW_THREADS` threads per row is mostly idle threads, and one
# thread per row is faster.
comptime _CUMSUM_MIN_ROW_SCAN_LEN = 128

# Threads per block of the one-thread-per-line kernels.
comptime _CUMSUM_SERIAL_BLOCK_SIZE = 256

# A line is split into chunks scanned by separate blocks (or threads) only when
# there are too few lines to occupy the GPU. Row-scan blocks per SM that count
# as "full", the blocks per SM to aim for when splitting, and the smallest
# chunk worth a block.
comptime _CUMSUM_FILL_BLOCKS_PER_SM = 2
comptime _CUMSUM_TARGET_BLOCKS_PER_SM = 8
comptime _CUMSUM_MIN_CHUNK_TILES = 4
# The same for the one-thread-per-line kernels, in threads per SM and elements.
comptime _CUMSUM_TARGET_THREADS_PER_SM = 2048
comptime _CUMSUM_MIN_SERIAL_CHUNK = 32


@inline(.always)
def cumsum[
    dtype: DType,
    exclusive: Bool,
    reverse: Bool,
    *,
    axis: Int,
](output: TileTensor[mut=True, dtype, ...], input: TileTensor[dtype, ...],):
    """
    Implements the CumSum operator from the ONNX spec:
    https://github.com/onnx/onnx/blob/main/docs/Operators.md#CumSum
    Computes cumulative sum of the input elements along the given axis.
    Cumulative sum can be inclusive or exclusive of the top element, and
    normal or reverse (direction along a given axis).

    Parameters:
        dtype: Type of the input and output tensors.
        exclusive: If set to True, return exclusive sum (top element not included).
        reverse: If set to True, perform cumsum operation in reverse direction.
        axis: The axis on which to perform the cumsum operation.

    Args:
        output: The output tensor.
        input: The input tensor.
    """
    comptime assert (
        input.rank == output.rank
    ), "input and output should have the same rank."

    comptime accum_type = DType.float64 if dtype == DType.float32 else get_accum_type[
        dtype
    ]()
    comptime assert (
        -input.rank <= axis < input.rank
    ), "Axis value must be in range [-rank, rank)"
    comptime axis_pos = axis if axis >= 0 else axis + input.rank

    var shape = coord_to_index_list(input.layout.shape_coord())

    var inner = 1
    var outer = 1
    var depth = 1
    for i in range(input.rank):
        if i < axis_pos:
            inner *= shape[i]
        elif i > axis_pos:
            outer *= shape[i]
        else:
            depth = shape[i]

    var output_data = output.reshape(row_major(output.num_elements()))
    var input_data = input.reshape(row_major(input.num_elements()))

    comptime simd_width = simd_width_of[accum_type]()
    for inner_index in range(inner):
        # SIMD lanes scan independent trailing positions without reassociating
        # the additions along the scan axis.
        def scan_chunk[
            width: Int
        ](outer_index: Int) {
            input_data, output_data, inner_index, depth, outer
        }:
            var accumulator = SIMD[accum_type, width](0)
            for depth_index in range(depth):
                var depth_index_adj = (
                    depth - 1 - depth_index if reverse else depth_index
                )
                var index = (
                    outer_index
                    + inner_index * depth * outer
                    + depth_index_adj * outer
                )

                comptime if exclusive:
                    output_data.store((index,), accumulator.cast[dtype]())
                    accumulator += input_data.load[width=width]((index,)).cast[
                        accum_type
                    ]()
                else:
                    accumulator += input_data.load[width=width]((index,)).cast[
                        accum_type
                    ]()
                    output_data.store((index,), accumulator.cast[dtype]())

        vectorize[simd_width](outer, scan_chunk)


def _gpu_accum_type[dtype: DType]() -> DType:
    # Warp shuffles move 32- and 64-bit lanes only, so narrow integers scan in
    # 32 bits. Truncating back to `dtype` gives the same wrapped result.
    comptime if dtype.is_integral() and bit_width_of[dtype]() < 32:
        return DType.int32 if dtype.is_signed() else DType.uint32
    return get_accum_type[dtype]()


@inline(.always)
def _cumsum_shape[axis: Int](input: ImmTileTensor[...]) -> Tuple[Int, Int, Int]:
    """Returns `(num_rows, axis_len, stride)` for a row-major `input`.

    `num_rows` is the product of the dims before `axis` and `stride` the
    product of the dims after it, so `input` reshapes to
    `[num_rows, axis_len, stride]`.
    """
    comptime axis_pos = axis if axis >= 0 else axis + input.rank
    var shape = coord_to_index_list(input.layout.shape_coord())
    var num_rows = 1
    var stride = 1
    comptime for i in range(input.rank):
        comptime if i < axis_pos:
            num_rows *= shape[i]
        elif i > axis_pos:
            stride *= shape[i]
    return (num_rows, shape[axis_pos], stride)


@inline(.always)
def _scan_pos[reverse: Bool](index: Int, axis_len: Int) -> Int:
    """Maps the `index`-th element in scan order to its position on the axis."""
    comptime if reverse:
        return axis_len - 1 - index
    return index


@inline(.always)
def _fold_lanes[
    accum_type: DType, exclusive: Bool, width: Int
](values: SIMD[accum_type, width], mut running: Scalar[accum_type]) -> SIMD[
    accum_type, width
]:
    """Folds `values` into `running` in lane order and returns the inclusive
    prefixes of the lanes, or the exclusive ones (`running` before each lane)
    when `exclusive`. Lanes past a chunk's end are zero, so a ragged tail adds
    nothing."""
    var prefixes = SIMD[accum_type, width](0)
    comptime for k in range(width):
        comptime if exclusive:
            prefixes[k] = running
            running += values[k]
        else:
            running += values[k]
            prefixes[k] = running
    return prefixes


@inline(.always)
def _row_scan_body[
    dtype: DType,
    accum_type: DType,
    exclusive: Bool,
    reverse: Bool,
    has_carry: Bool,
    threads: Int,
    items: Int,
    wide: Bool,
    OutLayoutType: TensorLayout,
    InLayoutType: TensorLayout,
    CarryLayoutType: TensorLayout,
](
    output: TileTensor[dtype, OutLayoutType, MutAnyOrigin],
    input: TileTensor[dtype, InLayoutType, ImmutAnyOrigin],
    carry: TileTensor[accum_type, CarryLayoutType, ImmutAnyOrigin],
    chunk_len: Int,
):
    """Scans one `chunk_len`-long chunk of a row of a `[num_rows, axis_len]`
    input; block `block_idx.x` takes chunk `block_idx.x % num_chunks` of row
    `block_idx.x // num_chunks`.

    `threads` threads each fold `items` consecutive elements -- as one wide
    access when `wide` and the addresses allow it, else one element at a time --
    and a block scan over the per-thread totals places each thread. The tile
    total carries into the next trip along the chunk; a chunk that fits in one
    tile has no next trip and skips that handoff. With `has_carry`, the sum of
    all earlier chunks of the row, `carry[row, chunk]`, seeds the first trip.
    """
    comptime assert input.flat_rank == 2 and output.flat_rank == 2
    comptime tile = threads * items
    var axis_len = Int(input.dim[1]())
    # Without a carry, the chunk is the whole row; skip the 64-bit division.
    var row = block_idx.x
    var chunk = 0
    comptime if has_carry:
        row, chunk = divmod(block_idx.x, ceildiv(axis_len, chunk_len))
    var begin = chunk * chunk_len
    var end = min(begin + chunk_len, axis_len)
    var tid = thread_idx.x
    # A wide access promises the alignment of the whole access, which is the
    # default `alignment` of `load`/`store`. `first`, `begin` and the row
    # starts are multiples of `items` elements, but a view into a larger buffer
    # need not start at one, so require it of both operands and fold element by
    # element otherwise.
    comptime wide_align = align_of[SIMD[dtype, items]]()
    var wide_ok = (
        (row * axis_len) % items == 0
        and begin % items == 0
        and Int(input.ptr) % wide_align == 0
        and Int(output.ptr) % wide_align == 0
    )
    var tile_total = stack_allocation[
        1, Scalar[accum_type], address_space=AddressSpace.SHARED
    ]()

    var carry_in: Scalar[accum_type] = 0
    comptime if has_carry:
        carry_in = carry[row, chunk]
    for tile_start in range(begin, end, tile):
        var first = tile_start + tid * items
        var vector = wide and wide_ok and first + items <= end
        # Gather the thread's `items` elements in scan order -- one access per
        # lane, one wide access for the whole group when the tile, the row and
        # the buffers line up -- then fold them in one place.
        var values = SIMD[accum_type, items](0)
        comptime if reverse:
            # A reverse scan reads the axis backwards, so the elements one
            # thread owns are not contiguous in memory.
            comptime for k in range(items):
                if first + k < end:
                    values[k] = input[
                        row, _scan_pos[reverse](first + k, axis_len)
                    ].cast[accum_type]()
        else:
            if vector:
                # One index per mode: a single flat index is distributed
                # across the modes, which only coincides with the element
                # offset when the row count is one.
                values = input.load[width=items](Coord(row, first)).cast[
                    accum_type
                ]()
            else:
                # Ragged tail of the chunk: element at a time.
                comptime for k in range(items):
                    if first + k < end:
                        values[k] = input[row, first + k].cast[accum_type]()
        var thread_sum: Scalar[accum_type] = 0
        var partial = _fold_lanes[accum_type, exclusive](values, thread_sum)

        var offset = carry_in + block.prefix_sum[
            block_size=threads, exclusive=True
        ](thread_sum)

        comptime if reverse:
            comptime for k in range(items):
                if first + k < end:
                    output[row, _scan_pos[reverse](first + k, axis_len)] = (
                        offset + partial[k]
                    ).cast[dtype]()
        else:
            if vector:
                var out_vec = (partial + SIMD[accum_type, items](offset)).cast[
                    dtype
                ]()
                output.store[width=items](Coord(row, first), out_vec)
            else:
                comptime for k in range(items):
                    if first + k < end:
                        output[row, first + k] = (offset + partial[k]).cast[
                            dtype
                        ]()

        # A chunk that fits in one tile has no next trip to carry into, so it
        # skips the handoff and its two barriers.
        if tile < chunk_len:
            if tid == threads - 1:
                tile_total[0] = offset + thread_sum
            barrier()
            carry_in = tile_total[0]
            # The last thread rewrites `tile_total` on the next trip.
            barrier()


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(_CUMSUM_SCAN_THREADS)
    )
)
@__name(t"cumsum_row_scan_{dtype}_{exclusive}_{reverse}_{has_carry}")
def _cumsum_row_scan_kernel[
    dtype: DType,
    accum_type: DType,
    exclusive: Bool,
    reverse: Bool,
    has_carry: Bool,
    OutLayoutType: TensorLayout,
    InLayoutType: TensorLayout,
    CarryLayoutType: TensorLayout,
](
    output: TileTensor[dtype, OutLayoutType, MutAnyOrigin],
    input: TileTensor[dtype, InLayoutType, ImmutAnyOrigin],
    carry: TileTensor[accum_type, CarryLayoutType, ImmutAnyOrigin],
    chunk_len_arg: Int32,
):
    """Scans a chunk of a long row with the wide one-trip geometry."""
    _row_scan_body[
        dtype,
        accum_type,
        exclusive,
        reverse,
        has_carry,
        _CUMSUM_SCAN_THREADS,
        _CUMSUM_SCAN_ITEMS,
        True,
        OutLayoutType,
        InLayoutType,
        CarryLayoutType,
    ](output, input, carry, Int(chunk_len_arg))


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(_CUMSUM_LONG_THREADS)
    )
)
@__name(t"cumsum_row_scan_long_{dtype}_{exclusive}_{reverse}_{has_carry}")
def _cumsum_row_scan_long_kernel[
    dtype: DType,
    accum_type: DType,
    exclusive: Bool,
    reverse: Bool,
    has_carry: Bool,
    OutLayoutType: TensorLayout,
    InLayoutType: TensorLayout,
    CarryLayoutType: TensorLayout,
](
    output: TileTensor[dtype, OutLayoutType, MutAnyOrigin],
    input: TileTensor[dtype, InLayoutType, ImmutAnyOrigin],
    carry: TileTensor[accum_type, CarryLayoutType, ImmutAnyOrigin],
    chunk_len_arg: Int32,
):
    """Scans a chunk of a long row: 1024 threads x 16 elements, one trip to
    16384 elements."""
    _row_scan_body[
        dtype,
        accum_type,
        exclusive,
        reverse,
        has_carry,
        _CUMSUM_LONG_THREADS,
        _CUMSUM_LONG_ITEMS,
        True,
        OutLayoutType,
        InLayoutType,
        CarryLayoutType,
    ](output, input, carry, Int(chunk_len_arg))


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(_CUMSUM_NARROW_THREADS)
    )
)
@__name(t"cumsum_row_scan_narrow_{dtype}_{exclusive}_{reverse}_{has_carry}")
def _cumsum_row_scan_narrow_kernel[
    dtype: DType,
    accum_type: DType,
    exclusive: Bool,
    reverse: Bool,
    has_carry: Bool,
    OutLayoutType: TensorLayout,
    InLayoutType: TensorLayout,
    CarryLayoutType: TensorLayout,
](
    output: TileTensor[dtype, OutLayoutType, MutAnyOrigin],
    input: TileTensor[dtype, InLayoutType, ImmutAnyOrigin],
    carry: TileTensor[accum_type, CarryLayoutType, ImmutAnyOrigin],
    chunk_len_arg: Int32,
):
    """Scans a chunk of a short row, one element per thread at a time."""
    _row_scan_body[
        dtype,
        accum_type,
        exclusive,
        reverse,
        has_carry,
        _CUMSUM_NARROW_THREADS,
        _CUMSUM_NARROW_ITEMS,
        False,
        OutLayoutType,
        InLayoutType,
        CarryLayoutType,
    ](output, input, carry, Int(chunk_len_arg))


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(_CUMSUM_BLOCK_SIZE)
    )
)
@__name(t"cumsum_row_total_{dtype}_{reverse}")
def _cumsum_row_total_kernel[
    dtype: DType,
    accum_type: DType,
    reverse: Bool,
    InLayoutType: TensorLayout,
    TotalLayoutType: TensorLayout,
](
    totals: TileTensor[accum_type, TotalLayoutType, MutAnyOrigin],
    input: TileTensor[dtype, InLayoutType, ImmutAnyOrigin],
    chunk_len_arg: Int32,
):
    """Writes the sum of each `chunk_len`-long chunk of a `[num_rows,
    axis_len]` input to `totals[row, chunk]`, one block per chunk."""
    comptime assert input.flat_rank == 2 and totals.flat_rank == 2
    comptime tile = _CUMSUM_BLOCK_SIZE * _CUMSUM_TOTAL_ITEMS
    var chunk_len = Int(chunk_len_arg)
    var axis_len = Int(input.dim[1]())
    var row, chunk = divmod(block_idx.x, ceildiv(axis_len, chunk_len))
    var begin = chunk * chunk_len
    var end = min(begin + chunk_len, axis_len)
    var tid = thread_idx.x

    var total: Scalar[accum_type] = 0
    for tile_start in range(begin, end, tile):
        # Consecutive threads read consecutive elements for coalescing.
        comptime for k in range(_CUMSUM_TOTAL_ITEMS):
            var index = tile_start + k * _CUMSUM_BLOCK_SIZE + tid
            if index < end:
                total += input[row, _scan_pos[reverse](index, axis_len)].cast[
                    accum_type
                ]()
    total = block.sum[block_size=_CUMSUM_BLOCK_SIZE, broadcast=False](total)
    if tid == 0:
        totals[row, chunk] = total


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(_CUMSUM_SERIAL_BLOCK_SIZE)
    )
)
@__name(t"cumsum_serial_{dtype}_{exclusive}_{reverse}_{has_carry}")
def _cumsum_serial_kernel[
    dtype: DType,
    accum_type: DType,
    exclusive: Bool,
    reverse: Bool,
    has_carry: Bool,
    OutLayoutType: TensorLayout,
    InLayoutType: TensorLayout,
    CarryLayoutType: TensorLayout,
](
    output: TileTensor[dtype, OutLayoutType, MutAnyOrigin],
    input: TileTensor[dtype, InLayoutType, ImmutAnyOrigin],
    carry: TileTensor[accum_type, CarryLayoutType, ImmutAnyOrigin],
    chunk_len_arg: Int32,
):
    """Scans one `chunk_len`-long chunk of one `[row, :, col]` line of a
    `[num_rows, axis_len, stride]` input per thread.

    Adjacent threads own adjacent columns, so when `stride > 1` each step
    along the axis is a coalesced load across the warp. With `has_carry`,
    `carry[row * stride + col, chunk]` is the sum of all earlier chunks of the
    line.
    """
    comptime assert input.flat_rank == 3 and output.flat_rank == 3
    var chunk_len = Int(chunk_len_arg)
    var axis_len = Int(input.dim[1]())
    var stride = Int(input.dim[2]())
    # Without a carry, the chunk is the whole line; skip the 64-bit division.
    var num_chunks = 1
    comptime if has_carry:
        num_chunks = ceildiv(axis_len, chunk_len)
    if global_idx.x >= Int(input.dim[0]()) * stride * num_chunks:
        return
    var row_chunk, col = divmod(global_idx.x, stride)
    var row = row_chunk
    var chunk = 0
    comptime if has_carry:
        row, chunk = divmod(row_chunk, num_chunks)
    var begin = chunk * chunk_len
    var end = min(begin + chunk_len, axis_len)

    var acc: Scalar[accum_type] = 0
    comptime if has_carry:
        acc = carry[row * stride + col, chunk]
    for i in range(begin, end):
        var pos = _scan_pos[reverse](i, axis_len)
        var val = input[row, pos, col].cast[accum_type]()
        comptime if exclusive:
            output[row, pos, col] = acc.cast[dtype]()
            acc += val
        else:
            acc += val
            output[row, pos, col] = acc.cast[dtype]()


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(_CUMSUM_SERIAL_BLOCK_SIZE)
    )
)
@__name(t"cumsum_serial_total_{dtype}_{reverse}")
def _cumsum_serial_total_kernel[
    dtype: DType,
    accum_type: DType,
    reverse: Bool,
    InLayoutType: TensorLayout,
    TotalLayoutType: TensorLayout,
](
    totals: TileTensor[accum_type, TotalLayoutType, MutAnyOrigin],
    input: TileTensor[dtype, InLayoutType, ImmutAnyOrigin],
    chunk_len_arg: Int32,
):
    """Writes the sum of each `chunk_len`-long chunk of each line of a
    `[num_rows, axis_len, stride]` input to `totals[row * stride + col, chunk]`,
    one thread per chunk."""
    comptime assert input.flat_rank == 3 and totals.flat_rank == 2
    var chunk_len = Int(chunk_len_arg)
    var axis_len = Int(input.dim[1]())
    var stride = Int(input.dim[2]())
    var num_chunks = ceildiv(axis_len, chunk_len)
    if global_idx.x >= Int(input.dim[0]()) * stride * num_chunks:
        return
    var row_chunk, col = divmod(global_idx.x, stride)
    var row, chunk = divmod(row_chunk, num_chunks)
    var begin = chunk * chunk_len
    var end = min(begin + chunk_len, axis_len)

    var total: Scalar[accum_type] = 0
    for i in range(begin, end):
        total += input[row, _scan_pos[reverse](i, axis_len), col].cast[
            accum_type
        ]()
    totals[row * stride + col, chunk] = total


def _enqueue_scan[
    dtype: DType,
    accum_type: DType,
    exclusive: Bool,
    reverse: Bool,
    has_carry: Bool,
](
    output: MutTileTensor[dtype, ...],
    input: ImmTileTensor[dtype, ...],
    carry: ImmTileTensor[accum_type, ...],
    num_rows: Int,
    axis_len: Int,
    stride: Int,
    chunk_len: Int,
    row_scan: Bool,
    wide: Bool,
    long_ok: Bool,
    ctx: DeviceContext,
) raises:
    """Scans every `chunk_len`-long chunk of `[num_rows, axis_len, stride]`.

    Without `has_carry` each line is one chunk starting at zero (`chunk_len`
    must be `axis_len`) and `carry` is never read. `row_scan` selects
    the block-per-chunk kernel, which needs `stride == 1`; `wide` selects the
    wide geometry over the narrow one within that kernel.
    """
    var num_chunks = ceildiv(axis_len, chunk_len)
    if row_scan:
        var rows_layout = row_major((num_rows, axis_len))
        var out_rows = output.reshape(rows_layout)
        var in_rows = input.reshape(rows_layout)
        # `_chunk_len` tiles a row with the geometry this shape takes.
        if wide:
            if long_ok and axis_len >= _CUMSUM_LONG_MIN_LEN:
                comptime kernel = _cumsum_row_scan_long_kernel[
                    dtype,
                    accum_type,
                    exclusive,
                    reverse,
                    has_carry,
                    out_rows.LayoutType,
                    in_rows.LayoutType,
                    carry.LayoutType,
                ]
                ctx.enqueue_function[kernel](
                    out_rows,
                    in_rows.as_imm(),
                    carry,
                    Int32(chunk_len),
                    grid_dim=num_rows * num_chunks,
                    block_dim=_CUMSUM_LONG_THREADS,
                )
                return
            comptime kernel = _cumsum_row_scan_kernel[
                dtype,
                accum_type,
                exclusive,
                reverse,
                has_carry,
                out_rows.LayoutType,
                in_rows.LayoutType,
                carry.LayoutType,
            ]
            ctx.enqueue_function[kernel](
                out_rows,
                in_rows.as_imm(),
                carry,
                Int32(chunk_len),
                grid_dim=num_rows * num_chunks,
                block_dim=_CUMSUM_SCAN_THREADS,
            )
            return
        comptime kernel = _cumsum_row_scan_narrow_kernel[
            dtype,
            accum_type,
            exclusive,
            reverse,
            has_carry,
            out_rows.LayoutType,
            in_rows.LayoutType,
            carry.LayoutType,
        ]
        ctx.enqueue_function[kernel](
            out_rows,
            in_rows.as_imm(),
            carry,
            Int32(chunk_len),
            grid_dim=num_rows * num_chunks,
            block_dim=_CUMSUM_NARROW_THREADS,
        )
        return

    var lines_layout = row_major((num_rows, axis_len, stride))
    var out_lines = output.reshape(lines_layout)
    var in_lines = input.reshape(lines_layout)
    comptime kernel = _cumsum_serial_kernel[
        dtype,
        accum_type,
        exclusive,
        reverse,
        has_carry,
        out_lines.LayoutType,
        in_lines.LayoutType,
        carry.LayoutType,
    ]
    ctx.enqueue_function[kernel](
        out_lines,
        in_lines.as_imm(),
        carry,
        Int32(chunk_len),
        grid_dim=ceildiv(
            num_rows * stride * num_chunks, _CUMSUM_SERIAL_BLOCK_SIZE
        ),
        block_dim=_CUMSUM_SERIAL_BLOCK_SIZE,
    )


def _enqueue_chunk_totals[
    dtype: DType, accum_type: DType, reverse: Bool
](
    totals: MutTileTensor[accum_type, ...],
    input: ImmTileTensor[dtype, ...],
    num_rows: Int,
    axis_len: Int,
    stride: Int,
    chunk_len: Int,
    row_scan: Bool,
    ctx: DeviceContext,
) raises:
    """Writes the per-chunk sums of `[num_rows, axis_len, stride]` to the
    `[num_rows * stride, num_chunks]` tensor `totals`, in scan order."""
    var num_chunks = ceildiv(axis_len, chunk_len)
    if row_scan:
        var in_rows = input.reshape(row_major((num_rows, axis_len)))
        comptime kernel = _cumsum_row_total_kernel[
            dtype,
            accum_type,
            reverse,
            in_rows.LayoutType,
            totals.LayoutType,
        ]
        ctx.enqueue_function[kernel](
            totals,
            in_rows.as_imm(),
            Int32(chunk_len),
            grid_dim=num_rows * num_chunks,
            block_dim=_CUMSUM_BLOCK_SIZE,
        )
        return

    var in_lines = input.reshape(row_major((num_rows, axis_len, stride)))
    comptime kernel = _cumsum_serial_total_kernel[
        dtype, accum_type, reverse, in_lines.LayoutType, totals.LayoutType
    ]
    ctx.enqueue_function[kernel](
        totals,
        in_lines.as_imm(),
        Int32(chunk_len),
        grid_dim=ceildiv(
            num_rows * stride * num_chunks, _CUMSUM_SERIAL_BLOCK_SIZE
        ),
        block_dim=_CUMSUM_SERIAL_BLOCK_SIZE,
    )


def _chunk_len(
    row_scan: Bool,
    num_lines: Int,
    axis_len: Int,
    sm_count: Int,
    wide: Bool,
    long_ok: Bool,
) -> Int:
    """Returns the axis span one block (or thread) scans; `axis_len` means the
    axis is not split.

    A line is split only when there are too few lines to occupy the GPU. The
    decision depends on shapes alone so that it replays under graph capture.
    """
    var chunk_len: Int
    if row_scan:
        if num_lines >= _CUMSUM_FILL_BLOCKS_PER_SM * sm_count:
            return axis_len
        # The tile of the geometry `_enqueue_scan` picks for this row.
        var tile: Int
        if not wide:
            tile = _CUMSUM_NARROW_THREADS * _CUMSUM_NARROW_ITEMS
        elif long_ok and axis_len >= _CUMSUM_LONG_MIN_LEN:
            tile = _CUMSUM_LONG_THREADS * _CUMSUM_LONG_ITEMS
        else:
            tile = _CUMSUM_SCAN_THREADS * _CUMSUM_SCAN_ITEMS
        var num_chunks = ceildiv(
            _CUMSUM_TARGET_BLOCKS_PER_SM * sm_count, num_lines
        )
        chunk_len = align_up(ceildiv(axis_len, num_chunks), tile)
        chunk_len = max(chunk_len, _CUMSUM_MIN_CHUNK_TILES * tile)
    else:
        if num_lines >= _CUMSUM_TARGET_THREADS_PER_SM * sm_count:
            return axis_len
        var num_chunks = ceildiv(
            _CUMSUM_TARGET_THREADS_PER_SM * sm_count, num_lines
        )
        chunk_len = max(ceildiv(axis_len, num_chunks), _CUMSUM_MIN_SERIAL_CHUNK)
    # Two chunks do not repay the extra launches.
    return axis_len if 2 * chunk_len >= axis_len else chunk_len


def _use_row_scan(
    num_lines: Int, axis_len: Int, stride: Int, sm_count: Int
) -> Bool:
    """Returns whether `num_lines` lines of `axis_len` take the row-scan
    kernel rather than the one-thread-per-line one.

    Row scan needs contiguous lines (`stride == 1`). Below
    `_CUMSUM_MIN_ROW_SCAN_LEN`, it is chosen only where the one-thread-per-line
    kernel would split the lines into chunks, which costs three launches and
    two scratch buffers where row scan takes one launch.
    """
    if stride != 1:
        return False
    return (
        axis_len >= _CUMSUM_MIN_ROW_SCAN_LEN
        or _chunk_len(False, num_lines, axis_len, sm_count, False, False)
        < axis_len
    )


def _cumsum_gpu[
    dtype: DType,
    exclusive: Bool,
    reverse: Bool,
    *,
    axis: Int,
](
    output: MutTileTensor[dtype, ...],
    input: ImmTileTensor[dtype, ...],
    ctx: DeviceContext,
) raises:
    # Unlike the CPU path, float32 accumulates in float32: float64 is slow on
    # most GPUs and unsupported on Apple silicon.
    comptime accum_type = _gpu_accum_type[dtype]()
    # Graph tensors are contiguous but do not carry static row-major strides,
    # so the views below are built from the base pointer, as on CPU.
    var num_rows, axis_len, stride = _cumsum_shape[axis](input)
    if num_rows * axis_len * stride == 0:
        return

    var num_lines = num_rows * stride
    var sm_count = ctx.default_device_info.sm_count
    var row_scan = _use_row_scan(num_lines, axis_len, stride, sm_count)
    # Floating rows keep the narrow geometry so their rounding sequence is
    # exactly the one before the wide geometry -- not a tighter error bound
    # (both geometries drift about the same, ~10 ulp of the running magnitude),
    # but bit-for-bit agreement with the previous revision, which is what the
    # in-tree GPU test's float32 `[1, 1_048_583]` case (atol 1e-3 against the
    # CPU's float64 reference) is calibrated for. Widening floating rows again
    # will hit that tolerance unless the comparison changes first. Integral
    # scans are exact in any grouping, and they are the shapes serving runs.
    comptime exact = dtype.is_integral()
    comptime long_ok = exact and bit_width_of[dtype]() <= _CUMSUM_LONG_MAX_WIDTH
    var wide_rows = exact and axis_len >= _CUMSUM_WIDE_MIN_LEN
    var chunk_len = _chunk_len(
        row_scan, num_lines, axis_len, sm_count, wide_rows, long_ok
    )
    # Stands in for `carry` wherever `has_carry` is False; never read.
    var no_carry = input.bitcast[accum_type]().reshape(row_major((1, 1)))
    if chunk_len == axis_len:
        _enqueue_scan[dtype, accum_type, exclusive, reverse, False](
            output,
            input,
            no_carry,
            num_rows,
            axis_len,
            stride,
            axis_len,
            row_scan,
            wide_rows,
            long_ok,
            ctx,
        )
        return

    # Split each line into chunks: sum every chunk, scan the sums into
    # per-chunk carries, then scan each chunk starting from its carry.
    var num_chunks = ceildiv(axis_len, chunk_len)
    var totals_buf = ctx.enqueue_create_buffer[accum_type](
        num_lines * num_chunks
    )
    var carries_buf = ctx.enqueue_create_buffer[accum_type](
        num_lines * num_chunks
    )
    var chunks_layout = row_major((num_lines, num_chunks))
    var totals = TileTensor(totals_buf, chunks_layout)
    var carries = TileTensor(carries_buf, chunks_layout)
    _enqueue_chunk_totals[dtype, accum_type, reverse](
        totals, input, num_rows, axis_len, stride, chunk_len, row_scan, ctx
    )
    _enqueue_scan[accum_type, accum_type, True, False, False](
        carries,
        totals.as_imm(),
        no_carry,
        num_lines,
        num_chunks,
        1,
        num_chunks,
        _use_row_scan(num_lines, num_chunks, 1, sm_count),
        wide_rows,
        long_ok,
        ctx,
    )
    _enqueue_scan[dtype, accum_type, exclusive, reverse, True](
        output,
        input,
        carries.as_imm(),
        num_rows,
        axis_len,
        stride,
        chunk_len,
        row_scan,
        wide_rows,
        long_ok,
        ctx,
    )


def cumsum[
    dtype: DType,
    exclusive: Bool,
    reverse: Bool,
    *,
    axis: Int,
    target: StaticString,
](
    output: TileTensor[mut=True, dtype, ...],
    input: TileTensor[dtype, ...],
    ctx: DeviceContext,
) raises:
    """Computes the cumulative sum of `input` along `axis` on `target`.

    See the CPU overload for the semantics. On GPU, float32 inputs accumulate
    in float32 rather than float64.

    Parameters:
        dtype: Type of the input and output tensors.
        exclusive: If set to True, return exclusive sum (top element not included).
        reverse: If set to True, perform cumsum operation in reverse direction.
        axis: The axis on which to perform the cumsum operation.
        target: The target device, `"cpu"` or a GPU target.

    Args:
        output: The output tensor.
        input: The input tensor.
        ctx: The device context used to enqueue GPU kernels.

    Raises:
        Error: If the GPU kernel launch fails.
    """
    comptime assert (
        input.rank == output.rank
    ), "input and output should have the same rank."
    comptime assert (
        -input.rank <= axis < input.rank
    ), "Axis value must be in range [-rank, rank)"
    with Trace[TraceLevel.OP, target=target](
        "cumsum", task_id=get_safe_task_id(ctx)
    ):
        comptime if is_cpu[target]():
            cumsum[dtype, exclusive, reverse, axis=axis](output, input)
        else:
            _cumsum_gpu[dtype, exclusive, reverse, axis=axis](
                output, input.as_imm(), ctx
            )
