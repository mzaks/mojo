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
"""Internal contract annotations for static verification.

These record conditions for analyses such as the `bounds-check-report` compiler
pass. They have no runtime effect: the compiler erases them when lowering to
LLVM, and the comptime interpreter skips them.
"""

from std.collections.string.string_span import _get_kgen_string


@inline(.nodebug)
def _ensures(cond: Bool):
    """Declares a postcondition, to be placed right before a function returns.

    Inside the function, static verification proves `cond` like any other
    obligation. At call sites of a function that is not inlined, it may assume
    `cond`, with the callee's arguments and returned values bound to the call.
    Capture "old" values in local variables at the start of the function.

    Args:
        cond: The condition that holds when the function returns.
    """
    var line, col, file_name = __mlir_op.`kgen.source_loc`[
        inlineCount=Int(0).__mlir_index__(),
        _type=Tuple[
            __mlir_type.index,
            __mlir_type.index,
            __mlir_type.`!kgen.string`,
        ],
    ]()
    __mlir_op.`kgen.obligation`[kind=_get_kgen_string["ensures"](), _type=None](
        cond._mlir_value, line, col, file_name
    )


@inline(.nodebug)
def _assume(cond: Bool):
    """Declares a trusted fact, such as a type invariant.

    Static verification takes `cond` as true where this is reached, without
    proving it. Use sparingly: a wrong assumption makes proofs unsound.

    Args:
        cond: The condition that is assumed to hold here.
    """
    var line, col, file_name = __mlir_op.`kgen.source_loc`[
        inlineCount=Int(0).__mlir_index__(),
        _type=Tuple[
            __mlir_type.index,
            __mlir_type.index,
            __mlir_type.`!kgen.string`,
        ],
    ]()
    __mlir_op.`kgen.assume`[kind=_get_kgen_string["invariant"](), _type=None](
        cond._mlir_value, line, col, file_name
    )
