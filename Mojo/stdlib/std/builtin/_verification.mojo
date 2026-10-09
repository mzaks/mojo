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
from std.traits import IsTriviallyMovable


@inline(.nodebug)
def _requires(cond: Bool):
    """Declares a precondition, to be placed at the start of a function.

    Callers must establish `cond`: static verification checks it at every call
    site (the call's arguments bound to the parameters), and assumes it inside
    the function itself.

    Args:
        cond: The condition that must hold when the function is called.
    """
    # The location of the call to the annotated function. It only resolves once
    # that function is inlined into a caller, which is how verification tells a
    # function's own precondition (unresolved: assumed) from the precondition
    # of an inlined callee (resolved: checked and reported at the call).
    var line, col, file_name = __mlir_op.`kgen.source_loc`[
        inlineCount=Int(1).__mlir_index__(),
        _type=Tuple[
            __mlir_type.index,
            __mlir_type.index,
            __mlir_type.`!kgen.string`,
        ],
    ]()
    __mlir_op.`kgen.obligation`[
        kind=_get_kgen_string["requires"](), _type=None
    ](cond._mlir_value, line, col, file_name)


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


@inline(.nodebug)
def _same_elements[
    T: AnyType
](dst: Pointer[T, _], src: Pointer[T, _], count: Int) -> Bool:
    """States, in a postcondition, that the `count` elements at `dst` on exit
    are the elements that were at `src` on entry.

    Use it in an `ensures` clause, typically with
    `src` an `old(...)` pointer, to say that a function keeps or moves elements
    it does not otherwise change. Unlike comparing elements with `==` in a
    quantifier, it works for any element type. It has no runtime effect.

    It compares the elements' bits, which a move keeps only when it is
    trivial: for a `T` that is not trivially movable it states nothing
    (it is `True`).

    Parameters:
        T: The element type.

    Args:
        dst: The elements on exit.
        src: The elements on entry.
        count: The number of elements.

    Returns:
        The statement, for the verifier.
    """
    comptime if not IsTriviallyMovable[T]:
        return True
    return Bool(
        mlir_value=__mlir_op.`kgen.contract.same_elements`[
            _type=__mlir_type.`!kgen.scalar<bool>`
        ](dst._mlir_value, src._mlir_value, count._mlir_value)
    )
