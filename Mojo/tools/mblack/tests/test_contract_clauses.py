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

"""Tests for the contract clauses of a function.

These are contracts for static verification (see
Mojo/proposals/function-contracts.md): `requires` and `ensures` clauses after
a function's trailing `where` clauses, with the same forms, including a
message.
"""

from tests.util import assert_mojo_format


def test_contract_on_one_line():
    source = "def get(xs: List[Int], i: Int) -> Int requires 0 <= i and i < len(xs): pass"
    expected = (
        "def get(xs: List[Int], i: Int) -> Int requires 0 <= i and i < len(xs):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_contract_clauses_get_their_own_lines():
    source = (
        "def make(count: Int) -> List[Int] requires count >= 0"
        " ensures len(result) == count: pass"
    )
    expected = (
        "def make(count: Int) -> List[Int]\n"
        "    requires count >= 0\n"
        "    ensures len(result) == count:\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_contract_clauses_stay_on_their_lines():
    source = (
        "def make(count: Int) -> List[Int]\n"
        "    requires count >= 0\n"
        "    ensures len(result) == count:\n"
        "    pass\n"
    )
    assert_mojo_format(source, source)


def test_where_clauses_line_up_with_contract():
    source = (
        "struct S:\n"
        "    def pop(mut self) -> Self.T where conforms_to(Self.T, Movable)"
        " requires len(self) > 0 ensures len(self) == old(len(self)) - 1: pass"
    )
    expected = (
        "struct S:\n"
        "    def pop(mut self) -> Self.T\n"
        "        where conforms_to(Self.T, Movable)\n"
        "        requires len(self) > 0\n"
        "        ensures len(self) == old(len(self)) - 1:\n"
        "        pass\n"
    )
    assert_mojo_format(source, expected)


def test_contract_clauses_with_messages():
    source = (
        'def f(a: Int, b: Int) requires a >= 0 else "a must not be negative"'
        ' requires (b > a, "b must exceed a"): pass'
    )
    expected = (
        "def f(a: Int, b: Int)\n"
        '    requires a >= 0 else "a must not be negative"\n'
        '    requires (b > a, "b must exceed a"):\n'
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_long_contract_clause_splits_inside_parentheses():
    source = (
        "def kernel(a: Pointer[Float32], n: Int32, k: Int32) requires k >= 0"
        " and a._extent() >= Int(k) * Int(n) and block_dim.x == tile_size"
        " and tile_size % WARP_SIZE == 0: pass"
    )
    expected = (
        "def kernel(a: Pointer[Float32], n: Int32, k: Int32)\n"
        "    requires (\n"
        "        k >= 0\n"
        "        and a._extent() >= Int(k) * Int(n)\n"
        "        and block_dim.x == tile_size\n"
        "        and tile_size % WARP_SIZE == 0\n"
        "    ):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_contract_after_exploded_arguments():
    source = (
        "def copy(dst: UnsafePointer[Float32, MutAnyOrigin], src:"
        " UnsafePointer[Float32, ImmutAnyOrigin], n: Int) requires n >= 0: pass"
    )
    expected = (
        "def copy(\n"
        "    dst: UnsafePointer[Float32, MutAnyOrigin],\n"
        "    src: UnsafePointer[Float32, ImmutAnyOrigin],\n"
        "    n: Int,\n"
        ")\n"
        "    requires n >= 0:\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_trait_method_contract():
    source = (
        "trait Sized:\n"
        "    def __len__(self) -> Int ensures result >= 0: ...\n"
    )
    expected = (
        "trait Sized:\n"
        "    def __len__(self) -> Int ensures result >= 0:\n"
        "        ...\n"
    )
    assert_mojo_format(source, expected)


def test_contract_keywords_are_names_elsewhere():
    source = (
        "def f(requires: Int) -> Int:\n"
        "    var ensures = requires + 1\n"
        "    return ensures\n"
    )
    assert_mojo_format(source, source)
