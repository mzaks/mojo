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

"""Tests for `where` clauses on runtime arguments.

These are contracts for static verification (see
Mojo/proposals/argument-contracts.md): `where` clauses after an argument's
name and type, with the same forms as trailing clauses, including several per
argument and a message.
"""

from tests.util import assert_mojo_format


def test_argument_where_clause():
    source = "def get(xs: List[Int], i: Int where 0 <= i and i < len(xs)) -> Int: pass"
    expected = (
        "def get(xs: List[Int], i: Int where 0 <= i and i < len(xs)) -> Int:\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_argument_where_clauses_split_at_arguments():
    source = (
        "def make(count: Int where count >= 0, out result: List[Int] where"
        " len(result) == count): pass"
    )
    expected = (
        "def make(\n"
        "    count: Int where count >= 0,\n"
        "    out result: List[Int] where len(result) == count,\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_argument_where_clause_with_messages():
    source = (
        'def f(a: Int where a >= 0 else "a must not be negative",'
        ' b: Int where (b > a, "b must exceed a")): pass'
    )
    expected = (
        "def f(\n"
        '    a: Int where a >= 0 else "a must not be negative",\n'
        '    b: Int where (b > a, "b must exceed a"),\n'
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_mut_self_where_clauses_with_old():
    source = (
        "struct S:\n"
        "    def pop(mut self where old(len(self)) > 0"
        " where len(self) == old(len(self)) - 1) -> Int: pass"
    )
    expected = (
        "struct S:\n"
        "    def pop(\n"
        "        mut self where old(len(self)) > 0"
        " where len(self) == old(len(self)) - 1,\n"
        "    ) -> Int:\n"
        "        pass\n"
    )
    assert_mojo_format(source, expected)


def test_argument_where_clause_with_default():
    source = "def f(a: Int where a > 0 = 1): pass"
    expected = "def f(a: Int where a > 0 = 1):\n    pass\n"
    assert_mojo_format(source, expected)
