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
# `Optional` for `verify-contracts`: `value()` and `take()` require a value
# to be held, which a check (`if o`, `o is not None`, `assert_true(o)`), a
# constructor, or a caller's postcondition establishes; see README.md.


from std.collections import LinkedList
from std.testing import assert_false, assert_true


def twice(var x: Int) -> Int:
    return x * 2


# --- must be PROVEN ---
def ok_checked_if(o: Optional[Int]) -> Int:
    if o:
        return o.value()
    return 0


def ok_checked_is_not_none(o: Optional[Int]) -> Int:
    if o is not None:
        return o.value()
    return 0


def ok_else_of_is_none(o: Optional[Int]) -> Int:
    if o is None:
        return 0
    return o.value()


def ok_not_inverted(o: Optional[Int]) -> Int:
    if not o:
        return 0
    return o.value()


def ok_constructed() -> Int:
    var o = Optional[Int](3)
    return o.value()


def ok_copied(o: Optional[Int]) -> Int:
    if o:
        var p = o.copy()
        return p.value()  # a copy holds what the original holds
    return 0


def ok_take_after_check(var o: Optional[Int]) -> Int:
    if o:
        return o.take()
    return 0


def ok_assigned_in_branch(var o: Optional[Int]) -> Int:
    if not o:
        o = Optional[Int](1)
    return o.value()  # held on both paths


def ok_while(var o: Optional[Int]) -> Int:
    var t = 0
    while o:
        t += o.value()
        o = None
    return t


def ok_early_return(o: Optional[Int]) raises -> Int:
    if not o:
        raise Error("empty")
    return o.value()


def ok_asserted(o: Optional[Int]) raises -> Int:
    assert_true(o)  # raises unless a value is held
    return o.value()


def ok_asserted_false(o: Optional[Int]) raises -> Int:
    assert_false(o)
    return 0 if not o else o.value()  # the branch is dead


def ok_mapped(var o: Optional[Int]) -> Int:
    if o:
        var m = o^.map(twice)
        return m.value()  # held exactly when `o` was
    return 0


def ok_mapped_closure(var o: Optional[Int]) -> Int:
    var calls = 0

    def count(var x: Int) {mut calls} -> Int:
        calls += 1
        return x

    if o:
        var m = o^.map(count)
        return m.value()
    return 0


def ok_init_with() -> Int:
    var o = Optional[Int](init_with=lambda () -> Int: 7)
    return o.value()  # built in place: always held


def ok_init_with_capture() -> Int:
    var calls = 0

    def make() {mut calls} -> Int:
        calls += 1
        return 7

    var o = Optional[Int](init_with=make)
    return o.value()


def ok_copied_pointer() -> Int:
    var x = 5
    var o: Optional[Pointer[Int, origin_of(x)]] = Optional(Pointer(to=x))
    var c = o.copied()
    return c.value()


def ok_maybe_pop(mut l: LinkedList[Int]) -> Int:
    if len(l) > 0:
        return l.maybe_pop().value()  # `maybe_pop` states when it holds one
    return 0


def ok_maybe_pop_index(mut l: LinkedList[Int]) -> Int:
    if len(l) > 2:
        return l.maybe_pop(1).value()  # always held, for an index in range
    return 0


def ok_from_u32() -> Int:
    return Int(Codepoint.from_u32(0x0399).value())  # a scalar value


# --- must stay UNPROVEN ---
def bad_unchecked(o: Optional[Int]) -> Int:
    return o.value()


def bad_none() -> Int:
    var o = Optional[Int]()
    return o.value()


def bad_wrong_branch(o: Optional[Int]) -> Int:
    if not o:
        return o.value()
    return 0


def bad_after_take(var o: Optional[Int]) -> Int:
    if o:
        var x = o.take()
        return x + o.value()  # empty since the `take`
    return 0


def bad_reassigned(var o: Optional[Int]) -> Int:
    if o:
        o = None
        return o.value()
    return 0


def bad_take_unchecked(var o: Optional[Int]) -> Int:
    return o.take()


def bad_mapped(var o: Optional[Int]) -> Int:
    var m = o^.map(twice)
    return m.value()  # empty when `o` was


def bad_asserted_false(o: Optional[Int]) raises -> Int:
    assert_false(o)
    return o.value()


def bad_capture_written(var o: Optional[Int]) -> Int:
    var xs: List[Int] = [1, 2, 3]

    def shrink() {mut xs} -> Int:
        xs.clear()
        return 1

    var p = Optional[Int](init_with=shrink)
    return xs[0] + p.value()  # `xs` was emptied by the closure


def bad_maybe_pop_empty(mut l: LinkedList[Int]) -> Int:
    return l.maybe_pop().value()  # empty for an empty list


def bad_maybe_pop_index(mut l: LinkedList[Int]) -> Int:
    return l.maybe_pop(1).value()  # the index may be out of range


def bad_from_u32(c: UInt32) -> Int:
    return Int(Codepoint.from_u32(c).value())  # may be a surrogate
