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
# Struct invariants as facts of the type, with `type-invariants=true`: a
# struct's invariant holds of every value of it that a function outside the
# struct reads, wherever the value is kept, and is checked where such a
# function writes a field of one; see README.md.

from std.memory.alloc import unsafe_alloc


# An `Int` that is not negative.
struct Nat(ImplicitlyCopyable, Movable) invariant self.value >= 0:
    var value: Int

    def __init__(out self, value: Int) requires value >= 0:
        self.value = value


struct Pair(Copyable, Movable):
    var first: Nat
    var second: Nat

    def __init__(out self, first: Nat, second: Nat):
        self.first = first
        self.second = second


# --- must be PROVEN ---
def ok_element(ns: List[Nat], xs: List[Int]) -> Int:
    if len(ns) > 0:
        var n = ns[0]  # an element of a list
        if n.value < len(xs):
            return xs[n.value]
    return 0


def ok_element_in_place(ns: List[Nat], xs: List[Int]) -> Int:
    if len(ns) > 0 and ns[0].value < len(xs):
        return xs[ns[0].value]
    return 0


def ok_field(p: Pair, xs: List[Int]) -> Int:
    if p.first.value < len(xs):
        return xs[p.first.value]  # a field of another struct
    return 0


def ok_write(mut p: Pair, a: Int):
    if a >= 0:
        p.second.value = a


def ok_write_element(mut ns: List[Nat], a: Int):
    if len(ns) > 0 and a >= 0:
        ns[0].value = a


def ok_behind_pointer(xs: List[Int]) -> Int:
    var p = unsafe_alloc[Nat](1)
    p.unsafe_write(Nat(1))
    var i = p[].value  # an element behind a pointer
    p.unsafe_free()
    if i < len(xs):
        return xs[i]
    return 0


def ok_write_behind_pointer(a: Int):
    var p = unsafe_alloc[Nat](1)
    p.unsafe_write(Nat(1))
    if a >= 0:
        p[].value = a
    p.unsafe_free()


# --- must be WARNED (at the write) ---
def bad_write(mut p: Pair, a: Int):
    p.second.value = a  # nothing is known about `a`


def bad_write_element(mut ns: List[Nat]):
    if len(ns) > 0:
        ns[0].value = -5


def bad_write_behind_pointer(a: Int):
    var p = unsafe_alloc[Nat](1)
    p.unsafe_write(Nat(1))
    p[].value = a
    p.unsafe_free()


def bad_write_local() -> Nat:
    var n = Nat(3)
    n.value = -5  # cannot hold
    return n


def main():
    var xs: List[Int] = [1, 2, 3]
    var ns: List[Nat] = [Nat(1), Nat(2)]
    var p = Pair(Nat(0), Nat(1))
    print(ok_element(ns, xs), ok_element_in_place(ns, xs), ok_field(p, xs))
    ok_write(p, 2)
    bad_write(p, 2)
    bad_write_element(ns)
    print(ok_behind_pointer(xs))
    ok_write_behind_pointer(2)
    bad_write_behind_pointer(2)
    print(bad_write_local().value)
