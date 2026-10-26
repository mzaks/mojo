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
# Local closures handed to library functions (`map`, `vectorize`,
# `deinit_with`) for `verify-contracts`: the library body is walked at the
# call, and the closure's where that body calls it; see README.md.

from std.algorithm import map, vectorize
from std.collections.dict import Dict, DictEntry
from std.hash import default_comp_time_hasher


struct Linear(not Deinitable else "use destroy()", Movable):
    var value: Int

    def __init__(out self, value: Int):
        self.value = value

    def destroy(deinit self):
        pass


# --- must be PROVEN ---
def ok_map_reads(xs: List[Int]) -> Int:
    var t = 0

    def f(i: Int) {mut t, imm}:
        t += xs[i]  # `i < size`, and `size` is `len(xs)`

    map(len(xs), f)
    return t


def ok_vectorize_reads(xs: List[Int]) -> Int:
    var t = 0

    def g[width: Int](i: Int) {mut t, imm}:
        t += xs[i]  # in the unrolled, the vector and the remainder loop

    vectorize[4](len(xs), g)
    return t


def ok_map_in_loop(xs: List[Int]) -> Int:
    var t = 0

    def f(i: Int) {mut t, imm}:
        t += xs[i]

    for _ in range(3):
        map(len(xs), f)
    return t


def ok_deinit_once() -> Int:
    var disposed = List[Int]()

    def d(var v: Int) {mut}:
        disposed.append(v)

    var o = Optional[Int](3)
    o^.deinit_with(d)  # calls `d` once: one element
    return disposed[0]


def ok_deinit_each(xs: List[Int]) -> Int:
    var disposed = List[Int]()
    disposed.append(1)

    def d(var v: Int) {mut}:
        disposed.append(v)

    var ys = xs.copy()
    ys^.deinit_with(d)  # a loop over the elements: the length only grows
    return disposed[0]


def ok_entry_direct(
    var entry: DictEntry[String, Linear, default_comp_time_hasher]
) -> Int:
    var disposed = List[Int]()

    def dispose(var key: String, var value: Linear) {mut}:
        disposed.append(value.value)
        value^.destroy()

    entry^.deinit_with(dispose)  # the key and value, once
    return disposed[0]


def ok_closure_handed_by_closure(
    var entry: DictEntry[String, Linear, default_comp_time_hasher]
) -> Int:
    var disposed = List[Int]()

    def dispose(var key: String, var value: Linear) {mut}:
        disposed.append(value.value)
        value^.destroy()

    def dispose_entry(
        var entry: DictEntry[String, Linear, default_comp_time_hasher]
    ) {mut}:
        entry^.deinit_with(dispose)  # `dispose` is a capture

    var o = Optional(entry^)
    if o:
        o^.deinit_with(dispose_entry)
        return disposed[0]
    o^.deinit_with(dispose_entry)
    return 0


def ok_dict_teardown(
    var d: Dict[String, Linear, default_comp_time_hasher]
) -> Int:
    var disposed = List[Int]()
    disposed.append(1)

    def dispose(var key: String, var value: Linear) {mut}:
        disposed.append(value.value)
        value^.destroy()

    d^.deinit_with(dispose)  # once per occupied slot
    return disposed[0]


# --- must stay UNPROVEN ---
def bad_map_past_end(xs: List[Int]) -> Int:
    var t = 0

    def f(i: Int) {mut t, imm}:
        t += xs[i + 1]  # `i + 1` is `len(xs)` in the last call

    map(len(xs), f)
    return t


def bad_map_clears(xs: List[Int]) -> Int:
    var t = List[Int]()
    t.append(1)

    def f(i: Int) {mut t, imm}:
        t.clear()

    map(len(xs), f)
    return t[0]  # cleared by the closure, if `xs` has elements


def bad_deinit_clears(xs: List[Int]) -> Int:
    var disposed = List[Int]()
    disposed.append(1)

    def d(var v: Int) {mut}:
        disposed.clear()

    var ys = xs.copy()
    ys^.deinit_with(d)
    return disposed[0]  # cleared by the closure


def bad_displaced_may_be_none(
    var d: Dict[String, Linear, default_comp_time_hasher]
) -> Int:
    var disposed = List[Int]()

    def dispose(var key: String, var value: Linear) {mut}:
        disposed.append(value.value)
        value^.destroy()

    def dispose_entry(
        var entry: DictEntry[String, Linear, default_comp_time_hasher]
    ) {mut}:
        entry^.deinit_with(dispose)

    var displaced = d.insert("k", Linear(1))
    displaced^.deinit_with(dispose_entry)  # nothing, when `k` was new
    d^.deinit_with(dispose)
    return disposed[0]  # `d` may have been empty
