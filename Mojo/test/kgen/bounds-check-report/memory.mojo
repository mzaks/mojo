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
# Lists in memory that calls may or may not write, for `bounds-check-report`;
# see README.md.


@inline(.never)
def clear_list(mut xs: List[Int]):
    xs.clear()


@inline(.never)
def clear_through[o: MutOrigin](p: Pointer[List[Int], o]):
    p[].clear()


struct Bag(Movable):
    var items: List[Int]

    def __init__(out self):
        self.items = [1, 2, 3]

    @inline(.never)
    def ok_get(self, i: Int) -> Int:
        if 0 <= i < len(self.items):
            _ = self.items[::-1]  # A call reading `self` in between.
            return self.items[i]
        return 0

    @inline(.never)
    def bad_get_after_clear(mut self, i: Int) -> Int:
        if 0 <= i < len(self.items):
            self.items.clear()
            return self.items[i]
        return 0


# --- must stay UNPROVEN ---
@inline(.never)
def bad_after_mut_call() -> Int:
    var vs: List[Int] = [1, 2, 3]
    clear_list(vs)
    return vs[2]


@inline(.never)
def bad_read_then_mut_call() -> Int:
    var vs: List[Int] = [1, 2, 3]
    _ = vs[1:-1:1]
    clear_list(vs)
    return vs[2]


@inline(.never)
def bad_escaped_pointer() -> Int:
    var vs: List[Int] = [1, 2, 3]
    var p = Pointer(to=vs)
    _ = vs[1:-1:1]
    clear_through(p)
    return vs[2]


# Lists of lists: the inner list's length is read from the outer list's heap
# buffer.
@inline(.never)
def bad_nested_short_child() -> Int:
    var list = List[List[Int]](capacity=1)
    var child = List[Int](capacity=2)
    child.append(10)
    list.append(child^)
    return list[0][1]


@inline(.never)
def bad_nested_wrong_child() -> Int:
    var list = List[List[Int]](capacity=2)
    var a = List[Int](capacity=3)
    a.append(1)
    a.append(2)
    a.append(3)
    var b = List[Int](capacity=1)
    b.append(1)
    list.append(a^)
    list.append(b^)
    return list[1][2]


@inline(.never)
def bad_nested_cleared() -> Int:
    var list = List[List[Int]](capacity=1)
    var child = List[Int](capacity=2)
    child.append(10)
    child.append(20)
    list.append(child^)
    list[0].clear()
    return list[0][1]


@inline(.never)
def bad_nested_overwritten_through_pointer() -> Int:
    var list = List[List[Int]](capacity=1)
    var child = List[Int](capacity=2)
    child.append(10)
    child.append(20)
    list.append(child^)
    var p = list.unsafe_ptr()
    p[0] = List[Int]()
    return list[0][1]


# Known limit: in bounds, but the second append reallocates and moves the
# first child with a copy loop, which the heap search does not follow.
@inline(.never)
def limit_nested_after_realloc() -> Int:
    var list = List[List[Int]](capacity=1)
    var a = List[Int](capacity=2)
    a.append(1)
    a.append(2)
    var b = List[Int](capacity=1)
    b.append(1)
    list.append(a^)
    list.append(b^)
    return list[0][1]


# --- must be PROVEN ---
@inline(.never)
def ok_strided_twice() -> Int:
    var vs: List[Int] = [1, 2, 3]
    var es = vs[1:-1:1]
    es = vs[::-1]
    return es[2]


@inline(.never)
def ok_read_calls_keep_length() -> Int:
    var vs: List[Int] = [1, 2, 3]
    _ = vs[1:-1:1]
    _ = vs[::-1]
    return vs[2]


@inline(.never)
def ok_nested_straight() -> Int:
    var list = List[List[Int]](capacity=1)
    var child = List[Int](capacity=2)
    child.append(10)
    child.append(20)
    list.append(child^)
    return list[0][1]


@inline(.never)
def ok_nested_two_children() -> Int:
    var list = List[List[Int]](capacity=2)
    var a = List[Int](capacity=1)
    a.append(1)
    var b = List[Int](capacity=3)
    b.append(1)
    b.append(2)
    b.append(3)
    list.append(a^)
    list.append(b^)
    return list[0][0] + list[1][2]


@inline(.never)
def ok_nested_other_list_written() -> Int:
    # Writes to another allocation, even of another element type, do not
    # touch the outer list's buffer.
    var list = List[List[Int]](capacity=1)
    var child = List[Int](capacity=2)
    child.append(10)
    child.append(20)
    list.append(child^)
    var other = List[Int](capacity=1)
    other.append(5)
    return list[0][1] + other[0]


def main():
    var bag = Bag()
    print(
        bad_after_mut_call(),
        bad_read_then_mut_call(),
        bad_escaped_pointer(),
        ok_strided_twice(),
        ok_read_calls_keep_length(),
        bag.ok_get(1),
        bad_nested_short_child(),
        bad_nested_wrong_child(),
        bad_nested_cleared(),
        bad_nested_overwritten_through_pointer(),
        limit_nested_after_realloc(),
        ok_nested_straight(),
        ok_nested_two_children(),
        ok_nested_other_list_written(),
        bag.bad_get_after_clear(1),
    )
