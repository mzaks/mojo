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
# `BitSet`, `Deque` and `LinkedList` for `verify-contracts`; see README.md.

from std.collections import BitSet, Deque, LinkedList


# --- must be PROVEN ---
def ok_bitset(mut b: BitSet[64], i: Int) -> Bool:
    if 0 <= i and i < 64:
        b.set(i)  # the bound is the size parameter
        b.toggle(i)
        return b.test(i)
    return False


def ok_bitset_generic[n: Int](mut b: BitSet[n], i: Int):
    if 0 <= i and i < n:
        b.clear(i)


def ok_deque_literal() -> Int:
    var d: Deque[Int] = [1, 2, 3]
    return d[2] + d[0]


def ok_deque_append() -> Int:
    var d: Deque[Int] = [1, 2]
    d.append(3)  # unbounded: one longer
    d.appendleft(0)
    return d[3]


def ok_deque_pop() raises -> Int:
    var d: Deque[Int] = [1, 2, 3]
    _ = d.pop()
    return d[1]


def ok_deque_insert() raises -> Int:
    var d: Deque[Int] = [1, 2, 3]
    d.insert(3, 4)
    return d[3]


def ok_deque_bounded(maxlen: Int) -> Int:
    var d = Deque[Int](maxlen=maxlen)
    d.append(1)  # an empty deque does not evict, bounded or not
    return d[0]


def ok_deque_loop(d: Deque[Int]) -> Int:
    var t = 0
    for i in range(len(d)):
        t += d[i]
    return t


def ok_linked_list() -> Int:
    var l: LinkedList[Int] = [1, 2, 3]
    l.append(4)
    l.insert(0, 0)
    return l.get_nth(4)


def ok_linked_list_pop() raises -> Int:
    var l: LinkedList[Int] = [1, 2, 3]
    _ = l.pop(1)
    return l.get_nth(1)


# --- must stay UNPROVEN ---
def bad_bitset(mut b: BitSet[64], i: Int):
    if 0 <= i and i <= 64:
        b.set(i)  # `i` may be 64


def bad_deque_past() -> Int:
    var d: Deque[Int] = [1, 2, 3]
    return d[3]


def bad_deque_bounded() -> Int:
    var d = Deque[Int](maxlen=2)
    d.append(1)
    d.append(2)
    d.append(3)  # full: evicts the first
    return d[2]


def bad_deque_after_pop() raises -> Int:
    var d: Deque[Int] = [1, 2, 3]
    _ = d.popleft()
    return d[2]  # one shorter


def bad_linked_list_insert_past() -> Int:
    var l: LinkedList[Int] = [1, 2, 3]
    l.insert(4, 0)  # past the end
    return 0


def bad_linked_list_get_past() -> Int:
    var l: LinkedList[Int] = [1, 2, 3]
    return l.get_nth(3)
