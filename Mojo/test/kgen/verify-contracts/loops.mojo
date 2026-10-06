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
# Loops for `verify-contracts`: invariants found at the loop head; see
# README.md.


from std.math import align_down

def change(mut xs: List[Int]):
    xs.clear()


# --- must be PROVEN ---
def ok_sum(xs: List[Int]) -> Int:
    var s = 0
    for i in range(len(xs)):
        s += xs[i]
    return s


def ok_sum_from(xs: List[Int], start: Int) -> Int:
    var s = 0
    if start >= 0:
        for i in range(start, len(xs)):
            s += xs[i]
    return s


def ok_while(xs: List[Int]) -> Int:
    var s = 0
    var i = 0
    while i < len(xs):
        s += xs[i]
        i += 1
    return s


def ok_pairs(xs: List[Int]) -> Int:
    var s = 0
    for i in range(len(xs)):
        for j in range(i):
            s += xs[i] * xs[j]
    return s


def ok_prefix(xs: List[Int], ys: List[Int]) -> Int:
    var s = 0
    if len(ys) >= len(xs):
        for i in range(len(xs)):
            s += xs[i] + ys[i]
    return s


def ok_early_break(xs: List[Int], stop: Int) -> Int:
    var i = 0
    while i < len(xs):
        if xs[i] == stop:
            break
        i += 1
    return i


def ok_reversed(xs: List[Int]) -> Int:
    var t = 0
    for i in reversed(range(len(xs))):
        t += xs[i]  # from `len(xs) - 1` down to 0
    return t


def ok_reversed_from(xs: List[Int], start: Int) -> Int:
    var t = 0
    if 0 <= start:
        for i in reversed(range(start, len(xs))):
            t += xs[i]  # down to `start`, inclusive
    return t


def ok_reversed_pairs(xs: List[Int]) -> Int:
    var t = 0
    for i in reversed(range(1, len(xs))):
        t += xs[i] - xs[i - 1]  # `i` never reaches 0
    return t


def ok_enumerate(xs: List[Int]) -> Int:
    var t = 0
    for i, x in enumerate(xs):
        t += x + xs[i]  # the count stays below the length
    return t


def ok_iterate_with_counter(xs: List[Int]) -> Int:
    var t = 0
    var i = 0
    for x in xs:
        t += x + xs[i]  # one element per step
        i += 1
    return t


def ok_enumerate_span(s: Span[Int, _]) -> Int:
    var t = 0
    for i, x in enumerate(s):
        t += x + s[i]
    return t


def ok_pop_in_loop() -> Int:
    var xs = List[Int]()
    for i in range(6):
        xs.append(i)
    var t = 0
    for _ in range(3, 6):
        t += xs.pop(3)  # `len(xs) + i` stays 9: at least 4 left
    return t


def ok_filled_then_read(n: Int) -> Int:
    var xs = List[Int]()
    for i in range(n):
        xs.append(i)  # `len(xs) - i` stays 0
    var t = 0
    for i in range(n):
        t += xs[i]
    return t


def small(x: Int where 0 <= x and x < 64) -> Int:
    return x


def ok_literal_elements() -> Int:
    var t = 0
    for x in [1, 10, 63]:
        t += small(x)  # each element of the literal
    return t


def ok_list_elements() -> Int:
    var xs: List[Int] = [4, 5, 6]
    var t = 0
    for x in xs:
        t += small(x)  # a reference to each element
    return t


def ok_strided(xs: List[Int], n: Int) -> Int:
    var sum = 0
    if n <= len(xs) and n < 1000:
        for i in range(0, n, 4):
            sum += xs[i]  # `i` is below `n`
    return sum


def ok_strided_from(xs: List[Int], t: Int) -> Int:
    var sum = 0
    if 0 <= t and t < 8 and len(xs) == 100:
        for i in range(t, 100, 8):
            sum += xs[i]
    return sum


def ok_strided_down(xs: List[Int]) -> Int:
    var sum = 0
    if len(xs) == 100:
        for i in range(99, 0, -3):
            sum += xs[i]
    return sum


def ok_strided_counter(xs: List[Int], n: Int) -> Int:
    var sum = 0
    var block = 0
    if 0 <= n and n <= len(xs) and n < 1000:
        # The loop's variable is unused: `block` counts its steps of 4.
        for _ in range(0, n, 4):
            sum += xs[block * 4]
            block += 1
    return sum


def ok_comptime_count(xs: List[Int]) -> Int:
    var sum = 0
    var at = 0
    if len(xs) == 3:
        comptime for _u in range(3):
            sum += xs[at]  # 0, 1, 2: a short `comptime for` is unrolled
            at += 1
    return sum


def ok_unrolled(xs: List[Int]) -> Int:
    var sum = 0
    var at = 0

    @__parameter
    def step():
        sum += xs[at]
        at += 1

    var n = len(xs)
    if n > 1000000:
        return 0
    # Two steps per iteration over the even part, then the rest: `at` is
    # the strided loop's cursor, and ends where the second loop starts.
    var paired = align_down(n, 2)
    for _outer in range(0, paired, 2):
        comptime for _u in range(2):
            step()
    for _rest in range(paired, n):
        step()
    return sum


# --- must stay UNPROVEN ---
def bad_one_past(xs: List[Int]) -> Int:
    var s = 0
    for i in range(len(xs) + 1):
        s += xs[i]  # `i == len(xs)` in the last iteration
    return s


def bad_other_list(xs: List[Int], ys: List[Int]) -> Int:
    var s = 0
    for i in range(len(xs)):
        s += ys[i]  # `ys` may be shorter
    return s


def bad_while_le(xs: List[Int]) -> Int:
    var s = 0
    var i = 0
    while i <= len(xs):
        s += xs[i]
        i += 1
    return s


def bad_changed_in_loop(mut xs: List[Int]) -> Int:
    var s = 0
    for i in range(len(xs)):
        s += xs[i]  # the first iteration is fine, a later one may not be
        change(xs)
    return s


def bad_after_loop(xs: List[Int]) -> Int:
    var i = 0
    while i < len(xs):
        i += 1
    return xs[i]  # `i == len(xs)` after the loop


def bad_reversed_one_past(xs: List[Int]) -> Int:
    var t = 0
    for i in reversed(range(len(xs))):
        t += xs[i + 1]  # `i + 1` is `len(xs)` on the first step
    return t


def bad_reversed_below(xs: List[Int]) -> Int:
    var t = 0
    for i in reversed(range(len(xs))):
        t += xs[i - 1]  # `i` reaches 0
    return t


def bad_enumerate_next(xs: List[Int]) -> Int:
    var t = 0
    for i, x in enumerate(xs):
        t += x + xs[i + 1]  # past the end on the last step
    return t


def bad_enumerate_other(xs: List[Int], ys: List[Int]) -> Int:
    var t = 0
    for i, x in enumerate(xs):
        t += x + ys[i]  # `ys` may be shorter
    return t


def bad_pop_in_loop() -> Int:
    var xs = List[Int]()
    for i in range(6):
        xs.append(i)
    var t = 0
    for _ in range(2, 6):
        t += xs.pop(3)  # the fourth pop leaves only 3
    return t


def bad_list_elements() -> Int:
    var xs: List[Int] = [4, 64]
    var t = 0
    for x in xs:
        t += small(x)  # 64 is out of range
    return t


def bad_unknown_elements(xs: List[Int]) -> Int:
    var t = 0
    for x in xs:
        t += small(x)  # nothing is known about the elements
    return t


def bad_strided_past(xs: List[Int], n: Int) -> Int:
    var sum = 0
    if n <= len(xs) and n < 1000:
        for i in range(0, n, 4):
            sum += xs[i + 1]  # `i + 1` may be `n`
    return sum


def bad_strided_counter(xs: List[Int], n: Int) -> Int:
    var sum = 0
    var block = 0
    if 0 <= n and n <= len(xs) and n < 1000:
        for _ in range(0, n, 4):
            sum += xs[block * 4]
            block += 2  # two blocks per step of 4
    return sum


def bad_strided_down(xs: List[Int]) -> Int:
    var sum = 0
    if len(xs) == 100:
        for i in range(100, 0, -3):
            sum += xs[i]  # starts at 100
    return sum


def bad_comptime_count(xs: List[Int]) -> Int:
    var sum = 0
    var at = 0
    if len(xs) == 3:
        comptime for _u in range(4):
            sum += xs[at]  # the fourth is past the end
            at += 1
    return sum


def bad_unrolled(xs: List[Int]) -> Int:
    var sum = 0
    var at = 0

    @__parameter
    def step():
        sum += xs[at]
        at += 1

    var n = len(xs)
    if n > 1000000:
        return 0
    var paired = align_down(n, 2)
    for _outer in range(0, paired, 2):
        comptime for _u in range(3):  # one more than the stride
            step()
    return sum


def bad_closure_counter(xs: List[Int]) -> Int:
    var sum = 0
    var at = 0

    @__parameter
    def step():
        sum += xs[at]
        at += 1  # written in the closure: the loop below changes it

    if len(xs) >= 1:
        for _ in range(5):
            step()
    return sum
