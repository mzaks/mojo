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
# Basic `bounds-check-report` examples; see README.md.
# Loop invariants relating two counters (`a + b == c`, `a - b == c`) for
# `bounds-check-report`; see README.md.


# --- must stay UNPROVEN ---
@inline(.never)
def bad_pop_loop_too_many() -> Int:
    var xs: List[Int] = [0, 1, 2, 3, 4, 5]
    for _ in range(3, 7):
        _ = xs.pop(3)  # the fourth pop has only 3 elements left
    return len(xs)


@inline(.never)
def bad_read_after_pop_loop() -> Int:
    var xs: List[Int] = [0, 1, 2, 3, 4, 5]
    for _ in range(3, 6):
        _ = xs.pop(3)
    return xs[3]  # 3 elements left


@inline(.never)
def bad_append_to_nonempty(n: Int) -> Int:
    if n < 0 or n > 1000:
        return 0
    var xs: List[Int] = [1, 2]
    for i in range(n):
        xs.append(i)
    return xs[n + 2]  # 2 + n elements


# --- should be PROVEN ---
@inline(.never)
def ok_pop_loop() -> Int:
    var xs: List[Int] = [0, 1, 2, 3, 4, 5]
    for _ in range(3, 6):
        _ = xs.pop(3)  # the length counts down as the index counts up
    return len(xs)


@inline(.never)
def ok_read_after_pop_loop() -> Int:
    var xs: List[Int] = [0, 1, 2, 3, 4, 5]
    for _ in range(3, 6):
        _ = xs.pop(3)
    return xs[2]


@inline(.never)
def ok_append_to_nonempty(n: Int) -> Int:
    if n < 0 or n > 1000:
        return 0
    var xs: List[Int] = [1, 2]
    for i in range(n):
        xs.append(i)
    return xs[n + 1]


def main():
    print(
        bad_pop_loop_too_many(),
        bad_read_after_pop_loop(),
        bad_append_to_nonempty(3),
        ok_pop_loop(),
        ok_read_after_pop_loop(),
        ok_append_to_nonempty(3),
    )
