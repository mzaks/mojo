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


# PROVEN: loop bound is `len(xs)`.
def sum_all(xs: List[Int]) -> Int:
    var s = 0
    for i in range(len(xs)):
        s += xs[i]
    return s


# PROVEN: guarded by the condition.
def get_or_zero(xs: List[Int], i: Int) -> Int:
    if i >= 0 and i < len(xs):
        return xs[i]
    return 0


# UNPROVEN: nothing is known about `i`.
def get(xs: List[Int], i: Int) -> Int:
    return xs[i]


def main():
    var xs: List[Int] = [1, 2, 3]
    print(sum_all(xs), get_or_zero(xs, 1), get(xs, 2))
