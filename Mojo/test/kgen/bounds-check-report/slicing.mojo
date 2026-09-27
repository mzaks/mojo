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
# Length contracts for copying and slicing `List`, and bounds of slices, for
# `bounds-check-report`; see README.md.


# --- must stay UNPROVEN ---
@inline(.never)
def bad_copy() -> Int:
    var vs: List[Int] = [1, 2, 3]
    var es = vs.copy()
    return es[3]  # one past the end of the copy


@inline(.never)
def bad_contiguous() -> Int:
    var vs: List[Int] = [1, 2, 3]
    var es = List(vs[1:])
    return es[2]  # the slice has 2 elements


@inline(.never)
def bad_slice_bounds() -> Int:
    var vs: List[Int] = [1, 2, 3]
    return len(vs[1:4])  # end is past the list


# Known limit: the length contract of `__getitem__(StridedSlice)` is proven,
# but its call site cannot evaluate `len(range(slice.indices(len)))` yet.
@inline(.never)
def limit_strided() -> Int:
    var vs: List[Int] = [1, 2, 3]
    var es = vs[::-1]
    return es[2]


# --- should be PROVEN ---
@inline(.never)
def ok_copy() -> Int:
    var vs: List[Int] = [1, 2, 3]
    var es = vs.copy()
    return es[2]


@inline(.never)
def ok_contiguous() -> Int:
    var vs: List[Int] = [1, 2, 3]
    var es = List(vs[1:])
    return es[1]


@inline(.never)
def ok_slice_bounds() -> Int:
    var vs: List[Int] = [1, 2, 3]
    return len(vs[0:3])


def main():
    print(
        bad_copy(),
        bad_contiguous(),
        bad_slice_bounds(),
        limit_strided(),
        ok_copy(),
        ok_contiguous(),
        ok_slice_bounds(),
    )
