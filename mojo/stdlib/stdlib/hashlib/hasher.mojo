# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
from ._ahash import AHasher
from ._fnv1a import Fnv1a
from sys import is_compile_time

alias default_hasher = DefaultHasher
alias default_comp_time_hasher = Fnv1a


struct DefaultHasher(Hasher):
    """
    DefaultHasher is a workaround type that delegates to AHasher at runtime and Fnv1a at compile time.
    This is necessary because AHasher is not yet usable at compile time. Once AHasher supports compile-time
    usage, this struct can be removed in favor of using AHasher directly.
    """

    var _ahasher: AHasher[SIMD[DType.uint64, 4](0)]
    var _fnv1a: Fnv1a

    fn __init__(out self):
        self._ahasher = AHasher[SIMD[DType.uint64, 4](0)]()
        self._fnv1a = Fnv1a()

    fn _update_with_bytes(
        mut self,
        data: UnsafePointer[
            UInt8, address_space = AddressSpace.GENERIC, mut=False, **_
        ],
        length: Int,
    ):
        @parameter
        if is_compile_time():
            self._fnv1a._update_with_bytes(data, length)
        else:
            self._ahasher._update_with_bytes(data, length)

    fn _update_with_simd(mut self, value: SIMD[_, _]):
        @parameter
        if is_compile_time():
            self._fnv1a._update_with_simd(value)
        else:
            self._ahasher._update_with_simd(value)

    fn update[T: Hashable](mut self, value: T):
        @parameter
        if is_compile_time():
            self._fnv1a.update(value)
        else:
            self._ahasher.update(value)

    fn finish(var self) -> UInt64:
        @parameter
        if is_compile_time():
            var hasher = self._fnv1a
            return hasher^.finish()
        else:
            var hasher = self._ahasher
            return hasher^.finish()


trait Hasher:
    fn __init__(out self):
        ...

    fn _update_with_bytes(
        mut self,
        data: UnsafePointer[
            UInt8, address_space = AddressSpace.GENERIC, mut=False, **_
        ],
        length: Int,
    ):
        ...

    fn _update_with_simd(mut self, value: SIMD[_, _]):
        ...

    fn update[T: Hashable](mut self, value: T):
        ...

    fn finish(var self) -> UInt64:
        ...
