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

"""Implements the [Fnv1a 64 bit variant](https://en.wikipedia.org/wiki/Fowler–Noll–Vo_hash_function) algorithm as a _Hasher type."""

from ._hasher import _Hasher, _HashableWithHasher
from memory import UnsafePointer


struct Fnv1a(_Hasher):
    """Fnv1a is a very simple algorithm with good quality, but sub optimal runtime for long inputs.
    It can be used for comp time hash value generation.

    References:

    - [Fnv1a 64 bit variant](https://en.wikipedia.org/wiki/Fowler–Noll–Vo_hash_function)
    """

    var _value: UInt64

    fn __init__(out self):
        """Initialize the hasher."""
        self._value = 0xCBF29CE484222325

    fn _update_with_bytes(
        mut self, data: UnsafePointer[UInt8, mut=False, **_], length: Int
    ):
        """Consume provided data to update the internal buffer.

        Args:
            data: Pointer to the byte array.
            length: The length of the byte array.
        """
        for i in range(length):
            self._value ^= data[i].cast[DType.uint64]()
            self._value *= 0x100000001B3

    fn _update_with_simd(mut self, value: SIMD[_, _]):
        """Update the buffer value with new data.

        Args:
            value: Value used for update.
        """

        @parameter
        if value.dtype.is_floating_point():
            v64 = value.to_bits().cast[DType.uint64]()
        else:
            v64 = value.cast[DType.uint64]()

        @parameter
        for i in range(0, v64.size):
            self._value ^= v64[i].cast[DType.uint64]()
            self._value *= 0x100000001B3

    fn update[T: _HashableWithHasher](mut self, value: T):
        """Update the buffer value with new hashable value.

        Parameters:
            T: Hashable type.

        Args:
            value: Value used for update.
        """
        value.__hash__(self)

    fn finish(owned self) -> UInt64:
        """Computes the hash value based on all the previously provided data.

        Returns:
            Final hash value.
        """
        return self._value
