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
# Integer conversions that may change the value, with
# `check-conversions=all`: one to a narrower type truncates, one to the other
# signedness reads the bits anew; see README.md.


# The bound of a masked copy in a 32-bit index type: the product of two
# sizes does not fit in general.
def bad_bound(rows: Int, stride: Int, distance: Int) -> Int32
    requires 0 <= rows and rows <= 100000
    requires 0 <= stride and stride <= 100000
    requires 0 <= distance and distance <= rows * stride:
    return Int32(rows * stride) - Int32(distance)


# With the rows clamped to those of a tile it does.
def ok_bound(rows: Int, stride: Int, distance: Int) -> Int32
    requires 0 <= rows and rows <= 128
    requires 0 <= stride and stride <= 100000
    requires 0 <= distance and distance <= rows * stride:
    return Int32(rows * stride) - Int32(distance)


def bad_narrow(n: Int) -> UInt8:
    return UInt8(n)


def ok_masked(n: Int) -> UInt8:
    return UInt8(n & 0xFF)


def ok_guarded(n: Int) -> UInt8:
    if 0 <= n and n < 256:
        return UInt8(n)
    return 0


# A negative number is a large unsigned one.
def bad_to_unsigned(n: Int) -> UInt:
    return UInt(n)


def ok_to_unsigned(n: Int) -> UInt requires n >= 0:
    return UInt(n)


def bad_to_signed(u: UInt) -> Int:
    return Int(u)


def bad_widen(n: Int32) -> UInt64:
    return UInt64(n)


# A wider type of the same signedness, or a wider signed one, holds every
# value: nothing to prove.
def ok_widen(n: Int32, u: UInt32) -> Int:
    return Int(Int64(n)) + Int(UInt64(u)) + Int(Int64(u))


# `ord` of a literal is its code point; of any string, a code point.
def ok_ascii() -> Byte:
    return Byte(ord("H"))


def bad_not_ascii() -> Byte:
    return Byte(ord("€"))


def ok_codepoint(s: StringSpan) raises -> UInt32:
    return UInt32(ord(s))


def bad_codepoint(s: StringSpan) raises -> UInt16:
    return UInt16(ord(s))


def main() raises:
    print(bad_bound(4, 8, 0), ok_bound(4, 8, 0))
    print(bad_narrow(1), ok_masked(1), ok_guarded(1))
    print(bad_to_unsigned(1), ok_to_unsigned(1), bad_to_signed(1))
    print(bad_widen(1), ok_widen(1, 1))
    print(ok_ascii(), bad_not_ascii())
    print(ok_codepoint("a"), bad_codepoint("a"))
