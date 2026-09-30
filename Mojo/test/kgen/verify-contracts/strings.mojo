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
# `String`, `StringSlice` and string literals for `verify-contracts`; see
# README.md.


# --- must be PROVEN ---
def ok_literal_bytes() -> Byte:
    return "Hello".as_bytes()[4]  # a literal's length is in its type


def ok_string_from_literal() -> Int:
    var s: String = "héllo"  # 6 bytes: `é` is two
    return s[byte=5].byte_length()


def ok_byte_index(s: String, i: Int) -> Int:
    if 0 <= i and i < s.byte_length():
        return s[byte=i].byte_length()
    return 0


def ok_byte_slice() -> Int:
    var s: String = "Hello"
    var t = s[byte=1:4]  # three bytes
    return t[byte=2].byte_length()


def ok_as_bytes() -> Byte:
    var s: String = "Hello"
    var b = s.as_bytes()
    return b[4]


def ok_as_bytes_mut() -> Byte:
    var s: String = "Hello"
    var b = s.unsafe_as_bytes_mut()
    return b[4]


def ok_append() -> Int:
    var s = String()
    s += "abc"
    s += "de"
    return s[byte=4].byte_length()


def ok_span_loop(s: StringSlice) -> Int:
    var t = 0
    var b = s.as_bytes()
    for i in range(s.byte_length()):
        t += Int(b[i])
    return t


# --- must stay UNPROVEN ---
def bad_byte_past() -> Int:
    var s: String = "héllo"
    return s[byte=6].byte_length()


def bad_byte_index(s: String, i: Int) -> Int:
    if 0 <= i and i <= s.byte_length():
        return s[byte=i].byte_length()  # `i` may be the length
    return 0


def bad_slice_past() -> Int:
    var s: String = "Hello"
    return s[byte=2:6].byte_length()


def bad_empty() -> Int:
    var s = String()
    return s[byte=0].byte_length()
