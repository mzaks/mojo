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

# Test that the run-time check of a `requires` clause follows the switch of
# assertions, `-D ASSERT`, and that `--contract-prover` does not turn checks
# on or off.

# RUN: mkdir -p %t

# Like a `debug_assert`, the clause is not checked by default.
# RUN: mojo build %s -o %t/default.ll --emit llvm
# RUN: FileCheck %s --check-prefix=OFF --input-file=%t/default.ll

# It is with all assertions on.
# RUN: mojo build %s -D ASSERT=all -o %t/all.ll --emit llvm
# RUN: FileCheck %s --check-prefix=ON --input-file=%t/all.ll

# It never is with assertions off.
# RUN: mojo build %s -D ASSERT=none -o %t/none.ll --emit llvm
# RUN: FileCheck %s --check-prefix=OFF --input-file=%t/none.ll

# The prover is a separate setting: naming the default changes nothing.
# RUN: mojo build %s --contract-prover=none -o %t/prover.ll --emit llvm
# RUN: FileCheck %s --check-prefix=OFF --input-file=%t/prover.ll
# RUN: mojo build %s -D ASSERT=all --contract-prover=none -o %t/both.ll --emit llvm
# RUN: FileCheck %s --check-prefix=ON --input-file=%t/both.ll

# An unknown prover is rejected with a diagnostic.
# RUN: not mojo build %s --contract-prover=smt -o %t/bad.ll --emit llvm 2>&1 \
# RUN:   | FileCheck %s --check-prefix=INVALID
# INVALID: invalid value 'smt' for '--contract-prover', expected one of: `none`, `z3`

# ON: precondition of 'half' does not hold: x % 2 == 0
# OFF-NOT: precondition of 'half' does not hold


@no_inline
def half(x: Int) -> Int requires x % 2 == 0:
    return x // 2


# A value the compiler does not know, so the check is not folded away.
@no_inline
def four() -> Int:
    return 4


def main():
    print(half(four()))
