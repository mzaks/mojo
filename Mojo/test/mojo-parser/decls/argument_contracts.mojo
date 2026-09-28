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

# Test `where` clauses on runtime arguments: contracts for static verification
# (see Mojo/proposals/argument-contracts.md). Each clause becomes a
# `kgen.requires` at the start of the body, whose region computes the
# condition from block arguments that stand for the function's arguments.

# RUN: %parse-mojo-isolated %s | FileCheck %s


# CHECK-LABEL: lit.fn @"one_clause
# CHECK: kgen.source_loc[0]
# CHECK: kgen.requires(%[[A:[a-z0-9_]+]] : {{.*}}) at(
# CHECK-NEXT: ^bb0(%{{[a-z0-9_]+}}: {{.*}}):
# CHECK-NOT: %[[A]]{{[ ,)]}}
# CHECK: kgen.contract.yield
def one_clause(a: Int where a > 0) -> Int:
    return a


# CHECK-LABEL: lit.fn @"with_message
# CHECK: kgen.requires({{.*}}) at({{.*}}) {message = "b must be positive"}
def with_message(b: Int where b > 0 else "b must be positive"):
    pass


# CHECK-LABEL: lit.fn @"paren_message
# CHECK: kgen.requires({{.*}}) at({{.*}}) {message = "b must be positive"}
def paren_message(b: Int where (b > 0, "b must be positive")):
    pass


# A clause may use other arguments, and an argument may have several clauses:
# one op per clause, each over all the function's arguments.
# CHECK-LABEL: lit.fn @"two_clauses
# CHECK: kgen.requires(%{{.*}}, %{{.*}} : {{.*}}, {{.*}})
# CHECK: kgen.requires(%{{.*}}, %{{.*}} : {{.*}}, {{.*}})
# CHECK-NOT: kgen.requires
# CHECK: lit.end_fn
def two_clauses(lo: Int, hi: Int where lo <= hi where hi < 100):
    pass


# CHECK-LABEL: lit.fn @"imm_and_var
# CHECK-COUNT-2: kgen.requires
def imm_and_var(imm a: Int where a >= 0, var b: Int where b >= a):
    pass


# CHECK-LABEL: lit.fn @"no_clauses
# CHECK-NOT: kgen.requires
# CHECK: lit.end_fn
def no_clauses(a: Int):
    pass
