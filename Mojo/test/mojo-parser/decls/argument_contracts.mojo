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


# Postconditions: a clause on an `out` argument becomes a `kgen.ensures` right
# before every return, over the arguments, the named result and the values of
# its `old(e)` calls, which a `kgen.old` computes on entry.

# CHECK-LABEL: lit.fn @"out_result
# CHECK-NOT: kgen.requires
# CHECK: kgen.ensures(%{{.*}}, %{{.*}} : {{.*}}, {{.*}})
# CHECK: kgen.contract.yield
# CHECK: lit.return
def out_result(a: Int, out r: Int where r > a):
    r = a + 1


# Every return gets its own copy.
# CHECK-LABEL: lit.fn @"two_returns
# CHECK: kgen.ensures
# CHECK: lit.return
# CHECK: kgen.ensures
# CHECK: lit.return
def two_returns(a: Int, out r: Int where r >= 0):
    if a > 0:
        r = a
        return
    r = 0


# On a `mut` argument, `old(e)` is `e` on entry: a `kgen.old` computes it, and
# its result is the last operand of the `kgen.ensures`.
# CHECK-LABEL: lit.fn @"mut_with_old
# CHECK-NOT: kgen.requires
# CHECK: %[[OLD:.*]] = kgen.old({{.*}}) -> !Int
# CHECK: kgen.contract.yield
# CHECK: kgen.ensures({{.*}}, %[[OLD]] : {{.*}})
# CHECK: lit.return
def mut_with_old(mut a: Int where a == old(a) + 1):
    a += 1


# Without `old`, a `mut` clause holds on entry and on exit.
# CHECK-LABEL: lit.fn @"mut_without_old
# CHECK: kgen.requires
# CHECK-NOT: kgen.old
# CHECK: kgen.ensures
# CHECK: lit.return
def mut_without_old(mut a: Int where a >= 0):
    a = 1


# Written only in terms of `old`, it is a precondition.
# CHECK-LABEL: lit.fn @"mut_only_old
# CHECK: kgen.requires
# CHECK-NOT: kgen.old
# CHECK-NOT: kgen.ensures
# CHECK: lit.end_fn
def mut_only_old(mut a: Int where old(a) > 0):
    a -= 1
