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

# Test the contract clauses of a function, for static verification (see
# Mojo/proposals/function-contracts.md). Each `requires` clause becomes a
# `kgen.requires` at the start of the body, whose region computes the
# condition from block arguments that stand for the function's arguments.
# Each `ensures` clause becomes a `kgen.ensures` before every return.

# RUN: %parse-mojo-isolated %s | FileCheck %s


# CHECK-LABEL: lit.fn @"one_clause
# CHECK: kgen.source_loc[0]
# CHECK: kgen.requires(%[[A:[a-z0-9_]+]] : {{.*}}) at(
# CHECK-NEXT: ^bb0(%{{[a-z0-9_]+}}: {{.*}}):
# CHECK-NOT: %[[A]]{{[ ,)]}}
# CHECK: kgen.contract.yield
def one_clause(a: Int) -> Int requires a > 0:
    return a


# CHECK-LABEL: lit.fn @"with_message
# CHECK: kgen.requires({{.*}}) at({{.*}}) {message = "b must be positive"}
def with_message(b: Int) requires b > 0 else "b must be positive":
    pass


# CHECK-LABEL: lit.fn @"paren_message
# CHECK: kgen.requires({{.*}}) at({{.*}}) {message = "b must be positive"}
def paren_message(b: Int) requires (b > 0, "b must be positive"):
    pass


# A clause may use every argument, and a function may have several clauses:
# one op per clause, each over all the function's arguments. A clause may
# start a new line.
# CHECK-LABEL: lit.fn @"two_clauses
# CHECK: kgen.requires(%{{.*}}, %{{.*}} : {{.*}}, {{.*}})
# CHECK: kgen.requires(%{{.*}}, %{{.*}} : {{.*}}, {{.*}})
# CHECK-NOT: kgen.requires
# CHECK: lit.end_fn
def two_clauses(lo: Int, hi: Int)
    requires lo <= hi
    requires hi < 100:
    pass


# The clauses follow the function's effects, result type and `where` clauses.
# CHECK-LABEL: lit.fn @"after_where
# CHECK-COUNT-2: kgen.requires
# CHECK: kgen.ensures
def after_where[T: AnyType](imm a: Int, var b: Int) raises -> Int
    where conforms_to(T, Copyable)
    requires a >= 0
    requires b >= a
    ensures result >= 0:
    return b


# CHECK-LABEL: lit.fn @"no_clauses
# CHECK-NOT: kgen.requires
# CHECK: lit.end_fn
def no_clauses(a: Int):
    pass


# The keywords are soft: ordinary names anywhere else.
# CHECK-LABEL: lit.fn @"soft_keywords
# CHECK-NOT: kgen.requires
# CHECK-NOT: kgen.ensures
# CHECK: lit.end_fn
def soft_keywords(requires: Int, result: Int) -> Int:
    var ensures = requires + result
    return ensures


# Postconditions: an `ensures` clause becomes a `kgen.ensures` right before
# every return, over the arguments and the result. A named `out` result is
# the function's local for it.

# CHECK-LABEL: lit.fn @"out_result
# CHECK-NOT: kgen.requires
# CHECK: kgen.ensures(%{{.*}}, %{{.*}} : {{.*}}, {{.*}})
# CHECK: kgen.contract.yield
# CHECK: lit.return
def out_result(a: Int, out r: Int) ensures r > a:
    r = a + 1


# `result` is the result. An unnamed one in a register goes through a local
# at each return, which the clause reads like a named `out` result.
# CHECK-LABEL: lit.fn @"arrow_result
# CHECK: %[[RESULT:.*]] = lit.var.decl "result" arg
# CHECK: lit.ref.store
# CHECK: kgen.ensures(%{{.*}}, %[[RESULT]] : {{.*}}, {{.*}})
# CHECK: kgen.contract.yield
# CHECK: lit.return
def arrow_result(a: Int) -> Int ensures result > a:
    return a + 1


# A result in memory is the function's result slot.
# CHECK-LABEL: lit.fn @"memory_result
# CHECK-NOT: lit.var.decl "result"
# CHECK: kgen.ensures(%{{.*}}, %{{.*}} : {{.*}}, {{.*}})
# CHECK: lit.return
def memory_result(n: Int) -> Pair ensures result.first == n:
    return Pair(n, n)


@fieldwise_init
struct Pair:
    var first: Int
    var second: Int


# `result` is another name for a named `out` result, unless an argument has
# that name.
# CHECK-LABEL: lit.fn @"result_of_out
# CHECK-COUNT-2: kgen.ensures
# CHECK: lit.return
def result_of_out(a: Int, out r: Int) ensures r > a ensures result > a:
    r = a + 1


# CHECK-LABEL: lit.fn @"result_argument
# CHECK-NOT: lit.var.decl "result"
# CHECK: kgen.ensures
# CHECK: lit.return
def result_argument(result: Int) ensures result >= 0 or result < 0:
    pass


# Every return gets its own copy.
# CHECK-LABEL: lit.fn @"two_returns
# CHECK: kgen.ensures
# CHECK: lit.return
# CHECK: kgen.ensures
# CHECK: lit.return
def two_returns(a: Int) -> Int ensures result >= 0:
    if a > 0:
        return a
    return 0


# In an `ensures` clause, `old(e)` is `e` on entry: a `kgen.old` inside the
# `kgen.ensures`, evaluated with memory as it is at the function's
# `kgen.contract.entry`.
# CHECK-LABEL: lit.fn @"mut_with_old
# CHECK-NOT: kgen.requires
# CHECK: %[[ENTRY:.*]] = kgen.contract.entry
# CHECK: kgen.ensures
# CHECK: kgen.old(%[[ENTRY]], {{.*}}) -> !Int
# CHECK: lit.return
def mut_with_old(mut a: Int) ensures a == old(a) + 1:
    a += 1


# A fact that holds on entry and on exit is stated twice. Without `old`,
# there is no entry marker.
# CHECK-LABEL: lit.fn @"mut_before_and_after
# CHECK: kgen.requires
# CHECK-NOT: kgen.contract.entry
# CHECK-NOT: kgen.old
# CHECK: kgen.ensures
# CHECK: lit.return
def mut_before_and_after(mut a: Int) requires a >= 0 ensures a >= 0:
    a = 1


# A `requires` clause reads a `mut` argument on entry.
# CHECK-LABEL: lit.fn @"mut_only_before
# CHECK: kgen.requires
# CHECK-NOT: kgen.contract.entry
# CHECK-NOT: kgen.old
# CHECK-NOT: kgen.ensures
# CHECK: lit.end_fn
def mut_only_before(mut a: Int) requires a > 0:
    a -= 1


# A required trait method keeps its contract: its body is the contract ops.
trait Counter:
    # CHECK-LABEL: lit.fn @"count
    # CHECK: kgen.ensures
    # CHECK-NEXT: ^bb0
    def count(self) -> Int ensures result >= 0: ...

    # CHECK-LABEL: lit.fn @"bump
    # CHECK: kgen.requires
    # CHECK: kgen.contract.entry
    # CHECK: kgen.ensures
    # CHECK: kgen.old
    def bump(mut self)
        requires self.count() < 100
        ensures self.count() == old(self.count()) + 1: ...


# `all([cond for i in range(...)])` in a contract is a quantifier: a
# `kgen.forall` over the index, whose region yields the condition.
# CHECK-LABEL: lit.fn @"all_nonneg
# CHECK: kgen.requires
# CHECK: kgen.forall({{ *}}%{{.*}} : !Int) -> !Bool
# CHECK: ^bb0(%{{.*}}: !Int):
# CHECK: kgen.contract.yield
def all_nonneg(n: Int) requires all([i * i >= 0 for i in range(n)]):
    pass


# With both bounds, and `old(e)` using the index: the `kgen.old` takes it as
# an operand.
# CHECK-LABEL: lit.fn @"shift_rest
# CHECK: %[[ENTRY:.*]] = kgen.contract.entry
# CHECK: kgen.ensures
# CHECK: kgen.forall({{ *}}%{{.*}} : !Int, %{{.*}} : !Int) -> !Bool
# CHECK: ^bb0(%[[I:.*]]: !Int):
# CHECK: kgen.old(%[[ENTRY]], {{.*}}%[[I]] : {{.*}}) -> !Int
def shift_rest(mut a: Int)
    ensures all([a + i == old(a) + i for i in range(1, 3)]):
    pass
