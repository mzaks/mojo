// RUN: kgen-opt -strip-contracts %s | FileCheck %s

// A precondition and its region go, the code around them stays.
// CHECK-LABEL: @requires
// CHECK-NOT: kgen.requires
// CHECK-NOT: kgen.contract.yield
// CHECK: hlcf.return %arg0
kgen.func @requires(%arg0: !kgen.scalar<index>) -> !kgen.scalar<index> {
  %line = kgen.param.constant = <1>
  %col = kgen.param.constant = <2>
  %file = kgen.param.constant: string = <"a.mojo">
  %zero = kgen.param.constant: scalar<index> = <0>
  kgen.requires(%arg0 : !kgen.scalar<index>) at(%line, %col, %file) : !kgen.string {
  ^bb0(%a: !kgen.scalar<index>):
    %c = pop.cmp ge(%a, %zero) : <1, index>
    kgen.contract.yield %c : !kgen.scalar<bool>
  }
  hlcf.return %arg0 : !kgen.scalar<index>
}

// A postcondition goes with the `kgen.old` in it, and the entry marker that
// only the `kgen.old` used goes too.
// CHECK-LABEL: @ensures
// CHECK-NOT: kgen.contract.entry
// CHECK-NOT: kgen.ensures
// CHECK-NOT: kgen.old
// CHECK: hlcf.return %arg0
kgen.func @ensures(%arg0: !kgen.scalar<index>) -> !kgen.scalar<index> {
  %line = kgen.param.constant = <1>
  %col = kgen.param.constant = <2>
  %file = kgen.param.constant: string = <"a.mojo">
  %entry = kgen.contract.entry
  kgen.ensures(%arg0 : !kgen.scalar<index>) at(%line, %col, %file) : !kgen.string {
  ^bb0(%a: !kgen.scalar<index>):
    %before = kgen.old(%entry, %a : !kgen.scalar<index>) -> !kgen.scalar<index> {
    ^bb0(%b: !kgen.scalar<index>):
      kgen.contract.yield %b : !kgen.scalar<index>
    }
    %c = pop.cmp ge(%a, %before) : <1, index>
    kgen.contract.yield %c : !kgen.scalar<bool>
  }
  hlcf.return %arg0 : !kgen.scalar<index>
}
