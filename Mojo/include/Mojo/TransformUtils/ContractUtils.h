//===----------------------------------------------------------------------===//
//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TRANSFORMUTILS_CONTRACTUTILS_H
#define KGEN_TRANSFORMUTILS_CONTRACTUTILS_H

#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
class Operation;
} // namespace mlir

namespace M::KGEN {

/// Marks the `hlcf.if` the parser emits for the run-time check of a
/// precondition (`if _contract_checks_enabled(): _check_contract(...)`), so
/// `verify-contracts` can leave it out for the calls it proves.
inline constexpr char kContractCheckAttrName[] = "kgen.contract_check";

/// Marks, until the parser has split the function, the `hlcf.if` it emits
/// next to such a check to assume the precondition: it goes to the copy of
/// the function that has the body.
inline constexpr char kContractAssumeAttrName[] = "kgen.contract_assume";

/// On a function the parser split for its checked preconditions: the symbol,
/// next to it, of its copy without the checks, which it only calls.
inline constexpr char kCheckedWrapperAttrName[] = "kgen.checked_wrapper";

/// What the name of that copy adds to the function's.
inline constexpr char kUncheckedSuffix[] = "$unchecked";

/// Give every contract op (`kgen.requires`, `kgen.ensures`, `kgen.old`) under
/// `root` that reads a stack slot through a pointer operand a snapshot of it
/// instead: the value is loaded right before the op and stored into a fresh
/// slot, which becomes the operand. The op still reads the value the slot held
/// at that point, and the original slot is left with loads and stores only, so
/// SROA and mem2reg can promote it. Snapshot slots are marked, and never
/// snapshotted again. (Defined in the Transforms library, which depends on the
/// dialects; TransformUtils cannot.)
void snapshotContractOperands(mlir::Operation *root);

/// The callee attribute of a call with the function it names, `from`,
/// replaced by `to`; `callee` itself if it does not name `from`. (Defined in
/// the Transforms library.)
mlir::Attribute retargetCallee(mlir::Attribute callee, mlir::SymbolRefAttr from,
                               mlir::SymbolRefAttr to);

} // namespace M::KGEN

#endif // KGEN_TRANSFORMUTILS_CONTRACTUTILS_H
