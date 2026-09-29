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

namespace mlir {
class Operation;
} // namespace mlir

namespace M::KGEN {

/// Give every contract op (`kgen.requires`, `kgen.ensures`, `kgen.old`) under
/// `root` that reads a stack slot through a pointer operand a snapshot of it
/// instead: the value is loaded right before the op and stored into a fresh
/// slot, which becomes the operand. The op still reads the value the slot held
/// at that point, and the original slot is left with loads and stores only, so
/// SROA and mem2reg can promote it. Snapshot slots are marked, and never
/// snapshotted again. (Defined in the Transforms library, which depends on the
/// dialects; TransformUtils cannot.)
void snapshotContractOperands(mlir::Operation *root);

} // namespace M::KGEN

#endif // KGEN_TRANSFORMUTILS_CONTRACTUTILS_H
