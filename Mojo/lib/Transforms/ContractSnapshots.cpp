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

#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/TransformUtils/ContractUtils.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace M;
using namespace M::KGEN;

/// Marks the slots `snapshotContractOperands` creates.
static constexpr llvm::StringLiteral kSnapshotAttr = "kgen.contract_snapshot";

/// The stack slot `ptr` points into, through field addresses and views.
static POP::StackAllocationOp stackSlotOf(Value ptr) {
  while (Operation *def = ptr.getDefiningOp()) {
    if (auto alloc = dyn_cast<POP::StackAllocationOp>(def))
      return alloc;
    if (auto gep = dyn_cast<StructGEPOp>(def))
      ptr = gep.getContainer();
    else if (auto cast = dyn_cast<POP::PointerBitcastOp>(def))
      ptr = cast.getInput();
    else if (auto view = dyn_cast<POP::UnionBitcastOp>(def))
      ptr = view.getValue();
    else if (auto element = dyn_cast<POP::ArrayGEPOp>(def))
      ptr = element.getArray();
    else
      return {};
  }
  return {};
}

void M::KGEN::snapshotContractOperands(Operation *root) {
  SmallVector<Operation *> contracts;
  root->walk([&](Operation *op) {
    if (isa<RequiresOp, EnsuresOp, OldOp>(op))
      contracts.push_back(op);
  });
  for (Operation *op : contracts) {
    for (OpOperand &operand : op->getOpOperands()) {
      auto type = dyn_cast<PointerType>(operand.get().getType());
      POP::StackAllocationOp slot = stackSlotOf(operand.get());
      if (!type || !slot || slot->hasAttr(kSnapshotAttr))
        continue;
      OpBuilder builder(op);
      Location loc = op->getLoc();
      Value value = POP::LoadOp::create(builder, loc, operand.get());
      auto snapshot = POP::StackAllocationOp::create(builder, loc, type);
      snapshot->setAttr(kSnapshotAttr, builder.getUnitAttr());
      POP::StoreOp::create(builder, loc, value, snapshot);
      operand.set(snapshot);
    }
  }
}
