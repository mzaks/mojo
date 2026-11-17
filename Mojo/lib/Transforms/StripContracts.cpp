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
//
// This file implements the strip-contracts pass, which removes the contract
// ops of a module once nothing after it reads them.
//
//===----------------------------------------------------------------------===//

#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/ToolCommon/KGENPasses.h"

using namespace mlir;
using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_STRIPCONTRACTS
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct StripContractsPass
    : M::KGEN::impl::StripContractsBase<StripContractsPass> {
  void runOnOperation() override;
};
} // namespace

void StripContractsPass::runOnOperation() {
  // The outermost contract ops: what is nested in one (a `kgen.old`, a
  // `kgen.forall`) goes with it.
  SmallVector<Operation *> contracts, entries;
  getOperation()->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isa<RequiresOp, EnsuresOp>(op)) {
      contracts.push_back(op);
      return WalkResult::skip();
    }
    if (isa<ContractEntryOp>(op))
      entries.push_back(op);
    return WalkResult::advance();
  });
  for (Operation *op : contracts)
    op->erase();
  numContracts += contracts.size();

  // The entry markers only served the `kgen.old`s of the postconditions.
  for (Operation *op : entries)
    if (op->use_empty())
      op->erase();
}
