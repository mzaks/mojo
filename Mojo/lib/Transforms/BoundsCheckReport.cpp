//===----------------------------------------------------------------------===//
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
//===----------------------------------------------------------------------===//
//
// Report-only prototype: try to prove every `kgen.obligation` with an SMT
// solver.
//
// Encoding (per function, into SMT-LIB text):
//   - Integer-like SSA values (index, uindex, fixed-width ints, bool) become
//     bitvector / Bool terms with exact (wrapping) semantics. Ops that are not
//     modeled, loads, calls, and non-integer values produce unconstrained
//     constants. Over-approximating is always sound: it can only make fewer
//     obligations provable.
//   - Structured control flow becomes reachability conditions: every block is
//     encoded under the condition under which it executes. Values that arrive
//     from several places (if/loop/try results, try handler arguments) are
//     fresh constants constrained by one implication per source
//     ("source reached => value = what the source sends").
//   - A loop's block arguments are fresh constants standing for "the state of
//     some iteration". They are constrained by invariants inferred with
//     Houdini: candidates are checked for initiation (loop entry) and
//     consecution (every `continue` edge), failing candidates are dropped, and
//     the process repeats until a fixpoint. Each check assumes the current
//     candidates of the loops enclosing the checked edge only, which keeps the
//     usual Houdini induction argument valid.
//
// An obligation is proven if "reached and condition false" is unsatisfiable
// under the invariants of its enclosing loops. It is "implied" if that only
// holds when additionally assuming the obligations that precede it on the
// same path.
//
//===----------------------------------------------------------------------===//

#include "Mojo/ToolCommon/KGENPasses.h"

#include "Mojo/HLCFDialect/HLCFOps.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/POPDialect/POPOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Threading.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"

#include <deque>
#include <map>
#include <set>

using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_BOUNDSCHECKREPORT
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {

//===----------------------------------------------------------------------===//
// Sorts
//===----------------------------------------------------------------------===//

/// The SMT sort of an SSA value. `None` values are not encoded at all.
struct Sort {
  /// `Ptr` is a 64-bit address, kept apart from `BV` so that no arithmetic
  /// or loop invariant template mixes pointers with integers.
  enum Kind { None, Bool, BV, Ptr } kind = None;
  unsigned width = 0;
  /// Default interpretation for ops whose semantics depend on signedness
  /// (comparisons, max/min, extension, division).
  bool isSigned = true;

  bool operator==(const Sort &o) const {
    return kind == o.kind && width == o.width;
  }
  std::string str() const {
    return kind == Bool ? "Bool" : "(_ BitVec " + std::to_string(width) + ")";
  }
};

Sort sortOf(Type type) {
  if (isa<PointerType>(type))
    return {Sort::Ptr, 64, false};
  if (isa<IndexType>(type))
    return {Sort::BV, 64, true};
  if (auto intTy = dyn_cast<IntegerType>(type)) {
    if (intTy.getWidth() == 1)
      return {Sort::Bool, 1, false};
    return {Sort::BV, intTy.getWidth(), !intTy.isUnsigned()};
  }
  if (auto simd = dyn_cast<SIMDType>(type)) {
    auto size = dyn_cast_or_null<IntegerAttr>(simd.getSize());
    if (!size || size.getInt() != 1)
      return {};
    std::optional<KGENDType> dtype = simd.getResolvedDType();
    if (!dtype)
      return {};
    if (dtype->isBool())
      return {Sort::Bool, 1, false};
    if (dtype->isIndex() || dtype->isUIndex())
      return {Sort::BV, 64, dtype->isIndex()};
    if (dtype->isSInt() || dtype->isUInt()) {
      ssize_t width = dtype->getWidthInBits(TargetInfoAttr());
      if (width <= 0)
        return {};
      return {Sort::BV, static_cast<unsigned>(width), dtype->isSInt()};
    }
  }
  return {};
}

std::string bvConst(const APInt &value) {
  return "(_ bv" + llvm::toString(value, 10, /*Signed=*/false) + " " +
         std::to_string(value.getBitWidth()) + ")";
}

std::string mkNot(StringRef a) {
  if (a == "true")
    return "false";
  if (a == "false")
    return "true";
  return ("(not " + a + ")").str();
}

std::string mkAnd(StringRef a, StringRef b) {
  if (a == "false" || b == "false")
    return "false";
  if (a == "true")
    return b.str();
  if (b == "true")
    return a.str();
  return ("(and " + a + " " + b + ")").str();
}

std::string mkOr(ArrayRef<std::string> terms) {
  SmallVector<StringRef> live;
  for (const std::string &t : terms) {
    if (t == "true")
      return "true";
    if (t != "false")
      live.push_back(t);
  }
  if (live.empty())
    return "false";
  if (live.size() == 1)
    return live.front().str();
  std::string result = "(or";
  for (StringRef t : live)
    result += (" " + t).str();
  return result + ")";
}

//===----------------------------------------------------------------------===//
// Loop, try and obligation bookkeeping
//===----------------------------------------------------------------------===//

using MaybeTerm = std::optional<std::string>;
struct LoopInfo;

/// A control transfer into a loop head, a loop exit or a try handler.
struct Edge {
  std::string reach;
  SmallVector<MaybeTerm> values;
  /// Loops whose invariants hold at the source of the edge (enclosing loops
  /// and loops completed before it).
  SmallVector<LoopInfo *> enclosing;
  /// Assumptions made before the source of the edge.
  std::string assumed = "true";
};

/// A Houdini candidate for a loop head: either `arg <rel> other` (e.g.
/// `i <= n`), or, for a relational template, `(rel arg other) == anchor`
/// with `rel` "bvadd" or "bvsub" and `anchor` the same expression over the
/// values at loop entry (e.g. `i + len == i0 + len0`).
struct Candidate {
  unsigned arg;
  std::string rel; // SMT-LIB comparison (e.g. "bvsle") or "bvadd"/"bvsub".
  /// Either another loop argument or a term available at the loop head.
  std::optional<unsigned> otherArg;
  /// The other term, or the anchor of a relational template.
  std::string otherTerm;
  bool alive = true;
  bool relational = false;
};

struct LoopInfo {
  unsigned id = 0;
  std::string reachIn;
  /// Assumptions made before the loop is entered.
  std::string assumedAtEntry = "true";
  SmallVector<MaybeTerm> args, inits;
  SmallVector<Sort> argSorts;
  SmallVector<LoopInfo *> enclosing;
  SmallVector<Edge> continues, breaks;
  /// Integer terms available at the loop head that the body compares against.
  SmallVector<std::pair<std::string, Sort>> boundTerms;
  std::vector<Candidate> candidates;

  /// Render `cand` with loop arguments replaced by `subst`.
  MaybeTerm render(const Candidate &cand, ArrayRef<MaybeTerm> subst) const {
    if (cand.arg >= subst.size() || !subst[cand.arg])
      return std::nullopt;
    if (cand.relational) {
      if (!cand.otherArg || *cand.otherArg >= subst.size() ||
          !subst[*cand.otherArg])
        return std::nullopt;
      return "(= (" + cand.rel + " " + *subst[cand.arg] + " " +
             *subst[*cand.otherArg] + ") " + cand.otherTerm + ")";
    }
    std::string other = cand.otherTerm;
    if (cand.otherArg) {
      if (*cand.otherArg >= subst.size() || !subst[*cand.otherArg])
        return std::nullopt;
      other = *subst[*cand.otherArg];
    }
    return "(" + cand.rel + " " + *subst[cand.arg] + " " + other + ")";
  }

  /// Conjunction of the currently alive candidates over the loop arguments.
  std::string invariant() const {
    std::string result = "true";
    for (const Candidate &cand : candidates)
      if (cand.alive)
        if (MaybeTerm t = render(cand, args))
          result = mkAnd(result, *t);
    return result;
  }
};

struct TryInfo {
  LIT::TryOp op;
  SmallVector<Edge> raises;
};

struct ObligationLocation {
  StringRef file;
  int64_t line = 0, col = 0;
};

std::optional<ObligationLocation> getObligationLocation(ObligationOp op) {
  IntegerAttr line, col;
  StringAttr file;
  if (!mlir::matchPattern(op.getLine(), mlir::m_Constant(&line)) ||
      !mlir::matchPattern(op.getCol(), mlir::m_Constant(&col)) ||
      !mlir::matchPattern(op.getFileName(), mlir::m_Constant(&file)))
    return std::nullopt;
  return ObligationLocation{file.getValue(), line.getInt(), col.getInt()};
}

StringRef getObligationKind(ObligationOp op) {
  if (auto kind = dyn_cast<StringAttr>(op.getKind()))
    return kind.getValue();
  return "<unknown>";
}

/// The source location of an op itself (e.g. a call), preferring the
/// innermost frame: the code as written at that point.
std::optional<ObligationLocation> locationOf(Location loc) {
  while (true) {
    if (auto callSite = dyn_cast<mlir::CallSiteLoc>(loc))
      loc = callSite.getCallee();
    else if (auto name = dyn_cast<mlir::NameLoc>(loc))
      loc = name.getChildLoc();
    else if (auto fused = dyn_cast<mlir::FusedLoc>(loc);
             fused && !fused.getLocations().empty())
      loc = fused.getLocations().front();
    else
      break;
  }
  if (auto file = dyn_cast<FileLineColLoc>(loc))
    return ObligationLocation{file.getFilename().getValue(), file.getLine(),
                              file.getColumn()};
  return std::nullopt;
}

/// A `requires` records the location of the call to the annotated function,
/// which only resolves once that function is inlined into a caller. Still
/// unresolved, it is the precondition of the function being analyzed itself:
/// assumed there, and checked at its call sites.
bool isOwnPrecondition(ObligationOp op) {
  return getObligationKind(op) == "requires" &&
         op.getLine().getDefiningOp<SourceLocOp>();
}

struct ObligationInfo {
  ObligationOp op;
  /// What to report: usually taken from `op`, but a callee's precondition
  /// checked at a call site is reported at the call.
  StringRef kind;
  std::optional<ObligationLocation> location;
  std::string reach, cond;
  SmallVector<LoopInfo *> enclosing;
  /// (reach, cond) of obligations preceding this one on the same path.
  SmallVector<std::pair<std::string, std::string>> earlier;
  /// Assumptions made before this obligation.
  std::string assumed;
};

//===----------------------------------------------------------------------===//
// Encoder
//===----------------------------------------------------------------------===//

class Encoder {
public:
  /// Declarations and global assertions, in dependency order.
  std::string prelude;
  std::deque<LoopInfo> loops;
  std::vector<ObligationInfo> obligations;
  /// Number of callee `ensures` assumed at call sites.
  unsigned contractsUsed = 0;

  explicit Encoder(const mlir::SymbolTable *symbols) : symbols(symbols) {}

  /// Record which IR value each unknown stands for and which names each
  /// defined term uses, and comment the script with it (for `dump-dir` and
  /// `explain`).
  bool annotate = false;

  /// The unknowns (with their IR descriptions) a term transitively depends
  /// on, e.g. the loads or call results an obligation's condition could not
  /// see through. Requires `annotate`.
  /// Returns (SMT name, description) pairs sorted by description.
  SmallVector<std::pair<std::string, std::string>>
  unknownInputs(StringRef expr) const {
    SmallVector<std::pair<std::string, std::string>> result;
    llvm::StringSet<> seen;
    SmallVector<std::string> worklist = namesIn(expr);
    while (!worklist.empty()) {
      std::string name = worklist.pop_back_val();
      if (!seen.insert(name).second)
        continue;
      if (auto it = unknowns.find(name); it != unknowns.end())
        result.push_back({name, it->second});
      if (auto it = deps.find(name); it != deps.end())
        worklist.append(it->second.begin(), it->second.end());
    }
    // Uninitialized reads are usually from arms that cannot be taken; list
    // them last so the cap on shown unknowns keeps the informative ones.
    llvm::sort(result, [](auto &a, auto &b) {
      bool ua = StringRef(a.second).contains("uninitialized read");
      bool ub = StringRef(b.second).contains("uninitialized read");
      return std::tie(ua, a.second, a.first) < std::tie(ub, b.second, b.first);
    });
    return result;
  }

  void encodeFunction(FuncOp func) {
    self = func;
    // Name the function in the script, for `dump-dir` debugging.
    prelude += ("; " + func.getSymName() + "\n").str();
    Region &body = func->getRegion(0);
    if (!body.empty())
      encodeBlock(body.front(), "true");
  }

private:
  const mlir::SymbolTable *symbols;
  FuncOp self;
  llvm::StringMap<std::string> unknowns;
  llvm::StringMap<SmallVector<std::string>> deps;

  /// The names of terms (`h12`, `v3`, `r7`, `a9`) used in an SMT expression.
  static SmallVector<std::string> namesIn(StringRef expr) {
    SmallVector<std::string> names;
    SmallVector<StringRef> tokens;
    expr.split(tokens, ' ', -1, /*KeepEmpty=*/false);
    for (StringRef token : tokens) {
      token = token.trim("()");
      if (token.size() < 2 || !StringRef("hvra").contains(token.front()))
        continue;
      if (llvm::all_of(token.drop_front(), llvm::isDigit))
        names.push_back(token.str());
    }
    return names;
  }

  /// A short description of an IR value: what defines it and where.
  std::string describe(Value value) {
    std::string text;
    llvm::raw_string_ostream os(text);
    if (ctx)
      os << "in " << ctx->callee << ": ";
    Location loc = value.getLoc();
    if (Operation *def = value.getDefiningOp()) {
      os << def->getName().getStringRef();
      if (auto call = dyn_cast<CallOp>(def))
        if (auto callee = dyn_cast<SymbolConstantAttr>(call.getCallee()))
          os << " "
             << callee.getSymbol()
                    .getRootReference()
                    .getValue()
                    .split('(')
                    .first;
      if (def->getNumResults() > 1)
        os << " result #" << cast<OpResult>(value).getResultNumber();
      loc = def->getLoc();
    } else {
      auto arg = cast<BlockArgument>(value);
      os << "argument #" << arg.getArgNumber() << " of "
         << arg.getOwner()->getParentOp()->getName().getStringRef();
      loc = arg.getOwner()->getParentOp()->getLoc();
    }
    if (std::optional<ObligationLocation> where = locationOf(loc))
      os << " at " << where->file << ":" << where->line << ":" << where->col;
    return text;
  }

  /// Record that `name` is an unknown standing for `description`.
  void noteUnknown(StringRef name, std::string description) {
    if (!annotate)
      return;
    prelude += ("; " + name + ": " + description + "\n").str();
    unknowns[name] = std::move(description);
  }
  unsigned counter = 0;
  DenseMap<Value, std::string> terms;
  DenseMap<std::pair<Value, unsigned>, std::string> extracts;
  /// Opaque multi-step accesses, keyed by aggregate and path.
  std::map<std::pair<void *, std::string>, std::string> paths;
  SmallVector<LoopInfo *> loopStack;
  /// Loops that have run to completion before the current point, in the
  /// current block and its ancestors. Their invariants held on their last
  /// iteration, so later code may assume them.
  SmallVector<LoopInfo *> completedLoops;
  SmallVector<TryInfo *> tryStack;
  SmallVector<std::pair<std::string, std::string>> activeObligations;

  /// A callee evaluated for one call site: its entry arguments stand for the
  /// call's operands, and its other values are computed on demand from their
  /// definitions (anything not modeled stays unconstrained).
  struct CallContext {
    /// The callee's name (for descriptions).
    std::string callee;
    /// The context the call itself is evaluated in (null: the function being
    /// analyzed); the callee's arguments are values there.
    CallContext *parent = nullptr;
    unsigned depth = 0;
    /// The call being evaluated.
    Operation *call = nullptr;
    /// Results of calls inside this callee, backed by nested contexts.
    DenseMap<Value, std::pair<CallContext *, Value>> results;
    /// Contexts of calls inside this callee (see `contextFor`).
    DenseMap<Operation *, CallContext *> calls;
    DenseMap<Value, Value> args;
    /// For an iteration of a loop (see `iterationContext`): the loop, the
    /// context it is evaluated in, and the iteration's number. `parent` is
    /// then the previous iteration (or `owner` for the first one), whose
    /// `hlcf.continue` operands are this iteration's block arguments.
    Operation *loop = nullptr;
    CallContext *owner = nullptr;
    unsigned iteration = 0;
    /// Iterations of loops in this context (see `iterationContext`).
    DenseMap<std::pair<Operation *, unsigned>, CallContext *> iterations;
    DenseMap<Value, std::string> terms;
    DenseMap<std::pair<Value, unsigned>, std::string> extracts;
    std::map<std::pair<void *, std::string>, std::string> paths;
  };
  std::deque<CallContext> contexts;
  /// The callee context being evaluated, or null for the function itself.
  CallContext *ctx = nullptr;
  /// Call results backed by a callee's returned value in a context.
  DenseMap<Value, std::pair<CallContext *, Value>> callResults;
  /// Contexts of calls in the function itself (see `contextFor`).
  DenseMap<Operation *, CallContext *> callContexts;
  /// Iterations of loops in the function itself (see `iterationContext`).
  DenseMap<std::pair<Operation *, unsigned>, CallContext *> loopIterations;

  /// The context `value` is evaluated in: an iteration context only
  /// evaluates the values of its loop's body, the others belong to the
  /// context the loop is in.
  CallContext *contextOf(Value value) {
    CallContext *c = ctx;
    while (c && c->loop &&
           !c->loop->getRegion(0).isAncestor(value.getParentRegion()))
      c = c->owner;
    return c;
  }

  /// If `value` is the result of a call whose callee has a single return,
  /// the context evaluating that callee and the returned value backing it.
  /// Calls met while evaluating a callee get nested contexts on demand.
  std::optional<std::pair<CallContext *, Value>> callResult(Value value) {
    if (ctx && ctx->loop)
      if (CallContext *owner = contextOf(value); owner != ctx) {
        CallContext *saved = ctx;
        ctx = owner;
        auto result = callResult(value);
        ctx = saved;
        return result;
      }
    auto &results = ctx ? ctx->results : callResults;
    if (auto it = results.find(value); it != results.end())
      return it->second;
    if (!ctx || ctx->depth >= kMaxNesting)
      return std::nullopt;
    auto call = value.getDefiningOp<CallOp>();
    std::optional<std::pair<FuncOp, HLCF::ReturnOp>> callee =
        call ? singleReturnCallee(call) : std::nullopt;
    if (!callee)
      return std::nullopt;
    CallContext &nested = contexts.emplace_back();
    nested.callee = calleeName(call);
    nested.call = call;
    nested.parent = ctx;
    nested.depth = ctx->depth + 1;
    Block &entry = callee->first->getRegion(0).front();
    for (auto [arg, operand] :
         llvm::zip(entry.getArguments(), call->getOperands()))
      nested.args[arg] = operand;
    for (auto [result, returned] :
         llvm::zip(call->getResults(), callee->second->getOperands()))
      results[result] = {&nested, returned};
    // Memory allocated in a callee is apart from all other allocations too.
    if (isAllocation(call))
      noteAllocation(call);
    return results.lookup(value);
  }

  static constexpr unsigned kMaxNesting = 4;

  /// The context evaluating the callee of `call` (which must have a single
  /// return), created on demand like the ones backing call results; null if
  /// there is none or the nesting limit is reached.
  CallContext *contextFor(CallOp call) {
    auto &map = ctx ? ctx->calls : callContexts;
    if (auto it = map.find(call); it != map.end())
      return it->second;
    CallContext *result = nullptr;
    if (call->getNumResults())
      if (auto res = callResult(call->getResult(0)))
        result = res->first;
    if (!result && ctx && ctx->depth < kMaxNesting)
      if (auto callee = singleReturnCallee(call)) {
        CallContext &nested = contexts.emplace_back();
        nested.callee = calleeName(call);
        nested.call = call;
        nested.parent = ctx;
        nested.depth = ctx->depth + 1;
        Block &entry = callee->first->getRegion(0).front();
        for (auto [arg, operand] :
             llvm::zip(entry.getArguments(), call->getOperands()))
          nested.args[arg] = operand;
        result = &nested;
      }
    map[call] = result;
    return result;
  }

  static std::string calleeName(CallOp call) {
    if (auto callee = dyn_cast<SymbolConstantAttr>(call.getCallee()))
      return callee.getSymbol()
          .getRootReference()
          .getValue()
          .split('(')
          .first.str();
    return "<callee>";
  }

  /// The callee of `call` if it has a body with exactly one return, at the end
  /// of its entry block, matching the call's operands and results.
  std::optional<std::pair<FuncOp, HLCF::ReturnOp>>
  singleReturnCallee(CallOp call) const {
    auto callee = dyn_cast<SymbolConstantAttr>(call.getCallee());
    if (!callee || !symbols)
      return std::nullopt;
    auto fn = symbols->lookup<FuncOp>(callee.getSymbol().getRootReference());
    if (!fn || fn == self || fn->getRegion(0).empty())
      return std::nullopt;
    Block &entry = fn->getRegion(0).front();
    auto ret = dyn_cast<HLCF::ReturnOp>(entry.getTerminator());
    if (!ret || entry.getNumArguments() != call->getNumOperands() ||
        ret->getNumOperands() != call->getNumResults())
      return std::nullopt;
    bool otherReturn =
        fn->walk([&](HLCF::ReturnOp r) {
            return r == ret ? WalkResult::advance() : WalkResult::interrupt();
          }).wasInterrupted();
    if (otherReturn)
      return std::nullopt;
    return std::make_pair(fn, ret);
  }

  DenseMap<Value, std::string> &termMap() { return ctx ? ctx->terms : terms; }
  std::map<std::pair<void *, std::string>, std::string> &pathMap() {
    return ctx ? ctx->paths : paths;
  }
  DenseMap<std::pair<Value, unsigned>, std::string> &extractMap() {
    return ctx ? ctx->extracts : extracts;
  }

  template <typename F>
  auto inContext(CallContext *context, F &&fn) {
    CallContext *saved = ctx;
    ctx = context;
    auto result = fn();
    ctx = saved;
    return result;
  }

  /// Loops whose invariants hold at the current point: the enclosing ones and
  /// the ones that completed before it.
  SmallVector<LoopInfo *> assumedLoops() const {
    SmallVector<LoopInfo *> result(loopStack.begin(), loopStack.end());
    result.append(completedLoops.begin(), completedLoops.end());
    return result;
  }

  std::string fresh(StringRef prefix) {
    return (prefix + Twine(counter++)).str();
  }

  /// The expression of every defined name, for `evalConst`.
  llvm::StringMap<std::string> definitions;
  llvm::StringMap<std::optional<APInt>> constants;

  /// The value of a term that is a constant (e.g. a loop counter in an
  /// unrolled iteration), through definitions: bit-vector literals and
  /// arithmetic, comparisons, Boolean connectives and `ite`. Booleans are
  /// 1-bit values. Nullopt if it is not constant or not understood.
  std::optional<APInt> evalConst(StringRef term) {
    size_t pos = 0;
    return evalAt(term, pos, 0);
  }

  static StringRef token(StringRef s, size_t &pos) {
    while (pos < s.size() && s[pos] == ' ')
      ++pos;
    size_t start = pos;
    while (pos < s.size() && s[pos] != ' ' && s[pos] != '(' && s[pos] != ')')
      ++pos;
    return s.slice(start, pos);
  }

  /// Skips one term starting at `pos`.
  static void skipTerm(StringRef s, size_t &pos) {
    while (pos < s.size() && s[pos] == ' ')
      ++pos;
    if (pos >= s.size())
      return;
    if (s[pos] != '(') {
      token(s, pos);
      return;
    }
    unsigned open = 0;
    do {
      if (s[pos] == '(')
        ++open;
      else if (s[pos] == ')')
        --open;
      ++pos;
    } while (pos < s.size() && open);
  }

  std::optional<APInt> evalAt(StringRef s, size_t &pos, unsigned depth) {
    while (pos < s.size() && s[pos] == ' ')
      ++pos;
    if (pos >= s.size() || depth > 256)
      return std::nullopt;
    if (s[pos] != '(') {
      StringRef name = token(s, pos);
      if (name == "true" || name == "false")
        return APInt(1, name == "true");
      if (auto it = constants.find(name); it != constants.end())
        return it->second;
      auto def = definitions.find(name);
      if (def == definitions.end())
        return std::nullopt;
      std::string expr = def->second;
      size_t inner = 0;
      std::optional<APInt> value = evalAt(expr, inner, depth + 1);
      constants[name] = value;
      return value;
    }
    ++pos; // '('
    auto close = [&](std::optional<APInt> v) -> std::optional<APInt> {
      while (pos < s.size() && s[pos] != ')')
        skipTerm(s, pos);
      ++pos;
      return v;
    };
    while (pos < s.size() && s[pos] == ' ')
      ++pos;
    // Indexed operators: `((_ extract hi lo) x)` and the extensions.
    if (s[pos] == '(') {
      ++pos;
      token(s, pos); // "_"
      StringRef op = token(s, pos);
      unsigned a = 0, b = 0;
      token(s, pos).getAsInteger(10, a);
      if (op == "extract")
        token(s, pos).getAsInteger(10, b);
      while (pos < s.size() && s[pos] != ')')
        ++pos;
      ++pos;
      std::optional<APInt> x = evalAt(s, pos, depth + 1);
      if (!x)
        return close(std::nullopt);
      if (op == "extract")
        return close(x->extractBits(a - b + 1, b));
      if (op == "zero_extend")
        return close(x->zext(x->getBitWidth() + a));
      if (op == "sign_extend")
        return close(x->sext(x->getBitWidth() + a));
      return close(std::nullopt);
    }
    StringRef op = token(s, pos);
    if (op == "_") {
      // `(_ bvN W)`.
      StringRef digits = token(s, pos).drop_front(2);
      unsigned width = 0;
      token(s, pos).getAsInteger(10, width);
      if (!width)
        return close(std::nullopt);
      return close(APInt(width, digits, 10));
    }
    if (op == "ite") {
      std::optional<APInt> c = evalAt(s, pos, depth + 1);
      if (!c)
        return close(std::nullopt);
      if (c->isOne()) {
        std::optional<APInt> v = evalAt(s, pos, depth + 1);
        return close(v);
      }
      skipTerm(s, pos);
      return close(evalAt(s, pos, depth + 1));
    }
    SmallVector<std::optional<APInt>> args;
    while (true) {
      while (pos < s.size() && s[pos] == ' ')
        ++pos;
      if (pos >= s.size() || s[pos] == ')')
        break;
      args.push_back(evalAt(s, pos, depth + 1));
    }
    ++pos;
    auto known = [&] {
      return llvm::all_of(args, [](auto &a) { return a.has_value(); });
    };
    if (op == "and" || op == "or") {
      bool isAnd = op == "and";
      bool unknown = false;
      for (auto &a : args) {
        if (!a)
          unknown = true;
        else if (a->isOne() != isAnd)
          return APInt(1, !isAnd);
      }
      if (unknown)
        return std::nullopt;
      return APInt(1, isAnd);
    }
    if (!known() || args.empty())
      return std::nullopt;
    const APInt &x = *args[0];
    auto boolean = [](bool b) { return APInt(1, b); };
    if (op == "not")
      return boolean(x.isZero());
    if (op == "bvneg")
      return -x;
    if (args.size() != 2 || x.getBitWidth() != args[1]->getBitWidth())
      return std::nullopt;
    const APInt &y = *args[1];
    if (op == "=")
      return boolean(x == y);
    if (op == "distinct")
      return boolean(x != y);
    if (op == "=>")
      return boolean(x.isZero() || y.isOne());
    if (op == "xor")
      return boolean(x != y);
    if (op == "bvadd")
      return x + y;
    if (op == "bvsub")
      return x - y;
    if (op == "bvmul")
      return x * y;
    if (op == "bvand")
      return x & y;
    if (op == "bvor")
      return x | y;
    if (op == "bvxor")
      return x ^ y;
    if (op == "bvslt")
      return boolean(x.slt(y));
    if (op == "bvsle")
      return boolean(x.sle(y));
    if (op == "bvsgt")
      return boolean(x.sgt(y));
    if (op == "bvsge")
      return boolean(x.sge(y));
    if (op == "bvult")
      return boolean(x.ult(y));
    if (op == "bvule")
      return boolean(x.ule(y));
    if (op == "bvugt")
      return boolean(x.ugt(y));
    if (op == "bvuge")
      return boolean(x.uge(y));
    return std::nullopt;
  }

  std::string declare(Sort sort, StringRef prefix = "h") {
    std::string name = fresh(prefix);
    prelude += "(declare-const " + name + " " + sort.str() + ")\n";
    return name;
  }

  std::string define(Sort sort, StringRef expr, StringRef prefix = "v") {
    std::string name = fresh(prefix);
    definitions[name] = expr.str();
    prelude +=
        ("(define-fun " + name + " () " + sort.str() + " " + expr + ")\n")
            .str();
    if (annotate)
      deps[name] = namesIn(expr);
    return name;
  }

  /// Name a Bool reachability condition (keeps formulas small).
  std::string reachName(StringRef expr) {
    if (expr == "true" || expr == "false" || expr.starts_with("r"))
      return expr.str();
    return define({Sort::Bool, 1, false}, expr, "r");
  }

  /// Only for definitions: constraints on fresh values that every other
  /// assignment can satisfy. Facts go through `addAssumption`.
  void assertGlobal(StringRef expr) {
    prelude += ("(assert " + expr + ")\n").str();
  }

  /// The conjunction of assumptions (`kgen.assume`, callee `ensures`) made so
  /// far, in program order. A check may only use the assumptions made before
  /// it: a fact that holds later (e.g. an invariant restored after a check)
  /// must not justify an earlier obligation.
  std::string assumed = "true";

  void addAssumption(StringRef reach, StringRef cond) {
    assumed =
        define({Sort::Bool, 1, false},
               mkAnd(assumed, ("(=> " + reach + " " + cond + ")").str()), "a");
  }

  /// The term of an encodable value. Values without a definition (block
  /// arguments, loads, calls, unmodeled ops) become unconstrained constants.
  MaybeTerm term(Value value) {
    Sort sort = sortOf(value.getType());
    if (sort.kind == Sort::None)
      return std::nullopt;
    if (ctx && ctx->loop)
      if (CallContext *owner = contextOf(value); owner != ctx)
        return inContext(owner, [&] { return term(value); });
    if (ctx) {
      // A callee argument is the caller's operand.
      auto arg = ctx->args.find(value);
      if (arg != ctx->args.end())
        return inContext(ctx->parent, [&] { return term(arg->second); });
    }
    // A call result backed by the callee's returned value.
    if (!termMap().count(value))
      if (auto res = callResult(value)) {
        auto [context, returned] = *res;
        MaybeTerm t = inContext(context, [&] { return term(returned); });
        if (t)
          termMap()[value] = *t;
        return t;
      }
    auto &map = termMap();
    auto it = map.find(value);
    if (it != map.end())
      return it->second;
    // Callee values are computed on demand; the function's own values are
    // encoded in program order.
    if (ctx)
      if (Operation *def = value.getDefiningOp()) {
        if (auto ifOp = dyn_cast<HLCF::IfOp>(def))
          encodeIfResultOnDemand(ifOp, cast<OpResult>(value));
        else if (auto loop = dyn_cast<HLCF::LoopOp>(def)) {
          if (MaybeTerm t = loopResultTerm(loop, cast<OpResult>(value)))
            map[value] = *t;
        } else
          encodeOp(def);
        auto it = map.find(value);
        if (it != map.end())
          return it->second;
      }
    std::string name = declare(sort);
    map[value] = name;
    noteUnknown(name, describe(value));
    // A string literal's address is static data, outside every allocation
    // (whose regions count from 1).
    if (Operation *def = value.getDefiningOp())
      if (def->getName().getStringRef() == "pop.string.address")
        assertGlobal("(= " + region(name) + " " + bvConst(APInt(64, 0)) + ")");
    return name;
  }

  std::string boolTerm(Value value) {
    MaybeTerm t = term(value);
    if (t && sortOf(value.getType()).kind == Sort::Bool)
      return *t;
    return declare({Sort::Bool, 1, false});
  }

  /// On demand (in a callee context), an if/elif/else whose arms all yield is
  /// a chain of `ite`s over the arms' conditions; conditions and yielded
  /// values are themselves computed on demand. Anything else inside the arms
  /// is ignored, which only loses information.
  void encodeIfResultOnDemand(HLCF::IfOp ifOp, OpResult result) {
    unsigned idx = result.getResultNumber();
    auto yielded = [&](Block &block) -> MaybeTerm {
      auto yield = dyn_cast<HLCF::YieldOp>(block.getTerminator());
      if (!yield || idx >= yield->getNumOperands())
        return std::nullopt;
      return term(yield->getOperand(idx));
    };
    // Arms in order: (condition, value); the else value comes last.
    SmallVector<std::pair<std::string, std::string>> arms;
    MaybeTerm thenValue = yielded(ifOp.getThenBlock());
    if (!thenValue)
      return;
    arms.push_back({boolTerm(ifOp.getCond()), *thenValue});
    auto elifs = ifOp.getElifRegions();
    for (unsigned i = 0; i + 1 < elifs.size(); i += 2) {
      Operation *condYield = elifs[i].front().getTerminator();
      if (!isa<HLCF::IfElifCondYieldOp>(condYield) ||
          !condYield->getNumOperands())
        return;
      MaybeTerm value = yielded(elifs[i + 1].front());
      if (!value)
        return;
      arms.push_back({boolTerm(condYield->getOperand(0)), *value});
    }
    MaybeTerm expr = yielded(ifOp.getElseBlock());
    if (!expr)
      return;
    for (auto &[cond, value] : llvm::reverse(arms))
      expr = "(ite " + cond + " " + value + " " + *expr + ")";
    setTerm(result, *expr);
  }

  void setTerm(Value value, StringRef expr) {
    std::string name = define(sortOf(value.getType()), expr);
    termMap()[value] = name;
    if (annotate)
      prelude += "; " + name + ": " + describe(value) + "\n";
  }

  //===--------------------------------------------------------------------===//
  // Straight-line ops
  //===--------------------------------------------------------------------===//

  /// The value of a scalar integer or boolean constant, as `width` bits.
  static std::optional<APInt> constantInt(Value value, unsigned width) {
    Attribute attr;
    if (!mlir::matchPattern(value, mlir::m_Constant(&attr)))
      return std::nullopt;
    return constantInt(attr, width);
  }

  static std::optional<APInt> constantInt(Attribute attr, unsigned width) {
    if (auto simd = dyn_cast<KGEN::SIMDAttr>(attr)) {
      if (simd.getValues().size() != 1)
        return std::nullopt;
      const DTypeValue &v = simd.getValues().front();
      KGENDType dtype = v.getDType();
      if (dtype.isBool())
        return APInt(width, v.getBoolVal() ? 1 : 0);
      if (dtype.isIndex() || dtype.isUIndex())
        return APInt(64, v.getIndexVal(), /*isSigned=*/true).sextOrTrunc(width);
      if (dtype.isSInt() || dtype.isUInt())
        return APInt(v.getIntVal().extOrTrunc(width));
      return std::nullopt;
    }
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      return intAttr.getValue().sextOrTrunc(width);
    if (auto boolAttr = dyn_cast<BoolAttr>(attr))
      return APInt(width, boolAttr.getValue() ? 1 : 0);
    return std::nullopt;
  }

  MaybeTerm constantTerm(Value value, Sort sort) {
    Attribute attr;
    if (!mlir::matchPattern(value, mlir::m_Constant(&attr)))
      return std::nullopt;
    return constantTerm(attr, sort);
  }

  static MaybeTerm constantTerm(Attribute attr, Sort sort) {
    std::optional<APInt> c = constantInt(attr, std::max(sort.width, 1u));
    if (!c)
      return std::nullopt;
    if (sort.kind == Sort::Bool)
      return std::string(c->isZero() ? "false" : "true");
    return bvConst(*c);
  }

  /// Bits of `input` reinterpreted/converted to `to` (int<->int, int<->bool).
  MaybeTerm convert(Value input, Sort to) {
    Sort from = sortOf(input.getType());
    MaybeTerm in = term(input);
    if (!in || from.kind == Sort::None || to.kind == Sort::None)
      return std::nullopt;
    if (from.kind == Sort::Bool && to.kind == Sort::Bool)
      return in;
    if (from.kind == Sort::Bool)
      return "(ite " + *in + " " + bvConst(APInt(to.width, 1)) + " " +
             bvConst(APInt(to.width, 0)) + ")";
    if (to.kind == Sort::Bool)
      return "(distinct " + *in + " " + bvConst(APInt(from.width, 0)) + ")";
    if (from.width == to.width)
      return in;
    if (from.width < to.width)
      return std::string(from.isSigned ? "((_ sign_extend "
                                       : "((_ zero_extend ") +
             std::to_string(to.width - from.width) + ") " + *in + ")";
    return "((_ extract " + std::to_string(to.width - 1) + " 0) " + *in + ")";
  }

  static std::optional<std::string> cmpOp(KGEN::CmpPredicate pred,
                                          bool isSigned) {
    switch (pred) {
    case CmpPredicate::EQ:
      return "=";
    case CmpPredicate::NE:
      return "distinct";
    case CmpPredicate::LT:
      return isSigned ? "bvslt" : "bvult";
    case CmpPredicate::LE:
      return isSigned ? "bvsle" : "bvule";
    case CmpPredicate::GT:
      return isSigned ? "bvsgt" : "bvugt";
    case CmpPredicate::GE:
      return isSigned ? "bvsge" : "bvuge";
    }
    return std::nullopt;
  }

  static std::string indexCmpOp(mlir::index::IndexCmpPredicate pred) {
    using P = mlir::index::IndexCmpPredicate;
    switch (pred) {
    case P::EQ:
      return "=";
    case P::NE:
      return "distinct";
    case P::SLT:
      return "bvslt";
    case P::SLE:
      return "bvsle";
    case P::SGT:
      return "bvsgt";
    case P::SGE:
      return "bvsge";
    case P::ULT:
      return "bvult";
    case P::ULE:
      return "bvule";
    case P::UGT:
      return "bvugt";
    case P::UGE:
      return "bvuge";
    }
    llvm_unreachable("unknown index predicate");
  }

  static constexpr int kUnwrap = -1;
  /// Access resolution may cross several call contexts, merges and views.
  static constexpr unsigned kMaxAccessDepth = 64;

  /// A place in a stack allocation: the allocation and the constant
  /// `kgen.struct.gep` field indices leading to it. Pointer and union views
  /// do not change the place.
  struct MemLoc {
    Value slot;
    SmallVector<int> path;
  };

  static std::optional<MemLoc> memLocation(Value ptr) {
    SmallVector<int> reversed;
    while (Operation *def = ptr.getDefiningOp()) {
      if (auto cast = dyn_cast<POP::PointerBitcastOp>(def)) {
        ptr = cast.getInput();
      } else if (auto view = dyn_cast<POP::UnionBitcastOp>(def)) {
        ptr = view.getValue();
      } else if (auto gep = dyn_cast<StructGEPOp>(def)) {
        auto index = dyn_cast<IntegerAttr>(gep.getIndexAttr());
        if (!index)
          return std::nullopt;
        reversed.push_back(index.getInt());
        ptr = gep.getContainer();
      } else {
        break;
      }
    }
    if (!ptr.getDefiningOp<POP::StackAllocationOp>() && !isEntryArgument(ptr))
      return std::nullopt;
    return MemLoc{ptr, SmallVector<int>(llvm::reverse(reversed))};
  }

  /// A pointer argument of a function's entry block: a place in the caller's
  /// memory. Two such arguments may alias each other, but not a local stack
  /// slot.
  static bool isEntryArgument(Value ptr) {
    auto arg = dyn_cast<BlockArgument>(ptr);
    return arg && isa<FuncOp>(arg.getOwner()->getParentOp()) &&
           arg.getOwner()->isEntryBlock();
  }

  /// The convention the callee of `call` declares for operand `index`.
  static std::optional<ArgConvention> argConvention(CallOp call,
                                                    unsigned index) {
    ArrayRef<ArgConvention> convs =
        call.getCalleeType().getBody().getArgConventions();
    if (index >= convs.size())
      return std::nullopt;
    return convs[index];
  }

  /// A function entry argument passed `imm_mem`: memory the caller lends
  /// read-only, which nothing may mutate while the function runs (the
  /// language forbids writes through it, and the caller's exclusive
  /// ownership forbids writes through other aliases for the call). Its places
  /// hold their entry values throughout the body. Trusted, like other
  /// language guarantees: unsafe code that casts the origin away breaks it.
  static bool isImmutableArgument(Value ptr) {
    if (!isEntryArgument(ptr))
      return false;
    auto arg = cast<BlockArgument>(ptr);
    auto func = cast<FuncOp>(arg.getOwner()->getParentOp());
    ArrayRef<ArgConvention> convs =
        func.getFuncTypeGenerator().getBody().getArgConventions();
    return arg.getArgNumber() < convs.size() &&
           convs[arg.getArgNumber()] == ArgConvention::ImmMem;
  }

  /// Whether stores to `a` and `b` (bases of places) may alias.
  static bool mayAlias(Value a, Value b) {
    return a == b || (isEntryArgument(a) && isEntryArgument(b));
  }

  static Value stackSlot(Value ptr) {
    std::optional<MemLoc> loc = memLocation(ptr);
    return loc ? loc->slot : Value();
  }

  /// Whether `prefix` is a prefix of `path`.
  static bool isPrefix(ArrayRef<int> prefix, ArrayRef<int> path) {
    return prefix.size() <= path.size() &&
           prefix == path.take_front(prefix.size());
  }

  /// The uses through which the address of a stack slot escapes: anything
  /// but plain loads, stores (as the address), views and field addresses of
  /// it, lifetime markers and `imm_mem` call operands (see
  /// `isImmutableArgument`: the callee only reads the slot, and any pointer it
  /// derives from it is immutable too).
  static SmallVector<Operation *> escapingUses(Value slot) {
    SmallVector<Operation *> escapes;
    SmallVector<Value> worklist = {slot};
    while (!worklist.empty()) {
      Value ptr = worklist.pop_back_val();
      for (OpOperand &use : ptr.getUses()) {
        Operation *user = use.getOwner();
        if (isa<POP::LoadOp>(user))
          continue;
        if (auto store = dyn_cast<POP::StoreOp>(user)) {
          if (&use != &store.getPtrMutable())
            escapes.push_back(user); // The address itself is stored.
          continue;
        }
        if (isa<POP::PointerBitcastOp, POP::UnionBitcastOp, StructGEPOp>(
                user)) {
          worklist.push_back(user->getResult(0));
          continue;
        }
        StringRef name = user->getName().getStringRef();
        if (name == "pop.stack_alloc.lifetime.start" ||
            name == "pop.stack_alloc.lifetime.end")
          continue;
        if (auto call = dyn_cast<CallOp>(user))
          if (argConvention(call, use.getOperandNumber()) ==
              ArgConvention::ImmMem)
            continue;
        escapes.push_back(user);
      }
    }
    return escapes;
  }

  /// Whether `later` certainly executes after `point` whenever both execute:
  /// their ancestors in a common block are ordered that way and no loop
  /// encloses that block (an iteration's `later` could precede the next
  /// iteration's `point`).
  static bool happensAfter(Operation *later, Operation *point) {
    DenseMap<Block *, Operation *> pointAncestors;
    for (Operation *op = point; op && !isa<FuncOp>(op); op = op->getParentOp())
      pointAncestors[op->getBlock()] = op;
    for (Operation *op = later; op && !isa<FuncOp>(op);
         op = op->getParentOp()) {
      auto it = pointAncestors.find(op->getBlock());
      if (it == pointAncestors.end())
        continue;
      if (it->second == op || !it->second->isBeforeInBlock(op))
        return false;
      for (Operation *anc = op->getParentOp(); anc && !isa<FuncOp>(anc);
           anc = anc->getParentOp())
        if (isa<HLCF::LoopOp>(anc))
          return false;
      return true;
    }
    return false;
  }

  /// Whether nothing but direct stores can have changed the slot before
  /// `point`: its address does not escape, or only escapes after `point`
  /// (e.g. into a closure created later), so no call or other pointer can
  /// write it before then.
  static bool isNonEscapingBefore(Value slot, Operation *point) {
    if (isEntryArgument(slot))
      return false; // The caller may have let it escape.
    // The escape at `point` itself (e.g. the call whose operand the value
    // is read for) has not happened yet either.
    return llvm::all_of(escapingUses(slot), [&](Operation *escape) {
      return escape == point || happensAfter(escape, point);
    });
  }

  /// Whether `op` (including nested ops) may write the place `loc`.
  static bool mayWrite(Operation *op, const MemLoc &loc, bool nonEscaping) {
    return op
        ->walk([&](Operation *nested) {
          if (auto store = dyn_cast<POP::StoreOp>(nested)) {
            std::optional<MemLoc> target = memLocation(store.getPtr());
            if (!target)
              return nonEscaping ? WalkResult::advance()
                                 : WalkResult::interrupt();
            if (target->slot == loc.slot && (isPrefix(target->path, loc.path) ||
                                             isPrefix(loc.path, target->path)))
              return WalkResult::interrupt();
            if (target->slot != loc.slot && mayAlias(target->slot, loc.slot))
              return WalkResult::interrupt();
            return WalkResult::advance();
          }
          // Lifetime markers write nothing: reading a place outside its
          // lifetime is undefined behavior, which the analysis excludes.
          StringRef name = nested->getName().getStringRef();
          if (name == "pop.stack_alloc.lifetime.start" ||
              name == "pop.stack_alloc.lifetime.end")
            return WalkResult::advance();
          if (nonEscaping || nested->getNumRegions() ||
              isa<POP::LoadOp, POP::StackAllocationOp, ObligationOp, AssumeOp,
                  CopyMarkerOp>(nested) ||
              mlir::isMemoryEffectFree(nested))
            return WalkResult::advance();
          return WalkResult::interrupt();
        })
        .wasInterrupted();
  }

  /// A load being forwarded: the place it reads, the type it loads and the
  /// access applied to the loaded value.
  struct Place {
    MemLoc loc;
    Type loaded;
    SmallVector<int> access;
    Sort sort;
    bool nonEscaping;
    Operation *slotDef;
  };

  enum class Search { Found, NotWritten, Fail };
  static constexpr unsigned kMaxMergeDepth = 32;

  /// Store-to-load forwarding for stack slots: the term for the value a load
  /// reads (with `access` applied), built from the stores before it, or
  /// nullopt. Places are compared by field path: a store of an enclosing
  /// struct forwards the loaded field of it, a store to a disjoint field does
  /// not interfere, and a partial write of the loaded place stops the search.
  /// Stored and loaded types may differ by a union view: storing a member and
  /// loading the union wraps it, the reverse unwraps it.
  ///
  /// The search walks backwards through the load's block and out through
  /// enclosing `if`s; it leaves a loop only if the loop never writes the place
  /// (a later iteration could have). An `if` that writes the place merges its
  /// arms: the value after it is an `ite` over the arms' conditions of each
  /// arm's value at its end (its last store, or the value before the `if` if
  /// the arm does not write the place). For a slot whose address escapes, any
  /// op that may write memory through an unknown pointer stops the search; for
  /// a non-escaping slot only stores to it do.
  MaybeTerm loadTerm(Operation *load, ArrayRef<int> access, Sort sort) {
    auto loadOp = dyn_cast<POP::LoadOp>(load);
    if (!loadOp)
      return std::nullopt;
    std::optional<MemLoc> loc = memLocation(loadOp.getPtr());
    if (!loc)
      return heapLoadTerm(loadOp, access, sort);
    // Loading a struct and taking a field of it reads the same as loading
    // that field: move leading field accesses into the place, so stores that
    // built the struct field by field are found.
    MemLoc place0 = *loc;
    Type loaded = loadOp.getResult().getType();
    size_t fields = 0;
    for (; fields < access.size() && access[fields] >= 0; ++fields) {
      auto structType = dyn_cast<StructType>(loaded);
      std::optional<SmallVector<Type>> elements =
          structType ? structType.getElementTypes() : std::nullopt;
      if (!elements || size_t(access[fields]) >= elements->size())
        break;
      loaded = (*elements)[access[fields]];
      place0.path.push_back(access[fields]);
    }
    Place place{place0,
                loaded,
                SmallVector<int>(access.drop_front(fields)),
                sort,
                isNonEscapingBefore(loc->slot, load),
                loc->slot.getDefiningOp()};
    return valueBefore(place, load, 0);
  }

  /// The value of the place just before `op`.
  MaybeTerm valueBefore(const Place &place, Operation *op, unsigned depth) {
    if (isImmutableArgument(place.loc.slot))
      return valueAtEntry(place, depth);
    Operation *cur = op;
    while (true) {
      auto [kind, found] = searchBlock(place, *cur->getBlock(), cur, depth);
      if (kind == Search::Found)
        return found;
      if (kind == Search::Fail)
        return std::nullopt;
      // Continue in the enclosing block, before the op containing `cur`.
      Operation *parent = cur->getParentOp();
      if (parent && isa<FuncOp>(parent))
        return valueAtCall(place, depth);
      if (!parent)
        return std::nullopt;
      if (auto ifOp = dyn_cast<HLCF::IfOp>(parent)) {
        if (elifConditionsMayWrite(place, ifOp))
          return std::nullopt;
      } else if (mayWrite(parent, place.loc, place.nonEscaping)) {
        return std::nullopt;
      }
      cur = parent;
    }
  }

  /// Reaching the entry of a callee evaluated for a call: a place in the
  /// callee's argument memory holds what the caller's memory held at the
  /// call, so continue the search in the caller, before the call.
  MaybeTerm valueAtCall(const Place &place, unsigned depth) {
    if (!ctx || !ctx->call || !isEntryArgument(place.loc.slot))
      return std::nullopt;
    auto arg = ctx->args.find(place.loc.slot);
    if (arg == ctx->args.end())
      return std::nullopt;
    std::optional<MemLoc> callerLoc = memLocation(arg->second);
    if (!callerLoc)
      return std::nullopt;
    Operation *call = ctx->call;
    Place callerPlace = place;
    callerPlace.loc.slot = callerLoc->slot;
    callerPlace.loc.path = callerLoc->path;
    callerPlace.loc.path.append(place.loc.path.begin(), place.loc.path.end());
    callerPlace.nonEscaping = isNonEscapingBefore(callerLoc->slot, call);
    callerPlace.slotDef = callerLoc->slot.getDefiningOp();
    return inContext(ctx->parent,
                     [&] { return valueBefore(callerPlace, call, depth + 1); });
  }

  /// The value of a place in an `imm_mem` argument, which is the same at
  /// every point of the body: the caller's value at the call when evaluating
  /// one, else one unknown shared by all reads of the place.
  MaybeTerm valueAtEntry(const Place &place, unsigned depth) {
    if (MaybeTerm t = valueAtCall(place, depth))
      return t;
    std::string key = "entry.", text;
    for (int step : place.loc.path) {
      key += std::to_string(step) + ".";
      text += " ." + std::to_string(step);
    }
    key += "|";
    for (int step : place.access) {
      key += (step == kUnwrap ? "u" : std::to_string(step)) + ".";
      text += step == kUnwrap ? " unwrap" : " ." + std::to_string(step);
    }
    // Loads of different types from one place are different reads.
    llvm::raw_string_ostream(key) << "|" << place.loaded.getAsOpaquePointer();
    auto [it, inserted] =
        pathMap().try_emplace({place.loc.slot.getAsOpaquePointer(), key}, "");
    if (inserted) {
      it->second = declare(place.sort);
      noteUnknown(it->second, "read-only argument memory" + text + " of " +
                                  describe(place.loc.slot));
    }
    return it->second;
  }

  /// Search `block` backwards from just before `from` (from its end if null)
  /// for the value of the place.
  std::pair<Search, MaybeTerm> searchBlock(const Place &place, Block &block,
                                           Operation *from, unsigned depth) {
    Operation *op =
        from ? from->getPrevNode() : (block.empty() ? nullptr : &block.back());
    for (; op; op = op->getPrevNode()) {
      if (op == place.slotDef) {
        // Nothing stored yet: an uninitialized read, whose value is simply
        // arbitrary (e.g. a union member for a tag that never occurs).
        std::string name = declare(place.sort);
        noteUnknown(name, "uninitialized read of " + describe(place.loc.slot));
        return {Search::Found, name};
      }
      if (auto store = dyn_cast<POP::StoreOp>(op)) {
        std::optional<MemLoc> target = memLocation(store.getPtr());
        if (!target) {
          if (place.nonEscaping)
            continue;
          return {Search::Fail, std::nullopt}; // May alias the place.
        }
        if (target->slot != place.loc.slot) {
          if (mayAlias(target->slot, place.loc.slot))
            return {Search::Fail, std::nullopt};
          continue;
        }
        if (isPrefix(target->path, place.loc.path)) {
          MaybeTerm t = fromStore(place, store, *target);
          return {t ? Search::Found : Search::Fail, t};
        }
        if (isPrefix(place.loc.path, target->path))
          return {Search::Fail, std::nullopt}; // Partial write.
        continue;                              // A disjoint field.
      }
      if (!mayWrite(op, place.loc, place.nonEscaping))
        continue;
      if (auto ifOp = dyn_cast<HLCF::IfOp>(op);
          ifOp && depth < kMaxMergeDepth) {
        MaybeTerm t = mergeIf(place, ifOp, depth + 1);
        return {t ? Search::Found : Search::Fail, t};
      }
      return {Search::Fail, std::nullopt};
    }
    return {Search::NotWritten, std::nullopt};
  }

  /// Whether the condition regions of an `hlcf.if`'s elif arms may write the
  /// place (they run before the arms).
  bool elifConditionsMayWrite(const Place &place, HLCF::IfOp ifOp) {
    auto elifs = ifOp.getElifRegions();
    for (unsigned i = 0; i < elifs.size(); i += 2)
      for (Operation &op : elifs[i].front())
        if (mayWrite(&op, place.loc, place.nonEscaping))
          return true;
    return false;
  }

  /// Whether control leaves through the end of an `if` arm: by a jump, or by
  /// a call that never returns (e.g. a failed `debug_assert`).
  bool armLeaves(Block &block) {
    if (isa<HLCF::BreakOp, HLCF::ContinueOp, HLCF::ReturnOp,
            HLCF::UnreachableOp, LIT::TryRaiseOp>(block.getTerminator()))
      return true;
    return llvm::any_of(block, [&](Operation &op) {
      auto call = dyn_cast<CallOp>(&op);
      return call && neverReturns(call);
    });
  }

  /// Whether a call never returns: its callee has no `hlcf.return` (it
  /// traps, loops forever or ends in `hlcf.unreachable`), or its entry block
  /// unconditionally makes such a call. Decided on the elaborated body, so it
  /// follows compile-time choices (a `debug_assert` in "warn" mode returns).
  bool neverReturns(CallOp call, unsigned depth = 0) {
    auto callee = dyn_cast<SymbolConstantAttr>(call.getCallee());
    if (!callee || !symbols || depth > kMaxNesting)
      return false;
    auto fn = symbols->lookup<FuncOp>(callee.getSymbol().getRootReference());
    if (!fn || fn->getRegion(0).empty())
      return false;
    if (auto it = noReturn.find(fn); it != noReturn.end())
      return it->second;
    noReturn[fn] = false; // Assumed while deciding (recursion).
    bool result = !fn->walk([](HLCF::ReturnOp) {
                       return WalkResult::interrupt();
                     }).wasInterrupted() ||
                  llvm::any_of(fn->getRegion(0).front(), [&](Operation &op) {
                    auto inner = dyn_cast<CallOp>(&op);
                    return inner && neverReturns(inner, depth + 1);
                  });
    noReturn[fn] = result;
    return result;
  }
  DenseMap<Operation *, bool> noReturn;

  /// The value of the place after an `hlcf.if` that writes it.
  MaybeTerm mergeIf(const Place &place, HLCF::IfOp ifOp, unsigned depth) {
    if (elifConditionsMayWrite(place, ifOp))
      return std::nullopt;
    std::optional<MaybeTerm> before; // Computed on first use.
    auto unknownIn = [&](StringRef why) {
      std::string name = declare(place.sort);
      std::string text;
      llvm::raw_string_ostream os(text);
      os << "stack slot value " << why << " of hlcf.if";
      if (std::optional<ObligationLocation> where = locationOf(ifOp.getLoc()))
        os << " at " << where->file << ":" << where->line << ":" << where->col;
      noteUnknown(name, text);
      return name;
    };
    auto armValue = [&](Block &block) -> std::string {
      Operation *terminator = block.getTerminator();
      // Control does not reach the load along an arm that leaves.
      if (armLeaves(block))
        return declare(place.sort);
      auto [kind, found] = searchBlock(place, block, terminator, depth);
      if (kind == Search::Found && found)
        return *found;
      if (kind == Search::NotWritten) {
        if (!before)
          before = valueBefore(place, ifOp, depth);
        if (*before)
          return **before;
        return unknownIn("before");
      }
      return unknownIn("in an arm");
    };
    SmallVector<std::pair<std::string, std::string>> arms;
    arms.push_back({boolTerm(ifOp.getCond()), armValue(ifOp.getThenBlock())});
    auto elifs = ifOp.getElifRegions();
    for (unsigned i = 0; i + 1 < elifs.size(); i += 2) {
      Operation *condYield = elifs[i].front().getTerminator();
      if (!isa<HLCF::IfElifCondYieldOp>(condYield) ||
          !condYield->getNumOperands())
        return std::nullopt;
      arms.push_back(
          {boolTerm(condYield->getOperand(0)), armValue(elifs[i + 1].front())});
    }
    std::string expr = armValue(ifOp.getElseBlock());
    for (auto &[cond, value] : llvm::reverse(arms))
      expr = "(ite " + cond + " " + value + " " + expr + ")";
    return expr;
  }

  /// The loaded value (with the place's access) given the store that last
  /// wrote the place or a place enclosing it.
  MaybeTerm fromStore(const Place &place, POP::StoreOp store,
                      const MemLoc &target) {
    Value stored = store.getArg();
    SmallVector<int> path(
        ArrayRef<int>(place.loc.path).drop_front(target.path.size()));
    ArrayRef<int> access = place.access;
    if (path.empty() && place.loaded != stored.getType()) {
      Type type = stored.getType();
      if (auto u = dyn_cast<POP::UnionType>(place.loaded);
          u && llvm::is_contained(u.getTypes(), type)) {
        // The load wraps the stored member; unwrapping it gives the member.
        if (access.empty() || access.front() != kUnwrap)
          return std::nullopt;
        access = access.drop_front();
      } else if (auto u = dyn_cast<POP::UnionType>(type);
                 u && llvm::is_contained(u.getTypes(), place.loaded)) {
        path.push_back(kUnwrap);
      } else {
        return std::nullopt;
      }
    }
    path.append(access.begin(), access.end());
    if (path.empty())
      return sortOf(stored.getType()) == place.sort ? term(stored)
                                                    : std::nullopt;
    return resolveAccess(stored, path, place.sort);
  }

  // Loop unrolling: the first iterations of a loop, evaluated like callees.
  // Values after a loop that exits within `kMaxUnroll` iterations are exact;
  // later exits leave them unknown.

  static constexpr unsigned kMaxUnroll = 8;

  /// A loop the unrolling handles: its body is one block ending in the only
  /// `hlcf.continue` for it, with one `hlcf.break` for it, reached through
  /// `hlcf.if` arms only (`exitPath`: their conditions and arms).
  struct LoopShape {
    HLCF::BreakOp exit;
    SmallVector<std::pair<Value, bool>> exitPath;
  };

  /// The loop a `hlcf.break` / `hlcf.continue` leaves or repeats.
  static Operation *targetLoop(Operation *jump, StringAttr label) {
    for (Operation *op = jump->getParentOp(); op; op = op->getParentOp())
      if (auto loop = dyn_cast<HLCF::LoopOp>(op))
        if (!label || loop.getLabelAttr() == label)
          return loop;
    return nullptr;
  }

  std::optional<LoopShape> loopShape(HLCF::LoopOp loop) {
    if (auto it = loopShapes.find(loop); it != loopShapes.end())
      return it->second;
    std::optional<LoopShape> &shape = loopShapes[loop];
    Region &body = loop->getRegion(0);
    if (!body.hasOneBlock() ||
        !isa<HLCF::ContinueOp>(body.front().getTerminator()))
      return std::nullopt;
    SmallVector<HLCF::BreakOp> breaks;
    unsigned continues = 0;
    body.walk([&](Operation *op) {
      if (auto jump = dyn_cast<HLCF::BreakOp>(op)) {
        if (targetLoop(op, jump.getLabelAttr()) == loop)
          breaks.push_back(jump);
      } else if (auto jump = dyn_cast<HLCF::ContinueOp>(op)) {
        if (targetLoop(op, jump.getLabelAttr()) == loop)
          ++continues;
      }
    });
    if (breaks.size() != 1 || continues != 1)
      return std::nullopt;
    LoopShape result{breaks.front(), {}};
    for (Operation *op = breaks.front(); op->getParentOp() != loop;) {
      Block *block = op->getBlock();
      auto ifOp = dyn_cast<HLCF::IfOp>(op->getParentOp());
      if (!ifOp || !ifOp.getElifRegions().empty())
        return std::nullopt;
      result.exitPath.push_back(
          {ifOp.getCond(), block == &ifOp.getThenBlock()});
      op = ifOp;
    }
    shape = result;
    return shape;
  }
  DenseMap<Operation *, std::optional<LoopShape>> loopShapes;

  /// The context evaluating iteration `k` of `loop` (in the current
  /// context), created on demand after the ones before it.
  CallContext *iterationContext(HLCF::LoopOp loop, unsigned k) {
    auto &map = ctx ? ctx->iterations : loopIterations;
    if (auto it = map.find({loop, k}); it != map.end())
      return it->second;
    CallContext *previous = k ? iterationContext(loop, k - 1) : nullptr;
    CallContext &iteration = contexts.emplace_back();
    std::string text;
    llvm::raw_string_ostream os(text);
    os << "iteration " << k << " of hlcf.loop";
    if (std::optional<ObligationLocation> where = locationOf(loop.getLoc()))
      os << " at " << where->file << ":" << where->line;
    iteration.callee = text;
    iteration.loop = loop;
    iteration.owner = ctx;
    iteration.iteration = k;
    iteration.parent = previous ? previous : ctx;
    iteration.depth = ctx ? ctx->depth : 0;
    Block &body = loop->getRegion(0).front();
    ValueRange carried =
        previous ? body.getTerminator()->getOperands() : loop->getOperands();
    for (auto [arg, value] : llvm::zip(body.getArguments(), carried))
      iteration.args[arg] = value;
    map[{loop, k}] = &iteration;
    return &iteration;
  }

  /// Whether iteration `k` leaves through the loop's `hlcf.break` (evaluated
  /// in the current context, which runs the loop).
  std::string exitsIn(HLCF::LoopOp loop, const LoopShape &shape, unsigned k) {
    CallContext *iteration = iterationContext(loop, k);
    return inContext(iteration, [&] {
      std::string cond = "true";
      for (auto &[value, arm] : shape.exitPath) {
        std::string c = boolTerm(value);
        cond = mkAnd(cond, arm ? c : mkNot(c));
      }
      return cond;
    });
  }

  /// The iterations that may be the one leaving the loop, in order, with
  /// their exit conditions: iterations whose exit condition is constant
  /// false are left out, and the list ends at one whose exit is certain
  /// (`certain` is then set; else later exits are possible).
  SmallVector<std::pair<unsigned, std::string>>
  exitingIterations(HLCF::LoopOp loop, const LoopShape &shape, bool &certain) {
    SmallVector<std::pair<unsigned, std::string>> result;
    certain = false;
    for (unsigned k = 0; k < kMaxUnroll; ++k) {
      std::string exits = exitsIn(loop, shape, k);
      std::optional<APInt> known = evalConst(exits);
      if (known && known->isZero())
        continue;
      result.push_back({k, exits});
      if (known) {
        certain = true;
        break;
      }
    }
    return result;
  }

  /// A loop result in a callee or iteration context: the `hlcf.break`
  /// operand of the iteration that exits.
  MaybeTerm loopResultTerm(HLCF::LoopOp loop, OpResult result) {
    std::optional<LoopShape> shape = loopShape(loop);
    Sort sort = sortOf(result.getType());
    if (!shape || sort.kind == Sort::None)
      return std::nullopt;
    bool certain;
    auto exits = exitingIterations(loop, *shape, certain);
    std::optional<std::string> expr;
    if (!certain) {
      expr = declare(sort);
      noteUnknown(*expr, "loop result after more than " +
                             std::to_string(kMaxUnroll) + " iterations of " +
                             describe(result));
    }
    for (auto &[k, cond] : llvm::reverse(exits)) {
      MaybeTerm value = inContext(iterationContext(loop, k), [&] {
        return term(shape->exit->getOperand(result.getResultNumber()));
      });
      if (!value)
        return std::nullopt;
      expr = expr ? "(ite " + cond + " " + *value + " " + *expr + ")" : *value;
    }
    if (!expr)
      return std::nullopt;
    return expr->front() == '(' ? define(sort, *expr) : *expr;
  }

  /// The loop's results in the function itself already have terms (bounded
  /// by the loop's invariants): tie them to the iteration that exits.
  void linkLoopResults(HLCF::LoopOp loop, const LoopShape &shape) {
    if (ctx || !linkedLoops.insert(loop).second)
      return;
    for (OpResult result : loop->getResults()) {
      MaybeTerm whole = term(result);
      if (!whole)
        continue;
      // Iterations after the exiting one never run: only the first exit
      // decides the results.
      std::string notYet = "true";
      bool certain;
      for (auto &[k, exits] : exitingIterations(loop, shape, certain)) {
        MaybeTerm value = inContext(iterationContext(loop, k), [&] {
          return term(shape.exit->getOperand(result.getResultNumber()));
        });
        if (value)
          assertGlobal("(=> " + mkAnd(notYet, exits) + " (= " + *whole + " " +
                       *value + "))");
        notYet = mkAnd(notYet, mkNot(exits));
      }
    }
  }
  DenseSet<Operation *> linkedLoops;

  // Heap memory: loads from anything but a local stack slot or an argument
  // place, e.g. the element of a list that holds lists. The value is built
  // lazily from the stores before the load, most recent first, each guarded
  // by "it wrote this address": `(ite (= addr store-addr) stored older)`.
  // Callees with a single return are searched through (their stores happened
  // before the call returned), `hlcf.if`s merge their arms, allocation and
  // freeing write nothing, and anything else that may write memory gives up.

  /// An address outside local stack slots: `index` elements of type
  /// `element` past the pointer `base` (both terms), then the field `path`
  /// inside that element. `root` is the pointer the address is derived from
  /// by offsets, views and field addresses of any type, which stay inside
  /// one allocation: it decides the region.
  struct HeapAddr {
    std::string base, index;
    Type element;
    SmallVector<int> path;
    std::string root;
  };

  /// A load from heap memory: its address, the type it loads, the access
  /// applied to the loaded value and the result's sort.
  struct HeapRead {
    HeapAddr addr;
    SmallVector<int> access;
    Sort sort;
    /// `hlcf.if` conditions known on the way to the read in the current
    /// context: it lies in an arm of an `if` on that condition. The same
    /// condition value decides every earlier `if` on it, e.g. the one that
    /// raises (lowered: `if c { store error }` then `if c { return } else
    /// { rest }`), so its other arm need not be searched.
    SmallVector<std::pair<Value, bool>> facts;

    HeapRead withoutFacts() const {
      HeapRead r = *this;
      r.facts.clear();
      return r;
    }
    std::optional<bool> fact(Value cond) const {
      for (auto &[value, known] : facts)
        if (value == cond)
          return known;
      return std::nullopt;
    }
  };

  static constexpr unsigned kMaxHeapDepth = 256;

  /// Whether `ptr` points into a stack slot, of the function itself or of a
  /// caller of the callee being evaluated. Heap memory never overlaps one.
  bool isStackAddress(Value ptr) {
    CallContext *c = ctx;
    while (true) {
      while (Operation *def = ptr.getDefiningOp()) {
        if (auto cast = dyn_cast<POP::PointerBitcastOp>(def))
          ptr = cast.getInput();
        else if (auto view = dyn_cast<POP::UnionBitcastOp>(def))
          ptr = view.getValue();
        else if (auto gep = dyn_cast<StructGEPOp>(def))
          ptr = gep.getContainer();
        else if (auto offset = dyn_cast<POP::OffsetOp>(def))
          ptr = offset.getPtr();
        else if (auto element = dyn_cast<POP::ArrayGEPOp>(def))
          ptr = element.getArray();
        else
          break;
      }
      if (ptr.getDefiningOp<POP::StackAllocationOp>())
        return true;
      if (!c || !isEntryArgument(ptr))
        return false;
      auto arg = c->args.find(ptr);
      if (arg == c->args.end())
        return false;
      ptr = arg->second;
      c = c->parent;
    }
  }

  /// The address `ptr` points to: constant field addresses inside an element,
  /// then offsets in units of that element, then the base pointer (whose
  /// bitcasts keep the address).
  std::optional<HeapAddr> heapAddress(Value ptr) {
    SmallVector<int> reversed;
    while (auto gep = ptr.getDefiningOp<StructGEPOp>()) {
      auto index = dyn_cast<IntegerAttr>(gep.getIndexAttr());
      if (!index)
        return std::nullopt;
      reversed.push_back(index.getInt());
      ptr = gep.getContainer();
    }
    Type element = cast<PointerType>(ptr.getType()).getElementType();
    std::string index = bvConst(APInt(64, 0));
    bool zero = true;
    while (true) {
      if (auto cast = ptr.getDefiningOp<POP::PointerBitcastOp>()) {
        if (llvm::cast<PointerType>(cast.getInput().getType())
                .getElementType() != element)
          break;
        ptr = cast.getInput();
      } else if (auto offset = ptr.getDefiningOp<POP::OffsetOp>()) {
        MaybeTerm i = term(offset.getIndex());
        if (!i)
          return std::nullopt;
        index = zero ? *i : "(bvadd " + index + " " + *i + ")";
        zero = false;
        ptr = offset.getPtr();
      } else {
        break;
      }
    }
    MaybeTerm base = term(ptr);
    if (!base)
      return std::nullopt;
    Value root = ptr;
    while (Operation *def = root.getDefiningOp()) {
      if (auto cast = dyn_cast<POP::PointerBitcastOp>(def))
        root = cast.getInput();
      else if (auto offset = dyn_cast<POP::OffsetOp>(def))
        root = offset.getPtr();
      else if (auto gep = dyn_cast<StructGEPOp>(def))
        root = gep.getContainer();
      else if (auto element = dyn_cast<POP::ArrayGEPOp>(def))
        root = element.getArray();
      else
        break;
    }
    MaybeTerm rootTerm = term(root);
    if (!rootTerm)
      return std::nullopt;
    return HeapAddr{*base, index, element,
                    SmallVector<int>(llvm::reverse(reversed)), *rootTerm};
  }

  /// An unknown for the read, described by why and where (`at`) it arises.
  std::string heapUnknown(const HeapRead &r, const Twine &why, Operation *at) {
    std::string name = declare(r.sort);
    std::string text;
    llvm::raw_string_ostream os(text);
    if (ctx)
      os << "in " << ctx->callee << ": ";
    os << "heap memory " << why;
    if (at) {
      os << " " << at->getName().getStringRef();
      if (std::optional<ObligationLocation> where = locationOf(at->getLoc()))
        os << " at " << where->file << ":" << where->line << ":" << where->col;
    }
    noteUnknown(name, text);
    return name;
  }

  /// The allocation `ptr` (a term) points into, as an uninterpreted function
  /// of the address. Allocations the heap search passes get distinct constant
  /// regions (`noteAllocation`); any other pointer's region is unknown, so it
  /// may point into any of them. Every allocation is treated as live: an
  /// address reused after a `free` would have two regions, which matters only
  /// to code that compares such pointers.
  std::string region(StringRef ptr) {
    if (!regionDeclared) {
      prelude += "(declare-fun region ((_ BitVec 64)) (_ BitVec 64))\n";
      regionDeclared = true;
    }
    return ("(region " + ptr + ")").str();
  }
  bool regionDeclared = false;
  unsigned regions = 0;
  std::set<std::pair<Operation *, CallContext *>> allocationsNoted;

  /// Give the memory returned by an allocation call its own region: done for
  /// every allocation in the function itself, for every one whose result a
  /// callee context resolves, and for every one the heap search passes.
  void noteAllocation(CallOp call) {
    if (!call->getNumResults() || !allocationsNoted.insert({call, ctx}).second)
      return;
    if (MaybeTerm ptr =
            resolveAccess(call->getResult(0), {0}, {Sort::Ptr, 64, false}))
      assertGlobal("(= " + region(*ptr) + " " + bvConst(APInt(64, ++regions)) +
                   ")");
  }

  static bool isAllocation(CallOp call) {
    return calleeName(call).starts_with("std::memory::alloc::alloc");
  }

  /// Ops that write no memory at all, heap or stack.
  static bool writesNothing(Operation *op) {
    StringRef name = op->getName().getStringRef();
    return isa<POP::LoadOp, POP::StackAllocationOp, ObligationOp, AssumeOp,
               CopyMarkerOp>(op) ||
           name == "pop.stack_alloc.lifetime.start" ||
           name == "pop.stack_alloc.lifetime.end" ||
           // Reading freed memory is undefined behavior, which the analysis
           // excludes.
           name == "pop.aligned_free" ||
           // A barrier orders memory accesses but writes nothing itself.
           name == "pop.fence" ||
           // Materializing a compile-time constant only produces a value.
           name == "kgen.param.materialize" ||
           // Control flow (`hlcf.yield`, `hlcf.break`, ...) moves values only.
           op->hasTrait<OpTrait::IsTerminator>() ||
           (!op->getNumRegions() && mlir::isMemoryEffectFree(op));
  }

  /// Whether `op` (including nested ops) may write heap memory.
  bool heapMayWrite(Operation *op) {
    return op
        ->walk([&](Operation *nested) {
          if (auto store = dyn_cast<POP::StoreOp>(nested))
            return isStackAddress(store.getPtr()) ? WalkResult::advance()
                                                  : WalkResult::interrupt();
          if (auto call = dyn_cast<CallOp>(nested))
            return isAllocation(call) ? WalkResult::advance()
                                      : WalkResult::interrupt();
          if (nested->getNumRegions() || writesNothing(nested))
            return WalkResult::advance();
          return WalkResult::interrupt();
        })
        .wasInterrupted();
  }

  MaybeTerm heapLoadTerm(POP::LoadOp load, ArrayRef<int> access, Sort sort) {
    if (isStackAddress(load.getPtr()))
      return std::nullopt;
    std::optional<HeapAddr> addr = heapAddress(load.getPtr());
    if (!addr)
      return std::nullopt;
    // Leading field accesses of the loaded value are part of the address,
    // like for stack slots.
    Type loaded = load.getResult().getType();
    size_t fields = 0;
    for (; fields < access.size() && access[fields] >= 0; ++fields) {
      auto structType = dyn_cast<StructType>(loaded);
      std::optional<SmallVector<Type>> elements =
          structType ? structType.getElementTypes() : std::nullopt;
      if (!elements || size_t(access[fields]) >= elements->size())
        break;
      loaded = (*elements)[access[fields]];
      addr->path.push_back(access[fields]);
    }
    HeapRead r{*addr, SmallVector<int>(access.drop_front(fields)), sort};
    return heapValueBefore(r, load, 0);
  }

  /// The value of the read just before `op`.
  MaybeTerm heapValueBefore(const HeapRead &r, Operation *op, unsigned depth) {
    if (depth > kMaxHeapDepth)
      return std::nullopt;
    std::string key;
    llvm::raw_string_ostream os(key);
    os << r.addr.base << "|" << r.addr.index << "|"
       << r.addr.element.getAsOpaquePointer() << "|" << r.sort.str();
    for (int step : r.addr.path)
      os << " " << step;
    os << "|";
    for (int step : r.access)
      os << " " << step;
    for (auto &[value, known] : r.facts)
      os << "|" << value.getAsOpaquePointer() << (known ? "t" : "f");
    auto memoKey = std::make_tuple(op, ctx, key);
    if (auto it = heapMemo.find(memoKey); it != heapMemo.end())
      return it->second;
    SmallVector<std::pair<std::string, std::string>> guards;
    MaybeTerm rest = heapScan(r, op, depth, guards);
    if (rest)
      for (auto &[cond, value] : llvm::reverse(guards))
        rest = "(ite " + cond + " " + value + " " + *rest + ")";
    // Name composite values: they are shared by every later read that
    // reaches this point, and copying them would grow exponentially.
    if (rest && rest->front() == '(')
      rest = define(r.sort, *rest);
    heapMemo[memoKey] = rest;
    return rest;
  }
  std::map<std::tuple<Operation *, CallContext *, std::string>, MaybeTerm>
      heapMemo;

  enum class HeapStep { Skip, Found, Fail };

  /// Scan backwards from just before `op` for stores that may have written
  /// the read, appending their guarded values to `guards` (most recent
  /// first); returns the value before the oldest one.
  MaybeTerm
  heapScan(const HeapRead &read, Operation *op, unsigned depth,
           SmallVectorImpl<std::pair<std::string, std::string>> &guards) {
    HeapRead r = read;
    for (Operation *cur = op;;) {
      for (Operation *prev = cur->getPrevNode(); prev;
           prev = prev->getPrevNode()) {
        // Passing an `if` one of whose arms always leaves (e.g. returns after
        // a raise) means the read's path took the other arm.
        if (auto ifOp = dyn_cast<HLCF::IfOp>(prev);
            ifOp && !r.fact(ifOp.getCond())) {
          if (armLeaves(ifOp.getThenBlock()))
            r.facts.push_back({ifOp.getCond(), false});
          else if (ifOp.getElifRegions().empty() &&
                   armLeaves(ifOp.getElseBlock()))
            r.facts.push_back({ifOp.getCond(), true});
        }
        MaybeTerm found;
        HeapStep step = heapStep(r, prev, depth, guards, found);
        if (step == HeapStep::Found)
          return found;
        if (step == HeapStep::Fail)
          return std::nullopt;
        // No `if` before a condition's definition can test it: forget facts
        // as soon as they cannot matter, so memoized values stay shared.
        llvm::erase_if(r.facts, [&](auto &fact) {
          return fact.first.getDefiningOp() == prev;
        });
      }
      llvm::erase_if(r.facts, [&](auto &fact) {
        return fact.first.getParentBlock() == cur->getBlock();
      });
      Operation *parent = cur->getParentOp();
      if (!parent)
        return std::nullopt;
      if (isa<FuncOp>(parent)) {
        // A callee's entry: continue in the caller, before the call.
        if (ctx && ctx->call) {
          CallContext *callee = ctx;
          return inContext(callee->parent, [&] {
            return heapValueBefore(r.withoutFacts(), callee->call, depth + 1);
          });
        }
        return heapUnknown(r, "at the entry of", parent);
      }
      if (auto ifOp = dyn_cast<HLCF::IfOp>(parent)) {
        // Leaving an arm: the value before the `if`, unless an elif
        // condition (which runs before the arm) may write it.
        auto elifs = ifOp.getElifRegions();
        for (unsigned i = 0; i < elifs.size(); i += 2)
          for (Operation &condOp : elifs[i].front())
            if (heapMayWrite(&condOp))
              return std::nullopt;
        HeapRead known = r;
        if (cur->getBlock() == &ifOp.getThenBlock())
          known.facts.push_back({ifOp.getCond(), true});
        else if (cur->getBlock() == &ifOp.getElseBlock() && elifs.empty())
          known.facts.push_back({ifOp.getCond(), false});
        return heapValueBefore(known, ifOp, depth + 1);
      }
      // The start of an iteration's body: what the previous iteration left,
      // or what the loop started with.
      if (ctx && ctx->loop == parent) {
        CallContext *iteration = ctx;
        if (iteration->iteration == 0)
          return inContext(iteration->owner, [&] {
            return heapValueBefore(r.withoutFacts(), parent, depth + 1);
          });
        Operation *next = parent->getRegion(0).front().getTerminator();
        return inContext(iteration->parent, [&] {
          return heapValueBefore(r.withoutFacts(), next, depth + 1);
        });
      }
      // A loop body could see a value an earlier iteration stored.
      if (heapMayWrite(parent))
        return std::nullopt;
      cur = parent;
    }
  }

  HeapStep
  heapStep(const HeapRead &r, Operation *op, unsigned depth,
           SmallVectorImpl<std::pair<std::string, std::string>> &guards,
           MaybeTerm &found) {
    if (auto store = dyn_cast<POP::StoreOp>(op))
      return heapStore(r, store, store.getPtr(), store.getArg(), guards, found);
    if (op->getName().getStringRef() == "pop.atomic.rmw")
      return heapStore(r, op, op->getOperand(0), Value(), guards, found);
    if (auto call = dyn_cast<CallOp>(op)) {
      if (isAllocation(call)) {
        noteAllocation(call);
        return HeapStep::Skip; // Fresh memory: writes no existing place.
      }
      auto callee = singleReturnCallee(call);
      CallContext *context = callee ? contextFor(call) : nullptr;
      if (!context)
        return HeapStep::Fail;
      // What the callee left there when it returned.
      found = inContext(context, [&] {
        return heapValueBefore(r.withoutFacts(), callee->second, depth + 1);
      });
      return found ? HeapStep::Found : HeapStep::Fail;
    }
    if (auto ifOp = dyn_cast<HLCF::IfOp>(op)) {
      if (!heapMayWrite(ifOp))
        return HeapStep::Skip;
      found = heapMergeIf(r, ifOp, depth + 1);
      return found ? HeapStep::Found : HeapStep::Fail;
    }
    if (auto loop = dyn_cast<HLCF::LoopOp>(op)) {
      if (!heapMayWrite(loop))
        return HeapStep::Skip;
      found = heapAfterLoop(r, loop, depth + 1);
      return found ? HeapStep::Found : HeapStep::Fail;
    }
    if (op->getNumRegions())
      return heapMayWrite(op) ? HeapStep::Fail : HeapStep::Skip;
    return writesNothing(op) ? HeapStep::Skip : HeapStep::Fail;
  }

  /// A write of `stored` (null: an unknown value, e.g. an atomic update) to
  /// `ptr` by `op`, met while scanning back for the read.
  HeapStep
  heapStore(const HeapRead &r, Operation *op, Value ptr, Value stored,
            SmallVectorImpl<std::pair<std::string, std::string>> &guards,
            MaybeTerm &found) {
    if (isStackAddress(ptr))
      return HeapStep::Skip;
    std::optional<HeapAddr> target = heapAddress(ptr);
    if (!target)
      return HeapStep::Fail;
    bool sameBase = target->base == r.addr.base;
    // Memory of different allocations never overlaps.
    std::string sameRegion =
        target->root == r.addr.root
            ? "true"
            : "(= " + region(target->root) + " " + region(r.addr.root) + ")";
    if (target->element != r.addr.element) {
      // Addresses counted in different element types are not compared: in
      // the same allocation, the write may overlap the read.
      std::string unknown = heapUnknown(r, "possibly overwritten by", op);
      if (sameBase) {
        found = unknown;
        return HeapStep::Found;
      }
      guards.push_back({sameRegion, unknown});
      return HeapStep::Skip;
    }
    std::string baseEq =
        sameBase ? "true" : "(= " + target->base + " " + r.addr.base + ")";
    std::string sameAddr =
        mkAnd(baseEq, target->index == r.addr.index
                          ? "true"
                          : "(= " + target->index + " " + r.addr.index + ")");
    // Different base pointers into one allocation may still overlap (one may
    // point into the other's memory at another offset): then the value is
    // unknown.
    auto clobbered = [&] {
      if (!sameBase)
        guards.push_back(
            {mkAnd("(not " + baseEq + ")", sameRegion),
             heapUnknown(r, "possibly overwritten through another pointer by",
                         op)});
    };
    if (isPrefix(target->path, r.addr.path)) {
      // The write covers the read field or an enclosing struct.
      MaybeTerm value;
      if (!stored) {
        value = heapUnknown(r, "written by", op);
      } else {
        SmallVector<int> path(r.addr.path.begin() + target->path.size(),
                              r.addr.path.end());
        path.append(r.access.begin(), r.access.end());
        value = path.empty()
                    ? (sortOf(stored.getType()) == r.sort ? term(stored)
                                                          : std::nullopt)
                    : resolveAccess(stored, path, r.sort);
      }
      if (!value)
        return HeapStep::Fail;
      if (sameAddr == "true") {
        found = value;
        return HeapStep::Found;
      }
      guards.push_back({sameAddr, *value});
      clobbered();
      return HeapStep::Skip;
    }
    if (isPrefix(r.addr.path, target->path)) {
      // A partial write of the read place.
      guards.push_back(
          {sameAddr, heapUnknown(r, "partially overwritten by", op)});
      clobbered();
      return HeapStep::Skip;
    }
    clobbered(); // Disjoint fields of one element never overlap.
    return HeapStep::Skip;
  }

  /// The value of the read after a loop that may write heap memory: what
  /// the exiting iteration left at its `hlcf.break`.
  MaybeTerm heapAfterLoop(const HeapRead &r, HLCF::LoopOp loop,
                          unsigned depth) {
    std::optional<LoopShape> shape = loopShape(loop);
    if (!shape)
      return std::nullopt;
    linkLoopResults(loop, *shape);
    bool certain;
    auto exits = exitingIterations(loop, *shape, certain);
    std::optional<std::string> expr;
    if (!certain)
      expr = heapUnknown(
          r, "after more than " + std::to_string(kMaxUnroll) + " iterations of",
          loop);
    for (auto &[k, cond] : llvm::reverse(exits)) {
      MaybeTerm value = inContext(iterationContext(loop, k), [&] {
        return heapValueBefore(r.withoutFacts(), shape->exit, depth + 1);
      });
      if (!value)
        value = heapUnknown(
            r, "at the exit of iteration " + std::to_string(k) + " of", loop);
      expr = expr ? "(ite " + cond + " " + *value + " " + *expr + ")" : *value;
    }
    if (!expr)
      return std::nullopt;
    return expr->front() == '(' ? define(r.sort, *expr) : *expr;
  }

  /// The value of the read after an `hlcf.if` that may write heap memory.
  MaybeTerm heapMergeIf(const HeapRead &r, HLCF::IfOp ifOp, unsigned depth) {
    auto armValue = [&](Block &block) -> std::string {
      Operation *terminator = block.getTerminator();
      // Control does not reach the read along an arm that leaves.
      if (armLeaves(block))
        return declare(r.sort);
      if (MaybeTerm t = heapValueBefore(r, terminator, depth))
        return *t;
      return heapUnknown(r, "written in an arm of", ifOp);
    };
    // An arm the read's path excludes is not searched.
    if (std::optional<bool> known = r.fact(ifOp.getCond())) {
      if (*known)
        return armValue(ifOp.getThenBlock());
      if (ifOp.getElifRegions().empty())
        return armValue(ifOp.getElseBlock());
    }
    SmallVector<std::pair<std::string, std::string>> arms;
    arms.push_back({boolTerm(ifOp.getCond()), armValue(ifOp.getThenBlock())});
    auto elifs = ifOp.getElifRegions();
    for (unsigned i = 0; i + 1 < elifs.size(); i += 2) {
      Operation *condYield = elifs[i].front().getTerminator();
      if (!isa<HLCF::IfElifCondYieldOp>(condYield) ||
          !condYield->getNumOperands())
        return std::nullopt;
      arms.push_back(
          {boolTerm(condYield->getOperand(0)), armValue(elifs[i + 1].front())});
    }
    std::string expr = armValue(ifOp.getElseBlock());
    for (auto &[cond, value] : llvm::reverse(arms))
      expr = "(ite " + cond + " " + value + " " + expr + ")";
    return expr;
  }

  /// The scalar reached from `aggregate` by the accesses in `path` (struct
  /// field indices or `kUnwrap`), following the ops that built the aggregate:
  /// `kgen.struct.create`, `pop.union.wrap` and `pop.select`. This keeps
  /// values visible through e.g. the `Optional` returned by iterators.
  MaybeTerm resolveAccess(Value aggregate, ArrayRef<int> path, Sort sort,
                          unsigned depth = 0) {
    if (ctx && ctx->loop)
      if (CallContext *owner = contextOf(aggregate); owner != ctx)
        return inContext(
            owner, [&] { return resolveAccess(aggregate, path, sort, depth); });
    if (path.empty()) {
      if (!(sortOf(aggregate.getType()) == sort))
        return std::nullopt;
      return term(aggregate);
    }
    if (depth > kMaxAccessDepth) {
      // Say so, rather than blaming whatever value an outer fallback picks.
      std::string name = declare(sort);
      noteUnknown(name, "access depth limit reached resolving " +
                            describe(aggregate));
      return name;
    }
    if (ctx) {
      auto arg = ctx->args.find(aggregate);
      if (arg != ctx->args.end())
        return inContext(ctx->parent, [&] {
          return resolveAccess(arg->second, path, sort, depth + 1);
        });
    }
    if (auto res = callResult(aggregate)) {
      auto [context, returned] = *res;
      return inContext(context, [&] {
        return resolveAccess(returned, path, sort, depth + 1);
      });
    }
    Operation *def = aggregate.getDefiningOp();
    if (!def)
      return opaqueField(aggregate, path, sort);
    if (isa<POP::LoadOp>(def))
      if (MaybeTerm t = loadTerm(def, path, sort))
        return t;
    Attribute constant;
    if (mlir::matchPattern(aggregate, mlir::m_Constant(&constant))) {
      // Walk into constant structs, e.g. the tag of a constant `None`.
      for (int step : path) {
        auto structAttr = dyn_cast<StructAttr>(constant);
        if (!structAttr || step < 0 ||
            static_cast<size_t>(step) >= structAttr.getValues().size())
          return std::nullopt;
        constant = structAttr.getValues()[step];
      }
      return constantTerm(constant, sort);
    }
    // An if/elif/else whose arms all yield: the access on the taken arm's
    // value, as a chain of `ite`s over the arms' conditions.
    if (auto ifOp = dyn_cast<HLCF::IfOp>(def)) {
      unsigned idx = cast<OpResult>(aggregate).getResultNumber();
      // An arm whose access does not resolve contributes an unknown, so the
      // other arms still count.
      auto yielded = [&](Block &block) -> MaybeTerm {
        auto yield = dyn_cast<HLCF::YieldOp>(block.getTerminator());
        if (!yield || idx >= yield->getNumOperands())
          return std::nullopt;
        if (MaybeTerm t =
                resolveAccess(yield->getOperand(idx), path, sort, depth + 1))
          return t;
        return declare(sort);
      };
      SmallVector<std::pair<std::string, std::string>> arms;
      bool complete = true;
      if (MaybeTerm t = yielded(ifOp.getThenBlock()))
        arms.push_back({boolTerm(ifOp.getCond()), *t});
      else
        complete = false;
      auto elifs = ifOp.getElifRegions();
      for (unsigned i = 0; complete && i + 1 < elifs.size(); i += 2) {
        Operation *condYield = elifs[i].front().getTerminator();
        MaybeTerm value = yielded(elifs[i + 1].front());
        if (!isa<HLCF::IfElifCondYieldOp>(condYield) ||
            !condYield->getNumOperands() || !value) {
          complete = false;
          break;
        }
        arms.push_back({boolTerm(condYield->getOperand(0)), *value});
      }
      MaybeTerm expr = complete ? yielded(ifOp.getElseBlock()) : std::nullopt;
      if (expr) {
        for (auto &[cond, value] : llvm::reverse(arms))
          expr = "(ite " + cond + " " + value + " " + *expr + ")";
        return expr;
      }
      return opaqueField(aggregate, path, sort);
    }
    // An aggregate that is itself a field or union member of another one.
    if (auto extract = dyn_cast<StructExtractOp>(def)) {
      if (auto index = dyn_cast<IntegerAttr>(extract.getIndexAttr())) {
        SmallVector<int> outer = {int(index.getInt())};
        outer.append(path.begin(), path.end());
        return resolveAccess(extract.getContainer(), outer, sort, depth + 1);
      }
    }
    if (def->getName().getStringRef() == "pop.union.unwrap") {
      SmallVector<int> outer = {kUnwrap};
      outer.append(path.begin(), path.end());
      return resolveAccess(def->getOperand(0), outer, sort, depth + 1);
    }
    if (auto create = dyn_cast<StructCreateOp>(def)) {
      if (path.front() < 0 ||
          static_cast<unsigned>(path.front()) >= create->getNumOperands())
        return std::nullopt;
      return resolveAccess(create->getOperand(path.front()), path.drop_front(),
                           sort, depth + 1);
    }
    if (def->getName().getStringRef() == "pop.union.wrap" &&
        path.front() == kUnwrap)
      return resolveAccess(def->getOperand(0), path.drop_front(), sort,
                           depth + 1);
    if (isa<POP::SelectOp, POP::SIMDSelectOp>(def)) {
      Value cond = def->getOperand(0);
      if (sortOf(cond.getType()).kind != Sort::Bool)
        return std::nullopt;
      MaybeTerm t = resolveAccess(def->getOperand(1), path, sort, depth + 1);
      MaybeTerm f = resolveAccess(def->getOperand(2), path, sort, depth + 1);
      if (!t && !f)
        return std::nullopt;
      return "(ite " + boolTerm(cond) + " " + (t ? *t : declare(sort)) + " " +
             (f ? *f : declare(sort)) + ")";
    }
    return opaqueField(aggregate, path, sort);
  }

  /// A field of an aggregate whose construction is not visible (a block
  /// argument, a call result, a load, ...): the same term every access to that
  /// field of that SSA value gets, so e.g. a callee's view of its argument
  /// agrees with the caller's view of the operand.
  std::optional<std::string> opaqueField(Value aggregate, ArrayRef<int> path,
                                         Sort sort) {
    if (path.empty())
      return std::nullopt;
    if (path.size() > 1 || path.front() < 0) {
      std::string key, text;
      for (int step : path) {
        key += (step == kUnwrap ? "u" : std::to_string(step)) + ".";
        text += step == kUnwrap ? " unwrap" : " ." + std::to_string(step);
      }
      auto &memo = pathMap();
      auto [it, inserted] =
          memo.try_emplace({aggregate.getAsOpaquePointer(), key}, "");
      if (inserted) {
        it->second = declare(sort);
        noteUnknown(it->second, "access" + text + " of " + describe(aggregate));
        noteStaticPointer(aggregate, it->second, sort);
      }
      return it->second;
    }
    auto &memo = extractMap();
    auto key = std::make_pair(aggregate, unsigned(path.front()));
    auto it = memo.find(key);
    if (it == memo.end()) {
      it = memo.try_emplace(key, declare(sort)).first;
      noteUnknown(it->second, "field " + std::to_string(path.front()) + " of " +
                                  describe(aggregate));
      noteStaticPointer(aggregate, it->second, sort);
    }
    return it->second;
  }

  /// A pointer inside a compile-time constant (`kgen.param.materialize`)
  /// points to static data, outside every allocation.
  void noteStaticPointer(Value aggregate, StringRef name, Sort sort) {
    Operation *def = aggregate.getDefiningOp();
    if (sort.kind == Sort::Ptr && def &&
        def->getName().getStringRef() == "kgen.param.materialize")
      assertGlobal("(= " + region(name) + " " + bvConst(APInt(64, 0)) + ")");
  }

  /// For a chain of `kgen.struct.extract` / `pop.union.unwrap` ending in
  /// `value`, the innermost aggregate and the accesses applied to it.
  static std::pair<Value, SmallVector<int>> accessPath(Value value) {
    SmallVector<int> reversed;
    while (Operation *def = value.getDefiningOp()) {
      if (auto extract = dyn_cast<StructExtractOp>(def)) {
        auto index = dyn_cast<IntegerAttr>(extract.getIndexAttr());
        if (!index)
          break;
        reversed.push_back(index.getInt());
        value = extract.getContainer();
      } else if (def->getName().getStringRef() == "pop.union.unwrap") {
        reversed.push_back(kUnwrap);
        value = def->getOperand(0);
      } else {
        break;
      }
    }
    return {value, SmallVector<int>(llvm::reverse(reversed))};
  }

  /// Encode a single-result op if its semantics are modeled; otherwise its
  /// result stays unconstrained (created lazily by `term`).
  void encodeOp(Operation *op) {
    if (op->getNumResults() != 1)
      return;
    Value result = op->getResult(0);
    Sort sort = sortOf(result.getType());
    if (sort.kind == Sort::None)
      return;

    if (op->hasTrait<OpTrait::ConstantLike>()) {
      if (MaybeTerm c = constantTerm(result, sort))
        setTerm(result, *c);
      return;
    }

    StringRef name = op->getName().getStringRef();
    if (isa<POP::PointerBitcastOp>(op)) {
      if (MaybeTerm t = term(op->getOperand(0)))
        termMap()[result] = *t; // The same address.
      return;
    }
    if (isa<POP::LoadOp>(op)) {
      MaybeTerm t = loadTerm(op, {}, sort);
      if (t) {
        if (t->front() == '(')
          setTerm(result, *t);
        else
          termMap()[result] = *t;
        return;
      }
    }
    auto operandTerms = [&]() {
      SmallVector<std::string> ts;
      for (Value operand : op->getOperands()) {
        MaybeTerm t = term(operand);
        if (!t || !(sortOf(operand.getType()) == sort))
          return SmallVector<std::string>();
        ts.push_back(*t);
      }
      return ts;
    };

    // Bit-preserving and converting casts.
    if (name == "pop.cast" || name == "pop.cast_to_builtin" ||
        name == "pop.cast_from_builtin" || name == "index.casts" ||
        name == "index.castu") {
      Value input = op->getOperand(0);
      Sort from = sortOf(input.getType());
      if (name == "index.casts")
        from.isSigned = true;
      if (name == "index.castu")
        from.isSigned = false;
      if (from.kind == Sort::None)
        return;
      // Honor the op-specific signedness for extension.
      Sort to = sort;
      MaybeTerm in = term(input);
      if (!in)
        return;
      if (from.kind == Sort::BV && to.kind == Sort::BV &&
          from.width < to.width) {
        setTerm(result, std::string(from.isSigned ? "((_ sign_extend "
                                                  : "((_ zero_extend ") +
                            std::to_string(to.width - from.width) + ") " + *in +
                            ")");
        return;
      }
      if (MaybeTerm t = convert(input, to)) {
        if (*t == *in)
          termMap()[result] = *in; // Pure aliasing, no new name needed.
        else
          setTerm(result, *t);
      }
      return;
    }

    if (sort.kind == Sort::BV) {
      static const llvm::StringMap<StringRef> binary = {
          {"pop.add", "bvadd"},      {"pop.sub", "bvsub"},
          {"pop.mul", "bvmul"},      {"index.add", "bvadd"},
          {"index.sub", "bvsub"},    {"index.mul", "bvmul"},
          {"pop.simd.and", "bvand"}, {"pop.simd.or", "bvor"},
          {"pop.simd.xor", "bvxor"}, {"index.and", "bvand"},
          {"index.or", "bvor"},      {"index.xor", "bvxor"}};
      auto it = binary.find(name);
      if (it != binary.end()) {
        SmallVector<std::string> ts = operandTerms();
        if (ts.size() == 2)
          setTerm(result,
                  ("(" + it->second + " " + ts[0] + " " + ts[1] + ")").str());
        return;
      }
      if (name == "pop.neg" || name == "pop.abs") {
        SmallVector<std::string> ts = operandTerms();
        if (ts.size() != 1)
          return;
        std::string neg = "(bvneg " + ts[0] + ")";
        if (name == "pop.neg")
          setTerm(result, neg);
        else
          setTerm(result, "(ite (bvslt " + ts[0] + " " +
                              bvConst(APInt(sort.width, 0)) + ") " + neg + " " +
                              ts[0] + ")");
        return;
      }
      if (name == "pop.max" || name == "pop.min" || name == "index.maxs" ||
          name == "index.mins" || name == "index.maxu" ||
          name == "index.minu") {
        SmallVector<std::string> ts = operandTerms();
        if (ts.size() != 2)
          return;
        bool isSigned =
            name.starts_with("pop.") ? sort.isSigned : name.ends_with("s");
        bool isMax = name.contains("max");
        std::string lt = std::string(isSigned ? "(bvslt " : "(bvult ") + ts[0] +
                         " " + ts[1] + ")";
        setTerm(result, "(ite " + lt + " " + (isMax ? ts[1] : ts[0]) + " " +
                            (isMax ? ts[0] : ts[1]) + ")");
        return;
      }
      // Shifts and divisions only by constants that avoid undefined cases.
      if (name == "pop.floordiv") {
        // Rounds towards negative infinity. Division by zero and signed
        // `MIN / -1` are undefined: the result is unconstrained there.
        SmallVector<std::string> ts = operandTerms();
        if (ts.size() != 2)
          return;
        std::string zero = bvConst(APInt(sort.width, 0));
        std::string undefined = "(= " + ts[1] + " " + zero + ")";
        std::string quotient = "(bvudiv " + ts[0] + " " + ts[1] + ")";
        if (sort.isSigned) {
          undefined = "(or " + undefined + " (and (= " + ts[0] + " " +
                      bvConst(APInt::getSignedMinValue(sort.width)) +
                      ") (= " + ts[1] + " " +
                      bvConst(APInt::getAllOnes(sort.width)) + ")))";
          std::string q = "(bvsdiv " + ts[0] + " " + ts[1] + ")";
          std::string r = "(bvsrem " + ts[0] + " " + ts[1] + ")";
          quotient = "(ite (and (distinct " + r + " " + zero +
                     ") (xor (bvslt " + ts[0] + " " + zero + ") (bvslt " +
                     ts[1] + " " + zero + "))) (bvsub " + q + " " +
                     bvConst(APInt(sort.width, 1)) + ") " + q + ")";
        }
        setTerm(result, "(ite " + undefined + " " + declare(sort) + " " +
                            quotient + ")");
        return;
      }
      if (name == "pop.shl" || name == "pop.shr") {
        SmallVector<std::string> ts = operandTerms();
        std::optional<APInt> rhs = constantInt(op->getOperand(1), sort.width);
        if (ts.size() != 2 || !rhs)
          return;
        APInt amount = *rhs;
        if (amount.uge(sort.width))
          return;
        StringRef shift =
            name == "pop.shl" ? "bvshl" : (sort.isSigned ? "bvashr" : "bvlshr");
        setTerm(result, ("(" + shift + " " + ts[0] + " " + ts[1] + ")").str());
        return;
      }
    }

    if (sort.kind == Sort::Bool) {
      if (auto cmp = dyn_cast<POP::CmpOp>(op)) {
        Sort operandSort = sortOf(cmp.getLhs().getType());
        MaybeTerm l = term(cmp.getLhs()), r = term(cmp.getRhs());
        if (!l || !r)
          return;
        if (operandSort.kind == Sort::Bool) {
          if (cmp.getPred() == CmpPredicate::EQ)
            setTerm(result, "(= " + *l + " " + *r + ")");
          else if (cmp.getPred() == CmpPredicate::NE)
            setTerm(result, "(distinct " + *l + " " + *r + ")");
          return;
        }
        if (std::optional<std::string> fn =
                cmpOp(cmp.getPred(), operandSort.isSigned))
          setTerm(result, "(" + *fn + " " + *l + " " + *r + ")");
        return;
      }
      if (auto cmp = dyn_cast<mlir::index::CmpOp>(op)) {
        MaybeTerm l = term(cmp.getLhs()), r = term(cmp.getRhs());
        if (l && r)
          setTerm(result,
                  "(" + indexCmpOp(cmp.getPred()) + " " + *l + " " + *r + ")");
        return;
      }
      static const llvm::StringMap<StringRef> logical = {
          {"pop.simd.and", "and"},
          {"pop.simd.or", "or"},
          {"pop.simd.xor", "xor"}};
      auto it = logical.find(name);
      if (it != logical.end()) {
        SmallVector<std::string> ts = operandTerms();
        if (ts.size() == 2)
          setTerm(result,
                  ("(" + it->second + " " + ts[0] + " " + ts[1] + ")").str());
        return;
      }
    }

    // `pop.select`, and `pop.simd.select` on scalars (same operand order).
    if (isa<POP::SelectOp, POP::SIMDSelectOp>(op)) {
      Value cond = op->getOperand(0), yes = op->getOperand(1),
            no = op->getOperand(2);
      if (sortOf(cond.getType()).kind != Sort::Bool)
        return;
      MaybeTerm c = term(cond), t = term(yes), f = term(no);
      if (c && t && f && sortOf(yes.getType()) == sort)
        setTerm(result, "(ite " + *c + " " + *t + " " + *f + ")");
      return;
    }

    if (isa<StructExtractOp>(op) || name == "pop.union.unwrap") {
      auto [aggregate, path] = accessPath(result);
      if (!path.empty())
        if (MaybeTerm t = resolveAccess(aggregate, path, sort)) {
          if (t->front() == '(')
            setTerm(result, *t);
          else
            termMap()[result] = *t;
          return;
        }
    }
    if (auto extract = dyn_cast<StructExtractOp>(op)) {
      auto index = dyn_cast<IntegerAttr>(extract.getIndexAttr());
      if (!index)
        return;
      Value container = extract.getContainer();
      unsigned idx = index.getInt();
      // The same field of the same struct value is the same value.
      auto key = std::make_pair(container, idx);
      auto &memo = extractMap();
      auto it = memo.find(key);
      if (it == memo.end()) {
        it = memo.try_emplace(key, declare(sort)).first;
        noteUnknown(it->second, "field " + std::to_string(idx) + " of " +
                                    describe(container));
      }
      termMap()[result] = it->second;
      return;
    }
  }

  //===--------------------------------------------------------------------===//
  // Control flow
  //===--------------------------------------------------------------------===//

  struct BlockResult {
    /// Condition under which control falls through the block's terminator.
    std::string fall;
    Operation *terminator = nullptr;
  };

  BlockResult encodeBlock(Block &block, std::string reach) {
    size_t activeSize = activeObligations.size();
    size_t completedSize = completedLoops.size();
    BlockResult result{reach, nullptr};
    for (Operation &op : block) {
      if (op.hasTrait<OpTrait::IsTerminator>()) {
        result = {handleTerminator(&op, reach), &op};
        break;
      }
      reach = encodeOperation(&op, reach);
      result.fall = reach;
    }
    activeObligations.resize(activeSize);
    completedLoops.resize(completedSize);
    return result;
  }

  SmallVector<MaybeTerm> termsOf(ValueRange values) {
    SmallVector<MaybeTerm> result;
    for (Value v : values)
      result.push_back(term(v));
    return result;
  }

  LoopInfo *findLoop(StringAttr label) {
    for (LoopInfo *loop : llvm::reverse(loopStack)) {
      auto op = loopOps.lookup(loop);
      if (!label || (op.getLabelAttr() && op.getLabelAttr() == label))
        return loop;
    }
    return nullptr;
  }

  std::string handleTerminator(Operation *op, StringRef reach) {
    std::string r = reachName(reach);
    if (auto cont = dyn_cast<HLCF::ContinueOp>(op)) {
      if (LoopInfo *loop = findLoop(cont.getLabelAttr()))
        loop->continues.push_back(
            {r, termsOf(cont.getOperands()), assumedLoops(), assumed});
      return "false";
    }
    if (auto brk = dyn_cast<HLCF::BreakOp>(op)) {
      if (LoopInfo *loop = findLoop(brk.getLabelAttr()))
        loop->breaks.push_back(
            {r, termsOf(brk.getOperands()), assumedLoops(), assumed});
      return "false";
    }
    if (auto raise = dyn_cast<LIT::TryRaiseOp>(op)) {
      for (TryInfo *t : llvm::reverse(tryStack))
        if (t->op.getLabelAttr() == raise.getLabelAttr()) {
          t->raises.push_back(
              {r, termsOf(raise.getOperands()), assumedLoops(), assumed});
          break;
        }
      return "false"; // An unmatched raise leaves the function.
    }
    if (isa<HLCF::ReturnOp, HLCF::UnreachableOp>(op))
      return "false";
    // yield, try.yield, elifcond.yield and unknown terminators fall through.
    return r;
  }

  std::string encodeOperation(Operation *op, std::string reach) {
    if (auto obligation = dyn_cast<ObligationOp>(op)) {
      std::string r = reachName(reach);
      std::string cond = boolTerm(obligation.getCond());
      if (isOwnPrecondition(obligation)) {
        addAssumption(r, cond);
        return reach;
      }
      obligations.push_back(
          {obligation, getObligationKind(obligation),
           getObligationLocation(obligation), r, cond, assumedLoops(),
           SmallVector<std::pair<std::string, std::string>>(activeObligations),
           assumed});
      activeObligations.push_back({r, cond});
      return reach;
    }
    if (auto assume = dyn_cast<AssumeOp>(op)) {
      std::string cond = boolTerm(assume.getCond());
      addAssumption(reachName(reach), cond);
      return reach;
    }
    if (auto call = dyn_cast<CallOp>(op)) {
      instantiateContracts(call, reach);
      if (isAllocation(call))
        noteAllocation(call);
    }
    if (auto ifOp = dyn_cast<HLCF::IfOp>(op))
      return encodeIf(ifOp, reach);
    if (auto loop = dyn_cast<HLCF::LoopOp>(op))
      return encodeLoop(loop, reach);
    if (auto tryOp = dyn_cast<LIT::TryOp>(op))
      return encodeTry(tryOp, reach);
    if (isa<mlir::FunctionOpInterface>(op))
      return reach; // Nested functions are encoded on their own.
    if (op->getNumRegions()) {
      // Unknown region op: its regions may run any number of times under the
      // current condition; block arguments and results stay unconstrained.
      // Each block gets its own free guard: blocks are not known to be
      // exclusive, so value definitions from different blocks must not be
      // able to contradict each other.
      for (Region &region : op->getRegions())
        for (Block &block : region)
          encodeBlock(block,
                      reachName(mkAnd(reach, declare({Sort::Bool, 1, false}))));
      return reach;
    }
    encodeOp(op);
    return reach;
  }

  /// Apply the contract of a (non-inlined) callee at a call to it: its
  /// `requires` become obligations of the caller, reported at the call, and
  /// its `ensures` are assumed after them. The callee's entry arguments are
  /// bound to the call's operands and its returned values back the call's
  /// results, so conditions over "self at exit" talk about the call's results.
  void instantiateContracts(CallOp call, StringRef reach) {
    if (ctx || !symbols)
      return;
    auto callee = dyn_cast<SymbolConstantAttr>(call.getCallee());
    if (!callee)
      return;
    auto fn = symbols->lookup<FuncOp>(callee.getSymbol().getRootReference());
    if (!fn || fn == self || fn->getRegion(0).empty())
      return;
    Block &entry = fn->getRegion(0).front();
    if (entry.getNumArguments() != call->getNumOperands())
      return;
    SmallVector<ObligationOp> preconditions, postconditions;
    for (Operation &op : entry)
      if (auto ob = dyn_cast<ObligationOp>(&op)) {
        if (isOwnPrecondition(ob))
          preconditions.push_back(ob);
        else if (getObligationKind(ob) == "ensures")
          postconditions.push_back(ob);
      }
    // A postcondition must hold on every normal exit: require exactly one
    // return, at the end of the entry block, preceded by the `ensures`.
    auto ret = dyn_cast<HLCF::ReturnOp>(entry.getTerminator());
    bool singleReturn =
        ret && ret->getNumOperands() == call->getNumResults() &&
        !fn->walk([&](HLCF::ReturnOp r) {
             return r == ret ? WalkResult::advance() : WalkResult::interrupt();
           }).wasInterrupted();
    if (!singleReturn)
      postconditions.clear();
    // With a single return, the call's results are the callee's returned
    // values even without contracts; they are evaluated on demand.
    if (preconditions.empty() && !singleReturn)
      return;

    CallContext &context = contexts.emplace_back();
    callContexts[call] = &context;
    context.call = call;
    context.callee =
        callee.getSymbol().getRootReference().getValue().split('(').first.str();
    for (auto [arg, operand] :
         llvm::zip(entry.getArguments(), call->getOperands()))
      context.args[arg] = operand;
    std::string r = reachName(reach);
    for (ObligationOp pre : preconditions) {
      std::string cond =
          inContext(&context, [&] { return boolTerm(pre.getCond()); });
      obligations.push_back(
          {pre, "requires", locationOf(call.getLoc()), r, cond, assumedLoops(),
           SmallVector<std::pair<std::string, std::string>>(activeObligations),
           assumed});
      activeObligations.push_back({r, cond});
      ++contractsUsed;
    }
    if (!singleReturn)
      return;
    for (auto [result, returned] :
         llvm::zip(call->getResults(), ret->getOperands()))
      callResults[result] = {&context, returned};
    for (ObligationOp post : postconditions) {
      std::string cond =
          inContext(&context, [&] { return boolTerm(post.getCond()); });
      addAssumption(r, cond);
      ++contractsUsed;
    }
  }

  /// Bind `results` to the values yielded by the arms that fall through.
  void bindResults(ValueRange results, ArrayRef<BlockResult> arms) {
    for (auto [i, res] : llvm::enumerate(results)) {
      MaybeTerm r = term(res);
      if (!r)
        continue;
      for (const BlockResult &arm : arms) {
        if (arm.fall == "false" || !arm.terminator ||
            i >= arm.terminator->getNumOperands())
          continue;
        if (MaybeTerm v = term(arm.terminator->getOperand(i)))
          assertGlobal("(=> " + arm.fall + " (= " + *r + " " + *v + "))");
      }
    }
  }

  std::string encodeIf(HLCF::IfOp ifOp, StringRef reach) {
    std::string c = boolTerm(ifOp.getCond());
    SmallVector<BlockResult> arms;
    arms.push_back(
        encodeBlock(ifOp.getThenBlock(), reachName(mkAnd(reach, c))));
    std::string rest = reachName(mkAnd(reach, mkNot(c)));
    auto elifs = ifOp.getElifRegions();
    for (unsigned i = 0; i + 1 < elifs.size(); i += 2) {
      BlockResult condArm = encodeBlock(elifs[i].front(), rest);
      std::string ck = "false";
      if (condArm.terminator && condArm.terminator->getNumOperands())
        ck = boolTerm(condArm.terminator->getOperand(0));
      arms.push_back(encodeBlock(elifs[i + 1].front(),
                                 reachName(mkAnd(condArm.fall, ck))));
      rest = reachName(mkAnd(condArm.fall, mkNot(ck)));
    }
    arms.push_back(encodeBlock(ifOp.getElseBlock(), rest));
    bindResults(ifOp.getResults(), arms);
    SmallVector<std::string> falls;
    for (const BlockResult &arm : arms)
      falls.push_back(arm.fall);
    return reachName(mkOr(falls));
  }

  DenseMap<LoopInfo *, HLCF::LoopOp> loopOps;

  std::string encodeLoop(HLCF::LoopOp loopOp, StringRef reach) {
    LoopInfo &loop = loops.emplace_back();
    loop.id = loops.size() - 1;
    loopOps[&loop] = loopOp;
    loop.reachIn = reachName(reach);
    loop.assumedAtEntry = assumed;
    loop.enclosing = assumedLoops();
    loop.inits = termsOf(loopOp.getOperands());
    Block &body = loopOp.getBody().front();
    for (BlockArgument arg : body.getArguments()) {
      loop.args.push_back(term(arg));
      loop.argSorts.push_back(sortOf(arg.getType()));
    }

    // Loop-invariant terms the body compares against are good bounds.
    llvm::SetVector<Value> bounds;
    loopOp.getBody().walk([&](Operation *op) {
      if (!isa<POP::CmpOp, mlir::index::CmpOp>(op))
        return;
      for (Value operand : op->getOperands()) {
        Region *region = operand.getParentRegion();
        if (!loopOp.getBody().isAncestor(region))
          bounds.insert(operand);
      }
    });
    for (Value bound : bounds)
      if (MaybeTerm t = term(bound))
        loop.boundTerms.push_back({*t, sortOf(bound.getType())});

    loopStack.push_back(&loop);
    BlockResult bodyResult = encodeBlock(body, loop.reachIn);
    loopStack.pop_back();

    SmallVector<std::string> exits;
    SmallVector<BlockResult> exitArms;
    for (Edge &brk : loop.breaks)
      exits.push_back(brk.reach);
    if (bodyResult.fall != "false")
      exits.push_back(loop.reachIn); // Unexpected fallthrough: be safe.
    for (auto [i, res] : llvm::enumerate(loopOp.getResults())) {
      MaybeTerm r = term(res);
      if (!r)
        continue;
      for (Edge &brk : loop.breaks)
        if (i < brk.values.size() && brk.values[i])
          assertGlobal("(=> " + brk.reach + " (= " + *r + " " + *brk.values[i] +
                       "))");
    }
    completedLoops.push_back(&loop);
    return reachName(mkOr(exits));
  }

  std::string encodeTry(LIT::TryOp tryOp, StringRef reach) {
    std::string r = reachName(reach);
    if (!tryOp.getFinallyRegions().empty()) {
      for (Region &region : tryOp->getRegions())
        for (Block &block : region)
          encodeBlock(block, r);
      return r;
    }
    // Calls in the try region might raise without an explicit `try.raise`.
    bool mayRaiseImplicitly = tryOp.getTryRegion()
                                  .walk([](Operation *op) {
                                    return isa<CallOp, CallIndirectOp>(op)
                                               ? WalkResult::interrupt()
                                               : WalkResult::advance();
                                  })
                                  .wasInterrupted();

    TryInfo info{tryOp, {}};
    tryStack.push_back(&info);
    BlockResult tryResult = encodeBlock(tryOp.getTryRegion().front(), r);
    tryStack.pop_back();

    // Except region: arguments come from the raises.
    Block &exceptBlock = tryOp.getExceptRegion().front();
    SmallVector<std::string> exceptReaches;
    for (Edge &raise : info.raises)
      exceptReaches.push_back(raise.reach);
    if (mayRaiseImplicitly)
      exceptReaches.push_back(r);
    for (auto [i, arg] : llvm::enumerate(exceptBlock.getArguments())) {
      MaybeTerm a = term(arg);
      if (!a)
        continue;
      for (Edge &raise : info.raises)
        if (i < raise.values.size() && raise.values[i])
          assertGlobal("(=> " + raise.reach + " (= " + *a + " " +
                       *raise.values[i] + "))");
    }

    // Else region: arguments come from the try region's normal completion.
    Block &elseBlock = tryOp.getElseRegion().front();
    for (auto [i, arg] : llvm::enumerate(elseBlock.getArguments())) {
      MaybeTerm a = term(arg);
      if (!a || tryResult.fall == "false" || !tryResult.terminator ||
          i >= tryResult.terminator->getNumOperands())
        continue;
      if (MaybeTerm v = term(tryResult.terminator->getOperand(i)))
        assertGlobal("(=> " + tryResult.fall + " (= " + *a + " " + *v + "))");
    }

    SmallVector<BlockResult> arms;
    arms.push_back(encodeBlock(exceptBlock, reachName(mkOr(exceptReaches))));
    arms.push_back(encodeBlock(elseBlock, tryResult.fall));
    bindResults(tryOp.getResults(), arms);
    return reachName(mkOr({arms[0].fall, arms[1].fall}));
  }
};

//===----------------------------------------------------------------------===//
// Solver
//===----------------------------------------------------------------------===//

enum class Answer { Unsat, Sat, Unknown };

/// The solver's reply to one query: the `check-sat` answer and, for queries
/// that asked for them, the values of named Boolean terms in the model.
struct Reply {
  Answer answer = Answer::Unknown;
  DenseMap<unsigned, bool> values;
  /// The raw output for the query (e.g. model values from `get-value`).
  std::string text;
};

/// Each query is preceded by `(echo "@@")`, so the output splits into one
/// segment per query regardless of error messages (e.g. `get-value` after
/// `unsat` reports that no model is available).
std::optional<std::vector<Reply>> parseReplies(StringRef output,
                                               size_t expected) {
  SmallVector<StringRef> segments;
  output.split(segments, "@@\n");
  if (segments.empty() || segments.size() - 1 != expected)
    return std::nullopt;
  std::vector<Reply> replies;
  for (StringRef segment : ArrayRef(segments).drop_front()) {
    Reply reply;
    StringRef answer = segment.split('\n').first.trim();
    if (answer == "unsat")
      reply.answer = Answer::Unsat;
    else if (answer == "sat")
      reply.answer = Answer::Sat;
    else if (answer == "unknown" || answer == "timeout")
      reply.answer = Answer::Unknown;
    else
      return std::nullopt; // Malformed script.
    // Values look like `((q0 true) (q1 false))`.
    StringRef rest = segment;
    while (true) {
      size_t pos = rest.find("(q");
      if (pos == StringRef::npos)
        break;
      rest = rest.drop_front(pos + 2);
      unsigned index;
      if (rest.consumeInteger(10, index))
        continue;
      rest = rest.ltrim();
      if (rest.starts_with("true"))
        reply.values[index] = true;
      else if (rest.starts_with("false"))
        reply.values[index] = false;
    }
    reply.text = segment.str();
    replies.push_back(std::move(reply));
  }
  return replies;
}

/// Runs z3 on `script` and returns one reply per query, or nullopt if the
/// solver could not be run or its output does not match.
std::optional<std::vector<Reply>> runZ3(StringRef z3, StringRef script,
                                        size_t expected, StringRef dumpDir,
                                        StringRef dumpName) {
  SmallString<128> scriptPath, outPath;
  if (!dumpDir.empty()) {
    scriptPath = dumpDir;
    llvm::sys::path::append(scriptPath, dumpName + ".smt2");
  } else if (llvm::sys::fs::createTemporaryFile("bounds", "smt2", scriptPath)) {
    return std::nullopt;
  }
  if (llvm::sys::fs::createTemporaryFile("bounds", "out", outPath))
    return std::nullopt;
  {
    std::error_code ec;
    llvm::raw_fd_ostream os(scriptPath, ec);
    if (ec)
      return std::nullopt;
    os << script;
  }
  std::optional<StringRef> redirects[] = {StringRef(""), StringRef(outPath),
                                          StringRef("")};
  int rc = llvm::sys::ExecuteAndWait(z3, {z3, "-smt2", scriptPath},
                                     std::nullopt, redirects);
  auto buffer = llvm::MemoryBuffer::getFile(outPath);
  llvm::sys::fs::remove(outPath);
  if (dumpDir.empty())
    llvm::sys::fs::remove(scriptPath);
  if (rc < 0 || !buffer)
    return std::nullopt;
  return parseReplies((*buffer)->getBuffer(), expected);
}

/// Builds one script out of independent `(push) ... (check-sat) (pop)` blocks.
struct QueryBatch {
  std::string text;
  size_t count = 0;

  /// Check `assumptions && goalNegation`. `named` terms are defined as
  /// `q0, q1, ...` inside the query (usable in `goalNegation`) and their
  /// values are requested when the query is satisfiable.
  /// `show` lists existing terms whose values are requested as well.
  void add(ArrayRef<std::string> assumptions, StringRef goalNegation,
           ArrayRef<std::string> named = {}, ArrayRef<std::string> show = {}) {
    text += "(echo \"@@\")\n(push 1)\n";
    for (const std::string &a : assumptions)
      if (a != "true")
        text += "(assert " + a + ")\n";
    for (auto [i, t] : llvm::enumerate(named))
      text += "(define-fun q" + std::to_string(i) + " () Bool " + t + ")\n";
    text += ("(assert " + goalNegation + ")\n(check-sat)\n").str();
    if (!named.empty()) {
      text += "(get-value (";
      for (unsigned i = 0; i < named.size(); ++i)
        text += " q" + std::to_string(i);
      text += "))\n";
    }
    if (!show.empty()) {
      text += "(get-value (";
      for (const std::string &name : show)
        text += " " + name;
      text += "))\n";
    }
    text += "(pop 1)\n";
    ++count;
  }
};

/// The model value z3 printed for `name` (from `get-value`), as a signed
/// decimal for bitvectors, or nullopt.
std::optional<std::string> modelValue(StringRef text, StringRef name) {
  std::string key = ("(" + name + " ").str();
  size_t pos = text.find(key);
  if (pos == StringRef::npos)
    return std::nullopt;
  StringRef value = text.drop_front(pos + key.size()).ltrim();
  value = value.take_until([](char c) { return c == ')' || c == '\n'; });
  value = value.trim();
  APInt bits;
  if (value.consume_front("#x") && !value.getAsInteger(16, bits))
    return llvm::toString(bits.zextOrTrunc(value.size() * 4), 10, true);
  if (value.consume_front("#b") && !value.getAsInteger(2, bits))
    return llvm::toString(bits.zextOrTrunc(value.size()), 10, true);
  return value.str();
}

/// Named invariant of each loop, defined once per script.
std::string invariantDefinitions(const std::deque<LoopInfo> &loops) {
  std::string defs;
  for (const LoopInfo &loop : loops)
    defs += "(define-fun inv" + std::to_string(loop.id) + " () Bool " +
            loop.invariant() + ")\n";
  return defs;
}

SmallVector<std::string> invariantsOf(ArrayRef<LoopInfo *> loops) {
  SmallVector<std::string> result;
  for (LoopInfo *loop : loops)
    result.push_back("inv" + std::to_string(loop->id));
  return result;
}

void generateCandidates(LoopInfo &loop) {
  constexpr size_t kMaxCandidates = 1024;
  static const char *rels[] = {"bvsle", "bvsge", "bvule", "bvuge"};
  auto add = [&](Candidate c) {
    if (loop.candidates.size() < kMaxCandidates)
      loop.candidates.push_back(std::move(c));
  };
  // Pairs of values that move in lockstep: their sum or difference keeps its
  // value from loop entry (e.g. an index counting up while a length counts
  // down).
  for (unsigned a = 0; a < loop.args.size(); ++a) {
    Sort sort = loop.argSorts[a];
    if (!loop.args[a] || sort.kind != Sort::BV || a >= loop.inits.size() ||
        !loop.inits[a])
      continue;
    for (unsigned b = a + 1; b < loop.args.size(); ++b) {
      if (!loop.args[b] || !(loop.argSorts[b] == sort) ||
          b >= loop.inits.size() || !loop.inits[b])
        continue;
      for (const char *op : {"bvadd", "bvsub"}) {
        Candidate c{a, op, b,
                    std::string("(") + op + " " + *loop.inits[a] + " " +
                        *loop.inits[b] + ")"};
        c.relational = true;
        add(std::move(c));
      }
    }
  }
  for (unsigned a = 0; a < loop.args.size(); ++a) {
    Sort sort = loop.argSorts[a];
    if (!loop.args[a] || sort.kind != Sort::BV)
      continue;
    SmallVector<std::string> others;
    auto addOther = [&](const std::string &t) {
      if (t != *loop.args[a] && !llvm::is_contained(others, t))
        others.push_back(t);
    };
    addOther(bvConst(APInt(sort.width, 0)));
    for (unsigned i = 0; i < loop.inits.size(); ++i)
      if (loop.inits[i] && loop.argSorts[i] == sort)
        addOther(*loop.inits[i]);
    for (auto &[t, s] : loop.boundTerms)
      if (s == sort)
        addOther(t);
    for (const std::string &other : others)
      for (const char *rel : rels)
        add({a, rel, std::nullopt, other});
    for (unsigned b = 0; b < loop.args.size(); ++b)
      if (b != a && loop.args[b] && loop.argSorts[b] == sort)
        for (const char *rel : rels)
          add({a, rel, b, ""});
  }
}

//===----------------------------------------------------------------------===//
// Reporting
//===----------------------------------------------------------------------===//

enum class Status { Proven, Implied, Unproven, Unreachable, SolverFailed };

struct BoundsCheckReportPass
    : impl::BoundsCheckReportBase<BoundsCheckReportPass> {
  using BoundsCheckReportBase::BoundsCheckReportBase;

  std::string z3;

  std::string header() const {
    return "(set-option :timeout " + std::to_string(timeoutMs) + ")\n";
  }

  /// Houdini: drop candidates until every remaining one is inductive. One
  /// query per loop edge asks for a model violating some candidate; every
  /// candidate that is false in that model is dropped.
  bool inferInvariants(Encoder &enc, StringRef name) const {
    for (LoopInfo &loop : enc.loops)
      generateCandidates(loop);
    for (unsigned round = 0; round < 64; ++round) {
      QueryBatch batch;
      std::vector<SmallVector<Candidate *>> owners;
      bool changed = false;
      auto addEdge = [&](LoopInfo &loop, ArrayRef<MaybeTerm> subst,
                         SmallVector<std::string> assume) {
        SmallVector<std::string> named;
        SmallVector<Candidate *> cands;
        for (Candidate &cand : loop.candidates) {
          if (!cand.alive)
            continue;
          if (MaybeTerm t = loop.render(cand, subst)) {
            named.push_back(*t);
            cands.push_back(&cand);
          } else {
            cand.alive = false;
            changed = true;
          }
        }
        if (named.empty())
          return;
        std::string all = "true";
        for (unsigned i = 0; i < named.size(); ++i)
          all = mkAnd(all, "q" + std::to_string(i));
        batch.add(assume, mkNot(all), named);
        owners.push_back(std::move(cands));
      };
      for (LoopInfo &loop : enc.loops) {
        SmallVector<std::string> assume = invariantsOf(loop.enclosing);
        assume.push_back(loop.reachIn);
        assume.push_back(loop.assumedAtEntry);
        addEdge(loop, loop.inits, assume);
        for (Edge &cont : loop.continues) {
          SmallVector<std::string> assume = invariantsOf(cont.enclosing);
          assume.push_back(cont.reach);
          assume.push_back(cont.assumed);
          addEdge(loop, cont.values, assume);
        }
      }
      if (batch.count == 0)
        return true;
      std::optional<std::vector<Reply>> replies = runZ3(
          z3,
          header() + enc.prelude + invariantDefinitions(enc.loops) + batch.text,
          batch.count, dumpDir, (name + ".houdini" + Twine(round)).str());
      if (!replies)
        return false;
      for (auto [cands, reply] : llvm::zip(owners, *replies)) {
        if (reply.answer == Answer::Unsat)
          continue;
        // Drop the candidates the counterexample falsifies. Without a usable
        // model (unknown, or no value is false) drop them all to guarantee
        // progress; dropping candidates is always sound.
        bool dropAll =
            reply.answer != Answer::Sat ||
            llvm::none_of(reply.values, [](auto &kv) { return !kv.second; });
        for (auto [i, cand] : llvm::enumerate(cands)) {
          auto it = reply.values.find(i);
          if (cand->alive &&
              (dropAll || (it != reply.values.end() && !it->second))) {
            cand->alive = false;
            changed = true;
          }
        }
      }
      if (!changed)
        return true;
    }
    // No fixpoint within the round limit: drop everything to stay sound.
    for (LoopInfo &loop : enc.loops)
      for (Candidate &cand : loop.candidates)
        cand.alive = false;
    return true;
  }

  struct FunctionReport {
    StringRef name;
    struct Result {
      StringRef kind;
      std::optional<ObligationLocation> location;
      Status status;
      /// With `explain`: the unknowns an unproven condition depends on.
      SmallVector<std::string> unknowns;
    };
    SmallVector<Result> results;
    bool solverFailed = false;
    unsigned contractsUsed = 0;
  };

  /// Encode one function, infer its loop invariants and check its
  /// obligations. Only reads the IR, so functions can run in parallel.
  FunctionReport analyzeFunction(FuncOp func, unsigned index,
                                 const mlir::SymbolTable &symbols) const {
    FunctionReport report;
    report.name = func.getSymName();
    Encoder enc(&symbols);
    enc.annotate = explain || !dumpDir.empty();
    enc.encodeFunction(func);
    std::string dumpName = "f" + std::to_string(index);
    bool solverOk = inferInvariants(enc, dumpName);

    // Three queries per obligation: is it reachable at all (guards against
    // vacuous proofs), is it provable on its own, and with earlier ones.
    // With `explain`, the second query also asks for a counterexample: the
    // values of the unknowns the condition depends on.
    constexpr size_t kMaxShown = 8;
    std::vector<SmallVector<std::pair<std::string, std::string>>> inputs;
    QueryBatch batch;
    for (ObligationInfo &ob : enc.obligations) {
      SmallVector<std::string> assume = invariantsOf(ob.enclosing);
      assume.push_back(ob.reach);
      assume.push_back(ob.assumed);
      batch.add(assume, "true");
      SmallVector<std::string> show;
      inputs.emplace_back();
      if (explain) {
        inputs.back() = enc.unknownInputs(ob.cond);
        for (auto &[name, description] : inputs.back())
          if (show.size() < kMaxShown)
            show.push_back(name);
      }
      batch.add(assume, mkNot(ob.cond), {}, show);
      for (auto &[reach, cond] : ob.earlier)
        assume.push_back("(=> " + reach + " " + cond + ")");
      batch.add(assume, mkNot(ob.cond));
    }
    std::optional<std::vector<Reply>> replies;
    if (solverOk)
      replies = runZ3(z3,
                      header() + enc.prelude + invariantDefinitions(enc.loops) +
                          batch.text,
                      batch.count, dumpDir, dumpName + ".obligations");
    report.solverFailed = !replies;
    report.contractsUsed = enc.contractsUsed;

    for (auto [i, ob] : llvm::enumerate(enc.obligations)) {
      Status status = Status::SolverFailed;
      if (replies) {
        if ((*replies)[3 * i].answer == Answer::Unsat)
          status = Status::Unreachable;
        else if ((*replies)[3 * i + 1].answer == Answer::Unsat)
          status = Status::Proven;
        else if (!ob.earlier.empty() &&
                 (*replies)[3 * i + 2].answer == Answer::Unsat)
          status = Status::Implied;
        else
          status = Status::Unproven;
      }
      SmallVector<std::string> unknowns;
      if (explain && status == Status::Unproven)
        for (auto &[name, description] : inputs[i]) {
          std::string line = description;
          if (std::optional<std::string> value =
                  modelValue((*replies)[3 * i + 1].text, name))
            line += "  (= " + *value + " in a counterexample)";
          unknowns.push_back(line);
        }
      report.results.push_back({ob.kind, ob.location, status, unknowns});
    }
    return report;
  }

  void runOnOperation() override {
    if (!z3Path.empty()) {
      z3 = z3Path;
    } else if (auto found = llvm::sys::findProgramByName("z3")) {
      z3 = *found;
    } else {
      getOperation().emitError("bounds-check-report: z3 not found in PATH");
      return signalPassFailure();
    }

    // Callee contracts are looked up by symbol; lookups only read the table.
    mlir::SymbolTable symbols(getOperation());

    // Functions with preconditions: calls to them are checked at the caller.
    DenseSet<Operation *> withPreconditions;
    getOperation().walk([&](FuncOp func) {
      if (!func->getRegion(0).empty() &&
          llvm::any_of(func->getRegion(0).front(), [](Operation &op) {
            auto ob = dyn_cast<ObligationOp>(&op);
            return ob && isOwnPrecondition(ob);
          }))
        withPreconditions.insert(func);
    });
    auto hasWork = [&](Operation *op) {
      if (isa<ObligationOp>(op))
        return true;
      auto call = dyn_cast<CallOp>(op);
      auto callee =
          call ? dyn_cast<SymbolConstantAttr>(call.getCallee()) : nullptr;
      return callee && withPreconditions.contains(symbols.lookup(
                           callee.getSymbol().getRootReference()));
    };

    SmallVector<FuncOp> funcs;
    getOperation().walk([&](FuncOp func) {
      if (!includeStdlib && func.getSymName().starts_with("std::"))
        return;
      if (func.walk([&](Operation *op) {
                return hasWork(op) ? WalkResult::interrupt()
                                   : WalkResult::advance();
              })
              .wasInterrupted())
        funcs.push_back(func);
    });

    // Each function spends its time in external solver processes; run them
    // concurrently and print the reports in module order afterwards.
    std::vector<FunctionReport> reports(funcs.size());
    mlir::parallelFor(&getContext(), 0, funcs.size(), [&](size_t i) {
      reports[i] = analyzeFunction(funcs[i], i, symbols);
    });

    unsigned total = 0, proven = 0, failed = 0, unreachable = 0;
    llvm::raw_ostream &os = llvm::errs();
    os << "\n=== bounds-check-report ===\n";
    for (FunctionReport &report : reports) {
      unsigned funcProven = 0, funcImplied = 0, funcTotal = 0,
               funcUnreachable = 0;
      for (auto &[kind, location, status, unknowns] : report.results) {
        if (status == Status::Implied) {
          ++funcImplied;
          continue;
        }
        if (status == Status::Unreachable) {
          ++funcUnreachable;
          continue;
        }
        ++funcTotal;
        funcProven += status == Status::Proven;
        failed += status == Status::SolverFailed;
      }
      total += funcTotal;
      proven += funcProven;
      unreachable += funcUnreachable;

      os << "@" << report.name << ": " << funcProven << "/" << funcTotal
         << " obligations proven";
      if (funcImplied)
        os << " (+" << funcImplied << " implied by earlier obligations)";
      if (funcUnreachable)
        os << " (+" << funcUnreachable << " unreachable)";
      if (report.contractsUsed)
        os << " (" << report.contractsUsed << " callee contracts applied)";
      if (report.solverFailed)
        os << "  [solver failed]";
      os << "\n";
      for (auto &[kind, location, status, unknowns] : report.results) {
        if (!verbose && (status == Status::Proven || status == Status::Implied))
          continue;
        os << "  "
           << (status == Status::Proven        ? "proven  "
               : status == Status::Implied     ? "implied "
               : status == Status::Unreachable ? "unreach "
                                               : "UNPROVEN")
           << "  " << kind << "  ";
        if (location)
          os << location->file << ":" << location->line << ":" << location->col;
        else
          os << "<unresolved location>";
        os << "\n";
        constexpr size_t kMaxUnknowns = 8;
        for (auto [n, unknown] : llvm::enumerate(unknowns)) {
          if (n == kMaxUnknowns) {
            os << "      ... and " << unknowns.size() - n << " more\n";
            break;
          }
          os << "      depends on unknown " << unknown << "\n";
        }
      }
    }

    os << "total: " << proven << "/" << total << " obligations proven";
    if (failed)
      os << " (" << failed << " not checked: solver failed)";
    if (unreachable)
      os << " (" << unreachable << " unreachable, not counted)";
    os << "\n\n";
    markAllAnalysesPreserved();
  }
};

} // namespace
