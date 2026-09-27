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
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"

#include <deque>

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
  enum Kind { None, Bool, BV } kind = None;
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

/// A Houdini candidate `arg <rel> other` for a loop head.
struct Candidate {
  unsigned arg;
  std::string rel; // SMT-LIB comparison, e.g. "bvsle"
  /// Either another loop argument or a term available at the loop head.
  std::optional<unsigned> otherArg;
  std::string otherTerm;
  bool alive = true;
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
  unsigned counter = 0;
  DenseMap<Value, std::string> terms;
  DenseMap<std::pair<Value, unsigned>, std::string> extracts;
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
    DenseMap<Value, Value> args;
    DenseMap<Value, std::string> terms;
    DenseMap<std::pair<Value, unsigned>, std::string> extracts;
  };
  std::deque<CallContext> contexts;
  /// The callee context being evaluated, or null for the function itself.
  CallContext *ctx = nullptr;
  /// Call results backed by a callee's returned value in a context.
  DenseMap<Value, std::pair<CallContext *, Value>> callResults;

  DenseMap<Value, std::string> &termMap() { return ctx ? ctx->terms : terms; }
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

  std::string declare(Sort sort, StringRef prefix = "h") {
    std::string name = fresh(prefix);
    prelude += "(declare-const " + name + " " + sort.str() + ")\n";
    return name;
  }

  std::string define(Sort sort, StringRef expr, StringRef prefix = "v") {
    std::string name = fresh(prefix);
    prelude +=
        ("(define-fun " + name + " () " + sort.str() + " " + expr + ")\n")
            .str();
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
    if (ctx) {
      // A callee argument is the caller's operand.
      auto arg = ctx->args.find(value);
      if (arg != ctx->args.end())
        return inContext(nullptr, [&] { return term(arg->second); });
    } else {
      // A call result backed by the callee's returned value.
      auto res = callResults.find(value);
      if (res != callResults.end() && !terms.count(value)) {
        auto [context, returned] = res->second;
        MaybeTerm t = inContext(context, [&] { return term(returned); });
        if (t)
          terms[value] = *t;
        return t;
      }
    }
    auto &map = termMap();
    auto it = map.find(value);
    if (it != map.end())
      return it->second;
    // Callee values are computed on demand; the function's own values are
    // encoded in program order.
    if (ctx)
      if (Operation *def = value.getDefiningOp()) {
        encodeOp(def);
        auto it = map.find(value);
        if (it != map.end())
          return it->second;
      }
    std::string name = declare(sort);
    map[value] = name;
    return name;
  }

  std::string boolTerm(Value value) {
    MaybeTerm t = term(value);
    if (t && sortOf(value.getType()).kind == Sort::Bool)
      return *t;
    return declare({Sort::Bool, 1, false});
  }

  void setTerm(Value value, StringRef expr) {
    termMap()[value] = define(sortOf(value.getType()), expr);
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

  /// The scalar reached from `aggregate` by the accesses in `path` (struct
  /// field indices or `kUnwrap`), following the ops that built the aggregate:
  /// `kgen.struct.create`, `pop.union.wrap` and `pop.select`. This keeps
  /// values visible through e.g. the `Optional` returned by iterators.
  MaybeTerm resolveAccess(Value aggregate, ArrayRef<int> path, Sort sort,
                          unsigned depth = 0) {
    if (path.empty()) {
      if (!(sortOf(aggregate.getType()) == sort))
        return std::nullopt;
      return term(aggregate);
    }
    if (depth > 16)
      return std::nullopt;
    if (ctx) {
      auto arg = ctx->args.find(aggregate);
      if (arg != ctx->args.end())
        return inContext(nullptr, [&] {
          return resolveAccess(arg->second, path, sort, depth + 1);
        });
    } else {
      auto res = callResults.find(aggregate);
      if (res != callResults.end()) {
        auto [context, returned] = res->second;
        return inContext(context, [&] {
          return resolveAccess(returned, path, sort, depth + 1);
        });
      }
    }
    Operation *def = aggregate.getDefiningOp();
    if (!def)
      return opaqueField(aggregate, path, sort);
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
    if (auto select = dyn_cast<POP::SelectOp>(def)) {
      if (sortOf(select.getCondition().getType()).kind != Sort::Bool)
        return std::nullopt;
      MaybeTerm t = resolveAccess(select.getTrueValue(), path, sort, depth + 1);
      MaybeTerm f =
          resolveAccess(select.getFalseValue(), path, sort, depth + 1);
      if (!t && !f)
        return std::nullopt;
      return "(ite " + boolTerm(select.getCondition()) + " " +
             (t ? *t : declare(sort)) + " " + (f ? *f : declare(sort)) + ")";
    }
    return opaqueField(aggregate, path, sort);
  }

  /// A field of an aggregate whose construction is not visible (a block
  /// argument, a call result, a load, ...): the same term every access to that
  /// field of that SSA value gets, so e.g. a callee's view of its argument
  /// agrees with the caller's view of the operand.
  std::optional<std::string> opaqueField(Value aggregate, ArrayRef<int> path,
                                         Sort sort) {
    if (path.size() != 1 || path.front() < 0)
      return std::nullopt;
    auto &memo = extractMap();
    auto key = std::make_pair(aggregate, unsigned(path.front()));
    auto it = memo.find(key);
    if (it == memo.end())
      it = memo.try_emplace(key, declare(sort)).first;
    return it->second;
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
      if (name == "pop.shl" || name == "pop.shr" || name == "pop.floordiv") {
        SmallVector<std::string> ts = operandTerms();
        std::optional<APInt> rhs = constantInt(op->getOperand(1), sort.width);
        if (ts.size() != 2 || !rhs)
          return;
        APInt amount = *rhs;
        if (name == "pop.floordiv") {
          if (amount.isZero() || (sort.isSigned && amount.isAllOnes()))
            return;
          if (!sort.isSigned) {
            setTerm(result, "(bvudiv " + ts[0] + " " + ts[1] + ")");
            return;
          }
          std::string zero = bvConst(APInt(sort.width, 0));
          std::string q = "(bvsdiv " + ts[0] + " " + ts[1] + ")";
          std::string r = "(bvsrem " + ts[0] + " " + ts[1] + ")";
          setTerm(result, "(ite (and (distinct " + r + " " + zero +
                              ") (xor (bvslt " + ts[0] + " " + zero +
                              ") (bvslt " + ts[1] + " " + zero + "))) (bvsub " +
                              q + " " + bvConst(APInt(sort.width, 1)) + ") " +
                              q + ")");
          return;
        }
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

    if (auto select = dyn_cast<POP::SelectOp>(op)) {
      if (sortOf(select.getCondition().getType()).kind != Sort::Bool)
        return;
      MaybeTerm c = term(select.getCondition());
      MaybeTerm t = term(select.getTrueValue());
      MaybeTerm f = term(select.getFalseValue());
      if (c && t && f && sortOf(select.getTrueValue().getType()) == sort)
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
      if (it == memo.end())
        it = memo.try_emplace(key, declare(sort)).first;
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
    if (auto call = dyn_cast<CallOp>(op))
      instantiateContracts(call, reach);
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
    if (preconditions.empty() && postconditions.empty())
      return;

    CallContext &context = contexts.emplace_back();
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
    if (postconditions.empty())
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
  void add(ArrayRef<std::string> assumptions, StringRef goalNegation,
           ArrayRef<std::string> named = {}) {
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
    text += "(pop 1)\n";
    ++count;
  }
};

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
  constexpr size_t kMaxCandidates = 256;
  static const char *rels[] = {"bvsle", "bvsge", "bvule", "bvuge"};
  auto add = [&](Candidate c) {
    if (loop.candidates.size() < kMaxCandidates)
      loop.candidates.push_back(std::move(c));
  };
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
    enc.encodeFunction(func);
    std::string dumpName = "f" + std::to_string(index);
    bool solverOk = inferInvariants(enc, dumpName);

    // Three queries per obligation: is it reachable at all (guards against
    // vacuous proofs), is it provable on its own, and with earlier ones.
    QueryBatch batch;
    for (ObligationInfo &ob : enc.obligations) {
      SmallVector<std::string> assume = invariantsOf(ob.enclosing);
      assume.push_back(ob.reach);
      assume.push_back(ob.assumed);
      batch.add(assume, "true");
      batch.add(assume, mkNot(ob.cond));
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
      report.results.push_back({ob.kind, ob.location, status});
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
      for (auto &[kind, location, status] : report.results) {
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
      for (auto &[kind, location, status] : report.results) {
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
