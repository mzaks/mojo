//===- VerifyContracts.cpp - Modular contract verification on LIT ---------===//
//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Verifies each `lit.fn` once, before elaboration and inlining, against the
// `where` contracts of the functions it calls (see
// Mojo/proposals/modular-verification.md):
//
// - the function's own `kgen.requires` clauses are assumed at its entry;
// - at every `lit.call` to a callee with `kgen.requires` clauses, the clauses,
//   instantiated with the call's arguments, are obligations at the call;
// - nothing else about a callee is used: its results are unknowns, and memory
//   reachable through its `mut` reference arguments is unknown after the call.
//
// Integer and Boolean operators (`SIMD` methods on a width-1 integer dtype,
// `Bool` conversions) are modelled directly as bit-vector and Boolean
// operations, and `len(x)` as an uninterpreted function of `x`'s value.
// Local variables are tracked through `lit.var.decl`, `lit.ref.store` and
// `lit.ref.load`, fields through `lit.ref.struct.ger`, and `ref` locals to
// what they refer to. Control flow: `hlcf.if`, `hlcf.return`, `lit.try`, and
// `hlcf.loop`, whose invariants are found with Houdini over small templates
// (bounds against 0, values before the loop, lengths, and other loop
// variables). `range` iteration is modelled as the stdlib defines it.
// Anything else (comptime control flow, loops with loop-carried values) is
// not analyzed yet: memory is unknown after it, and the obligations inside
// it are reported as not analyzed.
//
// Obligations are answered by an external `z3` process, one per function,
// with a deterministic resource limit per query and a wall-clock cap on the
// process. Results are reported as MLIR diagnostics on the calls.
//
//===----------------------------------------------------------------------===//

#include "Mojo/ToolCommon/KGENPasses.h"

#include "Mojo/HLCFDialect/HLCFOps.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/LITDialect/LITOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Regex.h"

#include <map>
#include <optional>
#include <set>

using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_VERIFYCONTRACTS
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {

using MaybeTerm = std::optional<std::string>;

/// An SMT sort: a bit-vector of some width, or Bool.
struct Sort {
  bool isBool = false;
  unsigned width = 64;
  bool isSigned = true;

  std::string str() const {
    return isBool ? "Bool" : "(_ BitVec " + std::to_string(width) + ")";
  }
};

std::string printed(Type type) {
  std::string text;
  llvm::raw_string_ostream os(text);
  type.print(os);
  return text;
}

std::string printed(Attribute attr) {
  std::string text;
  llvm::raw_string_ostream os(text);
  attr.print(os);
  return text;
}

/// The integer dtype named in printed IR (`dtype index`, `dtype = si32`), as
/// a sort.
std::optional<Sort> dtypeSort(StringRef text) {
  static llvm::Regex re("dtype (= )?([a-z0-9]+)");
  SmallVector<StringRef> m;
  if (!re.match(text, &m))
    return std::nullopt;
  StringRef name = m[2];
  if (name == "index")
    return Sort{false, 64, true};
  if (name == "uindex")
    return Sort{false, 64, false};
  bool isSigned = name.consume_front("si");
  if (!isSigned && !name.consume_front("ui"))
    return std::nullopt;
  unsigned width;
  if (name.getAsInteger(10, width) || !llvm::is_contained({8u, 16u, 32u, 64u},
                                                          width))
    return std::nullopt;
  return Sort{false, width, isSigned};
}

/// Whether printed parameter or type text names SIMD width 1.
bool isWidthOne(StringRef text) {
  return text.contains("{1}") || text.contains("_mlir_value = 1}");
}

/// The sort of a value of `type`: an integer (`Int`, `SIMD` of an integer
/// dtype and width 1, `index`, `kgen.scalar<index>`), a Boolean, or else an
/// opaque 64-bit handle (structs, references).
Sort sortOf(Type type) {
  std::string text = printed(type);
  StringRef t(text);
  if (t == "!lit.struct<@std::@builtin::@bool::@Bool>" ||
      t == "!kgen.scalar<bool>" || t == "i1")
    return {true, 1, false};
  if (t == "index")
    return {false, 64, true};
  // `Int` and other scalars, possibly behind a comptime alias.
  t.consume_front("!kgen.param<:meta<");
  if (t.starts_with("!lit.struct<@std::@simd::@SIMD<") && isWidthOne(t))
    if (std::optional<Sort> sort = dtypeSort(t))
      return *sort;
  if (t.starts_with("!kgen.scalar<"))
    if (std::optional<Sort> sort =
            dtypeSort(("dtype " + t.drop_front(13).drop_back(1)).str()))
      return *sort;
  return {false, 64, false};
}

std::string bvConst(int64_t value, unsigned width) {
  APInt bits(width, static_cast<uint64_t>(value), /*isSigned=*/true);
  return "(_ bv" + llvm::toString(bits, 10, /*Signed=*/false) + " " +
         std::to_string(width) + ")";
}

/// `callee`'s symbol as `a::b::c`, and its parameter values as text.
struct CalleeName {
  std::string path;
  SmallVector<std::string> params;
};

std::optional<CalleeName> calleeName(LIT::CallOp call) {
  auto symbol = dyn_cast<SymbolConstantAttr>(call.getCallee());
  if (!symbol)
    return std::nullopt;
  CalleeName name;
  SymbolRefAttr ref = symbol.getSymbol();
  name.path = ref.getRootReference().str();
  for (FlatSymbolRefAttr nested : ref.getNestedReferences())
    name.path += "::" + nested.getValue().str();
  for (TypedAttr param : symbol.getParamValues())
    name.params.push_back(printed(param));
  return name;
}

/// How a function is named in source: `List.__getitem__`, `ok_get`.
std::string displayName(LIT::FnOp fn) {
  std::string name;
  if (auto source = fn->getAttrOfType<StringAttr>("sourceName"))
    name = source.getValue().str();
  else if (fn.getSymName())
    name = fn.getSymName()->str();
  if (Operation *parent = fn->getParentOp())
    if (auto parentName = parent->getAttrOfType<StringAttr>(
            SymbolTable::getSymbolAttrName());
        parentName && !isa<LIT::PackageOp>(parent) &&
        parent->getName().getStringRef() != "lit.file_module")
      name = (parentName.getValue() + "." + name).str();
  return name;
}

//===----------------------------------------------------------------------===//
// Solver
//===----------------------------------------------------------------------===//

enum class Answer { Proven, Unproven, Unknown, NotAnalyzed };

/// Solver settings for one run of the pass.
struct SolverConfig {
  std::string z3;
  unsigned rlimit = 100000000;
  unsigned wallSeconds = 60;
  std::string dumpDir;
};

/// Runs z3 on `script` with a wall-clock cap; returns one answer per query
/// (queries are separated by `(echo "@@")`).
std::optional<SmallVector<Answer>> runZ3(const SolverConfig &config,
                                         StringRef script, StringRef dumpName) {
  SmallString<128> scriptPath, outPath;
  if (!config.dumpDir.empty()) {
    scriptPath = config.dumpDir;
    llvm::sys::path::append(scriptPath, dumpName + ".smt2");
  } else if (llvm::sys::fs::createTemporaryFile("verify", "smt2", scriptPath)) {
    return std::nullopt;
  }
  if (llvm::sys::fs::createTemporaryFile("verify", "out", outPath))
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
  StringRef z3 = config.z3;
  (void)llvm::sys::ExecuteAndWait(z3, {z3, "-smt2", scriptPath}, std::nullopt,
                                  redirects, config.wallSeconds);
  auto buffer = llvm::MemoryBuffer::getFile(outPath);
  llvm::sys::fs::remove(outPath);
  if (config.dumpDir.empty())
    llvm::sys::fs::remove(scriptPath);
  if (!buffer)
    return std::nullopt;
  // A process stopped at the cap answers nothing for the remaining queries,
  // which count as unknown.
  SmallVector<Answer> answers;
  SmallVector<StringRef> segments;
  (*buffer)->getBuffer().split(segments, "@@");
  for (StringRef segment : ArrayRef(segments).drop_front()) {
    StringRef answer = segment.trim().split('\n').first.trim();
    answers.push_back(answer == "unsat" ? Answer::Proven
                      : answer == "sat" ? Answer::Unproven
                                        : Answer::Unknown);
  }
  return answers;
}

/// One `(push) ... (check-sat) (pop)` block asking whether `goal` follows
/// from `assumptions`.
std::string query(ArrayRef<std::string> assumptions, StringRef goal,
                  unsigned rlimit) {
  std::string text = "(echo \"@@\")\n(push 1)\n";
  for (const std::string &a : assumptions)
    text += "(assert " + a + ")\n";
  text += ("(assert (not " + goal + "))\n(set-option :rlimit " +
           Twine(rlimit) + ")\n(check-sat)\n(set-option :rlimit 0)\n(pop 1)\n")
              .str();
  return text;
}

//===----------------------------------------------------------------------===//
// Encoding
//===----------------------------------------------------------------------===//

/// A place in memory: a root (a function argument or a local variable) and
/// a path of fields, each `/` followed by the field's printed attributes.
struct Loc {
  Value root;
  std::string path;
  bool operator<(const Loc &other) const {
    if (root.getAsOpaquePointer() != other.root.getAsOpaquePointer())
      return root.getAsOpaquePointer() < other.root.getAsOpaquePointer();
    return path < other.path;
  }
  bool operator==(const Loc &other) const {
    return root == other.root && path == other.path;
  }
};

/// What is known on a path through the function.
struct State {
  /// The path condition.
  std::string pc = "true";
  /// Values of places written on this path, or made unknown.
  std::map<Loc, std::string> env;
  /// Places a `ref` local refers to, by the local.
  std::map<Loc, Loc> refs;
  /// Places not in `env` (nor under a place in `env`) hold their value at
  /// entry if `epoch` is 0, and an unknown per place and epoch otherwise
  /// (after code whose writes are not tracked).
  unsigned epoch = 0;
  /// False once the path has left (returned, broken out, continued, raised).
  bool alive = true;
  /// The values of the region's terminator, when it yields.
  SmallVector<std::string> yields;
};

/// An obligation: `cond` must hold whenever `pc` does.
struct Obligation {
  std::string pc, cond;
  Location callLoc, clauseLoc;
  std::string callee;
  bool analyzed = true;
  /// The function's own postcondition, at a return (`callLoc`).
  bool postcondition = false;
};

/// A loop being walked: where its iterations continue and exit, and what
/// its body reads.
struct LoopFrame {
  StringRef label;
  SmallVector<State> continues, breaks;
  std::set<Loc> loaded;
  SmallVector<std::string> lengths;
  /// Conditions in the body (branches, comparisons, obligations): the
  /// invariants worth finding are about what they depend on.
  SmallVector<std::string> conditions;
};

/// A `lit.try` being walked: the paths that raise into its `except` region.
struct TryFrame {
  StringRef label;
  SmallVector<State> raises;
};

/// A candidate loop invariant: `lhs op rhs`, where `lhs` is a place and
/// `rhs` a place (`rhsLoc`) or a term that does not change in the loop.
struct Candidate {
  Loc lhs;
  std::string op;
  std::optional<Loc> rhsLoc;
  std::string rhsTerm;
  /// The candidate is about `len(lhs)`, not `lhs`.
  bool length = false;
};

class FunctionEncoder {
public:
  FunctionEncoder(LIT::FnOp fn, ModuleOp module,
                  SymbolTableCollection &symbols, const SolverConfig &solver,
                  std::string dumpPrefix)
      : fn(fn), module(module), symbols(symbols), solver(solver),
        dumpPrefix(std::move(dumpPrefix)) {}

  /// Encodes the function; returns false if it has no body.
  bool encode() {
    Region &body = fn.getFunctionBody();
    if (body.empty())
      return false;
    Block &entry = body.front();
    // Arguments: integers and Booleans are unknowns of their sort; references
    // are places, whose values at entry are unknowns.
    for (BlockArgument arg : entry.getArguments()) {
      if (isa<LIT::RefType>(arg.getType()))
        roots.push_back(arg);
      else
        values[arg] = declare(sortOf(arg.getType()), "a");
    }
    fn.walk([&](LIT::VarDeclOp decl) { roots.push_back(decl->getResult(0)); });
    State state;
    // The function's own preconditions hold at its entry.
    for (Operation &op : entry)
      if (auto req = dyn_cast<RequiresOp>(&op))
        if (MaybeTerm cond = instantiate(req.getBody(), req.getArgs(), state,
                                         {}, {}, nullptr, /*assumed=*/true))
          facts.push_back(*cond);
    entryState = state;
    walkBlock(entry, state);
    return true;
  }

  std::string script() const {
    std::string text = header();
    for (const Obligation &ob : obligations)
      if (ob.analyzed)
        text += query({ob.pc}, ob.cond, solver.rlimit);
    return text;
  }

  LIT::FnOp fn;
  SmallVector<Obligation> obligations;
  unsigned invariantsFound = 0, loopsAnalyzed = 0;

private:
  ModuleOp module;
  SymbolTableCollection &symbols;
  const SolverConfig &solver;
  std::string dumpPrefix;
  unsigned houdiniRuns = 0;

  DenseMap<Value, std::string> values;
  /// Fields of struct values the encoder builds (`range`s), by path.
  DenseMap<Value, std::map<std::string, std::string>> records;
  /// Places that reference values loaded from `ref` locals refer to.
  DenseMap<Value, Loc> derivedPlaces;
  std::map<Loc, std::string> entryValues;
  std::map<std::pair<Loc, unsigned>, std::string> epochValues;
  std::map<std::pair<std::string, std::string>, std::string> fieldValues;
  /// The struct value and field path each field value belongs to.
  std::map<std::string, std::pair<std::string, std::string>> fieldOwners;
  DenseMap<Value, Loc> refArgs; // Contract block arguments bound to places.
  /// The function's reference arguments and local variables.
  SmallVector<Value> roots;
  std::string prelude;
  SmallVector<std::string> facts;
  std::map<std::string, Sort> sorts;
  std::map<std::string, SmallVector<std::string>> deps;
  std::map<std::string, std::string> definitions;
  std::map<std::pair<std::string, std::string>, bool> dependsMemo;
  /// The state at the function's `kgen.contract.entry`, for its `old`s.
  std::optional<State> entryState;
  /// While evaluating a contract: where its `kgen.old`s look, whether its
  /// quantifiers are assumed, and the caller's values of the callee's
  /// arguments.
  State *oldState = nullptr;
  bool assuming = false;
  DenseMap<Value, Value> actuals;
  /// The call whose callee's postcondition is being assumed: its result.
  Value callResult;
  /// Element references `__getitem__` returned: the list's place and the
  /// index.
  DenseMap<Value, std::pair<Loc, std::string>> elements;
  std::set<std::string> elementFunctions;
  unsigned counter = 0;
  bool lenDeclared = false;
  /// Inside a contract region: calls there are evaluated, not checked.
  unsigned inContract = 0;
  SmallVector<LoopFrame *> loops;
  SmallVector<TryFrame *> tries;

  std::string header() const {
    std::string text = "; " + displayName(fn) + "\n" +
                       "(set-option :print-success false)\n" + prelude;
    for (const std::string &fact : facts)
      text += "(assert " + fact + ")\n";
    return text;
  }

  std::string declare(Sort sort, StringRef prefix = "u") {
    std::string name = (prefix + Twine(counter++)).str();
    prelude += "(declare-const " + name + " " + sort.str() + ")\n";
    sorts[name] = sort;
    return name;
  }

  std::string define(Sort sort, StringRef expr, StringRef prefix = "t") {
    std::string name = (prefix + Twine(counter++)).str();
    prelude += ("(define-fun " + name + " () " + sort.str() + " " + expr +
                ")\n")
                   .str();
    sorts[name] = sort;
    deps[name] = namesIn(expr);
    definitions[name] = expr.str();
    return name;
  }

  /// The names of terms (`t12`, `h3`) used in an expression.
  static SmallVector<std::string> namesIn(StringRef expr) {
    SmallVector<std::string> names;
    for (size_t i = 0; i < expr.size();) {
      if (llvm::isAlpha(expr[i]) && (i == 0 || !llvm::isAlnum(expr[i - 1]))) {
        size_t j = i + 1;
        while (j < expr.size() && llvm::isDigit(expr[j]))
          ++j;
        if (j > i + 1 && (j == expr.size() || !llvm::isAlnum(expr[j])))
          names.push_back(expr.slice(i, j).str());
        i = j;
        continue;
      }
      ++i;
    }
    return names;
  }

  /// The names `roots` depend on, through definitions.
  llvm::StringSet<> closure(ArrayRef<std::string> roots) const {
    llvm::StringSet<> seen;
    SmallVector<std::string> work;
    for (const std::string &root : roots)
      for (std::string &name : namesIn(root))
        work.push_back(std::move(name));
    while (!work.empty()) {
      std::string name = work.pop_back_val();
      if (!seen.insert(name).second)
        continue;
      if (auto it = deps.find(name); it != deps.end())
        work.append(it->second.begin(), it->second.end());
    }
    return seen;
  }

  Sort sortOfTerm(StringRef term) const {
    if (auto it = sorts.find(term.str()); it != sorts.end())
      return it->second;
    if (term == "true" || term == "false")
      return {true, 1, false};
    unsigned width;
    if (term.consume_front("(_ bv")) {
      term = term.drop_until([](char c) { return c == ' '; }).trim();
      if (!term.drop_back(1).getAsInteger(10, width))
        return {false, width, true};
    }
    return {false, 64, false};
  }

  std::string lenOf(StringRef handle) {
    if (!lenDeclared) {
      prelude += "(declare-fun len ((_ BitVec 64)) (_ BitVec 64))\n";
      lenDeclared = true;
    }
    std::string term = ("(len " + handle + ")").str();
    // Assumption: every `Sized` type in the stdlib has a non-negative length.
    facts.push_back("(bvsge " + term + " " + bvConst(0, 64) + ")");
    for (LoopFrame *frame : loops)
      frame->lengths.push_back(term);
    return term;
  }

  //===--------------------------------------------------------------------===//
  // Places
  //===--------------------------------------------------------------------===//

  /// The place a reference value denotes.
  std::optional<Loc> placeOf(Value ref) {
    if (auto it = refArgs.find(ref); it != refArgs.end())
      return it->second;
    if (auto it = derivedPlaces.find(ref); it != derivedPlaces.end())
      return it->second;
    if (isa<BlockArgument>(ref)) {
      if (ref.getParentRegion() == &fn.getFunctionBody())
        return Loc{ref, ""};
      return std::nullopt;
    }
    Operation *def = ref.getDefiningOp();
    if (isa<LIT::VarDeclOp>(def))
      return Loc{ref, ""};
    if (isa<LIT::RefImmutOp>(def) || isa<RebindOp>(def))
      return placeOf(def->getOperand(0));
    if (auto gep = dyn_cast<LIT::RefStructGEROp>(def)) {
      std::optional<Loc> base = placeOf(gep->getOperand(0));
      if (!base)
        return std::nullopt;
      // `/` separates fields: printed field attributes contain no `/`.
      base->path += "/" + printed(gep->getAttrDictionary());
      return base;
    }
    return std::nullopt;
  }

  Type placeType(const Loc &loc) {
    if (auto ref = dyn_cast<LIT::RefType>(loc.root.getType()))
      return ref.getElementType();
    return loc.root.getType();
  }

  std::string load(const Loc &loc, State &state, Sort sort) {
    for (LoopFrame *frame : loops)
      frame->loaded.insert(loc);
    if (auto it = state.env.find(loc); it != state.env.end())
      return it->second;
    // A field of a place whose value is known: a function of that value.
    StringRef path(loc.path);
    while (!path.empty()) {
      path = path.take_front(path.rfind('/'));
      auto it = state.env.find(Loc{loc.root, path.str()});
      if (it == state.env.end())
        continue;
      std::string rest = StringRef(loc.path).drop_front(path.size()).str();
      return fieldOf(it->second, rest, sort);
    }
    // A field of a root whose value is unknown: a function of that value.
    if (!loc.path.empty())
      return fieldOf(load(Loc{loc.root, ""}, state, Sort{false, 64, false}),
                     loc.path, sort);
    if (state.epoch) {
      auto [it, inserted] = epochValues.try_emplace({loc, state.epoch}, "");
      if (inserted)
        it->second = declare(sort, "h");
      return it->second;
    }
    auto [it, inserted] = entryValues.try_emplace(loc, "");
    if (inserted)
      it->second = declare(sort, "e");
    return it->second;
  }

  /// The value of field path `rest` of the struct value `value`.
  std::string fieldOf(const std::string &value, const std::string &rest,
                      Sort sort) {
    auto [field, inserted] = fieldValues.try_emplace({value, rest}, "");
    if (inserted) {
      field->second = declare(sort, "f");
      fieldOwners[field->second] = {value, rest};
    }
    return field->second;
  }

  std::string load(const Loc &loc, State &state, Type type) {
    return load(loc, state, sortOf(type));
  }

  void store(const Loc &loc, State &state, std::string value,
             Value stored = {}) {
    // Writing a place replaces the values of its fields...
    for (auto it = state.env.begin(); it != state.env.end();) {
      if (it->first.root == loc.root &&
          StringRef(it->first.path).starts_with(loc.path + "/"))
        it = state.env.erase(it);
      else
        ++it;
    }
    // ... and changes the values of the places containing it (`x` when `x.f`
    // is written), which become unknown.
    StringRef path(loc.path);
    while (!path.empty()) {
      path = path.take_front(path.rfind('/'));
      state.env[Loc{loc.root, path.str()}] = declare({false, 64, false}, "h");
    }
    state.env[loc] = std::move(value);
    state.refs.erase(loc);
    if (stored) {
      if (auto it = records.find(stored); it != records.end())
        for (auto &[field, term] : it->second)
          state.env[Loc{loc.root, loc.path + field}] = term;
      if (isa<LIT::RefType>(stored.getType()))
        if (std::optional<Loc> target = placeOf(stored))
          state.refs[loc] = *target;
    }
  }

  /// Everything reachable through `loc` becomes unknown.
  void havoc(const Loc &loc, State &state, Type type) {
    store(loc, state, declare(sortOf(type), "h"));
  }

  /// The top-level elements of a printed `#kgen<exprs[a, b]>` list.
  static SmallVector<StringRef> topLevelElements(StringRef text) {
    SmallVector<StringRef> elements;
    size_t open = text.find('[');
    if (open == StringRef::npos)
      return elements;
    int depth = 0;
    size_t start = open + 1;
    for (size_t i = start; i < text.size(); ++i) {
      char c = text[i];
      if (c == '<' || c == '[' || c == '(' || c == '{') {
        ++depth;
      } else if (c == '>' || c == ']' || c == ')' || c == '}') {
        if (depth == 0) {
          StringRef element = text.slice(start, i).trim();
          if (!element.empty())
            elements.push_back(element);
          break;
        }
        --depth;
      } else if (c == ',' && depth == 0) {
        elements.push_back(text.slice(start, i).trim());
        start = i + 1;
      }
    }
    return elements;
  }

  /// The name of the declaration a root's reference type's origin names
  /// (`xs`` in `#kgen.param.decl.ref<"xs`">`), quoted.
  static std::string quotedOriginName(Value root) {
    auto ref = dyn_cast<LIT::RefType>(root.getType());
    if (!ref)
      return "";
    std::string origin = printed(ref.getOrigin());
    StringRef text(origin);
    size_t pos = text.find("param.decl.ref<\"");
    if (pos == StringRef::npos)
      return "";
    text = text.drop_front(pos + strlen("param.decl.ref<"));
    size_t end = text.find('"', 1);
    if (end == StringRef::npos)
      return "";
    return text.take_front(end + 1).str();
  }

  /// The roots a mutable origin (printed) may reach; nullopt if it names none
  /// of the function's roots, so it may reach any of them.
  std::optional<SmallVector<Value>> rootsNamedBy(StringRef origin) {
    SmallVector<Value> named;
    for (Value root : roots) {
      std::string name = quotedOriginName(root);
      if (!name.empty() && origin.contains(name))
        named.push_back(root);
    }
    if (named.empty())
      return std::nullopt;
    return named;
  }

  /// Memory a mutable origin (printed) may reach becomes unknown.
  void havocOrigin(StringRef origin, State &state) {
    if (std::optional<SmallVector<Value>> named = rootsNamedBy(origin)) {
      for (Value root : *named)
        havoc(Loc{root, ""}, state, placeType(Loc{root, ""}));
      return;
    }
    havocAll(state);
  }

  void havocAll(State &state) {
    ++state.epoch;
    state.env.clear();
    state.refs.clear();
  }

  //===--------------------------------------------------------------------===//
  // Values
  //===--------------------------------------------------------------------===//

  std::string term(Value value, State &state) {
    if (auto it = values.find(value); it != values.end())
      return it->second;
    // A value from outside the region being evaluated (e.g. a constant).
    if (Operation *def = value.getDefiningOp())
      if (isa<ParamConstantOp>(def)) {
        evalOp(def, state);
        if (auto it = values.find(value); it != values.end())
          return it->second;
      }
    std::string fresh = declare(sortOf(value.getType()));
    values[value] = fresh;
    return fresh;
  }

  /// The value a reference refers to, for a call argument passed by
  /// reference (e.g. `len(xs)`).
  std::string valueThrough(Value ref, State &state) {
    if (std::optional<Loc> loc = placeOf(ref))
      return load(*loc, state,
                  cast<LIT::RefType>(ref.getType()).getElementType());
    return declare({false, 64, false}, "h");
  }

  void setResultsUnknown(Operation *op) {
    for (Value result : op->getResults())
      values[result] = declare(sortOf(result.getType()));
  }

  void noteCondition(StringRef cond) {
    for (LoopFrame *frame : loops)
      frame->conditions.push_back(cond.str());
  }

  //===--------------------------------------------------------------------===//
  // Merging paths
  //===--------------------------------------------------------------------===//

  /// The state after paths `states` join: values are `ite`s on their path
  /// conditions. Returns a dead state if none is alive.
  State merge(ArrayRef<State> states, SmallVector<std::string> *yields = {}) {
    SmallVector<const State *> live;
    for (const State &s : states)
      if (s.alive)
        live.push_back(&s);
    State out;
    if (live.empty()) {
      out.alive = false;
      return out;
    }
    if (live.size() == 1) {
      out = *live.front();
      if (yields)
        *yields = out.yields;
      out.yields.clear();
      return out;
    }
    for (const State *s : live)
      out.epoch = std::max(out.epoch, s->epoch);
    std::set<Loc> places;
    for (const State *s : live)
      for (auto &[loc, _] : s->env)
        places.insert(loc);
    for (const Loc &loc : places) {
      // The sort of the place, from any path that knows its value.
      Sort sort{false, 64, false};
      for (const State *s : live)
        if (auto it = s->env.find(loc); it != s->env.end()) {
          sort = sortOfTerm(it->second);
          break;
        }
      SmallVector<std::string> vals;
      for (const State *s : live)
        vals.push_back(load(loc, const_cast<State &>(*s), sort));
      out.env[loc] = ite(live, vals, sort);
    }
    // A `ref` local refers to the same place on every path, or to none.
    for (auto &[loc, target] : live.front()->refs)
      if (llvm::all_of(live, [&](const State *s) {
            auto it = s->refs.find(loc);
            return it != s->refs.end() && it->second == target;
          }))
        out.refs[loc] = target;
    std::string pc = "(or";
    for (const State *s : live)
      pc += " " + s->pc;
    out.pc = define({true, 1, false}, pc + ")", "r");
    if (yields) {
      yields->clear();
      size_t n = live.front()->yields.size();
      for (size_t i = 0; i < n; ++i) {
        SmallVector<std::string> vals;
        for (const State *s : live)
          vals.push_back(i < s->yields.size() ? s->yields[i] : "");
        if (llvm::is_contained(vals, "")) {
          yields->push_back(declare(sortOfTerm(live.front()->yields[i])));
          continue;
        }
        yields->push_back(ite(live, vals, sortOfTerm(vals.front())));
      }
    }
    return out;
  }

  std::string ite(ArrayRef<const State *> live, ArrayRef<std::string> vals,
                  Sort sort) {
    if (llvm::all_equal(vals))
      return vals.front();
    std::string acc = vals.back();
    for (size_t i = vals.size() - 1; i-- > 0;)
      acc = "(ite " + live[i]->pc + " " + vals[i] + " " + acc + ")";
    return define(sort, acc);
  }

  //===--------------------------------------------------------------------===//
  // Walking
  //===--------------------------------------------------------------------===//

  void walkBlock(Block &block, State &state) {
    for (Operation &op : block) {
      if (!state.alive)
        return;
      walkOp(&op, state);
    }
  }

  static StringRef labelOf(Operation *op) {
    if (auto label = op->getAttrOfType<StringAttr>("label"))
      return label.getValue();
    return "";
  }

  template <typename Frame>
  Frame *findFrame(SmallVectorImpl<Frame *> &frames, StringRef label) {
    for (Frame *frame : llvm::reverse(frames))
      if (label.empty() || frame->label == label)
        return frame;
    return nullptr;
  }

  void walkOp(Operation *op, State &state) {
    if (isa<RequiresOp>(op))
      return; // Assumed at entry.
    if (isa<ContractEntryOp>(op)) {
      if (!inContract)
        entryState = state;
      return;
    }
    if (auto ensures = dyn_cast<EnsuresOp>(op)) {
      if (!inContract)
        proveEnsures(ensures, state);
      return;
    }
    if (auto old = dyn_cast<OldOp>(op)) {
      evalOld(old, state);
      return;
    }
    if (auto forall = dyn_cast<ForallOp>(op)) {
      evalForall(forall, state);
      return;
    }
    if (op->getName().getStringRef() == "lit.error_return") {
      state.alive = false; // Raises out of the function.
      return;
    }
    if (isa<HLCF::ReturnOp>(op)) {
      state.alive = false;
      return;
    }
    if (isa<HLCF::YieldOp, ContractYieldOp, LIT::TryYieldOp>(op)) {
      state.yields.clear();
      for (Value operand : op->getOperands())
        state.yields.push_back(term(operand, state));
      return;
    }
    if (isa<HLCF::ContinueOp, HLCF::BreakOp>(op)) {
      LoopFrame *frame = findFrame(loops, labelOf(op));
      if (!frame || op->getNumOperands()) {
        notAnalyzed(op, state);
        state.alive = false;
        return;
      }
      (isa<HLCF::ContinueOp>(op) ? frame->continues : frame->breaks)
          .push_back(state);
      state.alive = false;
      return;
    }
    if (isa<LIT::TryRaiseOp>(op)) {
      TryFrame *frame = findFrame(tries, labelOf(op));
      if (!frame) {
        state.alive = false; // Raised out of the function.
        return;
      }
      frame->raises.push_back(state);
      state.alive = false;
      return;
    }
    if (auto ifOp = dyn_cast<HLCF::IfOp>(op)) {
      walkIf(ifOp, state);
      return;
    }
    if (auto loop = dyn_cast<HLCF::LoopOp>(op)) {
      walkLoop(loop, state);
      return;
    }
    if (auto tryOp = dyn_cast<LIT::TryOp>(op)) {
      walkTry(tryOp, state);
      return;
    }
    if (op->getNumRegions()) {
      notAnalyzed(op, state);
      return;
    }
    evalOp(op, state);
  }

  void walkIf(HLCF::IfOp ifOp, State &state) {
    if (!ifOp.getElifRegions().empty()) {
      notAnalyzed(ifOp, state);
      return;
    }
    std::string cond = term(ifOp.getCond(), state);
    noteCondition(cond);
    State thenState = state, elseState = state;
    thenState.pc = "(and " + state.pc + " " + cond + ")";
    elseState.pc = "(and " + state.pc + " (not " + cond + "))";
    thenState.yields.clear();
    elseState.yields.clear();
    walkBlock(ifOp.getThenBlock(), thenState);
    walkBlock(ifOp.getElseBlock(), elseState);
    SmallVector<std::string> results;
    State joined = merge({thenState, elseState}, &results);
    bindResults(ifOp, results);
    joined.yields = state.yields;
    state = std::move(joined);
  }

  void walkTry(LIT::TryOp tryOp, State &state) {
    if (tryOp->getNumRegions() < 2 || tryOp->getNumResults()) {
      notAnalyzed(tryOp, state);
      return;
    }
    TryFrame frame{labelOf(tryOp), {}};
    tries.push_back(&frame);
    State body = state;
    walkBlock(tryOp->getRegion(0).front(), body);
    tries.pop_back();
    SmallVector<State> after;
    // The paths that raised run the `except` region.
    State raised = merge(frame.raises);
    if (raised.alive) {
      walkBlock(tryOp->getRegion(1).front(), raised);
      after.push_back(std::move(raised));
    }
    // The paths that completed run the `else` region, if there is one.
    if (body.alive) {
      if (tryOp->getNumRegions() > 2 && !tryOp->getRegion(2).empty())
        walkBlock(tryOp->getRegion(2).front(), body);
      after.push_back(std::move(body));
    }
    State joined = merge(after);
    joined.yields = state.yields;
    state = std::move(joined);
  }

  /// The roots code may write; nullopt if it may write any of them.
  std::optional<llvm::DenseSet<Value>> writtenRoots(Operation *root) {
    llvm::DenseSet<Value> written;
    bool all = false;
    auto addOrigin = [&](StringRef origin) {
      if (std::optional<SmallVector<Value>> named = rootsNamedBy(origin))
        written.insert(named->begin(), named->end());
      else
        all = true;
    };
    auto addRef = [&](Value ref) {
      if (std::optional<Loc> loc = placeOf(ref))
        written.insert(loc->root);
      else if (auto type = dyn_cast<LIT::RefType>(ref.getType()))
        addOrigin(printed(type.getOrigin()));
      else
        all = true;
    };
    root->walk([&](Operation *op) {
      if (auto store = dyn_cast<LIT::RefStoreOp>(op)) {
        addRef(store->getOperand(1));
      } else if (auto call = dyn_cast<LIT::CallOp>(op)) {
        for (Value operand : call.getOperands())
          if (auto ref = dyn_cast<LIT::RefType>(operand.getType());
              ref && !ref.isMutableKnown(false))
            addRef(operand);
        // Keep the printed list alive while its elements are used.
        std::string origins = printed(call.getImplicitOriginsAttr());
        for (StringRef origin : topLevelElements(origins))
          if (!origin.ends_with(": !lit.origin<false>"))
            addOrigin(origin);
      } else if (op->getNumRegions() &&
                 !isa<HLCF::IfOp, HLCF::LoopOp, LIT::TryOp, RequiresOp,
                      EnsuresOp>(op)) {
        all = true;
      }
    });
    // Locals declared in the code are written before they are read.
    root->walk([&](LIT::VarDeclOp decl) { written.insert(decl->getResult(0)); });
    if (all)
      return std::nullopt;
    return written;
  }

  void walkLoop(HLCF::LoopOp loop, State &state) {
    if (loop->getNumResults() || loop->getNumOperands() ||
        loop.getBody().front().getNumArguments()) {
      notAnalyzed(loop, state);
      return;
    }
    // The loop head: what the loop may write holds an unknown there, bound
    // by the invariants found below.
    std::optional<llvm::DenseSet<Value>> written = writtenRoots(loop);
    State before = state;
    before.yields.clear();
    State head = before;
    head.pc = define({true, 1, false}, before.pc, "r");
    std::string headReach = head.pc;
    if (written) {
      for (Value root : *written) {
        Loc loc{root, ""};
        head.env[loc] = declare(sortOf(placeType(loc)), "l");
        for (auto it = head.env.begin(); it != head.env.end();)
          it = it->first.root == root && !it->first.path.empty()
                   ? head.env.erase(it)
                   : std::next(it);
        head.refs.erase(loc);
      }
    } else {
      havocAll(head);
    }
    State headAtEntry = head;
    LoopFrame frame{labelOf(loop), {}, {}, {}, {}, {}};
    loops.push_back(&frame);
    State body = head;
    walkBlock(loop.getBody().front(), body);
    loops.pop_back();
    if (body.alive) // Falling off the body starts the next iteration.
      frame.continues.push_back(body);
    ++loopsAnalyzed;
    findInvariants(frame, before, headAtEntry, headReach, written);
    State exit = merge(frame.breaks);
    exit.yields = state.yields;
    state = std::move(exit);
  }

  /// Houdini: keep the candidate invariants that hold on entry and are kept
  /// by every iteration, assuming all kept candidates; assert them at the
  /// loop head (under its reachability).
  void findInvariants(const LoopFrame &frame, State &before, State &head,
                      StringRef headReach,
                      const std::optional<llvm::DenseSet<Value>> &written) {
    // Integer places the body reads and may change, and that its conditions
    // depend on.
    llvm::StringSet<> relevant = closure(frame.conditions);
    SmallVector<Loc> places;
    for (const Loc &loc : frame.loaded) {
      if (written && !written->count(loc.root))
        continue;
      std::string h = load(loc, head, Sort{false, 64, false});
      Sort sort = sortOfTerm(h);
      if (sort.isBool || sort.width != 64 || !relevant.contains(h))
        continue;
      places.push_back(loc);
    }
    if (places.empty() && frame.lengths.empty())
      return;
    // Terms that do not change in the loop: values before it, and lengths
    // of lists it does not change.
    std::set<std::string> headNames;
    for (const Loc &loc : places)
      headNames.insert(load(loc, head, Sort{false, 64, true}));
    SmallVector<std::string> fixed;
    for (const Loc &loc : places)
      fixed.push_back(load(loc, before, Sort{false, 64, true}));
    for (const std::string &length : frame.lengths)
      if (llvm::none_of(headNames, [&](const std::string &name) {
            return llvm::is_contained(namesIn(length), name);
          }) &&
          llvm::any_of(namesIn(length), [&](const std::string &name) {
            return relevant.contains(name);
          }))
        fixed.push_back(length);
    std::sort(fixed.begin(), fixed.end());
    fixed.erase(std::unique(fixed.begin(), fixed.end()), fixed.end());
    // Lists the loop changes and whose lengths it reads: their lengths are
    // candidates too, against the same terms and the integer variables.
    SmallVector<Loc> lists;
    for (const Loc &loc : frame.loaded) {
      if ((written && !written->count(loc.root)) || !loc.path.empty())
        continue;
      std::string h = load(loc, head, Sort{false, 64, false});
      if (llvm::any_of(frame.lengths, [&](const std::string &length) {
            return llvm::is_contained(namesIn(length), h);
          }))
        lists.push_back(loc);
    }
    for (const Loc &loc : lists) {
      std::string before0 = load(loc, before, Sort{false, 64, false});
      fixed.push_back(lenOf(before0));
    }
    std::sort(fixed.begin(), fixed.end());
    fixed.erase(std::unique(fixed.begin(), fixed.end()), fixed.end());
    SmallVector<Candidate> candidates;
    for (const Loc &loc : lists) {
      candidates.push_back({loc, "bvsge", std::nullopt, bvConst(0, 64), true});
      for (const std::string &t : fixed)
        for (const char *op : {"bvsle", "bvsge"})
          candidates.push_back({loc, op, std::nullopt, t, true});
      for (const Loc &other : places)
        for (const char *op : {"bvsle", "bvslt", "bvsge"})
          candidates.push_back({loc, op, other, "", true});
    }
    for (const Loc &loc : places) {
      candidates.push_back({loc, "bvsge", std::nullopt, bvConst(0, 64)});
      for (const std::string &t : fixed)
        for (const char *op : {"bvsle", "bvslt", "bvsge"})
          candidates.push_back({loc, op, std::nullopt, t});
      for (const Loc &other : places)
        if (!(other == loc))
          for (const char *op : {"bvsle", "bvslt"})
            candidates.push_back({loc, op, other, ""});
    }
    auto render = [&](const Candidate &c, State &at) {
      std::string lhs = load(c.lhs, at, Sort{false, 64, true});
      if (c.length)
        lhs = lenOf(lhs);
      std::string rhs =
          c.rhsLoc ? load(*c.rhsLoc, at, Sort{false, 64, true}) : c.rhsTerm;
      return "(" + c.op + " " + lhs + " " + rhs + ")";
    };
    SmallVector<State *> ends;
    for (const State &s : frame.continues)
      ends.push_back(const_cast<State *>(&s));
    std::vector<bool> kept(candidates.size(), true);
    for (unsigned round = 0; round < 8; ++round) {
      // The queries, in the order `rebuild` emits them.
      SmallVector<size_t> asked;
      for (size_t i = 0; i < candidates.size(); ++i)
        if (kept[i])
          asked.append(1 + ends.size(), i);
      std::string text = rebuild(candidates, kept, before, head, ends, render);
      std::optional<SmallVector<Answer>> answers = runZ3(
          solver, text, dumpPrefix + ".houdini" + std::to_string(houdiniRuns++));
      bool changed = false;
      for (auto [k, i] : llvm::enumerate(asked)) {
        bool proven = answers && k < answers->size() &&
                      (*answers)[k] == Answer::Proven;
        if (!proven && kept[i]) {
          kept[i] = false;
          changed = true;
        }
      }
      if (!changed)
        break;
    }
    for (size_t i = 0; i < candidates.size(); ++i)
      if (kept[i]) {
        facts.push_back(
            ("(=> " + headReach + " " + render(candidates[i], head) + ")")
                .str());
        ++invariantsFound;
      }
  }

  /// The Houdini script for the kept candidates: rendering first (which may
  /// add definitions to the prelude), then the header, then the queries.
  template <typename Render>
  std::string rebuild(ArrayRef<Candidate> candidates,
                      const std::vector<bool> &kept, State &before,
                      State &head, ArrayRef<State *> ends, Render &render) {
    SmallVector<std::string> assumed;
    for (size_t i = 0; i < candidates.size(); ++i)
      if (kept[i])
        assumed.push_back(render(candidates[i], head));
    std::string queries;
    for (size_t i = 0; i < candidates.size(); ++i) {
      if (!kept[i])
        continue;
      // A candidate the solver cannot decide quickly is dropped, which is
      // sound: it is only not assumed.
      unsigned limit = std::max(1u, solver.rlimit / 20);
      queries += query({before.pc}, render(candidates[i], before), limit);
      for (State *end : ends) {
        SmallVector<std::string> assumptions{end->pc};
        assumptions.append(assumed.begin(), assumed.end());
        queries += query(assumptions, render(candidates[i], *end), limit);
      }
    }
    return header() + queries;
  }

  void bindResults(Operation *op, ArrayRef<std::string> terms) {
    for (auto [i, result] : llvm::enumerate(op->getResults()))
      values[result] = i < terms.size() ? terms[i]
                                        : declare(sortOf(result.getType()));
  }

  /// The function's own postcondition, where it returns.
  void proveEnsures(EnsuresOp ensures, State &state) {
    Operation *next = ensures->getNextNode();
    Location at = next ? next->getLoc() : ensures.getLoc();
    Obligation ob{state.pc, "false", at, ensures.getLoc(), displayName(fn)};
    ob.postcondition = true;
    State *old = entryState ? &*entryState : nullptr;
    if (MaybeTerm cond = instantiate(ensures.getBody(), ensures.getArgs(),
                                     state, {}, {}, old, /*assumed=*/false))
      ob.cond = *cond;
    else
      ob.analyzed = false;
    obligations.push_back(ob);
  }

  /// Code the encoder does not follow: memory is unknown after it, and its
  /// obligations are not analyzed.
  void notAnalyzed(Operation *op, State &state) {
    op->walk([&](LIT::CallOp call) {
      if (inContract)
        return;
      if (LIT::FnOp callee = lookup(call))
        if (!callee.getFunctionBody().empty())
          for (Operation &calleeOp : callee.getFunctionBody().front())
            if (auto req = dyn_cast<RequiresOp>(&calleeOp))
              obligations.push_back({state.pc, "true", call.getLoc(),
                                     req.getLoc(), displayName(callee),
                                     /*analyzed=*/false});
    });
    havocAll(state);
    setResultsUnknown(op);
  }

  LIT::FnOp lookup(LIT::CallOp call) {
    SymbolRefAttr callee = call.getDirectCallee();
    if (!callee)
      return {};
    // Symbols are absolute (`@std::@collections::...`): resolve them from the
    // module, since a `lit.fn` is a symbol table of its own.
    return dyn_cast_or_null<LIT::FnOp>(symbols.lookupSymbolIn(module, callee));
  }

  //===--------------------------------------------------------------------===//
  // Operations
  //===--------------------------------------------------------------------===//

  void evalOp(Operation *op, State &state) {
    if (auto cst = dyn_cast<ParamConstantOp>(op)) {
      evalConstant(cst);
      return;
    }
    if (isa<RebindOp>(op)) {
      Value input = op->getOperand(0), result = op->getResult(0);
      values[result] = term(input, state);
      if (auto it = records.find(input); it != records.end())
        records[result] = it->second;
      if (auto it = elements.find(input); it != elements.end())
        elements[result] = it->second;
      return;
    }
    if (auto storeOp = dyn_cast<LIT::RefStoreOp>(op)) {
      Value value = storeOp->getOperand(0), dest = storeOp->getOperand(1);
      if (auto it = elements.find(dest); it != elements.end()) {
        auto [list, index] = it->second;
        storeElement(list, index, term(value, state), state);
        return;
      }
      if (std::optional<Loc> loc = placeOf(dest))
        store(*loc, state, term(value, state), value);
      else if (auto ref = dyn_cast<LIT::RefType>(dest.getType()))
        havocOrigin(printed(ref.getOrigin()), state);
      else
        havocAll(state);
      return;
    }
    if (auto loadOp = dyn_cast<LIT::RefLoadOp>(op)) {
      Value ref = loadOp->getOperand(0), result = loadOp->getResult(0);
      if (auto it = elements.find(ref); it != elements.end()) {
        auto [list, index] = it->second;
        values[result] = elem(load(list, state, placeType(list)), index,
                              sortOf(result.getType()));
        return;
      }
      std::optional<Loc> loc = placeOf(ref);
      if (!loc) {
        setResultsUnknown(op);
        return;
      }
      values[result] = load(*loc, state, result.getType());
      // Loading a `ref` local gives a reference to what it refers to.
      if (auto it = state.refs.find(*loc); it != state.refs.end())
        derivedPlaces[result] = it->second;
      // Loading a struct whose fields are known gives a value with them.
      std::map<std::string, std::string> fields;
      for (auto &[place, term] : state.env)
        if (place.root == loc->root &&
            StringRef(place.path).starts_with(loc->path + "/"))
          fields[place.path.substr(loc->path.size())] = term;
      if (!fields.empty())
        records[result] = std::move(fields);
      return;
    }
    if (auto call = dyn_cast<LIT::CallOp>(op)) {
      evalCall(call, state);
      return;
    }
    if (isa<LIT::RefImmutOp>(op))
      if (auto it = elements.find(op->getOperand(0)); it != elements.end()) {
        auto element = it->second;
        elements[op->getResult(0)] = element;
      }
    // Declarations, lifetimes, debug info, and ops producing values not
    // modelled: results are unknown, memory unchanged.
    setResultsUnknown(op);
  }

  void evalConstant(ParamConstantOp cst) {
    Value result = cst->getResult(0);
    std::string text = printed(cst.getValue());
    Sort sort = sortOf(result.getType());
    // `Int` and `Bool` literals: `#lit.struct<{_mlir_value: scalar<index> =
    // 0}>` (possibly under a `rebind` to an alias), or `#kgen<simd true>`.
    static llvm::Regex structRe("^(#kgen.param.expr<rebind, )?#lit.struct<\\{"
                                "_mlir_value: scalar<[a-z0-9]+> = "
                                "(-?[0-9]+|true|false)\\}>");
    static llvm::Regex simdRe("^#kgen<simd (-?[0-9]+|true|false)>");
    SmallVector<StringRef> m;
    StringRef literal;
    if (structRe.match(text, &m))
      literal = m[2];
    else if (simdRe.match(text, &m))
      literal = m[1];
    if (sort.isBool && (literal == "true" || literal == "false")) {
      values[result] = literal.str();
    } else if (!sort.isBool && !literal.empty()) {
      int64_t v;
      if (!literal.getAsInteger(10, v))
        values[result] = bvConst(v, sort.width);
    }
    if (!values.count(result))
      values[result] = declare(sort);
  }

  void evalCall(LIT::CallOp call, State &state) {
    std::optional<CalleeName> name = calleeName(call);
    LIT::FnOp callee = lookup(call);
    bool hasBody = callee && !callee.getFunctionBody().empty();
    // A call to a function with preconditions: they are obligations here.
    if (hasBody && !inContract) {
      for (Operation &calleeOp : callee.getFunctionBody().front()) {
        auto req = dyn_cast<RequiresOp>(&calleeOp);
        if (!req)
          continue;
        Obligation ob{state.pc, "false", call.getLoc(), req.getLoc(),
                      displayName(callee)};
        if (MaybeTerm cond =
                instantiate(req.getBody(), req.getArgs(), state,
                            call.getOperands(), callee, nullptr, false)) {
          ob.cond = *cond;
          noteCondition(ob.cond);
        } else {
          ob.analyzed = false;
        }
        obligations.push_back(ob);
      }
    }
    if (name && evalBuiltin(call, *name, state))
      return;
    if (name && evalElementAccess(call, *name, state))
      return;
    // The call's effects: its results are unknown, and so is memory it may
    // write: through its mutable reference arguments, and through the
    // mutable origins it is given (e.g. inside a struct passed by value).
    State before = state;
    before.yields.clear();
    setResultsUnknown(call);
    for (Value operand : call.getOperands()) {
      auto ref = dyn_cast<LIT::RefType>(operand.getType());
      if (!ref || ref.isMutableKnown(false))
        continue;
      if (std::optional<Loc> loc = placeOf(operand))
        havoc(*loc, state, ref.getElementType());
      else
        havocOrigin(printed(ref.getOrigin()), state);
    }
    // Keep the printed list alive while its elements are used.
    std::string origins = printed(call.getImplicitOriginsAttr());
    for (StringRef origin : topLevelElements(origins))
      if (!origin.ends_with(": !lit.origin<false>"))
        havocOrigin(origin, state);
    if (hasBody)
      assumeEnsures(call, callee, before, state);
  }

  /// After a call, the callee's postcondition: its arguments as they are
  /// now, its `old`s as they were before the call. A raising callee's holds
  /// only where it did not raise (its first result is the raised flag).
  void assumeEnsures(LIT::CallOp call, LIT::FnOp callee, State &before,
                     State &state) {
    EnsuresOp ensures;
    callee.getFunctionBody().walk([&](EnsuresOp op) {
      if (!ensures)
        ensures = op;
    });
    if (!ensures)
      return;
    callResult = call->getNumResults() == 1 ? call->getResult(0) : Value();
    MaybeTerm cond =
        instantiate(ensures.getBody(), ensures.getArgs(), state,
                    call.getOperands(), callee, &before, /*assumed=*/true);
    callResult = {};
    if (!cond)
      return;
    std::string holds = *cond;
    std::string type = printed(call.getCallee().getType());
    if (StringRef(type).contains(" throws") && call->getNumResults() >= 1 &&
        sortOf(call->getResult(0).getType()).isBool)
      holds = "(=> (not " + values[call->getResult(0)] + ") " + holds + ")";
    state.pc = define({true, 1, false}, "(and " + state.pc + " " + holds + ")",
                      "r");
  }

  /// `_same_elements(a._data, b._data, n)` for lists `a` and `b`: their first
  /// `n` elements are equal, for every element sort read so far. Anything
  /// else is unknown (nothing when assumed, false to prove).
  std::string sameElements(const std::string &dst, const std::string &src,
                           const std::string &count) {
    auto dstField = fieldOwners.find(dst), srcField = fieldOwners.find(src);
    if (dstField == fieldOwners.end() || srcField == fieldOwners.end() ||
        !StringRef(dstField->second.second).contains("_data") ||
        dstField->second.second != srcField->second.second)
      return assuming ? "true" : "false";
    const std::string &a = dstField->second.first, &b = srcField->second.first;
    elem(a, bvConst(0, 64), {false, 64, true}); // At least `Int` elements.
    std::string k = declare({false, 64, true}, "k");
    std::string equal = "true";
    for (const std::string &fn : elementFunctions)
      equal = "(and " + equal + " (= (" + fn + " " + a + " " + k + ") (" + fn +
              " " + b + " " + k + ")))";
    std::string range = "(and (bvsle " + bvConst(0, 64) + " " + k +
                        ") (bvslt " + k + " " + count + "))";
    if (!assuming)
      return define({true, 1, false}, "(=> " + range + " " + equal + ")");
    return define({true, 1, false}, "(forall ((" + k +
                                        " (_ BitVec 64))) (=> " + range + " " +
                                        equal + "))");
  }

  /// `List.__getitem__`: a reference to an element, the list's place and an
  /// index. Reading it reads `elem(list, index)`; writing it makes a new list
  /// with the same length and the other elements unchanged.
  bool evalElementAccess(LIT::CallOp call, const CalleeName &name,
                         State &state) {
    StringRef path = name.path;
    if (!path.starts_with("std::collections::list::List::__getitem__[") ||
        call->getNumResults() != 1 || call.getNumOperands() < 1 ||
        !isa<LIT::RefType>(call->getResult(0).getType()))
      return false;
    std::optional<Loc> list = placeOf(call.getOperands()[0]);
    if (!list)
      return false;
    std::string index;
    // `xs[0]`: the index is a parameter, `!pop.int_literal`.
    static llvm::Regex literalRe("int_literal (-?[0-9]+)|"
                                 "#pop<int_literal (-?[0-9]+)>");
    SmallVector<StringRef> m;
    if (call.getNumOperands() == 2 &&
        sortOf(call.getOperands()[1].getType()).width == 64 &&
        !sortOf(call.getOperands()[1].getType()).isBool &&
        !path.contains("IntLiteral")) {
      index = term(call.getOperands()[1], state);
    } else if (call.getNumOperands() == 2) {
      // The literal's value is in the index's type (`IntLiteral[0]`).
      std::string type = printed(call.getOperands()[1].getType());
      if (literalRe.match(type, &m)) {
        int64_t v;
        StringRef digits = m[1].empty() ? m[2] : m[1];
        if (!digits.getAsInteger(10, v))
          index = bvConst(v, 64);
      }
    }
    if (index.empty())
      return false;
    Value result = call->getResult(0);
    values[result] = declare({false, 64, false}, "g");
    elements[result] = {*list, index};
    return true;
  }

  std::string elem(StringRef list, StringRef index, Sort sort) {
    std::string fn = "elem" + std::to_string(sort.isBool ? 1 : sort.width) +
                     (sort.isBool ? "b" : "");
    if (elementFunctions.insert(fn).second)
      prelude += "(declare-fun " + fn + " ((_ BitVec 64) (_ BitVec 64)) " +
                 sort.str() + ")\n";
    return define(sort, ("(" + fn + " " + list + " " + index + ")").str());
  }

  /// Writing element `index` of the list at `list` with `value`.
  void storeElement(const Loc &list, StringRef index, StringRef value,
                    State &state) {
    std::string old = load(list, state, placeType(list));
    std::string updated = declare({false, 64, false}, "h");
    Sort sort = sortOfTerm(value);
    std::string fn = "elem" + std::to_string(sort.isBool ? 1 : sort.width) +
                     (sort.isBool ? "b" : "");
    elem(updated, index, sort); // Declares the function.
    facts.push_back("(= " + lenOf(updated) + " " + lenOf(old) + ")");
    facts.push_back(("(= (" + fn + " " + updated + " " + index + ") " + value +
                     ")")
                        .str());
    facts.push_back(("(forall ((j (_ BitVec 64))) (! (=> (not (= j " + index +
                     ")) (= (" + fn + " " + updated + " j) (" + fn + " " + old +
                     " j))) :pattern ((" + fn + " " + updated + " j))))")
                        .str());
    store(list, state, updated);
  }

  /// Integer and Boolean operators, `len`, and `range` iteration.
  bool evalBuiltin(LIT::CallOp call, const CalleeName &name, State &state) {
    StringRef path = name.path;
    if (call->getNumResults() != 1)
      return false;
    Value result = call->getResult(0);
    auto operand = [&](unsigned i) {
      return term(call.getOperands()[i], state);
    };
    if (path.starts_with("std::builtin::bool::Bool::__mlir_bool__(") &&
        call.getNumOperands() == 1) {
      values[result] = operand(0);
      return true;
    }
    if (path.starts_with("std::builtin::len::len[") &&
        call.getNumOperands() == 1) {
      values[result] = define(
          {false, 64, true}, lenOf(valueThrough(call.getOperands()[0], state)));
      return true;
    }
    if (evalRange(call, name, state))
      return true;
    if (path.starts_with("std::builtin::_verification::_same_elements[") &&
        call.getNumOperands() == 3) {
      values[result] = sameElements(term(call.getOperands()[0], state),
                                    term(call.getOperands()[1], state),
                                    term(call.getOperands()[2], state));
      return true;
    }
    if (!path.starts_with("std::simd::SIMD::__") || name.params.size() < 2)
      return false;
    std::optional<Sort> sort = dtypeSort(name.params[0]);
    if (!sort || !isWidthOne(name.params[1]))
      return false;
    StringRef method = path.drop_front(strlen("std::simd::SIMD::"));
    method = method.take_until([](char c) { return c == '('; });
    static const std::pair<const char *, const char *> binary[] = {
        {"__add__", "bvadd"}, {"__sub__", "bvsub"}, {"__mul__", "bvmul"},
        {"__and__", "bvand"}, {"__or__", "bvor"},   {"__xor__", "bvxor"},
    };
    bool s = sort->isSigned;
    const std::pair<const char *, const char *> compare[] = {
        {"__lt__", s ? "bvslt" : "bvult"}, {"__le__", s ? "bvsle" : "bvule"},
        {"__gt__", s ? "bvsgt" : "bvugt"}, {"__ge__", s ? "bvsge" : "bvuge"},
    };
    if (call.getNumOperands() == 2) {
      for (auto &[m, smt] : binary) {
        if (method == m) {
          values[result] = define(*sort, "(" + std::string(smt) + " " +
                                             operand(0) + " " + operand(1) +
                                             ")");
          return true;
        }
        // In-place: `x += y` writes `x + y` through the reference `x`.
        if (method == ("__i" + StringRef(m).drop_front(2)).str()) {
          std::optional<Loc> loc = placeOf(call.getOperands()[0]);
          if (!loc)
            return false;
          std::string old = load(*loc, state, *sort);
          store(*loc, state,
                define(*sort, "(" + std::string(smt) + " " + old + " " +
                                  operand(1) + ")"));
          values[result] = declare(sortOf(result.getType()));
          return true;
        }
      }
      for (auto &[m, smt] : compare)
        if (method == m) {
          values[result] =
              define({true, 1, false}, "(" + std::string(smt) + " " +
                                           operand(0) + " " + operand(1) + ")");
          noteCondition(values[result]);
          return true;
        }
      if (method == "__eq__" || method == "__ne__") {
        std::string eq = "(= " + operand(0) + " " + operand(1) + ")";
        values[result] = define({true, 1, false},
                                method == "__eq__" ? eq : "(not " + eq + ")");
        return true;
      }
    }
    if (call.getNumOperands() == 1 && method == "__neg__") {
      values[result] = define(*sort, "(bvneg " + operand(0) + ")");
      return true;
    }
    return false;
  }

  /// `range(end)` and `range(start, end)` over an integer dtype, and their
  /// iterators' `__iter__` and `__next__`, as the stdlib defines them: the
  /// range holds a current value and an end (`max(end, 0)`, `max(start,
  /// end)`); `__next__` raises when they are equal, and otherwise returns the
  /// current value and increments it. The fields are named `/curr` and
  /// `/end` here.
  bool evalRange(LIT::CallOp call, const CalleeName &name, State &state) {
    StringRef path = name.path;
    Value result = call->getResult(0);
    auto max = [&](Sort sort, StringRef a, StringRef b) {
      std::string lt = sort.isSigned ? "bvslt" : "bvult";
      return define(sort, ("(ite (" + lt + " " + a + " " + b + ") " + b +
                           " " + a + ")")
                              .str());
    };
    if (path.starts_with("std::builtin::range::range[::DType](")) {
      if (name.params.empty())
        return false;
      std::optional<Sort> sort = dtypeSort(name.params[0]);
      if (!sort)
        return false;
      std::map<std::string, std::string> fields;
      if (call.getNumOperands() == 1) {
        fields["/curr"] = bvConst(0, sort->width);
        fields["/end"] =
            max(*sort, term(call.getOperands()[0], state), bvConst(0, sort->width));
      } else if (call.getNumOperands() == 2) {
        std::string start = term(call.getOperands()[0], state);
        fields["/curr"] = start;
        fields["/end"] = max(*sort, start, term(call.getOperands()[1], state));
      } else {
        return false;
      }
      values[result] = declare({false, 64, false}, "g");
      records[result] = std::move(fields);
      return true;
    }
    bool zero = path.starts_with("std::builtin::range::_ZeroStartingRange::");
    bool sequential = path.starts_with("std::builtin::range::_SequentialRange::");
    if (!zero && !sequential)
      return false;
    // The method name follows the struct's (and may itself contain `::` in
    // its mangled signature).
    StringRef method = path.drop_front(
        strlen(zero ? "std::builtin::range::_ZeroStartingRange::"
                    : "std::builtin::range::_SequentialRange::"));
    if (name.params.empty())
      return false;
    std::optional<Sort> sort = dtypeSort(name.params[0]);
    if (!sort || call.getNumOperands() < 1)
      return false;
    std::optional<Loc> self = placeOf(call.getOperands()[0]);
    if (!self)
      return false;
    Loc curr{self->root, self->path + "/curr"}, end{self->root, self->path + "/end"};
    if (method.starts_with("__iter__") && call.getNumOperands() == 1) {
      values[result] = declare({false, 64, false}, "g");
      records[result] = {{"/curr", load(curr, state, *sort)},
                         {"/end", load(end, state, *sort)}};
      return true;
    }
    if (method.starts_with("__next__") && call.getNumOperands() == 3) {
      std::optional<Loc> error = placeOf(call.getOperands()[1]);
      std::optional<Loc> out = placeOf(call.getOperands()[2]);
      if (!error || !out)
        return false;
      std::string c = load(curr, state, *sort);
      std::string e = load(end, state, *sort);
      std::string raised = define({true, 1, false}, "(= " + c + " " + e + ")");
      noteCondition(raised);
      store(curr, state,
            define(*sort, "(ite " + raised + " " + c + " (bvadd " + c + " " +
                              bvConst(1, sort->width) + "))"));
      // `store` makes the range itself unknown, not its other field.
      state.env[end] = e;
      store(*out, state, c);
      havoc(*error, state, placeType(*error));
      values[result] = raised;
      return true;
    }
    return false;
  }

  /// The condition of a contract region (`kgen.requires` or `kgen.ensures`)
  /// of `callee` (the function itself when `callee` is null), with its block
  /// arguments bound to the caller's `operands` (the function's own arguments
  /// when empty) in `state`. `kgen.old` inside it is evaluated in `old`.
  /// Quantifiers are real `forall`s when the condition is `assumed`, and a
  /// fresh index (enough to prove one) otherwise.
  MaybeTerm instantiate(Region &region, OperandRange args, State &state,
                        ValueRange operands, LIT::FnOp callee, State *old,
                        bool assumed) {
    Block &body = region.front();
    Block &calleeEntry = (callee ? callee : fn).getFunctionBody().front();
    DenseMap<Value, std::string> saved = values;
    DenseMap<Value, Value> savedActuals = actuals;
    State *savedOld = oldState;
    bool savedAssuming = assuming;
    SmallVector<Value> boundRefs;
    auto restore = [&] {
      for (Value ref : boundRefs)
        refArgs.erase(ref);
      // Values of the region's ops belong to this instantiation only.
      values = std::move(saved);
      actuals = std::move(savedActuals);
      oldState = savedOld;
      assuming = savedAssuming;
    };
    // The callee's arguments, as the caller's values (for `kgen.old`).
    if (callee)
      for (BlockArgument formal : calleeEntry.getArguments())
        if (formal.getArgNumber() < operands.size())
          actuals[formal] = operands[formal.getArgNumber()];
    for (auto [i, blockArg] : llvm::enumerate(body.getArguments())) {
      if (i >= args.size()) {
        restore();
        return std::nullopt;
      }
      Value formal = args[i];
      Value actual = callee ? actuals.lookup(formal) : formal;
      // A register `out` result is a local of the callee
      // (`lit.var.decl "r" arg`): at a call, the call's result.
      if (callee && !actual && isNamedResult(callee, formal) &&
          callResult && isa<LIT::RefType>(blockArg.getType())) {
        Loc loc{callResult, ""};
        state.env[loc] = term(callResult, state);
        refArgs[blockArg] = loc;
        boundRefs.push_back(blockArg);
        continue;
      }
      auto formalArg = dyn_cast<BlockArgument>(formal);
      bool ownLocal = !callee && formal.getDefiningOp<LIT::VarDeclOp>();
      if (!ownLocal &&
          (!formalArg || formalArg.getOwner() != &calleeEntry || !actual)) {
        restore();
        return std::nullopt;
      }
      if (!bind(blockArg, actual, state, boundRefs)) {
        restore();
        return std::nullopt;
      }
    }
    oldState = old;
    assuming = assumed;
    State inner = state;
    inner.yields.clear();
    ++inContract;
    walkBlock(body, inner);
    --inContract;
    MaybeTerm result;
    if (inner.yields.size() == 1)
      result = inner.yields.front();
    restore();
    return result;
  }

  static bool isNamedResult(LIT::FnOp callee, Value formal) {
    auto decl = formal.getDefiningOp<LIT::VarDeclOp>();
    auto named = callee->getAttrOfType<StringAttr>("namedResult");
    if (!decl || !named)
      return false;
    auto name = decl->getAttrOfType<StringAttr>("name");
    return !name || name.getValue() == named.getValue();
  }

  /// Binds a region's block argument to the caller's value `actual`: a
  /// reference to its place, a value to its term in `state`.
  bool bind(BlockArgument blockArg, Value actual, State &state,
            SmallVectorImpl<Value> &boundRefs) {
    if (isa<LIT::RefType>(blockArg.getType())) {
      std::optional<Loc> loc = placeOf(actual);
      if (!loc)
        return false;
      refArgs[blockArg] = *loc;
      boundRefs.push_back(blockArg);
      return true;
    }
    values[blockArg] = term(actual, state);
    return true;
  }

  /// `kgen.old(%entry, args...)`: its region evaluated on its arguments as
  /// they were at the entry token (`oldState`).
  void evalOld(OldOp old, State &state) {
    Value result = old->getResult(0);
    Region &region = old->getRegion(0);
    if (!oldState || region.empty() ||
        region.front().getNumArguments() + 1 != old->getNumOperands()) {
      values[result] = declare(sortOf(result.getType()));
      return;
    }
    SmallVector<Value> boundRefs;
    DenseMap<Value, std::string> saved = values;
    bool ok = true;
    for (auto [i, blockArg] : llvm::enumerate(region.front().getArguments())) {
      Value operand = old->getOperand(i + 1);
      // A callee's argument stands for the caller's value.
      if (Value actual = actuals.lookup(operand))
        operand = actual;
      ok = ok && bind(blockArg, operand, *oldState, boundRefs);
    }
    State inner = *oldState;
    inner.yields.clear();
    if (ok)
      walkBlock(region.front(), inner);
    for (Value ref : boundRefs)
      refArgs.erase(ref);
    std::string value = ok && inner.yields.size() == 1
                            ? inner.yields.front()
                            : declare(sortOf(result.getType()));
    values = std::move(saved);
    values[result] = value;
  }

  /// `kgen.forall(lo?, hi)`: its condition for every index in `[lo, hi)`.
  void evalForall(ForallOp forall, State &state) {
    Value result = forall->getResult(0);
    Region &region = forall->getRegion(0);
    auto unknown = [&] {
      // Assumed, nothing; to prove, false.
      values[result] = assuming ? "true" : "false";
    };
    if (region.empty() || region.front().getNumArguments() != 1 ||
        forall->getNumOperands() < 1 || forall->getNumOperands() > 2) {
      unknown();
      return;
    }
    BlockArgument index = region.front().getArgument(0);
    Sort sort = sortOf(index.getType());
    std::string lo = forall->getNumOperands() == 2
                         ? term(forall->getOperand(0), state)
                         : bvConst(0, sort.width);
    std::string hi = term(forall->getOperands().back(), state);
    unsigned firstName = counter;
    std::string k = declare(sort, "k");
    values[index] = k;
    State inner = state;
    inner.yields.clear();
    walkBlock(region.front(), inner);
    if (inner.yields.size() != 1) {
      unknown();
      return;
    }
    std::string body = inner.yields.front();
    std::string range = "(and (bvsle " + lo + " " + k + ") (bvslt " + k +
                        " " + hi + "))";
    if (!assuming) {
      // To prove it, one arbitrary index in the range suffices.
      values[result] =
          define({true, 1, false}, "(=> " + range + " " + body + ")");
      return;
    }
    // Assumed: a real quantifier over the body, with the definitions that
    // depend on the index inlined. An unknown made while evaluating the
    // body (e.g. a call's result) would be one value for every index, so
    // the quantifier is not assumed then.
    llvm::StringSet<> used = closure({body});
    for (auto &entry : used) {
      StringRef name = entry.getKey();
      unsigned id;
      if (name != k && StringRef("ugh").contains(name.front()) &&
          !name.drop_front(1).getAsInteger(10, id) && id >= firstName) {
        unknown();
        return;
      }
    }
    std::map<std::string, std::string> memo;
    std::string expanded = expandOver(body, k, memo);
    values[result] = define({true, 1, false},
                            "(forall ((" + k + " " + sort.str() + ")) (=> " +
                                range + " " + expanded + "))");
  }

  /// `term` with every definition that depends on `var` inlined, so `var`
  /// can be bound by a quantifier.
  std::string expandOver(StringRef term, StringRef var,
                         std::map<std::string, std::string> &memo) {
    std::string out;
    for (size_t i = 0; i < term.size();) {
      if (llvm::isAlpha(term[i]) && (i == 0 || !llvm::isAlnum(term[i - 1]))) {
        size_t j = i + 1;
        while (j < term.size() && llvm::isAlnum(term[j]))
          ++j;
        std::string name = term.slice(i, j).str();
        auto def = definitions.find(name);
        if (def != definitions.end() && dependsOn(name, var)) {
          auto [it, inserted] = memo.try_emplace(name, "");
          if (inserted)
            it->second = expandOver(def->second, var, memo);
          out += it->second;
        } else {
          out += name;
        }
        i = j;
        continue;
      }
      out += term[i++];
    }
    return out;
  }

  bool dependsOn(const std::string &name, StringRef var) {
    if (name == var)
      return true;
    auto key = std::make_pair(name, var.str());
    if (auto it = dependsMemo.find(key); it != dependsMemo.end())
      return it->second;
    dependsMemo[key] = false; // Definitions are acyclic.
    bool result = false;
    if (auto it = deps.find(name); it != deps.end())
      for (const std::string &dep : it->second)
        if (dependsOn(dep, var)) {
          result = true;
          break;
        }
    dependsMemo[key] = result;
    return result;
  }
};

bool inStdlib(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp())
    if (auto pkg = dyn_cast<LIT::PackageOp>(parent))
      if (pkg.getSymName() == "std")
        return true;
  return false;
}

struct VerifyContractsPass
    : impl::VerifyContractsBase<VerifyContractsPass> {
  using VerifyContractsBase::VerifyContractsBase;

  void runOnOperation() override {
    std::string z3 = z3Path;
    if (z3.empty()) {
      if (auto found = llvm::sys::findProgramByName("z3"))
        z3 = *found;
    }
    if (z3.empty()) {
      getOperation().emitError("verify-contracts: z3 not found");
      return signalPassFailure();
    }
    SymbolTableCollection symbols;
    SolverConfig solver{z3, rlimit, wallSeconds, dumpDir};
    unsigned total = 0, proven = 0, index = 0, loops = 0, invariants = 0;
    getOperation().walk([&](LIT::FnOp fn) {
      if (!includeStdlib && inStdlib(fn))
        return;
      std::string name = "f" + std::to_string(index++);
      FunctionEncoder enc(fn, getOperation(), symbols, solver, name);
      if (!enc.encode())
        return;
      loops += enc.loopsAnalyzed;
      invariants += enc.invariantsFound;
      if (enc.obligations.empty())
        return;
      std::optional<SmallVector<Answer>> answers =
          runZ3(solver, enc.script(), name);
      unsigned next = 0;
      for (Obligation &ob : enc.obligations) {
        Answer answer = Answer::NotAnalyzed;
        if (ob.analyzed)
          answer = answers && next < answers->size() ? (*answers)[next]
                                                     : Answer::Unknown;
        if (ob.analyzed)
          ++next;
        ++total;
        StringRef what = ob.postcondition ? "postcondition" : "precondition";
        if (answer == Answer::Proven) {
          ++proven;
          if (verbose)
            mlir::emitRemark(ob.callLoc)
                << what << " of '" << ob.callee << "' proven";
          continue;
        }
        auto diag = mlir::emitWarning(ob.callLoc);
        switch (answer) {
        case Answer::Unproven:
          diag << "cannot prove the " << what << " of '" << ob.callee << "'";
          break;
        case Answer::Unknown:
          diag << "the " << what << " of '" << ob.callee
               << "' was not decided within the solver limits";
          break;
        default:
          diag << "the " << what << " of '" << ob.callee
               << "' is not analyzed yet (this control flow is not supported)";
          break;
        }
        diag.attachNote(ob.clauseLoc) << what << " declared here";
      }
    });
    llvm::errs() << "verify-contracts: " << proven << "/" << total
                 << " obligations proven (" << loops << " loops, "
                 << invariants << " invariants)\n";
  }
};

} // namespace
