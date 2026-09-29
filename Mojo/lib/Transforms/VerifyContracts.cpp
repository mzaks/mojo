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
// `lit.ref.load`; control flow through `hlcf.if` and `hlcf.return`. Anything
// else (loops, `lit.try`, comptime control flow) is not analyzed yet: memory
// is unknown after it, and the obligations inside it are reported as not
// analyzed.
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
#include "llvm/ADT/StringExtras.h"
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

/// A place in memory: a root (a function argument or a local variable) and
/// a path of field names.
struct Loc {
  Value root;
  std::string path;
  bool operator<(const Loc &other) const {
    if (root.getAsOpaquePointer() != other.root.getAsOpaquePointer())
      return root.getAsOpaquePointer() < other.root.getAsOpaquePointer();
    return path < other.path;
  }
};

/// What is known on a path through the function.
struct State {
  /// The path condition.
  std::string pc = "true";
  /// Values of places written on this path, or made unknown.
  std::map<Loc, std::string> env;
  /// Places not in `env` hold their value at entry if `epoch` is 0, and an
  /// unknown per place and epoch otherwise (after code that is not analyzed).
  unsigned epoch = 0;
  /// False once the path has returned.
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
};

enum class Answer { Proven, Unproven, Unknown, NotAnalyzed };

class FunctionEncoder {
public:
  FunctionEncoder(LIT::FnOp fn, ModuleOp module,
                  SymbolTableCollection &symbols)
      : fn(fn), module(module), symbols(symbols) {}

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
      if (auto req = dyn_cast<RequiresOp>(&op)) {
        MaybeTerm cond =
            instantiate(req, req.getArgs(), state, /*caller=*/{});
        if (cond)
          facts.push_back(*cond);
      }
    walkBlock(entry, state);
    return true;
  }

  std::string script(unsigned rlimit) const {
    std::string text = "(set-option :print-success false)\n" + prelude;
    for (const std::string &fact : facts)
      text += "(assert " + fact + ")\n";
    for (const Obligation &ob : obligations) {
      if (!ob.analyzed)
        continue;
      text += "(echo \"@@\")\n(push 1)\n(assert " + ob.pc + ")\n(assert (not " +
              ob.cond + "))\n(set-option :rlimit " + std::to_string(rlimit) +
              ")\n(check-sat)\n(set-option :rlimit 0)\n(pop 1)\n";
    }
    return text;
  }

  LIT::FnOp fn;
  SmallVector<Obligation> obligations;

private:
  ModuleOp module;
  SymbolTableCollection &symbols;
  DenseMap<Value, std::string> values;
  std::map<Loc, std::string> entryValues;
  std::map<std::pair<Loc, unsigned>, std::string> epochValues;
  DenseMap<Value, Loc> refArgs; // Contract block arguments bound to places.
  /// The function's reference arguments and local variables.
  SmallVector<Value> roots;
  std::string prelude;
  SmallVector<std::string> facts;
  unsigned counter = 0;
  bool lenDeclared = false;
  /// Inside a contract region: calls there are evaluated, not checked.
  unsigned inContract = 0;

  std::string declare(Sort sort, StringRef prefix = "u") {
    std::string name = (prefix + Twine(counter++)).str();
    prelude += "(declare-const " + name + " " + sort.str() + ")\n";
    return name;
  }

  std::string define(Sort sort, StringRef expr) {
    std::string name = ("t" + Twine(counter++)).str();
    prelude += ("(define-fun " + name + " () " + sort.str() + " " + expr +
                ")\n")
                   .str();
    return name;
  }

  std::string lenOf(StringRef handle) {
    if (!lenDeclared) {
      prelude += "(declare-fun len ((_ BitVec 64)) (_ BitVec 64))\n";
      lenDeclared = true;
    }
    std::string term = ("(len " + handle + ")").str();
    // Assumption: every `Sized` type in the stdlib has a non-negative length.
    facts.push_back("(bvsge " + term + " " + bvConst(0, 64) + ")");
    return term;
  }

  //===--------------------------------------------------------------------===//
  // Places
  //===--------------------------------------------------------------------===//

  /// The place a reference value denotes.
  std::optional<Loc> placeOf(Value ref) {
    if (auto it = refArgs.find(ref); it != refArgs.end())
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

  std::string load(const Loc &loc, State &state, Type type) {
    if (auto it = state.env.find(loc); it != state.env.end())
      return it->second;
    if (state.epoch) {
      auto [it, inserted] = epochValues.try_emplace({loc, state.epoch}, "");
      if (inserted)
        it->second = declare(sortOf(type), "h");
      return it->second;
    }
    auto [it, inserted] = entryValues.try_emplace(loc, "");
    if (inserted)
      it->second = declare(sortOf(type), "e");
    return it->second;
  }

  void store(const Loc &loc, State &state, std::string value) {
    // Writing a place replaces the values of its fields.
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
      if (path.empty())
        break;
    }
    state.env[loc] = std::move(value);
  }

  /// Everything reachable through `loc` becomes unknown.
  void havoc(const Loc &loc, State &state, Type type) {
    store(loc, state, declare(sortOf(type), "h"));
  }

  //===--------------------------------------------------------------------===//
  // Values
  //===--------------------------------------------------------------------===//

  std::string term(Value value, State &state) {
    if (auto it = values.find(value); it != values.end())
      return it->second;
    // A value from outside the region being evaluated (e.g. a constant).
    if (Operation *def = value.getDefiningOp())
      if (auto cst = dyn_cast<ParamConstantOp>(def)) {
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
      return load(*loc, state, cast<LIT::RefType>(ref.getType()).getElementType());
    return declare({false, 64, false}, "h");
  }

  void setResultsUnknown(Operation *op) {
    for (Value result : op->getResults())
      values[result] = declare(sortOf(result.getType()));
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

  void walkOp(Operation *op, State &state) {
    if (isa<RequiresOp, EnsuresOp, ContractEntryOp>(op))
      return; // Handled at entry, or not yet (postconditions: stage 3).
    if (auto ret = dyn_cast<HLCF::ReturnOp>(op)) {
      state.alive = false;
      return;
    }
    if (isa<HLCF::YieldOp, ContractYieldOp>(op)) {
      state.yields.clear();
      for (Value operand : op->getOperands())
        state.yields.push_back(term(operand, state));
      return;
    }
    if (auto ifOp = dyn_cast<HLCF::IfOp>(op)) {
      walkIf(ifOp, state);
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
    State thenState = state, elseState = state;
    thenState.pc = "(and " + state.pc + " " + cond + ")";
    elseState.pc = "(and " + state.pc + " (not " + cond + "))";
    thenState.yields.clear();
    elseState.yields.clear();
    walkBlock(ifOp.getThenBlock(), thenState);
    walkBlock(ifOp.getElseBlock(), elseState);
    State merged;
    merged.epoch = std::max(thenState.epoch, elseState.epoch);
    if (!thenState.alive && !elseState.alive) {
      state.alive = false;
      return;
    }
    if (!elseState.alive || !thenState.alive) {
      State &live = thenState.alive ? thenState : elseState;
      live.pc = define({true, 1, false}, live.pc);
      bindResults(ifOp, live.yields);
      live.yields = state.yields;
      state = std::move(live);
      return;
    }
    // Both arms continue: values are `ite` on the condition.
    std::set<Loc> places;
    for (auto &[loc, _] : thenState.env)
      places.insert(loc);
    for (auto &[loc, _] : elseState.env)
      places.insert(loc);
    for (const Loc &loc : places) {
      Type type = placeType(loc);
      std::string a = load(loc, thenState, type);
      std::string b = load(loc, elseState, type);
      merged.env[loc] =
          a == b ? a : define(sortOf(type), "(ite " + cond + " " + a + " " + b + ")");
    }
    SmallVector<std::string> results;
    for (auto [i, result] : llvm::enumerate(ifOp->getResults())) {
      if (i >= thenState.yields.size() || i >= elseState.yields.size())
        break;
      results.push_back(define(sortOf(result.getType()),
                               "(ite " + cond + " " + thenState.yields[i] +
                                   " " + elseState.yields[i] + ")"));
    }
    bindResults(ifOp, results);
    merged.pc = state.pc;
    merged.yields = state.yields;
    state = std::move(merged);
  }

  Type placeType(const Loc &loc) {
    if (auto ref = dyn_cast<LIT::RefType>(loc.root.getType()))
      return ref.getElementType();
    return loc.root.getType();
  }

  void bindResults(Operation *op, ArrayRef<std::string> terms) {
    for (auto [i, result] : llvm::enumerate(op->getResults()))
      values[result] = i < terms.size() ? terms[i]
                                        : declare(sortOf(result.getType()));
  }

  /// Code the encoder does not follow: memory is unknown after it, and its
  /// obligations are not analyzed.
  void notAnalyzed(Operation *op, State &state) {
    op->walk([&](LIT::CallOp call) {
      if (inContract)
        return;
      if (LIT::FnOp callee = lookup(call))
        for (Operation &calleeOp : callee.getFunctionBody().front())
          if (auto req = dyn_cast<RequiresOp>(&calleeOp))
            obligations.push_back({state.pc, "true", call.getLoc(),
                                   req.getLoc(), displayName(callee),
                                   /*analyzed=*/false});
    });
    ++state.epoch;
    state.env.clear();
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
      values[op->getResult(0)] = term(op->getOperand(0), state);
      return;
    }
    if (auto store = dyn_cast<LIT::RefStoreOp>(op)) {
      Value dest = store->getOperand(1);
      if (std::optional<Loc> loc = placeOf(dest))
        this->store(*loc, state, term(store->getOperand(0), state));
      else if (auto ref = dyn_cast<LIT::RefType>(dest.getType()))
        havocOrigin(printed(ref.getOrigin()), state);
      else
        havocOrigin("", state);
      return;
    }
    if (auto loadOp = dyn_cast<LIT::RefLoadOp>(op)) {
      Value ref = loadOp->getOperand(0);
      if (std::optional<Loc> loc = placeOf(ref))
        values[loadOp->getResult(0)] =
            load(*loc, state, loadOp->getResult(0).getType());
      else
        setResultsUnknown(op);
      return;
    }
    if (auto call = dyn_cast<LIT::CallOp>(op)) {
      evalCall(call, state);
      return;
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
    if (name && evalBuiltin(call, *name, state))
      return;
    // A call to a function with preconditions: they are obligations here.
    LIT::FnOp callee = lookup(call);
    if (callee && !inContract && !callee.getFunctionBody().empty()) {
      for (Operation &calleeOp : callee.getFunctionBody().front()) {
        auto req = dyn_cast<RequiresOp>(&calleeOp);
        if (!req)
          continue;
        Obligation ob{state.pc, "false", call.getLoc(), req.getLoc(),
                      displayName(callee)};
        if (MaybeTerm cond = instantiate(req, req.getArgs(), state,
                                         call.getOperands(), callee))
          ob.cond = *cond;
        else
          ob.analyzed = false;
        obligations.push_back(ob);
      }
    }
    // The call's effects: its results are unknown, and so is memory it may
    // write: through its mutable reference arguments, and through the
    // mutable origins it is given (e.g. inside a struct passed by value).
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
      if (c == '<' || c == '[' || c == '(' || c == '{')
        ++depth;
      else if (c == '>' || c == ']' || c == ')' || c == '}') {
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

  /// Memory a mutable origin (printed) may reach becomes unknown: the roots
  /// whose declarations it names, or every root if it names none of them.
  void havocOrigin(StringRef origin, State &state) {
    bool matched = false;
    for (Value root : roots) {
      std::string name = quotedOriginName(root);
      if (name.empty() || !origin.contains(name))
        continue;
      havoc(Loc{root, ""}, state, placeType(Loc{root, ""}));
      matched = true;
    }
    if (!matched) {
      ++state.epoch;
      state.env.clear();
    }
  }

  /// Integer and Boolean operators, and `len`.
  bool evalBuiltin(LIT::CallOp call, const CalleeName &name, State &state) {
    StringRef path = name.path;
    if (call->getNumResults() != 1)
      return false;
    Value result = call->getResult(0);
    auto operand = [&](unsigned i) { return term(call.getOperands()[i], state); };
    if (path.starts_with("std::builtin::bool::Bool::__mlir_bool__(") &&
        call.getNumOperands() == 1) {
      values[result] = operand(0);
      return true;
    }
    if (path.starts_with("std::builtin::len::len[") &&
        call.getNumOperands() == 1) {
      values[result] = define({false, 64, true},
                              lenOf(valueThrough(call.getOperands()[0], state)));
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
      for (auto &[m, smt] : binary)
        if (method == m) {
          values[result] = define(*sort, "(" + std::string(smt) + " " +
                                             operand(0) + " " + operand(1) + ")");
          return true;
        }
      for (auto &[m, smt] : compare)
        if (method == m) {
          values[result] = define({true, 1, false},
                                  "(" + std::string(smt) + " " + operand(0) +
                                      " " + operand(1) + ")");
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

  /// The condition of a `kgen.requires` region of `callee` (the function
  /// itself when `callee` is null), with its block arguments bound to the
  /// caller's `operands` (the function's own arguments when empty).
  MaybeTerm instantiate(RequiresOp req, OperandRange args, State &state,
                        ValueRange operands, LIT::FnOp callee = {}) {
    Block &body = req.getBody().front();
    Block &calleeEntry = (callee ? callee : fn).getFunctionBody().front();
    DenseMap<Value, std::string> saved = values;
    SmallVector<Value> boundRefs;
    for (auto [i, blockArg] : llvm::enumerate(body.getArguments())) {
      if (i >= args.size())
        return std::nullopt;
      Value formal = args[i];
      auto formalArg = dyn_cast<BlockArgument>(formal);
      if (!formalArg || formalArg.getOwner() != &calleeEntry)
        return std::nullopt;
      Value actual = formal;
      if (callee) {
        if (formalArg.getArgNumber() >= operands.size())
          return std::nullopt;
        actual = operands[formalArg.getArgNumber()];
      }
      if (isa<LIT::RefType>(blockArg.getType())) {
        std::optional<Loc> loc = placeOf(actual);
        if (!loc)
          return std::nullopt;
        refArgs[blockArg] = *loc;
        boundRefs.push_back(blockArg);
      } else {
        values[blockArg] = term(actual, state);
      }
    }
    State inner = state;
    inner.yields.clear();
    ++inContract;
    walkBlock(body, inner);
    --inContract;
    for (Value ref : boundRefs)
      refArgs.erase(ref);
    MaybeTerm result;
    if (inner.yields.size() == 1)
      result = inner.yields.front();
    // Values of the region's ops belong to this instantiation only.
    values = std::move(saved);
    return result;
  }
};

//===----------------------------------------------------------------------===//
// Solver
//===----------------------------------------------------------------------===//

/// Runs z3 on `script` with a wall-clock cap; returns one answer per query.
std::optional<SmallVector<Answer>> runZ3(StringRef z3, StringRef script,
                                         unsigned capSeconds,
                                         StringRef dumpPath) {
  SmallString<128> scriptPath, outPath;
  if (!dumpPath.empty())
    scriptPath = dumpPath;
  else if (llvm::sys::fs::createTemporaryFile("verify", "smt2", scriptPath))
    return std::nullopt;
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
  int rc = llvm::sys::ExecuteAndWait(z3, {z3, "-smt2", scriptPath},
                                     std::nullopt, redirects, capSeconds);
  auto buffer = llvm::MemoryBuffer::getFile(outPath);
  llvm::sys::fs::remove(outPath);
  if (dumpPath.empty())
    llvm::sys::fs::remove(scriptPath);
  if (!buffer)
    return std::nullopt;
  SmallVector<Answer> answers;
  SmallVector<StringRef> segments;
  (*buffer)->getBuffer().split(segments, "@@");
  for (StringRef segment : ArrayRef(segments).drop_front()) {
    StringRef answer = segment.trim().split('\n').first.trim();
    answers.push_back(answer == "unsat" ? Answer::Proven
                      : answer == "sat" ? Answer::Unproven
                                        : Answer::Unknown);
  }
  // A process killed at the cap answers nothing for the rest.
  (void)rc;
  return answers;
}

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
    unsigned total = 0, proven = 0, index = 0;
    getOperation().walk([&](LIT::FnOp fn) {
      if (!includeStdlib && inStdlib(fn))
        return;
      FunctionEncoder enc(fn, getOperation(), symbols);
      if (!enc.encode() || enc.obligations.empty())
        return;
      SmallString<128> dump;
      if (!dumpDir.empty()) {
        dump = dumpDir;
        llvm::sys::path::append(dump, "f" + Twine(index++) + ".smt2");
      }
      std::optional<SmallVector<Answer>> answers =
          runZ3(z3, enc.script(rlimit), wallSeconds, dump);
      unsigned next = 0;
      for (Obligation &ob : enc.obligations) {
        Answer answer = Answer::NotAnalyzed;
        if (ob.analyzed)
          answer = answers && next < answers->size() ? (*answers)[next]
                                                     : Answer::Unknown;
        if (ob.analyzed)
          ++next;
        ++total;
        if (answer == Answer::Proven) {
          ++proven;
          if (verbose)
            mlir::emitRemark(ob.callLoc)
                << "precondition of '" << ob.callee << "' proven";
          continue;
        }
        auto diag = mlir::emitWarning(ob.callLoc);
        switch (answer) {
        case Answer::Unproven:
          diag << "cannot prove the precondition of '" << ob.callee << "'";
          break;
        case Answer::Unknown:
          diag << "the precondition of '" << ob.callee
               << "' was not decided within the solver limits";
          break;
        default:
          diag << "the precondition of '" << ob.callee
               << "' is not analyzed yet (loops and 'try' are not supported)";
          break;
        }
        diag.attachNote(ob.clauseLoc) << "precondition declared here";
      }
    });
    llvm::errs() << "verify-contracts: " << proven << "/" << total
                 << " preconditions proven\n";
  }
};

} // namespace
