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
#include "Mojo/Interpreter/InterpreterAttrs.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/POPDialect/POPAttrs.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/xxhash.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Regex.h"

#include <chrono>
#include <functional>
#include <future>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <signal.h>
#include <sys/wait.h>
#include <thread>

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

/// Whether the IR printer can print `entity`: it asserts on a member alias
/// whose sugared value is not a type (seen in max/kernels/src/algorithm,
/// where `--mlir-print-ir-after-all` asserts too).
template <typename Entity>
bool printable(Entity entity) {
  bool ok = true;
  entity.walk([&](SugarAttr sugar) {
    if (sugar.getKind() == SugarKind::MemberAlias &&
        !isa<TypeParamAttr>(sugar.getSugared()))
      ok = false;
  });
  return ok;
}

/// The printed IR of `type`, or "<unprintable>" (which no model matches,
/// so its values are opaque).
std::string printed(Type type) {
  if (!printable(type))
    return "<unprintable>";
  std::string text;
  llvm::raw_string_ostream os(text);
  type.print(os);
  return text;
}

std::string printed(Attribute attr) {
  if (!printable(attr))
    return "<unprintable>";
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

/// The elements of a printed list whose opening bracket was just consumed
/// from `text` (`a, b<c, d>]`), up to its closing `]` or `>`; `text` is
/// left after it. The arrow of a function type (`-> T`) is not a bracket.
SmallVector<StringRef> listElements(StringRef &text) {
  SmallVector<StringRef> elements;
  int depth = 0;
  size_t start = 0;
  for (size_t c = 0; c < text.size(); ++c) {
    char ch = text[c];
    if (ch == '>' && c > 0 && text[c - 1] == '-')
      continue;
    if (ch == '<' || ch == '[' || ch == '(' || ch == '{') {
      ++depth;
    } else if ((ch == ']' || ch == '>') && depth == 0) {
      if (!text.slice(start, c).trim().empty())
        elements.push_back(text.slice(start, c).trim());
      text = text.drop_front(c + 1);
      return elements;
    } else if (ch == '>' || ch == ')' || ch == '}' || ch == ']') {
      --depth;
    } else if (ch == ',' && depth == 0) {
      elements.push_back(text.slice(start, c).trim());
      start = c + 1;
    }
  }
  text = StringRef();
  return {};
}

/// List `which` (0 the shape, 1 the strides) of a printed `TileTensor`
/// layout type, as one printed type per mode; empty if it is not one.
SmallVector<StringRef> layoutList(StringRef layout, unsigned which) {
  size_t at = layout.find("@layout::@tile_layout::@Layout<");
  if (at == StringRef::npos)
    return {};
  StringRef rest = layout.drop_front(at);
  SmallVector<StringRef> elements;
  for (unsigned i = 0; i <= which; ++i) {
    size_t open = rest.find('[');
    if (open == StringRef::npos)
      return {};
    rest = rest.drop_front(open + 1);
    elements = listElements(rest);
  }
  return elements;
}

/// The value of a printed `ComptimeInt` type with a literal value
/// (`ComptimeInt<:T {:scalar<index> 8}>`). An unevaluated one
/// (`ComptimeInt<:T apply(f, 8, 4)>`, from `vectorize`) has its operands
/// printed, not its value, and gives none.
std::optional<int64_t> comptimeIntValue(StringRef type) {
  static llvm::Regex literal("\\{:scalar<index> (-?[0-9]+)\\}>$");
  SmallVector<StringRef> m;
  int64_t n;
  if (!type.starts_with("@std::@utils::@coord::@ComptimeInt<") ||
      type.contains("apply(") || !literal.match(type, &m) ||
      m[1].getAsInteger(10, n))
    return std::nullopt;
  return n;
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

/// Whether values of `type` are integers or Booleans the encoding models
/// exactly (not opaque handles).
bool isScalar(Type type) {
  std::string text = printed(type);
  StringRef t(text);
  if (sortOf(type).isBool || t == "index" || t.starts_with("!kgen.scalar<"))
    return true;
  t.consume_front("!kgen.param<:meta<");
  return t.starts_with("!lit.struct<@std::@simd::@SIMD<") && isWidthOne(t) &&
         dtypeSort(t).has_value();
}

std::string bvConst(int64_t value, unsigned width) {
  // Truncated to the width: a literal like 255 is a valid `UInt8`, and
  // wider values wrap as the dtype does.
  APInt bits = APInt(64, static_cast<uint64_t>(value)).trunc(width);
  return "(_ bv" + llvm::toString(bits, 10, /*Signed=*/false) + " " +
         std::to_string(width) + ")";
}

/// The local closure a call's callee names: the reference itself (`f()`),
/// or the one all of whose parameters the callee binds (`f[4]()`, to
/// `bound`). Null for any other callee.
Attribute closureCallee(Attribute callee, ArrayRef<TypedAttr> &bound) {
  if (auto bind = dyn_cast<BindParamsAttr>(callee)) {
    for (TypedAttr value : bind.getParamValues())
      if (isa<UnboundAttr>(value))
        return {};
    bound = bind.getParamValues();
    callee = bind.getGenerator();
  }
  return isa<ParamDeclRefAttr>(callee) ? callee : Attribute();
}

/// `callee`'s symbol as `a::b::c`, and its parameter values as text.
struct CalleeName {
  std::string path;
  SmallVector<std::string> params;
};

std::optional<CalleeName> calleeName(Attribute callee) {
  auto symbol = dyn_cast<SymbolConstantAttr>(callee);
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

/// The kernel a launch (`ctx.enqueue_function[kernel](...)`, whose callee
/// is `call`) runs, and the symbol naming it with its parameters: the
/// function among the call's parameters, behind the `rebind` and the thunk
/// that converts it to the declared `def` type (a function of its own,
/// named `def[...]`, whose parameters hold the kernel). `resolve` resolves
/// a parameter in the caller's scope.
std::pair<LIT::FnOp, SymbolConstantAttr>
launchedKernel(SymbolConstantAttr call, ModuleOp module,
               SymbolTableCollection &symbols,
               llvm::function_ref<TypedAttr(TypedAttr)> resolve) {
  std::pair<LIT::FnOp, SymbolConstantAttr> found;
  std::function<void(TypedAttr, int)> search = [&](TypedAttr param, int depth) {
    if (found.first || depth > 3)
      return;
    param = resolve(param);
    while (auto expr = dyn_cast<ParamOperatorAttr>(param)) {
      if (expr.getOpcode() != POC::Rebind || expr.getOperands().size() != 1)
        return;
      param = resolve(expr.getOperands()[0]);
    }
    auto symbol = dyn_cast<SymbolConstantAttr>(param);
    if (!symbol)
      return;
    auto fn = dyn_cast_or_null<LIT::FnOp>(
        symbols.lookupSymbolIn(module, symbol.getSymbol()));
    if (fn && !fn.getFunctionBody().empty() &&
        !symbol.getSymbol().getRootReference().getValue().starts_with("def[")) {
      found = {fn, symbol};
      return;
    }
    for (TypedAttr inner : symbol.getParamValues())
      search(inner, depth + 1);
  };
  for (TypedAttr param : call.getParamValues())
    search(param, 0);
  return found;
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
  /// The limit of each loop-invariant candidate's query (0: a twentieth
  /// of the function's query limit).
  unsigned houdiniRlimit = 0;
  unsigned wallSeconds = 60;
  std::string dumpDir;
  /// Where answers are cached by a hash of the script (empty: no cache).
  std::string cacheDir;
  /// The limit of each query whose goal is nonlinear (0: the query
  /// limit). Bit-vector products and quotients of unknowns can use up any
  /// limit where the integer retry decides them at once.
  unsigned nonlinearRlimit = 0;
  /// Whether `runCases` splits open queries over finite-domain unknowns.
  bool caseSplit = true;
  /// Whether an undecided invariant candidate is asked again with four
  /// times its limit.
  bool invariantRetry = false;
  /// Whether a query still undecided is asked over the integers with the
  /// products that provably do not overflow taken exactly
  /// (`exactIntegers`); with the integer retry.
  bool exactRetry = false;
  /// The scripts already run, or running, in this run of the pass.
  std::shared_ptr<struct ScriptMemo> memo = {};
};

/// Runs `program` and waits at most `seconds` for it; returns its exit
/// code, or -1 if it could not run or was killed at the cap. Not
/// `ExecuteAndWait`'s cap: that is the process-wide `alarm()`, which the
/// parallel solver runs overwrite (LLVM's own FIXME: "The alarm signal may
/// be delivered to another thread"), so z3 processes ran past it for an
/// hour and more. This polls the child instead.
int runWithCap(StringRef program, ArrayRef<StringRef> args,
               ArrayRef<std::optional<StringRef>> redirects, unsigned seconds) {
  bool failed = false;
  llvm::sys::ProcessInfo pi = llvm::sys::ExecuteNoWait(
      program, args, std::nullopt, redirects, 0, nullptr, &failed);
  if (failed || pi.Pid == llvm::sys::ProcessInfo::InvalidPid)
    return -1;
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
  auto pause = std::chrono::milliseconds(1);
  int status = 0;
  while (true) {
    pid_t done = ::waitpid(pi.Pid, &status, WNOHANG);
    if (done == pi.Pid)
      return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    if (done == -1 && errno != EINTR)
      return -1;
    if (std::chrono::steady_clock::now() >= deadline) {
      ::kill(pi.Pid, SIGKILL);
      while (::waitpid(pi.Pid, &status, 0) == -1 && errno == EINTR) {
      }
      return -1;
    }
    std::this_thread::sleep_for(pause);
    pause = std::min(pause * 2, std::chrono::milliseconds(50));
  }
}

/// The answers to the scripts of one run of the pass, by script: the same
/// script is run once (the instantiations of a kernel that differ in a
/// parameter its proof does not read have the same scripts), and whoever
/// asks while it runs waits for it.
struct ScriptMemo {
  using Answers = std::optional<SmallVector<Answer>>;
  std::mutex mutex;
  std::map<std::string, std::shared_future<Answers>> answers;
};

std::optional<SmallVector<Answer>>
runScript(const SolverConfig &config, StringRef script, StringRef dumpName);

/// Runs z3 on `script` with a wall-clock cap; returns one answer per query
/// (queries are separated by `(echo "@@")`). A script already run is not
/// run, or dumped, again.
std::optional<SmallVector<Answer>> runZ3(const SolverConfig &config,
                                         StringRef script, StringRef dumpName) {
  if (!config.memo)
    return runScript(config, script, dumpName);
  std::promise<ScriptMemo::Answers> promise;
  std::shared_future<ScriptMemo::Answers> running;
  {
    std::lock_guard<std::mutex> lock(config.memo->mutex);
    auto [it, inserted] = config.memo->answers.try_emplace(script.str());
    if (inserted)
      it->second = promise.get_future().share();
    else
      running = it->second;
  }
  if (running.valid())
    return running.get();
  ScriptMemo::Answers answers = runScript(config, script, dumpName);
  promise.set_value(answers);
  return answers;
}

std::optional<SmallVector<Answer>>
runScript(const SolverConfig &config, StringRef script, StringRef dumpName) {
  // The answers to a script are a function of the script and the solver:
  // the limits are deterministic, and runs stopped at the wall-clock cap are
  // not cached.
  size_t expected = script.count("(check-sat)");
  SmallString<128> cachePath;
  if (!config.cacheDir.empty()) {
    uint64_t hash = llvm::xxh3_64bits(script);
    // The solver: its path, and its binary's size and modification time,
    // so an upgraded z3 does not reuse answers.
    std::string solver = config.z3;
    llvm::sys::fs::file_status status;
    if (!llvm::sys::fs::status(config.z3, status))
      solver += "|" + std::to_string(status.getSize()) + "|" +
                std::to_string(llvm::sys::toTimeT(status.getLastModificationTime()));
    uint64_t solverHash = llvm::xxh3_64bits(solver);
    cachePath = config.cacheDir;
    llvm::sys::path::append(cachePath, llvm::utohexstr(hash) + "-" +
                                           llvm::utohexstr(solverHash) +
                                           ".answers");
    if (auto cached = llvm::MemoryBuffer::getFile(cachePath)) {
      SmallVector<Answer> answers;
      for (char c : (*cached)->getBuffer())
        if (c == 'p' || c == 'u' || c == 'k')
          answers.push_back(c == 'p'   ? Answer::Proven
                            : c == 'u' ? Answer::Unproven
                                       : Answer::Unknown);
      if (answers.size() == expected)
        return answers;
    }
  }
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
  int rc = runWithCap(z3, {z3, "-smt2", scriptPath}, redirects,
                      config.wallSeconds);
  if (rc != 0)
    cachePath.clear(); // Stopped (e.g. at the cap): do not cache.
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
  if (!cachePath.empty() && answers.size() == expected) {
    std::string text;
    for (Answer answer : answers)
      text += answer == Answer::Proven     ? 'p'
              : answer == Answer::Unproven ? 'u'
                                           : 'k';
    // Written to a temporary and renamed, so a parallel reader never sees a
    // partial file.
    SmallString<128> temp;
    int fd;
    if (!llvm::sys::fs::createUniqueFile(cachePath + ".%%%%%%", fd, temp)) {
      {
        llvm::raw_fd_ostream os(fd, /*shouldClose=*/true);
        os << text;
      }
      if (llvm::sys::fs::rename(temp, cachePath))
        llvm::sys::fs::remove(temp);
    }
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
// Integer retry
//===----------------------------------------------------------------------===//
//
// A query the solver answers `unknown` over bit-vectors (typically one with
// products of unknowns, `row * k + col < m * k`) is asked again over
// mathematical integers, which z3's nonlinear arithmetic handles far better.
// The translation is exact where it can be: a `w`-bit value is its signed
// value, arithmetic wraps (`mod 2^w`), unsigned operations work on the
// unsigned view, division keeps SMT-LIB's results for a zero divisor, and
// shifts, masks, extracts, extensions and concatenations by constants are
// arithmetic. Other bit operations become unknowns in range, which only
// makes a proof harder. So an `unsat` over the integers is an `unsat` over
// the bit-vectors; anything else stays unknown.

struct SExpr {
  std::string atom;
  std::vector<SExpr> list;
  bool isAtom() const { return list.empty() && !atom.empty(); }
};

/// Parses the scripts this pass writes: lists, atoms, string literals and
/// `;` comments (dropped).
std::optional<std::vector<SExpr>> parseSExprs(StringRef text) {
  std::vector<std::vector<SExpr>> stack(1);
  size_t i = 0;
  while (i < text.size()) {
    char c = text[i];
    if (c == ';') {
      while (i < text.size() && text[i] != '\n')
        ++i;
    } else if (isspace(static_cast<unsigned char>(c))) {
      ++i;
    } else if (c == '(') {
      stack.emplace_back();
      ++i;
    } else if (c == ')') {
      if (stack.size() < 2)
        return std::nullopt;
      SExpr e;
      e.list = std::move(stack.back());
      stack.pop_back();
      if (e.list.empty())
        e.atom = "()";
      stack.back().push_back(std::move(e));
      ++i;
    } else if (c == '"') {
      size_t j = text.find('"', i + 1);
      if (j == StringRef::npos)
        return std::nullopt;
      stack.back().push_back({text.slice(i, j + 1).str(), {}});
      i = j + 1;
    } else {
      size_t j = i;
      while (j < text.size() && !isspace(static_cast<unsigned char>(text[j])) &&
             text[j] != '(' && text[j] != ')' && text[j] != ';')
        ++j;
      stack.back().push_back({text.slice(i, j).str(), {}});
      i = j;
    }
  }
  if (stack.size() != 1)
    return std::nullopt;
  return std::move(stack.front());
}

std::string printSExpr(const SExpr &e) {
  if (e.list.empty())
    return e.atom;
  std::string text = "(";
  for (auto [k, child] : llvm::enumerate(e.list))
    text += (k ? " " : "") + printSExpr(child);
  return text + ")";
}

class IntTranslator {
public:
  /// The translated script, or nothing if it uses what is not translated.
  std::optional<std::string> translate(StringRef script) {
    std::optional<std::vector<SExpr>> commands = parseSExprs(script);
    if (!commands)
      return std::nullopt;
    for (const SExpr &cmd : *commands)
      if (!command(cmd))
        return std::nullopt;
    return out;
  }

  /// A defined term that is a sum, difference or product at the top: its
  /// translation without the wrapping, its width, and (for a candidate of
  /// the exact retry) whether it is a product of two terms that are not
  /// literals, or a sum or difference with such a product as an operand.
  struct Arithmetic {
    std::string exact;
    unsigned width;
    bool candidate;
  };
  /// Those of the commands translated so far, in order.
  std::vector<std::pair<std::string, Arithmetic>> arithmetic;
  /// Defined terms translated without the wrapping (see `exactIntegers`).
  std::set<std::string> exact;

  /// The translation so far.
  const std::string &text() const { return out; }

private:
  std::string out;
  /// The body of the definition being translated, and its translation
  /// without the wrapping if it is arithmetic at the top.
  const SExpr *top = nullptr;
  std::string topExact;
  std::set<std::string> products;
  /// Bit width of each constant and defined term (0: not a bit-vector).
  llvm::StringMap<unsigned> widths;
  /// Result width of each declared function.
  llvm::StringMap<unsigned> functionWidths;
  /// Variables bound by enclosing quantifiers, with their widths.
  SmallVector<std::pair<std::string, unsigned>> bound;
  /// Unknowns standing for untranslated terms, by the term's text, with the
  /// `push` depth they were declared at.
  std::map<std::string, std::pair<std::string, unsigned>> abstractions;
  std::string pending; // Their declarations, written before the command.
  /// Applications of declared functions whose range is asserted, with the
  /// `push` depth it was asserted at.
  std::set<std::pair<std::string, unsigned>> ranged;
  unsigned depth = 0, counter = 0;

  static std::string pow2(unsigned k) {
    return llvm::toString(APInt::getOneBitSet(k + 1, k), 10, false);
  }
  static std::string range(unsigned w, const std::string &x) {
    return "(and (<= (- " + pow2(w - 1) + ") " + x + ") (< " + x + " " +
           pow2(w - 1) + "))";
  }
  static std::string wrap(unsigned w, const std::string &x) {
    return "(- (mod (+ " + x + " " + pow2(w - 1) + ") " + pow2(w) + ") " +
           pow2(w - 1) + ")";
  }
  static std::string unsignedView(unsigned w, const std::string &x) {
    return "(ite (< " + x + " 0) (+ " + x + " " + pow2(w) + ") " + x + ")";
  }
  /// The signed value of the unsigned `w`-bit value `x`.
  static std::string signedView(unsigned w, const std::string &x) {
    return "(ite (>= " + x + " " + pow2(w - 1) + ") (- " + x + " " + pow2(w) +
           ") " + x + ")";
  }

  static std::optional<unsigned> bvSort(const SExpr &sort) {
    unsigned w;
    if (sort.list.size() == 3 && sort.list[0].atom == "_" &&
        sort.list[1].atom == "BitVec" &&
        !StringRef(sort.list[2].atom).getAsInteger(10, w) && w > 0)
      return w;
    return std::nullopt;
  }
  /// The sort of the translation: `Int` for a bit-vector, else as is.
  static std::optional<std::string> sortText(const SExpr &sort,
                                             unsigned &width) {
    if (std::optional<unsigned> w = bvSort(sort)) {
      width = *w;
      return std::string("Int");
    }
    width = 0;
    if (sort.atom == "Bool" || sort.atom == "Int")
      return sort.atom;
    return std::nullopt;
  }

  bool command(const SExpr &cmd) {
    if (cmd.list.empty())
      return false;
    StringRef head = cmd.list[0].atom;
    std::string text;
    if (head == "declare-const" && cmd.list.size() == 3) {
      unsigned w;
      std::optional<std::string> sort = sortText(cmd.list[2], w);
      if (!sort)
        return false;
      widths[cmd.list[1].atom] = w;
      text = "(declare-const " + cmd.list[1].atom + " " + *sort + ")\n";
      if (w)
        text += "(assert " + range(w, cmd.list[1].atom) + ")\n";
    } else if (head == "declare-fun" && cmd.list.size() == 4) {
      std::string args;
      for (const SExpr &arg : cmd.list[2].list) {
        unsigned w;
        std::optional<std::string> sort = sortText(arg, w);
        if (!sort)
          return false;
        args += (args.empty() ? "" : " ") + *sort;
      }
      unsigned w;
      std::optional<std::string> result = sortText(cmd.list[3], w);
      if (!result)
        return false;
      functionWidths[cmd.list[1].atom] = w;
      text = "(declare-fun " + cmd.list[1].atom + " (" + args + ") " +
             *result + ")\n";
    } else if (head == "define-fun" && cmd.list.size() == 5 &&
               cmd.list[2].list.empty()) {
      unsigned w;
      std::optional<std::string> sort = sortText(cmd.list[3], w);
      unsigned bodyWidth;
      top = &cmd.list[4];
      topExact.clear();
      std::optional<std::string> body =
          sort ? term(cmd.list[4], bodyWidth) : std::nullopt;
      top = nullptr;
      if (!body || bodyWidth != w)
        return false;
      if (!topExact.empty() && w && !depth) {
        const std::string &name = cmd.list[1].atom;
        const SExpr &e = cmd.list[4];
        unsigned lw;
        size_t terms = 0;
        bool overProduct = false;
        for (size_t i = 1; i < e.list.size(); ++i) {
          terms += !literal(e.list[i], lw);
          overProduct |= products.count(e.list[i].atom) != 0;
        }
        bool product = e.list[0].atom == "bvmul" && terms >= 2;
        if (product)
          products.insert(name);
        arithmetic.push_back({name, {topExact, w, product || overProduct}});
        if (exact.count(name))
          body = topExact;
      }
      widths[cmd.list[1].atom] = w;
      text = "(define-fun " + cmd.list[1].atom + " () " + *sort + " " + *body +
             ")\n";
    } else if (head == "assert" && cmd.list.size() == 2) {
      unsigned w;
      std::optional<std::string> body = term(cmd.list[1], w);
      if (!body || w)
        return false;
      text = "(assert " + *body + ")\n";
    } else if (head == "push") {
      ++depth;
      text = printSExpr(cmd) + "\n";
    } else if (head == "pop") {
      if (depth)
        --depth;
      for (auto it = abstractions.begin(); it != abstractions.end();)
        it = it->second.second > depth ? abstractions.erase(it) : std::next(it);
      for (auto it = ranged.begin(); it != ranged.end();)
        it = it->second > depth ? ranged.erase(it) : std::next(it);
      text = printSExpr(cmd) + "\n";
    } else if (head == "set-option" || head == "echo" ||
               head == "check-sat") {
      text = printSExpr(cmd) + "\n";
    } else {
      return false;
    }
    out += pending + text;
    pending.clear();
    return true;
  }

  std::optional<unsigned> widthOf(StringRef name) const {
    for (auto it = bound.rbegin(); it != bound.rend(); ++it)
      if (it->first == name)
        return it->second;
    auto it = widths.find(name);
    if (it == widths.end())
      return std::nullopt;
    return it->second;
  }

  bool mentionsBound(const SExpr &e) const {
    if (e.list.empty())
      return llvm::any_of(bound, [&](auto &b) { return b.first == e.atom; });
    return llvm::any_of(e.list, [&](const SExpr &c) { return mentionsBound(c); });
  }

  /// An unknown `w`-bit value in range for `e`, the same for the same text.
  std::optional<std::string> abstraction(const SExpr &e, unsigned w) {
    if (mentionsBound(e))
      return std::nullopt;
    std::string key = printSExpr(e);
    auto it = abstractions.find(key);
    if (it != abstractions.end())
      return it->second.first;
    std::string name = "ia" + std::to_string(counter++);
    abstractions[key] = {name, depth};
    pending += "(declare-const " + name + " Int)\n(assert " + range(w, name) +
               ")\n";
    return name;
  }

  static std::optional<APInt> literal(const SExpr &e, unsigned &w) {
    StringRef a = e.atom;
    if (e.list.size() == 3 && e.list[0].atom == "_" &&
        StringRef(e.list[1].atom).starts_with("bv")) {
      if (StringRef(e.list[2].atom).getAsInteger(10, w) || !w)
        return std::nullopt;
      APInt v;
      if (StringRef(e.list[1].atom).drop_front(2).getAsInteger(10, v))
        return std::nullopt;
      return v.zextOrTrunc(w);
    }
    if (a.starts_with("#x") || a.starts_with("#b")) {
      unsigned radix = a[1] == 'x' ? 16 : 2;
      w = (a.size() - 2) * (radix == 16 ? 4 : 1);
      APInt v;
      if (!w || a.drop_front(2).getAsInteger(radix, v))
        return std::nullopt;
      return v.zextOrTrunc(w);
    }
    return std::nullopt;
  }

  static std::optional<unsigned> smallConstant(const SExpr &e) {
    unsigned w;
    std::optional<APInt> v = literal(e, w);
    if (!v || v->getActiveBits() > 16)
      return std::nullopt;
    return static_cast<unsigned>(v->getZExtValue());
  }

  /// The translation of `e`, and its bit width (0 for a Boolean).
  std::optional<std::string> term(const SExpr &e, unsigned &w) {
    if (unsigned lw; std::optional<APInt> v = literal(e, lw)) {
      w = lw;
      return llvm::toString(*v, 10, /*Signed=*/true);
    }
    if (e.list.empty()) {
      if (e.atom == "true" || e.atom == "false") {
        w = 0;
        return e.atom;
      }
      std::optional<unsigned> width = widthOf(e.atom);
      if (!width)
        return std::nullopt;
      w = *width;
      return e.atom;
    }
    const SExpr &head = e.list[0];
    ArrayRef<SExpr> args = ArrayRef(e.list).drop_front();
    // Indexed operators: `((_ extract hi lo) x)`, `((_ zero_extend n) x)`.
    if (!head.list.empty() && head.list.size() >= 3 &&
        head.list[0].atom == "_" && args.size() == 1) {
      unsigned xw;
      std::optional<std::string> x = term(args[0], xw);
      if (!x || !xw)
        return std::nullopt;
      StringRef op = head.list[1].atom;
      unsigned a, b = 0;
      if (StringRef(head.list[2].atom).getAsInteger(10, a) ||
          (head.list.size() == 4 &&
           StringRef(head.list[3].atom).getAsInteger(10, b)))
        return std::nullopt;
      if (op == "sign_extend") {
        w = xw + a;
        return x;
      }
      if (op == "zero_extend") {
        w = xw + a;
        return a ? unsignedView(xw, *x) : *x;
      }
      if (op == "extract" && head.list.size() == 4 && a >= b && a < xw) {
        w = a - b + 1;
        return signedView(w, "(mod (div " + unsignedView(xw, *x) + " " +
                                 pow2(b) + ") " + pow2(w) + ")");
      }
      return std::nullopt;
    }
    StringRef op = head.atom;
    if (op == "forall" || op == "exists") {
      if (args.size() != 2)
        return std::nullopt;
      std::string vars, guard = "true";
      size_t mark = bound.size();
      for (const SExpr &v : args[0].list) {
        if (v.list.size() != 2)
          return std::nullopt;
        unsigned vw;
        std::optional<std::string> sort = sortText(v.list[1], vw);
        if (!sort)
          return std::nullopt;
        vars += "(" + v.list[0].atom + " " + *sort + ")";
        bound.push_back({v.list[0].atom, vw});
        if (vw)
          guard = "(and " + guard + " " + range(vw, v.list[0].atom) + ")";
      }
      unsigned bw;
      std::optional<std::string> body = term(args[1], bw);
      bound.resize(mark);
      if (!body || bw)
        return std::nullopt;
      w = 0;
      return "(" + op.str() + " (" + vars + ") (" +
             (op == "forall" ? "=>" : "and") + " " + guard + " " + *body +
             "))";
    }
    SmallVector<std::string> xs;
    SmallVector<unsigned> ws;
    for (const SExpr &arg : args) {
      unsigned aw;
      std::optional<std::string> x = term(arg, aw);
      if (!x)
        return std::nullopt;
      xs.push_back(*x);
      ws.push_back(aw);
    }
    auto joined = [&] {
      std::string text;
      for (const std::string &x : xs)
        text += " " + x;
      return text;
    };
    // Booleans and the structural operators.
    if (op == "and" || op == "or" || op == "not" || op == "=>" ||
        op == "xor") {
      w = 0;
      return "(" + op.str() + joined() + ")";
    }
    if (op == "=" || op == "distinct") {
      w = 0;
      return "(" + op.str() + joined() + ")";
    }
    if (op == "ite" && xs.size() == 3) {
      w = ws[1];
      return "(ite" + joined() + ")";
    }
    // A declared function: its arguments translated, its result in range
    // (asserted once per application, or wrapped under a quantifier).
    if (auto it = functionWidths.find(op); it != functionWidths.end()) {
      w = it->second;
      std::string app = "(" + op.str() + joined() + ")";
      if (!w)
        return app;
      if (mentionsBound(e))
        return wrap(w, app);
      if (ranged.insert({app, depth}).second)
        pending += "(assert " + range(w, app) + ")\n";
      return app;
    }
    if (xs.empty() || !ws[0])
      return std::nullopt;
    unsigned bw = ws[0];
    std::string a = xs[0], b = xs.size() > 1 ? xs[1] : "";
    auto compare = [&](StringRef cmp, bool isUnsigned) {
      w = 0;
      if (isUnsigned)
        return "(" + cmp.str() + " " + unsignedView(bw, a) + " " +
               unsignedView(bw, b) + ")";
      return "(" + cmp.str() + " " + a + " " + b + ")";
    };
    if (xs.size() == 2) {
      if (op == "bvslt") return compare("<", false);
      if (op == "bvsle") return compare("<=", false);
      if (op == "bvsgt") return compare(">", false);
      if (op == "bvsge") return compare(">=", false);
      if (op == "bvult") return compare("<", true);
      if (op == "bvule") return compare("<=", true);
      if (op == "bvugt") return compare(">", true);
      if (op == "bvuge") return compare(">=", true);
    }
    w = bw;
    auto arith = [&](const std::string &x) {
      if (&e == top)
        topExact = x;
      return wrap(bw, x);
    };
    if (op == "bvadd")
      return arith("(+" + joined() + ")");
    if (op == "bvmul")
      return arith("(*" + joined() + ")");
    if (op == "bvsub" && xs.size() == 2)
      return arith("(- " + a + " " + b + ")");
    if (op == "bvneg" && xs.size() == 1)
      return wrap(bw, "(- " + a + ")");
    if (xs.size() == 2 && (op == "bvudiv" || op == "bvurem")) {
      std::string ua = unsignedView(bw, a), ub = unsignedView(bw, b);
      std::string r = signedView(
          bw, "(" + std::string(op == "bvudiv" ? "div" : "mod") + " " + ua +
                  " " + ub + ")");
      // SMT-LIB: `x udiv 0` is all ones (-1), `x urem 0` is `x`.
      return "(ite (= " + b + " 0) " + (op == "bvudiv" ? "(- 1)" : a) + " " +
             r + ")";
    }
    if (xs.size() == 2 && (op == "bvsdiv" || op == "bvsrem")) {
      // Truncating division of the magnitudes, with the signs applied.
      std::string q = "(div (abs " + a + ") (abs " + b + "))";
      std::string tq = "(ite (= (< " + a + " 0) (< " + b + " 0)) " + q +
                       " (- " + q + "))";
      if (op == "bvsdiv")
        return "(ite (= " + b + " 0) (ite (< " + a + " 0) 1 (- 1)) " +
               wrap(bw, tq) + ")";
      return "(ite (= " + b + " 0) " + a + " (- " + a + " (* " + b + " " + tq +
             ")))";
    }
    if (op == "concat" && xs.size() == 2 && ws[1]) {
      w = bw + ws[1];
      return signedView(w, "(+ (* " + unsignedView(bw, a) + " " +
                               pow2(ws[1]) + ") " + unsignedView(ws[1], b) +
                               ")");
    }
    // Shifts and masks by constants.
    if (xs.size() == 2)
      if (std::optional<unsigned> k = smallConstant(args[1])) {
        if (op == "bvshl")
          return *k >= bw ? std::string("0")
                          : wrap(bw, "(* " + a + " " + pow2(*k) + ")");
        if (op == "bvlshr")
          return *k >= bw ? std::string("0")
                          : signedView(bw, "(div " + unsignedView(bw, a) +
                                               " " + pow2(*k) + ")");
        if (op == "bvashr")
          return "(div " + a + " " + pow2(std::min(*k, bw - 1)) + ")";
      }
    if (op == "bvand" && xs.size() == 2) {
      unsigned lw;
      if (std::optional<APInt> mask = literal(args[1], lw);
          mask && mask->isMask() && mask->countr_one() < bw)
        return signedView(bw, "(mod " + unsignedView(bw, a) + " " +
                                  pow2(mask->countr_one()) + ")");
    }
    if (StringRef(op).starts_with("bv"))
      return abstraction(e, bw);
    return std::nullopt;
  }

public:
  static std::string inRange(unsigned w, const std::string &x) {
    return range(w, x);
  }
};

/// The exact integer retry. Over the integers every sum and product still
/// wraps (`mod 2^64`), and that is what keeps the solver from a bound like
/// `row * n + col < k * n`: without the wrapping it is immediate. So, for
/// each query of `script` (a header, then `query` blocks): first, for each
/// product of two unknowns the header defines, and each sum over one, is
/// its exact value in range under the query's assumptions? Then the query
/// again, with the terms for which that was proven defined without the
/// wrapping. That is the same query: in every model of the assumptions
/// those terms have their exact values. Returns which queries are proven.
std::vector<bool> exactIntegers(const SolverConfig &config, StringRef script,
                                StringRef name) {
  StringRef marker = "(echo \"@@\")";
  size_t first = script.find(marker);
  if (first == StringRef::npos)
    return {};
  StringRef head = script.take_front(first);
  SmallVector<StringRef> queries;
  for (StringRef rest = script.drop_front(first); !rest.empty();) {
    size_t next = rest.find(marker, marker.size());
    queries.push_back(rest.take_front(next));
    rest = rest.drop_front(std::min(next, rest.size()));
  }
  std::vector<bool> proven(queries.size(), false);
  IntTranslator wrapped;
  if (!wrapped.translate(head))
    return proven;
  std::vector<std::pair<std::string, IntTranslator::Arithmetic>> candidates;
  for (auto &entry : wrapped.arithmetic)
    if (entry.second.candidate)
      candidates.push_back(entry);
  if (candidates.empty())
    return proven;
  // What each defined term names, to find what a goal depends on.
  std::map<std::string, std::vector<std::string>> uses;
  std::function<void(const SExpr &, std::vector<std::string> &)> atoms =
      [&](const SExpr &e, std::vector<std::string> &into) {
        if (e.list.empty())
          into.push_back(e.atom);
        for (const SExpr &child : e.list)
          atoms(child, into);
      };
  if (std::optional<std::vector<SExpr>> commands = parseSExprs(head))
    for (const SExpr &cmd : *commands)
      if (cmd.list.size() == 5 && cmd.list[0].atom == "define-fun")
        atoms(cmd.list[4], uses[cmd.list[1].atom]);
  std::set<std::string> candidateNames;
  for (auto &[term, a] : candidates)
    candidateNames.insert(term);
  // A query's candidates: those its goal depends on, and the products of
  // two terms that are not candidates themselves (the `k * n` of an extent
  // clause, which is among the assumptions); at most 16.
  auto relevant = [&](StringRef block) {
    std::set<std::string> cone;
    std::vector<std::string> work;
    size_t goal = block.rfind("(assert (not ");
    if (goal != StringRef::npos)
      if (std::optional<std::vector<SExpr>> g =
              parseSExprs(block.drop_front(goal).take_until(
                  [](char c) { return c == '\n'; })))
        for (const SExpr &e : *g)
          atoms(e, work);
    while (!work.empty()) {
      std::string n = work.back();
      work.pop_back();
      if (!cone.insert(n).second)
        continue;
      if (auto it = uses.find(n); it != uses.end())
        work.insert(work.end(), it->second.begin(), it->second.end());
    }
    std::vector<bool> take;
    size_t taken = 0;
    for (auto &[term, a] : candidates) {
      bool leaf = llvm::none_of(uses[term], [&](const std::string &n) {
        return candidateNames.count(n) != 0;
      });
      take.push_back(taken < 16 && (cone.count(term) || leaf));
      taken += take.back();
    }
    return take;
  };
  // Each candidate's exact value, over the exact values of the candidates
  // it is computed from (`t!x`): a sum over a product is in range only as
  // the sum over the exact product. It is the term's value where those
  // are in range too.
  std::string chained;
  for (auto &[term, a] : candidates) {
    std::string body;
    for (size_t i = 0; i < a.exact.size();) {
      size_t j = i;
      while (j < a.exact.size() && a.exact[j] != ' ' && a.exact[j] != '(' &&
             a.exact[j] != ')')
        ++j;
      if (j == i) {
        body += a.exact[i++];
        continue;
      }
      std::string token = a.exact.substr(i, j - i);
      body += token + (candidateNames.count(token) ? "!x" : "");
      i = j;
    }
    chained += "(define-fun " + term + "!x () Int " + body + ")\n";
  }
  // Each query's block over the integers, and the same block asking for a
  // candidate's range instead of the goal (the last `(assert (not ...))`).
  std::string intHead = wrapped.text();
  std::string ranges = intHead + chained;
  std::vector<bool> translated(queries.size(), false);
  std::vector<std::vector<bool>> takes(queries.size());
  for (auto [q, block] : llvm::enumerate(queries)) {
    IntTranslator one = wrapped;
    if (!one.translate(block))
      continue;
    std::string text = one.text().substr(intHead.size());
    size_t goal = text.rfind("(assert (not ");
    size_t end = goal == std::string::npos ? goal : text.find('\n', goal);
    if (end == std::string::npos)
      continue;
    translated[q] = true;
    takes[q] = relevant(block);
    // In range or not is quick to see over the integers, which can
    // otherwise run long without using up a limit.
    for (auto [c, candidate] : llvm::enumerate(candidates))
      if (takes[q][c])
        ranges += "(set-option :timeout 500)\n" + text.substr(0, goal) +
                  "(assert (not " +
                  IntTranslator::inRange(candidate.second.width,
                                         candidate.first + "!x") +
                  "))" + text.substr(end);
  }
  std::optional<SmallVector<Answer>> inRange =
      runZ3(config, ranges, (name + ".ranges").str());
  if (!inRange)
    return proven;
  std::string again;
  SmallVector<size_t> asked;
  size_t at = 0;
  for (size_t q = 0; q < queries.size(); ++q) {
    if (!translated[q])
      continue;
    // In definition order: a candidate is exact if it is in range and
    // the candidates it is computed from are exact.
    IntTranslator exact;
    for (auto [c, candidate] : llvm::enumerate(candidates)) {
      if (!takes[q][c])
        continue;
      bool inner = llvm::all_of(uses[candidate.first],
                                [&](const std::string &n) {
                                  return !candidateNames.count(n) ||
                                         exact.exact.count(n);
                                });
      if (at < inRange->size() && (*inRange)[at] == Answer::Proven && inner)
        exact.exact.insert(candidate.first);
      ++at;
    }
    if (exact.exact.empty() || !exact.translate(head) ||
        !exact.translate(queries[q]))
      continue;
    again += (asked.empty() ? "" : "(reset)\n") +
             std::string("(set-option :timeout 5000)\n") + exact.text();
    asked.push_back(q);
  }
  if (asked.empty())
    return proven;
  if (std::optional<SmallVector<Answer>> answers =
          runZ3(config, again, (name + ".exact").str()))
    for (auto [i, q] : llvm::enumerate(asked))
      proven[q] = i < answers->size() && (*answers)[i] == Answer::Proven;
  return proven;
}

//===----------------------------------------------------------------------===//
// Encoding
//===----------------------------------------------------------------------===//

/// The position of each value of the function being encoded, in walk
/// order: places are ordered by it, not by address, so the scripts (and the
/// answers cached for them) are the same from run to run.
thread_local DenseMap<Value, unsigned> *valueOrder = nullptr;

/// A place in memory: a root (a function argument or a local variable) and
/// a path of fields, each `/` followed by the field's printed attributes.
struct Loc {
  Value root;
  std::string path;
  bool operator<(const Loc &other) const {
    if (root != other.root) {
      // Values of other functions (a callee's contract region) get the next
      // position when first compared, which the deterministic walk makes the
      // same each run.
      if (valueOrder) {
        unsigned a = valueOrder->try_emplace(root, valueOrder->size())
                         .first->second;
        unsigned b = valueOrder->try_emplace(other.root, valueOrder->size())
                         .first->second;
        return a < b;
      }
      return root.getAsOpaquePointer() < other.root.getAsOpaquePointer();
    }
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
/// One step of how a function is instantiated: `symbol` (a call's callee,
/// with its parameter values) names `fn`, whose last `implicit` parameters
/// (implicit origins) it does not bind. For a local closure, `closure` is
/// its call's callee (`f[4]`) instead, binding its own parameters; the other
/// names in its body are the enclosing function's.
struct InstanceLink {
  SymbolConstantAttr symbol;
  LIT::FnOp fn;
  size_t implicit = 0;
  BindParamsAttr closure = {};

  ArrayRef<TypedAttr> values() const {
    return closure ? closure.getParamValues() : symbol.getParamValues();
  }
  Attribute callee() const {
    return closure ? Attribute(closure) : Attribute(symbol);
  }
};

struct Obligation {
  std::string pc, cond;
  Location callLoc, clauseLoc;
  std::string callee;
  bool analyzed = true;
  /// The function's own postcondition, at a return (`callLoc`).
  bool postcondition = false;
  /// For a refinement check, the implementation of the trait method
  /// `callee` that is held to its clauses.
  std::string implementation;
  /// What the obligation says, when it is not a callee's clause ("that the
  /// divisor is not 0").
  std::string claim;
};

/// A loop being walked: where its iterations continue and exit, and what
/// its body reads.
struct LoopFrame {
  StringRef label;
  /// A `comptime for`, which only its own `continue` and `break` target.
  bool comptime = false;
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
  /// `bvadd` or `bvsub`: the candidate is instead that `lhs` combined with
  /// `rhsLoc` keeps its value on entry (`len(xs) + i` when each iteration
  /// pops once and increments `i`).
  std::string combine = "";
  /// The candidate is instead that `lhs`, a strided range's cursor, is a
  /// whole number of steps (`rhsLoc`, as on entry) from where it started.
  bool stride = false;
  /// With `stride`: the candidate is instead that the cursor is as many
  /// steps from where it started as this variable is above its value on
  /// entry (a counter the body increments beside a strided loop).
  std::optional<Loc> counter;
};

/// A function's postcondition: its `kgen.ensures` clauses, which are
/// repeated before every return; the ones before the first return.
SmallVector<EnsuresOp> postconditionClauses(LIT::FnOp fn) {
  EnsuresOp first;
  fn.getFunctionBody().walk([&](EnsuresOp op) {
    if (!first)
      first = op;
  });
  SmallVector<EnsuresOp> clauses;
  for (Operation *op = first.getOperation();
       op && !op->hasTrait<OpTrait::IsTerminator>(); op = op->getNextNode())
    if (auto ensures = dyn_cast<EnsuresOp>(op))
      clauses.push_back(ensures);
  return clauses;
}

/// Which struct methods implement which trait methods, from the structs'
/// conformance tables (structs without parameters only): by
/// `trait|method|struct`, and that key by implementation.
struct TraitImpls {
  std::map<std::string, LIT::FnOp> byKey;
  DenseMap<Operation *, std::string> keyOf;
};

std::string implKey(TraitSymbolAttr trait, StringRef method,
                    StringRef structSymbol) {
  return printed(trait) + "|" + method.str() + "|" + structSymbol.str();
}

/// The symbol of the struct `fn` is a method of, if it has no parameters.
std::string plainStructOf(LIT::FnOp fn) {
  auto parent = fn->getParentOfType<LIT::StructDeclOp>();
  if (!parent || !parent.getParams().empty())
    return "";
  return printed(LIT::getFullyResolvedSymbolRef(parent));
}

class FunctionEncoder {
public:
  /// With `refines`, checks instead that `fn` implements the trait method
  /// `refines`: that the trait's precondition implies `fn`'s, and that `fn`
  /// establishes the trait's postcondition where it returns.
  FunctionEncoder(LIT::FnOp fn, ModuleOp module,
                  SymbolTableCollection &symbols, const SolverConfig &solver,
                  std::string dumpPrefix, const TraitImpls &impls,
                  LIT::FnOp refines = {})
      : fn(fn), module(module), symbols(symbols), solver(solver),
        dumpPrefix(std::move(dumpPrefix)), impls(impls), refines(refines),
        selfStruct(refines ? plainStructOf(fn) : "") {}

  /// Encodes the function; returns false if it has no body.
  bool encode() {
    Region &body = fn.getFunctionBody();
    if (body.empty())
      return false;
    fn->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (BlockArgument arg : block.getArguments())
            order.try_emplace(arg, order.size());
      for (Value result : op->getResults())
        order.try_emplace(result, order.size());
    });
    valueOrder = &order;
    llvm::scope_exit reset([] { valueOrder = nullptr; });
    fn.walk([&](LIT::FnOp nested) {
      if (nested != fn)
        if (auto decl = nested.getParamDeclAttr())
          closures[ParamDeclRefAttr::get(decl)] = nested;
    });
    Block &entry = body.front();
    // Arguments: integers and Booleans are unknowns of their sort; references
    // are places, whose values at entry are unknowns.
    for (BlockArgument arg : entry.getArguments()) {
      if (isa<LIT::RefType>(arg.getType()))
        roots.push_back(arg);
      else
        values[arg] = declare(sortOf(arg.getType()), "a");
    }
    fn.walk([&](LIT::VarDeclOp decl) {
      Value root = decl->getResult(0);
      roots.push_back(root);
      // Only ever loaded from, stored to and borrowed immutably (and its
      // lifetime marked), or passed to a range's own methods, which keep no
      // reference: nothing else writes it. That an immutable borrow is not
      // turned into a pointer that is written through is assumed.
      // A `kgen.rebind` of the reference is the same reference under
      // another type: its uses count too.
      std::function<bool(Value)> plain = [&](Value ref) {
        return llvm::all_of(ref.getUses(), [&](OpOperand &use) {
          Operation *user = use.getOwner();
          StringRef op = user->getName().getStringRef();
          if (auto call = dyn_cast<LIT::CallOp>(user)) {
            std::optional<CalleeName> name = calleeName(call.getCallee());
            return name &&
                   StringRef(name->path).starts_with("std::builtin::range::");
          }
          if (isa<RebindOp>(user))
            return plain(user->getResult(0));
          return isa<LIT::RefLoadOp>(user) || op == "lit.load.consume" ||
                 op == "lit.ref.immut" || op.starts_with("lit.var.lifetime.") ||
                 op.starts_with("lit.ownership.") ||
                 (isa<LIT::RefStoreOp>(user) && use.getOperandNumber() == 1);
        });
      };
      if (plain(root))
        unborrowed.insert(root);
    });
    // The function's own constraints on its parameters (`where N <= 8`): the
    // compiler rejects any instantiation that breaks them, so they hold in
    // the body. One the parameter model cannot read is an unknown.
    // They name the parameters by position (`#kgen.param.index.ref<0, i>`),
    // the body by name: the position is read as the declaration.
    if (auto pogs = fn.getFuncTypeGenerator().getParamListAttrs()) {
      ArrayRef<ParamDeclAttr> own = fn.getParams();
      positionalParams = &own;
      for (ConstraintAttr constraint : pogs.getBodyConstraints())
        facts.push_back(
            paramTerm(constraint.getProposition(), Sort{true, 1, false}));
      positionalParams = nullptr;
    }
    State state;
    // A caller through the trait establishes only the trait's precondition,
    // which must imply the implementation's own.
    if (refines) {
      for (Operation &op : refines.getFunctionBody().front())
        if (auto req = dyn_cast<RequiresOp>(&op))
          if (MaybeTerm cond =
                  instantiate(req.getBody(), req.getArgs(), state,
                              entry.getArguments(), refines, nullptr,
                              /*assumed=*/true))
            facts.push_back(*cond);
      for (Operation &op : entry)
        if (auto req = dyn_cast<RequiresOp>(&op)) {
          Obligation ob{state.pc, "false", fn.getLoc(), req.getLoc(),
                        displayName(refines)};
          ob.implementation = displayName(fn);
          if (MaybeTerm cond = instantiate(req.getBody(), req.getArgs(), state,
                                           {}, {}, nullptr, false))
            ob.cond = *cond;
          else
            ob.analyzed = false;
          obligations.push_back(ob);
        }
    }
    // The function's own preconditions hold at its entry (when refining,
    // after they are proven there: on the path, not as global facts).
    for (Operation &op : entry)
      if (auto req = dyn_cast<RequiresOp>(&op))
        if (MaybeTerm cond = instantiate(req.getBody(), req.getArgs(), state,
                                         {}, {}, nullptr, /*assumed=*/true)) {
          if (refines)
            state.pc = define({true, 1, false},
                              "(and " + state.pc + " " + *cond + ")", "r");
          else
            facts.push_back(*cond);
        }
    entryState = state;
    walkBlock(entry, state);
    // The body's own obligations are checked by the function's own run.
    if (refines)
      llvm::erase_if(obligations, [](const Obligation &ob) {
        return ob.implementation.empty();
      });
    return true;
  }

  /// The declarations and facts the queries share.
  std::string scriptHeader() const { return header(); }
  void solveCases(const std::string &text, SmallVectorImpl<Answer> &answers,
                  StringRef name, ArrayRef<bool> reask, bool exact = false) {
    runCases(text, answers, name, reask, exact);
  }
  bool obligationDependsOnCases(const Obligation &ob) {
    return dependsOnCases(ob.cond) || dependsOnCases(ob.pc);
  }

  std::string script() {
    std::string text = header();
    for (const Obligation &ob : obligations)
      if (ob.analyzed)
        text += query({ob.pc}, ob.cond, limitOf(ob));
    return text;
  }

  /// The query limit of every obligation.
  unsigned fullLimit() const {
    return queryRlimit ? queryRlimit : solver.rlimit;
  }

  /// The limit `ob` is first asked with: lower for a nonlinear goal
  /// (`nonlinear-rlimit`), which is then asked over the integers and only
  /// if that does not decide it, again with the full limit.
  unsigned limitOf(const Obligation &ob) {
    unsigned limit = fullLimit();
    if (solver.nonlinearRlimit && solver.nonlinearRlimit < limit &&
        nonlinear(ob.cond))
      limit = solver.nonlinearRlimit;
    return limit;
  }

  /// Whether `ob`'s goal multiplies or divides two unknowns.
  bool isNonlinear(const Obligation &ob) { return nonlinear(ob.cond); }

  /// Encodes the instantiation that `chain` names: a launched kernel with
  /// its parameters, then the function launching it with the parameters a
  /// call of it gives, and so on outwards. Each step's values are in the
  /// scope of the next. A value that is a parameter the next step does not
  /// bind (the outermost function's own) stays unbound: the parameter it
  /// gives is then an unknown, so what is proven holds for every value of
  /// it, and a parameter of one function cannot be read as a parameter of
  /// the same name of another.
  void bindInstance(ArrayRef<InstanceLink> chain) {
    // A launched kernel runs on the GPU: its own target is one.
    if (MaybeTerm gpu = targetPredicate("std::sys::info::is_gpu()",
                                        Sort{true, 1, false}))
      facts.push_back(*gpu);
    instanceFrames.clear();
    for (const InstanceLink &link : chain) {
      if (!link.closure) {
        instanceFrames.push_back(
            paramFrame(link.symbol, link.fn, link.implicit, ParamFrame{}));
        continue;
      }
      ParamFrame frame;
      frame.transparent = true;
      frame.scope = printed(link.closure);
      LIT::FnOp closure = link.fn;
      ArrayRef<ParamDeclAttr> decls = closure.getParams();
      decls = decls.drop_back(std::min(decls.size(), link.implicit));
      ArrayRef<TypedAttr> bound = link.closure.getParamValues();
      if (decls.size() == bound.size())
        for (auto [decl, value] : llvm::zip(decls, bound))
          frame.values[decl.getName()] = value;
      instanceFrames.push_back(std::move(frame));
    }
    for (size_t k = 0; k + 1 < instanceFrames.size(); ++k)
      instanceFrames[k].parent = &instanceFrames[k + 1];
    // Whether the frames from `frame` on give `name` a value: a closure's
    // frame passes the names it does not bind on to its caller's.
    auto bound = [](ParamFrame *frame, StringRef name) {
      for (; frame; frame = frame->transparent ? frame->parent : nullptr)
        if (frame->values.count(name))
          return true;
      return false;
    };
    for (size_t k = instanceFrames.size(); k-- > 0;) {
      ParamFrame *next = instanceFrames[k].parent;
      SmallVector<std::string> open;
      for (auto &entry : instanceFrames[k].values) {
        TypedAttr value = entry.second;
        while (auto upcast = dyn_cast<UpcastAttr>(value))
          value = upcast.getInputTypeValue();
        if (auto ref = dyn_cast<ParamDeclRefAttr>(value);
            ref && !bound(next, ref.getName()))
          open.push_back(entry.first().str());
      }
      for (const std::string &name : open)
        instanceFrames[k].values.erase(name);
    }
    rootParams = instanceFrames.empty() ? nullptr : &instanceFrames.front();
    params = rootParams;
  }

  /// After `bindInstance`: the name of an integer parameter of the kernel
  /// whose value in this instantiation is computed by a function outside
  /// the standard library that elaboration evaluates and the verifier does
  /// not (`config[1]` of `comptime config = _gemv_config[...]()`), if any.
  /// Such an instantiation is not verified: loops are unrolled and tiles
  /// sized by these values, so with them unknown next to nothing is proven,
  /// at the cost of the longest solver runs. It is what the launch compiles
  /// to only once the launcher's own parameters are given.
  std::optional<std::string> opaqueParameter() {
    return opaqueParameterOf(fn);
  }
  /// The same of `kernel`, with its parameters as `params` binds them (at
  /// its launch).
  std::optional<std::string> opaqueParameterOf(LIT::FnOp kernel) {
    for (ParamDeclAttr decl : kernel.getParams()) {
      auto ref = ParamDeclRefAttr::get(decl);
      std::string type = printed(ref.getType());
      if (!StringRef(type).contains("@std::@simd::@SIMD<") ||
          !StringRef(type).contains("{:dtype index}"))
        continue;
      TypedAttr value = resolveParam(ref);
      while (auto sugar = dyn_cast<SugarAttr>(value))
        value = sugar.getCanonical();
      bool opaque = false;
      value.walk([&](SymbolConstantAttr symbol) {
        if (opaque ||
            StringRef(printed(symbol.getSymbol())).starts_with("@std::"))
          return;
        opaque = isa_and_nonnull<LIT::FnOp>(
            symbols.lookupSymbolIn(module, symbol.getSymbol()));
      });
      if (opaque)
        return decl.getName().str();
    }
    return std::nullopt;
  }

  /// The resource limit of each query, if not the solver's.
  unsigned queryRlimit = 0;
  /// Whether integer divisors that may be 0 are obligations.
  bool checkDivision = false;
  /// Whether the compilation target's unknowns are declared.
  bool targetDeclared = false;
  /// Whether target predicates are read for a kernel at its launch.
  bool deviceView = false;

  LIT::FnOp fn;
  SmallVector<Obligation> obligations;
  unsigned invariantsFound = 0, loopsAnalyzed = 0;

private:
  ModuleOp module;
  SymbolTableCollection &symbols;
  const SolverConfig &solver;
  DenseMap<Value, unsigned> order;
  std::string dumpPrefix;
  const TraitImpls &impls;
  LIT::FnOp refines;
  /// When refining a method of a struct without parameters: the struct,
  /// which the trait's `Self` is.
  std::string selfStruct;
  unsigned houdiniRuns = 0;

  DenseMap<Value, std::string> values;
  /// Fields of struct values the encoder builds (`range`s), by path.
  DenseMap<Value, std::map<std::string, std::string>> records;
  /// Places that reference values loaded from `ref` locals refer to.
  DenseMap<Value, Loc> derivedPlaces;
  std::map<Loc, std::string> entryValues;
  std::map<std::pair<Loc, unsigned>, std::string> epochValues;
  std::map<std::pair<std::string, std::string>, std::string> fieldValues;
  /// The fields of values modelled constructors built (`storeFields`).
  std::map<std::pair<std::string, std::string>, std::string> builtFields;
  /// The struct value and field path each field value belongs to.
  std::map<std::string, std::pair<std::string, std::string>> fieldOwners;
  DenseMap<Value, Loc> refArgs; // Contract block arguments bound to places.
  /// The function's reference arguments and local variables.
  SmallVector<Value> roots;
  /// The local variables that are never borrowed (see `encode`).
  DenseSet<Value> unborrowed;
  /// What the last `writtenRoots` found written by name, also where the
  /// code may write anything a reference leads to.
  llvm::DenseSet<Value> storedRoots;
  /// The places of those roots it found written (the empty path: the whole
  /// root). A root declared in the code has none: it is written whole.
  DenseMap<Value, std::set<std::string>> storedPaths;
  std::string prelude;
  SmallVector<std::string> facts;
  std::map<std::string, Sort> sorts;
  std::map<std::string, SmallVector<std::string>> deps;
  std::map<std::string, std::string> parameterValues;
  std::map<std::string, std::string> definitions;
  /// Defined expressions (with their sort), to their names.
  std::map<std::string, std::string> defined;
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
  /// While evaluating a callee's contract: the call's values of the callee's
  /// parameters (its struct's, then its own), which are expressions in the
  /// caller's parameters (`parent`). `scope` keeps the callee's unknown
  /// parameter expressions apart from the caller's of the same name.
  /// A local closure's frame (`transparent`) holds its own parameters only:
  /// every other name in its body is its caller's, looked up in `parent`.
  struct ParamFrame {
    llvm::StringMap<TypedAttr> values;
    ParamFrame *parent = nullptr;
    std::string scope;
    bool transparent = false;
  };
  ParamFrame *params = nullptr;
  /// While the function's own constraints are read: its parameters, which
  /// they refer to by position.
  ArrayRef<ParamDeclAttr> *positionalParams = nullptr;
  /// The function's own parameters' frame: none, or an instantiation's.
  std::vector<ParamFrame> instanceFrames;
  ParamFrame *rootParams = nullptr;
  /// While a kernel's clause is instantiated at its launch: the launch's
  /// dimensions (`grid_dim_x`, `block_dim_y`, ...) that its GPU ids read.
  const std::map<std::string, std::string> *launchDims = nullptr;
  /// The `end` of each `comptime for` over `range(end)`, by the printed
  /// reference to its iterator parameter.
  std::map<std::string, std::string> comptimeRangeEnds;
  /// While a `comptime for` is walked unrolled: the current iteration's
  /// value, by the loop's iterator (`"iter`30"`).
  std::map<std::string, std::string> comptimeIterations;
  /// How many times the innermost unrolled body is walked.
  uint64_t unrolledBodies = 1;
  /// The references a variadic pack holds, by the pack's term.
  std::map<std::string, SmallVector<Value>> packRefs;
  /// Element references `__getitem__` returned: the list's place and the
  /// index.
  DenseMap<Value, std::pair<Loc, std::string>> elements;
  std::set<std::string> elementFunctions;
  bool tensorDimDeclared = false;
  bool pointerExtentDeclared = false;
  /// The GPU id constants declared (see `evalGpuId`).
  std::set<std::string> gpuIds;
  /// The results of `TileTensor.dim` (see `evalTileTensor`).
  llvm::DenseSet<Value> tensorDims;
  /// The valid extents (`_valid_dim`) of the partial tiles and views of
  /// them, by the tensor's term; any other tensor's are its `dim`s.
  std::map<std::string, SmallVector<std::string>> partialValid;
  /// The shapes of layout values `row_major(Coord(...))` built, by term.
  std::map<std::string, SmallVector<std::string>> layoutShapes;
  /// Local closures (`@parameter def` in the function), by the reference a
  /// call names them with: walked inline at their calls.
  DenseMap<Attribute, LIT::FnOp> closures;
  /// While a closure's body is walked at a call: where its returns and
  /// raises leave it.
  SmallVector<SmallVector<State> *> closureExits;
  /// The collection an iterator (by its place's root) iterates.
  DenseMap<Value, Loc> iterSources;
  /// The types of places named by a value of another type (an element an
  /// iterator yields, named by its `__next__` call).
  DenseMap<Value, Type> elementTypes;
  /// The functions standing for read-only trait method calls, by trait,
  /// method, type and signature (see `evalTraitQuery`).
  std::map<std::string, std::string> traitQueries;
  unsigned counter = 0;
  bool lenDeclared = false;
  /// Inside a contract region: calls there are evaluated, not checked.
  unsigned inContract = 0;
  SmallVector<LoopFrame *> loops;
  SmallVector<TryFrame *> tries;

  /// Case splitting: `answers` of the queries of `text`, upgraded to
  /// proven where every case proves them. A case is a combination of
  /// values of the finite-domain unknowns `text` declares, defined as those
  /// constants, so products and quotients by them fold (a bit-vector
  /// product of two unknowns is what z3 cannot decide within its limits).
  /// Only where an answer is unknown, and for at most 64 cases.
  /// With `exact`, each case is asked with `exactIntegers` instead (and
  /// the script as it is, where it has no finite-domain unknown).
  void runCases(const std::string &text, SmallVectorImpl<Answer> &answers,
                StringRef name, ArrayRef<bool> reask = {},
                bool exact = false) {
    // A refuted query may hold with the case facts (an exact quotient where
    // the script only bounds it): asked again where `reask` says it depends
    // on what the cases change.
    if (!solver.caseSplit)
      return;
    auto open = [&](size_t i) {
      return answers[i] == Answer::Unknown ||
             (answers[i] == Answer::Unproven && !caseFacts.empty() &&
              i < reask.size() && reask[i]);
    };
    bool any = false;
    for (size_t i = 0; i < answers.size(); ++i)
      any = any || open(i);
    if (!any)
      return;
    // The finite-domain unknowns the open queries depend on (through
    // definitions); others stay unknowns, which is sound.
    std::set<std::string> used;
    {
      size_t at = text.find("(echo \"@@\")");
      SmallVector<std::string> work;
      if (at != std::string::npos)
        for (const std::string &n : namesIn(StringRef(text).drop_front(at)))
          work.push_back(n);
      while (!work.empty()) {
        std::string n = work.pop_back_val();
        if (!used.insert(n).second)
          continue;
        if (auto def = definitions.find(n); def != definitions.end())
          for (const std::string &m : namesIn(def->second))
            work.push_back(m);
      }
    }
    // Every one the script declares (facts, such as loop invariants, may
    // relate them to the queries); only those the queries depend on if
    // that is more than 64 cases.
    SmallVector<std::pair<std::string, ArrayRef<int64_t>>> domains;
    size_t cases = 1;
    for (bool onlyUsed : {false, true}) {
      domains.clear();
      cases = 1;
      for (auto &[unknown, values] : finiteDomains) {
        std::string decl = "(declare-const " + unknown + " (_ BitVec 64))\n";
        if ((onlyUsed && !used.count(unknown)) ||
            text.find(decl) == std::string::npos)
          continue;
        domains.push_back({unknown, values});
        cases *= values.size();
      }
      if (cases <= 64)
        break;
    }
    if (exact && domains.empty()) {
      std::vector<bool> proven = exactIntegers(solver, text, name);
      for (auto [i, a] : llvm::enumerate(answers))
        if (open(i) && i < proven.size() && proven[i])
          a = Answer::Proven;
      return;
    }
    if (domains.empty() || cases > 64)
      return;
    std::string extra;
    for (const std::string &fact : caseFacts)
      extra += "(assert " + fact + ")\n";
    // The header, and each query (from its `(echo "@@")`).
    StringRef marker = "(echo \"@@\")";
    size_t firstQuery = text.find(marker.str());
    if (firstQuery == std::string::npos)
      return;
    std::string head = text.substr(0, firstQuery) + extra;
    SmallVector<StringRef> queries;
    for (StringRef rest = StringRef(text).drop_front(firstQuery);
         !rest.empty();) {
      size_t next = rest.find(marker, marker.size());
      queries.push_back(rest.take_front(next));
      rest = rest.drop_front(std::min(next, rest.size()));
    }
    if (queries.size() != answers.size())
      return;
    // In a case the finite-domain unknowns are constants, so what holds is
    // quick to prove: each query gets at most 5M (a refuted query re-asked
    // for the case facts would otherwise cost its full limit in every case).
    static llvm::Regex limitRe("\\(set-option :rlimit ([0-9]+)\\)");
    SmallVector<std::string> capped;
    for (StringRef q : queries) {
      std::string text = q.str();
      SmallVector<StringRef> m;
      uint64_t limit;
      if (limitRe.match(text, &m) && !m[1].getAsInteger(10, limit) &&
          limit > 5000000) {
        size_t at = text.find(m[0].str());
        text.replace(at, m[0].size(), "(set-option :rlimit 5000000)");
      }
      capped.push_back(std::move(text));
    }
    // Still to decide: open and not refuted by a case so far (a query a case
    // refutes is not asked again, so a real failure costs one or two cases).
    SmallVector<bool> pending;
    for (size_t i = 0; i < answers.size(); ++i)
      pending.push_back(open(i));
    // The script of case `c` for the queries `asked`.
    auto caseScript = [&](size_t c, ArrayRef<size_t> asked) {
      std::string t = head;
      for (size_t i : asked)
        t += capped[i];
      size_t rest = c;
      for (auto &[unknown, values] : domains) {
        int64_t v = values[rest % values.size()];
        rest /= values.size();
        std::string decl = "(declare-const " + unknown + " (_ BitVec 64))\n";
        t.replace(t.find(decl), decl.size(),
                  "(define-fun " + unknown + " () (_ BitVec 64) " +
                      bvConst(v, 64) + ")\n");
      }
      return t;
    };
    // The first case alone (a refuted query is dropped from the rest), then
    // the others in groups of 8 per solver process, separated by `(reset)`:
    // starting a process per case cost more than the cases' own work.
    for (size_t first = 0; first < cases;) {
      size_t last = first == 0 ? 1 : std::min(cases, first + 8);
      SmallVector<size_t> asked;
      for (size_t i = 0; i < pending.size(); ++i)
        if (pending[i])
          asked.push_back(i);
      if (asked.empty())
        break;
      if (exact) {
        for (size_t c = first; c < last && !asked.empty(); ++c) {
          std::vector<bool> proven = exactIntegers(
              solver, caseScript(c, asked), (name + ".case" + Twine(c)).str());
          SmallVector<size_t> still;
          for (auto [j, i] : llvm::enumerate(asked)) {
            if (j < proven.size() && proven[j])
              still.push_back(i);
            else
              pending[i] = false;
          }
          asked = std::move(still);
        }
        first = last;
        continue;
      }
      std::string t;
      for (size_t c = first; c < last; ++c)
        t += (c == first ? "" : "(reset)\n") + caseScript(c, asked);
      std::optional<SmallVector<Answer>> r =
          runZ3(solver, t, (name + ".case" + Twine(first)).str());
      if (!r || r->size() != asked.size() * (last - first))
        return;
      // A query undecided in a case is not proven either.
      for (size_t c = 0; c < last - first; ++c)
        for (auto [j, i] : llvm::enumerate(asked))
          if ((*r)[c * asked.size() + j] != Answer::Proven)
            pending[i] = false;
      first = last;
    }
    for (auto [i, a] : llvm::enumerate(answers))
      if (pending[i])
        a = Answer::Proven;
  }

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
    // The same expression is the same term: equal computations (a callee's
    // contract and its body evaluating the same helper calls) then share
    // names, which the solver sees without reasoning about them.
    std::string key = sort.str() + " " + expr.str();
    if (auto it = defined.find(key); it != defined.end())
      return it->second;
    std::string name = (prefix + Twine(counter++)).str();
    defined[key] = name;
    prelude += ("(define-fun " + name + " () " + sort.str() + " " + expr +
                ")\n")
                   .str();
    sorts[name] = sort;
    deps[name] = namesIn(expr);
    definitions[name] = expr.str();
    return name;
  }

  /// The elements of the application `(head a b ...)`, head first; empty
  /// for anything else.
  static SmallVector<StringRef> application(StringRef term) {
    SmallVector<StringRef> parts;
    term = term.trim();
    if (!term.consume_front("(") || !term.consume_back(")"))
      return parts;
    size_t start = 0;
    int depth = 0;
    for (size_t i = 0; i <= term.size(); ++i) {
      char c = i < term.size() ? term[i] : ' ';
      if (c == '(')
        ++depth;
      else if (c == ')')
        --depth;
      else if (c == ' ' && depth == 0) {
        if (i > start)
          parts.push_back(term.slice(start, i));
        start = i + 1;
      }
    }
    return parts;
  }

  /// The value of a 64-bit term that is a constant by its definition.
  std::optional<int64_t> constantInt(StringRef term, unsigned depth = 0) {
    term = term.trim();
    static llvm::Regex constRe("^\\(_ bv([0-9]+) 64\\)$");
    SmallVector<StringRef> m;
    uint64_t v;
    if (constRe.match(term, &m) && !m[1].getAsInteger(10, v))
      return (int64_t)v;
    if (auto it = definitions.find(term.str());
        it != definitions.end() && depth < 32)
      return constantInt(it->second, depth + 1);
    return std::nullopt;
  }

  /// Whether a Boolean term is decided by its definition alone: constants
  /// compared with constants, under `and`, `or`, `not` and `xor`. A
  /// launched instantiation's `comptime if` over its parameters is.
  std::optional<bool> decided(StringRef term, unsigned depth = 0) {
    term = term.trim();
    if (term == "true")
      return true;
    if (term == "false")
      return false;
    if (depth > 32)
      return std::nullopt;
    if (auto it = definitions.find(term.str()); it != definitions.end())
      return decided(it->second, depth + 1);
    SmallVector<StringRef> parts = application(term);
    if (parts.empty())
      return std::nullopt;
    StringRef head = parts[0];
    ArrayRef<StringRef> args = ArrayRef(parts).drop_front();
    if (head == "and" || head == "or") {
      bool all = true;
      for (StringRef arg : args) {
        std::optional<bool> v = decided(arg, depth + 1);
        if (v && *v == (head == "or"))
          return head == "or";
        all = all && v.has_value();
      }
      if (all)
        return head == "and";
      return std::nullopt;
    }
    if (head == "not" && args.size() == 1) {
      if (std::optional<bool> v = decided(args[0], depth + 1))
        return !*v;
      return std::nullopt;
    }
    if (head == "xor" && args.size() == 2) {
      std::optional<bool> a = decided(args[0], depth + 1),
                          b = decided(args[1], depth + 1);
      if (a && b)
        return *a != *b;
      return std::nullopt;
    }
    if (args.size() == 2 &&
        (head == "=" || head == "bvslt" || head == "bvsle" || head == "bvsgt" ||
         head == "bvsge")) {
      std::optional<int64_t> a = constantInt(args[0]), b = constantInt(args[1]);
      if (!a || !b)
        return std::nullopt;
      return head == "="       ? *a == *b
             : head == "bvslt" ? *a < *b
             : head == "bvsle" ? *a <= *b
             : head == "bvsgt" ? *a > *b
                               : *a >= *b;
    }
    return std::nullopt;
  }

  /// Whether `term` multiplies, divides or takes the remainder of two
  /// non-constant values anywhere in its definition.
  bool nonlinear(StringRef term) {
    if (auto it = nonlinearTerms.find(term); it != nonlinearTerms.end())
      return it->second;
    nonlinearTerms[term] = false; // Definitions are acyclic; guard anyway.
    bool result = false;
    auto def = definitions.find(term.str());
    if (def != definitions.end())
      if (std::optional<std::vector<SExpr>> exprs = parseSExprs(def->second))
        for (const SExpr &e : *exprs)
          result = result || nonlinear(e);
    nonlinearTerms[term] = result;
    return result;
  }
  bool nonlinear(const SExpr &e) {
    if (e.list.empty())
      return nonlinear(StringRef(e.atom));
    static const char *ops[] = {"bvmul",  "bvudiv", "bvsdiv",
                                "bvurem", "bvsrem", "bvsmod"};
    if (llvm::is_contained(ops, StringRef(e.list[0].atom))) {
      unsigned unknowns = 0;
      for (const SExpr &arg : ArrayRef(e.list).drop_front()) {
        bool constant = (arg.list.size() == 3 && arg.list[0].atom == "_") ||
                        StringRef(arg.atom).starts_with("#");
        unknowns += !constant;
      }
      if (unknowns >= 2)
        return true;
    }
    return llvm::any_of(ArrayRef(e.list).drop_front(),
                        [&](const SExpr &arg) { return nonlinear(arg); });
  }
  llvm::StringMap<bool> nonlinearTerms;

  /// Whether `term` depends on the compilation target (`target_*`).
  bool mentionsTarget(StringRef term) {
    if (term.contains("target_"))
      return true;
    if (auto it = targetTerms.find(term); it != targetTerms.end())
      return it->second;
    targetTerms[term] = false;
    bool result = false;
    if (auto def = definitions.find(term.str()); def != definitions.end()) {
      result = StringRef(def->second).contains("target_");
      for (const std::string &name : namesIn(def->second))
        if (!result && name != term)
          result = mentionsTarget(name);
    } else {
      for (const std::string &name : namesIn(term))
        if (!result && name != term)
          result = mentionsTarget(name);
    }
    targetTerms[term] = result;
    return result;
  }
  llvm::StringMap<bool> targetTerms;
  /// The `rank` and `flat_rank` of each layout type parameter seen, by name
  /// and `layoutKey`.
  std::map<std::pair<std::string, std::string>, std::string> layoutWitnesses;
  /// The terms of each generic layout's `static_shape[k]` and of its
  /// tensors' `dim[k]()`, by `layoutKey` and `k`.
  std::map<std::pair<std::string, int64_t>, SmallVector<std::string>>
      staticShapeTerms, dimTerms;
  /// A struct value made by writing one field of another: the other value
  /// and the field's path in it.
  std::map<std::string, std::pair<std::string, std::string>> updates;
  /// `Int(x)` of each value of a type that is not known to be `Int`.
  std::map<std::string, std::string> intConversions;
  /// The value of each field read of a tensor (`t.ptr`), by the tensor's
  /// term and the getter.
  std::map<std::pair<std::string, std::string>, std::string> tensorFields;
  /// The term of each generic layout's `static_stride[k]`, by `layoutKey`
  /// and `k`.
  std::map<std::pair<std::string, int64_t>, std::string> staticStrideTerms;
  /// Facts that are added once however often their terms are met.
  std::set<std::string> addedFacts;
  /// Unknowns that range over a few values (a SIMD width, the warp size),
  /// by name: what `runCases` substitutes.
  std::map<std::string, SmallVector<int64_t>> finiteDomains;
  /// The `simd_width_of` widths, by dtype and target (`simdWidth`).
  std::map<std::string, std::string> simdWidths;
  /// Facts asserted only in the cases of `runCases`, where they are cheap:
  /// exact unsigned quotients, whose divisor a case makes a constant.
  SmallVector<std::string> caseFacts;
  /// The quotients `caseFacts` make exact.
  std::set<std::string> caseQuotients;
  llvm::StringMap<bool> caseTerms;

  /// Whether a term depends (through definitions) on a finite-domain
  /// unknown or a quotient made exact in the cases: whether case splitting
  /// can change its answer.
  bool dependsOnCases(StringRef term) {
    if (finiteDomains.count(term.str()) || caseQuotients.count(term.str()))
      return true;
    if (auto it = caseTerms.find(term); it != caseTerms.end())
      return it->second;
    caseTerms[term] = false;
    bool result = false;
    auto def = definitions.find(term.str());
    for (const std::string &name :
         namesIn(def != definitions.end() ? StringRef(def->second) : term))
      if (!result && name != term)
        result = dependsOnCases(name);
    caseTerms[term] = result;
    return result;
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
    // `Sized.__len__` states it; the refinement check holds implementations
    // outside the stdlib to it (the stdlib's with `include-stdlib=true`).
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
    // A reference to an element (`xs[i]`, used as a place itself, as in
    // `xs[i][j]` or `xs[i].append(v)`): a place of its own, rooted at the
    // reference, whose value is the element (see `load` and `store`).
    if (elements.count(ref))
      return Loc{ref, ""};
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
    // The same place, seen as immutable, rebound, or with a wider origin.
    if (isa<LIT::RefImmutOp, RebindOp, LIT::RefUpcastOp>(def))
      return placeOf(def->getOperand(0));
    if (auto gep = dyn_cast<LIT::RefStructGEROp>(def)) {
      std::optional<Loc> base = placeOf(gep->getOperand(0));
      if (!base)
        return std::nullopt;
      // `/` separates fields: by name when the field has one (so the fields
      // built-ins write, like a slice's `start`, are the same places),
      // else by the printed attributes (which contain no `/`).
      if (auto field = gep->getAttrOfType<StringAttr>("field"))
        base->path += "/" + field.getValue().str();
      else
        base->path += "/" + printed(gep->getAttrDictionary());
      return base;
    }
    return std::nullopt;
  }

  Type placeType(const Loc &loc) {
    if (auto it = elementTypes.find(loc.root); it != elementTypes.end())
      return it->second;
    if (auto ref = dyn_cast<LIT::RefType>(loc.root.getType()))
      return ref.getElementType();
    return loc.root.getType();
  }

  std::string load(const Loc &loc, State &state, Sort sort) {
    for (LoopFrame *frame : loops)
      frame->loaded.insert(loc);
    if (auto it = state.env.find(loc); it != state.env.end())
      return it->second;
    // An element's place: the element, as the collection holds it now.
    if (loc.path.empty())
      if (auto it = elements.find(loc.root); it != elements.end()) {
        auto [list, index] = it->second;
        return elem(load(list, state, placeType(list)), index, sort);
      }
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
    if (auto it = fieldValues.find({value, rest}); it != fieldValues.end())
      return it->second;
    // A field of a field: the field at the joined path, so that `x.a`
    // loaded whole and then read at `.b` is `x.a.b`.
    if (auto it = fieldOwners.find(value); it != fieldOwners.end()) {
      std::pair<std::string, std::string> owner = it->second;
      return fieldOf(owner.first, owner.second + rest, sort);
    }
    // A value chosen by a condition (where two paths join, one of which
    // wrote a field): the field chosen by the same condition, which is the
    // one field where both paths hold the same.
    if (auto def = definitions.find(value); def != definitions.end()) {
      StringRef body(def->second);
      if (body.consume_front("(ite ") && body.consume_back(")")) {
        size_t b = body.rfind(' ');
        size_t a = b == StringRef::npos ? b : body.rfind(' ', b - 1);
        if (a != StringRef::npos) {
          StringRef cond = body.take_front(a), first = body.slice(a + 1, b),
                    second = body.drop_front(b + 1);
          auto plain = [](StringRef t) {
            return !t.empty() && !t.contains('(') && !t.contains(')');
          };
          if (plain(first) && plain(second)) {
            std::string x = fieldOf(first.str(), rest, sort);
            std::string y = fieldOf(second.str(), rest, sort);
            std::string joined =
                x == y ? x
                       : define(sort, ("(ite " + cond + " " + x + " " + y + ")")
                                          .str());
            fieldValues[{value, rest}] = joined;
            return joined;
          }
        }
      }
    }
    // A value that is another with one field written (`store`): its other
    // fields are that value's. Not a field the written one is part of.
    if (auto it = updates.find(value); it != updates.end()) {
      const std::string &written = it->second.second;
      if (rest != written && !StringRef(rest).starts_with(written + "/") &&
          !StringRef(written).starts_with(rest + "/"))
        return fieldOf(it->second.first, rest, sort);
    }
    // Through a field a constructor built it with (`/1` of a tuple literal,
    // for `/1/has`).
    for (size_t cut = rest.rfind('/'); cut != 0 && cut != std::string::npos;
         cut = rest.rfind('/', cut - 1))
      if (auto it = builtFields.find({value, rest.substr(0, cut)});
          it != builtFields.end())
        return fieldOf(it->second, rest.substr(cut), sort);
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
    // Writing a place replaces the values of its fields, and of the places
    // of its elements, which are read from it again...
    for (auto it = state.env.begin(); it != state.env.end();) {
      auto element = elements.find(it->first.root);
      if ((it->first.root == loc.root &&
           StringRef(it->first.path).starts_with(loc.path + "/")) ||
          (element != elements.end() && element->second.first.root == loc.root))
        it = state.env.erase(it);
      else
        ++it;
    }
    // ... and changes the values of the places containing it (`x` when `x.f`
    // is written): each is a new value that differs from the old one in
    // that field only.
    SmallVector<std::pair<std::string, std::string>> outer;
    StringRef path(loc.path);
    while (!path.empty()) {
      path = path.take_front(path.rfind('/'));
      outer.push_back(
          {path.str(),
           load(Loc{loc.root, path.str()}, state, Sort{false, 64, false})});
    }
    for (auto &[prefix, old] : outer) {
      std::string fresh = declare({false, 64, false}, "h");
      updates[fresh] = {old, loc.path.substr(prefix.size())};
      state.env[Loc{loc.root, prefix}] = fresh;
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
    // Writing an element's place writes the element: the collection gets
    // the element's new value (and keeps its length).
    if (auto it = elements.find(loc.root); it != elements.end()) {
      auto [list, index] = it->second;
      storeElement(list, index, state.env[Loc{loc.root, ""}], state);
    }
  }

  /// Everything reachable through `loc` becomes unknown.
  /// The field `loc` (not a root) holds an unknown: as `store`, without a
  /// value for it, whose sort is its reader's to know (a field's type is
  /// not tracked). It is then read as that field of its new parent.
  void forget(const Loc &loc, State &state) {
    SmallVector<std::pair<std::string, std::string>> outer;
    StringRef path(loc.path);
    while (!path.empty()) {
      path = path.take_front(path.rfind('/'));
      outer.push_back(
          {path.str(),
           load(Loc{loc.root, path.str()}, state, Sort{false, 64, false})});
    }
    for (auto it = state.env.begin(); it != state.env.end();)
      it = it->first.root == loc.root &&
                   (it->first.path == loc.path ||
                    StringRef(it->first.path).starts_with(loc.path + "/"))
               ? state.env.erase(it)
               : std::next(it);
    for (auto &[prefix, old] : outer) {
      std::string fresh = declare({false, 64, false}, "h");
      updates[fresh] = {old, loc.path.substr(prefix.size())};
      state.env[Loc{loc.root, prefix}] = fresh;
    }
    state.refs.erase(loc);
  }

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
    // The empty union (`#lit<origin.union >`, no origin at all) reaches no
    // root; `MutAnyOrigin` is `#lit.any.origin`, which this does not match.
    if (origin.starts_with("#lit<origin.union >"))
      return named;
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
  void havocOrigin(StringRef origin, State &state, bool confined = false) {
    if (std::optional<SmallVector<Value>> named = rootsNamedBy(origin)) {
      for (Value root : *named) {
        Loc loc{root, ""};
        // An interior origin (`xs["element"]` in source,
        // `#lit.interior.origin<xs, "element">` here) reaches what the root
        // owns (a collection's elements), not the root's own fields: its
        // length is kept.
        std::string name = quotedOriginName(root);
        size_t at = origin.find(name);
        if (at != StringRef::npos &&
            origin.take_front(at).ends_with(
                "#lit.interior.origin<#kgen.param.decl.ref<")) {
          std::string old = load(loc, state, placeType(loc));
          std::string fresh = declare({false, 64, false}, "h");
          facts.push_back("(= " + lenOf(fresh) + " " + lenOf(old) + ")");
          store(loc, state, fresh);
          continue;
        }
        havoc(loc, state, placeType(loc));
      }
      return;
    }
    // An origin that names no root (`MutAnyOrigin`, a tensor's) may reach
    // any memory a reference or pointer leads to: not the local variables
    // that are never borrowed, which keep what they hold. A call with
    // `confinedWrites` reaches no memory in the default address space.
    std::map<Loc, std::string> kept;
    for (auto &[loc, value] : state.env)
      if (unborrowed.count(loc.root) || (confined && genericRoot(loc.root)))
        kept.insert({loc, value});
    std::map<Loc, Loc> keptRefs;
    for (auto &[loc, target] : state.refs)
      if (unborrowed.count(loc.root) || (confined && genericRoot(loc.root)))
        keptRefs.insert({loc, target});
    havocAll(state);
    state.env = std::move(kept);
    state.refs = std::move(keptRefs);
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

  /// The innermost runtime loop (`comptime` false) or `comptime for` with
  /// `label` (any, if empty).
  LoopFrame *findLoop(StringRef label, bool comptime) {
    for (LoopFrame *frame : llvm::reverse(loops))
      if (frame->comptime == comptime &&
          (label.empty() || frame->label == label))
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
    if (isa<LIT::FnOp>(op))
      return; // A local closure's definition: walked where it is called.
    if (op->getName().getStringRef() == "lit.error_return") {
      if (!closureExits.empty())
        closureExits.back()->push_back(state); // Raises out of the closure.
      state.alive = false; // Raises out of the function.
      return;
    }
    // `comptime assert c`: the compiler checks `c` when it elaborates this
    // code (for each instantiation and target), so it holds from here on.
    // Nonlinear ones are not assumed (sound: assuming less): products of
    // unknown parameters in every query's path made a MAX kernel's
    // verification take 18 s instead of 3, with the same results.
    if (auto assertion = dyn_cast<ParamAssertOp>(op)) {
      std::string cond = paramTerm(assertion.getCond(), {true, 1, false});
      if (!nonlinear(cond))
        state.pc = define({true, 1, false},
                          "(and " + state.pc + " " + cond + ")", "r");
      return;
    }
    if (isa<HLCF::ReturnOp>(op)) {
      if (!closureExits.empty()) {
        closureExits.back()->push_back(state); // Returns from the closure.
        state.alive = false;
        return;
      }
      if (refines && !inContract)
        proveTraitEnsures(op, state);
      state.alive = false;
      return;
    }
    if (isa<HLCF::YieldOp, ContractYieldOp, LIT::TryYieldOp>(op)) {
      state.yields.clear();
      for (Value operand : op->getOperands())
        state.yields.push_back(term(operand, state));
      return;
    }
    StringRef opName = op->getName().getStringRef();
    if (opName == "hlcf.comptime.for.continue" ||
        opName == "hlcf.comptime.for.break") {
      LoopFrame *frame = findLoop("", /*comptime=*/true);
      if (!frame) {
        notAnalyzed(op, state);
        state.alive = false;
        return;
      }
      (opName.ends_with("continue") ? frame->continues : frame->breaks)
          .push_back(state);
      state.alive = false;
      return;
    }
    if (opName == "hlcf.unreachable") {
      state.alive = false;
      return;
    }
    if (opName == "hlcf.comptime.yield") {
      state.yields.clear();
      for (Value operand : op->getOperands())
        state.yields.push_back(term(operand, state));
      return;
    }
    if (opName == "hlcf.comptime.if") {
      walkComptimeIf(op, state);
      return;
    }
    if (opName == "hlcf.comptime.for") {
      walkComptimeFor(op, state);
      return;
    }
    if (isa<HLCF::ContinueOp, HLCF::BreakOp>(op)) {
      LoopFrame *frame = findLoop(labelOf(op), /*comptime=*/false);
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

  /// `hlcf.if`, with `elif` arms: those are pairs of regions, a condition
  /// (ending in `hlcf.if.elifcond.yield`) evaluated where every earlier
  /// condition was false, and its arm.
  void walkIf(HLCF::IfOp ifOp, State &state) {
    auto elifs = ifOp.getElifRegions();
    if (elifs.size() % 2) {
      notAnalyzed(ifOp, state);
      return;
    }
    SmallVector<State> arms;
    // Each arm's own condition, relative to the path before the `if`.
    SmallVector<std::string> local;
    std::string none = "true"; // No condition so far held.
    // Whether every arm ends where it started: alive, with no fact added to
    // its path condition.
    bool plain = true;
    auto arm = [&](Block &block, const State &from, StringRef cond) {
      State taken = from;
      taken.pc = ("(and " + from.pc + " " + cond + ")").str();
      std::string entry = taken.pc;
      taken.yields.clear();
      walkBlock(block, taken);
      plain &= taken.alive && taken.pc == entry;
      arms.push_back(std::move(taken));
      local.push_back(("(and " + none + " " + cond + ")").str());
      none = ("(and " + none + " (not " + cond + "))").str();
    };
    std::string cond = term(ifOp.getCond(), state);
    noteCondition(cond);
    arm(ifOp.getThenBlock(), state, cond);
    // Where no condition so far held.
    State rest = state;
    rest.pc = "(and " + state.pc + " (not " + cond + "))";
    rest.yields.clear();
    for (size_t i = 0; i < elifs.size(); i += 2) {
      Block &condBlock = elifs[i].front();
      for (Operation &op : condBlock.without_terminator()) {
        if (!rest.alive)
          break;
        walkOp(&op, rest);
      }
      auto yield = dyn_cast<HLCF::IfElifCondYieldOp>(condBlock.getTerminator());
      if (!rest.alive || !yield || yield->getNumOperands() != 1) {
        notAnalyzed(ifOp, state);
        return;
      }
      std::string elifCond = term(yield->getOperand(0), rest);
      noteCondition(elifCond);
      arm(elifs[i + 1].front(), rest, elifCond);
      rest.pc = "(and " + rest.pc + " (not " + elifCond + "))";
    }
    arm(ifOp.getElseBlock(), rest, "true");
    // Values are chosen by the arms' own conditions rather than their whole
    // paths, so the same choice made in two places is the same term. That is
    // sound: the path after the `if` excludes the arms that did not reach it.
    SmallVector<std::string> paths;
    for (auto [taken, cond] : llvm::zip(arms, local)) {
      paths.push_back(taken.pc);
      taken.pc = cond;
    }
    SmallVector<std::string> results;
    State joined = merge(arms, &results);
    bindResults(ifOp, results);
    // The arms' conditions cover every case, so where no arm added a fact
    // (a value chosen by a condition, `x if c else y`), the path condition
    // after the `if` is the one before it. Keeping it so keeps later facts
    // out of needless disjunctions.
    if (plain) {
      joined.pc = state.pc;
    } else {
      std::string pc = "(or";
      for (auto [taken, path] : llvm::zip(arms, paths))
        if (taken.alive)
          pc += " " + path;
      joined.pc = define({true, 1, false}, pc + ")", "r");
    }
    joined.yields = state.yields;
    state = std::move(joined);
  }

  /// `hlcf.comptime.if`: elaboration keeps the first arm whose condition (a
  /// parameter expression in `conds`) is true, or the else arm (the last
  /// region). Conditions are evaluated with `paramTerm`, so a literal or an
  /// expression over the function's parameters decides or constrains the
  /// arm; the same expression in two places is the same choice. As an
  /// expression (`x if comptime (c) else y`) its results are what the taken
  /// arm yields.
  void walkComptimeIf(Operation *op, State &state) {
    auto ifOp = dyn_cast<HLCF::ComptimeIfOp>(op);
    if (!ifOp || op->getNumRegions() != ifOp.getConds().size() + 1) {
      notAnalyzed(op, state);
      return;
    }
    SmallVector<State> arms;
    std::string none = "true"; // No earlier condition held.
    bool taken = false;        // An earlier condition is true by constants.
    for (auto [i, condAttr] : llvm::enumerate(ifOp.getConds())) {
      auto typed = dyn_cast<TypedAttr>(condAttr);
      std::string cond = typed ? paramTerm(typed, {true, 1, false})
                               : declare({true, 1, false}, "p");
      State arm = state;
      arm.yields.clear();
      arm.pc = "(and " + state.pc + " " + none + " " + cond + ")";
      // An arm whose condition is false by constants alone (a launched
      // instantiation's `comptime if unroll_factor == 1`) is not compiled.
      // It is still walked, so that its obligations have an answer in
      // every instantiation, but as unreachable: no invariants are
      // searched for its loops (`findInvariants`).
      std::optional<bool> known =
          taken ? std::optional<bool>(false) : decided(cond);
      if (known && !*known)
        arm.pc = "false";
      if (!op->getRegion(i).empty())
        walkBlock(op->getRegion(i).front(), arm);
      arms.push_back(std::move(arm));
      taken = taken || (known && *known);
      none = "(and " + none + " (not " + cond + "))";
    }
    State elseArm = state;
    elseArm.yields.clear();
    elseArm.pc = "(and " + state.pc + " " + none + ")";
    Region &elseRegion = op->getRegion(op->getNumRegions() - 1);
    if (taken)
      elseArm.pc = "false";
    if (!elseRegion.empty())
      walkBlock(elseRegion.front(), elseArm);
    arms.push_back(std::move(elseArm));
    SmallVector<std::string> results;
    State joined = merge(arms, &results);
    if (op->getNumResults()) {
      // An arm that yields fewer values than the results gives unknowns.
      if (results.size() != op->getNumResults())
        results.clear();
      bindResults(op, results);
    }
    joined.yields = state.yields;
    state = std::move(joined);
  }

  /// The scope of the printed parameter expression `expr` read in `frame`:
  /// the frame's, or inside local closures their caller's, since a closure's
  /// body names its caller's parameters. An expression that names a
  /// closure's own parameter (a body refers to parameters by name, printed
  /// quoted) is that call's alone.
  static std::string scopeKey(ParamFrame *frame, StringRef expr) {
    std::string own;
    for (; frame && frame->transparent; frame = frame->parent)
      if (llvm::any_of(frame->values, [&](const auto &entry) {
            return expr.contains(("\"" + entry.first() + "\"").str());
          }))
        own += frame->scope + "|";
    return (frame ? frame->scope + "|" : std::string()) + own;
  }

  /// The value of a parameter expression (printed): one unknown per
  /// expression in the function.
  std::string parameterValue(const std::string &expr, Sort sort) {
    std::string scope = scopeKey(params, expr);
    // The value of a `comptime for`'s iteration is one unknown per loop:
    // its uses print the same iterator with its element type as `Int` or
    // as the range's `Element` alias.
    std::string key = expr;
    static llvm::Regex iterRe("^#kgen\\.param\\.expr<apply_result_slot, .*"
                              "paramfor_next_value.*"
                              "#kgen\\.param\\.decl\\.ref<(\"iter[^\"]*\")>");
    SmallVector<StringRef> m;
    if (iterRe.match(expr, &m)) {
      // Unrolled: this iteration's value.
      if (auto it = comptimeIterations.find(m[1].str());
          it != comptimeIterations.end() && !sort.isBool && sort.width == 64)
        return it->second;
      key = "paramfor_next_value|" + m[1].str();
    } else if (!comptimeIterations.empty()) {
      // An unrolled iteration has a next value; any other expression of
      // its iterator is that iteration's alone.
      static llvm::Regex hasNextRe(
          "^#kgen\\.param\\.expr<apply, .*paramfor_has_next.*"
          "#kgen\\.param\\.decl\\.ref<(\"iter[^\"]*\")>");
      SmallVector<StringRef> h;
      if (sort.isBool && hasNextRe.match(expr, &h) &&
          comptimeIterations.count(h[1].str()))
        return "true";
      for (auto &[iter, value] : comptimeIterations)
        if (StringRef(expr).contains(iter))
          key += "|" + iter + "=" + value;
    }
    auto [it, inserted] =
        parameterValues.try_emplace(scope + key + "|" + sort.str(), "");
    if (inserted) {
      it->second = declare(sort, "p");
      // Name it in the script, for `dump-dir` debugging.
      prelude += "; " + it->second + ": " +
                 StringRef(expr).take_front(300).str() + "\n";
      // An iteration over `range(end)` is in `[0, end)`, also where it is
      // read inside a larger parameter expression (`jj * tile_n`).
      if (m.size() > 1 && !sort.isBool && sort.width == 64)
        if (auto end = comptimeRangeEnds.find("#kgen.param.decl.ref<" +
                                              m[1].str() + ">");
            end != comptimeRangeEnds.end())
          facts.push_back("(and (bvsle " + bvConst(0, 64) + " " + it->second +
                          ") (bvslt " + it->second + " " + end->second + "))");
    }
    return it->second;
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

  /// Whether a root is memory in the default address space: a local
  /// variable or an argument, not a reference into shared or global memory
  /// (`ref[AddressSpace.SHARED] self`, a `ref` local bound to such memory).
  /// The address space of a reference, where it is a number (also under
  /// sugar, as `AddressSpace.SHARED` prints).
  static std::optional<uint64_t> addressSpace(LIT::RefType ref) {
    Attribute space = ref.getAddressSpace();
    while (auto sugar = dyn_cast_or_null<SugarAttr>(space))
      space = sugar.getCanonical();
    if (auto number = dyn_cast_or_null<IntegerAttr>(space))
      return number.getValue().getZExtValue();
    return std::nullopt;
  }

  static bool genericRoot(Value root) {
    auto generic = [](Type type) {
      auto ref = dyn_cast<LIT::RefType>(type);
      if (!ref)
        return false;
      std::optional<uint64_t> space = addressSpace(ref);
      return space && *space == 0;
    };
    if (!generic(root.getType()))
      return false;
    Type element = cast<LIT::RefType>(root.getType()).getElementType();
    return !isa<LIT::RefType>(element) || generic(element);
  }

  /// Whether a reference is into another address space than the default.
  static bool otherSpace(LIT::RefType ref) {
    std::optional<uint64_t> space = addressSpace(ref);
    return space && *space != 0;
  }

  /// Whether what `call` may write through an origin that names no variable
  /// is confined to other address spaces than the default one: its only
  /// arguments that can be written through are pointers into shared or
  /// global memory (`async_copy(src, dst)`, a barrier's `arrive`). Such a
  /// pointer does not lead to a local variable or an argument.
  bool confinedWrites(LIT::CallOp call) {
    static llvm::Regex spaceRe("\\{([0-9]+)\\}\\}\\)?>>$");
    bool any = false;
    for (Value operand : call.getOperands()) {
      Type type = operand.getType();
      if (auto ref = dyn_cast<LIT::RefType>(type)) {
        if (ref.isMutableKnown(false))
          continue;
        // A mutable reference into shared or global memory (`ptr[]`).
        if (!otherSpace(ref))
          return false;
        any = true;
        continue;
      }
      std::string text = printed(type);
      StringRef t(text);
      if (!t.consume_front("!lit.struct<@std::@memory::@pointer::@Pointer<")) {
        if (!isScalar(type))
          return false;
        continue;
      }
      // Immutable: nothing is written through it.
      if (t.starts_with(
              ":!lit.struct<@std::@builtin::@bool::@Bool> {:scalar<bool> false}"))
        continue;
      SmallVector<StringRef> m;
      int64_t space;
      if (!spaceRe.match(text, &m) || m[1].getAsInteger(10, space) || !space)
        return false;
      any = true;
    }
    return any;
  }

  /// The roots code may write; nullopt if it may write any of them.
  std::optional<llvm::DenseSet<Value>> writtenRoots(Operation *root) {
    llvm::DenseSet<Value> written;
    bool all = false;
    // Whether the origin being added is of a call with `confinedWrites`.
    bool confined = false;
    auto addOrigin = [&](StringRef origin) {
      if (std::optional<SmallVector<Value>> named = rootsNamedBy(origin)) {
        written.insert(named->begin(), named->end());
      } else if (confined) {
        for (Value r : roots)
          if (!genericRoot(r))
            written.insert(r);
      } else {
        all = true;
      }
    };
    storedPaths.clear();
    auto addRef = [&](Value ref) {
      if (std::optional<Loc> loc = placeOf(ref)) {
        written.insert(loc->root);
        storedPaths[loc->root].insert(loc->path);
      } else if (auto type = dyn_cast<LIT::RefType>(ref.getType()))
        addOrigin(printed(type.getOrigin()));
      else
        all = true;
    };
    // The code, and the bodies of the local closures it calls: they are
    // walked at their calls and write the variables they capture.
    SmallVector<Operation *> work{root};
    llvm::SmallPtrSet<Operation *, 4> seen{root};
    while (!work.empty())
      work.pop_back_val()->walk([&](Operation *op) {
        if (auto store = dyn_cast<LIT::RefStoreOp>(op)) {
          auto ref = dyn_cast<LIT::RefType>(store->getOperand(1).getType());
          confined = ref && otherSpace(ref);
          addRef(store->getOperand(1));
          confined = false;
        } else if (auto call = dyn_cast<LIT::CallOp>(op)) {
          ArrayRef<TypedAttr> bound;
          if (auto closure =
                  closures.find(closureCallee(call.getCallee(), bound));
              closure != closures.end() &&
              seen.insert(closure->second.getOperation()).second)
            work.push_back(closure->second.getOperation());
          confined = confinedWrites(call);
          for (Value operand : call.getOperands())
            if (auto ref = dyn_cast<LIT::RefType>(operand.getType());
                ref && !ref.isMutableKnown(false))
              addRef(operand);
          // Keep the printed list alive while its elements are used.
          std::string origins = printed(call.getImplicitOriginsAttr());
          for (StringRef origin : topLevelElements(origins))
            if (!origin.ends_with(": !lit.origin<false>"))
              addOrigin(origin);
          confined = false;
        } else if (op->getNumRegions() && !seen.count(op) &&
                   !isa<HLCF::IfOp, HLCF::LoopOp, LIT::TryOp, RequiresOp,
                        EnsuresOp, OldOp, ForallOp>(op) &&
                   op->getName().getStringRef() != "hlcf.comptime.if" &&
                   op->getName().getStringRef() != "hlcf.comptime.for") {
          all = true;
        }
      });
    // Locals declared in the code are written before they are read.
    root->walk([&](LIT::VarDeclOp decl) { written.insert(decl->getResult(0)); });
    storedRoots = written;
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
    walkLoopBody(loop, loop.getBody().front(), /*comptime=*/false, state);
  }

  /// `hlcf.comptime.for`: its body (region 0) runs once per value of the
  /// comptime iterator, which elaboration unrolls. It is analyzed as a loop,
  /// once, for an arbitrary iteration: what the body computes from the
  /// iteration's parameter values is unknown, so what is proven holds for
  /// every one of them.
  void walkComptimeFor(Operation *loop, State &state) {
    // Over `range(end)`: each iteration's value is in `[0, end)`.
    Attribute decl = loop->getAttr("paramDecl");
    Attribute initial = loop->getAttr("initial");
    std::string iterator, count;
    if (decl && initial) {
      std::optional<std::string> end;
      initial.walk([&](ParamOperatorAttr apply) {
        ArrayRef<TypedAttr> ops = apply.getOperands();
        if (!end && apply.getOpcode() == POC::Apply && ops.size() == 2 &&
            StringRef(printed(ops[0]))
                .starts_with("#kgen.symbol.constant<@std::@builtin::@range::@"
                             "\"range[::DType](::SIMD[$0, 1])\""))
          if (std::string e = paramTerm(ops[1], Sort{false, 64, true});
              !sortOfTerm(e).isBool && sortOfTerm(e).width == 64)
            end = e;
      });
      // Its uses name it `#kgen.param.decl.ref<"iter`30">`.
      static llvm::Regex nameRe("param.decl \\*(\"[^\"]*\")");
      SmallVector<StringRef> m;
      std::string declText = printed(decl);
      if (end && nameRe.match(declText, &m)) {
        comptimeRangeEnds["#kgen.param.decl.ref<" + m[1].str() + ">"] = *end;
        iterator = m[1].str();
        count = *end;
      }
    }
    if (loop->getNumResults() || loop->getNumRegions() < 1 ||
        loop->getRegion(0).empty() ||
        loop->getRegion(0).front().getNumArguments()) {
      notAnalyzed(loop, state);
      return;
    }
    if (unrollComptimeFor(loop, iterator, count, state))
      return;
    walkLoopBody(loop, loop->getRegion(0).front(), /*comptime=*/true, state);
  }

  /// A `comptime for` over `range(count)` with a small constant count: its
  /// body, walked once per iteration with the iteration's value, as
  /// elaboration unrolls it. What the iterations do adds up exactly (a
  /// counter the body increments), where the loop abstraction keeps only
  /// what holds for an arbitrary iteration. Only a loop that changes an
  /// integer variable from outside it (directly or in a closure it calls):
  /// for any other the abstraction loses nothing, and unrolling every short
  /// loop more than doubled the solver's work on the kernels. At most 8
  /// iterations, and 16 bodies through nested loops.
  bool unrollComptimeFor(Operation *loop, const std::string &iterator,
                         const std::string &count, State &state) {
    static llvm::Regex constRe("^\\(_ bv([0-9]+) 64\\)$");
    SmallVector<StringRef> m;
    uint64_t n;
    if (iterator.empty() || !constRe.match(count, &m) ||
        m[1].getAsInteger(10, n) || n > 8 ||
        unrolledBodies * std::max<uint64_t>(n, 1) > 16 ||
        comptimeIterations.count(iterator))
      return false;
    std::optional<llvm::DenseSet<Value>> written = writtenRoots(loop);
    if (!written || llvm::none_of(*written, [&](Value root) {
          Operation *def = root.getDefiningOp();
          if (def && (loop->isAncestor(def) ||
                      def->getParentOfType<LIT::FnOp>() != fn))
            return false; // The loop's own local, or a closure's.
          return isIntPlace(placeType(Loc{root, ""}));
        }))
      return false;
    uint64_t outer = unrolledBodies;
    unrolledBodies *= std::max<uint64_t>(n, 1);
    Block &body = loop->getRegion(0).front();
    SmallVector<State> exits;
    State current = state;
    current.yields.clear();
    for (uint64_t k = 0; k < n && current.alive; ++k) {
      comptimeIterations[iterator] = bvConst(k, 64);
      // The body's parameter constants are this iteration's.
      loop->walk([&](ParamConstantOp cst) { values.erase(cst->getResult(0)); });
      LoopFrame frame{labelOf(loop), true, {}, {}, {}, {}, {}};
      loops.push_back(&frame);
      State iteration = current;
      walkBlock(body, iteration);
      loops.pop_back();
      // Falling off the body, or `continue`, starts the next iteration.
      if (iteration.alive)
        frame.continues.push_back(iteration);
      llvm::append_range(exits, frame.breaks);
      current = merge(frame.continues);
    }
    comptimeIterations.erase(iterator);
    loop->walk([&](ParamConstantOp cst) { values.erase(cst->getResult(0)); });
    unrolledBodies = outer;
    if (current.alive)
      exits.push_back(current);
    State exit = merge(exits);
    exit.yields = state.yields;
    state = std::move(exit);
    return true;
  }

  /// Whether all that `loop` (and the local closures it calls) does to the
  /// variable `root`, a range's iterator, is to call `__next__` on it: no
  /// store to it, and no other call that may write it.
  bool onlyAdvanced(Operation *loop, Value root) {
    bool only = true;
    SmallVector<Operation *> work{loop};
    llvm::SmallPtrSet<Operation *, 4> seen{loop};
    auto names = [&](Value ref) {
      std::optional<Loc> loc = placeOf(ref);
      return !loc || loc->root == root;
    };
    while (only && !work.empty())
      work.pop_back_val()->walk([&](Operation *op) {
        if (!only)
          return;
        if (auto store = dyn_cast<LIT::RefStoreOp>(op)) {
          only = !names(store->getOperand(1));
        } else if (auto call = dyn_cast<LIT::CallOp>(op)) {
          ArrayRef<TypedAttr> bound;
          if (auto closure =
                  closures.find(closureCallee(call.getCallee(), bound));
              closure != closures.end() &&
              seen.insert(closure->second.getOperation()).second)
            work.push_back(closure->second.getOperation());
          bool takes = false;
          for (Value operand : call.getOperands())
            if (auto ref = dyn_cast<LIT::RefType>(operand.getType());
                ref && !ref.isMutableKnown(false) && names(operand))
              takes = true;
          // The error and result slots of `__next__` are other variables;
          // a reference that names no known place may be this one.
          if (takes) {
            std::optional<CalleeName> name = calleeName(call.getCallee());
            std::optional<Loc> self = call.getNumOperands()
                                          ? placeOf(call.getOperands()[0])
                                          : std::nullopt;
            only = name && self && self->root == root &&
                   StringRef(name->path).starts_with("std::builtin::range::") &&
                   StringRef(name->path).contains("::__next__") &&
                   llvm::all_of(call.getOperands().drop_front(),
                                [&](Value operand) {
                                  std::optional<Loc> loc = placeOf(operand);
                                  return loc && loc->root != root;
                                });
          }
        }
      });
    return only;
  }

  void walkLoopBody(Operation *loop, Block &bodyBlock, bool comptime,
                    State &state) {
    // The loop head: what the loop may write holds an unknown there, bound
    // by the invariants found below.
    std::optional<llvm::DenseSet<Value>> written = writtenRoots(loop);
    DenseMap<Value, std::set<std::string>> fields = storedPaths;
    // Locals declared in the loop are written whole.
    loop->walk([&](LIT::VarDeclOp decl) { fields.erase(decl->getResult(0)); });
    State before = state;
    before.yields.clear();
    State head = before;
    head.pc = define({true, 1, false}, before.pc, "r");
    std::string headReach = head.pc;
    if (written) {
      // In the function's order, so the unknowns are named the same each run.
      SmallVector<Value> sorted(written->begin(), written->end());
      llvm::sort(sorted, [&](Value a, Value b) {
        return order.lookup(a) < order.lookup(b);
      });
      for (Value root : sorted) {
        Loc loc{root, ""};
        // Only fields of it are written (`self.stage = next`): the others
        // keep their values.
        if (auto paths = fields.find(root);
            paths != fields.end() && !paths->second.empty() &&
            !paths->second.count("")) {
          for (const std::string &path : paths->second)
            forget(Loc{root, path}, head);
          continue;
        }
        head.env[loc] = declare(sortOf(placeType(loc)), "l");
        for (auto it = head.env.begin(); it != head.env.end();)
          it = it->first.root == root && !it->first.path.empty()
                   ? head.env.erase(it)
                   : std::next(it);
        head.refs.erase(loc);
        // A range the loop only advances keeps its end and its step: they
        // are not candidates for the invariant search, which the two more
        // unknowns per strided loop made several times larger.
        for (const char *field : {"/end", "/step"})
          if (auto it = before.env.find(Loc{root, field});
              it != before.env.end() && onlyAdvanced(loop, root))
            head.env[Loc{root, field}] = it->second;
      }
    } else {
      // Anything a reference leads to: not a variable that is never
      // borrowed and that the loop does not write, nor the end and step of
      // such a range that it only advances.
      std::map<Loc, std::string> kept;
      for (auto &[loc, value] : head.env) {
        if (!unborrowed.count(loc.root))
          continue;
        if (!storedRoots.count(loc.root) ||
            ((loc.path == "/end" || loc.path == "/step") &&
             onlyAdvanced(loop, loc.root)))
          kept.insert({loc, value});
      }
      // What such a `ref` local refers to does not change either.
      std::map<Loc, Loc> keptRefs;
      for (auto &[loc, target] : head.refs)
        if (unborrowed.count(loc.root) && !storedRoots.count(loc.root))
          keptRefs.insert({loc, target});
      havocAll(head);
      head.env = std::move(kept);
      head.refs = std::move(keptRefs);
    }
    State headAtEntry = head;
    LoopFrame frame{labelOf(loop), comptime, {}, {}, {}, {}, {}};
    loops.push_back(&frame);
    State body = head;
    walkBlock(bodyBlock, body);
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
    // An unreachable loop (in a `comptime if` arm that is not compiled)
    // needs none.
    if (std::optional<bool> reachable = decided(before.pc);
        reachable && !*reachable)
      return;
    // Integer places the body reads and may change, and that its conditions
    // depend on.
    llvm::StringSet<> relevant = closure(frame.conditions);
    SmallVector<Loc> places, handles;
    SmallVector<std::string> keptTerms;
    for (const Loc &loc : frame.loaded) {
      if (written && !written->count(loc.root))
        continue;
      std::string h = load(loc, head, Sort{false, 64, false});
      Sort sort = sortOfTerm(h);
      if (sort.isBool || sort.width != 64)
        continue;
      // What the loop head kept from before the loop (a range's end) needs
      // no invariant.
      if (auto kept = before.env.find(loc);
          kept != before.env.end() && kept->second == h) {
        keptTerms.push_back(h); // Still a bound to compare the others with.
        continue;
      }
      if (!relevant.contains(h))
        continue;
      // A whole variable of another type (a list, whose length the
      // conditions read) is a handle, not an integer: only whether it still
      // holds its value on entry means something, not its bounds. (A
      // field's type is not tracked; fields are kept.)
      if (loc.path.empty() && !isScalar(placeType(loc))) {
        handles.push_back(loc);
        continue;
      }
      places.push_back(loc);
    }
    if (places.empty() && handles.empty() && frame.lengths.empty())
      return;
    // Terms that do not change in the loop: values before it, and lengths
    // of lists it does not change.
    std::set<std::string> headNames;
    for (const Loc &loc : places)
      headNames.insert(load(loc, head, Sort{false, 64, true}));
    SmallVector<std::string> fixed(keptTerms.begin(), keptTerms.end());
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
    // The integer parameters the loop's conditions depend on (`stage_cnt`
    // in `raw_next == Self.stage_cnt`): bounds as good as a value from
    // before the loop.
    for (const auto &[key, name] : parameterValues)
      if (relevant.contains(name))
        if (auto it = sorts.find(name);
            it != sorts.end() && !it->second.isBool && it->second.width == 64)
          fixed.push_back(name);
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
    // Whether an iteration may change the value at `loc`: some end of the
    // body holds another term for it than the loop head.
    auto changes = [&](const Loc &loc) {
      std::string h = load(loc, head, Sort{false, 64, false});
      return llvm::any_of(frame.continues, [&](const State &end) {
        return load(loc, const_cast<State &>(end), Sort{false, 64, false}) !=
               h;
      });
    };
    SmallVector<Candidate> candidates;
    for (const Loc &loc : handles)
      candidates.push_back(
          {loc, "=", std::nullopt, load(loc, before, Sort{false, 64, true})});
    for (const Loc &loc : lists) {
      bool resized = changes(loc);
      std::string own = lenOf(load(loc, before, Sort{false, 64, false}));
      candidates.push_back({loc, "bvsge", std::nullopt, bvConst(0, 64), true});
      for (const std::string &t : fixed) {
        // Not against another list's length: a list the loop changes is
        // bounded by its own length on entry and by the loop's variables.
        if (StringRef(t).starts_with("(len ") && t != own)
          continue;
        for (const char *op : {"bvsle", "bvsge"})
          candidates.push_back({loc, op, std::nullopt, t, true});
      }
      for (const Loc &other : places) {
        for (const char *op : {"bvsle", "bvslt", "bvsge"})
          candidates.push_back({loc, op, other, "", true});
        // Only a list and a variable the iteration both change: against
        // one it does not (an iterator's end, a bound), the template
        // restates a bound on the length.
        if (resized && changes(other))
          for (const char *combine : {"bvadd", "bvsub"})
            candidates.push_back({loc, "=", other, "", true, combine});
      }
    }
    for (const Loc &loc : places) {
      // The cursor of a strided range (`evalStridedRange`), by its step.
      if (StringRef(loc.path).ends_with("/start")) {
        Loc step{loc.root,
                 StringRef(loc.path).drop_back(strlen("/start")).str() +
                     "/step"};
        if (before.env.count(step)) {
          candidates.push_back({loc, "=", step, "", false, "", true});
          for (const Loc &other : places)
            if (!(other == loc) && other.root != loc.root && changes(other))
              candidates.push_back(
                  {loc, "=", step, "", false, "", true, other});
        }
      }
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
      if (c.stride) {
        std::string start0 = load(c.lhs, before, Sort{false, 64, true});
        std::string step0 = load(*c.rhsLoc, before, Sort{false, 64, true});
        if (c.counter)
          return "(= (bvsub " + load(c.lhs, at, Sort{false, 64, true}) + " " +
                 start0 + ") (bvmul (bvsub " +
                 load(*c.counter, at, Sort{false, 64, true}) + " " +
                 load(*c.counter, before, Sort{false, 64, true}) + ") " +
                 step0 + "))";
        return "(= (bvsrem (bvsub " + load(c.lhs, at, Sort{false, 64, true}) +
               " " + start0 + ") " + step0 + ") " + bvConst(0, 64) + ")";
      }
      if (!c.combine.empty()) {
        auto value = [&](State &in) {
          std::string lhs = load(c.lhs, in, Sort{false, 64, true});
          if (c.length)
            lhs = lenOf(lhs);
          return "(" + c.combine + " " + lhs + " " +
                 load(*c.rhsLoc, in, Sort{false, 64, true}) + ")";
        };
        return "(= " + value(at) + " " + value(before) + ")";
      }
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
    auto houdini = [&](ArrayRef<Candidate> set) {
      std::vector<bool> kept(set.size(), true);
      for (unsigned round = 0; round < 8; ++round) {
        // The queries, in the order `rebuild` emits them.
        SmallVector<size_t> asked;
        for (size_t i = 0; i < set.size(); ++i)
          if (kept[i])
            asked.append(1 + ends.size(), i);
        std::string text = rebuild(set, kept, before, head, ends, render);
        std::string run = dumpPrefix + ".houdini" + std::to_string(houdiniRuns++);
        std::optional<SmallVector<Answer>> answers = runZ3(solver, text, run);
        if (answers)
          runCases(text, *answers, run);
        // With `invariant-retry`: a candidate left undecided, and not
        // refuted, is asked once more alone with four times the limit.
        // After a large body (a kernel's unrolled iterations) the
        // invariants that relate a counter to the next loop's cursor need
        // slightly more than the limit (1.2M to 2M against 1M), on entry
        // or to be kept, and which ones changes with the script. Off by
        // default: the invariants it keeps turn obligations a kernel
        // without clauses fails quickly into ones asked up to the full
        // limit (gemv: 1380M of work without it, 2020M with it).
        std::vector<bool> open(set.size(), false), refuted(set.size(), false);
        if (answers)
          for (auto [k, i] : llvm::enumerate(asked))
            if (k < answers->size()) {
              if ((*answers)[k] == Answer::Unknown)
                open[i] = true;
              else if ((*answers)[k] != Answer::Proven)
                refuted[i] = true;
            }
        bool any = false;
        for (size_t i = 0; i < set.size(); ++i) {
          open[i] = open[i] && !refuted[i];
          any = any || open[i];
        }
        if (any && solver.invariantRetry) {
          SmallVector<size_t> again;
          for (size_t i = 0; i < set.size(); ++i)
            if (kept[i] && open[i])
              again.append(1 + ends.size(), i);
          std::string retry =
              rebuild(set, kept, before, head, ends, render, &open, 4);
          std::string name =
              dumpPrefix + ".houdini" + std::to_string(houdiniRuns++);
          std::optional<SmallVector<Answer>> more = runZ3(solver, retry, name);
          std::vector<bool> failed(set.size(), false);
          for (auto [k, i] : llvm::enumerate(again))
            if (!more || k >= more->size() || (*more)[k] != Answer::Proven)
              failed[i] = true;
          for (auto [k, i] : llvm::enumerate(asked))
            if (open[i] && !failed[i] && k < answers->size() &&
                (*answers)[k] == Answer::Unknown)
              (*answers)[k] = Answer::Proven;
        }
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
      for (size_t i = 0; i < set.size(); ++i)
        if (kept[i]) {
          facts.push_back(
              ("(=> " + headReach + " " + render(set[i], head) + ")")
                  .str());
          ++invariantsFound;
        }
    };
    // The same candidate from two templates (`len(xs) >= 0` from the bound
    // against 0 and from a fixed 0) is asked once.
    {
      std::set<std::string> seen;
      llvm::erase_if(candidates, [&](const Candidate &c) {
        return !seen.insert(render(c, head)).second;
      });
    }
    // The templates relating a length to a loop variable last, so that
    // `rebuild` can keep them out of the others' assumptions.
    std::stable_partition(candidates.begin(), candidates.end(),
                          [](const Candidate &c) { return c.combine.empty(); });
    houdini(candidates);
  }

  /// The Houdini script for the kept candidates: rendering first (which may
  /// add definitions to the prelude), then the header, then the queries; of
  /// the candidates in `only` alone (all kept ones are assumed), with
  /// `scale` times the limit, when asking again what was left undecided.
  template <typename Render>
  std::string
  rebuild(ArrayRef<Candidate> candidates, const std::vector<bool> &kept,
          State &before, State &head, ArrayRef<State *> ends, Render &render,
          const std::vector<bool> *only = nullptr, unsigned scale = 1) {
    // The templates relating a length to a loop variable (`combine`, last)
    // are assumed only in their own queries: assumed in every query, they
    // made z3 much slower on the whole script, although each query alone
    // stayed quick.
    SmallVector<std::string> assumed;
    size_t plain = 0;
    for (size_t i = 0; i < candidates.size(); ++i)
      if (kept[i]) {
        assumed.push_back(render(candidates[i], head));
        if (candidates[i].combine.empty())
          plain = assumed.size();
      }
    std::string queries;
    for (size_t i = 0; i < candidates.size(); ++i) {
      if (!kept[i] || (only && !(*only)[i]))
        continue;
      ArrayRef<std::string> premises(assumed);
      if (candidates[i].combine.empty())
        premises = premises.take_front(plain);
      // A candidate the solver cannot decide quickly is dropped, which is
      // sound: it is only not assumed. Within the function's own budget,
      // where it has one (a launched kernel's generic proof).
      unsigned limit =
          std::max(1u, (queryRlimit ? queryRlimit : solver.rlimit) / 20);
      if (solver.houdiniRlimit)
        limit = std::min(limit, solver.houdiniRlimit);
      if (scale > 1)
        limit = std::min<uint64_t>(uint64_t(limit) * scale,
                                   queryRlimit ? queryRlimit : solver.rlimit);
      queries += query({before.pc}, render(candidates[i], before), limit);
      for (State *end : ends) {
        SmallVector<std::string> assumptions{end->pc};
        assumptions.append(premises.begin(), premises.end());
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

  /// Where the implementation returns, the trait method's postcondition, on
  /// its arguments and the returned value (for a named result).
  void proveTraitEnsures(Operation *ret, State &state) {
    Block &entry = fn.getFunctionBody().front();
    callResult = ret->getNumOperands() == 1 ? ret->getOperand(0) : Value();
    State *old = entryState ? &*entryState : nullptr;
    for (EnsuresOp ensures : postconditionClauses(refines)) {
      Obligation ob{state.pc, "false", ret->getLoc(), ensures.getLoc(),
                    displayName(refines)};
      ob.postcondition = true;
      ob.implementation = displayName(fn);
      if (MaybeTerm cond =
              instantiate(ensures.getBody(), ensures.getArgs(), state,
                          entry.getArguments(), refines, old, false))
        ob.cond = *cond;
      else
        ob.analyzed = false;
      obligations.push_back(ob);
    }
    callResult = {};
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
    if (auto witness = dyn_cast<GetWitnessAttr>(call.getCallee())) {
      if (LIT::FnOp impl = selfImpl(witness))
        return impl;
      return traitMethod(witness.getTraitSymbol().getSymbol(),
                         witness.getWitnessName());
    }
    SymbolRefAttr callee = call.getDirectCallee();
    if (!callee)
      return {};
    // Symbols are absolute (`@std::@collections::...`): resolve them from the
    // module, since a `lit.fn` is a symbol table of its own.
    return dyn_cast_or_null<LIT::FnOp>(symbols.lookupSymbolIn(module, callee));
  }

  /// The call a struct's wrapper for an inherited trait default
  /// (`defaultFnRef`) forwards its arguments to, in order; null for any
  /// other function.
  static LIT::CallOp forwardedCall(LIT::FnOp fn) {
    if (!fn || !fn.getDefaultFnRefAttr() || fn.getFunctionBody().empty())
      return {};
    Block &entry = fn.getFunctionBody().front();
    LIT::CallOp call;
    for (auto op : entry.getOps<LIT::CallOp>()) {
      if (call)
        return {};
      call = op;
    }
    if (!call || call.getOperands() != ValueRange(entry.getArguments()))
      return {};
    return call;
  }

  /// When refining for a struct, a call of a trait method on `Self` (in the
  /// trait's clauses): the struct's own implementation, whose clauses then
  /// hold.
  LIT::FnOp selfImpl(GetWitnessAttr witness) {
    if (selfStruct.empty())
      return {};
    auto ref = dyn_cast<ParamDeclRefAttr>(resolveParam(witness.getTypeValue()));
    if (!ref || !ref.getName().getValue().starts_with("_Self"))
      return {};
    auto it = impls.byKey.find(implKey(witness.getTraitSymbol(),
                                       witness.getWitnessName(), selfStruct));
    return it == impls.byKey.end() ? LIT::FnOp() : it->second;
  }

  /// A generic call (`#kgen.get_witness<T, @Trait, "m($0)">`): the method's
  /// declaration in `trait` or the traits it refines, whose clauses every
  /// implementation provides.
  LIT::FnOp traitMethod(SymbolRefAttr trait, StringAttr method) {
    auto decl = dyn_cast_or_null<LIT::TraitDeclOp>(
        symbols.lookupSymbolIn(module, trait));
    if (!decl)
      return {};
    if (auto fn =
            dyn_cast_or_null<LIT::FnOp>(symbols.lookupSymbolIn(decl, method)))
      return fn;
    for (TraitSymbolAttr parent : decl.getImmediateParents())
      if (LIT::FnOp fn = traitMethod(parent.getSymbol(), method))
        return fn;
    return {};
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
        // A store through a reference into shared or global memory
        // (`smem_ptr[i] = v`) writes that memory only.
        havocOrigin(printed(ref.getOrigin()), state, otherSpace(ref));
      else
        havocAll(state);
      return;
    }
    // The field of an integer wrapper value is the value itself.
    if (auto extract = dyn_cast<LIT::StructExtractOp>(op);
        extract && intWrapper(extract.getContainer().getType())) {
      values[extract.getResult()] = term(extract.getContainer(), state);
      return;
    }
    // A field of a struct value (`self.ptr` of a struct passed by value):
    // the field the value was loaded with, else that field of the value,
    // the same wherever it is read from the same value.
    if (auto extract = dyn_cast<LIT::StructExtractOp>(op))
      if (auto field = op->getAttrOfType<StringAttr>("field")) {
        Value container = extract.getContainer(), result = extract.getResult();
        std::string path = "/" + field.getValue().str();
        Sort sort = sortOf(result.getType());
        std::map<std::string, std::string> inner;
        std::string known;
        if (auto it = records.find(container); it != records.end())
          for (auto &[name, term] : it->second) {
            if (name == path)
              known = term;
            else if (StringRef(name).starts_with(path + "/"))
              inner[name.substr(path.size())] = term;
          }
        std::string container_ = term(container, state);
        if (known.empty() && !sortOfTerm(container_).isBool &&
            sortOfTerm(container_).width == 64)
          known = fieldOf(container_, path, sort);
        if (!known.empty() && sortOfTerm(known).isBool == sort.isBool &&
            (sort.isBool || sortOfTerm(known).width == sort.width)) {
          values[result] = known;
          if (!inner.empty())
            records[result] = std::move(inner);
          return;
        }
      }
    if (isa<LIT::RefLoadOp>(op) ||
        op->getName().getStringRef() == "lit.load.consume") {
      Operation *loadOp = op;
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
    // An `IntLiteral`: its value is in its type.
    if (!values.count(result)) {
      static llvm::Regex intLiteralRe("IntLiteral ?<:!pop.int_literal (-?[0-9]+)>");
      std::string type = printed(result.getType());
      int64_t v;
      if (intLiteralRe.match(type, &m) && !m[1].getAsInteger(10, v))
        values[result] = bvConst(v, 64);
    }
    // A parameter expression.
    if (!values.count(result))
      values[result] = paramTerm(cst.getValue(), sort);
    // The value of an iteration of a `comptime for` over `range(end)`.
    if (!sort.isBool && sort.width == 64 && !comptimeRangeEnds.empty()) {
      std::string text = printed(cst.getValue());
      if (StringRef(text).contains("paramfor_next_value"))
        for (auto &[name, end] : comptimeRangeEnds)
          if (StringRef(text).contains(name)) {
            const std::string &v = values[result];
            facts.push_back("(and (bvsle " + bvConst(0, 64) + " " + v +
                            ") (bvslt " + v + " " + end + "))");
            break;
          }
    }
  }

  /// An integer or Boolean `SIMD` operator on terms, as its SMT expression.
  static MaybeTerm simdOperator(StringRef method, Sort sort,
                                ArrayRef<std::string> args, Sort &resultSort) {
    bool s = sort.isSigned;
    static const std::pair<const char *, const char *> binary[] = {
        {"__add__", "bvadd"}, {"__sub__", "bvsub"}, {"__mul__", "bvmul"},
        {"__and__", "bvand"}, {"__or__", "bvor"},   {"__xor__", "bvxor"},
    };
    const std::pair<const char *, const char *> compare[] = {
        {"__lt__", s ? "bvslt" : "bvult"}, {"__le__", s ? "bvsle" : "bvule"},
        {"__gt__", s ? "bvsgt" : "bvugt"}, {"__ge__", s ? "bvsge" : "bvuge"},
    };
    if (args.size() == 2) {
      for (auto &[m, smt] : binary)
        if (method == m) {
          resultSort = sort;
          return "(" + std::string(smt) + " " + args[0] + " " + args[1] + ")";
        }
      for (auto &[m, smt] : compare)
        if (method == m) {
          resultSort = {true, 1, false};
          return "(" + std::string(smt) + " " + args[0] + " " + args[1] + ")";
        }
      if (method == "__eq__" || method == "__ne__") {
        resultSort = {true, 1, false};
        std::string eq = "(= " + args[0] + " " + args[1] + ")";
        return method == "__eq__" ? eq : "(not " + eq + ")";
      }
    }
    if (args.size() == 2 && (method == "__floordiv__" || method == "__mod__")) {
      resultSort = sort;
      return floorDivision(method == "__mod__", sort, args[0], args[1]);
    }
    if (args.size() == 2 && method == "__ceildiv__") {
      resultSort = sort;
      return ceilDivision(sort, args[0], args[1]);
    }
    if (args.size() == 1 && method == "__neg__") {
      resultSort = sort;
      return "(bvneg " + args[0] + ")";
    }
    return std::nullopt;
  }

  /// With `check-division`, a division at `call` by `divisor` must not
  /// divide by 0. Mojo's integer `//` and `%` return 0 then (undocumented;
  /// `SIMD` masks the result), which is almost always a bug.
  void checkDivisor(LIT::CallOp call, StringRef divisor, State &state) {
    if (!checkDivision || inContract || isNonzeroLiteral(divisor) ||
        sortOfTerm(divisor).isBool)
      return;
    Obligation ob{state.pc, "false", call.getLoc(), call.getLoc(), "division"};
    ob.claim = "that the divisor is not 0";
    ob.cond = define({true, 1, false}, "(not (= " + divisor.str() + " " +
                                           bvConst(0, sortOfTerm(divisor).width) +
                                           "))");
    noteCondition(ob.cond);
    obligations.push_back(ob);
  }

  /// A partial tile (or a view of one) passed to a function other than the
  /// `TileTensor` methods whose clauses check `_valid_dim`: that function
  /// was verified for a tensor every element of which is backed, so the
  /// tile must not be partial here (its valid extents its `dim`s).
  void checkPartialTiles(LIT::CallOp call, const CalleeName &name,
                         State &state) {
    StringRef path = name.path;
    if (path.consume_front("layout::tile_tensor::TileTensor::"))
      for (const char *method :
           {"tile[", "vectorize[", "dim[", "_valid_dim[", "__getitem__",
            "__setitem__", "load[", "store[", "_indices_in_bounds",
            "_access_in_bounds", "_vectorize_in_bounds", "_tile_in_bounds",
            "_tile_coords_in_bounds", "_tile_shape_in_bounds"})
        if (path.starts_with(method))
          return;
    for (Value operand : call.getOperands()) {
      Type type = operand.getType();
      if (auto ref = dyn_cast<LIT::RefType>(type))
        type = ref.getElementType();
      auto st = dyn_cast<LIT::StructType>(type);
      if (!st || printed(st.getSymbol()) != "@layout::@tile_tensor::@TileTensor")
        continue;
      auto it = partialValid.find(tensorTerm(operand, state));
      if (it == partialValid.end())
        continue;
      std::string all = "true";
      for (auto [k, valid] : llvm::enumerate(it->second)) {
        MaybeTerm d = tensorDim(st.getParamValues(), operand, k, state);
        if (!d)
          return;
        all = "(and " + all + " (= " + valid + " " + *d + "))";
      }
      LIT::FnOp callee = lookup(call);
      Obligation ob{state.pc, "false", call.getLoc(), call.getLoc(),
                    callee ? displayName(callee) : name.path};
      ob.claim = "that the tile passed is not partial (every element backed)";
      ob.cond = define({true, 1, false}, all);
      noteCondition(ob.cond);
      obligations.push_back(ob);
    }
  }

  /// Whether `term` is a bit-vector literal other than 0 (`(_ bv4 64)`).
  static bool isNonzeroLiteral(StringRef term) {
    return term.starts_with("(_ bv") && !term.starts_with("(_ bv0 ");
  }

  /// `a // b` or `a % b` as Mojo defines them for integers: rounding towards
  /// negative infinity, so the remainder has the divisor's sign. SMT-LIB's
  /// signed division truncates, so a nonzero remainder of the other sign is
  /// corrected. Both are 0 for `b == 0`, as `SIMD` defines them (SMT-LIB's
  /// are not: an unsigned quotient of all ones, for one).
  static std::string floorDivision(bool remainder, Sort sort, StringRef a,
                                   StringRef b) {
    std::string w = std::to_string(sort.width);
    std::string zero = "(_ bv0 " + w + ")";
    std::string result;
    if (!sort.isSigned) {
      result = ("(" + Twine(remainder ? "bvurem" : "bvudiv") + " " + a + " " +
                b + ")")
                   .str();
    } else {
      std::string q = ("(bvsdiv " + a + " " + b + ")").str();
      std::string r = ("(bvsrem " + a + " " + b + ")").str();
      std::string adjust =
          ("(and (not (= " + r + " " + zero + ")) (not (= (bvslt " + a + " " +
           zero + ") (bvslt " + b + " " + zero + "))))")
              .str();
      result = remainder ? "(ite " + adjust + " (bvadd " + r + " " + b.str() +
                               ") " + r + ")"
                         : "(ite " + adjust + " (bvsub " + q + " (_ bv1 " + w +
                               ")) " + q + ")";
    }
    // The zero case only where the divisor may be 0: an `ite` on every
    // division made a MAX kernel's verification 5x slower (145 s against 27).
    if (isNonzeroLiteral(b))
      return result;
    return "(ite (= " + b.str() + " " + zero + ") " + zero + " " + result + ")";
  }

  /// `a.__ceildiv__(b)` as `SIMD` defines it: `-(a // -b)` when signed, the
  /// quotient plus one for a nonzero remainder when unsigned.
  static std::string ceilDivision(Sort sort, StringRef a, StringRef b) {
    if (sort.isSigned)
      return "(bvneg " +
             floorDivision(false, sort, a, ("(bvneg " + b + ")").str()) + ")";
    std::string w = std::to_string(sort.width);
    return "(bvadd " + floorDivision(false, sort, a, b) +
           " (ite (= " + floorDivision(true, sort, a, b) + " (_ bv0 " + w +
           ")) (_ bv0 " + w + ") (_ bv1 " + w + ")))";
  }

  /// The compilation target, as `std.sys.info` describes it: the triple is
  /// NVIDIA's, AMD's or Apple's GPU (at most one; on the host none), RDNA
  /// is an AMD GPU, and the build's accelerator (`--target-accelerator`,
  /// `_accelerator_arch()`) is absent or of one vendor. `is_gpu()`,
  /// `has_*_accelerator()`, `has_accelerator()` and `_resolve_warp_size()`
  /// (`WARP_SIZE`) are what their bodies make of these; the warp size of an
  /// accelerator the host only names is an unknown.
  MaybeTerm targetPredicate(StringRef path, Sort sort) {
    auto declared = [&](StringRef name) {
      std::string symbol = ("target_" + name).str();
      if (gpuIds.insert(symbol).second) {
        prelude += "(declare-const " + symbol + " Bool)\n";
        sorts[symbol] = Sort{true, 1, false};
      }
      return symbol;
    };
    // In the device view (a kernel's assertions at its launch) the GPU
    // triple is the build's accelerator's.
    auto flag = [&](StringRef name) {
      if (deviceView) {
        if (name == "nvidia_gpu")
          return declared("accelerator_nvidia");
        if (name == "amd_gpu")
          return declared("accelerator_amd");
        if (name == "apple_gpu")
          return declared("accelerator_apple");
        if (name == "amd_rdna")
          return declared("accelerator_amd_rdna");
      }
      return declared(name);
    };
    if (!targetDeclared) {
      targetDeclared = true;
      std::string nv = declared("nvidia_gpu"), amd = declared("amd_gpu"),
                  apple = declared("apple_gpu"), rdna = declared("amd_rdna");
      std::string accNv = declared("accelerator_nvidia"),
                  accAmd = declared("accelerator_amd"),
                  accApple = declared("accelerator_apple"),
                  accRdna = declared("accelerator_amd_rdna"),
                  acc = declared("accelerator");
      facts.push_back("(=> " + accRdna + " " + accAmd + ")");
      facts.push_back("(and (not (and " + nv + " " + amd + ")) (not (and " +
                      nv + " " + apple + ")) (not (and " + amd + " " + apple +
                      ")))");
      facts.push_back("(=> " + rdna + " " + amd + ")");
      facts.push_back("(and (not (and " + accNv + " " + accAmd +
                      ")) (not (and " + accNv + " " + accApple +
                      ")) (not (and " + accAmd + " " + accApple + ")))");
      facts.push_back("(=> (or " + accNv + " " + accAmd + " " + accApple +
                      ") " + acc + ")");
    }
    std::string gpu = "(or " + flag("nvidia_gpu") + " " + flag("amd_gpu") +
                      " " + flag("apple_gpu") + ")";
    StringRef info = "std::sys::info::";
    if (path.consume_front(info)) {
      if (!sort.isBool)
        return std::nullopt;
      if (path == "is_nvidia_gpu()")
        return flag("nvidia_gpu");
      if (path == "is_amd_gpu()")
        return flag("amd_gpu");
      if (path == "is_apple_gpu()")
        return flag("apple_gpu");
      if (path == "_is_amd_rdna()")
        return flag("amd_rdna");
      if (path == "is_gpu()")
        return define(sort, gpu);
      if (path == "has_nvidia_gpu_accelerator()")
        return define(sort, "(or " + flag("nvidia_gpu") + " " +
                                flag("accelerator_nvidia") + ")");
      if (path == "has_amd_gpu_accelerator()")
        return define(sort, "(or " + flag("amd_gpu") + " " +
                                flag("accelerator_amd") + ")");
      if (path == "has_apple_gpu_accelerator()")
        return define(sort, "(or " + flag("apple_gpu") + " " +
                                flag("accelerator_apple") + ")");
      if (path == "has_accelerator()")
        return define(sort, "(or " + gpu + " " + flag("accelerator") + ")");
      return std::nullopt;
    }
    if (path == "std::_gpu::globals::_resolve_warp_size()" && !sort.isBool &&
        sort.width == 64) {
      // One unknown: every `WARP_SIZE` of the function is the same.
      std::string other = "target_accelerator_warp_size";
      if (gpuIds.insert(other).second) {
        prelude += "(declare-const " + other + " (_ BitVec 64))\n";
        sorts[other] = sort;
        // Every `GPUInfo` the stdlib names says 32 or 64 (assumed).
        facts.push_back("(or (= " + other + " " + bvConst(32, 64) + ") (= " +
                        other + " " + bvConst(64, 64) + "))");
      }
      // One constant per view, so case splitting (`runCases`) can give it
      // each of its values.
      std::string w = deviceView ? "target_device_warp_size" : "target_warp_size";
      if (gpuIds.insert(w).second) {
        prelude += "(declare-const " + w + " (_ BitVec 64))\n";
        sorts[w] = sort;
        auto c = [&](int64_t v) { return bvConst(v, 64); };
        facts.push_back("(= " + w + " (ite " + flag("nvidia_gpu") + " " +
                        c(32) + " (ite " + flag("amd_rdna") + " " + c(32) +
                        " (ite " + flag("amd_gpu") + " " + c(64) + " (ite " +
                        flag("apple_gpu") + " " + c(32) + " (ite (not " +
                        flag("accelerator") + ") " + c(0) + " " + other +
                        "))))))");
        finiteDomains[w] = {0, 32, 64};
      }
      return w;
    }
    return std::nullopt;
  }

  /// A parameter expression's value: literals, and the integer and Boolean
  /// operators of parameter expressions (`add`, `lt`, `cond`, ...) and of
  /// `SIMD` (`apply` of `__ge__`, ...) over them. Anything else (a parameter,
  /// a value computed at comptime) is one unknown per expression, so the same
  /// expression in two places has the same value.
  std::string paramTerm(TypedAttr attr, Sort sort, unsigned depth = 0) {
    if (depth > 32)
      return parameterValue(printed(attr), sort);
    if (auto sugar = dyn_cast<SugarAttr>(attr))
      return paramTerm(sugar.getCanonical(), sort, depth + 1);
    // A layout's rank and flat rank (`t.rank`, the `rank` witness of its
    // layout type): one term per layout, for `num_elements()` and
    // `static_shape`.
    if (auto witness = dyn_cast<GetWitnessAttr>(attr);
        witness && !sort.isBool && sort.width == 64 &&
        (witness.getWitnessName() == "rank" ||
         witness.getWitnessName() == "flat_rank"))
      return layoutWitness(witness.getWitnessName(),
                           layoutKey(witness.getTypeValue()));
    // `static_shape[k]` of a layout type: its `k`th flat extent, or -1 where
    // that is known only at run time.
    if (auto bind = dyn_cast<BindParamsAttr>(attr);
        bind && !sort.isBool && sort.width == 64 &&
        bind.getParamValues().size() == 1)
      if (auto witness = dyn_cast<GetWitnessAttr>(bind.getGenerator());
          witness && witness.getWitnessName() == "static_shape") {
        std::string k = paramTerm(bind.getParamValues()[0], sort, depth + 1);
        static llvm::Regex constRe("^\\(_ bv([0-9]+) 64\\)$");
        SmallVector<StringRef> m;
        int64_t index;
        if (constRe.match(k, &m) && !m[1].getAsInteger(10, index)) {
          std::string term = parameterValue(printed(attr), sort);
          std::pair<std::string, int64_t> at{layoutKey(witness.getTypeValue()),
                                             index};
          auto &terms = staticShapeTerms[at];
          if (!llvm::is_contained(terms, term))
            terms.push_back(term);
          relateStaticShape(at);
          return term;
        }
      }
    // `static_stride[k]` of a layout type: its `k`th flat stride, or -1
    // where that is known only at run time. One term per layout and `k`,
    // which `_access_in_bounds` of a tile of the layout reads too.
    if (auto bind = dyn_cast<BindParamsAttr>(attr);
        bind && !sort.isBool && sort.width == 64 &&
        bind.getParamValues().size() == 1)
      if (auto witness = dyn_cast<GetWitnessAttr>(bind.getGenerator());
          witness && witness.getWitnessName() == "static_stride") {
        std::string k = paramTerm(bind.getParamValues()[0], sort, depth + 1);
        static llvm::Regex constRe("^\\(_ bv([0-9]+) 64\\)$");
        SmallVector<StringRef> m;
        int64_t index;
        if (constRe.match(k, &m) && !m[1].getAsInteger(10, index)) {
          auto [it, inserted] = staticStrideTerms.try_emplace(
              {layoutKey(witness.getTypeValue()), index}, "");
          if (inserted)
            it->second = parameterValue(printed(attr), sort);
          return it->second;
        }
      }
    if (auto index = dyn_cast<ParamIndexRefAttr>(attr);
        index && positionalParams && index.getDepth() == 0 &&
        index.getIndex() < positionalParams->size())
      return paramTerm(
          ParamDeclRefAttr::get((*positionalParams)[index.getIndex()]), sort,
          depth + 1);
    // A callee's parameter: the call's value for it, in the caller.
    if (auto ref = dyn_cast<ParamDeclRefAttr>(attr))
      for (ParamFrame *frame = params; frame;
           frame = frame->transparent ? frame->parent : nullptr)
        if (auto it = frame->values.find(ref.getName());
            it != frame->values.end()) {
          ParamFrame *callee = params;
          params = frame->parent;
          std::string t = paramTerm(it->second, sort, depth + 1);
          params = callee;
          return t;
        }
    if (auto simd = dyn_cast<SIMDAttr>(attr)) {
      ArrayRef<DTypeValue> vals = simd.getValues();
      if (vals.size() == 1) {
        if (sort.isBool)
          return vals[0].getBoolVal() ? "true" : "false";
        if (vals[0].getData().getBitWidth() == sort.width)
          return "(_ bv" +
                 llvm::toString(vals[0].getData(), 10, /*Signed=*/false) +
                 " " + std::to_string(sort.width) + ")";
      }
    }
    if (auto integer = dyn_cast<IntegerAttr>(attr)) {
      if (sort.isBool)
        return integer.getValue().isZero() ? "false" : "true";
      if (integer.getValue().getBitWidth() <= sort.width)
        return bvConst(integer.getValue().getSExtValue(), sort.width);
    }
    if (auto value = dyn_cast<LIT::LITStructAttr>(attr);
        value && value.getValues().size() == 1 &&
        std::get<0>(value.getValues()[0]).getValue() == "_mlir_value")
      return paramTerm(std::get<1>(value.getValues()[0]), sort, depth + 1);
    if (auto extract = dyn_cast<LIT::StructExtractAttr>(attr);
        extract && extract.getField().getValue() == "_mlir_value")
      return paramTerm(extract.getStructValue(), sort, depth + 1);
    // A dtype's code (`DType.__eq__` compares these).
    if (auto code = dyn_cast<POP::DTypeToUI8Attr>(attr);
        code && !sort.isBool && sort.width == 8)
      if (auto extract = dyn_cast<LIT::StructExtractAttr>(code.getValue());
          extract && extract.getField().getValue() == "_mlir_value") {
        std::string t = dtypeCode(extract.getStructValue());
        if (!t.empty())
          return t;
      }
    // A builtin value as a `SIMD` scalar (`Int(width)` of a `SIMDLength`).
    if (auto cast = dyn_cast<CastFromBuiltinAttr>(attr))
      return paramTerm(cast.getArg(), sort, depth + 1);
    // And back: the builtin value of an `Int` (`SIMDLength(W)` holds
    // `W._mlir_value`) is the `Int`.
    if (auto cast = dyn_cast<CastToBuiltinAttr>(attr); cast && depth < 16)
      if (auto extract = dyn_cast<LIT::StructExtractAttr>(cast.getArg());
          extract && extract.getField().getValue() == "_mlir_value" &&
          !sort.isBool && sort.width == 64 &&
          isScalar(extract.getStructValue().getType()) &&
          sortOf(extract.getStructValue().getType()).width == 64)
        return paramTerm(extract.getStructValue(), sort, depth + 1);
    // "All operands denote the same value": for integers and Booleans,
    // equality; anything else (types, structs) stays unknown.
    if (auto identical = dyn_cast<ParamIdenticalAttr>(attr);
        identical && sort.isBool && identical.getOperands().size() >= 2 &&
        llvm::all_of(identical.getOperands(), [](TypedAttr op) {
          return isScalar(op.getType());
        })) {
      ArrayRef<TypedAttr> ops = identical.getOperands();
      Sort operand = sortOf(ops[0].getType());
      std::string first = paramTerm(ops[0], operand, depth + 1);
      std::string all = "true";
      for (TypedAttr op : ops.drop_front())
        all = "(and " + all + " (= " + first + " " +
              paramTerm(op, operand, depth + 1) + "))";
      return define({true, 1, false}, all);
    }
    if (auto expr = dyn_cast<ParamOperatorAttr>(attr)) {
      ArrayRef<TypedAttr> ops = expr.getOperands();
      auto sub = [&](size_t i, Sort s) {
        return paramTerm(ops[i], s, depth + 1);
      };
      Sort operand = ops.empty() ? sort : sortOf(ops[0].getType());
      auto fold = [&](StringRef smt, Sort s) -> std::string {
        std::string acc = sub(0, s);
        for (size_t i = 1; i < ops.size(); ++i)
          acc = ("(" + smt + " " + acc + " " + sub(i, s) + ")").str();
        return acc;
      };
      switch (expr.getOpcode()) {
      case POC::Add:
        if (!sort.isBool && !ops.empty())
          return define(sort, fold("bvadd", sort));
        break;
      case POC::Mul:
        if (!sort.isBool && !ops.empty())
          return define(sort, fold("bvmul", sort));
        break;
      case POC::And:
      case POC::Or:
      case POC::Xor:
        if (!ops.empty()) {
          bool b = sort.isBool;
          StringRef smt = expr.getOpcode() == POC::And  ? (b ? "and" : "bvand")
                          : expr.getOpcode() == POC::Or ? (b ? "or" : "bvor")
                                                        : (b ? "xor" : "bvxor");
          return define(sort, fold(smt, sort));
        }
        break;
      case POC::EQ:
        if (ops.size() == 2)
          return define({true, 1, false}, "(= " + sub(0, operand) + " " +
                                               sub(1, operand) + ")");
        break;
      case POC::LT:
      case POC::LE:
        if (ops.size() == 2 && !operand.isBool)
          return define({true, 1, false},
                        "(" +
                            std::string(expr.getOpcode() == POC::LT ? "bvslt"
                                                                    : "bvsle") +
                            " " + sub(0, operand) + " " + sub(1, operand) + ")");
        break;
      case POC::Cond:
        if (ops.size() == 3)
          return define(sort, "(ite " + sub(0, {true, 1, false}) + " " +
                                  sub(1, sort) + " " + sub(2, sort) + ")");
        break;
      case POC::Rebind:
        if (ops.size() == 1)
          return sub(0, sort);
        break;
      case POC::Apply:
        if (ops.size() == 1)
          if (std::optional<CalleeName> name = calleeName(ops[0]))
            if (MaybeTerm t = targetPredicate(name->path, sort))
              return *t;
        if (ops.size() == 3 && sort.isBool)
          if (MaybeTerm t = dtypeMembership(ops))
            return *t;
        if (ops.size() == 1 && !sort.isBool && sort.width == 64)
          if (auto symbol = dyn_cast<SymbolConstantAttr>(ops[0]))
            if (MaybeTerm w = simdWidth(symbol))
              return *w;
        if (ops.size() == 1 && !sort.isBool && sort.width == 64)
          if (auto symbol = dyn_cast<SymbolConstantAttr>(ops[0]))
            if (MaybeTerm bytes = dtypeSize(symbol))
              return *bytes;
        // `Wrapper(n)` of an integer wrapper (`GEMVAlgorithm.GEMV_KERNEL`
        // is `GEMVAlgorithm(0)`): `n`.
        if (ops.size() == 2 && !sort.isBool)
          if (auto symbol = dyn_cast<SymbolConstantAttr>(ops[0]))
            if (auto init = dyn_cast_or_null<LIT::FnOp>(
                    symbols.lookupSymbolIn(module, symbol.getSymbol()));
                init && isFieldInit(init))
              return paramTerm(ops[1], sort, depth + 1);
        // `max_or_inf[dtype]()` and `min_or_neg_inf[dtype]()` of an integer
        // dtype (`Int.MAX`, `UInt8.MIN`): its bounds.
        if (ops.size() == 1)
          if (std::optional<CalleeName> name = calleeName(ops[0]);
              name && name->params.size() == 1)
            if (std::optional<Sort> intSort = dtypeSort(name->params[0]);
                intSort && !sort.isBool && intSort->width == sort.width) {
              StringRef path = name->path;
              bool max = path.starts_with("std::utils::numerics::max_or_inf[");
              if (max ||
                  path.starts_with("std::utils::numerics::min_or_neg_inf[")) {
                unsigned w = intSort->width;
                APInt bound = intSort->isSigned
                                  ? (max ? APInt::getSignedMaxValue(w)
                                         : APInt::getSignedMinValue(w))
                                  : (max ? APInt::getMaxValue(w)
                                         : APInt::getMinValue(w));
                return "(_ bv" + llvm::toString(bound, 10, /*Signed=*/false) +
                       " " + std::to_string(w) + ")";
              }
            }
        if (ops.size() >= 2)
          if (std::optional<CalleeName> name = calleeName(ops[0]);
              name && name->params.size() >= 2 &&
              StringRef(name->path).starts_with("std::simd::SIMD::__"))
            if (std::optional<Sort> simdSort = dtypeSort(name->params[0]);
                simdSort && isWidthOne(name->params[1])) {
              SmallVector<std::string> args;
              for (size_t i = 1; i < ops.size(); ++i)
                args.push_back(sub(i, *simdSort));
              StringRef method = StringRef(name->path)
                                     .drop_front(strlen("std::simd::SIMD::"))
                                     .take_until([](char c) { return c == '('; });
              Sort resultSort;
              if (MaybeTerm t = simdOperator(method, *simdSort, args, resultSort);
                  t && resultSort.isBool == sort.isBool &&
                  (sort.isBool || resultSort.width == sort.width))
                return define(sort, *t);
            }
        break;
      default:
        break;
      }
    }
    return parameterValue(printed(attr), sort);
  }

  /// A parameter value without its sugar and its `store_to_mem` wrappers
  /// (how values are passed by reference in parameter expressions).
  static TypedAttr unwrapValue(TypedAttr a) {
    while (true) {
      if (auto mem = dyn_cast<StoreToMemAttr>(a))
        a = mem.getValue();
      else if (auto sugar = dyn_cast<SugarAttr>(a))
        a = sugar.getCanonical();
      else
        return a;
    }
  }

  /// The code of a dtype literal (`{:dtype f32}`).
  static std::optional<uint64_t> dtypeLiteral(TypedAttr a) {
    a = unwrapValue(a);
    if (auto value = dyn_cast<LIT::LITStructAttr>(a);
        value && value.getValues().size() == 1)
      a = std::get<1>(value.getValues()[0]);
    if (auto constant = dyn_cast<DTypeConstantAttr>(a))
      return constant.getDType().getValue();
    return std::nullopt;
  }

  /// `simd_width_of[dtype, target]()` (the `DType` overload), as a
  /// parameter expression or a call (a clause calls it): the target's SIMD
  /// bit width (128 on every GPU target, at most 512 on CPUs, 0 for no
  /// target) divided by the dtype's (8 times its size, a power of two): 0 or
  /// a power of two, at most 256 (assumed). One term per dtype and target
  /// (resolved through call frames), so a clause's call and a body's
  /// `comptime` are the same width. As cases, not `w & (w - 1) == 0`: each
  /// case makes the width a constant; with the bit trick six of gemv's
  /// `vectorize` preconditions ran to the solver's limit.
  MaybeTerm simdWidth(SymbolConstantAttr symbol) {
    std::optional<CalleeName> name = calleeName(symbol);
    ArrayRef<TypedAttr> ps = symbol.getParamValues();
    if (!name || ps.size() < 2 ||
        !StringRef(name->path)
             .starts_with("std::sys::info::simd_width_of[!kgen.target,::DType,"))
      return std::nullopt;
    std::string key = "simd_width_of|" + layoutKey(unwrapValue(ps[1])) + "|" +
                      layoutKey(unwrapValue(ps[0]));
    auto [it, inserted] = simdWidths.try_emplace(key, "");
    if (!inserted)
      return it->second;
    std::string w = declare(Sort{false, 64, true}, "p");
    prelude += "; " + w + ": " + StringRef(key).take_front(300).str() + "\n";
    std::string fact = "(or (= " + w + " " + bvConst(0, 64) + ")";
    for (int64_t v = 1; v <= 256; v *= 2)
      fact += " (= " + w + " " + bvConst(v, 64) + ")";
    facts.push_back(fact + ")");
    finiteDomains[w] = {0, 1, 2, 4, 8, 16, 32, 64, 128, 256};
    it->second = w;
    // On the accelerator's target (`get_gpu_target()`): every GPU target
    // the stdlib lists has 128-bit vectors, so the width is 16 over the
    // dtype's size in bytes, and 0 only for a 256-bit dtype.
    if (std::string target = printed(ps[0]);
        StringRef(target).contains("accelerator_arch"))
      if (MaybeTerm bytes = dtypeBytes(ps[1])) {
        std::string cases = "(or";
        for (int64_t b = 1; b <= 32; b *= 2)
          cases += " (and (= " + *bytes + " " + bvConst(b, 64) + ") (= " + w +
                   " " + bvConst(16 / b, 64) + "))";
        facts.push_back(cases + ")");
        finiteDomains[w] = {0, 1, 2, 4, 8, 16};
      }
    return w;
  }

  /// `size_of[dtype]()` (the `DType` overload) as a parameter expression or
  /// a call (a clause calls it): the dtype's size in bytes, on every target.
  /// A literal's is in its name (`f32`, `bf16`, `si8`; `bool` is one byte,
  /// `index` eight); any other dtype's is a power of two up to 32 (`uint256`
  /// is the widest), one term per dtype expression.
  MaybeTerm dtypeSize(SymbolConstantAttr symbol) {
    std::optional<CalleeName> name = calleeName(symbol);
    ArrayRef<TypedAttr> ps = symbol.getParamValues();
    if (!name || ps.size() < 2 ||
        !StringRef(name->path)
             .starts_with("std::sys::info::size_of[!kgen.target,::DType,"))
      return std::nullopt;
    return dtypeBytes(ps[1]);
  }

  /// The size in bytes of the dtype a parameter value names (see
  /// `dtypeSize`).
  MaybeTerm dtypeBytes(TypedAttr param) {
    TypedAttr dtype = unwrapValue(resolveParam(unwrapValue(param)));
    std::string text = printed(dtype);
    // `{:dtype f32}` or `{_mlir_value: dtype = f32}`.
    static llvm::Regex literalRe("dtype (= )?([a-z]+)([0-9]*)[a-z0-9]*\\}");
    SmallVector<StringRef> m;
    if (literalRe.match(text, &m)) {
      unsigned bits;
      if (m[2] == "bool")
        return bvConst(1, 64);
      if (m[2] == "index" || m[2] == "uindex")
        return bvConst(8, 64);
      if (!m[3].getAsInteger(10, bits) && bits >= 8 && bits <= 256 &&
          llvm::isPowerOf2_32(bits))
        return bvConst(bits / 8, 64);
      return std::nullopt;
    }
    std::string key = "size_of|" + layoutKey(unwrapValue(param));
    auto [it, inserted] = simdWidths.try_emplace(key, "");
    if (!inserted)
      return it->second;
    std::string bytes = declare(Sort{false, 64, true}, "p");
    prelude +=
        "; " + bytes + ": " + StringRef(key).take_front(300).str() + "\n";
    std::string fact = "(or";
    for (int64_t v = 1; v <= 32; v *= 2)
      fact += " (= " + bytes + " " + bvConst(v, 64) + ")";
    facts.push_back(fact + ")");
    it->second = bytes;
    return bytes;
  }

  /// `dtype in (DType.float32, DType.bfloat16, ...)` as a parameter
  /// expression (`apply` of `Tuple.__contains__` to the tuple and the
  /// dtype): `dtype` is one of them. Only for tuples of dtype literals; the
  /// dtypes are compared as `DType.__eq__` does, by their `ui8` codes.
  MaybeTerm dtypeMembership(ArrayRef<TypedAttr> ops) {
    TypedAttr callee = ops[0];
    while (auto expr = dyn_cast<ParamOperatorAttr>(callee))
      if (expr.getOpcode() == POC::Rebind && expr.getOperands().size() == 1)
        callee = expr.getOperands()[0];
      else
        break;
    std::optional<CalleeName> name = calleeName(callee);
    if (!name ||
        !StringRef(name->path)
             .starts_with("std::builtin::tuple::Tuple::__contains__[") ||
        name->params.empty())
      return std::nullopt;
    // The tuple is built from a pack (`VariadicPack.__init__` applied to
    // the elements): every element must be a dtype literal.
    SmallVector<uint64_t> codes;
    bool found = false, literal = true;
    Attribute tuple = ops[1];
    tuple.walk([&](ParamOperatorAttr expr) {
      if (found || expr.getOpcode() != POC::Apply ||
          expr.getOperands().size() != 2)
        return;
      std::optional<CalleeName> init = calleeName(expr.getOperands()[0]);
      if (!init ||
          !StringRef(init->path)
               .starts_with("std::builtin::variadics::VariadicPack::__init__("))
        return;
      found = true;
      auto pack = dyn_cast<LIT::RefPackAttr>(expr.getOperands()[1]);
      if (!pack) {
        literal = false;
        return;
      }
      for (TypedAttr element : pack.getValues())
        if (std::optional<uint64_t> c = dtypeLiteral(element))
          codes.push_back(*c);
        else
          literal = false;
    });
    if (!found || !literal || codes.empty())
      return std::nullopt;
    std::string v = dtypeCode(ops[2]);
    if (v.empty())
      return std::nullopt;
    std::string any = "(or";
    for (uint64_t c : codes)
      any += " (= " + v + " " + bvConst(c, 8) + ")";
    return define({true, 1, false}, any + ")");
  }

  /// The `ui8` code of a `DType` value, as `DType.__eq__` compares dtypes:
  /// a literal's code, otherwise one unknown per dtype, the same for a
  /// callee's dtype parameter and the caller's value for it. Empty if the
  /// value is not a dtype.
  std::string dtypeCode(TypedAttr value) {
    ParamFrame *scope = nullptr;
    value = unwrapValue(resolveParam(unwrapValue(value), &scope));
    if (std::optional<uint64_t> c = dtypeLiteral(value))
      return bvConst(*c, 8);
    auto type = dyn_cast<LIT::StructType>(value.getType());
    if (!type || printed(type.getSymbol()) != "@std::@builtin::@dtype::@DType")
      return "";
    MLIRContext *ctx = value.getContext();
    TypedAttr code = POP::DTypeToUI8Attr::get(
        ctx,
        LIT::StructExtractAttr::get(value, StringAttr::get(ctx, "_mlir_value"),
                                    DTypeType::get(ctx)));
    ParamFrame *saved = params;
    params = scope;
    std::string term = parameterValue(printed(code), Sort{false, 8, false});
    params = saved;
    return term;
  }

  /// A call of a local closure: its body, walked here with the caller's
  /// state (it reads and writes the caller's variables it captures, by
  /// reference), its arguments the call's operands, its own parameters
  /// (`f[4]()`) the call's values. What its returns and raises leave is
  /// merged; its results are unknown.
  bool inlineClosure(LIT::CallOp call, State &state) {
    ArrayRef<TypedAttr> bound;
    Attribute ref = closureCallee(call.getCallee(), bound);
    auto it = closures.find(ref);
    if (it == closures.end() || closureExits.size() >= 4)
      return false;
    // Its implicit origin parameters come last; the call binds them apart.
    ArrayRef<ParamDeclAttr> decls = it->second.getParams();
    decls = decls.drop_back(
        std::min(decls.size(), call.getImplicitOrigins().size()));
    Block &entry = it->second.getFunctionBody().front();
    if (entry.getNumArguments() != call.getNumOperands() ||
        (!bound.empty() && bound.size() != decls.size()))
      return false;
    for (auto [arg, operand] : llvm::zip(entry.getArguments(), call.getOperands())) {
      if (isa<LIT::RefType>(arg.getType())) {
        if (std::optional<Loc> loc = placeOf(operand))
          refArgs[arg] = *loc;
      } else {
        values[arg] = term(operand, state);
      }
    }
    ParamFrame frame;
    frame.parent = params;
    frame.transparent = true;
    frame.scope = printed(call.getCallee());
    for (auto [decl, value] : llvm::zip(decls, bound))
      frame.values[decl.getName()] = value;
    ParamFrame *saved = params;
    if (!bound.empty()) {
      params = &frame;
      // Its parameter expressions are this call's: not what an earlier call
      // of it left.
      it->second->walk(
          [&](ParamConstantOp cst) { values.erase(cst->getResult(0)); });
    }
    SmallVector<State> exits;
    closureExits.push_back(&exits);
    State body = state;
    body.yields.clear();
    walkBlock(entry, body);
    closureExits.pop_back();
    params = saved;
    if (body.alive)
      exits.push_back(std::move(body));
    State joined = merge(exits);
    joined.yields = state.yields;
    state = std::move(joined);
    setResultsUnknown(call);
    return true;
  }

  void evalCall(LIT::CallOp call, State &state) {
    if (inlineClosure(call, state))
      return;
    std::optional<CalleeName> name = calleeName(call.getCallee());
    LIT::FnOp callee = lookup(call);
    ParamFrame frame = paramFrame(call, callee);
    // A struct's wrapper for an inherited default method has the default's
    // contract, its parameters bound by the wrapper's forwarding call.
    LIT::FnOp contract = callee;
    ParamFrame *contractFrame = &frame;
    ParamFrame forwarded;
    if (LIT::CallOp inner = forwardedCall(callee))
      if (LIT::FnOp target = lookup(inner)) {
        params = &frame;
        forwarded = paramFrame(inner, target);
        params = frame.parent;
        contract = target;
        contractFrame = &forwarded;
      }
    bool hasBody = contract && !contract.getFunctionBody().empty();
    // A call to a function with preconditions: they are obligations here.
    if (hasBody && !inContract) {
      for (Operation &calleeOp : contract.getFunctionBody().front()) {
        auto req = dyn_cast<RequiresOp>(&calleeOp);
        if (!req)
          continue;
        Obligation ob{state.pc, "false", call.getLoc(), req.getLoc(),
                      displayName(contract)};
        params = contractFrame;
        MaybeTerm cond =
            instantiate(req.getBody(), req.getArgs(), state, call.getOperands(),
                        contract, nullptr, false);
        params = frame.parent;
        if (cond) {
          ob.cond = *cond;
          noteCondition(ob.cond);
        } else {
          ob.analyzed = false;
        }
        obligations.push_back(ob);
      }
    }
    if (name && !inContract) {
      checkLaunch(call, *name, state);
      checkDereference(call, *name, callee, state);
      checkPartialTiles(call, *name, state);
    }
    if (name && evalReversedRange(call, *name, state))
      return;
    if (name && evalDim(call, *name, state))
      return;
    if (name && evalDivmod(call, *name, state))
      return;
    if (name && evalUnsignedDivision(call, *name, state))
      return;
    if (name && evalMinMax(call, *name, state))
      return;
    if (name && evalCeildiv(call, *name, state))
      return;
    if (callee && evalIntWrapperCall(call, callee, state))
      return;
    if (name && evalPointer(call, *name, state))
      return;
    if (name && evalElementWrite(call, *name, state))
      return;
    if (name && evalTupleLiteral(call, *name, state))
      return;
    if (name && evalListFromIterator(call, *name, state))
      return;
    if (name && evalTileTensor(call, *name, state))
      return;
    if (name && evalWarpShuffle(call, *name, state))
      return;
    if (name && evalGpuId(call, *name))
      return;
    if (auto symbol = dyn_cast<SymbolConstantAttr>(call.getCallee());
        symbol && call.getNumOperands() == 0 && call->getNumResults() == 1)
      if (sortOf(call->getResult(0).getType()).width == 64) {
        MaybeTerm w = simdWidth(symbol);
        if (!w)
          w = dtypeSize(symbol);
        if (w) {
          values[call->getResult(0)] = *w;
          return;
        }
      }
    if (name && callee && evalIteration(call, *name, callee, state))
      return;
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
    // Origins first: a mutable element reference argument (`xs[0]` in
    // `xs[0].append(v)`, given origin `xs["element"]`) then writes its new
    // value back into the collection as the origin left it.
    // Keep the printed list alive while its elements are used.
    std::string origins = printed(call.getImplicitOriginsAttr());
    // The origin of a mutable reference argument whose place is known is
    // covered by that argument's own havoc below: what the callee reaches
    // through a reference to one element (or field) is that element, not
    // its neighbours.
    std::set<std::string> covered;
    for (Value operand : call.getOperands())
      if (auto ref = dyn_cast<LIT::RefType>(operand.getType());
          ref && !ref.isMutableKnown(false) && placeOf(operand))
        covered.insert(printed(ref.getOrigin()));
    bool confined = confinedWrites(call);
    for (StringRef origin : topLevelElements(origins))
      if (!origin.ends_with(": !lit.origin<false>") &&
          !covered.count(origin.str()))
        havocOrigin(origin, state, confined);
    for (Value operand : call.getOperands()) {
      auto ref = dyn_cast<LIT::RefType>(operand.getType());
      if (!ref || ref.isMutableKnown(false))
        continue;
      if (std::optional<Loc> loc = placeOf(operand))
        havoc(*loc, state, ref.getElementType());
      else
        havocOrigin(printed(ref.getOrigin()), state, confined);
    }
    evalTraitQuery(call, callee, state);
    if (hasBody) {
      params = contractFrame;
      assumeEnsures(call, contract, before, state);
      params = frame.parent;
    }
    assumeAssertion(call, name, state);
    // Their operands as they were before the call (a reference argument it
    // may write is unknown after it).
    if (name) {
      recordTile(call, *name, before);
      recordRowMajor(call, *name, before);
      checkReshape(call, *name, before);
    }
    assumeListLiteral(call, name, before, state);
    if (callee)
      assumeCopy(call, callee, before, state);
    // `String(literal)`: as long as the literal. Its own contract cannot say
    // so: naming `String.byte_length()` there makes the declaration depend
    // on itself (through a `String` default argument in `SIMD.cast`).
    if (name &&
        StringRef(name->path)
            .starts_with("std::collections::string::string::String::__init__["
                         "!kgen.string](::StringLiteral[") &&
        call.getNumOperands() == 2 &&
        isa<LIT::RefType>(call.getOperands()[1].getType()))
      if (auto symbol = dyn_cast<SymbolConstantAttr>(call.getCallee());
          symbol && symbol.getParamValues().size() == 1)
        if (std::optional<int64_t> n =
                stringLength(resolveParam(symbol.getParamValues()[0]))) {
          std::string string = valueThrough(call.getOperands()[1], state);
          state.pc = define({true, 1, false},
                            "(and " + state.pc + " (= " + lenOf(string) + " " +
                                bvConst(*n, 64) + "))",
                            "r");
        }
  }

  /// A call of a trait method that only reads its arguments and returns an
  /// integer or a Boolean (`t.count()` on a generic `t`, or a struct's own
  /// implementation): a function of the arguments' values, one per method
  /// and type, so that equal arguments give equal results and the trait's
  /// clauses about it connect with the implementation's. Assumption: an
  /// implementation's result depends only on the values it is given, as
  /// `len(x)` is assumed to.
  void evalTraitQuery(LIT::CallOp call, LIT::FnOp callee, State &state) {
    if (call->getNumResults() != 1 || !isScalar(call->getResult(0).getType()) ||
        StringRef(printed(call.getCallee().getType())).contains(" throws"))
      return;
    std::string key;
    if (auto witness = dyn_cast<GetWitnessAttr>(call.getCallee()))
      key = selfImpl(witness)
                ? implKey(witness.getTraitSymbol(), witness.getWitnessName(),
                          selfStruct)
                : implKey(witness.getTraitSymbol(), witness.getWitnessName(),
                          printed(resolveParam(witness.getTypeValue())));
    else if (auto it = callee ? impls.keyOf.find(callee) : impls.keyOf.end();
             it != impls.keyOf.end())
      key = it->second;
    else
      return;
    SmallVector<std::string> args;
    for (Value operand : call.getOperands()) {
      auto ref = dyn_cast<LIT::RefType>(operand.getType());
      if (ref && !ref.isMutableKnown(false))
        return;
      args.push_back(ref ? valueThrough(operand, state) : term(operand, state));
      key += "|" + sortOfTerm(args.back()).str();
    }
    Sort sort = sortOf(call->getResult(0).getType());
    // `Sized.__len__` is `len`: `x.__len__()` and `len(x)` agree.
    if (StringRef(key).starts_with(
            "#kgen.trait_symbol<@std::@builtin::@len::@Sized>|__len__(") &&
        args.size() == 1 && sort.width == 64 && !sort.isBool) {
      values[call->getResult(0)] = define(sort, lenOf(args[0]));
      return;
    }
    key += "|" + sort.str();
    auto [it, inserted] = traitQueries.try_emplace(key, "");
    if (inserted) {
      it->second = "w" + std::to_string(counter++);
      prelude += "(declare-fun " + it->second + " (";
      for (const std::string &arg : args)
        prelude += sortOfTerm(arg).str() + " ";
      prelude += ") " + sort.str() + ")\n; " + it->second + ": " +
                 StringRef(key).take_front(300).str() + "\n";
    }
    std::string application = "(" + it->second;
    for (const std::string &arg : args)
      application += " " + arg;
    values[call->getResult(0)] =
        args.empty() ? it->second : define(sort, application + ")");
  }

  /// After `assert_true(c)`, `assert_false(c)`, `assert_equal(a, b)` or
  /// `assert_not_equal(a, b)` from `std.testing`, what it checked holds
  /// where it did not raise (each raises exactly when its check fails).
  /// Only for `Bool` conditions and for `Int`, `Bool` and integer scalar
  /// operands, whose equality is the terms': a type's own `__eq__` is not
  /// equality of its value.
  void assumeAssertion(LIT::CallOp call, const std::optional<CalleeName> &name,
                       State &state) {
    const char *prefix = "std::testing::testing::assert_";
    auto symbol = dyn_cast<SymbolConstantAttr>(call.getCallee());
    if (!name || !StringRef(name->path).starts_with(prefix) || !symbol ||
        symbol.getParamValues().empty() || call->getNumResults() < 1 ||
        !sortOf(call->getResult(0).getType()).isBool)
      return;
    StringRef kind =
        StringRef(name->path).drop_front(strlen(prefix)).take_until([](char c) {
          return c == '[' || c == '(';
        });
    TypedAttr type = resolveParam(symbol.getParamValues()[0]);
    auto value = [&](Value v) {
      return isa<LIT::RefType>(v.getType()) ? valueThrough(v, state)
                                            : term(v, state);
    };
    std::string fact;
    ValueRange ops = call.getOperands();
    if ((kind == "true" || kind == "false") && isBoolType(type) &&
        ops.size() >= 1) {
      std::string v = value(ops[0]);
      if (!sortOfTerm(v).isBool)
        return;
      fact = kind == "true" ? v : "(not " + v + ")";
    } else if ((kind == "equal" || kind == "not_equal") &&
               (isIntType(type) || isBoolType(type) || isIntegerScalar(type)) &&
               ops.size() >= 2) {
      std::string a = value(ops[0]), b = value(ops[1]);
      // A place whose value is unknown loads as a 64-bit term whatever its
      // type; terms of different sorts say nothing to each other.
      Sort sa = sortOfTerm(a), sb = sortOfTerm(b);
      if (sa.isBool != sb.isBool || sa.width != sb.width)
        return;
      std::string eq = "(= " + a + " " + b + ")";
      fact = kind == "equal" ? eq : "(not " + eq + ")";
    }
    if (fact.empty())
      return;
    state.pc = define({true, 1, false},
                      "(and " + state.pc + " (=> (not " +
                          values[call->getResult(0)] + ") " + fact + "))",
                      "r");
  }

  /// `x.copy()` through `Copyable`'s default (`Self(copy=self)`, a call
  /// through the trait the pass cannot follow): what the struct's own copy
  /// constructor states. Its arguments (`copy`, `out self`) are laid out as
  /// the wrapper's (`self`, `out result`).
  void assumeCopy(LIT::CallOp call, LIT::FnOp callee, State &before,
                  State &state) {
    auto fallback = callee->getAttrOfType<SymbolRefAttr>("defaultFnRef");
    if (!fallback ||
        !StringRef(printed(fallback))
             .starts_with("@std::@traits::@copyable::@Copyable::@\"copy("))
      return;
    auto parent = callee->getParentOfType<LIT::StructDeclOp>();
    if (!parent)
      return;
    for (auto ctor : parent.getBody()->getOps<LIT::FnOp>()) {
      if (!ctor.getSymName() ||
          !ctor.getSymName()->starts_with("__init__(copy:") ||
          ctor.getFunctionBody().empty() ||
          ctor.getFunctionBody().front().getNumArguments() !=
              call.getNumOperands())
        continue;
      ParamFrame frame = paramFrame(call, ctor);
      params = &frame;
      assumeEnsures(call, ctor, before, state);
      params = frame.parent;
      return;
    }
  }

  /// A list or array literal (`[a, b, c]`): its elements are the values it is given,
  /// as they were before the call (the stdlib moves them in, in order). Its
  /// contract states only its length; a generic element type has no `==`
  /// for a contract to state more with.
  void assumeListLiteral(LIT::CallOp call,
                         const std::optional<CalleeName> &name, State &before,
                         State &state) {
    if (!name ||
        !(StringRef(name->path)
              .starts_with("std::collections::list::List::__init__[") ||
          StringRef(name->path)
              .starts_with("std::collections::array::Array::__init__[")) ||
        !StringRef(name->path).contains("__list_literal__") ||
        call.getNumOperands() < 2)
      return;
    std::optional<Loc> packPlace = placeOf(call.getOperands()[0]);
    std::optional<Loc> out = placeOf(call.getOperands().back());
    if (!packPlace || !out)
      return;
    auto it = packRefs.find(load(*packPlace, before, placeType(*packPlace)));
    if (it == packRefs.end())
      return;
    std::string list = load(*out, state, placeType(*out));
    std::string holds = "true";
    for (auto [k, ref] : llvm::enumerate(it->second)) {
      std::optional<Loc> place = placeOf(ref);
      if (!place)
        return;
      std::string value = load(*place, before, placeType(*place));
      holds = "(and " + holds +
              " (= " + elem(list, bvConst(k, 64), sortOfTerm(value)) + " " +
              value + "))";
    }
    state.pc =
        define({true, 1, false}, "(and " + state.pc + " " + holds + ")", "r");
  }

  /// Whether a type parameter's value is `Bool`.
  static bool isBoolType(TypedAttr attr) {
    if (auto sugar = dyn_cast<SugarAttr>(attr))
      attr = sugar.getCanonical();
    auto type = dyn_cast<TypeParamAttr>(attr);
    return type && printed(type.getTypeValue()) ==
                       "!lit.struct<@std::@builtin::@bool::@Bool>";
  }

  /// Whether a type parameter's value is a width-1 integer `SIMD`
  /// (`UInt8`, `Int32`, ...).
  static bool isIntegerScalar(TypedAttr attr) {
    if (auto sugar = dyn_cast<SugarAttr>(attr))
      attr = sugar.getCanonical();
    auto type = dyn_cast<TypeParamAttr>(attr);
    if (!type)
      return false;
    std::string text = printed(type.getTypeValue());
    return StringRef(text).starts_with("!lit.struct<@std::@simd::@SIMD<") &&
           isWidthOne(text) && dtypeSort(text).has_value();
  }

  /// The parameters `call` binds for `callee`: the callee's struct's
  /// parameters, then its own but its implicit origins, in order. None when
  /// they do not line up.
  ParamFrame paramFrame(LIT::CallOp call, LIT::FnOp callee) {
    ParamFrame frame;
    frame.parent = params;
    // A trait method: its trait's `Self` is the type the call is made on.
    if (auto witness = dyn_cast<GetWitnessAttr>(call.getCallee())) {
      auto trait = callee ? callee->getParentOfType<LIT::TraitDeclOp>()
                          : LIT::TraitDeclOp();
      if (!trait)
        return frame;
      frame.scope = printed(witness);
      for (ParamDeclAttr decl : trait.getParams())
        if (decl.getName().getValue().starts_with("_Self"))
          frame.values[decl.getName()] = witness.getTypeValue();
      return frame;
    }
    auto symbol = dyn_cast<SymbolConstantAttr>(call.getCallee());
    if (!callee || !symbol)
      return frame;
    return paramFrame(symbol, callee, call.getImplicitOrigins().size(),
                      std::move(frame));
  }

  /// The parameters `symbol` binds of `callee`, whose last `implicit`
  /// parameters (implicit origins) it does not.
  ParamFrame paramFrame(SymbolConstantAttr symbol, LIT::FnOp callee,
                        size_t implicit, ParamFrame frame) {
    frame.scope = printed(symbol.getSymbol());
    SmallVector<ParamDeclAttr> decls;
    if (auto parent = callee->getParentOfType<LIT::StructDeclOp>())
      llvm::append_range(decls, parent.getParams());
    else if (auto trait = callee->getParentOfType<LIT::TraitDeclOp>())
      llvm::append_range(decls, trait.getParams());
    // The callee's implicit origin parameters come last; the call binds
    // them apart (`lit.call @f[mut *"x"]`).
    ArrayRef<ParamDeclAttr> own = callee.getParams();
    if (implicit <= own.size())
      llvm::append_range(decls, own.drop_back(implicit));
    ArrayRef<TypedAttr> values = symbol.getParamValues();
    if (decls.size() == values.size())
      for (auto [decl, value] : llvm::zip(decls, values))
        frame.values[decl.getName()] = value;
    return frame;
  }

  /// After a call, the callee's postcondition: its arguments as they are
  /// now, its `old`s as they were before the call. A raising callee's holds
  /// only where it did not raise (its first result is the raised flag).
  void assumeEnsures(LIT::CallOp call, LIT::FnOp callee, State &before,
                     State &state) {
    SmallVector<EnsuresOp> clauses = postconditionClauses(callee);
    if (clauses.empty())
      return;
    callResult = call->getNumResults() == 1 ? call->getResult(0) : Value();
    std::string holds = "true";
    for (EnsuresOp ensures : clauses)
      if (MaybeTerm cond = instantiate(ensures.getBody(), ensures.getArgs(),
                                       state, call.getOperands(), callee,
                                       &before, /*assumed=*/true))
        holds = "(and " + holds + " " + *cond + ")";
    callResult = {};
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

  /// Element access that returns a reference to an element (`List`'s,
  /// `Deque`'s and `Array`'s `__getitem__`, `LinkedList.get_nth`): the
  /// collection's place and an index. Reading it reads `elem(xs, index)`;
  /// writing it makes a new collection with the same length and the other
  /// elements unchanged. These take `ref self` but do not write it, so the
  /// collection is kept.
  bool evalElementAccess(LIT::CallOp call, const CalleeName &name,
                         State &state) {
    StringRef path = name.path;
    if (!(path.starts_with("std::collections::list::List::__getitem__[") ||
          path.starts_with("std::collections::list::List::unsafe_get") ||
          path.starts_with("std::collections::deque::Deque::__getitem__") ||
          path.starts_with("std::collections::array::Array::__getitem__") ||
          path.starts_with(
              "std::collections::linked_list::LinkedList::get_nth[")) ||
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
    } else if (call.getNumOperands() == 2 &&
               isa<LIT::RefType>(call.getOperands()[1].getType())) {
      // `get_nth[I: Indexer]`: an `Int` index, by reference.
      auto ref = cast<LIT::RefType>(call.getOperands()[1].getType());
      if (isInt(ref.getElementType()))
        index = valueThrough(call.getOperands()[1], state);
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

  /// Iterating a `List`, `Span`, `Array` or `Deque` forward, directly or
  /// through `enumerate`, as their iterators define it: an iterator holds a
  /// cursor (`/index`, from 0) and the collection's length when it was made
  /// (`/length`); `__next__` raises when the cursor reaches the length, and
  /// otherwise advances it (the element it yields is unknown). `enumerate`
  /// wraps one (`/_inner`) with a count (`/_count`, from `start`), and
  /// yields a tuple whose field `/0` is the count; a tuple's
  /// `__getitem_param__[k]` refers to its field `/k`.
  bool evalIteration(LIT::CallOp call, const CalleeName &name, LIT::FnOp callee,
                     State &state) {
    StringRef path = name.path;
    Sort sort{false, 64, true};
    static const char *collections[] = {
        "std::collections::list::List::__iter__[",
        "std::collections::span::Span::__iter__[",
        "std::collections::array::Array::__iter__[",
        "std::collections::deque::Deque::__iter__[",
        // Owned iteration (`for x in xs^`, a literal).
        "std::collections::list::List::__iter__(::List[$0]$)",
        "std::collections::array::Array::__iter__(::Array[$0, $1]$)",
        // A dictionary's keys, values and entries: its iterators count the
        // entries they have seen up to its length, skipping removed ones.
        "std::collections::dict::Dict::__iter__[",
        "std::collections::dict::Dict::keys[",
        "std::collections::dict::Dict::values[",
        "std::collections::dict::Dict::items[",
        "std::collections::dict::Dict::__reversed__[",
        // A linked list's nodes, from the head or (reversed) the tail.
        "std::collections::linked_list::LinkedList::__iter__[",
        "std::collections::linked_list::LinkedList::__iter__(::LinkedList[$0]$)",
        "std::collections::linked_list::LinkedList::__reversed__["};
    static const char *collectionTypes[] = {
        "@std::@collections::@list::@List<",
        "@std::@collections::@span::@Span<",
        "@std::@collections::@array::@Array<",
        "@std::@collections::@deque::@Deque<",
        "@std::@collections::@linked_list::@LinkedList<"};
    static const char *iterators[] = {
        "std::collections::list::_ListIter::",
        "std::collections::span::_SpanIter::",
        "std::collections::array::_ArrayIter::",
        "std::collections::deque::_DequeIter::",
        "std::collections::list::_ListIterOwned::",
        "std::collections::array::_ArrayIterOwned::",
        "std::collections::dict::_DictKeyIter::",
        "std::collections::dict::_DictValueIter::",
        "std::collections::dict::_DictEntryIter::",
        "std::collections::linked_list::_LinkedListIter::",
        "std::collections::linked_list::_LinkedListIterOwned::"};
    // The length of the collection at `v`, a reference.
    auto lengthOf = [&](Value v) -> MaybeTerm {
      std::optional<Loc> loc = placeOf(v);
      if (!loc)
        return std::nullopt;
      if (MaybeTerm n = loc->path.empty() ? arrayLength(*loc) : std::nullopt)
        return n;
      return define(sort, lenOf(load(*loc, state, placeType(*loc))));
    };
    // The result of a call: its register result or its `out` slot.
    auto produce = [&](std::map<std::string, std::string> fields) {
      if (call->getNumResults() == 1 &&
          !sortOf(call->getResult(0).getType()).isBool &&
          printed(call->getResult(0).getType()) != "!kgen.none") {
        values[call->getResult(0)] = declare({false, 64, false}, "g");
        records[call->getResult(0)] = std::move(fields);
        return true;
      }
      std::optional<Loc> out = placeOf(call.getOperands().back());
      if (!out)
        return false;
      storeFields(*out, state,
                  SmallVector<std::pair<std::string, std::string>>(
                      fields.begin(), fields.end()));
      setResultsUnknown(call);
      return true;
    };
    for (const char *prefix : collections)
      if (path.starts_with(prefix) && call.getNumOperands() >= 1 &&
          isa<LIT::RefType>(call.getOperands()[0].getType()) &&
          (call->getNumResults() == 1 || call.getNumOperands() == 2))
        if (MaybeTerm n = lengthOf(call.getOperands()[0])) {
          // The iterator's elements are the collection's (see `__next__`);
          // a dictionary's keys and values are not indexed, so stay unknown.
          if (call.getNumOperands() == 2 &&
              !path.starts_with("std::collections::dict::"))
            if (std::optional<Loc> src = placeOf(call.getOperands()[0]))
              if (std::optional<Loc> out = placeOf(call.getOperands()[1]))
                iterSources[out->root] = *src;
          return produce({{"/index", bvConst(0, 64)}, {"/length", *n}});
        }
    // `reversed(d)`, `reversed(d.values())`, `reversed(d.items())`: a new
    // count over the same dictionary, from no entries seen.
    if (path.starts_with("std::builtin::reversed::reversed[") &&
        call.getNumOperands() >= 1 &&
        StringRef(printed(call.getOperands()[0].getType()))
            .contains("@std::@collections::@dict::@"))
      if (std::optional<Loc> arg = placeOf(call.getOperands()[0])) {
        Loc length{arg->root, arg->path + "/length"};
        MaybeTerm n = state.env.count(length)
                          ? MaybeTerm(load(length, state, sort))
                          : StringRef(printed(call.getOperands()[0].getType()))
                                    .contains("@std::@collections::@dict::@Dict<")
                                ? lengthOf(call.getOperands()[0])
                                : std::nullopt;
        if (n)
          return produce({{"/index", bvConst(0, 64)}, {"/length", *n}});
      }
    // `enumerate(xs, start=)` of such a collection.
    if (path.starts_with("std::iter::__init__::enumerate[") &&
        call.getNumOperands() >= 2 && !name.params.empty() &&
        llvm::any_of(collectionTypes, [&](const char *type) {
          return StringRef(name.params[0]).contains(type);
        }))
      if (MaybeTerm n = lengthOf(call.getOperands()[0]))
        return produce({{"/_inner/index", bvConst(0, 64)},
                        {"/_inner/length", *n},
                        {"/_count", term(call.getOperands()[1], state)}});
    // An iterator's next element, forward only.
    bool isIterator = llvm::any_of(iterators, [&](const char *prefix) {
      return path.starts_with(prefix);
    });
    bool isEnumerate = path.starts_with("std::iter::__init__::_Enumerate::");
    if (!isIterator && !isEnumerate)
      return tupleField(call, name, state);
    // The method, after its struct's name (its mangled signature may itself
    // contain `::`).
    StringRef method = path;
    for (const char *prefix : iterators)
      if (path.starts_with(prefix))
        method = path.drop_front(strlen(prefix));
    if (isEnumerate)
      method = path.drop_front(strlen("std::iter::__init__::_Enumerate::"));
    std::optional<Loc> self =
        call.getNumOperands() ? placeOf(call.getOperands()[0]) : std::nullopt;
    if (!self)
      return false;
    std::string inner = isEnumerate ? "/_inner" : "";
    Loc index{self->root, self->path + inner + "/index"},
        length{self->root, self->path + inner + "/length"},
        count{self->root, self->path + "/_count"};
    // An owned iterator (`_ListIterOwned`) only goes forward; a dictionary's
    // counts the entries it has seen in either direction.
    bool owned = path.contains("IterOwned::");
    bool dict = path.starts_with("std::collections::dict::");
    // A linked list's walks its nodes, from the tail when reversed: it
    // yields `len(l)` elements as long as its nodes are as many as its
    // size, which its methods keep (an assumption about the stdlib).
    bool linked = path.starts_with("std::collections::linked_list::");
    bool backward = false;
    if (isIterator && !owned && !dict) {
      ParamFrame frame = paramFrame(call, callee);
      auto forward = frame.values.find("forward");
      if (forward == frame.values.end())
        return false;
      backward = !StringRef(printed(forward->second)).contains("true");
      if (backward && !linked)
        return false;
    } else if (isEnumerate && !llvm::any_of(iterators, [&](const char *prefix) {
                 StringRef type = StringRef(prefix).drop_back(2);
                 return !name.params.empty() &&
                        StringRef(name.params[0])
                            .contains(type.rsplit("::").second);
               })) {
      return false;
    }
    if (method.starts_with("__iter__")) {
      std::map<std::string, std::string> fields = {
          {inner + "/index", load(index, state, sort)},
          {inner + "/length", load(length, state, sort)}};
      if (isEnumerate)
        fields["/_count"] = load(count, state, sort);
      return produce(std::move(fields));
    }
    if (!method.starts_with("__next__") || call.getNumOperands() != 3 ||
        call->getNumResults() != 1)
      return false;
    std::optional<Loc> error = placeOf(call.getOperands()[1]);
    std::optional<Loc> out = placeOf(call.getOperands()[2]);
    if (!error || !out)
      return false;
    std::string i = load(index, state, sort), n = load(length, state, sort);
    std::string raised =
        define({true, 1, false}, "(bvsge " + i + " " + n + ")");
    noteCondition(raised);
    auto step = [&](const Loc &loc, const std::string &value) {
      store(loc, state,
            define(sort, "(ite " + raised + " " + value + " (bvadd " + value +
                             " " + bvConst(1, 64) + "))"));
    };
    std::string c = isEnumerate ? load(count, state, sort) : "";
    step(index, i);
    state.env[length] = n;
    auto source = iterSources.find(self->root);
    // The element at the cursor, counted from the end when going backward.
    std::string position =
        backward ? define(sort, "(bvsub (bvsub " + n + " " + bvConst(1, 64) +
                                    ") " + i + ")")
                 : i;
    if (isEnumerate) {
      step(count, c);
      storeFields(*out, state, {{"/0", c}});
    } else if (source != iterSources.end() && owned) {
      // An owned iterator yields element `i` itself...
      std::string collection =
          load(source->second, state, placeType(source->second));
      store(*out, state, elem(collection, position, sortOf(placeType(*out))));
    } else if (auto ref = dyn_cast<LIT::RefType>(placeType(*out));
               ref && source != iterSources.end()) {
      // ... a borrowing one a reference to it, the element's place (named
      // by the call, with the element's type).
      havoc(*out, state, placeType(*out));
      Value element = call->getResult(0);
      elements[element] = {source->second, position};
      elementTypes[element] = ref.getElementType();
      state.refs[*out] = Loc{element, ""};
    } else {
      havoc(*out, state, placeType(*out));
    }
    havoc(*error, state, placeType(*error));
    values[call->getResult(0)] = raised;
    return true;
  }

  /// `List(iterator)` of an iterator the pass models: as long as the
  /// iterator has elements left (`length - index`). The elements are
  /// unknown.
  bool evalListFromIterator(LIT::CallOp call, const CalleeName &name,
                            State &state) {
    if (!StringRef(name.path).starts_with(
            "std::collections::list::List::__init__[::Iterable & ") ||
        call.getNumOperands() != 2)
      return false;
    std::optional<Loc> it = placeOf(call.getOperands()[0]);
    std::optional<Loc> out = placeOf(call.getOperands()[1]);
    if (!it || !out)
      return false;
    Loc length{it->root, it->path + "/length"}, index{it->root, it->path + "/index"};
    if (!state.env.count(length) || !state.env.count(index))
      return false;
    Sort sort{false, 64, true};
    std::string remaining = define(
        sort, "(bvsub " + load(length, state, sort) + " " +
                  load(index, state, sort) + ")");
    std::string list = declare({false, 64, false}, "g");
    store(*out, state, list);
    state.pc = define({true, 1, false},
                      "(and " + state.pc + " (= " + lenOf(list) + " " +
                          remaining + "))",
                      "r");
    setResultsUnknown(call);
    return true;
  }

  /// `tuple[k]` (`Tuple.__getitem_param__[k]`): a reference to field `/k` of
  /// the tuple's place.
  bool tupleField(LIT::CallOp call, const CalleeName &name, State &state) {
    if (!StringRef(name.path).starts_with(
            "std::builtin::tuple::Tuple::__getitem_param__[") ||
        call.getNumOperands() != 1 || call->getNumResults() != 1 ||
        !isa<LIT::RefType>(call->getResult(0).getType()))
      return false;
    std::optional<Loc> tuple = placeOf(call.getOperands()[0]);
    auto symbol = dyn_cast<SymbolConstantAttr>(call.getCallee());
    if (!tuple || !symbol)
      return false;
    // The index is the callee's own first parameter, after the tuple's.
    std::optional<int64_t> k;
    for (TypedAttr param : symbol.getParamValues()) {
      static llvm::Regex indexRe("^#lit.struct<\\{_mlir_value: scalar<index> "
                                 "= ([0-9]+)\\}>");
      SmallVector<StringRef> m;
      int64_t v;
      if (indexRe.match(printed(param), &m) && !m[1].getAsInteger(10, v)) {
        k = v;
        break;
      }
    }
    if (!k)
      return false;
    Value result = call->getResult(0);
    values[result] = declare({false, 64, false}, "g");
    derivedPlaces[result] =
        Loc{tuple->root, tuple->path + "/" + std::to_string(*k)};
    return true;
  }

  /// `Tuple(*pack)`, as a tuple literal builds it (`(n, None)`): field `/k`
  /// is the value the pack's `k`th reference refers to.
  bool evalTupleLiteral(LIT::CallOp call, const CalleeName &name,
                        State &state) {
    if (!StringRef(name.path).starts_with("std::builtin::tuple::Tuple::__init__[") ||
        !StringRef(name.path).contains("](*$0)") || call.getNumOperands() != 2)
      return false;
    std::optional<Loc> packPlace = placeOf(call.getOperands()[0]);
    std::optional<Loc> out = placeOf(call.getOperands()[1]);
    if (!packPlace || !out)
      return false;
    auto it = packRefs.find(load(*packPlace, state, placeType(*packPlace)));
    if (it == packRefs.end())
      return false;
    SmallVector<std::pair<std::string, std::string>> fields;
    for (auto [k, ref] : llvm::enumerate(it->second)) {
      std::optional<Loc> place = placeOf(ref);
      if (!place)
        return false;
      fields.push_back({"/" + std::to_string(k),
                        load(*place, state, placeType(*place))});
    }
    storeFields(*out, state, fields);
    setResultsUnknown(call);
    return true;
  }

  /// Dimension `k` of the `TileTensor` `tensor`, whose type parameters are
  /// `params` (its layout is the fourth): the static size where the layout's
  /// shape says `ComptimeInt[n]`, otherwise `tdim(tensor, k)`, not negative.
  MaybeTerm tensorDim(ArrayRef<TypedAttr> params, Value tensor, int64_t k,
                      State &state) {
    if (params.size() < 4 || k < 0)
      return std::nullopt;
    // The shape is the first list of the layout type's parameters.
    std::string layout = printed(resolveParam(params[3]));
    SmallVector<StringRef> shape = layoutList(layout, 0);
    if ((size_t)k < shape.size())
      if (std::optional<int64_t> n = comptimeIntValue(shape[k]))
        return bvConst(*n, 64);
    if (MaybeTerm n = comptimeShape(params[3], k))
      return n;
    std::string value = isa<LIT::RefType>(tensor.getType())
                            ? valueThrough(tensor, state)
                            : term(tensor, state);
    if (sortOfTerm(value).width != 64 || sortOfTerm(value).isBool)
      return std::nullopt;
    if (!tensorDimDeclared) {
      prelude += "(declare-fun tdim ((_ BitVec 64) (_ BitVec 64)) "
                 "(_ BitVec 64))\n";
      tensorDimDeclared = true;
    }
    std::string d = "(tdim " + value + " " + bvConst(k, 64) + ")";
    facts.push_back("(bvsge " + d + " " + bvConst(0, 64) + ")");
    std::string dim = define(Sort{false, 64, true}, d);
    std::pair<std::string, int64_t> at{layoutKey(params[3]), k};
    auto &terms = dimTerms[at];
    if (!llvm::is_contained(terms, dim))
      terms.push_back(dim);
    relateStaticShape(at);
    return dim;
  }

  /// A layout type parameter, as the frame it belongs to and its printed
  /// value there: the same layout reached through a callee's parameters has
  /// the same key, and two callees' `Self.LayoutType` different ones.
  std::string layoutKey(TypedAttr layout) {
    ParamFrame *scope = nullptr;
    TypedAttr resolved = resolveParam(layout, &scope);
    std::string text = printed(resolved);
    // A closed expression (no parameter in it: a concrete layout, the
    // accelerator's target) is the same in every frame.
    bool closed = !StringRef(text).contains("param.decl.ref") &&
                  !StringRef(text).contains("param.index.ref");
    std::string key = closed ? std::string() : scopeKey(scope, text);
    return (key.empty() ? "|" : key) + text;
  }

  /// The `rank` or `flat_rank` witness of a layout (by `layoutKey`): one
  /// unknown per layout.
  std::string layoutWitness(StringRef name, const std::string &layout) {
    auto [it, inserted] = layoutWitnesses.try_emplace({name.str(), layout});
    if (inserted) {
      it->second = declare(Sort{false, 64, true}, "p");
      prelude += "; " + it->second + ": " + name.str() + " of " +
                 StringRef(layout).take_front(300).str() + "\n";
    }
    return it->second;
  }

  /// `static_shape[k]` is the `k`th extent of the flattened shape, -1 where
  /// it is known only at run time; `dim[k]()` is the `k`th outer mode's
  /// extent. They are the same extent where no mode is nested (`flat_rank
  /// == rank`), and then a known `static_shape[k]` is `dim[k]()`.
  void relateStaticShape(const std::pair<std::string, int64_t> &at) {
    auto shapes = staticShapeTerms.find(at);
    auto dims = dimTerms.find(at);
    if (shapes == staticShapeTerms.end() || dims == dimTerms.end())
      return;
    std::string flat = "(= " + layoutWitness("flat_rank", at.first) + " " +
                       layoutWitness("rank", at.first) + ")";
    for (const std::string &shape : shapes->second)
      for (const std::string &dim : dims->second) {
        std::string fact = "(=> (and " + flat + " (bvsge " + shape + " " +
                           bvConst(0, 64) + ")) (= " + dim + " " + shape + "))";
        if (addedFacts.insert(fact).second)
          facts.push_back(fact);
      }
  }

  /// Extent `k` of a `TileTensor` layout type parameter whose shape says
  /// `ComptimeInt[v]` for a parameter expression `v` (`BM` in a function
  /// generic over it), as a term in the scope the layout came from.
  MaybeTerm comptimeShape(TypedAttr layoutParam, int64_t k) {
    ParamFrame *scope = nullptr;
    auto layout = dyn_cast<TypeParamAttr>(resolveParam(layoutParam, &scope));
    auto type = layout ? dyn_cast<LIT::StructType>(layout.getTypeValue())
                       : LIT::StructType();
    if (!type || type.getParamValues().empty() ||
        printed(type.getSymbol()) != "@layout::@tile_layout::@Layout")
      return std::nullopt;
    auto shape = dyn_cast<ParamListAttr>(type.getParamValues()[0]);
    if (!shape || k >= (int64_t)shape.getValues().size())
      return std::nullopt;
    ParamFrame *saved = params;
    params = scope;
    MaybeTerm n = comptimeIntTerm(shape.getValues()[k]);
    params = saved;
    return n;
  }

  /// The value of a `ComptimeInt[v]` type parameter, for a parameter
  /// expression `v`, as a term in the current scope.
  MaybeTerm comptimeIntTerm(TypedAttr mode) {
    auto type = dyn_cast<TypeParamAttr>(mode);
    auto comptimeInt = type ? dyn_cast<LIT::StructType>(type.getTypeValue())
                            : LIT::StructType();
    if (!comptimeInt || comptimeInt.getParamValues().size() != 1 ||
        printed(comptimeInt.getSymbol()) !=
            "@std::@utils::@coord::@ComptimeInt")
      return std::nullopt;
    TypedAttr value = comptimeInt.getParamValues()[0];
    Sort sort = sortOf(value.getType());
    if (sort.isBool || sort.width != 64)
      return std::nullopt;
    return paramTerm(value, sort);
  }

  /// Element `k` of the `Coord` value `coord`, whose type is `type`, as a
  /// 64-bit term: a literal `ComptimeInt`'s value, or the integer the
  /// coordinate was built with (`Coord(*values)`), widened by its
  /// signedness. None for any other element, so that a struct's value is
  /// never taken for an integer.
  MaybeTerm coordElement(StringRef type, const std::string &coord, size_t k) {
    if (std::optional<int64_t> n = comptimeIntValue(type))
      return bvConst(*n, 64);
    // An integer, or the index of a `comptime for` (the range's `Element`).
    if (!type.starts_with("@std::@simd::@SIMD<") &&
        !type.contains("\"Element\", @std::@simd::@SIMD<"))
      return std::nullopt;
    auto it = builtFields.find({coord, "/" + std::to_string(k)});
    if (it == builtFields.end())
      return std::nullopt;
    std::string c = it->second;
    Sort sort = sortOfTerm(c);
    if (sort.isBool || sort.width > 64)
      return std::nullopt;
    if (sort.width < 64)
      c = "((_ " + std::string(sort.isSigned ? "sign" : "zero") + "_extend " +
          std::to_string(64 - sort.width) + ") " + c + ")";
    return c;
  }

  /// The element types of a `Coord` from its `#kgen.param_list` parameter.
  std::optional<SmallVector<StringRef>> coordTypes(StringRef printedList) {
    if (!printedList.consume_front("#kgen.param_list<"))
      return std::nullopt;
    return listElements(printedList);
  }

  /// The integers of a parameter list (`#kgen.param_list<2, 4>`, or of
  /// parameters `BM, BN` in a generic function), such as a `*sizes: Int`
  /// parameter, as terms.
  std::optional<SmallVector<std::string>> intList(TypedAttr param) {
    // The list's elements are in the scope of the frame it came from (a
    // caller's `BM`, passed on as `tile[BM, 16]`).
    ParamFrame *scope = nullptr;
    auto list = dyn_cast<ParamListAttr>(resolveParam(param, &scope));
    if (!list)
      return std::nullopt;
    ParamFrame *saved = params;
    params = scope;
    llvm::scope_exit restore([&] { params = saved; });
    SmallVector<std::string> values;
    for (TypedAttr value : list.getValues()) {
      std::string v = paramTerm(value, Sort{false, 64, true});
      if (sortOfTerm(v).isBool || sortOfTerm(v).width != 64)
        return std::nullopt;
      values.push_back(v);
    }
    return values;
  }

  /// `TileTensor` from the `layout` package: `dim[k]()`, the extent of
  /// dimension `k`, is the static size where the layout's shape says
  /// `ComptimeInt[n]`, and otherwise a function of the tensor's value (one
  /// per dimension, not negative). `_coord_in_bounds(i, n)`, the clause of
  /// its indexing, is `0 <= i < n` for an `Int` or `ComptimeInt` coordinate
  /// and true for a tuple one.
  bool evalTileTensor(LIT::CallOp call, const CalleeName &name, State &state) {
    StringRef path = name.path;
    auto symbol = dyn_cast<SymbolConstantAttr>(call.getCallee());
    if (!symbol || call->getNumResults() != 1)
      return false;
    Value result = call->getResult(0);
    Sort sort{false, 64, true};
    // An integer parameter, as a type argument (`{:scalar<index> 8}`) or a
    // value (`{_mlir_value: scalar<index> = 0}`); the number is group 2.
    static llvm::Regex intValue(
        "(\\{:scalar<index> |scalar<index> = )(-?[0-9]+)\\}");
    // `Int(t.dim[k]())`: the extent, whatever the tensor's index dtype (an
    // extent is not negative and fits).
    if (path.starts_with("std::simd::SIMD::__init__[::DType](::SIMD[") &&
        call.getNumOperands() == 1 && tensorDims.count(call.getOperands()[0])) {
      Sort target = sortOf(result.getType());
      if (target.isBool || target.width != 64)
        return false;
      values[result] = term(call.getOperands()[0], state);
      return true;
    }
    // `rebind[T](x)`: the same value under another type.
    if (path.starts_with("std::builtin::rebind::rebind[") &&
        call.getNumOperands() == 1 &&
        !isa<LIT::RefType>(call.getOperands()[0].getType())) {
      std::string value = term(call.getOperands()[0], state);
      Sort to = sortOf(result.getType()), from = sortOfTerm(value);
      if (to.isBool == from.isBool && (to.isBool || to.width == from.width)) {
        values[result] = value;
        if (auto it = records.find(call.getOperands()[0]); it != records.end())
          records[result] = it->second;
        return true;
      }
    }
    // `Int(x)` of an `Int` (in a function generic over an `Intable` index):
    // `x`.
    if (path.starts_with("std::simd::SIMD::__init__[::Intable") &&
        call.getNumOperands() == 1) {
      Value operand = call.getOperands()[0];
      Type from = operand.getType();
      if (auto ref = dyn_cast<LIT::RefType>(from))
        from = ref.getElementType();
      bool isInt = false;
      if (auto param = dyn_cast<KGEN::ParamType>(from))
        isInt = isIntType(resolveParam(param.getParam()));
      else
        isInt = isIntPlace(from);
      Sort target = sortOf(result.getType());
      if (isInt && !target.isBool && target.width == 64) {
        std::string value = isa<LIT::RefType>(operand.getType())
                                ? valueThrough(operand, state)
                                : term(operand, state);
        if (!sortOfTerm(value).isBool && sortOfTerm(value).width == 64) {
          values[result] = value;
          return true;
        }
      }
      // Of any other `Intable` value: one result per value (the conversion
      // is assumed to be a function of the value), so a clause's
      // `Int(index)` is the body's.
      if (!target.isBool && target.width == 64) {
        std::string value = isa<LIT::RefType>(operand.getType())
                                ? valueThrough(operand, state)
                                : term(operand, state);
        auto [it, inserted] = intConversions.try_emplace(value, "");
        if (inserted)
          it->second = declare(target);
        values[result] = it->second;
        return true;
      }
    }
    // `t.ptr` (and any other field read through `__getattr_param__`): a
    // function of the tensor's value and the field, so a clause's
    // `t.ptr._extent()` is about the pointer the body reads.
    if (path.starts_with(
            "layout::tile_tensor::TileTensor::__getattr_param__[") &&
        call.getNumOperands() == 1) {
      std::string tensor = tensorTerm(call.getOperands()[0], state);
      Sort field = sortOf(result.getType());
      if (sortOfTerm(tensor).isBool || sortOfTerm(tensor).width != 64)
        return false;
      auto [it, inserted] = tensorFields.try_emplace(
          {tensor, printed(call.getCallee())}, "");
      if (inserted)
        it->second = declare(field);
      values[result] = it->second;
      return true;
    }
    // The implicit conversion of a mutable tensor to an immutable one: the
    // same tensor (layout and storage), so the same dimensions.
    if (path.starts_with("layout::tile_tensor::TileTensor::__init__[") &&
        path.contains("](::TileTensor[") && call.getNumOperands() == 1 &&
        !isa<LIT::RefType>(call.getOperands()[0].getType())) {
      values[result] = term(call.getOperands()[0], state);
      return true;
    }
    // `t.num_elements()`: the product of its dimensions (its rank from its
    // layout's shape).
    if (path.starts_with("layout::tile_tensor::TileTensor::num_elements(") &&
        call.getNumOperands() == 1) {
      ArrayRef<TypedAttr> params = symbol.getParamValues();
      if (params.size() < 4)
        return false;
      auto product = [&](size_t rank) -> MaybeTerm {
        std::string p;
        for (size_t k = 0; k < rank; ++k) {
          MaybeTerm d = tensorDim(params, call.getOperands()[0], k, state);
          if (!d)
            return std::nullopt;
          p = k ? "(bvmul " + p + " " + *d + ")" : *d;
        }
        return p;
      };
      std::string layout = printed(resolveParam(params[3]));
      size_t rank = layoutList(layout, 0).size();
      if (rank != 0) {
        MaybeTerm p = product(rank);
        if (!p)
          return false;
        values[result] = define(sort, *p);
        return true;
      }
      // A generic layout: by its rank where one was seen (`comptime assert
      // t.rank == 2`), for ranks up to 4.
      auto it = layoutWitnesses.find({"rank", layoutKey(params[3])});
      if (it == layoutWitnesses.end())
        return false;
      std::string n = declare(sort, "n");
      for (size_t r = 1; r <= 4; ++r)
        if (MaybeTerm p = product(r))
          facts.push_back("(=> (= " + it->second + " " + bvConst(r, 64) +
                          ") (= " + n + " " + *p + "))");
      values[result] = n;
      return true;
    }
    if (path.starts_with("layout::tile_tensor::_coord_in_bounds[") &&
        call.getNumOperands() == 2 && !symbol.getParamValues().empty()) {
      std::string type = printed(resolveParam(symbol.getParamValues()[0]));
      SmallVector<StringRef> m;
      std::string bound = term(call.getOperands()[1], state), v;
      if (StringRef(type).contains("@std::@utils::@coord::@Coord<")) {
        values[result] = "true";
        return true;
      }
      if (StringRef(type).contains("@std::@utils::@coord::@ComptimeInt<") &&
          intValue.match(type, &m)) {
        int64_t n;
        if (m[2].getAsInteger(10, n))
          return false;
        v = bvConst(n, 64);
      } else if (isIntType(resolveParam(symbol.getParamValues()[0]))) {
        v = term(call.getOperands()[0], state);
      } else {
        return false;
      }
      if (sortOfTerm(v).width != 64 || sortOfTerm(bound).width != 64)
        return false;
      values[result] = define({true, 1, false},
                              "(and (bvsle " + bvConst(0, 64) + " " + v +
                                  ") (bvslt " + v + " " + bound + "))");
      return true;
    }
    if (path.starts_with(
            "layout::tile_tensor::TileTensor::dim[::SIMD[DType.int, 1]](") &&
        call.getNumOperands() == 1) {
      ArrayRef<TypedAttr> params = symbol.getParamValues();
      SmallVector<StringRef> m;
      int64_t k;
      std::string last =
          params.empty() ? "" : printed(resolveParam(params.back()));
      if (!intValue.match(last, &m) || m[2].getAsInteger(10, k))
        return false;
      MaybeTerm d = tensorDim(params, call.getOperands()[0], k, state);
      if (!d)
        return false;
      tensorDims.insert(result);
      values[result] = *d;
      return true;
    }
    // `t._valid_dim[k]()`: the valid extent (`validExtent`).
    if (path.starts_with("layout::tile_tensor::TileTensor::_valid_dim[") &&
        call.getNumOperands() == 1) {
      ArrayRef<TypedAttr> params = symbol.getParamValues();
      SmallVector<StringRef> m;
      int64_t k;
      std::string last =
          params.empty() ? "" : printed(resolveParam(params.back()));
      if (!intValue.match(last, &m) || m[2].getAsInteger(10, k))
        return false;
      MaybeTerm d = validExtent(params, call.getOperands()[0], k, state);
      if (!d)
        return false;
      values[result] = *d;
      return true;
    }
    // `t._indices_in_bounds(*items)`, the clause of writing an element:
    // every index `i` of the pack in `[0, dim[i])`.
    if (path.starts_with(
            "layout::tile_tensor::TileTensor::_indices_in_bounds[") &&
        call.getNumOperands() == 2) {
      Value pack = call.getOperands()[1];
      while (auto rebind = pack.getDefiningOp<RebindOp>())
        pack = rebind->getOperand(0);
      if (!isa<LIT::RefType>(pack.getType()))
        return false;
      auto refs = packRefs.find(valueThrough(pack, state));
      if (refs == packRefs.end())
        return false;
      std::string all = "true";
      for (auto [k, ref] : llvm::enumerate(refs->second)) {
        std::optional<Loc> place = placeOf(ref);
        // An `Int` (possibly behind an alias), as `index` takes it.
        if (!place || !isIntPlace(placeType(*place)))
          return false;
        std::string v = load(*place, state, sort);
        MaybeTerm d = validExtent(symbol.getParamValues(), call.getOperands()[0],
                                  k, state);
        if (!d)
          return false;
        all = "(and " + all + " (bvsle " + bvConst(0, 64) + " " + v +
              ") (bvslt " + v + " " + *d + "))";
      }
      values[result] = define({true, 1, false}, all);
      return true;
    }
    // `t._tile_in_bounds[*sizes](*coords)`, the clause of taking a tile:
    // every coordinate `c` of the pack not negative for a positive size
    // (`sizes` is the ninth parameter). The tile may extend past the edge:
    // its valid extents say how much of it is backed (`recordTile`).
    if (path.starts_with("layout::tile_tensor::TileTensor::_tile_in_bounds[") &&
        call.getNumOperands() == 2) {
      ArrayRef<TypedAttr> params = symbol.getParamValues();
      if (params.size() < 9)
        return false;
      std::optional<SmallVector<std::string>> sizes = intList(params[8]);
      std::optional<SmallVector<std::string>> coords =
          intPack(call.getOperands()[1], state);
      if (!sizes || !coords || coords->size() != sizes->size())
        return false;
      std::string all = "true";
      for (auto [c, size] : llvm::zip(*coords, *sizes))
        all = "(and " + all + " (ite (bvsgt " + size + " " + bvConst(0, 64) +
              ") (bvsle " + bvConst(0, 64) + " " + c + ") true))";
      values[result] = define({true, 1, false}, all);
      return true;
    }
    // `t._tile_coords_in_bounds[*sizes](coords)` and
    // `t._tile_shape_in_bounds(shape, coords)`, the clauses of taking a tile
    // with `Coord` coordinates: as `_tile_in_bounds`, with each dimension's
    // coordinate the `Coord`'s element, and its size from the integer list
    // `sizes` (the ninth parameter) or the shape `Coord`'s element. The
    // `Coord`s' element types are the type lists among the parameters after
    // the tensor's eight (and the sizes). A nested layout, another rank or a
    // tuple element is true, as in the helpers.
    bool fromSizes = path.starts_with(
        "layout::tile_tensor::TileTensor::_tile_coords_in_bounds[");
    if ((fromSizes && call.getNumOperands() == 2) ||
        (path.starts_with(
             "layout::tile_tensor::TileTensor::_tile_shape_in_bounds") &&
         call.getNumOperands() == 3)) {
      ArrayRef<TypedAttr> params = symbol.getParamValues();
      std::optional<SmallVector<std::string>> sizes;
      if (fromSizes && (params.size() < 9 || !(sizes = intList(params[8]))))
        return false;
      SmallVector<std::string> lists;
      for (size_t i = fromSizes ? 9 : 8;
           i < params.size() && lists.size() < (fromSizes ? 1u : 2u); ++i) {
        std::string text = printed(resolveParam(params[i]));
        if (StringRef(text).starts_with("#kgen.param_list<@") ||
            StringRef(text).starts_with("#kgen.param_list<!"))
          lists.push_back(text);
      }
      if (lists.size() != (fromSizes ? 1u : 2u))
        return false;
      std::optional<SmallVector<StringRef>> shapeTypes;
      if (!fromSizes && !(shapeTypes = coordTypes(lists[0])))
        return false;
      std::optional<SmallVector<StringRef>> elementTypes =
          coordTypes(lists.back());
      SmallVector<StringRef> shape =
          layoutList(printed(resolveParam(params[3])), 0);
      if (!elementTypes || shape.empty())
        return false;
      size_t rank = shape.size();
      bool nested = llvm::any_of(shape, [](StringRef mode) {
        return mode.starts_with("@std::@utils::@coord::@Coord<");
      });
      if (nested || elementTypes->size() != rank ||
          (sizes ? sizes->size() : shapeTypes->size()) != rank) {
        values[result] = "true";
        return true;
      }
      auto valueOf = [&](Value v) {
        return isa<LIT::RefType>(v.getType()) ? valueThrough(v, state)
                                              : term(v, state);
      };
      std::string shapeValue = fromSizes ? "" : valueOf(call.getOperands()[1]);
      std::string coordValue = valueOf(call.getOperands().back());
      auto tuple = [](StringRef type) {
        return type.starts_with("@std::@utils::@coord::@Coord<");
      };
      std::string all = "true", zero = bvConst(0, 64);
      for (size_t k = 0; k < rank; ++k) {
        StringRef ct = (*elementTypes)[k];
        if (tuple(ct) || (!fromSizes && tuple((*shapeTypes)[k])))
          continue;
        MaybeTerm size = fromSizes
                             ? MaybeTerm((*sizes)[k])
                             : coordElement((*shapeTypes)[k], shapeValue, k);
        MaybeTerm c = coordElement(ct, coordValue, k);
        MaybeTerm d = validExtent(params, call.getOperands()[0], k, state);
        if (!size || !c || !d)
          return false;
        all = "(and " + all + " (ite (bvsgt " + *size + " " + zero +
              ") (and (bvsle " + zero + " " + *c + ") (bvslt " + *c +
              " (bvsdiv " + *d + " " + *size + "))) true))";
      }
      values[result] = define({true, 1, false}, all);
      return true;
    }
    // `t._access_in_bounds[width](coord)`, the clause of `load` and
    // `store`: for a flat layout without vectorization, indexed by one
    // value per dimension, every integer coordinate in `[0, dim)`, and with
    // a width above one the last at most `dim - width` and its stride 1.
    // Anything else is true, as in the helper. The width is the ninth
    // parameter, the coordinate's element types the tenth.
    if (path.starts_with(
            "layout::tile_tensor::TileTensor::_access_in_bounds[") &&
        call.getNumOperands() == 2) {
      ArrayRef<TypedAttr> params = symbol.getParamValues();
      if (params.size() < 10)
        return false;
      // The width: a literal, or a parameter's value (`load[simd_width]`).
      static llvm::Regex widthRe("_mlir_value = ([0-9]+)\\}");
      SmallVector<StringRef> m;
      std::string widthText = printed(resolveParam(params[8]));
      int64_t literalWidth;
      std::string width;
      if (widthRe.match(widthText, &m) && !m[1].getAsInteger(10, literalWidth))
        width = bvConst(literalWidth, 64);
      else
        width = paramTerm(params[8], Sort{false, 64, true});
      std::string typesText = printed(resolveParam(params[9]));
      StringRef types(typesText);
      if (!types.consume_front("#kgen.param_list<"))
        return false;
      SmallVector<StringRef> elementTypes = listElements(types);
      ParamFrame *layoutScope = nullptr;
      TypedAttr layoutAttr = resolveParam(params[3], &layoutScope);
      std::string layout = printed(layoutAttr);
      SmallVector<StringRef> shape = layoutList(layout, 0);
      SmallVector<StringRef> strides = layoutList(layout, 1);
      // A tile of a generic layout: its strides are not a list but the
      // parent's (`_NestedTileResultStrideTypes[Parent]`, which names the
      // parent's `__stride_types`), mode for mode where the parent is
      // flat. Whether the last is 1 is then the parent's `static_stride`.
      std::string parentUnit;
      if (!shape.empty() && StringRef(layout).contains("\"__stride_types\"")) {
        TypedAttr parent;
        bool one = true;
        layoutAttr.walk([&](GetWitnessAttr witness) {
          if (witness.getWitnessName() != "__stride_types")
            return;
          if (!parent)
            parent = witness.getTypeValue();
          else if (parent != witness.getTypeValue())
            one = false;
        });
        if (!parent || !one)
          return false;
        ParamFrame *saved = this->params;
        this->params = layoutScope;
        std::string key = layoutKey(parent);
        this->params = saved;
        int64_t last = shape.size() - 1;
        auto [it, inserted] = staticStrideTerms.try_emplace({key, last}, "");
        if (inserted) {
          it->second = declare(Sort{false, 64, true}, "p");
          prelude += "; " + it->second + ": static_stride[" +
                     std::to_string(last) + "] of " +
                     StringRef(key).take_front(300).str() + "\n";
        }
        const std::string &stride = it->second;
        // A runtime stride (-1) is not known to be 1, nor is the stride
        // of a tile of a nested parent.
        parentUnit =
            "(ite (and (= " + layoutWitness("flat_rank", key) + " " +
            layoutWitness("rank", key) + ") (= " + layoutWitness("rank", key) +
            " " + bvConst(shape.size(), 64) + ")) (or (= " + stride + " " +
            bvConst(1, 64) + ") (and (bvslt " + stride + " " + bvConst(0, 64) +
            ") " + declare({true, 1, false}) + ")) " +
            declare({true, 1, false}) + ")";
      }
      // The engine's element width (`DefaultEngine[element_width=n]`); of
      // another engine (a tile's `OffsetResultType`) it is not known, and
      // the helper is true where it is not 1.
      static llvm::Regex engineRe(
          "^#kgen.type<!lit.struct<@layout::@tensor_engine::@"
          "(DefaultEngine|DevicePointerEngine)<.*\\{:scalar<index> "
          "([0-9]+)\\}>>> :");
      std::string engine = printed(resolveParam(params[5]));
      int64_t elementWidth = 1;
      bool engineKnown =
          engineRe.match(engine, &m) && !m[2].getAsInteger(10, elementWidth);
      if (shape.empty() ||
          (parentUnit.empty() &&
           (strides.size() != shape.size() || !engineKnown)))
        return false;
      bool nested = llvm::any_of(shape, [](StringRef mode) {
        return mode.starts_with("@std::@utils::@coord::@Coord<");
      });
      if (nested || elementTypes.size() != shape.size() || elementWidth != 1) {
        values[result] = "true";
        return true;
      }
      Value coord = call.getOperands()[1];
      std::string coordValue = isa<LIT::RefType>(coord.getType())
                                   ? valueThrough(coord, state)
                                   : term(coord, state);
      std::string all = "true";
      size_t last = shape.size() - 1;
      for (auto [k, type] : llvm::enumerate(elementTypes)) {
        if (type.starts_with("@std::@utils::@coord::@Coord<"))
          continue;
        MaybeTerm element = coordElement(type, coordValue, k);
        if (!element)
          return false;
        std::string c = *element;
        MaybeTerm d = validExtent(params, call.getOperands()[0], k, state);
        if (!d)
          return false;
        std::string zero = bvConst(0, 64);
        std::string scalar =
            "(and (bvsle " + zero + " " + c + ") (bvslt " + c + " " + *d + "))";
        if (k == last) {
          // A runtime stride is not known to be 1.
          std::string unit = parentUnit;
          if (unit.empty()) {
            std::optional<int64_t> stride = comptimeIntValue(strides[k]);
            unit = !stride        ? declare({true, 1, false})
                   : *stride == 1 ? "true"
                                  : "false";
          }
          std::string wide = "(and " + unit + " (bvsle " + zero + " " + c +
                             ") (bvsle " + c + " (bvsub " + *d + " " + width +
                             ")))";
          all = "(and " + all + " (ite (bvsgt " + width + " " + bvConst(1, 64) +
                ") " + wide + " " + scalar + "))";
        } else {
          all = "(and " + all + " " + scalar + ")";
        }
      }
      if (!engineKnown)
        all = "(or " + declare({true, 1, false}) + " " + all + ")";
      values[result] = define({true, 1, false}, all);
      return true;
    }
    // `Coord(*values)`: field `/k` of the coordinate is the `k`th value
    // where it is an integer (a `ComptimeInt`'s value is in its type).
    if (path.starts_with("std::utils::coord::Coord::__init__[") &&
        path.contains("](*$0)") && call.getNumOperands() == 1) {
      std::optional<Loc> packPlace = placeOf(call.getOperands()[0]);
      if (!packPlace)
        return false;
      auto it = packRefs.find(load(*packPlace, state, placeType(*packPlace)));
      if (it == packRefs.end())
        return false;
      std::string built = declare({false, 64, false}, "g");
      for (auto [k, ref] : llvm::enumerate(it->second)) {
        std::optional<Loc> place = placeOf(ref);
        if (place && isa<KGEN::ParamType>(placeType(*place)) &&
            isIntPlace(placeType(*place))) {
          // The index of a `comptime for`.
          setBuiltField(built, "/" + std::to_string(k),
                        load(*place, state, Sort{false, 64, true}));
          continue;
        }
        if (!place || !isScalar(placeType(*place)) ||
            sortOf(placeType(*place)).isBool)
          continue;
        setBuiltField(built, "/" + std::to_string(k),
                      load(*place, state, sortOf(placeType(*place))));
      }
      values[result] = built;
      return true;
    }
    // `Coord(tuple)`, as a tuple literal converts (`t.load((i, j))`): the
    // fields the tuple was built with.
    if (path.starts_with("std::utils::coord::Coord::__init__(::Tuple[") &&
        call.getNumOperands() == 1) {
      Value tuple = call.getOperands()[0];
      std::string from = isa<LIT::RefType>(tuple.getType())
                             ? valueThrough(tuple, state)
                             : term(tuple, state);
      SmallVector<std::pair<std::string, std::string>> fields;
      for (auto it = builtFields.lower_bound({from, ""});
           it != builtFields.end() && it->first.first == from; ++it)
        fields.push_back({it->first.second, it->second});
      std::string built = declare({false, 64, false}, "g");
      for (auto &[field, value] : fields)
        setBuiltField(built, field, value);
      values[result] = built;
      return true;
    }
    // `t._vectorize_in_bounds[*sizes]()`, the clause of vectorizing: every
    // dimension a multiple of its size, where that is more than one.
    if (path.starts_with(
            "layout::tile_tensor::TileTensor::_vectorize_in_bounds[") &&
        call.getNumOperands() == 1) {
      ArrayRef<TypedAttr> params = symbol.getParamValues();
      std::optional<SmallVector<std::string>> sizes;
      if (params.size() < 9 || !(sizes = intList(params[8])))
        return false;
      std::string all = "true";
      for (auto [k, size] : llvm::enumerate(*sizes)) {
        MaybeTerm d = tensorDim(params, call.getOperands()[0], k, state);
        if (!d)
          return false;
        all = "(and " + all + " (ite (bvsgt " + size + " " + bvConst(1, 64) +
              ") (= (bvsrem " + *d + " " + size + ") " + bvConst(0, 64) +
              ") true))";
      }
      values[result] = define({true, 1, false}, all);
      return true;
    }
    // `t.distribute[thread_layout](tid)`: a view with `dim // threads`
    // elements per dimension (stride `stride * threads`, starting at the
    // thread's coordinate `(tid // s) % threads`, so within the tensor for
    // any `tid`). Its layout type may hold them unevaluated, or computed at
    // run time; `threads` is the thread layout's shape, the ninth parameter.
    // `distribute_with_offset` returns the same view as the first of a
    // tuple (with the thread's coordinates and offset, unknown here).
    bool withOffset = path.starts_with(
        "layout::tile_tensor::TileTensor::distribute_with_offset[");
    if ((withOffset ||
         path.starts_with("layout::tile_tensor::TileTensor::distribute[")) &&
        call.getNumOperands() == 2) {
      ArrayRef<TypedAttr> params = symbol.getParamValues();
      if (params.size() < 9)
        return false;
      ParamFrame *scope = nullptr;
      auto threads = dyn_cast<ParamListAttr>(resolveParam(params[8], &scope));
      SmallVector<StringRef> shape =
          layoutList(printed(resolveParam(params[3])), 0);
      if (!threads || shape.empty() ||
          threads.getValues().size() != shape.size() ||
          llvm::any_of(shape, [](StringRef mode) {
            return mode.starts_with("@std::@utils::@coord::@Coord<");
          }))
        return false;
      SmallVector<std::string> counts;
      {
        ParamFrame *saved = this->params;
        this->params = scope;
        for (TypedAttr mode : threads.getValues())
          if (MaybeTerm n = comptimeIntTerm(mode))
            counts.push_back(*n);
        this->params = saved;
      }
      if (counts.size() != shape.size())
        return false;
      SmallVector<std::string> dims;
      for (size_t k = 0; k < shape.size(); ++k) {
        MaybeTerm d = tensorDim(params, call.getOperands()[0], k, state);
        if (!d)
          return false;
        dims.push_back("(bvsdiv " + *d + " " + counts[k] + ")");
      }
      std::string view = declare({false, 64, false}, "g");
      if (withOffset) {
        std::string tuple = declare({false, 64, false}, "g");
        setBuiltField(tuple, "/0", view);
        values[result] = tuple;
      } else {
        values[result] = view;
      }
      if (!tensorDimDeclared) {
        prelude += "(declare-fun tdim ((_ BitVec 64) (_ BitVec 64)) "
                   "(_ BitVec 64))\n";
        tensorDimDeclared = true;
      }
      for (auto [k, d] : llvm::enumerate(dims))
        facts.push_back("(=> (bvsgt " + counts[k] + " " + bvConst(0, 64) +
                        ") (= (tdim " + view + " " + bvConst(k, 64) + ") " + d +
                        "))");
      return true;
    }
    // `t.vectorize[*sizes]()`: a view with `ceildiv(dim, size)` vectors per
    // dimension (its layout type gives them unevaluated).
    if (path.starts_with("layout::tile_tensor::TileTensor::vectorize[") &&
        call.getNumOperands() == 1) {
      ArrayRef<TypedAttr> params = symbol.getParamValues();
      std::optional<SmallVector<std::string>> sizes;
      if (params.size() < 9 || !(sizes = intList(params[8])))
        return false;
      SmallVector<std::string> dims;
      for (auto [k, s] : llvm::enumerate(*sizes)) {
        MaybeTerm d = tensorDim(params, call.getOperands()[0], k, state);
        if (!d)
          return false;
        dims.push_back("(ite (= (bvsrem " + *d + " " + s + ") " +
                       bvConst(0, 64) + ") (bvsdiv " + *d + " " + s +
                       ") (bvadd (bvsdiv " + *d + " " + s + ") " +
                       bvConst(1, 64) + "))");
      }
      setResultsUnknown(call);
      std::string view = values[result];
      if (sortOfTerm(view).width != 64 || sortOfTerm(view).isBool)
        return false;
      if (!tensorDimDeclared) {
        prelude += "(declare-fun tdim ((_ BitVec 64) (_ BitVec 64)) "
                   "(_ BitVec 64))\n";
        tensorDimDeclared = true;
      }
      // For a positive size; a view of another has no extents to state.
      for (auto [k, d] : llvm::enumerate(dims))
        facts.push_back("(=> (bvsgt " + (*sizes)[k] + " " + bvConst(0, 64) +
                        ") (= (tdim " + view + " " + bvConst(k, 64) + ") " + d +
                        "))");
      // A view of a partial tile: a vector is valid where all its elements
      // are, so `valid // size` of them.
      std::string parent = tensorTerm(call.getOperands()[0], state);
      if (auto it = partialValid.find(parent);
          it != partialValid.end() && it->second.size() == sizes->size()) {
        SmallVector<std::string> valid;
        for (auto [v, size] : llvm::zip(it->second, *sizes))
          valid.push_back(define(sort, "(ite (bvsgt " + size + " " +
                                           bvConst(0, 64) + ") (bvsdiv " + v +
                                           " " + size + ") " + v + ")"));
        partialValid[view] = std::move(valid);
      }
      return true;
    }
    return false;
  }

  /// A tensor argument's value, through a reference.
  std::string tensorTerm(Value tensor, State &state) {
    return isa<LIT::RefType>(tensor.getType()) ? valueThrough(tensor, state)
                                               : term(tensor, state);
  }

  /// `_valid_dim[k]()` of a tensor whose type parameters are `params`: the
  /// recorded extent of a partial tile (or a view of one), else `dim[k]()`.
  MaybeTerm validExtent(ArrayRef<TypedAttr> params, Value tensor, int64_t k,
                        State &state) {
    if (auto it = partialValid.find(tensorTerm(tensor, state));
        it != partialValid.end() && k >= 0 && (size_t)k < it->second.size())
      return it->second[k];
    return tensorDim(params, tensor, k, state);
  }

  /// Whether a place holds a 64-bit integer: an `Int`, or a type that is
  /// one by a parameter expression (the index of a `comptime for` over a
  /// range has the range's `Element` type).
  bool isIntPlace(Type type) {
    if (auto param = dyn_cast<KGEN::ParamType>(type))
      return isIntType(param.getParam());
    return isScalar(type) && !sortOf(type).isBool && sortOf(type).width == 64;
  }

  /// The integers a variadic `*args: Int` pack holds (its references'
  /// values), in order.
  std::optional<SmallVector<std::string>> intPack(Value pack, State &state) {
    while (auto rebind = pack.getDefiningOp<RebindOp>())
      pack = rebind->getOperand(0);
    if (!isa<LIT::RefType>(pack.getType()))
      return std::nullopt;
    auto refs = packRefs.find(valueThrough(pack, state));
    if (refs == packRefs.end())
      return std::nullopt;
    SmallVector<std::string> ints;
    for (Value ref : refs->second) {
      std::optional<Loc> place = placeOf(ref);
      if (!place || !isIntPlace(placeType(*place)))
        return std::nullopt;
      ints.push_back(load(*place, state, Sort{false, 64, true}));
    }
    return ints;
  }

  /// After `t.tile[*sizes](*coords)` (the `Int` coordinates form): the
  /// tile's valid extents, `clamp(valid(t) - c * size, 0, size)` per
  /// dimension, so a tile past the edge (partial) is indexed only where it
  /// is backed.
  void recordTile(LIT::CallOp call, const CalleeName &name, State &state) {
    StringRef path = name.path;
    auto symbol = dyn_cast<SymbolConstantAttr>(call.getCallee());
    if (!symbol || call->getNumResults() != 1 || call.getNumOperands() != 2 ||
        !path.starts_with("layout::tile_tensor::TileTensor::tile[") ||
        !path.contains("*::SIMD[DType.int, 1]"))
      return;
    ArrayRef<TypedAttr> params = symbol.getParamValues();
    if (params.size() < 9)
      return;
    std::optional<SmallVector<std::string>> sizes = intList(params[8]);
    std::optional<SmallVector<std::string>> coords =
        intPack(call.getOperands()[1], state);
    std::string tile = term(call->getResult(0), state);
    if (!sizes || !coords || coords->size() != sizes->size() ||
        sortOfTerm(tile).isBool || sortOfTerm(tile).width != 64)
      return;
    Sort sort{false, 64, true};
    std::string zero = bvConst(0, 64);
    SmallVector<std::string> valid;
    for (auto [k, size] : llvm::enumerate(*sizes)) {
      MaybeTerm parent = validExtent(params, call.getOperands()[0], k, state);
      if (!parent)
        return;
      const std::string &c = (*coords)[k];
      std::string rest = define(sort, "(bvsub " + *parent + " (bvmul " + c +
                                          " " + size + "))");
      std::string clamped = define(
          sort, "(ite (bvslt " + rest + " " + zero + ") " + zero +
                    " (ite (bvslt " + size + " " + rest + ") " + size + " " +
                    rest + "))");
      // A whole tile (`0 <= c < valid // size`, the old clause) is valid
      // throughout: stated so, the solver need not multiply `c * size`.
      valid.push_back(define(
          sort, "(ite (and (bvsgt " + size + " " + zero + ") (bvsle " + zero +
                    " " + c + ") (bvslt " + c + " (bvsdiv " + *parent + " " +
                    size + "))) " + size + " " + clamped + ")"));
    }
    partialValid[tile] = std::move(valid);
  }

  /// After `row_major(coord)`: the layout's shape is the coordinate's
  /// elements (`Coord(n, k)` built with integers).
  void recordRowMajor(LIT::CallOp call, const CalleeName &name,
                      State &state) {
    auto symbol = dyn_cast<SymbolConstantAttr>(call.getCallee());
    if (!symbol || call->getNumResults() != 1 || call.getNumOperands() != 1 ||
        !StringRef(name.path).starts_with("layout::tile_layout::row_major[") ||
        symbol.getParamValues().empty())
      return;
    // Kept alive while its elements are used.
    std::string typeList = printed(resolveParam(symbol.getParamValues()[0]));
    std::optional<SmallVector<StringRef>> types = coordTypes(typeList);
    if (!types)
      return;
    std::string coord = tensorTerm(call.getOperands()[0], state);
    SmallVector<std::string> shape;
    for (auto [k, type] : llvm::enumerate(*types)) {
      MaybeTerm e = coordElement(type, coord, k);
      if (!e)
        return;
      shape.push_back(*e);
    }
    layoutShapes[term(call->getResult(0), state)] = std::move(shape);
  }

  /// The number of elements of a tensor's valid extents: their product, for
  /// a rank its layout gives or a `rank` seen (up to 4).
  MaybeTerm validCount(ArrayRef<TypedAttr> params, Value tensor,
                       State &state) {
    if (params.size() < 4)
      return std::nullopt;
    auto product = [&](size_t rank) -> MaybeTerm {
      std::string p;
      for (size_t k = 0; k < rank; ++k) {
        MaybeTerm d = validExtent(params, tensor, k, state);
        if (!d)
          return std::nullopt;
        p = k ? "(bvmul " + p + " " + *d + ")" : *d;
      }
      return p;
    };
    size_t rank = layoutList(printed(resolveParam(params[3])), 0).size();
    if (rank)
      return product(rank);
    auto it = layoutWitnesses.find({"rank", layoutKey(params[3])});
    if (it == layoutWitnesses.end())
      return std::nullopt;
    std::string n = declare(Sort{false, 64, true}, "n");
    for (size_t r = 1; r <= 4; ++r)
      if (MaybeTerm p = product(r))
        facts.push_back("(=> (= " + it->second + " " + bvConst(r, 64) +
                        ") (= " + n + " " + *p + "))");
    return n;
  }

  /// `t.reshape(layout)` for a layout `row_major(Coord(...))` built: the
  /// view's dimensions are the layout's shape, and an obligation that it
  /// covers no more elements than `t` validly holds (the view reads from
  /// `t`'s first element on; assumed: `t` has no zero stride). Any other
  /// layout leaves the view's dimensions unknown, as before.
  void checkReshape(LIT::CallOp call, const CalleeName &name, State &state) {
    auto symbol = dyn_cast<SymbolConstantAttr>(call.getCallee());
    if (!symbol || call->getNumResults() != 1 || call.getNumOperands() != 2 ||
        !StringRef(name.path).starts_with(
            "layout::tile_tensor::TileTensor::reshape["))
      return;
    auto it = layoutShapes.find(tensorTerm(call.getOperands()[1], state));
    std::string view = term(call->getResult(0), state);
    if (it == layoutShapes.end() || sortOfTerm(view).isBool ||
        sortOfTerm(view).width != 64)
      return;
    if (!tensorDimDeclared) {
      prelude += "(declare-fun tdim ((_ BitVec 64) (_ BitVec 64)) "
                 "(_ BitVec 64))\n";
      tensorDimDeclared = true;
    }
    std::string size;
    for (auto [k, d] : llvm::enumerate(it->second)) {
      facts.push_back("(= (tdim " + view + " " + bvConst(k, 64) + ") " + d +
                      ")");
      size = k ? "(bvmul " + size + " " + d + ")" : d;
    }
    if (inContract || size.empty())
      return;
    Obligation ob{state.pc, "false", call.getLoc(), call.getLoc(),
                  "TileTensor.reshape"};
    ob.claim = "that the new layout covers no more elements than the tensor";
    if (MaybeTerm count =
            validCount(symbol.getParamValues(), call.getOperands()[0], state))
      ob.cond = define({true, 1, false}, "(bvsle " + size + " " + *count + ")");
    else
      ob.analyzed = false;
    noteCondition(ob.cond);
    obligations.push_back(ob);
  }

  /// The extent of the pointer value `p`, `Pointer._extent()`: how many
  /// elements are valid from it, an unknown per pointer, not negative.
  MaybeTerm pointerExtent(Value p, State &state) {
    std::string value = isa<LIT::RefType>(p.getType()) ? valueThrough(p, state)
                                                       : term(p, state);
    if (sortOfTerm(value).isBool || sortOfTerm(value).width != 64)
      return std::nullopt;
    return define(Sort{false, 64, true}, extentOf(value));
  }

  /// `pext(value)`, not negative.
  std::string extentOf(const std::string &value) {
    if (!pointerExtentDeclared) {
      prelude += "(declare-fun pext ((_ BitVec 64)) (_ BitVec 64))\n";
      pointerExtentDeclared = true;
    }
    std::string e = "(pext " + value + ")";
    facts.push_back("(bvsge " + e + " " + bvConst(0, 64) + ")");
    return e;
  }

  /// A struct with a single integer field and no parameters (an enum-like
  /// wrapper such as `GEMVAlgorithm`, a file descriptor): its values are
  /// represented by that integer. Null if `type` is not one.
  LIT::StructDeclOp intWrapper(Type type) {
    auto st = dyn_cast<LIT::StructType>(type);
    if (!st || !st.getParamValues().empty())
      return {};
    auto decl = dyn_cast_or_null<LIT::StructDeclOp>(
        symbols.lookupSymbolIn(module, st.getSymbol()));
    if (!decl)
      return {};
    SmallVector<LIT::StructFieldOp> fields;
    for (Region &region : decl->getRegions())
      for (Block &block : region)
        for (auto field : block.getOps<LIT::StructFieldOp>())
          fields.push_back(field);
    if (fields.size() != 1)
      return {};
    Type fieldType = fields.front().getType();
    Sort sort = sortOf(fieldType);
    if (!isScalar(fieldType) || sort.isBool || sort.width != 64)
      return {};
    return decl;
  }

  /// Whether `init` is the constructor of an integer wrapper from its one
  /// integer argument: it stores that argument into the field and returns
  /// the value, and does nothing else.
  bool isFieldInit(LIT::FnOp init) {
    if (init.getFunctionBody().empty() ||
        !init->getParentOfType<LIT::StructDeclOp>())
      return false;
    Block &body = init.getFunctionBody().front();
    if (body.getNumArguments() != 1 || init->getNumResults() > 1)
      return false;
    Value arg = body.getArgument(0);
    unsigned stores = 0;
    for (Operation &op : body) {
      StringRef opName = op.getName().getStringRef();
      if (opName == "lit.ref.store") {
        Value stored = op.getOperand(0);
        for (Operation *def = stored.getDefiningOp();
             def && def->getName().getStringRef() == "kgen.rebind";
             def = stored.getDefiningOp())
          stored = def->getOperand(0);
        if (stored != arg)
          return false;
        ++stores;
        continue;
      }
      if (!llvm::is_contained(
              {StringRef("lit.var.decl"), StringRef("lit.ref.struct.ger"),
               StringRef("kgen.rebind"), StringRef("lit.load.consume"),
               StringRef("lit.var.lifetime.start"),
               StringRef("lit.var.lifetime.end"), StringRef("hlcf.return"),
               StringRef("debuginfo.value"), StringRef("debuginfo.kill")},
              opName))
        return false;
    }
    if (stores != 1)
      return false;
    // The value returned is of an integer wrapper type.
    for (Operation &op : body)
      if (isa<HLCF::ReturnOp>(op))
        return op.getNumOperands() == 1 &&
               intWrapper(op.getOperand(0).getType());
    return false;
  }

  /// A call of an integer wrapper's own function: its constructor from the
  /// integer is that integer; a method whose body only reads fields,
  /// rebinds, calls and returns (`__eq__`, `__is__`, `__ne__`) is
  /// evaluated in place, with its arguments' values (so `k is
  /// GEMVAlgorithm.GEMV_KERNEL` is `k == 0`).
  bool evalIntWrapperCall(LIT::CallOp call, LIT::FnOp callee, State &state) {
    auto decl = callee->getParentOfType<LIT::StructDeclOp>();
    if (!decl || callee.getFunctionBody().empty() ||
        call->getNumResults() != 1)
      return false;
    // Only functions of integer wrappers.
    bool ofWrapper = false;
    for (Type t : callee.getFunctionBody().front().getArgumentTypes())
      ofWrapper = ofWrapper || intWrapper(t);
    if (!ofWrapper && !intWrapper(call->getResult(0).getType()))
      return false;
    if (isFieldInit(callee) && call.getNumOperands() == 1) {
      values[call->getResult(0)] = term(call.getOperands()[0], state);
      return true;
    }
    Block &body = callee.getFunctionBody().front();
    if (!llvm::hasSingleElement(callee.getFunctionBody()) ||
        body.getNumArguments() != call.getNumOperands() ||
        !body.getOps<RequiresOp>().empty())
      return false;
    for (Value operand : call.getOperands())
      if (isa<LIT::RefType>(operand.getType()))
        return false;
    for (Operation &op : body)
      if (!isa<LIT::StructExtractOp, LIT::CallOp, HLCF::ReturnOp>(op) &&
          !llvm::is_contained({StringRef("kgen.rebind"),
                               StringRef("kgen.param.constant")},
                              op.getName().getStringRef()))
        return false;
    for (auto [arg, operand] : llvm::zip(body.getArguments(), call.getOperands()))
      values[arg] = term(operand, state);
    for (Operation &op : body) {
      if (auto ret = dyn_cast<HLCF::ReturnOp>(op)) {
        if (ret.getNumOperands() != 1)
          return false;
        values[call->getResult(0)] = term(ret.getOperand(0), state);
        return true;
      }
      walkOp(&op, state);
    }
    return false;
  }

  /// `p[]`: the pointer's extent is at least 1, the precondition of
  /// `p[unsafe_offset=0]`. Stated here, not as a clause: one on `self` of
  /// this method (it returns a reference) makes code generation fail
  /// (`StackReuse`: "was supposed to be elidable").
  void checkDereference(LIT::CallOp call, const CalleeName &name,
                        LIT::FnOp callee, State &state) {
    if (!StringRef(name.path).starts_with(
            "std::memory::pointer::Pointer::__getitem__(::Pointer[") ||
        call.getNumOperands() != 1)
      return;
    Obligation ob{state.pc, "false", call.getLoc(),
                  callee ? callee.getLoc() : call.getLoc(),
                  "Pointer.__getitem__"};
    if (MaybeTerm e = pointerExtent(call.getOperands()[0], state)) {
      ob.cond = define({true, 1, false},
                       "(bvsle " + bvConst(1, 64) + " " + *e + ")");
      noteCondition(ob.cond);
    } else {
      ob.analyzed = false;
    }
    obligations.push_back(ob);
  }

  /// `Pointer._extent()`, and `p._offset_in_bounds[width](offset)`, the
  /// clause of accessing `width` elements at `offset`:
  /// `0 <= offset <= extent - width` (the width is the last parameter).
  /// Also the pointers whose extents no clause can state: `List.unsafe_ptr()`
  /// and `Span.unsafe_ptr()` point to at least `len(self)` elements (a
  /// list's capacity and a span's memory can be longer), `Array.unsafe_ptr()`
  /// to at least its size; casts of the origin or the address space (the
  /// implicit conversion of a mutable pointer to an immutable one, an
  /// ordinary call, among them) are the same pointer.
  bool evalPointer(LIT::CallOp call, const CalleeName &name, State &state) {
    StringRef path = name.path;
    auto symbol = dyn_cast<SymbolConstantAttr>(call.getCallee());
    if (!symbol || call->getNumResults() != 1)
      return false;
    Value result = call->getResult(0);
    if ((path.starts_with("std::collections::list::List::unsafe_ptr[") ||
         path.starts_with("std::collections::span::Span::unsafe_ptr(")) &&
        call.getNumOperands() == 1) {
      Value self = call.getOperands()[0];
      std::string value = isa<LIT::RefType>(self.getType())
                              ? valueThrough(self, state)
                              : term(self, state);
      if (sortOfTerm(value).isBool || sortOfTerm(value).width != 64)
        return false;
      std::string pointer = declare({false, 64, false}, "ptr");
      facts.push_back("(bvsge " + extentOf(pointer) + " " + lenOf(value) + ")");
      values[result] = pointer;
      return true;
    }
    if (path.starts_with("std::collections::array::Array::unsafe_ptr[") &&
        call.getNumOperands() == 1) {
      std::optional<Loc> loc = placeOf(call.getOperands()[0]);
      MaybeTerm n;
      if (loc && loc->path.empty())
        n = arrayLength(*loc);
      // An array that is a field (`self.a_engine.unsafe_ptr()`): its length
      // is in the type of the reference it is called on.
      if (!n)
        if (auto ref = dyn_cast<LIT::RefType>(call.getOperands()[0].getType()))
          if (auto type = dyn_cast<LIT::StructType>(ref.getElementType());
              type && type.getParamValues().size() == 2 &&
              printed(type.getSymbol()) == "@std::@collections::@array::@Array") {
            TypedAttr size = type.getParamValues()[1];
            Sort sort = sortOf(size.getType());
            if (!sort.isBool && sort.width == 64)
              n = paramTerm(size, sort);
          }
      if (!n)
        return false;
      std::string pointer = declare({false, 64, false}, "ptr");
      facts.push_back("(bvsge " + extentOf(pointer) + " " + *n + ")");
      values[result] = pointer;
      return true;
    }
    if (!path.consume_front("std::memory::pointer::Pointer::"))
      return false;
    // Casts that keep the element type and the address. (`unsafe_bitcast`
    // changes the element type, so its extent stays unknown.)
    bool samePointer =
        (path.starts_with("__init__[") && path.contains("](::Pointer[")) ||
        path.starts_with("as_imm(") ||
        path.starts_with("as_unsafe_any_origin(") ||
        path.starts_with("unsafe_as_noalias(") ||
        path.starts_with("unsafe_mut_cast[") ||
        path.starts_with("unsafe_origin_cast[") ||
        path.starts_with("address_space_cast[") ||
        path.starts_with("unsafe_address_space_cast[");
    if (samePointer && call.getNumOperands() == 1 &&
        !isa<LIT::RefType>(call.getOperands()[0].getType())) {
      values[result] = term(call.getOperands()[0], state);
      return true;
    }
    // `Pointer(to=x)` points to at least `x` (an element of an array may
    // have more after it). Not a clause: one on this constructor makes the
    // compile-time interpreter read a null `to` (`Pointer(to=...)` in
    // `Tuple.__getitem__`).
    if (path.starts_with("__init__(to:") && call.getNumOperands() == 1) {
      std::string pointer = declare({false, 64, false}, "ptr");
      facts.push_back("(bvsge " + extentOf(pointer) + " " + bvConst(1, 64) +
                      ")");
      values[result] = pointer;
      return true;
    }
    if (path.starts_with("_extent(") && call.getNumOperands() == 1) {
      MaybeTerm e = pointerExtent(call.getOperands()[0], state);
      if (!e)
        return false;
      values[result] = *e;
      return true;
    }
    if (path.starts_with("_offset_in_bounds[") && call.getNumOperands() == 2 &&
        !symbol.getParamValues().empty()) {
      MaybeTerm e = pointerExtent(call.getOperands()[0], state);
      std::string width =
          paramTerm(symbol.getParamValues().back(), Sort{false, 64, true});
      std::string offset = term(call.getOperands()[1], state);
      if (!e || sortOfTerm(width).width != 64 || sortOfTerm(width).isBool ||
          sortOfTerm(offset).width != 64 || sortOfTerm(offset).isBool)
        return false;
      values[result] =
          define({true, 1, false}, "(and (bvsle " + bvConst(0, 64) + " " +
                                       offset + ") (bvsle " + offset +
                                       " (bvsub " + *e + " " + width + ")))");
      return true;
    }
    return false;
  }

  /// `ceildiv(a, b)` of two integers (`Int`, or a `SIMD` scalar of an
  /// integer dtype): their `__ceildiv__`, which it calls. Generic over its
  /// type, it returns through an out slot, its last operand.
  bool evalCeildiv(LIT::CallOp call, const CalleeName &name, State &state) {
    auto symbol = dyn_cast<SymbolConstantAttr>(call.getCallee());
    if (!StringRef(name.path).starts_with("std::math::math::ceildiv[") ||
        !symbol || symbol.getParamValues().size() != 1 ||
        call.getNumOperands() != 3)
      return false;
    std::optional<Loc> out = placeOf(call.getOperands()[2]);
    if (!out)
      return false;
    TypedAttr type = resolveParam(symbol.getParamValues()[0]);
    std::optional<Sort> sort;
    if (isIntType(type))
      sort = Sort{false, 64, true};
    else if (auto param = dyn_cast<TypeParamAttr>(type))
      if (std::string text = printed(param.getTypeValue());
          StringRef(text).starts_with("!lit.struct<@std::@simd::@SIMD<") &&
          isWidthOne(text))
        sort = dtypeSort(text);
    if (!sort || sort->isBool)
      return false;
    SmallVector<std::string, 2> args;
    for (Value v : call.getOperands().take_front(2)) {
      std::string a = isa<LIT::RefType>(v.getType()) ? valueThrough(v, state)
                                                     : term(v, state);
      if (sortOfTerm(a).isBool || sortOfTerm(a).width != sort->width)
        return false;
      args.push_back(a);
    }
    checkDivisor(call, args[1], state);
    store(*out, state, define(*sort, ceilDivision(*sort, args[0], args[1])));
    setResultsUnknown(call);
    return true;
  }

  /// `min(a, b)` and `max(a, b)` of two integers (`Int`, or a `SIMD` scalar
  /// of an integer dtype): the smaller and the larger, as their dtype
  /// compares. `align_down(a, b)` and `align_up(a, b)` of two such: `a // b`
  /// and `ceildiv(a, b)` times `b`, as the stdlib computes them (by zero as
  /// `//` is here).
  bool evalMinMax(LIT::CallOp call, const CalleeName &name, State &state) {
    StringRef path = name.path;
    bool isMin = path ==
                 "std::math::math::min[::DType,::SIMDLength](::SIMD[$0, "
                 "$1],::SIMD[$0, $1])";
    bool isMax = path ==
                 "std::math::math::max[::DType,::SIMDLength](::SIMD[$0, "
                 "$1],::SIMD[$0, $1])";
    bool isDown = path.starts_with(
        "std::math::math::align_down[::DType,::SIMDLength](::SIMD[$0, $1],"
        "::SIMD[$0, $1])");
    bool isUp = path.starts_with(
        "std::math::math::align_up[::DType,::SIMDLength](::SIMD[$0, $1],"
        "::SIMD[$0, $1])");
    if ((!isMin && !isMax && !isDown && !isUp) || call.getNumOperands() != 2 ||
        call->getNumResults() != 1)
      return false;
    std::string type = printed(call->getResult(0).getType());
    if (!StringRef(type).starts_with("!lit.struct<@std::@simd::@SIMD<") ||
        !isWidthOne(type))
      return false;
    std::optional<Sort> sort = dtypeSort(type);
    if (!sort || sort->isBool)
      return false;
    SmallVector<std::string, 2> args;
    for (Value v : call.getOperands()) {
      std::string a = isa<LIT::RefType>(v.getType()) ? valueThrough(v, state)
                                                     : term(v, state);
      if (sortOfTerm(a).isBool || sortOfTerm(a).width != sort->width)
        return false;
      args.push_back(a);
    }
    if (isDown || isUp) {
      std::string quotient =
          isDown ? floorDivision(/*remainder=*/false, *sort, args[0], args[1])
                 : ceilDivision(*sort, args[0], args[1]);
      values[call->getResult(0)] =
          define(*sort, "(bvmul " + quotient + " " + args[1] + ")");
      return true;
    }
    std::string less =
        std::string(sort->isSigned ? "(bvslt " : "(bvult ") +
        (isMin ? args[0] + " " + args[1] : args[1] + " " + args[0]) + ")";
    values[call->getResult(0)] =
        define(*sort, "(ite " + less + " " + args[0] + " " + args[1] + ")");
    return true;
  }

  /// `divmod(a, b)` of two `Int`s and `udivmod(a, b)` (and its unchecked
  /// form): a tuple of the quotient and the remainder (fields `/0`, `/1`),
  /// rounding toward negative infinity, or for `udivmod` of the arguments
  /// as unsigned. Division by zero is as for `//` and `%`.
  bool evalDivmod(LIT::CallOp call, const CalleeName &name, State &state) {
    StringRef path = name.path;
    bool isUnsigned =
        path == "std::math::uutils::udivmod(::SIMD[DType.int, 1],::SIMD[DType."
                "int, 1])" ||
        path == "std::math::uutils::udivmod_unchecked(::SIMD[DType.int, 1],::"
                "SIMD[DType.int, 1])";
    bool isSigned = false;
    if (!isUnsigned && path.starts_with("std::math::math::divmod[")) {
      auto symbol = dyn_cast<SymbolConstantAttr>(call.getCallee());
      isSigned = symbol && symbol.getParamValues().size() == 1 &&
                 isIntType(resolveParam(symbol.getParamValues()[0]));
    }
    if ((!isUnsigned && !isSigned) || call.getNumOperands() != 2 ||
        call->getNumResults() != 1)
      return false;
    SmallVector<std::string, 2> args;
    for (Value v : call.getOperands())
      args.push_back(isa<LIT::RefType>(v.getType()) ? valueThrough(v, state)
                                                    : term(v, state));
    Sort sort{false, 64, true};
    for (const std::string &a : args)
      if (sortOfTerm(a).isBool || sortOfTerm(a).width != 64)
        return false;
    Sort as{false, 64, !isUnsigned};
    // `udivmod_unchecked` states `b > 0` as a precondition.
    if (!path.contains("_unchecked("))
      checkDivisor(call, args[1], state);
    std::string built = declare({false, 64, false}, "g");
    std::string q = define(sort, floorDivision(false, as, args[0], args[1]));
    std::string r = define(sort, floorDivision(true, as, args[0], args[1]));
    setBuiltField(built, "/0", q);
    setBuiltField(built, "/1", r);
    // What a solver needs to bound them (division by a symbolic divisor is
    // nonlinear): unsigned, `q * b + r == a` and `r < b` for `b != 0`.
    if (isUnsigned)
      facts.push_back("(=> (distinct " + args[1] + " " + bvConst(0, 64) +
                      ") (and (= (bvadd (bvmul " + q + " " + args[1] + ") " +
                      r + ") " + args[0] + ") (bvult " + r + " " + args[1] +
                      ") (bvule " + q + " " + args[0] + ")))");
    values[call->getResult(0)] = built;
    return true;
  }

  /// `ufloordiv(a, b)`, `udiv_unchecked(a, b)` and `uceildiv(a, b)` of two
  /// `Int`s: bounded, `0 <= q <= a` for `a >= 0`, which holds for any
  /// divisor allowed (`q` is 0 for a zero one; `udiv_unchecked` requires
  /// `b > 0`). Not exact: an exact quotient times
  /// an unknown made a Houdini script of a MAX kernel take 909 s instead of
  /// 4 s (11.7 s as shift and mask by the literal 32).
  bool evalUnsignedDivision(LIT::CallOp call, const CalleeName &name,
                            State &state) {
    StringRef path = name.path;
    StringRef args2 = "(::SIMD[DType.int, 1],::SIMD[DType.int, 1])";
    if (path != ("std::math::uutils::ufloordiv" + args2).str() &&
        path != ("std::math::uutils::udiv_unchecked" + args2).str() &&
        path != ("std::math::uutils::uceildiv" + args2).str())
      return false;
    if (call.getNumOperands() != 2 || call->getNumResults() != 1)
      return false;
    Value dividend = call.getOperands()[0];
    std::string a = isa<LIT::RefType>(dividend.getType())
                        ? valueThrough(dividend, state)
                        : term(dividend, state);
    if (sortOfTerm(a).isBool || sortOfTerm(a).width != 64)
      return false;
    // `udiv_unchecked` states `b > 0` as a precondition (by 0 it is
    // undefined); the others return 0 for it.
    if (!path.starts_with("std::math::uutils::udiv_unchecked")) {
      Value divisor = call.getOperands()[1];
      checkDivisor(call,
                   isa<LIT::RefType>(divisor.getType())
                       ? valueThrough(divisor, state)
                       : term(divisor, state),
                   state);
    }
    std::string q = declare({false, 64, true}, "q");
    facts.push_back("(=> (bvsle " + bvConst(0, 64) + " " + a +
                    ") (and (bvsle " + bvConst(0, 64) + " " + q + ") (bvsle " +
                    q + " " + a + ")))");
    // By 0 the quotient is 0 (`udiv_unchecked` requires `b > 0` instead).
    Value divisorValue = call.getOperands()[1];
    std::string b = isa<LIT::RefType>(divisorValue.getType())
                        ? valueThrough(divisorValue, state)
                        : term(divisorValue, state);
    if (!sortOfTerm(b).isBool && sortOfTerm(b).width == 64) {
      facts.push_back("(=> (= " + b + " " + bvConst(0, 64) + ") (= " + q +
                      " " + bvConst(0, 64) + "))");
      std::string exact = path.starts_with("std::math::uutils::uceildiv")
                              ? ceilDivision(Sort{false, 64, false}, a, b)
                              : "(bvudiv " + a + " " + b + ")";
      caseQuotients.insert(q);
      caseFacts.push_back("(=> (and (bvsle " + bvConst(0, 64) + " " + a +
                          ") (bvsgt " + b + " " + bvConst(0, 64) + ")) (= " +
                          q + " " + exact + "))");
    }
    values[call->getResult(0)] = q;
    return true;
  }

  /// `Dim(x)`, `Dim(x, y)`, `Dim(x, y, z)` and `Dim(tuple)` from MAX's GPU
  /// host API: fields `/x`, `/y` and `/z`, an omitted axis 1. An argument is
  /// an `Int` (its value) or an `IntLiteral` (its value, in its type); a
  /// tuple's elements are the fields its literal was built with.
  bool evalDim(LIT::CallOp call, const CalleeName &name, State &state) {
    StringRef path = name.path;
    auto symbol = dyn_cast<SymbolConstantAttr>(call.getCallee());
    if (!path.starts_with("max::gpu::host::dim::Dim::__init__[") || !symbol ||
        call->getNumResults() != 1)
      return false;
    ArrayRef<TypedAttr> types = symbol.getParamValues();
    bool fromTuple = path.contains("](::Tuple[");
    if (call.getNumOperands() != (fromTuple ? 1 : types.size()) ||
        types.empty() || types.size() > 3)
      return false;
    std::string tuple;
    if (fromTuple) {
      Value t = call.getOperands()[0];
      tuple = isa<LIT::RefType>(t.getType()) ? valueThrough(t, state)
                                             : term(t, state);
    }
    SmallVector<std::string, 3> axes;
    for (auto [k, param] : llvm::enumerate(types)) {
      TypedAttr type = resolveParam(param);
      std::string v;
      if (std::optional<int64_t> n = intLiteralType(type)) {
        v = bvConst(*n, 64);
      } else if (!isIntType(type)) {
        return false;
      } else if (fromTuple) {
        auto it = builtFields.find({tuple, "/" + std::to_string(k)});
        if (it == builtFields.end())
          return false;
        v = it->second;
      } else {
        Value x = call.getOperands()[k];
        v = isa<LIT::RefType>(x.getType()) ? valueThrough(x, state)
                                           : term(x, state);
      }
      if (sortOfTerm(v).isBool || sortOfTerm(v).width != 64)
        return false;
      axes.push_back(v);
    }
    while (axes.size() < 3)
      axes.push_back(bvConst(1, 64));
    std::string built = declare({false, 64, false}, "g");
    for (auto [k, axis] : llvm::enumerate(axes))
      setBuiltField(built, std::string("/") + "xyz"[k], axis);
    values[call->getResult(0)] = built;
    return true;
  }

  /// A kernel launch, `ctx.enqueue_function[kernel](*args, grid_dim=g,
  /// block_dim=b)`: the kernel's preconditions are obligations here, with
  /// its arguments the launch's and its `grid_dim` and `block_dim` the
  /// launch's dimensions. They are not analyzed where the arguments or
  /// dimensions are not known.
  void checkLaunch(LIT::CallOp call, const CalleeName &name, State &state) {
    auto symbol = dyn_cast<SymbolConstantAttr>(call.getCallee());
    if (!symbol ||
        !StringRef(name.path).contains("DeviceContext::enqueue_function[") ||
        call.getNumOperands() < 4)
      return;
    auto [kernel, kernelSymbol] =
        launchedKernel(symbol, module, symbols,
                       [&](TypedAttr param) { return resolveParam(param); });
    if (!kernel || kernel.getFunctionBody().empty())
      return;
    // The arguments: the references of the pack, the first operand that is
    // one (after `self`, and after the compiled function `f` in
    // `enqueue_function(f, *args, ...)`).
    auto valueOf = [&](Value v) {
      return isa<LIT::RefType>(v.getType()) ? valueThrough(v, state)
                                            : term(v, state);
    };
    SmallVector<Value> refs;
    bool known = false;
    size_t next = call.getNumOperands();
    for (size_t i = 1; i < call.getNumOperands(); ++i)
      if (auto it = packRefs.find(valueOf(call.getOperands()[i]));
          it != packRefs.end()) {
        refs = it->second;
        known = true;
        next = i + 1;
        break;
      }
    // The dimensions: the first two `Dim`s after it, as `Dim` built them.
    // Operands between the pack and them are host arguments
    // (`host_arg=`), the kernel's last arguments.
    SmallVector<Value, 2> dimOperands;
    for (size_t i = next; i < call.getNumOperands() && dimOperands.size() < 2;
         ++i) {
      Type type = call.getOperands()[i].getType();
      if (auto ref = dyn_cast<LIT::RefType>(type))
        type = ref.getElementType();
      if (StringRef(printed(type)).contains("@max::@gpu::@host::@dim::@Dim"))
        dimOperands.push_back(call.getOperands()[i]);
      else if (dimOperands.empty() && known)
        refs.push_back(call.getOperands()[i]);
    }
    if (dimOperands.size() != 2)
      known = false;
    // A `DeviceBuffer` passed to the kernel becomes a pointer to its
    // elements (its `device_type`): the extent of that pointer is the
    // buffer's length.
    for (Value ref : refs) {
      Type type = ref.getType();
      if (auto r = dyn_cast<LIT::RefType>(type))
        type = r.getElementType();
      if (!StringRef(printed(type))
               .contains("@max::@gpu::@host::@device_context::@DeviceBuffer<"))
        continue;
      std::string value = valueOf(ref);
      if (!sortOfTerm(value).isBool && sortOfTerm(value).width == 64)
        facts.push_back("(= " + extentOf(value) + " " + lenOf(value) + ")");
    }
    std::map<std::string, std::string> dims;
    const char *kinds[] = {"grid_dim_", "block_dim_"};
    for (auto [k, dim] : llvm::enumerate(dimOperands)) {
      std::string value = valueOf(dim);
      for (char axis : {'x', 'y', 'z'}) {
        auto it = builtFields.find({value, std::string("/") + axis});
        if (it == builtFields.end())
          known = false;
        else
          dims[std::string(kinds[k]) + axis] = it->second;
      }
    }
    ParamFrame frame = paramFrame(kernelSymbol, kernel, 0, ParamFrame{});
    frame.parent = params;
    // A launch whose instantiation is not verified (`opaqueParameter`,
    // reported where the instantiations are): its clauses, over the same
    // unknown values, are not asked either.
    {
      ParamFrame *saved = params;
      params = &frame;
      bool opaque = opaqueParameterOf(kernel).has_value();
      params = saved;
      if (opaque)
        return;
    }
    for (RequiresOp req :
         kernel.getFunctionBody().front().getOps<RequiresOp>()) {
      Obligation ob{state.pc, "false", call.getLoc(), req.getLoc(),
                    displayName(kernel)};
      MaybeTerm cond;
      if (known) {
        ParamFrame *saved = params;
        params = &frame;
        launchDims = &dims;
        cond = instantiate(req.getBody(), req.getArgs(), state,
                           ValueRange(refs), kernel, nullptr, false);
        launchDims = nullptr;
        params = saved;
      }
      if (cond) {
        ob.cond = *cond;
        noteCondition(ob.cond);
      } else {
        ob.analyzed = false;
      }
      obligations.push_back(ob);
    }
    // The kernel's hardware contract, its top-level `comptime assert`s that
    // name the target (a GPU predicate, `WARP_SIZE`), for the target it is
    // compiled for: the build's accelerator (assumption). The launcher runs
    // on the host (`DeviceContext` is host-only). Its other asserts are the
    // compiler's to check where the kernel is compiled.
    for (auto assertion :
         kernel.getFunctionBody().front().getOps<ParamAssertOp>()) {
      Obligation ob{state.pc, "false", call.getLoc(), assertion.getLoc(),
                    displayName(kernel)};
      ob.claim = "the assertion of '" + displayName(kernel) +
                 "' for the accelerator it is launched on";
      ParamFrame *saved = params;
      params = &frame;
      deviceView = true;
      std::string cond = paramTerm(assertion.getCond(), {true, 1, false});
      deviceView = false;
      params = saved;
      if (!mentionsTarget(cond))
        continue;
      std::string host = targetPredicate("std::sys::info::is_gpu()",
                                         {true, 1, false})
                             .value_or("false");
      ob.pc = define({true, 1, false},
                     "(and " + state.pc + " (not " + host + "))", "r");
      ob.cond = cond;
      noteCondition(ob.cond);
      obligations.push_back(ob);
    }
  }

  /// `warp.broadcast(x)` and the unmasked `shuffle_idx/up/down/xor(x, k)`:
  /// `x` as another lane of the warp has it (or this lane's own), so as
  /// SOME thread of the same block has it. That is `x`'s term with what
  /// varies by thread renamed to fresh copies (`inOtherThread`); which lane
  /// is not needed. Where `x` depends on what cannot be copied (a load, an
  /// unknown result), the value stays unknown.
  bool evalWarpShuffle(LIT::CallOp call, const CalleeName &name,
                       State &state) {
    StringRef path = name.path;
    const char *prefix = "std::_gpu::primitives::warp::";
    if (!path.consume_front(prefix) || call->getNumResults() != 1)
      return false;
    unsigned operands = path.starts_with("broadcast[") ? 1 : 2;
    if (!(path.starts_with("broadcast[") || path.starts_with("shuffle_idx[") ||
          path.starts_with("shuffle_up[") ||
          path.starts_with("shuffle_down[") ||
          path.starts_with("shuffle_xor[")) ||
        call.getNumOperands() != operands)
      return false;
    Value val = call.getOperands()[0];
    if (isa<LIT::RefType>(val.getType()))
      return false;
    std::string x = term(val, state);
    Sort sort = sortOfTerm(x);
    if (sort.isBool || sort.width != sortOf(call->getResult(0).getType()).width)
      return false;
    if (MaybeTerm other = inOtherThread(x))
      values[call->getResult(0)] = *other;
    else
      setResultsUnknown(call);
    return true;
  }

  /// `term` as another thread of the same block computes it: kernel
  /// parameters, block ids and dimensions and the target are the same;
  /// thread ids, `lane_id`, arguments (a device helper may be called with
  /// thread-dependent ones) and bounded quotients become fresh copies, with
  /// every fact about them copied too. Nothing if `term` depends on any
  /// other unknown.
  MaybeTerm inOtherThread(const std::string &term) {
    static llvm::Regex uniform(
        "^(p[0-9]+|gpu_(block_idx|block_dim|grid_dim)_[xyz]|target_.*)$");
    static llvm::Regex varying("^(a[0-9]+|q[0-9]+|gpu_thread_idx_[xyz]|"
                               "gpu_lane_id)$");
    std::map<std::string, std::string> copied;
    std::set<std::string> uncopyable; // Never shared: copying them fails.
    bool failed = false;
    std::function<std::string(StringRef)> rewrite;
    auto copyName = [&](const std::string &name) -> std::string {
      if (auto it = copied.find(name); it != copied.end())
        return it->second;
      if (uncopyable.count(name)) {
        failed = true;
        return name;
      }
      auto sort = sorts.find(name);
      if (sort == sorts.end())
        return name; // A function or keyword.
      std::string result = name;
      if (auto def = definitions.find(name); def != definitions.end()) {
        bool before = failed;
        failed = false;
        std::string expr = rewrite(def->second);
        if (failed) {
          uncopyable.insert(name);
          return name;
        }
        failed = before;
        if (expr != def->second)
          result = define(sort->second, expr);
      } else if (uniform.match(name)) {
        result = name;
      } else if (varying.match(name)) {
        result = declare(sort->second, "o");
      } else {
        uncopyable.insert(name);
        failed = true;
        return name;
      }
      copied[name] = result;
      return result;
    };
    rewrite = [&](StringRef expr) -> std::string {
      std::string out;
      size_t i = 0;
      while (i < expr.size() && !failed) {
        char c = expr[i];
        if (isalpha(static_cast<unsigned char>(c)) || c == '_') {
          size_t j = i;
          while (j < expr.size() &&
                 (isalnum(static_cast<unsigned char>(expr[j])) ||
                  expr[j] == '_'))
            ++j;
          out += copyName(expr.slice(i, j).str());
          i = j;
        } else {
          out += c;
          ++i;
        }
      }
      return out;
    };
    std::string result = rewrite(term);
    if (failed)
      return std::nullopt;
    // The facts about what was renamed hold for the other thread too (they
    // are the semantics of ids and models, not this thread's path). Copying
    // one may rename more; repeat until nothing changes.
    std::set<size_t> done;
    for (bool changed = true; changed;) {
      changed = false;
      for (size_t k = 0; k < facts.size(); ++k) {
        if (done.count(k))
          continue;
        bool mentions = false;
        StringRef fact(facts[k]);
        for (size_t i = 0; i < fact.size() && !mentions;) {
          if (!isalpha(static_cast<unsigned char>(fact[i])) && fact[i] != '_') {
            ++i;
            continue;
          }
          size_t j = i;
          while (j < fact.size() &&
                 (isalnum(static_cast<unsigned char>(fact[j])) ||
                  fact[j] == '_'))
            ++j;
          auto it = copied.find(fact.slice(i, j).str());
          mentions = it != copied.end() && it->second != it->first;
          i = j;
        }
        if (!mentions)
          continue;
        done.insert(k);
        size_t before = copied.size();
        std::string original = facts[k];
        std::string copy = rewrite(original);
        if (failed) {
          failed = false; // This fact names what cannot be copied: skip it.
          continue;
        }
        if (copy != original)
          facts.push_back(copy);
        changed = changed || copied.size() != before;
      }
    }
    return result;
  }

  /// The unknown of a GPU id on an axis (`thread_idx`, `block_idx`,
  /// `block_dim`, `grid_dim`), declared with its limits at its first use.
  std::string gpuIdSymbol(StringRef which, char axis) {
    std::string symbol = ("gpu_" + which + "_" + Twine(axis)).str();
    if (gpuIds.insert(symbol).second) {
      prelude += "(declare-const " + symbol + " (_ BitVec 64))\n";
      sorts[symbol] = Sort{false, 64, true};
      std::string dim = ("gpu_block_dim_" + Twine(axis)).str();
      std::string grid = ("gpu_grid_dim_" + Twine(axis)).str();
      if (which == "thread_idx")
        facts.push_back("(and (bvsle " + bvConst(0, 64) + " " + symbol +
                        ") (bvslt " + symbol + " " + dim + "))");
      else if (which == "block_idx")
        facts.push_back("(and (bvsle " + bvConst(0, 64) + " " + symbol +
                        ") (bvslt " + symbol + " " + grid + "))");
      else if (which == "block_dim")
        facts.push_back("(and (bvsle " + bvConst(1, 64) + " " + symbol +
                        ") (bvsle " + symbol + " " + bvConst(1024, 64) + "))");
      else
        facts.push_back("(and (bvsle " + bvConst(1, 64) + " " + symbol +
                        ") (bvslt " + symbol + " " +
                        bvConst(int64_t(1) << 31, 64) + "))");
    }
    return symbol;
  }

  /// GPU ids (`thread_idx.x`, `block_idx.y`, `block_dim.z`, `grid_dim.x`,
  /// `global_idx.x`): one value per id and axis in a function, so reads
  /// agree, with the launch limits every supported GPU has (assumptions):
  /// `0 <= thread_idx < block_dim <= 1024` and `0 <= block_idx < grid_dim <
  /// 2^31`. `global_idx` is `block_idx * block_dim + thread_idx`, as the
  /// stdlib defines it.
  bool evalGpuId(LIT::CallOp call, const CalleeName &name) {
    StringRef path = name.path;
    const char *prefix = "std::_gpu::primitives::id::";
    if (!path.starts_with(prefix) || call->getNumResults() != 1)
      return false;
    StringRef rest = path.drop_front(strlen(prefix));
    // `lane_id()`: below the warp size (`WARP_SIZE`), 32 or 64 on
    // supported GPUs.
    if (rest.starts_with("lane_id(") && call.getNumOperands() == 0 &&
        !launchDims) {
      std::string symbol = "gpu_lane_id";
      if (gpuIds.insert(symbol).second) {
        prelude += "(declare-const " + symbol + " (_ BitVec 64))\n";
        sorts[symbol] = Sort{false, 64, true};
        facts.push_back("(and (bvsle " + bvConst(0, 64) + " " + symbol +
                        ") (bvslt " + symbol + " " + bvConst(64, 64) + "))");
        if (MaybeTerm warp = targetPredicate(
                "std::_gpu::globals::_resolve_warp_size()",
                Sort{false, 64, true}))
          facts.push_back("(=> (bvsgt " + *warp + " " + bvConst(0, 64) +
                          ") (bvslt " + symbol + " " + *warp + "))");
      }
      values[call->getResult(0)] = symbol;
      return true;
    }
    // `warp_id()`: `thread_idx.x` divided by the warp size, unsigned, for
    // a warp size of 32 or 64. Broadcast within the warp, it is the same
    // value.
    if ((rest.starts_with("warp_id[") || rest.starts_with("_warp_id[")) &&
        call.getNumOperands() == 0 && !launchDims &&
        sortOf(call->getResult(0).getType()).width == 64) {
      MaybeTerm warp = targetPredicate(
          "std::_gpu::globals::_resolve_warp_size()", Sort{false, 64, true});
      if (!warp)
        return false;
      gpuIdSymbol("block_dim", 'x');
      gpuIdSymbol("grid_dim", 'x');
      std::string thread = gpuIdSymbol("thread_idx", 'x');
      // By shifts for the warp sizes GPUs have, and unknown for any other:
      // a quotient of two unknowns made every query that reads it a
      // nonlinear one, asked three times (blockwise_fp8: 150M more work to
      // leave the same obligations open).
      std::string other = declare(Sort{false, 64, true}, "u");
      values[call->getResult(0)] =
          define(Sort{false, 64, true},
                 "(ite (= " + *warp + " " + bvConst(32, 64) + ") (bvlshr " +
                     thread + " " + bvConst(5, 64) + ") (ite (= " + *warp +
                     " " + bvConst(64, 64) + ") (bvlshr " + thread + " " +
                     bvConst(6, 64) + ") " + other + "))");
      return true;
    }
    StringRef kind = rest.take_until([](char c) { return c == ':'; });
    if (!rest.drop_front(kind.size()).starts_with("::__getattr_param__["))
      return false;
    // The axis: the string parameter (`"x"`).
    std::optional<char> axis;
    for (const std::string &param : name.params)
      for (const char *a : {"\"x\"", "\"y\"", "\"z\""})
        if (StringRef(param).contains(a))
          axis = a[1];
    if (!axis || sortOf(call->getResult(0).getType()).width != 64)
      return false;
    // In a kernel's clause at its launch: the launch's dimensions; the
    // indices of a thread are unknowns, which only makes the clause harder
    // to prove.
    if (launchDims) {
      std::string key;
      if (kind == "_BlockDim")
        key = "block_dim_";
      else if (kind == "_GridDim")
        key = "grid_dim_";
      key += *axis;
      auto it = launchDims->find(key);
      values[call->getResult(0)] =
          it != launchDims->end() ? it->second : declare(Sort{false, 64, true});
      return true;
    }
    auto id = [&](StringRef which) { return gpuIdSymbol(which, *axis); };
    // Declare the dimensions first: the ids' facts name them.
    id("block_dim");
    id("grid_dim");
    Value result = call->getResult(0);
    if (kind == "_ThreadIdx")
      values[result] = id("thread_idx");
    else if (kind == "_BlockIdx")
      values[result] = id("block_idx");
    else if (kind == "_BlockDim")
      values[result] = id("block_dim");
    else if (kind == "_GridDim")
      values[result] = id("grid_dim");
    else if (kind == "_GlobalIdx")
      values[result] = define(Sort{false, 64, true},
                              "(bvadd (bvmul " + id("block_idx") + " " +
                                  id("block_dim") + ") " + id("thread_idx") +
                                  ")");
    else
      return false;
    return true;
  }

  /// `List.unsafe_set(i, v)`: writing element `i`, as `xs[i] = v` does.
  bool evalElementWrite(LIT::CallOp call, const CalleeName &name,
                        State &state) {
    if (!StringRef(name.path).starts_with(
            "std::collections::list::List::unsafe_set(") ||
        call.getNumOperands() != 3 || call->getNumResults() != 0)
      return false;
    std::optional<Loc> list = placeOf(call.getOperands()[0]);
    Value idx = call.getOperands()[1], value = call.getOperands()[2];
    if (!list || sortOfTerm(term(idx, state)).width != 64)
      return false;
    storeElement(*list, term(idx, state),
                 isa<LIT::RefType>(value.getType()) ? valueThrough(value, state)
                                                    : term(value, state),
                 state);
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
    if ((path.starts_with("std::builtin::bool::Bool::__mlir_bool__(") ||
         path.starts_with("std::builtin::bool::Bool::__bool__(")) &&
        call.getNumOperands() == 1) {
      values[result] = operand(0);
      return true;
    }
    // `Bool`'s operators (`not x` is `__invert__`).
    if (path.starts_with("std::builtin::bool::Bool::__") &&
        sortOf(result.getType()).isBool) {
      StringRef method = path.drop_front(strlen("std::builtin::bool::Bool::"))
                             .take_until([](char c) { return c == '('; });
      static const std::pair<const char *, const char *> binary[] = {
          {"__and__", "and"},
          {"__or__", "or"},
          {"__xor__", "xor"},
          {"__eq__", "="}};
      if (call.getNumOperands() == 1 && method == "__invert__") {
        values[result] = define({true, 1, false}, "(not " + operand(0) + ")");
        return true;
      }
      if (call.getNumOperands() == 2) {
        for (auto &[m, smt] : binary)
          if (method == m) {
            values[result] = define({true, 1, false},
                                    "(" + std::string(smt) + " " + operand(0) +
                                        " " + operand(1) + ")");
            return true;
          }
        if (method == "__ne__") {
          values[result] =
              define({true, 1, false},
                     "(not (= " + operand(0) + " " + operand(1) + "))");
          return true;
        }
      }
    }
    if (path.starts_with("std::builtin::len::len[") &&
        call.getNumOperands() == 1) {
      // An `Array`'s length is its size parameter, in its type (an
      // expression in the parameters of the function the place is in).
      if (std::optional<Loc> loc = placeOf(call.getOperands()[0]);
          loc && loc->path.empty())
        if (MaybeTerm n = arrayLength(*loc)) {
          values[result] = *n;
          return true;
        }
      values[result] = define(
          {false, 64, true}, lenOf(valueThrough(call.getOperands()[0], state)));
      return true;
    }
    if (evalRange(call, name, state))
      return true;
    if (evalOptional(call, name, state))
      return true;
    // A string's length in bytes: `len` of its value (`String` has no
    // `__len__` of its own); a literal's, from its type.
    if ((path.starts_with(
             "std::collections::string::string::String::byte_length(") ||
         path.starts_with("std::collections::string::string_span::StringSpan::"
                          "byte_length(")) &&
        call.getNumOperands() == 1) {
      Value self = call.getOperands()[0];
      std::string value = isa<LIT::RefType>(self.getType())
                              ? valueThrough(self, state)
                              : term(self, state);
      values[result] = define({false, 64, true}, lenOf(value));
      return true;
    }
    if (path.starts_with(
            "std::builtin::string_literal::StringLiteral::byte_length("))
      if (auto symbol = dyn_cast<SymbolConstantAttr>(call.getCallee());
          symbol && symbol.getParamValues().size() == 1)
        if (std::optional<int64_t> n =
                stringLength(resolveParam(symbol.getParamValues()[0]))) {
          values[result] = bvConst(*n, 64);
          return true;
        }
    // `index(x)` of an `Int` (passed by reference): `x`; of an
    // `IntLiteral`: its value, in its type.
    if (path.starts_with("std::builtin::int::index[") &&
        call.getNumOperands() == 1 &&
        isa<LIT::RefType>(call.getOperands()[0].getType()))
      if (auto symbol = dyn_cast<SymbolConstantAttr>(call.getCallee());
          symbol && symbol.getParamValues().size() == 1) {
        TypedAttr type = resolveParam(symbol.getParamValues()[0]);
        if (isIntType(type)) {
          values[result] = valueThrough(call.getOperands()[0], state);
          return true;
        }
        if (std::optional<int64_t> v = intLiteralType(type)) {
          values[result] = bvConst(*v, 64);
          return true;
        }
      }
    // `Int(literal)`: the literal's value.
    if (path.starts_with("std::simd::SIMD::__init__[!pop.int_literal](") &&
        call.getNumOperands() == 1) {
      values[result] = term(call.getOperands()[0], state);
      return true;
    }
    // The pack of a heterogeneous variadic call (a tuple literal): the
    // references it is built from (`lit.ref.pack.create`).
    if (path.starts_with("std::builtin::variadics::VariadicPack::__init__(") &&
        call.getNumOperands() == 1)
      if (Operation *create = call.getOperands()[0].getDefiningOp();
          create && create->getName().getStringRef() == "lit.ref.pack.create") {
        std::string pack = declare({false, 64, false}, "g");
        packRefs[pack] = SmallVector<Value>(create->getOperands());
        values[result] = pack;
        return true;
      }
    // The pack of a variadic call (`[1, 2, 3]`): as long as the array of
    // references it is built from (`array<3, ...>`).
    if ((path.starts_with("std::builtin::variadics::VariadicList::__init__[") ||
         path.starts_with(
             "std::builtin::variadics::VariadicListMem::__init__[")) &&
        call.getNumOperands() == 1) {
      static llvm::Regex arrayRe("array<([0-9]+),");
      SmallVector<StringRef> m;
      std::string type = printed(call.getOperands()[0].getType());
      int64_t n;
      if (arrayRe.match(type, &m) && !m[1].getAsInteger(10, n)) {
        std::string pack = declare({false, 64, false}, "g");
        facts.push_back("(= " + lenOf(pack) + " " + bvConst(n, 64) + ")");
        values[result] = pack;
        // The references the pack holds, from the array stored into the
        // variable it is built from (`pop.array.create [%a, %b]`).
        if (std::optional<Loc> array = placeOf(call.getOperands()[0]);
            array && array->path.empty())
          for (Operation *user : array->root.getUsers())
            if (auto store = dyn_cast<LIT::RefStoreOp>(user);
                store && store->getOperand(1) == array->root)
              if (Operation *create = store->getOperand(0).getDefiningOp();
                  create &&
                  create->getName().getStringRef() == "pop.array.create" &&
                  create->getNumOperands() == (unsigned)n)
                packRefs[pack] = SmallVector<Value>(create->getOperands());
        return true;
      }
    }
    if (path.starts_with("std::builtin::_verification::_same_elements[") &&
        call.getNumOperands() == 3) {
      values[result] = sameElements(term(call.getOperands()[0], state),
                                    term(call.getOperands()[1], state),
                                    term(call.getOperands()[2], state));
      return true;
    }
    // An integer conversion (`Int(u8)`, `UInt8(i)`): extended by the
    // source's signedness, or truncated, as integer `cast` does.
    // The parameters resolved: in a callee's clause they may name its own
    // (`Int(offset)` for an `offset: Scalar` of any dtype).
    auto resolved = [&](size_t i) {
      auto symbol = dyn_cast<SymbolConstantAttr>(call.getCallee());
      return symbol && i < symbol.getParamValues().size()
                 ? printed(resolveParam(symbol.getParamValues()[i]))
                 : name.params[i];
    };
    if (path.starts_with("std::simd::SIMD::__init__[::DType](::SIMD[") &&
        call.getNumOperands() == 1 && name.params.size() == 3 &&
        isWidthOne(resolved(1)))
      if (std::optional<Sort> to = dtypeSort(resolved(0)),
          from = dtypeSort(resolved(2));
          to && from && sortOfTerm(operand(0)).width == from->width &&
          !sortOfTerm(operand(0)).isBool) {
        std::string v = operand(0);
        std::string w;
        if (to->width > from->width)
          w = "((_ " + std::string(from->isSigned ? "sign" : "zero") +
              "_extend " + std::to_string(to->width - from->width) + ") " + v +
              ")";
        else if (to->width < from->width)
          w = "((_ extract " + std::to_string(to->width - 1) + " 0) " + v + ")";
        else
          w = v;
        values[result] = define(*to, w);
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
    if (call.getNumOperands() == 2 &&
        (method == "__floordiv__" || method == "__mod__")) {
      checkDivisor(call, operand(1), state);
      values[result] = define(*sort, floorDivision(method == "__mod__", *sort,
                                                   operand(0), operand(1)));
      return true;
    }
    if (call.getNumOperands() == 2 && method == "__ceildiv__") {
      checkDivisor(call, operand(1), state);
      values[result] =
          define(*sort, ceilDivision(*sort, operand(0), operand(1)));
      return true;
    }
    // In place (`x //= y`): not modelled, but its divisor is checked.
    if (call.getNumOperands() == 2 &&
        (method == "__ifloordiv__" || method == "__imod__"))
      checkDivisor(call, operand(1), state);
    if (call.getNumOperands() == 1 && method == "__neg__") {
      values[result] = define(*sort, "(bvneg " + operand(0) + ")");
      return true;
    }
    return false;
  }

  /// A parameter value with the callee parameters it names replaced by the
  /// values their calls bind, as far as they are known.
  TypedAttr resolveParam(TypedAttr attr, ParamFrame **scope = nullptr) {
    // A type passed where a wider trait is expected: the type itself.
    while (auto upcast = dyn_cast<UpcastAttr>(attr))
      attr = upcast.getInputTypeValue();
    // The frame the result's own parameter references belong to.
    if (scope)
      *scope = params;
    for (ParamFrame *frame = params; frame; frame = frame->parent) {
      while (auto upcast = dyn_cast<UpcastAttr>(attr))
        attr = upcast.getInputTypeValue();
      auto ref = dyn_cast<ParamDeclRefAttr>(attr);
      if (!ref)
        break;
      auto it = frame->values.find(ref.getName());
      if (it == frame->values.end()) {
        if (!frame->transparent)
          break;
        // Not a closure's own parameter: its caller's.
        if (scope)
          *scope = frame->parent;
        continue;
      }
      attr = it->second;
      if (scope)
        *scope = frame->parent;
    }
    while (auto upcast = dyn_cast<UpcastAttr>(attr))
      attr = upcast.getInputTypeValue();
    return attr;
  }

  /// Whether a type parameter's value is `Int`.
  static bool isIntType(TypedAttr attr) {
    // Through aliases, and through a type given by a parameter expression
    // (`comptime for k in range(n)`: `k` is the range's `Element`, `Int`).
    for (int depth = 0; depth < 8; ++depth) {
      if (auto sugar = dyn_cast<SugarAttr>(attr)) {
        attr = sugar.getCanonical();
        continue;
      }
      auto type = dyn_cast<TypeParamAttr>(attr);
      if (!type)
        return false;
      if (auto param = dyn_cast<KGEN::ParamType>(type.getTypeValue())) {
        attr = param.getParam();
        continue;
      }
      return isInt(type.getTypeValue());
    }
    return false;
  }

  /// The value of an `IntLiteral` type parameter's literal.
  static std::optional<int64_t> intLiteralType(TypedAttr attr) {
    if (auto sugar = dyn_cast<SugarAttr>(attr))
      attr = sugar.getCanonical();
    auto type = dyn_cast<TypeParamAttr>(attr);
    if (!type)
      return std::nullopt;
    static llvm::Regex literalRe(
        "^!lit.struct<@std::@builtin::@int_literal::@IntLiteral"
        "<:!pop.int_literal (-?[0-9]+)>>$");
    SmallVector<StringRef> m;
    int64_t v;
    if (!literalRe.match(printed(type.getTypeValue()), &m) ||
        m[1].getAsInteger(10, v))
      return std::nullopt;
    return v;
  }

  /// The length in bytes of a `!kgen.string` parameter value.
  static std::optional<int64_t> stringLength(TypedAttr attr) {
    if (auto sugar = dyn_cast<SugarAttr>(attr))
      attr = sugar.getCanonical();
    if (auto str = dyn_cast<StringAttr>(attr))
      return str.getValue().size();
    return std::nullopt;
  }

  /// Whether `type` is `Int` (possibly through an alias).
  static bool isInt(Type type) {
    static llvm::Regex intRe("^!lit.struct<@std::@simd::@SIMD<:[^ ]* "
                             "\\{:dtype index\\}, :[^ ]* \\{1\\}>>$");
    return intRe.match(printed(type));
  }

  /// The length of the `Array` at `loc`, from its type.
  MaybeTerm arrayLength(const Loc &loc) {
    auto type = dyn_cast<LIT::StructType>(placeType(loc));
    if (!type || type.getParamValues().size() != 2 ||
        printed(type.getSymbol()) != "@std::@collections::@array::@Array")
      return std::nullopt;
    TypedAttr size = type.getParamValues()[1];
    Sort sort = sortOf(size.getType());
    if (sort.isBool || sort.width != 64)
      return std::nullopt;
    // The type is the caller's when the place is (a callee's contract binds
    // its arguments to the caller's places).
    ParamFrame *saved = params;
    Operation *owner =
        isa<BlockArgument>(loc.root)
            ? cast<BlockArgument>(loc.root).getOwner()->getParentOp()
            : loc.root.getDefiningOp();
    if (owner && (owner == fn.getOperation() ||
                  owner->getParentOfType<LIT::FnOp>() == fn))
      params = rootParams;
    std::string n = paramTerm(size, sort);
    params = saved;
    return n;
  }

  /// Field `field` of the struct value `built`, which a modelled
  /// constructor made: it keeps it wherever the value is moved or copied.
  void setBuiltField(const std::string &built, const std::string &field,
                     const std::string &value) {
    fieldValues[{built, field}] = value;
    builtFields[{built, field}] = value;
  }

  /// Writes a struct value with known fields to `loc`: the struct itself is
  /// a fresh value, its fields the given terms.
  void storeFields(const Loc &loc, State &state,
                   ArrayRef<std::pair<std::string, std::string>> fields) {
    std::string built = declare({false, 64, false}, "g");
    store(loc, state, built);
    // The value keeps its fields wherever it is moved or copied to.
    for (auto &[field, value] : fields) {
      state.env[Loc{loc.root, loc.path + field}] = value;
      fieldValues[{built, field}] = value;
      builtFields[{built, field}] = value;
    }
  }

  /// `Optional` (of an integer or Boolean) as the stdlib defines it: whether
  /// it holds a value (`/has`) and the value (`/val`); its constructors from
  /// a value, from `None` and by copy, `or_else` and `__bool__`. And
  /// `ContiguousSlice`'s constructor from two optional bounds, into its
  /// `start` and `end` fields. The results are out-result slots, the
  /// callee's last operand.
  bool evalOptional(LIT::CallOp call, const CalleeName &name, State &state) {
    StringRef path = name.path;
    const char *optional = "std::collections::optional::Optional::";
    const char *slice = "std::builtin::builtin_slice::ContiguousSlice::";
    const char *strided = "std::builtin::builtin_slice::StridedSlice::";
    bool isOptional = path.starts_with(optional);
    bool isSlice = path.starts_with(slice);
    bool isStrided = path.starts_with(strided);
    if (!isOptional && !isSlice && !isStrided)
      return false;
    StringRef method = path.drop_front(strlen(isOptional ? optional
                                              : isSlice  ? slice
                                                         : strided));
    ValueRange ops = call.getOperands();
    auto place = [&](Value v) { return placeOf(v); };
    Sort flag{true, 1, false};
    // The element sort, from `Optional[Int]`'s parameter.
    Sort valueSort{false, 64, true};
    if (!name.params.empty())
      if (std::optional<Sort> sort = dtypeSort(name.params[0]))
        valueSort = *sort;
    auto has = [&](const Loc &opt) {
      return load(Loc{opt.root, opt.path + "/has"}, state, flag);
    };
    auto val = [&](const Loc &opt) {
      return load(Loc{opt.root, opt.path + "/val"}, state, valueSort);
    };
    if (isSlice) {
      if (!method.starts_with("__init__(::Optional") || ops.size() < 3)
        return false;
      std::optional<Loc> start = place(ops[0]), end = place(ops[1]),
                         out = place(ops.back());
      if (!start || !end || !out)
        return false;
      storeFields(*out, state,
                  {{"/start/has", has(*start)}, {"/start/val", val(*start)},
                   {"/end/has", has(*end)}, {"/end/val", val(*end)}});
      setResultsUnknown(call);
      return true;
    }
    // `StridedSlice(start, end, stride)`: its inner `Slice`'s fields, with a
    // step that is always present.
    if (isStrided) {
      if (!method.starts_with("__init__(::Optional") || ops.size() < 4)
        return false;
      std::optional<Loc> start = place(ops[0]), end = place(ops[1]),
                         out = place(ops.back());
      if (!start || !end || !out || sortOf(ops[2].getType()).width != 64 ||
          sortOf(ops[2].getType()).isBool)
        return false;
      storeFields(*out, state,
                  {{"/_inner/start/has", has(*start)},
                   {"/_inner/start/val", val(*start)},
                   {"/_inner/end/has", has(*end)},
                   {"/_inner/end/val", val(*end)},
                   {"/_inner/step/has", "true"},
                   {"/_inner/step/val", term(ops[2], state)}});
      setResultsUnknown(call);
      return true;
    }
    if (method.starts_with("__bool__(") && ops.size() == 1 &&
        call->getNumResults() == 1) {
      std::optional<Loc> self = place(ops[0]);
      if (!self)
        return false;
      values[call->getResult(0)] = has(*self);
      return true;
    }
    if (ops.size() < 2)
      return false;
    std::optional<Loc> out = place(ops.back());
    if (!out)
      return false;
    if (method.starts_with("__init__(None)")) {
      storeFields(*out, state, {{"/has", "false"}});
    } else if (method.starts_with("__init__(copy:")) {
      std::optional<Loc> src = place(ops[0]);
      if (!src)
        return false;
      storeFields(*out, state, {{"/has", has(*src)}, {"/val", val(*src)}});
    } else if (method.starts_with("__init__($0$)")) {
      std::optional<Loc> value = place(ops[0]);
      if (!value)
        return false;
      storeFields(*out, state,
                  {{"/has", "true"}, {"/val", load(*value, state, valueSort)}});
    } else if (method.starts_with("or_else(") && ops.size() == 3) {
      std::optional<Loc> self = place(ops[0]), fallback = place(ops[1]);
      if (!self || !fallback)
        return false;
      store(*out, state,
            define(valueSort, "(ite " + has(*self) + " " + val(*self) + " " +
                                  load(*fallback, state, valueSort) + ")"));
    } else {
      return false;
    }
    setResultsUnknown(call);
    return true;
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
      } else if (call.getNumOperands() == 3 && sort->width == 64 &&
                 sort->isSigned) {
        // `range(start, end, step)`, a forward `_StridedRange`: a zero step
        // makes it the empty range from 0 with step 1.
        std::string step = term(call.getOperands()[2], state);
        std::string zero = bvConst(0, 64);
        std::string degenerate =
            define({true, 1, false}, "(= " + step + " " + zero + ")");
        auto unless = [&](const std::string &v, const std::string &other) {
          return define(*sort,
                        "(ite " + degenerate + " " + other + " " + v + ")");
        };
        fields["/start"] = unless(term(call.getOperands()[0], state), zero);
        fields["/end"] = unless(term(call.getOperands()[1], state), zero);
        fields["/step"] = unless(step, bvConst(1, 64));
        fields["/idx"] = zero;
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

  /// The forward `_StridedRange` over `Int` (`range(start, end, step)`), its
  /// `__iter__` and `__next__` as the stdlib defines them: `__next__` raises
  /// once a step has wrapped (`/idx`) or the cursor (`/start`) has reached
  /// the end in the step's direction, and otherwise returns the cursor and
  /// advances it by the step.
  bool evalStridedRange(LIT::CallOp call, const CalleeName &name,
                        State &state) {
    StringRef path = name.path;
    const char *strided = "std::builtin::range::_StridedRange::";
    std::optional<Sort> dtype = dtypeSort(name.params[0]);
    if (!dtype || dtype->isBool || dtype->width != 64 || !dtype->isSigned ||
        !StringRef(name.params[1]).contains("= true") ||
        call.getNumOperands() < 1)
      return false;
    Sort sort = *dtype;
    StringRef method = path.drop_front(strlen(strided));
    std::optional<Loc> self = placeOf(call.getOperands()[0]);
    if (!self)
      return false;
    Loc start{self->root, self->path + "/start"},
        end{self->root, self->path + "/end"},
        step{self->root, self->path + "/step"},
        idx{self->root, self->path + "/idx"};
    if (method.starts_with("__iter__") && call.getNumOperands() == 1 &&
        call->getNumResults() == 1) {
      Value result = call->getResult(0);
      values[result] = declare({false, 64, false}, "g");
      records[result] = {{"/start", load(start, state, sort)},
                         {"/end", load(end, state, sort)},
                         {"/step", load(step, state, sort)},
                         {"/idx", load(idx, state, sort)}};
      return true;
    }
    if (method.starts_with("__next__") && call.getNumOperands() == 3 &&
        call->getNumResults() == 1) {
      std::optional<Loc> error = placeOf(call.getOperands()[1]);
      std::optional<Loc> out = placeOf(call.getOperands()[2]);
      if (!error || !out)
        return false;
      std::string s = load(start, state, sort), e = load(end, state, sort),
                  st = load(step, state, sort), i = load(idx, state, sort);
      std::string zero = bvConst(0, 64);
      std::string raised =
          define({true, 1, false}, "(or (not (= " + i + " " + zero +
                                       ")) (ite (bvsgt " + st + " " + zero +
                                       ") (bvsge " + s + " " + e + ") (bvsge " +
                                       e + " " + s + ")))");
      noteCondition(raised);
      std::string next = "(bvadd " + s + " " + st + ")";
      std::string wrapped = "(xor (bvslt " + next + " " + s + ") (bvslt " + st +
                            " " + zero + "))";
      std::string newStart =
          define(sort, "(ite " + raised + " " + s + " " + next + ")");
      std::string newIdx =
          define(sort, "(ite " + raised + " " + i + " (ite " + wrapped + " " +
                           bvConst(1, 64) + " " + zero + "))");
      store(start, state, newStart);
      store(idx, state, newIdx);
      // `store` makes the range itself unknown, not its other fields.
      state.env[start] = newStart;
      state.env[idx] = newIdx;
      state.env[end] = e;
      state.env[step] = st;
      store(*out, state, s);
      havoc(*error, state, placeType(*error));
      values[call->getResult(0)] = raised;
      return true;
    }
    return false;
  }

  /// `reversed(range(...))` over `Int`, and the reversed iterator's
  /// `__iter__` and `__next__`. The stdlib walks from `end - 1` down to the
  /// range's start, inclusive, flagging exhaustion; this models the same
  /// sequence mirrored from the forward range, with an exclusive cursor
  /// (`/curr`, starting at the forward range's end) and the lower bound
  /// (`/end`, its start): `__next__` raises when they are equal, and
  /// otherwise decrements the cursor and returns it.
  bool evalReversedRange(LIT::CallOp call, const CalleeName &name,
                         State &state) {
    StringRef path = name.path;
    Sort sort{false, 64, true};
    if (path.starts_with(
            "std::builtin::reversed::reversed[::ReversibleRange")) {
      StringRef type = name.params.empty() ? "" : StringRef(name.params[0]);
      if (call.getNumOperands() != 2 || name.params.size() != 1 ||
          !(type.contains("@std::@builtin::@range::@_ZeroStartingRange<") ||
            type.contains("@std::@builtin::@range::@_SequentialRange<")) ||
          !type.contains("{:dtype index}>"))
        return false;
      std::optional<Loc> range = placeOf(call.getOperands()[0]);
      std::optional<Loc> out = placeOf(call.getOperands()[1]);
      if (!range || !out)
        return false;
      std::string start =
          load(Loc{range->root, range->path + "/curr"}, state, sort);
      std::string end =
          load(Loc{range->root, range->path + "/end"}, state, sort);
      storeFields(*out, state, {{"/curr", end}, {"/end", start}});
      setResultsUnknown(call);
      return true;
    }
    const char *strided = "std::builtin::range::_StridedRange::";
    if (!path.starts_with(strided) || name.params.size() < 2)
      return false;
    if (evalStridedRange(call, name, state))
      return true;
    std::optional<Sort> dtype = dtypeSort(name.params[0]);
    if (!dtype || dtype->isBool || dtype->width != 64 || !dtype->isSigned ||
        !StringRef(name.params[1]).contains("= false") ||
        call.getNumOperands() < 1)
      return false;
    StringRef method = path.drop_front(strlen(strided));
    std::optional<Loc> self = placeOf(call.getOperands()[0]);
    if (!self)
      return false;
    Loc curr{self->root, self->path + "/curr"},
        end{self->root, self->path + "/end"};
    if (method.starts_with("__iter__") && call.getNumOperands() == 1 &&
        call->getNumResults() == 1) {
      Value result = call->getResult(0);
      values[result] = declare({false, 64, false}, "g");
      records[result] = {{"/curr", load(curr, state, sort)},
                         {"/end", load(end, state, sort)}};
      return true;
    }
    if (method.starts_with("__next__") && call.getNumOperands() == 3 &&
        call->getNumResults() == 1) {
      std::optional<Loc> error = placeOf(call.getOperands()[1]);
      std::optional<Loc> out = placeOf(call.getOperands()[2]);
      if (!error || !out)
        return false;
      std::string c = load(curr, state, sort);
      std::string e = load(end, state, sort);
      std::string raised = define({true, 1, false}, "(= " + c + " " + e + ")");
      noteCondition(raised);
      std::string next =
          define(sort, "(ite " + raised + " " + c + " (bvsub " + c + " " +
                           bvConst(1, sort.width) + "))");
      store(curr, state, next);
      // `store` makes the range itself unknown, not its other field.
      state.env[end] = e;
      store(*out, state, next);
      havoc(*error, state, placeType(*error));
      values[call->getResult(0)] = raised;
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
    // Calls inside the contract assumed their callees' postconditions into
    // the path condition of `inner`: they belong to the clause, as facts
    // when it is assumed and as premises when it is proven.
    if (result && inner.pc != state.pc && sortOfTerm(*result).isBool)
      result =
          define({true, 1, false}, "(" + std::string(assumed ? "and" : "=>") +
                                       " " + inner.pc + " " + *result + ")");
    restore();
    return result;
  }

  /// Whether `formal`, an operand of the callee's contract, is its result: a
  /// local of the callee (a named `out` result, `lit.var.decl "r" arg`, or a
  /// constructor's `self`, `lit.var.decl "self" initoutarg`). A contract's
  /// operands are the function's arguments and its result, so any local is
  /// the result.
  static bool isNamedResult(LIT::FnOp callee, Value formal) {
    auto decl = formal.getDefiningOp<LIT::VarDeclOp>();
    return decl && decl->getParentOfType<LIT::FnOp>() == callee;
  }

  /// Binds a region's block argument to the caller's value `actual`: a
  /// reference to its place, a value to its term in `state`.
  bool bind(BlockArgument blockArg, Value actual, State &state,
            SmallVectorImpl<Value> &boundRefs) {
    if (isa<LIT::RefType>(blockArg.getType())) {
      std::optional<Loc> loc = placeOf(actual);
      // A reference to a place that is not tracked (`self` of a struct in
      // shared memory): left unbound, so what the clause reads through it
      // is unknown, and the rest of the clause still counts.
      if (!loc)
        return actual && isa<LIT::RefType>(actual.getType());
      refArgs[blockArg] = *loc;
      boundRefs.push_back(blockArg);
      return true;
    }
    // A value argument given by reference (a launch's argument pack).
    values[blockArg] = isa<LIT::RefType>(actual.getType())
                           ? valueThrough(actual, state)
                           : term(actual, state);
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
    // What the region's calls guarantee about the entry values (`count() >=
    // 0` in `old(self.count())`) holds here too; like the facts of calls in
    // the clause itself, it becomes a fact or premise of the clause.
    if (ok && inner.pc != oldState->pc)
      state.pc =
          define({true, 1, false}, "(and " + state.pc + " " + inner.pc + ")",
                 "r");
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

/// A required trait method: a declaration, whose clauses every
/// implementation provides, with nothing to verify of its own.
bool isRequiredTraitMethod(LIT::FnOp fn) {
  return isa_and_nonnull<LIT::TraitDeclOp>(fn->getParentOp()) &&
         !fn.isDefaultedTraitFn();
}

/// A struct's synthesized wrapper for an inherited trait default: it only
/// forwards to the default, whose precondition is its own (callers are held
/// to it), so there is nothing to verify of its own.
bool isDefaultWrapper(LIT::FnOp fn) {
  return fn.getDefaultFnRefAttr() != nullptr;
}

/// Code from an imported package (the stdlib, `layout`, ...), not from the
/// file being checked.
bool inLibrary(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp())
    if (isa<LIT::PackageOp>(parent))
      return true;
  return false;
}

/// The top-level package `op` is in (`nn` for `nn.softmax`), or "".
StringRef packageOf(Operation *op) {
  StringRef name;
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp())
    if (auto package = dyn_cast<LIT::PackageOp>(parent))
      name = package.getSymName();
  return name;
}

struct VerifyContractsPass
    : impl::VerifyContractsBase<VerifyContractsPass> {
  using VerifyContractsBase::VerifyContractsBase;

  void runOnOperation() override {
    if (!intTranslateFile.empty()) {
      auto buffer = llvm::MemoryBuffer::getFile(intTranslateFile);
      std::optional<std::string> ints;
      if (buffer)
        ints = IntTranslator().translate((*buffer)->getBuffer());
      if (!ints) {
        getOperation().emitError("verify-contracts: cannot translate ")
            << intTranslateFile;
        return signalPassFailure();
      }
      std::error_code ec;
      llvm::raw_fd_ostream os(intTranslateFile + ".int.smt2", ec);
      if (ec)
        return signalPassFailure();
      os << *ints;
      return;
    }
    std::string z3 = z3Path;
    if (z3.empty()) {
      if (auto found = llvm::sys::findProgramByName("z3"))
        z3 = *found;
    }
    if (z3.empty()) {
      emitError(getOperation().getLoc(), "verify-contracts: z3 not found");
      return signalPassFailure();
    }
    if (!dumpFn.empty())
      getOperation().walk([&](LIT::FnOp fn) {
        if (displayName(fn) == dumpFn)
          fn.print(llvm::errs());
      });
    SolverConfig solver{z3, rlimit, houdiniRlimit, wallSeconds, dumpDir,
                        cacheDir};
    if (intRetry)
      solver.nonlinearRlimit = nonlinearRlimit;
    solver.exactRetry = intRetry;
    solver.caseSplit = caseSplit;
    solver.invariantRetry = invariantRetry;
    solver.memo = std::make_shared<ScriptMemo>();
    if (!cacheDir.empty())
      (void)llvm::sys::fs::create_directories(cacheDir);
    // The scripts are run from `dump-dir`: one that cannot be written to
    // would leave every query unanswered.
    if (!dumpDir.empty())
      if (std::error_code ec = llvm::sys::fs::create_directories(dumpDir)) {
        // By location: an error on the module prints all of it.
        emitError(getOperation().getLoc(),
                  "verify-contracts: cannot create dump-dir '" + dumpDir +
                      "': " + ec.message());
        return signalPassFailure();
      }
    // Functions are verified independently, in parallel; their results are
    // reported afterwards, in order.
    // Each function, and each implementation of a trait method with
    // clauses against them (`refines`).
    struct Job {
      LIT::FnOp fn, refines;
    };
    SmallVector<Job> fns;
    // Functions from the file, and from imported packages when asked.
    auto selected = [&](Operation *op) {
      return includeStdlib || !inLibrary(op) ||
             llvm::is_contained(packages, packageOf(op));
    };
    // A local closure only ever called directly is verified where it is
    // called, with its caller's state (`inlineClosure`); one that escapes
    // (a parameter of another call: `vectorize[f]`, a launch) on its own.
    auto calledOnly = [](LIT::FnOp fn) {
      LIT::FnOp outer = fn->getParentOfType<LIT::FnOp>();
      auto decl = fn.getParamDeclAttr();
      if (!outer || !decl)
        return false;
      Attribute ref = ParamDeclRefAttr::get(decl);
      bool escapes = false;
      outer->walk([&](Operation *op) {
        if (escapes || op == fn.getOperation())
          return;
        // A direct call names the closure as its callee; any other
        // mention (a callee's parameter, `f[elementwise_lambda_fn=g]`) passes
        // it on.
        auto call = dyn_cast<LIT::CallOp>(op);
        for (NamedAttribute attr : op->getAttrs()) {
          if (call && attr.getValue() == call.getCalleeAttr()) {
            // `f()`, or `f[4]()` with all of its parameters, none of which
            // mentions it.
            ArrayRef<TypedAttr> bound;
            size_t own = fn.getParams().size();
            own -= std::min(own, call.getImplicitOrigins().size());
            if (closureCallee(call.getCallee(), bound) == ref &&
                (bound.empty() || bound.size() == own)) {
              for (TypedAttr value : bound)
                value.walk([&](Attribute a) {
                  if (a == ref)
                    escapes = true;
                });
              continue;
            }
          }
          attr.getValue().walk([&](Attribute a) {
            if (a == ref)
              escapes = true;
          });
        }
      });
      return !escapes;
    };
    getOperation().walk([&](LIT::FnOp fn) {
      if (selected(fn) && !isRequiredTraitMethod(fn) && !isDefaultWrapper(fn) &&
          !calledOnly(fn))
        fns.push_back({fn, {}});
    });
    ModuleOp module = getOperation();
    TraitImpls impls;
    {
      SymbolTableCollection symbols;
      getOperation().walk([&](ConformanceOp conformance) {
        if (!selected(conformance))
          return;
        auto trait = dyn_cast_or_null<LIT::TraitDeclOp>(symbols.lookupSymbolIn(
            module, conformance.getTraitSymbol().getSymbol()));
        if (!trait)
          return;
        for (auto witness : conformance.getBody().getOps<WitnessOp>()) {
          auto method = dyn_cast_or_null<LIT::FnOp>(
              symbols.lookupSymbolIn(trait, witness.getSymNameAttr()));
          auto value = dyn_cast<SymbolConstantAttr>(witness.getValue());
          if (!method || !value || method.getFunctionBody().empty())
            continue;
          auto impl = dyn_cast_or_null<LIT::FnOp>(
              symbols.lookupSymbolIn(module, value.getSymbol()));
          if (std::string owner = impl ? plainStructOf(impl) : "";
              !owner.empty()) {
            std::string key = implKey(conformance.getTraitSymbol(),
                                      witness.getSymName(), owner);
            impls.byKey[key] = impl;
            impls.keyOf[impl] = key;
          }
          bool hasClauses = !method.getFunctionBody()
                                 .getOps<RequiresOp>()
                                 .empty() ||
                            !postconditionClauses(method).empty();
          if (impl && hasClauses && !isDefaultWrapper(impl))
            fns.push_back({impl, method});
        }
      });
    }
    // Generic kernels as they are launched (`enqueue_function[kernel[16,
    // 16]](...)`, possibly with parameters of the launching function), by
    // kernel: what the generic proof leaves open is checked for each.
    struct Instance {
      SmallVector<InstanceLink> chain;
      Location launch;
      // The outermost call that gave parameters, when not the launch.
      std::optional<Location> via;
    };
    DenseMap<Operation *, SmallVector<Instance>> instances;
    {
      SymbolTableCollection symbols;
      DenseSet<Operation *> verified;
      for (Job &job : fns)
        if (!job.refines)
          verified.insert(job.fn);
      // The launches, by the callee's symbol alone: printing every call's
      // parameters (`calleeName`) costs more than the rest of this scan.
      struct Launch {
        InstanceLink kernel;
        LIT::FnOp launcher;
        Location loc;
      };
      SmallVector<Launch> launches;
      module.walk([&](LIT::CallOp call) {
        auto symbol = dyn_cast<SymbolConstantAttr>(call.getCallee());
        ArrayRef<FlatSymbolRefAttr> nested =
            symbol ? symbol.getSymbol().getNestedReferences()
                   : ArrayRef<FlatSymbolRefAttr>();
        if (nested.size() < 2 ||
            !nested.back().getValue().starts_with("enqueue_function[") ||
            !nested[nested.size() - 2].getValue().ends_with("DeviceContext"))
          return;
        auto [kernel, instance] = launchedKernel(
            symbol, module, symbols, [](TypedAttr param) { return param; });
        if (kernel && verified.count(kernel) &&
            !instance.getParamValues().empty())
          launches.push_back({{instance, kernel, 0},
                              call->getParentOfType<LIT::FnOp>(),
                              call.getLoc()});
      });
      // Whether a parameter of the calling function is in it.
      auto openValues = [](ArrayRef<TypedAttr> values) {
        bool found = false;
        for (TypedAttr value : values)
          value.walk([&](ParamDeclRefAttr) { found = true; });
        return found;
      };
      auto open = [&](SymbolConstantAttr symbol) {
        return openValues(symbol.getParamValues());
      };
      // The same of a chain's last step; through a closure's call, which
      // gives only the closure's own parameters, of the step before it too.
      auto openChain = [&](ArrayRef<InstanceLink> chain) {
        for (const InstanceLink &link : llvm::reverse(chain)) {
          if (openValues(link.values()))
            return true;
          if (!link.closure)
            break;
        }
        return false;
      };
      // The calls of the functions whose parameters a launch (or a call
      // on the way to it) passes on, up to three levels out.
      struct Caller {
        InstanceLink link;
        LIT::FnOp from;
        Location loc;
      };
      DenseMap<Operation *, SmallVector<Caller>> callers;
      DenseSet<Operation *> wanted;
      for (Launch &launch : launches)
        if (launch.launcher && open(launch.kernel.symbol))
          wanted.insert(launch.launcher);
      // A launch in a local closure with parameters of its own
      // (`_rows_per_warp[rows]`): the closure's calls give them, the
      // function they are in the rest.
      for (Operation *op :
           SmallVector<Operation *>(wanted.begin(), wanted.end())) {
        auto closure = cast<LIT::FnOp>(op);
        auto decl = closure.getParamDeclAttr();
        auto outer = closure->getParentOfType<LIT::FnOp>();
        if (!decl || !outer || !calledOnly(closure))
          continue;
        Attribute ref = ParamDeclRefAttr::get(decl);
        outer->walk([&](LIT::CallOp call) {
          ArrayRef<TypedAttr> bound;
          auto bind = dyn_cast<BindParamsAttr>(call.getCallee());
          if (!bind || closureCallee(bind, bound) != ref)
            return;
          auto from = call->getParentOfType<LIT::FnOp>();
          InstanceLink link{
              {}, closure, call.getImplicitOrigins().size(), bind};
          callers[closure].push_back({link, from, call.getLoc()});
          if (from)
            wanted.insert(from);
        });
      }
      for (int level = 0; level < 3 && !wanted.empty(); ++level) {
        llvm::StringSet<> names;
        for (Operation *fn : wanted)
          if (auto name = cast<LIT::FnOp>(fn).getSymName())
            names.insert(*name);
        DenseSet<Operation *> next;
        module.walk([&](LIT::CallOp call) {
          auto symbol = dyn_cast<SymbolConstantAttr>(call.getCallee());
          if (!symbol ||
              !names.contains(symbol.getSymbol().getLeafReference().getValue()))
            return;
          auto fn = dyn_cast_or_null<LIT::FnOp>(
              symbols.lookupSymbolIn(module, symbol.getSymbol()));
          if (!fn || !wanted.count(fn))
            return;
          auto from = call->getParentOfType<LIT::FnOp>();
          callers[fn].push_back({{symbol, fn, call.getImplicitOrigins().size()},
                                 from,
                                 call.getLoc()});
          if (from && open(symbol) && !callers.count(from))
            next.insert(from);
        });
        wanted = std::move(next);
      }
      // Each launch, through every chain of calls that gives its open
      // parameters; at most 32 instantiations per kernel.
      std::set<std::string> seen;
      std::function<void(SmallVector<InstanceLink>, LIT::FnOp, Location,
                         std::optional<Location>, int)>
          expand = [&](SmallVector<InstanceLink> chain, LIT::FnOp outer,
                       Location launch, std::optional<Location> via,
                       int depth) {
            auto it = outer ? callers.find(outer) : callers.end();
            if (depth < 3 && openChain(chain) && it != callers.end() &&
                !it->second.empty()) {
              for (const Caller &caller : it->second) {
                SmallVector<InstanceLink> longer = chain;
                longer.push_back(caller.link);
                expand(std::move(longer), caller.from, launch, caller.loc,
                       depth + 1);
              }
              return;
            }
            std::string key;
            for (const InstanceLink &link : chain)
              key += printed(link.callee()) + "|";
            SmallVector<Instance> &list = instances[chain.front().fn];
            if (list.size() < 32 && seen.insert(key).second)
              list.push_back({std::move(chain), launch, via});
          };
      for (Launch &launch : launches)
        expand({launch.kernel}, launch.launcher, launch.loc, std::nullopt, 0);
    }
    struct Result {
      SmallVector<Obligation> obligations;
      SmallVector<Answer> answers;
      unsigned loops = 0, invariants = 0;
    };
    // Encodes and solves one function (or instantiation) into `result`.
    auto verify = [&](FunctionEncoder &enc, Result &result,
                      const std::string &name) {
      if (!enc.encode())
        return;
      result.loops = enc.loopsAnalyzed;
      result.invariants = enc.invariantsFound;
      if (enc.obligations.empty())
        return;
      std::optional<SmallVector<Answer>> answers =
          runZ3(solver, enc.script(), name);
      // Still open: case splitting over the finite-domain unknowns, before
      // the full limit (a case makes them constants, so it is quick where
      // the full limit runs to its end).
      auto cases = [&] {
        SmallVector<size_t> open;
        SmallVector<bool> reask;
        std::string text = enc.scriptHeader();
        size_t k = 0;
        for (const Obligation &ob : enc.obligations) {
          if (!ob.analyzed)
            continue;
          if (k < answers->size() && ((*answers)[k] == Answer::Unknown ||
                                      (*answers)[k] == Answer::Unproven)) {
            open.push_back(k);
            reask.push_back(enc.obligationDependsOnCases(ob));
            text += query({ob.pc}, ob.cond, enc.limitOf(ob));
          }
          ++k;
        }
        if (!open.empty()) {
          SmallVector<Answer> sub;
          for (size_t k : open)
            sub.push_back((*answers)[k]);
          enc.solveCases(text, sub, name + ".cases", reask);
          for (auto [i, k] : llvm::enumerate(open))
            if (sub[i] == Answer::Proven)
              (*answers)[k] = Answer::Proven;
        }
      };

      // Queries answered `unknown`: once more over the integers (with the
      // limit they were asked with), and those asked with a lower limit and
      // still open, again with the full one.
      if (intRetry && answers) {
        SmallVector<const Obligation *> analyzed;
        for (const Obligation &ob : enc.obligations)
          if (ob.analyzed)
            analyzed.push_back(&ob);
        auto retry = [&](bool lowered, StringRef suffix) {
          SmallVector<size_t> open;
          std::string text = enc.scriptHeader();
          for (auto [k, ob] : llvm::enumerate(analyzed)) {
            unsigned limit = enc.limitOf(*ob);
            if (k >= answers->size() || (*answers)[k] != Answer::Unknown ||
                (lowered && limit == enc.fullLimit()))
              continue;
            open.push_back(k);
            // Over the integers, nonlinear queries can run for long without
            // using up their limit; one asked with a lower limit gets 1 s
            // there, as what stays open is asked again with the full limit.
            bool capped = !lowered && limit < enc.fullLimit();
            if (capped)
              text += "(set-option :timeout 1000)\n";
            text +=
                query({ob->pc}, ob->cond, lowered ? enc.fullLimit() : limit);
            if (capped)
              text += "(set-option :timeout 0)\n";
          }
          if (open.empty())
            return;
          std::optional<std::string> ints;
          if (!lowered)
            ints = IntTranslator().translate(text);
          if (!lowered && !ints)
            return;
          if (std::optional<SmallVector<Answer>> retried =
                  runZ3(solver, lowered ? text : *ints, name + suffix.str()))
            for (auto [i, k] : llvm::enumerate(open))
              if (i < retried->size() &&
                  (lowered || (*retried)[i] == Answer::Proven))
                (*answers)[k] = (*retried)[i];
        };
        cases();
        retry(/*lowered=*/false, ".int");
        // Still open with a product in the goal: over the integers with
        // exact products, in each case of the finite-domain unknowns.
        if (solver.exactRetry) {
          SmallVector<size_t> open;
          std::string text = enc.scriptHeader();
          for (auto [k, ob] : llvm::enumerate(analyzed))
            if (k < answers->size() && (*answers)[k] == Answer::Unknown &&
                enc.isNonlinear(*ob)) {
              open.push_back(k);
              text += query({ob->pc}, ob->cond, enc.limitOf(*ob));
            }
          if (!open.empty()) {
            SmallVector<Answer> sub(open.size(), Answer::Unknown);
            enc.solveCases(text, sub, name + ".exact", {}, /*exact=*/true);
            for (auto [i, k] : llvm::enumerate(open))
              if (sub[i] == Answer::Proven)
                (*answers)[k] = Answer::Proven;
          }
        }
        retry(/*lowered=*/true, ".full");
      } else if (answers) {
        cases();
      }
      unsigned next = 0;
      for (Obligation &ob : enc.obligations) {
        Answer answer = Answer::NotAnalyzed;
        if (ob.analyzed) {
          answer = answers && next < answers->size() ? (*answers)[next]
                                                     : Answer::Unknown;
          ++next;
        }
        result.answers.push_back(answer);
      }
      result.obligations = std::move(enc.obligations);
    };
    std::vector<Result> results(fns.size());
    // A launched generic kernel is verified only for its launched
    // instantiations, unless `generic-launched`: a proof for every value of
    // its parameters can be far slower (nonlinear in them) than one for the
    // values launched.
    auto launched = [&](size_t i) {
      return !fns[i].refines && instances.count(fns[i].fn);
    };
    llvm::DefaultThreadPool pool(llvm::hardware_concurrency());
    for (size_t i = 0; i < fns.size(); ++i) {
      if (launched(i) && !genericLaunched)
        continue;
      pool.async([&, i] {
        SymbolTableCollection symbols;
        std::string name = "f" + std::to_string(i);
        FunctionEncoder enc(fns[i].fn, module, symbols, solver, name, impls,
                            fns[i].refines);
        enc.checkDivision = checkDivision;
        if (launched(i))
          enc.queryRlimit = genericRlimit;
        verify(enc, results[i], name);
      });
    }
    pool.wait();
    // The instantiations of the launched kernels: all of them, or with
    // `generic-launched`, those whose generic proof left obligations open.
    struct InstanceJob {
      size_t job, instance;
      Result result;
      /// The parameter that makes it opaque (`opaqueParameter`): skipped.
      std::optional<std::string> opaque = std::nullopt;
    };
    std::vector<InstanceJob> instanceJobs;
    for (size_t i = 0; i < fns.size(); ++i) {
      auto it = fns[i].refines ? instances.end() : instances.find(fns[i].fn);
      if (it == instances.end() ||
          (genericLaunched && llvm::all_of(results[i].answers, [](Answer a) {
             return a == Answer::Proven;
           })))
        continue;
      for (size_t j = 0; j < it->second.size(); ++j)
        instanceJobs.push_back({i, j, {}});
    }
    for (size_t n = 0; n < instanceJobs.size(); ++n)
      pool.async([&, n] {
        InstanceJob &job = instanceJobs[n];
        SymbolTableCollection symbols;
        std::string name =
            "f" + std::to_string(job.job) + ".i" + std::to_string(job.instance);
        FunctionEncoder enc(fns[job.job].fn, module, symbols, solver, name,
                            impls);
        enc.checkDivision = checkDivision;
        enc.bindInstance(instances.find(fns[job.job].fn.getOperation())
                             ->second[job.instance]
                             .chain);
        if ((job.opaque = enc.opaqueParameter()))
          return;
        verify(enc, job.result, name);
      });
    pool.wait();
    for (InstanceJob &job : instanceJobs)
      if (job.opaque) {
        LIT::FnOp kernel = fns[job.job].fn;
        const Instance &at = instances[kernel.getOperation()][job.instance];
        auto diag = mlir::emitWarning(at.launch);
        diag << "'" << displayName(kernel)
             << "' is not verified for the instantiation launched here: its "
                "parameter '"
             << *job.opaque
             << "' is computed by a function the verifier does not evaluate";
        if (at.via)
          diag.attachNote(*at.via) << "with the parameters given here";
      }
    // An obligation's answer in each instantiation of its function: the
    // obligation with the same call, clause and occurrence.
    using Key = std::tuple<const void *, const void *, unsigned>;
    auto keys = [](const Result &result) {
      std::map<std::pair<const void *, const void *>, unsigned> seen;
      SmallVector<Key> out;
      for (const Obligation &ob : result.obligations) {
        std::pair<const void *, const void *> at{
            ob.callLoc.getAsOpaquePointer(), ob.clauseLoc.getAsOpaquePointer()};
        out.push_back({at.first, at.second, seen[at]++});
      }
      return out;
    };
    std::map<size_t, SmallVector<std::pair<size_t, std::map<Key, Answer>>>>
        instanceAnswers;
    for (InstanceJob &job : instanceJobs) {
      if (job.opaque)
        continue;
      std::map<Key, Answer> answers;
      for (auto [key, answer] : llvm::zip(keys(job.result), job.result.answers))
        answers[key] = answer;
      instanceAnswers[job.job].push_back({job.instance, std::move(answers)});
    }
    // A kernel verified only for its instantiations: its obligations are
    // those of its instantiations, in order of first appearance, each
    // standing for its worst answer among them (the per-instantiation
    // answers then decide what is reported).
    if (!genericLaunched)
      for (InstanceJob &job : instanceJobs) {
        if (job.opaque)
          continue;
        Result &result = results[job.job];
        std::set<Key> have;
        for (const Key &key : keys(result))
          have.insert(key);
        if (result.obligations.empty()) {
          result.loops = job.result.loops;
          result.invariants = job.result.invariants;
        }
        for (auto [key, ob, answer] : llvm::zip(
                 keys(job.result), job.result.obligations, job.result.answers))
          if (have.insert(key).second) {
            result.obligations.push_back(ob);
            result.answers.push_back(Answer::NotAnalyzed);
          }
      }
    // The worst of an obligation's answers across instantiations.
    auto rank = [](Answer a) {
      switch (a) {
      case Answer::Unproven:
        return 3;
      case Answer::Unknown:
        return 2;
      case Answer::NotAnalyzed:
        return 1;
      default:
        return 0;
      }
    };
    unsigned total = 0, proven = 0, forInstances = 0, loops = 0, invariants = 0;
    for (auto [i, result] : llvm::enumerate(results)) {
      loops += result.loops;
      invariants += result.invariants;
      SmallVector<Key> resultKeys = keys(result);
      for (auto [k, ob, answer] :
           llvm::enumerate(result.obligations, result.answers)) {
        ++total;
        StringRef what = ob.postcondition ? "postcondition" : "precondition";
        // A refinement: `Box.get`'s precondition follows from
        // `Counter.get`'s; `Box.bump` establishes `Counter.bump`'s
        // postcondition.
        std::string claim =
            !ob.claim.empty()             ? ob.claim
            : ob.implementation.empty() ? ""
            : ob.postcondition
                ? "that '" + ob.implementation + "' establishes the " +
                      what.str() + " of '" + ob.callee + "'"
                : "that the precondition of '" + ob.implementation +
                      "' follows from that of '" + ob.callee + "'";
        if (answer == Answer::Proven) {
          ++proven;
          if (verbose && !claim.empty())
            mlir::emitRemark(ob.callLoc) << "proven " << claim;
          else if (verbose)
            mlir::emitRemark(ob.callLoc)
                << what << " of '" << ob.callee << "' proven";
          continue;
        }
        // Open in general: proven, or not, for each launched instantiation.
        SmallVector<std::pair<size_t, Answer>> perInstance;
        if (auto it = instanceAnswers.find(i); it != instanceAnswers.end())
          for (auto &[instance, answers] : it->second) {
            auto found = answers.find(resultKeys[k]);
            // An instantiation that has this call but fewer occurrences
            // of it unrolled a loop fewer times (`tile_n` iterations): it
            // has nothing to prove for this one.
            Key first = resultKeys[k];
            std::get<2>(first) = 0;
            if (found == answers.end() && answers.count(first))
              continue;
            perInstance.push_back({instance, found == answers.end()
                                                 ? Answer::NotAnalyzed
                                                 : found->second});
          }
        // Reported by the worst instantiation, if it is not checked
        // generically.
        if (!genericLaunched && launched(i) && !perInstance.empty()) {
          answer = Answer::Proven;
          for (auto &p : perInstance)
            if (rank(p.second) > rank(answer))
              answer = p.second;
        }
        if (!perInstance.empty() && llvm::all_of(perInstance, [](auto &p) {
              return p.second == Answer::Proven;
            })) {
          ++proven;
          ++forInstances;
          if (verbose)
            mlir::emitRemark(ob.callLoc)
                << what << " of '" << ob.callee << "' proven for the "
                << perInstance.size() << " launched instantiation"
                << (perInstance.size() == 1 ? "" : "s");
          continue;
        }
        auto diag = mlir::emitWarning(ob.callLoc);
        switch (answer) {
        case Answer::Unproven:
          if (!claim.empty())
            diag << "cannot prove " << claim;
          else
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
        if (ob.claim.empty())
          diag.attachNote(ob.clauseLoc) << what << " declared here";
        else if (ob.clauseLoc != ob.callLoc)
          diag.attachNote(ob.clauseLoc) << "stated here";
        Operation *fn = fns[i].fn.getOperation();
        for (auto [instance, instanceAnswer] : perInstance)
          if (instanceAnswer != Answer::Proven) {
            const Instance &at = instances[fn][instance];
            diag.attachNote(at.launch)
                << (instanceAnswer == Answer::Unknown
                        ? "not decided within the solver limits"
                        : "not proven")
                << " for the instantiation launched here"
                << (genericLaunched ? " either" : "");
            if (at.via)
              diag.attachNote(*at.via) << "with the parameters given here";
          }
      }
    }
    llvm::errs() << "verify-contracts: " << proven << "/" << total
                 << " obligations proven (";
    if (forInstances)
      llvm::errs() << forInstances << " only for the launched instantiations; ";
    llvm::errs() << loops << " loops, " << invariants << " invariants)\n";
  }
};

} // namespace
