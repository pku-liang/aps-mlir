#ifndef SCEV_ANALYSIS_H
#define SCEV_ANALYSIS_H

#include "TOR/TOR.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

using std::unique_ptr;

namespace mlir {

namespace tor {

/**
 * @brief A monomial. It has few symbols in most cases.
 *
 */
class Monomial {
public:
  llvm::SmallVector<Value, 2> symbols;
  Monomial() { symbols.clear(); }

  Monomial(Value v) { symbols.push_back(v); }

  static bool cmp(Value a, Value b) { return a.getImpl() < b.getImpl(); }

  bool is_constant() const { return symbols.size() == 0; }

  void canonicalize() { std::sort(symbols.begin(), symbols.end(), cmp); }

  Monomial mul(const Monomial &x) const;

  /**
   * @brief Smaller in order;
   *
   */
  friend bool operator<(const Monomial &x, const Monomial &y) {
    if (x.symbols.size() != y.symbols.size())
      return x.symbols.size() < y.symbols.size();
    for (auto xy : llvm::zip(x.symbols, y.symbols)) {
      auto x_value = std::get<0>(xy);
      auto y_value = std::get<1>(xy);
      if (x_value != y_value)
        return cmp(x_value, y_value);
    }
    return false;
  }

  /**
   * @brief belongs to the same monomial
   *
   * @param x
   * @param y
   * @return true
   * @return false
   */
  bool operator==(const Monomial &y) const {
    if (symbols.size() != y.symbols.size())
      return false;
    for (auto xy : llvm::zip(symbols, y.symbols)) {
      auto x_value = std::get<0>(xy);
      auto y_value = std::get<1>(xy);
      if (x_value != y_value)
        return false;
    }
    return true;
  }

  bool operator!=(const Monomial &y) const { return !(*this == y); }

  /**
   * @brief A dumb simple hash function
   *
   * @return std::size_t
   */
  std::size_t signature() const {
    std::size_t result = 1;
    for (auto x : symbols)
      result = result * ((std::size_t)x.getImpl());
    return result;
  }
};

struct HashMonomial {
  std::size_t operator()(const Monomial &s) const noexcept {
    return s.signature();
  }
};

/**
 * @brief Multivariate Polynomial used in SCEV
 *
 */
class Polynomial {
public:
  std::unordered_map<Monomial, int, HashMonomial> monomials;

public:
  Polynomial(int c = 0) { monomials.insert({Monomial(), c}); }

  Polynomial(Value v, int c = 1) { monomials.insert({Monomial(v), c}); }

  bool is_constant() const {
    return monomials.size() == 1 && monomials.begin()->first.is_constant();
  }

  std::optional<int> get_constant() const {
    if (!is_constant())
      return std::nullopt;
    return std::optional<int>(monomials.begin()->second);
  }

  void canonicalize();

  void inplace_add(const Polynomial &lhs);

  void inplace_sub(const Polynomial &lhs);

  static Polynomial add(const Polynomial &lhs1, const Polynomial &lhs2);

  static Polynomial sub(const Polynomial &lhs1, const Polynomial &lhs2);

  static Polynomial mul(const Polynomial &lhs1, const Polynomial &lhs2);

  static Polynomial div(const Polynomial &lhs1, const Polynomial &lhs2);

  bool operator==(const Polynomial &y) const;

  bool operator!=(const Polynomial &y) const { return !(*this == y); }

  void dump(llvm::raw_ostream &os) const {
    for (const auto &kv : monomials)
      os << kv.second << "(" << kv.first.signature() << "), ";
  }
};

struct DistanceResult {
  enum {
    NoSolution,
    Always,
    Solution,
    Unknown,
  } type;
  int dist;
};

class SCEVChain {
public:
  Polynomial coeff;
  enum { OP_ADD, OP_MUL, OP_NONE } op;
  bool hasDivOp = false;
  unique_ptr<SCEVChain> rest;

  SCEVChain() : coeff(0) {
    op = OP_NONE;
    rest = nullptr;
  }

  static unique_ptr<SCEVChain> from_constant(Polynomial c) {
    auto scev = std::make_unique<SCEVChain>();
    scev->coeff = c;
    scev->op = OP_NONE;
    return scev;
  }

  /**
   * @brief Build scevchain from given coefficients
   *
   * @param coeffs
   * @return SCEVChain*
   */
  static unique_ptr<SCEVChain>
  from_array(const std::vector<Polynomial> &coeffs);

  /**
   * @brief Compute the value of this scala at certain iteration
   *
   * @param iteration
   * @return int
   */
  std::optional<int> compute_value_at(int iteration);

  /**
   * @brief Return a copy of the current scev
   *
   * @return SCEVChain*
   */
  unique_ptr<SCEVChain> copy() const {
    auto ret = std::make_unique<SCEVChain>();
    ret->coeff = this->coeff;
    ret->op = this->op;
    ret->hasDivOp = this->hasDivOp;
    if (this->rest.get() != nullptr)
      ret->rest = this->rest->copy();
    return ret;
  }

  /**
   * @brief {x, +, y, +, 0} -> {x, +, y}
   *
   */
  void prune() {
    if (this->rest != nullptr && this->rest->coeff == 0 &&
        this->rest->rest == nullptr) {
      this->rest = nullptr;
    }
  }

  bool hasDiv();

  /**
   * @brief lhs1 == lhs2
   *
   * @param lhs1
   * @param lhs2
   * @return true
   * @return false
   */
  static bool equal(const SCEVChain *lhs1, const SCEVChain *lhs2);

  /**
   * @brief lhs1 + lhs2
   *
   * @param lhs1
   * @param lhs2
   * @return SCEVChain*
   */
  static unique_ptr<SCEVChain> add(const SCEVChain *lhs1,
                                   const SCEVChain *lhs2);

  /**
   * @brief lhs + c
   *
   * @param lhs
   * @param c
   * @return SCEVChain*
   */
  static unique_ptr<SCEVChain> add_const(const SCEVChain *lhs,
                                         const Polynomial &c);

  /**
   * @brief -lhs
   *
   * @param lhs
   * @return unique_ptr<SCEVChain> -lhs
   */
  static unique_ptr<SCEVChain> neg(const SCEVChain *lhs);

  /**
   * @brief lhs1 - lhs2
   *
   * @param lhs1
   * @param lhs2
   * @return SCEVChain*
   */
  static unique_ptr<SCEVChain> minus(const SCEVChain *lhs1,
                                     const SCEVChain *lhs2);

  /**
   * @brief lhs - c
   *
   * @param lhs
   * @param c
   * @return SCEVChain*
   */
  static unique_ptr<SCEVChain> minus_const(const SCEVChain *lhs,
                                           const Polynomial &c);

  /**
   * @brief lhs1 * lhs2
   *
   * @param lhs1
   * @param lhs2
   * @return SCEVChain*
   */
  static unique_ptr<SCEVChain> mul(const SCEVChain *lhs1,
                                   const SCEVChain *lhs2);

  /**
   * @brief lhs * c
   *
   * @param lhs
   * @param c
   * @return SCEVChain*
   */
  static unique_ptr<SCEVChain> mul_const(const SCEVChain *lhs,
                                         const Polynomial &c);

  /**
   * @brief lhs / c
   *
   * @param lhs
   * @param c
   * @return SCEVChain*
   */
  static unique_ptr<SCEVChain> div_const(const SCEVChain *lhs,
                                         const Polynomial &c);

  /**
   * @brief Find a 'd' such that a(i+d) = b(i)
   *
   * @param a
   * @param b
   * @return std::optional<int>, nullopt if cannot compute
   */
  static DistanceResult query_distance(SCEVChain *a, SCEVChain *b);

  /**
   * @brief Find a 'd' such that some value in range a(i+d) ~ (a(i+d) + lenA - 1)
   *        overlap with range  = b(i) ~ b(i) + lenB - 1
   *
   * @param a
   * @param b
   * @return std::optional<int>, nullopt if cannot compute
   */
  static DistanceResult query_distance(int lenA, SCEVChain *a, int lenB, SCEVChain *b);

  void dump(llvm::raw_ostream &os) const {
    os << "{";
    coeff.dump(os);
    os << ", +, ";
    if (rest.get() != nullptr)
      rest->dump(os);
    os << "}";
    os << hasDivOp << " ";
  }
};

/**
 * @brief This information is computed during SCEV analysis
 * Each variable can have at most one iteration arguments.
 */
struct ComputeInfo {
  enum { Composite, Pure } type;
  struct {
    Value v;
    int sign;
  } var;

  unique_ptr<SCEVChain> scev;

  ComputeInfo() = default;

  ComputeInfo(int ty, Value var_v, int var_sign, unique_ptr<SCEVChain> s)
      : type(static_cast<decltype(type)>(ty)), var({var_v, var_sign}),
        scev(std::move(s)) {}

  ComputeInfo(const ComputeInfo &x)
      : type(x.type), var(x.var), scev(x.scev->copy()) {}

  void operator=(const ComputeInfo &x) {
    type = x.type;
    var = x.var;
    scev = x.scev->copy();
  }

  inline bool is_pure() const { return type == Pure; }
};

class SCEVAnalysis {
private:
  llvm::DenseMap<Value, unique_ptr<SCEVChain>> scev_results;

  llvm::DenseMap<Value, ComputeInfo> compute_results;

  Operation *containingLoop;

  std::optional<ComputeInfo> get_info(Value v);

  bool is_outside(Value v);

  void traverse(Block &block);

  void compute_for(ForOp forOp);

  void compute_while(WhileOp whileOp);

  void compute_scev();

public:
  SCEVAnalysis() { containingLoop = nullptr; }

  SCEVAnalysis(Operation *op) {
    containingLoop = op;
    compute_scev();
  }

  SCEVChain *get_scev_chain(Value value);

  bool isInvalidated();
};

std::optional<Polynomial> get_poly_from_value(Value v);

} // namespace tor

} // namespace mlir

#endif