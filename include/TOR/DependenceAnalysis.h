#ifndef DEPENDENCE_ANALYSIS_H
#define DEPENDENCE_ANALYSIS_H

#include "TOR/SCEVAnalysis.h"
#include "mlir/IR/Operation.h"
#include <iostream>
#include <memory>
#include <optional>
#include <unordered_map>

namespace mlir {
namespace tor {

struct DependenceResult {
  enum { NotDependent, Dependent, Unknown, Always } type;
  int dist;
  void dump(llvm::raw_ostream &os) {
    if (type == Unknown)
      os << "Dependence is unknown\n";
    else if (type == Dependent)
      os << "Dependent with distance = " << dist << "\n";
    else if (type == Always)
      os << "Dependent in all iterations\n";
    else
      os << "Not dependent\n";
  }
};

struct ValueHash {
  size_t operator()(const Value &v) const noexcept {
    return std::hash<uint64_t>{}((uint64_t)(v.getImpl()));
  }
};

/**
 * @brief This class provides an interface to SCEVAnalysis
 * @todo Currently can't figure out a way to incorporate mlir analysis manager
 * Should add that into the it;
 */
class DependenceAnalysis {
private:
  std::unordered_map<Operation *, SCEVAnalysis> cache;
  std::unordered_map<Value, unique_ptr<SCEVChain>, ValueHash> dangling_value;

public:
  using RetTy = DependenceResult;

  /**
   * @brief Get the scev analysis for this op;
   *
   * @param op
   * @return SCEVAnalysis&
   */
  SCEVAnalysis &get_analysis(Operation *op);

  SCEVChain *get_dangling_value(Value v);

  std::optional<SCEVChain *> get_scev_chain(Value v);

  DependenceResult query_scev_chain(int lenA, SCEVChain *a, int lenB, SCEVChain *b);

  /**
   * @brief Value Range [a , a + lenA) equals to Value Range [b , b + lenB) after x-iterations.
   *
   * @param a
   * @param b
   * @return RetTy
   */
  RetTy get_distance(int lenA, Value a, int lenB, Value b, Operation *loop);
};

} // namespace tor
} // namespace mlir
#endif