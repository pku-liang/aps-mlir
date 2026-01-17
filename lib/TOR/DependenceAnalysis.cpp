#include "TOR/DependenceAnalysis.h"
#include "Schedule/DbgHelper.h"
#include "mlir/Analysis/AliasAnalysis/LocalAliasAnalysis.h"
#include "mlir/IR/Operation.h"
#include <optional>

using namespace std;

namespace mlir {
namespace tor {

/**
 * @brief Get the parent loop op. Return nullopt if defined in toplevel function
 *
 * @param v
 * @return optional<Operation*>
 */
optional<Operation *> get_parent_loop_op(Value v) {
  auto op = v.getParentRegion()->getParentOp();
  while (op && !llvm::isa<ForOp>(op) && !llvm::isa<WhileOp>(op) &&
         !llvm::isa<FuncOp>(op))
    op = op->getParentOp();

  if (!op)
    return nullopt;

  if (llvm::isa<FuncOp>(op))
    return nullopt;

  return op;
}

Operation *get_parent_ctrl_op(Value v) {
  auto op = v.getParentRegion()->getParentOp();
  while (op && !llvm::isa<ForOp>(op) && !llvm::isa<WhileOp>(op) &&
         !llvm::isa<FuncOp>(op) && !llvm::isa<DesignOp>(op))
    op = op->getParentOp();
  assert(op != nullptr);
  return op;
}

SCEVAnalysis &DependenceAnalysis::get_analysis(Operation *op) {
  if (cache.find(op) == cache.end())
    cache[op] = SCEVAnalysis(op);
  return cache[op];
}

SCEVChain *DependenceAnalysis::get_dangling_value(Value v) {
  if (dangling_value.find(v) == dangling_value.end()) {
    auto poly = get_poly_from_value(v);
    assert(poly.has_value());
    dangling_value[v] = SCEVChain::from_constant(poly.value());
  }
  return dangling_value[v].get();
}

optional<SCEVChain *> DependenceAnalysis::get_scev_chain(Value v) {
  auto op = get_parent_loop_op(v);
  if (op.has_value()) {
    auto &scev_analysis = get_analysis(op.value());
    auto result = scev_analysis.get_scev_chain(v);
    if (result != nullptr) {
      return result;
    }
  }

  return get_dangling_value(v);
}

DependenceResult DependenceAnalysis::query_scev_chain(int lenA, SCEVChain *a, int lenB, SCEVChain *b) {
  DistanceResult result;
  if (lenA == 1 && lenB == 1) {
    result = SCEVChain::query_distance(a, b);
  } else {
    result = SCEVChain::query_distance(lenA, a, lenB, b);
  }

  if (result.type == DistanceResult::Unknown)
    return {DependenceResult::Unknown, 0};

  if (result.type == DistanceResult::Always)
    return {DependenceResult::Always, 0};

  if (result.type == DistanceResult::NoSolution || result.dist > 0)
    return {DependenceResult::NotDependent, 0};

  return {DependenceResult::Dependent, -result.dist};
}

/**
 * @brief Check if value range [*a*, *a+len*) equals to *b* after *d* iterations of *loop*
 *        in the current iteration, i.e. a(t) = b(t+d).
 *
 * @param a
 * @param b
 * @param loop
 * @return DependenceResult
 */
DependenceResult DependenceAnalysis::get_distance(int lenA, Value a, int lenB, Value b,
                                                  Operation *loop) {
  auto parent_ctrl_a = get_parent_ctrl_op(a);
  auto parent_ctrl_b = get_parent_ctrl_op(b);
  // First check if a and b belong to the same loop

  if (parent_ctrl_a == parent_ctrl_b && parent_ctrl_a == loop) {
    auto scev_chain_a = get_scev_chain(a);
    auto scev_chain_b = get_scev_chain(b);

    if (!scev_chain_a.has_value() || !scev_chain_b.has_value())
      return {DependenceResult::Unknown, 0};

    return query_scev_chain(lenA, scev_chain_a.value(), lenB, scev_chain_b.value());
  }

  assert(parent_ctrl_a->isAncestor(loop));
  assert(parent_ctrl_b->isAncestor(loop));

  // Then check if a or b in the outer loop
  optional<SCEVChain *> scev_chain_a, scev_chain_b;
  scev_chain_a =
      parent_ctrl_a == loop ? get_scev_chain(a) : get_dangling_value(a);
  scev_chain_b =
      parent_ctrl_b == loop ? get_scev_chain(b) : get_dangling_value(b);

  if (!scev_chain_a.has_value() || !scev_chain_b.has_value())
    return {DependenceResult::Unknown, 0};

  return query_scev_chain(lenA, scev_chain_a.value(), lenB, scev_chain_b.value());
}

} // namespace tor
} // namespace mlir