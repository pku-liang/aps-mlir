/**
 * @file SCEVAnalysis.cpp
 * @author Jin Luo
 * @brief This file implements SCEV analysis on tor.for operations
 * @version 0.1
 * @date 2023-08-15
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "TOR/SCEVAnalysis.h"
#include "Schedule/DbgHelper.h"
#include "TOR/TOR.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "scev-analysis"

using namespace std;

namespace mlir {
namespace tor {

Monomial Monomial::mul(const Monomial &x) const {
  Monomial result;
  result.symbols.insert(result.symbols.end(), symbols.begin(), symbols.end());
  result.symbols.insert(result.symbols.end(), x.symbols.begin(),
                        x.symbols.end());
  result.canonicalize();
  return result;
}

void Polynomial::canonicalize() {
  for (auto iter = monomials.begin(); iter != monomials.end();) {
    if (iter->second == 0 && !iter->first.is_constant()) {
      iter = monomials.erase(iter);
    } else {
      iter++;
    }
  }
}

void Polynomial::inplace_add(const Polynomial &lhs) {
  for (const auto &x : lhs.monomials) {
    auto &v = monomials[x.first];
    v += x.second;
    if (v == 0 && !x.first.is_constant())
      monomials.erase(x.first);
  }
}

void Polynomial::inplace_sub(const Polynomial &lhs) {
  for (const auto &x : lhs.monomials) {
    auto &v = monomials[x.first];
    v -= x.second;
    if (v == 0 && !x.first.is_constant())
      monomials.erase(x.first);
  }
}

Polynomial Polynomial::add(const Polynomial &lhs1, const Polynomial &lhs2) {
  Polynomial result = lhs1;
  result.inplace_add(lhs2);
  return result;
}

Polynomial Polynomial::sub(const Polynomial &lhs1, const Polynomial &lhs2) {
  Polynomial result = lhs1;
  result.inplace_sub(lhs2);
  return result;
}

Polynomial Polynomial::mul(const Polynomial &lhs1, const Polynomial &lhs2) {
  Polynomial result = Polynomial();
  for (const auto &x : lhs1.monomials)
    for (const auto &y : lhs2.monomials) {
      auto prod = x.first.mul(y.first);
      result.monomials[prod] += x.second * y.second;
    }
  result.canonicalize();
  return result;
}

Polynomial Polynomial::div(const Polynomial &lhs1, const Polynomial &lhs2) {
  Polynomial result = Polynomial();
  auto divisorConstant = lhs2.get_constant();
  assert(divisorConstant.has_value() && "Divisor must be a constant polynomial");
  int divisor = divisorConstant.value();
  for (const auto &x : lhs1.monomials) {
    result.monomials[x.first] = x.second / divisor;
  }
  result.canonicalize();
  return result;
}

bool Polynomial::operator==(const Polynomial &y) const {
  if (monomials.size() != y.monomials.size())
    return false;
  for (auto xy : llvm::zip(monomials, y.monomials)) {
    auto &xm = std::get<0>(xy);
    auto &ym = std::get<1>(xy);
    if (xm.first != ym.first)
      return false;
    if (xm.second != ym.second)
      return false;
  }
  return true;
}

unique_ptr<SCEVChain>
SCEVChain::from_array(const std::vector<Polynomial> &coeffs) {
  unique_ptr<SCEVChain> cur = nullptr;
  for (auto x : llvm::reverse(coeffs)) {
    auto nxt = SCEVChain::from_constant(x);
    nxt->rest = std::move(cur);
    nxt->op = OP_ADD;
    cur = std::move(nxt);
  }
  return cur;
}

optional<int> SCEVChain::compute_value_at(int iteration) {
  SCEVChain *current = this;
  int result = 0, binom = 1, idx = 0;
  while (current != nullptr) {
    auto c = current->coeff.get_constant();
    if (!c.has_value())
      return nullopt;
    result += c.value() * binom;
    binom = binom * (iteration - idx) / (idx + 1);
    idx = idx + 1;
    current = current->rest.get();
  }
  return optional<int>(result);
}

bool SCEVChain::hasDiv() {
  SCEVChain *current = this;
  while (current != nullptr) {
    if (current->hasDivOp)
      return true;
    current = current->rest.get();
  }
  return false;
}

bool SCEVChain::equal(const SCEVChain *lhs1, const SCEVChain *lhs2) {
  // should be careful
  if (lhs1 == nullptr && lhs2 == nullptr)
    return true;

  if (lhs1 == nullptr || lhs2 == nullptr)
    return false;

  if (lhs1->coeff != lhs2->coeff)
    return false;

  if (lhs1->hasDivOp != lhs2->hasDivOp) {
    return false;
  }
  return SCEVChain::equal(lhs1->rest.get(), lhs2->rest.get());
}

unique_ptr<SCEVChain> SCEVChain::add(const SCEVChain *lhs1,
                                     const SCEVChain *lhs2) {
  auto ret = std::make_unique<SCEVChain>();
  if (lhs1 == nullptr && lhs2 == nullptr)
    return nullptr;
  if (lhs1 == nullptr)
    return lhs2->copy();
  if (lhs2 == nullptr)
    return lhs1->copy();

  ret->coeff = Polynomial::add(lhs1->coeff, lhs2->coeff);
  ret->op = OP_ADD;
  ret->rest = SCEVChain::add(lhs1->rest.get(), lhs2->rest.get());
  ret->prune();
  ret->hasDivOp = lhs1->hasDivOp | lhs2->hasDivOp;
  return ret;
}

unique_ptr<SCEVChain> SCEVChain::add_const(const SCEVChain *lhs1,
                                           const Polynomial &c) {
  if (lhs1 == nullptr)
    return SCEVChain::from_constant(c);

  auto ret = lhs1->copy();
  ret->coeff = Polynomial::add(ret->coeff, c);
  return ret;
}

unique_ptr<SCEVChain> SCEVChain::neg(const SCEVChain *lhs) {
  if (lhs == nullptr)
    return nullptr;

  auto ret = std::make_unique<SCEVChain>();

  ret->coeff = Polynomial::sub(0, lhs->coeff);
  ret->op = lhs->op;
  ret->hasDivOp = lhs->hasDivOp;
  if (lhs->op == OP_ADD) {
    ret->rest = SCEVChain::neg(lhs->rest.get());
  } else {
    ret->rest = lhs->rest->copy();
  }

  return ret;
}

unique_ptr<SCEVChain> SCEVChain::minus(const SCEVChain *lhs1,
                                       const SCEVChain *lhs2) {
  auto ret = std::make_unique<SCEVChain>();
  if (lhs1 == nullptr && lhs2 == nullptr)
    return nullptr;
  if (lhs1 == nullptr)
    return SCEVChain::neg(lhs2)->copy();
  if (lhs2 == nullptr)
    return lhs1->copy();

  ret->coeff = Polynomial::sub(lhs1->coeff, lhs2->coeff);
  ret->op = OP_ADD;
  ret->rest = SCEVChain::minus(lhs1->rest.get(), lhs2->rest.get());
  ret->prune();
  ret->hasDivOp = lhs1->hasDivOp | lhs2->hasDivOp;
  return ret;
}

unique_ptr<SCEVChain> SCEVChain::minus_const(const SCEVChain *lhs1,
                                             const Polynomial &c) {
  if (lhs1 == nullptr)
    return SCEVChain::from_constant(Polynomial::sub(0, c));

  auto ret = lhs1->copy();
  ret->coeff = Polynomial::sub(ret->coeff, c);
  return ret;
}

unique_ptr<SCEVChain> SCEVChain::mul(const SCEVChain *lhs1,
                                     const SCEVChain *lhs2) {
  if (lhs1 == nullptr || lhs2 == nullptr)
    return SCEVChain::from_constant(0);

  auto ret = std::make_unique<SCEVChain>();
  ret->coeff = Polynomial::mul(lhs1->coeff, lhs2->coeff);
  ret->op = OP_ADD;
  ret->hasDivOp = lhs1->hasDivOp | lhs2->hasDivOp;
  auto tmp1 = SCEVChain::mul_const(lhs1->rest.get(), lhs2->coeff);
  auto tmp2 = SCEVChain::mul_const(lhs2->rest.get(), lhs2->coeff);
  auto tmp3 = SCEVChain::add(tmp1.get(), tmp2.get());
  auto tmp4 = SCEVChain::mul(lhs1->rest.get(), lhs2->rest.get());
  ret->rest = SCEVChain::add(tmp3.get(), tmp4.get());
  /* ret->rest = SCEVChain::add(
      SCEVChain::add(SCEVChain::mul_const(lhs1->rest.get(), lhs2->coeff).get(),
                     SCEVChain::mul_const(lhs2->rest.get(),
     lhs2->coeff).get()).get(), SCEVChain::mul(lhs1->rest.get(),
     lhs2->rest.get()).get()); */
  ret->prune();

  return ret;
}

unique_ptr<SCEVChain> SCEVChain::mul_const(const SCEVChain *lhs,
                                           const Polynomial &c) {
  // Special case for c = 0
  if (lhs == nullptr || c == 0)
    return SCEVChain::from_constant(0);

  auto ret = lhs->copy();
  SCEVChain *cur = ret.get();
  while (cur != nullptr) {
    cur->coeff = Polynomial::mul(cur->coeff, c);
    cur = cur->rest.get();
  }

  return ret;
}

unique_ptr<SCEVChain> SCEVChain::div_const(const SCEVChain *lhs1,
                                           const Polynomial &c) {
  if (lhs1 == nullptr) {
    return SCEVChain::from_constant(0);
  }

  auto ret = std::make_unique<SCEVChain>();
  ret->hasDivOp = true;
  ret->coeff = Polynomial::div(lhs1->coeff, c);
  ret->op = OP_ADD;
  auto tmp = SCEVChain::div_const(lhs1->rest.get(), c);
  ret->rest = std::move(tmp);
  ret->prune();

  return ret;
}

DistanceResult SCEVChain::query_distance(SCEVChain *a, SCEVChain *b) {
  // The ultimate goal is to find every i, j such that a(i) == b(j)
  // Currently trying is solve a(i + d) = b(i)
  // Let a = {c1, +, stride1}, b = {c2, +, stride2}, there is a constant
  // d iff stride1 == stride2

  auto &c1 = a->coeff, &c2 = b->coeff;
  auto stride1 = a->rest.get(), stride2 = b->rest.get();

  if (!SCEVChain::equal(stride1, stride2))
    return {DistanceResult::Unknown, 0};

  if (stride1 == nullptr && c1 == c2)
    return {DistanceResult::Always, 0};

  auto diff = Polynomial::sub(c2, c1);
  if (!diff.is_constant())
    return {DistanceResult::Unknown, 0};

  auto c = diff.get_constant().value();
  if (c == 0)
    return {DistanceResult::Solution, 0};

  auto stride = stride1;
  if (stride == nullptr)
    return {DistanceResult::NoSolution, 0};

  if (stride->rest != nullptr)
    return {DistanceResult::Unknown, 0};

  if (!stride->coeff.is_constant())
    return {DistanceResult::Unknown, 0};

  int st_c = stride->coeff.get_constant().value();
  if (st_c == 0 || c % st_c != 0)
    return {DistanceResult::NoSolution, 0};

  return {DistanceResult::Solution, c / st_c};
}

DistanceResult SCEVChain::query_distance(int lenA, SCEVChain *a, int lenB, SCEVChain *b) {
  // The ultimate goal is to find every i, j such that a(i) == b(j)
  // Currently trying is solve a(i + d) = b(i)
  // Let a = {c1, +, stride1}, b = {c2, +, stride2}, there is a constant
  // d iff stride1 == stride2
  // todo
  auto &c1 = a->coeff, &c2 = b->coeff;
  auto stride1 = a->rest.get(), stride2 = b->rest.get();

  if (!SCEVChain::equal(stride1, stride2))
    return {DistanceResult::Unknown, 0};

  auto diff = Polynomial::sub(c2, c1);
  int lenDiff = std::abs(lenB - lenA);
  if (stride1 == nullptr) {
    if (diff.is_constant()) {
      if (std::abs(diff.get_constant().value()) < lenDiff) {
        return {DistanceResult::Always, 0};
      }
    }
  }

  if (stride1 == nullptr && c1 == c2)
    return {DistanceResult::Always, 0};

  if (!diff.is_constant())
    return {DistanceResult::Unknown, 0};

  auto c = diff.get_constant().value();
  if (std::abs(c) <= lenDiff)
    return {DistanceResult::Solution, 0};

  if (stride1 == nullptr)
    return {DistanceResult::NoSolution, 0};

  if (stride1->rest != nullptr)
    return {DistanceResult::Unknown, 0};

  if (!stride1->coeff.is_constant())
    return {DistanceResult::Unknown, 0};

  int st_c = stride1->coeff.get_constant().value();
  if (st_c == 0 || c % st_c != 0)
    return {DistanceResult::NoSolution, 0};

  return {DistanceResult::Solution, c / st_c};
}

Operation *get_single_yield(Operation *op) {
  if (auto forOp = llvm::dyn_cast<ForOp>(op)) {
    return &forOp.getBody()->getOperations().back();
  }
  if (auto whileOp = llvm::dyn_cast<WhileOp>(op)) {
    return &whileOp.getAfter().getBlocks().begin()->back();
  }
  llvm::errs() << "Function get_single_yield should not be called on op ";
  op->dump();
  assert(0);
}

optional<int> get_const_from_value(Value v) {
  Operation *op = v.getDefiningOp();
  if (op == nullptr) {
    // v is not a direct result of an operation.
    return nullopt;
  }

  if (auto constIndexOp = llvm::dyn_cast<arith::ConstantIndexOp>(op))
    return optional<int>(constIndexOp.value());
  if (auto constIntOp = llvm::dyn_cast<arith::ConstantIntOp>(op))
    return optional<int>(constIntOp.value());

  return nullopt;
}

/**
 * @brief Search through the definition of v. So that we can compute the
 * relationship between N+i+1 and N+i+2
 * @param v
 * @return Polynomial
 */
Polynomial traverse_def_tree(Value v) {
  auto op = v.getDefiningOp();
  if (op == nullptr)
    return Polynomial(v);

  auto c = get_const_from_value(v);
  if (c.has_value())
    return Polynomial(c.value());

  if (auto addIOp = llvm::dyn_cast<AddIOp>(op)) {
    auto lhs1 = addIOp.getOperand(0);
    auto lhs2 = addIOp.getOperand(1);
    auto c1 = get_const_from_value(lhs1);
    auto c2 = get_const_from_value(lhs2);

    if (!c1.has_value() && !c2.has_value())
      goto not_expandable;

    auto poly1 = traverse_def_tree(lhs1);
    auto poly2 = traverse_def_tree(lhs2);

    return Polynomial::add(poly1, poly2);
  } else if (auto subIOp = llvm::dyn_cast<SubIOp>(op)) {
    auto lhs1 = subIOp.getOperand(0);
    auto lhs2 = subIOp.getOperand(1);
    auto c1 = get_const_from_value(lhs1);
    auto c2 = get_const_from_value(lhs2);

    if (!c1.has_value() && !c2.has_value())
      goto not_expandable;

    auto poly1 = traverse_def_tree(lhs1);
    auto poly2 = traverse_def_tree(lhs2);

    return Polynomial::sub(poly1, poly2);
  } else if (auto mulIOp = llvm::dyn_cast<MulIOp>(op)) {
    auto lhs1 = mulIOp.getOperand(0);
    auto lhs2 = mulIOp.getOperand(1);
    auto c1 = get_const_from_value(lhs1);
    auto c2 = get_const_from_value(lhs2);

    if (!c1.has_value() && !c2.has_value())
      goto not_expandable;

    auto poly1 = traverse_def_tree(lhs1);
    auto poly2 = traverse_def_tree(lhs2);

    return Polynomial::mul(poly1, poly2);
  }

not_expandable:
  return Polynomial(v);
}

optional<Polynomial> get_poly_from_value(Value v) {
  if (!v.getType().isIntOrIndex())
    return nullopt;
  auto c = get_const_from_value(v);
  if (c.has_value())
    return optional<Polynomial>(c.value());
  return traverse_def_tree(v);
}

bool SCEVAnalysis::is_outside(Value v) {
  auto op = v.getParentRegion()->getParentOp();
  while (op) {
    if (op == containingLoop)
      return false;
    op = op->getParentOp();
  }
  return true;
}

optional<ComputeInfo> SCEVAnalysis::get_info(Value v) {
  auto c = get_const_from_value(v);

  if (c.has_value()) {
    return ComputeInfo(ComputeInfo::Pure, Value(), 1,
                       SCEVChain::from_constant(c.value()));
  }

  if (is_outside(v))
    return ComputeInfo(ComputeInfo::Pure, Value(), 1,
                       SCEVChain::from_constant(v));

  if (compute_results.find(v) != compute_results.end()) {
    return compute_results.lookup(v);
  }

  return nullopt;
}

void SCEVAnalysis::traverse(Block &block) {
  for (auto &op : block.getOperations()) {
    if (auto mulIOp = llvm::dyn_cast<MulIOp>(op)) {
      auto lhs1 = get_info(mulIOp.getLhs());
      auto lhs2 = get_info(mulIOp.getRhs());
      auto result = mulIOp.getResult();
      // For MulIOp, we can only handle the multiplication of two known chrec
      if (lhs1.has_value() && lhs2.has_value() && lhs1.value().is_pure() &&
          lhs2.value().is_pure()) {
        auto result_scev =
            SCEVChain::mul(lhs1.value().scev.get(), lhs2.value().scev.get());
        compute_results[result] =
            ComputeInfo(ComputeInfo::Pure, Value(), 1, std::move(result_scev));
      } else {
        assert(compute_results.find(result) == compute_results.end());
      }
    } else if (auto addIOp = llvm::dyn_cast<AddIOp>(op)) {
      auto lhs1 = get_info(addIOp.getLhs());
      auto lhs2 = get_info(addIOp.getRhs());
      auto result = addIOp.getResult();
      bool lhs1_is_pure = lhs1.has_value() && lhs1.value().is_pure();
      bool lhs2_is_pure = lhs2.has_value() && lhs2.value().is_pure();

      // If both two operands are not pure scev, then we cannot handle this case
      if (!lhs1_is_pure && !lhs2_is_pure) {
        assert(compute_results.find(result) == compute_results.end());
        continue;
      }
      if (!lhs1.has_value() || !lhs2.has_value())
        continue;

      auto result_scev =
          SCEVChain::add(lhs1.value().scev.get(), lhs2.value().scev.get());

      if (lhs1_is_pure && lhs2_is_pure) {
        compute_results[result] =
            ComputeInfo(ComputeInfo::Pure, Value(), 1, std::move(result_scev));
      } else if (lhs1_is_pure) {
        compute_results[result] =
            ComputeInfo(ComputeInfo::Composite, lhs2.value().var.v,
                        lhs1.value().var.sign, std::move(result_scev));

      } else if (lhs2_is_pure) {
        compute_results[result] =
            ComputeInfo(ComputeInfo::Composite, lhs1.value().var.v,
                        lhs1.value().var.sign, std::move(result_scev));
      }

    } else if (auto subIOp = llvm::dyn_cast<SubIOp>(op)) {
      auto lhs1 = get_info(subIOp.getLhs());
      auto lhs2 = get_info(subIOp.getRhs());
      auto result = subIOp.getResult();
      bool lhs1_is_pure = lhs1.has_value() && lhs1.value().is_pure();
      bool lhs2_is_pure = lhs2.has_value() && lhs2.value().is_pure();

      // If both two operands are not pure scev, then we cannot handle this case
      if (!lhs1_is_pure && !lhs2_is_pure) {
        // assert(compute_results.find(result) == compute_results.end());
        continue;
      }
      if (!lhs1.has_value() || !lhs2.has_value())
        continue;

      auto result_scev =
          SCEVChain::minus(lhs1.value().scev.get(), lhs2.value().scev.get());

      if (lhs1_is_pure && lhs2_is_pure) {
        compute_results[result] =
            ComputeInfo(ComputeInfo::Pure, Value(), 1, std::move(result_scev));
      } else if (lhs1_is_pure) {
        compute_results[result] =
            ComputeInfo(ComputeInfo::Composite, lhs2.value().var.v,
                        -lhs2.value().var.sign, std::move(result_scev));
      } else if (lhs2_is_pure) {
        compute_results[result] =
            ComputeInfo(ComputeInfo::Composite, lhs1.value().var.v,
                        lhs1.value().var.sign, std::move(result_scev));
      }
    } else if (auto ifOp = llvm::dyn_cast<IfOp>(op)) {
      if (!ifOp.getThenRegion().empty())
        traverse(*ifOp.getThenRegion().getBlocks().begin());
      if (!ifOp.getElseRegion().empty())
        traverse(*ifOp.getElseRegion().getBlocks().begin());
      // TODO: How to handle Yield op? In most case it would be OK to dismiss
      // it.
    } else if (auto forOp = llvm::dyn_cast<ForOp>(op)) {
      // Here we don't calculate SCEV for inner loop
      traverse(*forOp.getRegion().getBlocks().begin());
    } else if (auto whileOp = llvm::dyn_cast<WhileOp>(op)) {
      traverse(*whileOp.getBefore().getBlocks().begin());
      traverse(*whileOp.getAfter().getBlocks().begin());
    } else if (auto shLIOp = llvm::dyn_cast<arith::ShLIOp>(op)) {
      auto lhs1 = get_info(shLIOp.getLhs());
      auto lhs2 = get_const_from_value(shLIOp.getRhs());
      auto result = shLIOp.getResult();

      if (lhs1.has_value() && lhs2.has_value() && lhs1.value().is_pure()) {
        auto value = lhs2.value();
        auto result_scev = SCEVChain::mul(lhs1.value().scev.get(), SCEVChain::from_constant(1 << value).get());
        compute_results[result] = ComputeInfo(ComputeInfo::Pure, Value(), 1, std::move(result_scev));
      } else {
        assert(compute_results.find(result) == compute_results.end());
      }
    } else if (auto divOp = llvm::dyn_cast<arith::DivUIOp>(op)) {
      auto lhs1 = get_info(divOp.getLhs());
      auto lhs2 = get_const_from_value(divOp.getRhs());
      auto result = divOp.getResult();

      if (lhs1.has_value() && lhs2.has_value() && lhs1.value().is_pure()) {
        auto value = lhs2.value();
        auto result_scev = SCEVChain::div_const(lhs1.value().scev.get(), value);
        compute_results[result] = ComputeInfo(ComputeInfo::Pure, Value(), 1, std::move(result_scev));
      } else {
        assert(compute_results.find(result) == compute_results.end());
      }
    }
  }
} // namespace tor

void SCEVAnalysis::compute_for(ForOp forOp) {
  /* compute scev_chain for initial args */
  auto indVar = forOp.getInductionVar();
  auto lb = get_poly_from_value(forOp.getLowerBound());
  auto step = get_poly_from_value(forOp.getStep());
  if (lb.has_value() && step.has_value())
    scev_results[indVar] = SCEVChain::from_array({lb.value(), step.value()});

  int updated = false;
  do {
    updated = false;
    for (auto iter_arg : forOp.getBody()->getArguments()) {
      if (scev_results.find(iter_arg) != scev_results.end())
        compute_results[iter_arg] = ComputeInfo(ComputeInfo::Pure, Value(), 1,
                                                scev_results[iter_arg]->copy());
      else
        compute_results[iter_arg] = ComputeInfo(
            ComputeInfo::Composite, iter_arg, 1, SCEVChain::from_constant(0));
    }
    

    traverse(*forOp.getRegion().getBlocks().begin());
    auto yieldOp =
        llvm::dyn_cast<YieldOp>(get_single_yield(forOp.getOperation()));
    for (const auto &iter_arg : llvm::enumerate(forOp.getRegionIterArgs())) {
      auto idx = iter_arg.index();
      Value v = iter_arg.value();
      Value operand = forOp.getOperand(idx + 1);

      auto init_value = get_poly_from_value(operand);
      if (!init_value.has_value())
        continue;

      Value prev = yieldOp.getOperand(idx);

      if (compute_results.find(prev) == compute_results.end())
        continue;

      auto &info = compute_results[prev];
      if (!info.is_pure() && info.var.v == v) {
        if (info.var.sign == -1) {
          // v' = -v + chrec. How to handle this?
          continue;
        } else {
          // v' = v + chrec;
          auto result = SCEVChain::from_constant(init_value.value());
          result->op = SCEVChain::OP_ADD;
          result->rest = std::move(info.scev);
          scev_results[v] = std::move(result);
        }
      } else {
        // v0 = init_value, v' = chrec
        // TODO: How to handle this. It seems difficult to allow chrec shifting
        continue;
      }
    }

    for (auto &kv : compute_results) {
      if (scev_results.find(kv.first) == scev_results.end() &&
          kv.second.is_pure()) {
        scev_results[kv.first] = std::move(kv.second.scev);
        updated = true;
      }
    }
    compute_results.clear();
  } while (updated == true);
} // namespace tor

void SCEVAnalysis::compute_while(WhileOp whileOp) {
  auto &beforeRegion = whileOp.getBefore();
  auto &afterRegion = whileOp.getAfter();
  bool updated = false;

  do {
    updated = false;
    for (auto iter_arg : beforeRegion.getArguments()) {
      if (scev_results.find(iter_arg) != scev_results.end())
        compute_results[iter_arg] = ComputeInfo(ComputeInfo::Pure, Value(), 1,
                                                scev_results[iter_arg]->copy());
      else
        compute_results[iter_arg] = ComputeInfo(
            ComputeInfo::Composite, iter_arg, 1, SCEVChain::from_constant(0));
    }

    compute_results.clear();
    traverse(*beforeRegion.getBlocks().begin());

    auto condOp = llvm::dyn_cast<ConditionOp>(
        beforeRegion.getBlocks().begin()->getOperations().back());

    for (auto after_arg : llvm::enumerate(afterRegion.getArguments())) {
      int idx = after_arg.index();
      auto arg = after_arg.value();
      auto operand = condOp.getOperand(idx + 1);
      if (compute_results.find(operand) != compute_results.end())
        compute_results[arg] = compute_results[operand];
    }

    traverse(*afterRegion.getBlocks().begin());

    auto yieldOp =
        llvm::dyn_cast<YieldOp>(get_single_yield(whileOp.getOperation()));
    for (const auto &iter_arg : llvm::enumerate(beforeRegion.getArguments())) {
      auto idx = iter_arg.index();
      Value v = iter_arg.value();
      Value operand = whileOp.getOperand(idx);

      auto init_value = get_poly_from_value(operand);
      if (!init_value.has_value())
        continue;

      Value prev = yieldOp.getOperand(idx);

      if (compute_results.find(prev) == compute_results.end())
        continue;

      auto &info = compute_results[prev];
      if (!info.is_pure() && info.var.v == v) {
        if (info.var.sign == -1) {
          // v' = -v + chrec. How to handle this?
          continue;
        } else {
          // v' = v + chrec;
          auto result = SCEVChain::from_constant(init_value.value());
          result->op = SCEVChain::OP_ADD;
          result->rest = std::move(info.scev);
          scev_results[v] = std::move(result);
        }
      } else {
        // v0 = init_value, v' = chrec
        // TODO: How to handle this. It seems difficult to allow chrec shifting
        continue;
      }
    }

    for (auto &kv : compute_results) {
      if (scev_results.find(kv.first) == scev_results.end() &&
          kv.second.is_pure()) {
        scev_results[kv.first] = std::move(kv.second.scev);
        updated = true;
      }
    }

    compute_results.clear();
  } while (updated);
}

void SCEVAnalysis::compute_scev() {
  if (auto forOp = llvm::dyn_cast<ForOp>(containingLoop))
    compute_for(forOp);
  else if (auto whileOp = llvm::dyn_cast<WhileOp>(containingLoop))
    compute_while(whileOp);
}

SCEVChain *SCEVAnalysis::get_scev_chain(Value value) {
  if (scev_results.find(value) != scev_results.end()) {
    return scev_results[value].get();
  } else {
    scev_results[value] = SCEVChain::from_constant(value);
    return scev_results[value].get();
  }
}

bool SCEVAnalysis::isInvalidated() { return false; }

} // namespace tor
} // namespace mlir