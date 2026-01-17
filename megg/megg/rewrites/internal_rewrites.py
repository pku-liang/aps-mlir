"""
Internal rewrite rules for common mathematical operations using egglog.

This module defines algebraic simplification rules, constant folding rules,
and other mathematical transformations that can be applied directly in the
e-graph without external MLIR passes.
"""

import egglog
from megg.egraph.term import Term, LitTerm, eclass_ty
from collections.abc import Iterable
from egglog import *
import logging
logger = logging.getLogger(__name__)
def basic_math_laws() -> egglog.Ruleset:
    """
    Get essential mathematical rewrite rules that are safe and practical.

    Following the pattern from example.py, this focuses on specific useful
    patterns rather than exhaustive algebraic identities to avoid solver crashes.
    """
    # Define type strings (following example.py pattern)
    i32_ty = egglog.String('i32')
    index_ty = egglog.String('index')
    # f32_ty = egglog.String('f32')
    # f64_ty = egglog.String('f64')

    # Define variables
    a, b, c, d = vars_("a b c d", Term)

    rules = []

    # NOTE: Index cast rewrites are disabled because they interfere with
    # skeleton matching by changing the e-graph structure too aggressively.
    # For matching to work, the Term structure must exactly match the pattern,
    # and these rewrites break that invariant.
    
    
    # i32, (x + y) >> 1 ==> (x & y) + ((x ^ y) >> 1)
    rules.append(
        rewrite(
            Term.shrsi(
                Term.add(a, b, i32_ty),
                Term.lit(LitTerm.int(egglog.i64(1)), i32_ty),
                i32_ty
            )
        ).to(
            Term.add(
                Term.and_(a, b, i32_ty),
                Term.shrsi(
                    Term.xor_(a, b, i32_ty),
                    Term.lit(LitTerm.int(egglog.i64(1)), i32_ty),
                    i32_ty
                ),
                i32_ty
            )
        )
    )

    # Apply rules for each type separately (following example.py pattern)
    for ty in [i32_ty, index_ty]:

        # ====================================================================
        # SAFE DISTRIBUTIVITY/FACTORING PATTERNS (following example.py)
        # ====================================================================
        
        # 加法交换律
        rules.append(
            rewrite(
                Term.add(a, b, ty)
            ).to(
                Term.add(b, a, ty)
            )
        )
        # 乘法交换律
        rules.append(
            rewrite(
                Term.mul(a, b, ty)
            ).to(
                Term.mul(b, a, ty)
            )
        )
        # 加法结合律
        rules.append(
            rewrite(
                Term.add(Term.add(a, b, ty), c, ty)
            ).to(
                Term.add(a, Term.add(b, c, ty), ty)
            )
        )
        # 乘法结合律
        rules.append(
            rewrite(
                Term.mul(Term.mul(a, b, ty), c, ty)
            ).to(
                Term.mul(a, Term.mul(b, c, ty), ty)
            )
        )
    
        # (a*c)+(c*b) <=> (a+b)*c - factoring out common factor
        rules.append(
            rewrite(
                Term.add(Term.mul(a, c, ty), Term.mul(c, b, ty), ty)
            ).to(
                Term.mul(Term.add(a, b, ty), c, ty)
            )
        )

        # (c*a)+(b*c) <=> (a+b)*c - factoring with different order
        rules.append(
            rewrite(
                Term.add(Term.mul(c, a, ty), Term.mul(b, c, ty), ty)
            ).to(
                Term.mul(Term.add(a, b, ty), c, ty)
            )
        )

        # (a*c)+(b*c) <=> (a+b)*c - standard factoring
        rules.append(
            rewrite(
                Term.add(Term.mul(a, c, ty), Term.mul(b, c, ty), ty)
            ).to(
                Term.mul(Term.add(a, b, ty), c, ty)
            )
        )

        # (c*a)+(c*b) <=> c*(a+b) - factor out left multiplier
        rules.append(
            rewrite(
                Term.add(Term.mul(c, a, ty), Term.mul(c, b, ty), ty)
            ).to(
                Term.mul(c, Term.add(a, b, ty), ty)
            )
        )
        
        # x1 * x1 + x2 * x2 - 2 * x1 * x2 = (x1 - x2) * (x1 - x2)
        rules.append(
            rewrite(
                Term.sub(
                    Term.add(
                        Term.mul(a, a, ty),
                        Term.mul(b, b, ty),
                        ty
                    ),
                    Term.mul(
                        Term.lit(LitTerm.int(egglog.i64(2)), ty),
                        Term.mul(a, b, ty),
                        ty
                    ),
                    ty
                )
            ).to(
                Term.mul(
                    Term.sub(a, b, ty),
                    Term.sub(a, b, ty),
                    ty
                )
            )
        )
        
        # x1 * z - x2 * z = (x1 - x2) * z
        rules.append(
            rewrite(
                Term.sub(
                    Term.mul(a, c, ty),
                    Term.mul(b, c, ty),
                    ty
                )
            ).to(
                Term.mul(
                    Term.sub(a, b, ty),
                    c,
                    ty
                )
            )
        )

        # ====================================================================
        # SAFE IDENTITY RULES
        # ====================================================================

        # x + 0 = x (additive identity)
        rules.append(
            rewrite(
                Term.add(a, Term.lit(LitTerm.int(egglog.i64(0)), ty), ty)
            ).to(a)
        )

        # x * 1 = x (multiplicative identity)
        rules.append(
            rewrite(
                Term.mul(a, Term.lit(LitTerm.int(egglog.i64(1)), ty), ty)
            ).to(a)
        )

        # x * 0 = 0 (zero property)
        rules.append(
            rewrite(
                Term.mul(a, Term.lit(LitTerm.int(egglog.i64(0)), ty), ty)
            ).to(Term.lit(LitTerm.int(egglog.i64(0)), ty))
        )

        # ====================================================================
        # SAFE SIMPLIFICATION PATTERNS
        # ====================================================================

        # Double negation: -(-x) = x
        rules.append(
            rewrite(Term.neg(Term.neg(a, ty), ty)).to(a)
        )

        # Subtraction to addition: x - y = x + (-y)
        rules.append(
            rewrite(Term.sub(a, b, ty)).to(Term.add(a, Term.neg(b, ty), ty))
        )

        # Addition with negation: x + (-y) = x - y
        rules.append(
            rewrite(Term.add(a, Term.neg(b, ty), ty)).to(Term.sub(a, b, ty))
        )

        # Self subtraction: x - x = 0
        rules.append(
            rewrite(Term.sub(a, a, ty)).to(Term.lit(LitTerm.int(egglog.i64(0)), ty))
        )

        # Additive inverse: x + (-x) = 0
        rules.append(
            rewrite(Term.add(a, Term.neg(a, ty), ty)).to(Term.lit(LitTerm.int(egglog.i64(0)), ty))
        )

        # Additive inverse (commutative): (-x) + x = 0
        rules.append(
            rewrite(Term.add(Term.neg(a, ty), a, ty)).to(Term.lit(LitTerm.int(egglog.i64(0)), ty))
        )

        if ty == i32_ty:
            (ci,) = vars_("ci", egglog.i64)
            const_ci = Term.lit(LitTerm.int(ci), ty)

            # Cancel a matching mul/div by the same non-zero constant: (x * c) / c -> x
            rules.append(
                rewrite(
                    Term.div(
                        Term.mul(a, const_ci, ty),
                        const_ci,
                        ty,
                    )
                ).to(a)
            )

            # Same rule when the constant appears on the left of the multiplication.
            rules.append(
                rewrite(
                    Term.div(
                        Term.mul(const_ci, a, ty),
                        const_ci,
                        ty,
                    )
                ).to(a)
            )

            # ====================================================================
            # SHIFT TO MULTIPLICATION LAWS: x << n => x * 2^n
            # ====================================================================
            # For common shift amounts, convert to multiplication by power of 2
            # This is beneficial as multiplications may be optimized or replaced
            # by custom instructions more easily than shifts
            for n in range(1, 5):  # Support shifts from 1 to 4 bits
                power_of_2 = 1 << n  # Calculate 2^n
                rules.append(
                    rewrite(
                        Term.shl(a, Term.lit(LitTerm.int(egglog.i64(n)), ty), ty)
                    ).to(
                        Term.mul(a, Term.lit(LitTerm.int(egglog.i64(power_of_2)), ty), ty)
                    )
                )
                # DISABLED: Reverse rule (Mul -> Shl) interferes with component matching
                # Component rewrites expect Mul form, not Shl form
                # rules.append(
                #     rewrite(
                #         Term.mul(a, Term.lit(LitTerm.int(egglog.i64(power_of_2)), ty), ty)
                #     ).to(
                #         Term.shl(a, Term.lit(LitTerm.int(egglog.i64(n)), ty), ty)
                #     )
                # )


    basic_math_laws_min = egglog.ruleset(*rules, name='basic_math_laws_min')

    ops = [Term.add, Term.mul, Term.sub, Term.div]

    # ---- 声明一个类型函数：给每个 Term 一个结果类型（作为谓词/guard使用）----


    # ---- 3) 用“关系”做 guard 的条件重写（正确写法）----
    @basic_math_laws_min.register
    def _pushdown_index_cast(x: Term, y: Term, ai: i64) -> Iterable[RewriteOrRule]:
        
        for op in ops:
            # 双边 index_cast 下推
            yield rewrite(
                Term.index_cast(op(x, y, i32_ty), index_ty)
            ).to(
                op(Term.index_cast(x, index_ty), Term.index_cast(y, index_ty), index_ty),
                eclass_ty(x) == i32_ty, eclass_ty(y) == i32_ty
            )
            # 双边 i32 下推
            yield rewrite(
                Term.index_cast(op(x, y, index_ty), i32_ty)
            ).to(
                op(Term.index_cast(x, i32_ty), Term.index_cast(y, i32_ty), i32_ty),
                eclass_ty(x) == index_ty, eclass_ty(y) == index_ty
            )
            
            # 单边 index_cast 下推
            yield rewrite(
                Term.index_cast(op(x, y, i32_ty), index_ty)
            ).to(
                op(Term.index_cast(x, index_ty), y, index_ty),
                eclass_ty(x) == i32_ty, eclass_ty(y) == index_ty
            )
            # 单边 i32 下推
            yield rewrite(
                Term.index_cast(op(x, y, index_ty), i32_ty)
            ).to(
                op(Term.index_cast(x, i32_ty), y, i32_ty),
                eclass_ty(x) == index_ty, eclass_ty(y) == i32_ty
            )
        
        # index(lit)
        yield rewrite(
            Term.index_cast(Term.lit(LitTerm.int(ai), i32_ty), index_ty)
        ).to(
            Term.lit(LitTerm.int(ai), index_ty)
        )
        yield rewrite(
            Term.index_cast(Term.lit(LitTerm.int(ai), index_ty), i32_ty)
        ).to(
            Term.lit(LitTerm.int(ai), i32_ty)
        )
        
        # index(loop_index)
        yield rewrite(
            Term.index_cast(Term.loop_index(ai, i32_ty), index_ty)
        ).to(
            Term.loop_index(ai, index_ty)
        )
        yield rewrite(
            Term.index_cast(Term.loop_index(ai, index_ty), i32_ty)
        ).to(
            Term.loop_index(ai, i32_ty)
        )
        
        # index(index_cast(x)) -> x
        yield rewrite(
            Term.index_cast(Term.index_cast(x, index_ty), i32_ty)
        ).to(x,
            eclass_ty(x) == i32_ty
        )
        
        yield rewrite(
            Term.index_cast(Term.index_cast(x, i32_ty), index_ty)
        ).to(x,
            eclass_ty(x) == index_ty
        )

    return basic_math_laws_min


# type rules
def type_annotation_ruleset() -> egglog.Ruleset:
        """Create type annotation rules for e-graph terms."""
        # Add `eclass_ty` type annotation
        e, x, y, z = egglog.vars_("e x y z", Term)
        t = egglog.var("t", egglog.String)  # type
        i = egglog.var("i", egglog.i64)
        l = egglog.var("l", LitTerm)
        stmts = egglog.var("stmts", egglog.Vec[Term])

        # Register type annotation rules
        type_rules = [
            (Term.lit(l, t), []),
            (Term.arg(i, t), []),
            (Term.loop_index(i, t), []),
            (Term.loop_iter_arg(i, i, t), []),
            (Term.block_arg(i, i, t), []),
        ]

        # Binary arithmetic operators (MLIR arith dialect)
        binary_ops = [
            "add",    # add (handles both addi and addf)
            "sub",    # subtract (handles both subi and subf)
            "mul",    # multiply (handles both muli and mulf)
            "div",    # divide (handles both divi and divf)
            "rem",    # remainder
            "shl",    # shift left
            "shrsi",  # shift right signed
            "shrui",  # shift right unsigned
            # Note: cmp is handled separately because it has 4 arguments (includes predicate)
        ]

        # Unary arithmetic operators (MLIR arith dialect)
        unary_ops = [
            "neg",    # negate
        ]

        # Cast operators (MLIR arith dialect)
        cast_ops = [
            "index_cast",  # index to/from integer
            "sitofp",      # signed int to float
            "uitofp",      # unsigned int to float
            "fptosi",      # float to signed int
            "fptoui",      # float to unsigned int
            "extsi",       # sign extend
            "extui",       # zero extend
            "trunci",      # truncate integer
            "bitcast",     # bitcast
        ]

        for op in binary_ops:
            type_rules.append((getattr(Term, op)(x, y, t), []))

        for op in unary_ops:
            type_rules.append((getattr(Term, op)(x, t), []))

        for op in cast_ops:
            type_rules.append((getattr(Term, op)(x, t), []))

        # Comparison operation with predicate
        pred = egglog.var("pred", egglog.String)
        type_rules.append((Term.cmp(x, y, pred, t), []))

        # Create type annotation rules similar to original func_to_term.py
        rules = [egglog.rule(egglog.eq(e).to(term_pattern)).then(
            egglog.set_(eclass_ty(e)).to(t)) for term_pattern, _ in type_rules]

        return egglog.ruleset(*rules, name='type_annotation')

# ====================================================================
# Constant Folding Rules
# ====================================================================
def constant_folding_laws() -> egglog.Ruleset:
    """
    修正后的常量折叠 ruleset。
    要点：在模式中把字面值内部的 i64/f64 绑定成变量 (ai, bi, af, bf)，
    不要对 LitTerm 本身尝试访问 .value。
    """
    i32_ty = egglog.String("i32")
    f64_ty = egglog.String("f64")

    # 这里绑定的是原子字面值（i64 / f64），不是整个 LitTerm
    ai, bi = vars_("ai bi", egglog.i64)
    af, bf = vars_("af bf", egglog.f64)

    rules = []

    # ---------- integer (i32) ----------
    rules.append(
        rewrite(
            Term.add(Term.lit(LitTerm.int(ai), i32_ty), Term.lit(LitTerm.int(bi), i32_ty), i32_ty)
        ).to(
            Term.lit(LitTerm.int(ai + bi), i32_ty)
        )
    )

    rules.append(
        rewrite(
            Term.sub(Term.lit(LitTerm.int(ai), i32_ty), Term.lit(LitTerm.int(bi), i32_ty), i32_ty)
        ).to(
            Term.lit(LitTerm.int(ai - bi), i32_ty)
        )
    )

    rules.append(
        rewrite(
            Term.mul(Term.lit(LitTerm.int(ai), i32_ty), Term.lit(LitTerm.int(bi), i32_ty), i32_ty)
        ).to(
            Term.lit(LitTerm.int(ai * bi), i32_ty)
        )
    )

    rules.append(
        rewrite(
            Term.neg(Term.lit(LitTerm.int(ai), i32_ty), i32_ty)
        ).to(
            Term.sub(
                Term.lit(LitTerm.int(egglog.i64(0)), i32_ty),
                Term.lit(LitTerm.int(ai), i32_ty),
                i32_ty
            )
        )
    )

    # ---------- float (f64) ----------
    rules.append(
        rewrite(
            Term.add(Term.lit(LitTerm.float(af), f64_ty), Term.lit(LitTerm.float(bf), f64_ty), f64_ty)
        ).to(
            Term.lit(LitTerm.float(af + bf), f64_ty)
        )
    )

    rules.append(
        rewrite(
            Term.mul(Term.lit(LitTerm.float(af), f64_ty), Term.lit(LitTerm.float(bf), f64_ty), f64_ty)
        ).to(
            Term.lit(LitTerm.float(af * bf), f64_ty)
        )
    )

    rules.append(
        rewrite(
            Term.neg(Term.lit(LitTerm.float(af), f64_ty), f64_ty)
        ).to(
            Term.sub(
                Term.lit(LitTerm.float(egglog.f64(0.0)), f64_ty),
                Term.lit(LitTerm.float(af), f64_ty),
                f64_ty
            )
        )
    )


    return egglog.ruleset(*rules, name="constant_folding_laws")


# ====================================================================
# Registration Function
# ====================================================================
def register_internal_rewrites(egraph: EGraph, include_constant_folding: bool = False) -> int:
    """
    Register all internal mathematical rewrite rules with the given e-graph.

    Args:
        egraph: The e-graph to register rules with
        include_constant_folding: Whether to include constant folding rules
    """
    rule_cnt = 0

    # 1) Register basic mathematical laws
    math_laws = basic_math_laws()
    egraph.run(math_laws.saturate())
    rule_cnt += len(math_laws.__egg_ruleset__.rules)

    # 2) Optionally register constant folding rules
    if include_constant_folding:
        constant_rules = constant_folding_laws()
        rule_cnt += len(constant_rules.__egg_ruleset__.rules)
        egraph.run(constant_rules.saturate())
        logger.info("Registered basic mathematical laws + constant folding rules")
    else:
        logger.info("Registered basic mathematical laws")

    return rule_cnt


# Export main functions
__all__ = [
    'basic_math_laws',
    'constant_folding_laws',
    'type_annotation_ruleset',
    'register_internal_rewrites',
]


def test_basic_laws():
    """
    Test function to demonstrate basic mathematical laws functionality.

    This creates expressions and tests algebraic simplifications.
    """
    import os
    from megg.utils import get_temp_dir

    # Get temp dir (uses MEGG_TEMP_DIR env var or /tmp/megg)
    tmp_dir = str(get_temp_dir())

    # Set TMPDIR to our temp folder
    os.environ["TMPDIR"] = tmp_dir

    egraph = EGraph(save_egglog_string=True)

    # Register the basic mathematical laws
    register_internal_rewrites(egraph, include_constant_folding=False)

    # Test basic algebraic law: x + (-x) = 0
    x = egraph.let("x", Term.arg(egglog.i64(0), egglog.String("i32")))
    neg_x = egraph.let("neg_x", Term.neg(x, egglog.String("i32")))
    x_plus_neg_x = egraph.let("x_plus_neg_x", Term.add(x, neg_x, egglog.String("i32")))

    # Create expected result: 0
    zero = egraph.let("zero", Term.lit(LitTerm.int(egglog.i64(0)), egglog.String("i32")))

    # Run the mathematical laws again to apply them to the new terms
    math_laws = basic_math_laws()
    egraph.run(math_laws.saturate())

    # Check if x + (-x) equals 0
    result = egraph.check_bool(egglog.eq(x_plus_neg_x).to(zero))

    logger.info(f"Basic law test: x + (-x) = 0? {result}")
    logger.info(f"E-graph has {len([line for line in egraph.as_egglog_string.split('\\n') if line.strip()])} lines")

    return result


if __name__ == "__main__":
    # Run the test when file is executed directly
    test_basic_laws()
