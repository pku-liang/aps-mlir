"""自定义指令匹配的 rewrite 规则生成

核心思路：
1. 复用 FuncToTerms 的 _operation_to_term 和 _block_to_term 逻辑
2. 从 block 获取 Vec[serialized_term]（控制流 + 副作用操作）
3. 对每个 serialized term：
   - 控制流（For/If/While）→ 保存进 skeleton，递归处理
   - 普通操作（Yield/Return）→ 转换成 pattern tree，生成 rewrite
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import re
import egglog

from megg.egraph.term import Term, LitTerm
from megg.egraph.datatype import DataType
from megg.egraph.func_to_terms import (
    FuncToTerms,
    mlir_type_to_egraph_ty_string,
    mlir_type_to_megg_type,
)
from megg.utils.mlir_utils import MModule, MOperation, MValue, MBlock
import logging

logger = logging.getLogger(__name__)


def normalize_pattern_module(pattern_module: MModule, verbose: bool = False) -> MModule:
    """
    Normalize pattern module by running it through Megg optimization pipeline.

    This applies internal rewrites (algebraic laws, constant folding) to the pattern
    functions, extracts the optimized representation, and reconstructs as MLIR.

    The goal is to produce a canonical pattern representation with:
    - Redundant operations removed (e.g., unnecessary index_cast)
    - Constants folded
    - Expressions simplified

    Args:
        pattern_module: Input MLIR module containing pattern functions
        verbose: Enable verbose logging

    Returns:
        Normalized MLIR module
    """
    if verbose:
        logger.info("=== Starting Pattern Normalization ===")
        logger.info(f"Input pattern module:\n{pattern_module}")

    try:
        from megg.egraph.func_to_terms import FuncToTerms
        from megg.egraph.megg_egraph import MeggEGraph
        from megg.egraph.extract import Extractor, AstSize
        from megg.egraph.terms_to_func import ExprTreeToMLIR
        from megg.rewrites.internal_rewrites import basic_math_laws, constant_folding_laws, type_annotation_ruleset
        import egglog

        # Create output module
        normalized_module = MModule("module {}")

        # Process each function in the pattern module
        for func_op in pattern_module.get_functions():
            func_name = func_op.symbol_name

            if verbose:
                logger.info(f"\n--- Normalizing pattern function: {func_name} ---")

            # Step 1: Parse to e-graph
            if verbose:
                logger.info("  Step 1: Parsing to e-graph...")
            egraph = egglog.EGraph()
            transformer = FuncToTerms.transform(func_op, egraph)

            if verbose:
                logger.info(f"    ✓ Parsed to e-graph with {len(transformer.ssa_to_term)} SSA values")

            # Step 2: Apply internal rewrites
            if verbose:
                logger.info("  Step 2: Applying internal rewrites...")

            math_laws = basic_math_laws()
            const_laws = constant_folding_laws()
            type_rules = type_annotation_ruleset()

            # Run multiple rounds to allow full simplification
            for round_idx in range(5):
                egraph.run(math_laws.saturate())
                egraph.run(const_laws.saturate())
                egraph.run(type_rules.saturate())

            if verbose:
                logger.info("    ✓ Applied internal rewrites (5 rounds)")

            # Step 3: Extract optimized representation
            if verbose:
                logger.info("  Step 3: Extracting optimized representation...")

            megg_egraph = MeggEGraph.from_egraph(egraph, func_transformer=transformer)
            extractor = Extractor(megg_egraph, AstSize())

            # Extract from all root eclasses
            output_exprs = []
            for i, eclass_id in enumerate(megg_egraph.root_eclasses):
                result = extractor.find_best(eclass_id)
                output_exprs.append(result.expr)
                if verbose:
                    logger.info(f"    ✓ Extracted output {i} with cost {result.cost}")

            # Extract from top block if available
            body_exprs = list(output_exprs)
            top_block_term = getattr(transformer, 'top_block', None)
            if top_block_term is not None:
                try:
                    top_block_id = transformer.get_id_of_term(top_block_term)
                    if top_block_id is not None:
                        term_to_eclass = getattr(megg_egraph, 'term_id_to_eclass', {})
                        top_block_eclass = term_to_eclass.get(top_block_id)
                        if top_block_eclass:
                            top_block_result = extractor.find_best(top_block_eclass)
                            body_exprs = [top_block_result.expr]
                            if verbose:
                                logger.info(f"    ✓ Extracted top block with cost {top_block_result.cost}")
                except Exception as e:
                    if verbose:
                        logger.debug(f"    Could not extract top block: {e}")

            # Step 4: Reconstruct MLIR
            if verbose:
                logger.info("  Step 4: Reconstructing MLIR...")

            if body_exprs:
                normalized_func = ExprTreeToMLIR.reconstruct(
                    original_func=func_op,
                    body_exprs=body_exprs,
                    output_terms=output_exprs or body_exprs,
                    target_module=normalized_module
                )

                if normalized_func:
                    normalized_module.append_to_module(normalized_func)
                    if verbose:
                        logger.info(f"    ✓ Reconstructed function {func_name}")
                else:
                    # Fallback to original
                    normalized_module.append_to_module(func_op)
                    if verbose:
                        logger.warning(f"    ⚠ Reconstruction failed, using original")
            else:
                # Fallback to original
                normalized_module.append_to_module(func_op)
                if verbose:
                    logger.warning(f"    ⚠ No expressions extracted, using original")

        if verbose:
            logger.info("\n=== Pattern Normalization Complete ===")
            logger.info(f"Normalized pattern module:\n{normalized_module}")

        return normalized_module

    except Exception as e:
        logger.warning(f"Pattern normalization failed: {e}, using original module")
        if verbose:
            import traceback
            logger.debug(f"Traceback:\n{traceback.format_exc()}")
        return pattern_module


def _extract_constant_value(mlir_value: MValue) -> Optional[int]:
    """从 MLIR value 中提取常量整数值

    Args:
        mlir_value: MLIR SSA value

    Returns:
        常量整数值，如果不是常量则返回 None
    """
    if mlir_value is None:
        return None

    # 获取定义该 value 的 operation
    defining_op = mlir_value.get_defining_op()
    if defining_op is None:
        return None

    # 检查是否是 arith.constant
    op_name = defining_op.name if hasattr(defining_op, 'name') else str(defining_op.type)
    if 'constant' not in op_name.lower():
        return None

    # 方法 1: 尝试从 operation 字符串中解析
    try:
        op_str = str(defining_op)
        # 例如: "arith.constant 4 : index" → 提取 "4"
        # 或: "%c4 = arith.constant 4 : index" → 提取 "4"
        match = re.search(r'constant\s+(-?\d+)\s*:', op_str)
        if match:
            value = int(match.group(1))
            logger.debug(f"Extracted constant value {value} from {op_str}")
            return value
    except Exception as e:
        logger.debug(f"Failed to extract constant from string: {e}")

    # 方法 2: 尝试从 attributes 获取
    try:
        if hasattr(defining_op, 'attributes'):
            for attr_name in ['value', 'constant_value']:
                attr = defining_op.get_attribute(attr_name)
                if attr is not None:
                    # 尝试转换为整数
                    if hasattr(attr, 'value'):
                        return int(attr.value)
                    else:
                        return int(attr)
    except Exception as e:
        logger.debug(f"Failed to extract constant from attributes: {e}")

    return None


def _get_var_name(var) -> str:
    """Return a consistent string identifier for egglog variables."""
    name = getattr(var, "name", None)
    if isinstance(name, str):
        return name

    text = str(var)
    if text.startswith('<Var ') and text.endswith('>'):
        return text[5:-1].strip()
    return text


def simplify_pattern_str(pattern_str: str) -> str:
    """Simplify a pattern string for display.

    Converts verbose Term representation to a simplified format:
    - Term.yield_(Vec[Term](Term.add(...))) -> yield(add(...))
    - Term.store(x, y, ...) -> store(x, y, ...)
    - Removes egglog type annotations like String("..."), i64(...)
    """
    import re

    s = pattern_str

    # Remove Term. prefix
    s = re.sub(r'\bTerm\.', '', s)

    # Remove Vec[Term](...) wrapper, keep contents
    s = re.sub(r'Vec\[Term\]\(([^)]*)\)', r'\1', s)

    # Simplify egglog.String("...") -> "..."
    s = re.sub(r'egglog\.String\("([^"]*)"\)', r'"\1"', s)
    s = re.sub(r'String\("([^"]*)"\)', r'"\1"', s)

    # Simplify egglog.i64(...) -> just the number
    s = re.sub(r'egglog\.i64\((\d+)\)', r'\1', s)
    s = re.sub(r'i64\((\d+)\)', r'\1', s)

    # Remove type annotations like "__expr_index", "__expr_void"
    s = re.sub(r'"__expr_\w+"', '...', s)

    # Simplify variable names: _arg0, _arg1 -> arg0, arg1
    s = re.sub(r'\b_arg(\d+)\b', r'arg\1', s)

    # Remove trailing underscores from variable names
    s = re.sub(r'\b(\w+)_\b', r'\1', s)

    # Clean up extra whitespace
    s = re.sub(r'\s+', ' ', s).strip()

    return s


def _extract_used_args_from_pattern(pattern: Term, arg_vars: List[Term]) -> List[Term]:
    """
    Extract which argument variables are actually used in the pattern.

    This is important for patterns where some function arguments are unused.
    For example, horner3(arg0, arg1) only uses arg0 in the computation.

    Args:
        pattern: The pattern term
        arg_vars: All available argument variables

    Returns:
        List of argument variables that appear in the pattern
    """
    pattern_str = str(pattern)
    used_args = []

    for var in arg_vars:
        var_name = _get_var_name(var)
        # Check if this variable appears in the pattern
        # Use word boundary to avoid partial matches
        if re.search(r'\b' + re.escape(var_name) + r'\b', pattern_str):
            used_args.append(var)

    return used_args


@dataclass
class SkeletonStmt:
    """Block 中的一个 statement（Vec 元素）"""
    name: str
    pattern_term: Optional[Term] = None  # 叶子 pattern
    nested_skeleton: Optional['SkeletonNode'] = None  # 嵌套控制流
    operand_terms: List[Term] = field(default_factory=list)  # 参与匹配的操作数

    def is_leaf(self) -> bool:
        return self.pattern_term is not None

    def is_nested(self) -> bool:
        return self.nested_skeleton is not None


@dataclass
class SkeletonBlock:
    """一个 block（包含 statements）"""
    name: str
    statements: List[SkeletonStmt] = field(default_factory=list)


@dataclass
class SkeletonNode:
    """控制流节点（包含一个或多个 blocks）

    重要：控制流参数（condition/init_values）需要被验证以确保pattern匹配正确！

    例如：
    - scf.for: Term.for_with_carry(start, end, step, idx, init_vals, body_block, ty)
      - init_vals 需要验证 (存储为 condition_term)
      - body_block 的 Vec 结构存储在 blocks 中

    - scf.if: Term.if_(cond, then_block, else_block, ty)
      - cond 需要验证 (存储为 condition_term) - 包含关键的predicate信息！
      - then_block 和 else_block 的 Vec 结构存储在 blocks 中

    - scf.while: Term.while_(init_vals, cond, body, ty)
      - init_vals 和 cond 都需要验证
    """
    container_type: str  # "func.body", "scf.for", "scf.if", "scf.while"
    blocks: List[SkeletonBlock] = field(default_factory=list)
    result_type: Optional[str] = None
    # 控制流参数 (需要验证的部分)
    condition_term: Optional[Term] = None  # scf.if的condition, scf.while的condition
    init_values_term: Optional[Term] = None  # scf.for的init_vals, scf.while的init_vals
    # Bug Fix #2: 添加循环边界约束
    loop_bounds: Optional[Dict[str, Any]] = None  # scf.for的lower/upper/step (存储常量值)


@dataclass
class Skeleton:
    """完整的控制流骨架"""
    instr_name: str
    root: SkeletonNode
    leaf_patterns: Dict[str, Term] = field(default_factory=dict)
    leaf_operands: Dict[str, List[Term]] = field(default_factory=dict)
    arg_vars: List = field(default_factory=list)  # 函数参数的 generic variables
    result_type: Optional[egglog.String] = None  # 函数返回类型
    arg_var_to_index: Dict[str, int] = field(default_factory=dict)
    arg_types: Dict[str, DataType] = field(default_factory=dict)
    operand_constraints: List[Tuple[str, int, str]] = field(default_factory=list)
    has_side_effects: bool = False
    clobbers: List[str] = field(default_factory=list)

    def format_tree(self) -> str:
        """Format skeleton as a tree structure like for(for(stmt0), for(stmt1, stmt2))"""
        return self._format_node(self.root)

    def _format_node(self, node: SkeletonNode) -> str:
        """Recursively format a skeleton node."""
        # Get short name for container type
        type_name = node.container_type.replace("scf.", "").replace("func.", "")

        children = []
        for block in node.blocks:
            for stmt in block.statements:
                if stmt.is_nested():
                    # Recursively format nested control flow
                    children.append(self._format_node(stmt.nested_skeleton))
                else:
                    # Leaf pattern - just show the name
                    children.append(stmt.name)

        if children:
            return f"{type_name}({', '.join(children)})"
        else:
            return type_name

    def add_leaf_pattern(self, name: str, pattern: Term, operands: Optional[List[Term]] = None):
        full_name = f"{self.instr_name}_{name}"
        self.leaf_patterns[full_name] = pattern
        operands = operands or []

        arg_var_set = set(self.arg_vars) if self.arg_vars else set()

        filtered_operands: List[Term] = [op for op in operands if op in arg_var_set]

        if not filtered_operands and self.arg_vars:
            # Extract operands from pattern structure instead of regex matching
            # This preserves the correct order of arguments
            filtered_operands = self._extract_arg_vars_from_pattern(pattern)

        self.leaf_operands[full_name] = filtered_operands

    def _extract_arg_vars_from_pattern(self, pattern: Term) -> List[Term]:
        """
        Extract argument variables from pattern Term in the correct order.

        Parses the pattern string representation and extracts variables
        in the order they appear in the pattern.
        """
        result = []
        pattern_str = str(pattern)

        # Create a mapping from variable names to variable objects
        var_name_to_var: Dict[str, Term] = {}
        for var in self.arg_vars:
            var_name = _get_var_name(var)
            var_name_to_var[var_name] = var

        # Find all variable occurrences in order using regex
        # Match variable names as whole words
        var_pattern = r'\b(' + '|'.join(re.escape(_get_var_name(v)) for v in self.arg_vars) + r')\b'
        matches = re.finditer(var_pattern, pattern_str)

        seen = set()
        for match in matches:
            var_name = match.group(1)
            if var_name not in seen:
                seen.add(var_name)
                result.append(var_name_to_var[var_name])

        return result

    def add_operand_constraint(self, component_name: str, operand_idx: int, var_name: str):
        constraint = (component_name, operand_idx, var_name)
        if constraint not in self.operand_constraints:
            self.operand_constraints.append(constraint)


def _populate_operand_constraints(skeleton: Skeleton) -> None:
    """Populate argument index mapping and operand equality constraints."""
    if skeleton.arg_vars:
        skeleton.arg_var_to_index = {
            _get_var_name(var): idx for idx, var in enumerate(skeleton.arg_vars)
        }
    else:
        skeleton.arg_var_to_index = {}

    for component_name, operands in skeleton.leaf_operands.items():
        for idx, term in enumerate(operands):
            var_name = _get_var_name(term)
            skeleton.add_operand_constraint(component_name, idx, var_name)


def _specialize_leaf_pattern(
    skeleton: Skeleton,
    component_name: str,
    pattern: Term,
    operand_terms: List[Term],
) -> Tuple[Term, List[Term]]:
    """Make component rewrites more specific when the pattern is ambiguous."""

    # Default: return original pattern/operands
    if len(operand_terms) != 1:
        return pattern, operand_terms

    operand = operand_terms[0]
    var_name = _get_var_name(operand)
    if not var_name:
        return pattern, operand_terms

    arg_index = skeleton.arg_var_to_index.get(var_name)
    if arg_index is None:
        return pattern, operand_terms

    dtype = skeleton.arg_types.get(var_name)
    if dtype is None:
        return pattern, operand_terms

    # Normalize pattern string (egglog prints variables with leading underscore).
    pattern_str = str(pattern).replace(" ", "")
    candidates = {
        f'Term.yield_(Vec[Term]({var_name}),"void")',
        f'Term.yield_(Vec[Term](_{var_name}),"void")',
    }
    if pattern_str not in candidates:
        return pattern, operand_terms

    # Specialize to a concrete argument term so each component rewrite is unique.
    arg_term = Term.arg(egglog.i64(arg_index), egglog.String(str(dtype)))
    specialized_pattern = Term.yield_(egglog.Vec[Term](arg_term), egglog.String("void"))

    return specialized_pattern, [arg_term]


def _instruction_name(func_op: MOperation) -> str:
    """提取函数名"""
    return func_op.symbol_name

def _process_block_statements(
    block: MBlock,
    block_name: str,
    helper: FuncToTerms,
    ssa_to_term: Dict[MValue, Term],
    instr_name: str,
    stmt_counter: int
) -> Tuple[SkeletonBlock, int]:
    """处理 block 中的 statements（Vec 元素）

    策略：
    1. 遍历 block 中的所有 operations
    2. 识别 Vec 元素（控制流 + 副作用操作）
    3. 对每个 Vec 元素：
       - 控制流 → 递归处理，生成 nested skeleton
       - 副作用操作（yield/store）→ 用 _operation_to_term 转换生成 pattern tree
    """
    skeleton_block = SkeletonBlock(name=block_name)
    print(f"\n[DEBUG] >>> Enter block '{block_name}'")

    # 局部 SSA 映射副本，避免覆盖上层
    local_ssa_to_term = dict(ssa_to_term)
    block_args = block.arguments
    print(f"block_args: {[arg._value for arg in block_args]}")
    unmapped_args = [arg for arg in block_args if arg not in local_ssa_to_term]

    # 映射 block 参数（for/while 的迭代变量等）
    print(f"unmapped_args: {unmapped_args}")
    if len(unmapped_args) > 0:
        raw_block = getattr(block, '_block', None)
        block_id = id(raw_block) if raw_block is not None else id(block)
        block_tag = f"{block_name}_{block_id}"
        for i, arg in enumerate(unmapped_args):
            if i == 0:
                var_name = f"{block_tag}_loop_idx"
            else:
                var_name = f"{block_tag}_acc{i-1}"
            local_ssa_to_term[arg] = egglog.var(var_name, Term)

    # ==== 第一轮：建立 SSA 映射 ====
    operations = block.get_operations()
    print(f"[DEBUG] Block {block_name} has {len(operations)} ops")

    for op in operations:
        # 控制流操作先占位（递归处理在第二轮）
        if op.name in ['scf.for', 'scf.if', 'scf.while']:
            for i, res in enumerate(op.results):
                cf_var = egglog.var(f"cf_{len(local_ssa_to_term)}_{i}", Term)
                local_ssa_to_term[res] = cf_var
            print(f"  [DEBUG] control-flow placeholder for {op.name}")
            continue

        operand_terms = [local_ssa_to_term[o] for o in op.operands if o in local_ssa_to_term]
        try:
            result_term = helper._operation_to_term(op, operand_terms)
            if result_term is not None:
                if op.results:
                    local_ssa_to_term[op.results[0]] = result_term
                else:
                    # 对无返回值操作也保存（store/yield）
                    local_ssa_to_term[op] = result_term
            print(f"  [DEBUG] op={op.name} mapped -> {result_term}")
        except Exception as e:
            print(f"  [WARN] _operation_to_term failed for {op.name}: {e}")

    # ==== 第二轮：提取 Vec 元素 ====
    for op in operations:
        # 控制流操作：递归生成 nested skeleton
        if op.name in ['scf.for', 'scf.if', 'scf.while']:
            stmt_name = f"{block_name}_stmt{stmt_counter}"
            stmt_counter += 1
            nested_node, stmt_counter = _process_control_flow(
                op,
                op.name,
                helper,
                local_ssa_to_term,
                instr_name,
                stmt_counter
            )
            stmt = SkeletonStmt(name=stmt_name, nested_skeleton=nested_node)
            skeleton_block.statements.append(stmt)
            print(f"    Statement {stmt_name}: {op.name} (nested)")
            continue

        # 仅将副作用操作视为 Vec 元素（store/yield）
        is_vec_element = op.name in ['scf.yield', 'memref.store', 'memref.alloc', 'memref.alloca']
        if not is_vec_element:
            continue

        stmt_name = f"{block_name}_stmt{stmt_counter}"
        stmt_counter += 1
        pattern_term = None

        # ==== 获取 pattern term ====
        if op.results:
            pattern_term = local_ssa_to_term.get(op.results[0])
        elif op in local_ssa_to_term:
            pattern_term = local_ssa_to_term[op]
        
        print(f" pattern_term for {stmt_name}: {pattern_term}")

        # ==== 收集操作数 ====
        stmt_operands: List[Term] = []
        for operand in op.operands:
            if operand in local_ssa_to_term:
                stmt_operands.append(local_ssa_to_term[operand])
            else:
                print(f"    [WARN] operand {operand} not mapped in {stmt_name}")

        # 跳过占位 yield（cf_）
        if op.name == 'scf.yield':
            if len(op.operands)==0:
                print(f"    [DEBUG] Skip placeholder yield {stmt_name}: {op}")
                continue

        stmt = SkeletonStmt(
            name=stmt_name,
            pattern_term=pattern_term,
            operand_terms=stmt_operands
        )
        skeleton_block.statements.append(stmt)
        print(f"    Statement {stmt_name}: {op.name} -> {pattern_term}")

    print(f"[DEBUG] <<< Exit block '{block_name}' with {len(skeleton_block.statements)} statements\n")
    return skeleton_block, stmt_counter



def _process_control_flow(
    control_op: MOperation,
    control_type: str,
    helper: FuncToTerms,
    ssa_to_term: Dict[MValue, Term],
    instr_name: str,
    stmt_counter: int
) -> Tuple[SkeletonNode, int]:
    """处理控制流操作，生成 nested skeleton node

    Args:
        control_op: 控制流 operation（scf.for/if/while）
        control_type: 控制流类型
        helper: FuncToTerms helper
        ssa_to_term: SSA 到 Term 的映射
        instr_name: 指令名
        stmt_counter: Statement 计数器

    Returns:
        (skeleton_node, updated_stmt_counter)
    """
    regions = control_op.get_regions()

    if control_type == "scf.for":
        # scf.for: 1 个 region (body)
        node = SkeletonNode(container_type="scf.for")

        # Bug Fix: 提取循环边界常量
        # scf.for 的 operands: [lower_bound, upper_bound, step, init_values...]
        if control_op.operands and len(control_op.operands) >= 3:
            lower_value = control_op.operands[0]
            upper_value = control_op.operands[1]
            step_value = control_op.operands[2]

            lower_bound = _extract_constant_value(lower_value)
            upper_bound = _extract_constant_value(upper_value)
            step = _extract_constant_value(step_value)

            if lower_bound is not None and upper_bound is not None and step is not None:
                node.loop_bounds = {
                    'lower': lower_bound,
                    'upper': upper_bound,
                    'step': step
                }
                logger.info(f"  Extracted loop bounds: lower={lower_bound}, upper={upper_bound}, step={step}")
            else:
                logger.debug(f"  Could not extract all loop bounds (lower={lower_bound}, upper={upper_bound}, step={step})")

        body_mlir_block = regions[0].get_blocks()[0]
        # Use canonical block name expected by the skeleton matcher.
        body_block, stmt_counter = _process_block_statements(
            body_mlir_block,
            "body",
            helper,
            ssa_to_term,
            instr_name,
            stmt_counter
        )
        node.blocks.append(body_block)

    elif control_type == "scf.if":
        # scf.if: 2 个 regions (then, else)
        node = SkeletonNode(container_type="scf.if")

        # 提取 condition (scf.if的第一个operand)
        if control_op.operands and len(control_op.operands) > 0:
            condition_value = control_op.operands[0]
            condition_term = ssa_to_term.get(condition_value)
            if condition_term is not None:
                node.condition_term = condition_term
                print(f"  Extracted scf.if condition: {condition_term}")

        then_mlir_block = regions[0].get_blocks()[0]
        then_block, stmt_counter = _process_block_statements(
            then_mlir_block,
            "then",
            helper,
            ssa_to_term,
            instr_name,
            stmt_counter
        )
        node.blocks.append(then_block)

        if len(regions) > 1:
            else_mlir_block = regions[1].get_blocks()[0]
            else_block, stmt_counter = _process_block_statements(
                else_mlir_block,
                "else",
                helper,
                ssa_to_term,
                instr_name,
                stmt_counter
            )
            node.blocks.append(else_block)

    elif control_type == "scf.while":
        # scf.while: 2 个 regions (before, after)
        node = SkeletonNode(container_type="scf.while")

        before_mlir_block = regions[0].get_blocks()[0]
        before_block, stmt_counter = _process_block_statements(
            before_mlir_block,
            "before",
            helper,
            ssa_to_term,
            instr_name,
            stmt_counter
        )
        node.blocks.append(before_block)

        if len(regions) > 1:
            after_mlir_block = regions[1].get_blocks()[0]
            after_block, stmt_counter = _process_block_statements(
                after_mlir_block,
                "after",
                helper,
                ssa_to_term,
                instr_name,
                stmt_counter
            )
            node.blocks.append(after_block)

    else:
        raise ValueError(f"Unknown control flow type: {control_type}")

    return node, stmt_counter


def _build_skeleton_from_func(func_op: MOperation) -> Tuple[Optional[Skeleton], Optional[Tuple[Term, egglog.String, List]]]:
    """从函数构建 skeleton 或 simple pattern

    策略：
    1. 创建 FuncToTerms helper（复用 _operation_to_term）
    2. 创建 generic variables for arguments
    3. 从 entry block 开始处理 statements
    4. 检测是简单计算模式还是复杂控制流

    Returns:
        (skeleton, simple_pattern_with_type_and_args)
        - 如果是简单模式：(None, (pattern_term, result_type, arg_vars))
        - 如果是复杂模式：(skeleton, None)
    """
    instr_name = _instruction_name(func_op)
    print(f"Extracting skeleton for {instr_name}")

    # Step 1: 创建 helper
    egraph = egglog.EGraph()
    helper = FuncToTerms(
        func=func_op,
        egraph=egraph,
        next_id=0,
        ssa_to_id={},
        ssa_to_term={},
        loop_to_term={},
        top_block=None
    )

    # Step 2: 创建 generic variables for arguments
    entry_block = func_op.get_regions()[0].get_blocks()[0]
    num_args = len(entry_block.arguments)
    arg_vars: List[Term] = []
    arg_types_map: Dict[str, DataType] = {}

    if num_args > 0:
        arg_names = ' '.join(f"arg{i}" for i in range(num_args))
        arg_vars = list(egglog.vars_(arg_names, Term))

    ssa_to_term: Dict[MValue, Term] = {}
    for i, arg in enumerate(entry_block.arguments):
        if i >= len(arg_vars):
            var = egglog.var(f"arg{i}", Term)
            arg_vars.append(var)
        else:
            var = arg_vars[i]

        ssa_to_term[arg] = var

        var_name = _get_var_name(var)
        try:
            arg_type = mlir_type_to_megg_type(arg.type)
        except Exception:
            arg_type = None
        if arg_type is not None:
            arg_types_map[var_name] = arg_type

    # Step 3: 检测是否为简单计算模式
    # 简单模式：无控制流、无副作用（除了 func.return）
    operations = entry_block.get_operations()

    has_control_flow = False
    has_side_effects = False
    return_op = None

    for op in operations:
        if op.name in ['scf.for', 'scf.if', 'scf.while']:
            has_control_flow = True
            # FIXME: 暂时不考虑memref.load作为副作用
        elif op.name in ['scf.yield', 'memref.store', 'scf.cond', 'memref.alloc', 'memref.alloca']:
            has_side_effects = True
        elif op.name == 'func.return':
            return_op = op

    # 情况1：简单计算模式（无控制流、无副作用）
    if not has_control_flow and not has_side_effects and return_op is not None:
        print(f"  Detected simple computation pattern (no control flow/side effects)")

        # 构建 SSA 映射
        local_ssa_to_term = dict(ssa_to_term)

        for op in operations:
            if op.name == 'func.return':
                continue

            # 准备 operands
            operand_terms = []
            for operand in op.operands:
                if operand in local_ssa_to_term:
                    operand_terms.append(local_ssa_to_term[operand])

            # 转换 operation
            try:
                result_term = helper._operation_to_term(op, operand_terms)
                if result_term is not None and op.results:
                    local_ssa_to_term[op.results[0]] = result_term
            except Exception:
                pass

        # 提取 return 的返回值作为 pattern 和 result type
        if return_op.operands:
            return_value = return_op.operands[0]
            if return_value in local_ssa_to_term:
                pattern = local_ssa_to_term[return_value]
                # 提取返回值类型
                result_type = mlir_type_to_egraph_ty_string(return_value.type)
                print(f"  Extracted simple pattern from func.return with type {result_type}")
                print(f"  Pattern uses {len(arg_vars)} argument variables")
                return None, (pattern, result_type, arg_vars)

        print(f"  Warning: Failed to extract pattern from func.return")
        return None, None

    # 情况2：复杂控制流模式
    print(f"  Detected complex control flow pattern")

    # Step 3: 处理 entry block
    # Root 永远是 func.body (对应 MeggEGraph 的 top_block)
    root = SkeletonNode(container_type="func.body")

    body_block, _ = _process_block_statements(
        entry_block,
        "body",
        helper,
        ssa_to_term,
        instr_name,
        0
    )
    root.blocks.append(body_block)

    print(f"  Skeleton root: func.body with {len(body_block.statements)} statements")
    for stmt in body_block.statements:
        if stmt.is_nested():
            print(f"    - {stmt.name}: {stmt.nested_skeleton.container_type} (nested)")
        else:
            print(f"    - {stmt.name}: leaf")

    # Step 4: 优化 skeleton root
    # 如果 func.body 只包含一个 nested control flow，提升为 root
    # 这样可以直接匹配控制流节点，而不需要匹配 func.body wrapper
    if len(body_block.statements) == 1 and body_block.statements[0].is_nested():
        root = body_block.statements[0].nested_skeleton
        print(f"  Promoted skeleton root to: {root.container_type}")

    # Step 4: 提取函数返回类型
    # 从函数签名获取返回类型
    result_type = None
    if return_op and return_op.operands:
        result_type = mlir_type_to_egraph_ty_string(return_op.operands[0].type)

    # Step 5: 构建 skeleton（包含 arg_vars 和 result_type）
    skeleton = Skeleton(
        instr_name=instr_name,
        root=root,
        arg_vars=arg_vars,
        result_type=result_type
    )
    skeleton.arg_types = arg_types_map
    skeleton.has_side_effects = has_side_effects
    skeleton.clobbers = ['memory'] if has_side_effects else []

    # Step 6: 递归提取所有 leaf patterns
    _extract_leaf_patterns_recursive(skeleton.root, skeleton)

    # Step 7: 构建操作数约束，确保匹配时能够验证参数一致性
    _populate_operand_constraints(skeleton)

    print(f"Built skeleton for {instr_name} with {len(skeleton.leaf_patterns)} leaf patterns")
    print(f"  Skeleton has {len(arg_vars)} argument variables and result type {result_type}")

    return skeleton, None


def _extract_leaf_patterns_recursive(node: SkeletonNode, skeleton: Skeleton):
    """递归提取所有 leaf patterns"""
    for block in node.blocks:
        for stmt in block.statements:
            if stmt.is_leaf():
                # 叶子 pattern
                skeleton.add_leaf_pattern(stmt.name, stmt.pattern_term, stmt.operand_terms)
            elif stmt.is_nested():
                # 递归处理嵌套控制流
                _extract_leaf_patterns_recursive(stmt.nested_skeleton, skeleton)


def build_ruleset_from_module(module: MModule, normalize: bool = True, verbose: bool = False) -> Tuple[egglog.Ruleset, List[Skeleton]]:
    """从模块构建 ruleset 和 skeletons

    Args:
        module: Input MLIR module containing pattern functions
        normalize: If True, normalize patterns through Megg optimization first (default: True)
        verbose: Enable verbose logging for normalization

    Returns:
        (ruleset, skeletons)
    """
    # Normalize pattern module through Megg optimization pipeline
    # This applies internal rewrites to produce a canonical pattern representation
    if normalize:
        print("[Pattern Normalization] Normalizing pattern module...")
        module = normalize_pattern_module(module, verbose=verbose)
        print(module)
        print("[Pattern Normalization] ✓ Pattern normalization complete")

    rewrites: List[egglog.Rewrite] = []
    skeletons: List[Skeleton] = []

    for func_op in module.get_functions():
        instr_name = _instruction_name(func_op)

        try:
            # 构建 skeleton 或 simple pattern
            print(f"Processing function: {instr_name}")
            skeleton, simple_pattern_with_type_and_args = _build_skeleton_from_func(func_op)

            if simple_pattern_with_type_and_args is not None:
                # 情况1：简单计算模式 - 生成直接的 rewrite 规则
                # pattern → custom_instr(name, arg_vars, result_type)
                pattern, result_type, arg_vars = simple_pattern_with_type_and_args

                # 只包含实际在pattern中使用的参数
                used_args = _extract_used_args_from_pattern(pattern, arg_vars)

                if len(used_args) < len(arg_vars):
                    unused_count = len(arg_vars) - len(used_args)
                    print(f"  Note: {unused_count} unused argument(s) excluded from pattern (e.g., horner3 only uses arg0, not arg1)")

                operands_vec = egglog.Vec[Term](*used_args) if used_args else egglog.Vec[Term]()
                custom_instr = Term.custom_instr(
                    egglog.String(instr_name),
                    operands_vec,
                    result_type
                )
                rewrite = egglog.rewrite(pattern).to(custom_instr)
                rewrites.append(rewrite)
                print(f"  Added simple pattern rewrite: {instr_name} with type {result_type} and {len(used_args)} operands (out of {len(arg_vars)} function args)")

            elif skeleton is not None:
                # 情况2：复杂控制流模式 - 生成 component rewrites + skeleton
                # Print skeleton tree format
                print(f"  [Skeleton] {skeleton.format_tree()}")

                for full_name, pattern in skeleton.leaf_patterns.items():
                    operand_terms = skeleton.leaf_operands.get(full_name, [])
                    if not operand_terms:
                        print(
                            f"  Skipping component rewrite '{full_name}' due to empty operands"
                        )
                        continue

                    specialized_pattern, specialized_operands = _specialize_leaf_pattern(
                        skeleton,
                        full_name,
                        pattern,
                        operand_terms,
                    )

                    operand_vec = (
                        egglog.Vec[Term](*specialized_operands)
                        if specialized_operands else egglog.Vec[Term]()
                    )
                    comp_instr = Term.component_instr(
                        egglog.String(full_name),
                        operand_vec,
                        egglog.String("void")
                    )
                    # Check pattern complexity before adding
                    pattern_str = str(specialized_pattern)
                    pattern_len = len(pattern_str)

                    # Print simplified rewrite: pattern -> component
                    simplified = simplify_pattern_str(pattern_str)
                    # Extract short component name
                    short_name = full_name.replace(f"{instr_name}_", "")
                    print(f"  [Rewrite] {simplified} -> {short_name}")

                    if pattern_len > 5000:
                        print(f"    ⚠️  WARNING: Very large pattern ({pattern_len} chars), may cause performance issues")

                    rewrite = egglog.rewrite(specialized_pattern).to(comp_instr)
                    rewrites.append(rewrite)

                    # 🧪 MANUAL TEST: 手动添加使用字符串常量的rewrite来测试类型匹配
                    if full_name == "gemm_4x4_body_stmt4":
                        print(f"  🧪 Adding manual test rewrite with string constants (not variables)")
                        # 原始的pattern使用变量:
                        # MemRefStore(_cf_18_0, _arg1, Add(Mul(loop_idx, Lit(4), __expr_index), loop_idx, __expr_index), __expr_void)
                        # 手动版本使用字符串常量:
                        # MemRefStore(_cf_18_0, _arg1, Add(Mul(loop_idx, Lit(4), "index"), loop_idx, "index"), "void")

                        cf_var = egglog.var("_cf_manual", Term)
                        arg1_var = egglog.var("_arg1_manual", Term)
                        loop_idx_var = egglog.var("_loop_idx_manual", Term)
                        
                        all_var_1 = egglog.var("_all_manual", Term)
                        all_var_2 = egglog.var("_all2_manual", Term)

                        manual_pattern = Term.store(
                            cf_var,
                            arg1_var,
                            Term.add(
                                Term.mul(all_var_1, all_var_2, egglog.String("index")),
                                loop_idx_var,
                                egglog.String("index")
                            ),
                            egglog.String("void")
                        )

                        manual_comp_instr = Term.component_instr(
                            egglog.String("gemm_4x4_body_stmt4_MANUAL_TEST"),
                            egglog.Vec[Term](arg1_var),
                            egglog.String("void")
                        )

                        manual_rewrite = egglog.rewrite(manual_pattern).to(manual_comp_instr)
                        rewrites.append(manual_rewrite)
                        print(f"  ✅ Added manual test rewrite: gemm_4x4_body_stmt4_MANUAL_TEST")

                skeletons.append(skeleton)
                print(f"  Added skeleton: {skeleton}")

        except Exception as e:
            import traceback
            print(f"Warning: Failed to process function {instr_name}: {e}")
            traceback.print_exc()
            continue

    # 创建 ruleset
    if rewrites:
        ruleset = egglog.ruleset(*rewrites, name="match_rewrite")
    else:
        ruleset = egglog.ruleset(name="match_rewrite")

    print(f"Total rewrites: {len(rewrites)}")

    return ruleset, skeletons



if __name__ == "__main__":
    a = egglog.var("a", Term)
    (b,) = egglog.vars_("b", Term)

    print(a)  # 看看打印出来是什么
    print(b)
