"""
MeggEGraph: Python-based E-graph structure for greedy extraction.

This module provides a simplified e-graph data structure constructed from
serialized egglog e-graph dictionaries, with greedy extraction capabilities.

For advanced extraction with customizable cost functions, use the
`extract` module which provides:
- CostFunction interface for defining custom costs
- Extractor class with proper greedy algorithm from egg
- Built-in cost functions: AstSize, AstDepth, OpWeightedCost, etc.

Example:
    from megg.egraph.extract import Extractor, AstSize, OpWeightedCost

    egraph = MeggEGraph.from_egraph(egglog_graph)

    # Use built-in cost function
    extractor = Extractor(egraph, AstSize())
    result = extractor.find_best()

    # Or define custom costs per operation
    custom_cost = OpWeightedCost({'Mul': 10.0, 'Add': 1.0})
    extractor = Extractor(egraph, custom_cost)
    result = extractor.find_best()
"""

from __future__ import annotations
from typing import TypedDict, Dict, List, Optional, Any, TYPE_CHECKING
import re
import os
from collections import deque
from dataclasses import dataclass, field
from megg.egraph.datatype import DataType
from megg.egraph.func_to_terms import _parse_type_from_string
import json
import logging

if TYPE_CHECKING:
    from megg.rewrites.match_rewrites import Skeleton, SkeletonNode, SkeletonStmt

logger = logging.getLogger(__name__)

# TypedDict definitions matching egglog serialization format
class ENodeDict(TypedDict):
    """Serialized e-node representation."""
    children: list[str]  # List of eclass IDs
    cost: float          # Cost of this node
    eclass: str          # Parent eclass ID
    op: str              # Operation name (e.g., "Add", "Lit", "Arg")
    subsumed: bool       # Whether this node is subsumed


class EClassDict(TypedDict):
    """Serialized e-class metadata."""
    type: str  # Type annotation


class EGraphDict(TypedDict):
    """Serialized e-graph representation."""
    nodes: dict[str, ENodeDict]       # Map from node ID to node data
    root_eclasses: list[str]          # List of root eclass IDs
    class_data: dict[str, EClassDict] # Map from eclass ID to eclass data


@dataclass
class MeggENode:
    """E-node in MeggEGraph."""
    node_id: str
    op: str
    children: List[str]  # List of eclass IDs
    eclass: str
    node_type: str  # "impl", "arg", "lit", or "unimplemented"
    value: Optional[Any] = None # for lit nodes, we need to store the value


@dataclass
class MeggEClass:
    """E-class in MeggEGraph."""
    eclass_id: str
    nodes: List[MeggENode]
    dtype: Optional[DataType] = None


@dataclass
class ExpressionNode:
    """Expression tree node extracted from e-graph."""
    op: str
    node_id: str
    eclass_id: str
    children: List[ExpressionNode]
    cost: float
    dtype: Optional[DataType] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for MLIR conversion."""
        return {
            'op': self.op,
            'node_id': self.node_id,
            'eclass_id': self.eclass_id,
            'children': [c.to_dict() for c in self.children],
            'cost': self.cost,
            'type': self.dtype,
            'metadata': self.metadata
        }


@dataclass
class MeggEGraph:
    """
    Python-based E-graph for greedy extraction and custom instruction insertion.

    Constructed from serialized egglog e-graph dictionary.
    Can also add new nodes for custom instructions.
    """

    enodes: Dict[str, MeggENode]
    eclasses: Dict[str, MeggEClass]
    root_eclasses: List[str]

    # Mapping from numeric term IDs (id_of) to eclass IDs – used for
    # reconstructing specific pieces such as the function's top-level block.
    term_id_to_eclass: Dict[int, str] = field(default_factory=dict)

    # Control flow node index for fast skeleton matching
    cf_index: Dict[str, List[MeggENode]] = field(default_factory=dict)

    # Control flow operation types to index
    CF_OPS = [
        "Term.for_with_carry",
        "Term.for_",
        "Term.if_",
        "Term.while_",
        "Term.block"
    ]


    def _set_eclass_dtype(self, eclass_name: str, dtype: DataType):
        eclass = self.eclasses[eclass_name]
        if eclass.dtype is not None:
            assert eclass.dtype == dtype
        else:
            eclass.dtype = dtype

    def add_custom_instr_node(self, eclass_id: str, instr_name: str, operands: List[str] = None, result_type: str = "void") -> str:
        """
        在指定的 eclass 中添加一个 custom_instr node

        Args:
            eclass_id: 目标 eclass ID（例如 "Term-17"）
            instr_name: 自定义指令名称（例如 "complex_mul_add"）
            operands: Operand eclasses（例如 ["Term-0", "Term-1"]）
            result_type: 返回类型（例如 "i32"）

        Returns:
            新创建的 node ID
        """
        if eclass_id not in self.eclasses:
            raise ValueError(f"EClass {eclass_id} not found")

        operands = operands or []

        # 创建 Vec 节点用于存储操作数
        vec_node_idx = len(self.enodes)
        while True:
            vec_node_id = f"custom-vec-{vec_node_idx}"
            if vec_node_id not in self.enodes:
                break
            vec_node_idx += 1

        vec_eclass_idx = len(self.eclasses)
        while True:
            vec_eclass_id = f"Vec_Term-custom-{vec_eclass_idx}"
            if vec_eclass_id not in self.eclasses:
                break
            vec_eclass_idx += 1

        vec_node = MeggENode(
            node_id=vec_node_id,
            op="Vec",
            children=list(operands),
            eclass=vec_eclass_id,
            node_type="impl"
        )
        self.enodes[vec_node_id] = vec_node
        self.eclasses[vec_eclass_id] = MeggEClass(
            eclass_id=vec_eclass_id,
            nodes=[vec_node],
            dtype=None
        )

        # 创建新的 node ID
        custom_idx = len(self.enodes)
        while True:
            node_id = f"custom-instr-{instr_name}-{custom_idx}"
            if node_id not in self.enodes:
                break
            custom_idx += 1

        # 解析 result_type 为 DataType
        from megg.egraph.func_to_terms import _parse_type_from_string
        try:
            dtype = _parse_type_from_string(result_type)
        except Exception:
            from megg.egraph.datatype import VoidType
            dtype = VoidType()

        custom_node = MeggENode(
            node_id=node_id,
            op="Term.custom_instr",
            children=[vec_eclass_id],
            eclass=eclass_id,
            node_type="impl",
            value=instr_name
        )

        self.enodes[node_id] = custom_node

        eclass = self.eclasses[eclass_id]
        eclass.nodes.append(custom_node)

        if eclass.dtype is None:
            eclass.dtype = dtype

        logger.info(
            f"Added custom_instr node '{instr_name}' to eclass {eclass_id} with {len(operands)} operands and type {result_type}"
        )
        return node_id

    def _build_control_flow_index(self):
        """
        Build control flow node index for fast skeleton matching.

        This method indexes all control flow nodes (for, if, while, block)
        by their operation type, enabling O(1) lookup instead of O(N) traversal.

        Time complexity: O(N) where N = total nodes (one-time cost)
        Space complexity: O(cf_nodes) where cf_nodes = control flow nodes
        """
        from collections import defaultdict
        self.cf_index = defaultdict(list)

        cf_count = 0
        for enode in self.enodes.values():
            if enode.op in self.CF_OPS:
                self.cf_index[enode.op].append(enode)
                cf_count += 1

        logger.info(f"Built control flow index: {cf_count} nodes across {len(self.cf_index)} types")
        for op_type, nodes in self.cf_index.items():
            logger.debug(f"  {op_type}: {len(nodes)} nodes")

    @classmethod
    def from_dict(cls, egraph_dict: EGraphDict) -> "MeggEGraph":
        enodes: Dict[str, MeggENode] = {}
        eclasses: Dict[str, MeggEClass] = {}
        unimplemented_enodes: Dict[str, List[str]] = {}

        @staticmethod
        def get_boxed_inner(node_name: str, egraph_dict: EGraphDict) -> str:
            name = egraph_dict["nodes"][node_name]["op"].strip('"')
            if name.startswith("Boxed(") and name.endswith(")"):
                name = name[len("Boxed("):-1].strip('"')
            return name

        @staticmethod
        def get_args(node_name: str, egraph_dict: EGraphDict) -> List[str]:
            vec_node = egraph_dict["nodes"][node_name]
            if vec_node["op"] != "Vec":
                return []
            return [egraph_dict["nodes"][c]["eclass"] for c in vec_node["children"]]

        @staticmethod
        def get_node_dtype(node_name: str, node_json: ENodeDict, egraph_dict: EGraphDict) -> Optional[DataType]:
            if not node_json["children"]:
                return None
            last_child = node_json["children"][-1]
            val = get_boxed_inner(last_child, egraph_dict)
            return _parse_type_from_string(val) if val else None
        
        @staticmethod
        def _convert_children_to_eclasses(children: List[str], egraph_dict: EGraphDict) -> List[str]:
            """Convert enode children to their eclass IDs."""
            return [egraph_dict["nodes"][child]["eclass"] for child in children]

        # === Pass 1: create empty e-classes ===
        for class_name, class_json in egraph_dict['class_data'].items():
            class_type = class_json.get('type', 'unknown')
            # Check for Term types (handles both 'Term' and 'megg.egraph.term.Term')
            if not ('Term' in class_type or 'Vec' in class_type):
                continue
            eclasses[class_name] = MeggEClass(eclass_id=class_name, nodes=[], dtype=None)

        # === Pass 2: add all enodes ===
        for node_id, node_data in egraph_dict['nodes'].items():
            eclass = node_data['eclass']
            op = node_data['op']
            children = node_data['children']
            node_type = 'unimplemented'
            value = None
            dtype = None

            if op.startswith("Term."):
                op_stripped = op[len("Term."):]

                if op_stripped in ['arg', 'loop_index']:
                    value = get_boxed_inner(children[0], egraph_dict) if children else None
                    children = []
                    node_type = 'arg'
                elif op_stripped in ['loop_iter_arg', 'block_arg']:
                    _id = get_boxed_inner(children[0], egraph_dict) if children else None
                    idx = get_boxed_inner(children[1], egraph_dict) if len(children) > 1 else None
                    value = (_id, idx)
                    children = []
                    node_type = 'arg'
                elif op_stripped == 'lit':
                    if children:
                        lit_node_id = children[0]
                        lit_children = egraph_dict["nodes"][lit_node_id]["children"]
                        if lit_children:
                            value = get_boxed_inner(lit_children[0], egraph_dict)
                    children = []
                    node_type = 'lit'
                else:
                    children = [egraph_dict["nodes"][c]["eclass"] for c in children]
                    node_type = 'impl'
                    # cmp, 将predicate作为value存储
                    if op_stripped == 'cmp':
                        predcate_node = node_data['children'][2] if len(node_data['children']) > 2 else None
                        predicate = get_boxed_inner(predcate_node, egraph_dict) if predcate_node else None
                        # predicate is like "pred_enum : type", we only want the enum part
                        if predicate and ':' in predicate:
                            predicate = predicate.split(':')[0].strip()
                        value = predicate
                    elif op_stripped == 'while_':
                        # 需要将while两个block的id存进value
                        cond_block_node = node_data['children'][1] if len(node_data['children']) > 0 else None
                        body_block_node = node_data['children'][2] if len(node_data['children']) > 1 else None
                        # 从每个block的第一个节点获取id
                        cond_child = egraph_dict["nodes"][cond_block_node]["children"][0] if cond_block_node and cond_block_node in egraph_dict["nodes"] else None
                        body_child = egraph_dict["nodes"][body_block_node]["children"][0] if body_block_node and body_block_node in egraph_dict["nodes"] else None
                        cond_block_id = get_boxed_inner(cond_child, egraph_dict) if cond_child else None    
                        body_block_id = get_boxed_inner(body_child, egraph_dict) if body_block_node else None
                        value = (cond_block_id, body_block_id)
                        print(f"while_ node {node_id} with value {value}")
                    elif op_stripped == 'affine_for':
                        lower_map_node = node_data['children'][4] if len(node_data['children']) > 4 else None
                        upper_map_node = node_data['children'][5] if len(node_data['children']) > 5 else None
                        lower_map = get_boxed_inner(lower_map_node, egraph_dict) if lower_map_node else None
                        upper_map = get_boxed_inner(upper_map_node, egraph_dict) if upper_map_node else None
                        value = {
                            'lower_map': lower_map,
                            'upper_map': upper_map,
                            'has_iter_args': False,
                        }
                    elif op_stripped == 'affine_for_with_carry':
                        lower_map_node = node_data['children'][4] if len(node_data['children']) > 4 else None
                        upper_map_node = node_data['children'][5] if len(node_data['children']) > 5 else None
                        lower_map = get_boxed_inner(lower_map_node, egraph_dict) if lower_map_node else None
                        upper_map = get_boxed_inner(upper_map_node, egraph_dict) if upper_map_node else None
                        value = {
                            'lower_map': lower_map,
                            'upper_map': upper_map,
                            'has_iter_args': True,
                        }
                    elif op_stripped == 'custom_instr':
                        # Children: name, operands vec, type string
                        name_node = node_data['children'][0] if len(node_data['children']) > 0 else None
                        instr_name = get_boxed_inner(name_node, egraph_dict) if name_node else None
                        value = instr_name
                    elif op_stripped == 'component_instr':
                        # Children: name, operands vec, type string (same as custom_instr)
                        name_node = node_data['children'][0] if len(node_data['children']) > 0 else None
                        comp_name = get_boxed_inner(name_node, egraph_dict) if name_node else None
                        value = comp_name
                    elif op_stripped == 'get_global':
                        # get_global, 将global name作为value存储
                        print(f"Processing get_global node {node_id} with children {children}")
                        # 第一个的子节点是global name
                        global_name_node = node_data['children'][0] if len(node_data['children']) > 0 else None
                        value = get_boxed_inner(global_name_node, egraph_dict) if global_name_node else None
                        print(f"get_global node {node_id} with value {value}")
                
                dtype = get_node_dtype(node_id, node_data, egraph_dict)
                if dtype and eclass in eclasses:
                    eclasses[eclass].dtype = dtype
                        # Vec_Term
            elif op == "Vec":
                # Vector of terms - convert children to eclasses
                node_type = 'impl'
                children=_convert_children_to_eclasses(node_data['children'], egraph_dict)
                
            else:
                if eclass not in unimplemented_enodes:
                    unimplemented_enodes[eclass] = []
                unimplemented_enodes[eclass].append(node_id)
            print(op)
            enodes[node_id] = MeggENode(
                node_id=node_id,
                op=op,
                children=children,
                eclass=eclass,
                node_type=node_type,
                value=value
            )
            if eclass in eclasses and node_type != 'unimplemented':
                eclasses[eclass].nodes.append(enodes[node_id])

        root_eclasses = [eclass_id for eclass_id in egraph_dict.get('root_eclasses', []) ]

        instance = cls(enodes=enodes, eclasses=eclasses, root_eclasses=root_eclasses)
        instance.unimplemented_enodes = unimplemented_enodes

        # Build control flow index for fast skeleton matching
        instance._build_control_flow_index()

        return instance

    @classmethod
    def from_egraph(cls, egraph, n_inline_leaves: int = 0,
                    split_primitive_outputs: bool = False,
                    func_transformer = None) -> MeggEGraph:
        """
        Construct MeggEGraph from egglog EGraph instance.

        Args:
            egraph: egglog.EGraph instance
            n_inline_leaves: Parameter for serialization
            split_primitive_outputs: Parameter for serialization
            func_transformer: Optional FuncToTerms transformer for proper root eclass mapping

        Returns:
            MeggEGraph instance
        """
        # Serialize the e-graph
        serialized = egraph._serialize(
            n_inline_leaves=n_inline_leaves,
            split_primitive_outputs=split_primitive_outputs
        )
        egraph_json_str = serialized.to_json()
        from megg.utils import get_temp_dir
        tmp_dir = get_temp_dir()
        # save the json string to a file for debugging
        with open(tmp_dir / "egraph_debug.json", "w") as f:
            f.write(egraph_json_str)

        egraph_dict: EGraphDict = json.loads(egraph_json_str)

        megg_egraph = cls.from_dict(egraph_dict)

        # dump to json for debugging
        megg_egraph.dump_to_json(str(tmp_dir / "megg_egraph_debug.json"))
        

        # Find root eclasses using the proper method from implegraph.py
        # The correct approach: Use the FuncToTerms transformer to map output_terms to eclasses
        
        root_eclasses = []
        
        # Build id_to_eclass mapping from egraph_dict
        id_to_eclass: Dict[int, str] = {}
        for node_name, node_json in egraph_dict['nodes'].items():
            if node_json['op'] == 'id_of':
                # Extract ID from eclass name (format: "i64-123")
                eclass_str = node_json['eclass']
                if eclass_str.startswith('i64-'):
                    try:
                        numeric_id = int(eclass_str[len('i64-'):])
                        # Get the Term eclass from the child
                        child_node = node_json['children'][0]
                        if child_node in egraph_dict['nodes']:
                            term_eclass = egraph_dict['nodes'][child_node]['eclass']
                            id_to_eclass[numeric_id] = term_eclass
                    except (ValueError, IndexError):
                        pass
        # Expose the id->eclass mapping early so callers can always access it,
        # even if we return before reaching the fallback logic below.
        megg_egraph.term_id_to_eclass = id_to_eclass

        if func_transformer and hasattr(func_transformer, 'top_block'):
            logger.info(f"Using func_transformer.top_block to find root eclass")
            
            try:
                top_block = func_transformer.top_block
                print(f"ssa_to_term:", func_transformer.ssa_to_term)
                # Get the ID of the top_block term
                top_block_id = func_transformer.get_id_of_term(top_block)
                print(f"Top block term ID: {top_block_id}")
                print(f"id_to_eclass mapping: {id_to_eclass}")
                if top_block_id is not None:
                    # Find the eclass for this ID in the serialized graph
                    if top_block_id in id_to_eclass:
                        root_eclass = id_to_eclass[top_block_id]
                        if root_eclass in megg_egraph.eclasses:
                            root_eclasses.append(root_eclass)
                            logger.info(f"Found top_block ID {top_block_id} -> eclass {root_eclass}")
                            megg_egraph.root_eclasses = root_eclasses
                            logger.info(f"Successfully found root eclass via func_transformer.top_block: {root_eclass}")
                            return megg_egraph
                    else:
                        logger.warning(f"top_block ID {top_block_id} not found in id_to_eclass mapping")
                else:
                    logger.warning("Could not get ID for top_block term")
                    
            except Exception as e:
                logger.warning(f"Failed to get eclass from top_block: {e}")

        # Fallback: Find top-level Term.block eclasses (blocks not used as children)
        if not root_eclasses:
            logger.info("Attempting fallback: finding top-level Term.block eclasses")
            print(f"DEBUG: Starting block detection fallback")

            # Find all Term.block eclasses
            # Note: serialized op names include type info, e.g., "Term.block(·, Boxed("i32"))"
            block_eclasses = set()
            node_count = 0
            for node_id, node_data in egraph_dict['nodes'].items():
                node_count += 1
                op = node_data['op']
                # Check if op starts with 'Term.block'
                if isinstance(op, str) and op.startswith('Term.block'):
                    print(f"DEBUG: Found block node: {op}, eclass: {node_data['eclass']}")
                    eclass_id = node_data['eclass']
                    if eclass_id in megg_egraph.eclasses:
                        block_eclasses.add(eclass_id)
                        print(f"DEBUG: Added to block_eclasses: {eclass_id}")
                    else:
                        print(f"DEBUG: Eclass {eclass_id} not in megg_egraph.eclasses")

            print(f"DEBUG: Checked {node_count} nodes, found {len(block_eclasses)} block eclasses: {block_eclasses}")

            # Find blocks used as children (nested blocks)
            used_block_eclasses = set()
            for node_id, node_data in egraph_dict['nodes'].items():
                for child_node_id in node_data['children']:
                    if child_node_id in egraph_dict['nodes']:
                        child_eclass = egraph_dict['nodes'][child_node_id]['eclass']
                        if child_eclass in block_eclasses:
                            used_block_eclasses.add(child_eclass)
                            print(f"DEBUG: Block eclass {child_eclass} is used as child")

            print(f"DEBUG: Found {len(used_block_eclasses)} nested blocks: {used_block_eclasses}")

            # Top-level blocks (not used as children) are function bodies
            top_level_blocks = block_eclasses - used_block_eclasses
            print(f"DEBUG: Top-level blocks: {top_level_blocks}")

            if top_level_blocks:
                root_eclasses.extend(list(top_level_blocks))
                logger.info(f"Found {len(top_level_blocks)} top-level block(s) as roots: {top_level_blocks}")

        if root_eclasses:
            megg_egraph.root_eclasses = root_eclasses
            logger.info(f"Successfully found {len(root_eclasses)} root eclasses: {root_eclasses}")
        else:
            logger.warning("No root eclasses found from serialized graph")


        # Build control flow index for fast skeleton matching
        megg_egraph._build_control_flow_index()

        return megg_egraph

    def extract_greedy(self, eclass_id: Optional[str] = None, cost_function=None) -> ExpressionNode:
        """
        Greedy extraction algorithm using extract.py with customizable cost functions.

        DEPRECATED: This method is maintained for backward compatibility.
        For full control, use extract.Extractor directly:

            from megg.egraph.extract import Extractor, AstSize
            extractor = Extractor(megg_egraph, AstSize())
            result = extractor.find_best()

        Args:
            eclass_id: E-class ID to extract from (uses first root if None)
            cost_function: Cost function to use (default: AstSize). If provided,
                          uses extract.Extractor. If None, uses legacy algorithm.

        Returns:
            ExpressionNode representing the extracted expression tree
        """
        if cost_function is not None:
            # Use new extract.py with custom cost function
            try:
                from megg.egraph.extract import Extractor
                extractor = Extractor(self, cost_function)
                result = extractor.find_best(eclass_id)
                return result.expr
            except ImportError:
                # Fall back to legacy if extract.py not available
                pass

        # Legacy extraction (for backward compatibility)
        if eclass_id is None:
            if not self.root_eclasses:
                raise ValueError("No root eclasses available for extraction")
            eclass_id = self.root_eclasses[0]

        # Memoization for extracted eclasses
        memo: Dict[str, ExpressionNode] = {}

        return self._extract_eclass(eclass_id, memo)

    def extract_all_roots(self) -> List[ExpressionNode]:
        """
        Extract expression trees for all root eclasses.

        Returns:
            List of ExpressionNode for each root eclass
        """
        return [self.extract_greedy(eclass_id) for eclass_id in self.root_eclasses]

    def extract_with_cost(self, cost_function, eclass_id: Optional[str] = None):
        """
        Extract using extract.py with a custom cost function.

        This is the RECOMMENDED way to perform extraction with custom costs.

        Args:
            cost_function: A CostFunction instance (e.g., AstSize(), OpWeightedCost(...))
            eclass_id: E-class to extract from (default: first root)

        Returns:
            ExtractionResult with cost, expr, and eclass_id

        Example:
            from megg.egraph.extract import AstSize, OpWeightedCost

            # Use built-in cost function
            result = egraph.extract_with_cost(AstSize())

            # Use custom operation weights
            custom_cost = OpWeightedCost({'Mul': 10.0, 'Add': 1.0})
            result = egraph.extract_with_cost(custom_cost)

            print(f"Cost: {result.cost}")
            print(f"Expression: {result.expr}")
        """
        from megg.egraph.extract import Extractor
        extractor = Extractor(self, cost_function)
        return extractor.find_best(eclass_id)

    def _normalize_op_name(self, op: str) -> str:
        """
        Normalize operation name from egglog format to MLIR format.

        Examples:
            'Term.add' -> 'Add'
            'Term.mul' -> 'Mul'
            'Term.arg' -> 'Arg'
            'Lit' -> 'Lit' (unchanged)
        """
        # Remove 'Term.' prefix if present
        if op.startswith('Term.'):
            op = op[5:]  # Remove 'Term.'

        # Capitalize first letter
        if op and op[0].islower():
            op = op[0].upper() + op[1:]

        # Normalize legacy control-flow suffix naming
        if op == 'For_':
            op = 'For'

        return op

    def _extract_eclass(self, eclass_id: str, memo: Dict[str, ExpressionNode]) -> ExpressionNode:
        """
        Recursively extract the lowest-cost term from an e-class.

        Args:
            eclass_id: E-class ID to extract
            memo: Memoization cache for extracted eclasses

        Returns:
            ExpressionNode for the lowest-cost term
        """
        # Check memo first
        if eclass_id in memo:
            return memo[eclass_id]

        # Get the e-class
        if eclass_id not in self.eclasses:
            raise ValueError(f"E-class {eclass_id} not found in e-graph")

        eclass = self.eclasses[eclass_id]


        # Select the lowest-cost node (greedy choice)
        best_node = None

        # Recursively extract children
        # NOTE: children contains node_ids, not eclass_ids
        # We need to map node_id -> eclass_id first
        children_exprs = []
        for child_node_id in best_node.children:
            # Get the eclass_id from the child node
            if child_node_id not in self.enodes:
                raise ValueError(f"Child node {child_node_id} not found in e-graph")
            child_eclass_id = self.enodes[child_node_id].eclass
            child_expr = self._extract_eclass(child_eclass_id, memo)
            children_exprs.append(child_expr)

        # Normalize operation name (remove Term. prefix, capitalize)
        normalized_op = self._normalize_op_name(best_node.op)

        # Create expression node
        expr_node = ExpressionNode(
            op=normalized_op,
            node_id=best_node.node_id,
            eclass_id=eclass_id,
            children=children_exprs,
            cost=best_node.cost,
            dtype=eclass.dtype,
            metadata={'node_type': best_node.node_type}
        )

        # Memoize
        memo[eclass_id] = expr_node

        return expr_node

    def dump_to_json(self, filepath: str) -> None:
        """
        Dump MeggEGraph to JSON file for debugging.

        Args:
            filepath: Path to output JSON file
        """
        data = {
            'enodes': {
                node_id: {
                    'op': enode.op,
                    'children': enode.children,
                    'eclass': enode.eclass,
                    'node_type': enode.node_type
                } for node_id, enode in self.enodes.items()
            },
            'eclasses': {
                eclass_id: {
                    'dtype': str(eclass.dtype) if eclass.dtype else None,
                    'nodes': [node.op for node in eclass.nodes]
                } for eclass_id, eclass in self.eclasses.items()
            },
            'root_eclasses': self.root_eclasses
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the e-graph.

        Returns:
            Dictionary with e-graph statistics
        """
        total_nodes = len(self.enodes)
        total_eclasses = len(self.eclasses)

        # Count nodes by type
        nodes_by_type = {}
        for node in self.enodes.values():
            nodes_by_type[node.node_type] = nodes_by_type.get(node.node_type, 0) + 1


        return {
            'total_nodes': total_nodes,
            'total_eclasses': total_eclasses,
            'nodes_by_type': nodes_by_type,
            'root_eclasses': len(self.root_eclasses)
        }

# ============================================================================
# Skeleton Matcher for Complex Instruction Matching
# ============================================================================

class SkeletonMatcher:
    """
    Optimized skeleton matcher using control flow index.

    Strategy:
    1. Use cf_index for O(1) candidate lookup (instead of O(N) traversal)
    2. Only verify components exist (no recursive structure validation)
    3. Lightweight verification for maximum performance

    Performance:
    - Before: O(N × skeletons) where N = total eclasses
    - After: O(candidates × components) where candidates << N
    """

    # Skeleton type to operation mapping
    # Note: Some skeleton types can map to multiple operations (e.g., scf.for -> for_ or for_with_carry)
    SKELETON_TYPE_TO_OP = {
        "scf.for": ["Term.for_", "Term.for_with_carry"],
        "scf.if": ["Term.if_"],
        "scf.while": ["Term.while_"],
        "func.body": ["Term.block"]
    }

    def __init__(self, egraph: MeggEGraph):
        self.egraph = egraph

    def match_skeleton(self, skeleton: 'Skeleton') -> List[tuple]:
        """
        Match skeleton using indexed search.

        Returns:
            List[(eclass_id, var_bindings)]
            - eclass_id: matched control flow node's eclass
            - var_bindings: variable → operand eclass mapping
        """
        matches = []

        # Step 1: Use index to find candidates (O(1) lookup)
        op_types = self.SKELETON_TYPE_TO_OP.get(skeleton.root.container_type)
        if op_types is None:
            logger.warning(f"Unknown skeleton type: {skeleton.root.container_type}")
            return matches

        # Collect candidates from all possible operation types
        candidates = []
        for op_type in op_types:
            candidates.extend(self.egraph.cf_index.get(op_type, []))

        logger.info(f"Skeleton '{skeleton.instr_name}': "
                   f"found {len(candidates)} candidates for {skeleton.root.container_type}")
        logger.debug(f"  Op type queried: {op_type}")
        logger.debug(f"  Expected components: {list(skeleton.leaf_patterns.keys())}")

        stmt_to_full_name: Dict[str, str] = {}
        for full_name in skeleton.leaf_patterns.keys():
            suffix = full_name[len(skeleton.instr_name) + 1:] if full_name.startswith(f"{skeleton.instr_name}_") else full_name
            stmt_to_full_name[suffix] = full_name

        required_components = {
            comp_name for comp_name, _, _ in skeleton.operand_constraints
        }

        # Step 2: For each candidate, verify components and constraints
        for candidate_node in candidates:
            bindings: Dict[str, str] = {}
            if not self._verify_skeleton_node(candidate_node, skeleton.root, bindings, skeleton, stmt_to_full_name):
                logger.debug(f"  Skeleton verification failed for eclass {candidate_node.eclass}")
                continue

            logger.info(f"  Bindings after skeleton verification: {bindings}")
            logger.info(f"  stmt_to_full_name mapping: {stmt_to_full_name}")

            component_operands: Dict[str, List[str]] = {}
            for stmt_name, stmt_eclass in bindings.items():
                full_name = stmt_to_full_name.get(stmt_name)
                logger.info(f"    Mapping stmt '{stmt_name}' to full_name '{full_name}'")
                if full_name:
                    operands = self._extract_component_operands(stmt_eclass)
                    logger.info(f"      Extracted {len(operands)} operands: {operands}")
                    component_operands[full_name] = operands

            logger.info(f"  component_operands keys: {list(component_operands.keys())}")
            logger.info(f"  Expected components: {list(skeleton.leaf_patterns.keys())}")

            missing = required_components - set(component_operands.keys())
            if missing:
                logger.info(f"  Missing components after verification: {missing}")
                continue

            var_bindings = self._verify_operand_constraints(component_operands, skeleton)
            if var_bindings is None:
                logger.debug(f"  Operand constraints failed for eclass {candidate_node.eclass}")
                continue

            matches.append((candidate_node.eclass, var_bindings))
            logger.info(f"✓ Matched skeleton '{skeleton.instr_name}' at eclass {candidate_node.eclass}")

        logger.info(f"Skeleton '{skeleton.instr_name}': {len(matches)} total matches")
        return matches

    def _verify_operand_constraints(self, component_operands: Dict[str, List[str]],
                                    skeleton: 'Skeleton') -> Optional[Dict[str, str]]:
        """Verify operand equality constraints and return variable bindings."""
        if not skeleton.operand_constraints:
            return {}

        var_bindings: Dict[str, str] = {}

        for comp_name, operand_idx, var_name in skeleton.operand_constraints:
            operands = component_operands.get(comp_name)
            if operands is None or operand_idx >= len(operands):
                logger.debug(f"  ✗ Component '{comp_name}' missing operand index {operand_idx}")
                return None

            operand_eclass = self._canonicalize_operand_eclass(operands[operand_idx])
            if not self._validate_operand_type(var_name, operand_eclass, skeleton):
                logger.debug(
                    f"  ✗ Type check failed for variable '{var_name}' bound to eclass {operand_eclass}"
                )
                return None
            bound = var_bindings.get(var_name)
            if bound is None:
                var_bindings[var_name] = operand_eclass
            elif bound != operand_eclass:
                logger.debug(f"  ✗ Constraint failed: variable '{var_name}' bound to {bound} != {operand_eclass}")
                return None

        return var_bindings

    def _validate_operand_type(self, var_name: str, operand_eclass: str,
                                skeleton: 'Skeleton') -> bool:
        """Ensure operand types match expected argument types when available."""
        if not skeleton.arg_var_to_index or var_name not in skeleton.arg_var_to_index:
            return True

        expected_type = skeleton.arg_types.get(var_name)
        if expected_type is None:
            return True

        operand_class = self.egraph.eclasses.get(operand_eclass)
        if operand_class is None:
            logger.debug(f"  ✗ Missing eclass '{operand_eclass}' while type checking '{var_name}'")
            return False

        actual_type = operand_class.dtype
        if actual_type is None:
            logger.debug(
                f"  Warning: operand eclass '{operand_eclass}' has unknown dtype; skipping strict check for '{var_name}'"
            )
            return True

        if actual_type != expected_type:
            logger.debug(
                "  ✗ Type mismatch for '%s': expected %s, got %s",
                var_name,
                expected_type,
                actual_type,
            )
            return False

        return True

    def _canonicalize_operand_eclass(self, operand_eclass: str, max_depth: int = 3) -> str:
        """Attempt to map operand eclass to canonical argument eclass if reachable."""
        if operand_eclass not in self.egraph.eclasses:
            return operand_eclass

        visited = set()
        queue = deque([(operand_eclass, 0)])

        while queue:
            current, depth = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            eclass = self.egraph.eclasses.get(current)
            if eclass is None:
                continue

            if any(node.op == "Term.arg" for node in eclass.nodes):
                return current

            if depth >= max_depth:
                continue

            for node in eclass.nodes:
                for child in node.children:
                    if child in self.egraph.eclasses and child not in visited:
                        queue.append((child, depth + 1))

        return operand_eclass

    def _extract_component_operands(self, stmt_eclass: str) -> List[str]:
        """Extract operand eclasses from a component instruction statement."""
        eclass = self.egraph.eclasses.get(stmt_eclass)
        if eclass is None:
            return []

        for node in eclass.nodes:
            if node.op != "Term.component_instr":
                continue

            best_children: Optional[List[str]] = None
            best_score = -1

            for child in node.children:
                if child not in self.egraph.eclasses:
                    continue

                vec_eclass = self.egraph.eclasses[child]
                for vec_node in vec_eclass.nodes:
                    if vec_node.op != "Vec":
                        continue

                    candidates = list(vec_node.children)
                    if not candidates:
                        continue

                    # Prefer operand vectors whose children come from argument eclasses
                    score = 0
                    for candidate in candidates:
                        ec = self.egraph.eclasses.get(candidate)
                        if not ec:
                            continue
                        if any(child_node.op == "Term.arg" for child_node in ec.nodes):
                            score += 1

                    if score > best_score:
                        best_children = candidates
                        best_score = score

            if best_children is not None:
                return best_children

        return []

    def _extract_body_eclass(self, control_node: MeggENode, container_type: str) -> Optional[str]:
        """
        Extract body block eclass from control flow node.

        Child index mapping:
        - Term.for_with_carry: children[5] (body)
        - Term.for_: children[4] (body)
        - Term.if_: children[1] (then_branch)
        - Term.while_: children[1] (before_block)
        - Term.block: node.eclass (itself)

        Args:
            control_node: Control flow node
            container_type: Skeleton container type ("scf.for", "scf.if", etc.)

        Returns:
            Body block eclass ID, or None if not found
        """
        if container_type == "func.body":
            return control_node.eclass
        elif container_type == "scf.for":
            # ForWithCarry has 7 children, For has 6
            if len(control_node.children) >= 6:
                return control_node.children[5] if len(control_node.children) == 7 else control_node.children[4]
        elif container_type == "scf.if":
            if len(control_node.children) >= 2:
                return control_node.children[1]  # then_branch
        elif container_type == "scf.while":
            if len(control_node.children) >= 2:
                return control_node.children[1]  # before_block

        return None

    def _find_component_name(self, stmt_eclass: str) -> Optional[str]:
        """
        Extract component_instr name from statement eclass.

        Args:
            stmt_eclass: Statement eclass ID

        Returns:
            Component name if found, None otherwise
        """
        eclass = self.egraph.eclasses.get(stmt_eclass)
        if eclass is None:
            return None

        for node in eclass.nodes:
            if node.op == "Term.component_instr":
                # Component name is stored in value field
                return node.value

        return None

    def _is_control_flow_match(self, node: MeggENode, skeleton_node) -> bool:
        """Check whether an e-graph node matches the expected skeleton container type."""
        expected_ops = self.SKELETON_TYPE_TO_OP.get(skeleton_node.container_type)
        if expected_ops is None:
            return False
        return node.op in expected_ops

    def _verify_loop_bounds(self, code_node: 'MeggENode', expected_bounds: Dict[str, int]) -> bool:
        """验证循环边界是否匹配 (Bug Fix #2)

        Args:
            code_node: Code 中的 for 循环节点（Term.for_with_carry 或 Term.for_）
            expected_bounds: Pattern 中期望的循环边界 {'lower': 0, 'upper': 4, 'step': 1}

        Returns:
            True 如果边界匹配，False 否则
        """
        # code_node.children 包含: [lower, upper, step, index_var, ...]
        if len(code_node.children) < 3:
            logger.debug(f"    Loop node has insufficient children: {len(code_node.children)}")
            return False

        lower_eclass = code_node.children[0]
        upper_eclass = code_node.children[1]
        step_eclass = code_node.children[2]

        # 从 eclass 中提取常量值
        actual_lower = self._extract_constant_from_eclass(lower_eclass)
        actual_upper = self._extract_constant_from_eclass(upper_eclass)
        actual_step = self._extract_constant_from_eclass(step_eclass)

        # 比较
        if actual_lower != expected_bounds.get('lower'):
            logger.debug(f"    Lower bound mismatch: {actual_lower} != {expected_bounds['lower']}")
            return False

        if actual_upper != expected_bounds.get('upper'):
            logger.debug(f"    Upper bound mismatch: {actual_upper} != {expected_bounds['upper']}")
            return False

        if actual_step != expected_bounds.get('step'):
            logger.debug(f"    Step mismatch: {actual_step} != {expected_bounds['step']}")
            return False

        logger.debug(f"    ✓ Loop bounds matched: lower={actual_lower}, upper={actual_upper}, step={actual_step}")
        return True

    def _extract_constant_from_eclass(self, eclass_id: str) -> Optional[int]:
        """从 eclass 中提取常量整数值 (Bug Fix #2)

        Args:
            eclass_id: E-class ID

        Returns:
            常量整数值，如果不是常量则返回 None
        """
        eclass = self.egraph.eclasses.get(eclass_id)
        if eclass is None:
            logger.debug(f"    Eclass {eclass_id} not found")
            return None

        # 在 eclass 的所有节点中查找 Term.lit
        for node in eclass.nodes:
            if node.op == "Term.lit":
                # node.value 可能存储了字面量值
                if node.value is not None:
                    try:
                        # 尝试解析整数
                        value_str = str(node.value)
                        # 可能的格式: "LitTerm.int(4)", "4", 等
                        import re
                        match = re.search(r'(\d+)', value_str)
                        if match:
                            return int(match.group(1))
                        return int(node.value)
                    except (ValueError, TypeError) as e:
                        logger.debug(f"    Failed to parse constant from value '{node.value}': {e}")

                # 尝试从 children 中获取（可能是 LitTerm.int(value)）
                if len(node.children) > 0:
                    # 第一个 child 可能是实际的常量值
                    try:
                        lit_term_eclass = node.children[0]
                        lit_eclass = self.egraph.eclasses.get(lit_term_eclass)
                        if lit_eclass:
                            for lit_node in lit_eclass.nodes:
                                # 查找 LitTerm.int 节点
                                if 'int' in lit_node.op.lower():
                                    if lit_node.value is not None:
                                        return int(lit_node.value)
                    except Exception as e:
                        logger.debug(f"    Failed to extract from children: {e}")

        logger.debug(f"    No constant found in eclass {eclass_id}")
        return None

    def _verify_condition_match(self, code_eclass: str, pattern_term) -> bool:
        """
        Verify that code's condition eclass matches the pattern term.

        This recursively checks the structure of the condition pattern tree,
        including all constants like predicate in Term.cmp.

        Args:
            code_eclass: Eclass ID of the condition in the code
            pattern_term: Pattern Term from skeleton (e.g., Term.cmp(arg0, arg1, "uge", ty))

        Returns:
            True if the pattern matches, False otherwise
        """
        # 获取pattern的字符串表示
        pattern_text = str(pattern_term)
        cmp_str = self._extract_cmp_pattern_string(pattern_text)
        pattern_str = cmp_str if cmp_str is not None else pattern_text
        logger.debug(f"    Verifying condition: pattern={pattern_str}, code_eclass={code_eclass}")

        # 获取code eclass的所有nodes
        eclass = self.egraph.eclasses.get(code_eclass)
        if eclass is None:
            logger.debug(f"    ✗ Code eclass {code_eclass} not found")
            return False

        # 尝试在eclass中找到匹配pattern的node
        # 策略: 检查pattern的op类型和关键参数(如predicate)
        for node in eclass.nodes:
            if self._node_matches_pattern(node, pattern_term, pattern_str):
                logger.debug(f"    ✓ Found matching node: {node.op}")
                return True

        logger.debug(f"    ✗ No matching node found in eclass {code_eclass}")
        logger.debug(f"      Available nodes: {[n.op for n in eclass.nodes]}")
        return False

    def _node_matches_pattern(self, node: MeggENode, pattern_term, pattern_str: str) -> bool:
        """
        Check if an e-graph node matches a pattern term.

        For Term.cmp, this includes checking the predicate value stored in node.value.
        """
        # 从pattern字符串中提取操作类型
        # 例如: "Term.cmp(arg0, arg1, "uge", "i1")" -> "Term.cmp"
        if not pattern_str.startswith("Term."):
            return False

        # 提取操作名 (例如: "Term.cmp(...)" -> "cmp")
        op_end = pattern_str.find("(")
        if op_end == -1:
            return False

        pattern_op = pattern_str[:op_end].strip()  # "Term.cmp"

        # 检查node的op是否匹配
        if node.op != pattern_op:
            return False

        # 对于Term.cmp,需要额外检查predicate
        if pattern_op == "Term.cmp":
            expected_pred = self._extract_cmp_predicate(pattern_str)
            if expected_pred is not None:
                actual_pred = node.value  # predicate存储在value中
                if actual_pred != expected_pred:
                    logger.debug(
                        "      ✗ Predicate mismatch: expected '%s', got '%s'",
                        expected_pred,
                        actual_pred,
                    )
                    return False
                logger.debug(f"      ✓ Predicate matched: '{expected_pred}'")

        # TODO: 递归验证children (如果需要完整的pattern tree matching)
        # 目前只验证根节点的op和关键参数(predicate)

        return True

    def _extract_cmp_pattern_string(self, term_str: str) -> Optional[str]:
        """Return canonical `Term.cmp(...)` portion from a pattern string."""
        idx = term_str.find("Term.cmp(")
        if idx == -1:
            return None

        start = idx
        depth = 0
        in_string = False
        escape = False

        for i in range(idx, len(term_str)):
            ch = term_str[i]

            if escape:
                escape = False
                continue

            if ch == '\\' and in_string:
                escape = True
                continue

            if ch == '"':
                in_string = not in_string
                continue

            if not in_string:
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                    if depth == 0:
                        return term_str[start:i + 1]

        return term_str[start:]

    def _extract_cmp_predicate(self, pattern_str: str) -> Optional[str]:
        """Safely extract the predicate (3rd argument) from a Term.cmp pattern string."""
        if not pattern_str.startswith("Term.cmp"):
            return None

        start = pattern_str.find('(')
        end = pattern_str.rfind(')')
        if start == -1 or end == -1 or end <= start:
            return None

        args_str = pattern_str[start + 1:end]
        args: List[str] = []
        current: List[str] = []
        depth = 0
        in_string = False
        escape = False

        for ch in args_str:
            if escape:
                current.append(ch)
                escape = False
                continue

            if ch == '\\' and in_string:
                current.append(ch)
                escape = True
                continue

            if ch == '"':
                in_string = not in_string
                current.append(ch)
                continue

            if not in_string:
                if ch in '([{':
                    depth += 1
                elif ch in ')]}':
                    if depth > 0:
                        depth -= 1
                elif ch == ',' and depth == 0:
                    args.append(''.join(current).strip())
                    current = []
                    continue

            current.append(ch)

        if current:
            args.append(''.join(current).strip())

        if len(args) >= 3:
            predicate_expr = args[2].strip()
            if predicate_expr.startswith('"') and predicate_expr.endswith('"'):
                return predicate_expr.strip('"')
            match = re.match(r'"([^\"]+)"', predicate_expr)
            if match:
                return match.group(1)

        logger.debug(
            "      ⚠ Unable to parse predicate from pattern '%s' (args=%s)",
            pattern_str,
            args,
        )
        return None

    # ============================================================================
    # DEPRECATED METHODS - Kept for compatibility, will be removed in future
    # ============================================================================

    def _verify_skeleton_node(self,
                              node: MeggENode,
                              skeleton_node,
                              bindings: Dict[str, str],
                              skeleton: 'Skeleton',
                              stmt_to_full_name: Dict[str, str]) -> bool:
        """
        Verify skeleton node structure and control flow parameters.

        Critical: Also verifies control flow parameters (condition, init_values) to ensure pattern匹配正确性!
        """
        # 验证控制流类型
        if not self._is_control_flow_match(node, skeleton_node):
            logger.info(f"  Control flow type mismatch: {node.op} vs {skeleton_node.container_type}")
            return False

        logger.info(f"  ✓ Matched control flow type: {skeleton_node.container_type} (op={node.op})")

        # Bug Fix: 验证循环边界 (对于 scf.for)
        if skeleton_node.container_type == "scf.for" and skeleton_node.loop_bounds:
            if not self._verify_loop_bounds(node, skeleton_node.loop_bounds):
                logger.info(f"  ✗ Loop bounds mismatch")
                logger.info(f"     Expected: {skeleton_node.loop_bounds}")
                return False
            logger.info(f"  ✓ Loop bounds matched: {skeleton_node.loop_bounds}")

        # 验证控制流参数 (condition/init_values)
        if skeleton_node.condition_term is not None:
            # scf.if的condition在children[0]
            # scf.while的condition在children[1] (after init_vals)
            if skeleton_node.container_type == "scf.if":
                if len(node.children) < 1:
                    logger.info(f"  ✗ scf.if node missing condition (no children)")
                    return False
                code_condition_eclass = node.children[0]
                if not self._verify_condition_match(code_condition_eclass, skeleton_node.condition_term):
                    logger.info(f"  ✗ scf.if condition mismatch")
                    logger.info(f"     Expected pattern: {skeleton_node.condition_term}")
                    logger.info(f"     Code eclass: {code_condition_eclass}")
                    return False
                logger.info(f"  ✓ scf.if condition matched: {skeleton_node.condition_term}")

        logger.info(f"  Verifying skeleton node {skeleton_node.container_type} with {len(skeleton_node.blocks)} blocks")

        # 对于每个 block，验证其 statements
        for skeleton_block in skeleton_node.blocks:
            # 获取对应的 MLIR block eclass
            block_eclass_id = self._find_block_for_skeleton(node, skeleton_node.container_type, skeleton_block.name)
            if block_eclass_id is None:
                logger.info(f"  ✗ Block '{skeleton_block.name}' not found in node {node.op}")
                return False

            logger.info(f"  ✓ Found block '{skeleton_block.name}' at eclass {block_eclass_id}")

            # 获取 block 的 Vec 元素
            block_stmts = self._get_block_statements(block_eclass_id)
            if block_stmts is None:
                logger.info(f"  ✗ No statements in block '{skeleton_block.name}'")
                return False

            logger.info(f"  Block '{skeleton_block.name}' has {len(block_stmts)} statements, expecting {len(skeleton_block.statements)}")

            # 验证 block 中的每个 statement
            if not self._verify_block_statements(block_stmts,
                                                 skeleton_block.statements,
                                                 bindings,
                                                 skeleton,
                                                 stmt_to_full_name):
                logger.info(f"  ✗ Statement verification failed for block '{skeleton_block.name}'")
                return False

        logger.debug(f"Successfully verified skeleton node {skeleton_node.container_type}")
        return True

    def _verify_block_statements(self,
                                block_stmts: List[str],
                                skeleton_stmts,
                                bindings: Dict[str, str],
                                skeleton: 'Skeleton',
                                stmt_to_full_name: Dict[str, str]) -> bool:
        """验证 block 的 statements 是否匹配 skeleton"""
        # 简单实现：检查每个 skeleton_stmt 是否能在 block_stmts 中找到
        # TODO: 可以添加顺序验证

        logger.info(f"    Verifying {len(skeleton_stmts)} skeleton statements against {len(block_stmts)} block statements")

        last_matched_index = -1

        for skeleton_stmt in skeleton_stmts:
            if skeleton_stmt.is_leaf():
                # 叶子 statement：查找对应的 component_instr
                # skeleton_stmt.name 如 "body_stmt1"
                # 完整名称应该从 pattern_term 获取或从 skeleton 获取
                # 但我们在 leaf_patterns 中有完整名称
                # 需要找到这个 statement 对应的完整名称
                logger.info(f"    Looking for leaf statement '{skeleton_stmt.name}'")

                # Check if this component has any constraints (needs to be verified)
                # If it has no constraints, we can skip the matching requirement
                # This handles the case where empty-operand components were skipped during rewrite generation
                full_name = stmt_to_full_name.get(skeleton_stmt.name)
                if full_name is None and skeleton.instr_name:
                    full_name = f"{skeleton.instr_name}_{skeleton_stmt.name}"

                has_constraints = any(
                    comp_name == full_name for comp_name, _, _ in skeleton.operand_constraints
                )

                if not has_constraints:
                    logger.info(f"      ⊘ Skipping leaf '{skeleton_stmt.name}' (no constraints, component may be empty-operand)")
                    continue

                found = False
                for idx in range(last_matched_index + 1, len(block_stmts)):
                    stmt_eclass = block_stmts[idx]
                    # 尝试匹配 component_instr
                    if self._is_component_match_by_stmt(stmt_eclass, skeleton_stmt):
                        bindings[skeleton_stmt.name] = stmt_eclass
                        logger.info(f"      ✓ Found leaf '{skeleton_stmt.name}' at {stmt_eclass}")
                        last_matched_index = idx
                        found = True
                        break

                if not found:
                    logger.info(f"      ✗ Leaf statement '{skeleton_stmt.name}' not found")
                    return False

            elif skeleton_stmt.is_nested():
                # 嵌套控制流：递归验证
                logger.info(f"    Looking for nested statement '{skeleton_stmt.name}' (type={skeleton_stmt.nested_skeleton.container_type})")
                found = False
                for idx in range(last_matched_index + 1, len(block_stmts)):
                    stmt_eclass = block_stmts[idx]
                    # 获取这个 eclass 的 node
                    eclass = self.egraph.eclasses.get(stmt_eclass)
                    if eclass:
                        for nested_node in eclass.nodes:
                            if self._verify_skeleton_node(nested_node, skeleton_stmt.nested_skeleton, bindings, skeleton, stmt_to_full_name):
                                bindings[skeleton_stmt.name] = stmt_eclass
                                logger.info(f"      ✓ Found nested '{skeleton_stmt.name}' at {stmt_eclass}")
                                last_matched_index = idx
                                found = True
                                break
                    if found:
                        break

                if not found:
                    logger.debug(f"Nested control flow '{skeleton_stmt.name}' not found")
                    return False

        return True

    def _is_component_match_by_stmt(self, stmt_eclass_id: str, skeleton_stmt) -> bool:
        """检查 statement eclass 是否匹配 skeleton_stmt（通过 component_instr）"""
        eclass = self.egraph.eclasses.get(stmt_eclass_id)
        if eclass is None:
            return False

        # 查找 ComponentInstr 节点
        # 注意：序列化后的 op 名称是 "Term.component_instr"
        for node in eclass.nodes:
            if node.op == "Term.component_instr":
                # component_instr 的第一个参数是名称（String）
                # children[0] 应该是 String eclass
                if len(node.children) > 0:
                    name_eclass_id = node.children[0]
                    # 尝试获取 String 值
                    # 简单实现：skeleton_stmt.name 应该包含在 component name 中
                    # component name 格式："{instr_name}_{stmt_name}"
                    # 例如：complex_mul_add_body_stmt1
                    # skeleton_stmt.name: body_stmt1
                    logger.info(f"        Found component_instr with name eclass: {name_eclass_id}")
                    # TODO: 更精确的名称匹配
                    # 现在简单地假设只要找到 component_instr 就匹配
                    return True

        return False

    def _find_block_for_skeleton(self, node: MeggENode, container_type: str, block_name: str) -> Optional[str]:
        """
        根据 skeleton block name 找到对应的 MLIR block eclass

        - scf.for: 只有 "body" block
        - scf.if: "then" 和 "else" blocks
        - scf.while: "before" 和 "after" blocks
        - func.body: 只有 "body" block
        """
        if container_type == "func.body" or container_type == "scf.for":
            # 只有一个 body block
            if block_name == "body":
                return self._find_body_block(node, container_type)

        elif container_type == "scf.if":
            # then 和 else blocks
            if block_name == "then" and len(node.children) >= 2:
                return node.children[1]
            elif block_name == "else" and len(node.children) >= 3:
                return node.children[2]

        elif container_type == "scf.while":
            # before 和 after blocks
            if block_name == "before" and len(node.children) >= 2:
                return node.children[1]
            elif block_name == "after" and len(node.children) >= 3:
                return node.children[2]

        return None

    def _find_body_block(self, control_node: MeggENode, control_type: str) -> Optional[str]:
        """
        从控制流节点中找到body block的eclass

        对于不同的控制流类型，block的位置不同：
        - func.body: Term.block本身就是block，返回其eclass
        - scf.for: Term.for_(lb, ub, step, loop_id, block, type) -> block是第5个child (index=4)
        - scf.for_with_carry: Term.for_with_carry(lb, ub, step, loop_idx, iter_args, block, type) -> block是第6个(index=5)
        - scf.if: Term.if_(cond, then_block, else_block, type) -> then_block是第2个child (index=1)
        - scf.while: Term.while_(init_args, before_block, after_block, type) -> before_block是第2个(index=1)
        """
        if control_type == "func.body":
            # func.body的节点本身就是Block，返回其eclass
            return control_node.eclass
        elif control_type == "scf.for":
            # For: [lb, ub, step, loop_id, block, type] (6 children)
            # ForWithCarry: [lb, ub, step, loop_idx, iter_args, block, type] (7 children)
            if len(control_node.children) == 6:
                return control_node.children[4]  # For
            elif len(control_node.children) == 7:
                return control_node.children[5]  # ForWithCarry
        elif control_type == "scf.if":
            # If: [cond, then_block, else_block, type]
            if len(control_node.children) >= 2:
                return control_node.children[1]  # then_block
        elif control_type == "scf.while":
            # While: [init_args, before_block, after_block, type]
            if len(control_node.children) >= 2:
                return control_node.children[1]  # before_block

        return None

    def _get_block_statements(self, block_eclass_id: str) -> Optional[List[str]]:
        """
        从block eclass中提取语句列表

        Block结构: Term.block(block_id, Vec[stmts], scope_name)
        需要找到Vec的children
        """
        eclass = self.egraph.eclasses.get(block_eclass_id)
        if eclass is None:
            logger.info(f"      ✗ Block eclass '{block_eclass_id}' not found")
            return None

        # 在eclass中查找Block节点
        for node in eclass.nodes:
            if node.op == "Term.block":
                # Block的children: [block_id, stmts_vec, scope_name]
                if len(node.children) >= 2:
                    stmts_vec_eclass = node.children[1]
                    statements = self._extract_vec_elements(stmts_vec_eclass)
                    _yield = self.egraph.eclasses.get(statements[-1]).nodes[0]
                    if _yield.op == 'Term.yield_':
                        # check if yield has no operands
                        for op in _yield.children:
                            op = self.egraph.eclasses.get(op)
                            if op is None:
                                continue
                            # op is a vec
                            assert op.nodes[0].op == 'Vec'
                            if len(op.nodes[0].children)==0:
                                statements=statements[:-1]
                                break
                    logger.info(f"      Found {len(statements)} statements in Vec at {stmts_vec_eclass}")
                    return statements

        logger.info(f"      ✗ No Term.block node in eclass '{block_eclass_id}'")
        return None

    def _extract_vec_elements(self, vec_eclass_id: str) -> List[str]:
        """
        从Vec eclass中提取所有元素的eclass列表

        Vec在egglog中的children就是所有元素
        """
        eclass = self.egraph.eclasses.get(vec_eclass_id)
        if eclass is None:
            return []

        # 查找Vec节点
        for node in eclass.nodes:
            if node.op == "Vec":
                # Vec的children就是所有元素
                return node.children

        # 如果没有找到Vec节点，返回空列表
        return []

    def _is_component_match(self, stmt_eclass_id: str, instr_name: str, comp_name: str) -> bool:
        """
        检查语句eclass是否匹配组件名称

        匹配条件：eclass中包含一个ComponentInstr节点，名称为{instr_name}_{comp_name}
        """
        eclass = self.egraph.eclasses.get(stmt_eclass_id)
        if eclass is None:
            return False

        expected_name = f"{instr_name}_{comp_name}"

        for node in eclass.nodes:
            # 查找ComponentInstr而不是CustomInstr
            if node.op == "ComponentInstr":
                # ComponentInstr的结构: component_instr(name, operands, type)
                # 我们需要检查第一个child（name）是否匹配
                # 在序列化后，name通常作为value存储
                if hasattr(node, 'value') and node.value:
                    # value可能是 String或直接的字符串
                    node_name = str(node.value).strip('"')
                    if node_name == expected_name:
                        return True

        return False

    def _verify_order(self, component_positions: Dict[str, int],
                     order_constraints: List[tuple]) -> bool:
        """
        验证组件在block中的顺序是否满足约束

        order_constraints: [("stmt0", "stmt1")] 表示stmt0必须在stmt1之前
        """
        for comp_a, comp_b in order_constraints:
            if comp_a not in component_positions or comp_b not in component_positions:
                return False
            if component_positions[comp_a] >= component_positions[comp_b]:
                return False  # 违反顺序约束

        return True
