"""
Greedy Extraction from E-graphs with Customizable Cost Functions.

This module implements the greedy extraction algorithm from egg (extract.rs),
adapted for Python and the MeggEGraph representation.
"""

from __future__ import annotations
from typing import Dict, Callable, Optional, TypeVar, Generic, Any, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

from megg.egraph.megg_egraph import MeggEGraph, MeggENode, MeggEClass, ExpressionNode


# Type variables
Cost = TypeVar('Cost')  # Cost can be any comparable type (float, int, etc.)


class CostFunction(ABC, Generic[Cost]):
    """
    Abstract base class for cost functions.
    
    Cost functions determine the cost of e-nodes during extraction.
    The cost should be monotonic: a parent node's cost should be >= its children's costs.
    """

    @abstractmethod
    def cost(self, node: MeggENode, costs: Callable[[str], Cost]) -> Cost:
        """
        Calculate the cost of an e-node.

        Args:
            node: The e-node to cost
            costs: Function to get the cost of a child e-class by ID

        Returns:
            The cost of this node
        """
        pass

    def cost_rec(self, expr: ExpressionNode) -> Cost:
        """
        Calculate the total cost of an expression tree.
        
        As provided, this just recursively calls `cost` all the way down the expression.
        """
        cost_map: Dict[str, Cost] = {}
        
        def compute_cost(node: ExpressionNode) -> Cost:
            if node.eclass_id in cost_map:
                return cost_map[node.eclass_id]
                
            child_costs = [compute_cost(child) for child in node.children]
            
            # Create a mock MeggENode for cost calculation
            mock_node = MeggENode(
                node_id=node.node_id,
                op=node.op,
                children=[child.eclass_id for child in node.children],
                eclass=node.eclass_id,
                node_type=node.metadata.get('node_type', 'expr')
            )
            
            def cost_lookup(eclass_id: str) -> Cost:
                for child in node.children:
                    if child.eclass_id == eclass_id:
                        return compute_cost(child)
                raise ValueError(f"Child eclass {eclass_id} not found")
            
            node_cost = self.cost(mock_node, cost_lookup)
            cost_map[node.eclass_id] = node_cost
            return node_cost
        
        return compute_cost(expr)


class AstSize(CostFunction[int]):
    """
    Simple cost function that counts total AST size.
    
    Each node has cost 1 + sum of children costs.
    This encourages smaller expressions.
    """

    def cost(self, node: MeggENode, costs: Callable[[str], int]) -> int:
        total = 1
        for child_eclass in node.children:
            total += costs(child_eclass)
        return total


class AstDepth(CostFunction[int]):
    """
    Simple cost function that counts maximum AST depth.
    
    Each node has cost 1 + max of children costs.
    This encourages shallower expressions.
    """

    def cost(self, node: MeggENode, costs: Callable[[str], int]) -> int:
        max_child = 0
        for child_eclass in node.children:
            max_child = max(max_child, costs(child_eclass))
        return 1 + max_child


class OpWeightedCost(CostFunction[float]):
    """
    Cost function with custom weights per operation.
    
    Allows assigning different costs to different operations.
    Useful for preferring certain operations over others.
    """

    def __init__(self, op_costs: Dict[str, float], default_cost: float = 1.0):
        """
        Initialize with operation-specific costs.

        Args:
            op_costs: Dictionary mapping operation names to costs
            default_cost: Default cost for operations not in op_costs
        """
        self.op_costs = op_costs
        self.default_cost = default_cost

    def cost(self, node: MeggENode, costs: Callable[[str], float]) -> float:
        # Get the base cost for this operation
        op_cost = self.op_costs.get(node.op, self.default_cost)

        # Add children costs
        total = op_cost
        for child_eclass in node.children:
            total += costs(child_eclass)

        return total


class ConstantFoldingCost(CostFunction[float]):
    """
    Cost function that strongly prefers constant-folded expressions.
    
    Literals have cost 1.0, all other operations have higher costs.
    This encourages the extractor to select constant expressions when available.
    """

    def cost(self, node: MeggENode, costs: Callable[[str], float]) -> float:
        if node.op == 'Lit':
            return 1.0

        # Non-literal operations have base cost + children
        base_cost = 5.0
        total = base_cost
        for child_eclass in node.children:
            total += costs(child_eclass)

        return total
    
    
# MeggCostFunction implementations
class MeggCost(CostFunction[float]):
    """
    Megg-specific cost function that assigns different costs to different operations.

    This cost function is similar to AstSize, but takes into account the specific
    operation type and assigns different costs based on the computational complexity
    of each operation.

    Cost categories:
    - Literals and arguments: 1.0 (cheapest)
    - Arithmetic operations: 2.0-10.0 (based on complexity)
    - Control flow: 10.0-20.0 (more expensive)
    - Memory operations: 5.0-15.0 (moderate to expensive)
    - Custom instructions: 15.0 (expensive, hardware-specific)
    """

    # Default operation costs
    OP_COSTS = {
        # Literals and basic values (cheapest)
        'LitTerm.int': 1.0,
        'LitTerm.float': 1.0,
        'Term.arg': 1.0,
        'Term.loop_index': 1.0,
        'Term.loop_iter_arg': 1.0,
        'Term.block_arg': 1.0,

        # Unary operations
        'Term.neg': 2.0,

        # Binary arithmetic operations (basic)
        'Term.add': 2.0,
        'Term.sub': 2.0,

        # Binary arithmetic operations (complex)
        'Term.mul': 4.0,
        'Term.div': 8.0,
        'Term.rem': 8.0,

        # Cast operations (moderate cost)
        'Term.index_cast': 100.0,
        'Term.sitofp': 5.0,
        'Term.uitofp': 5.0,
        'Term.fptosi': 5.0,
        'Term.fptoui': 5.0,
        'Term.extsi': 3.0,
        'Term.extui': 3.0,
        'Term.trunci': 3.0,
        'Term.bitcast': 2.0,

        # Comparison operations
        'Term.cmp': 3.0,

        # Control flow operations (expensive)
        'Term.if_': 10.0,
        'Term.while_': 20.0,
        'Term.for_': 15.0,
        'Term.for_with_carry': 18.0,
        'Term.affine_for': 15.0,
        'Term.affine_for_with_carry': 18.0,
        'Term.block': 5.0,
        'Term.yield_': 3.0,
        'Term.affine_yield': 3.0,
        'Term.condition': 5.0,

        # Memory operations (moderate to expensive)
        'Term.alloc': 10.0,
        'Term.get_global': 5.0,
        'Term.store': 8.0,
        'Term.load': 6.0,

        # Return operations
        'Term.return_': 2.0,

        # Custom instructions (cheap!!! - these are real instructions)
        'Term.custom_instr': 0.1,

        # Component instructions (INFINITE COST - should never be extracted!)
        # These are intermediate matching artifacts, not real instructions
        'Term.component_instr': float('inf'),

        # Vec operations (low cost, just containers)
        'Vec': 0.0,
    }

    def __init__(self, custom_costs: Optional[Dict[str, float]] = None):
        """
        Initialize MeggCost with optional custom operation costs.

        Args:
            custom_costs: Optional dictionary to override default costs for specific operations
        """
        self.op_costs = self.OP_COSTS.copy()
        if custom_costs:
            self.op_costs.update(custom_costs)

    def cost(self, node: MeggENode, costs: Callable[[str], float]) -> float:
        """
        Calculate the cost of a node based on its operation type and children.

        Args:
            node: The e-node to cost
            costs: Function to get the cost of a child e-class by ID

        Returns:
            The total cost of this node (operation cost + sum of children costs)
        """
        # Get the base cost for this operation
        op_cost = self.op_costs.get(node.op, 5.0)  # Default 5.0 for unknown ops

        # Add children costs
        total = op_cost
        for child_eclass in node.children:
            total += costs(child_eclass)

        return total


class AffineCost(CostFunction[float]):
    """
    Cost function that strongly prefers affine operations.

    Affine operations are linear arithmetic operations that can be represented as:
        a*x + b*y + c

    These include:
    - Addition, subtraction (linear)
    - Multiplication (linear when one operand is constant)
    - Negation (linear)
    - Literals/constants (affine)
    - Index casting, extension, truncation (affine transformations)

    Non-affine operations (divisions, shifts, bitwise, control flow, etc.) are
    heavily penalized to discourage their use in the extracted expression.

    Cost assignment:
    - Affine operations: 0.0 (preferred)
    - All other operations: 10000.0 (strongly discouraged)
    """

    # Affine operations (cost = 0)
    AFFINE_OPS = {
        # Arithmetic operations
        'Term.add',
        'Term.sub',
        'Term.mul',
        'Term.neg',

        # Literals and values
        'LitTerm.int',
        'LitTerm.float',
        'Term.arg',
        'Term.loop_index',
        'Term.loop_iter_arg',
        'Term.block_arg',

        # Cast operations (affine transformations)
        'Term.index_cast',
        'Term.extsi',
        'Term.extui',
        'Term.trunci',

        # Vec operations (containers, zero cost)
        'Vec',
    }

    # Non-affine penalty cost
    NON_AFFINE_COST = 10000.0

    def __init__(self, affine_cost: float = 0.0, non_affine_cost: float = 10000.0):
        """
        Initialize AffineCost with customizable costs.

        Args:
            affine_cost: Cost for affine operations (default: 0.0)
            non_affine_cost: Cost for non-affine operations (default: 10000.0)
        """
        self.affine_cost = affine_cost
        self.non_affine_cost = non_affine_cost

    def cost(self, node: MeggENode, costs: Callable[[str], float]) -> float:
        """
        Calculate the cost of a node based on whether it's affine or not.

        Args:
            node: The e-node to cost
            costs: Function to get the cost of a child e-class by ID

        Returns:
            The total cost of this node (operation cost + sum of children costs)
        """
        # Check if this operation is affine
        if node.op in self.AFFINE_OPS:
            op_cost = self.affine_cost
        else:
            op_cost = self.non_affine_cost

        # Add children costs
        total = op_cost
        for child_eclass in node.children:
            total += costs(child_eclass)

        return total


@dataclass
class ExtractionResult(Generic[Cost]):
    """Result of extraction from an e-class."""
    cost: Cost
    expr: ExpressionNode
    eclass_id: str


class Extractor(Generic[Cost]):
    """
    Greedy extractor for e-graphs with customizable cost functions.
    
    This implements the extraction algorithm from egg (extract.rs):
    1. Iteratively compute costs for all e-classes
    2. For each e-class, select the e-node with minimum cost  
    3. Recursively extract the best expression from each e-class
    """

    def __init__(self, egraph: MeggEGraph, cost_function: CostFunction[Cost]):
        """
        Create a new Extractor with the given cost function.

        Args:
            egraph: The e-graph to extract from
            cost_function: The cost function to use for extraction
        """
        self.egraph = egraph
        self.cost_function = cost_function
        self.costs: Dict[str, (Cost, MeggENode)] = {}
        self.logger = logging.getLogger(__name__)
        # Perform extraction on creation (like Rust implementation)
        self.find_costs()

    def find_best(self, eclass_id: Optional[str] = None) -> ExtractionResult[Cost]:
        """
        Find the cheapest (lowest cost) expression in the given e-class.

        Args:
            eclass_id: E-class ID to extract from (uses first root if None)

        Returns:
            ExtractionResult with cost and expression tree
        """
        if eclass_id is None:
            if not self.egraph.root_eclasses:
                raise ValueError("No root eclasses available")
            eclass_id = self.egraph.root_eclasses[0]

        if eclass_id not in self.costs:
            raise ValueError(f"E-class {eclass_id} has no computed cost")

        cost, best_node = self.costs[eclass_id]

        # Recursively build the expression tree
        expr = self.build_recexpr(eclass_id, best_node)

        return ExtractionResult(cost=cost, expr=expr, eclass_id=eclass_id)

    def find_best_node(self, eclass_id: str) -> MeggENode:
        """
        Find the cheapest e-node in the given e-class.

        Args:
            eclass_id: E-class ID

        Returns:
            The e-node with lowest cost
        """
        if eclass_id not in self.costs:
            raise ValueError(f"E-class {eclass_id} has no computed cost")

        _, best_node = self.costs[eclass_id]
        return best_node

    def find_best_cost(self, eclass_id: str) -> Cost:
        """
        Find the cost of the term that would be extracted from this e-class.

        Args:
            eclass_id: E-class ID

        Returns:
            The cost of the best expression
        """
        if eclass_id not in self.costs:
            raise ValueError(f"E-class {eclass_id} has no computed cost")

        cost, _ = self.costs[eclass_id]
        return cost

    def build_recexpr(self, eclass_id: str, node: MeggENode) -> ExpressionNode:
        """
        Build expression tree recursively from best nodes.
        
        This is similar to Rust's build_recexpr method.
        """
        # Recursively extract children
        children_exprs = []
        for child_eclass in node.children:
            if child_eclass not in self.costs:
                self.logger.warning(f"Child e-class {child_eclass} has no cost")
                continue
                
            _, child_best_node = self.costs[child_eclass]
            child_expr = self.build_recexpr(child_eclass, child_best_node)
            children_exprs.append(child_expr)

        # Get cost and convert to float for storage
        cost_val = self.costs[eclass_id][0]
        cost_float = float(cost_val) if isinstance(cost_val, (int, float)) else 0.0
        
        # Get type annotation
        eclass = self.egraph.eclasses[eclass_id]
        
        # Normalize operation name (remove Term. prefix, capitalize)
        normalized_op = self.normalize_op_name(node.op)

        en = ExpressionNode(
            op=normalized_op,
            node_id=node.node_id,
            eclass_id=eclass_id,
            children=children_exprs,
            cost=cost_float,
            dtype=eclass.dtype,
            metadata={'node_type': node.node_type}
        )
        # if the node is a literal, we can also store its value in metadata
        if node.node_type == 'lit':
            en.metadata['value'] = node.value
        elif node.node_type == 'arg':
            en.metadata['arg_info'] = node.value
        elif node.op == 'Term.cmp':
            en.metadata['cmp_info'] = node.value
        elif node.op == 'Term.while_':
            en.metadata['while_info'] = node.value
        elif node.op == 'Term.get_global':
            en.metadata['global_name'] = node.value
        elif node.op in ('Term.affine_for', 'Term.affine_for_with_carry'):
            en.metadata['affine_info'] = node.value or {}
        elif node.op == 'Term.custom_instr':
            en.metadata['instr_name'] = node.value
        print(f"metadata: {en.metadata}")
        
        return en

    def node_total_cost(self, node: MeggENode) -> Optional[Cost]:
        """
        Calculate total cost of a node if all children have costs.
        
        Returns None if any child e-class doesn't have a computed cost yet.
        """
    
        
        def is_legal_node(op: str) -> bool:
            # print(f"Checking legality of op: {op}")
            if op.startswith("String") or op.startswith("i64"):
                return False
            return True
        # self.logger.info(f"Before filtering, children: {node.children}")
        node.children = [child for child in node.children if is_legal_node(child)]
        # self.logger.info(f"After filtering, children: {node.children}")
        # Check if all children e-classes have computed costs
        for child_eclass in node.children:
            
            if child_eclass not in self.costs:
                # self.logger.info(f"Child e-class {child_eclass} has no computed cost yet")
                return None
        
        # All children have costs, compute this node's cost
        def cost_lookup(eclass_id: str) -> Cost:
            cost, _ = self.costs[eclass_id]
            return cost
        
        return self.cost_function.cost(node, cost_lookup)

    def find_costs(self):
        """
        Iteratively compute costs for all e-classes.
        
        This is the core of the greedy extraction algorithm from egg.
        It performs fixed-point iteration until all costs stabilize.
        """
        did_something = True
        iterations = 0
        max_iterations = len(self.egraph.eclasses) * 10  # Safety limit

        while did_something and iterations < max_iterations:
            did_something = False
            iterations += 1

            for eclass_id, eclass in self.egraph.eclasses.items():
                # Try to find the best node for this e-class
                new_cost_node = self.make_pass(eclass)

                if new_cost_node is None:
                    continue

                new_cost, new_node = new_cost_node

                # Update if we found a better cost
                if eclass_id not in self.costs:
                    self.costs[eclass_id] = (new_cost, new_node)
                    did_something = True
                else:
                    old_cost, _ = self.costs[eclass_id]
                    if self.compare_costs(new_cost, old_cost) < 0:
                        self.costs[eclass_id] = (new_cost, new_node)
                        did_something = True

        # Log warnings for e-classes without costs
        for eclass_id in self.egraph.eclasses:
            # nodes_num = len(self.egraph.eclasses[eclass_id].nodes)
            # if nodes_num == 0:
            #     continue
            if eclass_id not in self.costs:
                self.logger.warning(
                    f"Failed to compute cost for eclass {eclass_id}: {self.egraph.eclasses[eclass_id].nodes}"
                )
            else:
                pass
                # self.logger.info(f"E-class {eclass_id} cost: {self.costs[eclass_id][0]}")

        self.logger.info(f"Cost computation converged in {iterations} iterations")

    def make_pass(self, eclass: MeggEClass) -> Optional[tuple[Cost, MeggENode]]:
        """
        Try to find the best node in an e-class for this iteration.

        Args:
            eclass: The e-class to process

        Returns:
            (cost, node) tuple for the best node, or None if no costs available
        """
        best_cost = None
        best_node = None

        for node in eclass.nodes:

            # Try to compute total cost for this node
            node_cost = self.node_total_cost(node)
            # self.logger.info(f"Node {node.op} in eclass {eclass.eclass_id} has cost {node_cost}")
            if node_cost is None:
                continue

            # Update best if this is better
            if best_cost is None or self.compare_costs(node_cost, best_cost) < 0:
                best_cost = node_cost
                best_node = node

        if best_cost is not None and best_node is not None:
            return (best_cost, best_node)

        return None

    def normalize_op_name(self, op: str) -> str:
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

        return op

    def compare_costs(self, a: Cost, b: Cost) -> int:
        """
        Compare two costs.

        Args:
            a: First cost
            b: Second cost

        Returns:
            -1 if a < b, 0 if a == b, 1 if a > b
        """
        try:
            if a < b:
                return -1
            elif a > b:
                return 1
            else:
                return 0
        except TypeError:
            # If comparison fails, treat as equal
            self.logger.warning(f"Cost comparison failed for {a} and {b}")
            return 0


# Example usage and testing
if __name__ == "__main__":
    import json
    from megg.egraph.megg_egraph import EGraphDict

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Testing Greedy Extractor with Cost Functions")
    print("=" * 60)

    # Create a test e-graph: (a + b) + 5
    # Children field contains eclass_ids (matching Rust implementation)

    egraph_dict: EGraphDict = {
        'nodes': {
            'node_a': {
                'children': [],
                'cost': 1.0,
                'eclass': 'eclass_a',
                'op': 'Term.arg',
                'subsumed': False,
                'node_type': 'impl'
            },
            'node_b': {
                'children': [],
                'cost': 1.0,
                'eclass': 'eclass_b', 
                'op': 'Term.arg',
                'subsumed': False,
                'node_type': 'impl'
            },
            'node_lit_5': {
                'children': ['node_int_5'],  # 指向包含实际值的节点
                'cost': 1.0,
                'eclass': 'eclass_5',
                'op': 'Term.lit',
                'subsumed': False,
                'node_type': 'impl'
            },
            'node_int_5': {
                'children': [],
                'cost': 1.0,
                'eclass': 'eclass_5',  # 关键修改：和 node_lit_5 在同一个 eclass
                'op': 'LitTerm.int',
                'subsumed': False,
                'node_type': 'lit'
            },
            'node_add_ab': {
                'children': ['eclass_a', 'eclass_b'],  # 使用 eclass_ids
                'cost': 3.0,
                'eclass': 'eclass_ab',
                'op': 'Term.add',
                'subsumed': False,
                'node_type': 'impl'
            },
            'node_add_result': {
                'children': ['eclass_ab', 'eclass_5'],  # 使用 eclass_ids
                'cost': 5.0,
                'eclass': 'eclass_result',
                'op': 'Term.add',
                'subsumed': False,
                'node_type': 'impl'
            }
        },
        'root_eclasses': ['eclass_result'],
        'class_data': {
            'eclass_a': {'type': 'i32'},
            'eclass_b': {'type': 'i32'},
            'eclass_5': {'type': 'i32'},
            'eclass_ab': {'type': 'i32'},
            'eclass_result': {'type': 'i32'}
        }
    }
    megg_egraph = MeggEGraph.from_dict(egraph_dict)

    # Test 1: AstSize cost function
    print("\n1. Using AstSize cost function:")
    extractor_size = Extractor(megg_egraph, AstSize())
    result_size = extractor_size.find_best()
    print(f"   Cost: {result_size.cost}")
    print(f"   Expression: {result_size.expr.op}")

    # Test 2: AstDepth cost function  
    print("\n2. Using AstDepth cost function:")
    extractor_depth = Extractor(megg_egraph, AstDepth())
    result_depth = extractor_depth.find_best()
    print(f"   Cost: {result_depth.cost}")

    # Test 3: OpWeightedCost - prefer additions
    print("\n3. Using OpWeightedCost (Add=1.0, Mul=100.0):")
    weighted_cost = OpWeightedCost({'Add': 1.0, 'Mul': 100.0}, default_cost=5.0)
    extractor_weighted = Extractor(megg_egraph, weighted_cost)
    result_weighted = extractor_weighted.find_best()
    print(f"   Cost: {result_weighted.cost}")

    # Test 4: ConstantFoldingCost - prefer literals
    print("\n4. Using ConstantFoldingCost:")
    const_cost = ConstantFoldingCost()
    extractor_const = Extractor(megg_egraph, const_cost)
    result_const = extractor_const.find_best()
    print(f"   Cost: {result_const.cost}")

    print("\n" + "=" * 60)
    print("All extraction tests completed!")
    print("=" * 60)
