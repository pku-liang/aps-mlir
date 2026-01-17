from typing import List, Optional, Union, Tuple
from contextlib import contextmanager
from megg.mlir_utils import MLIRIRBuilder
from megg.utils import MOperation, MType, MBlock, MRegion, MValue, MModule


def Singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@Singleton
class IRBuilder:

    @staticmethod
    def _to_value(operand: Union[MValue, MOperation]) -> MValue:
        """Convert MOperation to MValue if needed"""
        if isinstance(operand, MOperation):
            return MValue(operand._op)
        elif isinstance(operand, MValue):
            return operand
        else:
            raise TypeError(f"Expected MValue or MOperation, got {type(operand)}")
        
    def __init__(self):
        self._builder = MLIRIRBuilder()

    def i1(self) -> MType:
        """Get i1 type"""
        return MType(self._builder.get_i1_type())

    def i32(self) -> MType:
        """Get i32 type"""
        return MType(self._builder.get_i32_type())

    def i64(self) -> MType:
        """Get i64 type"""
        return MType(self._builder.get_i64_type())

    def f32(self) -> MType:
        """Get f32 type"""
        return MType(self._builder.get_f32_type())

    def f64(self) -> MType:
        """Get f64 type"""
        return MType(self._builder.get_f64_type())

    def index(self) -> MType:
        """Get index type"""
        return MType(self._builder.get_index_type())

    def integer(self, width: int) -> MType:
        """Get integer type with specified width"""
        return MType(self._builder.get_integer_type(width))

    def llvm_ptr(self) -> MType:
        """Get LLVM opaque pointer type (!llvm.ptr)"""
        return MType(self._builder.get_llvm_ptr_type())

    def memref(self, shape: List[int], element_type: MType) -> MType:
        """Get memref type with shape and element type"""
        element_type = element_type._type
        return MType(self._builder.get_memref_type(shape, element_type))

    def function_type(self, inputs: List[MType], results: List[MType]) -> MType:
        """Create function type with input and result types"""
        inputs = [t._type for t in inputs]
        results = [t._type for t in results]
        return MType(self._builder.get_function_type(inputs, results))

    def create_function(self, name: str, arg_types: List[MType],
                        result_types: List[MType] = None) -> tuple[MOperation, MBlock]:
        if result_types is None:
            result_types = []
        arg_types = [arg._type for arg in arg_types]
        result_types = [r._type for r in result_types]
        func_op, entry_block = self._builder.create_function(
            name, arg_types, result_types)
        func_op = MOperation(func_op)
        entry_block = MBlock(entry_block)
        func_op.entry_block = entry_block  #TODO add member to avoid GC, should replace with a function type rather than operator?
        return func_op, entry_block

    # Constants
    def constant(self, value: Union[int, float], type: Optional[MType] = None) -> MValue:
        """Create a constant value"""
        type_str = str(type._type)
        if isinstance(value, float):
            if 'f32' in type_str:
                result = self._builder.create_constant_f32(value)
            else:
                result = self._builder.create_constant_f64(value)
        else:  # int
            if 'i64' in type_str:
                result = self._builder.create_constant_i64(value)
            elif 'index' in type_str:
                result = self._builder.create_constant_index(value)
            else:
                result = self._builder.create_constant_i32(value)
        return MValue(result)

    # Arithmetic operations
    def add(self, lhs: Union[MValue, MOperation], rhs: Union[MValue, MOperation]) -> MValue:
        """Add two values (integer or float based on operand types)"""
        lhs = self._to_value(lhs)
        rhs = self._to_value(rhs)
        lhs_type_str = str(lhs.type).strip()
        if 'f' == lhs_type_str[0]:
            result = self._builder.create_add_f(lhs._value, rhs._value)
        else:
            result = self._builder.create_add_i(lhs._value, rhs._value)
        return MValue(result)

    def sub(self, lhs: Union[MValue, MOperation], rhs: Union[MValue, MOperation]) -> MValue:
        """Subtract two values (integer or float based on operand types)"""
        lhs = self._to_value(lhs)
        rhs = self._to_value(rhs)
        lhs_type_str = str(lhs.type)
        if 'f32' in lhs_type_str or 'f64' in lhs_type_str:
            result = self._builder.create_sub_f(lhs._value, rhs._value)
        else:
            result = self._builder.create_sub_i(lhs._value, rhs._value)
        return MValue(result)

    def mul(self, lhs: Union[MValue, MOperation], rhs: Union[MValue, MOperation]) -> MValue:
        """Multiply two values (integer or float based on operand types)"""
        lhs = self._to_value(lhs)
        rhs = self._to_value(rhs)
        lhs_type_str = str(lhs.type)
        if 'f32' in lhs_type_str or 'f64' in lhs_type_str:
            result = self._builder.create_mul_f(lhs._value, rhs._value)
        else:
            result = self._builder.create_mul_i(lhs._value, rhs._value)
        return MValue(result)

    def div(self, lhs: Union[MValue, MOperation], rhs: Union[MValue, MOperation], signed: bool = True) -> MValue:
        """Divide two values"""
        lhs = self._to_value(lhs)
        rhs = self._to_value(rhs)
        lhs_type_str = str(lhs.type)
        if 'f32' in lhs_type_str or 'f64' in lhs_type_str:
            result = self._builder.create_div_f(lhs._value, rhs._value)
        elif signed:
            result = self._builder.create_div_si(lhs._value, rhs._value)
        else:
            result = self._builder.create_div_ui(lhs._value, rhs._value)
        return MValue(result)

    def add_i(self, lhs: Union[MValue, MOperation], rhs: Union[MValue, MOperation]) -> MValue:
        """Integer addition"""
        lhs = self._to_value(lhs)
        rhs = self._to_value(rhs)
        result = self._builder.create_add_i(lhs._value, rhs._value)
        return MValue(result)

    def sub_i(self, lhs: Union[MValue, MOperation], rhs: Union[MValue, MOperation]) -> MValue:
        """Integer subtraction"""
        lhs = self._to_value(lhs)
        rhs = self._to_value(rhs)
        result = self._builder.create_sub_i(lhs._value, rhs._value)
        return MValue(result)

    def mul_i(self, lhs: Union[MValue, MOperation], rhs: Union[MValue, MOperation]) -> MValue:
        """Integer multiplication"""
        lhs = self._to_value(lhs)
        rhs = self._to_value(rhs)
        result = self._builder.create_mul_i(lhs._value, rhs._value)
        return MValue(result)

    def div_si(self, lhs: Union[MValue, MOperation], rhs: Union[MValue, MOperation]) -> MValue:
        """Signed integer division"""
        lhs = self._to_value(lhs)
        rhs = self._to_value(rhs)
        result = self._builder.create_div_si(lhs._value, rhs._value)
        return MValue(result)

    def div_ui(self, lhs: Union[MValue, MOperation], rhs: Union[MValue, MOperation]) -> MValue:
        """Unsigned integer division"""
        lhs = self._to_value(lhs)
        rhs = self._to_value(rhs)
        result = self._builder.create_div_ui(lhs._value, rhs._value)
        return MValue(result)

    def add_f(self, lhs: Union[MValue, MOperation], rhs: Union[MValue, MOperation]) -> MValue:
        """Float addition"""
        lhs = self._to_value(lhs)
        rhs = self._to_value(rhs)
        result = self._builder.create_add_f(lhs._value, rhs._value)
        return MValue(result)

    def sub_f(self, lhs: Union[MValue, MOperation], rhs: Union[MValue, MOperation]) -> MValue:
        """Float subtraction"""
        lhs = self._to_value(lhs)
        rhs = self._to_value(rhs)
        result = self._builder.create_sub_f(lhs._value, rhs._value)
        return MValue(result)

    def mul_f(self, lhs: Union[MValue, MOperation], rhs: Union[MValue, MOperation]) -> MValue:
        """Float multiplication"""
        lhs = self._to_value(lhs)
        rhs = self._to_value(rhs)
        result = self._builder.create_mul_f(lhs._value, rhs._value)
        return MValue(result)

    def div_f(self, lhs: Union[MValue, MOperation], rhs: Union[MValue, MOperation]) -> MValue:
        """Float division"""
        lhs = self._to_value(lhs)
        rhs = self._to_value(rhs)
        result = self._builder.create_div_f(lhs._value, rhs._value)
        return MValue(result)

    def rem(self, lhs: Union[MValue, MOperation], rhs: Union[MValue, MOperation], signed: bool = True) -> MValue:
        """Remainder operation (modulo)"""
        lhs = self._to_value(lhs)
        rhs = self._to_value(rhs)
        lhs_type_str = str(lhs.type)
        if 'f32' in lhs_type_str or 'f64' in lhs_type_str:
            result = self._builder.create_rem_f(lhs._value, rhs._value)
        elif signed:
            result = self._builder.create_rem_si(lhs._value, rhs._value)
        else:
            result = self._builder.create_rem_ui(lhs._value, rhs._value)
        return MValue(result)

    def and_(self, lhs: Union[MValue, MOperation], rhs: Union[MValue, MOperation]) -> MValue:
        """Bitwise AND"""
        lhs = self._to_value(lhs)
        rhs = self._to_value(rhs)
        result = self._builder.create_and_i(lhs._value, rhs._value)
        return MValue(result)

    def or_(self, lhs: Union[MValue, MOperation], rhs: Union[MValue, MOperation]) -> MValue:
        """Bitwise OR"""
        lhs = self._to_value(lhs)
        rhs = self._to_value(rhs)
        result = self._builder.create_or_i(lhs._value, rhs._value)
        return MValue(result)

    def xor_(self, lhs: Union[MValue, MOperation], rhs: Union[MValue, MOperation]) -> MValue:
        """Bitwise XOR"""
        lhs = self._to_value(lhs)
        rhs = self._to_value(rhs)
        result = self._builder.create_xor_i(lhs._value, rhs._value)
        return MValue(result)

    def shl(self, lhs: Union[MValue, MOperation], rhs: Union[MValue, MOperation]) -> MValue:
        """Shift left"""
        lhs = self._to_value(lhs)
        rhs = self._to_value(rhs)
        result = self._builder.create_shl_i(lhs._value, rhs._value)
        return MValue(result)

    def shrsi(self, lhs: Union[MValue, MOperation], rhs: Union[MValue, MOperation]) -> MValue:
        """Shift right signed"""
        lhs = self._to_value(lhs)
        rhs = self._to_value(rhs)
        result = self._builder.create_shrsi_i(lhs._value, rhs._value)
        return MValue(result)

    def shrui(self, lhs: Union[MValue, MOperation], rhs: Union[MValue, MOperation]) -> MValue:
        """Shift right unsigned"""
        lhs = self._to_value(lhs)
        rhs = self._to_value(rhs)
        result = self._builder.create_shrui_i(lhs._value, rhs._value)
        return MValue(result)

    def cmpi(self, predicate: str, lhs: Union[MValue, MOperation], rhs: Union[MValue, MOperation]) -> MValue:
        """Integer comparison"""
        lhs = self._to_value(lhs)
        rhs = self._to_value(rhs)
        result = self._builder.create_cmpi(predicate, lhs._value, rhs._value)
        return MValue(result)

    def cmpf(self, predicate: str, lhs: Union[MValue, MOperation], rhs: Union[MValue, MOperation]) -> MValue:
        """Float comparison"""
        lhs = self._to_value(lhs)
        rhs = self._to_value(rhs)
        result = self._builder.create_cmpf(predicate, lhs._value, rhs._value)
        return MValue(result)

    def cmp(self, predicate: str, lhs: Union[MValue, MOperation], rhs: Union[MValue, MOperation]) -> MValue:
        """Generic comparison (automatically determines float vs int)"""
        lhs = self._to_value(lhs)
        type_str = str(lhs.type)
        is_float = any(ft in type_str for ft in ['f16', 'f32', 'f64'])
        if is_float:
            return self.cmpf(predicate, lhs, rhs)
        else:
            return self.cmpi(predicate, lhs, rhs)

    def select(self, condition: Union[MValue, MOperation],
               true_value: Union[MValue, MOperation],
               false_value: Union[MValue, MOperation]) -> MValue:
        """Select operation (ternary conditional: condition ? true_value : false_value)"""
        condition = self._to_value(condition)
        true_value = self._to_value(true_value)
        false_value = self._to_value(false_value)
        result = self._builder.create_select(condition._value, true_value._value, false_value._value)
        return MValue(result)

    def index_cast(self, value: Union[MValue, MOperation], target_type: MType) -> MValue:
        """Cast to/from index type"""
        value = self._to_value(value)
        target_type = target_type._type
        result = self._builder.create_index_cast(value._value, target_type)
        return MValue(result)

    def ptrtoint(self, ptr: Union[MValue, MOperation], int_type: MType) -> MValue:
        """Convert LLVM pointer to integer (llvm.ptrtoint)"""
        ptr = self._to_value(ptr)
        int_type = int_type._type
        result = self._builder.create_ptrtoint(ptr._value, int_type)
        return MValue(result)

    def sitofp(self, value: Union[MValue, MOperation], target_type: MType) -> MValue:
        """Signed integer to floating point"""
        value = self._to_value(value)
        target_type = target_type._type
        result = self._builder.create_si_to_fp(value._value, target_type)
        return MValue(result)

    def fptosi(self, value: Union[MValue, MOperation], target_type: MType) -> MValue:
        """Floating point to signed integer"""
        value = self._to_value(value)
        target_type = target_type._type
        result = self._builder.create_fp_to_si(value._value, target_type)
        return MValue(result)

    def fptoui(self, value: Union[MValue, MOperation], target_type: MType) -> MValue:
        """Floating point to unsigned integer"""
        value = self._to_value(value)
        target_type = target_type._type
        result = self._builder.create_fp_to_ui(value._value, target_type)
        return MValue(result)

    def uitofp(self, value: Union[MValue, MOperation], target_type: MType) -> MValue:
        """Unsigned integer to floating point"""
        value = self._to_value(value)
        target_type = target_type._type
        result = self._builder.create_ui_to_fp(value._value, target_type)
        return MValue(result)

    def extsi(self, value: Union[MValue, MOperation], target_type: MType) -> MValue:
        """Sign extend integer"""
        value = self._to_value(value)
        target_type = target_type._type
        result = self._builder.create_ext_si(value._value, target_type)
        return MValue(result)

    def extui(self, value: Union[MValue, MOperation], target_type: MType) -> MValue:
        """Zero extend integer"""
        value = self._to_value(value)
        target_type = target_type._type
        result = self._builder.create_ext_ui(value._value, target_type)
        return MValue(result)

    def trunci(self, value: Union[MValue, MOperation], target_type: MType) -> MValue:
        """Truncate integer"""
        value = self._to_value(value)
        target_type = target_type._type
        result = self._builder.create_trunc_i(value._value, target_type)
        return MValue(result)

    def bitcast(self, value: Union[MValue, MOperation], target_type: MType) -> MValue:
        """Bitcast between types"""
        value = self._to_value(value)
        target_type = target_type._type
        result = self._builder.create_bitcast(value._value, target_type)
        return MValue(result)

    def ret(self, values: List[Union[MValue, MOperation]] = None) -> MOperation:
        """Create return operation"""
        if values is None:
            values = []
        vals = []
        for val in values:
            val = self._to_value(val)
            vals.append(val._value)
        result = self._builder.create_return(vals)
        return MOperation(result)

    def func_return(self, values: List[Union[MValue, MOperation]] = None) -> MOperation:
        """Create func.return operation"""
        if values is None:
            values = []
        vals = []
        for val in values:
            val = self._to_value(val)
            vals.append(val._value)
        result = self._builder.create_func_return(vals)
        return MOperation(result)

    def scf_if(self, condition: Union[MValue, MOperation], result_types: List[MType] = None, has_else: bool = True) -> MOperation:
        """Create SCF if operation"""
        if result_types is None:
            result_types = []
        condition = self._to_value(condition)
        unwrapped_types = []
        for t in result_types:
            unwrapped_types.append(t._type)
        result = self._builder.create_scf_if(
            condition._value, unwrapped_types, has_else)
        return MOperation(result)

    def scf_while(self, result_types: List[MType], init_vals: List[Union[MValue, MOperation]]) -> MOperation:
        """Create SCF while loop"""
        unwrapped_types = []
        for t in result_types:
            unwrapped_types.append(t._type)
        unwrapped_vals = []
        for v in init_vals:
            v = self._to_value(v)
            unwrapped_vals.append(v._value)
        result = self._builder.create_scf_while(
            unwrapped_types, unwrapped_vals)
        return MOperation(result)

    def scf_for(self, lower: Union[MValue, MOperation], upper: Union[MValue, MOperation],
                step: Union[MValue, MOperation], iter_args: List[Union[MValue, MOperation]] = None) -> MOperation:
        """Create SCF for loop"""
        if iter_args is None:
            iter_args = []
        lower = self._to_value(lower)
        upper = self._to_value(upper)
        step = self._to_value(step)
        args = []
        for arg in iter_args:
            arg = self._to_value(arg)
            args.append(arg._value)
        result = self._builder.create_scf_for(
            lower._value, upper._value, step._value, args)
        return MOperation(result)

    def scf_yield(self, values: List[Union[MValue, MOperation]] = None) -> MOperation:
        """Create SCF yield operation"""
        if values is None:
            values = []
        vals = []
        for v in values:
            v = self._to_value(v)
            vals.append(v._value)
        result = self._builder.create_scf_yield(vals)
        return MOperation(result)

    def scf_condition(self, condition: Union[MValue, MOperation], values: List[Union[MValue, MOperation]] = None) -> MOperation:
        """Create SCF condition operation"""
        if values is None:
            values = []
        condition = self._to_value(condition)
        vals = []
        for v in values:
            v = self._to_value(v)
            vals.append(v._value)
        result = self._builder.create_scf_condition(
            condition._value, vals)
        return MOperation(result)

    # Loop constructs
    def affine_for(self, lower: int, upper: int, step: int = 1) -> MOperation:
        """Create affine for loop"""
        result = self._builder.create_affine_for(lower, upper, step)
        return MOperation(result)

    def affine_yield(self, values: List[Union[MValue, MOperation]] = None) -> MOperation:
        """Create affine yield operation"""
        if values is None:
            values = []
        vals = []
        for v in values:
            v = self._to_value(v)
            vals.append(v._value)
        result = self._builder.create_affine_yield(vals)
        return MOperation(result)

    # Memory operations
    def alloc(self, memref_type: MType) -> MOperation:
        """Allocate memory (heap allocation)"""
        memref_type = memref_type._type
        result = self._builder.create_memref_alloc(memref_type)
        return MOperation(result)

    def alloca(self, memref_type: MType) -> MOperation:
        """Allocate memory on stack (alloca)"""
        memref_type = memref_type._type
        result = self._builder.create_memref_alloca(memref_type)
        return MOperation(result)

    def store(self, value: Union[MValue, MOperation], memref: Union[MValue, MOperation],
              indices: List[Union[MValue, MOperation]]) -> None:
        """Store value to memory"""
        value = self._to_value(value)
        memref = self._to_value(memref)
        idx_vals = []
        for idx in indices:
            idx = self._to_value(idx)
            idx_vals.append(idx._value)
        self._builder.create_memref_store(value._value, memref._value, idx_vals)

    def load(self, memref: Union[MValue, MOperation], indices: List[Union[MValue, MOperation]]) -> MValue:
        """Load value from memory"""
        memref = self._to_value(memref)
        idx_vals = []
        for idx in indices:
            idx = self._to_value(idx)
            idx_vals.append(idx._value)
        result = self._builder.create_memref_load(memref._value, idx_vals)
        return MValue(result)

    def get_global(self, memref_type: MType, symbol_name: str) -> MOperation:
        """Get global memref"""
        memref_type = memref_type._type
        result = self._builder.create_memref_get_global(
            memref_type, symbol_name)
        return MOperation(result)

    def extract_aligned_pointer_as_index(self, memref: Union[MValue, MOperation]) -> MValue:
        """Extract aligned pointer from memref as index"""
        memref = self._to_value(memref)
        result = self._builder.create_memref_extract_aligned_pointer_as_index(
            memref._value)
        return MValue(result)

    # LLVM operations
    def inline_asm(self, asm_string: str, constraints: str,
                   operands: List[Union[MValue, MOperation]],
                   result_type: MType,
                   has_side_effects: bool = False) -> MValue:
        """Create LLVM inline assembly operation"""
        # Convert operands to raw values
        operand_vals = []
        for op in operands:
            op = self._to_value(op)
            operand_vals.append(op._value)

        # Create inline asm
        result = self._builder.create_inline_asm(
            asm_string,
            constraints,
            operand_vals,
            result_type._type,
            has_side_effects
        )
        return MValue(result)

    def create_block(self, region, arg_types: List[MType] = None) -> MBlock:
        """Create a new block in the region"""
        if arg_types is None:
            arg_types = []
        block = self._builder.create_block(region, arg_types)
        return MBlock(block)

    @contextmanager
    def set_insertion_point_to_start(self, target: Union[MOperation, MBlock]):
        """Set insertion point to start of function/block"""
        if isinstance(target, MOperation):
            # Use the function-specific method for operations
            self._builder.set_insertion_point_to_start_func(target._op)
        elif isinstance(target, MBlock):
            # Use the block-specific method for blocks
            self._builder.set_insertion_point_to_start_block(target._block)
        else:
            # Assume it's a raw MLIR operation
            self._builder.set_insertion_point_to_start_func(target)
        yield

    @contextmanager
    def set_insertion_point_to_end(self, target: Union[MOperation, MBlock]):
        """Set insertion point to end of function/block"""
        # Save current insertion point
        saved_ip = self._builder.save_insertion_point()
        try:
            if isinstance(target, MOperation):
                # Use the function-specific method for operations
                self._builder.set_insertion_point_to_end_func(target._op)
            elif isinstance(target, MBlock):
                # Use the block-specific method for blocks
                self._builder.set_insertion_point_to_end_block(target._block)
            yield
        finally:
            # Restore insertion point
            self._builder.restore_insertion_point(saved_ip)

    @contextmanager
    def set_insertion_point(self, target: Union[MOperation, MBlock]):
        """Set insertion point before operation or at block start"""
        if isinstance(target, MOperation):
            self._builder.set_insertion_point(target._op)
        elif isinstance(target, MBlock):
            # For blocks, use set_insertion_point_to_start_block
            self._builder.set_insertion_point_to_start_block(target._block)
        yield

    @contextmanager
    def set_insertion_point_after(self, target: Union[MOperation]):
        """Set insertion point after operation"""
        if isinstance(target, MOperation):
            self._builder.set_insertion_point_after(target._op)
        else:
            self._builder.set_insertion_point_after(target)
        yield

    @contextmanager
    def set_context(self, ctx):
        self._builder.set_context(ctx)
        yield

    # Helper methods for accessing operation regions and results
    def get_if_then_block(self, if_op: MOperation) -> MBlock:
        """Get the then block of an if operation"""
        raw_op = if_op._op
        # get_blocks() returns raw MLIRBlock objects, need to wrap
        then_region = raw_op.get_regions()[0]
        blocks = then_region.get_blocks()
        return MBlock(blocks[0])

    def get_if_else_block(self, if_op: MOperation) -> MBlock:
        """Get the else block of an if operation"""
        raw_op = if_op._op
        # Get the else region (second region, index 1)
        else_region = raw_op.get_regions()[1]
        # Get the first block from the else region
        else_blocks = else_region.get_blocks()
        if not else_blocks:
            raise ValueError("scf.if else region has no blocks")
        return MBlock(else_blocks[0])

    def get_for_body_block(self, for_op: MOperation) -> MBlock:
        """Get the body block of a for loop"""
        raw_op = for_op._op
        raw_op = for_op
        return raw_op.get_regions()[0].get_blocks()[0]

    def get_while_before_block(self, while_op: MOperation) -> MBlock:
        """Get the before block of a while loop"""
        raw_op = while_op._op
        return MBlock(raw_op.get_regions()[0].get_blocks()[0])

    def get_while_after_block(self, while_op: MOperation) -> MBlock:
        """Get the after block of a while loop"""
        raw_op = while_op._op
        # Access the second region (after region) and first block
        # get_blocks() already returns List[MBlock], don't wrap again
        return MBlock(raw_op.get_regions()[1].get_blocks()[0])

    def get_operation_result(self, op: MOperation, index: int = 0) -> Optional[MValue]:
        """Get a result value from an operation by index"""
        raw_op = op._op
        if index < len(raw_op.get_results()):
            return MValue(raw_op.get_results()[index])
        return None
