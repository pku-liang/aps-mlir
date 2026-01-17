"""
CADL Parser module

Provides the main parsing functionality using Lark parser generator.
Converts parse trees into AST nodes matching the Rust implementation.
"""

from pathlib import Path
from typing import Optional
import re
try:
    from lark import Lark, Transformer
    from lark.exceptions import UnexpectedToken, UnexpectedCharacters, ParseError
except ImportError:
    raise ImportError("lark package is required. Install with: pip install lark")
from .ast import *


class CADLParseError(Exception):
    """Enhanced CADL parse error with formatting"""
    
    def __init__(self, message: str, line: int, column: int, filename: Optional[str] = None, 
                 source_lines: Optional[list] = None, suggestion: Optional[str] = None):
        self.message = message
        self.line = line
        self.column = column
        self.filename = filename
        self.source_lines = source_lines or []
        self.suggestion = suggestion
        super().__init__(self.format_error())
    
    def format_error(self) -> str:
        """Format a pretty error message with source context"""
        lines = []
        
        # Header
        if self.filename:
            lines.append(f"Error in {self.filename}:")
        else:
            lines.append("Parse Error:")
        
        lines.append("")
        
        # Main error message
        lines.append(f"  {self.message}")
        lines.append("")
        
        # Source context
        if self.source_lines and self.line > 0:
            # Show 1-2 lines before and after the error
            start_line = max(1, self.line - 1)
            end_line = min(len(self.source_lines), self.line + 1)
            
            for line_num in range(start_line, end_line + 1):
                line_content = self.source_lines[line_num - 1] if line_num <= len(self.source_lines) else ""
                prefix = "  → " if line_num == self.line else "    "
                lines.append(f"{prefix}{line_num:4} | {line_content}")
                
                # Add pointer to error column
                if line_num == self.line and self.column > 0:
                    pointer_line = "    " + " " * 5 + " " * (self.column - 1) + "^"
                    lines.append(pointer_line)
        
        # Location info
        lines.append("")
        lines.append(f"  at line {self.line}, column {self.column}")
        
        # Suggestion if available
        if self.suggestion:
            lines.append("")
            lines.append(f"  💡 Suggestion: {self.suggestion}")
        
        return "\n".join(lines)


def analyze_source_context(source_lines: list, line: int, column: int, token_value: Optional[str] = None) -> tuple[str, str]:
    """Analyze source context to provide detailed error messages"""
    if line <= 0 or line > len(source_lines):
        return "Parse error", "Invalid line number"
    
    current_line = source_lines[line - 1]
    
    # Check for common patterns
    
    # 1. Missing semicolon patterns
    if line > 1:
        prev_line = source_lines[line - 2].strip()
        if (prev_line.endswith(('42', ')', ']', 'true', 'false')) and 
            not prev_line.endswith(';') and 
            not prev_line.endswith('{') and 
            not prev_line.endswith(',') and
            'let ' in prev_line or '=' in prev_line):
            return "Missing semicolon", f"Add ';' at the end of line {line - 1}"
    
    # 2. Invalid identifier patterns
    if token_value and token_value.isdigit():
        if column > 1:
            before = current_line[:column-1].strip().split()
            if before and before[-1] in ['rtype', 'flow', 'fn', 'static']:
                return "Invalid identifier", f"Identifiers cannot start with numbers. Use a letter or underscore instead of '{token_value}'"
    
    # 3. Keyword as identifier
    keywords = ['let', 'if', 'else', 'fn', 'while', 'with', 'do', 'in', 'spawn', 
                'thread', 'return', 'break', 'continue', 'match', 'sel', 'true', 
                'false', 'regfile', 'static', 'rtype', 'flow', 'Instance']
    if token_value in keywords:
        return "Reserved keyword", f"'{token_value}' is a reserved keyword and cannot be used as an identifier"
    
    # 4. Missing colon in declarations
    if ':' in current_line and token_value:
        parts = current_line.split(':')
        if len(parts) >= 2 and token_value in parts[0]:
            # Check if there's missing colon before type
            type_part = parts[1].strip().split()[0] if parts[1].strip() else ""
            if type_part.startswith(('u', 'i', 'f')) and type_part[1:].isdigit():
                return "Missing colon", "Variable declarations need ':' between name and type"
    
    # 5. Unterminated string detection
    if '"' in current_line:
        quotes = current_line.count('"')
        if quotes % 2 != 0:
            return "Unterminated string", "Missing closing quote '\"' for string literal"
    
    # 6. Function parameter syntax errors (check before bracket analysis)
    if 'fn ' in current_line and '(' in current_line:
        # Look for patterns like "fn name(param: type -> type2)" which is wrong
        paren_content = current_line[current_line.find('('):current_line.rfind(')')+1] if '(' in current_line and ')' in current_line else ""
        if '->' in paren_content:
            # This is likely a function parameter error - arrow should be outside parentheses
            return "Invalid function parameter", "Return type arrow '->' should be outside parameter list: 'fn name(param: type) -> (return_type)'"
    
    # 7. Mismatched brackets detection
    bracket_pairs = {'(': ')', '[': ']', '{': '}'}
    stack = []
    for i, char in enumerate(current_line):
        if char in bracket_pairs:
            stack.append((char, i))
        elif char in bracket_pairs.values():
            if not stack:
                return "Extra closing bracket", f"Unexpected '{char}' without matching opening bracket"
            last_open, pos = stack.pop()
            if bracket_pairs[last_open] != char:
                return "Mismatched brackets", f"Expected '{bracket_pairs[last_open]}' but found '{char}'"
    
    if stack:
        missing_close = bracket_pairs[stack[-1][0]]
        return "Unclosed bracket", f"Missing closing '{missing_close}'"
    
    # 7. Invalid number literal patterns
    if "'" in current_line and token_value:
        # Check for malformed width specifiers like 5'b10201 (invalid binary)
        width_pattern = r"(\d+)'([bBoOdDhH])([0-9a-fA-F]+)"
        match = re.search(width_pattern, current_line)
        if match:
            width, base, digits = match.groups()
            base_lower = base.lower()
            if base_lower == 'b' and not all(d in '01' for d in digits):
                return "Invalid binary literal", f"Binary literals can only contain 0 and 1, found '{digits}'"
            elif base_lower == 'o' and not all(d in '01234567' for d in digits):
                return "Invalid octal literal", f"Octal literals can only contain 0-7, found '{digits}'"
            elif base_lower in ['h'] and not all(d in '0123456789abcdefABCDEF' for d in digits):
                return "Invalid hexadecimal literal", f"Hexadecimal literals can only contain 0-9 and A-F, found '{digits}'"
    
    # 8. Missing arrow in function declarations
    if ('fn ' in current_line or 'rtype ' in current_line) and '(' in current_line and ')' in current_line:
        if '->' not in current_line and '{' in current_line:
            return "Missing return type", "Function declarations need '-> (return_type)' before the body"
    
    # 9. Empty array size
    if '[' in current_line and ']' in current_line:
        array_match = re.search(r'\[\s*([^;]*?)\s*;\s*([^]]*?)\s*\]', current_line)
        if array_match:
            element_type, size = array_match.groups()
            if not element_type.strip():
                return "Missing array element type", "Array declarations need an element type like '[u32; 5]'"
            if not size.strip():
                return "Missing array size", "Array declarations need a size like '[u32; 5]'"
    
    # Default fallback
    return "Syntax error", "Check the syntax around this location"


def format_lark_error(e: Exception, source: str, filename: Optional[str] = None) -> CADLParseError:
    """Convert Lark parsing exception to pretty CADLParseError"""
    source_lines = source.splitlines()
    
    if isinstance(e, UnexpectedToken):
        line = e.line
        column = e.column
        token = e.token
        expected = e.expected
        
        token_value = token.value if hasattr(token, 'value') else str(token)
        
        # Analyze the context for detailed error detection
        error_type, suggestion = analyze_source_context(source_lines, line, column, token_value)
        
        # Create detailed message based on analysis
        if error_type == "Missing semicolon":
            message = f"Missing semicolon before '{token_value}'"
        elif error_type == "Invalid identifier":
            message = f"Invalid identifier '{token_value}'"
        elif error_type == "Reserved keyword":
            message = f"Cannot use reserved keyword '{token_value}' as identifier"
        elif error_type == "Unterminated string":
            message = "Unterminated string literal"
        elif error_type in ["Unclosed bracket", "Extra closing bracket", "Mismatched brackets"]:
            message = f"Bracket error: {error_type.lower()}"
        elif error_type.startswith("Invalid") and "literal" in error_type:
            message = f"Malformed number literal"
        elif error_type == "Missing return type":
            message = "Missing return type in function declaration"
        elif error_type == "Invalid function parameter":
            message = "Invalid function parameter syntax"
        elif error_type.startswith("Missing array"):
            message = f"Array declaration error: {error_type.lower()}"
        else:
            # Fallback to enhanced generic message
            message = f"Unexpected '{token_value}'"
            
            # Enhanced suggestions based on expected tokens
            if 'SEMICOLON' in expected and 'RBRACE' in expected:
                suggestion = "Missing semicolon ';' to end statement, or '}' to close block"
            elif 'COLON' in expected:
                suggestion = "Missing ':' in type annotation (e.g., 'name: type')"
            elif 'ASSIGN' in expected:
                suggestion = "Missing '=' for assignment"
            elif 'COMMA' in expected:
                suggestion = "Missing ',' to separate items in a list"
            elif any(op in expected for op in ['OP_PLUS', 'OP_MINUS', 'OP_MULTIPLY']):
                suggestion = "Expression appears incomplete - missing operator or operand"
        
        return CADLParseError(message, line, column, filename, source_lines, suggestion)
    
    elif isinstance(e, UnexpectedCharacters):
        line = e.line
        column = e.column
        char = source[e.pos_in_stream] if e.pos_in_stream is not None and e.pos_in_stream < len(source) else '?'
        
        # Analyze context for character-level errors
        error_type, suggestion = analyze_source_context(source_lines, line, column)
        
        if error_type == "Unterminated string":
            message = "Unterminated string literal"
        else:
            message = f"Unexpected character '{char}'"
            
            # Enhanced character-specific suggestions
            if char == '_' and column == 1:
                suggestion = "Identifiers starting with '_' might be reserved (like _irf, _mem)"
            elif char in '"\'':
                suggestion = "Check for unterminated string literals or incorrect quote usage"
            elif char in '()[]{}':
                suggestion = "Check for mismatched brackets or parentheses"
            elif char.isdigit():
                suggestion = "Numbers cannot start identifiers - use a letter or underscore"
            elif not char.isprintable():
                suggestion = f"Invalid character (code: {ord(char)}) - check file encoding"
        
        return CADLParseError(message, line, column, filename, source_lines, suggestion)
    
    else:
        # Enhanced generic parse error
        message = f"Parse error: {str(e)}"
        suggestion = "Check overall file syntax and structure"
        return CADLParseError(message, 1, 1, filename, source_lines, suggestion)


class CADLTransformer(Transformer):
    """Transformer to convert Lark parse tree to CADL AST"""

    def _validate_burst_operation(self, lhs: Expr, rhs: Expr) -> None:
        """
        Validate burst operation constraints:
        - Burst lengths must be compile-time constants (literals)
        """
        def check_burst_length(expr: Expr, side_name: str) -> None:
            """Check if expr is a burst operation with non-constant length"""
            if isinstance(expr, RangeSliceExpr):
                # Check if this is a burst operation
                if isinstance(expr.expr, IdentExpr):
                    if expr.expr.name in ("_burst_read", "_burst_write"):
                        if expr.length is None:
                            raise ValueError(f"Burst operation on {side_name} must have explicit length")
                        # Require length to be a literal expression
                        if not isinstance(expr.length, LitExpr):
                            raise ValueError(
                                f"Burst operation length on {side_name} must be a compile-time constant literal. "
                                f"Got: {type(expr.length).__name__}"
                            )

        # Check both sides
        check_burst_length(lhs, "LHS")
        check_burst_length(rhs, "RHS")

    # Literals and identifiers
    def number_lit(self, items):
        literal_str = str(items[0])
        literal = parse_literal_from_string(literal_str)
        return LitExpr(literal)

    def string_lit(self, items):
        return StringLitExpr(str(items[0]).strip('"'))

    def true_lit(self, items):
        return LitExpr("true")

    def false_lit(self, items):
        return LitExpr("false")

    def identifier(self, items):
        return IdentExpr(str(items[0]))

    # Type system
    def single_type(self, items):
        basic_type = parse_basic_type_from_string(str(items[0]))
        return DataType_Single(basic_type)

    def array_type(self, items):
        # Grammar: LBRACKET VARTYPE (SEMICOLON NUMBER_LIT)* RBRACKET
        element_type = parse_basic_type_from_string(str(items[1]))  # Skip LBRACKET
        dimensions = []
        # Extract dimensions from SEMICOLON NUMBER_LIT pairs
        i = 2
        while i < len(items) - 1:  # Skip final RBRACKET
            if hasattr(items[i], 'type') and items[i].type == 'SEMICOLON':
                dimensions.append(int(str(items[i + 1])))
                i += 2
            else:
                i += 1
        return DataType_Array(element_type, dimensions)

    def instance_type(self, items):
        return DataType_Instance()

    def basic_type(self, items):
        return CompoundType_Basic(items[0])

    # Function arguments
    def fn_arg(self, items):
        name = str(items[0])
        # items[1] is the COLON token, items[2] is the compound_type
        type_info = items[2]
        return FnArg(name, type_info)

    def fn_arg_list(self, items):
        # Filter out comma tokens and return only FnArg objects
        return [item for item in items if isinstance(item, FnArg)]

    # With bindings
    def with_binding(self, items):
        name = str(items[0])      # IDENTIFIER
        type_name = str(items[2]) # VARTYPE (skip COLON)
        basic_type = parse_basic_type_from_string(type_name)
        init_expr = items[5] if len(items) > 5 and items[5] is not None else None  # first expr
        next_expr = items[7] if len(items) > 7 and items[7] is not None else None  # second expr
        return WithBinding(name, basic_type, init_expr, next_expr)

    # Expressions - Binary operations
    def add_op(self, items):
        return BinaryExpr(BinaryOp.ADD, items[0], items[2])

    def sub_op(self, items):
        return BinaryExpr(BinaryOp.SUB, items[0], items[2])

    def mul_op(self, items):
        return BinaryExpr(BinaryOp.MUL, items[0], items[2])

    def div_op(self, items):
        return BinaryExpr(BinaryOp.DIV, items[0], items[2])

    def rem_op(self, items):
        return BinaryExpr(BinaryOp.REM, items[0], items[2])

    def eq_op(self, items):
        return BinaryExpr(BinaryOp.EQ, items[0], items[2])

    def ne_op(self, items):
        return BinaryExpr(BinaryOp.NE, items[0], items[2])

    def lt_op(self, items):
        return BinaryExpr(BinaryOp.LT, items[0], items[2])

    def le_op(self, items):
        return BinaryExpr(BinaryOp.LE, items[0], items[2])

    def gt_op(self, items):
        return BinaryExpr(BinaryOp.GT, items[0], items[2])

    def ge_op(self, items):
        return BinaryExpr(BinaryOp.GE, items[0], items[2])

    def and_op(self, items):
        return BinaryExpr(BinaryOp.AND, items[0], items[2])

    def or_op(self, items):
        return BinaryExpr(BinaryOp.OR, items[0], items[2])

    def lshift_op(self, items):
        return BinaryExpr(BinaryOp.LSHIFT, items[0], items[2])

    def rshift_op(self, items):
        return BinaryExpr(BinaryOp.RSHIFT, items[0], items[2])

    def bit_and_op(self, items):
        return BinaryExpr(BinaryOp.BIT_AND, items[0], items[2])

    def bit_or_op(self, items):
        return BinaryExpr(BinaryOp.BIT_OR, items[0], items[2])

    def bit_xor_op(self, items):
        return BinaryExpr(BinaryOp.BIT_XOR, items[0], items[2])

    # Expressions - Unary operations
    def neg_op(self, items):
        return UnaryExpr(UnaryOp.NEG, items[1])  # Skip OP_MINUS token

    def not_op(self, items):
        return UnaryExpr(UnaryOp.NOT, items[1])  # Skip OP_NOT token

    def bit_not_op(self, items):
        return UnaryExpr(UnaryOp.BIT_NOT, items[1])  # Skip OP_BIT_NOT token

    def signed_cast(self, items):
        # items = [CAST_TOKEN, LPAREN, expr, RPAREN]
        return UnaryExpr(UnaryOp.SIGNED_CAST, items[2])

    def unsigned_cast(self, items):
        # items = [CAST_TOKEN, LPAREN, expr, RPAREN]
        return UnaryExpr(UnaryOp.UNSIGNED_CAST, items[2])

    def f32_cast(self, items):
        # items = [CAST_TOKEN, LPAREN, expr, RPAREN]
        return UnaryExpr(UnaryOp.F32_CAST, items[2])

    def f64_cast(self, items):
        # items = [CAST_TOKEN, LPAREN, expr, RPAREN]
        return UnaryExpr(UnaryOp.F64_CAST, items[2])

    def int_cast(self, items):
        # items = [CAST_TOKEN, LPAREN, expr, RPAREN]
        return UnaryExpr(UnaryOp.INT_CAST, items[2])

    def uint_cast(self, items):
        # items = [CAST_TOKEN, LPAREN, expr, RPAREN]
        return UnaryExpr(UnaryOp.UINT_CAST, items[2])

    # Complex expressions
    def call_expr(self, items):
        name = str(items[0])  # IDENTIFIER
        # items[1] is LPAREN, items[2] is expr_list (optional), items[3] is RPAREN
        args = items[2] if len(items) > 3 and items[2] else []
        return CallExpr(name, args)

    def index_expr(self, items):
        expr = items[0]
        indices = items[2]  # Skip LBRACKET token, get expr_list
        return IndexExpr(expr, indices)

    def slice_expr(self, items):
        expr = items[0]
        start = items[2]  # Skip LBRACKET token
        end = items[4]    # Skip COLON token
        return SliceExpr(expr, start, end)

    def range_slice_expr(self, items):
        # Grammar: postfix_expr LBRACKET expr OP_PLUS COLON expr? RBRACKET
        expr = items[0]
        start = items[2]  # Skip LBRACKET token
        # items[3] is OP_PLUS, items[4] is COLON
        length = items[5] if len(items) > 6 and items[5] is not None else None
        return RangeSliceExpr(expr, start, length)

    def paren_expr(self, items):
        # items = [LPAREN, expr_list, RPAREN]
        expr_list = items[1] if len(items) > 1 else []
        if isinstance(expr_list, list):
            if len(expr_list) == 1:
                return expr_list[0]
            return TupleExpr(expr_list)
        else:
            return expr_list

    def if_expr(self, items):
        # items = [KW_IF, condition, LBRACE, then_branch, RBRACE, KW_ELSE, LBRACE, else_branch, RBRACE]
        condition = items[1]      # Skip KW_IF
        then_branch = items[3]    # Skip LBRACE
        else_branch = items[7]    # Skip KW_ELSE, LBRACE
        return IfExpr(condition, then_branch, else_branch)

    def select_expr(self, items):
        # Grammar: KW_SEL LBRACE sel_arm+ RBRACE
        # items[0] = KW_SEL token
        # items[1] = LBRACE token
        # items[2:-1] = sel_arm tuples (condition, value)
        # items[-1] = RBRACE token

        # Filter out tokens, keep only sel_arm tuples
        arms_raw = [item for item in items if isinstance(item, tuple)]

        if len(arms_raw) == 0:
            raise ValueError("select expression must have at least one arm")

        if len(arms_raw) == 1:
            # Only one arm - use it as default with no conditional arms
            arms = []
            default = arms_raw[0][1]  # Value part of the only arm
        else:
            # Multiple arms - all but last are conditional, last is default
            arms = arms_raw[:-1]  # List of (condition, value) tuples
            default = arms_raw[-1][1]  # Value part of last arm (ignore its condition)

        return SelectExpr(arms, default)

    def aggregate_expr(self, items):
        # Grammar: LBRACE expr_list RBRACE
        expr_list = items[1]  # Skip LBRACE, get expr_list, skip RBRACE
        return AggregateExpr(expr_list)

    def sel_arm(self, items):
        # Grammar: expr COLON expr COMMA
        # items[0] = condition expr
        # items[1] = COLON token
        # items[2] = value expr
        # items[3] = COMMA token
        return (items[0], items[2])

    def expr_list(self, items):
        # Filter out COMMA tokens, keep only expressions
        return [item for item in items if not (hasattr(item, 'type') and item.type == 'COMMA')]

    # Statements
    def expr_stmt(self, items):
        return ExprStmt(items[0])

    def assign_stmt(self, items):
        # Grammar: KW_LET? expr (COLON data_type)? ASSIGN expr SEMICOLON
        is_let = any(hasattr(item, 'type') and item.type == 'KW_LET' for item in items)
        
        # Find indices of key tokens
        assign_idx = next(i for i, item in enumerate(items) if hasattr(item, 'type') and item.type == 'ASSIGN')
        
        if is_let:
            lhs = items[1]  # expr after KW_LET
        else:
            lhs = items[0]  # first expr
            
        # RHS is the expression after ASSIGN (before SEMICOLON)
        rhs = items[assign_idx + 1]
        
        # Type annotation is between COLON and ASSIGN if present
        type_annotation = None
        colon_idx = next((i for i, item in enumerate(items) if hasattr(item, 'type') and item.type == 'COLON'), None)
        if colon_idx is not None and colon_idx < assign_idx:
            type_annotation = items[colon_idx + 1]
        
        # Type checking rule: let statements must have explicit type annotations
        if is_let and type_annotation is None:
            raise ValueError("'let' statements require explicit type annotation. Use 'let var: type = value;'")

        # Validate burst operation constraints
        self._validate_burst_operation(lhs, rhs)

        return AssignStmt(is_let, lhs, rhs, type_annotation)

    def return_stmt(self, items):
        # items = [KW_RETURN, expr_list, SEMICOLON]
        expr_list = items[1] if len(items) > 1 else []
        return ReturnStmt(expr_list if isinstance(expr_list, list) else [expr_list])

    def guard_stmt(self, items):
        condition = items[0]
        stmt = items[1]
        return GuardStmt(condition, stmt)

    def do_while_stmt(self, items):
        # Grammar: KW_WITH with_binding* KW_DO body KW_WHILE expr SEMICOLON
        bindings = []
        
        # Find KW_DO to separate bindings from body
        do_idx = next(i for i, item in enumerate(items) if hasattr(item, 'type') and item.type == 'KW_DO')
        while_idx = next(i for i, item in enumerate(items) if hasattr(item, 'type') and item.type == 'KW_WHILE')
        
        # Extract bindings (between KW_WITH and KW_DO)
        for i in range(1, do_idx):  # Skip KW_WITH
            if isinstance(items[i], WithBinding):
                bindings.append(items[i])
        
        # Extract body (between KW_DO and KW_WHILE)
        body = items[do_idx + 1]  # Should be the transformed body
        
        # Extract condition (between KW_WHILE and SEMICOLON)
        condition = items[while_idx + 1]
        
        return DoWhileStmt(bindings, body, condition)

    def directive_stmt(self, items):
        # Grammar: LBRACKET_BRACKET IDENTIFIER (LPAREN expr RPAREN)? RBRACKET_BRACKET
        # items[0] = [[, items[1] = IDENTIFIER, items[2] = (, items[3] = expr, items[4] = ), items[5] = ]]
        name = str(items[1])  # IDENTIFIER is at index 1
        expr = items[3] if len(items) > 4 else None  # expr is at index 3 if present (after LPAREN)
        return DirectiveStmt(name, expr)

    def spawn_stmt(self, items):
        return SpawnStmt(items)

    def static_stmt(self, items):
        return StaticStmt(items[0])

    def thread_stmt(self, items):
        return ThreadStmt(items[0])

    # Body
    def empty_body(self, items):
        return None

    def block_body(self, items):
        # Filter out LBRACE and RBRACE tokens, return only statements
        return [item for item in items if isinstance(item, Stmt)]

    # Static and thread definitions
    def static(self, items):
        # Expected structure: attribute* KW_STATIC IDENTIFIER COLON data_type (ASSIGN expr)? SEMICOLON
        # Extract attributes from beginning
        attrs = []
        idx = 0
        while idx < len(items) and isinstance(items[idx], tuple):
            attrs.append(items[idx])
            idx += 1

        # Now parse the rest: KW_STATIC IDENTIFIER COLON data_type ...
        # items[idx] is KW_STATIC
        name = str(items[idx + 1])  # IDENTIFIER token
        type_info = items[idx + 3]  # data_type (already transformed)
        expr = items[idx + 5] if len(items) > idx + 5 and items[idx + 5] is not None else None  # expr (already transformed)

        # Convert attributes to dict
        attr_dict = dict(attrs) if attrs else {}

        return Static(name, type_info, expr, attr_dict)

    # Flow definition
    def default_flow(self, items):
        attrs = []
        idx = 0
        
        # Extract attributes
        while idx < len(items) and isinstance(items[idx], tuple):
            attrs.append(items[idx])
            idx += 1
        
        # Skip KW_FLOW token
        name_idx = idx + 1
        
        name = str(items[name_idx])
        
        # Find the inputs and body by looking for the right types
        inputs = []
        body = None
        
        for i in range(name_idx + 1, len(items)):
            item = items[i]
            if isinstance(item, list) and len(item) > 0 and all(isinstance(arg, FnArg) for arg in item):
                # This is the fn_arg_list
                inputs = item
            elif item is None:
                # This is an empty body
                body = item
                break
            elif isinstance(item, list):
                # This could be the body (list of statements)
                body = item
                break
        
        flow_attrs = FlowAttributes.from_tuples(attrs)
        input_pairs = [(arg.id, arg.ty.to_basic()) for arg in inputs]
        
        return Flow(name, FlowKind.DEFAULT, input_pairs, flow_attrs, body)

    def rtype_flow(self, items):
        attrs = []
        idx = 0
        
        # Extract attributes
        while idx < len(items) and isinstance(items[idx], tuple):
            attrs.append(items[idx])
            idx += 1
        
        # Skip KW_RTYPE token
        name_idx = idx + 1
        
        name = str(items[name_idx])
        
        # Find the inputs and body by looking for the right types
        inputs = []
        body = None
        
        for i in range(name_idx + 1, len(items)):
            item = items[i]
            if isinstance(item, list) and len(item) > 0 and all(isinstance(arg, FnArg) for arg in item):
                # This is the fn_arg_list
                inputs = item
            elif item is None:
                # This is an empty body
                body = item
                break
            elif isinstance(item, list):
                # This could be the body (list of statements)
                body = item
                break
        
        flow_attrs = FlowAttributes.from_tuples(attrs)
        input_pairs = [(arg.id, arg.ty.to_basic()) for arg in inputs]
        
        return Flow(name, FlowKind.RTYPE, input_pairs, flow_attrs, body)

    # Regfile definition
    def regfile(self, items):
        name = str(items[1])      # Skip KW_REGFILE
        width = int(str(items[3])) # Skip LPAREN  
        depth = int(str(items[5])) # Skip COMMA
        return Regfile(name, width, depth)


    # Processor parts
    def proc_part(self, items):
        part = items[0]
        if isinstance(part, Regfile):
            return RegfilePart(part)
        elif isinstance(part, Flow):
            return FlowPart(part)
        elif isinstance(part, Static):
            return StaticPart(part)
        return part

    # Main processor
    def proc(self, items):
        return Proc.from_parts(items)

    def simple_attr(self, items):
        # items = [HASH, LBRACKET, IDENTIFIER, RBRACKET]
        attr_name = items[2].value  # IDENTIFIER value
        return (attr_name, None)

    def param_attr(self, items):
        # items = [HASH, LBRACKET, IDENTIFIER, LPAREN, attr_expr, RPAREN, RBRACKET]
        attr_name = items[2].value  # IDENTIFIER value
        attr_expr = items[4]  # attr_expr (could be expr or array_literal)
        return (attr_name, attr_expr)

    def array_literal(self, items):
        # Grammar: LBRACKET attr_expr_list RBRACKET
        # items[0] = LBRACKET token
        # items[1] = attr_expr_list (list of expressions)
        # items[2] = RBRACKET token
        expr_list = items[1] if len(items) > 1 else []
        return ArrayLiteralExpr(expr_list if isinstance(expr_list, list) else [expr_list])

    def attr_expr_list(self, items):
        # Filter out COMMA tokens, keep only expressions
        return [item for item in items if not (hasattr(item, 'type') and item.type == 'COMMA')]

    def start(self, items):
        return items[0]


class CADLParser:
    """Main CADL parser class"""

    def __init__(self):
        """Initialize the parser with Lark grammar"""
        grammar_path = Path(__file__).parent / "grammar.lark"
        with open(grammar_path, 'r') as f:
            grammar = f.read()
        
        self.parser = Lark(
            grammar,
            parser='lalr',  # Using LALR parser for better performance with transformers
            start='start'
        )
        self.transformer = CADLTransformer()

    def parse(self, source: str, filename: Optional[str] = None) -> Proc:
        """Parse CADL source code into AST"""
        try:
            parse_tree = self.parser.parse(source)
            result = self.transformer.transform(parse_tree)
            return result
        except (UnexpectedToken, UnexpectedCharacters, ParseError) as e:
            # Convert Lark errors to pretty CADL errors (no chaining to hide traceback)
            raise format_lark_error(e, source, filename)
        except Exception as e:
            # Handle transformer errors and other issues
            raise CADLParseError(f"Internal error: {e}", 1, 1, filename, source.splitlines())


# Global parser instance
_parser = None


def get_parser() -> CADLParser:
    """Get or create global parser instance"""
    global _parser
    if _parser is None:
        _parser = CADLParser()
    return _parser


def parse_proc(source: str, filename: Optional[str] = None) -> Proc:
    """Parse a CADL processor from source code
    
    Args:
        source: CADL source code string
        filename: Optional filename for error reporting
        
    Returns:
        Proc: Parsed processor AST
        
    Raises:
        CADLParseError: On parse errors (with pretty formatting)
    """
    parser = get_parser()
    try:
        return parser.parse(source, filename)
    except CADLParseError:
        # Re-raise our pretty errors as-is
        raise
    except Exception as e:
        # Wrap any other errors
        raise CADLParseError(f"Unexpected error: {e}", 1, 1, filename, source.splitlines())