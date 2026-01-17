# Syntax Error Test Cases

This directory contains CADL files with various syntax errors for testing error handling in the parser.

## Test Files

1. **`missing_semicolon.cadl`** - Missing semicolons in various contexts
2. **`mismatched_brackets.cadl`** - Unmatched brackets, braces, and parentheses
3. **`invalid_identifiers.cadl`** - Invalid identifier names and keyword usage
4. **`malformed_literals.cadl`** - Invalid number literal formats
5. **`invalid_syntax.cadl`** - General syntax structure errors
6. **`unterminated_strings.cadl`** - Unterminated strings and comments
7. **`invalid_types.cadl`** - Invalid type declarations
8. **`malformed_expressions.cadl`** - Incomplete or malformed expressions
9. **`invalid_attributes.cadl`** - Malformed attribute syntax
10. **`invalid_statements.cadl`** - Invalid statement structures
11. **`mixed_errors.cadl`** - Multiple error types in one file
12. **`empty_constructs.cadl`** - Empty or incomplete language constructs

## Usage

These files are designed to fail parsing and can be used to test:
- Error message quality
- Error recovery mechanisms
- Parser robustness
- Error reporting accuracy

Try parsing them with:
```bash
pixi run parse examples/wont_pass/<filename>.cadl
```

Each should produce helpful error messages indicating what went wrong and where.