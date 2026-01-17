#!/usr/bin/env python3
"""
Example usage of the debug infrastructure.

This demonstrates various ways to use the debug printing system
with different log levels and configurations.
"""

from cadl_frontend.debug import (
    dbg_print,
    dbg_debug,
    dbg_info,
    dbg_warning,
    dbg_error,
    dbg_critical,
    set_debug_level,
    set_module_debug_level,
    enable_debug_colors,
    DebugLevel,
    debug_function,
    debug_level_context,
)


def example_basic_usage():
    """Example 1: Basic debug printing with different levels."""
    print("\n=== Example 1: Basic Debug Printing ===\n")

    # Set global level to INFO (will show INFO, WARNING, ERROR, CRITICAL)
    set_debug_level(DebugLevel.INFO)

    dbg_debug("This debug message won't be shown")
    dbg_info("Starting parser initialization")
    dbg_warning("Deprecated syntax detected")
    dbg_error("Failed to parse expression")
    dbg_critical("Fatal error: Cannot continue")


def example_module_specific_levels():
    """Example 2: Per-module debug levels."""
    print("\n=== Example 2: Module-Specific Debug Levels ===\n")

    # Set global level to WARNING
    set_debug_level(DebugLevel.WARNING)

    # But allow DEBUG for parser module
    set_module_debug_level("parser", DebugLevel.DEBUG)

    # This won't show (global level is WARNING)
    dbg_debug("Generic debug message")

    # This will show (parser module allows DEBUG)
    dbg_debug("Parser: tokenizing input", module="parser")
    dbg_info("Parser: found 42 tokens", module="parser")

    # This won't show (transformer uses global level)
    dbg_debug("Transformer: building AST", module="transformer")

    # This will show (WARNING level)
    dbg_warning("Transformer: deprecated construct", module="transformer")


def example_convenience_functions():
    """Example 3: Using convenience functions."""
    print("\n=== Example 3: Convenience Functions ===\n")

    set_debug_level(DebugLevel.DEBUG)

    # Convenience functions for each level
    dbg_debug("Detailed debugging information")
    dbg_info("Processing file: example.cadl")
    dbg_warning("Line 42: Consider using newer syntax")
    dbg_error("Syntax error at line 100")
    dbg_critical("Out of memory!")


@debug_function(level=DebugLevel.DEBUG)
def parse_cadl_file(filename: str) -> int:
    """Example function with automatic entry/exit logging."""
    dbg_info(f"Parsing file: {filename}", module="parser")

    # Simulate parsing
    token_count = 42

    dbg_info(f"Found {token_count} tokens", module="parser")

    return token_count


def example_function_decorator():
    """Example 4: Function decorator for entry/exit logging."""
    print("\n=== Example 4: Function Decorator ===\n")

    set_debug_level(DebugLevel.DEBUG)

    result = parse_cadl_file("example.cadl")
    print(f"Result: {result} tokens\n")


def example_context_manager():
    """Example 5: Temporary debug level changes."""
    print("\n=== Example 5: Context Manager ===\n")

    set_debug_level(DebugLevel.WARNING)

    dbg_debug("This won't show (level is WARNING)")

    # Temporarily enable DEBUG level
    with debug_level_context(DebugLevel.DEBUG):
        dbg_debug("This shows (temporarily enabled DEBUG)")
        dbg_info("Still inside debug context")

    dbg_debug("This won't show again (back to WARNING)")
    dbg_warning("This will show (WARNING level)")


def example_realistic_parser():
    """Example 6: Realistic parser debugging scenario."""
    print("\n=== Example 6: Realistic Parser Scenario ===\n")

    # Configure debug levels for different modules
    set_debug_level(DebugLevel.INFO)
    set_module_debug_level("parser.lexer", DebugLevel.DEBUG)
    set_module_debug_level("parser.transformer", DebugLevel.WARNING)

    # Lexer produces detailed debug output
    dbg_debug("Lexer: Starting tokenization", module="parser.lexer")
    dbg_debug("Lexer: Token 1: KEYWORD 'flow'", module="parser.lexer")
    dbg_debug("Lexer: Token 2: IDENTIFIER 'main'", module="parser.lexer")
    dbg_info("Lexer: Tokenization complete (42 tokens)", module="parser.lexer")

    # Parser produces info-level output
    dbg_debug("Parser: This debug won't show", module="parser.main")
    dbg_info("Parser: Building parse tree", module="parser.main")
    dbg_info("Parser: Parse tree complete", module="parser.main")

    # Transformer only shows warnings and above
    dbg_debug("Transformer: Debug won't show", module="parser.transformer")
    dbg_info("Transformer: Info won't show", module="parser.transformer")
    dbg_warning("Transformer: Deprecated node type", module="parser.transformer")
    dbg_error("Transformer: Invalid AST node", module="parser.transformer")


def example_custom_formatting():
    """Example 7: Custom formatting options."""
    print("\n=== Example 7: Custom Formatting ===\n")

    set_debug_level(DebugLevel.DEBUG)

    # Multiple arguments with custom separator
    dbg_print("Value1", "Value2", "Value3", level=DebugLevel.INFO, sep=" | ")

    # Inline values
    x, y = 10, 20
    dbg_info("Computing:", x, "+", y, "=", x + y)


def example_color_output():
    """Example 8: Colored output."""
    print("\n=== Example 8: Colored Output ===\n")

    # Enable colors (default is enabled)
    enable_debug_colors(True)
    set_debug_level(DebugLevel.DEBUG)

    dbg_debug("Debug messages appear in dim gray")
    dbg_info("Info messages appear in blue")
    dbg_warning("Warning messages appear in yellow")
    dbg_error("Error messages appear in red")
    dbg_critical("Critical messages appear in bold red")

    # Disable colors
    print("\nWith colors disabled:")
    enable_debug_colors(False)
    dbg_warning("Warning without colors")


def example_error_handling():
    """Example 9: Error handling with debug output."""
    print("\n=== Example 9: Error Handling ===\n")

    set_debug_level(DebugLevel.DEBUG)

    @debug_function(level=DebugLevel.DEBUG)
    def risky_function():
        dbg_info("Attempting risky operation")
        raise ValueError("Something went wrong!")

    try:
        risky_function()
    except ValueError as e:
        dbg_error(f"Caught exception: {e}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("Debug Infrastructure Usage Examples")
    print("=" * 60)

    example_basic_usage()
    example_module_specific_levels()
    example_convenience_functions()
    example_function_decorator()
    example_context_manager()
    example_realistic_parser()
    example_custom_formatting()
    example_color_output()
    example_error_handling()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
