"""
Pragma Parser for Megg Compiler

This module parses #pragma megg optimize directives in C source code
and splits the code into target functions (to be optimized by Megg)
and rest functions (to be compiled by LLVM).
"""

from typing import List, Set, Tuple, Optional
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PragmaParser:
    """Parse #pragma megg optimize directives and split C code."""

    # Pragma pattern to match
    PRAGMA_PATTERN = r'#pragma\s+megg\s+optimize'

    # Function definition pattern (simplified)
    FUNC_PATTERN = r'^\s*(\w+\s+)+(\w+)\s*\([^)]*\)\s*\{'

    def __init__(self, source_file: str):
        """
        Initialize pragma parser.

        Args:
            source_file: Path to C source file
        """
        self.source_file = Path(source_file)
        if not self.source_file.exists():
            raise FileNotFoundError(f"Source file not found: {source_file}")

        self.source_code = self._read_source()
        self.lines = self.source_code.split('\n')

    def _read_source(self) -> str:
        """Read source file content."""
        with open(self.source_file, 'r') as f:
            return f.read()

    def find_marked_functions(self) -> List[Tuple[str, int, int]]:
        """
        Find all functions marked with #pragma megg optimize.

        Returns:
            List of tuples: (function_name, start_line, end_line)
        """
        marked_functions = []

        i = 0
        while i < len(self.lines):
            line = self.lines[i]

            # Skip comments - check if line is a comment
            stripped = line.strip()
            if stripped.startswith('//') or stripped.startswith('*') or stripped.startswith('/*'):
                i += 1
                continue

            # Check if this line contains the pragma
            if re.search(self.PRAGMA_PATTERN, line):
                logger.info(f"Found pragma at line {i+1}: {line.strip()}")

                # Find the next function definition
                func_info = self._find_next_function(i + 1)
                if func_info:
                    func_name, start_line, end_line = func_info
                    marked_functions.append((func_name, start_line, end_line))
                    logger.info(f"  Marked function: {func_name} (lines {start_line+1}-{end_line+1})")
                    i = end_line  # Skip to end of function
                else:
                    logger.warning(f"  No function found after pragma at line {i+1}")

            i += 1

        logger.info(f"Found {len(marked_functions)} marked function(s)")
        return marked_functions

    def _find_next_function(self, start_line: int) -> Optional[Tuple[str, int, int]]:
        """
        Find the next function definition starting from start_line.

        Args:
            start_line: Line number to start searching from

        Returns:
            Tuple of (function_name, start_line, end_line) or None
        """
        # Find function declaration
        func_start = None
        func_name = None

        for i in range(start_line, len(self.lines)):
            line = self.lines[i]

            # Skip empty lines and comments
            stripped = line.strip()
            if not stripped or stripped.startswith('//') or stripped.startswith('/*'):
                continue

            # Check if this line starts a function
            match = re.search(self.FUNC_PATTERN, line)
            if match:
                func_name = match.group(2)
                func_start = i
                break

        if func_start is None:
            return None

        # Find function end by counting braces
        func_end = self._find_function_end(func_start)

        if func_end is None:
            logger.warning(f"Could not find end of function {func_name}")
            return None

        return (func_name, func_start, func_end)

    def _find_function_end(self, start_line: int) -> Optional[int]:
        """
        Find the end of a function by counting braces.

        Args:
            start_line: Line number where function starts

        Returns:
            End line number or None
        """
        brace_count = 0
        in_function = False

        for i in range(start_line, len(self.lines)):
            line = self.lines[i]

            # Count opening and closing braces
            for char in line:
                if char == '{':
                    brace_count += 1
                    in_function = True
                elif char == '}':
                    brace_count -= 1

                    # Function ends when braces balance
                    if in_function and brace_count == 0:
                        return i

        return None

    def split_code(self, marked_functions: List[Tuple[str, int, int]]) -> Tuple[str, str]:
        """
        Split source code into target.c (marked functions) and rest.c (other code).

        Args:
            marked_functions: List of (function_name, start_line, end_line) tuples

        Returns:
            Tuple of (target_code, rest_code)
        """
        if not marked_functions:
            logger.warning("No marked functions found, returning original code in rest.c")
            return "", self.source_code

        # Extract includes and declarations
        includes = self._extract_includes()

        # Build set of marked function line ranges
        marked_ranges = set()
        for _, start, end in marked_functions:
            for line_num in range(start, end + 1):
                marked_ranges.add(line_num)

        # Split code
        target_lines = []
        rest_lines = []

        for i, line in enumerate(self.lines):
            # Skip pragma directives in output
            if re.search(self.PRAGMA_PATTERN, line):
                continue

            if i in marked_ranges:
                target_lines.append(line)
            else:
                rest_lines.append(line)

        # Build target.c with includes
        target_code = self._build_target_code(includes, target_lines, marked_functions)

        # Build rest.c with extern declarations
        rest_code = self._build_rest_code(includes, rest_lines, marked_functions)

        return target_code, rest_code

    def _extract_includes(self) -> List[str]:
        """Extract #include directives from source code."""
        includes = []
        for line in self.lines:
            stripped = line.strip()
            if stripped.startswith('#include'):
                includes.append(line)
        return includes

    def _build_target_code(self, includes: List[str], target_lines: List[str],
                           marked_functions: List[Tuple[str, int, int]]) -> str:
        """
        Build target.c with includes and marked functions.

        Args:
            includes: Include directives
            target_lines: Lines of marked functions
            marked_functions: Function metadata

        Returns:
            Complete target.c content
        """
        parts = []

        # Only add basic includes (stdint.h, stddef.h) - no stdio!
        # Polygeist cannot handle stdio.h and complex system headers
        basic_includes = [inc for inc in includes if 'stdint.h' in inc or 'stddef.h' in inc]

        if not basic_includes:
            # If no basic includes found, add stdint.h by default
            parts.append('#include <stdint.h>')
        else:
            parts.extend(basic_includes)

        parts.append("")

        # Add function code
        parts.extend(target_lines)

        return '\n'.join(parts)

    def _build_rest_code(self, includes: List[str], rest_lines: List[str],
                        marked_functions: List[Tuple[str, int, int]]) -> str:
        """
        Build rest.c with extern declarations for marked functions.

        Args:
            includes: Include directives
            rest_lines: Lines of non-marked code
            marked_functions: Function metadata

        Returns:
            Complete rest.c content
        """
        parts = []

        # Add includes
        if includes:
            parts.extend(includes)
            parts.append("")

        # Add extern declarations for marked functions
        # Note: This is a simplified version that extracts signatures from original code
        parts.append("// External declarations for optimized functions")
        for func_name, start, _ in marked_functions:
            signature = self._extract_function_signature(func_name, start)
            if signature:
                parts.append(f"extern {signature};")
        parts.append("")

        # Add rest of the code
        parts.extend(rest_lines)

        return '\n'.join(parts)

    def _extract_function_signature(self, func_name: str, start_line: int) -> Optional[str]:
        """
        Extract function signature from declaration.

        Args:
            func_name: Function name
            start_line: Line where function starts

        Returns:
            Function signature string (e.g., "int foo(int a, int b)")
        """
        # Simple approach: extract from start line to opening brace
        signature_parts = []

        for i in range(start_line, min(start_line + 10, len(self.lines))):
            line = self.lines[i]
            signature_parts.append(line)

            if '{' in line:
                # Found opening brace, extract signature
                full_text = ' '.join(signature_parts)
                # Remove everything from { onwards
                signature = full_text.split('{')[0].strip()
                return signature

        return None


def test_pragma_parser():
    """Simple test for pragma parser."""
    test_code = """
#include <stdio.h>

#pragma megg optimize
int tricky(int a) {
    if (a > 0) {
        while (a--) {
            printf("%d\\n", a);
        }
    }
    return a;
}

int other() {
    return 0;
}


"""

    # Write test file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
        f.write(test_code)
        test_file = f.name

    try:
        parser = PragmaParser(test_file)
        marked = parser.find_marked_functions()
        print(f"Marked functions: {marked}")

        target_code, rest_code = parser.split_code(marked)
        print("\n=== TARGET.C ===")
        print(target_code)
        print("\n=== REST.C ===")
        print(rest_code)
    finally:
        import os
        os.unlink(test_file)


if __name__ == "__main__":
    test_pragma_parser()
