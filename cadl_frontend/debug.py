"""
Debug infrastructure for CADL frontend.

Provides configurable debug printing with log levels that can be controlled
globally or per-module.

Usage:
    from cadl_frontend.debug import dbg_print, set_debug_level, DebugLevel

    # Set global debug level
    set_debug_level(DebugLevel.INFO)

    # Use in code
    dbg_print("Starting parser", level=DebugLevel.DEBUG)
    dbg_print("Parsed 10 flows", level=DebugLevel.INFO)
    dbg_print("Warning: deprecated syntax", level=DebugLevel.WARNING)
    dbg_print("Failed to parse", level=DebugLevel.ERROR)
"""

import sys
from enum import IntEnum
from typing import Optional, TextIO, Any
from functools import wraps


class DebugLevel(IntEnum):
    """Debug levels ordered by severity."""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    OFF = 100  # Disable all debug output


class DebugConfig:
    """Global debug configuration."""
    def __init__(self):
        self.level = DebugLevel.WARNING  # Default: only warnings and above
        self.output = sys.stderr  # Default output stream
        self.show_module = True  # Show module name in output
        self.show_level = True  # Show level name in output
        self.color_enabled = True  # Enable colored output
        self.module_levels = {}  # Per-module debug levels

    def set_level(self, level: DebugLevel) -> None:
        """Set global debug level."""
        self.level = level

    def set_module_level(self, module_name: str, level: DebugLevel) -> None:
        """Set debug level for specific module."""
        self.module_levels[module_name] = level

    def get_effective_level(self, module_name: Optional[str] = None) -> DebugLevel:
        """Get effective debug level for module (or global)."""
        if module_name and module_name in self.module_levels:
            return self.module_levels[module_name]
        return self.level

    def set_output(self, output: TextIO) -> None:
        """Set output stream for debug messages."""
        self.output = output

    def enable_colors(self, enabled: bool = True) -> None:
        """Enable or disable colored output."""
        self.color_enabled = enabled


# Global configuration instance
_config = DebugConfig()


# ANSI color codes
class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright foreground colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"


# Level-specific colors
LEVEL_COLORS = {
    DebugLevel.DEBUG: Colors.BRIGHT_BLACK,
    DebugLevel.INFO: Colors.BRIGHT_BLUE,
    DebugLevel.WARNING: Colors.BRIGHT_YELLOW,
    DebugLevel.ERROR: Colors.BRIGHT_RED,
    DebugLevel.CRITICAL: Colors.RED + Colors.BOLD,
}

LEVEL_NAMES = {
    DebugLevel.DEBUG: "DEBUG",
    DebugLevel.INFO: "INFO",
    DebugLevel.WARNING: "WARNING",
    DebugLevel.ERROR: "ERROR",
    DebugLevel.CRITICAL: "CRITICAL",
}


def dbg_print(
    *args: Any,
    level: DebugLevel = DebugLevel.DEBUG,
    module: Optional[str] = None,
    sep: str = " ",
    end: str = "\n",
    **kwargs
) -> None:
    """
    Print debug message if current debug level permits.

    Args:
        *args: Values to print
        level: Debug level of this message
        module: Module name (for per-module filtering)
        sep: Separator between arguments
        end: String appended after the last value
        **kwargs: Additional keyword arguments passed to print()
    """
    effective_level = _config.get_effective_level(module)

    # Check if we should print this message
    if level < effective_level:
        return

    # Build prefix
    prefix_parts = []

    if _config.color_enabled:
        color = LEVEL_COLORS.get(level, "")
        prefix_parts.append(color)

    if _config.show_level:
        level_name = LEVEL_NAMES.get(level, f"LEVEL{level}")
        prefix_parts.append(f"[{level_name}]")

    if _config.show_module and module:
        prefix_parts.append(f"[{module}]")

    if prefix_parts:
        prefix = " ".join(prefix_parts) + ": "
    else:
        prefix = ""

    # Format message
    message = sep.join(str(arg) for arg in args)

    # Add color reset if needed
    if _config.color_enabled:
        suffix = Colors.RESET
    else:
        suffix = ""

    # Print to configured output
    print(prefix + message + suffix, end=end, file=_config.output, **kwargs)


def set_debug_level(level: DebugLevel) -> None:
    """Set global debug level."""
    _config.set_level(level)


def set_module_debug_level(module_name: str, level: DebugLevel) -> None:
    """Set debug level for specific module."""
    _config.set_module_level(module_name, level)


def set_debug_output(output: TextIO) -> None:
    """Set output stream for debug messages."""
    _config.set_output(output)


def enable_debug_colors(enabled: bool = True) -> None:
    """Enable or disable colored output."""
    _config.enable_colors(enabled)


def get_debug_level() -> DebugLevel:
    """Get current global debug level."""
    return _config.level


def debug_function(level: DebugLevel = DebugLevel.DEBUG):
    """
    Decorator to add entry/exit debug logging to functions.

    Usage:
        @debug_function(level=DebugLevel.DEBUG)
        def my_function(x, y):
            return x + y
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            module = func.__module__
            func_name = func.__qualname__

            dbg_print(
                f"→ Entering {func_name}",
                level=level,
                module=module
            )

            try:
                result = func(*args, **kwargs)
                dbg_print(
                    f"← Exiting {func_name}",
                    level=level,
                    module=module
                )
                return result
            except Exception as e:
                dbg_print(
                    f"✗ Exception in {func_name}: {e}",
                    level=DebugLevel.ERROR,
                    module=module
                )
                raise

        return wrapper
    return decorator


# Convenience functions for specific levels
def dbg_debug(*args, module: Optional[str] = None, **kwargs) -> None:
    """Print debug message at DEBUG level."""
    dbg_print(*args, level=DebugLevel.DEBUG, module=module, **kwargs)


def dbg_info(*args, module: Optional[str] = None, **kwargs) -> None:
    """Print debug message at INFO level."""
    dbg_print(*args, level=DebugLevel.INFO, module=module, **kwargs)


def dbg_warning(*args, module: Optional[str] = None, **kwargs) -> None:
    """Print debug message at WARNING level."""
    dbg_print(*args, level=DebugLevel.WARNING, module=module, **kwargs)


def dbg_error(*args, module: Optional[str] = None, **kwargs) -> None:
    """Print debug message at ERROR level."""
    dbg_print(*args, level=DebugLevel.ERROR, module=module, **kwargs)


def dbg_critical(*args, module: Optional[str] = None, **kwargs) -> None:
    """Print debug message at CRITICAL level."""
    dbg_print(*args, level=DebugLevel.CRITICAL, module=module, **kwargs)


# Context manager for temporary debug level changes
class debug_level_context:
    """
    Context manager to temporarily change debug level.

    Usage:
        with debug_level_context(DebugLevel.DEBUG):
            # Debug output enabled for this block
            dbg_print("Detailed debug info")
    """
    def __init__(self, level: DebugLevel, module: Optional[str] = None):
        self.level = level
        self.module = module
        self.old_level = None

    def __enter__(self):
        if self.module:
            self.old_level = _config.module_levels.get(self.module)
            _config.set_module_level(self.module, self.level)
        else:
            self.old_level = _config.level
            _config.set_level(self.level)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _ = (exc_type, exc_val, exc_tb)  # Unused but required by context manager protocol
        if self.module:
            if self.old_level is None:
                _config.module_levels.pop(self.module, None)
            else:
                _config.set_module_level(self.module, self.old_level)
        else:
            _config.set_level(self.old_level)
        return False
