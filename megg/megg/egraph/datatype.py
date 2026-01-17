"""
Minimal data types for MLIR to egglog term conversion.
"""
from typing import Union
from dataclasses import dataclass
from enum import Enum, auto


@dataclass(frozen=True)
class DataType:
    """Base class for data types."""
    pass


@dataclass(frozen=True)
class IntType(DataType):
    """
    Integer type with configurable width and signedness.
    """
    width: int
    signed: bool = False
    
    def __post_init__(self):
        if self.width <= 0:
            raise ValueError(f"Integer width must be positive, got {self.width}")
    
    def __str__(self) -> str:
        prefix = "i" if self.signed else "u"
        return f"{prefix}{self.width}"
    
@dataclass(frozen=True)
class IndexType(DataType):
    """MLIR Index type, used for memref indices and loops."""

    def __str__(self):
        return "index"



@dataclass(frozen=True)
class FloatType(DataType):
    """
    Floating point type with standard precisions.
    """
    class Precision(Enum):
        HALF = auto()    # 16-bit
        SINGLE = auto()  # 32-bit
        DOUBLE = auto()  # 64-bit
    
    precision: Precision
    
    def __str__(self) -> str:
        if self.precision == FloatType.Precision.HALF:
            return "f16"
        elif self.precision == FloatType.Precision.SINGLE:
            return "f32"
        elif self.precision == FloatType.Precision.DOUBLE:
            return "f64"
        else:
            return f"float<{self.precision}>"

    @classmethod
    def float_type(cls, width: Union[str, int]):
        if isinstance(width, str):
            width = width.lower()
            if width in ("half", "f16"):
                return cls(cls.Precision.HALF)  
            elif width in ("f32", "float"):
                return cls(cls.Precision.SINGLE)
            elif width in ("double", "f64"):
                return cls(cls.Precision.DOUBLE)
            else:
                raise ValueError(f"Unsupported float type string: {width}")
        
        elif isinstance(width, int):
            if width == 16:
                return cls(cls.Precision.HALF)
            elif width == 32:
                return cls(cls.Precision.SINGLE)
            elif width == 64:
                return cls(cls.Precision.DOUBLE)
            else:
                raise ValueError(f"Unsupported float width: {width} bits. Supported: 16, 32, 64")

@dataclass(frozen=True)
class TupleType(DataType):
    """Tuple/product type for multiple values."""
    element_types: tuple[DataType, ...]

    def __str__(self):
        elements = ", ".join(str(t) for t in self.element_types)
        return f"({elements})"


@dataclass(frozen=True)
class VectorType(DataType):
    """Vector type with element type and size."""
    element_type: DataType
    size: int

    def __str__(self):
        return f"vector<{self.size}x{self.element_type}>"


@dataclass(frozen=True)
class TensorType(DataType):
    """Tensor type with shape and element type."""
    shape: tuple[int, ...]
    element_type: DataType

    def __str__(self):
        shape_str = "x".join(str(d) for d in self.shape)
        return f"tensor<{shape_str}x{self.element_type}>"


@dataclass(frozen=True)
class MemRefType(DataType):
    """MemRef type for memory references."""
    shape: tuple[int, ...]
    element_type: DataType

    def __str__(self):
        shape_str = "x".join(str(d) for d in self.shape)
        return f"memref<{shape_str}x{self.element_type}>"


@dataclass(frozen=True)
class VoidType(DataType):
    """Void type for operations with no return value."""

    def __str__(self):
        return "void"