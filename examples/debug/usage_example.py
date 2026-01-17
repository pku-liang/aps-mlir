#!/usr/bin/env python3
"""
Example usage of the CADL frontend parser
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cadl_frontend import parse_proc
from cadl_frontend.ast import *


def main():
    """Main example function"""
    # Load example CADL file
    example_file = Path(__file__).parent / "simple.cadl"
    
    if not example_file.exists():
        print("Example file not found!")
        return
    
    # Read and parse the CADL source
    with open(example_file, 'r') as f:
        source = f.read()
    
    print("Parsing CADL source:")
    print("-" * 40)
    print(source)
    print("-" * 40)
    
    try:
        # Parse the source code
        ast = parse_proc(source, str(example_file))
        
        print("\nParsing successful!")
        print(f"Found {len(ast.regfiles)} regfiles")
        print(f"Found {len(ast.flows)} flows")
        print(f"Found {len(ast.functions)} functions") 
        print(f"Found {len(ast.statics)} static variables")
        
        # Print details about each component
        if ast.regfiles:
            print("\nRegfiles:")
            for name, regfile in ast.regfiles.items():
                print(f"  {name}: {regfile.width}x{regfile.depth}")
        
        if ast.flows:
            print("\nFlows:")
            for name, flow in ast.flows.items():
                print(f"  {name} ({flow.kind.value}): {len(flow.inputs)} inputs")
        
        if ast.functions:
            print("\nFunctions:")
            for name, function in ast.functions.items():
                print(f"  {name}: {len(function.args)} args -> {len(function.ret)} returns")
        
        if ast.statics:
            print("\nStatic variables:")
            for name, static in ast.statics.items():
                print(f"  {name}: {type(static.ty).__name__}")
                
    except Exception as e:
        print(f"\nParse error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())