"""
ASP-DAC Tutorial: Megg Compiler Visualization
Interactive demonstration of CADL → Custom Instruction compilation
"""

import streamlit as st
import json
from pathlib import Path
import sys
import re

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "megg"))

st.set_page_config(
    page_title="Megg Compiler Tutorial",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import visualization components
from components import (
    render_pipeline_diagram,
    render_egraph_growth,
    render_skeleton_tree,
    render_matching_animation,
    render_rewrite_rules_diagram,
    render_external_passes_diagram
)

# Custom CSS for better code display
st.markdown("""
<style>
    .stCode { font-size: 14px; }
    .step-header {
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        padding: 10px 20px;
        border-radius: 5px;
        color: white;
        margin-bottom: 20px;
    }
    .highlight-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def load_example_data(example_name: str) -> dict:
    """Load all files for an example."""
    example_dir = PROJECT_ROOT / "examples" / "diff_match" / example_name

    data = {"name": example_name}

    # Load files if they exist
    file_mappings = {
        "cadl": f"{example_name}.cadl",
        "c_code": f"{example_name}.c",
        "mlir_pattern": f"{example_name}.mlir",
        "test_c": f"test_{example_name}.c",
        "stats": f"{example_name}.stats.json",
        "encoding": f"{example_name}.json",
        "asm": f"{example_name}.asm",
    }

    for key, filename in file_mappings.items():
        filepath = example_dir / filename
        if filepath.exists():
            if filename.endswith('.json'):
                with open(filepath) as f:
                    data[key] = json.load(f)
            else:
                with open(filepath) as f:
                    data[key] = f.read()

    return data


def get_available_examples() -> list:
    """Get list of available examples."""
    examples_dir = PROJECT_ROOT / "examples" / "diff_match"
    examples = []
    for d in examples_dir.iterdir():
        if d.is_dir() and (d / f"{d.name}.cadl").exists():
            examples.append(d.name)
    return sorted(examples)


def render_step1_cadl_to_c(data: dict):
    """Step 1: CADL to C conversion visualization."""
    st.markdown("### Step 1: CADL → C Conversion")
    st.markdown("""
    **CADL (Computer Architecture Description Language)** describes custom instructions
    with hardware semantics. The `cadl2c` tool converts this to equivalent C code for
    software simulation and pattern matching.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📝 CADL Source")
        if "cadl" in data:
            st.code(data["cadl"], language="rust")
        else:
            st.warning("CADL file not found")

    with col2:
        st.markdown("#### 🔄 Generated C Code")
        if "c_code" in data:
            st.code(data["c_code"], language="c")
        else:
            st.warning("C file not found")

    # Highlight key transformations
    with st.expander("🔍 Key Transformations", expanded=True):
        st.markdown("""
        | CADL Construct | C Equivalent |
        |----------------|--------------|
        | `with i: u32 = (0, i_) do {...} while (cond)` | `for (i = 0; cond; i = i_) {...}` |
        | `[[unroll(N)]]` | Loop body replicated N times |
        | `matrix[i * 4 + j]` | Array indexing (preserved) |
        | `_burst_read[addr +: N]` | Memory load operations |
        | `_irf[rs1]` | Register file access |
        """)


def render_step2_hybrid_rewrite(data: dict):
    """Step 2: Hybrid rewrite visualization."""
    st.markdown("### Step 2: Hybrid Rewrite (E-Graph + MLIR)")

    st.markdown("""
    The Megg compiler uses a **hybrid approach**:
    - **Internal Rewrites**: Algebraic rules in e-graph (commutativity, associativity, etc.)
    - **External Rewrites**: MLIR passes (loop-unroll, loop-tile, etc.)
    """)

    # Show statistics
    if "stats" in data:
        stats = data["stats"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Internal Rewrites", stats.get("internal_rewrites", 0))
        with col2:
            external = stats.get("external_rewrites", [])
            st.metric("External Passes", len(external))
        with col3:
            st.metric("Custom Matches", stats.get("custom_rewrites", 0))
        with col4:
            st.metric("Time (s)", f"{stats.get('time_elapsed', 0):.2f}")

        # E-graph growth visualization
        st.markdown("#### E-Graph Growth")
        initial = stats.get("initial_egraph_statistics", {})
        final = stats.get("egraph_statistics", {})
        render_egraph_growth(initial, final)

        # Show rewrite rules
        st.markdown("#### Rewrite Rules")
        render_rewrite_rules_diagram()

        # Show external passes applied
        render_external_passes_diagram(external)


def render_step3_skeleton(data: dict):
    """Step 3: Skeleton extraction visualization."""
    st.markdown("### Step 3: MLIR → Skeleton & Components")

    st.markdown("""
    The pattern MLIR is decomposed into:
    - **Skeleton**: Control flow structure (loops, conditions)
    - **Components**: Computation patterns within each block
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📐 Pattern MLIR")
        if "mlir_pattern" in data:
            st.code(data["mlir_pattern"], language="mlir")
        else:
            st.warning("Pattern MLIR not found")

    with col2:
        st.markdown("#### 🦴 Extracted Skeleton")
        if "mlir_pattern" in data:
            render_skeleton_tree(data["mlir_pattern"], data["name"])
        else:
            st.warning("Cannot extract skeleton")

    with st.expander("🔍 Skeleton Structure Explanation", expanded=True):
        st.markdown("""
        **Skeleton** captures the loop structure:
        ```python
        ForSkeleton(
            lower=0, upper=2, step=1,    # Outer loop bounds
            body=ForSkeleton(            # Nested inner loop
                lower=0, upper=4, step=1,
                body=Component(...)      # Actual computation
            )
        )
        ```

        **Component** captures the computation pattern:
        - Load operations: `memref.load %arg0[%idx]`
        - Arithmetic: `arith.muli`, `arith.addi`
        - Store operations: `memref.store`
        - Iter args for reductions
        """)


def generate_skeleton_repr(data: dict) -> str:
    """Generate a skeleton representation from the MLIR pattern."""
    if "mlir_pattern" not in data:
        return "# No pattern available"

    mlir = data["mlir_pattern"]

    # Simple parsing to extract structure
    lines = []
    lines.append(f"# Skeleton for {data['name']}")
    lines.append("")

    # Count loops
    for_count = mlir.count("scf.for")
    if_count = mlir.count("scf.if")

    lines.append(f"structure:")
    lines.append(f"  loops: {for_count}")
    lines.append(f"  conditionals: {if_count}")
    lines.append("")

    # Extract loop bounds from MLIR
    import re
    for_matches = re.findall(r'scf\.for %\w+ = (%\w+) to (%\w+) step (%\w+)', mlir)

    lines.append("loop_structure:")
    for i, (lower, upper, step) in enumerate(for_matches):
        lines.append(f"  loop_{i}:")
        lines.append(f"    lower: {lower}")
        lines.append(f"    upper: {upper}")
        lines.append(f"    step: {step}")

    lines.append("")
    lines.append("components:")

    # Extract key operations
    if "memref.load" in mlir:
        load_count = mlir.count("memref.load")
        lines.append(f"  - load_ops: {load_count}")
    if "memref.store" in mlir:
        store_count = mlir.count("memref.store")
        lines.append(f"  - store_ops: {store_count}")
    if "arith.muli" in mlir:
        lines.append(f"  - multiply: arith.muli")
    if "arith.addi" in mlir:
        lines.append(f"  - add: arith.addi")
    if "iter_args" in mlir:
        lines.append(f"  - reduction: iter_args (accumulator)")

    return "\n".join(lines)


def render_step4_matching(data: dict):
    """Step 4: Matching process visualization."""
    st.markdown("### Step 4: Pattern Matching & Replacement")

    st.markdown("""
    The e-graph matching process:
    1. **Normalize** source MLIR to canonical form
    2. **Extract skeleton** from pattern and source
    3. **Match** components using e-graph equivalence
    4. **Replace** matched subgraph with `custom_instr`
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📥 Input C Code (to be optimized)")
        if "test_c" in data:
            # Extract just the target function
            test_c = data["test_c"]
            # Find the pragma-marked function
            lines = test_c.split('\n')
            in_func = False
            func_lines = []
            for line in lines:
                if '#pragma megg optimize' in line or in_func:
                    in_func = True
                    func_lines.append(line)
                    if in_func and line.strip() == '}' and len(func_lines) > 5:
                        break
            if func_lines:
                st.code('\n'.join(func_lines), language="c")
            else:
                st.code(test_c[:2000] + "...", language="c")
        else:
            st.warning("Test C file not found")

    with col2:
        st.markdown("#### 📤 Output (with custom instruction)")
        if "asm" in data:
            # Extract relevant assembly
            asm = data["asm"]
            # Find the function
            func_name = data["name"]
            lines = asm.split('\n')
            in_func = False
            func_lines = []
            for line in lines:
                if f'<{func_name}' in line:
                    in_func = True
                if in_func:
                    func_lines.append(line)
                    if line.strip().startswith('ret') or (in_func and len(func_lines) > 30):
                        break
            if func_lines:
                st.code('\n'.join(func_lines[:40]), language="asm")
            else:
                st.code("# Assembly not found for this function", language="asm")
        else:
            st.warning("Assembly file not found")

    # Show encoding information
    if "encoding" in data:
        st.markdown("#### 🔢 Custom Instruction Encoding")
        enc = data["encoding"]

        for instr_name, encoding in enc.items():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.code(f"opcode: {encoding.get('opcode', 'N/A')}", language="yaml")
            with col2:
                st.code(f"funct3: {encoding.get('funct3', '0x0')}", language="yaml")
            with col3:
                st.code(f"funct7: {encoding.get('funct7', 'N/A')}", language="yaml")

    # Matching visualization
    if "stats" in data and "mlir_pattern" in data:
        stats = data["stats"]
        matches = stats.get("custom_rewrites", 0)
        mlir = data["mlir_pattern"]

        # Count operations in pattern
        ops_count = sum([
            mlir.count("arith."),
            mlir.count("memref."),
            mlir.count("scf."),
            mlir.count("index_cast")
        ])

        render_matching_animation(
            source_ops=ops_count,
            matched_ops=matches,
            custom_instr=f"custom_instr(\"{data['name']}_vv\", %arg0, %arg1)"
        )

    with st.expander("🎯 Matching Process Details", expanded=True):
        if "stats" in data:
            stats = data["stats"]
            matches = stats.get("custom_rewrites", 0)

            st.markdown(f"""
            **Matching Summary:**
            - Total custom patterns matched: **{matches}**
            - Each match replaces a complex computation with a single instruction

            **Process:**
            ```
            Source MLIR          Pattern MLIR
                ↓                     ↓
            ┌─────────┐         ┌─────────┐
            │ E-Graph │ ←match→ │Skeleton │
            │ (351    │         │+Compon- │
            │  nodes) │         │  ents   │
            └─────────┘         └─────────┘
                     ↓
               custom_instr({data['name']}_vv, %arg0, %arg1)
            ```
            """)


def render_overview():
    """Render the overview page."""
    st.markdown("## 🔧 Megg Compiler: Custom Instruction Synthesis")

    st.markdown("""
    This tutorial demonstrates how the **Megg compiler** automatically matches
    C code patterns to custom RISC-V instructions defined in **CADL**.
    """)

    # Pipeline diagram (visual)
    st.markdown("### Compilation Pipeline")
    render_pipeline_diagram()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        **Step 1: CADL → C**

        Convert hardware description to C semantics
        """)

    with col2:
        st.markdown("""
        **Step 2: Hybrid Rewrite**

        Apply internal (e-graph) + external (MLIR) optimizations
        """)

    with col3:
        st.markdown("""
        **Step 3: Skeleton**

        Extract control flow structure for matching
        """)

    with col4:
        st.markdown("""
        **Step 4: Matching**

        Find and replace with custom instructions
        """)


def main():
    st.title("🎓 ASP-DAC Tutorial: Megg Compiler")
    st.markdown("*Interactive Custom Instruction Synthesis Demo*")

    # Sidebar
    with st.sidebar:
        st.markdown("## Navigation")

        # Example selector
        examples = get_available_examples()
        selected_example = st.selectbox(
            "Select Example",
            examples,
            index=examples.index("vgemv3d") if "vgemv3d" in examples else 0
        )

        st.markdown("---")

        # Step selector
        step = st.radio(
            "Tutorial Step",
            ["Overview", "Step 1: CADL → C", "Step 2: Hybrid Rewrite",
             "Step 3: Skeleton", "Step 4: Matching"],
            index=0
        )

        st.markdown("---")
        st.markdown(f"**Example:** `{selected_example}`")

        # Quick stats
        data = load_example_data(selected_example)
        if "stats" in data:
            stats = data["stats"]
            st.markdown("**Quick Stats:**")
            st.markdown(f"- Rewrites: {stats.get('internal_rewrites', 0)}")
            st.markdown(f"- Matches: {stats.get('custom_rewrites', 0)}")

    # Main content
    data = load_example_data(selected_example)

    if step == "Overview":
        render_overview()
    elif step == "Step 1: CADL → C":
        render_step1_cadl_to_c(data)
    elif step == "Step 2: Hybrid Rewrite":
        render_step2_hybrid_rewrite(data)
    elif step == "Step 3: Skeleton":
        render_step3_skeleton(data)
    elif step == "Step 4: Matching":
        render_step4_matching(data)

    # Footer
    st.markdown("---")
    st.markdown("*ASP-DAC 2025 Tutorial - Megg: E-Graph Based Custom Instruction Synthesis*")


if __name__ == "__main__":
    main()
