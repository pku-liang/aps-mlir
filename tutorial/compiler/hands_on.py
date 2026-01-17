"""
Hands-on interactive mode for ASP-DAC Tutorial.
Allows participants to run compilation steps and see intermediate results.
"""

import streamlit as st
import streamlit.components.v1 as components
import subprocess
import tempfile
import json
import re
import html
from pathlib import Path
import shutil
import os
import sys

# tutorial/compiler/ -> need to go up 2 levels to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "megg"))


def get_project_root():
    return PROJECT_ROOT


def render_cadl_code(cadl_code: str, height: int = 400):
    """Render CADL code with syntax highlighting using custom HTML/CSS."""
    import uuid
    placeholders = {}

    def make_placeholder(match, css_class):
        token = f"__PLACEHOLDER_{uuid.uuid4().hex}__"
        content = html.escape(match.group(1))
        placeholders[token] = f'<span class="{css_class}">{content}</span>'
        return token

    highlighted = cadl_code

    # Step 1: Replace comments and strings first
    # Single-line comments
    highlighted = re.sub(r'(//[^\n]*)', lambda m: make_placeholder(m, 'cadl-comment'), highlighted)
    # Strings
    highlighted = re.sub(r'("(?:[^"\\]|\\.)*")', lambda m: make_placeholder(m, 'cadl-string'), highlighted)

    # Step 2: Escape HTML entities
    highlighted = html.escape(highlighted)

    # Step 3: Apply highlighting patterns
    patterns = [
        # Attributes like #[opcode(...)]
        (r'(#\[[^\]]+\])', r'<span class="cadl-attr">\1</span>'),
        # Types
        (r'\b(u5|u8|u16|u32|u64|i8|i16|i32|i64|bool)\b', r'<span class="cadl-type">\1</span>'),
        # Keywords
        (r'\b(static|let|with|do|while|if|else|return|rtype|itype|fn)\b', r'<span class="cadl-keyword">\1</span>'),
        # Special variables
        (r'(_irf|_burst_read|_burst_write|_mem)', r'<span class="cadl-special">\1</span>'),
        # Numbers (binary, hex, decimal)
        (r"\b(\d+'[bhdoBHDO][0-9a-fA-F_]+|\d+)\b", r'<span class="cadl-number">\1</span>'),
    ]

    for pattern, replacement in patterns:
        highlighted = re.sub(pattern, replacement, highlighted)

    # Step 4: Restore placeholders
    for token, replacement in placeholders.items():
        highlighted = highlighted.replace(token, replacement)

    html_content = f'''
    <style>
        .cadl-container {{
            background-color: #f8f8f8;
            color: #333333;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.5;
            padding: 12px;
            border-radius: 6px;
            border: 1px solid #e0e0e0;
            overflow: auto;
            max-height: {height}px;
            white-space: pre;
        }}
        .cadl-keyword {{ color: #0000ff; font-weight: bold; }}
        .cadl-type {{ color: #267f99; }}
        .cadl-attr {{ color: #af00db; }}
        .cadl-special {{ color: #795e26; }}
        .cadl-number {{ color: #098658; }}
        .cadl-string {{ color: #a31515; }}
        .cadl-comment {{ color: #008000; font-style: italic; }}
    </style>
    <div class="cadl-container">{highlighted}</div>
    '''

    components.html(html_content, height=height + 30, scrolling=True)


def render_c_code(c_code: str, height: int = 400):
    """Render C code with syntax highlighting using custom HTML/CSS."""
    # Use placeholder tokens to avoid regex conflicts with inserted HTML
    import uuid
    placeholders = {}

    def make_placeholder(match, css_class):
        token = f"__PLACEHOLDER_{uuid.uuid4().hex}__"
        # Escape the matched content for HTML safety
        content = html.escape(match.group(1))
        placeholders[token] = f'<span class="{css_class}">{content}</span>'
        return token

    highlighted = c_code

    # Step 1: Replace comments and strings first (they can contain other syntax)
    # Single-line comments
    highlighted = re.sub(r'(//[^\n]*)', lambda m: make_placeholder(m, 'c-comment'), highlighted)
    # Multi-line comments
    highlighted = re.sub(r'(/\*[\s\S]*?\*/)', lambda m: make_placeholder(m, 'c-comment'), highlighted)
    # Strings
    highlighted = re.sub(r'("(?:[^"\\]|\\.)*")', lambda m: make_placeholder(m, 'c-string'), highlighted)
    # Character literals
    highlighted = re.sub(r"('(?:[^'\\]|\\.)*')", lambda m: make_placeholder(m, 'c-string'), highlighted)

    # Step 2: Escape HTML entities in the remaining code
    highlighted = html.escape(highlighted)

    # Step 3: Apply other highlighting patterns
    patterns = [
        # Preprocessor directives
        (r'(#\w+)', r'<span class="c-preprocessor">\1</span>'),
        # Types
        (r'\b(void|int|char|short|long|float|double|signed|unsigned|uint8_t|uint16_t|uint32_t|uint64_t|int8_t|int16_t|int32_t|int64_t|size_t|bool)\b',
         r'<span class="c-type">\1</span>'),
        # Keywords
        (r'\b(if|else|for|while|do|switch|case|default|break|continue|return|goto|struct|union|enum|typedef|static|extern|const|volatile|inline|sizeof)\b',
         r'<span class="c-keyword">\1</span>'),
        # Numbers (hex, float, int)
        (r'\b(0x[0-9a-fA-F]+|0b[01]+|\d+\.?\d*(?:[eE][+-]?\d+)?[fFlLuU]*)\b', r'<span class="c-number">\1</span>'),
    ]

    for pattern, replacement in patterns:
        highlighted = re.sub(pattern, replacement, highlighted)

    # Step 4: Restore placeholders
    for token, replacement in placeholders.items():
        highlighted = highlighted.replace(token, replacement)

    html_content = f'''
    <style>
        .c-container {{
            background-color: #f8f8f8;
            color: #333333;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.5;
            padding: 12px;
            border-radius: 6px;
            border: 1px solid #e0e0e0;
            overflow: auto;
            max-height: {height}px;
            white-space: pre;
        }}
        .c-keyword {{ color: #0000ff; font-weight: bold; }}
        .c-type {{ color: #267f99; }}
        .c-preprocessor {{ color: #af00db; }}
        .c-number {{ color: #098658; }}
        .c-string {{ color: #a31515; }}
        .c-comment {{ color: #008000; font-style: italic; }}
    </style>
    <div class="c-container">{highlighted}</div>
    '''

    components.html(html_content, height=height + 30, scrolling=True)


def render_mlir_code(mlir_code: str, height: int = 400):
    """Render MLIR code with syntax highlighting using custom HTML/CSS."""
    # Escape HTML entities first
    escaped = html.escape(mlir_code)

    # Define highlighting patterns (order matters - more specific first)
    patterns = [
        # Comments (// ...)
        (r'(//[^\n]*)', r'<span class="mlir-comment">\1</span>'),
        # Strings
        (r'("(?:[^"\\]|\\.)*")', r'<span class="mlir-string">\1</span>'),
        # Types (i32, f32, index, memref, tensor, etc.)
        (r'\b(i\d+|f\d+|bf16|index|none)\b', r'<span class="mlir-type">\1</span>'),
        (r'\b(memref|tensor|vector|tuple|complex)(&lt;[^&]*&gt;)?', r'<span class="mlir-type">\1\2</span>'),
        # Dialect prefixes and operations (func.func, arith.addi, scf.for, etc.)
        (r'\b(func|arith|scf|affine|memref|linalg|tensor|vector|builtin|llvm|cf|math)\.(\w+)\b',
         r'<span class="mlir-dialect">\1</span>.<span class="mlir-op">\2</span>'),
        # Keywords
        (r'\b(module|func|return|yield|if|else|for|while|iter_args|step|to)\b',
         r'<span class="mlir-keyword">\1</span>'),
        # Block arguments and SSA values (%name, %0, %arg0)
        (r'(%[\w\d_]+)', r'<span class="mlir-ssa">\1</span>'),
        # Block labels (^bb0:)
        (r'(\^[\w\d_]+:?)', r'<span class="mlir-block">\1</span>'),
        # Attributes (#map, @func_name)
        (r'(#[\w\d_]+)', r'<span class="mlir-attr">\1</span>'),
        (r'(@[\w\d_]+)', r'<span class="mlir-symbol">\1</span>'),
        # Numbers
        (r'\b(\d+\.?\d*(?:e[+-]?\d+)?)\b', r'<span class="mlir-number">\1</span>'),
        # Arrows
        (r'(-&gt;)', r'<span class="mlir-arrow">\1</span>'),
    ]

    highlighted = escaped
    for pattern, replacement in patterns:
        highlighted = re.sub(pattern, replacement, highlighted)

    html_content = f'''
    <style>
        .mlir-container {{
            background-color: #f8f8f8;
            color: #333333;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.5;
            padding: 12px;
            border-radius: 6px;
            border: 1px solid #e0e0e0;
            overflow: auto;
            max-height: {height}px;
            white-space: pre;
        }}
        .mlir-keyword {{ color: #0000ff; font-weight: bold; }}
        .mlir-type {{ color: #267f99; }}
        .mlir-dialect {{ color: #af00db; }}
        .mlir-op {{ color: #795e26; }}
        .mlir-ssa {{ color: #001080; }}
        .mlir-block {{ color: #a31515; }}
        .mlir-attr {{ color: #098658; }}
        .mlir-symbol {{ color: #0070c1; }}
        .mlir-number {{ color: #098658; }}
        .mlir-string {{ color: #a31515; }}
        .mlir-comment {{ color: #008000; font-style: italic; }}
        .mlir-arrow {{ color: #333333; }}
    </style>
    <div class="mlir-container">{highlighted}</div>
    '''

    components.html(html_content, height=height + 30, scrolling=True)


def get_available_examples() -> list:
    """Get list of available examples (must have both .cadl and test_*.c)."""
    cadl_dir = PROJECT_ROOT / "tutorial" / "cadl" / "compiler"
    csrc_dir = PROJECT_ROOT / "tutorial" / "csrc"
    examples = []
    for f in cadl_dir.iterdir():
        if f.is_file() and f.suffix == ".cadl":
            # Only include if corresponding test_*.c exists
            test_c = csrc_dir / f"test_{f.stem}.c"
            if test_c.exists():
                examples.append(f.stem)
    return sorted(examples)


def load_example_data(example_name: str) -> dict:
    """Load all files for an example."""
    cadl_dir = PROJECT_ROOT / "tutorial" / "cadl" / "compiler"
    csrc_dir = PROJECT_ROOT / "tutorial" / "csrc"
    mlir_dir = PROJECT_ROOT / "tutorial" / "mlir"
    output_dir = PROJECT_ROOT / "tutorial" / "outputs" / "compile_logs"

    data = {"name": example_name}

    cadl_name = example_name
    source_mappings = {
        "cadl": cadl_dir / f"{cadl_name}.cadl",
        "test_c": csrc_dir / f"test_{example_name}.c",
        "c_code": csrc_dir / f"{cadl_name}.c",
        "mlir_pattern": mlir_dir / f"{cadl_name}.mlir",
    }

    # Compiled artifacts from output directory (use cadl_name for output files)
    output_mappings = {
        "stats": output_dir / f"{cadl_name}.stats.json",
        "encoding": output_dir / f"{cadl_name}.json",
        "asm": output_dir / f"{cadl_name}.asm",
    }

    for key, filepath in {**source_mappings, **output_mappings}.items():
        if filepath.exists():
            if filepath.suffix == '.json':
                with open(filepath) as f:
                    data[key] = json.load(f)
            else:
                with open(filepath) as f:
                    data[key] = f.read()

    return data


def run_command(cmd: list, cwd: str = None, env: dict = None) -> tuple:
    """Run a command and return (success, stdout, stderr)."""
    try:
        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        result = subprocess.run(
            cmd,
            cwd=cwd or str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
            env=full_env
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out (60s limit)"
    except Exception as e:
        return False, "", str(e)


def run_cadl2c(cadl_code: str, work_dir: Path) -> tuple:
    """Run CADL to C conversion."""
    cadl_file = work_dir / "input.cadl"
    cadl_file.write_text(cadl_code)

    # Run cadl2c
    cmd = ["python", "aps-frontend", "cadl2c", str(cadl_file)]
    success, stdout, stderr = run_command(cmd)

    # Read generated C file if exists
    c_file = work_dir / "input.c"
    c_code = ""
    if c_file.exists():
        c_code = c_file.read_text()
    elif success and stdout:
        c_code = stdout

    return success, c_code, stderr


def run_polygeist(c_code: str, work_dir: Path) -> tuple:
    """Convert C to MLIR using Polygeist."""
    c_file = work_dir / "target.c"
    c_file.write_text(c_code)
    mlir_file = work_dir / "target.mlir"

    cmd = [
        "cgeist", "-c", "-S",
        str(c_file),
        "-O3", "--memref-fullrank",
        "-o", str(mlir_file)
    ]

    success, stdout, stderr = run_command(cmd)

    mlir_code = ""
    if success and mlir_file.exists():
        mlir_code = mlir_file.read_text()

    return success, mlir_code, stderr


def parse_skeleton_from_output(compile_output: str) -> dict:
    """Parse skeleton and components info from megg compile output.

    New format parses:
    - [Skeleton] for(for(stmt0, stmt1), for(stmt2))
    - [Rewrite] yield(add(mul(...))) -> body_stmt0
    - [Loop Hints] From ISAX pattern:
    - [External] loop decisions
    """
    import re

    result = {
        "instr_name": "",
        "skeleton_tree_str": "",  # e.g., "for(for(stmt0, stmt1), for(stmt2))"
        "rewrites": [],  # List of {"pattern": str, "component": str}
        "num_leaf_patterns": 0,
        "loop_hints": [],  # List of {"name": str, "trip": int, "nested": int, "level": int}
        "external_decisions": [],  # List of {"loop": str, "hint": str, "decision": str}
        "raw_skeleton_info": ""
    }

    lines = compile_output.split('\n')
    skeleton_lines = []

    for line in lines:
        # Extracting skeleton for {instr_name}
        if 'Extracting skeleton for' in line:
            match = re.search(r'Extracting skeleton for\s+(\S+)', line)
            if match:
                result["instr_name"] = match.group(1)
            skeleton_lines.append(line.strip())

        # [Skeleton] for(for(stmt0), for(stmt1))
        elif '[Skeleton]' in line:
            match = re.search(r'\[Skeleton\]\s+(.+)', line)
            if match:
                result["skeleton_tree_str"] = match.group(1).strip()
            skeleton_lines.append(line.strip())

        # [Rewrite] pattern -> component
        elif '[Rewrite]' in line:
            match = re.search(r'\[Rewrite\]\s+(.+?)\s+->\s+(\S+)', line)
            if match:
                result["rewrites"].append({
                    "pattern": match.group(1).strip(),
                    "component": match.group(2).strip()
                })
            skeleton_lines.append(line.strip())

        # [Loop Hints] loop_0: trip=4, nested=1, level=0
        elif re.match(r'\s*(loop_\d+):', line):
            match = re.search(r'(loop_\d+):\s*trip=(\d+),\s*nested=(\d+),\s*level=(-?\d+)', line)
            if match:
                result["loop_hints"].append({
                    "name": match.group(1),
                    "trip": int(match.group(2)),
                    "nested": int(match.group(3)),
                    "level": int(match.group(4))
                })

        # [External] loop_0 (trip=16, nested=1) vs loop_0 (trip=4, nested=1) → loop-unroll(4)
        elif '[External]' in line:
            skeleton_lines.append(line.strip())
            # Parse the decision
            match = re.search(r'\[External\]\s+(loop_\d+)\s+\(trip=(\d+),\s*nested=(\d+)\)\s+(?:vs|==)\s+(\S+)\s+\(trip=(\d+),\s*nested=(\d+)\)\s+→\s+(.+)', line)
            if match:
                result["external_decisions"].append({
                    "app_loop": match.group(1),
                    "app_trip": int(match.group(2)),
                    "app_nested": int(match.group(3)),
                    "hint_name": match.group(4),
                    "hint_trip": int(match.group(5)),
                    "hint_nested": int(match.group(6)),
                    "decision": match.group(7).strip()
                })

        # Built skeleton for {instr_name} with N leaf patterns
        elif 'Built skeleton for' in line:
            skeleton_lines.append(line.strip())
            match = re.search(r'with\s+(\d+)\s+leaf patterns', line)
            if match:
                result["num_leaf_patterns"] = int(match.group(1))

    result["raw_skeleton_info"] = '\n'.join(skeleton_lines)

    return result


def parse_insn_encoding(insn_hex: int) -> dict:
    """Parse RISC-V R-type instruction encoding.

    R-type format:
    - bits [6:0]   = opcode
    - bits [11:7]  = rd
    - bits [14:12] = funct3
    - bits [19:15] = rs1
    - bits [24:20] = rs2
    - bits [31:25] = funct7
    """
    return {
        "opcode": insn_hex & 0x7F,
        "rd": (insn_hex >> 7) & 0x1F,
        "funct3": (insn_hex >> 12) & 0x7,
        "rs1": (insn_hex >> 15) & 0x1F,
        "rs2": (insn_hex >> 20) & 0x1F,
        "funct7": (insn_hex >> 25) & 0x7F,
    }


def check_asm_matches_encoding(asm_code: str, encoding: dict) -> dict:
    """Check if .insn instructions in ASM match the expected encoding.

    Returns:
        {
            "matched_insns": [(hex_value, parsed_encoding), ...],
            "unmatched_insns": [(hex_value, parsed_encoding), ...],
            "expected_encoding": {...}
        }
    """
    import re

    result = {
        "matched_insns": [],
        "unmatched_insns": [],
        "expected_encoding": encoding
    }

    if not encoding:
        return result

    # Get expected opcode and funct7 from encoding
    expected_opcode = None
    expected_funct7 = None

    for instr_name, enc in encoding.items():
        if "opcode" in enc:
            expected_opcode = int(enc["opcode"], 16) if isinstance(enc["opcode"], str) else enc["opcode"]
        if "funct7" in enc:
            expected_funct7 = int(enc["funct7"], 16) if isinstance(enc["funct7"], str) else enc["funct7"]
        break  # Just use first instruction's encoding

    # Find all .insn instructions in ASM
    # Format: .insn 4, 0xXXXXXXXX
    insn_pattern = r'\.insn\s+\d+,\s*0x([0-9a-fA-F]+)'

    for match in re.finditer(insn_pattern, asm_code):
        hex_str = match.group(1)
        try:
            insn_hex = int(hex_str, 16)
            parsed = parse_insn_encoding(insn_hex)

            # Check if this matches our custom instruction
            opcode_match = (expected_opcode is None or parsed["opcode"] == expected_opcode)
            funct7_match = (expected_funct7 is None or parsed["funct7"] == expected_funct7)

            if opcode_match and funct7_match:
                result["matched_insns"].append((hex_str, parsed))
            else:
                result["unmatched_insns"].append((hex_str, parsed))
        except ValueError:
            pass

    return result


def run_compile_sh_handson(example_name: str) -> tuple:
    """Run pixi run compile-handson and return (success, snapshots_path, output)."""
    output_dir = PROJECT_ROOT / "tutorial" / "outputs" / "compile_logs"

    # Build paths for tutorial examples
    cadl_name = example_name
    cadl_file = f"tutorial/cadl/compiler/{cadl_name}.cadl"
    test_c_file = f"tutorial/csrc/test_{example_name}.c"
    output_exe = f"tutorial/outputs/{cadl_name}.riscv"

    # Check input files exist
    if not (PROJECT_ROOT / cadl_file).exists():
        return False, None, f"Error: CADL file not found: {cadl_file}"
    if not (PROJECT_ROOT / test_c_file).exists():
        return False, None, f"Error: Test C file not found: {test_c_file}"

    # Run pixi run compile-handson
    cmd = ["pixi", "run", "compile-handson", cadl_file, test_c_file, output_exe]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=180,  # 3 minutes for full pipeline
        )

        # Snapshots file is moved to output/compile_logs/ by compile.sh
        snapshots_path = output_dir / f"{cadl_name}.snapshots.json"

        output = result.stdout + "\n" + result.stderr

        if result.returncode == 0 and snapshots_path.exists():
            return True, snapshots_path, output
        else:
            return False, None, f"Return code: {result.returncode}\n{output}"

    except subprocess.TimeoutExpired:
        return False, None, "Command timed out (180s limit)"
    except Exception as e:
        return False, None, str(e)


# ============================================================
# STEP 1: ISAX Pattern Generation (CADL -> MLIR)
# ============================================================

def run_step1_cadl_to_mlir(example_data: dict):
    """Execute CADL to MLIR conversion (called from sidebar)."""
    default_cadl = example_data.get("cadl", "")

    if not default_cadl:
        st.session_state["step1_status"] = ("error", "No CADL file")
        st.rerun()
        return

    work_dir = Path(tempfile.mkdtemp(prefix="tutorial_"))

    # Step 1: CADL to C
    cadl_file = work_dir / "input.cadl"
    cadl_file.write_text(default_cadl)

    cmd = ["python", "aps-frontend", "cadl2c", str(cadl_file)]
    success, stdout, stderr = run_command(cmd)

    c_file = work_dir / "input.c"
    if c_file.exists():
        c_code = c_file.read_text()
    elif success and stdout:
        c_code = stdout
    else:
        st.session_state["step1_status"] = ("error", "CADL→C failed")
        st.session_state["step1_generated_mlir"] = ""
        st.rerun()
        return

    # Step 2: C to MLIR
    success, mlir_code, error = run_polygeist(c_code, work_dir)

    if success and mlir_code:
        st.session_state["step1_generated_mlir"] = mlir_code
        st.session_state["step2_pattern_mlir"] = mlir_code
        st.session_state["step1_status"] = ("success", "Done!")
        st.rerun()
    else:
        st.session_state["step1_status"] = ("error", "C→MLIR failed")
        st.session_state["step1_generated_mlir"] = ""
        st.rerun()


def render_step1():
    """Step 1: ISAX Pattern Generation - CADL to MLIR conversion."""
    st.markdown("## Step 1: ISAX Pattern Generation")

    example_data = st.session_state.get("example_data", {})

    st.info("Convert CADL definition to MLIR pattern (via Polygeist). Click **Run** in the sidebar to execute.")

    st.markdown("### CADL -> MLIR")

    default_cadl = example_data.get("cadl", "// No CADL file found for this example")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**CADL Source**")
        if default_cadl:
            render_cadl_code(default_cadl, height=400)
        else:
            st.warning("No CADL file found for this example")

    with col2:
        st.markdown("**ISAX Pattern MLIR**")

        display_mlir = st.session_state.get("step1_generated_mlir", "")

        if display_mlir:
            render_mlir_code(display_mlir, height=400)
        else:
            st.info("Click **Run** in the sidebar to convert CADL to MLIR, or **Load Example** to use pre-generated MLIR.")


# ============================================================
# STEP 2: Application Matching
# ============================================================

def render_skeleton_and_components(skeleton_info: dict):
    """Render skeleton as control flow graph with component rewrites."""
    import streamlit as st

    instr_name = skeleton_info.get("instr_name", "")
    skeleton_tree_str = skeleton_info.get("skeleton_tree_str", "")
    rewrites = skeleton_info.get("rewrites", [])
    num_leaf_patterns = skeleton_info.get("num_leaf_patterns", 0)
    raw_info = skeleton_info.get("raw_skeleton_info", "")

    # If no skeleton info, show a message
    if not instr_name and not skeleton_tree_str and not rewrites:
        st.info("No skeleton info found (simple pattern mode or not yet run)")
        return

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        if instr_name:
            st.metric("Instruction", instr_name)
    with col2:
        st.metric("Leaf Patterns", num_leaf_patterns)
    with col3:
        st.metric("Rewrites", len(rewrites))

    st.markdown("---")

    # Skeleton (Control Flow Graph)
    st.markdown("**Skeleton (Control Flow Graph):**")
    if skeleton_tree_str:
        st.code(skeleton_tree_str, language="text")
    else:
        st.info("No skeleton structure (simple pattern)")

    st.markdown("---")

    # Component Rewrites
    st.markdown("**Component Rewrites:**")
    if rewrites:
        for i, rw in enumerate(rewrites):
            pattern = rw.get("pattern", "")
            component = rw.get("component", "")

            with st.expander(f"**{component}**", expanded=True):
                st.code(f"{pattern}\n    -> {component}", language="text")
    else:
        st.info("No component rewrites found")

    # Raw info in expander
    if raw_info:
        with st.expander("Raw Output (Debug)", expanded=False):
            st.code(raw_info, language="bash")


def render_skeleton_and_components_inline(skeleton_info: dict):
    """Render skeleton and components in a compact inline format for side-by-side display."""
    import streamlit as st

    instr_name = skeleton_info.get("instr_name", "")
    skeleton_tree_str = skeleton_info.get("skeleton_tree_str", "")
    rewrites = skeleton_info.get("rewrites", [])
    num_leaf_patterns = skeleton_info.get("num_leaf_patterns", 0)
    raw_info = skeleton_info.get("raw_skeleton_info", "")

    # If no skeleton info, show a message
    if not instr_name and not skeleton_tree_str and not rewrites:
        st.info("No skeleton info found (simple pattern mode or not yet run)")
        return

    # Header with instruction name
    if instr_name:
        st.markdown(f"**Instruction: `{instr_name}`**")

    # Skeleton (Control Flow Graph)
    st.markdown("**Skeleton:**")
    if skeleton_tree_str:
        st.code(skeleton_tree_str, language="text")
    else:
        st.caption("(simple pattern)")

    # Component Rewrites - compact list
    st.markdown(f"**Components ({len(rewrites)}):**")
    if rewrites:
        for rw in rewrites:
            pattern = rw.get("pattern", "")
            component = rw.get("component", "")
            with st.expander(f"`{component}`", expanded=False):
                st.code(pattern, language="text")
    else:
        st.caption("No component rewrites")

    # Raw info in expander (optional)
    if raw_info:
        with st.expander("Debug Info", expanded=False):
            st.code(raw_info, language="bash")


def render_external_loop_transforms(skeleton_info: dict):
    """Render external loop transformation decisions."""
    import streamlit as st

    loop_hints = skeleton_info.get("loop_hints", [])
    external_decisions = skeleton_info.get("external_decisions", [])

    if not loop_hints and not external_decisions:
        st.info("No external loop transformations")
        return

    # Deduplicate app loops by loop name
    app_loops = {}
    for d in external_decisions:
        loop_name = d.get("app_loop", "")
        if loop_name and loop_name not in app_loops:
            app_loops[loop_name] = {
                "trip": d.get("app_trip", 0),
                "nested": d.get("app_nested", 0)
            }

    # Two columns: App loops vs ISAX loops
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Application Loop Features:**")
        if app_loops:
            for name, info in app_loops.items():
                st.code(f"{name}: trip={info['trip']}, nested={info['nested']}", language="text")
        else:
            st.info("No app loop info")

    with col2:
        st.markdown("**ISAX Pattern Loop Features:**")
        if loop_hints:
            for h in loop_hints:
                st.code(f"{h.get('name', '')}: trip={h.get('trip', 0)}, nested={h.get('nested', 0)}", language="text")
        else:
            st.info("No pattern loop info")

    # Decision rules explanation
    st.markdown("---")
    st.markdown("**Megg Decision Rules:**")
    st.markdown("""
- `app_trip > pattern_trip` → **loop-unroll**(factor = app_trip / pattern_trip)
- `app_nested < pattern_nested` → **loop-tile** (add nesting level)
- `app_nested > pattern_nested` → **loop-unroll-jam** (reduce nesting)
- `app == pattern` → no transformation needed
""")

    # Applied transformations (deduplicated by app_loop)
    st.markdown("---")
    st.markdown("**Applied Transformations:**")
    if external_decisions:
        seen_loops = set()
        for d in external_decisions:
            app_loop = d.get("app_loop", "")
            if app_loop in seen_loops:
                continue
            seen_loops.add(app_loop)

            decision = d.get("decision", "")
            hint = d.get("hint_name", "")
            if "no change" in decision or "no transform" in decision:
                st.info(f"`{app_loop}` ← {hint}: {decision}")
            else:
                st.success(f"`{app_loop}` ← {hint}: **{decision}**")
    else:
        st.info("No transformations applied")


def run_matching_with_custom_code(test_code: str, pattern_mlir: str, example_name: str) -> tuple:
    """Run matching with custom test code."""
    import tempfile

    output_dir = PROJECT_ROOT / "tutorial" / "outputs" / "compile_logs"

    # Create temp directory for custom code
    work_dir = Path(tempfile.mkdtemp(prefix="megg_custom_"))

    # Write custom test code
    custom_test_file = work_dir / f"test_{example_name}.c"
    custom_test_file.write_text(test_code)

    # Write pattern MLIR
    custom_pattern_file = work_dir / f"{example_name}.mlir"
    custom_pattern_file.write_text(pattern_mlir)

    # Copy encoding JSON if exists
    encoding_json = output_dir / f"{example_name}.json"
    if encoding_json.exists():
        import shutil
        shutil.copy(encoding_json, work_dir / f"{example_name}.json")

    output_path = work_dir / f"{example_name}.out"

    # Build command
    cmd = [
        "pixi", "run", "python", "./megg-opt.py",
        "--mode", "c-e2e",
        str(custom_test_file),
        "--custom-instructions", str(custom_pattern_file),
        "-o", str(output_path),
        "--keep-intermediate",
        "--handson"
    ]

    # Add encoding if exists
    custom_encoding = work_dir / f"{example_name}.json"
    if custom_encoding.exists():
        cmd.extend(["--encoding-json", str(custom_encoding)])

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT}/python:{PROJECT_ROOT}/3rdparty/llvm-project/install/python_packages/mlir_core"

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
            env=env
        )

        output = result.stdout + "\n" + result.stderr

        # Look for output files
        snapshots_path = work_dir / f"{example_name}.snapshots.json"
        asm_path = work_dir / f"{example_name}.asm"

        asm_code = ""
        if asm_path.exists():
            asm_code = asm_path.read_text()

        if result.returncode == 0 and snapshots_path.exists():
            return True, snapshots_path, output, asm_code, str(work_dir)
        else:
            return False, None, output, asm_code, str(work_dir)

    except subprocess.TimeoutExpired:
        return False, None, "Command timed out (120s limit)", "", str(work_dir)
    except Exception as e:
        return False, None, str(e), "", str(work_dir)


def run_step2_matching(example_data: dict):
    """Execute E-graph matching (called from sidebar)."""
    example_name = example_data.get("name", "unknown")
    output_dir = PROJECT_ROOT / "tutorial" / "outputs" / "compile_logs"

    cadl_name = example_name

    # Check prerequisites
    pattern_mlir = st.session_state.get("step2_pattern_mlir", "") or example_data.get("mlir_pattern", "")
    csrc_dir = PROJECT_ROOT / "tutorial" / "csrc"
    test_c_file = csrc_dir / f"test_{example_name}.c"

    if not test_c_file.exists():
        st.session_state["step2_status"] = ("error", "No app code")
        st.rerun()
        return
    if not pattern_mlir:
        st.session_state["step2_status"] = ("error", "No pattern")
        st.rerun()
        return

    success, snapshots_path, output = run_compile_sh_handson(example_name)

    # Reload ASM (use cadl_name for output files)
    asm_path = output_dir / f"{cadl_name}.asm"
    if asm_path.exists():
        st.session_state["asm_code"] = asm_path.read_text()

    st.session_state["compile_output"] = output

    # Parse skeleton and components from output
    skeleton_info = parse_skeleton_from_output(output)
    st.session_state["skeleton_info"] = skeleton_info

    if success and snapshots_path:
        with open(snapshots_path) as f:
            snapshots_data = json.load(f)

        st.session_state["snapshots_data"] = snapshots_data
        st.session_state["matching_complete"] = True

        # Load stats (use cadl_name for output files)
        stats_path = output_dir / f"{cadl_name}.stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                st.session_state["compile_stats"] = json.load(f)

        st.session_state["step2_status"] = ("success", "Done!")
        st.rerun()
    else:
        st.session_state["step2_status"] = ("error", "Failed")
        st.session_state["matching_complete"] = False
        st.rerun()


def render_step2():
    """Step 2: Application Matching with E-graph optimization."""
    st.markdown("## Step 2: Application Matching")

    example_data = st.session_state.get("example_data", {})
    example_name = example_data.get("name", "unknown")

    st.info("Match application code against ISAX pattern using E-graph optimization. Click **Run E-graph Matching** in the sidebar to execute.")

    # Load pattern MLIR data
    # Priority: 1) From Step 1, 2) From file, 3) Generate from example C code
    if "step2_pattern_mlir" in st.session_state:
        pattern_mlir = st.session_state["step2_pattern_mlir"]
        st.success("Using pattern from Step 1")
    elif example_data.get("mlir_pattern", ""):
        pattern_mlir = example_data.get("mlir_pattern", "")
    else:
        # Try to generate from example C code if available
        pattern_mlir = ""
        c_code = example_data.get("c_code", "")
        if c_code:
            st.info("No pattern MLIR found. You can generate it from the example C code below, or complete Step 1 first.")

    csrc_dir = PROJECT_ROOT / "tutorial" / "csrc"
    output_dir = PROJECT_ROOT / "tutorial" / "outputs" / "compile_logs"
    test_c_file = csrc_dir / f"test_{example_name}.c"

    # Read current file content
    current_test_c = ""
    if test_c_file.exists():
        current_test_c = test_c_file.read_text()

    # ============ Input Section ============
    st.markdown("---")
    st.markdown("### Input")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Application Code** (`test_{example_name}.c`)")
        if current_test_c:
            render_c_code(current_test_c, height=350)
        else:
            st.warning("No test_*.c file found")

    with col2:
        st.markdown("**ISAX Pattern (*.mlir)**")
        if pattern_mlir:
            render_mlir_code(pattern_mlir, height=350)
        else:
            st.warning("No pattern MLIR. Complete Step 1 first, or generate from example C code.")
            # Offer to generate from example C code
            c_code = example_data.get("c_code", "")
            if c_code:
                st.markdown("**Example ISAX C Code available:**")
                with st.expander("View C Code", expanded=False):
                    st.code(c_code, language="c")
                if st.button("Generate MLIR from Example C", key="gen_mlir_from_c"):
                    with st.spinner("Running Polygeist..."):
                        work_dir = Path(tempfile.mkdtemp(prefix="tutorial_"))
                        success, mlir_code, error = run_polygeist(c_code, work_dir)
                        if success and mlir_code:
                            st.session_state["step2_pattern_mlir"] = mlir_code
                            st.success("Generated! Refreshing...")
                            st.rerun()
                        else:
                            st.error(f"Generation failed: {error}")

    # ============ Helper functions ============
    from megg.utils import get_temp_dir
    tmp_dir = get_temp_dir()

    def compress_svg(svg_path):
        """Compress SVG using scour library. Returns path to compressed SVG in /tmp."""
        import tempfile
        import hashlib
        from pathlib import Path
        from scour import scour

        # Generate unique filename based on original path and mtime
        mtime = svg_path.stat().st_mtime
        file_hash = hashlib.md5(f"{svg_path}:{mtime}".encode()).hexdigest()[:12]
        compressed_path = Path(tempfile.gettempdir()) / f"svg_compressed_{file_hash}.svg"

        # Return cached version if exists
        if compressed_path.exists():
            return compressed_path

        # Scour options for aggressive compression
        options = scour.sanitizeOptions(options=None)
        options.remove_metadata = True
        options.remove_descriptive_elements = True
        options.strip_comments = True
        options.enable_viewboxing = True
        options.indent_type = None
        options.newlines = False

        with open(svg_path, 'r') as f_in:
            compressed = scour.scourString(f_in.read(), options)

        compressed_path.write_text(compressed)
        return compressed_path

    def render_svg(svg_path, expanded=True):
        """Render an SVG file with Ctrl+scroll zoom support."""
        import streamlit.components.v1 as components
        import hashlib

        if svg_path.exists():
            try:
                # Use compressed SVG
                compressed_path = compress_svg(svg_path)
                svg_content = compressed_path.read_text()
                # Generate unique ID for this SVG container
                svg_id = hashlib.md5(str(svg_path).encode()).hexdigest()[:8]

                html_content = f'''
                <div id="svg-container-{svg_id}" style="
                    overflow: auto;
                    max-height: 500px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    padding: 10px;
                    background: white;
                ">
                    <div id="svg-wrapper-{svg_id}" style="
                        transform-origin: top left;
                        transition: transform 0.1s ease-out;
                    ">
                        {svg_content}
                    </div>
                </div>
                <div style="font-size: 12px; color: #666; margin-top: 4px;">
                    Ctrl + Scroll to zoom
                </div>
                <script>
                    (function() {{
                        const container = document.getElementById('svg-container-{svg_id}');
                        const wrapper = document.getElementById('svg-wrapper-{svg_id}');
                        let scale = 1;
                        const minScale = 0.2;
                        const maxScale = 5;

                        container.addEventListener('wheel', function(e) {{
                            if (e.ctrlKey) {{
                                e.preventDefault();
                                const delta = e.deltaY > 0 ? 0.9 : 1.1;
                                scale = Math.min(maxScale, Math.max(minScale, scale * delta));
                                wrapper.style.transform = 'scale(' + scale + ')';
                            }}
                        }}, {{ passive: false }});
                    }})();
                </script>
                '''
                components.html(html_content, height=550, scrolling=True)
                return True
            except Exception as e:
                st.warning(f"Could not load SVG: {e}")
                return False
        else:
            st.info("SVG not available. Run matching to generate.")
            return False

    # ============ E-graph data ============
    snapshots_data = st.session_state.get("snapshots_data", {})
    snapshots = snapshots_data.get("snapshots", []) if isinstance(snapshots_data, dict) else []
    skeleton_info = st.session_state.get("skeleton_info", {})
    matching_complete = st.session_state.get("matching_complete", False)

    # Extract phase map
    phase_map = {}
    for s in snapshots:
        phase_map[s['phase_name']] = s

    def get_egraph_stats(phase_name):
        if phase_name in phase_map:
            egraph_stats = phase_map[phase_name].get('egraph_stats', {})
            for func_name, stats in egraph_stats.items():
                if isinstance(stats, dict):
                    return stats.get('total_eclasses', 0), stats.get('total_nodes', 0)
        return 0, 0

    # ============ Section 1: Initial E-graph ============
    st.markdown("---")
    st.markdown("### 1. Initial E-graph")

    if matching_complete:
        init_ec, init_en = get_egraph_stats('0_egraph_init')
        st.metric("E-classes / E-nodes", f"{init_ec} / {init_en}")

        # Show application MLIR (from Polygeist)
        affine_mlir_files = sorted(tmp_dir.glob("affine_extracted_*.mlir"), key=lambda p: p.stat().st_mtime, reverse=True)
        if affine_mlir_files:
            latest_affine = affine_mlir_files[0]
            with st.expander("Application MLIR (Polygeist output)", expanded=False):
                mlir_content = latest_affine.read_text()
                render_mlir_code(mlir_content, height=300)

        with st.expander("Initial E-graph Visualization", expanded=True):
            render_svg(tmp_dir / "egraph_before_internal.svg")
    else:
        st.info("Run E-graph Matching to see the initial e-graph")

    # ============ Section 2: Hybrid Rewrite ============
    st.markdown("---")
    st.markdown("### 2. Hybrid Rewrite")

    if matching_complete:
        # ---- 2.1 Internal Rewrites ----
        st.markdown("#### 2.1 Internal Rewrites")
        st.markdown("Algebraic equivalence laws applied to the e-graph")

        internal_ec, internal_en = get_egraph_stats('1_internal_rewrites')
        init_ec, _ = get_egraph_stats('0_egraph_init')
        delta_ec = internal_ec - init_ec

        # Get internal rewrite count
        internal_phase = phase_map.get('1_internal_rewrites', {})
        internal_stats = internal_phase.get('cumulative_stats', {})
        internal_count = internal_stats.get('internal_rewrites', 0)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Internal Rewrites Applied", internal_count)
        with col2:
            st.metric("E-classes / E-nodes", f"{internal_ec} / {internal_en}",
                     delta=f"+{delta_ec}" if delta_ec > 0 else str(delta_ec))

        # List of internal rewrite rules
        st.markdown("**Applied Rules:**")
        st.markdown("""
- `(a + b) * c ↔ a * c + b * c` (distributivity)
- `a * 0 → 0`, `a * 1 → a` (identity)
- `a + 0 → a` (identity)
- Associativity, commutativity rules
        """)

        with st.expander("E-graph after Internal Rewrites", expanded=False):
            render_svg(tmp_dir / "egraph_after_internal.svg")

        # ---- 2.2 External Rewrites ----
        st.markdown("---")
        st.markdown("#### 2.2 External Rewrites")
        st.markdown("Loop transformations to match ISAX pattern structure")

        external_ec, external_en = get_egraph_stats('2_external_rewrites')
        delta_ec = external_ec - internal_ec

        # Get external rewrite details
        external_phase = phase_map.get('2_external_rewrites', {})
        external_stats = external_phase.get('cumulative_stats', {})
        external_count = external_stats.get('external_rewrites', 0)
        external_details = external_stats.get('external_details', [])

        col1, col2 = st.columns(2)
        with col1:
            st.metric("E-classes / E-nodes", f"{external_ec} / {external_en}",
                     delta=f"+{delta_ec}" if delta_ec > 0 else str(delta_ec))
        with col2:
            st.metric("Passes Applied", external_count)

        # Show applied passes
        if external_details:
            st.markdown("**Applied Passes:**")
            for detail in external_details:
                pass_name = detail.get('pass', 'unknown')
                param = detail.get('parameter', '')
                value = detail.get('value', '')
                st.code(f"{pass_name}({param}={value})", language="text")

        # External Loop Transformations details
        st.markdown("**Loop Transformation Decisions:**")
        render_external_loop_transforms(skeleton_info)

        # Show MLIR comparison: before vs after transformation
        st.markdown("**MLIR Comparison:**")

        # Find files
        affine_mlir_files = sorted(tmp_dir.glob("affine_extracted_*.mlir"), key=lambda p: p.stat().st_mtime, reverse=True)
        loop_mlir_files = sorted(tmp_dir.glob("loop_*_loop-*.mlir"), key=lambda p: p.stat().st_mtime, reverse=True)

        if affine_mlir_files or loop_mlir_files:
            with st.expander("Before vs After Loop Transformation", expanded=False):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Before (Extracted)**")
                    if affine_mlir_files:
                        st.caption(affine_mlir_files[0].name)
                        render_mlir_code(affine_mlir_files[0].read_text(), height=250)
                    else:
                        st.info("No extracted MLIR found")

                with col2:
                    st.markdown("**After (Transformed)**")
                    if loop_mlir_files:
                        st.caption(loop_mlir_files[0].name)
                        render_mlir_code(loop_mlir_files[0].read_text(), height=250)
                    else:
                        st.info("No transformed MLIR found")
        else:
            st.info("No MLIR files found for comparison")

        with st.expander("E-graph after External Rewrites", expanded=False):
            render_svg(tmp_dir / "egraph_before_custom.svg")
    else:
        st.info("Run E-graph Matching to see hybrid rewrite results")

    # ============ Section 3: Skeleton-Component Matching ============
    st.markdown("---")
    st.markdown("### 3. Skeleton-Component Matching")

    if matching_complete:
        st.markdown("Match application patterns against ISAX skeleton and components")

        # Two-column layout: MLIR on left, Skeleton/Components on right
        col_mlir, col_skeleton = st.columns(2)

        with col_mlir:
            st.markdown("**ISAX Pattern MLIR**")
            # Get pattern MLIR from session state or example data
            example_data = st.session_state.get("example_data", {})
            pattern_mlir = st.session_state.get("step2_pattern_mlir", "") or example_data.get("mlir_pattern", "")
            if pattern_mlir:
                render_mlir_code(pattern_mlir, height=400)
            else:
                st.info("No pattern MLIR available")

        with col_skeleton:
            # Skeleton & Components (inline version)
            render_skeleton_and_components_inline(skeleton_info)

        # Custom matching result
        st.markdown("---")
        st.markdown("#### Matching Result")

        custom_phase = phase_map.get('3_custom_rewrites', {})
        custom_stats = custom_phase.get('cumulative_stats', {})
        custom_matches = custom_stats.get('custom_rewrites', 0)
        custom_ec, custom_en = get_egraph_stats('3_custom_rewrites')

        col1, col2 = st.columns(2)
        with col1:
            if custom_matches > 0:
                st.success(f"**{custom_matches} component rewrite(s) matched!**")
            else:
                st.warning("No component rewrites matched")
        with col2:
            delta_ec = custom_ec - get_egraph_stats('2_external_rewrites')[0]
            st.metric("After Custom Rewrites", f"{custom_ec} / {custom_en}",
                     delta=f"+{delta_ec}" if delta_ec > 0 else str(delta_ec))

        # E-graph after custom rewrites
        with st.expander("E-graph after Custom Rewrites", expanded=False):
            render_svg(tmp_dir / "egraph_after_custom.svg")
    else:
        st.info("Run E-graph Matching to see skeleton-component matching")

    # ============ Section 4: Extraction ============
    st.markdown("---")
    st.markdown("### 4. Extraction")

    if matching_complete:
        st.markdown("Extract the best expression from e-graph and generate MLIR with custom instructions")

        # Find the optimized MLIR file (with llvm.inline_asm)
        optimized_mlir_files = sorted(tmp_dir.glob("megg_e2e_*/optimized.mlir"), key=lambda p: p.stat().st_mtime, reverse=True)

        if optimized_mlir_files:
            optimized_mlir_path = optimized_mlir_files[0]
            optimized_mlir_content = optimized_mlir_path.read_text()

            # Count llvm.inline_asm occurrences
            inline_asm_count = optimized_mlir_content.count("llvm.inline_asm")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Custom Instructions", inline_asm_count)
            with col2:
                st.caption(f"File: {optimized_mlir_path.name}")

            st.markdown("**Extracted MLIR with `llvm.inline_asm`:**")
            render_mlir_code(optimized_mlir_content, height=350)
        else:
            st.info("No optimized MLIR found. Run matching to generate.")
    else:
        st.info("Run E-graph Matching to see extracted MLIR")

    # ============ Section 5: Result ============
    st.markdown("---")
    st.markdown("### 5. Result")

    matching_complete = st.session_state.get("matching_complete", False)
    display_asm = st.session_state.get("asm_code", "")

    # Get encoding from example data
    encoding = example_data.get("encoding", {})

    if matching_complete and display_asm:
        # Check if .insn matches expected encoding
        match_result = check_asm_matches_encoding(display_asm, encoding)
        matched_insns = match_result["matched_insns"]
        unmatched_insns = match_result["unmatched_insns"]

        if matched_insns:
            st.success(f"**Match SUCCESS**: {len(matched_insns)} custom instruction(s) with correct encoding found!")

            # Show expected encoding
            if encoding:
                for instr_name, enc in encoding.items():
                    st.markdown(f"**Expected encoding for `{instr_name}`:**")
                    st.code(f"opcode: {enc.get('opcode', 'N/A')}, funct7: {enc.get('funct7', 'N/A')}", language="yaml")

            # Show matched instructions
            st.markdown("**Matched Instructions:**")
            for hex_val, parsed in matched_insns:
                st.code(f"0x{hex_val}: opcode=0x{parsed['opcode']:02x}, funct7=0x{parsed['funct7']:02x}", language="yaml")
        elif ".insn" in display_asm:
            st.warning("**Match FAILED**: .insn found but encoding doesn't match!")
            if encoding:
                for instr_name, enc in encoding.items():
                    st.markdown(f"**Expected encoding for `{instr_name}`:**")
                    st.code(f"opcode: {enc.get('opcode', 'N/A')}, funct7: {enc.get('funct7', 'N/A')}", language="yaml")
            if unmatched_insns:
                st.markdown("**Found .insn instructions (wrong encoding):**")
                for hex_val, parsed in unmatched_insns[:3]:
                    st.code(f"0x{hex_val}: opcode=0x{parsed['opcode']:02x}, funct7=0x{parsed['funct7']:02x}", language="yaml")
        else:
            st.warning("**Match FAILED**: No custom instruction (.insn) found in assembly")

        # Extract function-specific assembly
        lines = display_asm.split('\n')
        func_lines = []
        in_func = False
        for line in lines:
            if f'<{example_name}' in line or f'<test_{example_name}' in line:
                in_func = True
            if in_func:
                func_lines.append(line)
                if line.strip().startswith('ret') or len(func_lines) > 100:
                    break

        if func_lines:
            st.markdown(f"**Function `{example_name}` assembly:**")
            st.code('\n'.join(func_lines), language="asm")
        else:
            st.markdown("**Full assembly (first 100 lines):**")
            st.code('\n'.join(lines[:100]), language="asm")
    else:
        st.info("Run E-graph Matching to generate assembly output")
        st.markdown("**No matching result yet**")


# ============================================================
# Overview
# ============================================================

def render_overview():
    """Render the overview/welcome page."""
    st.markdown("### Overview")

    st.markdown("""
    Welcome to the **ASP-DAC Hands-on Tutorial** on Custom Instruction Synthesis with E-graph!

    This tutorial demonstrates how to use **Megg** (MLIR E-graph) compiler to automatically
    match and optimize code using custom RISC-V instructions.
    """)

    st.markdown("---")

    # Architecture diagram
    st.markdown("### System Architecture")
    arch_img = Path(__file__).parent / "figs" / "architecture.png"
    st.image(str(arch_img), use_container_width=True)

    st.markdown("---")

    # Tutorial steps
    st.markdown("### Tutorial Steps")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **E-graph Playground**

        Learn the basics of e-graphs and equality saturation
        using the interactive egglog demo.
        """)

    with col2:
        st.markdown("""
        **Step 1: ISAX Pattern**

        Convert CADL definition to MLIR patterns
        for matching.

        - CADL → MLIR
        """)

    with col3:
        st.markdown("""
        **Step 2: Matching**

        Apply e-graph optimization to match patterns
        and generate optimized code.

        - E-graph construction
        - Pattern matching
        - Code generation
        """)

    st.markdown("---")

    # Key concepts
    st.markdown("### Key Concepts")

    with st.expander("What is an E-graph?", expanded=False):
        st.markdown("""
        An **E-graph** (equivalence graph) is a data structure that compactly represents
        many equivalent programs. It enables **equality saturation**, where we apply
        rewrite rules to explore all equivalent forms simultaneously, then extract
        the optimal one.
        """)


    st.markdown("---")

    # Input Code Display
    st.markdown("### Input Code")

    example_data = st.session_state.get("example_data", {})
    example_name = example_data.get("name", "unknown")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Application C Code** (`test_{example_name}.c`)")
        test_c = example_data.get("test_c", "// No test C file found for this example")
        st.code(test_c, language="c")

    with col2:
        st.markdown(f"**ISAX Definition** (`{example_name}.cadl`)")
        cadl_code = example_data.get("cadl", "// No file found for this example")
        st.code(cadl_code, language="rust")

    st.markdown("---")

    # Output Display
    st.markdown("### Output")
    st.markdown("After compilation, the custom instruction is generated:")
    output_asm = """800001a8 <vgemv3d_vv>:
800001a8:	62b5752b          	.insn	4, 0x62b5752b
800001ac:	00000513          	li	a0,0
800001b0:	00008067          	ret"""
    st.code(output_asm, language="asm")


# ============================================================
# E-graph Playground
# ============================================================

def render_egglog_playground():
    """Render the egglog playground page with embedded demo."""
    st.markdown("### E-graph Playground")
    st.markdown("*Interactive egglog environment - try equality saturation in your browser!*")

    st.markdown("""
    This playground embeds the official [egglog demo](https://egraphs-good.github.io/egglog-demo/).
    You can write egglog programs, run them, and visualize e-graphs.

    **Quick Start:**
    - Click **Load Example** below to load the arithmetic simplification example
    - Press `Ctrl+Enter` or click "Run" to execute
    - View the e-graph visualization on the right
    """)

    # Example code for the tutorial
    example_code = ''';;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; 1. Language definition
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(datatype Expr
  (Mul Expr Expr)
  (Div Expr Expr)
  (Shl Expr Expr)
  (Const i64)
  (Var String))

;; 2. Expression: (a * 2) / 2

(let start
  (Div (Mul (Var "a") (Const 2))
       (Const 2)))

;; rule 1
;; a * 2  <=>  a << 1
(rewrite (Mul a (Const 2))
         (Shl a (Const 1)))
(run 5)
;; extract
(extract start)

;; rule 2
;; (a * b) / c <=> a * (b / c)
(rewrite (Div (Mul a b) c)
         (Mul a (Div b c)))
(run 5)
;; extract
(extract start)

;; rule 3
;; a / a => 1
(rewrite (Div a a)
         (Const 1))
(run 5)
;; extract
(extract start)

;; rule 4
;; a * 1 => a
(rewrite (Mul a (Const 1)) a)
(run 5)
;; extract
(extract start)
'''

    # Escape the code for JavaScript
    escaped_code = json.dumps(example_code)

    st.markdown("---")

    # Create HTML component with iframe and load button
    import streamlit.components.v1 as components

    html_content = f'''
    <style>
        .egglog-container {{
            width: 100%;
        }}
        .load-btn {{
            background-color: #ff4b4b;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            margin-bottom: 10px;
        }}
        .load-btn:hover {{
            background-color: #ff3333;
        }}
        .load-btn:disabled {{
            background-color: #ccc;
            cursor: not-allowed;
        }}
        .status {{
            font-size: 12px;
            color: #666;
            margin-left: 10px;
        }}
        #egglog-iframe {{
            width: 100%;
            height: 750px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
    </style>

    <div class="egglog-container">
        <button class="load-btn" id="load-btn" onclick="loadExample()">Load Example: Arithmetic Simplification</button>
        <span class="status" id="status"></span>
        <br><br>
        <iframe id="egglog-iframe" src="https://egraphs-good.github.io/egglog-demo/"></iframe>
    </div>

    <script type="module">
        // Import LZMA compression from egglog-demo
        const lzmaModule = await import('https://egraphs-good.github.io/egglog-demo/lzma-url.mjs');
        const compressUrlSafe = lzmaModule.compressUrlSafe;

        const exampleCode = {escaped_code};

        window.loadExample = function() {{
            const btn = document.getElementById('load-btn');
            const status = document.getElementById('status');
            const iframe = document.getElementById('egglog-iframe');

            btn.disabled = true;
            status.textContent = 'Compressing...';

            try {{
                const compressed = compressUrlSafe(exampleCode);
                const url = 'https://egraphs-good.github.io/egglog-demo/?program=' + compressed;
                iframe.src = url;
                status.textContent = 'Loaded!';
                setTimeout(() => {{ status.textContent = ''; }}, 2000);
            }} catch (e) {{
                status.textContent = 'Error: ' + e.message;
            }}

            btn.disabled = false;
        }};
    </script>
    '''

    components.html(html_content, height=850, scrolling=False)

    st.markdown("---")

    # Show the example code for reference
    with st.expander("View Example Code", expanded=False):
        st.markdown("""
        This example demonstrates how e-graphs can simplify `(a * 2) / 2` to just `a`
        through equality saturation by applying algebraic rewrite rules.
        """)
        st.code(example_code, language="lisp")


# ============================================================
# Main
# ============================================================

def main():
    st.set_page_config(
        page_title="Megg Hands-on Tutorial",
        page_icon="",
        layout="wide"
    )

    st.title("ASP-DAC Hands-on Tutorial")
    st.markdown("*Interactive Custom Instruction Synthesis with E-graph*")

    # Navigation
    with st.sidebar:
        st.markdown("## Navigation")

        # Example selector
        examples = get_available_examples()
        selected_example = st.selectbox(
            "Select Example:",
            examples,
            index=examples.index("vgemv3d") if "vgemv3d" in examples else 0,
            key="selected_example"
        )

        # Clear previous results when example changes
        prev_example = st.session_state.get("_prev_example", None)
        if prev_example != selected_example:
            # Clear Step 1 results
            st.session_state.pop("step1_generated_mlir", None)
            st.session_state.pop("step1_status", None)
            st.session_state.pop("step2_pattern_mlir", None)
            # Clear Step 2 results
            st.session_state.pop("step2_status", None)
            st.session_state.pop("snapshots_data", None)
            st.session_state.pop("matching_complete", None)
            st.session_state.pop("asm_code", None)
            st.session_state.pop("compile_output", None)
            st.session_state.pop("skeleton_info", None)
            st.session_state.pop("compile_stats", None)
            # Remember current example
            st.session_state["_prev_example"] = selected_example

        # Load example data
        example_data = load_example_data(selected_example)
        st.session_state["example_data"] = example_data

        st.markdown("---")
        st.markdown("### Steps")

        # Custom step selector with Run buttons inline
        # Initialize step in session state
        if "current_step" not in st.session_state:
            st.session_state["current_step"] = "Overview"

        # Overview
        if st.button("Overview", key="nav_overview", use_container_width=True,
                     type="primary" if st.session_state["current_step"] == "Overview" else "secondary"):
            st.session_state["current_step"] = "Overview"
            st.rerun()

        # E-graph Playground
        if st.button("E-graph Playground", key="nav_playground", use_container_width=True,
                     type="primary" if st.session_state["current_step"] == "E-graph Playground" else "secondary"):
            st.session_state["current_step"] = "E-graph Playground"
            st.rerun()

        # Step 1 with Run button (vertically centered)
        st.markdown("---")
        col1a, col1b = st.columns([4, 1], vertical_alignment="center")
        with col1a:
            if st.button("Step 1: ISAX Pattern", key="nav_step1", use_container_width=True,
                         type="primary" if st.session_state["current_step"] == "Step 1: ISAX Pattern" else "secondary"):
                st.session_state["current_step"] = "Step 1: ISAX Pattern"
                st.session_state.pop("step1_status", None)  # Clear status on nav
                st.rerun()
        with col1b:
            if st.button("▶", key="run_step1", type="primary", use_container_width=True,
                         help="Run CADL to MLIR"):
                st.session_state["current_step"] = "Step 1: ISAX Pattern"
                st.session_state["step1_status"] = ("info", "Running...")
                st.rerun()

        # Show Step 1 status
        step1_status = st.session_state.get("step1_status")
        if step1_status:
            status_type, status_msg = step1_status
            if status_type == "success":
                st.success(status_msg)
            elif status_type == "error":
                st.error(status_msg)
            elif status_type == "info":
                st.info(status_msg)
                # Actually run the task after showing status
                run_step1_cadl_to_mlir(example_data)

        # Step 2 with Run button (vertically centered)
        col2a, col2b = st.columns([4, 1], vertical_alignment="center")
        with col2a:
            if st.button("Step 2: Matching", key="nav_step2", use_container_width=True,
                         type="primary" if st.session_state["current_step"] == "Step 2: Matching" else "secondary"):
                st.session_state["current_step"] = "Step 2: Matching"
                st.session_state.pop("step2_status", None)  # Clear status on nav
                st.rerun()
        with col2b:
            if st.button("▶", key="run_step2", type="primary", use_container_width=True,
                         help="Run E-graph Matching"):
                st.session_state["current_step"] = "Step 2: Matching"
                st.session_state["step2_status"] = ("info", "Running...")
                st.rerun()

        # Show Step 2 status
        step2_status = st.session_state.get("step2_status")
        if step2_status:
            status_type, status_msg = step2_status
            if status_type == "success":
                st.success(status_msg)
            elif status_type == "error":
                st.error(status_msg)
            elif status_type == "info":
                st.info(status_msg)
                # Actually run the task after showing status
                run_step2_matching(example_data)

        st.markdown("---")
        if st.button("Reset All"):
            for key in list(st.session_state.keys()):
                if key != "selected_example":
                    del st.session_state[key]
            st.rerun()

    # Render selected step
    step = st.session_state.get("current_step", "Overview")
    if step == "Overview":
        render_overview()
    elif step == "E-graph Playground":
        render_egglog_playground()
    elif step == "Step 1: ISAX Pattern":
        render_step1()
    elif step == "Step 2: Matching":
        render_step2()


if __name__ == "__main__":
    main()
