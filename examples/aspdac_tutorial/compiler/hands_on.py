"""
Hands-on interactive mode for ASP-DAC Tutorial.
Allows participants to run compilation steps and see intermediate results.
"""

import streamlit as st
import subprocess
import tempfile
import json
from pathlib import Path
import shutil
import os
import sys

# examples/aspdac_tutorial/compiler/ -> need to go up 3 levels to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "megg"))


def get_project_root():
    return PROJECT_ROOT


def get_available_examples() -> list:
    """Get list of available examples."""
    examples_dir = PROJECT_ROOT / "examples" / "diff_match"
    examples = []
    for d in examples_dir.iterdir():
        if d.is_dir() and (d / f"{d.name}.cadl").exists():
            examples.append(d.name)
    return sorted(examples)


def load_example_data(example_name: str) -> dict:
    """Load all files for an example."""
    example_dir = PROJECT_ROOT / "examples" / "diff_match" / example_name

    data = {"name": example_name}

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


def run_compile_sh_handson(example_name: str, example_dir: Path) -> tuple:
    """Run compile.sh with --handson flag and return (success, snapshots_path, output)."""
    test_c = example_dir / f"test_{example_name}.c"
    pattern_mlir = example_dir / f"{example_name}.mlir"
    encoding_json = example_dir / f"{example_name}.json"
    output_path = example_dir / f"{example_name}.out"

    if not test_c.exists():
        return False, None, f"test file not found: {test_c}"
    if not pattern_mlir.exists():
        return False, None, f"pattern file not found: {pattern_mlir}"

    # Build megg-opt.py command
    cmd = [
        "pixi", "run", "python", "./megg-opt.py",
        "--mode", "c-e2e",
        str(test_c),
        "--custom-instructions", str(pattern_mlir),
        "-o", str(output_path),
        "--keep-intermediate",
        "--handson"
    ]

    # Add encoding if exists
    if encoding_json.exists():
        cmd.extend(["--encoding-json", str(encoding_json)])

    # Set environment
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

        # Look for snapshots file
        snapshots_path = example_dir / f"{example_name}.snapshots.json"

        output = result.stdout + "\n" + result.stderr

        if result.returncode == 0 and snapshots_path.exists():
            return True, snapshots_path, output
        else:
            return False, None, f"Return code: {result.returncode}\n{output}"

    except subprocess.TimeoutExpired:
        return False, None, "Command timed out (120s limit)"
    except Exception as e:
        return False, None, str(e)


# ============================================================
# STEP 1: ISAX Pattern Generation (CADL -> C -> MLIR -> Skeleton)
# ============================================================

def render_step1():
    """Step 1: ISAX Pattern Generation - Three sub-steps."""
    st.markdown("## Step 1: ISAX Pattern Generation")

    example_data = st.session_state.get("example_data", {})
    example_name = example_data.get("name", "unknown")

    st.info("Generate ISAX pattern from CADL description through C to MLIR, then extract skeleton and components")

    # Sub-step selector
    sub_step = st.radio(
        "Select Sub-step:",
        ["1.1 CADL -> C", "1.2 C -> MLIR"],
        horizontal=True,
        key="step1_substep"
    )

    if sub_step == "1.1 CADL -> C":
        render_step1_1_cadl_to_c(example_data, example_name)
    elif sub_step == "1.2 C -> MLIR":
        render_step1_2_c_to_mlir(example_data, example_name)


def render_step1_1_cadl_to_c(example_data: dict, example_name: str):
    """Sub-step 1.1: CADL to C conversion."""
    st.markdown("### 1.1 CADL -> C (ISAX Definition)")

    default_cadl = example_data.get("cadl", "// No CADL file found for this example")
    default_c = example_data.get("c_code", "")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**CADL Source**")
        st.code(default_cadl, language="rust")

        cadl_code = st.text_area(
            "Edit CADL code:",
            value=default_cadl,
            height=250,
            key=f"cadl_input_{example_name}",
            label_visibility="collapsed"
        )

        col1a, col1b = st.columns(2)
        with col1a:
            if st.button("Run CADL to C", type="primary", key="run_cadl2c"):
                with st.spinner("Converting CADL to C..."):
                    work_dir = Path(tempfile.mkdtemp(prefix="tutorial_"))
                    success, c_code, error = run_cadl2c(cadl_code, work_dir)

                    if success and c_code:
                        st.session_state["step1_generated_c"] = c_code
                        st.success("Conversion successful!")
                    else:
                        st.error(f"Conversion failed: {error}")
                        st.session_state["step1_generated_c"] = ""
        with col1b:
            if st.button("Load Example C", key="load_example_c"):
                st.session_state["step1_generated_c"] = default_c
                st.rerun()

    with col2:
        st.markdown("**Generated ISAX C Code**")

        # Show empty if not run yet, otherwise show generated
        display_c = st.session_state.get("step1_generated_c", "")

        if display_c:
            st.code(display_c, language="c")
            if st.button("Use in Step 1.2", key="copy_to_step1_2"):
                st.session_state["step1_2_c_code"] = display_c
                st.success("Copied! Go to Step 1.2")
        else:
            st.info("Run CADL to C conversion to see output here, or load example C code.")


def render_step1_2_c_to_mlir(example_data: dict, example_name: str):
    """Sub-step 1.2: C to MLIR conversion."""
    st.markdown("### 1.2 C -> MLIR (ISAX Pattern)")

    example_c = example_data.get("c_code", "// No C file found")
    default_mlir = example_data.get("mlir_pattern", "")

    # Use C from step 1.1 if available
    input_c = st.session_state.get("step1_2_c_code", example_c)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**ISAX C Code**")
        st.code(input_c, language="c")

        c_code = st.text_area(
            "Edit C code:",
            value=input_c,
            height=200,
            key=f"c_input_{example_name}",
            label_visibility="collapsed"
        )

        col1a, col1b = st.columns(2)
        with col1a:
            if st.button("Run C to MLIR", type="primary", key="run_polygeist"):
                with st.spinner("Running Polygeist..."):
                    work_dir = Path(tempfile.mkdtemp(prefix="tutorial_"))
                    success, mlir_code, error = run_polygeist(c_code, work_dir)

                    if success and mlir_code:
                        st.session_state["step1_generated_mlir"] = mlir_code
                        st.success("Conversion successful!")
                    else:
                        st.error(f"Conversion failed: {error}")
                        st.session_state["step1_generated_mlir"] = ""
        with col1b:
            if st.button("Load Example MLIR", key="load_example_mlir"):
                st.session_state["step1_generated_mlir"] = default_mlir
                st.rerun()

    with col2:
        st.markdown("**ISAX Pattern MLIR**")

        # Show empty if not run yet
        display_mlir = st.session_state.get("step1_generated_mlir", "")

        if display_mlir:
            st.code(display_mlir, language="mlir")
            if st.button("Use in Step 1.3", key="copy_to_step1_3"):
                st.session_state["step1_3_mlir"] = display_mlir
                st.success("Copied! Go to Step 1.3")
        else:
            st.info("Run C to MLIR conversion to see output here, or load example MLIR.")


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


def run_matching_with_custom_code(test_code: str, pattern_mlir: str, example_name: str, example_dir: Path) -> tuple:
    """Run matching with custom test code."""
    import tempfile

    # Create temp directory for custom code
    work_dir = Path(tempfile.mkdtemp(prefix="megg_custom_"))

    # Write custom test code
    custom_test_file = work_dir / f"test_{example_name}.c"
    custom_test_file.write_text(test_code)

    # Write pattern MLIR
    custom_pattern_file = work_dir / f"{example_name}.mlir"
    custom_pattern_file.write_text(pattern_mlir)

    # Copy encoding JSON if exists
    encoding_json = example_dir / f"{example_name}.json"
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


def render_step2():
    """Step 2: Application Matching with E-graph optimization."""
    st.markdown("## Step 2: Application Matching")

    example_data = st.session_state.get("example_data", {})
    example_name = example_data.get("name", "unknown")

    st.info("Match application code against ISAX pattern using E-graph optimization")

    # Load data
    if "step2_pattern_mlir" in st.session_state:
        pattern_mlir = st.session_state["step2_pattern_mlir"]
        st.success("Using pattern from Step 1")
    else:
        pattern_mlir = example_data.get("mlir_pattern", "")

    example_dir = PROJECT_ROOT / "examples" / "diff_match" / example_name
    test_c_file = example_dir / f"test_{example_name}.c"

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

        # Toggle between view and edit mode
        edit_mode = st.checkbox("Edit Mode", key="edit_mode_toggle")

        if edit_mode:
            # Editable text area
            edited_code = st.text_area(
                "Edit test code:",
                value=current_test_c,
                height=350,
                key="test_code_editor",
                label_visibility="collapsed"
            )

            col1a, col1b = st.columns(2)
            with col1a:
                if st.button("Save Changes", type="primary", key="save_test_code"):
                    try:
                        test_c_file.write_text(edited_code)
                        st.success(f"Saved to {test_c_file}")
                        # Reload example data
                        st.session_state["example_data"]["test_c"] = edited_code
                    except Exception as e:
                        st.error(f"Failed to save: {e}")
            with col1b:
                if st.button("Discard Changes", key="discard_test_code"):
                    st.rerun()

            test_c = edited_code
        else:
            # Rendered code view
            if current_test_c:
                st.code(current_test_c, language="c")
            else:
                st.warning("No test_*.c file found")
            test_c = current_test_c

    with col2:
        st.markdown("**ISAX Pattern (*.mlir)**")
        if pattern_mlir:
            st.code(pattern_mlir, language="mlir")
        else:
            st.warning("No pattern MLIR. Complete Step 1 first.")

    # ============ Run Matching ============
    st.markdown("---")

    st.caption(f"Runs: `examples/diff_match/{example_name}/compile.sh` with --handson")

    if st.button("Run E-graph Matching", type="primary", key="run_matching"):
        if not test_c:
            st.error("No application code found")
        elif not pattern_mlir:
            st.error("No pattern MLIR. Complete Step 1 first.")
        else:
            with st.spinner("Running Megg E2E compiler with --handson mode..."):
                # Always run with example files (which may have been edited)
                success, snapshots_path, output = run_compile_sh_handson(example_name, example_dir)

                # Reload ASM
                asm_path = example_dir / f"{example_name}.asm"
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

                    # Load stats
                    stats_path = example_dir / f"{example_name}.stats.json"
                    if stats_path.exists():
                        with open(stats_path) as f:
                            st.session_state["compile_stats"] = json.load(f)

                    st.success("Matching complete!")
                else:
                    st.error(f"Matching failed")
                    st.session_state["matching_complete"] = False

    # Show compile output
    compile_output = st.session_state.get("compile_output")
    if compile_output:
        with st.expander("Compiler Output", expanded=False):
            st.code(compile_output, language="bash")

    # ============ Helper functions ============
    from megg.utils import get_temp_dir
    tmp_dir = get_temp_dir()

    def render_svg(svg_path, expanded=True):
        """Render an SVG file with Ctrl+scroll zoom support."""
        import streamlit.components.v1 as components
        import hashlib

        if svg_path.exists():
            try:
                svg_content = svg_path.read_text()
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
                st.code(mlir_content, language="mlir")

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
                        st.code(affine_mlir_files[0].read_text(), language="mlir")
                    else:
                        st.info("No extracted MLIR found")

                with col2:
                    st.markdown("**After (Transformed)**")
                    if loop_mlir_files:
                        st.caption(loop_mlir_files[0].name)
                        st.code(loop_mlir_files[0].read_text(), language="mlir")
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

        # Skeleton & Components
        render_skeleton_and_components(skeleton_info)

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

    # ============ Section 4: Result ============
    st.markdown("---")
    st.markdown("### 4. Result")

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

    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │                        Megg Compiler                            │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐    │
    │   │    ISAX     │      │   C Code    │      │   C Code    │    │
    │   │  (Pattern)  │      │  (Pattern)  │      │   (App)     │    │
    │   └──────┬──────┘      └──────┬──────┘      └──────┬──────┘    │
    │          │                    │                    │           │
    │          ▼                    ▼                    ▼           │
    │   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐    │
    │   │  ISAX→C     │      │  Polygeist  │      │  Polygeist  │    │
    │   │  Frontend   │      │   (C→MLIR)  │      │   (C→MLIR)  │    │
    │   └──────┬──────┘      └──────┬──────┘      └──────┬──────┘    │
    │          │                    │                    │           │
    │          ▼                    ▼                    ▼           │
    │   ┌──────────────────────────────────────────────────────┐     │
    │   │                    MLIR                              │     │
    │   └──────────────────────────┬───────────────────────────┘     │
    │                              │                                 │
    │                              ▼                                 │
    │   ┌──────────────────────────────────────────────────────┐     │
    │   │                   E-graph                            │     │
    │   │  ┌────────────────────────────────────────────────┐  │     │
    │   │  │  Internal Rewrites (algebraic laws)            │  │     │
    │   │  │  External Rewrites (loop transforms)           │  │     │
    │   │  │  Custom Rewrites (pattern matching)            │  │     │
    │   │  └────────────────────────────────────────────────┘  │     │
    │   └──────────────────────────┬───────────────────────────┘     │
    │                              │                                 │
    │                              ▼                                 │
    │   ┌──────────────────────────────────────────────────────┐     │
    │   │              Optimized MLIR + Custom Instr           │     │
    │   └──────────────────────────┬───────────────────────────┘     │
    │                              │                                 │
    │                              ▼                                 │
    │   ┌──────────────────────────────────────────────────────┐     │
    │   │                 RISC-V Assembly                      │     │
    │   └──────────────────────────────────────────────────────┘     │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    ```
    """)

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

        Define custom instructions and convert them
        to MLIR patterns for matching.

        - ISAX → C (Frontend)
        - C → MLIR (Polygeist)
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

    with st.expander("What is Megg?", expanded=False):
        st.markdown("""
        **Megg** (MLIR E-graph) is our compiler that uses e-graphs to optimize MLIR code.
        It can automatically detect patterns that match custom instructions and replace
        them, enabling transparent acceleration with custom hardware.
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
    - Write egglog code in the editor
    - Press `Ctrl+Enter` or click "Run" to execute
    - View the e-graph visualization on the right
    """)

    st.markdown("---")

    # Embed the egglog demo using iframe
    import streamlit.components.v1 as components
    components.iframe(
        "https://egraphs-good.github.io/egglog-demo/",
        height=800,
        scrolling=True
    )

    st.markdown("---")

    # Example code section
    st.markdown("### Example: Arithmetic Simplification")
    st.markdown("""
    Copy the code below into the playground above to see how e-graphs can simplify
    `(a * 2) / 2` to just `a` through equality saturation.
    """)

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

        # Load example data
        example_data = load_example_data(selected_example)
        st.session_state["example_data"] = example_data

        st.markdown("---")

        step = st.radio(
            "Select Step:",
            [
                "Overview",
                "E-graph Playground",
                "Step 1: ISAX Pattern",
                "Step 2: Matching"
            ]
        )

        st.markdown("---")
        st.markdown("### Pipeline")
        st.markdown("""
        **Step 1: ISAX Pattern**
        - 1.1 CADL -> C
        - 1.2 C -> MLIR

        **Step 2: Matching**
        - E-graph Optimization
        - Skeleton & Components
        - Pattern Matching
        - Code Generation
        """)

        st.markdown("---")
        if st.button("Reset All"):
            for key in list(st.session_state.keys()):
                if key != "selected_example":
                    del st.session_state[key]
            st.rerun()

    # Render selected step
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
