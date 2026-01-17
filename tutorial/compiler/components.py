"""
Visualization components for the tutorial demo.
"""

import streamlit as st
import re


def render_pipeline_diagram():
    """Render the compilation pipeline as an interactive diagram."""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 30px; border-radius: 15px; color: white;">
        <h3 style="text-align: center; margin-bottom: 20px;">Megg Compilation Pipeline</h3>
        <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap;">
            <div style="text-align: center; padding: 15px;">
                <div style="background: rgba(255,255,255,0.2); padding: 20px; border-radius: 10px;
                            min-width: 100px;">
                    <div style="font-size: 24px;">📝</div>
                    <div style="font-weight: bold;">CADL</div>
                    <div style="font-size: 12px;">Hardware Spec</div>
                </div>
            </div>
            <div style="font-size: 24px;">→</div>
            <div style="text-align: center; padding: 15px;">
                <div style="background: rgba(255,255,255,0.2); padding: 20px; border-radius: 10px;
                            min-width: 100px;">
                    <div style="font-size: 24px;">💻</div>
                    <div style="font-weight: bold;">C Code</div>
                    <div style="font-size: 12px;">Software Impl</div>
                </div>
            </div>
            <div style="font-size: 24px;">→</div>
            <div style="text-align: center; padding: 15px;">
                <div style="background: rgba(255,255,255,0.2); padding: 20px; border-radius: 10px;
                            min-width: 100px;">
                    <div style="font-size: 24px;">🔧</div>
                    <div style="font-weight: bold;">MLIR</div>
                    <div style="font-size: 12px;">IR Pattern</div>
                </div>
            </div>
            <div style="font-size: 24px;">→</div>
            <div style="text-align: center; padding: 15px;">
                <div style="background: rgba(255,255,255,0.2); padding: 20px; border-radius: 10px;
                            min-width: 100px;">
                    <div style="font-size: 24px;">🌳</div>
                    <div style="font-weight: bold;">E-Graph</div>
                    <div style="font-size: 12px;">Equivalences</div>
                </div>
            </div>
            <div style="font-size: 24px;">→</div>
            <div style="text-align: center; padding: 15px;">
                <div style="background: rgba(255,255,255,0.2); padding: 20px; border-radius: 10px;
                            min-width: 100px;">
                    <div style="font-size: 24px;">⚡</div>
                    <div style="font-weight: bold;">RISC-V</div>
                    <div style="font-size: 12px;">Custom Instr</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_egraph_growth(initial: dict, final: dict):
    """Render e-graph growth visualization."""
    import streamlit as st

    initial_classes = initial.get("num_eclasses", 0)
    initial_nodes = initial.get("num_enodes", 0)
    final_classes = final.get("num_eclasses", 0)
    final_nodes = final.get("num_enodes", 0)

    growth_classes = ((final_classes - initial_classes) / initial_classes * 100) if initial_classes > 0 else 0
    growth_nodes = ((final_nodes - initial_nodes) / initial_nodes * 100) if initial_nodes > 0 else 0

    st.markdown(f"""
    <div style="display: flex; justify-content: space-around; margin: 20px 0;">
        <div style="text-align: center; padding: 20px; background: #e8f4ea; border-radius: 10px; flex: 1; margin: 0 10px;">
            <div style="font-size: 14px; color: #666;">Initial E-Graph</div>
            <div style="font-size: 32px; font-weight: bold; color: #2e7d32;">{initial_classes}</div>
            <div style="font-size: 12px; color: #888;">e-classes</div>
            <div style="font-size: 20px; color: #4caf50; margin-top: 5px;">{initial_nodes}</div>
            <div style="font-size: 12px; color: #888;">e-nodes</div>
        </div>
        <div style="display: flex; align-items: center; font-size: 40px; color: #1976d2;">
            →
        </div>
        <div style="text-align: center; padding: 20px; background: #e3f2fd; border-radius: 10px; flex: 1; margin: 0 10px;">
            <div style="font-size: 14px; color: #666;">Final E-Graph</div>
            <div style="font-size: 32px; font-weight: bold; color: #1565c0;">{final_classes}</div>
            <div style="font-size: 12px; color: #888;">e-classes (+{growth_classes:.0f}%)</div>
            <div style="font-size: 20px; color: #2196f3; margin-top: 5px;">{final_nodes}</div>
            <div style="font-size: 12px; color: #888;">e-nodes (+{growth_nodes:.0f}%)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_skeleton_tree(mlir_code: str, name: str):
    """Render skeleton as a tree structure."""
    # Count structures (for summary display)
    _ = mlir_code.count("scf.for")  # for_count
    _ = mlir_code.count("scf.if")   # if_count
    load_count = mlir_code.count("memref.load")
    store_count = mlir_code.count("memref.store")
    mul_count = mlir_code.count("arith.muli")
    add_count = mlir_code.count("arith.addi")

    # Extract loop bounds
    for_matches = re.findall(r'scf\.for %(\w+) = (%\w+) to (%\w+) step (%\w+)', mlir_code)

    tree_html = f"""
    <div style="background: #f5f5f5; padding: 20px; border-radius: 10px; font-family: monospace;">
        <div style="color: #1976d2; font-weight: bold; margin-bottom: 15px;">
            Skeleton: {name}
        </div>
        <div style="margin-left: 0px; border-left: 3px solid #4caf50; padding-left: 15px;">
            <div style="color: #388e3c;">func.func @{name}</div>
    """

    indent = 1
    for i, (var, lower, upper, step) in enumerate(for_matches):
        margin = 20 * (indent + i)
        tree_html += f"""
            <div style="margin-left: {margin}px; border-left: 3px solid #ff9800; padding-left: 15px; margin-top: 10px;">
                <div style="color: #f57c00;">scf.for %{var} = {lower} to {upper} step {step}</div>
        """

    # Add leaf nodes (operations)
    leaf_margin = 20 * (indent + len(for_matches))
    tree_html += f"""
        <div style="margin-left: {leaf_margin}px; margin-top: 10px; padding: 10px;
                    background: #fff; border-radius: 5px; border: 1px solid #ddd;">
            <div style="color: #666; font-size: 12px;">Components:</div>
            <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 5px;">
    """

    if load_count > 0:
        tree_html += f'<span style="background: #e3f2fd; padding: 5px 10px; border-radius: 15px; font-size: 12px;">load x{load_count}</span>'
    if store_count > 0:
        tree_html += f'<span style="background: #fce4ec; padding: 5px 10px; border-radius: 15px; font-size: 12px;">store x{store_count}</span>'
    if mul_count > 0:
        tree_html += f'<span style="background: #f3e5f5; padding: 5px 10px; border-radius: 15px; font-size: 12px;">mul x{mul_count}</span>'
    if add_count > 0:
        tree_html += f'<span style="background: #e8f5e9; padding: 5px 10px; border-radius: 15px; font-size: 12px;">add x{add_count}</span>'

    tree_html += """
            </div>
        </div>
    """

    # Close loop divs
    for _ in for_matches:
        tree_html += "</div>"

    tree_html += """
        </div>
    </div>
    """

    st.markdown(tree_html, unsafe_allow_html=True)


def render_matching_animation(source_ops: int, matched_ops: int, custom_instr: str):
    """Render matching result visualization."""
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                padding: 25px; border-radius: 15px; margin: 20px 0;">
        <div style="text-align: center; margin-bottom: 20px;">
            <span style="font-size: 18px; font-weight: bold; color: #333;">Pattern Matching Result</span>
        </div>
        <div style="display: flex; justify-content: center; align-items: center; gap: 30px;">
            <div style="text-align: center;">
                <div style="background: #ffcdd2; padding: 30px; border-radius: 10px; min-width: 150px;">
                    <div style="font-size: 36px; font-weight: bold; color: #c62828;">{source_ops}</div>
                    <div style="font-size: 14px; color: #666;">MLIR Operations</div>
                </div>
            </div>
            <div style="font-size: 40px; color: #4caf50;">⟹</div>
            <div style="text-align: center;">
                <div style="background: #c8e6c9; padding: 30px; border-radius: 10px; min-width: 150px;">
                    <div style="font-size: 36px; font-weight: bold; color: #2e7d32;">1</div>
                    <div style="font-size: 14px; color: #666;">Custom Instruction</div>
                </div>
            </div>
        </div>
        <div style="text-align: center; margin-top: 20px; padding: 15px;
                    background: rgba(255,255,255,0.7); border-radius: 10px;">
            <code style="font-size: 16px; color: #1565c0;">{custom_instr}</code>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_code_diff(before: str, after: str, before_label: str = "Before", after_label: str = "After"):
    """Render code side by side with highlighting."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"#### {before_label}")
        st.code(before, language="c")

    with col2:
        st.markdown(f"#### {after_label}")
        st.code(after, language="asm")


def render_rewrite_rules_diagram():
    """Render internal rewrite rules as a visual diagram."""
    st.markdown("""
    <div style="background: #fff; padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0;">
        <h4 style="color: #1976d2; margin-bottom: 15px;">Internal Rewrite Rules (E-Graph)</h4>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
            <div style="background: #e3f2fd; padding: 15px; border-radius: 8px;">
                <div style="font-weight: bold; color: #1565c0; margin-bottom: 5px;">Commutativity</div>
                <code style="font-size: 12px;">Add(x,y) ↔ Add(y,x)</code>
            </div>
            <div style="background: #f3e5f5; padding: 15px; border-radius: 8px;">
                <div style="font-weight: bold; color: #7b1fa2; margin-bottom: 5px;">Associativity</div>
                <code style="font-size: 12px;">Add(Add(x,y),z) ↔ Add(x,Add(y,z))</code>
            </div>
            <div style="background: #e8f5e9; padding: 15px; border-radius: 8px;">
                <div style="font-weight: bold; color: #2e7d32; margin-bottom: 5px;">Identity</div>
                <code style="font-size: 12px;">Add(x,0) → x</code>
            </div>
            <div style="background: #fff3e0; padding: 15px; border-radius: 8px;">
                <div style="font-weight: bold; color: #e65100; margin-bottom: 5px;">Strength Reduction</div>
                <code style="font-size: 12px;">Mul(x,2) → Shl(x,1)</code>
            </div>
            <div style="background: #fce4ec; padding: 15px; border-radius: 8px;">
                <div style="font-weight: bold; color: #c2185b; margin-bottom: 5px;">Constant Folding</div>
                <code style="font-size: 12px;">Add(3,4) → 7</code>
            </div>
            <div style="background: #e0f7fa; padding: 15px; border-radius: 8px;">
                <div style="font-weight: bold; color: #00838f; margin-bottom: 5px;">Neg Elimination</div>
                <code style="font-size: 12px;">Add(x,Neg(y)) → Sub(x,y)</code>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_external_passes_diagram(passes: list):
    """Render external MLIR passes."""
    st.markdown("""
    <div style="background: #fff; padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0; margin-top: 20px;">
        <h4 style="color: #388e3c; margin-bottom: 15px;">External Rewrite Passes (MLIR)</h4>
    """, unsafe_allow_html=True)

    if passes:
        for p in passes:
            pass_name = p.get('pass', 'unknown')
            param = p.get('parameter', '')
            value = p.get('value', '')

            st.markdown(f"""
            <div style="background: #e8f5e9; padding: 15px; border-radius: 8px; margin: 10px 0;
                        display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-weight: bold; color: #2e7d32;">{pass_name}</span>
                </div>
                <div>
                    <code style="background: #c8e6c9; padding: 5px 10px; border-radius: 5px;">
                        {param}={value}
                    </code>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="color: #666; text-align: center; padding: 20px;">
                No external passes applied
            </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
