# ASP-DAC Tutorial Demo

Interactive visualization for the Megg compiler tutorial at ASP-DAC 2025.

## Quick Start

### Mode 1: Presentation Mode (展示模式)
```bash
pixi run tutorial-demo
```
静态展示所有例子的中间结果，适合讲解流程。

### Mode 2: Hands-on Mode (动手模式)
```bash
pixi run tutorial-hands-on
```
参与者可以：
- 修改 CADL/C/MLIR 代码
- 点击 Run 按钮执行每个步骤
- 实时查看中间结果

Then open http://localhost:8501 in your browser.

## Features

### Step 1: CADL → C Conversion
- Side-by-side view of CADL hardware description and generated C code
- Highlights key syntax transformations

### Step 2: Hybrid Rewrite
- E-graph growth visualization (before/after)
- Internal rewrite rules display (algebraic laws)
- External MLIR passes applied

### Step 3: Skeleton Extraction
- MLIR pattern code view
- Visual skeleton tree representation
- Component breakdown (loads, stores, arithmetic)

### Step 4: Pattern Matching
- Input C code vs output assembly comparison
- Custom instruction encoding display
- Matching animation showing operation reduction

## Available Examples

- `vgemv3d` - 4x4 Matrix-Vector Multiply
- `lerp` - Linear Interpolation
- `horner3` - Polynomial Evaluation (Horner's method)
- `v3ddist_vv` - 3D Vector Distance
- `avg_r` - Averaging with Rounding
- `q15_mulr` - Q15 Fixed-point Multiply with Rounding
- And more...

## For Presenters

1. Use the sidebar to navigate between steps
2. Switch examples using the dropdown
3. Each step shows relevant intermediate representations
4. Expanders contain detailed explanations for audience Q&A

## Deployment

For Streamlit Cloud deployment:
1. Push to GitHub
2. Connect to Streamlit Cloud
3. Set main file path: `tutorial_demo/app.py`
