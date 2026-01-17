# Pure Computation Pattern Tests for diff_match

本目录包含用于验证 e-graph 代数优化能力的纯计算模式测试。

## 测试用例概览

### 1. LERP - 线性插值

**目标**: 验证 e-graph 对分配/结合律、操作数重排与跨式子 CSE 的统一与抽取

**算法**: `r = a + t*(b-a)`，其中 t=5

**文件**:
- `lerp/lerp.c` - CADL 生成的 C 实现
- `lerp/lerp.mlir` - cpp2mlir 生成的模式定义
- `lerp/lerp.json` - RISC-V 编码定义
- `lerp/test_lerp.c` - 测试程序（10个测试用例）
- `lerp/compile.sh` - 编译脚本

**测试用例**: 包括正常值、边界值、负数等10个测试

**期望**: e-graph 能抽取到乘加相邻（MAC 友好）的形态

---

### 2. HORNER3 - 三次多项式求值

**目标**: 展示全局重结合带来的乘法树高度降低与更规整的 mul→add→mul→add 流水

**算法**: `4x³ + 3x² + 2x + 1`（朴素展开形式）

**文件**:
- `horner3/horner3.c` - CADL 生成的 C 实现
- `horner3/horner3.mlir` - cpp2mlir 生成的模式定义
- `horner3/horner3.json` - RISC-V 编码定义
- `horner3/test_horner3.c` - 测试程序（10个测试用例）
- `horner3/compile.sh` - 编译脚本

**测试用例**: 测试 x=0,1,2,3,-1,-2,5,10,-3,4 等值

**期望**: e-graph 抽取乘法树高度更低、乘加相邻的实现

---

### 3. AVG_R - 位运算平均值

**目标**: 验证位运算优化和 CSE 能力

**算法**: `(x & y) + ((x ^ y) >> 1)`（位运算实现的平均值）

**文件**:
- `avg_r/avg_r.c` - CADL 生成的 C 实现
- `avg_r/avg_r.mlir` - cpp2mlir 生成的模式定义
- `avg_r/avg_r.json` - RISC-V 编码定义
- `avg_r/test_avg_r.c` - 测试程序（10个测试用例）
- `avg_r/compile.sh` - 编译脚本

**测试用例**: 包括相同值、差异值、边界值等10个测试

**期望**: e-graph 识别位运算模式并优化

---

### 4. Q15_MULR - Q15 定点数乘法

**目标**: 验证定点数乘法的复杂计算模式

**算法**: Q15 定点数乘法 `(x*y + 16384) >> 15`

**文件**:
- `q15_mulr/q15_mulr.c` - CADL 生成的 C 实现
- `q15_mulr/q15_mulr.mlir` - cpp2mlir 生成的模式定义
- `q15_mulr/q15_mulr.json` - RISC-V 编码定义
- `q15_mulr/test_q15_mulr.c` - 测试程序（10个测试用例）
- `q15_mulr/compile.sh` - 编译脚本

**测试用例**: 测试各种 Q15 定点数乘法场景

**期望**: e-graph 优化定点数运算流水线

---

## 目录结构

```
tests/diff_match/
├── README.md              # 本文件
├── lerp/                  # 线性插值
│   ├── lerp.c            # CADL 生成的实现
│   ├── lerp.cadl         # CADL 源码
│   ├── lerp.mlir         # cpp2mlir 生成的模式
│   ├── lerp.json         # 编码定义
│   ├── test_lerp.c       # 测试程序
│   └── compile.sh        # 编译脚本
├── horner3/               # Horner 多项式
│   ├── horner3.c
│   ├── horner3.cadl
│   ├── horner3.mlir
│   ├── horner3.json
│   ├── test_horner3.c
│   └── compile.sh
├── avg_r/                 # 位运算平均值
│   ├── avg_r.c
│   ├── avg_r.cadl
│   ├── avg_r.mlir
│   ├── avg_r.json
│   ├── test_avg_r.c
│   └── compile.sh
├── q15_mulr/              # Q15 定点数乘法
│   ├── q15_mulr.c
│   ├── q15_mulr.cadl
│   ├── q15_mulr.mlir
│   ├── q15_mulr.json
│   ├── test_q15_mulr.c
│   └── compile.sh
├── v3ddist_vs/            # （已有）3D 距离计算（向量-标量）
├── v3ddist_vv/            # （已有）3D 距离计算（向量-向量）
├── vcovmat3d_vs/          # （已有）协方差矩阵
└── vgemv3d/               # （已有）矩阵-向量乘法
```

## 使用方法

### 编译单个测试

```bash
# LERP
cd tests/diff_match/lerp
./compile.sh

# HORNER3
cd tests/diff_match/horner3
./compile.sh

# AVG_R
cd tests/diff_match/avg_r
./compile.sh

# Q15_MULR
cd tests/diff_match/q15_mulr
./compile.sh
```

### 运行测试

```bash
# 运行 LERP 测试
./tests/diff_match/lerp/lerp.out

# 运行 HORNER3 测试
./tests/diff_match/horner3/horner3.out

# 运行 AVG_R 测试
./tests/diff_match/avg_r/avg_r.out

# 运行 Q15_MULR 测试
./tests/diff_match/q15_mulr/q15_mulr.out
```

### 查看反汇编

```bash
# 查看生成的自定义指令
cat tests/diff_match/lerp/lerp.asm | grep -A5 -B5 "\.insn"
cat tests/diff_match/horner3/horner3.asm | grep -A5 -B5 "\.insn"
cat tests/diff_match/avg_r/avg_r.asm | grep -A5 -B5 "\.insn"
cat tests/diff_match/q15_mulr/q15_mulr.asm | grep -A5 -B5 "\.insn"
```

## 生成工具链

### C → MLIR 转换

使用 `cpp2mlir` 工具将 C 代码转换为 MLIR：

```bash
python python/megg/utils/cpp2mlir \
  --input tests/diff_match/lerp/lerp.c \
  --output tests/diff_match/lerp/lerp.mlir \
  --verbose
```

该工具会：
1. 使用 mlsynthesis 将 C 转换为初始 MLIR
2. 应用 mlir-opt 规范化 passes（canonicalize、cse、lower-affine）
3. 生成优化后的 MLIR

### 测试程序特点

所有 `test_*.c` 文件都遵循统一的模式：
- 使用 `#pragma megg optimize` 标记优化函数
- 包含 `marchid.h` 和 `riscv-pk/encoding.h`
- 使用 `asm volatile("fence")` 确保内存顺序
- 包含 10 个测试用例，涵盖各种场景
- 返回值：0=成功，1=失败

## E-graph 优势分析

| 模式 | E-graph 能力 | 演示内容 |
|------|-------------|---------|
| LERP | 分配律 | `a+t*(b-a)` 的乘加优化 |
| HORNER3 | 重结合优化 | 降低乘法树高度 |
| AVG_R | 位运算优化 | `(x&y) + ((x^y)>>1)` 模式识别 |
| Q15_MULR | 定点数优化 | 复杂算术流水线优化 |

## 编码方案

所有指令使用 RISC-V 自定义扩展格式：

- **opcode**: `0001011` (自定义-0)
- **funct3**:
  - `001`: LERP
  - `010`: HORNER3
  - `011`: AVG_R
  - `100`: Q15_MULR
- **funct7**: `0000001`

## 测试结果验证

成功标准：
1. ✅ **编译成功**: 生成 RISC-V 可执行文件
2. ✅ **MLIR 生成**: cpp2mlir 正确生成 MLIR
3. ✅ **指令匹配**: e-graph 成功匹配自定义指令模式
4. ✅ **编码正确**: 生成的 `.insn` 指令格式正确
5. ✅ **测试通过**: 所有 10 个测试用例返回预期结果

## 参考文档

- [docs/PURE_COMPUTATION_PATTERN_PROPOSAL.md](../../docs/PURE_COMPUTATION_PATTERN_PROPOSAL.md) - 完整提案（英文）
- [docs/PURE_COMPUTATION_PATTERN_PROPOSAL_CN.md](../../docs/PURE_COMPUTATION_PATTERN_PROPOSAL_CN.md) - 完整提案（中文）
- [docs/test_case.md](../../docs/test_case.md) - 测试用例规格

## 工具链

- **mlsynthesis**: C 到 MLIR 的初始转换
- **mlir-opt**: MLIR 规范化和优化
- **megg-opt.py**: Megg 优化器（e-graph 匹配和自定义指令生成）
- **RISC-V 工具链**: RISC-V 32-bit 编译和链接

---

**创建日期**: 2025-01-09
**最后更新**: 2025-01-09
**状态**: 已实现并测试
