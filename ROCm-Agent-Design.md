# ROCm-Agent：面向 AMD GPU 的高性能 HIP 内核生成 Agent

## 1. 项目概述

基于强化学习训练 Agent，自动将 PyTorch 模型转化为 AMD GPU 上的高性能 HIP 内核。参考 CUDA-Agent 论文的 Agent 循环（实现 → 编译 → 验证 → 性能分析 → 迭代优化），适配 ROCm 7.x 工具链。

### 1.1 目标架构：gfx1201（RDNA 4）

当前版本专注于 **gfx1201（RDNA 4, R9700）** 单一架构优化。

设计理由：

1. **单机闭环**：R9700（32GB）可同时完成模型训练和 HIP 内核编译/验证/profiling，无需多机协调
2. **RDNA 4 新特性**：原生支持 Flash Attention（ck_tile/01_fmha/）、完整 Tile GEMM、改进的 WMMA 和 GDDR7 带宽
3. **训练效率**：Agent 专注单一架构，可在有限 RL 训练中更快收敛

**未来扩展**：为新架构创建 `<arch>/SKILL.md` + `<arch>/HIP_REFS.md`，复用相同训练框架。

### 1.2 硬件规格

| 规格 | 参数 |
|------|------|
| GPU | AMD Radeon R9700（RDNA 4, gfx1201） |
| 显存 | 32GB GDDR7 |
| 线程模型 | Wavefront = 32（与 CUDA Warp 一致） |
| 矩阵加速 | WMMA（AI Accelerators，改进版） |
| 软件栈 | ROCm 7.x + hipcc |

**显存分配**：

| 项目 | 显存 |
|------|------|
| GPTQ 4-bit 模型（30B MoE） | ~18 GB |
| LoRA 训练（optimizer + activations） | ~4 GB |
| HIP 内核评估（临时，测试模型很小） | ~0.5 GB |
| **峰值** | **~22.5 GB / 32 GB** |

### 1.3 内核实现策略

**优先级 1：AMD 库融合调用**——hipBLASLt 完成 GEMM+Bias+Activation 一步融合

**优先级 2：AMD 库单算子调用**——rocBLAS（GEMM）、MIOpen（Conv/BN/Pooling/Softmax/LN）、rocPRIM（Reduction/Sort）

**优先级 3：手写 HIP 内核**——库函数无法覆盖的跨算子融合

---

## 2. 系统架构

### 2.1 单机训练流程

```
R9700 (单卡 32GB, gfx1201)
┌──────────────────────────────────────────────┐
│  TRL GRPOTrainer                             │
│    1. GPTQ 推理：生成 HIP 内核代码            │
│    2. 本地评估：                              │
│       ├─ hipcc compile (CPU)                 │
│       ├─ verify correctness (GPU, ~0.5GB)    │
│       └─ profile performance (GPU, ~0.5GB)   │
│    3. GRPO 更新：LoRA 梯度回传               │
└──────────────────────────────────────────────┘
```

训练循环：

1. TRL GRPOTrainer 用 GPTQ 模型生成 HIP 内核代码
2. 本地调用 HipKernelInteraction 执行 hipcc 编译 + 验证 + profiling
3. 返回真实奖励（compile/verify/profile 结果）
4. GRPO 梯度更新（LoRA）

### 2.2 目录结构

```
ROCm-Agent/
├── ROCm-Agent-Design.md            # 本文档
├── requirements.txt                # Python 依赖
├── LICENSE
├── .gitignore
│
├── tools/                          # ── 所有脚本工具 ──
│   ├── train_grpo.py               #   GRPO 训练（单卡 GPTQ + LoRA + 本地评估）
│   ├── hip_kernel_interaction.py   #   HIP 内核交互代理（compile→verify→profile→reward）
│   ├── compile.py                  #   hipcc 编译
│   ├── compile.sh                  #   编译包装器
│   ├── verify.py                   #   正确性验证
│   ├── bench.py                    #   性能对比（Eager / torch.compile / HIP）
│   └── prepare_data.py             #   数据预处理：dataset → 训练格式
│
├── agent_workdir/
│   ├── gfx1201/                    # ── RDNA 4 架构目录 ──
│   │   ├── SKILL.md                #   Agent 行为指令
│   │   ├── HIP_REFS.md             #   示例索引（含兼容性标注）
│   │   ├── model_new.py            #   Agent 输出：优化后的模型
│   │   └── kernels/                #   Agent 输出：HIP 内核
│   │       ├── *.hip
│   │       └── *_binding.cpp
│   ├── binding.cpp                 # ── 共享基础设施 ──
│   ├── binding_registry.h
│   └── model.py                    #   原始 PyTorch 模型（只读，每道题变化）
│
├── models/                         # 模型权重（.gitignore）
├── data/                           # 训练数据（.gitignore）
├── checkpoints/                    # 训练检查点（.gitignore）
└── rocm-libraries/                 # ROCm 官方库参考（.gitignore）
```

### 2.3 架构配置文件

| 文件 | 作用 |
|------|------|
| `gfx1201/SKILL.md` | Agent 行为指令：三级优先级 + RDNA 4 硬件优化清单 + gfx1201 兼容的参考代码索引 |
| `gfx1201/HIP_REFS.md` | 完整示例索引：标注每个示例对 gfx1201 的兼容性（✅/⚠️），按优先级和训练难度分类 |

扩展到新架构时，只需创建对应的 `<arch>/SKILL.md` + `<arch>/HIP_REFS.md` 目录。

### 2.4 参考代码来源：rocm-libraries

使用 [ROCm/rocm-libraries rocm-7.2.0](https://github.com/ROCm/rocm-libraries/releases/tag/rocm-7.2.0) 作为 HIP 编程参考。

### 2.5 CUDA → HIP 速查表

| CUDA | HIP (RDNA 4) |
|------|---------------|
| `#include <cuda_runtime.h>` | `#include <hip/hip_runtime.h>` |
| `cudaStream_t` | `hipStream_t` |
| `__syncthreads()` / `<<<blocks, threads>>>` | 不变 |
| warp = 32 | **wavefront = 32（一致）** |
| `__shfl_sync(mask, val, src)` | `__shfl(val, src)` |
| cuBLAS / cuDNN / CUB | rocBLAS+hipBLASLt / MIOpen / rocPRIM |
| nvcc / `TORCH_CUDA_ARCH_LIST=9.0` | hipcc / `--offload-arch=gfx1201` |

---

## 3. 数据集：ROCm-Agent-Ops

RL 训练只需"题目"（model.py），不需要 HIP 参考答案。Agent 从编译/验证/profiling 环境反馈中自主学习。

**构建流水线**：

1. **种子算子挖掘**：从 torch/transformers 库挖掘，复用 CUDA-Agent-Ops-6K
2. **LLM 组合合成**：最多 5 个算子顺序组合为融合任务
3. **R9700 上执行过滤**：Eager/Compile 可执行、确定性、非平凡输出、运行时间 1-100ms

**课程学习标注**：简单（单算子）~30%、中等（2-3 算子含 GEMM/Conv）~45%、困难（4-5 算子或完整模块）~25%

**规模**：~4,500-5,000 条 model.py

---

## 4. 训练方案

### 4.1 双模型对照实验（MoE vs Dense）

| 属性 | 模型 A：MoE 代码专精 | 模型 B：Dense 通用最强代码 |
|------|----------------------|--------------------------|
| 模型 | **Qwen3-Coder-30B-A3B** Q4 | **Qwen3.5-27B** Q4 |
| 架构 | MoE（128 专家，8 激活） | **Dense（全参激活）** |
| 总参数 / 激活参数 | 30B / **3B** | 27B / **27B** |
| 量化显存 | ~18GB | ~17GB |
| 训练峰值 | ~22GB（R9700 32GB 内） | ~21GB |
| 许可证 | Apache 2.0 | Apache 2.0 |

### 4.2 训练架构（单机）

```
R9700 (单卡 32GB, gfx1201)
├── GPTQ 4-bit 模型 (~18GB)
├── LoRA 训练层 (~4GB)
├── TRL GRPOTrainer (生成 + 训练)
└── HipKernelInteraction (本地 compile/verify/profile)
```

奖励函数（真实编译/运行结果）：

| 条件 | 奖励 |
|------|------|
| 编译或验证失败 | -1 |
| 正确输出 | +1 |
| 比 Eager 基线快 >5% | +2 |
| 比 Eager 和 Compile 都快 >5% | +3 |

### 4.3 训练启动命令

```bash
conda activate ra
cd /path/to/ROCm-Agent

# 1. 准备数据
python3 tools/prepare_data.py \
  --input data/CUDA-Agent-Ops-6K/data.parquet \
  --output data/rocm_agent_ops/ \
  --arch gfx1201 \
  --skill-path agent_workdir/gfx1201/SKILL.md

# 2. 启动训练
python3 tools/train_grpo.py \
  --model models/Qwen3-Coder-30B-A3B-Instruct \
  --batch-size 2 \
  --num-generations 4 \
  --gradient-accumulation 4 \
  --max-completion-length 2048 \
  --lr 1e-6 \
  --epochs 5
```

---

## 5. 评估方案

**基准**：自建 ROCm-Bench，从数据集划分 500 条测试集在 R9700 上评估。

**指标**：Pass Rate、Faster Rate vs. Compile、Geomean Speedup。

**目标**：Pass Rate >85%，Faster Rate vs. Compile >50%，Speedup >1.2x。

---

## 6. 里程碑

| 阶段 | 交付物 |
|------|--------|
| **M1: 环境搭建** | ROCm 7.x 工具链在 R9700 上跑通，compile/verify/profile 验证通过 |
| **M2: 数据集** | ROCm-Agent-Ops（~4500 条 model.py） |
| **M3: 模型 A 训练** | MoE 30B Q4 GRPO 训练 |
| **M4: 模型 B 训练** | Dense 27B Q4 GRPO 训练 |
| **M5: 评估开源** | A/B 对比报告、最优模型和数据集开源 |

---

## 7. 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| R9700 ROCm 兼容性（RDNA 4 较新） | M1 全面验证 PyTorch + GPTQ 加载；Docker 固定 ROCm 版本 |
| 单卡显存紧张（训练 + 评估同时） | gradient_checkpointing=True；batch_size=2；评估模型很小 |
| ROCm 库架构支持不统一 | 按架构分目录，标注兼容性；Agent 仅引用 gfx1201 兼容示例 |
| Agent 手写内核难度高 | 课程学习渐进；标准算子优先调库 |
| 编译/profiling 与训练争抢 GPU | 评估阶段在训练间隙运行，不并发；测试模型显存占用小 |
| 训练不稳定 | 课程学习（简单→困难），监控 KL 散度和奖励曲线 |

---

## 8. 开源组件

| 组件 | 方案 | 许可证 |
|------|------|--------|
| 基座模型 A | Qwen3-Coder-30B-A3B（Q4，MoE） | Apache 2.0 |
| 基座模型 B | Qwen3.5-27B（Q4，Dense） | Apache 2.0 |
| RL 框架 | TRL (GRPOTrainer) + accelerate | Apache 2.0 |
| 量化 | GPTQ（auto-gptq） | MIT |
| 微调 | PEFT (LoRA) | Apache 2.0 |
| AMD 库 | rocBLAS / hipBLASLt / MIOpen / rocPRIM | MIT |
| 数据集 | CUDA-Agent-Ops-6K（复用） | — |
| 参考代码 | rocm-libraries rocm-7.2.0 | MIT |
