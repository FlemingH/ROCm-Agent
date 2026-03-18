# ROCm-Agent：面向 AMD GPU 的高性能 HIP 内核生成 Agent

## 1. 项目概述

基于强化学习训练 Agent，自动将 PyTorch 模型转化为 AMD GPU 上的高性能 HIP 内核。参考 CUDA-Agent 论文的 Agent 架构，适配 ROCm 7.x 工具链，在单卡 R9700 (32GB) 上完成训练。

### 1.1 目标架构：gfx1201（RDNA 4）

当前版本专注于 **gfx1201（RDNA 4, R9700）** 单一架构优化。

**未来扩展**：为新架构创建 `agent_workdir/<arch>/SKILL.md` + `<arch>/ref_snippets.py`，tools 目录通过 `--arch` 参数自动适配。

### 1.2 硬件规格

| 规格 | 参数 |
|------|------|
| GPU | AMD Radeon AI PRO R9700（RDNA 4, gfx1201） |
| 显存 | 34.2 GB GDDR7 |
| 线程模型 | Wavefront = 32 |
| 软件栈 | ROCm 7.2 + hipcc + PyTorch 2.9.1+rocm7.2.0 |

**显存分配（Qwen3-8B BF16 训练）**：

| 项目 | 显存 |
|------|------|
| 模型权重 (8.2B × BF16) | 16.4 GB |
| LoRA + 优化器 | ~0.1 GB |
| 梯度 + 激活值 (gradient checkpointing) | ~5-8 GB |
| KV Cache (生成阶段) | ~3-5 GB |
| **峰值** | **~27-29 GB / 34.2 GB** |

---

## 2. 系统架构

### 2.1 训练流程

```
R9700 (单卡 34.2GB, gfx1201)
┌──────────────────────────────────────────────────────┐
│  TRL GRPOTrainer                                     │
│    1. BF16 推理：生成 HIP 内核代码 (4 completions)     │
│    2. 并行评估 (4 workers)：                          │
│       ├─ hipcc compile → 分级奖励 (-0.9 ~ -0.25)     │
│       ├─ verify correctness → 奖励 0.0               │
│       └─ profile performance → 奖励 +1 ~ +3          │
│    3. GRPO 更新：LoRA 梯度回传                        │
└──────────────────────────────────────────────────────┘
```

### 2.2 提示词架构

```
[system] SKILL.md (84行)
  ├─ 输出格式模板：.hip + _binding.cpp + model_new.py
  └─ 规则：REGISTER_BINDING, extern "C", state_dict 兼容

[user]
  ├─ 指令 + model.py 代码（来自 CUDA-Agent-Ops-6K）
  └─ 参考 HIP 内核（来自 rocm-libraries，按算子匹配注入，70.7% 覆盖率）

[assistant] <think></think>（thinking 禁用）→ 生成 3 个代码文件
```

### 2.3 目录结构

```
ROCm-Agent/
├── ROCm-Agent-Design.md
├── requirements.txt
├── LICENSE / .gitignore
│
├── tools/                              # 通用脚本（架构无关）
│   ├── train_grpo.py                   #   GRPO 训练 + 并行 reward
│   ├── hip_kernel_interaction.py       #   编译/验证/profiling + 分级奖励
│   ├── prepare_data.py                 #   数据预处理 + 参考代码注入
│   ├── compile.py                      #   hipcc 编译
│   ├── verify.py                       #   正确性验证
│   └── bench.py                        #   性能对比
│
├── agent_workdir/
│   ├── binding.cpp                     # 共享基础设施
│   ├── binding_registry.h
│   ├── model.py                        # 原始模型（只读）
│   └── gfx1201/                        # RDNA 4 架构配置
│       ├── SKILL.md                    #   输出格式模板 + 规则
│       ├── ref_snippets.py             #   从 rocm-libraries 实时提取参考代码
│       ├── model_new.py                #   Agent 输出（运行时生成）
│       └── kernels/                    #   Agent 输出（运行时生成）
│
├── rocm-libraries/                     # rocm-libraries 7.2.0 源码（.gitignore）
├── models/                             # 模型权重（.gitignore）
├── data/                               # 训练数据（.gitignore）
├── checkpoints/                        # 训练检查点（.gitignore）
└── logs/                               # 训练日志（.gitignore）
```

### 2.4 参考代码注入

`ref_snippets.py` 从 rocm-libraries 7.2.0 实时提取 29 个代码片段，按训练样本的算子类型匹配注入 prompt：

| 来源 | 片段数 | 覆盖算子 |
|------|--------|---------|
| composablekernel (CK) | 14 | 激活函数、LayerNorm、RMSNorm、MaxPool |
| miopen | 4 | BatchNorm、Softmax、GroupNorm、Conv |
| rocprim + rocblas | 2 | Reduce、GEMM |
| 硬编码简单内核 | 9 | clamp/where/pow/sign/leaky_relu/hardswish 等 |

覆盖率：70.7% 的训练样本有参考代码注入。

### 2.5 CUDA → HIP 速查表

| CUDA | HIP (RDNA 4) |
|------|---------------|
| `#include <cuda_runtime.h>` | `#include <hip/hip_runtime.h>` |
| `cudaStream_t` | `hipStream_t` |
| `__syncthreads()` / `<<<blocks, threads>>>` | 不变 |
| warp = 32 | wavefront = 32（一致） |
| `c10::cuda::getCurrentCUDAStream()` | 不变（ROCm 兼容层） |
| `input.is_cuda()` | 不变（ROCm 下返回 true） |

---

## 3. 数据集

复用 [CUDA-Agent-Ops-6K](https://huggingface.co/datasets/ASKDESC/CUDA-Agent-Ops-6K)（6000 条 model.py），经 `prepare_data.py` 处理为训练格式：

- 训练集：5400 条，验证集：600 条
- 难度分布：easy 4.6% / medium 91.3% / hard 4.0%
- 每条样本包含：system prompt (SKILL.md) + user prompt (model.py + 参考代码)

---

## 4. 训练方案

### 4.1 模型选型

| 属性 | 当前方案 |
|------|---------|
| 模型 | **Qwen3-8B** (Dense) |
| 精度 | BF16（原生，无量化） |
| 显存 | 16.4 GB 模型 + ~12 GB 训练 = ~29 GB 峰值 |
| LoRA | r=8, alpha=16, target=q_proj+v_proj, 3.8M 可训练参数 |
| Thinking 模式 | 禁用（修改 chat_template 强制跳过） |

选型理由：
- 8B Dense BF16 比 30B MoE GPTQ 有更高的 GPU 利用率和采样多样性
- 8B 模型能产生有效的 RL 信号（reward 从 -1.0 提升到 +0.02）
- 训练时间 ~20 小时（vs 30B 的 5 周）

### 4.2 奖励体系（分级）

| 阶段 | 奖励 | 说明 |
|------|------|------|
| 无代码输出 | -1.0 | |
| 部分文件 | -0.9 ~ -0.75 | 按文件完整度 |
| 3 文件齐全，编译失败 | -0.5 ~ -0.25 | 语法错误 vs 链接错误 |
| 编译通过，验证失败 | 0.0 | |
| 验证通过 | +1.0 | |
| 比 Eager 快 >5% | +2.0 | |
| 比 Eager 和 Compile 都快 >5% | +3.0 | |

额外：reward ±0.1 随机噪声，确保 GRPO advantage 非零。

### 4.3 训练配置

| 参数 | 值 |
|------|-----|
| 有效批量 | batch=2 × grad_accum=4 = 8 |
| 每 prompt 生成数 | num_generations=4 |
| 补全长度 | max_completion_length=1024 |
| 采样温度 | temperature=1.0 |
| 学习率 | 1e-6 |
| 训练轮次 | 2 epochs (1350 steps) |
| 奖励并行 | 4 ProcessPoolExecutor workers |
| 保存间隔 | 每 100 步 |
| 预估时间 | ~20 小时 |

### 4.4 训练启动命令

```bash
conda activate ra
cd /home/test/ROCm-Agent

# 0. 安装依赖（首次运行）
pip install trl accelerate transformers peft safetensors tokenizers \
  sentencepiece datasets huggingface_hub pyarrow ninja cmake

# 1. 下载数据集、模型和 rocm-libraries（首次运行）
huggingface-cli download ASKDESC/CUDA-Agent-Ops-6K \
  --local-dir data/CUDA-Agent-Ops-6K --repo-type dataset
huggingface-cli download Qwen/Qwen3-8B \
  --local-dir models/Qwen3-8B
git clone --depth 1 --branch rocm-7.2.0 \
  https://github.com/ROCm/rocm-libraries.git rocm-libraries

# 2. 禁用 Qwen3 thinking 模式（首次运行）
python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B')
old = \"{%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\\\n\\\\n</think>\\\\n\\\\n' }}\n    {%- endif %}\"
tok.chat_template = tok.chat_template.replace(old, \"{{- '<think>\\\\n\\\\n</think>\\\\n\\\\n' }}\")
tok.save_pretrained('models/Qwen3-8B')
"

# 3. 准备数据（参考代码自动从 rocm-libraries 注入）
python3 tools/prepare_data.py \
  --input data/CUDA-Agent-Ops-6K/data.parquet \
  --output data/rocm_agent_ops/ \
  --arch gfx1201

# 4. 启动训练（在 tmux 中运行）
tmux new -s train
python3 tools/train_grpo.py \
  --model models/Qwen3-8B \
  --epochs 2 \
  --batch-size 2 \
  --num-generations 4 \
  --gradient-accumulation 4 \
  --max-completion-length 1024 \
  --lr 1e-6 \
  --temperature 1.0 \
  --reward-workers 4 \
  --save-steps 100 \
  --output-dir checkpoints/grpo-8b-final \
  2>&1 | tee logs/train.log
```

---

## 5. 评估方案

**基准**：自建 ROCm-Bench，从数据集划分 500 条测试集在 R9700 上评估。

**指标**：Pass Rate、Faster Rate vs. Compile、Geomean Speedup。

**目标**：Pass Rate >85%，Faster Rate vs. Compile >50%，Speedup >1.2x。

---

## 6. 优化路线图：A → B → C 三阶段

### 6.1 总览

| | 阶段 A（当前） | 阶段 B（硬编码修复） | 阶段 C（自主 Agent） |
|---|---|---|---|
| 模式 | 单轮生成 | 固定 2 轮（生成+修复） | 模型自主 N 轮 |
| 谁控制流程 | 无决策 | 代码硬编码 | 模型决策 |
| 模型能力 | 从零写代码 | + 看错误修代码 | + 调试/搜索/优化 |
| 训练框架 | TRL GRPOTrainer | 继承 GRPOTrainer | 自研训练循环 |
| 每步时间 | ~140s | ~210s | ~700s |
| 总训练时间 | ~52h | ~78h | ~260h |
| 改动量 | 已完成 | ~200 行 | ~500 行新代码 |
| Pass rate 预期 | 5-15% | 20-40% | 50-80% |

### 6.2 阶段决策流程

```
阶段 A 训练完成（~52h）
    ↓
  评估 pass rate
    ├─ > 15% → 写推理脚本做多轮推理（路径 1，零成本），可能够用
    └─ < 15% → 进入阶段 B
                  ↓
               收集失败样本 + 实施硬编码修复 + 重新训练（~78h）
                  ↓
               评估 pass rate
                  ├─ > 30% → 够用
                  └─ < 30% → 进入阶段 C（自研训练循环，~260h）
```

### 6.3 阶段 A：单轮生成（当前）

```
[prompt] → model.generate() → [v1 代码] → evaluate() → reward → GRPO update
```

参见第 4 节训练方案。

### 6.4 阶段 B：硬编码修复管线

**A → B 需要收集的数据**：

用阶段 A 训练后的 LoRA 模型，对数据集中每个 model.py 生成代码，记录三元组：

```
(model.py, v1 代码, 编译/验证反馈, reward)
```

- 新建 `tools/collect_trajectories.py` 实现
- 目标：~3000 条失败样本 + ~500 条成功样本

**A → B 需要修改的代码**：

| 文件 | 修改内容 |
|------|---------|
| `tools/prepare_data.py` | 支持构造修复 prompt：70% 单轮 + 30% 修复样本 |
| `tools/train_grpo.py` | 继承 GRPOTrainer，对 reward<0 的补全触发修复轮 |
| 新建 `tools/collect_trajectories.py` | 收集失败/成功样本 |
| `tools/hip_kernel_interaction.py` | 无需修改（已支持多轮） |

**修复轮流程**（硬编码在 train_grpo.py 中）：

```
v1 = model.generate(prompt)
r1, feedback, stage = evaluate(v1)

if r1 >= 1.0:                        # 已通过 → 不修复
    return r1
elif stage in ("compile_error", "verify_error"):
    fix_prompt = prompt + [assistant: v1] + [user: feedback]
    v2 = model.generate(fix_prompt)   # 模型修复
    r2, _, _ = evaluate(v2)
    return max(r1, r2)               # 取更好的 reward
else:
    return r1
```

从 CUDA-Agent 保留的能力：**调试修复**（看错误 → 修代码）
去掉的能力：自主搜索、参数调优、回滚修复（硬编码替代模型决策）

### 6.5 阶段 C：自主 Agent（远期）

**B → C 需要的代码**：

| 文件 | 内容 |
|------|------|
| 新建 `tools/train_agent_rl.py` | 自研 PPO/GRPO 训练循环，支持多轮 episode（~500 行） |
| 新建 `tools/agent_tools.py` | BashTool / EditTool / GlobTool 实现 |
| `agent_workdir/gfx1201/SKILL.md` | 扩展为完整工作流手册（类似 CUDA-Agent 375 行版） |

**Agent 循环**（类似 CUDA-Agent 的 ReAct 模式）：

```
while not done and turn < max_turns:
    action = model.generate(conversation)        # 模型决定下一步
    if action.type == "bash":
        result = execute_bash(action.command)     # 编译/验证/profiling
    elif action.type == "edit":
        result = edit_file(action.path, action.content)
    conversation.append(action, result)
    done, reward = check_completion()

# RL 更新：对整条轨迹计算 advantage
loss = -advantage * sum(log_probs_all_turns)
```

**预期**：模型学会完整的开发流程（分析→实现→调试→优化），接近 CUDA-Agent 论文效果。
**代价**：自研训练循环，放弃 TRL 便利性，训练时间 5x+。

---

## 7. 与 CUDA-Agent 的差异（阶段 A 视角）

| | CUDA-Agent | ROCm-Agent |
|---|---|---|
| 硬件 | Nvidia (CUDA) | **AMD (HIP/ROCm)** |
| 模型 | 30B+ MoE GPTQ | **8B Dense BF16** |
| SKILL.md | 375 行完整操作手册 | **84 行格式模板 + 规则** |
| 内核参考 | 无（模型自主推理） | **从 rocm-libraries 实时注入（70.7% 覆盖）** |
| 交互模式 | 多轮迭代（最多 20 轮） | **单轮生成**（TRL GRPO 限制） |
| 奖励体系 | 二元（-1/+1/+2/+3） | **分级（-1.0 到 +3.0，8 级）** |
| 训练框架 | 自研训练循环 | **TRL GRPOTrainer** |
| 训练时间 | 多卡多天 | **单卡 ~20 小时** |

---

## 8. 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| 8B 模型代码生成能力有限 | 注入 rocm-libraries 参考代码；分级奖励降低学习门槛 |
| 单轮生成无法迭代修复 | 未来可展开为多轮 prompt（方向 2）或换多轮 RL 框架 |
| 采样多样性不足（低 entropy） | temperature=1.0 + reward noise ±0.1 |
| Qwen3 thinking 消耗 tokens | 修改 chat_template 强制禁用 |
| GPTQ 在 ROCm 上只有 TorchQuantLinear | 改用 BF16 原生模型，避开量化后端限制 |

---

## 9. 开源组件

| 组件 | 方案 | 许可证 |
|------|------|--------|
| 基座模型 | Qwen3-8B (BF16, Dense) | Qwen License |
| RL 框架 | TRL (GRPOTrainer) + accelerate | Apache 2.0 |
| 微调 | PEFT (LoRA) | Apache 2.0 |
| 参考代码 | rocm-libraries 7.2.0 (CK, MIOpen, rocPRIM, rocBLAS) | MIT |
| 数据集 | CUDA-Agent-Ops-6K（复用） | — |
