# ROCm-Agent：面向 AMD GPU 的高性能 HIP 内核生成 Agent

基于 GRPO 强化学习训练 Agent，自动将 PyTorch 模型转化为 AMD GPU 上的高性能 HIP 内核。

---

## 1. 硬件与架构

### 1.1 目标 GPU

| 架构 | 系列 | 显卡 | 显存 | 状态 |
|------|------|------|------|------|
| gfx1100 | RDNA 3 | 4× W7800 | 32 GB 每卡 | **4 卡 vLLM TP=2** |

### 1.2 4 卡角色与设备映射（当前正式命令）

| 物理 GPU | 角色 | 命令映射 | 说明 |
|----------|------|----------|------|
| 0 | vLLM TP shard 0 | `CUDA_VISIBLE_DEVICES=2,3` | 当前机器上 `CUDA 2/3 -> 物理 0/1` |
| 1 | vLLM TP shard 1 | `CUDA_VISIBLE_DEVICES=2,3` | `tensor_parallel_size=2` |
| 2 | 训练 | `CUDA_VISIBLE_DEVICES=0,1 --train-gpu 1` | 当前机器上 `CUDA 1 -> 物理 2` |
| 3 | 评估 | `--eval-gpu 3` | `hip_kernel_interaction.py` 使用物理卡号 |

---

## 2. 4 卡训练数据流

```
                    ┌────────────────────────────────────┐
                    │  Host vLLM Server (物理 GPU 0 + 1)  │
                    │  tools/vllm_serve.py, TP = 2       │
                    │  PagedAttention + Continuous Batch  │
  ┌── HTTP req ───→ │  BF16 autoregressive decoding      │
  │                 └───────────────┬────────────────────┘
  │                                 │ HTTP: completions
  │                                 ↓
  │   ┌──────────────────────────────────────────────────┐
  │   │  Host GPU 2: TRL GRPOTrainer                     │
  │   │  use_vllm=True, vllm_mode="server"               │
  │   │                                                  │
  │   │  1. 发送 batch prompts → vLLM 生成 completions   │
  │   │  2. 提交 completions → CPU 编译 + GPU 3 验证     │
  │   │  3. 收集 rewards                                 │
  │   │  4. Forward → GRPO loss → Backward → 更新 LoRA   │
  │   │  5. HTTP: 同步 LoRA 权重 → vLLM                  │
  │   └──────┬──────────────┬────────────────────────────┘
  │          │              │
  ←── sync ──┘              │ 提交编译/验证任务
                            ↓
              ┌──────────────────────────┐
              │  CPU: 4× hipcc workers   │──→ 编译结果
              │  ProcessPoolExecutor     │
              └──────────────────────────┘
                            │ 编译通过后
                            ↓
              ┌──────────────────────────┐
              │  GPU 3: verify + bench   │──→ 奖励分数 (-1.0 ~ +3.0)
              │  HIP_VISIBLE_DEVICES=3   │
              │  物理隔离，不影响训练     │
              └──────────────────────────┘
```

**当前正式配置**（4 卡流水线）：

| 阶段 | 当前配置 | 说明 |
|------|----------|------|
| LLM 生成 | vLLM TP=2, `max_model_len=8192` | 物理 GPU 0+1 |
| hipcc 编译 | 4× CPU workers | 与生成/训练并行 |
| GPU eval | `--eval-gpu 3` | 物理 GPU 3 |
| 训练 | `--train-gpu 1` | 物理 GPU 2 |

---

## 3. 目录结构

```
ROCm-Agent/
├── tools/                              # 通用脚本（架构无关）
│   ├── train_grpo.py                   #   GRPO 训练（本地/vLLM 双模式）
│   ├── hip_kernel_interaction.py       #   编译/验证/profiling + 分级奖励
│   ├── prepare_data.py                 #   数据预处理 + 参考代码注入
│   ├── compile.py / verify.py / bench.py
│
├── agent_workdir/
│   ├── binding.cpp / binding_registry.h
│   └── gfx1100/                        # RDNA 3：SKILL.md + ref_snippets.py
│
│   ├── vllm_serve.py                  #   vLLM 推理服务启动器（spawn 修复）
│
├── rocm-libraries/                     # 7.2.0 源码（.gitignore）
├── models/ / data/ / checkpoints/ / logs/
```

---

## 4. 模型与训练

### 4.1 模型

| 属性 | 值 |
|------|-----|
| 模型 | `TeichAI/Qwen3-4B-Thinking-2507-DeepSeek-v3.2-Speciale-Code-Distill` |
| LoRA | r=8, alpha=16, q_proj+v_proj, 3.8M 可训练参数 |
| Thinking | 保留模型自带 `chat_template`，不额外修改 |

### 4.2 奖励体系

| 阶段 | 奖励 |
|------|------|
| 无代码 / 部分文件 | -1.0 ~ -0.75 |
| 编译失败（语法/链接） | -0.5 ~ -0.25 |
| 编译通过，验证失败 | 0.0 |
| 验证通过 | +1.0 |
| 比 Eager 快 >5% | +2.0 |
| 比 Eager 和 Compile 都快 >5% | +3.0 |

额外 ±0.1 随机噪声确保 GRPO advantage 非零。

### 4.3 训练参数

| 参数 | 值 |
|------|-----|
| 有效批量 | batch=1 × grad_accum=4 = 4 |
| 生成数 / prompt | num_generations=2 |
| 补全长度 | 128 |
| 温度 | 1.0 |
| 学习率 | 1e-6 |
| 轮次 | 2 epochs（约 1350 updates / epoch） |
| 编译并行 | 4 CPU workers |
| vLLM 上下文 | `max_model_len=8192` |

### 4.4 提示词架构

```
[system] SKILL.md (84行)
  ├─ 输出模板：fused_kernel.hip + fused_kernel_binding.cpp + model_new.py
  └─ 规则：REGISTER_BINDING, extern "C", state_dict 兼容

[user] 指令 + model.py + 参考 HIP 内核（rocm-libraries 按算子匹配注入，70.7% 覆盖）

[assistant] 生成 3 个代码文件
```

### 4.5 数据集

[CUDA-Agent-Ops-6K](https://huggingface.co/datasets/ASKDESC/CUDA-Agent-Ops-6K)（6000 条），split 为 5400 训练 + 600 验证。

---

## 5. 启动命令

```bash
# ═══ 首次准备 ═══

# 安装依赖
pip install -r requirements.txt

# 下载数据集 + 模型 + rocm-libraries
huggingface-cli download ASKDESC/CUDA-Agent-Ops-6K \
  --local-dir data/CUDA-Agent-Ops-6K --repo-type dataset
huggingface-cli download TeichAI/Qwen3-4B-Thinking-2507-DeepSeek-v3.2-Speciale-Code-Distill \
  --local-dir models/Qwen3-4B-Thinking-2507-DeepSeek-v3.2-Speciale-Code-Distill
git clone --depth 1 --branch rocm-7.2.0 \
  https://github.com/ROCm/rocm-libraries.git rocm-libraries

# 准备训练数据
python3 tools/prepare_data.py \
  --input data/CUDA-Agent-Ops-6K/data.parquet \
  --output data/rocm_agent_ops/ --arch gfx1100

# ═══ 安装 vLLM（首次）═══

# 1. 安装 vllm ROCm 预编译包
pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/

# 2. 安装 amdsmi（ROCm 平台检测）
pip install /opt/rocm-7.2.0/share/amd_smi/

# 3. 安装 flash-attn（ROCm 版）
pip install https://wheels.vllm.ai/rocm/bcf2be96120005e9aea171927f85055a6a5c0cf6/flash_attn-2.8.3-cp312-cp312-manylinux_2_35_x86_64.whl

# 4. 安装 vLLM 运行时依赖
pip install -r requirements.txt

# ═══ 4 卡 vLLM 训练（当前正式命令，TP=2） ═══

# 可选：启动前清理残留进程
pkill -9 -f "tools/train_grpo.py" 2>/dev/null || true
pkill -9 -f "tools/vllm_serve.py" 2>/dev/null || true
pkill -9 -f "VLLM::|EngineCore" 2>/dev/null || true

# 终端 1：启动 vLLM 推理服务（物理 GPU 0+1；当前机器上 CUDA 2/3 -> 物理 0/1）
CUDA_VISIBLE_DEVICES=2,3 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
PYTHONUNBUFFERED=1 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
python3 tools/vllm_serve.py \
  --model models/Qwen3-4B-Thinking-2507-DeepSeek-v3.2-Speciale-Code-Distill \
  --tensor_parallel_size 2 \
  --dtype bfloat16 \
  --gpu_memory_utilization 0.82 \
  --max_model_len 8192 \
  --enforce_eager \
  --port 8000 \
  > logs/vllm-teichai-2507-qwen3-4b-tp2.log 2>&1

# 终端 2：启动 GRPO 训练（物理 GPU 2 训练，物理 GPU 3 评估）
# 注意：当前机器上 CUDA_VISIBLE_DEVICES=0,1 中的 --train-gpu 1 对应物理 GPU 2
CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
python3 tools/train_grpo.py \
  --model models/Qwen3-4B-Thinking-2507-DeepSeek-v3.2-Speciale-Code-Distill \
  --use-vllm \
  --vllm-port 8000 \
  --train-gpu 1 \
  --eval-gpu 3 \
  --arch gfx1100 \
  --epochs 2 \
  --batch-size 1 \
  --num-generations 2 \
  --gradient-accumulation 4 \
  --max-completion-length 128 \
  --lr 1e-6 \
  --temperature 1.0 \
  --reward-workers 4 \
  --save-steps 50 \
  --output-dir checkpoints/grpo-teichai-2507-qwen3-4b-tp2 \
  > logs/train-teichai-2507-qwen3-4b-tp2.log 2>&1
```

---

## 6. 路线图

| | 阶段 A（当前） | 阶段 B（修复管线） | 阶段 C（自主 Agent） |
|---|---|---|---|
| 模式 | 单轮生成 | 2 轮（生成+修复） | 模型自主 N 轮 |
| 训练框架 | TRL GRPOTrainer | 继承 GRPOTrainer | 自研训练循环 |
| 改动量 | 已完成 | ~200 行 | ~500 行 |
| Pass rate | 5-15% | 20-40% | 50-80% |

---

## 7. 开源组件

| 组件 | 方案 | 许可证 |
|------|------|--------|
| 基座模型 | `TeichAI/Qwen3-4B-Thinking-2507-DeepSeek-v3.2-Speciale-Code-Distill` | 见模型仓库说明 |
| RL 框架 | TRL GRPOTrainer | Apache 2.0 |
| 微调 | PEFT LoRA | Apache 2.0 |
| 推理加速 | vLLM 0.17.1+rocm700（主机模式，TP=2） | Apache 2.0 |
| 参考代码 | rocm-libraries 7.2.0 | MIT |
| 数据集 | CUDA-Agent-Ops-6K | — |
