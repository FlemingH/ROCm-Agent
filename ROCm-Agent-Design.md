# ROCm-Agent：面向 AMD GPU 的高性能 HIP 内核生成 Agent

基于 GRPO 强化学习训练 Agent，自动将 PyTorch 模型转化为 AMD GPU 上的高性能 HIP 内核。

---

## 1. 硬件与架构

### 1.1 目标 GPU

| 架构 | 系列 | 显卡 | 显存 | 状态 |
|------|------|------|------|------|
| gfx1100 | RDNA 3 | 4× W7800 | 32 GB 每卡 | **推荐（4 卡 Docker vLLM）** |
| gfx1201 | RDNA 4 | R9700 | 34.2 GB | 备选（单卡，需禁用 hipBLASLt） |

通过 `--arch` 参数和 `agent_workdir/<arch>/` 目录自动适配不同架构。

### 1.2 4 卡显存分配（Qwen3-8B BF16）

| GPU | 角色 | 显存 | 说明 |
|-----|------|------|------|
| 0 | vLLM TP shard 0 | ~10 / 32 GB | 模型半体 + KV cache |
| 1 | vLLM TP shard 1 | ~10 / 32 GB | 模型半体 + KV cache |
| 2 | 训练 | ~22 / 32 GB | 模型 16.4GB + LoRA + 梯度 + 激活值 |
| 3 | 评估 | ~2-4 / 32 GB | verify/bench 按需加载，物理隔离 |

---

## 2. 4 卡训练数据流

```
                    ┌────────────────────────────────────┐
                    │  Docker Container (GPU 0 + 1)      │
                    │  trl vllm-serve, TP = 2            │
                    │  PagedAttention + Continuous Batch  │
  ┌── HTTP req ───→ │  BF16 autoregressive decoding      │
  │                 └───────────────┬────────────────────┘
  │                                 │ HTTP: completions (8条)
  │                                 ↓
  │   ┌──────────────────────────────────────────────────┐
  │   │  Host GPU 2: TRL GRPOTrainer                     │
  │   │  use_vllm=True, vllm_mode="server"               │
  │   │                                                  │
  │   │  1. 发送 batch prompts → vLLM 生成 completions   │
  │   │  2. 提交 completions → CPU 编译 + GPU 3 验证     │
  │   │  3. 收集 rewards                                 │
  │   │  4. Forward → GRPO loss → Backward → 更新 LoRA   │
  │   │  5. HTTP: 同步 LoRA 权重 (~7.6MB) → vLLM         │
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

**单步时序**（4 卡 vLLM vs 单卡）：

| 阶段 | 单卡串行 | 4 卡流水线 |
|------|---------|-----------|
| LLM 生成 | ~120s (66%) | ~25-35s (vLLM TP=2) |
| hipcc 编译 | ~40s (22%) | 与生成并行 |
| GPU eval | ~14s (8%) | 与下轮生成并行 |
| 训练 | ~9s (5%) | 与下轮生成并行 |
| **单步总时间** | **~135s** | **~40-50s** |
| **完整训练** | **~50h** | **~15-19h** |

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
│   ├── gfx1100/                        # RDNA 3：SKILL.md + ref_snippets.py
│   └── gfx1201/                        # RDNA 4：SKILL.md + ref_snippets.py
│
├── docker/Dockerfile.vllm              # AMD vLLM + TRL Docker 镜像
├── scripts/start-vllm.sh              # 启动 Docker vLLM 服务
│
├── rocm-libraries/                     # 7.2.0 源码（.gitignore）
├── models/ / data/ / checkpoints/ / logs/
```

---

## 4. 模型与训练

### 4.1 模型

| 属性 | 值 |
|------|-----|
| 模型 | Qwen3-8B (Dense, BF16) |
| LoRA | r=8, alpha=16, q_proj+v_proj, 3.8M 可训练参数 |
| Thinking | 禁用（修改 chat_template 强制跳过） |

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
| 有效批量 | batch=2 × grad_accum=4 = 8 |
| 生成数 / prompt | num_generations=4 |
| 补全长度 | 1024 |
| 温度 | 1.0 |
| 学习率 | 1e-6 |
| 轮次 | 2 epochs (1350 steps) |
| 编译并行 | 4 CPU workers |
| 预估时间 | 4 卡 ~15-19h / 单卡 ~50h |

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
huggingface-cli download Qwen/Qwen3-8B --local-dir models/Qwen3-8B
git clone --depth 1 --branch rocm-7.2.0 \
  https://github.com/ROCm/rocm-libraries.git rocm-libraries

# 禁用 Qwen3 thinking
python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('models/Qwen3-8B')
old = \"{%- if enable_thinking is defined and enable_thinking is false %}\\n        {{- '<think>\\\\n\\\\n</think>\\\\n\\\\n' }}\\n    {%- endif %}\"
tok.chat_template = tok.chat_template.replace(old, \"{{- '<think>\\\\n\\\\n</think>\\\\n\\\\n' }}\")
tok.save_pretrained('models/Qwen3-8B')
"

# 准备训练数据
python3 tools/prepare_data.py \
  --input data/CUDA-Agent-Ops-6K/data.parquet \
  --output data/rocm_agent_ops/ --arch gfx1100

# ═══ 4 卡 Docker vLLM 训练（推荐）═══

bash scripts/start-vllm.sh models/Qwen3-8B 2    # Docker vLLM on GPU 0+1
python3 tools/train_grpo.py \
  --model models/Qwen3-8B \
  --use-vllm --train-gpu 2 --eval-gpu 3 \
  --arch gfx1100 \
  --epochs 2 --batch-size 2 --num-generations 4 \
  --gradient-accumulation 4 --lr 1e-6 --temperature 1.0 \
  --reward-workers 4 --save-steps 50 \
  --output-dir checkpoints/grpo-8b-vllm \
  2>&1 | tee logs/train-vllm.log
docker stop rocm-agent-vllm && docker rm rocm-agent-vllm

# ═══ 单卡训练（R9700 gfx1201 备选）═══

sudo mv /opt/rocm-7.2.0/lib/libhipblaslt.so.1 /opt/rocm-7.2.0/lib/libhipblaslt.so.1.bak
python3 tools/train_grpo.py \
  --model models/Qwen3-8B --arch gfx1201 \
  --epochs 2 --batch-size 2 --num-generations 4 \
  --gradient-accumulation 4 --lr 1e-6 --temperature 1.0 \
  --reward-workers 4 --save-steps 50 \
  --output-dir checkpoints/grpo-8b-final \
  2>&1 | tee logs/train.log
sudo mv /opt/rocm-7.2.0/lib/libhipblaslt.so.1.bak /opt/rocm-7.2.0/lib/libhipblaslt.so.1
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

## 7. 风险与缓解

| 风险 | 缓解 |
|------|------|
| 生成瓶颈（单卡 66%） | Docker vLLM TP=2 加速 3-5× |
| hipBLASLt gfx1201 崩溃 | 单卡禁用 libhipblaslt.so；gfx1100 无此 bug |
| HIP 内核越界写入训练 GPU | --eval-gpu 物理隔离到 GPU 3 |
| gfx1100 无 Flash Attention | --enforce-eager 禁用 CUDA Graph |
| Docker 与主机 GPU 冲突 | ROCR_VISIBLE_DEVICES 隔离 |
| 8B 模型代码能力有限 | 注入 rocm-libraries 参考代码 + 分级奖励 |
| 采样多样性不足 | temperature=1.0 + reward noise ±0.1 |

---

## 8. 开源组件

| 组件 | 方案 | 许可证 |
|------|------|--------|
| 基座模型 | Qwen3-8B (BF16) | Qwen License |
| RL 框架 | TRL GRPOTrainer | Apache 2.0 |
| 微调 | PEFT LoRA | Apache 2.0 |
| 推理加速 | vLLM Docker (rocm/vllm-dev navi) | Apache 2.0 |
| 参考代码 | rocm-libraries 7.2.0 | MIT |
| 数据集 | CUDA-Agent-Ops-6K | — |
