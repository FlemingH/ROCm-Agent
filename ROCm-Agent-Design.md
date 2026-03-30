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
| 3 | 评估 | `--eval-gpu 0` | `hip_kernel_interaction.py` 使用评测 GPU |

*(注：上述映射保证了 vLLM 和 训练/评估进程在显存和算力上严格物理隔离，避免 OOM 和相互干扰。)*

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
              │  物理隔离，不影响训练     │
              └──────────────────────────┘
```

---

## 3. 核心机制设计

### 3.1 模型简化输出 (单文件合约)
为降低小模型的认知负荷，防止生成内容超长被截断，模型被要求**只输出 1 个文件**：`fused_kernel.hip`。
系统的 `hip_kernel_interaction.py` 会在后台自动完成以下后处理：
1. 自动注入 `#include <hip/hip_runtime.h>`。
2. 自动合成 `fused_kernel_binding.cpp`（PyTorch C++ 扩展胶水代码）。
3. 自动正则替换原始模型生成 `model_new.py`。

### 3.2 保姆级提示词 (SKILL.md)
* **代码骨架**：直接提供带 `__launch_bounds__(256)` 的 `__global__` 函数与 `extern "C"` 启动器分离的完美模板。
* **STRICTLY FORBIDDEN**：设立严格红线（禁止嵌套函数、禁止动态内存分配、禁止引入 torch 头文件）。
* **gfx1100 架构规则**：明确标注 wavefront=32、LDS 64KB/CU（≤32KB 保证双 workgroup 占用率）、dispatch gap ≈1.7µs（鼓励 kernel fusion）、避免寄存器溢出。
* **共享内存归约模板**：提供完整的 tree reduction 代码（`__shared__ float sdata[256]` + `__syncthreads`），直接适用于 softmax、layernorm、sum、max 等需要归约的算子。
* **性能优化指南**：教授模型使用 `float4` 向量化访存、`#pragma unroll`、快速数学函数（`__expf`、`__fdividef`、`__frsqrt_rn`）、以及 warp-level `__shfl_xor` 归约。
* **Token 预算**：SKILL.md 约 1232 tokens（系统提示），最坏情况含参考代码的 prompt 约 5513 tokens，加上 1024 补全长度 = 6537 tokens，在 8192 的 vLLM 上下文内有 1655 tokens 余量。

### 3.3 极简参考代码 (ref_snippets.py)
摒弃了复杂的 AMD 官方宏定义源码，改为提供硬编码的、与 `SKILL.md` 模板完美契合的 C++ 算子实现示例，帮助模型快速跨过 0.0 分的编译鸿沟。

**覆盖的算子类别**：
| 类别 | 算子示例 | 实现方式 |
|------|----------|----------|
| 逐元素激活 | ReLU, SiLU, GELU, Sigmoid, Tanh, ELU | `__launch_bounds__(256)` + grid-stride loop + 快速数学 (`__expf`, `__fdividef`) |
| 归一化 | LayerNorm, RMSNorm, BatchNorm, GroupNorm | 共享内存 2-pass 归约（mean→variance→normalize / sum_sq→inv_rms） |
| Softmax | softmax, log_softmax | 数值稳定 3-pass 归约（max→sum_exp→normalize）使用 `__shared__` |
| 归约 | torch.sum, torch.mean | 共享内存 tree reduction |
| 线性/卷积 | Linear, Conv1d/2d/3d, GEMM | 权重近似（sandbox 约束下的简化实现） |

**关键改进**（相较初始版本）：
* Softmax 从错误的单 `expf()` 改为数值稳定的 3-pass 共享内存实现
* LayerNorm/RMSNorm 从空操作 `output=input` 改为正确的共享内存归约实现
* 所有 kernel 模板加入 `__launch_bounds__(256)` 引导编译器优化寄存器分配
* 激活函数使用 `__expf`/`__fdividef` 快速数学替代标准库函数

---

## 4. 模型与训练参数

### 4.1 模型

| 属性 | 值 |
|------|-----|
| 模型 | `janhq/Jan-code-4b` |
| 特点 | 针对代码生成的专精模型，C++ 遵循能力极强 |
| LoRA | r=8, alpha=16, q_proj+v_proj |

### 4.2 奖励体系

| 阶段 | 奖励 | 说明 |
|------|------|------|
| 无代码 / 部分文件 | -1.0 | 格式错误，未提取到代码块 |
| 编译失败 (Syntax) | -0.5 | 存在基础的 C++ 语法错误 |
| 编译失败 (Linker) | -0.25 | 语法正确，但入口函数名 (`launch_my_kernel`) 错误 |
| 编译通过，验证失败 | 0.0 | 成功上机，但计算逻辑或数值精度不对 |
| 验证通过 | +1.0 | 逻辑正确，结果与 PyTorch 完全一致 |
| 比 Eager 快 >5% | +2.0 | 性能超越原生 PyTorch 底层算子 |
| 极致优化 | +3.0 | 性能超越原生，并击败 `torch.compile` |

### 4.3 训练参数 (防 OOM 极限配置)

| 参数 | 值 | 说明 |
|------|-----|------|
| 有效批量 | 8 | `batch=1` × `grad_accum=8` |
| 生成数 / prompt | `num_generations=4` | 每个 prompt 采样 4 次 |
| 补全长度 | `max_completion_length=1024` | 配合单文件输出和精准停机足够 |
| 温度 | `0.5` | 鼓励在固定模板内探索不同的数学优化写法 |
| vLLM 上下文 | `max_model_len=8192` | vLLM 独立推理，不受训练显存限制 |

---

## 5. 启动命令

```bash
# ═══ 首次准备 ═══

# 安装依赖
pip install -r requirements.txt

# 下载数据集 + 模型
huggingface-cli download ASKDESC/CUDA-Agent-Ops-6K \
  --local-dir data/CUDA-Agent-Ops-6K --repo-type dataset
huggingface-cli download janhq/Jan-code-4b \
  --local-dir models/Jan-code-4b

# 准备训练数据 (动态注入 SKILL 模板和精简参考代码)
python3 tools/prepare_data.py \
  --input data/CUDA-Agent-Ops-6K/data.parquet \
  --output data/rocm_agent_ops/ \
  --arch gfx1100 \
  --skill agent_workdir/gfx1100/SKILL.md

# ═══ 安装 vLLM（首次）═══

# 1. 安装 vllm ROCm 预编译包
pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/

# 2. 安装 flash-attn（ROCm 版）
pip install https://wheels.vllm.ai/rocm/bcf2be96120005e9aea171927f85055a6a5c0cf6/flash_attn-2.8.3-cp312-cp312-manylinux_2_35_x86_64.whl

# ═══ 4 卡 vLLM 训练（当前正式命令） ═══

# 彻底清理环境防止死锁
pkill -9 -f "train_grpo.py" || true
pkill -9 -f "vllm" || true

# 终端 1：启动 vLLM 推理服务（物理 GPU 0+1）
CUDA_VISIBLE_DEVICES=2,3 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
nohup python tools/vllm_serve.py \
  --model models/Jan-code-4b \
  --tensor_parallel_size 2 \
  --dtype bfloat16 \
  --gpu_memory_utilization 0.82 \
  --max_model_len 8192 \
  --enforce_eager \
  --port 8000 \
  --trust-remote-code > logs/vllm-jan-code-4b.log 2>&1 &

# 等待 vLLM 启动成功后...

# 终端 2：启动 GRPO 训练（物理 GPU 2 训练，物理 GPU 3 评估）
CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
nohup python tools/train_grpo.py \
  --model models/Jan-code-4b \
  --use-vllm \
  --vllm-port 8000 \
  --train-gpu 1 \
  --eval-gpu 0 \
  --arch gfx1100 \
  --batch-size 1 \
  --num-generations 4 \
  --gradient-accumulation 8 \
  --max-completion-length 1024 \
  --temperature 0.5 \
  --conservative-eos-stop \
  --output-dir checkpoints/grpo-jan-code-4b-b2 \
  > logs/train-jan-code-4b-b2.log 2>&1 &
```

---

## 6. 开源组件

| 组件 | 方案 |
|------|------|
| 基座模型 | `janhq/Jan-code-4b` |
| RL 框架 | TRL GRPOTrainer |
| 推理加速 | vLLM 0.17.1+rocm700（主机模式，TP=2） |
| 数据集 | CUDA-Agent-Ops-6K |