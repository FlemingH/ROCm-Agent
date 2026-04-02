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
| 3 | 评估 | `--eval-gpu 1` | `hip_kernel_interaction.py` 使用评测 GPU |

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

### 3.1 模型简化输出 (单文件合约) + 动态 Binding + 权重传递
为降低小模型的认知负荷，防止生成内容超长被截断，模型被要求**只输出 1 个文件**：`fused_kernel.hip`。
系统的 `hip_kernel_interaction.py` 会在后台自动完成以下后处理：
1. 自动注入 `#include <hip/hip_runtime.h>`。
2. **动态解析** `launch_fused_kernel()` 签名，自动合成参数匹配的 `fused_kernel_binding.cpp`。支持标量参数（`float slope`, `int dim`）和指针/张量参数（`const float* weight`）。
3. **动态权重传递**：`verify.py` 使用 `DynamicModelNew(Model)` 代替静态 `model_new.py`。该类继承原始 Model（保证 `load_state_dict(strict=True)` 成功），在 `forward()` 中自动提取 float32 权重张量传给 HIP 扩展。
4. **Prompt 权重注入**：`prepare_data.py` 自动实例化 Model，提取 `state_dict` 中的 float32 张量名和形状，注入到 prompt 中，指导 Agent 在 kernel 签名中添加对应的 `const float*` 参数。

**关键改进**：
- 早期版本使用硬编码 binding（仅支持 `(output, input, size, stream)` 4 参数签名），导致仅 ~12.5% 数据集可用。
- 动态 binding 通过 `parse_kernel_extra_params()` 自动适配标量参数。
- **权重传递机制**将可用数据从 12.5% 提升到 ~44.6%（+shape-preserving 有权重样本 1300 条）。Agent 仍只输出单文件 `fused_kernel.hip`，训练时间增幅 <5%。

### 3.2 保姆级提示词 (SKILL.md) — 极简版

> **b3 教训**：原始 SKILL.md 包含 12 条规则 + 5 段代码示例 = 1232 tokens，4B 模型因认知过载导致 100% clipped_ratio、29% 编译失败率。简化后 323 tokens， 规则从 12→7 条。

* **代码骨架**：直接提供带 `__launch_bounds__(256)` 的 `__global__` 函数与 `extern "C"` 启动器分离的完美模板。
* **7 条精简规则**：禁止动态内存/torch 头文件/Python 代码、`__global__` 必须顶层定义、使用 `__expf`/`__fdividef` 快速数学、全部 fuse 为单 kernel、输出限制 600 tokens。
* **Token 预算**：SKILL.md 约 323 tokens（系统提示），最坏 prompt 约 3520 tokens + 1024 补全 = 4544 tokens，8192 上下文内有 3648 tokens 余量。
* **设计原则**：*不在 SKILL 中教优化理论*，让模型通过 ref_snippets 示例学习正确模式，通过 reward signal 强化高性能写法。


### 3.3 极简参考代码 (ref_snippets.py) — Stage 1 基础版

提供硬编码的、与 SKILL.md 模板完美契合的简洁 C++ 算子实现，帮助模型跨过编译鸿沟。

**覆盖的算子类别**：
| 类别 | 算子 (16 种) | 实现方式 |
|------|----------|----------|
| 逐元素激活 (×11) | relu, gelu, silu, sigmoid, tanh, elu, exp, log, abs, neg, clamp | `__launch_bounds__(256)` + grid-stride loop + 快速数学 |
| 归一化 (×2) | LayerNorm, RMSNorm | 共享内存 2-pass 归约 |
| Softmax (×1) | softmax, log_softmax | 数值稳定 3-pass 归约 |
| 归约 (×1) | sum, mean | 共享内存 tree reduction |
| 线性 (×1) | Linear, Conv, GEMM | 权重近似简化实现 |

**op 映射**：`_OP_MAP` 包含 40+ PyTorch op 名到 16 个 snippet key 的映射，每个 prompt 最多注入 3 个 snippet（`max_snippets=3`）。

### 3.4 两阶段训练策略

基于 b3 失败分析（认知过载）和 CK/MIOpen 优化研究，采用分阶段策略：

#### Stage 1 — 基础正确性（b4）
- **SKILL.md**：极简版 323 tokens，7 条规则
- **ref_snippets**：基础版，所有 reduction 使用 `__shared__ float s[256]` 全 LDS 归约
- **训练数据**：`data/rocm_agent_ops_v2/`（avg 726 tokens/prompt）
- **目标指标**：clipped_ratio < 70%，compile success > 80%，avg reward > 0.0

#### Stage 2 — 性能优化（b5，基于 b4 checkpoint）
- **SKILL.md**：保持不变（323 tokens），不增加认知负荷
- **ref_snippets**：升级为 CK/MIOpen 优化版，包含 3 项核心优化
- **目标指标**：更多 +2/+3 reward（比 baseline / torch.compile 快）

**核心原理**：模型不需要理解优化理论，只需模仿 ref_snippet 模式。当模仿优化版代码获得 +2/+3 reward，GRPO 自动强化该模式。

### 3.5 Stage 2 优化策略（来源与 gfx1100 验证）

以下优化策略从 AMD rocm-libraries 和 Confluence 文档中提取，已在 gfx1100 W7800 上全部编译验证通过。

#### 优化 1：float4 向量化内存访问（来源：CK 论文 §5）

| 属性 | 说明 |
|------|------|
| **来源** | CK Paper: "vectorized global memory accesses from tensor descriptor and tile distribution, choosing per-lane vector widths" |
| **原理** | 每个线程一次加载/存储 128-bit (4×float)，将 memory transactions 减少 4× |
| **适用算子** | 所有逐元素 unary ops（ReLU, SiLU, GELU, Sigmoid 等） |
| **gfx1100** | ✅ RDNA3 支持 128-bit 对齐的 global load/store，无特殊限制 |
| **代码复杂度** | 低 — 仅改变加载模式，kernel 结构不变 |
| **token 开销** | 比基础版多 ~50 tokens（增加 `float4` cast 和边界处理） |

```cpp
// 基础版: 逐元素
for (int i = idx; i < size; i += stride) output[i] = f(input[i]);

// 优化版: float4 向量化
int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
float4 v = *reinterpret_cast<const float4*>(&input[idx]);
// ... apply f(v.x), f(v.y), f(v.z), f(v.w)
*reinterpret_cast<float4*>(&output[idx]) = result;
```

#### 优化 2：Warp Shuffle Reduction（来源：MIOpen warp_reduce.hpp + block_reduce.hpp）

| 属性 | 说明 |
|------|------|
| **来源** | MIOpen `src/kernels/warp_reduce.hpp`: 使用 `__shfl_down` 做 warp 内归约；`block_reduce.hpp`: 先 warp shuffle → 仅 num_warps 个元素写入 LDS → 再 warp shuffle |
| **原理** | 将 shared memory 用量从 `float[256]` 降至 `float[8]`（256线程/32warpSize=8 warps），减少 LDS bank conflict 和 `__syncthreads` 次数 |
| **适用算子** | LayerNorm, RMSNorm, Softmax, reduce_sum, reduce_mean |
| **gfx1100** | ✅ RDNA3 warpSize=32，`__shfl_down` 原生支持，CK 论文确认用 `ds_bpermute` intrinsic 做 warp-level reduction |
| **代码复杂度** | 中 — 需要 `warp_id/lane_id` 计算和 2 级归约 |
| **token 开销** | 比基础版多 ~40 tokens（增加 warp helper 函数） |

```cpp
// MIOpen 模式: warp_reduce → shared[num_warps] → warp_reduce
__device__ float warp_reduce_sum(float val) {
    for (int d = 16; d >= 1; d >>= 1)  // warpSize/2=16 for gfx1100
        val += __shfl_down(val, d);
    return val;
}
// block_reduce: shared memory 仅需 [blockDim/32] = [8] 个元素
```

#### 优化 3：Fused Residual + Normalization（来源：MIOpen MIOpenAddLayerNorm.cpp）

| 属性 | 说明 |
|------|------|
| **来源** | MIOpen `src/kernels/MIOpenAddLayerNorm.cpp`: 将 `y = LayerNorm(x1 + x2)` 融合为单个 kernel，避免中间 tensor 的 global memory round-trip |
| **原理** | CK "atomize first, compose later" — 在归约循环中直接计算 `x1[i] + x2[i]`，读取两个输入但只写一个输出 |
| **适用算子** | 训练数据中含 residual + LayerNorm/RMSNorm 的 fusion 样本 |
| **gfx1100** | ✅ 纯算术融合，无架构限制 |
| **代码复杂度** | 低 — 在现有 LayerNorm kernel 中多加一次加法 |
| **token 开销** | 比基础版 LayerNorm 多 ~20 tokens |

#### gfx1100 架构兼容性总结

| 优化策略 | gfx1100 支持 | 依赖特性 | 编译验证 |
|----------|:---:|----------|:---:|
| float4 向量化 | ✅ | 128-bit global load/store | ✅ hipcc --offload-arch=gfx1100 |
| Warp Shuffle (`__shfl_down`) | ✅ | warpSize=32, `ds_bpermute` | ✅ hipcc --offload-arch=gfx1100 |
| Fused Residual+Norm | ✅ | 纯算术，无特殊指令 | ✅ hipcc --offload-arch=gfx1100 |

> **排除的策略**：CK 的 Tile Distribution / Tensor Coordinate Transform / MFMA fragment mapping 等高级优化因需要 C++ template 元编程，远超 4B 模型的代码生成能力，不纳入训练范围。

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
| 编译通过，验证失败 | 0.0 ~ 0.8 | 成功上机，若形状相同则根据 MSE 给出 0.05~0.8 的连续分数，否则 0.0 |
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

# 准备训练数据 (动态注入 SKILL 模板、参考代码、权重信息)
python3 tools/prepare_data.py \
  --input data/CUDA-Agent-Ops-6K/data.parquet \
  --output data/rocm_agent_ops_v5/ \
  --arch gfx1100 \
  --skill agent_workdir/gfx1100/SKILL.md
# Note: prepare_data.py now auto-extracts weight tensor info from each Model
# and injects it into the prompt (e.g. "const float* bn_weight shape=(64)")

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
nohup python -u tools/vllm_serve.py \
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
nohup python -u tools/train_grpo.py \
  --model models/Jan-code-4b \
  --use-vllm \
  --vllm-port 8000 \
  --train-gpu 1 \
  --eval-gpu 1 \
  --arch gfx1100 \
  --batch-size 1 \
  --num-generations 4 \
  --gradient-accumulation 8 \
  --reward-workers 8 \
  --max-completion-length 2048 \
  --temperature 0.7 \
  --conservative-eos-stop \
  --train-data data/rocm_agent_ops_v5/train.parquet \
  --output-dir checkpoints/grpo-jan-code-4b-b22 \
  > logs/train-b22.log 2>&1 &
```

---

## 6. 开源组件

| 组件 | 方案 |
|------|------|
| 基座模型 | `janhq/Jan-code-4b` |
| RL 框架 | TRL GRPOTrainer |
| 推理加速 | vLLM 0.17.1+rocm700（主机模式，TP=2） |
