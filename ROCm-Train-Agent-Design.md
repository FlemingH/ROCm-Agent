# ROCm-Train-Agent：面向 AMD GPU 的高性能 HIP 内核生成 Agent

基于 GRPO 强化学习训练 Agent，自动将 PyTorch 模型转化为 AMD GPU 上的高性能 HIP 内核。

---

## 1. 硬件与架构

### 1.1 目标 GPU

| 架构 | 系列 | 显卡 | 显存 | 状态 |
|------|------|------|------|------|
| gfx1100 | RDNA 3 | 4× W7800 | 32 GB 每卡 | **4 卡 vLLM TP=2** |

### 1.2 4 卡角色与设备映射

为了避免 PyTorch 和 vLLM 的环境冲突，以及训练时反向传播的 OOM 崩溃，系统采用了严格的物理隔离架构：

| 物理 GPU | 角色 | 环境变量配置 | 说明 |
|----------|------|----------|------|
| 0 | vLLM TP shard 0 | `HIP_VISIBLE_DEVICES=2,3` | vLLM 进程看到 GPU 0/1 对应物理 GPU 2/3 |
| 1 | vLLM TP shard 1 | `HIP_VISIBLE_DEVICES=2,3` | 提供模型推理生成 |
| 2 | 训练 | `HIP_VISIBLE_DEVICES=0,1 --train-gpu 0` | 训练进程 GPU 0 对应物理 GPU 2 |
| 3 | 评估 | `--eval-gpu 1` | 在空闲副卡（物理3）上开辟独立沙盒进行编译、验证和压测 |

---

## 2. 4 卡训练数据流图

```text
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
  │   │  3. 收集 rewards (MSE + 性能测速)                 │
  │   │  4. Forward → GRPO loss → Backward → 更新 LoRA   │
  │   │  5. HTTP: 同步 LoRA 权重 → vLLM                  │
  │   └──────┬──────────────┬────────────────────────────┘
  │          │              │
  ←── sync ──┘              │ 提交 8 并发编译/验证任务
                            ↓
              ┌──────────────────────────┐
              │  CPU: 8× hipcc workers   │──→ 编译结果
              │  ProcessPoolExecutor     │
              └──────────────────────────┘
                            │ 编译通过后
                            ↓
              ┌──────────────────────────┐
              │  GPU 3: verify + bench   │──→ 奖励分数 (-1.0 ~ +3.0)
              │  eval-gpu=1 (严格物理隔离) │
              └──────────────────────────┘
```

---

## 3. 核心加速与优化策略 (当前生效)

为了让 4B 级别的专精代码小模型能够胜任复杂的 C++ HIP 算子编写，我们在训练流程中实施了以下关键护航技术：

### 2.1 极简输出合约 (单文件生成)
模型仅需输出核心的 `fused_kernel.hip`，外围脏活由后台脚本自动补全：
* **头文件注入：** 自动补充 `#include <hip/hip_runtime.h>`。
* **动态绑定 (Dynamic Binding)：** 自动解析模型代码的 C++ 函数签名，合成 `fused_kernel_binding.cpp`，将 Python 的张量内存指针无缝连接至 HIP。
* **动态权重提取 (Weight Passing)：** 自动抓取原生 PyTorch 模型的 `float32` 权重，并通过提示词喂给大模型，解决了包含 `Conv2d` 和 `Linear` 层的数学无解问题。

### 2.2 连续 MSE 卷面分机制 (Reward Shaping)
摒弃了非黑即白的 `0.0` 验证惩罚。只要模型写出的 C++ 代码成功编译运行，且输出张量维度 (Shape) 匹配标准答案：
系统会计算它与真实数值的均方误差 (MSE)，并基于公式 `0.8 / (1.0 + sqrt(MSE))` 发放 **`0.05 ~ 0.8`** 的渐进分。
这极大地挽救了“数学超纲题”中的梯度信号，引导模型逐步修正算法逻辑。

### 2.3 保姆级硬件优化指南 (SKILL.md)
在提示词中直接注入了针对 RDNA3 (gfx1100) 架构的满血版优化教典：
* **`STRICTLY FORBIDDEN` 禁令**：强制禁止嵌套函数、Python代码和分配动态内存。
* **`__launch_bounds__(256)`**：强制限制寄存器溢出。
* **Wavefront = 32**：指明架构的 Warp 尺寸。
* **`float4` 向量化读取**：教授如何利用 128-bit 宽总线实现显存 IO 翻倍。
* **快速数学指令**：强制使用 `__expf`, `__frsqrt_rn` 等底层调用。

### 2.4 纯净的沙盒隔离防污染机制
放弃了高并发但极其危险的 Python 内存 `dlopen` 模块直调。每次模型验证时，系统均拉起独立的 Subprocess 环境进行干净的编译和测试，测完立即 `empty_cache()` 并物理销毁文件，彻底杜绝了动态库挂载污染（C-Extension Caching Bug），确保每一个打分真实可靠。

---

## 3. 模型与训练参数

### 3.1 模型状态

| 属性 | 值 |
|------|-----|
| 模型 | `janhq/Jan-code-4b` (强代码理解力) |
| LoRA | r=8, alpha=16, q_proj+v_proj, 约 23.6M trainable params |

### 3.2 阶梯奖励体系

| 阶段 | 奖励 | 说明 |
|------|------|------|
| 无代码 / 部分文件 | -1.0 | 格式错误，未包含完整 Markdown 代码块 |
| 编译失败 (Syntax) | -0.5 | 存在基础 C++ 语法错误 |
| 编译失败 (Linker) | -0.25 | 语法正确，但入口函数签名 (`launch_my_kernel`) 错误 |
| 验证数值错误 | 0.05~0.8| 运行成功，形状对齐，按误差均方根非线性给分 (连续 MSE 奖励) |
| 验证完全通过 | +1.0 | 逻辑完美正确，数值与 PyTorch 原生算子一致 |
| 性能击败 Eager | +2.0 | 压测耗时低于 PyTorch 原生 C++ 底层算子 5% |
| 性能击败 Compile | +3.0 | 极限提速，同时快过原生和 `torch.compile` Triton 编译器 |

### 3.3 训练参数 (防 OOM 黄金配比)

| 参数 | 值 | 说明 |
|------|-----|------|
| 并发评估 Worker | `8` | 平衡 PCIe 总线负载与 CPU 并发上限 |
| 有效批量 | 8 | `batch=1` × `grad_accum=8` (TRL 1.0.0 要求 grad_accum 可被 num_generations 整除) |
| 生成数 / prompt | `num_generations=4` | |
| 补全长度 | `2048` | 提供充裕的容错思考空间，借助 EOS 自动停机节省时间 |
| 探索温度 | `0.7` | 高温探索，鼓励探索高级优化（如 float4, shfl_xor） |

---

## 4. 标准启动流水线

```bash
# ═══ 首次准备 ═══
pip install -r requirements.txt
huggingface-cli download ASKDESC/CUDA-Agent-Ops-6K --local-dir data/CUDA-Agent-Ops-6K --repo-type dataset
huggingface-cli download janhq/Jan-code-4b --local-dir models/Jan-code-4b

# 打包数据 (将提取权重结构并注入 Prompt)
python3 tools/prepare_data.py \
  --input data/CUDA-Agent-Ops-6K/data.parquet \
  --output data/rocm_agent_ops_v5/ \
  --arch gfx1100 \
  --skill agent_workdir/gfx1100/SKILL.md

# ═══ 训练启动（B26 TRL 1.0.0 + ROCm vLLM） ═══

# 0. ROCm 必需环境变量（已写入 ~/.bashrc，新终端自动生效）
#    FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
#    HIP_FORCE_DEV_KERNARG=1  HSA_NO_SCRATCH_RECLAIM=1
#    SAFETENSORS_FAST_GPU=1   TORCH_BLAS_PREFER_HIPBLASLT=1
#    RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1

# 1. 彻底清理环境防止死锁
pkill -9 -f "train_grpo.py" || true
pkill -9 -f "vllm" || true

# 2. 终端 1：启动 vLLM 推理服务（物理 GPU 0+1）
HIP_VISIBLE_DEVICES=2,3 \
nohup python -u tools/vllm_serve.py \
  --model models/Jan-code-4b \
  --tensor_parallel_size 2 \
  --dtype bfloat16 \
  --gpu_memory_utilization 0.82 \
  --max_model_len 8192 \
  --enforce_eager \
  --port 8000 \
  > logs/vllm-jan-code-4b.log 2>&1 &

# 等待 vLLM 显示 Uvicorn running 启动成功后...
# curl http://localhost:8000/health/  → {"status":"ok"}

# 3. 终端 2：启动 GRPO 训练（物理 GPU 2 训练，物理 GPU 3 评估）
#    注意：TRL 1.0.0 要求 generation_batch_size (= batch × grad_accum) 可被 num_generations 整除
HIP_VISIBLE_DEVICES=0,1 \
nohup python -u tools/train_grpo.py \
  --model models/Jan-code-4b \
  --use-vllm \
  --vllm-port 8000 \
  --train-gpu 0 \
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
  --output-dir checkpoints/grpo-jan-code-4b-b26 \
  > logs/train-b26.log 2>&1 &
```
