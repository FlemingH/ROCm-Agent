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
| 0 | 训练 | `CUDA_VISIBLE_DEVICES=0,1 --train-gpu 0` | 训练进程的 GPU 0 对应物理 GPU 0，负责反向传播更新权重 |
| 1 | 评估 | `--eval-gpu 1` | 训练进程的 GPU 1 对应物理 GPU 1，在空闲副卡上开辟独立沙盒进行编译、验证和压测 |
| 2 | vLLM TP shard 0 | `CUDA_VISIBLE_DEVICES=2,3` | vLLM 进程看到的 GPU 0 对应物理 GPU 2，提供模型推理生成 |
| 3 | vLLM TP shard 1 | `CUDA_VISIBLE_DEVICES=2,3` | vLLM 进程看到的 GPU 1 对应物理 GPU 3，提供模型推理生成 |

---

## 2. 4 卡训练数据流图

```text
                    ┌────────────────────────────────────┐
                    │  Host vLLM Server (物理 GPU 2 + 3)  │
                    │  tools/vllm_serve.py, TP = 2       │
                    │  PagedAttention + Continuous Batch  │
  ┌── HTTP req ───→ │  BF16 autoregressive decoding      │
  │                 └───────────────┬────────────────────┘
  │                                 │ HTTP: completions
  │                                 ↓
  │   ┌──────────────────────────────────────────────────┐
  │   │  Host GPU 0: TRL GRPOTrainer                     │
  │   │  use_vllm=True, vllm_mode="server"               │
  │   │                                                  │
  │   │  1. 发送 batch prompts → vLLM 生成 completions   │
  │   │  2. 提交 completions → CPU 编译 + GPU 1 验证     │
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
              │  GPU 1: verify + bench   │──→ 奖励分数 (-1.0 ~ +3.0)
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

# 2. 终端 1：启动 vLLM 推理服务（物理 GPU 2+3）
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
  > logs/vllm-jan-code-4b.log 2>&1 &

# 等待 vLLM 显示 Uvicorn running 启动成功后...
# curl http://localhost:8000/health/  → {"status":"ok"}

# 3. 终端 2：启动 GRPO 训练（物理 GPU 0 训练，物理 GPU 1 评估）
#    注意：TRL 1.0.0 要求 generation_batch_size (= batch × grad_accum) 可被 num_generations 整除
CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
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

---

## 5. 训练过程与踩坑记录 (Training Process & Pitfalls)

在 ROCm-Agent 的整个迭代与训练过程中，我们经历了多轮重构与调优（一直迭代至 B26 稳定版），最终打造出了当前的高并发、高稳定、高容错训练管线。以下是核心的踩坑与解决记录：

### 5.1 显存 OOM 与严格物理隔离
* **坑**：最初尝试 8B 模型并开启 GRPO 时，即使在 32GB 显存的 W7800 上也会迅速遭遇 OOM。
* **解**：
  1. 切换至 4B 级别模型 (`janhq/Jan-code-4b`)，兼顾代码理解能力与显存占用。
  2. 采用**严格的物理显卡隔离**架构：将负责大批量并发生成的 vLLM 部署在 GPU 0+1，主训练进程部署在 GPU 2，编译与验证沙盒隔离在 GPU 3。辅以 `batch-size=1` 与 `gradient-accumulation=8`，彻底根治了 OOM 问题。

### 5.2 多进程并发死锁 (Deadlocks)
* **坑**：在拉起 8 个 CPU Worker 并发进行内核编译与测试时，Python 的 `multiprocessing` 默认使用 `fork` 模式。这连带复制了主进程的 vLLM 和 CUDA 状态，导致底层 C++ 驱动锁状态错乱，引发了严重的 NCCL/RCCL 通信死锁与 `BrokenProcessPool` 崩溃。
* **解**：在 `train_grpo.py` 入口强制启用 `spawn` 模式 (`multiprocessing.set_start_method("spawn", force=True)`)，确保每个评测 Worker 都是全新纯净的 Python 进程，不继承任何危险的父进程状态。

### 5.3 动态库挂载污染 (C-Extension Caching Bug)
* **坑**：为了加速打分，曾尝试将编译验证环节（`verify.py`）改为进程内 (`in-process`) `import` 调用。但 Python 的 `dlopen` 会永久缓存同名动态库，导致模型即使生成了新代码，底层调用的依然是旧版 `hip_extension.so` 缓存，奖励分数被彻底打乱。
* **解**：退回到纯正的 `subprocess.run` 沙盒隔离执行，并为每次生成的动态库赋予唯一的哈希后缀 (`hip_ext_UUID.so`)。牺牲了 1 秒的进程启动时间，但换来了 100% 绝对正确的奖励验证反馈。

### 5.4 输出格式失控与截断假象 (Truncation)
* **坑**：模型早期经常陷入长篇大论的 `<think>` 思考过程，导致输出轻松顶破 2048 Token 上限被截断。
* **解**：
  1. 缩减并重构 `SKILL.md`，要求绝对的单文件代码块输出。
  2. 在 vLLM 中显式配置停机词 `<END_OF_OUTPUT>`。模型一写完立即熔断推理，将每条数据的均长压缩到了 300~400 tokens，节省了巨量算力。
  3. （注：TRL 日志中的 `clipped_ratio=1.0` 仅仅是因为停机词非原生 EOS 触发的日志判断 Bug，底层梯度传播依然完全正常）。

### 5.5 TRL FSDP+PEFT 官方 Bug 
* **坑**：尝试开启 FSDP 多卡切分训练架构时，引发 vLLM 端频繁崩溃死锁（报错：`No module named 'base_model'`）。
* **解**：排查 TRL `v1.0.0` 源码发现，其 `_sync_fsdp1_params_to_vllm` 同步函数中漏写了对 LoRA 特有权重的过滤逻辑。最终我们果断撤回 FSDP，退回单卡反向传播架构，规避了这一第三方库的致命隐雷。

### 5.6 硬件瞬时故障与完美断点续训 (Checkpoint Resume)
* **坑**：在 B26 跑了整整 57 个小时（94% 进度）后，高压运行的 GPU 发生偶发硬件报错 (`HIP error: unspecified launch failure`)，训练中断。
* **解**：依赖我们预置的 `save_steps=50` 自动存档机制，系统完美留存了 Step 2500 的模型切片。通过为 Trainer 添加 `--resume-from-checkpoint` 参数，成功无缝加载优化器状态并断点续训，将前 50 小时的算力心血全部抢救了回来，顺利奔向 2700 步终点。

### 5.7 训练结果简述
经过上述密集的趟坑与调优，ROCm-Agent 最终跑出了难以置信的稳定性与有效性：
在稳定的断点续训与评估循环中，模型生成的 C++ 代码**基础语法合规率达到了恐怖的 100%**（未出现过 -0.5 惩罚）。凭借精心设计的**连续 MSE 卷面分机制**，模型可以在算理尚未绝对精确时持续获得 `0.05~0.8` 的正向连续梯度。它稳步摸索 RDNA3 的底层特性，并开始成功写出附带 `__launch_bounds__(256)`、网格步长循环与基础数据类型的有效算子，证明了基于 RLHF 让大模型自主写 HIP 高性能计算内核的路径是彻底走得通的！
