# ROCm-Agent：面向 AMD GPU 全架构的高性能 HIP 内核生成 Agent

## 1. 项目概述

基于强化学习训练 Agent，自动将 PyTorch 模型转化为 AMD GPU 上的高性能 HIP 内核。参考 CUDA-Agent 论文的 Agent 循环（实现 → 编译 → 验证 → 性能分析 → 迭代优化），按架构分版本适配 ROCm 7.x 工具链。首版聚焦 RDNA 3（gfx1100 / Radeon Pro W7800），后续扩展至 RDNA 4 等消费级/工作站架构。

### 1.1 架构专注设计

**当前版本专注于 gfx1100（RDNA 3）单一架构优化。**

这是一个有意为之的设计决策，原因如下：

1. **ROCm 库的架构差异性**：AMD 的 ROCm 生态中，不同库对不同架构的支持程度差异显著。例如 Composable Kernel 的 Flash Attention（ck_tile/fmha）仅支持 gfx9 和 gfx12（RDNA 4），不支持 gfx11（RDNA 3）；部分高级融合算子仅支持特定架构。一个"通用"Agent 无法有效处理这些架构差异。
2. **硬件特性的架构绑定**：不同 RDNA 代际之间硬件特性差异显著（如 WMMA 指令版本、缓存层次、wavefront 宽度），优化策略不可互换。
3. **训练效率**：Agent 专注单一架构，可以在有限的 RL 训练中更快收敛到高质量的内核生成策略，而非在多种架构间分散学习资源。

**未来扩展路线**：

- **短期**：为 RDNA 4 创建 `gfx1200/SKILL.md` + `gfx1200/HIP_REFS.md`，复用相同的 Agent 训练框架
- **中期**：实现架构自动检测（`rocminfo` → 自动选择对应 SKILL），单一 Agent 部署时根据目标 GPU 加载对应配置
- **长期**：随 ROCm 库架构支持逐步统一（如 Composable Kernel 扩展 gfx11 FMHA 支持），合并架构配置，训练跨架构通用 Agent

### 1.2 目标硬件


| 规格      | 参数                                 |
| ------- | ---------------------------------- |
| 架构      | RDNA 3（gfx1100）                    |
| 显存      | 4 × 32GB GDDR6 ECC = **128GB 总显存** |
| FP32 算力 | 45.2 TFLOPS × 4                    |
| 线程模型    | **Wavefront = 32**（与 CUDA Warp 一致） |
| 高速缓存    | LDS 64KB/CU + 96MB Infinity Cache  |
| 矩阵加速    | WMMA（AI Accelerators）              |
| 软件栈     | **ROCm 7.x** + hipcc               |


### 1.2 内核实现策略

**优先级 1：AMD 库融合调用**——hipBLASLt 完成 GEMM+Bias+Activation 一步融合

**优先级 2：AMD 库单算子调用**——rocBLAS（GEMM）、MIOpen（Conv/BN/Pooling/Softmax/LN）、rocPRIM（Reduction/Sort）

**优先级 3：手写 HIP 内核**——库函数无法覆盖的跨算子融合，使用 `__global`__ HIP 内核。RDNA 3 wavefront=32 与 CUDA warp=32 一致，预训练中的 CUDA 编程知识可直接迁移。

---

## 2. Agent 工作环境

### 2.1 目录结构

```
ROCm-Agent/
├── rocm-libraries/                 # ROCm 官方库源码（rocm-7.2.0，只读参考）
│   └── projects/                   # rocblas / hipblaslt / miopen / rocprim / composablekernel / rocwmma
├── tools/                          # ── 所有脚本统一存放 ──
│   ├── compile.py                  #   hipcc 编译（架构由 PYTORCH_ROCM_ARCH 传入）
│   ├── compile.sh                  #   编译包装器
│   ├── verify.py                   #   正确性验证
│   ├── profile.py                  #   性能对比
│   ├── hip_kernel_interaction.py   #   RL 交互代理（编译→验证→跑分→打分）
│   ├── interaction_config.yaml     #   verl interaction 配置
│   ├── prepare_data.py             #   数据预处理：dataset → verl 格式
│   └── filter_trajectories.py      #   轨迹过滤：挑选高质量样本
└── agent_workdir/
    ├── gfx1100/                    # ── 架构目录（RDNA 3）──
    │   ├── SKILL.md                #   Agent 行为指令（架构专属）
    │   ├── HIP_REFS.md             #   示例索引（含兼容性标注）
    │   ├── model_new.py            #   Agent 输出：优化后的模型
    │   └── kernels/                #   Agent 输出：HIP 内核
    │       ├── *.hip               #     内核文件
    │       └── *_binding.cpp       #     Python 绑定
    ├── binding.cpp                 # ── 共享基础设施（所有架构复用）──
    ├── binding_registry.h          #   自动注册机制
    └── model.py                    #   原始 PyTorch 模型（只读，每道题变化）
```

目录结构分三层：`tools/` 放所有脚本工具，`<arch>/` 放架构专属配置和产出，`agent_workdir/` 根目录放共享基础设施。扩展新架构只需新建 `gfx1200/` 等同级目录。

### 2.2 架构配置文件

每个目标架构有两个配置文件，放在 `agent_workdir/` 中：


| 文件                    | 作用                                                     |
| --------------------- | ------------------------------------------------------ |
| `gfx1100/SKILL.md`    | Agent 行为指令：三级优先级 + RDNA 3 硬件优化清单 + gfx1100 兼容的参考代码快速索引 |
| `gfx1100/HIP_REFS.md` | 完整示例索引：标注每个示例对 gfx1100 的兼容性（✅/⚠️），按优先级和训练难度分类          |


扩展到新架构时，只需创建对应的 `<arch>/SKILL.md` + `<arch>/HIP_REFS.md` 目录，无需修改 Agent 训练框架。

### 2.3 参考代码来源：rocm-libraries

使用 [ROCm/rocm-libraries rocm-7.2.0](https://github.com/ROCm/rocm-libraries/releases/tag/rocm-7.2.0) 作为 HIP 编程参考（与运行环境 ROCm 7.2 版本匹配）。优势：

- 官方实现，API 用法权威准确
- 覆盖全部 AMD 库（hipBLASLt / rocBLAS / MIOpen / rocPRIM / Composable Kernel / rocWMMA）
- `gfx1100/HIP_REFS.md` 中标注了每个示例对 gfx1100 的兼容性，Agent 仅使用兼容的示例作为参考

### 2.4 CUDA → HIP 速查表


| CUDA                                        | HIP (RDNA 3)                            |
| ------------------------------------------- | --------------------------------------- |
| `#include <cuda_runtime.h>`                 | `#include <hip/hip_runtime.h>`          |
| `cudaStream_t`                              | `hipStream_t`                           |
| `__syncthreads()` / `<<<blocks, threads>>>` | 不变                                      |
| warp = 32                                   | **wavefront = 32（一致）**                  |
| `__shfl_sync(mask, val, src)`               | `__shfl(val, src)`                      |
| Shared Memory / L2                          | LDS (64KB/CU) / **96MB Infinity Cache** |
| cuBLAS / cuDNN / CUB                        | rocBLAS+hipBLASLt / MIOpen / rocPRIM    |
| Tensor Core / WMMA                          | **WMMA (AI Accelerators)**              |
| nvcc / `TORCH_CUDA_ARCH_LIST=9.0`           | hipcc / `--offload-arch=gfx1100`        |


上下文消耗：gfx1100/SKILL.md + 按需从 rocm-libraries 读取的代码片段 + 速查表 ≈ 4-8K tokens（占 64K 上下文约 8-12%）。

---

## 3. 数据集：ROCm-Agent-Ops

RL 训练只需"题目"（model.py），不需要 HIP 参考答案。Agent 从编译/验证/profiling 环境反馈中自主学习。

**构建流水线**（复用 CUDA-Agent 论文方法）：

1. **种子算子挖掘**：从 torch/transformers 库挖掘，复用 CUDA-Agent-Ops-6K
2. **LLM 组合合成**：最多 5 个算子顺序组合为融合任务
3. **W7800 上执行过滤**：Eager/Compile 可执行、确定性、非平凡输出、运行时间 1-100ms、去 KernelBench 污染

**课程学习标注**：简单（单算子）~~30%、中等（2-3 算子含 GEMM/Conv）~~45%、困难（4-5 算子或完整模块）~25%

**规模**：~4,500-5,000 条 model.py + rocm-libraries 官方示例作为上下文学习参考。

---

## 4. 训练方案

### 4.1 双模型对照实验（MoE vs Dense）


| 属性         | 模型 A：MoE 代码专精              | 模型 B：Dense 通用最强代码                        |
| ---------- | -------------------------- | ---------------------------------------- |
| 模型         | **Qwen3-Coder-30B-A3B** Q4 | **Qwen3.5-27B** Q4                       |
| 架构         | MoE（128 专家，8 激活）           | **Dense（全参激活）**                          |
| 总参数 / 激活参数 | 30B / **3B**               | 27B / **27B**                            |
| 代码基准       | SWE-bench ~55-65%（估）       | **SWE-bench 72.4%, LiveCodeBench 80.7%** |
| 量化显存       | ~18GB                      | ~17GB                                    |
| GRPO 每卡占用  | 9GB，剩余 23GB                | 8.5GB，剩余 23.5GB                          |
| 训练上下文      | **128K**                   | **128-192K**                             |
| 最大交互轮数     | **~40 轮**                  | **~50 轮**                                |
| Rollout 速度 | ~60s/2K tokens（3B 激活，快）    | ~150s/2K tokens（27B 全激活，慢）               |
| LoRA 训练效率  | MoE 部分专家 LoRA 更新不均匀        | **全参数 LoRA 均匀更新**                        |
| 部署         | 单卡（18GB）                   | 单卡（17GB）                                 |
| 许可证        | Apache 2.0                 | Apache 2.0                               |


**实验核心问题**：MoE（3B 激活，代码专精训练，rollout 快）vs Dense（27B 全激活，通用最强代码能力，rollout 慢），哪种架构更适合 RL 训练 HIP 内核生成 Agent？

**对比公平性**：两者上下文（128K）、交互轮数（40-50 轮）、显存占用（每卡 ~9GB）、部署方式（单卡）均接近，差异集中在架构和训练效率上。

### 4.2 训练流水线

全部在 4x W7800 上完成，使用 GRPO（无需 Critic，适配 128GB 总显存），verl 框架：

```
阶段 1: 单轮 GRPO 热身
  输入 model.py → Agent 一次性生成 HIP 内核 → 环境给出奖励
  奖励: -1（错误）/ +1（正确）/ +2（比 Eager 快 >5%）/ +3（比 Eager 和 Compile 都快 >5%）

阶段 2: Actor 初始化
  采样轨迹 → RFT 过滤高质量轨迹 → Actor 监督微调

阶段 3: 多轮 Agentic RL
  Agent 多轮循环: model.py → HIP 内核 → 编译 → 验证 → profiling → 迭代优化
  课程学习: 前期简单算子为主 → 中期增加 GEMM/Conv 融合 → 后期全难度
```

### 4.3 训练说明

两个模型串行训练，共用 4x W7800。主要瓶颈在 Agentic RL 阶段（占 ~~80% 时间），受限于 hipcc 编译速度（~~30s/次）和多轮串行交互。模型 B（Dense 27B）rollout 速度约为模型 A（MoE 3B 激活）的 1/2.5，Agentic RL 阶段耗时约 1.7-2 倍。

**可能的加速手段**：增加 W7800 卡数（8 卡可减半）、减少最大交互轮数、编译缓存复用已编译的 .so、模型 B 可用 speculative decoding 加速 rollout。

### 4.4 训练启动命令（模型 A 完整流程）

奖励函数：-1（编译/验证失败）、+1（正确）、+2（比 Eager 快 >5%）、+3（比 Eager 和 Compile 都快 >5%）。

模型 B（Qwen3.5-27B）使用相同命令，差异参数在注释中标注。

```bash
# ========== 0. 环境 ==========
conda activate ra
pip install -r requirements.txt
# vLLM ROCm wheel 需要单独安装（跳过 torch 版本检查，卸载不兼容的 flashinfer）：
pip install --no-deps "https://wheels.vllm.ai/rocm/95c0f928cdeeaa21c4906e73cee6a156e1b3b995/vllm-0.17.1%2Brocm700-cp312-cp312-manylinux_2_35_x86_64.whl"
pip uninstall -y flashinfer-python torch-c-dlpack-ext 2>/dev/null

# ========== 1. 准备数据 ==========
python3 tools/prepare_data.py \
  --input data/CUDA-Agent-Ops-6K/data.parquet \
  --output data/rocm_agent_ops/ \
  --arch gfx1100 \
  --skill-path agent_workdir/gfx1100/SKILL.md

# ========== 2. 阶段 1：单轮 GRPO 热身 ==========
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=data/rocm_agent_ops/train.parquet \
  data.val_files=data/rocm_agent_ops/val.parquet \
  data.return_raw_chat=True \
  data.train_batch_size=256 \
  data.max_prompt_length=4096 \
  data.max_response_length=8192 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path=models/Qwen3-Coder-30B-A3B-Instruct \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n=5 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.total_epochs=5 \
  trainer.save_freq=20 \
  trainer.test_freq=5 \
  trainer.project_name=rocm_agent \
  trainer.experiment_name=phase1_moe_grpo
# 模型 B 差异：model.path=models/Qwen3.5-27B, batch=128, mini_batch=32, exp=phase1_dense_grpo

# ========== 3. 阶段 2：RFT 过滤 + 监督微调 ==========
python3 tools/filter_trajectories.py \
  --input checkpoints/phase1_moe_grpo/trajectories/ \
  --output data/rocm_agent_ops/rft_filtered.parquet \
  --min-reward 2

python3 -m verl.trainer.main_sft \
  data.train_files=data/rocm_agent_ops/rft_filtered.parquet \
  model.path=models/Qwen3-Coder-30B-A3B-Instruct \
  model.enable_gradient_checkpointing=True \
  optim.lr=5e-7 \
  trainer.n_gpus_per_node=4 \
  trainer.total_epochs=3 \
  trainer.save_path=checkpoints/phase2_moe_rft \
  trainer.project_name=rocm_agent \
  trainer.experiment_name=phase2_moe_rft

# ========== 4. 阶段 3：多轮 Agentic RL ==========
# Interaction 配置见 tools/interaction_config.yaml
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=data/rocm_agent_ops/train.parquet \
  data.val_files=data/rocm_agent_ops/val.parquet \
  data.return_raw_chat=True \
  data.train_batch_size=128 \
  data.max_prompt_length=4096 \
  data.max_response_length=16384 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path=checkpoints/phase2_moe_rft \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n=3 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.multi_turn.enable=true \
  actor_rollout_ref.rollout.multi_turn.interaction_config_path=tools/interaction_config.yaml \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=20 \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=20 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.total_epochs=15 \
  trainer.save_freq=10 \
  trainer.test_freq=5 \
  trainer.project_name=rocm_agent \
  trainer.experiment_name=phase3_moe_agentic_rl
# 模型 B 差异：model.path=checkpoints/phase2_dense_rft, batch=64, mini_batch=16, n=2, exp=phase3_dense_agentic_rl
```

---

## 5. 评估方案

**基准**：自建 ROCm-Bench，从数据集划分 500 条测试集在 W7800 上评估。参考基准：MIOpen/rocBLAS 官方库性能。

**指标**：Pass Rate、Faster Rate vs. Compile、Geomean Speedup。

**目标**：Pass Rate >85%（理想 >92%），Faster Rate vs. Compile >50%（理想 >70%），Speedup >1.2x（理想 >1.5x）。

**A/B 对比维度**：


| 对比维度                                    | 模型 A（MoE 30B，128K，40 轮） | 模型 B（Dense 27B，128K，50 轮） |
| --------------------------------------- | ----------------------- | ------------------------- |
| 整体指标（Pass Rate / Faster Rate / Speedup） | —                       | —                         |
| 按难度分层表现（简单/中等/困难）                       | —                       | —                         |
| 首次生成即正确比例                               | —                       | —                         |
| 平均迭代轮数                                  | —                       | —                         |
| 因上下文耗尽失败比例                              | —                       | —                         |
| 单样本平均训练时间                               | —                       | —                         |


实验假设：

- 若模型 B 整体明显优于 A → Dense 全激活（27B）+ 均匀 LoRA 更新的优势大于 MoE（3B 激活）的推理速度优势
- 若模型 A 整体接近或优于 B → MoE 代码专精训练可弥补激活参数少的劣势，且训练效率显著更高
- 若两者在不同难度上各有胜负 → 可考虑混合策略（简单任务用 MoE 快速处理，困难任务用 Dense 深度优化）

---

## 6. 里程碑


| 阶段              | 交付物                                              |
| --------------- | ------------------------------------------------ |
| **M1: 环境搭建**    | ROCm 7.x 工具链 + rocm-libraries 参考代码在 4x W7800 上跑通 |
| **M2: 数据集**     | ROCm-Agent-Ops（~4500 条 model.py）                 |
| **M3: 模型 A 训练** | MoE 30B Q4 三阶段 GRPO 训练                           |
| **M4: 模型 B 训练** | Dense 27B Q4 三阶段 GRPO 训练                         |
| **M5: 评估开源**    | A/B 对比报告、最优模型和数据集开源                              |


---

## 7. 风险与缓解


| 风险                             | 缓解措施                                                                     |
| ------------------------------ | ------------------------------------------------------------------------ |
| W7800 ROCm 7.x 兼容性             | M1 全面验证；Docker 固定 ROCm 版本                                                |
| RDNA 3 RL 训练框架适配               | M1 验证 BitsAndBytes/verl；不兼容则改 GPTQ/AWQ                                   |
| **ROCm 库架构支持不统一**              | **按架构分目录（`<arch>/SKILL.md` + `<arch>/HIP_REFS.md`），标注兼容性；Agent 仅引用兼容示例** |
| **gfx1100 缺少 Flash Attention** | **CK 经典示例有 WMMA 版自注意力（32_...wmma_fp16.cpp）；关注 CK 后续版本对 gfx11 FMHA 的支持**  |
| Agent 手写内核难度高                  | 课程学习渐进；标准算子优先调库                                                          |
| 编译/profiling 速度慢               | 异步 rollout 批量收集轨迹                                                        |
| 上下文学习不足以掌握 HIP                 | 增加 rocm-libraries 参考代码片段；必要时回退少量 SFT                                     |
| 标准编译器已足够好                      | 聚焦编译器优化不好的算子融合场景                                                         |
| 训练不稳定                          | 多阶段热身，监控 KL 散度和奖励曲线                                                      |


---

## 8. 开源组件


| 组件       | 方案                                     | 许可证        |
| -------- | -------------------------------------- | ---------- |
| 基座模型 A   | Qwen3-Coder-30B-A3B（Q4，MoE）            | Apache 2.0 |
| 基座模型 B   | Qwen3.5-27B（Q4，Dense）                  | Apache 2.0 |
| RL 框架    | verl                                   | Apache 2.0 |
| 量化       | bitsandbytes / GPTQ                    | MIT        |
| 微调       | PEFT (LoRA)                            | Apache 2.0 |
| 推理       | vLLM                                   | Apache 2.0 |
| AMD 库    | rocBLAS / hipBLASLt / MIOpen / rocPRIM | MIT        |
| 数据集      | CUDA-Agent-Ops-6K（复用）                  | —          |
| Agent 环境 | 自建（参考 OpenHands）                       | —          |
| 参考代码     | rocm-libraries rocm-7.2.0（官方源码）        | MIT        |


