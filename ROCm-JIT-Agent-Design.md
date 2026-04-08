# ROCm-JIT-Agent：AI-in-the-Loop 本地算子锻造引擎 (Kernel Forge)

## 1. 愿景与目标用户 (The Vision & Target Audience)

**“打破云端算力霸权，让端侧开发者在消费级显卡上，就地锻造出压榨硬件极限的底层算子。”**

传统的 AI 算子优化是一个漫长且痛苦的离线过程，需要顶尖的工程师手写 C++/HIP、调试内存对齐、解决寄存器溢出。更致命的是，这些能力往往被锁死在财力量雄厚的机房或需要联网调用的超大杯闭源模型（如 GPT-4o）中。

**ROCm-JIT-Agent** 敏锐地切入了**“算力下沉与端侧 AI 爆发”**的蓝海。它的核心产品故事并非面向普通终端消费者的实时加速，而是**专为开源极客、中小型科研团队、边缘计算（Edge AI）部署工程师打造的“本地高性能算子锻造炉”**。

通过 GRPO 强化学习，我们成功将 AMD RDNA 架构的底层汇编智慧，压缩到了一个仅需极低显存即可运行的 **4B 专精代码模型**中。这意味着，任何开发者都能在他们那张单薄的 7900 XTX 甚至 7900 GRE 上，于完全离线、隐私安全的环境下，为自己的创新算法（如最新的 Sparse Attention 或魔改的量化解包）瞬间生成比肩官方驱动工程师水准的极速算子。

### 1.1 极客感十足的交互体验

用户无需离开熟悉的 PyTorch 环境，只需在性能瓶颈的模块上添加一个简单的装饰器 `@rocm_jit_agent.optimize`：

```python
import torch
import rocm_jit_agent

@rocm_jit_agent.optimize(target="gfx1100", backend="local:Jan-code-4b")
def custom_attention_or_swish(x, weight, bias):
    # 用户熟悉的、可能很慢的 PyTorch 逻辑
    return torch.sigmoid(x) * weight + bias

# --- Benchmark (类似 GPU MODE 风格) ---
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[128 * i for i in range(2, 50)],
        line_arg='provider',
        line_vals=['torch-native', 'torch-compile', 'rocm-jit-agent'],
        line_names=["Torch (Eager)", "Torch (Compiled)", "ROCm-JIT-Agent"],
        styles=[('blue', '-'), ('green', '--'), ('red', '-')],
        ylabel="GB/s",
        plot_name="swish_performance",
        args={'M': 4096},
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    w = torch.randn(N, device='cuda', dtype=torch.float32)
    b = torch.randn(N, device='cuda', dtype=torch.float32)
    
    if provider == 'torch-native':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.sigmoid(x) * w + b)
    if provider == 'torch-compile':
        compiled_fn = torch.compile(lambda: torch.sigmoid(x) * w + b)
        ms, min_ms, max_ms = triton.testing.do_bench(compiled_fn)
    if provider == 'rocm-jit-agent':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: custom_attention_or_swish(x, w, b))
        
    gbps = lambda ms: 3 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(print_data=True, save_path='.')
```

**运行表现：**
当该前沿算子被部署或训练脚本**第一次**运行触发时，控制台将由于后台的 JIT 试错产生十几秒的启动延迟（在长达数天的科研训练或端侧一次性部署面前，这十几秒微不足道），并弹出一个极具硬核感的命令行进度提示：
```text
[rocm_jit_agent] 侦测到未优化的 PyTorch 算子，正在唤醒 4B 本地专精模型伴随编译...
[rocm_jit_agent] AI 分析张量签名: input(1024x1024, fp32), weight(1024, fp32)...
[rocm_jit_agent] [Iter 1/4] 生成 RDNA3 (gfx1100) 底层 HIP 算子... 编译成功.
[rocm_jit_agent] rocprofv3 硬件级探测: 发现 FMA 乘加融合丢失，VGPR过载，强制指令 AI 重构...
[rocm_jit_agent] [Iter 2/4] 代码重构中... 验证完美通过! MSE=0.0.
[rocm_jit_agent] ✨ 锻造成功！Torch Eager: 144.5us -> AI HIP Kernel: 15.2us (提速 9.5x)
[rocm_jit_agent] 💾 已将极速算子硬编码至 ~/.rocm_jit_agent_cache/，后续所有的训练/推理 Epoch 将零延迟介入。
```

随后的所有调用，将直接挂载并执行这个被 AI “压榨”到硬件极限的 `.so` 动态库。

---

## 2. 核心架构设计：四层解耦外壳 (Model-Agnostic Shell)

为了实现极高的鲁棒性与生态兼容性，该系统 采用了**“模型无关（Model-Agnostic）”**的架构设计。模型只负责根据 Prompt 吐出代码字符串，而外壳全权接管整个工程管线。

### 2.1 第一层：前端拦截与 AST 分析层
*   **职责**：在 Python 运行时动态拦截带有 `@rocm_jit_agent.optimize` 的函数。
*   **机制**：通过 Python 的 `inspect` 模块提取函数源码的 AST（抽象语法树），抓取输入张量的 `shape` 和 `dtype`，自动合成包含算子逻辑和权重参数（如 `weight`, `bias`）信息的初始 User Prompt。

### 2.2 第二层：LLM 网关适配层 (Gateway)
*   **职责**：屏蔽背后大语言模型的差异，实现云端/本地的无缝切换。
*   **机制**：提供统一的推理接口。支持本地专精的小模型（如基于 GRPO 训练的 `grpo-jan-code-4b-b26`，它能在 8GB 显存内流畅运行，专治各种底层格式不服），也支持在面对极度复杂的图网络空间归约算子时，通过 API 临时调用云端的超大参数模型（如 Claude 3.5 Sonnet 或 Qwen-Max）。

### 2.3 第三层：双阶段状态机与提示词引擎 (Two-Phase State Machine)
这是整个引擎的“大脑”，也是顶尖算子调优专家的 AI 化克隆：
*   **Phase 1: 正确性对齐 (Correctness)**
    *   模型生成第一版雏形代码。
    *   外壳在沙盒中执行并计算与原生 PyTorch 标答的 MSE 误差。
    *   **兜底机制**：如果模型犯了头文件遗漏、函数签名错乱等低级错误，外壳的 AST 解析器会**强行拦截并自动修复**。如果遭遇 Segfault，外壳捕获 `hipcc` 日志，命令模型进行针对性的 Debug 重写。
*   **Phase 2: rocprofv3 靶向性能榨取 (Performance-Guided Tuning)**
    *   正确性达标后，外壳静默唤醒 AMD 的硬件分析器 `rocprofv3`。
    *   提取真实硅片指标：寄存器溢出（Spilling）、L2 Cache 未命中率、核心指令触发率。
    *   **降维指导**：外壳将这些枯燥的十六进制日志，翻译成极具压迫感的 Prompt。例如：“*你的 VGPR 占用高达 128 个，直接拖垮了 Occupancy（并发度），立刻删减局部变量*” 或 “*警告：当前计算未触发 FMA 指令！必须将单独的 ADD/MUL 重构为 fmaf()*”。这种硬核反馈逼迫 4B 小模型写出超越人类普通手写水平的汇编级优化代码。

### 2.4 第四层：纯净沙盒与缓存生命周期层 (Sandbox & JIT Cache)
*   **职责**：保证宿主 Python 进程的绝对安全。
*   **机制**：所有的编译、链接和跑分都在独立的 `subprocess` 临时目录中进行，防止 C-Extension 的段错误（Segfault）挂死用户的训练进程。
*   **热替换**：一旦某个 Kernel 在多轮“对线”中胜出，将其 `.so` 永久缓存在本地（Key 为源码哈希+张量签名）。下次遇到相同算子，绕过大模型，达成 O(1) 的极速启动。

---

## 3. 设计哲学：为什么不用云端，而是死磕本地 4B 训练？

**为什么不在每次运行遇到了未优化的算子时，都发网络请求让 Claude 3.5 帮忙改 Bug？**
*   **企业级的隐私隔离**：当端侧团队在调试医学大模型或自动驾驶内部量化方案时，他们不希望将模型代码和权重签名发送到任何公有云。
*   **训练保下限**：通过 GRPO 强化学习，我们让仅有 4B 参数量的本地基础模型（Base）强制戒掉了预训练语料中的“CUDA 幻觉”（比如瞎写 `__h2f`），并形成了苛刻的输出格式肌肉记忆。训练的目的是让它成为一个**“首发良率极高、甚至能在离线显卡上秒出结果”的懂行下士**。
*   **外壳拔上限**：面对极其苛刻的 3.0 分（需要同时超越 Eager 和 Triton 编译器），单靠小模型的直觉（Zero-Shot）是不现实的。外壳通过引入 `rocprofv3` 的底层硅片级反馈，扮演了一个**冷酷的监工**，用高压 Prompt 指导模型进行靶向的汇编级重构，榨干消费级显卡的最后 10% 带宽。

---

## 4. 未来演进路线 (Roadmap to 2.0)

当前的 1.0 版本使用 HIP (C++) 证明了整个工程闭环的可行性和惊人的提速潜力。随着生态的发展，该引擎将迎来以下维度的降维打击：

### 4.1 全面拥抱 Triton 抽象 (Triton Support)
*   **背景**：让一个小模型精准写出管理寄存器溢出的 HIP C++ 代码过于残忍。
*   **升级**：下一版将支持输出 **OpenAI Triton** Python 代码。因为没有分号、类型和繁琐的指针绑定，大模型的首发成功率（One-Shot Accuracy）将从目前的徘徊提升至 95% 以上。此时，外壳的优化策略将变为利用 `@triton.autotune` 自动搜索 `BLOCK_SIZE` 和 `num_warps`。

### 4.2 更大规模模型的无缝接入 (Scaling Laws)
*   由于采用了模型无关的外壳设计，我们可以随时接入专为代码推理蒸馏的 32B 甚至更大参数模型（如 Qwen2.5-Coder-32B）。外壳不需要改动任何一行逻辑，就能借助大模型的深厚数学功底，去攻克目前 4B 模型无法驾驭的空间归约难题（如 LayerNorm 或复杂的非对称量化解包）。

### 4.3 RAG 驱动的“非标算子军火库”
*   当遭遇业界刚刚发布的全新算法（例如各种复数神经网络，或者 1.5-bit Ternary 压缩等 AMD 官方尚未支持的特性）时，外壳可以在接入 4B/LLM 之前，先自动检索向量数据库中最新的 GitHub 算法实现片段，作为 Few-shot Prompt 喂给模型。这不仅抹平了 AMD 滞后于 Nvidia 的开源生态劣势，更让端侧机器获得了在没有官方驱动下，靠 JIT“死磕”新算法的破局能力。
