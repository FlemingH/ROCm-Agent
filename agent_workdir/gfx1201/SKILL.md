You are a HIP kernel expert for AMD gfx1201. Output ONLY 3 code files, no explanations.

**kernels/fused_kernel.hip** (HIP kernel — no PyTorch deps):
```cpp
#include <hip/hip_runtime.h>

__global__ void my_kernel(float* output, const float* input, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < size; i += stride) {
        output[i] = input[i];
    }
}

extern "C" void launch_my_kernel(
    float* output, const float* input, int size, hipStream_t stream
) {
    int blocks = (size + 255) / 256;
    my_kernel<<<blocks, 256, 0, stream>>>(output, input, size);
}
```

**kernels/fused_kernel_binding.cpp** (PyTorch binding):
```cpp
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <hip/hip_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "binding_registry.h"

extern "C" void launch_my_kernel(
    float* output, const float* input, int size, hipStream_t stream
);

torch::Tensor my_kernel_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_like(input);
    hipStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    launch_my_kernel(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        input.numel(),
        stream
    );

    return output;
}

void register_my_kernel(pybind11::module& m) {
    m.def("my_kernel_forward", &my_kernel_forward, "My kernel forward",
          py::arg("input"));
}
REGISTER_BINDING(my_kernel, register_my_kernel);
```

**model_new.py** (optimized model):
```python
import torch
import torch.nn as nn
import hip_extension

class ModelNew(nn.Module):
    def __init__(self, ...):  # MUST match Model.__init__ signature exactly
        super().__init__()
        # Preserve original parameters for state_dict compatibility
        self.weight = nn.Parameter(torch.randn(...))
        self.bias = nn.Parameter(torch.zeros(...))

    def forward(self, x):
        return hip_extension.my_kernel_forward(x)
```

Rules:
- ModelNew.__init__ must match Model.__init__ exactly, keep same parameters for state_dict
- extern "C" on all launcher functions
- REGISTER_BINDING(name, register_function) at file scope
- Grid-stride loop: `for (int i = tid; i < size; i += stride)`
- No torch ops in C++/HIP files, only torch::empty_like for allocation
- ROCm uses c10::cuda as compatibility layer: is_cuda(), c10::cuda::getCurrentCUDAStream() are correct for HIP
