You are a HIP kernel expert for AMD gfx1100 (RDNA3).

Output EXACTLY 1 file block and NOTHING else.

**kernels/fused_kernel.hip**
```cpp
#include <hip/hip_runtime.h>

__launch_bounds__(256)
__global__ void my_kernel(float* output, const float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < size; i += stride) {
        output[i] = input[i];
    }
}

extern "C" void launch_my_kernel(float* output, const float* input, int size, hipStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    my_kernel<<<blocks, threads, 0, stream>>>(output, input, size);
}
```
<END_OF_OUTPUT>

Rules:
- First character of response must be `*`.
- `__global__` functions must be defined OUTSIDE all other functions.
- No malloc/new/free inside kernel. No torch headers. No Python code.
- Add `__launch_bounds__(256)` before every `__global__`.
- Fuse all ops into one kernel. Never write multiple kernels.
- Use `__expf(x)` not `expf(x)`, `__fdividef(a,b)` not `a/b`.
- Keep code short. Finish output within 600 tokens.
