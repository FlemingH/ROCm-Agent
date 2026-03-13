#include <hip/hip_runtime.h>

template<int THREADS>
__global__ void axpby_kernel(float* out, const float* a, const float* b, float alpha, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < size; i += stride) {
        out[i] = alpha * a[i] + b[i];
    }
}

extern "C" void axpby_launcher(
    float* out, const float* a, const float* b,
    float alpha, int size, int config, hipStream_t stream
) {
    if (size <= 0) return;
    switch (config) {
        case 1: {
            int blocks = (size + 127) / 128;
            axpby_kernel<128><<<blocks, 128, 0, stream>>>(out, a, b, alpha, size);
            break;
        }
        case 2: {
            int blocks = (size + 511) / 512;
            axpby_kernel<512><<<blocks, 512, 0, stream>>>(out, a, b, alpha, size);
            break;
        }
        default: {
            int blocks = (size + 255) / 256;
            axpby_kernel<256><<<blocks, 256, 0, stream>>>(out, a, b, alpha, size);
            break;
        }
    }
}
