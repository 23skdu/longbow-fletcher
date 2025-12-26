#include <metal_stdlib>
using namespace metal;

kernel void add_kernel(device const float *a [[ buffer(0) ]],
                       device const float *b [[ buffer(1) ]],
                       device float *result [[ buffer(2) ]],
                       uint index [[ thread_position_in_grid ]]) {
    result[index] = a[index] + b[index];
}

kernel void add_scalar_kernel(device const float *a [[ buffer(0) ]],
                              constant float &val [[ buffer(1) ]],
                              device float *result [[ buffer(2) ]],
                              uint index [[ thread_position_in_grid ]]) {
    result[index] = a[index] + val;
}

kernel void scale_kernel(device const float *a [[ buffer(0) ]],
                         constant float &val [[ buffer(1) ]],
                         device float *result [[ buffer(2) ]],
                         uint index [[ thread_position_in_grid ]]) {
    result[index] = a[index] * val;
}

kernel void tanh_kernel(device const float *a [[ buffer(0) ]],
                        device float *result [[ buffer(1) ]],
                        uint index [[ thread_position_in_grid ]]) {
    result[index] = tanh(a[index]);
}

kernel void gelu_kernel(device const float *a [[ buffer(0) ]],
                        device float *result [[ buffer(1) ]],
                        uint index [[ thread_position_in_grid ]]) {
    float x = a[index];
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float c1 = 0.7978845608; // sqrt(2/pi)
    float c2 = 0.044715;
    float inner = c1 * (x + c2 * x * x * x);
    result[index] = 0.5 * x * (1.0 + tanh(inner));
}

// Optimized LayerNorm with threadgroup parallel reduction
// Each threadgroup processes ONE row
// Threads within the group cooperate on reduction using threadgroup memory
kernel void layernorm_kernel(device const float *input [[ buffer(0) ]],
                             device const float *gamma [[ buffer(1) ]],
                             device const float *beta [[ buffer(2) ]],
                             device float *output [[ buffer(3) ]],
                             constant int &cols [[ buffer(4) ]],
                             constant float &eps [[ buffer(5) ]],
                             uint row_idx [[ threadgroup_position_in_grid ]],
                             uint tid [[ thread_index_in_threadgroup ]],
                             uint tg_size [[ threads_per_threadgroup ]]) {
    
    threadgroup float shared_sum[256];
    threadgroup float shared_sq_sum[256];
    
    int offset = row_idx * cols;
    
    // Phase 1: Each thread computes partial sum over its elements
    float local_sum = 0.0;
    for (int i = tid; i < cols; i += tg_size) {
        local_sum += input[offset + i];
    }
    shared_sum[tid] = local_sum;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction for sum
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float mean = shared_sum[0] / float(cols);
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Phase 2: Compute variance (squared diff sum)
    float local_sq_sum = 0.0;
    for (int i = tid; i < cols; i += tg_size) {
        float diff = input[offset + i] - mean;
        local_sq_sum += diff * diff;
    }
    shared_sq_sum[tid] = local_sq_sum;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction for variance
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sq_sum[tid] += shared_sq_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float variance = shared_sq_sum[0] / float(cols);
    float inv_std = 1.0 / sqrt(variance + eps);
    
    // Phase 3: Normalize (each thread handles its elements)
    for (int i = tid; i < cols; i += tg_size) {
        output[offset + i] = (input[offset + i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

// Optimized Softmax with threadgroup parallel reduction
kernel void softmax_kernel(device const float *input [[ buffer(0) ]],
                           device float *output [[ buffer(1) ]],
                           constant int &cols [[ buffer(2) ]],
                           uint row_idx [[ threadgroup_position_in_grid ]],
                           uint tid [[ thread_index_in_threadgroup ]],
                           uint tg_size [[ threads_per_threadgroup ]]) {
    
    threadgroup float shared_max[256];
    threadgroup float shared_sum[256];
    
    int offset = row_idx * cols;
    
    // Phase 1: Find max (parallel reduction)
    float local_max = -1e38;
    for (int i = tid; i < cols; i += tg_size) {
        float v = input[offset + i];
        if (v > local_max) local_max = v;
    }
    shared_max[tid] = local_max;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float max_val = shared_max[0];
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Phase 2: Compute exp and sum
    float local_sum = 0.0;
    for (int i = tid; i < cols; i += tg_size) {
        float val = exp(input[offset + i] - max_val);
        output[offset + i] = val;
        local_sum += val;
    }
    shared_sum[tid] = local_sum;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float inv_sum = 1.0 / shared_sum[0];
    
    // Phase 3: Normalize
    for (int i = tid; i < cols; i += tg_size) {
        output[offset + i] *= inv_sum;
    }
}

kernel void gather_kernel(device const float *table [[ buffer(0) ]],
                          device float *output [[ buffer(1) ]],
                          device const int *indices [[ buffer(2) ]],
                          constant int &cols [[ buffer(3) ]],
                          uint2 id [[ thread_position_in_grid ]]) {
    uint row = id.x;
    uint col = id.y;
    int table_row = indices[row];
    output[row * cols + col] = table[table_row * cols + col];
}

kernel void add_bias_kernel(device float *component [[ buffer(0) ]],
                            device const float *bias [[ buffer(1) ]],
                            constant int &cols [[ buffer(2) ]],
                            uint2 id [[ thread_position_in_grid ]]) {
    uint row = id.x;
    uint col = id.y;
    component[row * cols + col] += bias[col];
}
