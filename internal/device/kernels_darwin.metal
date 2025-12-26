#include <metal_stdlib>
using namespace metal;

// ============ FP32 Kernels ============

kernel void add_kernel(device const float *a [[ buffer(0) ]],
                       device const float *b [[ buffer(1) ]],
                       device float *result [[ buffer(2) ]],
                       uint index [[ thread_position_in_grid ]]) {
    result[index] = a[index] + b[index];
}

// ============ FP16 Kernels ============

kernel void add_kernel_f16(device const half *a [[ buffer(0) ]],
                           device const half *b [[ buffer(1) ]],
                           device half *result [[ buffer(2) ]],
                           uint index [[ thread_position_in_grid ]]) {
    result[index] = a[index] + b[index];
}

kernel void add_scalar_kernel_f16(device const half *a [[ buffer(0) ]],
                                  constant half &val [[ buffer(1) ]],
                                  device half *result [[ buffer(2) ]],
                                  uint index [[ thread_position_in_grid ]]) {
    result[index] = a[index] + val;
}

kernel void scale_kernel_f16(device const half *a [[ buffer(0) ]],
                             constant half &val [[ buffer(1) ]],
                             device half *result [[ buffer(2) ]],
                             uint index [[ thread_position_in_grid ]]) {
    result[index] = a[index] * val;
}

kernel void tanh_kernel_f16(device const half *a [[ buffer(0) ]],
                            device half *result [[ buffer(1) ]],
                            uint index [[ thread_position_in_grid ]]) {
    result[index] = half(tanh(float(a[index])));
}

kernel void gelu_kernel_f16(device const half *a [[ buffer(0) ]],
                            device half *result [[ buffer(1) ]],
                            uint index [[ thread_position_in_grid ]]) {
    float x = float(a[index]);
    float c1 = 0.7978845608;
    float c2 = 0.044715;
    float inner = c1 * (x + c2 * x * x * x);
    result[index] = half(0.5 * x * (1.0 + tanh(inner)));
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

// Fused Add+GELU kernel to reduce dispatch overhead
kernel void add_gelu_kernel(device const float *a [[ buffer(0) ]],
                            device const float *b [[ buffer(1) ]],
                            device float *result [[ buffer(2) ]],
                            uint index [[ thread_position_in_grid ]]) {
    float x = a[index] + b[index];
    float c1 = 0.7978845608;
    float c2 = 0.044715;
    float inner = c1 * (x + c2 * x * x * x);
    result[index] = 0.5 * x * (1.0 + tanh(inner));
}

// Fused Add+Tanh kernel
kernel void add_tanh_kernel(device const float *a [[ buffer(0) ]],
                            device const float *b [[ buffer(1) ]],
                            device float *result [[ buffer(2) ]],
                            uint index [[ thread_position_in_grid ]]) {
    result[index] = tanh(a[index] + b[index]);
}

// FP16 Add Bias (add vector b to each row of A)
kernel void add_bias_kernel_f16(device half *matrix [[ buffer(0) ]],
                                device const half *bias [[ buffer(1) ]],
                                device half *result [[ buffer(2) ]],
                                constant int &cols [[ buffer(3) ]],
                                uint2 gid [[ thread_position_in_grid ]]) {
    // Note: matrix is rows*cols. Bias is size cols.
    // We launch a 2D grid: (cols, rows).
    // gid.x is col index, gid.y is row index.
    
    // We don't have rows arg here, gid check handles bounds implicitly if grid is sized right.
    // Wait, let's pass dimensions to check bounds.
    
    // Actually, simple is best: 1D flat ? 
    // Bias add is row-dependent. index % cols == bias element. Or better: (index / cols) = row. 
    // (index % cols) = col.
    // But division is slow.
    // 2D dispatch is better.
    
    int index = gid.y * cols + gid.x;
    result[index] = matrix[index] + bias[gid.x];
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

// FP16 LayerNorm: half I/O, float accumulation for stability
kernel void layernorm_kernel_f16(device const half *input [[ buffer(0) ]],
                                 device const half *gamma [[ buffer(1) ]],
                                 device const half *beta [[ buffer(2) ]],
                                 device half *output [[ buffer(3) ]],
                                 constant int &cols [[ buffer(4) ]],
                                 constant float &eps [[ buffer(5) ]],
                                 uint row_idx [[ threadgroup_position_in_grid ]],
                                 uint tid [[ thread_index_in_threadgroup ]],
                                 uint tg_size [[ threads_per_threadgroup ]]) {
    
    threadgroup float shared_sum[256];
    threadgroup float shared_sq_sum[256];
    
    int offset = row_idx * cols;
    
    // Phase 1: Sum
    float local_sum = 0.0;
    for (int i = tid; i < cols; i += tg_size) {
        local_sum += float(input[offset + i]);
    }
    shared_sum[tid] = local_sum;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float mean = shared_sum[0] / float(cols);
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Phase 2: Variance
    float local_sq_sum = 0.0;
    for (int i = tid; i < cols; i += tg_size) {
        float diff = float(input[offset + i]) - mean;
        local_sq_sum += diff * diff;
    }
    shared_sq_sum[tid] = local_sq_sum;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sq_sum[tid] += shared_sq_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float variance = shared_sq_sum[0] / float(cols);
    float inv_std = 1.0 / sqrt(variance + eps);
    
    // Phase 3: Normalize and store as half
    for (int i = tid; i < cols; i += tg_size) {
        float norm = (float(input[offset + i]) - mean) * inv_std;
        output[offset + i] = half(norm * float(gamma[i]) + float(beta[i]));
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

// FP16 Softmax with half precision I/O, float reductions for accuracy
kernel void softmax_kernel_f16(device const half *input [[ buffer(0) ]],
                               device half *output [[ buffer(1) ]],
                               constant int &cols [[ buffer(2) ]],
                               uint row_idx [[ threadgroup_position_in_grid ]],
                               uint tid [[ thread_index_in_threadgroup ]],
                               uint tg_size [[ threads_per_threadgroup ]]) {
    
    threadgroup float shared_max[256];
    threadgroup float shared_sum[256];
    
    int offset = row_idx * cols;
    
    // Phase 1: Find max (use float for accuracy)
    float local_max = -1e38;
    for (int i = tid; i < cols; i += tg_size) {
        float v = float(input[offset + i]);
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
    
    // Phase 2: Compute exp and sum (float for accuracy)
    float local_sum = 0.0;
    for (int i = tid; i < cols; i += tg_size) {
        float val = exp(float(input[offset + i]) - max_val);
        output[offset + i] = half(val);
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
    
    // Phase 3: Normalize and store as half
    for (int i = tid; i < cols; i += tg_size) {
        output[offset + i] = half(float(output[offset + i]) * inv_sum);
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

// ============ Nomic-specific Kernels ============

// RoPE (Rotary Positional Embeddings) FP16
// Assumes input is (batch*seq, num_heads * head_dim)
// Grid: (head_dim/2, num_heads, batch*seq)
kernel void rope_kernel_f16(device half *data [[ buffer(0) ]],
                            constant int &head_dim [[ buffer(1) ]],
                            constant int &num_heads [[ buffer(2) ]],
                            constant int &seq_len [[ buffer(3) ]],
                            uint3 gid [[ thread_position_in_grid ]]) {
    uint i = gid.x; // feature pair index (0 to head_dim/2 - 1)
    uint h = gid.y; // head index
    uint b_s = gid.z; // (batch * seq) index
    
    uint seq_idx = b_s % seq_len;
    uint offset = b_s * (num_heads * head_dim) + h * head_dim;
    
    float theta = (float)seq_idx * pow(10000.0, -2.0 * (float)i / (float)head_dim);
    float cos_theta = cos(theta);
    float sin_theta = sin(theta);
    
    half x1 = data[offset + i];
    half x2 = data[offset + i + head_dim/2];
    
    data[offset + i] = half((float)x1 * cos_theta - (float)x2 * sin_theta);
    data[offset + i + head_dim/2] = half((float)x1 * sin_theta + (float)x2 * cos_theta);
}

// SwiGLU Activation FP16
// Assumes input is (N, 2*inter_size), output is (N, inter_size)
// Grid: (inter_size, N)
kernel void swiglu_kernel_f16(device const half *input [[ buffer(0) ]],
                              device half *output [[ buffer(1) ]],
                              constant int &inter_size [[ buffer(2) ]],
                              uint2 gid [[ thread_position_in_grid ]]) {
    uint i = gid.x; // index in inter_size
    uint n = gid.y; // index in N
    
    int row_offset_in = n * (2 * inter_size);
    int row_offset_out = n * inter_size;
    
    float x = (float)input[row_offset_in + i];
    float y = (float)input[row_offset_in + i + inter_size];
    
    // Swish(x) = x * sigmoid(x)
    float swish_x = x / (1.0f + exp(-x));
    output[row_offset_out + i] = half(swish_x * y);
}

