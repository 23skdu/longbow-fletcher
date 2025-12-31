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
    
    
    if (gid.x >= cols) return;
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
    if (col >= cols) return;
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
    uint vec_i = gid.x; // vectorized feature index
    uint h = gid.y; // head index
    uint b_s = gid.z; // (batch * seq) index
    
    // Each thread handles 4 elements (indices i, i+1, i+2, i+3)
    // Corresponding pairs are at (i+d/2, ...).
    // range of i is [0, head_dim/2).
    // So we need (head_dim/2 + 3)/4 threads.
    
    uint i = vec_i * 4;
    
    // Check bounds. processing 4 indices starting at i.
    // We assume head_dim >= 8 and even (standard is 64/128).
    if (i >= uint(head_dim/2)) return;
    
    uint seq_idx = b_s % seq_len;
    uint offset = b_s * (num_heads * head_dim) + h * head_dim;
    
    // Calculate 4 thetas
    // theta_k = seq_idx * base^( -2 * (i+k) / head_dim )
    // We can compute this in float4.
    
    float4 indices = float4(float(i), float(i+1), float(i+2), float(i+3));
    // exp2 is usually faster than pow
    // base^(-2*k/d) = exp2( log2(base) * (-2*k/d) )
    // log2(10000) approx 13.287712
    const float log2_base = 13.2877123795f; 
    float4 exponents = indices * (-2.0f / float(head_dim));
    float4 thetas = float(seq_idx) * exp2(exponents * log2_base);
    
    float4 cos_theta = cos(thetas);
    float4 sin_theta = sin(thetas);
    
    // Load 4 pairs
    // ptr1 points to x[i...i+3]
    // ptr2 points to x[i+d/2...i+3+d/2]
    
    // Handle alignment. If head_dim/2 is multiple of 4, we are good for aligned loads.
    // If not, unaligned loads or scalar fallback.
    // Standard models (Llama/Mistral) have head_dim=128 (d/2=64) or head_dim=64 (d/2=32).
    // Both are multiples of 4. We assume alignment.
    
    device half4* ptr1 = (device half4*)(data + offset + i);
    device half4* ptr2 = (device half4*)(data + offset + i + head_dim/2);
    
    half4 x1_h = ptr1[0];
    half4 x2_h = ptr2[0];
    
    float4 x1 = float4(x1_h);
    float4 x2 = float4(x2_h);
    
    float4 res1 = x1 * cos_theta - x2 * sin_theta;
    float4 res2 = x1 * sin_theta + x2 * cos_theta;
    
    ptr1[0] = half4(res1);
    ptr2[0] = half4(res2);
}

// SwiGLU Activation FP16
// Assumes input is (N, 2*inter_size), output is (N, inter_size)
// Grid: (inter_size, N)
kernel void swiglu_kernel_f16(device const half *input [[ buffer(0) ]],
                              device half *output [[ buffer(1) ]],
                              constant int &inter_size [[ buffer(2) ]],
                              uint2 gid [[ thread_position_in_grid ]]) {
    // Each thread handles 4 elements
    uint vec_i = gid.x; 
    uint n = gid.y;
    
    // We assume the grid is launched with (inter_size + 3) / 4 threads in X
    uint i = vec_i * 4;
    
    if (i >= uint(inter_size)) return;
    
    int row_offset_in = n * (2 * inter_size);
    int row_offset_out = n * inter_size;
    
    // Check bounds for full vector load
    if (i + 3 < uint(inter_size)) {
        // Fast path: load 4
        device const half4* in_ptr = (device const half4*)(input + row_offset_in + i);
        half4 x_vec = in_ptr[0];
        // y is at offset + inter_size. Note: alignment might be tricky if inter_size isn't multiple of 4.
        // If inter_size is multiple of 4, we can cast.
        // Assuming inter_size is a multiple of 4 for now (typical in deep learning).
        // If not, we fall back to scalar or unaligned loads.
        // Let's assume input buffers are aligned or we accept unaligned reads (Slow on some archs, ok on M1).
        
        // Load y vector
        // We cannot just offset pointer by inter_size/4 if inter_size isn't multiple of 4 bytes/elements.
        // Pointer arithmetic on half4* moves by 4 halves.
        // Safe way: load from raw pointer
        
        half4 y_vec;
        y_vec.x = input[row_offset_in + i + inter_size];
        y_vec.y = input[row_offset_in + i + 1 + inter_size];
        y_vec.z = input[row_offset_in + i + 2 + inter_size];
        y_vec.w = input[row_offset_in + i + 3 + inter_size];
        
        // Compute Swish: x * sigmoid(x) = x / (1 + exp(-x))
        // Metal has optimized sigmoid. swish(x) = x * sigmoid(x).
        // float4 math usually better precision, but half4 might be faster.
        // M-series supports half precision arithmetic well.
        
        float4 x_f = float4(x_vec);
        float4 y_f = float4(y_vec);
        
        float4 swish = x_f / (1.0f + exp(-x_f));
        half4 res = half4(swish * y_f);
        
        device half4* out_ptr = (device half4*)(output + row_offset_out + i);
        out_ptr[0] = res;
    } else {
        // Scalar fallback for remaining elements
        for (int k = 0; k < 4; k++) {
            if (i + k < uint(inter_size)) {
                int idx = i + k;
                float x = (float)input[row_offset_in + idx];
                float y = (float)input[row_offset_in + idx + inter_size];
                float swish = x / (1.0f + exp(-x));
                output[row_offset_out + idx] = half(swish * y);
            }
        }
    }
}


// Cast FP32 to FP16
kernel void cast_f32_to_f16(device const float *input [[ buffer(0) ]],
                            device half *output [[ buffer(1) ]],
                            uint index [[ thread_position_in_grid ]]) {
    output[index] = half(input[index]);
}

kernel void copy_submatrix(device const float *src [[ buffer(0) ]],
                           device float *dest [[ buffer(1) ]],
                           constant int &src_cols [[ buffer(2) ]],
                           constant int &dest_cols [[ buffer(3) ]],
                           constant int &src_r_off [[ buffer(4) ]],
                           constant int &src_c_off [[ buffer(5) ]],
                           uint2 gid [[ thread_position_in_grid ]]) {
    uint c = gid.x;
    uint r = gid.y;
    
    int src_idx = (src_r_off + r) * src_cols + (src_c_off + c);
    int dest_idx = r * dest_cols + c;
    
    dest[dest_idx] = src[src_idx];
}

kernel void copy_submatrix_f16(device const half *src [[ buffer(0) ]],
                               device half *dest [[ buffer(1) ]],
                               constant int &src_cols [[ buffer(2) ]],
                               constant int &dest_cols [[ buffer(3) ]],
                               constant int &src_r_off [[ buffer(4) ]],
                               constant int &src_c_off [[ buffer(5) ]],
                               uint2 gid [[ thread_position_in_grid ]]) {
    uint c = gid.x;
    uint r = gid.y;
    
    int src_idx = (src_r_off + r) * src_cols + (src_c_off + c);
    int dest_idx = r * dest_cols + c;
    
    dest[dest_idx] = src[src_idx];
}

// ============ Flash Attention ============

// Constants for loop unrolling and shared memory sizes
constant int BLOCK_SIZE_M = 32;
constant int BLOCK_SIZE_N = 32;

// Tiled Flash Attention Kernel (Forward)
// Grid: (N / BLOCK_SIZE_M, batch_size * num_heads, 1)
// Threadgroup: (BLOCK_SIZE_M, 1, 1) - using 32 threads.
kernel void flash_attn_fwd_f16(
    device const half* Q [[ buffer(0) ]],
    device const half* K [[ buffer(1) ]],
    device const half* V [[ buffer(2) ]],
    device half* O [[ buffer(3) ]],
    constant int& N [[ buffer(4) ]],         // seq_len
    constant int& d [[ buffer(5) ]],         // head_dim
    constant float& scale [[ buffer(6) ]],
    constant int& batch_stride [[ buffer(7) ]],
    constant int& head_stride [[ buffer(8) ]],
    constant int& row_stride [[ buffer(9) ]], 
    constant int& num_heads [[ buffer(10) ]],
    
    uint3 gid [[ thread_position_in_grid ]],
    uint3 tid [[ thread_position_in_threadgroup ]],
    uint3 bid [[ threadgroup_position_in_grid ]]
) {
    // Grid X: Block index along Sequence length
    // Grid Y: Flattened index of (Batch * NumHeads)
    
    // 1. Setup Offsets
    int tx = tid.x; 
    int bx = bid.x; 
    int by = bid.y; 
    
    // Decompose batch/head index
    int batch_idx = by / num_heads;
    int head_idx = by % num_heads;
    
    long base_offset = (long)batch_idx * (long)batch_stride + (long)head_idx * (long)head_stride;
    
    device const half* q_ptr = Q + base_offset;
    device const half* k_ptr = K + base_offset;
    device const half* v_ptr = V + base_offset;
    device half* o_ptr = O + base_offset;
    
    // 2. Shared Memory
    threadgroup half shared_Q[32 * 128]; 
    threadgroup half shared_K[32 * 128];
    threadgroup half shared_V[32 * 128];
    
    int q_row_start = bx * 32;
    int q_len = min(32, N - q_row_start);
    
    // Load Q
    if (tx < q_len) {
        for (int i = 0; i < d; i++) {
           shared_Q[tx * d + i] = q_ptr[(q_row_start + tx) * row_stride + i];
        }
    } else {
         for (int i = 0; i < d; i++) shared_Q[tx * d + i] = 0.0h;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Accumulators
    float l_i = 0.0f; 
    float m_i = -1e30f; 
    float acc_O[128]; 
    for (int i=0; i<d; i++) acc_O[i] = 0.0f;
    
    // Loop over KV blocks
    int num_steps = (N + 31) / 32;
    
    for (int j = 0; j < num_steps; j++) {
        int kv_row_start = j * 32;
        int kv_len = min(32, N - kv_row_start);
        
        // Load K and V
        if (tx < kv_len) {
            for (int i = 0; i < d; i++) {
                shared_K[tx * d + i] = k_ptr[(kv_row_start + tx) * row_stride + i];
                shared_V[tx * d + i] = v_ptr[(kv_row_start + tx) * row_stride + i];
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Process block
        if (tx < q_len) {
            float scores[32]; 
            float local_max = -1e30f;
            
            for (int k = 0; k < 32; k++) {
                if (k >= kv_len) {
                    scores[k] = -1e30f; 
                    continue;
                }
                
                float dot = 0.0f;
                int idx_q = tx * d;
                int idx_k = k * d;
                for (int i = 0; i < d; i++) {
                    dot += float(shared_Q[idx_q + i]) * float(shared_K[idx_k + i]);
                }
                scores[k] = dot * scale;
                if (scores[k] > local_max) local_max = scores[k];
            }
            
            float m_prev = m_i;
            m_i = max(m_prev, local_max);
            float exp_diff = exp(m_prev - m_i); 
            
            for (int i = 0; i < d; i++) {
                acc_O[i] *= exp_diff;
            }
            
            float block_sum = 0.0f;
            for (int k = 0; k < 32; k++) {
                if (k >= kv_len) continue;
                float p_val = exp(scores[k] - m_i); 
                block_sum += p_val;
                
                int idx_v = k * d;
                for (int i = 0; i < d; i++) {
                    acc_O[i] += p_val * float(shared_V[idx_v + i]);
                }
            }
            
            l_i = l_i * exp_diff + block_sum;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write Output
    if (tx < q_len) {
        float inv_l = 1.0f / l_i;
        int row_idx = (q_row_start + tx);
        for (int i = 0; i < d; i++) {
            o_ptr[row_idx * row_stride + i] = half(acc_O[i] * inv_l);
        }
    }
}
