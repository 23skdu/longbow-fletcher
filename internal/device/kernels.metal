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

// LayerNorm
// Input: matrix (rows x cols)
// Each thread handles ONE ROW (inefficient for large rows, but simple for now)
// Or use threadgroup memory parallel reduction?
// For logic simplicity: 1 kernel per element?
// LayerNorm requires Mean/Var of the row. Use 2 passes or 1 pass per thread-per-row?
// Let's implement a simple "Thread per array element" kernel which unfortunately is hard for LayerNorm
// because we need row statistics.
// Alternative: "Thread per row".
kernel void layernorm_kernel(device const float *input [[ buffer(0) ]],
                             device const float *gamma [[ buffer(1) ]],
                             device const float *beta [[ buffer(2) ]],
                             device float *output [[ buffer(3) ]],
                             constant int &cols [[ buffer(4) ]],
                             constant float &eps [[ buffer(5) ]],
                             uint row_idx [[ thread_position_in_grid ]]) {
                             
    int offset = row_idx * cols;
    
    // 1. Calculate Mean
    float sum = 0.0;
    for (int i = 0; i < cols; i++) {
        sum += input[offset + i];
    }
    float mean = sum / float(cols);
    
    // 2. Calculate Variance
    float sum_sq_diff = 0.0;
    for (int i = 0; i < cols; i++) {
        float diff = input[offset + i] - mean;
        sum_sq_diff += diff * diff;
    }
    float variance = sum_sq_diff / float(cols);
    float inv_std = 1.0 / sqrt(variance + eps);
    
    // 3. Normalize
    for (int i = 0; i < cols; i++) {
        output[offset + i] = (input[offset + i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

kernel void softmax_kernel(device const float *input [[ buffer(0) ]],
                           device float *output [[ buffer(1) ]],
                           constant int &cols [[ buffer(2) ]],
                           uint row_idx [[ thread_position_in_grid ]]) {
    int offset = row_idx * cols;
    
    // 1. Find Max
    float max_val = -1e38; // float min
    for (int i = 0; i < cols; i++) {
        float v = input[offset + i];
        if (v > max_val) max_val = v;
    }
    
    // 2. Sum Exp
    float sum_exp = 0.0;
    for (int i = 0; i < cols; i++) {
        float val = exp(input[offset + i] - max_val);
        output[offset + i] = val; // Temp store
        sum_exp += val;
    }
    
    // 3. Normalize
    float inv_sum = 1.0 / sum_exp;
    for (int i = 0; i < cols; i++) {
        output[offset + i] *= inv_sum;
    }
}

kernel void gather_kernel(device const float *table [[ buffer(0) ]],
                          device float *output [[ buffer(1) ]],
                          device const int *indices [[ buffer(2) ]],
                          constant int &cols [[ buffer(3) ]],
                          uint2 id [[ thread_position_in_grid ]]) {
    // 2D dispatch: x = output_row_idx (batch item), y = col_idx (feature)
    // Actually, simple dispatch: 1D over entire output elements?
    // Or 1D over rows?
    // Let's use 2D dispatch: (rows, cols)
    
    // We assume indices length = output rows.
    uint row = id.x;
    uint col = id.y;
    
    // id.x is index in 'indices' array.
    int table_row = indices[row];
    
    // table index
    // table is (vocab_size x cols)
    // access: table_row * cols + col
    output[row * cols + col] = table[table_row * cols + col];
}

kernel void add_bias_kernel(device float *component [[ buffer(0) ]],
                            device const float *bias [[ buffer(1) ]],
                            constant int &cols [[ buffer(2) ]],
                            uint2 id [[ thread_position_in_grid ]]) {
    uint row = id.x;
    uint col = id.y;
    
    // Bounds check? Logic usually guarantees sizing if grid is exact.
    // If we use dispatchThreads, Metal handles bounds if threadgroup is larger than grid?
    // dispatchThreads sends exact grid size generally? 
    // Wait, if threadgroup size is fixed (e.g. 32x32), and grid is 100x100.
    // Metal (if using dispatchThreads API) handles boundaries if we use [[thread_position_in_grid]].
    // BUT we must allow for partial threadgroups.
    // If using `dispatchThreads:threadsPerThreadgroup:`, Metal clamps threads?
    // No, standard Metal compute shader:
    // We should do bounds check if we suspect overrun.
    // But let's assume valid id for now.
    
    component[row * cols + col] += bias[col];
}
