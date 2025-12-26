#ifndef METAL_BRIDGE_H
#define METAL_BRIDGE_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pointer types
typedef void *MetalContextRef;
typedef void *MetalBufferRef;

// Setup
MetalContextRef Metal_Init(const char *libSource);
void Metal_FreeContext(MetalContextRef ctx);

// Buffer Management
MetalBufferRef Metal_Alloc(MetalContextRef ctx, int size);
void Metal_FreeBuffer(MetalContextRef ctx, MetalBufferRef buf);
void Metal_CopyToDevice(MetalBufferRef buf, int offset, const void *data,
                        int size);
void Metal_CopyToHost(MetalBufferRef buf, int offset, void *data, int size);
void Metal_Memset(MetalBufferRef buf, int offset, int value, int size);
// Index is absolute or relative? Let's say relative to buffer. But here we take
// buf directly. Let's keep SetAt absolute for now or add offset. Better:
// Metal_SetAt(buf, offset, val).
void Metal_SetAt(MetalBufferRef buf, int offset, float val);

// Ops
void Metal_Add(MetalContextRef ctx, MetalBufferRef a, int offA,
               MetalBufferRef b, int offB, MetalBufferRef result, int offRes,
               int count);
void Metal_AddScalar(MetalContextRef ctx, MetalBufferRef a, int offA, float val,
                     MetalBufferRef result, int offRes, int count);
void Metal_Scale(MetalContextRef ctx, MetalBufferRef a, int offA, float val,
                 MetalBufferRef result, int offRes, int count);
void Metal_Scale_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                     uint16_t val, MetalBufferRef result, int offRes,
                     int count);
void Metal_Tanh(MetalContextRef ctx, MetalBufferRef a, int offA,
                MetalBufferRef result, int offRes, int count);
void Metal_Tanh_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                    MetalBufferRef result, int offRes, int count);
void Metal_Gelu(MetalContextRef ctx, MetalBufferRef a, int offA,
                MetalBufferRef result, int offRes, int count);
void Metal_Gelu_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                    MetalBufferRef result, int offRes, int count);
void Metal_LayerNorm(MetalContextRef ctx, MetalBufferRef input, int offIn,
                     MetalBufferRef gamma, int offGamma, MetalBufferRef beta,
                     int offBeta, MetalBufferRef result, int offRes, int rows,
                     int cols, float eps);
void Metal_Softmax(MetalContextRef ctx, MetalBufferRef input, int offIn,
                   MetalBufferRef result, int offRes, int rows, int cols);
void Metal_Softmax_F16(MetalContextRef ctx, MetalBufferRef input, int offIn,
                       MetalBufferRef result, int offRes, int rows, int cols);
void Metal_LayerNorm_F16(MetalContextRef ctx, MetalBufferRef input, int offIn,
                         MetalBufferRef gamma, int offGamma,
                         MetalBufferRef beta, int offBeta,
                         MetalBufferRef result, int offRes, int rows, int cols,
                         float eps);
void Metal_AddBias_F16(MetalContextRef ctx, MetalBufferRef matrix, int offMat,
                       MetalBufferRef bias, int offBias, MetalBufferRef result,
                       int offRes, int rows, int cols);
void Metal_Linear_Graph(MetalContextRef ctx, MetalBufferRef input, int offIn,
                        int rows, int inCols, MetalBufferRef weight,
                        int offWeight, int outCols, MetalBufferRef bias,
                        int offBias, MetalBufferRef result, int offRes);
void Metal_LinearActivation_Graph(MetalContextRef ctx, MetalBufferRef input,
                                  int offIn, int rows, int inCols,
                                  MetalBufferRef weight, int offWeight,
                                  int outCols, MetalBufferRef bias, int offBias,
                                  MetalBufferRef result, int offRes,
                                  int activationType);
void Metal_Attention_Graph(MetalContextRef ctx, MetalBufferRef q, int offQ,
                           MetalBufferRef k, int offK, MetalBufferRef v,
                           int offV, MetalBufferRef result, int offRes,
                           int batchSize, int seqLen, int hiddenSize,
                           float scale);
void Metal_Gather(MetalContextRef ctx, MetalBufferRef table, int offTable,
                  MetalBufferRef indices, int offIndices, MetalBufferRef output,
                  int offOut, int indicesCount, int cols);
void Metal_AddBias(MetalContextRef ctx, MetalBufferRef component, int offComp,
                   MetalBufferRef bias, int offBias, int rows, int cols);

// Matrix Mul (MPS)
// C = A * B. A (r x inner), B (inner x c), C (r x c)
// Transpose flags? For now assume standard A x B.
// We support A * B^T via flags if needed, or caller handles it.
// The Go code uses Transpose T() often.
// MPSMatrixMultiplication supports transpose.
// Let's forward transpose flags.
void Metal_MatMul(MetalContextRef ctx, MetalBufferRef a, int offA, bool transA,
                  MetalBufferRef b, int offB, bool transB, MetalBufferRef c,
                  int offC, int M, int N, int K);

// FP16 MatMul for 2x GPU performance (requires FP16 buffers)
void Metal_MatMul_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                      bool transA, MetalBufferRef b, int offB, bool transB,
                      MetalBufferRef c, int offC, int M, int N, int K);

// Batched MatMul for attention: process multiple sequences at once
// Assumes uniform matrix sizes. batchCount = number of independent MatMuls.
// strideA/B/C = byte offset between consecutive matrices in the batch
void Metal_BatchedMatMul(MetalContextRef ctx, MetalBufferRef a, int offA,
                         int strideA, bool transA, MetalBufferRef b, int offB,
                         int strideB, bool transB, MetalBufferRef c, int offC,
                         int strideC, int M, int N, int K, int batchCount);

void Metal_Synchronize(MetalContextRef ctx);

#ifdef __cplusplus
}
#endif

#endif
