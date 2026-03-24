#ifndef CUDA_BRIDGE_H
#define CUDA_BRIDGE_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pointer types
typedef void *CudaContextRef;
typedef void *CudaBufferRef;

// Setup
// Device Management
CudaContextRef Cuda_Init();
int Cuda_GetDeviceCount();
void Cuda_SetDevice(CudaContextRef ctx, int deviceId);
void Cuda_FreeContext(CudaContextRef ctx);

// Buffer Management
CudaBufferRef Cuda_Alloc(CudaContextRef ctx, int size);
void Cuda_FreeBuffer(CudaContextRef ctx, CudaBufferRef buf);
void Cuda_CopyToDevice(CudaBufferRef buf, int offset, const void *data,
                       int size);
void Cuda_CopyToHost(CudaBufferRef buf, int offset, void *data, int size);
void *Cuda_GetBufferContents(CudaBufferRef buf);

// Ops - MatX Fused
void Cuda_Linear_Fused(CudaContextRef ctx, CudaBufferRef input, int rows,
                       int inCols, CudaBufferRef weight, int outCols,
                       CudaBufferRef bias, CudaBufferRef result,
                       int activation);

void Cuda_LayerNorm(CudaContextRef ctx, CudaBufferRef input,
                    CudaBufferRef gamma, CudaBufferRef beta,
                    CudaBufferRef result, int rows, int cols, float eps);

void Cuda_AddLayerNorm(CudaContextRef ctx, CudaBufferRef residual,
                       CudaBufferRef gamma, CudaBufferRef beta,
                       CudaBufferRef result, int rows, int cols, float eps);

void Cuda_Softmax(CudaContextRef ctx, CudaBufferRef input, CudaBufferRef result,
                  int rows, int cols);

void Cuda_Gather(CudaContextRef ctx, CudaBufferRef table, CudaBufferRef indices,
                 CudaBufferRef output, int indicesCount, int cols);

void Cuda_Attention_Fused(CudaContextRef ctx, CudaBufferRef q, CudaBufferRef k,
                          CudaBufferRef v, CudaBufferRef result, int batchSize,
                          int seqLen, int hiddenSize, float scale);

void Cuda_ApplyRoPE(CudaContextRef ctx, CudaBufferRef data, int batchSize,
                    int seqLen, int numHeads, int headDim);

void Cuda_Synchronize(CudaContextRef ctx);

// Memory Info
void Cuda_GetMemoryInfo(CudaContextRef ctx, int64_t *free, int64_t *total);

// Tensor Operations
void Cuda_Add(CudaContextRef ctx, CudaBufferRef a, CudaBufferRef b, CudaBufferRef result, int size);
void Cuda_AddScalar(CudaContextRef ctx, CudaBufferRef a, float val, CudaBufferRef result, int size);
void Cuda_Scale(CudaContextRef ctx, CudaBufferRef a, float val, CudaBufferRef result, int size);
void Cuda_Tanh(CudaContextRef ctx, CudaBufferRef input, CudaBufferRef result, int size);

// Cast operations
void Cuda_Cast_F32_to_F16(CudaContextRef ctx, CudaBufferRef input, CudaBufferRef result, int size);
void Cuda_Cast_F16_to_F32(CudaContextRef ctx, CudaBufferRef input, CudaBufferRef result, int size);

// Slice operation (extract sub-matrix)
void Cuda_Slice(CudaContextRef ctx, CudaBufferRef input, CudaBufferRef output, 
                int srcRow, int srcCol, int rows, int cols, int srcCols);

// Paste operation (copy sub-matrix into destination)
void Cuda_Paste(CudaContextRef ctx, CudaBufferRef dst, CudaBufferRef src,
                int dstRow, int dstCol, int srcRow, int srcCol, 
                int rows, int cols, int dstCols, int srcCols);

#ifdef __cplusplus
}
#endif

#endif
