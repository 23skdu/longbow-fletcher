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

#ifdef __cplusplus
}
#endif

#endif
