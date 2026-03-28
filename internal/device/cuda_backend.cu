#include "cuda_bridge.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <math.h>

struct CudaContext {
  cudaStream_t stream;
  cublasHandle_t cublasHandle;
};

#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
  } while(0)

#define BLOCK_SIZE 256

__device__ float warpReduceSum(float val) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__global__ void addKernel(float* a, float* b, float* result, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = a[idx] + b[idx];
  }
}

__global__ void addScalarKernel(float* a, float val, float* result, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = a[idx] + val;
  }
}

__global__ void scaleKernel(float* a, float val, float* result, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = a[idx] * val;
  }
}

__global__ void tanhKernel(float* input, float* result, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = tanhf(input[idx]);
  }
}

__global__ void expKernel(float* input, float* result, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = expf(input[idx]);
  }
}

__global__ void softmaxKernel(float* input, float* result, int rows, int cols) {
  int row = blockIdx.x;
  if (row >= rows) return;
  
  float maxVal = -INFINITY;
  for (int j = 0; j < cols; j++) {
    maxVal = fmaxf(maxVal, input[row * cols + j]);
  }
  
  float sum = 0.0f;
  for (int j = 0; j < cols; j++) {
    sum += expf(input[row * cols + j] - maxVal);
  }
  
  for (int j = 0; j < cols; j++) {
    result[row * cols + j] = expf(input[row * cols + j] - maxVal) / sum;
  }
}

__global__ void layerNormKernel(float* input, float* gamma, float* beta, 
                                float* result, int rows, int cols, float eps) {
  int row = blockIdx.x;
  if (row >= rows) return;
  
  float mean = 0.0f;
  for (int j = 0; j < cols; j++) {
    mean += input[row * cols + j];
  }
  mean /= cols;
  
  float variance = 0.0f;
  for (int j = 0; j < cols; j++) {
    float diff = input[row * cols + j] - mean;
    variance += diff * diff;
  }
  variance /= cols;
  
  float std = sqrtf(variance + eps);
  
  for (int j = 0; j < cols; j++) {
    float normalized = (input[row * cols + j] - mean) / std;
    result[row * cols + j] = normalized * gamma[j] + beta[j];
  }
}

__global__ void castF32toF16Kernel(float* input, __half* result, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = __float2half(input[idx]);
  }
}

__global__ void castF16toF32Kernel(__half* input, float* result, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = __half2float(input[idx]);
  }
}

__global__ void castF32toF64Kernel(float* input, double* result, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = (double)input[idx];
  }
}

__global__ void castF64toF32Kernel(double* input, float* result, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = (float)input[idx];
  }
}

__global__ void castF32toI32Kernel(float* input, int32_t* result, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = (int32_t)roundf(input[idx]);
  }
}

__global__ void castI32toF32Kernel(int32_t* input, float* result, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = (float)input[idx];
  }
}

__global__ void castF32toI64Kernel(float* input, int64_t* result, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = (int64_t)roundf(input[idx]);
  }
}

__global__ void castI64toF32Kernel(int64_t* input, float* result, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = (float)input[idx];
  }
}

__global__ void castF32toU32Kernel(float* input, uint32_t* result, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = (uint32_t)max(0.0f, roundf(input[idx]));
  }
}

__global__ void castU32toF32Kernel(uint32_t* input, float* result, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = (float)input[idx];
  }
}

__global__ void castF32toU64Kernel(float* input, uint64_t* result, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = (uint64_t)max(0.0f, roundf(input[idx]));
  }
}

__global__ void castU64toF32Kernel(uint64_t* input, float* result, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = (float)input[idx];
  }
}

__global__ void castF32toI8Kernel(float* input, int8_t* result, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = (int8_t)max(-128.0f, min(127.0f, roundf(input[idx])));
  }
}

__global__ void castI8toF32Kernel(int8_t* input, float* result, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = (float)input[idx];
  }
}

__global__ void castF32toU8Kernel(float* input, uint8_t* result, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = (uint8_t)max(0.0f, min(255.0f, roundf(input[idx])));
  }
}

__global__ void castU8toF32Kernel(uint8_t* input, float* result, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = (float)input[idx];
  }
}

__global__ void addLayerNormKernel(float* residual, float* gamma, float* beta,
                                   float* result, int rows, int cols, float eps) {
  int row = blockIdx.x;
  if (row >= rows) return;
  
  float mean = 0.0f;
  for (int j = 0; j < cols; j++) {
    mean += residual[row * cols + j];
  }
  mean /= cols;
  
  float variance = 0.0f;
  for (int j = 0; j < cols; j++) {
    float diff = residual[row * cols + j] - mean;
    variance += diff * diff;
  }
  variance /= cols;
  
  float std = sqrtf(variance + eps);
  
  for (int j = 0; j < cols; j++) {
    float normalized = (residual[row * cols + j] - mean) / std;
    result[row * cols + j] = normalized * gamma[j] + beta[j];
  }
}

extern "C" {

CudaContextRef Cuda_Init() {
  CudaContext *ctx = new CudaContext();
  if (cudaStreamCreate(&ctx->stream) != cudaSuccess) {
    delete ctx;
    return nullptr;
  }
  cublasCreate(&ctx->cublasHandle);
  cublasSetStream(ctx->cublasHandle, ctx->stream);
  return (CudaContextRef)ctx;
}

int Cuda_GetDeviceCount() {
  int count;
  cudaGetDeviceCount(&count);
  return count;
}

void Cuda_SetDevice(CudaContextRef ctx, int deviceId) {
  cudaSetDevice(deviceId);
}

void Cuda_FreeContext(CudaContextRef ctx) {
  CudaContext *c = (CudaContext *)ctx;
  cudaStreamDestroy(c->stream);
  cublasDestroy(c->cublasHandle);
  delete c;
}

CudaBufferRef Cuda_Alloc(CudaContextRef ctx, int size) {
  void *ptr;
  if (cudaMalloc(&ptr, size) != cudaSuccess) {
    return nullptr;
  }
  return (CudaBufferRef)ptr;
}

void Cuda_FreeBuffer(CudaContextRef ctx, CudaBufferRef buf) { cudaFree(buf); }

void Cuda_CopyToDevice(CudaBufferRef buf, int offset, const void *data,
                       int size) {
  cudaMemcpy((char *)buf + offset, data, size, cudaMemcpyHostToDevice);
}

void Cuda_CopyToHost(CudaBufferRef buf, int offset, void *data, int size) {
  cudaMemcpy(data, (char *)buf + offset, size, cudaMemcpyDeviceToHost);
}

void Cuda_CopyDeviceToDevice(CudaBufferRef dst, CudaBufferRef src, int size) {
  cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
}

void *Cuda_GetBufferContents(CudaBufferRef buf) {
  return (void *)buf;
}

void Cuda_Linear_Fused(CudaContextRef ctx, CudaBufferRef input, int rows,
                       int inCols, CudaBufferRef weight, int outCols,
                       CudaBufferRef bias, CudaBufferRef result,
                       int activation) {
  CudaContext *c = (CudaContext *)ctx;
  float alpha = 1.0f;
  float beta = bias ? 1.0f : 0.0f;
  
  cublasSgemm(c->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
              outCols, rows, inCols,
              &alpha,
              (float *)weight, outCols,
              (float *)input, inCols,
              &beta,
              (float *)result, outCols);
}

void Cuda_LayerNorm(CudaContextRef ctx, CudaBufferRef input,
                    CudaBufferRef gamma, CudaBufferRef beta,
                    CudaBufferRef result, int rows, int cols, float eps) {
  int blocks = rows;
  layerNormKernel<<<blocks, 1, 0, ((CudaContext*)ctx)->stream>>>(
    (float*)input, (float*)gamma, (float*)beta,
    (float*)result, rows, cols, eps);
}

void Cuda_Softmax(CudaContextRef ctx, CudaBufferRef input, CudaBufferRef result,
                  int rows, int cols) {
  softmaxKernel<<<rows, 1, 0, ((CudaContext*)ctx)->stream>>>(
    (float*)input, (float*)result, rows, cols);
}

void Cuda_Gather(CudaContextRef ctx, CudaBufferRef table, CudaBufferRef indices,
                 CudaBufferRef output, int indicesCount, int cols) {
  cudaMemcpy(output, table, indicesCount * cols * sizeof(float), cudaMemcpyDeviceToDevice);
}

void Cuda_Attention_Fused(CudaContextRef ctx, CudaBufferRef q, CudaBufferRef k,
                          CudaBufferRef v, CudaBufferRef result, int batchSize,
                          int seqLen, int hiddenSize, float scale) {
  CudaContext *c = (CudaContext *)ctx;
  
  int headDim = hiddenSize / (hiddenSize / 64);
  int numHeads = hiddenSize / headDim;
  int totalHeads = batchSize * numHeads;
  
  int scoresSize = totalHeads * seqLen * seqLen;
  float* scores = NULL;
  cudaMalloc(&scores, scoresSize * sizeof(float));
  
  float alpha = scale;
  float beta = 0.0f;
  
  cublasSgemm(c->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
              seqLen, seqLen, headDim,
              &alpha,
              (float*)k, headDim,
              (float*)q, headDim,
              &beta,
              scores, seqLen);
  
  int blocks = (scoresSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
  expKernel<<<blocks, BLOCK_SIZE, 0, c->stream>>>((float*)scores, (float*)scores, scoresSize);
  
  softmaxKernel<<<totalHeads, 1, 0, c->stream>>>(scores, scores, totalHeads, seqLen);
  
  alpha = 1.0f;
  beta = 0.0f;
  cublasSgemm(c->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
              headDim, seqLen, seqLen,
              &alpha,
              (float*)v, headDim,
              (float*)scores, seqLen,
              &beta,
              (float*)result, headDim);
  
  cudaFree(scores);
}

void Cuda_ApplyRoPE(CudaContextRef ctx, CudaBufferRef data, int batchSize,
                    int seqLen, int numHeads, int headDim) {
}

void Cuda_Synchronize(CudaContextRef ctx) {
  CudaContext *c = (CudaContext *)ctx;
  cudaStreamSynchronize(c->stream);
}

void Cuda_GetMemoryInfo(CudaContextRef ctx, int64_t *free, int64_t *total) {
  size_t free_mem, total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);
  *free = (int64_t)free_mem;
  *total = (int64_t)total_mem;
}

void Cuda_Add(CudaContextRef ctx, CudaBufferRef a, CudaBufferRef b, CudaBufferRef result, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  addKernel<<<blocks, BLOCK_SIZE, 0, ((CudaContext*)ctx)->stream>>>(
    (float*)a, (float*)b, (float*)result, size);
}

void Cuda_AddScalar(CudaContextRef ctx, CudaBufferRef a, float val, CudaBufferRef result, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  addScalarKernel<<<blocks, BLOCK_SIZE, 0, ((CudaContext*)ctx)->stream>>>(
    (float*)a, val, (float*)result, size);
}

void Cuda_Scale(CudaContextRef ctx, CudaBufferRef a, float val, CudaBufferRef result, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  scaleKernel<<<blocks, BLOCK_SIZE, 0, ((CudaContext*)ctx)->stream>>>(
    (float*)a, val, (float*)result, size);
}

void Cuda_Tanh(CudaContextRef ctx, CudaBufferRef input, CudaBufferRef result, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  tanhKernel<<<blocks, BLOCK_SIZE, 0, ((CudaContext*)ctx)->stream>>>(
    (float*)input, (float*)result, size);
}

void Cuda_Cast_F32_to_F16(CudaContextRef ctx, CudaBufferRef input, CudaBufferRef result, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  castF32toF16Kernel<<<blocks, BLOCK_SIZE, 0, ((CudaContext*)ctx)->stream>>>(
    (float*)input, (__half*)result, size);
}

void Cuda_Cast_F16_to_F32(CudaContextRef ctx, CudaBufferRef input, CudaBufferRef result, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error before kernel: %s\n", cudaGetErrorString(err));
  }
  castF16toF32Kernel<<<blocks, BLOCK_SIZE, 0, ((CudaContext*)ctx)->stream>>>(
    (__half*)input, (float*)result, size);
  cudaStreamSynchronize(((CudaContext*)ctx)->stream);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error after Cast_F16_to_F32 kernel: %s\n", cudaGetErrorString(err));
  }
}

void Cuda_Cast_F32_to_F64(CudaContextRef ctx, CudaBufferRef input, CudaBufferRef result, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  castF32toF64Kernel<<<blocks, BLOCK_SIZE, 0, ((CudaContext*)ctx)->stream>>>(
    (float*)input, (double*)result, size);
}

void Cuda_Cast_F64_to_F32(CudaContextRef ctx, CudaBufferRef input, CudaBufferRef result, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  castF64toF32Kernel<<<blocks, BLOCK_SIZE, 0, ((CudaContext*)ctx)->stream>>>(
    (double*)input, (float*)result, size);
}

void Cuda_Cast_F32_to_I32(CudaContextRef ctx, CudaBufferRef input, CudaBufferRef result, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  castF32toI32Kernel<<<blocks, BLOCK_SIZE, 0, ((CudaContext*)ctx)->stream>>>(
    (float*)input, (int32_t*)result, size);
}

void Cuda_Cast_I32_to_F32(CudaContextRef ctx, CudaBufferRef input, CudaBufferRef result, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  castI32toF32Kernel<<<blocks, BLOCK_SIZE, 0, ((CudaContext*)ctx)->stream>>>(
    (int32_t*)input, (float*)result, size);
}

void Cuda_Cast_F32_to_I64(CudaContextRef ctx, CudaBufferRef input, CudaBufferRef result, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  castF32toI64Kernel<<<blocks, BLOCK_SIZE, 0, ((CudaContext*)ctx)->stream>>>(
    (float*)input, (int64_t*)result, size);
}

void Cuda_Cast_I64_to_F32(CudaContextRef ctx, CudaBufferRef input, CudaBufferRef result, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  castI64toF32Kernel<<<blocks, BLOCK_SIZE, 0, ((CudaContext*)ctx)->stream>>>(
    (int64_t*)input, (float*)result, size);
}

void Cuda_Cast_F32_to_U32(CudaContextRef ctx, CudaBufferRef input, CudaBufferRef result, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  castF32toU32Kernel<<<blocks, BLOCK_SIZE, 0, ((CudaContext*)ctx)->stream>>>(
    (float*)input, (uint32_t*)result, size);
}

void Cuda_Cast_U32_to_F32(CudaContextRef ctx, CudaBufferRef input, CudaBufferRef result, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  castU32toF32Kernel<<<blocks, BLOCK_SIZE, 0, ((CudaContext*)ctx)->stream>>>(
    (uint32_t*)input, (float*)result, size);
}

void Cuda_Cast_F32_to_U64(CudaContextRef ctx, CudaBufferRef input, CudaBufferRef result, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  castF32toU64Kernel<<<blocks, BLOCK_SIZE, 0, ((CudaContext*)ctx)->stream>>>(
    (float*)input, (uint64_t*)result, size);
}

void Cuda_Cast_U64_to_F32(CudaContextRef ctx, CudaBufferRef input, CudaBufferRef result, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  castU64toF32Kernel<<<blocks, BLOCK_SIZE, 0, ((CudaContext*)ctx)->stream>>>(
    (uint64_t*)input, (float*)result, size);
}

void Cuda_Cast_F32_to_I8(CudaContextRef ctx, CudaBufferRef input, CudaBufferRef result, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  castF32toI8Kernel<<<blocks, BLOCK_SIZE, 0, ((CudaContext*)ctx)->stream>>>(
    (float*)input, (int8_t*)result, size);
}

void Cuda_Cast_I8_to_F32(CudaContextRef ctx, CudaBufferRef input, CudaBufferRef result, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  castI8toF32Kernel<<<blocks, BLOCK_SIZE, 0, ((CudaContext*)ctx)->stream>>>(
    (int8_t*)input, (float*)result, size);
}

void Cuda_Cast_F32_to_U8(CudaContextRef ctx, CudaBufferRef input, CudaBufferRef result, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  castF32toU8Kernel<<<blocks, BLOCK_SIZE, 0, ((CudaContext*)ctx)->stream>>>(
    (float*)input, (uint8_t*)result, size);
}

void Cuda_Cast_U8_to_F32(CudaContextRef ctx, CudaBufferRef input, CudaBufferRef result, int size) {
  int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  castU8toF32Kernel<<<blocks, BLOCK_SIZE, 0, ((CudaContext*)ctx)->stream>>>(
    (uint8_t*)input, (float*)result, size);
}

void Cuda_Slice(CudaContextRef ctx, CudaBufferRef input, CudaBufferRef output,
                int srcRow, int srcCol, int rows, int cols, int srcCols) {
  cudaMemcpy2D(output, cols * sizeof(float),
               (char*)input + (srcRow * srcCols + srcCol) * sizeof(float), srcCols * sizeof(float),
               cols * sizeof(float), rows, cudaMemcpyDeviceToDevice);
}

void Cuda_Paste(CudaContextRef ctx, CudaBufferRef dst, CudaBufferRef src,
                int dstRow, int dstCol, int srcRow, int srcCol,
                int rows, int cols, int dstCols, int srcCols) {
  cudaMemcpy2D((char*)dst + (dstRow * dstCols + dstCol) * sizeof(float), dstCols * sizeof(float),
               (char*)src + (srcRow * srcCols + srcCol) * sizeof(float), srcCols * sizeof(float),
               cols * sizeof(float), rows, cudaMemcpyDeviceToDevice);
}

void Cuda_AddLayerNorm(CudaContextRef ctx, CudaBufferRef residual,
                       CudaBufferRef gamma, CudaBufferRef beta,
                       CudaBufferRef result, int rows, int cols, float eps) {
  int blocks = rows;
  addLayerNormKernel<<<blocks, 1, 0, ((CudaContext*)ctx)->stream>>>(
    (float*)residual, (float*)gamma, (float*)beta,
    (float*)result, rows, cols, eps);
}

}
