#include "cuda_bridge.h"
#include <cuda_runtime.h>
#include <iostream>
#include <matx.h>

using namespace matx;

// Internal context structure
struct CudaContext {
  cudaStream_t stream;
  // We could add cached plans or other MatX state here
};

extern "C" {

CudaContextRef Cuda_Init() {
  CudaContext *ctx = new CudaContext();
  if (cudaStreamCreate(&ctx->stream) != cudaSuccess) {
    delete ctx;
    return nullptr;
  }
  cublasCreate(&ctx->cublasHandle);
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

void *Cuda_GetBufferContents(CudaBufferRef buf) {
  return (void *)buf; // Direct access (managed or device)
}

void Cuda_Linear_Fused(CudaContextRef ctx, CudaBufferRef input, int rows,
                       int inCols, CudaBufferRef weight, int outCols,
                       CudaBufferRef bias, CudaBufferRef result,
                       int activation) {
  CudaContext *c = (CudaContext *)ctx;

  auto in_view = make_tensor<float>((float *)input, {rows, inCols});
  auto wt_view = make_tensor<float>((float *)weight, {inCols, outCols});
  auto res_view = make_tensor<float>((float *)result, {rows, outCols});

  if (bias) {
    auto bias_view = make_tensor<float>((float *)bias, {outCols});
    (res_view = matmul(in_view, wt_view) + bias_view).run(c->stream);
  } else {
    (res_view = matmul(in_view, wt_view)).run(c->stream);
  }

  if (activation == 1) { // GELU
    (res_view = gelu(res_view)).run(c->stream);
  } else if (activation == 4) { // SwiGLU
    // x * swish(y)
    // res_view has [rows, outCols]
    int half = outCols / 2;
    auto slice1 = res_view.Slice({0, 0}, {rows, half});
    auto slice2 = res_view.Slice({0, half}, {rows, outCols});
    (slice1 = slice1 * (slice2 * sigmoid(slice2))).run(c->stream);
  }
}

void Cuda_LayerNorm(CudaContextRef ctx, CudaBufferRef input,
                    CudaBufferRef gamma, CudaBufferRef beta,
                    CudaBufferRef result, int rows, int cols, float eps) {
  CudaContext *c = (CudaContext *)ctx;

  auto in_view = make_tensor<float>((float *)input, {rows, cols});
  auto g_view = make_tensor<float>((float *)gamma, {cols});
  auto b_view = make_tensor<float>((float *)beta, {cols});
  auto res_view = make_tensor<float>((float *)result, {rows, cols});

  // MatX LayerNorm fusion
  (res_view = scale(g_view, (in_view - mean<1>(in_view)) *
                                rsqrt(var<1>(in_view) + eps)) +
              b_view)
      .run(c->stream);
}

void Cuda_Softmax(CudaContextRef ctx, CudaBufferRef input, CudaBufferRef result,
                  int rows, int cols) {
  CudaContext *c = (CudaContext *)ctx;
  auto in_view = make_tensor<float>((float *)input, {rows, cols});
  auto res_view = make_tensor<float>((float *)result, {rows, cols});

  (res_view = softmax(in_view)).run(c->stream);
}

void Cuda_Gather(CudaContextRef ctx, CudaBufferRef table, CudaBufferRef indices,
                 CudaBufferRef output, int indicesCount, int cols) {
  CudaContext *c = (CudaContext *)ctx;
  auto t_view = make_tensor<float>((float *)table,
                                   {indicesCount, cols}); // simplified view
  auto i_view = make_tensor<int>((int *)indices, {indicesCount});
  auto o_view = make_tensor<float>((float *)output, {indicesCount, cols});

  // MatX Gather
  // (o_view = gather<0>(t_view, i_view)).run(c->stream);
}

void Cuda_Attention_Fused(CudaContextRef ctx, CudaBufferRef q, CudaBufferRef k,
                          CudaBufferRef v, CudaBufferRef result, int batchSize,
                          int seqLen, int hiddenSize, float scale) {
  CudaContext *c = (CudaContext *)ctx;

  auto q_view = make_tensor<float>((float *)q, {batchSize, seqLen, hiddenSize});
  auto k_view = make_tensor<float>((float *)k, {batchSize, seqLen, hiddenSize});
  auto v_view = make_tensor<float>((float *)v, {batchSize, seqLen, hiddenSize});
  auto res_view =
      make_tensor<float>((float *)result, {batchSize, seqLen, hiddenSize});

  // Scores = Softmax( (Q * K^T) * scale )
  // We use MatX's lazy evaluation for multi-head attention fusion
  auto scores = softmax(matmul(q_view, transpose(k_view, {0, 2, 1})) * scale);

  // Context = Scores * V
  (res_view = matmul(scores, v_view)).run(c->stream);
}

void Cuda_ApplyRoPE(CudaContextRef ctx, CudaBufferRef data, int batchSize,
                    int seqLen, int numHeads, int headDim) {
  CudaContext *c = (CudaContext *)ctx;
  auto d_view = make_tensor<float>((float *)data,
                                   {batchSize * seqLen, numHeads, headDim});

  // RoPE in MatX can be expressed as a transform or a custom kernel if not
  // built-in. For now, we provide the structure.
}

// ... existing functions ...

void Cuda_Linear_Fused_F16(CudaContextRef ctx, CudaBufferRef input, int rows,
                           int inCols, CudaBufferRef weight, int outCols,
                           CudaBufferRef bias, CudaBufferRef result,
                           int activation) {
  CudaContext *c = (CudaContext *)ctx;

  auto in_view = make_tensor<__half>((__half *)input, {rows, inCols});
  auto wt_view = make_tensor<__half>((__half *)weight, {inCols, outCols});
  auto res_view = make_tensor<__half>((__half *)result, {rows, outCols});

  if (bias) {
    auto bias_view = make_tensor<__half>((__half *)bias, {outCols});
    (res_view = matmul(in_view, wt_view) + bias_view).run(c->stream);
  } else {
    (res_view = matmul(in_view, wt_view)).run(c->stream);
  }

  if (activation == 1) { // GELU
    (res_view = gelu(res_view)).run(c->stream);
  } else if (activation == 4) { // SwiGLU
    int half = outCols / 2;
    auto slice1 = res_view.Slice({0, 0}, {rows, half});
    auto slice2 = res_view.Slice({0, half}, {rows, outCols});
    (slice1 = slice1 * (slice2 * sigmoid(slice2))).run(c->stream);
  }
}

void Cuda_LayerNorm_F16(CudaContextRef ctx, CudaBufferRef input,
                        CudaBufferRef gamma, CudaBufferRef beta,
                        CudaBufferRef result, int rows, int cols, float eps) {
  CudaContext *c = (CudaContext *)ctx;

  // Mixed precision LayerNorm: Input halves, Compute float, Output halves used
  // automatically by some libs? MatX allows mixed types? Use __half everywhere
  // for now.
  auto in_view = make_tensor<__half>((__half *)input, {rows, cols});
  auto g_view = make_tensor<__half>((__half *)gamma, {cols});
  auto b_view = make_tensor<__half>((__half *)beta, {cols});
  auto res_view = make_tensor<__half>((__half *)result, {rows, cols});

  // Cast eps to __half
  __half eps_h = (__half)eps;

  (res_view = scale(g_view, (in_view - mean<1>(in_view)) *
                                rsqrt(var<1>(in_view) + eps_h)) +
              b_view)
      .run(c->stream);
}

void Cuda_Softmax_F16(CudaContextRef ctx, CudaBufferRef input,
                      CudaBufferRef result, int rows, int cols) {
  CudaContext *c = (CudaContext *)ctx;
  auto in_view = make_tensor<__half>((__half *)input, {rows, cols});
  auto res_view = make_tensor<__half>((__half *)result, {rows, cols});

  (res_view = softmax(in_view)).run(c->stream);
}

void Cuda_Gather_F16(CudaContextRef ctx, CudaBufferRef table,
                     CudaBufferRef indices, CudaBufferRef output,
                     int indicesCount, int cols) {
  // Not implemented fully yet, placeholder
}

void Cuda_Attention_Fused_F16(CudaContextRef ctx, CudaBufferRef q,
                              CudaBufferRef k, CudaBufferRef v,
                              CudaBufferRef result, int batchSize, int seqLen,
                              int hiddenSize, float scale) {
  CudaContext *c = (CudaContext *)ctx;

  auto q_view =
      make_tensor<__half>((__half *)q, {batchSize, seqLen, hiddenSize});
  auto k_view =
      make_tensor<__half>((__half *)k, {batchSize, seqLen, hiddenSize});
  auto v_view =
      make_tensor<__half>((__half *)v, {batchSize, seqLen, hiddenSize});
  auto res_view =
      make_tensor<__half>((__half *)result, {batchSize, seqLen, hiddenSize});

  __half scale_h = (__half)scale;

  auto scores = softmax(matmul(q_view, transpose(k_view, {0, 2, 1})) * scale_h);
  (res_view = matmul(scores, v_view)).run(c->stream);
}

void Cuda_ApplyRoPE_F16(CudaContextRef ctx, CudaBufferRef data, int batchSize,
                        int seqLen, int numHeads, int headDim) {
  // Placeholder
}

void Cuda_Synchronize(CudaContextRef ctx) {
  CudaContext *c = (CudaContext *)ctx;
  cudaStreamSynchronize(c->stream);
}

} // extern "C"
