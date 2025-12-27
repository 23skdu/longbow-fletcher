#import "metal_bridge.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
// MPSGraph import removed as we are not using it to assume strict dependency
// avoidance #import
// <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

@interface MetalWrapper : NSObject
@property(strong) id<MTLDevice> device;
@property(strong) id<MTLCommandQueue> commandQueue;
@property(strong) id<MTLLibrary> library;
// FP32 Pipelines
@property(strong) id<MTLComputePipelineState> pipelineAdd;
@property(strong) id<MTLComputePipelineState> pipelineAddScalar;
@property(strong) id<MTLComputePipelineState> pipelineScale;
@property(strong) id<MTLComputePipelineState> pipelineTanh;
@property(strong) id<MTLComputePipelineState> pipelineGelu;
@property(strong) id<MTLComputePipelineState> pipelineLayerNorm;
@property(strong) id<MTLComputePipelineState> pipelineSoftmax;
@property(strong) id<MTLComputePipelineState> pipelineGather;
@property(strong) id<MTLComputePipelineState> pipelineAddBias;
// FP16 Pipelines
@property(strong) id<MTLComputePipelineState> pipelineAdd_F16;
@property(strong) id<MTLComputePipelineState> pipelineAddScalar_F16;
@property(strong) id<MTLComputePipelineState> pipelineScale_F16;
@property(strong) id<MTLComputePipelineState> pipelineTanh_F16;
@property(strong) id<MTLComputePipelineState> pipelineGelu_F16;
@property(strong) id<MTLComputePipelineState> pipelineSoftmax_F16;
@property(strong) id<MTLComputePipelineState> pipelineLayerNorm_F16;
@property(strong) id<MTLComputePipelineState> pipelineAddBias_F16;
@property(strong) id<MTLComputePipelineState> pipelineRope_F16;
@property(strong) id<MTLComputePipelineState> pipelineSwiglu_F16;

@property(strong) id<MTLCommandBuffer> currentCommandBuffer;
@property(strong) id<MTLComputeCommandEncoder> currentEncoder;
@property(strong) id<MTLCommandBuffer> lastMPSBuffer;
@end

@implementation MetalWrapper
- (void)ensureEncoder {
  @synchronized(self) {
    if (!self.currentCommandBuffer) {
      self.currentCommandBuffer = [self.commandQueue commandBuffer];
    }
    if (!self.currentEncoder) {
      self.currentEncoder = [self.currentCommandBuffer computeCommandEncoder];
    }
  }
}
- (void)stopEncoder {
  @synchronized(self) {
    if (self.currentEncoder) {
      [self.currentEncoder endEncoding];
      self.currentEncoder = nil;
    }
  }
}

- (void)flush {
  @synchronized(self) {
    if (self.currentEncoder) {
      [self.currentEncoder endEncoding];
      self.currentEncoder = nil;
    }
    if (self.currentCommandBuffer) {
      [self.currentCommandBuffer commit];
      self.currentCommandBuffer = nil;
    }
  }
}
- (void)fullSync {
  @synchronized(self) {
    [self flush];
    if (self.lastMPSBuffer) {
      self.lastMPSBuffer = nil;
    }
    id<MTLCommandBuffer> barrier = [self.commandQueue commandBuffer];
    [barrier commit];
    [barrier waitUntilCompleted];
  }
}
@end

MetalContextRef Metal_Init(const char *libSource) {
  MetalWrapper *ctx = [[MetalWrapper alloc] init];
  ctx.device = MTLCreateSystemDefaultDevice();
  if (!ctx.device) {
    printf("Error: Metal device not found\n");
    return NULL;
  }
  printf("Metal Device: %s\n", [ctx.device.name UTF8String]);
  ctx.commandQueue = [ctx.device newCommandQueue];

  NSError *error = nil;
  NSString *src = [NSString stringWithUTF8String:libSource];
  ctx.library = [ctx.device newLibraryWithSource:src options:nil error:&error];
  if (error) {
    printf("Error compiling Metal headers: %s\n",
           [[error localizedDescription] UTF8String]);
    return NULL;
  }

// Helper macro for loading kernels
#define LOAD(name, prop)                                                       \
  ctx.prop = [ctx.device                                                       \
      newComputePipelineStateWithFunction:[ctx.library                         \
                                              newFunctionWithName:name]        \
                                    error:&error];                             \
  if (error)                                                                   \
    printf("Failed to load %s: %s\n", [name UTF8String],                       \
           [[error localizedDescription] UTF8String]);

  LOAD(@"add_kernel", pipelineAdd);
  LOAD(@"add_scalar_kernel", pipelineAddScalar);
  LOAD(@"scale_kernel", pipelineScale);
  LOAD(@"tanh_kernel", pipelineTanh);
  LOAD(@"gelu_kernel", pipelineGelu);
  LOAD(@"layernorm_kernel", pipelineLayerNorm);
  LOAD(@"softmax_kernel", pipelineSoftmax);
  LOAD(@"gather_kernel", pipelineGather);
  LOAD(@"add_bias_kernel", pipelineAddBias);

  LOAD(@"add_kernel_f16", pipelineAdd_F16);
  LOAD(@"add_scalar_kernel_f16", pipelineAddScalar_F16);
  LOAD(@"scale_kernel_f16", pipelineScale_F16);
  LOAD(@"tanh_kernel_f16", pipelineTanh_F16);
  LOAD(@"gelu_kernel_f16", pipelineGelu_F16);
  LOAD(@"softmax_kernel_f16", pipelineSoftmax_F16);
  LOAD(@"layernorm_kernel_f16", pipelineLayerNorm_F16);
  LOAD(@"add_bias_kernel_f16", pipelineAddBias_F16);
  LOAD(@"rope_kernel_f16", pipelineRope_F16);
  LOAD(@"swiglu_kernel_f16", pipelineSwiglu_F16);

  return (__bridge_retained MetalContextRef)ctx;
}

void Metal_FreeContext(MetalContextRef ctx) {
  if (ctx) {
    MetalWrapper *wrapper = (__bridge_transfer MetalWrapper *)ctx;
    [wrapper flush];
    wrapper = nil;
  }
}

MetalBufferRef Metal_Alloc(MetalContextRef ctx, int size) {
  MetalWrapper *wrapper = (__bridge MetalWrapper *)ctx;
  id<MTLBuffer> buf =
      [wrapper.device newBufferWithLength:size
                                  options:MTLResourceStorageModeShared];
  return (__bridge_retained MetalBufferRef)buf;
}
void Metal_FreeBuffer(MetalContextRef ctx, MetalBufferRef buf) {
  if (buf) {
    id<MTLBuffer> buffer = (__bridge_transfer id<MTLBuffer>)buf;
    buffer = nil;
  }
}
void Metal_CopyToDevice(MetalBufferRef buf, int offset, const void *data,
                        int size) {
  id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)buf;
  memcpy([buffer contents] + offset, data, size);
}
void Metal_CopyToHost(MetalBufferRef buf, int offset, void *data, int size) {
  id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)buf;
  memcpy(data, [buffer contents] + offset, size);
}
void Metal_Memset(MetalBufferRef buf, int offset, int value, int size) {
  id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)buf;
  memset([buffer contents] + offset, value, size);
}
void *Metal_GetBufferContents(MetalBufferRef buf) {
  id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)buf;
  return [buffer contents];
}
void Metal_SetAt(MetalBufferRef buf, int offset, float val) {
  id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)buf;
  float *ptr = (float *)((char *)[buffer contents] + offset);
  *ptr = val;
}

// Compute Kernels
#define ENCODE(wrapper, pipeline)                                              \
  [wrapper ensureEncoder];                                                     \
  [wrapper.currentEncoder setComputePipelineState:wrapper.pipeline];

void Metal_Add(MetalContextRef ctx, MetalBufferRef a, int offA,
               MetalBufferRef b, int offB, MetalBufferRef result, int offRes,
               int count) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineAdd);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)a offset:offA atIndex:0];
  if (b)
    [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)b
                         offset:offB
                        atIndex:1];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:2];
  [c.currentEncoder
            dispatchThreads:MTLSizeMake(count, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(
                                MIN(512, c.pipelineAdd
                                             .maxTotalThreadsPerThreadgroup),
                                1, 1)];
}
// ... AddScalar, Scale, Tanh, Gelu, etc implementing similar pattern.
// For brevity, I will only implement what's strictly needed for the Plan B
// execution, OR I have to assume the caller expects ALL of them. The caller IS
// main.go / metal_darwin.go. I must implement ALL functions declared in
// metal_bridge.h. To save space, I will implement them concisely.

void Metal_AddScalar(MetalContextRef ctx, MetalBufferRef a, int offA, float val,
                     MetalBufferRef result, int offRes, int count) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineAddScalar);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)a offset:offA atIndex:0];
  [c.currentEncoder setBytes:&val length:sizeof(float) atIndex:1];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:2];
  [c.currentEncoder
            dispatchThreads:MTLSizeMake(count, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(
                                MIN(512, c.pipelineAddScalar
                                             .maxTotalThreadsPerThreadgroup),
                                1, 1)];
}

void Metal_Scale(MetalContextRef ctx, MetalBufferRef a, int offA, float val,
                 MetalBufferRef result, int offRes, int count) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineScale);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)a offset:offA atIndex:0];
  [c.currentEncoder setBytes:&val length:sizeof(float) atIndex:1];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:2];
  [c.currentEncoder
            dispatchThreads:MTLSizeMake(count, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(
                                MIN(512, c.pipelineScale
                                             .maxTotalThreadsPerThreadgroup),
                                1, 1)];
}

void Metal_Scale_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                     uint16_t val, MetalBufferRef result, int offRes,
                     int count) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineScale_F16);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)a offset:offA atIndex:0];
  [c.currentEncoder setBytes:&val length:sizeof(uint16_t) atIndex:1];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:2];
  [c.currentEncoder
            dispatchThreads:MTLSizeMake(count, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(
                                MIN(512, c.pipelineScale_F16
                                             .maxTotalThreadsPerThreadgroup),
                                1, 1)];
}

void Metal_Tanh(MetalContextRef ctx, MetalBufferRef a, int offA,
                MetalBufferRef result, int offRes, int count) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineTanh);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)a offset:offA atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:1];
  [c.currentEncoder
            dispatchThreads:MTLSizeMake(count, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(
                                MIN(512, c.pipelineTanh
                                             .maxTotalThreadsPerThreadgroup),
                                1, 1)];
}
void Metal_Gelu(MetalContextRef ctx, MetalBufferRef a, int offA,
                MetalBufferRef result, int offRes, int count) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineGelu);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)a offset:offA atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:1];
  [c.currentEncoder
            dispatchThreads:MTLSizeMake(count, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(
                                MIN(512, c.pipelineGelu
                                             .maxTotalThreadsPerThreadgroup),
                                1, 1)];
}
void Metal_Tanh_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                    MetalBufferRef result, int offRes, int count) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineTanh_F16);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)a offset:offA atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:1];
  [c.currentEncoder
            dispatchThreads:MTLSizeMake(count, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(
                                MIN(512, c.pipelineTanh_F16
                                             .maxTotalThreadsPerThreadgroup),
                                1, 1)];
}
void Metal_Gelu_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                    MetalBufferRef result, int offRes, int count) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineGelu_F16);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)a offset:offA atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:1];
  [c.currentEncoder
            dispatchThreads:MTLSizeMake(count, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(
                                MIN(512, c.pipelineGelu_F16
                                             .maxTotalThreadsPerThreadgroup),
                                1, 1)];
}

void Metal_LayerNorm(MetalContextRef ctx, MetalBufferRef input, int offIn,
                     MetalBufferRef gamma, int offGamma, MetalBufferRef beta,
                     int offBeta, MetalBufferRef result, int offRes, int rows,
                     int cols, float eps) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineLayerNorm);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)input
                       offset:offIn
                      atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)gamma
                       offset:offGamma
                      atIndex:1];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)beta
                       offset:offBeta
                      atIndex:2];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:3];
  [c.currentEncoder setBytes:&cols length:sizeof(int) atIndex:4];
  [c.currentEncoder setBytes:&eps length:sizeof(float) atIndex:5];
  [c.currentEncoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}
void Metal_LayerNorm_F16(MetalContextRef ctx, MetalBufferRef input, int offIn,
                         MetalBufferRef gamma, int offGamma,
                         MetalBufferRef beta, int offBeta,
                         MetalBufferRef result, int offRes, int rows, int cols,
                         float eps) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineLayerNorm_F16);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)input
                       offset:offIn
                      atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)gamma
                       offset:offGamma
                      atIndex:1];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)beta
                       offset:offBeta
                      atIndex:2];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:3];
  [c.currentEncoder setBytes:&cols length:sizeof(int) atIndex:4];
  [c.currentEncoder setBytes:&eps length:sizeof(float) atIndex:5];
  [c.currentEncoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}
void Metal_Softmax(MetalContextRef ctx, MetalBufferRef input, int offIn,
                   MetalBufferRef result, int offRes, int rows, int cols) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineSoftmax);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)input
                       offset:offIn
                      atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:1];
  [c.currentEncoder setBytes:&cols length:sizeof(int) atIndex:2];
  [c.currentEncoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}
void Metal_Softmax_F16(MetalContextRef ctx, MetalBufferRef input, int offIn,
                       MetalBufferRef result, int offRes, int rows, int cols) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineSoftmax_F16);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)input
                       offset:offIn
                      atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:1];
  [c.currentEncoder setBytes:&cols length:sizeof(int) atIndex:2];
  [c.currentEncoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}
void Metal_AddBias(MetalContextRef ctx, MetalBufferRef component, int offComp,
                   MetalBufferRef bias, int offBias, int rows, int cols) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineAddBias);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)component
                       offset:offComp
                      atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)bias
                       offset:offBias
                      atIndex:1];
  [c.currentEncoder setBytes:&cols length:sizeof(int) atIndex:2];
  [c.currentEncoder
            dispatchThreads:MTLSizeMake(rows, cols, 1)
      threadsPerThreadgroup:MTLSizeMake(1,
                                        MIN((NSUInteger)cols,
                                            c.pipelineAddBias
                                                .maxTotalThreadsPerThreadgroup),
                                        1)];
}
void Metal_AddBias_F16(MetalContextRef ctx, MetalBufferRef matrix, int offMat,
                       MetalBufferRef bias, int offBias, MetalBufferRef result,
                       int offRes, int rows, int cols) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineAddBias_F16);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)matrix
                       offset:offMat
                      atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)bias
                       offset:offBias
                      atIndex:1];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:2];
  [c.currentEncoder setBytes:&cols length:sizeof(int) atIndex:3];
  MTLSize gridSize = MTLSizeMake(cols, rows, 1);
  NSUInteger w = c.pipelineAddBias_F16.threadExecutionWidth;
  NSUInteger h = c.pipelineAddBias_F16.maxTotalThreadsPerThreadgroup / w;
  [c.currentEncoder dispatchThreads:gridSize
              threadsPerThreadgroup:MTLSizeMake(w, h, 1)];
}
void Metal_Gather(MetalContextRef ctx, MetalBufferRef table, int offTable,
                  MetalBufferRef indices, int offIndices, MetalBufferRef output,
                  int offOut, int indicesCount, int cols) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineGather);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)table
                       offset:offTable
                      atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)output
                       offset:offOut
                      atIndex:1];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)indices
                       offset:offIndices
                      atIndex:2];
  [c.currentEncoder setBytes:&cols length:sizeof(int) atIndex:3];
  MTLSize gridSize = MTLSizeMake(indicesCount, cols, 1);
  MTLSize threadgroupSize = MTLSizeMake(
      1, MIN(c.pipelineGather.maxTotalThreadsPerThreadgroup, (NSUInteger)cols),
      1);
  [c.currentEncoder dispatchThreads:gridSize
              threadsPerThreadgroup:threadgroupSize];
}

void Metal_ApplyRoPE_F16(MetalContextRef ctx, MetalBufferRef data, int offData,
                         int batchSize, int seqLen, int numHeads, int headDim) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  ENCODE(mc, pipelineRope_F16);
  [mc.currentEncoder setBuffer:(__bridge id<MTLBuffer>)data
                        offset:offData
                       atIndex:0];
  [mc.currentEncoder setBytes:&headDim length:sizeof(int) atIndex:1];
  [mc.currentEncoder setBytes:&numHeads length:sizeof(int) atIndex:2];
  [mc.currentEncoder setBytes:&seqLen length:sizeof(int) atIndex:3];
  MTLSize gridSize = MTLSizeMake(headDim / 2, numHeads, batchSize * seqLen);
  NSUInteger maxThreads = mc.pipelineRope_F16.maxTotalThreadsPerThreadgroup;
  MTLSize threadgroupSize =
      MTLSizeMake(MIN((NSUInteger)(headDim / 2), maxThreads), 1, 1);
  [mc.currentEncoder dispatchThreads:gridSize
               threadsPerThreadgroup:threadgroupSize];
}

void Metal_SwiGLU_F16(MetalContextRef ctx, MetalBufferRef input, int offIn,
                      MetalBufferRef output, int offOut, int n, int interSize) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  ENCODE(mc, pipelineSwiglu_F16);
  [mc.currentEncoder setBuffer:(__bridge id<MTLBuffer>)input
                        offset:offIn
                       atIndex:0];
  [mc.currentEncoder setBuffer:(__bridge id<MTLBuffer>)output
                        offset:offOut
                       atIndex:1];
  [mc.currentEncoder setBytes:&interSize length:sizeof(int) atIndex:2];
  MTLSize gridSize = MTLSizeMake(interSize, n, 1);
  NSUInteger maxThreads = mc.pipelineSwiglu_F16.maxTotalThreadsPerThreadgroup;
  MTLSize threadgroupSize =
      MTLSizeMake(MIN((NSUInteger)interSize, maxThreads), 1, 1);
  [mc.currentEncoder dispatchThreads:gridSize
               threadsPerThreadgroup:threadgroupSize];
}

// Matrix Multiplication
// Matrix Multiplication
void Metal_MatMul(MetalContextRef ctx, MetalBufferRef a, int offA, bool transA,
                  MetalBufferRef b, int offB, bool transB, MetalBufferRef c,
                  int offC, int M, int N, int K) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;

  [mc flush];

  @synchronized(mc) {
    if (!mc.currentCommandBuffer) {
      mc.currentCommandBuffer = [mc.commandQueue commandBuffer];
    }
  }

  MPSMatrixDescriptor *descA = [MPSMatrixDescriptor
      matrixDescriptorWithRows:(transA ? K : M)
                       columns:(transA ? M : K)rowBytes:(transA ? M : K) *
                               sizeof(float)
                      dataType:MPSDataTypeFloat32];
  MPSMatrixDescriptor *descB = [MPSMatrixDescriptor
      matrixDescriptorWithRows:(transB ? N : K)
                       columns:(transB ? K : N)rowBytes:(transB ? K : N) *
                               sizeof(float)
                      dataType:MPSDataTypeFloat32];
  MPSMatrixDescriptor *descC =
      [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                            columns:N
                                           rowBytes:N * sizeof(float)
                                           dataType:MPSDataTypeFloat32];
  MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)a
                                               offset:offA
                                           descriptor:descA];
  MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)b
                                               offset:offB
                                           descriptor:descB];
  MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)c
                                               offset:offC
                                           descriptor:descC];
  MPSMatrixMultiplication *mul =
      [[MPSMatrixMultiplication alloc] initWithDevice:mc.device
                                        transposeLeft:transA
                                       transposeRight:transB
                                           resultRows:M
                                        resultColumns:N
                                      interiorColumns:K
                                                alpha:1.0
                                                 beta:0.0];
  [mul encodeToCommandBuffer:mc.currentCommandBuffer
                  leftMatrix:matA
                 rightMatrix:matB
                resultMatrix:matC];
  [mc flush];
}

void Metal_MatMul_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                      bool transA, MetalBufferRef b, int offB, bool transB,
                      MetalBufferRef c, int offC, int M, int N, int K) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;

  [mc flush];

  @synchronized(mc) {
    if (!mc.currentCommandBuffer) {
      mc.currentCommandBuffer = [mc.commandQueue commandBuffer];
    }
  }

  MPSMatrixDescriptor *descA = [MPSMatrixDescriptor
      matrixDescriptorWithRows:(transA ? K : M)
                       columns:(transA ? M : K)rowBytes:(transA ? M : K) *
                               sizeof(uint16_t)
                      dataType:MPSDataTypeFloat16];
  MPSMatrixDescriptor *descB = [MPSMatrixDescriptor
      matrixDescriptorWithRows:(transB ? N : K)
                       columns:(transB ? K : N)rowBytes:(transB ? K : N) *
                               sizeof(uint16_t)
                      dataType:MPSDataTypeFloat16];
  MPSMatrixDescriptor *descC =
      [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                            columns:N
                                           rowBytes:N * sizeof(uint16_t)
                                           dataType:MPSDataTypeFloat16];
  MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)a
                                               offset:offA
                                           descriptor:descA];
  MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)b
                                               offset:offB
                                           descriptor:descB];
  MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)c
                                               offset:offC
                                           descriptor:descC];
  MPSMatrixMultiplication *mul =
      [[MPSMatrixMultiplication alloc] initWithDevice:mc.device
                                        transposeLeft:transA
                                       transposeRight:transB
                                           resultRows:M
                                        resultColumns:N
                                      interiorColumns:K
                                                alpha:1.0
                                                 beta:0.0];
  [mul encodeToCommandBuffer:mc.currentCommandBuffer
                  leftMatrix:matA
                 rightMatrix:matB
                resultMatrix:matC];
  [mc flush];
}

void Metal_BatchedMatMul(MetalContextRef ctx, MetalBufferRef a, int offA,
                         int strideA, bool transA, MetalBufferRef b, int offB,
                         int strideB, bool transB, MetalBufferRef c, int offC,
                         int strideC, int M, int N, int K, int batchCount) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;

  [mc flush];

  @synchronized(mc) {
    if (!mc.currentCommandBuffer) {
      mc.currentCommandBuffer = [mc.commandQueue commandBuffer];
    }
  }

  MPSMatrixDescriptor *descA = [MPSMatrixDescriptor
      matrixDescriptorWithRows:(transA ? K : M)
                       columns:(transA ? M : K)rowBytes:(transA ? M : K) *
                               sizeof(float)
                      dataType:MPSDataTypeFloat32];
  MPSMatrixDescriptor *descB = [MPSMatrixDescriptor
      matrixDescriptorWithRows:(transB ? N : K)
                       columns:(transB ? K : N)rowBytes:(transB ? K : N) *
                               sizeof(float)
                      dataType:MPSDataTypeFloat32];
  MPSMatrixDescriptor *descC =
      [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                            columns:N
                                           rowBytes:N * sizeof(float)
                                           dataType:MPSDataTypeFloat32];
  MPSMatrixMultiplication *mul =
      [[MPSMatrixMultiplication alloc] initWithDevice:mc.device
                                        transposeLeft:transA
                                       transposeRight:transB
                                           resultRows:M
                                        resultColumns:N
                                      interiorColumns:K
                                                alpha:1.0
                                                 beta:0.0];
  for (int i = 0; i < batchCount; i++) {
    int batchOffA = offA + i * strideA;
    int batchOffB = offB + i * strideB;
    int batchOffC = offC + i * strideC;
    MPSMatrix *matA =
        [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)a
                                   offset:batchOffA
                               descriptor:descA];
    MPSMatrix *matB =
        [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)b
                                   offset:batchOffB
                               descriptor:descB];
    MPSMatrix *matC =
        [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)c
                                   offset:batchOffC
                               descriptor:descC];
    [mul encodeToCommandBuffer:mc.currentCommandBuffer
                    leftMatrix:matA
                   rightMatrix:matB
                  resultMatrix:matC];
  }
  [mc flush];
}

void Metal_BatchedMatMul_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                             int strideA, bool transA, MetalBufferRef b,
                             int offB, int strideB, bool transB,
                             MetalBufferRef c, int offC, int strideC, int M,
                             int N, int K, int batchCount) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;

  [mc flush];

  @synchronized(mc) {
    if (!mc.currentCommandBuffer) {
      mc.currentCommandBuffer = [mc.commandQueue commandBuffer];
    }
  }

  MPSMatrixDescriptor *descA = [MPSMatrixDescriptor
      matrixDescriptorWithRows:(transA ? K : M)
                       columns:(transA ? M : K)rowBytes:(transA ? M : K) *
                               sizeof(uint16_t)
                      dataType:MPSDataTypeFloat16];
  MPSMatrixDescriptor *descB = [MPSMatrixDescriptor
      matrixDescriptorWithRows:(transB ? N : K)
                       columns:(transB ? K : N)rowBytes:(transB ? K : N) *
                               sizeof(uint16_t)
                      dataType:MPSDataTypeFloat16];
  MPSMatrixDescriptor *descC =
      [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                            columns:N
                                           rowBytes:N * sizeof(uint16_t)
                                           dataType:MPSDataTypeFloat16];
  MPSMatrixMultiplication *mul =
      [[MPSMatrixMultiplication alloc] initWithDevice:mc.device
                                        transposeLeft:transA
                                       transposeRight:transB
                                           resultRows:M
                                        resultColumns:N
                                      interiorColumns:K
                                                alpha:1.0
                                                 beta:0.0];
  for (int i = 0; i < batchCount; i++) {
    int batchOffA = offA + i * strideA;
    int batchOffB = offB + i * strideB;
    int batchOffC = offC + i * strideC;
    MPSMatrix *matA =
        [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)a
                                   offset:batchOffA
                               descriptor:descA];
    MPSMatrix *matB =
        [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)b
                                   offset:batchOffB
                               descriptor:descB];
    MPSMatrix *matC =
        [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)c
                                   offset:batchOffC
                               descriptor:descC];
  }
  [mc flush];
}

unsigned long long Metal_GetAllocatedSize(void *ctx) {
  if (!ctx)
    return 0;
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  if (@available(macOS 10.13, *)) {
    return [mc.device currentAllocatedSize];
  }
  return 0;
}

unsigned long long Metal_GetRecommendMaxWorkingSetSize(void *ctx) {
  if (!ctx)
    return 0;
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  if (@available(macOS 10.13, *)) {
    return [mc.device recommendedMaxWorkingSetSize];
  }
  return 0;
}

// Plan B Implementations
void Metal_Linear_Graph(MetalContextRef ctx, MetalBufferRef input, int offIn,
                        int rows, int inCols, MetalBufferRef weight,
                        int offWeight, int outCols, MetalBufferRef bias,
                        int offBias, MetalBufferRef result, int offRes) {
  Metal_MatMul_F16(ctx, input, offIn, false, weight, offWeight, false, result,
                   offRes, rows, outCols, inCols);
  Metal_AddBias_F16(ctx, result, offRes, bias, offBias, result, offRes, rows,
                    outCols);
}

void Metal_LinearActivation_Graph(MetalContextRef ctx, MetalBufferRef input,
                                  int offIn, int rows, int inCols,
                                  MetalBufferRef weight, int offWeight,
                                  int outCols, MetalBufferRef bias, int offBias,
                                  MetalBufferRef result, int offRes,
                                  int activationType) {
  Metal_MatMul_F16(ctx, input, offIn, false, weight, offWeight, false, result,
                   offRes, rows, outCols, inCols);
  Metal_AddBias_F16(ctx, result, offRes, bias, offBias, result, offRes, rows,
                    outCols);
  int count = rows * outCols;
  if (activationType == 1)
    Metal_Gelu_F16(ctx, result, offRes, result, offRes, count);
  else if (activationType == 2)
    Metal_Tanh_F16(ctx, result, offRes, result, offRes, count);
  else if (activationType == 3)
    Metal_Softmax_F16(ctx, result, offRes, result, offRes, rows, outCols);
}

void Metal_Attention_Graph(MetalContextRef ctx, MetalBufferRef q, int offQ,
                           MetalBufferRef k, int offK, MetalBufferRef v,
                           int offV, MetalBufferRef result, int offRes,
                           int batchSize, int seqLen, int hiddenSize,
                           float scale) {
  int scoresSize = batchSize * seqLen * seqLen * sizeof(uint16_t);
  MetalBufferRef scoresBuf = Metal_Alloc(ctx, scoresSize);

  int strideQ = seqLen * hiddenSize * sizeof(uint16_t);
  int strideK = seqLen * hiddenSize * sizeof(uint16_t);
  int strideScores = seqLen * seqLen * sizeof(uint16_t);
  int strideV = seqLen * hiddenSize * sizeof(uint16_t);
  int strideOut = seqLen * hiddenSize * sizeof(uint16_t);

  // Q * K^T -> Scores
  Metal_BatchedMatMul_F16(ctx, q, offQ, strideQ, false, k, offK, strideK, true,
                          scoresBuf, 0, strideScores, seqLen, seqLen,
                          hiddenSize, batchSize);

  // Scale
  __fp16 scaleF16 = (__fp16)scale;
  uint16_t scaleBits = *(uint16_t *)&scaleF16;
  Metal_Scale_F16(ctx, scoresBuf, 0, scaleBits, scoresBuf, 0,
                  batchSize * seqLen * seqLen);

  // Softmax
  Metal_Softmax_F16(ctx, scoresBuf, 0, scoresBuf, 0, batchSize * seqLen,
                    seqLen);

  // Scores * V -> Result
  Metal_BatchedMatMul_F16(ctx, scoresBuf, 0, strideScores, false, v, offV,
                          strideV, false, result, offRes, strideOut, seqLen,
                          hiddenSize, seqLen, batchSize);

  Metal_FreeBuffer(ctx, scoresBuf);
}

void Metal_Synchronize(MetalContextRef ctx) {
  MetalWrapper *wrapper = (__bridge MetalWrapper *)ctx;
  [wrapper fullSync];
}
