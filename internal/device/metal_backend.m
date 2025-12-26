#import "metal_bridge.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

@interface GraphContext : NSObject
@property(strong) MPSGraph *graph;
@property(strong) MPSGraphTensor *input;
@property(strong) MPSGraphTensor *weight;
@property(strong) MPSGraphTensor *bias;
@property(strong) MPSGraphTensor *output;
@end

@implementation GraphContext
@end

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
// FP16 Pipelines (for 2x GPU performance)
@property(strong) id<MTLComputePipelineState> pipelineAdd_F16;
@property(strong) id<MTLComputePipelineState> pipelineAddScalar_F16;
@property(strong) id<MTLComputePipelineState> pipelineScale_F16;
@property(strong) id<MTLComputePipelineState> pipelineTanh_F16;
@property(strong) id<MTLComputePipelineState> pipelineGelu_F16;
@property(strong) id<MTLComputePipelineState> pipelineSoftmax_F16;
@property(strong) id<MTLComputePipelineState> pipelineLayerNorm_F16;
@property(strong) id<MTLComputePipelineState> pipelineAddBias_F16;

// Async command batching
@property(strong) id<MTLCommandBuffer> currentCommandBuffer;
@property(strong) id<MTLComputeCommandEncoder> currentEncoder;
// Track last MPS buffer for synchronization
@property(strong) id<MTLCommandBuffer> lastMPSBuffer;
// Cache for compiled MPSGraph executables
// Cache for compiled MPSGraph executables (stores GraphContext)
@property(strong) NSMutableDictionary<NSString *, GraphContext *> *graphCache;
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

- (void)flush {
  @synchronized(self) {
    if (self.currentEncoder) {
      [self.currentEncoder endEncoding];
      self.currentEncoder = nil;
    }
    if (self.currentCommandBuffer) {
      [self.currentCommandBuffer commit];
      [self.currentCommandBuffer waitUntilCompleted];
      self.currentCommandBuffer = nil;
    }
  }
}

- (void)fullSync {
  @synchronized(self) {
    // First flush compute encoder
    [self flush];
    // Then wait for any pending MPS operations
    if (self.lastMPSBuffer) {
      [self.lastMPSBuffer waitUntilCompleted];
      self.lastMPSBuffer = nil;
    }
  }
}

@end

MetalContextRef Metal_Init(const char *libSource) {
  MetalWrapper *ctx = [[MetalWrapper alloc] init];
  ctx.graphCache = [NSMutableDictionary dictionary];
  ctx.device = MTLCreateSystemDefaultDevice();
  if (!ctx.device) {
    printf("Error: Metal device not found\n");
    return NULL;
  }

  ctx.commandQueue = [ctx.device newCommandQueue];

  NSError *error = nil;
  NSString *src = [NSString stringWithUTF8String:libSource];

  ctx.library = [ctx.device newLibraryWithSource:src options:nil error:&error];
  if (error) {
    printf("Error compiling Metal headers: %s\n",
           [[error localizedDescription] UTF8String]);
    return NULL;
  }

  ctx.pipelineAdd = [ctx.device
      newComputePipelineStateWithFunction:[ctx.library
                                              newFunctionWithName:@"add_kernel"]
                                    error:&error];
  if (error)
    printf("Failed to load add_kernel: %s\n",
           [[error localizedDescription] UTF8String]);

  ctx.pipelineAddScalar =
      [ctx.device newComputePipelineStateWithFunction:
                      [ctx.library newFunctionWithName:@"add_scalar_kernel"]
                                                error:&error];
  ctx.pipelineScale =
      [ctx.device newComputePipelineStateWithFunction:
                      [ctx.library newFunctionWithName:@"scale_kernel"]
                                                error:&error];
  ctx.pipelineTanh =
      [ctx.device newComputePipelineStateWithFunction:
                      [ctx.library newFunctionWithName:@"tanh_kernel"]
                                                error:&error];
  ctx.pipelineGelu =
      [ctx.device newComputePipelineStateWithFunction:
                      [ctx.library newFunctionWithName:@"gelu_kernel"]
                                                error:&error];
  ctx.pipelineLayerNorm =
      [ctx.device newComputePipelineStateWithFunction:
                      [ctx.library newFunctionWithName:@"layernorm_kernel"]
                                                error:&error];
  ctx.pipelineSoftmax =
      [ctx.device newComputePipelineStateWithFunction:
                      [ctx.library newFunctionWithName:@"softmax_kernel"]
                                                error:&error];
  ctx.pipelineGather =
      [ctx.device newComputePipelineStateWithFunction:
                      [ctx.library newFunctionWithName:@"gather_kernel"]
                                                error:&error];
  ctx.pipelineAddBias =
      [ctx.device newComputePipelineStateWithFunction:
                      [ctx.library newFunctionWithName:@"add_bias_kernel"]
                                                error:&error];

  // Initialize FP16 pipelines
  ctx.pipelineAdd_F16 =
      [ctx.device newComputePipelineStateWithFunction:
                      [ctx.library newFunctionWithName:@"add_kernel_f16"]
                                                error:&error];
  ctx.pipelineAddScalar_F16 =
      [ctx.device newComputePipelineStateWithFunction:
                      [ctx.library newFunctionWithName:@"add_scalar_kernel_f16"]
                                                error:&error];
  ctx.pipelineScale_F16 =
      [ctx.device newComputePipelineStateWithFunction:
                      [ctx.library newFunctionWithName:@"scale_kernel_f16"]
                                                error:&error];
  ctx.pipelineTanh_F16 =
      [ctx.device newComputePipelineStateWithFunction:
                      [ctx.library newFunctionWithName:@"tanh_kernel_f16"]
                                                error:&error];
  ctx.pipelineGelu_F16 =
      [ctx.device newComputePipelineStateWithFunction:
                      [ctx.library newFunctionWithName:@"gelu_kernel_f16"]
                                                error:&error];
  ctx.pipelineSoftmax_F16 =
      [ctx.device newComputePipelineStateWithFunction:
                      [ctx.library newFunctionWithName:@"softmax_kernel_f16"]
                                                error:&error];
  ctx.pipelineLayerNorm_F16 =
      [ctx.device newComputePipelineStateWithFunction:
                      [ctx.library newFunctionWithName:@"layernorm_kernel_f16"]
                                                error:&error];
  ctx.pipelineAddBias_F16 =
      [ctx.device newComputePipelineStateWithFunction:
                      [ctx.library newFunctionWithName:@"add_bias_kernel_f16"]
                                                error:&error];

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

void Metal_SetAt(MetalBufferRef buf, int offset, float val) {
  id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)buf;
  float *ptr = (float *)((char *)[buffer contents] + offset);
  *ptr = val;
}

// ============== ASYNC COMPUTE OPERATIONS ==============

void Metal_Add(MetalContextRef ctx, MetalBufferRef a, int offA,
               MetalBufferRef b, int offB, MetalBufferRef result, int offRes,
               int count) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  [c ensureEncoder];

  [c.currentEncoder setComputePipelineState:c.pipelineAdd];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)a offset:offA atIndex:0];
  if (b)
    [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)b
                         offset:offB
                        atIndex:1];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:2];

  MTLSize gridSize = MTLSizeMake(count, 1, 1);
  NSUInteger threadGroupSize =
      MIN(512, c.pipelineAdd.maxTotalThreadsPerThreadgroup);
  MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

  [c.currentEncoder dispatchThreads:gridSize
              threadsPerThreadgroup:threadgroupSize];
}

void Metal_AddScalar(MetalContextRef ctx, MetalBufferRef a, int offA, float val,
                     MetalBufferRef result, int offRes, int count) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  [c ensureEncoder];

  [c.currentEncoder setComputePipelineState:c.pipelineAddScalar];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)a offset:offA atIndex:0];
  [c.currentEncoder setBytes:&val length:sizeof(float) atIndex:1];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:2];

  MTLSize gridSize = MTLSizeMake(count, 1, 1);
  NSUInteger threadGroupSize =
      MIN(512, c.pipelineAddScalar.maxTotalThreadsPerThreadgroup);
  MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

  [c.currentEncoder dispatchThreads:gridSize
              threadsPerThreadgroup:threadgroupSize];
}

void Metal_Scale(MetalContextRef ctx, MetalBufferRef a, int offA, float val,
                 MetalBufferRef result, int offRes, int count) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  [c ensureEncoder];

  [c.currentEncoder setComputePipelineState:c.pipelineScale];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)a offset:offA atIndex:0];
  [c.currentEncoder setBytes:&val length:sizeof(float) atIndex:1];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:2];

  MTLSize gridSize = MTLSizeMake(count, 1, 1);
  NSUInteger threadGroupSize =
      MIN(512, c.pipelineScale.maxTotalThreadsPerThreadgroup);
  MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

  [c.currentEncoder dispatchThreads:gridSize
              threadsPerThreadgroup:threadgroupSize];
}

// FP16 Scale for 2x performance
void Metal_Scale_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                     uint16_t val, MetalBufferRef result, int offRes,
                     int count) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  [c ensureEncoder];

  [c.currentEncoder setComputePipelineState:c.pipelineScale_F16];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)a offset:offA atIndex:0];
  [c.currentEncoder setBytes:&val length:sizeof(uint16_t) atIndex:1];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:2];

  MTLSize gridSize = MTLSizeMake(count, 1, 1);
  NSUInteger threadGroupSize =
      MIN(512, c.pipelineScale_F16.maxTotalThreadsPerThreadgroup);
  MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

  [c.currentEncoder dispatchThreads:gridSize
              threadsPerThreadgroup:threadgroupSize];
}

void Metal_Tanh(MetalContextRef ctx, MetalBufferRef a, int offA,
                MetalBufferRef result, int offRes, int count) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  [c ensureEncoder];

  [c.currentEncoder setComputePipelineState:c.pipelineTanh];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)a offset:offA atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:1];

  MTLSize gridSize = MTLSizeMake(count, 1, 1);
  NSUInteger threadGroupSize =
      MIN(512, c.pipelineTanh.maxTotalThreadsPerThreadgroup);
  MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

  [c.currentEncoder dispatchThreads:gridSize
              threadsPerThreadgroup:threadgroupSize];
}

void Metal_Gelu(MetalContextRef ctx, MetalBufferRef a, int offA,
                MetalBufferRef result, int offRes, int count) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  [c ensureEncoder];

  [c.currentEncoder setComputePipelineState:c.pipelineGelu];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)a offset:offA atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:1];

  MTLSize gridSize = MTLSizeMake(count, 1, 1);
  NSUInteger threadGroupSize =
      MIN(512, c.pipelineGelu.maxTotalThreadsPerThreadgroup);
  MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

  [c.currentEncoder dispatchThreads:gridSize
              threadsPerThreadgroup:threadgroupSize];
}

// FP16 Tanh for 2x performance
void Metal_Tanh_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                    MetalBufferRef result, int offRes, int count) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  [c ensureEncoder];

  [c.currentEncoder setComputePipelineState:c.pipelineTanh_F16];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)a offset:offA atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:1];

  MTLSize gridSize = MTLSizeMake(count, 1, 1);
  NSUInteger threadGroupSize =
      MIN(512, c.pipelineTanh_F16.maxTotalThreadsPerThreadgroup);
  MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

  [c.currentEncoder dispatchThreads:gridSize
              threadsPerThreadgroup:threadgroupSize];
}

// FP16 Gelu for 2x performance
void Metal_Gelu_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                    MetalBufferRef result, int offRes, int count) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  [c ensureEncoder];

  [c.currentEncoder setComputePipelineState:c.pipelineGelu_F16];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)a offset:offA atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:1];

  MTLSize gridSize = MTLSizeMake(count, 1, 1);
  NSUInteger threadGroupSize =
      MIN(512, c.pipelineGelu_F16.maxTotalThreadsPerThreadgroup);
  MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

  [c.currentEncoder dispatchThreads:gridSize
              threadsPerThreadgroup:threadgroupSize];
}

void Metal_LayerNorm(MetalContextRef ctx, MetalBufferRef input, int offIn,
                     MetalBufferRef gamma, int offGamma, MetalBufferRef beta,
                     int offBeta, MetalBufferRef result, int offRes, int rows,
                     int cols, float eps) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  [c ensureEncoder];

  [c.currentEncoder setComputePipelineState:c.pipelineLayerNorm];
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

  // Dispatch one threadgroup per row, with 256 threads per group for parallel
  // reduction
  MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
  MTLSize threadgroupsPerGrid = MTLSizeMake(rows, 1, 1);

  [c.currentEncoder dispatchThreadgroups:threadgroupsPerGrid
                   threadsPerThreadgroup:threadgroupSize];
}

void Metal_Softmax(MetalContextRef ctx, MetalBufferRef input, int offIn,
                   MetalBufferRef result, int offRes, int rows, int cols) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  [c ensureEncoder];

  [c.currentEncoder setComputePipelineState:c.pipelineSoftmax];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)input
                       offset:offIn
                      atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:1];
  [c.currentEncoder setBytes:&cols length:sizeof(int) atIndex:2];

  // Dispatch one threadgroup per row, with 256 threads per group for parallel
  // reduction
  MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
  MTLSize threadgroupsPerGrid = MTLSizeMake(rows, 1, 1);

  [c.currentEncoder dispatchThreadgroups:threadgroupsPerGrid
                   threadsPerThreadgroup:threadgroupSize];
}

void Metal_LayerNorm_F16(MetalContextRef ctx, MetalBufferRef input, int offIn,
                         MetalBufferRef gamma, int offGamma,
                         MetalBufferRef beta, int offBeta,
                         MetalBufferRef result, int offRes, int rows, int cols,
                         float eps) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  [c ensureEncoder];

  [c.currentEncoder setComputePipelineState:c.pipelineLayerNorm_F16];
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

  MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
  MTLSize threadgroupsPerGrid = MTLSizeMake(rows, 1, 1);

  [c.currentEncoder dispatchThreadgroups:threadgroupsPerGrid
                   threadsPerThreadgroup:threadgroupSize];
}

void Metal_AddBias_F16(MetalContextRef ctx, MetalBufferRef matrix, int offMat,
                       MetalBufferRef bias, int offBias, MetalBufferRef result,
                       int offRes, int rows, int cols) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  [c ensureEncoder];

  [c.currentEncoder setComputePipelineState:c.pipelineAddBias_F16];
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

  // 2D dispatch: (cols, rows)
  MTLSize gridSize = MTLSizeMake(cols, rows, 1);
  NSUInteger w = c.pipelineAddBias_F16.threadExecutionWidth;
  NSUInteger h = c.pipelineAddBias_F16.maxTotalThreadsPerThreadgroup / w;
  MTLSize threadgroupSize = MTLSizeMake(w, h, 1);

  [c.currentEncoder dispatchThreads:gridSize
              threadsPerThreadgroup:threadgroupSize];
}

// FP16 Softmax for 2x performance
void Metal_Softmax_F16(MetalContextRef ctx, MetalBufferRef input, int offIn,
                       MetalBufferRef result, int offRes, int rows, int cols) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  [c ensureEncoder];

  [c.currentEncoder setComputePipelineState:c.pipelineSoftmax_F16];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)input
                       offset:offIn
                      atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:1];
  [c.currentEncoder setBytes:&cols length:sizeof(int) atIndex:2];

  MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
  MTLSize threadgroupsPerGrid = MTLSizeMake(rows, 1, 1);

  [c.currentEncoder dispatchThreadgroups:threadgroupsPerGrid
                   threadsPerThreadgroup:threadgroupSize];
}

void Metal_Gather(MetalContextRef ctx, MetalBufferRef table, int offTable,
                  MetalBufferRef indices, int offIndices, MetalBufferRef output,
                  int offOut, int indicesCount, int cols) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  [c ensureEncoder];

  [c.currentEncoder setComputePipelineState:c.pipelineGather];
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
  NSUInteger maxThreads = c.pipelineGather.maxTotalThreadsPerThreadgroup;
  NSUInteger w = 1;
  NSUInteger h = MIN(maxThreads, (NSUInteger)cols);

  MTLSize threadgroupSize = MTLSizeMake(w, h, 1);

  [c.currentEncoder dispatchThreads:gridSize
              threadsPerThreadgroup:threadgroupSize];
}

void Metal_AddBias(MetalContextRef ctx, MetalBufferRef component, int offComp,
                   MetalBufferRef bias, int offBias, int rows, int cols) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  [c ensureEncoder];

  [c.currentEncoder setComputePipelineState:c.pipelineAddBias];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)component
                       offset:offComp
                      atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)bias
                       offset:offBias
                      atIndex:1];
  [c.currentEncoder setBytes:&cols length:sizeof(int) atIndex:2];

  MTLSize gridSize = MTLSizeMake(rows, cols, 1);
  NSUInteger maxThreads = c.pipelineAddBias.maxTotalThreadsPerThreadgroup;
  MTLSize threadgroupSize = MTLSizeMake(1, MIN(cols, maxThreads), 1);

  [c.currentEncoder dispatchThreads:gridSize
              threadsPerThreadgroup:threadgroupSize];
}

void Metal_MatMul(MetalContextRef ctx, MetalBufferRef a, int offA, bool transA,
                  MetalBufferRef b, int offB, bool transB, MetalBufferRef c,
                  int offC, int M, int N, int K) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;

  // Flush current encoder before MPS operations
  [mc flush];

  id<MTLCommandBuffer> buffer = [mc.commandQueue commandBuffer];

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

  [mul encodeToCommandBuffer:buffer
                  leftMatrix:matA
                 rightMatrix:matB
                resultMatrix:matC];

  [buffer commit];
  // Store for sync in flush
  mc.lastMPSBuffer = buffer;
}

// FP16 MatMul for 2x GPU performance
void Metal_MatMul_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                      bool transA, MetalBufferRef b, int offB, bool transB,
                      MetalBufferRef c, int offC, int M, int N, int K) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;

  [mc flush];

  id<MTLCommandBuffer> buffer = [mc.commandQueue commandBuffer];

  // Use Float16 data type for 2x performance
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

  [mul encodeToCommandBuffer:buffer
                  leftMatrix:matA
                 rightMatrix:matB
                resultMatrix:matC];

  [buffer commit];
  mc.lastMPSBuffer = buffer;
}

void Metal_BatchedMatMul(MetalContextRef ctx, MetalBufferRef a, int offA,
                         int strideA, bool transA, MetalBufferRef b, int offB,
                         int strideB, bool transB, MetalBufferRef c, int offC,
                         int strideC, int M, int N, int K, int batchCount) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;

  // Flush before MPS
  [mc flush];

  id<MTLCommandBuffer> buffer = [mc.commandQueue commandBuffer];

  // Create descriptors with batch info
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

  // Create multiply operation
  MPSMatrixMultiplication *mul =
      [[MPSMatrixMultiplication alloc] initWithDevice:mc.device
                                        transposeLeft:transA
                                       transposeRight:transB
                                           resultRows:M
                                        resultColumns:N
                                      interiorColumns:K
                                                alpha:1.0
                                                 beta:0.0];

  // Process batches - MPS doesn't have native batch support,
  // but we can encode multiple operations to one command buffer
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

    [mul encodeToCommandBuffer:buffer
                    leftMatrix:matA
                   rightMatrix:matB
                  resultMatrix:matC];
  }

  [buffer commit];
  // Don't wait - let GPU pipeline run asynchronously
}

// Helper to run graph with inputs
void runGraph(MetalWrapper *mc, MPSGraph *graph,
              NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds,
              NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results) {
  [graph runWithMTLCommandQueue:mc.commandQueue
                          feeds:feeds
               targetOperations:nil
              resultsDictionary:results];
  // Async execution
}

void Metal_Linear_Graph(MetalContextRef ctx, MetalBufferRef input, int offIn,
                        int rows, int inCols, MetalBufferRef weight,
                        int offWeight, int outCols, MetalBufferRef bias,
                        int offBias, MetalBufferRef result, int offRes) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  [mc flush];

  NSString *key =
      [NSString stringWithFormat:@"L_%d_%d_%d", rows, inCols, outCols];
  GraphContext *gctx = mc.graphCache[key];

  if (!gctx) {
    gctx = [[GraphContext alloc] init];
    gctx.graph = [[MPSGraph alloc] init];

    gctx.input = [gctx.graph placeholderWithShape:@[ @(rows), @(inCols) ]
                                         dataType:MPSDataTypeFloat16
                                             name:nil];
    gctx.weight = [gctx.graph placeholderWithShape:@[ @(inCols), @(outCols) ]
                                          dataType:MPSDataTypeFloat16
                                              name:nil];
    gctx.bias = [gctx.graph placeholderWithShape:@[ @(outCols) ]
                                        dataType:MPSDataTypeFloat16
                                            name:nil];

    MPSGraphTensor *matmul =
        [gctx.graph matrixMultiplicationWithPrimaryTensor:gctx.input
                                          secondaryTensor:gctx.weight
                                                     name:nil];
    gctx.output = [gctx.graph additionWithPrimaryTensor:matmul
                                        secondaryTensor:gctx.bias
                                                   name:nil];

    mc.graphCache[key] = gctx;
  }

  MPSGraphTensorData *inputData = [[MPSGraphTensorData alloc]
      initWithMTLBuffer:(__bridge id<MTLBuffer>)input
                  shape:@[ @(rows), @(inCols) ]
               dataType:MPSDataTypeFloat16];
  MPSGraphTensorData *weightData = [[MPSGraphTensorData alloc]
      initWithMTLBuffer:(__bridge id<MTLBuffer>)weight
                  shape:@[ @(inCols), @(outCols) ]
               dataType:MPSDataTypeFloat16];
  MPSGraphTensorData *biasData =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:(__bridge id<MTLBuffer>)bias
                                              shape:@[ @(outCols) ]
                                           dataType:MPSDataTypeFloat16];
  MPSGraphTensorData *resultData = [[MPSGraphTensorData alloc]
      initWithMTLBuffer:(__bridge id<MTLBuffer>)result
                  shape:@[ @(rows), @(outCols) ]
               dataType:MPSDataTypeFloat16];

  runGraph(
      mc, gctx.graph,
      @{gctx.input : inputData, gctx.weight : weightData, gctx.bias : biasData},
      @{gctx.output : resultData});
}

void Metal_LinearActivation_Graph(MetalContextRef ctx, MetalBufferRef input,
                                  int offIn, int rows, int inCols,
                                  MetalBufferRef weight, int offWeight,
                                  int outCols, MetalBufferRef bias, int offBias,
                                  MetalBufferRef result, int offRes,
                                  int activationType) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  [mc flush];

  NSString *key = [NSString stringWithFormat:@"LA_%d_%d_%d_%d", rows, inCols,
                                             outCols, activationType];
  GraphContext *gctx = mc.graphCache[key];

  if (!gctx) {
    gctx = [[GraphContext alloc] init];
    gctx.graph = [[MPSGraph alloc] init];

    gctx.input = [gctx.graph placeholderWithShape:@[ @(rows), @(inCols) ]
                                         dataType:MPSDataTypeFloat16
                                             name:nil];
    gctx.weight = [gctx.graph placeholderWithShape:@[ @(inCols), @(outCols) ]
                                          dataType:MPSDataTypeFloat16
                                              name:nil];
    gctx.bias = [gctx.graph placeholderWithShape:@[ @(outCols) ]
                                        dataType:MPSDataTypeFloat16
                                            name:nil];

    MPSGraphTensor *curr =
        [gctx.graph matrixMultiplicationWithPrimaryTensor:gctx.input
                                          secondaryTensor:gctx.weight
                                                     name:nil];
    curr = [gctx.graph additionWithPrimaryTensor:curr
                                 secondaryTensor:gctx.bias
                                            name:nil];

    if (activationType == 1) { // GELU
      MPSGraphTensor *const0_5 =
          [gctx.graph constantWithScalar:0.5 dataType:MPSDataTypeFloat16];
      MPSGraphTensor *const0_707 =
          [gctx.graph constantWithScalar:0.70710678
                                dataType:MPSDataTypeFloat16];
      MPSGraphTensor *one = [gctx.graph constantWithScalar:1.0
                                                  dataType:MPSDataTypeFloat16];

      MPSGraphTensor *inner =
          [gctx.graph multiplicationWithPrimaryTensor:curr
                                      secondaryTensor:const0_707
                                                 name:nil];
      MPSGraphTensor *erfVal = [gctx.graph erfWithTensor:inner name:nil];
      MPSGraphTensor *add = [gctx.graph additionWithPrimaryTensor:one
                                                  secondaryTensor:erfVal
                                                             name:nil];
      MPSGraphTensor *scale =
          [gctx.graph multiplicationWithPrimaryTensor:curr
                                      secondaryTensor:const0_5
                                                 name:nil];
      curr = [gctx.graph multiplicationWithPrimaryTensor:scale
                                         secondaryTensor:add
                                                    name:nil];
    } else if (activationType == 2) { // Tanh
      curr = [gctx.graph tanhWithTensor:curr name:nil];
    } else if (activationType == 3) { // Softmax
      curr = [gctx.graph softMaxWithTensor:curr axis:-1 name:nil];
    }
    gctx.output = curr;

    mc.graphCache[key] = gctx;
  }

  MPSGraphTensorData *inputData = [[MPSGraphTensorData alloc]
      initWithMTLBuffer:(__bridge id<MTLBuffer>)input
                  shape:@[ @(rows), @(inCols) ]
               dataType:MPSDataTypeFloat16];
  MPSGraphTensorData *weightData = [[MPSGraphTensorData alloc]
      initWithMTLBuffer:(__bridge id<MTLBuffer>)weight
                  shape:@[ @(inCols), @(outCols) ]
               dataType:MPSDataTypeFloat16];
  MPSGraphTensorData *biasData =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:(__bridge id<MTLBuffer>)bias
                                              shape:@[ @(outCols) ]
                                           dataType:MPSDataTypeFloat16];
  MPSGraphTensorData *resultData = [[MPSGraphTensorData alloc]
      initWithMTLBuffer:(__bridge id<MTLBuffer>)result
                  shape:@[ @(rows), @(outCols) ]
               dataType:MPSDataTypeFloat16];

  runGraph(
      mc, gctx.graph,
      @{gctx.input : inputData, gctx.weight : weightData, gctx.bias : biasData},
      @{gctx.output : resultData});
}

void Metal_Attention_Graph(MetalContextRef ctx, MetalBufferRef q, int offQ,
                           MetalBufferRef k, int offK, MetalBufferRef v,
                           int offV, MetalBufferRef result, int offRes,
                           int batchSize, int seqLen, int hiddenSize,
                           float scale) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  [mc flush];

  NSString *key = [NSString
      stringWithFormat:@"Att_%d_%d_%d", batchSize, seqLen, hiddenSize];
  GraphContext *gctx = mc.graphCache[key]; // Reuse GraphContext

  if (!gctx) {
    gctx = [[GraphContext alloc] init];
    gctx.graph = [[MPSGraph alloc] init];

    // Placeholders (Flattened inputs)
    gctx.input = [gctx.graph
        placeholderWithShape:@[ @(batchSize * seqLen), @(hiddenSize) ]
                    dataType:MPSDataTypeFloat16
                        name:nil]; // Q
    gctx.weight = [gctx.graph
        placeholderWithShape:@[ @(batchSize * seqLen), @(hiddenSize) ]
                    dataType:MPSDataTypeFloat16
                        name:nil]; // K
    gctx.bias = [gctx.graph
        placeholderWithShape:@[ @(batchSize * seqLen), @(hiddenSize) ]
                    dataType:MPSDataTypeFloat16
                        name:nil]; // V

    // Reshape to (Batch, Seq, Hidden)
    MPSGraphTensor *q3d =
        [gctx.graph reshapeTensor:gctx.input
                        withShape:@[ @(batchSize), @(seqLen), @(hiddenSize) ]
                             name:nil];
    MPSGraphTensor *k3d =
        [gctx.graph reshapeTensor:gctx.weight
                        withShape:@[ @(batchSize), @(seqLen), @(hiddenSize) ]
                             name:nil];
    MPSGraphTensor *v3d =
        [gctx.graph reshapeTensor:gctx.bias
                        withShape:@[ @(batchSize), @(seqLen), @(hiddenSize) ]
                             name:nil];

    // Transpose K -> (Batch, Hidden, Seq)
    MPSGraphTensor *kT = [gctx.graph transposeTensor:k3d
                                           dimension:-1
                                       withDimension:-2
                                                name:nil];

    // Q * K^T -> (Batch, Seq, Seq)
    MPSGraphTensor *scores =
        [gctx.graph matrixMultiplicationWithPrimaryTensor:q3d
                                          secondaryTensor:kT
                                                     name:nil];

    // Scale
    MPSGraphTensor *scaleTensor =
        [gctx.graph constantWithScalar:scale dataType:MPSDataTypeFloat16];
    scores = [gctx.graph multiplicationWithPrimaryTensor:scores
                                         secondaryTensor:scaleTensor
                                                    name:nil];

    // Softmax
    scores = [gctx.graph softMaxWithTensor:scores axis:-1 name:nil];

    // Context = Scores * V -> (Batch, Seq, Hidden)
    MPSGraphTensor *context =
        [gctx.graph matrixMultiplicationWithPrimaryTensor:scores
                                          secondaryTensor:v3d
                                                     name:nil];

    // Reshape back to (Batch*Seq, Hidden)
    gctx.output =
        [gctx.graph reshapeTensor:context
                        withShape:@[ @(batchSize * seqLen), @(hiddenSize) ]
                             name:nil];

    mc.graphCache[key] = gctx;
  }

  NSArray *shape = @[ @(batchSize * seqLen), @(hiddenSize) ];

  MPSGraphTensorData *qData =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:(__bridge id<MTLBuffer>)q
                                              shape:shape
                                           dataType:MPSDataTypeFloat16];
  MPSGraphTensorData *kData =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:(__bridge id<MTLBuffer>)k
                                              shape:shape
                                           dataType:MPSDataTypeFloat16];
  MPSGraphTensorData *vData =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:(__bridge id<MTLBuffer>)v
                                              shape:shape
                                           dataType:MPSDataTypeFloat16];
  MPSGraphTensorData *resData = [[MPSGraphTensorData alloc]
      initWithMTLBuffer:(__bridge id<MTLBuffer>)result
                  shape:shape
               dataType:MPSDataTypeFloat16];

  runGraph(mc, gctx.graph,
           @{gctx.input : qData, gctx.weight : kData, gctx.bias : vData},
           @{gctx.output : resData});
}

void Metal_Synchronize(MetalContextRef ctx) {
  MetalWrapper *wrapper = (__bridge MetalWrapper *)ctx;
  [wrapper fullSync];
}
