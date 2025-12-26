#import "metal_bridge.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

@interface MetalWrapper : NSObject
@property(strong) id<MTLDevice> device;
@property(strong) id<MTLCommandQueue> commandQueue;
@property(strong) id<MTLLibrary> library;
@property(strong) id<MTLComputePipelineState> pipelineAdd;
@property(strong) id<MTLComputePipelineState> pipelineAddScalar;
@property(strong) id<MTLComputePipelineState> pipelineScale;
@property(strong) id<MTLComputePipelineState> pipelineTanh;
@property(strong) id<MTLComputePipelineState> pipelineGelu;
@property(strong) id<MTLComputePipelineState> pipelineLayerNorm;
@property(strong) id<MTLComputePipelineState> pipelineSoftmax;
@property(strong) id<MTLComputePipelineState> pipelineGather;
@property(strong) id<MTLComputePipelineState> pipelineAddBias;

// Async command batching
@property(strong) id<MTLCommandBuffer> currentCommandBuffer;
@property(strong) id<MTLComputeCommandEncoder> currentEncoder;
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

@end

MetalContextRef Metal_Init(const char *libSource) {
  MetalWrapper *ctx = [[MetalWrapper alloc] init];
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
  [buffer waitUntilCompleted];
}

void Metal_Synchronize(MetalContextRef ctx) {
  MetalWrapper *wrapper = (__bridge MetalWrapper *)ctx;
  [wrapper flush];
}
