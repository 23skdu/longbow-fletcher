// go:build metal
//  +build metal

#import "metal_bridge.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

@interface MetalWrapper : NSObject
@property(strong) id<MTLDevice> device;
@property(strong) id<MTLCommandQueue> commandQueue;
@property(strong) id<MTLLibrary> library;
@property(strong) NSMutableDictionary<NSString *, id> *bertGraphCache;

// Compute Pipelines
@property(strong) id<MTLComputePipelineState> pipelineAdd;
@property(strong) id<MTLComputePipelineState> pipelineAddScalar;
@property(strong) id<MTLComputePipelineState> pipelineScale;
@property(strong) id<MTLComputePipelineState> pipelineTanh;
@property(strong) id<MTLComputePipelineState> pipelineGelu;
@property(strong) id<MTLComputePipelineState> pipelineLayerNorm;
@property(strong) id<MTLComputePipelineState> pipelineAddLayerNorm;
@property(strong) id<MTLComputePipelineState> pipelineSoftmax;
@property(strong) id<MTLComputePipelineState> pipelineGather;
@property(strong) id<MTLComputePipelineState> pipelineAddBias;

@property(strong) id<MTLComputePipelineState> pipelineAdd_F16;
@property(strong) id<MTLComputePipelineState> pipelineAddScalar_F16;
@property(strong) id<MTLComputePipelineState> pipelineScale_F16;
@property(strong) id<MTLComputePipelineState> pipelineTanh_F16;
@property(strong) id<MTLComputePipelineState> pipelineGelu_F16;
@property(strong) id<MTLComputePipelineState> pipelineSoftmax_F16;
@property(strong) id<MTLComputePipelineState> pipelineLayerNorm_F16;
@property(strong) id<MTLComputePipelineState> pipelineAddLayerNorm_F16;
@property(strong) id<MTLComputePipelineState> pipelineAddBias_F16;
@property(strong) id<MTLComputePipelineState> pipelineAddBiasGelu_F16;
@property(strong) id<MTLComputePipelineState> pipelineAddBiasTanh_F16;
@property(strong) id<MTLComputePipelineState> pipelineRope_F16;
@property(strong) id<MTLComputePipelineState> pipelineSwiglu_F16;
@property(strong) id<MTLComputePipelineState> pipelineCast_F32_to_F16;
@property(strong) id<MTLComputePipelineState> pipelineCast_F16_to_F32;
@property(strong) id<MTLComputePipelineState> pipelineCopySubmatrix;
@property(strong) id<MTLComputePipelineState> pipelineCopySubmatrix_F16;
@property(strong) id<MTLComputePipelineState> pipelineFlashAttn;
@property(strong) id<MTLComputePipelineState> pipelineCheckNaN_F32;
@property(strong) id<MTLComputePipelineState> pipelineCheckNaN_F16;

@property(strong) id<MTLCommandBuffer> currentCommandBuffer;
@property(strong) id<MTLComputeCommandEncoder> currentEncoder;
@property(strong) id<MTLCommandBuffer> lastCommittedBuffer;
@end

@implementation MetalWrapper
- (void)stopEncoder {
  @synchronized(self) {
    if (self.currentEncoder) {
      [self.currentEncoder endEncoding];
      self.currentEncoder = nil;
    }
  }
}

- (void)ensureEncoder {
  @synchronized(self) {
    if (!self.currentCommandBuffer) {
      self.currentCommandBuffer = [self.commandQueue commandBuffer];
    }
    if (!self.currentEncoder) {
      if (@available(macOS 10.14, *)) {
        MTLComputePassDescriptor *desc =
            [MTLComputePassDescriptor computePassDescriptor];
        desc.dispatchType = MTLDispatchTypeConcurrent;
        self.currentEncoder = [self.currentCommandBuffer
            computeCommandEncoderWithDescriptor:desc];
      } else {
        // Fallback for older macOS
        self.currentEncoder = [self.currentCommandBuffer computeCommandEncoder];
      }
    }
  }
}

- (void)ensureCommandBuffer {
  @synchronized(self) {
    if (!self.currentCommandBuffer) {
      self.currentCommandBuffer = [self.commandQueue commandBuffer];
    }
  }
}

- (void)flush {
  @synchronized(self) {
    [self stopEncoder];
    if (self.currentCommandBuffer) {
      self.lastCommittedBuffer = self.currentCommandBuffer;
      [self.currentCommandBuffer commit];
      self.currentCommandBuffer = nil;
    }
  }
}

- (void)fullSync {
  @synchronized(self) {
    [self flush];
    if (self.lastCommittedBuffer) {
      [self.lastCommittedBuffer waitUntilCompleted];
      self.lastCommittedBuffer = nil;
    }
  }
}

- (BOOL)isCompleted {
  @synchronized(self) {
    if (self.currentCommandBuffer)
      return NO;
    if (!self.lastCommittedBuffer)
      return YES;
    return self.lastCommittedBuffer.status >= MTLCommandBufferStatusCompleted;
  }
}
@end

static id<MTLComputePipelineState> loadPipeline(MetalWrapper *ctx,
                                                NSString *name) {
  id<MTLFunction> fn = [ctx.library newFunctionWithName:name];
  if (!fn)
    return nil;
  return [ctx.device newComputePipelineStateWithFunction:fn error:nil];
}

MetalContextRef Metal_Init(const char *libSource) {
  MetalWrapper *ctx = [[MetalWrapper alloc] init];
  ctx.device = MTLCreateSystemDefaultDevice();
  if (!ctx.device)
    return NULL;
  ctx.commandQueue = [ctx.device newCommandQueue];
  ctx.bertGraphCache = [NSMutableDictionary dictionary];

  NSError *error = nil;
  NSString *src = [NSString stringWithUTF8String:libSource];
  ctx.library = [ctx.device newLibraryWithSource:src options:nil error:&error];
  if (error) {
    NSLog(@"Metal compilation error: %@", error);
    return NULL;
  }

  ctx.pipelineAdd = loadPipeline(ctx, @"add_kernel");
  ctx.pipelineAddScalar = loadPipeline(ctx, @"add_scalar_kernel");
  ctx.pipelineScale = loadPipeline(ctx, @"scale_kernel");
  ctx.pipelineTanh = loadPipeline(ctx, @"tanh_kernel");
  ctx.pipelineGelu = loadPipeline(ctx, @"gelu_kernel");
  ctx.pipelineLayerNorm = loadPipeline(ctx, @"layernorm_kernel");
  ctx.pipelineAddLayerNorm = loadPipeline(ctx, @"add_layernorm_kernel");
  ctx.pipelineSoftmax = loadPipeline(ctx, @"softmax_kernel");
  ctx.pipelineGather = loadPipeline(ctx, @"gather_kernel");
  ctx.pipelineAddBias = loadPipeline(ctx, @"add_bias_kernel");

  ctx.pipelineAdd_F16 = loadPipeline(ctx, @"add_kernel_f16");
  ctx.pipelineAddScalar_F16 = loadPipeline(ctx, @"add_scalar_kernel_f16");
  ctx.pipelineScale_F16 = loadPipeline(ctx, @"scale_kernel_f16");
  ctx.pipelineTanh_F16 = loadPipeline(ctx, @"tanh_kernel_f16");
  ctx.pipelineGelu_F16 = loadPipeline(ctx, @"gelu_approx_kernel_f16");
  ctx.pipelineSoftmax_F16 = loadPipeline(ctx, @"softmax_kernel_f16");
  ctx.pipelineLayerNorm_F16 = loadPipeline(ctx, @"layernorm_kernel_f16");
  ctx.pipelineAddLayerNorm_F16 = loadPipeline(ctx, @"add_layernorm_kernel_f16");
  ctx.pipelineAddBias_F16 = loadPipeline(ctx, @"add_bias_kernel_f16");
  ctx.pipelineAddBiasGelu_F16 = loadPipeline(ctx, @"add_bias_gelu_kernel_f16");
  ctx.pipelineAddBiasTanh_F16 = loadPipeline(ctx, @"add_bias_tanh_kernel_f16");
  ctx.pipelineRope_F16 = loadPipeline(ctx, @"rope_kernel_f16");
  ctx.pipelineSwiglu_F16 = loadPipeline(ctx, @"swiglu_kernel_f16");
  ctx.pipelineCast_F32_to_F16 = loadPipeline(ctx, @"cast_f32_to_f16");
  ctx.pipelineCast_F16_to_F32 = loadPipeline(ctx, @"cast_f16_to_f32");
  ctx.pipelineCopySubmatrix = loadPipeline(ctx, @"copy_submatrix");
  ctx.pipelineCopySubmatrix_F16 = loadPipeline(ctx, @"copy_submatrix_f16");
  ctx.pipelineFlashAttn = loadPipeline(ctx, @"flash_attn_fwd_f16");
  ctx.pipelineCheckNaN_F32 = loadPipeline(ctx, @"check_nan_f32");
  ctx.pipelineCheckNaN_F16 = loadPipeline(ctx, @"check_nan_f16");

  return (__bridge_retained MetalContextRef)ctx;
}

void Metal_Free(MetalContextRef ctx) {
  if (ctx) {
    MetalWrapper *wrapper = (__bridge_transfer MetalWrapper *)ctx;
    [wrapper fullSync];
    wrapper = nil;
  }
}

MetalBufferRef Metal_Alloc(MetalContextRef ctx, int size) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  id<MTLBuffer> buf =
      [mc.device newBufferWithLength:size options:MTLResourceStorageModeShared];
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
  memcpy((char *)[buffer contents] + offset, data, size);
  [buffer didModifyRange:NSMakeRange(offset, size)];
}

void Metal_CopyToHost(MetalBufferRef buf, int offset, void *data, int size) {
  id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)buf;
  memcpy(data, (char *)[buffer contents] + offset, size);
}

void Metal_Blit(MetalContextRef ctx, MetalBufferRef src, int srcOff,
                MetalBufferRef dst, int dstOff, int len) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;

  // End current compute encoder if active
  if (c.currentEncoder) {
    [c.currentEncoder endEncoding];
    c.currentEncoder = nil;
  }
  [c ensureCommandBuffer];

  // Use Blit Encoder
  id<MTLBlitCommandEncoder> blit = [c.currentCommandBuffer blitCommandEncoder];
  [blit copyFromBuffer:(__bridge id<MTLBuffer>)src
           sourceOffset:srcOff
               toBuffer:(__bridge id<MTLBuffer>)dst
      destinationOffset:dstOff
                   size:len];
  [blit endEncoding];

  // Note: We ended the compute encoder, so next compute op will start a new
  // one. This is fine. Blit implicitly synchronizes with Compute if on same
  // buffer in same command buffer? Metal tracks hazards.
}

void *Metal_GetBufferContents(MetalBufferRef buf) {
  id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)buf;
  return [buffer contents];
}

void Metal_SetAt(MetalBufferRef buf, int o, float v) {
  *(float *)((char *)[(__bridge id<MTLBuffer>)buf contents] + o) = v;
}
void Metal_Memset(MetalBufferRef buf, int o, int v, int s) {
  memset((char *)[(__bridge id<MTLBuffer>)buf contents] + o, v, s);
}

void Metal_ExtractBytes(MetalBufferRef buf, int offset, void *dest, int size) {
  id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)buf;
  memcpy(dest, (char *)[buffer contents] + offset, size);
}

// Forward Declarations
void Metal_MatMul_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                      bool transA, MetalBufferRef b, int offB, bool transB,
                      MetalBufferRef c, int offC, int M, int N, int K);
void Metal_MatMul_F16_F32(MetalContextRef ctx, MetalBufferRef a, int offA,
                          bool transA, MetalBufferRef b, int offB, bool transB,
                          MetalBufferRef c, int offC, int M, int N, int K);

// Kernels Implementation
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
  [c.currentEncoder dispatchThreads:MTLSizeMake(count, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(MIN(count, 512), 1, 1)];
}

void Metal_Cast_F32_to_F16(MetalContextRef ctx, MetalBufferRef input, int offIn,
                           MetalBufferRef output, int offOut, int count) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineCast_F32_to_F16);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)input
                       offset:offIn
                      atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)output
                       offset:offOut
                      atIndex:1];
  [c.currentEncoder dispatchThreads:MTLSizeMake(count, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(MIN(count, 512), 1, 1)];
}

void Metal_Cast_F16_to_F32(MetalContextRef ctx, MetalBufferRef input, int offIn,
                           MetalBufferRef output, int offOut, int count) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineCast_F16_to_F32);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)input
                       offset:offIn
                      atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)output
                       offset:offOut
                      atIndex:1];
  [c.currentEncoder dispatchThreads:MTLSizeMake(count, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(MIN(count, 512), 1, 1)];
}

void Metal_AddScalar(MetalContextRef ctx, MetalBufferRef a, int offA, float val,
                     MetalBufferRef result, int offRes, int count) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineAddScalar);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)a offset:offA atIndex:0];
  [c.currentEncoder setBytes:&val length:4 atIndex:1];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:2];
  [c.currentEncoder dispatchThreads:MTLSizeMake(count, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(MIN(count, 512), 1, 1)];
}

void Metal_Scale(MetalContextRef ctx, MetalBufferRef a, int offA, float val,
                 MetalBufferRef result, int offRes, int count) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineScale);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)a offset:offA atIndex:0];
  [c.currentEncoder setBytes:&val length:4 atIndex:1];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:2];
  [c.currentEncoder dispatchThreads:MTLSizeMake(count, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(MIN(count, 512), 1, 1)];
}

void Metal_Tanh(MetalContextRef ctx, MetalBufferRef a, int offA,
                MetalBufferRef result, int offRes, int count) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineTanh);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)a offset:offA atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:1];
  [c.currentEncoder dispatchThreads:MTLSizeMake(count, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(MIN(count, 512), 1, 1)];
}

void Metal_Gelu(MetalContextRef ctx, MetalBufferRef a, int offA,
                MetalBufferRef result, int offRes, int count) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineGelu);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)a offset:offA atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:1];
  [c.currentEncoder dispatchThreads:MTLSizeMake(count, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(MIN(count, 512), 1, 1)];
}

void Metal_LayerNorm(MetalContextRef ctx, MetalBufferRef in, int offIn,
                     MetalBufferRef gamma, int offGamma, MetalBufferRef beta,
                     int offBeta, MetalBufferRef result, int offRes, int rows,
                     int cols, float eps) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineLayerNorm);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)in
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
  [c.currentEncoder setBytes:&cols length:4 atIndex:4];
  [c.currentEncoder setBytes:&eps length:4 atIndex:5];
  [c.currentEncoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void Metal_AddLayerNorm(MetalContextRef ctx, MetalBufferRef in, int offIn,
                        MetalBufferRef residual, int offResid,
                        MetalBufferRef gamma, int offGamma, MetalBufferRef beta,
                        int offBeta, MetalBufferRef result, int offRes,
                        int rows, int cols, float eps) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineAddLayerNorm);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)in
                       offset:offIn
                      atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)residual
                       offset:offResid
                      atIndex:1];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)gamma
                       offset:offGamma
                      atIndex:2];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)beta
                       offset:offBeta
                      atIndex:3];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:4];
  [c.currentEncoder setBytes:&cols length:4 atIndex:5];
  [c.currentEncoder setBytes:&eps length:4 atIndex:6];
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
  [c.currentEncoder setBytes:&cols length:4 atIndex:2];
  [c.currentEncoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
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
  [c.currentEncoder setBytes:&cols length:4 atIndex:3];
  [c.currentEncoder dispatchThreads:MTLSizeMake(indicesCount, cols, 1)
              threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
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
  [c.currentEncoder setBytes:&cols length:4 atIndex:2];
  [c.currentEncoder dispatchThreads:MTLSizeMake(rows, cols, 1)
              threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
}

// FP16 Kernels
void Metal_Add_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                   MetalBufferRef b, int offB, MetalBufferRef result,
                   int offRes, int count) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineAdd_F16);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)a offset:offA atIndex:0];
  if (b)
    [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)b
                         offset:offB
                        atIndex:1];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:2];
  [c.currentEncoder dispatchThreads:MTLSizeMake(count, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(MIN(count, 512), 1, 1)];
}

void Metal_AddScalar_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                         uint16_t val, MetalBufferRef result, int offRes,
                         int count) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineAddScalar_F16);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)a offset:offA atIndex:0];
  [c.currentEncoder setBytes:&val length:2 atIndex:1];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:2];
  [c.currentEncoder dispatchThreads:MTLSizeMake(count, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(MIN(count, 512), 1, 1)];
}

void Metal_Scale_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                     uint16_t val, MetalBufferRef result, int offRes,
                     int count) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineScale_F16);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)a offset:offA atIndex:0];
  [c.currentEncoder setBytes:&val length:2 atIndex:1];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:2];
  [c.currentEncoder dispatchThreads:MTLSizeMake(count, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(MIN(count, 512), 1, 1)];
}

void Metal_Tanh_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                    MetalBufferRef result, int offRes, int count) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineTanh_F16);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)a offset:offA atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:1];
  [c.currentEncoder dispatchThreads:MTLSizeMake(count, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(MIN(count, 512), 1, 1)];
}

void Metal_Gelu_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                    MetalBufferRef result, int offRes, int count) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineGelu_F16);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)a offset:offA atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:1];
  [c.currentEncoder dispatchThreads:MTLSizeMake(count, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(MIN(count, 512), 1, 1)];
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
  [c.currentEncoder setBytes:&cols length:4 atIndex:2];
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
  [c.currentEncoder setBytes:&cols length:4 atIndex:4];
  [c.currentEncoder setBytes:&eps length:4 atIndex:5];
  [c.currentEncoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void Metal_AddLayerNorm_F16(MetalContextRef ctx, MetalBufferRef in, int offIn,
                            MetalBufferRef residual, int offResid,
                            MetalBufferRef gamma, int offGamma,
                            MetalBufferRef beta, int offBeta,
                            MetalBufferRef result, int offRes, int rows,
                            int cols, float eps) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineAddLayerNorm_F16);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)in
                       offset:offIn
                      atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)residual
                       offset:offResid
                      atIndex:1];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)gamma
                       offset:offGamma
                      atIndex:2];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)beta
                       offset:offBeta
                      atIndex:3];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:4];
  [c.currentEncoder setBytes:&cols length:4 atIndex:5];
  [c.currentEncoder setBytes:&eps length:4 atIndex:6];
  [c.currentEncoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
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
  [c.currentEncoder setBytes:&cols length:4 atIndex:3];
  [c.currentEncoder dispatchThreads:MTLSizeMake(cols, rows, 1)
              threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
}

void Metal_AddBiasGelu_F16(MetalContextRef ctx, MetalBufferRef matrix,
                           int offMat, MetalBufferRef bias, int offBias,
                           MetalBufferRef result, int offRes, int rows,
                           int cols) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineAddBiasGelu_F16);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)matrix
                       offset:offMat
                      atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)bias
                       offset:offBias
                      atIndex:1];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:2];
  [c.currentEncoder setBytes:&cols length:4 atIndex:3];
  [c.currentEncoder dispatchThreads:MTLSizeMake(cols, rows, 1)
              threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
}

void Metal_AddBiasTanh_F16(MetalContextRef ctx, MetalBufferRef matrix,
                           int offMat, MetalBufferRef bias, int offBias,
                           MetalBufferRef result, int offRes, int rows,
                           int cols) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineAddBiasTanh_F16);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)matrix
                       offset:offMat
                      atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)bias
                       offset:offBias
                      atIndex:1];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:offRes
                      atIndex:2];
  [c.currentEncoder setBytes:&cols length:4 atIndex:3];
  [c.currentEncoder dispatchThreads:MTLSizeMake(cols, rows, 1)
              threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
}

void Metal_ApplyRoPE_F16(MetalContextRef ctx, MetalBufferRef data, int offData,
                         int batchSize, int seqLen, int numHeads, int headDim) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineRope_F16);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)data
                       offset:offData
                      atIndex:0];
  [c.currentEncoder setBytes:&headDim length:4 atIndex:1];
  [c.currentEncoder setBytes:&numHeads length:4 atIndex:2];
  [c.currentEncoder setBytes:&seqLen length:4 atIndex:3];
  // Vectorized kernel handles 4 pairs (8 elements) per thread
  int threadsX = (headDim / 2 + 3) / 4;
  [c.currentEncoder
            dispatchThreads:MTLSizeMake(threadsX, numHeads, batchSize * seqLen)
      threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
}

void Metal_SwiGLU_F16(MetalContextRef ctx, MetalBufferRef input, int offIn,
                      MetalBufferRef output, int offOut, int n, int interSize) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineSwiglu_F16);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)input
                       offset:offIn
                      atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)output
                       offset:offOut
                      atIndex:1];
  [c.currentEncoder setBytes:&interSize length:4 atIndex:2];
  // Vectorized kernel handles 4 elements per thread
  int threadsX = (interSize + 3) / 4;
  [c.currentEncoder dispatchThreads:MTLSizeMake(threadsX, n, 1)
              threadsPerThreadgroup:MTLSizeMake(MIN(threadsX, 512), 1, 1)];
}

// Matrix Multiplication (MPS)
void Metal_MatMul(MetalContextRef ctx, MetalBufferRef a, int offA, bool transA,
                  MetalBufferRef b, int offB, bool transB, MetalBufferRef c,
                  int offC, int M, int N, int K) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  [mc stopEncoder];
  [mc ensureCommandBuffer];
  MPSMatrixDescriptor *dA = [MPSMatrixDescriptor
      matrixDescriptorWithRows:(transA ? K : M)
                       columns:(transA ? M : K)rowBytes:(transA ? M : K) * 4
                      dataType:MPSDataTypeFloat32];
  MPSMatrixDescriptor *dB = [MPSMatrixDescriptor
      matrixDescriptorWithRows:(transB ? N : K)
                       columns:(transB ? K : N)rowBytes:(transB ? K : N) * 4
                      dataType:MPSDataTypeFloat32];
  MPSMatrixDescriptor *dC =
      [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                            columns:N
                                           rowBytes:N * 4
                                           dataType:MPSDataTypeFloat32];
  MPSMatrix *mA = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)a
                                             offset:offA
                                         descriptor:dA];
  MPSMatrix *mB = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)b
                                             offset:offB
                                         descriptor:dB];
  MPSMatrix *mC = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)c
                                             offset:offC
                                         descriptor:dC];
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
                  leftMatrix:mA
                 rightMatrix:mB
                resultMatrix:mC];
}

void Metal_MatMul_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                      bool transA, MetalBufferRef b, int offB, bool transB,
                      MetalBufferRef c, int offC, int M, int N, int K) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  [mc stopEncoder];
  [mc ensureCommandBuffer];
  MPSMatrixDescriptor *dA = [MPSMatrixDescriptor
      matrixDescriptorWithRows:(transA ? K : M)
                       columns:(transA ? M : K)rowBytes:(transA ? M : K) * 2
                      dataType:MPSDataTypeFloat16];
  MPSMatrixDescriptor *dB = [MPSMatrixDescriptor
      matrixDescriptorWithRows:(transB ? N : K)
                       columns:(transB ? K : N)rowBytes:(transB ? K : N) * 2
                      dataType:MPSDataTypeFloat16];
  MPSMatrixDescriptor *dC =
      [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                            columns:N
                                           rowBytes:N * 2
                                           dataType:MPSDataTypeFloat16];
  MPSMatrix *mA = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)a
                                             offset:offA
                                         descriptor:dA];
  MPSMatrix *mB = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)b
                                             offset:offB
                                         descriptor:dB];
  MPSMatrix *mC = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)c
                                             offset:offC
                                         descriptor:dC];
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
                  leftMatrix:mA
                 rightMatrix:mB
                resultMatrix:mC];
}

void Metal_BatchedMatMul(MetalContextRef ctx, MetalBufferRef a, int offA,
                         int strideA, bool transA, MetalBufferRef b, int offB,
                         int strideB, bool transB, MetalBufferRef c, int offC,
                         int strideC, int M, int N, int K, int batchCount) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  [mc stopEncoder];
  [mc ensureCommandBuffer];
  MPSMatrixDescriptor *dA = [MPSMatrixDescriptor
      matrixDescriptorWithRows:(transA ? K : M)
                       columns:(transA ? M : K)rowBytes:(transA ? M : K) * 4
                      dataType:MPSDataTypeFloat32];
  MPSMatrixDescriptor *dB = [MPSMatrixDescriptor
      matrixDescriptorWithRows:(transB ? N : K)
                       columns:(transB ? K : N)rowBytes:(transB ? K : N) * 4
                      dataType:MPSDataTypeFloat32];
  MPSMatrixDescriptor *dC =
      [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                            columns:N
                                           rowBytes:N * 4
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
    MPSMatrix *mA = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)a
                                               offset:offA + i * strideA
                                           descriptor:dA];
    MPSMatrix *mB = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)b
                                               offset:offB + i * strideB
                                           descriptor:dB];
    MPSMatrix *mC = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)c
                                               offset:offC + i * strideC
                                           descriptor:dC];
    [mul encodeToCommandBuffer:mc.currentCommandBuffer
                    leftMatrix:mA
                   rightMatrix:mB
                  resultMatrix:mC];
  }
}

void Metal_BatchedMatMul_F16(MetalContextRef ctx, MetalBufferRef a, int offA,
                             int strideA, bool transA, MetalBufferRef b,
                             int offB, int strideB, bool transB,
                             MetalBufferRef c, int offC, int strideC, int M,
                             int N, int K, int batchCount) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  [mc stopEncoder];
  [mc ensureCommandBuffer];
  MPSMatrixDescriptor *dA = [MPSMatrixDescriptor
      matrixDescriptorWithRows:(transA ? K : M)
                       columns:(transA ? M : K)rowBytes:(transA ? M : K) * 2
                      dataType:MPSDataTypeFloat16];
  MPSMatrixDescriptor *dB = [MPSMatrixDescriptor
      matrixDescriptorWithRows:(transB ? N : K)
                       columns:(transB ? K : N)rowBytes:(transB ? K : N) * 2
                      dataType:MPSDataTypeFloat16];
  MPSMatrixDescriptor *dC =
      [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                            columns:N
                                           rowBytes:N * 2
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
    MPSMatrix *mA = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)a
                                               offset:offA + i * strideA
                                           descriptor:dA];
    MPSMatrix *mB = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)b
                                               offset:offB + i * strideB
                                           descriptor:dB];
    MPSMatrix *mC = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)c
                                               offset:offC + i * strideC
                                           descriptor:dC];
    [mul encodeToCommandBuffer:mc.currentCommandBuffer
                    leftMatrix:mA
                   rightMatrix:mB
                  resultMatrix:mC];
  }
}
// Mixed Precision MatMul: F16 Inputs -> F32 Output (for Scores accumulation)
void Metal_BatchedMatMul_F16_F32(MetalContextRef ctx, MetalBufferRef a,
                                 int offA, int strideA, bool transA,
                                 MetalBufferRef b, int offB, int strideB,
                                 bool transB, MetalBufferRef c, int offC,
                                 int strideC, int M, int N, int K,
                                 int batchCount) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  [mc stopEncoder];
  [mc ensureCommandBuffer];

  // Inputs are F16
  MPSMatrixDescriptor *dA = [MPSMatrixDescriptor
      matrixDescriptorWithRows:(transA ? K : M)
                       columns:(transA ? M : K)rowBytes:(transA ? M : K) * 2
                      dataType:MPSDataTypeFloat16];
  MPSMatrixDescriptor *dB = [MPSMatrixDescriptor
      matrixDescriptorWithRows:(transB ? N : K)
                       columns:(transB ? K : N)rowBytes:(transB ? K : N) * 2
                      dataType:MPSDataTypeFloat16];

  // Output is F32 (!), strides passed are in bytes (or elements? Go sends bytes
  // usually? Wait, stride arguments in Metal_BatchedMatMul_F16 are passed as
  // just ints. In Go `Metal_Attention_Graph` calc: `strideQ = seqLen *
  // hiddenSize * 2`. So stride is BYTES. dA rowBytes check above uses element
  // count * 2? `matrixDescriptorWithRows` rowBytes argument expects BYTES. My
  // check `(transA ? M : K) * 2` calculates bytes assuming dense-packed row.
  // The passed `stride` args are batch strides (bytes between matrices).
  // Yes.

  MPSMatrixDescriptor *dC = [MPSMatrixDescriptor
      matrixDescriptorWithRows:M
                       columns:N
                      rowBytes:N * 4 // FP32 dense row bytes
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
    MPSMatrix *mA = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)a
                                               offset:offA + i * strideA
                                           descriptor:dA];
    MPSMatrix *mB = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)b
                                               offset:offB + i * strideB
                                           descriptor:dB];
    MPSMatrix *mC = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)c
                                               offset:offC + i * strideC
                                           descriptor:dC];
    [mul encodeToCommandBuffer:mc.currentCommandBuffer
                    leftMatrix:mA
                   rightMatrix:mB
                  resultMatrix:mC];
  }
}

void Metal_MatMul_F16_F32(MetalContextRef ctx, MetalBufferRef a, int offA,
                          bool transA, MetalBufferRef b, int offB, bool transB,
                          MetalBufferRef c, int offC, int M, int N, int K) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  [mc stopEncoder];
  [mc ensureCommandBuffer];
  MPSMatrixDescriptor *dA = [MPSMatrixDescriptor
      matrixDescriptorWithRows:(transA ? K : M)
                       columns:(transA ? M : K)rowBytes:(transA ? M : K) * 2
                      dataType:MPSDataTypeFloat16];
  MPSMatrixDescriptor *dB = [MPSMatrixDescriptor
      matrixDescriptorWithRows:(transB ? N : K)
                       columns:(transB ? K : N)rowBytes:(transB ? K : N) * 2
                      dataType:MPSDataTypeFloat16];
  MPSMatrixDescriptor *dC =
      [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                            columns:N
                                           rowBytes:N * 4
                                           dataType:MPSDataTypeFloat32];
  MPSMatrix *mA = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)a
                                             offset:offA
                                         descriptor:dA];
  MPSMatrix *mB = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)b
                                             offset:offB
                                         descriptor:dB];
  MPSMatrix *mC = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)c
                                             offset:offC
                                         descriptor:dC];
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
                  leftMatrix:mA
                 rightMatrix:mB
                resultMatrix:mC];
}
void Metal_MatMul_F16_F32_Stride(MetalContextRef ctx, MetalBufferRef a,
                                 int offA, int strideA, MetalBufferRef b,
                                 int offB, int strideB, MetalBufferRef c,
                                 int offC, int M, int N, int K) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  [mc stopEncoder];
  [mc ensureCommandBuffer];
  MPSMatrixDescriptor *dA =
      [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                            columns:K
                                           rowBytes:strideA
                                           dataType:MPSDataTypeFloat16];
  MPSMatrixDescriptor *dB =
      [MPSMatrixDescriptor matrixDescriptorWithRows:N
                                            columns:K
                                           rowBytes:strideB
                                           dataType:MPSDataTypeFloat16];
  MPSMatrixDescriptor *dC =
      [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                            columns:N
                                           rowBytes:N * 4
                                           dataType:MPSDataTypeFloat32];
  MPSMatrix *mA = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)a
                                             offset:offA
                                         descriptor:dA];
  MPSMatrix *mB = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)b
                                             offset:offB
                                         descriptor:dB];
  MPSMatrix *mC = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)c
                                             offset:offC
                                         descriptor:dC];
  MPSMatrixMultiplication *mul =
      [[MPSMatrixMultiplication alloc] initWithDevice:mc.device
                                        transposeLeft:false
                                       transposeRight:true
                                           resultRows:M
                                        resultColumns:N
                                      interiorColumns:K
                                                alpha:1.0
                                                 beta:0.0];
  [mul encodeToCommandBuffer:mc.currentCommandBuffer
                  leftMatrix:mA
                 rightMatrix:mB
                resultMatrix:mC];
}

void Metal_MatMul_F16_Stride(MetalContextRef ctx, MetalBufferRef a, int offA,
                             int strideA, MetalBufferRef b, int offB,
                             int strideB, MetalBufferRef result, int offRes,
                             int strideRes, int M, int N, int K) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  [mc stopEncoder];
  [mc ensureCommandBuffer];
  MPSMatrixDescriptor *dA =
      [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                            columns:K
                                           rowBytes:strideA
                                           dataType:MPSDataTypeFloat16];
  MPSMatrixDescriptor *dB =
      [MPSMatrixDescriptor matrixDescriptorWithRows:K
                                            columns:N
                                           rowBytes:strideB
                                           dataType:MPSDataTypeFloat16];
  MPSMatrixDescriptor *dC =
      [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                            columns:N
                                           rowBytes:strideRes
                                           dataType:MPSDataTypeFloat16];
  MPSMatrix *mA = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)a
                                             offset:offA
                                         descriptor:dA];
  MPSMatrix *mB = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)b
                                             offset:offB
                                         descriptor:dB];
  MPSMatrix *mC =
      [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)result
                                 offset:offRes
                             descriptor:dC];
  MPSMatrixMultiplication *mul =
      [[MPSMatrixMultiplication alloc] initWithDevice:mc.device
                                        transposeLeft:false
                                       transposeRight:false
                                           resultRows:M
                                        resultColumns:N
                                      interiorColumns:K
                                                alpha:1.0
                                                 beta:0.0];
  [mul encodeToCommandBuffer:mc.currentCommandBuffer
                  leftMatrix:mA
                 rightMatrix:mB
                resultMatrix:mC];
}

// Composite Ops
void Metal_Attention_Graph_v3(MetalContextRef ctx, MetalBufferRef q, int offQ,
                              MetalBufferRef k, int offK, MetalBufferRef v,
                              int offV, MetalBufferRef result, int offRes,
                              int batchSize, int seqLen, int hiddenSize,
                              int numHeads, float scale) {
  int headDim = hiddenSize / numHeads;
  int totalHeads = batchSize * numHeads;

  // printf("AttentionGraph: batch=%d seq=%d hidden=%d heads=%d headDim=%d\n",
  // batchSize, seqLen, hiddenSize, numHeads, headDim);

  // 1. Accumulate Scores in FP32
  int scoresElems = totalHeads * seqLen * seqLen;
  int scoresSize = scoresElems * 4; // FP32
  MetalBufferRef scoresBuf = Metal_Alloc(ctx, scoresSize);

  int strideQ = seqLen * hiddenSize * 2;
  int strideK = seqLen * hiddenSize * 2;
  int strideV = seqLen * hiddenSize * 2;
  int strideScores = seqLen * seqLen * 4;

  // Step 1: Q * K^T -> Scores (FP32)
  for (int b = 0; b < batchSize; b++) {
    for (int h = 0; h < numHeads; h++) {
      int headOff = h * headDim * 2;
      int i = b * numHeads + h;
      for (int s = 0; s < seqLen; s++) {
        // Query weight is at: b * strideQ + s * hiddenSize * 2 + headOff
        // but for MatMul we can pass the start of the head's matrix and a row
        // stride. Wait, our internal MatMul takes simple offsets. We need to be
        // careful: the heads are interleaved in the last dimension. Q: [Batch,
        // Seq, Heads, HeadDim] -> but stored as [Batch, Seq, HiddenSize]
      }
      // Actually, simple offset for the head start is enough IF row stride is
      // hiddenSize*2
      Metal_MatMul_F16_F32_Stride(
          ctx, q, offQ + b * strideQ + headOff, hiddenSize * 2, k,
          offK + b * strideK + headOff, hiddenSize * 2, scoresBuf,
          i * strideScores, seqLen, seqLen, headDim);
    }
  }

  // Scale (F32)
  Metal_Scale(ctx, scoresBuf, 0, scale, scoresBuf, 0, scoresElems);
  Metal_Synchronize(ctx);

  // Softmax (F32)
  Metal_Softmax(ctx, scoresBuf, 0, scoresBuf, 0, totalHeads * seqLen, seqLen);
  Metal_Synchronize(ctx);

  // Cast Scores F32 -> F16
  int scoresSizeF16 = scoresElems * 2;
  MetalBufferRef scoresBufF16 = Metal_Alloc(ctx, scoresSizeF16);

  Metal_Cast_F32_to_F16(ctx, scoresBuf, 0, scoresBufF16, 0, scoresElems);

  // Scores(F16) * V(F16) -> Result(F16)
  int strideScoresF16 = seqLen * seqLen * 2;

  for (int b = 0; b < batchSize; b++) {
    for (int h = 0; h < numHeads; h++) {
      int headOff = h * headDim * 2;
      int i = b * numHeads + h;
      Metal_MatMul_F16_Stride(ctx, scoresBufF16, i * strideScoresF16,
                              seqLen * 2, v, offV + b * strideV + headOff,
                              hiddenSize * 2, result,
                              offRes + b * strideQ + headOff, hiddenSize * 2,
                              seqLen, headDim, seqLen);
    }
  }

  Metal_Synchronize(ctx);
  Metal_FreeBuffer(ctx, scoresBuf);
  Metal_FreeBuffer(ctx, scoresBufF16);
}

// Fused Attention - Single kernel dispatch for better performance
// Fused Attention - Single kernel dispatch for better performance
void Metal_FusedAttention_F16(MetalContextRef ctx, MetalBufferRef q, int offQ,
                              MetalBufferRef k, int offK, MetalBufferRef v,
                              int offV, MetalBufferRef result, int offRes,
                              int batchSize, int seqLen, int hiddenSize,
                              int numHeads, float scale) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  int headDim = hiddenSize / numHeads;
  int totalHeads = batchSize * numHeads;

  // Allocate temporary buffer for attention scores
  int scoresSize = totalHeads * seqLen * seqLen * 2; // FP16
  MetalBufferRef scoresBuf = Metal_Alloc(ctx, scoresSize);

  int strideQ = seqLen * hiddenSize * 2;
  int strideK = seqLen * hiddenSize * 2;
  int strideScores = seqLen * seqLen * 2;
  int strideV = seqLen * hiddenSize * 2;

  // Step 1: Q × K^T with scale fused
  [mc stopEncoder];
  [mc ensureCommandBuffer];

  // Q is (seqLen, headDim) but sliced from (seqLen, hiddenSize)
  MPSMatrixDescriptor *dQ =
      [MPSMatrixDescriptor matrixDescriptorWithRows:seqLen
                                            columns:headDim
                                           rowBytes:hiddenSize * 2
                                           dataType:MPSDataTypeFloat16];
  // K is (seqLen, headDim)
  MPSMatrixDescriptor *dK =
      [MPSMatrixDescriptor matrixDescriptorWithRows:seqLen
                                            columns:headDim
                                           rowBytes:hiddenSize * 2
                                           dataType:MPSDataTypeFloat16];
  // Scores is (seqLen, seqLen)
  MPSMatrixDescriptor *dScores =
      [MPSMatrixDescriptor matrixDescriptorWithRows:seqLen
                                            columns:seqLen
                                           rowBytes:seqLen * 2
                                           dataType:MPSDataTypeFloat16];

  MPSMatrixMultiplication *mulQK =
      [[MPSMatrixMultiplication alloc] initWithDevice:mc.device
                                        transposeLeft:false
                                       transposeRight:true
                                           resultRows:seqLen
                                        resultColumns:seqLen
                                      interiorColumns:headDim
                                                alpha:scale // Fuse scale here!
                                                 beta:0.0];

  // Batch loop for Q×K^T
  for (int b = 0; b < batchSize; b++) {
    for (int h = 0; h < numHeads; h++) {
      int headOffset = h * headDim * 2;
      int i = b * numHeads + h;

      MPSMatrix *mQ =
          [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)q
                                     offset:offQ + b * strideQ + headOffset
                                 descriptor:dQ];
      MPSMatrix *mK =
          [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)k
                                     offset:offK + b * strideK + headOffset
                                 descriptor:dK];
      MPSMatrix *mScores =
          [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)scoresBuf
                                     offset:i * strideScores
                                 descriptor:dScores];
      [mulQK encodeToCommandBuffer:mc.currentCommandBuffer
                        leftMatrix:mQ
                       rightMatrix:mK
                      resultMatrix:mScores];
    }
  }

  // Step 2: Softmax
  Metal_Softmax_F16(ctx, scoresBuf, 0, scoresBuf, 0, totalHeads * seqLen,
                    seqLen);

  // Step 3: Scores × V
  // Scores(F16) [totalHeads, seqLen, seqLen] * V(F16) [totalHeads, seqLen,
  // headDim] -> Result(F16) [totalHeads, seqLen, headDim]
  Metal_BatchedMatMul_F16(ctx, scoresBuf, 0, strideScores, false, v, offV,
                          strideV / numHeads, false, result, offRes,
                          strideQ / numHeads, seqLen, headDim, seqLen,
                          totalHeads);

  Metal_FreeBuffer(ctx, scoresBuf);
}

void Metal_FusedAttention_VarLen_F16(
    MetalContextRef ctx, MetalBufferRef q, int offQ, MetalBufferRef k, int offK,
    MetalBufferRef v, int offV, MetalBufferRef result, int offRes, int *lengths,
    int batchSize, int hiddenSize, int numHeads, float scale) {
  MetalWrapper *mc = (__bridge MetalWrapper *)ctx;
  int headDim = hiddenSize / numHeads;

  // Calculate total scores size required (totalHeads * seqLen^2)
  int totalScoresParams = 0;
  for (int i = 0; i < batchSize; i++) {
    int l = lengths[i];
    totalScoresParams += l * l * numHeads;
  }

  // FP16
  int scoresSize = totalScoresParams * 2;
  MetalBufferRef scoresBuf = Metal_Alloc(ctx, scoresSize);

  [mc stopEncoder];
  [mc ensureCommandBuffer];

  // Pass 1: Q * K^T
  int currentIO_Offset = 0;
  int currentScores_Offset = 0;
  for (int i = 0; i < batchSize; i++) {
    int seqLen = lengths[i];
    if (seqLen == 0)
      continue;

    MPSMatrixMultiplication *mulQK =
        [[MPSMatrixMultiplication alloc] initWithDevice:mc.device
                                          transposeLeft:false
                                         transposeRight:true
                                             resultRows:seqLen
                                          resultColumns:seqLen
                                        interiorColumns:headDim
                                                  alpha:scale
                                                   beta:0.0];

    MPSMatrixDescriptor *dQ =
        [MPSMatrixDescriptor matrixDescriptorWithRows:seqLen
                                              columns:headDim
                                             rowBytes:hiddenSize * 2
                                             dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor *dScores =
        [MPSMatrixDescriptor matrixDescriptorWithRows:seqLen
                                              columns:seqLen
                                             rowBytes:seqLen * 2
                                             dataType:MPSDataTypeFloat16];

    for (int h = 0; h < numHeads; h++) {
      int headOffset = h * headDim * 2;
      MPSMatrix *mQ = [[MPSMatrix alloc]
          initWithBuffer:(__bridge id<MTLBuffer>)q
                  offset:offQ + currentIO_Offset * 2 + headOffset
              descriptor:dQ];
      MPSMatrix *mK = [[MPSMatrix alloc]
          initWithBuffer:(__bridge id<MTLBuffer>)k
                  offset:offK + currentIO_Offset * 2 + headOffset
              descriptor:dQ];
      MPSMatrix *mScores =
          [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)scoresBuf
                                     offset:currentScores_Offset * 2
                                 descriptor:dScores];

      [mulQK encodeToCommandBuffer:mc.currentCommandBuffer
                        leftMatrix:mQ
                       rightMatrix:mK
                      resultMatrix:mScores];
      currentScores_Offset += seqLen * seqLen;
    }
    currentIO_Offset += seqLen * hiddenSize;
  }

  // Pass 2: Softmax
  currentScores_Offset = 0;
  for (int i = 0; i < batchSize; i++) {
    int seqLen = lengths[i];
    if (seqLen == 0)
      continue;
    for (int h = 0; h < numHeads; h++) {
      int byteOffScores = currentScores_Offset * 2;
      Metal_Softmax_F16(ctx, scoresBuf, byteOffScores, scoresBuf, byteOffScores,
                        seqLen, seqLen);
      currentScores_Offset += seqLen * seqLen;
    }
  }
  [mc stopEncoder]; // Stop encoder before next MPS pass

  // Pass 3: Scores * V
  currentIO_Offset = 0;
  currentScores_Offset = 0;
  for (int i = 0; i < batchSize; i++) {
    int seqLen = lengths[i];
    if (seqLen == 0)
      continue;

    MPSMatrixMultiplication *mulSV =
        [[MPSMatrixMultiplication alloc] initWithDevice:mc.device
                                          transposeLeft:false
                                         transposeRight:false
                                             resultRows:seqLen
                                          resultColumns:headDim
                                        interiorColumns:seqLen
                                                  alpha:1.0
                                                   beta:0.0];

    MPSMatrixDescriptor *dScores =
        [MPSMatrixDescriptor matrixDescriptorWithRows:seqLen
                                              columns:seqLen
                                             rowBytes:seqLen * 2
                                             dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor *dV =
        [MPSMatrixDescriptor matrixDescriptorWithRows:seqLen
                                              columns:headDim
                                             rowBytes:hiddenSize * 2
                                             dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor *dO = dV;

    for (int h = 0; h < numHeads; h++) {
      int headOffset = h * headDim * 2;
      MPSMatrix *mScores =
          [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)scoresBuf
                                     offset:currentScores_Offset * 2
                                 descriptor:dScores];
      MPSMatrix *mV = [[MPSMatrix alloc]
          initWithBuffer:(__bridge id<MTLBuffer>)v
                  offset:offV + currentIO_Offset * 2 + headOffset
              descriptor:dV];
      MPSMatrix *mO = [[MPSMatrix alloc]
          initWithBuffer:(__bridge id<MTLBuffer>)result
                  offset:offRes + currentIO_Offset * 2 + headOffset
              descriptor:dO];

      [mulSV encodeToCommandBuffer:mc.currentCommandBuffer
                        leftMatrix:mScores
                       rightMatrix:mV
                      resultMatrix:mO];

      currentScores_Offset += seqLen * seqLen;
    }
    currentIO_Offset += seqLen * hiddenSize;
  }

  Metal_FreeBuffer(ctx, scoresBuf);
}

void Metal_FlashAttention(MetalContextRef ctx, MetalBufferRef Q, int offQ,
                          MetalBufferRef K, int offK, MetalBufferRef V,
                          int offV, MetalBufferRef O, int offO, int N, int d,
                          float scale, int batch_stride, int head_stride,
                          int row_stride, int num_heads, int total_batches) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineFlashAttn);

  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)Q offset:offQ atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)K offset:offK atIndex:1];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)V offset:offV atIndex:2];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)O offset:offO atIndex:3];

  [c.currentEncoder setBytes:&N length:4 atIndex:4];
  [c.currentEncoder setBytes:&d length:4 atIndex:5];
  [c.currentEncoder setBytes:&scale length:4 atIndex:6];
  [c.currentEncoder setBytes:&batch_stride length:4 atIndex:7];
  [c.currentEncoder setBytes:&head_stride length:4 atIndex:8];
  [c.currentEncoder setBytes:&row_stride length:4 atIndex:9];
  [c.currentEncoder setBytes:&num_heads length:4 atIndex:10];

  int blocks_n = (N + 31) / 32;
  [c.currentEncoder dispatchThreadgroups:MTLSizeMake(blocks_n, total_batches, 1)
                   threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

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
  // Optimize by fusing AddBias + Activation if possible
  // Metal_Linear_Graph calls MatMul + AddBias.
  // We can do MatMul + AddBiasActivation.

  // 1. MatMul
  Metal_MatMul_F16(ctx, input, offIn, false, weight, offWeight, false, result,
                   offRes, rows, outCols, inCols);

  // 2. AddBias + Activation
  if (activationType == 1) { // GELU
    Metal_AddBiasGelu_F16(ctx, result, offRes, bias, offBias, result, offRes,
                          rows, outCols);
  } else if (activationType == 2) { // Tanh
    Metal_AddBiasTanh_F16(ctx, result, offRes, bias, offBias, result, offRes,
                          rows, outCols);
  } else {
    // Fallback or just AddBias
    Metal_AddBias_F16(ctx, result, offRes, bias, offBias, result, offRes, rows,
                      outCols);
    if (activationType == 3) { // Softmax
      Metal_Softmax_F16(ctx, result, offRes, result, offRes, rows, outCols);
    }
  }
}

// Misc
void Metal_Synchronize(MetalContextRef ctx) {
  [(__bridge MetalWrapper *)ctx fullSync];
}
bool Metal_IsCompleted(MetalContextRef ctx) {
  return [(__bridge MetalWrapper *)ctx isCompleted];
}
unsigned long long Metal_GetAllocatedSize(MetalContextRef ctx) {
  return [((__bridge MetalWrapper *)ctx).device currentAllocatedSize];
}
unsigned long long Metal_GetRecommendMaxWorkingSetSize(MetalContextRef ctx) {
  return [((__bridge MetalWrapper *)ctx).device recommendedMaxWorkingSetSize];
}
void Metal_Tanh_Graph(MetalContextRef ctx, MetalBufferRef i, int oI,
                      MetalBufferRef r, int oR, int c) {
  Metal_Tanh_F16(ctx, i, oI, r, oR, c);
}
void Metal_Gelu_Graph(MetalContextRef ctx, MetalBufferRef i, int oI,
                      MetalBufferRef r, int oR, int c) {
  Metal_Gelu_F16(ctx, i, oI, r, oR, c);
}

// CopySubmatrix Implementations
void Metal_CopySubmatrix(MetalContextRef ctx, MetalBufferRef src, int offSrc,
                         int srcCols, MetalBufferRef dest, int offDest,
                         int destCols, int srcRowOff, int srcColOff, int rows,
                         int cols) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineCopySubmatrix);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)src
                       offset:offSrc
                      atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)dest
                       offset:offDest
                      atIndex:1];

  [c.currentEncoder setBytes:&srcCols length:4 atIndex:2];
  [c.currentEncoder setBytes:&destCols length:4 atIndex:3];
  [c.currentEncoder setBytes:&srcRowOff length:4 atIndex:4];
  [c.currentEncoder setBytes:&srcColOff length:4 atIndex:5];

  // Grid: (cols, rows, 1)
  [c.currentEncoder
            dispatchThreads:MTLSizeMake(cols, rows, 1)
      threadsPerThreadgroup:MTLSizeMake(MIN(cols, 32), MIN(rows, 32), 1)];
}

void Metal_CopySubmatrix_F16(MetalContextRef ctx, MetalBufferRef src,
                             int offSrc, int srcCols, MetalBufferRef dest,
                             int offDest, int destCols, int srcRowOff,
                             int srcColOff, int rows, int cols) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineCopySubmatrix_F16);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)src
                       offset:offSrc
                      atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)dest
                       offset:offDest
                      atIndex:1];

  [c.currentEncoder setBytes:&srcCols length:4 atIndex:2];
  [c.currentEncoder setBytes:&destCols length:4 atIndex:3];
  [c.currentEncoder setBytes:&srcRowOff length:4 atIndex:4];
  [c.currentEncoder setBytes:&srcColOff length:4 atIndex:5];

  // Grid: (cols, rows, 1)
  [c.currentEncoder
            dispatchThreads:MTLSizeMake(cols, rows, 1)
      threadsPerThreadgroup:MTLSizeMake(MIN(cols, 32), MIN(rows, 32), 1)];
}

void Metal_CheckNaN_F32(MetalContextRef ctx, MetalBufferRef input, int offIn,
                        int count, MetalBufferRef result) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineCheckNaN_F32);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)input
                       offset:offIn
                      atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:0
                      atIndex:1];
  [c.currentEncoder dispatchThreads:MTLSizeMake(count, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(MIN(count, 512), 1, 1)];
}

void Metal_CheckNaN_F16(MetalContextRef ctx, MetalBufferRef input, int offIn,
                        int count, MetalBufferRef result) {
  MetalWrapper *c = (__bridge MetalWrapper *)ctx;
  ENCODE(c, pipelineCheckNaN_F16);
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)input
                       offset:offIn
                      atIndex:0];
  [c.currentEncoder setBuffer:(__bridge id<MTLBuffer>)result
                       offset:0
                      atIndex:1];
  [c.currentEncoder dispatchThreads:MTLSizeMake(count, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(MIN(count, 512), 1, 1)];
}

