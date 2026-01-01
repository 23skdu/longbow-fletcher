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

// Graph Context for Caching
@interface BertGraphContext : NSObject
@property(strong) MPSGraph *graph;
@property(strong) MPSGraphTensor *tInput;
@property(strong) MPSGraphTensor *tWq, *tWk, *tWv, *tWout;
@property(strong) MPSGraphTensor *tWinter, *tWoutFFN;
@property(strong) MPSGraphTensor *tBq, *tBk, *tBv, *tBout;
@property(strong) MPSGraphTensor *tBinter, *tBoutFFN;
@property(strong) MPSGraphTensor *tGammaAttn, *tBetaAttn;
@property(strong) MPSGraphTensor *tGammaFFN, *tBetaFFN;
@property(strong) MPSGraphTensor *tFinal; // Output Tensor
@end

@implementation BertGraphContext
@end

// Helper to create MPSGraphTensorData, handling offset by copying if necessary
MPSGraphTensorData *dataFromBuffer(MetalContextRef ctx, MetalBufferRef buf,
                                   int offset, NSArray<NSNumber *> *shape,
                                   MPSDataType dtype) {
  id<MTLBuffer> mtlBuf = (__bridge id<MTLBuffer>)buf;
  MetalWrapper *wrapper = (__bridge MetalWrapper *)ctx;

  // CPU Readback Debug
  if (shape.count == 3 && shape[2].intValue == 64) { // Only Input
    uint16_t *cpuPtr = (uint16_t *)mtlBuf.contents;
    fprintf(stderr, "DEBUG CPU Readback Input: 0x%04x 0x%04x off=%d addr=%p\n",
            cpuPtr[0 + offset / 2], cpuPtr[1 + offset / 2], offset, mtlBuf);
  }

  int elemSize = (dtype == MPSDataTypeFloat16) ? 2 : 4;

  // Reshape Logic: Flatten Rank 3 [B, S, H] -> Rank 2 [B*S, H]
  // Or handle arbitrary Rank 2.
  NSUInteger rows = 1;
  NSUInteger cols = 1;

  if (shape.count >= 2) {
    cols = shape.lastObject.unsignedIntegerValue;
    for (NSUInteger i = 0; i < shape.count - 1; i++) {
      rows *= shape[i].unsignedIntegerValue;
    }
  } else {
    cols = shape[0].unsignedIntegerValue; // Rank 1 -> [1, C] or [C, 1]?
    // Rank 1 usually bias [H]. Treat as [1, H] Matrix.
    rows = 1;
  }

  // Calculate Padding Requirements
  BOOL needsPadding = NO;
  NSUInteger rowBytesPacked = cols * elemSize;
  NSUInteger rowBytesPadded = (rowBytesPacked + 255) & ~255;

  // Force Padding if mismatch AND rows > 1.
  // Exception: If rows=1, packed=padded for access purposes?
  // But MatrixDescriptor requires rowBytes>=packed.
  // If we specify Padded rowBytes, we MUST have allocated enough space.

  if (rowBytesPadded > rowBytesPacked && rows > 1) {
    needsPadding = YES;
  }
  // Also enforce padding if offset > 0 (to align to 0).
  if (offset > 0)
    needsPadding = YES;

  id<MTLBuffer> targetBuf = mtlBuf; // Default
  NSUInteger targetRowBytes = rowBytesPacked;

  if (needsPadding) {
    targetRowBytes = rowBytesPadded;
    NSUInteger allocSize = rows * targetRowBytes;
    id<MTLBuffer> tmp =
        [wrapper.device newBufferWithLength:allocSize
                                    options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cb = [wrapper.commandQueue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];

    for (NSUInteger i = 0; i < rows; i++) {
      [blit copyFromBuffer:mtlBuf
               sourceOffset:offset + (i * rowBytesPacked)
                   toBuffer:tmp
          destinationOffset:(i * targetRowBytes)
                       size:rowBytesPacked];
    }

    [blit endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    targetBuf = tmp;
  }

  // Create MPSMatrix
  MPSMatrixDescriptor *desc =
      [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                            columns:cols
                                           rowBytes:targetRowBytes
                                           dataType:dtype];
  MPSMatrix *matrix = [[MPSMatrix alloc] initWithBuffer:targetBuf
                                             descriptor:desc];

  // Debug Inspection of Matrix Buffer (Verify Input -0.5)
  // Only for [B*S, H] -> likely rows=8, cols=64
  if (rows == 8 && cols == 64) {
    uint16_t *ptr = (uint16_t *)targetBuf.contents;
    fprintf(stderr, "DEBUG Matrix Input: 0x%04x 0x%04x\n", ptr[0], ptr[1]);
  }

  // Wrap Matrix
  return [[MPSGraphTensorData alloc] initWithMPSMatrix:matrix];
}

void Metal_BertLayer_Graph(
    MetalContextRef ctx, MetalBufferRef input, int offIn, MetalBufferRef q,
    int offQ, MetalBufferRef k, int offK, MetalBufferRef v, int offV,
    MetalBufferRef out, int offOut, MetalBufferRef inter, int offInter,
    MetalBufferRef outFFN, int offOutFFN, MetalBufferRef biasQ, int offBiasQ,
    MetalBufferRef biasK, int offBiasK, MetalBufferRef biasV, int offBiasV,
    MetalBufferRef biasOut, int offBiasOut, MetalBufferRef biasInter,
    int offBiasInter, MetalBufferRef biasOutFFN, int offBiasOutFFN,
    MetalBufferRef gammaAttn, int offGammaAttn, MetalBufferRef betaAttn,
    int offBetaAttn, MetalBufferRef gammaFFN, int offGammaFFN,
    MetalBufferRef betaFFN, int offBetaFFN, MetalBufferRef result, int offRes,
    int batchSize, int seqLen, int hiddenSize, int numHeads,
    int intermediateSize, float eps) {
  if (@available(macOS 11.0, *)) {
    MetalWrapper *wrapper = (__bridge MetalWrapper *)ctx;
    int headDim = hiddenSize / numHeads;

    NSString *cacheKey =
        [NSString stringWithFormat:@"bert_%d_%d", batchSize, seqLen];
    BertGraphContext *context = nil; // DISABLE CACHE

    if (!context) {
      context = [[BertGraphContext alloc] init];
      MPSGraph *graph = [[MPSGraph alloc] init];

      // Define Shapes (Rank 2 for Inputs to allow MPSMatrix usage)
      NSArray *shapeInput2D = @[ @(batchSize * seqLen), @(hiddenSize) ];
      NSArray *shape3D = @[ @(batchSize), @(seqLen), @(hiddenSize) ];
      NSArray *shapeWeight = @[ @(hiddenSize), @(hiddenSize) ];
      NSArray *shapeBiasRank2 = @[ @(1), @(hiddenSize) ];            // [1, H]
      NSArray *shapeBiasRank1 = @[ @(hiddenSize) ];                  // [H]
      NSArray *shapeBiasInterRank2 = @[ @(1), @(intermediateSize) ]; // [1, I]
      NSArray *shapeBiasInterRank1 = @[ @(intermediateSize) ];       // [I]

      // Inputs (Rank 2)
      context.tInput = [graph placeholderWithShape:shapeInput2D
                                          dataType:MPSDataTypeFloat16
                                              name:@"input"];
      context.tWq = [graph placeholderWithShape:shapeWeight
                                       dataType:MPSDataTypeFloat16
                                           name:@"Wq"];
      context.tWk = [graph placeholderWithShape:shapeWeight
                                       dataType:MPSDataTypeFloat16
                                           name:@"Wk"];
      context.tWv = [graph placeholderWithShape:shapeWeight
                                       dataType:MPSDataTypeFloat16
                                           name:@"Wv"];
      context.tWout = [graph placeholderWithShape:shapeWeight
                                         dataType:MPSDataTypeFloat16
                                             name:@"Wout"];
      context.tWinter =
          [graph placeholderWithShape:@[ @(hiddenSize), @(intermediateSize) ]
                             dataType:MPSDataTypeFloat16
                                 name:@"Winter"];
      context.tWoutFFN =
          [graph placeholderWithShape:@[ @(intermediateSize), @(hiddenSize) ]
                             dataType:MPSDataTypeFloat16
                                 name:@"WoutFFN"];

      context.tBq = [graph placeholderWithShape:shapeBiasRank2
                                       dataType:MPSDataTypeFloat16
                                           name:@"Bq"];
      context.tBk = [graph placeholderWithShape:shapeBiasRank2
                                       dataType:MPSDataTypeFloat16
                                           name:@"Bk"];
      context.tBv = [graph placeholderWithShape:shapeBiasRank2
                                       dataType:MPSDataTypeFloat16
                                           name:@"Bv"];
      context.tBout = [graph placeholderWithShape:shapeBiasRank2
                                         dataType:MPSDataTypeFloat16
                                             name:@"Bout"];
      context.tBinter = [graph placeholderWithShape:shapeBiasInterRank2
                                           dataType:MPSDataTypeFloat16
                                               name:@"Binter"];
      context.tBoutFFN = [graph placeholderWithShape:shapeBiasRank2
                                            dataType:MPSDataTypeFloat16
                                                name:@"BoutFFN"];

      context.tGammaAttn = [graph placeholderWithShape:shapeBiasRank2
                                              dataType:MPSDataTypeFloat16
                                                  name:@"GammaAttn"];
      context.tBetaAttn = [graph placeholderWithShape:shapeBiasRank2
                                             dataType:MPSDataTypeFloat16
                                                 name:@"BetaAttn"];
      context.tGammaFFN = [graph placeholderWithShape:shapeBiasRank2
                                             dataType:MPSDataTypeFloat16
                                                 name:@"GammaFFN"];
      context.tBetaFFN = [graph placeholderWithShape:shapeBiasRank2
                                            dataType:MPSDataTypeFloat16
                                                name:@"BetaFFN"];

      // --- Construction ---
      // 0. Reshape Input to 3D
      MPSGraphTensor *input3D = [graph reshapeTensor:context.tInput
                                           withShape:shape3D
                                                name:nil];

      // 0b. Reshape Biases/Gamma/Beta to Rank 1
      MPSGraphTensor *bq = [graph reshapeTensor:context.tBq
                                      withShape:shapeBiasRank1
                                           name:nil];
      MPSGraphTensor *bk = [graph reshapeTensor:context.tBk
                                      withShape:shapeBiasRank1
                                           name:nil];
      MPSGraphTensor *bv = [graph reshapeTensor:context.tBv
                                      withShape:shapeBiasRank1
                                           name:nil];
      MPSGraphTensor *bout = [graph reshapeTensor:context.tBout
                                        withShape:shapeBiasRank1
                                             name:nil];
      MPSGraphTensor *binter = [graph reshapeTensor:context.tBinter
                                          withShape:shapeBiasInterRank1
                                               name:nil];
      MPSGraphTensor *boutFFN = [graph reshapeTensor:context.tBoutFFN
                                           withShape:shapeBiasRank1
                                                name:nil];

      MPSGraphTensor *gammaAttn = [graph reshapeTensor:context.tGammaAttn
                                             withShape:shapeBiasRank1
                                                  name:nil];
      MPSGraphTensor *betaAttn = [graph reshapeTensor:context.tBetaAttn
                                            withShape:shapeBiasRank1
                                                 name:nil];
      MPSGraphTensor *gammaFFN = [graph reshapeTensor:context.tGammaFFN
                                            withShape:shapeBiasRank1
                                                 name:nil];
      MPSGraphTensor *betaFFN = [graph reshapeTensor:context.tBetaFFN
                                           withShape:shapeBiasRank1
                                                name:nil];

      // 1. Projections (Use input3D and Reshaped Biases)
      MPSGraphTensor *q =
          [graph matrixMultiplicationWithPrimaryTensor:input3D
                                       secondaryTensor:context.tWq
                                                  name:nil];
      q = [graph additionWithPrimaryTensor:q secondaryTensor:bq name:nil];

      MPSGraphTensor *k =
          [graph matrixMultiplicationWithPrimaryTensor:input3D
                                       secondaryTensor:context.tWk
                                                  name:nil];
      k = [graph additionWithPrimaryTensor:k secondaryTensor:bk name:nil];

      MPSGraphTensor *v =
          [graph matrixMultiplicationWithPrimaryTensor:context.tInput
                                       secondaryTensor:context.tWv
                                                  name:nil];
      v = [graph additionWithPrimaryTensor:v
                           secondaryTensor:context.tBv
                                      name:nil];

      // 2. Reshape [B, S, H] -> [B, S, NumHeads, HeadDim]
      NSArray *shapeHeads =
          @[ @(batchSize), @(seqLen), @(numHeads), @(headDim) ];
      MPSGraphTensor *qH = [graph reshapeTensor:q
                                      withShape:shapeHeads
                                           name:nil];
      MPSGraphTensor *kH = [graph reshapeTensor:k
                                      withShape:shapeHeads
                                           name:nil];
      MPSGraphTensor *vH = [graph reshapeTensor:v
                                      withShape:shapeHeads
                                           name:nil];

      // 3. Transpose -> [B, NumHeads, S, HeadDim]
      qH = [graph transposeTensor:qH dimension:1 withDimension:2 name:nil];
      kH = [graph transposeTensor:kH dimension:1 withDimension:2 name:nil];
      vH = [graph transposeTensor:vH dimension:1 withDimension:2 name:nil];

      // 4. Attention
      MPSGraphTensor *kT = [graph transposeTensor:kH
                                        dimension:2
                                    withDimension:3
                                             name:nil];
      MPSGraphTensor *scores =
          [graph matrixMultiplicationWithPrimaryTensor:qH
                                       secondaryTensor:kT
                                                  name:nil];
      MPSGraphTensor *scale =
          [graph constantWithScalar:(1.0f / sqrtf((float)headDim))
                           dataType:MPSDataTypeFloat16];
      scores = [graph multiplicationWithPrimaryTensor:scores
                                      secondaryTensor:scale
                                                 name:nil];
      scores = [graph softMaxWithTensor:scores axis:3 name:nil];

      MPSGraphTensor *attn = [graph matrixMultiplicationWithPrimaryTensor:scores
                                                          secondaryTensor:vH
                                                                     name:nil];

      // 5. Transpose Back -> [B, S, NumHeads, HeadDim]
      attn = [graph transposeTensor:attn dimension:1 withDimension:2 name:nil];
      attn = [graph reshapeTensor:attn withShape:shape3D name:nil];

      // 6. Output Proj
      // Note: Output Proj weights are [H, H]. Attn is [B, S, H].
      // We can multiply 3D * 2D directly? yes.
      // But we changed Input Placeholders to Rank 2.
      // Do we keep internal heavy ops Rank 3? Yes.
      // So Output Proj is Rank 3.

      MPSGraphTensor *outProj =
          [graph matrixMultiplicationWithPrimaryTensor:attn
                                       secondaryTensor:context.tWout
                                                  name:nil];
      outProj = [graph additionWithPrimaryTensor:outProj
                                 secondaryTensor:context.tBout
                                            name:nil];

      // 7. Residual + Norm 1 (Manual LayerNorm)
      MPSGraphTensor *res1 = [graph additionWithPrimaryTensor:context.tInput
                                              secondaryTensor:outProj
                                                         name:nil];
      MPSGraphTensor *mean1 = [graph meanOfTensor:res1 axes:@[ @(2) ] name:nil];
      MPSGraphTensor *var1 = [graph varianceOfTensor:res1
                                                axes:@[ @(2) ]
                                                name:nil];
      MPSGraphTensor *sub1 = [graph subtractionWithPrimaryTensor:res1
                                                 secondaryTensor:mean1
                                                            name:nil];
      MPSGraphTensor *epsT = [graph constantWithScalar:eps
                                              dataType:MPSDataTypeFloat16];
      MPSGraphTensor *std1 =
          [graph squareRootWithTensor:[graph additionWithPrimaryTensor:var1
                                                       secondaryTensor:epsT
                                                                  name:nil]
                                 name:nil];
      MPSGraphTensor *norm1 = [graph divisionWithPrimaryTensor:sub1
                                               secondaryTensor:std1
                                                          name:nil];
      norm1 = [graph multiplicationWithPrimaryTensor:norm1
                                     secondaryTensor:context.tGammaAttn
                                                name:nil];
      norm1 = [graph additionWithPrimaryTensor:norm1
                               secondaryTensor:context.tBetaAttn
                                          name:nil];

      // 8. FFN
      MPSGraphTensor *ffn1 =
          [graph matrixMultiplicationWithPrimaryTensor:norm1
                                       secondaryTensor:context.tWinter
                                                  name:nil];
      ffn1 = [graph additionWithPrimaryTensor:ffn1
                              secondaryTensor:context.tBinter
                                         name:nil];

      // GELU Approx: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
      MPSGraphTensor *c05 = [graph constantWithScalar:0.5
                                             dataType:MPSDataTypeFloat16];
      MPSGraphTensor *c1 = [graph constantWithScalar:1.0
                                            dataType:MPSDataTypeFloat16];
      MPSGraphTensor *cS =
          [graph constantWithScalar:0.7978845608
                           dataType:MPSDataTypeFloat16]; // sqrt(2/pi)
      MPSGraphTensor *cCoeff = [graph constantWithScalar:0.044715
                                                dataType:MPSDataTypeFloat16];

      MPSGraphTensor *x3 =
          [graph multiplicationWithPrimaryTensor:ffn1
                                 secondaryTensor:
                                     [graph multiplicationWithPrimaryTensor:ffn1
                                                            secondaryTensor:ffn1
                                                                       name:nil]
                                            name:nil];
      MPSGraphTensor *poly = [graph
          additionWithPrimaryTensor:ffn1
                    secondaryTensor:[graph
                                        multiplicationWithPrimaryTensor:x3
                                                        secondaryTensor:cCoeff
                                                                   name:nil]
                               name:nil];
      MPSGraphTensor *arg = [graph multiplicationWithPrimaryTensor:poly
                                                   secondaryTensor:cS
                                                              name:nil];
      MPSGraphTensor *tanh = [graph tanhWithTensor:arg name:nil];
      MPSGraphTensor *gelu = [graph
          multiplicationWithPrimaryTensor:ffn1
                          secondaryTensor:
                              [graph
                                  multiplicationWithPrimaryTensor:c05
                                                  secondaryTensor:
                                                      [graph
                                                          additionWithPrimaryTensor:
                                                              c1
                                                                    secondaryTensor:
                                                                        tanh
                                                                               name:
                                                                                   nil]
                                                             name:nil]
                                     name:nil];

      MPSGraphTensor *ffn2 =
          [graph matrixMultiplicationWithPrimaryTensor:gelu
                                       secondaryTensor:context.tWoutFFN
                                                  name:nil];
      ffn2 = [graph additionWithPrimaryTensor:ffn2
                              secondaryTensor:context.tBoutFFN
                                         name:nil];

      // 9. Residual + Norm 2
      MPSGraphTensor *res2 = [graph additionWithPrimaryTensor:norm1
                                              secondaryTensor:ffn2
                                                         name:nil];
      MPSGraphTensor *mean2 = [graph meanOfTensor:res2 axes:@[ @(2) ] name:nil];
      MPSGraphTensor *var2 = [graph varianceOfTensor:res2
                                                axes:@[ @(2) ]
                                                name:nil];
      MPSGraphTensor *sub2 = [graph subtractionWithPrimaryTensor:res2
                                                 secondaryTensor:mean2
                                                            name:nil];
      MPSGraphTensor *std2 =
          [graph squareRootWithTensor:[graph additionWithPrimaryTensor:var2
                                                       secondaryTensor:epsT
                                                                  name:nil]
                                 name:nil];
      MPSGraphTensor *norm2 = [graph divisionWithPrimaryTensor:sub2
                                               secondaryTensor:std2
                                                          name:nil];
      norm2 = [graph multiplicationWithPrimaryTensor:norm2
                                     secondaryTensor:context.tGammaFFN
                                                name:nil];
      MPSGraphTensor *final = [graph additionWithPrimaryTensor:norm2
                                               secondaryTensor:context.tBetaFFN
                                                          name:nil];
      // Reshape to 2D [B*S, H] to simplify export (and match Rank 2 Input
      // convention)
      final = [graph reshapeTensor:final
                         withShape:@[ @(batchSize * seqLen), @(hiddenSize) ]
                              name:nil];
      context.tFinal = final; // Save Output Tensor (Rank 2)

      context.graph = graph;
      wrapper.bertGraphCache[cacheKey] = context;
    }

    // --- Execution ---
    // Define Shapes for MPSMatrix inputs (Rank 2)
    NSArray *shapeInput2D = @[ @(batchSize * seqLen), @(hiddenSize) ];
    NSArray *shapeWeight = @[ @(hiddenSize), @(hiddenSize) ];
    NSArray *shapeBiasRank2 = @[ @(1), @(hiddenSize) ];

    NSArray *shapeInter = @[ @(hiddenSize), @(intermediateSize) ];
    NSArray *shapeOutFFN = @[ @(intermediateSize), @(hiddenSize) ];
    NSArray *shapeBiasInterRank2 = @[ @(1), @(intermediateSize) ];

    NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
    // Inputs (Rank 2)
    feeds[context.tInput] =
        dataFromBuffer(ctx, input, offIn, shapeInput2D, MPSDataTypeFloat16);

    // Weights (Rank 2)
    feeds[context.tWq] =
        dataFromBuffer(ctx, q, offQ, shapeWeight, MPSDataTypeFloat16);
    feeds[context.tWk] =
        dataFromBuffer(ctx, k, offK, shapeWeight, MPSDataTypeFloat16);
    feeds[context.tWv] =
        dataFromBuffer(ctx, v, offV, shapeWeight, MPSDataTypeFloat16);
    feeds[context.tWout] =
        dataFromBuffer(ctx, out, offOut, shapeWeight, MPSDataTypeFloat16);
    feeds[context.tWinter] =
        dataFromBuffer(ctx, inter, offInter, shapeInter, MPSDataTypeFloat16);
    feeds[context.tWoutFFN] =
        dataFromBuffer(ctx, outFFN, offOutFFN, shapeOutFFN, MPSDataTypeFloat16);

    // Biases (Rank 2)
    feeds[context.tBq] = dataFromBuffer(ctx, biasQ, offBiasQ, shapeBiasRank2,
                                        MPSDataTypeFloat16);
    feeds[context.tBk] = dataFromBuffer(ctx, biasK, offBiasK, shapeBiasRank2,
                                        MPSDataTypeFloat16);
    feeds[context.tBv] = dataFromBuffer(ctx, biasV, offBiasV, shapeBiasRank2,
                                        MPSDataTypeFloat16);
    feeds[context.tBout] = dataFromBuffer(ctx, biasOut, offBiasOut,
                                          shapeBiasRank2, MPSDataTypeFloat16);
    feeds[context.tBinter] = dataFromBuffer(
        ctx, biasInter, offBiasInter, shapeBiasInterRank2, MPSDataTypeFloat16);
    feeds[context.tBoutFFN] = dataFromBuffer(
        ctx, biasOutFFN, offBiasOutFFN, shapeBiasRank2, MPSDataTypeFloat16);

    // Norm Params (Rank 2)
    feeds[context.tGammaAttn] = dataFromBuffer(
        ctx, gammaAttn, offGammaAttn, shapeBiasRank2, MPSDataTypeFloat16);
    feeds[context.tBetaAttn] = dataFromBuffer(
        ctx, betaAttn, offBetaAttn, shapeBiasRank2, MPSDataTypeFloat16);
    feeds[context.tGammaFFN] = dataFromBuffer(
        ctx, gammaFFN, offGammaFFN, shapeBiasRank2, MPSDataTypeFloat16);
    feeds[context.tBetaFFN] = dataFromBuffer(
        ctx, betaFFN, offBetaFFN, shapeBiasRank2, MPSDataTypeFloat16);

    // Check if we can target output buffer directly
    NSMutableDictionary *resultsDict = [NSMutableDictionary dictionary];

    // 1. Create Descriptor for Output
    // Need MPSNDArrayDescriptor. F16=6
    // [8, 64]
    // We assume we can create MPSNDArray from buffer

    // Check selector existence
    SEL runSel = @selector
        (runWithMTLCommandQueue:feeds:targetOperations:resultsDictionary:);
    if ([context.graph respondsToSelector:runSel]) {
      fprintf(stderr, "DEBUG: MPSGraph supports resultsDictionary!\n");

      // 1. Alloc Padded Tmp Buffer (256-byte align rows)
      // Rank 2 [8, 64]. 64 F16 = 128 bytes. Align->256 bytes.
      // Total = 8 * 256 = 2048 bytes.
      NSUInteger rowBytes = hiddenSize * 2;
      NSUInteger alignedRowBytes = (rowBytes + 255) & ~255;
      NSUInteger totalBytes = MAX(seqLen * alignedRowBytes, 2048);

      id<MTLBuffer> tmp =
          [wrapper.device newBufferWithLength:totalBytes
                                      options:MTLResourceStorageModeShared];

      if (tmp) {
        memset(tmp.contents, 0xAA, totalBytes);

        // Use MPSMatrix to enforce Padded Stride (256 bytes)
        MPSMatrixDescriptor *outDesc =
            [MPSMatrixDescriptor matrixDescriptorWithRows:batchSize * seqLen
                                                  columns:hiddenSize
                                                 rowBytes:alignedRowBytes
                                                 dataType:MPSDataTypeFloat16];
        MPSMatrix *outMatrix = [[MPSMatrix alloc] initWithBuffer:tmp
                                                          offset:0
                                                      descriptor:outDesc];

        MPSGraphTensorData *outputData =
            [[MPSGraphTensorData alloc] initWithMPSMatrix:outMatrix];
        // Note: initWithMPSMatrix implicitly sets shape/dtype from matrix
        resultsDict[context.tFinal] = outputData;

        id<MTLCommandBuffer> cb = [wrapper.commandQueue commandBuffer];

        // Force compute of final tensor's operation
        NSArray *ops = @[ context.tFinal.operation ];

        [context.graph runWithMTLCommandQueue:wrapper.commandQueue
                                        feeds:feeds
                             targetOperations:ops
                            resultsDictionary:resultsDict];

        // Blit Padded->Packed
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        for (int i = 0; i < seqLen; i++) {
          [blit copyFromBuffer:tmp
                   sourceOffset:i * rowBytes
                       toBuffer:(__bridge id<MTLBuffer>)result
              destinationOffset:offRes + (i * rowBytes)
                           size:rowBytes];
        }
        [blit endEncoding];

        [cb commit];
        [cb waitUntilCompleted];

        fprintf(stderr, "DEBUG: Graph executed + Blit Repack.\n");

        // Inspect Tmp
        uint16_t *ptr = (uint16_t *)tmp.contents;
        fprintf(stderr, "DEBUG Padded Memory Inspection:\n");
        // Print 32 lines of 8 shorts (16 bytes) -> 512 bytes
        // Total 2048 bytes. Print chunks.
        for (int k = 0; k < 64; k++) {
          // Print offset 0, 128, 256, 512...
          // k * 32 bytes?
          // Let's print Rows from 0 to 7.
          // Row 0 start 0. Row 1 start 256 (Padded) or 128 (Packed).
        }

        for (int r = 0; r < 8; r++) {
          // Print Padded Start and Packed Start candidates
          int offPad = r * 256 / 2;  // shorts
          int offPack = r * 128 / 2; // shorts
          fprintf(stderr, "Row %d: Pad(0x%04x) Pack(0x%04x)\n", r, ptr[offPad],
                  ptr[offPack]);
        }
        fflush(stderr);
      } else {
        fprintf(stderr, "Error: Tmp Alloc Failed\n");
      }
    } else {
      fprintf(stderr, "DEBUG: MPSGraph DOES NOT support resultsDictionary. "
                      "Fallback... failed.\n");
      // Fallback to standard run + broken export?
      // Or try to use standard run and hope it wrote to resultsDict? No.
      id<MTLCommandBuffer> cb = [wrapper.commandQueue commandBuffer];
      NSDictionary *results =
          [context.graph runWithMTLCommandQueue:wrapper.commandQueue
                                          feeds:feeds
                                  targetTensors:@[ context.tFinal ]
                               targetOperations:nil];
      [cb commit];
      [cb waitUntilCompleted];
      // ...
    }

    // Inspect Result Buffer (Not Tmp)
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)result;
    if (buf) {
      uint16_t *ptr = (uint16_t *)buf.contents;
      if (ptr) { // Only if Shared? Result is likely Private/Managed?
                 // If Private, we can't read on CPU.
                 // Assume Private.
                 // We can Blit to a Debug Tmp if needed.
      }
    }
  }
}
