#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Metal error: no accessible GPU device\n";
            return 1;
        }
        std::cout << "Metal device: " << device.name.UTF8String << std::endl;
        NSString *source = @"#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "kernel void probe(device int *values [[buffer(0)]], uint i [[thread_position_in_grid]]) { values[i] = values[i] * values[i] - 1; }\n";
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library) {
            std::cerr << "Metal compilation error: " << error.localizedDescription.UTF8String << '\n';
            return 1;
        }
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"probe"] error:&error];
        if (!pipeline) {
            std::cerr << "Metal pipeline error: " << error.localizedDescription.UTF8String << '\n';
            return 1;
        }
        int input[] = {-1, 0, 1, 3};
        id<MTLBuffer> buffer = [device newBufferWithBytes:input length:sizeof(input) options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!buffer || !queue || !command || !encoder) {
            std::cerr << "Metal error: failed to allocate dispatch resources\n";
            return 1;
        }
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffer offset:0 atIndex:0];
        [encoder dispatchThreads:MTLSizeMake(4, 1, 1) threadsPerThreadgroup:MTLSizeMake(4, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            std::cerr << "Metal execution error: " << command.error.localizedDescription.UTF8String << '\n';
            return 1;
        }
        const int *result = static_cast<const int *>(buffer.contents);
        for (int i = 0; i < 4; ++i) {
            if (result[i] != input[i] * input[i] - 1) {
                std::cerr << "Metal arithmetic mismatch at " << i << '\n';
                return 1;
            }
        }
        std::cout << "PASS: four integer arithmetic results from a completed Metal dispatch\n";
        return 0;
    }
}
