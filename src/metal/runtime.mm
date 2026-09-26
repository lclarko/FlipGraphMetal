#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "runtime.h"
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <string>
#include <vector>
#include <limits>
#include <mach-o/dyld.h>
#include <CommonCrypto/CommonDigest.h>

#if defined(METAL_SOURCE_DIR) && defined(METAL_LIBRARY_NAME)
#error Select either source shaders or a packaged library, not both
#elif !defined(METAL_SOURCE_DIR) && (!defined(METAL_LIBRARY_NAME) || !defined(METAL_LIBRARY_SHA256))
#error Build with a generated Metal library header or explicitly set METAL_SOURCE_DIR
#endif

#ifndef METAL_SOURCE_DIR
static id<MTLLibrary> loadPackagedLibrary(id<MTLDevice> device) {
    uint32_t length = 0;
    _NSGetExecutablePath(nullptr, &length);
    std::vector<char> executable(length);
    if (!length || _NSGetExecutablePath(executable.data(), &length) != 0)
        throw std::runtime_error("Metal: cannot locate executable for shader loading");
    NSString *directory = [[[NSString stringWithUTF8String:executable.data()]
                           stringByResolvingSymlinksInPath] stringByDeletingLastPathComponent];
    NSString *path = [directory stringByAppendingPathComponent:@METAL_LIBRARY_NAME];
    NSError *error = nil;
    NSData *bytes = [NSData dataWithContentsOfFile:path options:0 error:&error];
    if (!bytes)
        throw std::runtime_error(std::string("Metal: cannot read packaged shader library ") +
                                 path.UTF8String + "; keep the matching shaders directory with the executable");
    if (!bytes.length || bytes.length > std::numeric_limits<CC_LONG>::max())
        throw std::runtime_error("Metal: invalid packaged shader library size");
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(bytes.bytes, static_cast<CC_LONG>(bytes.length), digest);
    const char hex[] = "0123456789abcdef";
    std::string checksum;
    for (unsigned char byte : digest) {
        checksum += hex[byte >> 4];
        checksum += hex[byte & 15];
    }
    if (checksum != METAL_LIBRARY_SHA256)
        throw std::runtime_error("Metal: packaged shader library does not match this executable; rebuild or reinstall the complete package");
    // Load the verified bytes, avoiding another path read after the hash check.
    dispatch_data_t data = dispatch_data_create(bytes.bytes, bytes.length, nullptr,
                                                DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    if (!data) throw std::runtime_error("Metal: shader library allocation failed");
    id<MTLLibrary> library = [device newLibraryWithData:data error:&error];
    if (!library)
        throw std::runtime_error(std::string("Metal: cannot load packaged shader library: ") +
                                 (error ? error.localizedDescription.UTF8String : "unknown error"));
    std::cout << "Metal library: " << METAL_LIBRARY_NAME << " SHA256 " << checksum << std::endl;
    return library;
}
#endif

class MetalRuntime {
public:
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    id<MTLBuffer> compactBuffers[3] = {};
    std::unordered_map<const void *, id<MTLBuffer>> buffers;
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;

    MetalRuntime() {
        device = MTLCreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("Metal: no accessible GPU device");
        queue = [device newCommandQueue];
        if (!queue) throw std::runtime_error("Metal: command queue allocation failed");
        std::cout << "Metal device: " << device.name.UTF8String << std::endl;
#ifdef METAL_SOURCE_DIR
        NSMutableString *source = [NSMutableString string];
#ifdef METAL_F2
        [source appendString:@"#define METAL_F2\n"];
#endif
#ifdef METAL_TESTING
        [source appendString:@"#define METAL_TESTING\n"];
#endif
        for (NSString *name in @[@"core.h", @"addition.h", @"flip_set.h", @"scheme_integer.h", @"scheme_z2.h", @"pairs_counter.h", @"additions_reducer.h",
#ifndef METAL_F2
            @"compact.h",
#endif
            @"controlled.h", @"controlled_capture.h", @"controlled_packed.h", @"kernels.metal", @"controlled_kernels.metal", @"reduction_kernels.metal"
#ifdef METAL_TESTING
            , @"test_kernels.metal"
#endif
        ]) {
            NSString *path = [@METAL_SOURCE_DIR stringByAppendingPathComponent:name];
            NSError *error = nil;
            NSString *part = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error];
            if (!part) throw std::runtime_error(error.localizedDescription.UTF8String);
            [source appendString:[part stringByReplacingOccurrencesOfString:@"#pragma once" withString:@""]];
            [source appendString:@"\n"];
        }
        NSError *error = nil;
        MTLCompileOptions *options = [MTLCompileOptions new];
        options.mathMode = MTLMathModeSafe;
        options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        library = [device newLibraryWithSource:source options:options error:&error];
        if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
        std::cout << "Metal library: runtime source" << std::endl;
#else
        library = loadPackagedLibrary(device);
#endif
    }
};

static MetalRuntime &runtime() {
    static MetalRuntime instance;
    return instance;
}

void *metalAllocateBytes(size_t size) {
    auto &r = runtime();
    if (!size || size > r.device.maxBufferLength) throw std::runtime_error("Metal: unsupported buffer size");
    id<MTLBuffer> buffer = [r.device newBufferWithLength:size options:MTLResourceStorageModeShared];
    if (!buffer) throw std::runtime_error("Metal: buffer allocation failed");
    std::memset(buffer.contents, 0, size);
    r.buffers[buffer.contents] = buffer;
    return buffer.contents;
}

void metalFree(void *pointer) {
    if (pointer && runtime().buffers.erase(pointer) != 1) throw std::runtime_error("Metal: unknown allocation");
}

void metalLaunch(const char *name, size_t threads, size_t blockSize, std::initializer_list<MetalArgument> arguments) {
    @autoreleasepool {
        auto &r = runtime();
        auto &pipeline = r.pipelines[name];
        if (!pipeline) {
            NSError *error = nil;
            id<MTLFunction> function = [r.library newFunctionWithName:[NSString stringWithUTF8String:name]];
            if (!function) throw std::runtime_error(std::string("Metal: unknown kernel ") + name);
            pipeline = [r.device newComputePipelineStateWithFunction:function error:&error];
            if (!pipeline) throw std::runtime_error(error.localizedDescription.UTF8String);
        }
        if (!threads || !blockSize || blockSize > pipeline.maxTotalThreadsPerThreadgroup)
            throw std::runtime_error("Metal: unsupported --block-size or dispatch size");
        id<MTLCommandBuffer> command = [r.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) throw std::runtime_error("Metal: dispatch allocation failed");
        command.label = [NSString stringWithUTF8String:name];
        [encoder setComputePipelineState:pipeline];
        id<MTLBuffer> errors = [r.device newBufferWithLength:threads * sizeof(int) options:MTLResourceStorageModeShared];
        if (!errors) throw std::runtime_error("Metal: validation buffer allocation failed");
        std::memset(errors.contents, 0, errors.length);
        [encoder setBuffer:errors offset:0 atIndex:30];
        size_t index = 0;
        for (const auto &arg : arguments) {
            if (arg.buffer) {
                auto it = r.buffers.find(arg.pointer);
                if (it == r.buffers.end()) throw std::runtime_error("Metal: kernel argument is not a GPU buffer");
                [encoder setBuffer:it->second offset:0 atIndex:index];
            } else {
                [encoder setBytes:arg.pointer length:arg.size atIndex:index];
            }
            index++;
        }
        if (std::strcmp(name, "randomWalkCompactKernel") == 0) {
            if (blockSize != 32 || index != 13)
                throw std::runtime_error("Metal: unsupported compact walk dispatch");
            const size_t bytesPerWorker[3] = {6000, 4200, 6000};
            size_t maxWorkers = r.device.maxBufferLength / 6000; // largest compact per-worker buffer
            if (threads > maxWorkers - maxWorkers % 32)
                throw std::runtime_error("Metal: compact storage buffer size exceeds device limit");
            size_t workers = (threads + 31) / 32 * 32;
            for (size_t buffer = 0; buffer < 3; buffer++) {
                size_t length = workers * bytesPerWorker[buffer];
                if (!r.compactBuffers[buffer] || r.compactBuffers[buffer].length < length)
                    r.compactBuffers[buffer] = [r.device newBufferWithLength:length options:MTLResourceStorageModeShared];
                if (!r.compactBuffers[buffer])
                    throw std::runtime_error("Metal: compact storage buffer allocation failed");
                [encoder setBuffer:r.compactBuffers[buffer] offset:0 atIndex:13 + buffer];
            }
        }
        [encoder dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(blockSize, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted)
            throw std::runtime_error(command.error.localizedDescription.UTF8String);
        const int *status = static_cast<const int *>(errors.contents);
        for (size_t i = 0; i < threads; i++)
            if (status[i]) {
                std::string reason = status[i] == 2 ? "reducer capacity or expression error" : "tensor validation failed";
                if (std::strcmp(name, "randomWalkCompactKernel") == 0 && status[i] == 2)
                    reason = "unsupported compact walk configuration";
                if (std::strcmp(name, "randomWalkCompactKernel") == 0 && status[i] == 3)
                    reason = "unsupported compact scheme representation";
                if (status[i] == 4) reason = "flip candidate capacity exceeded (500 pairs per factor)";
                throw std::runtime_error("Metal " + reason + " in " + name + " at worker " + std::to_string(i));
            }
        std::cout << "Metal dispatch " << name << ": " << threads << " threads, "
                  << (command.GPUEndTime - command.GPUStartTime) * 1000 << " ms GPU" << std::endl;
    }
}
