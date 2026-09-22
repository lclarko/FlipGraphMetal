#pragma once
#include <cstddef>
#include <initializer_list>
#include <type_traits>
#include <new>
#include <stdexcept>
#include <vector>

void *metalAllocateBytes(size_t size);
void metalFree(void *pointer);
struct MetalArgument {
    const void *pointer;
    size_t size;
    bool buffer;
};
void metalLaunch(const char *name, size_t threads, size_t blockSize, std::initializer_list<MetalArgument> arguments);
template <class T> void metalAllocate(T **pointer, size_t size) {
    static_assert(std::is_trivially_copyable<T>::value && std::is_trivially_destructible<T>::value);
    if (size % sizeof(T)) throw std::runtime_error("Metal: invalid allocation element count");
    *pointer = static_cast<T *>(metalAllocateBytes(size));
    for (size_t i = 0; i < size / sizeof(T); i++) new (*pointer + i) T();
}
template <class T> MetalArgument metalArgument(T *pointer) {
    return {pointer, 0, true};
}
template <class T> MetalArgument metalArgument(const T &value) {
    return {&value, sizeof(T), false};
}
template <class... Args> void metalDispatch(const char *name, size_t threads, size_t blockSize, const Args &...args) {
    metalLaunch(name, threads, blockSize, {metalArgument(args)...});
}
