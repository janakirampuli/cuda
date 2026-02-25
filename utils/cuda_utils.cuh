#pragma once

// Common utilities for the CUDA samples in this repo.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>

namespace cuda_utils {

// -----------------------------
// Error checking
// -----------------------------

inline void check_cuda(cudaError_t result, const char* expr, const char* file, int line) {
    if (result != cudaSuccess) {
        std::fprintf(stderr,
                     "CUDA error: %s\n  expr: %s\n  file: %s:%d\n",
                     cudaGetErrorString(result), expr, file, line);
        std::fflush(stderr);
        std::abort();
    }
}

} // namespace cuda_utils

#define CUDA_CHECK(expr) ::cuda_utils::check_cuda((expr), #expr, __FILE__, __LINE__)

namespace cuda_utils {

// -----------------------------
// Math helpers
// -----------------------------

template <typename T>
constexpr T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

// -----------------------------
// Timing (CPU wall-clock)
// -----------------------------

// Seconds since an arbitrary epoch; monotonic where supported.
inline double time_sec() {
    timespec ts{};
#if defined(CLOCK_MONOTONIC)
    clock_gettime(CLOCK_MONOTONIC, &ts);
#else
    clock_gettime(CLOCK_REALTIME, &ts);
#endif
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

// -----------------------------
// Validation helpers
// -----------------------------

// -----------------------------
// Random init helpers
// -----------------------------

inline void init_vector(float* vec, int n) {
    for (int i = 0; i < n; ++i) {
        vec[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    }
}

inline void init_matrix(float* mat, int rows, int cols) {
    const int n = rows * cols;
    for (int i = 0; i < n; ++i) {
        mat[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    }
}

inline bool allclose_f32(const float* ref,
                         const float* got,
                         std::size_t n,
                         float atol = 1e-4f,
                         float rtol = 0.0f,
                         std::size_t* first_bad_index = nullptr) {
    for (std::size_t i = 0; i < n; ++i) {
        const float a = ref[i];
        const float b = got[i];
        const float diff = std::fabs(a - b);
        const float tol = atol + rtol * std::fabs(a);
        if (diff > tol || std::isnan(diff)) {
            if (first_bad_index) *first_bad_index = i;
            return false;
        }
    }
    return true;
}

} // namespace cuda_utils
