/*
 *  Copyright (c) 2025 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

/**
 * @file
 * @brief SIMDe-accelerated SAD functions
 *
 * This file provides SAD (Sum of Absolute Differences) implementations using
 * SIMDe library for cross-platform SIMD compatibility. These implementations
 * serve as portable alternatives when native SIMD instructions are unavailable.
 */

#ifndef VPX_VPX_DSP_SIMDE_SAD_SIMDE_H_
#define VPX_VPX_DSP_SIMDE_SAD_SIMDE_H_

#include "vpx_config.h"
#include "vpx/vpx_integer.h"

#if defined(CONFIG_SIMDE) && CONFIG_SIMDE
#include <simde/x86/avx2.h>
#include <simde/x86/sse2.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Declare the SIMDe-based SAD functions that mirror the existing x86 implementations

unsigned int vpx_sad64x64_simde(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride);

unsigned int vpx_sad64x32_simde(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride);

unsigned int vpx_sad32x64_simde(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride);

unsigned int vpx_sad32x32_simde(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride);

unsigned int vpx_sad32x16_simde(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride);

unsigned int vpx_sad16x32_simde(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride);

unsigned int vpx_sad16x16_simde(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride);

unsigned int vpx_sad16x8_simde(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride);

unsigned int vpx_sad8x16_simde(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride);

unsigned int vpx_sad8x8_simde(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride);

unsigned int vpx_sad8x4_simde(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride);

unsigned int vpx_sad4x8_simde(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride);

unsigned int vpx_sad4x4_simde(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride);

// SAD AVG functions using SIMDe
unsigned int vpx_sad64x64_avg_simde_avx2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride,
    const uint8_t *second_pred);

unsigned int vpx_sad64x32_avg_simde_avx2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride,
    const uint8_t *second_pred);

unsigned int vpx_sad32x64_avg_simde_avx2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride,
    const uint8_t *second_pred);

unsigned int vpx_sad32x32_avg_simde_avx2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride,
    const uint8_t *second_pred);

unsigned int vpx_sad32x16_avg_simde_avx2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride,
    const uint8_t *second_pred);

unsigned int vpx_sad16x32_avg_simde_sse2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride,
    const uint8_t *second_pred);

unsigned int vpx_sad16x16_avg_simde_sse2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride,
    const uint8_t *second_pred);

unsigned int vpx_sad16x8_avg_simde_sse2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride,
    const uint8_t *second_pred);

unsigned int vpx_sad8x16_avg_simde_sse2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride,
    const uint8_t *second_pred);

unsigned int vpx_sad8x8_avg_simde_sse2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride,
    const uint8_t *second_pred);

unsigned int vpx_sad8x4_avg_simde_sse2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride,
    const uint8_t *second_pred);

unsigned int vpx_sad4x8_avg_simde_sse2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride,
    const uint8_t *second_pred);

unsigned int vpx_sad4x4_avg_simde_sse2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride,
    const uint8_t *second_pred);

// SAD skip functions using SIMDe
unsigned int vpx_sad_skip_64x64_simde_avx2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride);

unsigned int vpx_sad_skip_64x32_simde_avx2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride);

unsigned int vpx_sad_skip_32x64_simde_avx2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride);

unsigned int vpx_sad_skip_32x32_simde_avx2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride);

unsigned int vpx_sad_skip_32x16_simde_avx2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride);

unsigned int vpx_sad_skip_16x32_simde_sse2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride);

unsigned int vpx_sad_skip_16x16_simde_sse2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride);

unsigned int vpx_sad_skip_16x8_simde_sse2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride);

unsigned int vpx_sad_skip_8x16_simde_sse2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride);

unsigned int vpx_sad_skip_8x8_simde_sse2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride);

unsigned int vpx_sad_skip_8x4_simde_sse2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride);

unsigned int vpx_sad_skip_4x8_simde_sse2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride);

unsigned int vpx_sad_skip_4x4_simde_sse2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride);

// SAD 4D functions using SIMDe
void vpx_sad64x64x4d_simde(const uint8_t *src_ptr, int src_stride,
                           const uint8_t *const ref_array[4], int ref_stride,
                           uint32_t sad_array[4]);

void vpx_sad64x32x4d_simde(const uint8_t *src_ptr, int src_stride,
                           const uint8_t *const ref_array[4], int ref_stride,
                           uint32_t sad_array[4]);

void vpx_sad32x64x4d_simde(const uint8_t *src_ptr, int src_stride,
                           const uint8_t *const ref_array[4], int ref_stride,
                           uint32_t sad_array[4]);

void vpx_sad32x32x4d_simde(const uint8_t *src_ptr, int src_stride,
                           const uint8_t *const ref_array[4], int ref_stride,
                           uint32_t sad_array[4]);

void vpx_sad32x16x4d_simde(const uint8_t *src_ptr, int src_stride,
                           const uint8_t *const ref_array[4], int ref_stride,
                           uint32_t sad_array[4]);

void vpx_sad16x32x4d_simde(const uint8_t *src_ptr, int src_stride,
                           const uint8_t *const ref_array[4], int ref_stride,
                           uint32_t sad_array[4]);

void vpx_sad16x16x4d_simde(const uint8_t *src_ptr, int src_stride,
                           const uint8_t *const ref_array[4], int ref_stride,
                           uint32_t sad_array[4]);

void vpx_sad16x8x4d_simde(const uint8_t *src_ptr, int src_stride,
                          const uint8_t *const ref_array[4], int ref_stride,
                          uint32_t sad_array[4]);

void vpx_sad8x16x4d_simde(const uint8_t *src_ptr, int src_stride,
                          const uint8_t *const ref_array[4], int ref_stride,
                          uint32_t sad_array[4]);

void vpx_sad8x8x4d_simde(const uint8_t *src_ptr, int src_stride,
                         const uint8_t *const ref_array[4], int ref_stride,
                         uint32_t sad_array[4]);

void vpx_sad8x4x4d_simde(const uint8_t *src_ptr, int src_stride,
                         const uint8_t *const ref_array[4], int ref_stride,
                         uint32_t sad_array[4]);

void vpx_sad4x8x4d_simde(const uint8_t *src_ptr, int src_stride,
                         const uint8_t *const ref_array[4], int ref_stride,
                         uint32_t sad_array[4]);

void vpx_sad4x4x4d_simde(const uint8_t *src_ptr, int src_stride,
                         const uint8_t *const ref_array[4], int ref_stride,
                         uint32_t sad_array[4]);

// For high bitdepth functions
#if CONFIG_VP9_HIGHBITDEPTH
unsigned int vpx_highbd_sad64x64_simde_avx2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride);

unsigned int vpx_highbd_sad64x32_simde_avx2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride);

unsigned int vpx_highbd_sad32x64_simde_avx2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride);

unsigned int vpx_highbd_sad32x32_simde_avx2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride);

unsigned int vpx_highbd_sad32x16_simde_avx2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride);

unsigned int vpx_highbd_sad16x32_simde_avx2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride);

unsigned int vpx_highbd_sad16x16_simde_avx2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride);

unsigned int vpx_highbd_sad16x8_simde_avx2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride);

unsigned int vpx_highbd_sad8x16_simde_avx2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride);

unsigned int vpx_highbd_sad8x8_simde_avx2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride);

unsigned int vpx_highbd_sad8x4_simde_avx2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride);

unsigned int vpx_highbd_sad4x8_simde_avx2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride);

unsigned int vpx_highbd_sad4x4_simde_avx2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride);

// High bitdepth AVG functions
unsigned int vpx_highbd_sad64x64_avg_simde_avx2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride,
    const uint8_t *second_pred);

unsigned int vpx_highbd_sad32x32_avg_simde_avx2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride,
    const uint8_t *second_pred);

unsigned int vpx_highbd_sad16x16_avg_simde_avx2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride,
    const uint8_t *second_pred);

unsigned int vpx_highbd_sad8x8_avg_simde_avx2(
    const uint8_t *src_ptr, int src_stride, 
    const uint8_t *ref_ptr, int ref_stride,
    const uint8_t *second_pred);

#endif // CONFIG_VP9_HIGHBITDEPTH

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VPX_DSP_X86_SAD_SIMDE_H_