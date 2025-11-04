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
 * @brief SIMDe-accelerated SAD functions (4d variants)
 *
 * This file provides SAD (Sum of Absolute Differences) implementations using
 * SIMDe library for cross-platform SIMD compatibility. These implementations
 * serve as portable alternatives when native SIMD instructions are unavailable.
 */

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_ports/mem.h"

#if defined(CONFIG_SIMDE) && CONFIG_SIMDE

#include <simde/x86/sse2.h>
#include <simde/x86/ssse3.h>
#include <simde/x86/avx2.h>

static INLINE void sad_calc_4_simde(const uint8_t *src, int src_stride,
                                   const uint8_t *const refs[4], int ref_stride,
                                   int height, uint32_t *sad) {
  int r, i;
  uint32_t sum_sad[4] = {0};

  for (r = 0; r < height; r++) {
    for (i = 0; i < 4; i++) {
      const uint8_t *src_ptr = src + r * src_stride;
      const uint8_t *ref_ptr = refs[i] + r * ref_stride;
      // For 4x4 blocks, we compute the SAD directly using scalar code
      // since SIMD doesn't provide significant benefits for such small blocks
      unsigned int block_sad = 0;
      for (int j = 0; j < 4; j++) {
        for (int k = 0; k < 4; k++) {
          block_sad += abs(src_ptr[j * src_stride + k] - ref_ptr[j * ref_stride + k]);
        }
      }
      sum_sad[i] += block_sad;
    }
  }

  sad[0] = sum_sad[0];
  sad[1] = sum_sad[1];
  sad[2] = sum_sad[2];
  sad[3] = sum_sad[3];
}

static INLINE void sad_calc_8xh_simde(const uint8_t *src, int src_stride,
                                   const uint8_t *const refs[4], int ref_stride,
                                   int height, uint32_t *sad) {
  int r, i;
  uint32_t sum_sad[4] = {0};

  for (r = 0; r < height; r++) {
    for (i = 0; i < 4; i++) {
      const uint8_t *src_ptr = src + r * src_stride;
      const uint8_t *ref_ptr = refs[i] + r * ref_stride;
      // Use SIMDe intrinsics for processing 8xH blocks (load 8 bytes at a time)
      simde__m128i src_reg = simde_mm_loadl_epi64((const simde__m128i *)src_ptr);
      simde__m128i ref_reg = simde_mm_loadl_epi64((const simde__m128i *)ref_ptr);
      simde__m128i sad_reg = simde_mm_sad_epu8(ref_reg, src_reg);
      simde__m128i sum_vec = simde_mm_set_epi32(0, 0, 0, sum_sad[i]);
      sum_vec = simde_mm_add_epi32(sum_vec, sad_reg);
      sum_sad[i] += (uint32_t)(simde_mm_cvtsi128_si32(sum_vec) +
                               simde_mm_cvtsi128_si32(simde_mm_srli_si128(sum_vec, 8)));
    }
  }

  sad[0] = sum_sad[0];
  sad[1] = sum_sad[1];
  sad[2] = sum_sad[2];
  sad[3] = sum_sad[3];
}

static INLINE void sad_calc_16_simde(const uint8_t *src, int src_stride,
                                    const uint8_t *const refs[4], int ref_stride,
                                    int height, uint32_t *sad) {
  int r, i;
  uint32_t sum_sad[4] = {0};

  for (r = 0; r < height; r++) {
    for (i = 0; i < 4; i++) {
      const uint8_t *src_ptr = src + r * src_stride;
      const uint8_t *ref_ptr = refs[i] + r * ref_stride;
      // Process 16 bytes at a time using SIMDe
      simde__m128i src0_reg = simde_mm_loadu_si128((const simde__m128i *)src_ptr);
      simde__m128i ref0_reg = simde_mm_loadu_si128((const simde__m128i *)ref_ptr);
      simde__m128i src1_reg = simde_mm_loadu_si128((const simde__m128i *)(src_ptr + 8));
      simde__m128i ref1_reg = simde_mm_loadu_si128((const simde__m128i *)(ref_ptr + 8));
      
      simde__m128i sad0_reg = simde_mm_sad_epu8(ref0_reg, src0_reg);
      simde__m128i sad1_reg = simde_mm_sad_epu8(ref1_reg, src1_reg);
      simde__m128i sad_total = simde_mm_add_epi32(sad0_reg, sad1_reg);
      
      simde__m128i sum_vec = simde_mm_set_epi32(0, 0, 0, sum_sad[i]);
      sum_vec = simde_mm_add_epi32(sum_vec, sad_total);
      sum_sad[i] += (uint32_t)(simde_mm_cvtsi128_si32(sum_vec) +
                               simde_mm_cvtsi128_si32(simde_mm_srli_si128(sum_vec, 8)));
    }
  }

  sad[0] = sum_sad[0];
  sad[1] = sum_sad[1];
  sad[2] = sum_sad[2];
  sad[3] = sum_sad[3];
}

static INLINE void sad_calc_32_simde(const uint8_t *src, int src_stride,
                                    const uint8_t *const refs[4], int ref_stride,
                                    int height, uint32_t *sad) {
  int r, i;
  uint32_t sum_sad[4] = {0};

  for (r = 0; r < height; r++) {
    for (i = 0; i < 4; i++) {
      const uint8_t *src_ptr = src + r * src_stride;
      const uint8_t *ref_ptr = refs[i] + r * ref_stride;
      // Process 32 bytes at a time using SIMDe (two 16-byte chunks)
      simde__m128i src0_reg = simde_mm_loadu_si128((const simde__m128i *)src_ptr);
      simde__m128i ref0_reg = simde_mm_loadu_si128((const simde__m128i *)ref_ptr);
      simde__m128i src1_reg = simde_mm_loadu_si128((const simde__m128i *)(src_ptr + 16));
      simde__m128i ref1_reg = simde_mm_loadu_si128((const simde__m128i *)(ref_ptr + 16));
      
      simde__m128i sad0_reg = simde_mm_sad_epu8(ref0_reg, src0_reg);
      simde__m128i sad1_reg = simde_mm_sad_epu8(ref1_reg, src1_reg);
      simde__m128i sad_total = simde_mm_add_epi32(sad0_reg, sad1_reg);
      
      simde__m128i sum_vec = simde_mm_set_epi32(0, 0, 0, sum_sad[i]);
      sum_vec = simde_mm_add_epi32(sum_vec, sad_total);
      sum_sad[i] += (uint32_t)(simde_mm_cvtsi128_si32(sum_vec) +
                               simde_mm_cvtsi128_si32(simde_mm_srli_si128(sum_vec, 8)));
    }
  }

  sad[0] = sum_sad[0];
  sad[1] = sum_sad[1];
  sad[2] = sum_sad[2];
  sad[3] = sum_sad[3];
}

static INLINE void sad_calc_64_simde(const uint8_t *src, int src_stride,
                                    const uint8_t *const refs[4], int ref_stride,
                                    int height, uint32_t *sad) {
  int r, i;
  uint32_t sum_sad[4] = {0};

  for (r = 0; r < height; r++) {
    for (i = 0; i < 4; i++) {
      const uint8_t *src_ptr = src + r * src_stride;
      const uint8_t *ref_ptr = refs[i] + r * ref_stride;
      // Process 64 bytes at a time using SIMDe (four 16-byte chunks)
      simde__m128i src0_reg = simde_mm_loadu_si128((const simde__m128i *)src_ptr);
      simde__m128i ref0_reg = simde_mm_loadu_si128((const simde__m128i *)ref_ptr);
      simde__m128i src1_reg = simde_mm_loadu_si128((const simde__m128i *)(src_ptr + 16));
      simde__m128i ref1_reg = simde_mm_loadu_si128((const simde__m128i *)(ref_ptr + 16));
      simde__m128i src2_reg = simde_mm_loadu_si128((const simde__m128i *)(src_ptr + 32));
      simde__m128i ref2_reg = simde_mm_loadu_si128((const simde__m128i *)(ref_ptr + 32));
      simde__m128i src3_reg = simde_mm_loadu_si128((const simde__m128i *)(src_ptr + 48));
      simde__m128i ref3_reg = simde_mm_loadu_si128((const simde__m128i *)(ref_ptr + 48));
      
      simde__m128i sad0_reg = simde_mm_sad_epu8(ref0_reg, src0_reg);
      simde__m128i sad1_reg = simde_mm_sad_epu8(ref1_reg, src1_reg);
      simde__m128i sad2_reg = simde_mm_sad_epu8(ref2_reg, src2_reg);
      simde__m128i sad3_reg = simde_mm_sad_epu8(ref3_reg, src3_reg);
      simde__m128i sad_total = simde_mm_add_epi32(
                                simde_mm_add_epi32(sad0_reg, sad1_reg),
                                simde_mm_add_epi32(sad2_reg, sad3_reg));
      
      simde__m128i sum_vec = simde_mm_set_epi32(0, 0, 0, sum_sad[i]);
      sum_vec = simde_mm_add_epi32(sum_vec, sad_total);
      sum_sad[i] += (uint32_t)(simde_mm_cvtsi128_si32(sum_vec) +
                               simde_mm_cvtsi128_si32(simde_mm_srli_si128(sum_vec, 8)));
    }
  }

  sad[0] = sum_sad[0];
  sad[1] = sum_sad[1];
  sad[2] = sum_sad[2];
  sad[3] = sum_sad[3];
}

// Exported functions
void vpx_sad4x4x4d_simde(const uint8_t *src_ptr, int src_stride,
                              const uint8_t *const ref_array[4], int ref_stride,
                              uint32_t sad_array[4]) {
  sad_calc_4_simde(src_ptr, src_stride, ref_array, ref_stride, 4, sad_array);
}

void vpx_sad4x8x4d_simde(const uint8_t *src_ptr, int src_stride,
                              const uint8_t *const ref_array[4], int ref_stride,
                              uint32_t sad_array[4]) {
  sad_calc_4_simde(src_ptr, src_stride, ref_array, ref_stride, 8, sad_array);
}

void vpx_sad8x4x4d_simde(const uint8_t *src_ptr, int src_stride,
                              const uint8_t *const ref_array[4], int ref_stride,
                              uint32_t sad_array[4]) {
  sad_calc_8xh_simde(src_ptr, src_stride, ref_array, ref_stride, 4, sad_array);
}

void vpx_sad8x8x4d_simde(const uint8_t *src_ptr, int src_stride,
                              const uint8_t *const ref_array[4], int ref_stride,
                              uint32_t sad_array[4]) {
  sad_calc_8xh_simde(src_ptr, src_stride, ref_array, ref_stride, 8, sad_array);
}

void vpx_sad8x16x4d_simde(const uint8_t *src_ptr, int src_stride,
                               const uint8_t *const ref_array[4], int ref_stride,
                               uint32_t sad_array[4]) {
  sad_calc_8xh_simde(src_ptr, src_stride, ref_array, ref_stride, 16, sad_array);
}

void vpx_sad16x16x4d_simde(const uint8_t *src_ptr, int src_stride,
                                const uint8_t *const ref_array[4], int ref_stride,
                                uint32_t sad_array[4]) {
  sad_calc_16_simde(src_ptr, src_stride, ref_array, ref_stride, 16, sad_array);
}

void vpx_sad16x8x4d_simde(const uint8_t *src_ptr, int src_stride,
                               const uint8_t *const ref_array[4], int ref_stride,
                               uint32_t sad_array[4]) {
  sad_calc_16_simde(src_ptr, src_stride, ref_array, ref_stride, 8, sad_array);
}

void vpx_sad16x32x4d_simde(const uint8_t *src_ptr, int src_stride,
                                const uint8_t *const ref_array[4], int ref_stride,
                                uint32_t sad_array[4]) {
  sad_calc_16_simde(src_ptr, src_stride, ref_array, ref_stride, 32, sad_array);
}

void vpx_sad32x32x4d_simde(const uint8_t *src_ptr, int src_stride,
                                const uint8_t *const ref_array[4], int ref_stride,
                                uint32_t sad_array[4]) {
  sad_calc_32_simde(src_ptr, src_stride, ref_array, ref_stride, 32, sad_array);
}

void vpx_sad32x16x4d_simde(const uint8_t *src_ptr, int src_stride,
                                const uint8_t *const ref_array[4], int ref_stride,
                                uint32_t sad_array[4]) {
  sad_calc_32_simde(src_ptr, src_stride, ref_array, ref_stride, 16, sad_array);
}

void vpx_sad32x64x4d_simde(const uint8_t *src_ptr, int src_stride,
                                const uint8_t *const ref_array[4], int ref_stride,
                                uint32_t sad_array[4]) {
  sad_calc_32_simde(src_ptr, src_stride, ref_array, ref_stride, 64, sad_array);
}

void vpx_sad64x64x4d_simde(const uint8_t *src_ptr, int src_stride,
                                const uint8_t *const ref_array[4], int ref_stride,
                                uint32_t sad_array[4]) {
  sad_calc_64_simde(src_ptr, src_stride, ref_array, ref_stride, 64, sad_array);
}

void vpx_sad64x32x4d_simde(const uint8_t *src_ptr, int src_stride,
                                const uint8_t *const ref_array[4], int ref_stride,
                                uint32_t sad_array[4]) {
  sad_calc_64_simde(src_ptr, src_stride, ref_array, ref_stride, 32, sad_array);
}

#endif  // CONFIG_SIMDE