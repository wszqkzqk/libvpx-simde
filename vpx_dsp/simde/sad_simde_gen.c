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

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_ports/mem.h"

#if defined(CONFIG_SIMDE) && CONFIG_SIMDE

#include <simde/x86/sse2.h>
#include <simde/x86/ssse3.h>

// SAD 4xH function using SIMDe
static INLINE unsigned int sad4xh_simde(const uint8_t *src_ptr, int src_stride,
                                        const uint8_t *ref_ptr, int ref_stride,
                                        int height) {
  int i;
  unsigned int sad = 0;
  simde__m128i src_reg, ref_reg, sad_reg;
  simde__m128i sum_sad = simde_mm_setzero_si128();

  for (i = 0; i < height; i++) {
    // Load 4 bytes from each buffer
    src_reg = simde_mm_set_epi32(0, 0, 0, *(int *)src_ptr);
    ref_reg = simde_mm_set_epi32(0, 0, 0, *(int *)ref_ptr);
    sad_reg = simde_mm_sad_epu8(ref_reg, src_reg);
    sum_sad = simde_mm_add_epi32(sum_sad, sad_reg);
    src_ptr += src_stride;
    ref_ptr += ref_stride;
  }

  sad = (unsigned int)(simde_mm_cvtsi128_si32(sum_sad) +
                       simde_mm_cvtsi128_si32(simde_mm_srli_si128(sum_sad, 8)));
  return sad;
}

static INLINE unsigned int sad8xh_simde(const uint8_t *src_ptr, int src_stride,
                                        const uint8_t *ref_ptr, int ref_stride,
                                        int height) {
  int i;
  unsigned int sad = 0;
  simde__m128i src_reg, ref_reg, sad_reg;
  simde__m128i sum_sad = simde_mm_setzero_si128();

  for (i = 0; i < height; i++) {
    src_reg = simde_mm_loadl_epi64((const simde__m128i *)src_ptr);
    ref_reg = simde_mm_loadl_epi64((const simde__m128i *)ref_ptr);
    sad_reg = simde_mm_sad_epu8(ref_reg, src_reg);
    sum_sad = simde_mm_add_epi32(sum_sad, sad_reg);
    src_ptr += src_stride;
    ref_ptr += ref_stride;
  }

  sad = (unsigned int)(simde_mm_cvtsi128_si32(sum_sad) +
                       simde_mm_cvtsi128_si32(simde_mm_srli_si128(sum_sad, 8)));
  return sad;
}

static INLINE unsigned int sad16xh_simde(const uint8_t *src_ptr, int src_stride,
                                         const uint8_t *ref_ptr, int ref_stride,
                                         int height) {
  int i;
  unsigned int sad = 0;
  simde__m128i src1_reg, src2_reg, ref1_reg, ref2_reg;
  simde__m128i sum_sad = simde_mm_setzero_si128();

  for (i = 0; i < height; i++) {
    src1_reg = simde_mm_loadu_si128((const simde__m128i *)src_ptr);
    ref1_reg = simde_mm_loadu_si128((const simde__m128i *)ref_ptr);
    src2_reg = simde_mm_srli_si128(src1_reg, 8);
    ref2_reg = simde_mm_srli_si128(ref1_reg, 8);
    src1_reg = simde_mm_sad_epu8(ref1_reg, src1_reg);
    src2_reg = simde_mm_sad_epu8(ref2_reg, src2_reg);
    sum_sad = simde_mm_add_epi32(sum_sad, simde_mm_add_epi32(src1_reg, src2_reg));
    src_ptr += src_stride;
    ref_ptr += ref_stride;
  }

  sad = (unsigned int)(simde_mm_cvtsi128_si32(sum_sad) +
                       simde_mm_cvtsi128_si32(simde_mm_srli_si128(sum_sad, 8)));
  return sad;
}

// Exported functions implementing SAD for different sizes using SIMDe
unsigned int vpx_sad4x4_simde(const uint8_t *src_ptr, int src_stride,
                                   const uint8_t *ref_ptr, int ref_stride) {
  return sad4xh_simde(src_ptr, src_stride, ref_ptr, ref_stride, 4);
}

unsigned int vpx_sad4x8_simde(const uint8_t *src_ptr, int src_stride,
                                   const uint8_t *ref_ptr, int ref_stride) {
  return sad4xh_simde(src_ptr, src_stride, ref_ptr, ref_stride, 8);
}

unsigned int vpx_sad4x16_simde(const uint8_t *src_ptr, int src_stride,
                                     const uint8_t *ref_ptr, int ref_stride) {
  return sad4xh_simde(src_ptr, src_stride, ref_ptr, ref_stride, 16);
}

unsigned int vpx_sad8x4_simde(const uint8_t *src_ptr, int src_stride,
                                   const uint8_t *ref_ptr, int ref_stride) {
  return sad8xh_simde(src_ptr, src_stride, ref_ptr, ref_stride, 4);
}

unsigned int vpx_sad8x8_simde(const uint8_t *src_ptr, int src_stride,
                                   const uint8_t *ref_ptr, int ref_stride) {
  return sad8xh_simde(src_ptr, src_stride, ref_ptr, ref_stride, 8);
}

unsigned int vpx_sad8x16_simde(const uint8_t *src_ptr, int src_stride,
                                     const uint8_t *ref_ptr, int ref_stride) {
  return sad8xh_simde(src_ptr, src_stride, ref_ptr, ref_stride, 16);
}

unsigned int vpx_sad16x8_simde(const uint8_t *src_ptr, int src_stride,
                                     const uint8_t *ref_ptr, int ref_stride) {
  return sad16xh_simde(src_ptr, src_stride, ref_ptr, ref_stride, 8);
}

unsigned int vpx_sad16x16_simde(const uint8_t *src_ptr, int src_stride,
                                      const uint8_t *ref_ptr, int ref_stride) {
  return sad16xh_simde(src_ptr, src_stride, ref_ptr, ref_stride, 16);
}

unsigned int vpx_sad16x32_simde(const uint8_t *src_ptr, int src_stride,
                                      const uint8_t *ref_ptr, int ref_stride) {
  return sad16xh_simde(src_ptr, src_stride, ref_ptr, ref_stride, 32);
}

#endif  // CONFIG_SIMDE