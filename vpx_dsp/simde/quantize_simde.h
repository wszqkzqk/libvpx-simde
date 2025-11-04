/*
 *  Copyright (c) 2025 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_SIMDE_QUANTIZE_SIMDE_H_
#define VPX_VPX_DSP_SIMDE_QUANTIZE_SIMDE_H_

#include "vpx_dsp/simde/simde_headers.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/vpx_dsp_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// SIMDE implementations of quantization-related functions

// 128-bit operations
static INLINE simde__m128i load_tran_low_simde(const tran_low_t *input) {
    // Load 8 tran_low_t values (which are typically int16_t) 
    // into a SIMD register using SIMDE
    const int16_t *src = (const int16_t *)input;
    return simde_mm_loadu_si128((const simde__m128i *)src);
}

static INLINE simde__m128i load_tran_low_simde_aligned(const tran_low_t *input) {
    const int16_t *src = (const int16_t *)input;
    return simde_mm_load_si128((const simde__m128i *)src);
}

static INLINE void store_tran_low_simde(simde__m128i values, tran_low_t *output) {
    // Store 8 tran_low_t values from a SIMD register using SIMDE
    int16_t *dst = (int16_t *)output;
    simde_mm_storeu_si128((simde__m128i *)dst, values);
}

static INLINE void store_tran_low_simde_aligned(simde__m128i values, tran_low_t *output) {
    int16_t *dst = (int16_t *)output;
    simde_mm_store_si128((simde__m128i *)dst, values);
}

static INLINE void store_zero_tran_low_simde(tran_low_t *output) {
    simde__m128i zero = simde_mm_setzero_si128();
    store_tran_low_simde(zero, output);
}

// 256-bit operations (for AVX2)
static INLINE simde__m256i load_tran_low_avx2_simde(const tran_low_t *input) {
    // Load 16 tran_low_t values (which are typically int16_t) 
    // into a SIMD register using SIMDE AVX2
    const int16_t *src = (const int16_t *)input;
    return simde_mm256_loadu_si256((const simde__m256i *)src);
}

static INLINE simde__m256i load_tran_low_avx2_simde_aligned(const tran_low_t *input) {
    const int16_t *src = (const int16_t *)input;
    return simde_mm256_load_si256((const simde__m256i *)src);
}

static INLINE void store_tran_low_avx2_simde(simde__m256i values, tran_low_t *output) {
    // Store 16 tran_low_t values from an AVX2 register using SIMDE
    int16_t *dst = (int16_t *)output;
    simde_mm256_storeu_si256((simde__m256i *)dst, values);
}

static INLINE void store_zero_avx2_tran_low_simde(tran_low_t *output) {
    simde__m256i zero = simde_mm256_setzero_si256();
    store_tran_low_avx2_simde(zero, output);
}

// The function pointers should be specified by the calling context rather than globally defined
// Different architectures (SSE2 vs AVX2) expect different register types (128-bit vs 256-bit)
// The calling code should use the appropriate function variant for its context

// Helper functions for loading int16 values
static INLINE simde__m128i load_simde(const int16_t *input) {
    return simde_mm_load_si128((const simde__m128i *)input);
}

static INLINE void store_simde(int16_t *output, simde__m128i values) {
    simde_mm_store_si128((simde__m128i *)output, values);
}

static INLINE void storeu_simde(int16_t *output, simde__m128i values) {
    simde_mm_storeu_si128((simde__m128i *)output, values);
}

// Additional utility functions for transforms
static INLINE void load_tran_low8_simde(const tran_low_t *input, simde__m128i *out) {
    const int16_t *src = (const int16_t *)input;
    *out = simde_mm_load_si128((const simde__m128i *)src);
}

static INLINE void loadu_tran_low8_simde(const tran_low_t *input, simde__m128i *out) {
    const int16_t *src = (const int16_t *)input;
    *out = simde_mm_loadu_si128((const simde__m128i *)src);
}

static INLINE void store_tran_low8_simde(simde__m128i values, tran_low_t *output) {
    int16_t *dst = (int16_t *)output;
    simde_mm_store_si128((simde__m128i *)dst, values);
}

static INLINE void storeu_tran_low8_simde(simde__m128i values, tran_low_t *output) {
    int16_t *dst = (int16_t *)output;
    simde_mm_storeu_si128((simde__m128i *)dst, values);
}

// Additional functions required for VP9 quantize
static INLINE simde__m128i scan_for_eob_simde(simde__m128i *qcoeff0, simde__m128i *qcoeff1,
                                   const int16_t *scan, const int index,
                                   const simde__m128i zero) {
  const simde__m128i zero_coeff0 = simde_mm_cmpeq_epi16(*qcoeff0, zero);
  const simde__m128i zero_coeff1 = simde_mm_cmpeq_epi16(*qcoeff1, zero);
  const simde__m128i scan0 = simde_mm_loadu_si128((const simde__m128i *)(scan + index));
  const simde__m128i scan1 = simde_mm_loadu_si128((const simde__m128i *)(scan + index + 8));
  simde__m128i eob0, eob1;
  eob0 = simde_mm_andnot_si128(zero_coeff0, scan0);
  eob1 = simde_mm_andnot_si128(zero_coeff1, scan1);
  return simde_mm_max_epi16(eob0, eob1);
}

static INLINE int16_t accumulate_eob_simde(simde__m128i eob) {
  simde__m128i eob_shuffled;
  eob_shuffled = simde_mm_srli_si128(eob, 8);
  eob = simde_mm_max_epi16(eob, eob_shuffled);
  eob_shuffled = simde_mm_srli_si128(eob, 4);
  eob = simde_mm_max_epi16(eob, eob_shuffled);
  eob_shuffled = simde_mm_srli_si128(eob, 2);
  eob = simde_mm_max_epi16(eob, eob_shuffled);
  return (int16_t)simde_mm_cvtsi128_si32(eob);
}


// Macro for function selection - don't define globally as generic load_tran_low
// Use specific function names depending on the context

#ifdef __cplusplus
}  // extern "C"
#endif


// Additional helper function
#endif  // VPX_VPX_DSP_SIMDE_QUANTIZE_SIMDE_H_