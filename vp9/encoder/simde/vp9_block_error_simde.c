/*
 *  Copyright (c) 2025 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "vpx_dsp/simde/simde_headers.h"
#include "./vp9_rtcd.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/vpx_dsp_common.h"

int64_t vp9_block_error_simde(const tran_low_t *coeff, const tran_low_t *dqcoeff,
                          intptr_t block_size, int64_t *ssz) {
#if defined(__x86_64__) || defined(_M_X64)
  // SSE2 implementation
  simde__m128i sse_acc = simde_mm_setzero_si128();      // sse accumulator
  simde__m128i ssz_acc = simde_mm_setzero_si128();      // ssz accumulator
  simde__m128i zero = simde_mm_setzero_si128();         // dedicated zero register
  simde__m128i uq, dq, uq2, dq2;
  intptr_t size = block_size;

  while (size > 0) {
    // Load 16 tran_low_t values (each tran_low_t is typically int16_t, so 8 elements per register)
    if (size >= 8) {
      uq = load_tran_low(coeff);
      dq = load_tran_low(dqcoeff);
      coeff += 8;
      dqcoeff += 8;
      size -= 8;
    } else {
      // Handle remaining elements
      uq = simde_mm_setzero_si128();
      dq = simde_mm_setzero_si128();
      for (int i = 0; i < size; i++) {
        ((int16_t*)&uq)[i] = coeff[i];
        ((int16_t*)&dq)[i] = dqcoeff[i];
      }
      coeff += size;
      dqcoeff += size;
      size = 0;
    }

    // Convert to int16_t for processing
    simde__m128i c16 = simde_mm_packs_epi32(uq, uq);
    simde__m128i d16 = simde_mm_packs_epi32(dq, dq);
    
    // Calculate differences
    simde__m128i diff = simde_mm_sub_epi16(c16, d16);
    simde__m128i csq = simde_mm_mullo_epi16(c16, c16);   // coeff squared

    // Square differences
    diff = simde_mm_mullo_epi16(diff, diff);

    // Sum squares for error
    simde__m128i diff_sq_lo = simde_mm_unpacklo_epi16(diff, zero);
    simde__m128i diff_sq_hi = simde_mm_unpackhi_epi16(diff, zero);
    simde__m128i coeff_sq_lo = simde_mm_unpacklo_epi16(csq, zero);
    simde__m128i coeff_sq_hi = simde_mm_unpackhi_epi16(csq, zero);

    sse_acc = simde_mm_add_epi32(sse_acc, diff_sq_lo);
    sse_acc = simde_mm_add_epi32(sse_acc, diff_sq_hi);
    ssz_acc = simde_mm_add_epi32(ssz_acc, coeff_sq_lo);
    ssz_acc = simde_mm_add_epi32(ssz_acc, coeff_sq_hi);
  }

  // Sum up the accumulator
  int64_t sse = 0, ssz_sum = 0;
  simde__m128i temp;

  // Sum SSE accumulator
  temp = simde_mm_srli_si128(sse_acc, 8);
  sse_acc = simde_mm_add_epi32(sse_acc, temp);
  temp = simde_mm_srli_si128(sse_acc, 4);
  sse_acc = simde_mm_add_epi32(sse_acc, temp);
  sse = simde_mm_cvtsi128_si32(sse_acc);

  // Sum SSZ accumulator
  temp = simde_mm_srli_si128(ssz_acc, 8);
  ssz_acc = simde_mm_add_epi32(ssz_acc, temp);
  temp = simde_mm_srli_si128(ssz_acc, 4);
  ssz_acc = simde_mm_add_epi32(ssz_acc, temp);
  ssz_sum = simde_mm_cvtsi128_si32(ssz_acc);

  *ssz = ssz_sum;
  return sse;
#elif defined(__aarch64__) || defined(_M_ARM64)
  // ARM NEON implementation for fallback
  int64_t error = 0, sqcoeff = 0;
  for (intptr_t i = 0; i < block_size; i++) {
    const int diff = coeff[i] - dqcoeff[i];
    error += diff * diff;
    sqcoeff += coeff[i] * coeff[i];
  }
  *ssz = sqcoeff;
  return error;
#else
  // Pure C implementation for fallback
  int64_t error = 0, sqcoeff = 0;
  for (intptr_t i = 0; i < block_size; i++) {
    const int diff = coeff[i] - dqcoeff[i];
    error += diff * diff;
    sqcoeff += coeff[i] * coeff[i];
  }
  *ssz = sqcoeff;
  return error;
#endif
}

int64_t vp9_block_error_fp_simde(const tran_low_t *coeff, const tran_low_t *dqcoeff,
                             int block_size) {
#if defined(__x86_64__) || defined(_M_X64)
  // SSE2 implementation for block error fp
  simde__m128i sse_acc = simde_mm_setzero_si128();      // sse accumulator
  simde__m128i zero = simde_mm_setzero_si128();         // dedicated zero register
  simde__m128i uq, dq;
  int size = block_size;

  while (size > 0) {
    // Load 8 tran_low_t values (each tran_low_t is typically int16_t)
    if (size >= 8) {
      uq = load_tran_low(coeff);
      dq = load_tran_low(dqcoeff);
      coeff += 8;
      dqcoeff += 8;
      size -= 8;
    } else {
      // Handle remaining elements
      uq = simde_mm_setzero_si128();
      dq = simde_mm_setzero_si128();
      for (int i = 0; i < size; i++) {
        ((int16_t*)&uq)[i] = coeff[i];
        ((int16_t*)&dq)[i] = dqcoeff[i];
      }
      coeff += size;
      dqcoeff += size;
      size = 0;
    }

    // Convert to int16_t for processing
    simde__m128i c16 = simde_mm_packs_epi32(uq, uq);
    simde__m128i d16 = simde_mm_packs_epi32(dq, dq);
    
    // Calculate differences
    simde__m128i diff = simde_mm_sub_epi16(c16, d16);

    // Square differences
    diff = simde_mm_mullo_epi16(diff, diff);

    // Sum squares for error
    simde__m128i diff_sq_lo = simde_mm_unpacklo_epi16(diff, zero);
    simde__m128i diff_sq_hi = simde_mm_unpackhi_epi16(diff, zero);

    sse_acc = simde_mm_add_epi32(sse_acc, diff_sq_lo);
    sse_acc = simde_mm_add_epi32(sse_acc, diff_sq_hi);
  }

  // Sum up the accumulator
  int64_t sse = 0;
  simde__m128i temp;

  // Sum SSE accumulator
  temp = simde_mm_srli_si128(sse_acc, 8);
  sse_acc = simde_mm_add_epi32(sse_acc, temp);
  temp = simde_mm_srli_si128(sse_acc, 4);
  sse_acc = simde_mm_add_epi32(sse_acc, temp);
  sse = simde_mm_cvtsi128_si32(sse_acc);

  return sse;
#else
  // Pure C implementation for fallback
  int64_t error = 0;
  for (int i = 0; i < block_size; i++) {
    const int diff = coeff[i] - dqcoeff[i];
    error += diff * diff;
  }
  return error;
#endif
}