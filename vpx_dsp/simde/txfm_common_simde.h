/*
 *  Copyright (c) 2025 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_SIMDE_TXFM_COMMON_SIMDE_H_
#define VPX_VPX_DSP_SIMDE_TXFM_COMMON_SIMDE_H_

#include "vpx_dsp/simde/simde_headers.h"

// Define helper functions that were originally in x86/txfm_common_sse2.h but using SIMDE equivalents

#define pair_set_epi16(a, b) \
  simde_mm_set_epi16((int16_t)(b), (int16_t)(a), (int16_t)(b), (int16_t)(a), \
                     (int16_t)(b), (int16_t)(a), (int16_t)(b), (int16_t)(a))

#define pair_set_epi32(a, b) \
  simde_mm_set_epi32((int)(b), (int)(a), (int)(b), (int)(a))

#define dual_set_epi16(a, b) \
  simde_mm_set_epi16((int16_t)(b), (int16_t)(b), (int16_t)(b), (int16_t)(b), \
                     (int16_t)(a), (int16_t)(a), (int16_t)(a), (int16_t)(a))

#define octa_set_epi16(a, b, c, d, e, f, g, h) \
  simde_mm_setr_epi16((int16_t)(a), (int16_t)(b), (int16_t)(c), (int16_t)(d), \
                      (int16_t)(e), (int16_t)(f), (int16_t)(g), (int16_t)(h))

// Helper transpose functions using SIMDE - in-place version to match the usage in code
static INLINE void transpose_16bit_8x8_simde(simde__m128i *const in) {
  // Save original values before overwriting
  const simde__m128i t0 = in[0], t1 = in[1], t2 = in[2], t3 = in[3];
  const simde__m128i t4 = in[4], t5 = in[5], t6 = in[6], t7 = in[7];
  
  // Unpack 16 bit elements resulting in:
  const simde__m128i a0 = simde_mm_unpacklo_epi16(t0, t1);
  const simde__m128i a1 = simde_mm_unpacklo_epi16(t2, t3);
  const simde__m128i a2 = simde_mm_unpacklo_epi16(t4, t5);
  const simde__m128i a3 = simde_mm_unpacklo_epi16(t6, t7);
  const simde__m128i a4 = simde_mm_unpackhi_epi16(t0, t1);
  const simde__m128i a5 = simde_mm_unpackhi_epi16(t2, t3);
  const simde__m128i a6 = simde_mm_unpackhi_epi16(t4, t5);
  const simde__m128i a7 = simde_mm_unpackhi_epi16(t6, t7);

  // Unpack 32 bit elements resulting in:
  const simde__m128i b0 = simde_mm_unpacklo_epi32(a0, a1);
  const simde__m128i b1 = simde_mm_unpackhi_epi32(a0, a1);
  const simde__m128i b2 = simde_mm_unpacklo_epi32(a2, a3);
  const simde__m128i b3 = simde_mm_unpackhi_epi32(a2, a3);
  const simde__m128i b4 = simde_mm_unpacklo_epi32(a4, a5);
  const simde__m128i b5 = simde_mm_unpackhi_epi32(a4, a5);
  const simde__m128i b6 = simde_mm_unpacklo_epi32(a6, a7);
  const simde__m128i b7 = simde_mm_unpackhi_epi32(a6, a7);

  // Unpack 64 bit elements resulting in transposed matrix
  in[0] = simde_mm_unpacklo_epi64(b0, b2);
  in[1] = simde_mm_unpackhi_epi64(b0, b2);
  in[2] = simde_mm_unpacklo_epi64(b1, b3);
  in[3] = simde_mm_unpackhi_epi64(b1, b3);
  in[4] = simde_mm_unpacklo_epi64(b4, b6);
  in[5] = simde_mm_unpackhi_epi64(b4, b6);
  in[6] = simde_mm_unpacklo_epi64(b5, b7);
  in[7] = simde_mm_unpackhi_epi64(b5, b7);
}

static INLINE void transpose_16bit_16x16_simde(simde__m128i *const left,
                                              simde__m128i *const right) {
  simde__m128i tbuf[8];
  transpose_16bit_8x8_simde(left);
  transpose_16bit_8x8_simde(right);
  transpose_16bit_8x8_simde(left + 8);
  transpose_16bit_8x8_simde(right + 8);
 
  // Now swap middle 8x8 sections
  for (int i = 0; i < 8; i++) {
    tbuf[i] = left[i + 8];
    left[i + 8] = right[i];
    right[i] = tbuf[i];
  }
}

// Additional transpose functions
static INLINE void transpose4x4_16bit_simde(simde__m128i *in, simde__m128i *out) {
  // Unpack 16 bit elements. Goes from:
  // in[0]: 00 01 02 03  XX XX XX XX
  // in[1]: 10 11 12 13  XX XX XX XX
  // in[2]: 20 21 22 23  XX XX XX XX
  // in[3]: 30 31 32 33  XX XX XX XX
  // to:
  // a0:    00 10 01 11  02 12 03 13
  // a1:    20 30 21 31  22 32 23 33
  const simde__m128i a0 = simde_mm_unpacklo_epi16(in[0], in[1]);
  const simde__m128i a1 = simde_mm_unpacklo_epi16(in[2], in[3]);

  // Unpack 32 bit elements resulting in:
  // out[0]: 00 10 20 30  01 11 21 31
  // out[1]: 02 12 22 32  03 13 23 33
  // out[2]: 00 10 20 30  01 11 21 31
  // out[3]: 02 12 22 32  03 13 23 33
  // Actually, let's simplify by just using in as out for in-place
  out[0] = simde_mm_unpacklo_epi32(a0, a1);
  out[1] = simde_mm_unpackhi_epi32(a0, a1);
  out[2] = simde_mm_unpacklo_epi32(a0, a1);
  out[3] = simde_mm_unpackhi_epi32(a0, a1);
}

// Sign manipulation function - returns result with sign inverted if sign mask has bits set

// Sign manipulation function - returns result with sign inverted if sign mask has bits set
static INLINE simde__m128i invert_sign_simde(simde__m128i a, simde__m128i sign) {
  a = simde_mm_xor_si128(a, sign);
  return simde_mm_sub_epi16(a, sign);
}


// Functions for storing output using SIMDE
static INLINE void store_output_simde(const simde__m128i *poutput, tran_low_t *dst_ptr) {
#if CONFIG_VP9_HIGHBITDEPTH
  const simde__m128i zero = simde_mm_setzero_si128();
  const simde__m128i sign_bits = simde_mm_cmplt_epi16(*poutput, zero);
  simde__m128i out0 = simde_mm_unpacklo_epi16(*poutput, sign_bits);
  simde__m128i out1 = simde_mm_unpackhi_epi16(*poutput, sign_bits);
  simde_mm_store_si128((simde__m128i *)(dst_ptr), out0);
  simde_mm_store_si128((simde__m128i *)(dst_ptr + 4), out1);
#else
  simde_mm_store_si128((simde__m128i *)(dst_ptr), *poutput);
#endif  // CONFIG_VP9_HIGHBITDEPTH
}

#endif  // VPX_VPX_DSP_SIMDE_TXFM_COMMON_SIMDE_H_