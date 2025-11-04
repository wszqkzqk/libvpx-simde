/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

// This file contains SIMD implementations using SIMDE
// When CONFIG_SIMDE is enabled, this provides SIMD functionality without native instruction set extensions

#include "vpx_dsp_rtcd.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/simde/simde_headers.h"

// Sign extend 16 bit to 32 bit helper function
static INLINE void sign_extend_16bit_to_32bit_simde(simde__m128i in, simde__m128i zero,
                                                    simde__m128i *out_lo,
                                                    simde__m128i *out_hi) {
  const simde__m128i sign_bits = simde_mm_cmplt_epi16(in, zero);
  *out_lo = simde_mm_unpacklo_epi16(in, sign_bits);
  *out_hi = simde_mm_unpackhi_epi16(in, sign_bits);
}
#include "vpx_dsp/simde/bitdepth_conversion_simde.h"
#include "vpx_ports/mem.h"



void vpx_minmax_8x8_simde(const uint8_t *s, int p, const uint8_t *d, int dp,
                         int *min, int *max) {
  simde__m128i u0, s0, d0, diff, maxabsdiff, minabsdiff, negdiff, absdiff0, absdiff;
  u0 = simde_mm_setzero_si128();
  // Row 0
  s0 = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((const simde__m128i *)(s)), u0);
  d0 = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((const simde__m128i *)(d)), u0);
  diff = simde_mm_subs_epi16(s0, d0);
  negdiff = simde_mm_subs_epi16(u0, diff);
  absdiff0 = simde_mm_max_epi16(diff, negdiff);
  // Row 1
  s0 = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((const simde__m128i *)(s + p)), u0);
  d0 = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((const simde__m128i *)(d + dp)), u0);
  diff = simde_mm_subs_epi16(s0, d0);
  negdiff = simde_mm_subs_epi16(u0, diff);
  absdiff = simde_mm_max_epi16(diff, negdiff);
  maxabsdiff = simde_mm_max_epi16(absdiff0, absdiff);
  minabsdiff = simde_mm_min_epi16(absdiff0, absdiff);
  // Row 2
  s0 = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((const simde__m128i *)(s + 2 * p)), u0);
  d0 = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((const simde__m128i *)(d + 2 * dp)), u0);
  diff = simde_mm_subs_epi16(s0, d0);
  negdiff = simde_mm_subs_epi16(u0, diff);
  absdiff = simde_mm_max_epi16(diff, negdiff);
  maxabsdiff = simde_mm_max_epi16(maxabsdiff, absdiff);
  minabsdiff = simde_mm_min_epi16(minabsdiff, absdiff);
  // Row 3
  s0 = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((const simde__m128i *)(s + 3 * p)), u0);
  d0 = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((const simde__m128i *)(d + 3 * dp)), u0);
  diff = simde_mm_subs_epi16(s0, d0);
  negdiff = simde_mm_subs_epi16(u0, diff);
  absdiff = simde_mm_max_epi16(diff, negdiff);
  maxabsdiff = simde_mm_max_epi16(maxabsdiff, absdiff);
  minabsdiff = simde_mm_min_epi16(minabsdiff, absdiff);
  // Row 4
  s0 = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((const simde__m128i *)(s + 4 * p)), u0);
  d0 = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((const simde__m128i *)(d + 4 * dp)), u0);
  diff = simde_mm_subs_epi16(s0, d0);
  negdiff = simde_mm_subs_epi16(u0, diff);
  absdiff = simde_mm_max_epi16(diff, negdiff);
  maxabsdiff = simde_mm_max_epi16(maxabsdiff, absdiff);
  minabsdiff = simde_mm_min_epi16(minabsdiff, absdiff);
  // Row 5
  s0 = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((const simde__m128i *)(s + 5 * p)), u0);
  d0 = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((const simde__m128i *)(d + 5 * dp)), u0);
  diff = simde_mm_subs_epi16(s0, d0);
  negdiff = simde_mm_subs_epi16(u0, diff);
  absdiff = simde_mm_max_epi16(diff, negdiff);
  maxabsdiff = simde_mm_max_epi16(maxabsdiff, absdiff);
  minabsdiff = simde_mm_min_epi16(minabsdiff, absdiff);
  // Row 6
  s0 = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((const simde__m128i *)(s + 6 * p)), u0);
  d0 = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((const simde__m128i *)(d + 6 * dp)), u0);
  diff = simde_mm_subs_epi16(s0, d0);
  negdiff = simde_mm_subs_epi16(u0, diff);
  absdiff = simde_mm_max_epi16(diff, negdiff);
  maxabsdiff = simde_mm_max_epi16(maxabsdiff, absdiff);
  minabsdiff = simde_mm_min_epi16(minabsdiff, absdiff);
  // Row 7
  s0 = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((const simde__m128i *)(s + 7 * p)), u0);
  d0 = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((const simde__m128i *)(d + 7 * dp)), u0);
  diff = simde_mm_subs_epi16(s0, d0);
  negdiff = simde_mm_subs_epi16(u0, diff);
  absdiff = simde_mm_max_epi16(diff, negdiff);
  maxabsdiff = simde_mm_max_epi16(maxabsdiff, absdiff);
  minabsdiff = simde_mm_min_epi16(minabsdiff, absdiff);

  maxabsdiff = simde_mm_max_epi16(maxabsdiff, simde_mm_srli_si128(maxabsdiff, 8));
  maxabsdiff = simde_mm_max_epi16(maxabsdiff, simde_mm_srli_epi64(maxabsdiff, 32));
  maxabsdiff = simde_mm_max_epi16(maxabsdiff, simde_mm_srli_epi64(maxabsdiff, 16));
  *max = simde_mm_extract_epi16(maxabsdiff, 0);

  minabsdiff = simde_mm_min_epi16(minabsdiff, simde_mm_srli_si128(minabsdiff, 8));
  minabsdiff = simde_mm_min_epi16(minabsdiff, simde_mm_srli_epi64(minabsdiff, 32));
  minabsdiff = simde_mm_min_epi16(minabsdiff, simde_mm_srli_epi64(minabsdiff, 16));
  *min = simde_mm_extract_epi16(minabsdiff, 0);
}

unsigned int vpx_avg_8x8_simde(const uint8_t *s, int p) {
  simde__m128i s0, s1, u0;
  unsigned int avg = 0;
  u0 = simde_mm_setzero_si128();
  s0 = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((const simde__m128i *)(s)), u0);
  s1 = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((const simde__m128i *)(s + p)), u0);
  s0 = simde_mm_adds_epu16(s0, s1);
  s1 = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((const simde__m128i *)(s + 2 * p)), u0);
  s0 = simde_mm_adds_epu16(s0, s1);
  s1 = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((const simde__m128i *)(s + 3 * p)), u0);
  s0 = simde_mm_adds_epu16(s0, s1);
  s1 = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((const simde__m128i *)(s + 4 * p)), u0);
  s0 = simde_mm_adds_epu16(s0, s1);
  s1 = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((const simde__m128i *)(s + 5 * p)), u0);
  s0 = simde_mm_adds_epu16(s0, s1);
  s1 = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((const simde__m128i *)(s + 6 * p)), u0);
  s0 = simde_mm_adds_epu16(s0, s1);
  s1 = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((const simde__m128i *)(s + 7 * p)), u0);
  s0 = simde_mm_adds_epu16(s0, s1);

  s0 = simde_mm_adds_epu16(s0, simde_mm_srli_si128(s0, 8));
  s0 = simde_mm_adds_epu16(s0, simde_mm_srli_epi64(s0, 32));
  s0 = simde_mm_adds_epu16(s0, simde_mm_srli_epi64(s0, 16));
  avg = simde_mm_extract_epi16(s0, 0);
  return (avg + 32) >> 6;
}

unsigned int vpx_avg_4x4_simde(const uint8_t *s, int p) {
  simde__m128i s0, s1, u0;
  unsigned int avg = 0;
  u0 = simde_mm_setzero_si128();
  s0 = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((const simde__m128i *)(s)), u0);
  s1 = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((const simde__m128i *)(s + p)), u0);
  s0 = simde_mm_adds_epu16(s0, s1);
  s1 = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((const simde__m128i *)(s + 2 * p)), u0);
  s0 = simde_mm_adds_epu16(s0, s1);
  s1 = simde_mm_unpacklo_epi8(simde_mm_loadl_epi64((const simde__m128i *)(s + 3 * p)), u0);
  s0 = simde_mm_adds_epu16(s0, s1);

  s0 = simde_mm_adds_epu16(s0, simde_mm_srli_si128(s0, 4));
  s0 = simde_mm_adds_epu16(s0, simde_mm_srli_epi64(s0, 16));
  avg = simde_mm_extract_epi16(s0, 0);
  return (avg + 8) >> 4;
}

#if CONFIG_VP9_HIGHBITDEPTH
unsigned int vpx_highbd_avg_8x8_simde(const uint8_t *s8, int p) {
  simde__m128i s0, s1;
  unsigned int avg;
  const uint16_t *s = CONVERT_TO_SHORTPTR(s8);
  const simde__m128i zero = simde_mm_setzero_si128();
  s0 = simde_mm_loadu_si128((const simde__m128i *)(s));
  s1 = simde_mm_loadu_si128((const simde__m128i *)(s + p));
  s0 = simde_mm_adds_epu16(s0, s1);
  s1 = simde_mm_loadu_si128((const simde__m128i *)(s + 2 * p));
  s0 = simde_mm_adds_epu16(s0, s1);
  s1 = simde_mm_loadu_si128((const simde__m128i *)(s + 3 * p));
  s0 = simde_mm_adds_epu16(s0, s1);
  s1 = simde_mm_loadu_si128((const simde__m128i *)(s + 4 * p));
  s0 = simde_mm_adds_epu16(s0, s1);
  s1 = simde_mm_loadu_si128((const simde__m128i *)(s + 5 * p));
  s0 = simde_mm_adds_epu16(s0, s1);
  s1 = simde_mm_loadu_si128((const simde__m128i *)(s + 6 * p));
  s0 = simde_mm_adds_epu16(s0, s1);
  s1 = simde_mm_loadu_si128((const simde__m128i *)(s + 7 * p));
  s0 = simde_mm_adds_epu16(s0, s1);
  s1 = simde_mm_unpackhi_epi16(s0, zero);
  s0 = simde_mm_unpacklo_epi16(s0, zero);
  s0 = simde_mm_add_epi32(s0, s1);
  s0 = simde_mm_add_epi32(s0, simde_mm_srli_si128(s0, 8));
  s0 = simde_mm_add_epi32(s0, simde_mm_srli_si128(s0, 4));
  avg = (unsigned int)simde_mm_cvtsi128_si32(s0);

  return (avg + 32) >> 6;
}

unsigned int vpx_highbd_avg_4x4_simde(const uint8_t *s8, int p) {
  simde__m128i s0, s1;
  unsigned int avg;
  const uint16_t *s = CONVERT_TO_SHORTPTR(s8);
  s0 = simde_mm_loadl_epi64((const simde__m128i *)(s));
  s1 = simde_mm_loadl_epi64((const simde__m128i *)(s + p));
  s0 = simde_mm_adds_epu16(s0, s1);
  s1 = simde_mm_loadl_epi64((const simde__m128i *)(s + 2 * p));
  s0 = simde_mm_adds_epu16(s0, s1);
  s1 = simde_mm_loadl_epi64((const simde__m128i *)(s + 3 * p));
  s0 = simde_mm_adds_epu16(s0, s1);
  s0 = simde_mm_add_epi16(s0, simde_mm_srli_si128(s0, 4));
  s0 = simde_mm_add_epi16(s0, simde_mm_srli_si128(s0, 2));
  avg = simde_mm_extract_epi16(s0, 0);

  return (avg + 8) >> 4;
}
#endif  // CONFIG_VP9_HIGHBITDEPTH

static void hadamard_col8_simde(simde__m128i *in, int iter) {
  simde__m128i a0 = in[0];
  simde__m128i a1 = in[1];
  simde__m128i a2 = in[2];
  simde__m128i a3 = in[3];
  simde__m128i a4 = in[4];
  simde__m128i a5 = in[5];
  simde__m128i a6 = in[6];
  simde__m128i a7 = in[7];

  simde__m128i b0 = simde_mm_add_epi16(a0, a1);
  simde__m128i b1 = simde_mm_sub_epi16(a0, a1);
  simde__m128i b2 = simde_mm_add_epi16(a2, a3);
  simde__m128i b3 = simde_mm_sub_epi16(a2, a3);
  simde__m128i b4 = simde_mm_add_epi16(a4, a5);
  simde__m128i b5 = simde_mm_sub_epi16(a4, a5);
  simde__m128i b6 = simde_mm_add_epi16(a6, a7);
  simde__m128i b7 = simde_mm_sub_epi16(a6, a7);

  a0 = simde_mm_add_epi16(b0, b2);
  a1 = simde_mm_add_epi16(b1, b3);
  a2 = simde_mm_sub_epi16(b0, b2);
  a3 = simde_mm_sub_epi16(b1, b3);
  a4 = simde_mm_add_epi16(b4, b6);
  a5 = simde_mm_add_epi16(b5, b7);
  a6 = simde_mm_sub_epi16(b4, b6);
  a7 = simde_mm_sub_epi16(b5, b7);

  if (iter == 0) {
    b0 = simde_mm_add_epi16(a0, a4);
    b7 = simde_mm_add_epi16(a1, a5);
    b3 = simde_mm_add_epi16(a2, a6);
    b4 = simde_mm_add_epi16(a3, a7);
    b2 = simde_mm_sub_epi16(a0, a4);
    b6 = simde_mm_sub_epi16(a1, a5);
    b1 = simde_mm_sub_epi16(a2, a6);
    b5 = simde_mm_sub_epi16(a3, a7);

    a0 = simde_mm_unpacklo_epi16(b0, b1);
    a1 = simde_mm_unpacklo_epi16(b2, b3);
    a2 = simde_mm_unpackhi_epi16(b0, b1);
    a3 = simde_mm_unpackhi_epi16(b2, b3);
    a4 = simde_mm_unpacklo_epi16(b4, b5);
    a5 = simde_mm_unpacklo_epi16(b6, b7);
    a6 = simde_mm_unpackhi_epi16(b4, b5);
    a7 = simde_mm_unpackhi_epi16(b6, b7);

    b0 = simde_mm_unpacklo_epi32(a0, a1);
    b1 = simde_mm_unpacklo_epi32(a4, a5);
    b2 = simde_mm_unpackhi_epi32(a0, a1);
    b3 = simde_mm_unpackhi_epi32(a4, a5);
    b4 = simde_mm_unpacklo_epi32(a2, a3);
    b5 = simde_mm_unpacklo_epi32(a6, a7);
    b6 = simde_mm_unpackhi_epi32(a2, a3);
    b7 = simde_mm_unpackhi_epi32(a6, a7);

    in[0] = simde_mm_unpacklo_epi64(b0, b1);
    in[1] = simde_mm_unpackhi_epi64(b0, b1);
    in[2] = simde_mm_unpacklo_epi64(b2, b3);
    in[3] = simde_mm_unpackhi_epi64(b2, b3);
    in[4] = simde_mm_unpacklo_epi64(b4, b5);
    in[5] = simde_mm_unpackhi_epi64(b4, b5);
    in[6] = simde_mm_unpacklo_epi64(b6, b7);
    in[7] = simde_mm_unpackhi_epi64(b6, b7);
  } else {
    in[0] = simde_mm_add_epi16(a0, a4);
    in[7] = simde_mm_add_epi16(a1, a5);
    in[3] = simde_mm_add_epi16(a2, a6);
    in[4] = simde_mm_add_epi16(a3, a7);
    in[2] = simde_mm_sub_epi16(a0, a4);
    in[6] = simde_mm_sub_epi16(a1, a5);
    in[1] = simde_mm_sub_epi16(a2, a6);
    in[5] = simde_mm_sub_epi16(a3, a7);
  }
}

static INLINE void hadamard_8x8_simde(const int16_t *src_diff,
                                     ptrdiff_t src_stride, tran_low_t *coeff,
                                     int is_final) {
  simde__m128i src[8];
  src[0] = simde_mm_load_si128((const simde__m128i *)src_diff);
  src[1] = simde_mm_load_si128((const simde__m128i *)(src_diff += src_stride));
  src[2] = simde_mm_load_si128((const simde__m128i *)(src_diff += src_stride));
  src[3] = simde_mm_load_si128((const simde__m128i *)(src_diff += src_stride));
  src[4] = simde_mm_load_si128((const simde__m128i *)(src_diff += src_stride));
  src[5] = simde_mm_load_si128((const simde__m128i *)(src_diff += src_stride));
  src[6] = simde_mm_load_si128((const simde__m128i *)(src_diff += src_stride));
  src[7] = simde_mm_load_si128((const simde__m128i *)(src_diff + src_stride));

  hadamard_col8_simde(src, 0);
  hadamard_col8_simde(src, 1);

  if (is_final) {
    store_tran_low_simde(src[0], coeff);
    coeff += 8;
    store_tran_low_simde(src[1], coeff);
    coeff += 8;
    store_tran_low_simde(src[2], coeff);
    coeff += 8;
    store_tran_low_simde(src[3], coeff);
    coeff += 8;
    store_tran_low_simde(src[4], coeff);
    coeff += 8;
    store_tran_low_simde(src[5], coeff);
    coeff += 8;
    store_tran_low_simde(src[6], coeff);
    coeff += 8;
    store_tran_low_simde(src[7], coeff);
  } else {
    int16_t *coeff16 = (int16_t *)coeff;
    simde_mm_store_si128((simde__m128i *)coeff16, src[0]);
    coeff16 += 8;
    simde_mm_store_si128((simde__m128i *)coeff16, src[1]);
    coeff16 += 8;
    simde_mm_store_si128((simde__m128i *)coeff16, src[2]);
    coeff16 += 8;
    simde_mm_store_si128((simde__m128i *)coeff16, src[3]);
    coeff16 += 8;
    simde_mm_store_si128((simde__m128i *)coeff16, src[4]);
    coeff16 += 8;
    simde_mm_store_si128((simde__m128i *)coeff16, src[5]);
    coeff16 += 8;
    simde_mm_store_si128((simde__m128i *)coeff16, src[6]);
    coeff16 += 8;
    simde_mm_store_si128((simde__m128i *)coeff16, src[7]);
  }
}

void vpx_hadamard_8x8_simde(const int16_t *src_diff, ptrdiff_t src_stride,
                           tran_low_t *coeff) {
  hadamard_8x8_simde(src_diff, src_stride, coeff, 1);
}

static INLINE void hadamard_16x16_simde(const int16_t *src_diff,
                                       ptrdiff_t src_stride, tran_low_t *coeff,
                                       int is_final) {
#if CONFIG_VP9_HIGHBITDEPTH
  // For high bitdepths, it is unnecessary to store_tran_low
  // (mult/unpack/store), then load_tran_low (load/pack) the same memory in the
  // next stage.  Output to an intermediate buffer first, then store_tran_low_simde()
  // in the final stage.
  DECLARE_ALIGNED(32, int16_t, temp_coeff[16 * 16]);
  int16_t *t_coeff = temp_coeff;
#else
  int16_t *t_coeff = coeff;
#endif
  int16_t *coeff16 = (int16_t *)coeff;
  int idx;
  for (idx = 0; idx < 4; ++idx) {
    const int16_t *src_ptr =
        src_diff + (idx >> 1) * 8 * src_stride + (idx & 0x01) * 8;
    hadamard_8x8_simde(src_ptr, src_stride, (tran_low_t *)(t_coeff + idx * 64),
                      0);
  }

  for (idx = 0; idx < 64; idx += 8) {
    simde__m128i coeff0 = simde_mm_load_si128((const simde__m128i *)t_coeff);
    simde__m128i coeff1 = simde_mm_load_si128((const simde__m128i *)(t_coeff + 64));
    simde__m128i coeff2 = simde_mm_load_si128((const simde__m128i *)(t_coeff + 128));
    simde__m128i coeff3 = simde_mm_load_si128((const simde__m128i *)(t_coeff + 192));

    simde__m128i b0 = simde_mm_add_epi16(coeff0, coeff1);
    simde__m128i b1 = simde_mm_sub_epi16(coeff0, coeff1);
    simde__m128i b2 = simde_mm_add_epi16(coeff2, coeff3);
    simde__m128i b3 = simde_mm_sub_epi16(coeff2, coeff3);

    b0 = simde_mm_srai_epi16(b0, 1);
    b1 = simde_mm_srai_epi16(b1, 1);
    b2 = simde_mm_srai_epi16(b2, 1);
    b3 = simde_mm_srai_epi16(b3, 1);

    coeff0 = simde_mm_add_epi16(b0, b2);
    coeff1 = simde_mm_add_epi16(b1, b3);
    coeff2 = simde_mm_sub_epi16(b0, b2);
    coeff3 = simde_mm_sub_epi16(b1, b3);

    if (is_final) {
      store_tran_low_simde(coeff0, coeff);
      store_tran_low_simde(coeff1, coeff + 64);
      store_tran_low_simde(coeff2, coeff + 128);
      store_tran_low_simde(coeff3, coeff + 192);
      coeff += 8;
    } else {
      simde_mm_store_si128((simde__m128i *)coeff16, coeff0);
      simde_mm_store_si128((simde__m128i *)(coeff16 + 64), coeff1);
      simde_mm_store_si128((simde__m128i *)(coeff16 + 128), coeff2);
      simde_mm_store_si128((simde__m128i *)(coeff16 + 192), coeff3);
      coeff16 += 8;
    }

    t_coeff += 8;
  }
}

void vpx_hadamard_16x16_simde(const int16_t *src_diff, ptrdiff_t src_stride,
                             tran_low_t *coeff) {
  hadamard_16x16_simde(src_diff, src_stride, coeff, 1);
}

void vpx_hadamard_32x32_simde(const int16_t *src_diff, ptrdiff_t src_stride,
                             tran_low_t *coeff) {
#if CONFIG_VP9_HIGHBITDEPTH
  // For high bitdepths, it is unnecessary to store_tran_low
  // (mult/unpack/store), then load_tran_low (load/pack) the same memory in the
  // next stage.  Output to an intermediate buffer first, then store_tran_low_simde()
  // in the final stage.
  DECLARE_ALIGNED(32, int16_t, temp_coeff[32 * 32]);
  int16_t *t_coeff = temp_coeff;
#else
  int16_t *t_coeff = coeff;
#endif
  int idx;
  simde__m128i coeff0_lo, coeff1_lo, coeff2_lo, coeff3_lo, b0_lo, b1_lo, b2_lo,
      b3_lo;
  simde__m128i coeff0_hi, coeff1_hi, coeff2_hi, coeff3_hi, b0_hi, b1_hi, b2_hi,
      b3_hi;
  simde__m128i b0, b1, b2, b3;
  const simde__m128i zero = simde_mm_setzero_si128();
  for (idx = 0; idx < 4; ++idx) {
    const int16_t *src_ptr =
        src_diff + (idx >> 1) * 16 * src_stride + (idx & 0x01) * 16;
    hadamard_16x16_simde(src_ptr, src_stride,
                        (tran_low_t *)(t_coeff + idx * 256), 0);
  }

  for (idx = 0; idx < 256; idx += 8) {
    simde__m128i coeff0 = simde_mm_load_si128((const simde__m128i *)t_coeff);
    simde__m128i coeff1 = simde_mm_load_si128((const simde__m128i *)(t_coeff + 256));
    simde__m128i coeff2 = simde_mm_load_si128((const simde__m128i *)(t_coeff + 512));
    simde__m128i coeff3 = simde_mm_load_si128((const simde__m128i *)(t_coeff + 768));

    // Sign extend 16 bit to 32 bit.
    sign_extend_16bit_to_32bit_simde(coeff0, zero, &coeff0_lo, &coeff0_hi);
    sign_extend_16bit_to_32bit_simde(coeff1, zero, &coeff1_lo, &coeff1_hi);
    sign_extend_16bit_to_32bit_simde(coeff2, zero, &coeff2_lo, &coeff2_hi);
    sign_extend_16bit_to_32bit_simde(coeff3, zero, &coeff3_lo, &coeff3_hi);

    b0_lo = simde_mm_add_epi32(coeff0_lo, coeff1_lo);
    b0_hi = simde_mm_add_epi32(coeff0_hi, coeff1_hi);

    b1_lo = simde_mm_sub_epi32(coeff0_lo, coeff1_lo);
    b1_hi = simde_mm_sub_epi32(coeff0_hi, coeff1_hi);

    b2_lo = simde_mm_add_epi32(coeff2_lo, coeff3_lo);
    b2_hi = simde_mm_add_epi32(coeff2_hi, coeff3_hi);

    b3_lo = simde_mm_sub_epi32(coeff2_lo, coeff3_lo);
    b3_hi = simde_mm_sub_epi32(coeff2_hi, coeff3_hi);

    b0_lo = simde_mm_srai_epi32(b0_lo, 2);
    b1_lo = simde_mm_srai_epi32(b1_lo, 2);
    b2_lo = simde_mm_srai_epi32(b2_lo, 2);
    b3_lo = simde_mm_srai_epi32(b3_lo, 2);

    b0_hi = simde_mm_srai_epi32(b0_hi, 2);
    b1_hi = simde_mm_srai_epi32(b1_hi, 2);
    b2_hi = simde_mm_srai_epi32(b2_hi, 2);
    b3_hi = simde_mm_srai_epi32(b3_hi, 2);

    b0 = simde_mm_packs_epi32(b0_lo, b0_hi);
    b1 = simde_mm_packs_epi32(b1_lo, b1_hi);
    b2 = simde_mm_packs_epi32(b2_lo, b2_hi);
    b3 = simde_mm_packs_epi32(b3_lo, b3_hi);

    coeff0 = simde_mm_add_epi16(b0, b2);
    coeff1 = simde_mm_add_epi16(b1, b3);
    store_tran_low_simde(coeff0, coeff);
    store_tran_low_simde(coeff1, coeff + 256);

    coeff2 = simde_mm_sub_epi16(b0, b2);
    coeff3 = simde_mm_sub_epi16(b1, b3);
    store_tran_low_simde(coeff2, coeff + 512);
    store_tran_low_simde(coeff3, coeff + 768);

    coeff += 8;
    t_coeff += 8;
  }
}

int vpx_satd_simde(const tran_low_t *coeff, int length) {
  int i;
  const simde__m128i zero = simde_mm_setzero_si128();
  simde__m128i accum = zero;

  for (i = 0; i < length; i += 8) {
    const simde__m128i src_line = load_tran_low_simde(coeff);
    const simde__m128i inv = simde_mm_sub_epi16(zero, src_line);
    const simde__m128i abs = simde_mm_max_epi16(src_line, inv);  // abs(src_line)
    const simde__m128i abs_lo = simde_mm_unpacklo_epi16(abs, zero);
    const simde__m128i abs_hi = simde_mm_unpackhi_epi16(abs, zero);
    const simde__m128i sum = simde_mm_add_epi32(abs_lo, abs_hi);
    accum = simde_mm_add_epi32(accum, sum);
    coeff += 8;
  }

  {  // cascading summation of accum
    simde__m128i hi = simde_mm_srli_si128(accum, 8);
    accum = simde_mm_add_epi32(accum, hi);
    hi = simde_mm_srli_epi64(accum, 32);
    accum = simde_mm_add_epi32(accum, hi);
  }

  return simde_mm_cvtsi128_si32(accum);
}

void vpx_int_pro_row_simde(int16_t hbuf[16], const uint8_t *ref,
                          const int ref_stride, const int height) {
  int idx;
  simde__m128i zero = simde_mm_setzero_si128();
  simde__m128i src_line = simde_mm_loadu_si128((const simde__m128i *)ref);
  simde__m128i s0 = simde_mm_unpacklo_epi8(src_line, zero);
  simde__m128i s1 = simde_mm_unpackhi_epi8(src_line, zero);
  simde__m128i t0, t1;
  int height_1 = height - 1;
  ref += ref_stride;

  for (idx = 1; idx < height_1; idx += 2) {
    src_line = simde_mm_loadu_si128((const simde__m128i *)ref);
    t0 = simde_mm_unpacklo_epi8(src_line, zero);
    t1 = simde_mm_unpackhi_epi8(src_line, zero);
    s0 = simde_mm_adds_epu16(s0, t0);
    s1 = simde_mm_adds_epu16(s1, t1);
    ref += ref_stride;

    src_line = simde_mm_loadu_si128((const simde__m128i *)ref);
    t0 = simde_mm_unpacklo_epi8(src_line, zero);
    t1 = simde_mm_unpackhi_epi8(src_line, zero);
    s0 = simde_mm_adds_epu16(s0, t0);
    s1 = simde_mm_adds_epu16(s1, t1);
    ref += ref_stride;
  }

  src_line = simde_mm_loadu_si128((const simde__m128i *)ref);
  t0 = simde_mm_unpacklo_epi8(src_line, zero);
  t1 = simde_mm_unpackhi_epi8(src_line, zero);
  s0 = simde_mm_adds_epu16(s0, t0);
  s1 = simde_mm_adds_epu16(s1, t1);

  if (height == 64) {
    s0 = simde_mm_srai_epi16(s0, 5);
    s1 = simde_mm_srai_epi16(s1, 5);
  } else if (height == 32) {
    s0 = simde_mm_srai_epi16(s0, 4);
    s1 = simde_mm_srai_epi16(s1, 4);
  } else {
    s0 = simde_mm_srai_epi16(s0, 3);
    s1 = simde_mm_srai_epi16(s1, 3);
  }

  simde_mm_storeu_si128((simde__m128i *)hbuf, s0);
  hbuf += 8;
  simde_mm_storeu_si128((simde__m128i *)hbuf, s1);
}

int16_t vpx_int_pro_col_simde(const uint8_t *ref, const int width) {
  simde__m128i zero = simde_mm_setzero_si128();
  simde__m128i src_line = simde_mm_loadu_si128((const simde__m128i *)ref);
  simde__m128i s0 = simde_mm_sad_epu8(src_line, zero);
  simde__m128i s1;
  int i;

  for (i = 16; i < width; i += 16) {
    ref += 16;
    src_line = simde_mm_loadu_si128((const simde__m128i *)ref);
    s1 = simde_mm_sad_epu8(src_line, zero);
    s0 = simde_mm_adds_epu16(s0, s1);
    s0 = simde_mm_adds_epu16(s0, s1);
  }

  s1 = simde_mm_srli_si128(s0, 8);
  s0 = simde_mm_adds_epu16(s0, s1);

  return simde_mm_extract_epi16(s0, 0);
}

int vpx_vector_var_simde(const int16_t *ref, const int16_t *src, const int bwl) {
  int idx;
  int width = 4 << bwl;
  int16_t mean;
  simde__m128i v0 = simde_mm_loadu_si128((const simde__m128i *)ref);
  simde__m128i v1 = simde_mm_load_si128((const simde__m128i *)src);
  simde__m128i diff = simde_mm_subs_epi16(v0, v1);
  simde__m128i sum = diff;
  simde__m128i sse = simde_mm_madd_epi16(diff, diff);

  ref += 8;
  src += 8;

  for (idx = 8; idx < width; idx += 8) {
    v0 = simde_mm_loadu_si128((const simde__m128i *)ref);
    v1 = simde_mm_load_si128((const simde__m128i *)src);
    diff = simde_mm_subs_epi16(v0, v1);

    sum = simde_mm_add_epi16(sum, diff);
    v0 = simde_mm_madd_epi16(diff, diff);
    sse = simde_mm_add_epi32(sse, v0);

    ref += 8;
    src += 8;
  }

  v0 = simde_mm_srli_si128(sum, 8);
  sum = simde_mm_add_epi16(sum, v0);
  v0 = simde_mm_srli_epi64(sum, 32);
  sum = simde_mm_add_epi16(sum, v0);
  v0 = simde_mm_srli_epi32(sum, 16);
  sum = simde_mm_add_epi16(sum, v0);

  v1 = simde_mm_srli_si128(sse, 8);
  sse = simde_mm_add_epi32(sse, v1);
  v1 = simde_mm_srli_epi64(sse, 32);
  sse = simde_mm_add_epi32(sse, v1);

  mean = (int16_t)simde_mm_extract_epi16(sum, 0);

  return simde_mm_cvtsi128_si32(sse) - ((mean * mean) >> (bwl + 2));
}