/*
 *  Copyright (c) 2025 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>
#include "vpx_dsp/simde/simde_headers.h"

#include "./vp9_rtcd.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_dsp/simde/bitdepth_conversion_simde.h"
#include "vpx_dsp/simde/quantize_simde.h"
#include "vp9/common/vp9_scan.h"
#include "vp9/encoder/vp9_block.h"

// Zero fill 8 positions in the output buffer.
static VPX_FORCE_INLINE void store_zero_tran_low_avx2_simde(tran_low_t *a) {
  const simde__m256i zero = simde_mm256_setzero_si256();
#if CONFIG_VP9_HIGHBITDEPTH
  simde_mm256_storeu_si256((simde__m256i *)(a), zero);
  simde_mm256_storeu_si256((simde__m256i *)(a + 8), zero);
#else
  simde_mm256_storeu_si256((simde__m256i *)(a), zero);
#endif
}

static VPX_FORCE_INLINE void load_fp_values_avx2_simde(
    const struct macroblock_plane *mb_plane, simde__m256i *round, simde__m256i *quant,
    const int16_t *dequant_ptr, simde__m256i *dequant) {
  *round = simde_mm256_castsi128_si256(
      simde_mm_load_si128((const simde__m128i *)mb_plane->round_fp));
  *round = simde_mm256_permute4x64_epi64(*round, 0x54);
  *quant = simde_mm256_castsi128_si256(
      simde_mm_load_si128((const simde__m128i *)mb_plane->quant_fp));
  *quant = simde_mm256_permute4x64_epi64(*quant, 0x54);
  *dequant =
      simde_mm256_castsi128_si256(simde_mm_load_si128((const simde__m128i *)dequant_ptr));
  *dequant = simde_mm256_permute4x64_epi64(*dequant, 0x54);
}

static VPX_FORCE_INLINE simde__m256i get_max_lane_eob(const int16_t *iscan,
                                                 simde__m256i v_eobmax,
                                                 simde__m256i v_mask) {
#if CONFIG_VP9_HIGHBITDEPTH
  const simde__m256i v_iscan = simde_mm256_permute4x64_epi64(
      simde_mm256_loadu_si256((const simde__m256i *)iscan), 0xD8);
#else
  const simde__m256i v_iscan = simde_mm256_loadu_si256((const simde__m256i *)iscan);
#endif
  const simde__m256i v_nz_iscan = simde_mm256_and_si256(v_iscan, v_mask);
  return simde_mm256_max_epi16(v_eobmax, v_nz_iscan);
}

static VPX_FORCE_INLINE uint16_t get_max_eob(simde__m256i eob256) {
  const simde__m256i eob_lo = eob256;
  // Copy upper 128 to lower 128
  const simde__m256i eob_hi = simde_mm256_permute2x128_si256(eob256, eob256, 0X81);
  simde__m256i eob = simde_mm256_max_epi16(eob_lo, eob_hi);
  simde__m256i eob_s = simde_mm256_shuffle_epi32(eob, 0xe);
  eob = simde_mm256_max_epi16(eob, eob_s);
  eob_s = simde_mm256_shufflelo_epi16(eob, 0xe);
  eob = simde_mm256_max_epi16(eob, eob_s);
  eob_s = simde_mm256_shufflelo_epi16(eob, 1);
  eob = simde_mm256_max_epi16(eob, eob_s);
#if defined(_MSC_VER) && (_MSC_VER < 1910)
  return simde_mm_cvtsi128_si32(simde_mm256_extracti128_si256(eob, 0)) & 0xffff;
#else
  return (uint16_t)simde_mm256_extract_epi16(eob, 0);
#endif
}

static VPX_FORCE_INLINE void quantize_fp_16(
    const simde__m256i *round, const simde__m256i *quant, const simde__m256i *dequant,
    const simde__m256i *thr, const tran_low_t *coeff_ptr, const int16_t *iscan_ptr,
    tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr, simde__m256i *eob_max) {
  const simde__m256i coeff = load_tran_low_avx2_simde(coeff_ptr);
  const simde__m256i abs_coeff = simde_mm256_abs_epi16(coeff);
  const int32_t nzflag =
      simde_mm256_movemask_epi8(simde_mm256_cmpgt_epi16(abs_coeff, *thr));

  if (nzflag) {
    const simde__m256i tmp_rnd = simde_mm256_adds_epi16(abs_coeff, *round);
    const simde__m256i abs_qcoeff = simde_mm256_mulhi_epi16(tmp_rnd, *quant);
    const simde__m256i qcoeff = simde_mm256_sign_epi16(abs_qcoeff, coeff);
    const simde__m256i dqcoeff = simde_mm256_mullo_epi16(qcoeff, *dequant);
    const simde__m256i nz_mask =
        simde_mm256_cmpgt_epi16(abs_qcoeff, simde_mm256_setzero_si256());
    store_tran_low_avx2_simde(qcoeff, qcoeff_ptr);
    store_tran_low_avx2_simde(dqcoeff, dqcoeff_ptr);

    *eob_max = get_max_lane_eob(iscan_ptr, *eob_max, nz_mask);
  } else {
    store_zero_tran_low_avx2_simde(qcoeff_ptr);
    store_zero_tran_low_avx2_simde(dqcoeff_ptr);
  }
}

void vp9_quantize_fp_avx2_simde(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                          const struct macroblock_plane *const mb_plane,
                          tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                          const int16_t *dequant_ptr, uint16_t *eob_ptr,
                          const struct ScanOrder *const scan_order) {
  simde__m256i round, quant, dequant, thr;
  simde__m256i eob_max = simde_mm256_setzero_si256();
  const int16_t *iscan = scan_order->iscan;

  coeff_ptr += n_coeffs;
  iscan += n_coeffs;
  qcoeff_ptr += n_coeffs;
  dqcoeff_ptr += n_coeffs;
  n_coeffs = -n_coeffs;

  // Setup global values
  load_fp_values_avx2_simde(mb_plane, &round, &quant, dequant_ptr, &dequant);
  thr = simde_mm256_setzero_si256();

  quantize_fp_16(&round, &quant, &dequant, &thr, coeff_ptr + n_coeffs,
                 iscan + n_coeffs, qcoeff_ptr + n_coeffs,
                 dqcoeff_ptr + n_coeffs, &eob_max);

  n_coeffs += 8 * 2;

  // remove dc constants
  dequant = simde_mm256_permute2x128_si256(dequant, dequant, 0x31);
  quant = simde_mm256_permute2x128_si256(quant, quant, 0x31);
  round = simde_mm256_permute2x128_si256(round, round, 0x31);
  thr = simde_mm256_srai_epi16(dequant, 1);

  // AC only loop
  while (n_coeffs < 0) {
    quantize_fp_16(&round, &quant, &dequant, &thr, coeff_ptr + n_coeffs,
                   iscan + n_coeffs, qcoeff_ptr + n_coeffs,
                   dqcoeff_ptr + n_coeffs, &eob_max);
    n_coeffs += 8 * 2;
  }

  *eob_ptr = get_max_eob(eob_max);
}

// Enable this flag when matching the optimized code to
// vp9_quantize_fp_32x32_c(). Disabled, the optimized code will match the
// existing ssse3 code and quantize_fp_32x32_nz_c().
//
// #define MATCH_VP9_QUANTIZE_FP_32X32_C

#ifndef MATCH_VP9_QUANTIZE_FP_32X32_C
static VPX_FORCE_INLINE void quantize_fp_32x32_16_no_nzflag(
    const simde__m256i *round, const simde__m256i *quant, const simde__m256i *dequant,
    const simde__m256i *thr, const tran_low_t *coeff_ptr, const int16_t *iscan_ptr,
    tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr, simde__m256i *eob_max) {
  const simde__m256i coeff = load_tran_low_avx2_simde(coeff_ptr);
  const simde__m256i abs_coeff = simde_mm256_abs_epi16(coeff);
  const simde__m256i tmp_rnd = simde_mm256_adds_epi16(abs_coeff, *round);
  const simde__m256i abs_qcoeff = simde_mm256_mulhi_epi16(tmp_rnd, *quant);
  const simde__m256i qcoeff = simde_mm256_sign_epi16(abs_qcoeff, coeff);
  const simde__m256i abs_dqcoeff =
      simde_mm256_srli_epi16(simde_mm256_mullo_epi16(abs_qcoeff, *dequant), 1);
  const simde__m256i dqcoeff = simde_mm256_sign_epi16(abs_dqcoeff, coeff);
  const simde__m256i nz_mask =
      simde_mm256_cmpgt_epi16(abs_qcoeff, simde_mm256_setzero_si256());
  store_tran_low_avx2_simde(qcoeff, qcoeff_ptr);
  store_tran_low_avx2_simde(dqcoeff, dqcoeff_ptr);

  *eob_max = get_max_lane_eob(iscan_ptr, *eob_max, nz_mask);
  (void)thr;
}
#endif

static VPX_FORCE_INLINE void quantize_fp_32x32_16(
    const simde__m256i *round, const simde__m256i *quant, const simde__m256i *dequant,
    const simde__m256i *thr, const tran_low_t *coeff_ptr, const int16_t *iscan_ptr,
    tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr, simde__m256i *eob_max) {
  const simde__m256i coeff = load_tran_low_avx2_simde(coeff_ptr);
  const simde__m256i abs_coeff = simde_mm256_abs_epi16(coeff);
  const simde__m256i thr_mask = simde_mm256_cmpgt_epi16(abs_coeff, *thr);
  const int32_t nzflag = simde_mm256_movemask_epi8(thr_mask);

  if (nzflag) {
#ifdef MATCH_VP9_QUANTIZE_FP_32X32_C
    const simde__m256i tmp_rnd =
        simde_mm256_and_si256(simde_mm256_adds_epi16(abs_coeff, *round), thr_mask);
#else
    const simde__m256i tmp_rnd = simde_mm256_adds_epi16(abs_coeff, *round);
#endif
    const simde__m256i abs_qcoeff = simde_mm256_mulhi_epi16(tmp_rnd, *quant);
    const simde__m256i qcoeff = simde_mm256_sign_epi16(abs_qcoeff, coeff);
    const simde__m256i abs_dqcoeff =
        simde_mm256_srli_epi16(simde_mm256_mullo_epi16(abs_qcoeff, *dequant), 1);
    const simde__m256i dqcoeff = simde_mm256_sign_epi16(abs_dqcoeff, coeff);
    const simde__m256i nz_mask =
        simde_mm256_cmpgt_epi16(abs_qcoeff, simde_mm256_setzero_si256());
    store_tran_low_avx2_simde(qcoeff, qcoeff_ptr);
    store_tran_low_avx2_simde(dqcoeff, dqcoeff_ptr);

    *eob_max = get_max_lane_eob(iscan_ptr, *eob_max, nz_mask);
  } else {
    store_zero_tran_low_avx2_simde(qcoeff_ptr);
    store_zero_tran_low_avx2_simde(dqcoeff_ptr);
  }
}

void vp9_quantize_fp_32x32_avx2_simde(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                                const struct macroblock_plane *const mb_plane,
                                tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                                const int16_t *dequant_ptr, uint16_t *eob_ptr,
                                const struct ScanOrder *const scan_order) {
  simde__m256i round, quant, dequant, thr;
  simde__m256i eob_max = simde_mm256_setzero_si256();
  const int16_t *iscan = scan_order->iscan;

  coeff_ptr += n_coeffs;
  iscan += n_coeffs;
  qcoeff_ptr += n_coeffs;
  dqcoeff_ptr += n_coeffs;
  n_coeffs = -n_coeffs;

  // Setup global values
  load_fp_values_avx2_simde(mb_plane, &round, &quant, dequant_ptr, &dequant);
  thr = simde_mm256_srli_epi16(dequant, 2);
  quant = simde_mm256_slli_epi16(quant, 1);
  {
    const simde__m256i rnd = simde_mm256_set1_epi16((int16_t)1);
    round = simde_mm256_add_epi16(round, rnd);
    round = simde_mm256_srai_epi16(round, 1);
  }

#ifdef MATCH_VP9_QUANTIZE_FP_32X32_C
  // Subtracting 1 here eliminates a simde_mm256_cmpeq_epi16() instruction when
  // calculating the zbin mask.
  thr = simde_mm256_sub_epi16(thr, simde_mm256_set1_epi16(1));
  quantize_fp_32x32_16(&round, &quant, &dequant, &thr, coeff_ptr + n_coeffs,
                       iscan + n_coeffs, qcoeff_ptr + n_coeffs,
                       dqcoeff_ptr + n_coeffs, &eob_max);
#else
  quantize_fp_32x32_16_no_nzflag(
      &round, &quant, &dequant, &thr, coeff_ptr + n_coeffs, iscan + n_coeffs,
      qcoeff_ptr + n_coeffs, dqcoeff_ptr + n_coeffs, &eob_max);
#endif

  n_coeffs += 8 * 2;

  // remove dc constants
  dequant = simde_mm256_permute2x128_si256(dequant, dequant, 0x31);
  quant = simde_mm256_permute2x128_si256(quant, quant, 0x31);
  round = simde_mm256_permute2x128_si256(round, round, 0x31);
  thr = simde_mm256_permute2x128_si256(thr, thr, 0x31);

  // AC only loop
  while (n_coeffs < 0) {
    quantize_fp_32x32_16(&round, &quant, &dequant, &thr, coeff_ptr + n_coeffs,
                         iscan + n_coeffs, qcoeff_ptr + n_coeffs,
                         dqcoeff_ptr + n_coeffs, &eob_max);
    n_coeffs += 8 * 2;
  }

  *eob_ptr = get_max_eob(eob_max);
}

#if CONFIG_VP9_HIGHBITDEPTH
static VPX_FORCE_INLINE simde__m256i mm256_mul_shift_epi32_logscale(const simde__m256i *x,
                                                               const simde__m256i *y,
                                                               int log_scale) {
  simde__m256i prod_lo = simde_mm256_mul_epi32(*x, *y);
  simde__m256i prod_hi = simde_mm256_srli_epi64(*x, 32);
  const simde__m256i mult_hi = simde_mm256_srli_epi64(*y, 32);
  const simde__m256i mask = simde_mm256_set_epi32(0, -1, 0, -1, 0, -1, 0, -1);
  prod_hi = simde_mm256_mul_epi32(prod_hi, mult_hi);
  prod_lo = simde_mm256_srli_epi64(prod_lo, 16 - log_scale);
  prod_lo = simde_mm256_and_si256(prod_lo, mask);
  prod_hi = simde_mm256_srli_epi64(prod_hi, 16 - log_scale);
  prod_hi = simde_mm256_slli_epi64(prod_hi, 32);
  return simde_mm256_or_si256(prod_lo, prod_hi);
}

static VPX_FORCE_INLINE simde__m256i highbd_init_256(const int16_t *val_ptr) {
  const simde__m128i v = simde_mm_load_si128((const simde__m128i *)val_ptr);
  const simde__m128i zero = simde_mm_setzero_si128();
  const simde__m128i dc = simde_mm_unpacklo_epi16(v, zero);
  const simde__m128i ac = simde_mm_unpackhi_epi16(v, zero);
  return simde_mm256_insertf128_si256(simde_mm256_castsi128_si256(dc), ac, 1);
}

static VPX_FORCE_INLINE void highbd_load_fp_values(
    const struct macroblock_plane *mb_plane, simde__m256i *round, simde__m256i *quant,
    const int16_t *dequant_ptr, simde__m256i *dequant) {
  *round = highbd_init_256(mb_plane->round_fp);
  *quant = highbd_init_256(mb_plane->quant_fp);
  *dequant = highbd_init_256(dequant_ptr);
}

static VPX_FORCE_INLINE simde__m256i highbd_get_max_lane_eob(
    const int16_t *iscan_ptr, simde__m256i eobmax, simde__m256i nz_mask) {
  const simde__m256i packed_nz_mask =
      simde_mm256_packs_epi32(nz_mask, simde_mm256_setzero_si256());
  const simde__m256i packed_nz_mask_perm =
      simde_mm256_permute4x64_epi64(packed_nz_mask, 0xD8);
  const simde__m256i iscan =
      simde_mm256_castsi128_si256(simde_mm_loadu_si128((const simde__m128i *)iscan_ptr));
  const simde__m256i nz_iscan = simde_mm256_and_si256(iscan, packed_nz_mask_perm);
  return simde_mm256_max_epi16(eobmax, nz_iscan);
}

static VPX_FORCE_INLINE void highbd_quantize_fp(
    const simde__m256i *round, const simde__m256i *quant, const simde__m256i *dequant,
    const tran_low_t *coeff_ptr, const int16_t *iscan_ptr,
    tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr, simde__m256i *eob) {
  const simde__m256i coeff = simde_mm256_loadu_si256((const simde__m256i *)coeff_ptr);
  const simde__m256i abs_coeff = simde_mm256_abs_epi32(coeff);
  const simde__m256i tmp_rnd = simde_mm256_add_epi32(abs_coeff, *round);
  const simde__m256i abs_q = mm256_mul_shift_epi32_logscale(&tmp_rnd, quant, 0);
  const simde__m256i abs_dq = simde_mm256_mullo_epi32(abs_q, *dequant);
  const simde__m256i q = simde_mm256_sign_epi32(abs_q, coeff);
  const simde__m256i dq = simde_mm256_sign_epi32(abs_dq, coeff);
  const simde__m256i nz_mask = simde_mm256_cmpgt_epi32(abs_q, simde_mm256_setzero_si256());

  simde_mm256_storeu_si256((simde__m256i *)qcoeff_ptr, q);
  simde_mm256_storeu_si256((simde__m256i *)dqcoeff_ptr, dq);

  *eob = highbd_get_max_lane_eob(iscan_ptr, *eob, nz_mask);
}

void vp9_highbd_quantize_fp_avx2_simde(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                                 const struct macroblock_plane *const mb_plane,
                                 tran_low_t *qcoeff_ptr,
                                 tran_low_t *dqcoeff_ptr,
                                 const int16_t *dequant_ptr, uint16_t *eob_ptr,
                                 const struct ScanOrder *const scan_order) {
  const int step = 8;
  simde__m256i round, quant, dequant;
  simde__m256i eob_max = simde_mm256_setzero_si256();
  const int16_t *iscan = scan_order->iscan;

  coeff_ptr += n_coeffs;
  iscan += n_coeffs;
  qcoeff_ptr += n_coeffs;
  dqcoeff_ptr += n_coeffs;
  n_coeffs = -n_coeffs;

  // Setup global values
  highbd_load_fp_values(mb_plane, &round, &quant, dequant_ptr, &dequant);

  highbd_quantize_fp(&round, &quant, &dequant, coeff_ptr + n_coeffs,
                     iscan + n_coeffs, qcoeff_ptr + n_coeffs,
                     dqcoeff_ptr + n_coeffs, &eob_max);

  n_coeffs += step;

  // remove dc constants
  dequant = simde_mm256_permute2x128_si256(dequant, dequant, 0x31);
  quant = simde_mm256_permute2x128_si256(quant, quant, 0x31);
  round = simde_mm256_permute2x128_si256(round, round, 0x31);

  // AC only loop
  while (n_coeffs < 0) {
    highbd_quantize_fp(&round, &quant, &dequant, coeff_ptr + n_coeffs,
                       iscan + n_coeffs, qcoeff_ptr + n_coeffs,
                       dqcoeff_ptr + n_coeffs, &eob_max);
    n_coeffs += step;
  }

  *eob_ptr = get_max_eob(eob_max);
}

static VPX_FORCE_INLINE void highbd_quantize_fp_32x32(
    const simde__m256i *round, const simde__m256i *quant, const simde__m256i *dequant,
    const simde__m256i *thr, const tran_low_t *coeff_ptr, const int16_t *iscan_ptr,
    tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr, simde__m256i *eob) {
  const simde__m256i coeff = simde_mm256_loadu_si256((const simde__m256i *)coeff_ptr);
  const simde__m256i abs_coeff = simde_mm256_abs_epi32(coeff);
  const simde__m256i thr_mask = simde_mm256_cmpgt_epi32(abs_coeff, *thr);
  const simde__m256i tmp_rnd =
      simde_mm256_and_si256(simde_mm256_add_epi32(abs_coeff, *round), thr_mask);
  const simde__m256i abs_q = mm256_mul_shift_epi32_logscale(&tmp_rnd, quant, 0);
  const simde__m256i abs_dq =
      simde_mm256_srli_epi32(simde_mm256_mullo_epi32(abs_q, *dequant), 1);
  const simde__m256i q = simde_mm256_sign_epi32(abs_q, coeff);
  const simde__m256i dq = simde_mm256_sign_epi32(abs_dq, coeff);
  const simde__m256i nz_mask = simde_mm256_cmpgt_epi32(abs_q, simde_mm256_setzero_si256());

  simde_mm256_storeu_si256((simde__m256i *)qcoeff_ptr, q);
  simde_mm256_storeu_si256((simde__m256i *)dqcoeff_ptr, dq);

  *eob = highbd_get_max_lane_eob(iscan_ptr, *eob, nz_mask);
}

void vp9_highbd_quantize_fp_32x32_avx2_simde(
    const tran_low_t *coeff_ptr, intptr_t n_coeffs,
    const struct macroblock_plane *const mb_plane, tran_low_t *qcoeff_ptr,
    tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr,
    const struct ScanOrder *const scan_order) {
  const int step = 8;
  simde__m256i round, quant, dequant, thr;
  simde__m256i eob_max = simde_mm256_setzero_si256();
  const int16_t *iscan = scan_order->iscan;

  coeff_ptr += n_coeffs;
  iscan += n_coeffs;
  qcoeff_ptr += n_coeffs;
  dqcoeff_ptr += n_coeffs;
  n_coeffs = -n_coeffs;

  // Setup global values
  highbd_load_fp_values(mb_plane, &round, &quant, dequant_ptr, &dequant);
  thr = simde_mm256_srli_epi32(dequant, 2);
  // Subtracting 1 here eliminates a simde_mm256_cmpeq_epi32() instruction when
  // calculating the zbin mask.
  thr = simde_mm256_sub_epi32(thr, simde_mm256_set1_epi32(1));
  quant = simde_mm256_slli_epi32(quant, 1);
  round = simde_mm256_srai_epi32(simde_mm256_add_epi32(round, simde_mm256_set1_epi32(1)), 1);

  highbd_quantize_fp_32x32(&round, &quant, &dequant, &thr, coeff_ptr + n_coeffs,
                           iscan + n_coeffs, qcoeff_ptr + n_coeffs,
                           dqcoeff_ptr + n_coeffs, &eob_max);

  n_coeffs += step;

  // remove dc constants
  dequant = simde_mm256_permute2x128_si256(dequant, dequant, 0x31);
  quant = simde_mm256_permute2x128_si256(quant, quant, 0x31);
  round = simde_mm256_permute2x128_si256(round, round, 0x31);
  thr = simde_mm256_permute2x128_si256(thr, thr, 0x31);

  // AC only loop
  while (n_coeffs < 0) {
    highbd_quantize_fp_32x32(
        &round, &quant, &dequant, &thr, coeff_ptr + n_coeffs, iscan + n_coeffs,
        qcoeff_ptr + n_coeffs, dqcoeff_ptr + n_coeffs, &eob_max);
    n_coeffs += step;
  }

  *eob_ptr = get_max_eob(eob_max);
}
#endif  // CONFIG_VP9_HIGHBITDEPTH