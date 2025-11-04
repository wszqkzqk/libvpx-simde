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

#include "./vp9_rtcd.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_dsp/simde/bitdepth_conversion_simde.h"
#include "vp9/common/vp9_scan.h"
#include "vp9/encoder/vp9_block.h"
#include "vpx_dsp/simde/simde_headers.h"

static INLINE void load_fp_values(const struct macroblock_plane *const mb_plane,
                                  simde__m128i *round, simde__m128i *quant,
                                  const int16_t *dequant_ptr,
                                  simde__m128i *dequant) {
  *round = simde_mm_load_si128((const simde__m128i *)mb_plane->round_fp);
  *quant = simde_mm_load_si128((const simde__m128i *)mb_plane->quant_fp);
  *dequant = simde_mm_load_si128((const simde__m128i *)dequant_ptr);
}

void vp9_quantize_fp_simde(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                          const struct macroblock_plane *const mb_plane,
                          tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                          const int16_t *dequant_ptr, uint16_t *eob_ptr,
                          const struct ScanOrder *const scan_order) {
  const simde__m128i zero = simde_mm_setzero_si128();
  simde__m128i thr;
  int nzflag;
  int index = 16;
  simde__m128i round, quant, dequant;
  simde__m128i coeff0, coeff1, coeff0_sign, coeff1_sign;
  simde__m128i qcoeff0, qcoeff1;
  simde__m128i eob;
  const int16_t *iscan = scan_order->iscan;

  // Setup global values.
  load_fp_values(mb_plane, &round, &quant, dequant_ptr, &dequant);

  // Do DC and first 15 AC.
  coeff0 = load_tran_low_simde(coeff_ptr);
  coeff1 = load_tran_low_simde(coeff_ptr + 8);

  // Poor man's abs().
  coeff0_sign = simde_mm_srai_epi16(coeff0, 15);
  coeff1_sign = simde_mm_srai_epi16(coeff1, 15);
  qcoeff0 = invert_sign_simde(coeff0, coeff0_sign);
  qcoeff1 = invert_sign_simde(coeff1, coeff1_sign);

  qcoeff0 = simde_mm_adds_epi16(qcoeff0, round);
  qcoeff0 = simde_mm_mulhi_epi16(qcoeff0, quant);

  round = simde_mm_unpackhi_epi64(round, round);
  quant = simde_mm_unpackhi_epi64(quant, quant);

  qcoeff1 = simde_mm_adds_epi16(qcoeff1, round);
  qcoeff1 = simde_mm_mulhi_epi16(qcoeff1, quant);

  // Reinsert signs.
  qcoeff0 = invert_sign_simde(qcoeff0, coeff0_sign);
  qcoeff1 = invert_sign_simde(qcoeff1, coeff1_sign);

  store_tran_low_simde(qcoeff0, qcoeff_ptr);
  store_tran_low_simde(qcoeff1, qcoeff_ptr + 8);

  qcoeff0 = simde_mm_mullo_epi16(qcoeff0, dequant);
  dequant = simde_mm_unpackhi_epi64(dequant, dequant);
  qcoeff1 = simde_mm_mullo_epi16(qcoeff1, dequant);

  store_tran_low_simde(qcoeff0, dqcoeff_ptr);
  store_tran_low_simde(qcoeff1, dqcoeff_ptr + 8);

  eob = scan_for_eob_simde(&qcoeff0, &qcoeff1, iscan, 0, zero);

  thr = simde_mm_srai_epi16(dequant, 1);

  // AC only loop.
  while (index < n_coeffs) {
    coeff0 = load_tran_low_simde(coeff_ptr + index);
    coeff1 = load_tran_low_simde(coeff_ptr + index + 8);

    // Poor man's abs().
    coeff0_sign = simde_mm_srai_epi16(coeff0, 15);
    coeff1_sign = simde_mm_srai_epi16(coeff1, 15);
    qcoeff0 = invert_sign_simde(coeff0, coeff0_sign);
    qcoeff1 = invert_sign_simde(coeff1, coeff1_sign);

    nzflag = simde_mm_movemask_epi8(simde_mm_cmpgt_epi16(qcoeff0, thr)) |
             simde_mm_movemask_epi8(simde_mm_cmpgt_epi16(qcoeff1, thr));

    if (nzflag) {
      simde__m128i eob0;
      qcoeff0 = simde_mm_adds_epi16(qcoeff0, round);
      qcoeff1 = simde_mm_adds_epi16(qcoeff1, round);
      qcoeff0 = simde_mm_mulhi_epi16(qcoeff0, quant);
      qcoeff1 = simde_mm_mulhi_epi16(qcoeff1, quant);

      // Reinsert signs.
      qcoeff0 = invert_sign_simde(qcoeff0, coeff0_sign);
      qcoeff1 = invert_sign_simde(qcoeff1, coeff1_sign);

      store_tran_low_simde(qcoeff0, qcoeff_ptr + index);
      store_tran_low_simde(qcoeff1, qcoeff_ptr + index + 8);

      qcoeff0 = simde_mm_mullo_epi16(qcoeff0, dequant);
      qcoeff1 = simde_mm_mullo_epi16(qcoeff1, dequant);

      store_tran_low_simde(qcoeff0, dqcoeff_ptr + index);
      store_tran_low_simde(qcoeff1, dqcoeff_ptr + index + 8);

      eob0 = scan_for_eob_simde(&qcoeff0, &qcoeff1, iscan, index, zero);
      eob = simde_mm_max_epi16(eob, eob0);
    } else {
      store_zero_tran_low_simde(qcoeff_ptr + index);
      store_zero_tran_low_simde(qcoeff_ptr + index + 8);

      store_zero_tran_low_simde(dqcoeff_ptr + index);
      store_zero_tran_low_simde(dqcoeff_ptr + index + 8);
    }

    index += 16;
  }

  *eob_ptr = accumulate_eob_simde(eob);
}