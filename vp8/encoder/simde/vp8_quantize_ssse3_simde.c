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
#include "./vp8_rtcd.h"
#include "vp8/encoder/block.h"
#include "vpx_ports/bitops.h" /* get_msb */

void vp8_fast_quantize_b_ssse3_simde(BLOCK *b, BLOCKD *d) {
  int eob, mask;

  simde__m128i z0 = simde_mm_load_si128((const simde__m128i *)(b->coeff));
  simde__m128i z1 = simde_mm_load_si128((const simde__m128i *)(b->coeff + 8));
  simde__m128i round0 = simde_mm_load_si128((const simde__m128i *)(b->round));
  simde__m128i round1 = simde_mm_load_si128((const simde__m128i *)(b->round + 8));
  simde__m128i quant_fast0 = simde_mm_load_si128((const simde__m128i *)(b->quant_fast));
  simde__m128i quant_fast1 = simde_mm_load_si128((const simde__m128i *)(b->quant_fast + 8));
  simde__m128i dequant0 = simde_mm_load_si128((const simde__m128i *)(d->dequant));
  simde__m128i dequant1 = simde_mm_load_si128((const simde__m128i *)(d->dequant + 8));

  simde__m128i sz0, sz1, x, x0, x1, y0, y1, zeros, abs0, abs1;

  DECLARE_ALIGNED(16, const uint8_t,
                  pshufb_zig_zag_mask[16]) = { 0, 1,  4,  8,  5, 2,  3,  6,
                                               9, 12, 13, 10, 7, 11, 14, 15 };
  simde__m128i zig_zag = simde_mm_load_si128((const simde__m128i *)pshufb_zig_zag_mask);

  /* sign of z: z >> 15 */
  sz0 = simde_mm_srai_epi16(z0, 15);
  sz1 = simde_mm_srai_epi16(z1, 15);

  /* x = abs(z) */
  x0 = simde_mm_abs_epi16(z0);
  x1 = simde_mm_abs_epi16(z1);

  /* x += round */
  x0 = simde_mm_add_epi16(x0, round0);
  x1 = simde_mm_add_epi16(x1, round1);

  /* y = (x * quant) >> 16 */
  y0 = simde_mm_mulhi_epi16(x0, quant_fast0);
  y1 = simde_mm_mulhi_epi16(x1, quant_fast1);

  /* ASM saves Y for EOB */
  /* I think we can ignore that because adding the sign doesn't change anything
   * and multiplying 0 by dequant is OK as well */
  abs0 = y0;
  abs1 = y1;

  /* Restore the sign bit. */
  y0 = simde_mm_xor_si128(y0, sz0);
  y1 = simde_mm_xor_si128(y1, sz1);
  x0 = simde_mm_sub_epi16(y0, sz0);
  x1 = simde_mm_sub_epi16(y1, sz1);

  /* qcoeff = x */
  simde_mm_store_si128((simde__m128i *)(d->qcoeff), x0);
  simde_mm_store_si128((simde__m128i *)(d->qcoeff + 8), x1);

  /* x * dequant */
  x0 = simde_mm_mullo_epi16(x0, dequant0);
  x1 = simde_mm_mullo_epi16(x1, dequant1);

  /* dqcoeff = x * dequant */
  simde_mm_store_si128((simde__m128i *)(d->dqcoeff), x0);
  simde_mm_store_si128((simde__m128i *)(d->dqcoeff + 8), x1);

  zeros = simde_mm_setzero_si128();

  x0 = simde_mm_cmpgt_epi16(abs0, zeros);
  x1 = simde_mm_cmpgt_epi16(abs1, zeros);

  x = simde_mm_packs_epi16(x0, x1);

  x = simde_mm_shuffle_epi8(x, zig_zag);

  mask = simde_mm_movemask_epi8(x);

  /* x2 is needed to increase the result from non-zero masks by 1,
   * +1 is needed to mask undefined behavior for a null argument,
   * the result of get_msb(1) is 0 */
  eob = get_msb(mask * 2 + 1);

  *d->eob = eob;
}