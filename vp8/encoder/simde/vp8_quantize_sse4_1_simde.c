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
#include "vpx_ports/bitops.h" /* get_lsb */
#include "vpx_ports/compiler_attributes.h"

// Unsigned shift overflow is disabled for the use of ~1U << eob with ymask.
VPX_NO_UNSIGNED_SHIFT_CHECK void vp8_regular_quantize_b_sse4_1_simde(BLOCK *b,
                                                                     BLOCKD *d) {
  int eob = -1;
  short *zbin_boost_ptr = b->zrun_zbin_boost;
  simde__m128i zbin_boost0 = simde_mm_load_si128((const simde__m128i *)(zbin_boost_ptr));
  simde__m128i zbin_boost1 = simde_mm_load_si128((const simde__m128i *)(zbin_boost_ptr + 8));
  simde__m128i x0, x1, y0, y1, x_minus_zbin0, x_minus_zbin1, dqcoeff0, dqcoeff1;
  simde__m128i quant_shift0 = simde_mm_load_si128((const simde__m128i *)(b->quant_shift));
  simde__m128i quant_shift1 = simde_mm_load_si128((const simde__m128i *)(b->quant_shift + 8));
  simde__m128i z0 = simde_mm_load_si128((const simde__m128i *)(b->coeff));
  simde__m128i z1 = simde_mm_load_si128((const simde__m128i *)(b->coeff + 8));
  simde__m128i zbin_extra = simde_mm_cvtsi32_si128(b->zbin_extra);
  simde__m128i zbin0 = simde_mm_load_si128((const simde__m128i *)(b->zbin));
  simde__m128i zbin1 = simde_mm_load_si128((const simde__m128i *)(b->zbin + 8));
  simde__m128i round0 = simde_mm_load_si128((const simde__m128i *)(b->round));
  simde__m128i round1 = simde_mm_load_si128((const simde__m128i *)(b->round + 8));
  simde__m128i quant0 = simde_mm_load_si128((const simde__m128i *)(b->quant));
  simde__m128i quant1 = simde_mm_load_si128((const simde__m128i *)(b->quant + 8));
  simde__m128i dequant0 = simde_mm_load_si128((const simde__m128i *)(d->dequant));
  simde__m128i dequant1 = simde_mm_load_si128((const simde__m128i *)(d->dequant + 8));
  simde__m128i qcoeff0, qcoeff1, t0, t1, x_shuf0, x_shuf1;
  uint32_t mask, ymask;
  DECLARE_ALIGNED(16, static const uint8_t,
                  zig_zag_mask[16]) = { 0, 1,  4,  8,  5, 2,  3,  6,
                                        9, 12, 13, 10, 7, 11, 14, 15 };
  DECLARE_ALIGNED(16, uint16_t, qcoeff[16]) = { 0 };

  /* Duplicate to all lanes. */
  zbin_extra = simde_mm_shufflelo_epi16(zbin_extra, 0);
  zbin_extra = simde_mm_unpacklo_epi16(zbin_extra, zbin_extra);

  /* x = abs(z) */
  x0 = simde_mm_abs_epi16(z0);
  x1 = simde_mm_abs_epi16(z1);

  /* zbin[] + zbin_extra */
  zbin0 = simde_mm_add_epi16(zbin0, zbin_extra);
  zbin1 = simde_mm_add_epi16(zbin1, zbin_extra);

  /* In C x is compared to zbin where zbin = zbin[] + boost + extra. Rebalance
   * the equation because boost is the only value which can change:
   * x - (zbin[] + extra) >= boost */
  x_minus_zbin0 = simde_mm_sub_epi16(x0, zbin0);
  x_minus_zbin1 = simde_mm_sub_epi16(x1, zbin1);

  /* All the remaining calculations are valid whether they are done now with
   * simd or later inside the loop one at a time. */
  x0 = simde_mm_add_epi16(x0, round0);
  x1 = simde_mm_add_epi16(x1, round1);

  y0 = simde_mm_mulhi_epi16(x0, quant0);
  y1 = simde_mm_mulhi_epi16(x1, quant1);

  y0 = simde_mm_add_epi16(y0, x0);
  y1 = simde_mm_add_epi16(y1, x1);

  /* Instead of shifting each value independently we convert the scaling
   * factor with 1 << (16 - shift) so we can use multiply/return high half. */
  y0 = simde_mm_mulhi_epi16(y0, quant_shift0);
  y1 = simde_mm_mulhi_epi16(y1, quant_shift1);

  /* Restore the sign. */
  y0 = simde_mm_sign_epi16(y0, z0);
  y1 = simde_mm_sign_epi16(y1, z1);

  {
    const simde__m128i zig_zag_i16_0 =
        simde_mm_setr_epi8(0, 1, 2, 3, 8, 9, 14, 15, 10, 11, 4, 5, 6, 7, 12, 13);
    const simde__m128i zig_zag_i16_1 =
        simde_mm_setr_epi8(0, 1, 6, 7, 8, 9, 2, 3, 14, 15, 4, 5, 10, 11, 12, 13);

    /* The first part of the zig zag needs a value
     * from x_minus_zbin1 and vice versa. */
    t1 = simde_mm_alignr_epi8(x_minus_zbin1, x_minus_zbin1, 2);
    t0 = simde_mm_blend_epi16(x_minus_zbin0, t1, 0x80);
    t1 = simde_mm_blend_epi16(t1, x_minus_zbin0, 0x80);
    x_shuf0 = simde_mm_shuffle_epi8(t0, zig_zag_i16_0);
    x_shuf1 = simde_mm_shuffle_epi8(t1, zig_zag_i16_1);
  }

  /* Check if y is nonzero and put it in zig zag order. */
  t0 = simde_mm_packs_epi16(y0, y1);
  t0 = simde_mm_cmpeq_epi8(t0, simde_mm_setzero_si128());
  t0 = simde_mm_shuffle_epi8(t0, simde_mm_load_si128((const simde__m128i *)zig_zag_mask));
  ymask = simde_mm_movemask_epi8(t0) ^ 0xffff;

  for (;;) {
    t0 = simde_mm_cmpgt_epi16(zbin_boost0, x_shuf0);
    t1 = simde_mm_cmpgt_epi16(zbin_boost1, x_shuf1);
    t0 = simde_mm_packs_epi16(t0, t1);
    mask = simde_mm_movemask_epi8(t0);
    mask = ~mask & ymask;
    if (!mask) break;
    /* |eob| will contain the index of the next found element where:
     * boost[i - old_eob - 1] <= x[zigzag[i]] && y[zigzag[i]] != 0 */
    eob = get_lsb(mask);
    /* Need to clear the mask from processed elements so that
     * they are no longer counted in the next iteration. */
    ymask &= ~1U << eob;
    /* It's safe to read ahead of this buffer if struct VP8_COMP has at
     * least 32 bytes before the zrun_zbin_boost_* fields (it has 384).
     * Any data read outside of the buffer is masked by the updated |ymask|. */
    zbin_boost0 = simde_mm_loadu_si128((const simde__m128i *)(zbin_boost_ptr - eob - 1));
    zbin_boost1 = simde_mm_loadu_si128((const simde__m128i *)(zbin_boost_ptr - eob + 7));
    qcoeff[zig_zag_mask[eob]] = 0xffff;
  }

  qcoeff0 = simde_mm_load_si128((const simde__m128i *)(qcoeff));
  qcoeff1 = simde_mm_load_si128((const simde__m128i *)(qcoeff + 8));
  qcoeff0 = simde_mm_and_si128(qcoeff0, y0);
  qcoeff1 = simde_mm_and_si128(qcoeff1, y1);

  simde_mm_store_si128((simde__m128i *)(d->qcoeff), qcoeff0);
  simde_mm_store_si128((simde__m128i *)(d->qcoeff + 8), qcoeff1);

  dqcoeff0 = simde_mm_mullo_epi16(qcoeff0, dequant0);
  dqcoeff1 = simde_mm_mullo_epi16(qcoeff1, dequant1);

  simde_mm_store_si128((simde__m128i *)(d->dqcoeff), dqcoeff0);
  simde_mm_store_si128((simde__m128i *)(d->dqcoeff + 8), dqcoeff1);

  *d->eob = eob + 1;
}