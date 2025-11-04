/*
 *  Copyright (c) 2025 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_DSP_SIMDE_SIMDE_COMMON_H_
#define VPX_DSP_SIMDE_SIMDE_COMMON_H_

#include "vpx_dsp/simde/simde_headers.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/vpx_dsp_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Common SIMD operations (avoiding conflicting macros)
static INLINE simde__m128i abs_epi16_simde(simde__m128i x) {
  simde__m128i sign = simde_mm_srai_epi16(x, 15);
  return simde_mm_sub_epi16(simde_mm_xor_si128(x, sign), sign);
}

static INLINE simde__m128i abs_diff_epi16_simde(simde__m128i a, simde__m128i b) {
  simde__m128i diff = simde_mm_sub_epi16(a, b);
  simde__m128i sign = simde_mm_srai_epi16(diff, 15);
  return simde_mm_sub_epi16(simde_mm_xor_si128(diff, sign), sign);
}

// Define useful macros (avoiding duplicates)
#define DECLARE_ALIGNED_SIMDE(alignment, type, name) \
  __attribute__((aligned(alignment))) type name

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_DSP_SIMDE_SIMDE_COMMON_H_