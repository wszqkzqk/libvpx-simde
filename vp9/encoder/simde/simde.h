/*
 *  Copyright (c) 2025 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VP9_ENCODER_SIMDE_H_
#define VP9_ENCODER_SIMDE_H_

#include "vpx_config.h"
#include "vp9/encoder/vp9_block.h"
#include "vp9/common/vp9_scan.h"

#if defined(CONFIG_SIMDE) && CONFIG_SIMDE

#ifdef __cplusplus
extern "C" {
#endif

void vp9_quantize_fp_simde(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                          const struct macroblock_plane *const mb_plane,
                          tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                          const int16_t *dequant_ptr, uint16_t *eob_ptr,
                          const struct ScanOrder *const scan_order);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // CONFIG_SIMDE

#endif  // VP9_ENCODER_SIMDE_H_