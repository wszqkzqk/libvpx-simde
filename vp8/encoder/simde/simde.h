/*
 *  Copyright (c) 2025 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VP8_ENCODER_SIMDE_H_
#define VP8_ENCODER_SIMDE_H_

#include "vpx_config.h"
#include "vp8/encoder/block.h"
#include "vp8/encoder/denoising.h"

#if defined(CONFIG_SIMDE) && CONFIG_SIMDE

#ifdef __cplusplus
extern "C" {
#endif

void vp8_regular_quantize_b_simde(BLOCK *b, BLOCKD *d);
void vp8_fast_quantize_b_simde(BLOCK *b, BLOCKD *d);
int vp8_denoiser_filter_simde(unsigned char *mc_running_avg_y,
                             int mc_avg_y_stride, unsigned char *running_avg_y,
                             int avg_y_stride, unsigned char *sig,
                             int sig_stride, unsigned int motion_magnitude,
                             int increase_denoising);
int vp8_denoiser_filter_uv_simde(unsigned char *mc_running_avg,
                                int mc_avg_stride, unsigned char *running_avg,
                                int avg_stride, unsigned char *sig,
                                int sig_stride, unsigned int motion_magnitude,
                                int increase_denoising);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // CONFIG_SIMDE

#endif  // VP8_ENCODER_SIMDE_H_