/*
 *  Copyright (c) 2025 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

/**
 * @file
 * @brief SIMDe-accelerated forward transform functions header
 *
 * This file declares the forward transform implementations using SIMDe library
 * for cross-platform SIMD compatibility.
 */

#ifndef VPX_VPX_DSP_SIMDE_FWD_TXFM_SIMDE_H_
#define VPX_VPX_DSP_SIMDE_FWD_TXFM_SIMDE_H_

#include "./vpx_config.h"
#include "vpx/vpx_integer.h"

#if defined(CONFIG_SIMDE) && CONFIG_SIMDE

#ifdef __cplusplus
extern "C" {
#endif

// Forward DCT declarations using SIMDe
void vpx_fdct4x4_1_simde(const int16_t *input, tran_low_t *output, int stride);
void vpx_fdct8x8_1_simde(const int16_t *input, tran_low_t *output, int stride);
void vpx_fdct16x16_1_simde(const int16_t *input, tran_low_t *output, int stride);
void vpx_fdct32x32_1_simde(const int16_t *input, tran_low_t *output, int stride);

// Full DCT implementations using SIMDe
void vpx_fdct4x4_simde(const int16_t *input, tran_low_t *output, int stride);
void vpx_fdct8x8_simde(const int16_t *input, tran_low_t *output, int stride);
void vpx_fdct16x16_simde(const int16_t *input, tran_low_t *output, int stride);
void vpx_fdct32x32_rd_simde(const int16_t *input, tran_low_t *output, int stride);
void vpx_fdct32x32_simde(const int16_t *input, tran_low_t *output, int stride);

#if CONFIG_VP9_HIGHBITDEPTH
void vpx_highbd_fdct4x4_simde(const int16_t *input, tran_low_t *output, int stride);
void vpx_highbd_fdct8x8_simde(const int16_t *input, tran_low_t *output, int stride);
void vpx_highbd_fdct16x16_simde(const int16_t *input, tran_low_t *output, int stride);
void vpx_highbd_fdct32x32_rd_simde(const int16_t *input, tran_low_t *output, int stride);
void vpx_highbd_fdct32x32_simde(const int16_t *input, tran_low_t *output, int stride);
#endif  // CONFIG_VP9_HIGHBITDEPTH

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // CONFIG_SIMDE
#endif  // VPX_VPX_DSP_SIMDE_FWD_TXFM_SIMDE_H_