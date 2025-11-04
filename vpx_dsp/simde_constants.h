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
 * @brief SIMDe constants for cross-platform SIMD implementation
 *
 * This file defines constants and functions for runtime detection of SIMDe
 * support, which provides portable SIMD implementations when native SIMD
 * instructions are unavailable.
 */

#ifndef VPX_DSP_SIMDE_CONSTANTS_H_
#define VPX_DSP_SIMDE_CONSTANTS_H_

#include "vpx/vpx_integer.h"

#ifdef __cplusplus
extern "C" {
#endif

// SIMDe capability flags
typedef enum {
  SIMDE_CAPS_SSE = 0x01,      /**< SIMDe SSE support */
  SIMDE_CAPS_SSE2 = 0x02,     /**< SIMDe SSE2 support */
  SIMDE_CAPS_SSE3 = 0x04,     /**< SIMDe SSE3 support */
  SIMDE_CAPS_SSSE3 = 0x08,    /**< SIMDe SSSE3 support */
  SIMDE_CAPS_SSE4_1 = 0x10,   /**< SIMDe SSE4.1 support */
  SIMDE_CAPS_SSE4_2 = 0x20,   /**< SIMDe SSE4.2 support */
  SIMDE_CAPS_AVX = 0x40,      /**< SIMDe AVX support */
  SIMDE_CAPS_AVX2 = 0x80,     /**< SIMDe AVX2 support */
  SIMDE_CAPS_ALL = 0xFF       /**< All SIMDe capabilities */
} SIMDECaps;

/**
 * Checks if SIMDe is supported and available on the current system
 * @return 1 if SIMDe is available, 0 otherwise
 */
static INLINE int vpx_check_simde_support(void) {
#ifdef VPX_USE_SIMDE
  return 1;
#else
  return 0;
#endif
}

/**
 * Gets the SIMDe capability flags available on this system
 * @return Bitmask of available SIMDe capabilities
 */
static INLINE int vpx_get_simde_caps(void) {
  int caps = 0;
#ifdef VPX_USE_SIMDE
  // When SIMDe is enabled and no native SIMD is available, 
  // SIMDe provides equivalent functionality
  caps |= SIMDE_CAPS_SSE2;   // SIMDe provides SSE2-equivalent functionality
  caps |= SIMDE_CAPS_SSE;    // SIMDe provides SSE-equivalent functionality
  caps |= SIMDE_CAPS_SSSE3;  // SIMDe provides SSSE3-equivalent functionality
  caps |= SIMDE_CAPS_SSE4_1; // SIMDe provides SSE4.1-equivalent functionality
  caps |= SIMDE_CAPS_AVX;    // SIMDe provides AVX-equivalent functionality
  caps |= SIMDE_CAPS_AVX2;   // SIMDe provides AVX2-equivalent functionality
#endif
  return caps;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_DSP_SIMDE_CONSTANTS_H_