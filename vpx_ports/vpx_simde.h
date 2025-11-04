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
 * @brief SIMDe support and runtime detection for cross-platform SIMD
 *
 * This file provides headers and functions for SIMDe (SIMD Everywhere) library,
 * which offers portable SIMD implementations across architectures without
 * native SIMD support.
 */

#ifndef VPX_VPX_PORTS_VPX_SIMDE_H_
#define VPX_VPX_PORTS_VPX_SIMDE_H_

#include "vpx_config.h"
#include "vpx/vpx_integer.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Check if SIMDe support is compiled in
 */
#if defined(CONFIG_SIMDE) && defined(VPX_SIMD_SIMDE)
#define VPX_SIMDE_AVAILABLE 1

#include <simde/simde-features.h>
#include <simde/x86/sse2.h>
#include <simde/x86/ssse3.h>
#include <simde/x86/avx.h>
#include <simde/x86/avx2.h>

// Include SIMDe without native aliases to avoid conflicts with native SIMD
// Do not define SIMDE_ENABLE_NATIVE_ALIASES

// When SIMDE is enabled, disable the native SIMD implementations for cross-platform compatibility
#if defined(VPX_SIMD_SIMDE)
#undef HAVE_MMX
#undef HAVE_SSE
#undef HAVE_SSE2
#undef HAVE_SSE3
#undef HAVE_SSSE3
#undef HAVE_SSE4_1
#undef HAVE_SSE4_2
#undef HAVE_AVX
#undef HAVE_AVX2
#undef HAVE_AVX512
#endif

#else // CONFIG_SIMDE not defined
#define VPX_SIMDE_AVAILABLE 0
#endif // CONFIG_SIMDE

/**
 * Check if SIMDe should be used instead of native SIMD implementations.
 * This is typically when native SIMD is not available but SIMDe is enabled.
 */
static INLINE int vpx_use_simde(void) {
#if defined(CONFIG_SIMDE) && defined(VPX_SIMD_SIMDE)
    // When SIMDe is configured, it should be used in preference to native implementations
    return 1;
#else
    return 0; // SIMDe not available
#endif
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VPX_PORTS_VPX_SIMDE_H_