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
 * @brief SIMDe common header for SIMD implementations
 *
 * This header provides common SIMDe includes for DSP functions.
 * Use of native aliases is avoided to prevent conflicts with native SIMD.
 */

#ifndef VPX_VPX_DSP_SIMDE_H_
#define VPX_VPX_DSP_SIMDE_H_

#include "vpx_config.h"

#if defined(VPX_SIMD_SIMDE)
#include <simde/simde-features.h>
#include <simde/x86/sse2.h>
#include <simde/x86/ssse3.h>
#include <simde/x86/avx.h>
#include <simde/x86/avx2.h>
#include <simde/x86/avx512.h>

// Define SIMDe-specific helper macros and functions that don't use native aliases

#endif  // VPX_SIMD_SIMDE

#endif  // VPX_VPX_DSP_SIMDE_H_