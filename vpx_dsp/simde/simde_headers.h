/*
 *  Copyright (c) 2025 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

/*
 * This file provides SIMDE headers for libvpx SIMD implementations.
 * When CONFIG_SIMDE is enabled, this includes SIMDE headers instead of native ones.
 */

#ifndef VPX_DSP_X86_SIMDE_HEADERS_H_
#define VPX_DSP_X86_SIMDE_HEADERS_H_

#include "vpx_config.h"

#if defined(CONFIG_SIMDE) && CONFIG_SIMDE
/* Use SIMDE headers instead of native ones */
#include <simde/x86/sse2.h>
#include <simde/x86/ssse3.h>
#include <simde/x86/avx.h>
#include <simde/x86/avx2.h>
/* Include other required SIMDE headers as needed */
#include <simde/x86/avx512.h>

/* Include custom SIMDE utility headers */
#include "vpx_dsp/simde/quantize_simde.h"
#include "vpx_dsp/simde/bitdepth_conversion_simde.h"
#include "vpx_dsp/simde/txfm_common_simde.h"
#include "vpx_dsp/simde/sad_simde.h"
#include "vpx_dsp/simde/simde_common.h"
#include "vpx_ports/mem.h"
#else
/* Use native headers */
#include <emmintrin.h>   /* SSE2 */
#include <pmmintrin.h>   /* SSSE3 */
#include <tmmintrin.h>   /* SSSE3 */
#include <smmintrin.h>   /* SSE4.1 */
#include <immintrin.h>   /* AVX, AVX2 */
#endif

#endif  // VPX_DSP_X86_SIMDE_HEADERS_H_