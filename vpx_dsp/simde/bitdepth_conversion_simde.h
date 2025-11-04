/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_DSP_SIMDE_BITDEPTH_CONVERSION_SIMDE_H_
#define VPX_DSP_SIMDE_BITDEPTH_CONVERSION_SIMDE_H_

#include "vpx_dsp/simde/simde_headers.h"
#if defined(CONFIG_SIMDE) && CONFIG_SIMDE
#define CONVERT_TO_SHORTPTR(x) ((uint16_t *)(((uintptr_t)(x)) << 1))
#else
#define CONVERT_TO_SHORTPTR(x) ((uint16_t *)(((uintptr_t)(x)) << 1))
#endif

#endif  // VPX_DSP_SIMDE_BITDEPTH_CONVERSION_SIMDE_H_