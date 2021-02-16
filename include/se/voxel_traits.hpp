/*

Copyright 2016 Emanuele Vespa, Imperial College London 

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 

*/

#ifndef _VOXEL_TRAITS_
#define _VOXEL_TRAITS_
#include "utils/math_utils.h"

#ifndef SE_NEARPLANE
#define SE_NEARPLANE 0.1f
#define SE_FARPLANE 10.0f
#endif

namespace se {

template <class VoxelTraits>
struct voxel_traits{ };

/******************************************************************************
*
* KFusion Truncated Signed Distance Function voxel traits
*
****************************************************************************/

typedef struct {
	float x;
	float y;
} SDF;

template <> struct voxel_traits<SDF> {
	typedef SDF value_type;
	static inline value_type empty() { return {1.f, -1.f}; }
	static inline value_type initValue() { return {1.f, 0.f}; }
};

/******************************************************************************
*
* Bayesian Fusion voxel traits and algorithm specific defines
*
****************************************************************************/

typedef struct {
	float x;
	double y;
} BayesianFusion;

template <> struct voxel_traits<BayesianFusion> {
	typedef BayesianFusion value_type;
	static inline value_type empty() { return {0.f, 0.f}; }
	static inline value_type initValue() { return {0.f, 0.f}; }
};

}

#endif
