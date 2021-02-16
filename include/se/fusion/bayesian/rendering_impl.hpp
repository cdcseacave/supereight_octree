/*
 *
 * Copyright 2016 Emanuele Vespa, Imperial College London 
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
 *
 * */
#ifndef FUSION_BAYESIAN_RENDERING_HPP
#define FUSION_BAYESIAN_RENDERING_HPP
#include "../../utils/math_utils.h"
#include "../../octree.hpp"
#include <type_traits>

namespace se {

template <typename T>
inline Eigen::Vector4f
raycast(const se::Octree<T> &volume, 
    const Eigen::Vector3f origin, const Eigen::Vector3f direction, 
    const float tnear, const float tfar, const float, const float step, 
    const float) { 

  const float inverseVoxelSize = 1.f / step;
  auto select_occupancy = [](const auto& val){ return val.x; };
  if (tnear < tfar) {
    float t = tnear;
    float stepsize = step;
    float f_t = volume.interp(inverseVoxelSize * (origin + direction * t), select_occupancy);
    float f_tt = 0;

    // if we are not already in it
    if (f_t <= SURF_BOUNDARY) { 
      for (; t < tfar; t += stepsize) {
        const Eigen::Vector3f position = inverseVoxelSize * (origin + direction * t);
        const Eigen::Vector3i scaled_pos = position.cast<int>();
        Volume<OFusion>::value_type data =
            volume.get_fine(scaled_pos.x(), scaled_pos.y(), scaled_pos.z());
        if(data.x > -100.f && data.y > 0.f){
          f_tt = volume.interp(position, select_occupancy);
        }
        if (f_tt > SURF_BOUNDARY) break;
        f_t = f_tt;
      }            
      if (f_tt > SURF_BOUNDARY) {
        // got it, calculate accurate intersection
        t = t - stepsize * (f_tt - SURF_BOUNDARY) / (f_tt - f_t);
        Eigen::Vector4f res = (origin + direction * t).homogeneous();
        res.w() = t;
        return res;
      }
    }
  }
  return Eigen::Vector4f::Constant(0);
}

}

#endif
