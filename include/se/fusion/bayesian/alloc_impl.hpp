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
 * 
 * */
#ifndef FUSION_BAYESIAN_ALLOC_H
#define FUSION_BAYESIAN_ALLOC_H
#include "../../utils/math_utils.h"

namespace se {

/* Compute step size based on distance travelled along the ray */ 
static inline float compute_stepsize(const float dist_travelled, const float hf_band,
    const float voxelSize) {
  float new_step;
  float half = hf_band * 0.5f;
  if(dist_travelled < hf_band) new_step = voxelSize;
  else if(dist_travelled < hf_band + half) new_step = 10.f * voxelSize; 
  else new_step = 30.f * voxelSize;
  return new_step;
}

/* Compute octree level given a step size */ 
static inline int step_to_depth(const float step, const int max_depth, 
    const float voxelsize) {
  return static_cast<int>(floorf(std::log2f(voxelsize/step)) + max_depth);
}

template <typename FieldType, 
          template <typename> class OctreeT, typename HashType,
          typename StepF, typename DepthF>
size_t buildOctantList(HashType* allocationList, size_t reserved,
    OctreeT<FieldType>& volume, const Eigen::Matrix4f& pose, 
    const float *depthmap, const Eigen::Vector2i &imageSize, 
    StepF compute_stepsize, DepthF step_to_depth, float mu) {

  const float voxelSize = volume.dim() / volume.size();
  const float inverseVoxelSize = 1 / voxelSize;
  const int size = volume.size();
  const int max_depth = log2(size);
  const int leaves_depth = max_depth - se::math::log2_const(OctreeT<FieldType>::blockSide);

#ifdef _OPENMP
  std::atomic<unsigned int> voxelCount;
  std::atomic<unsigned int> leavesCount;
#else
  unsigned int voxelCount;
#endif

  const Eigen::Vector3f camera = pose.topRightCorner<3, 1>();
  voxelCount = 0;
#pragma omp parallel for
  for (int y = 0; y < imageSize.y(); ++y) {
    for (int x = 0; x < imageSize.x(); ++x) {
      const float depth = depthmap[x + y*imageSize.x()];
      if(depth <= 0)
        continue;
      int tree_depth = max_depth; 
      float stepsize = voxelSize;

      #if 1
      const float band = 4 * mu;
      #else
      const float band = 2 * se::math::clamp(mu * se::math::sq(depth), 2 * voxelSize, 8 * voxelSize);
      #endif

      const Eigen::Vector3f worldVertex = (pose * Eigen::Vector3f((x + 0.5f) * depth, 
            (y + 0.5f) * depth, depth).homogeneous()).head<3>();
      const Eigen::Vector3f direction = (camera - worldVertex).normalized();
      const Eigen::Vector3f origin = worldVertex - (band * 0.5f) * direction;
      const float dist = (camera - origin).norm(); 
      Eigen::Vector3f step = direction*stepsize;

      Eigen::Vector3f voxelPos = origin;
      for(float travelled = 0.f; travelled < dist; travelled += stepsize){
        Eigen::Vector3f voxelScaled = (voxelPos * inverseVoxelSize).array().floor();
        if((voxelScaled.x() < size) && (voxelScaled.y() < size) &&
           (voxelScaled.z() < size) && (voxelScaled.x() >= 0) &&
           (voxelScaled.y() >= 0)   && (voxelScaled.z() >= 0)){
          const Eigen::Vector3i voxel = voxelScaled.cast<int>();
          auto node_ptr = volume.fetch_octant(voxel.x(), voxel.y(), voxel.z(), 
              tree_depth);
          if(!node_ptr){
            const unsigned idx = voxelCount++;
            if(idx < reserved) {
              allocationList[idx] = volume.hash(voxel.x(), voxel.y(), voxel.z(), 
                std::min(tree_depth, leaves_depth));
            }
          } else if(tree_depth >= leaves_depth) { 
            static_cast<se::VoxelBlock<FieldType>*>(node_ptr)->active(true);
          }
        }
        stepsize = compute_stepsize(travelled, band, voxelSize);  
        tree_depth = step_to_depth(stepsize, max_depth, voxelSize);
        
        step = direction*stepsize;
        voxelPos +=step;
      }
    }
  }
  return (size_t)voxelCount >= reserved ? reserved : (size_t)voxelCount;
}

}

#endif
