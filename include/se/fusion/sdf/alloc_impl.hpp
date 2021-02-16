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
#ifndef FUSION_SDF_ALLOC_H
#define FUSION_SDF_ALLOC_H
#include "../../utils/math_utils.h"
#include "../../node.hpp"
#include "../../utils/morton_utils.hpp"

namespace se {

// Given a depth map and camera matrix it computes the list of
// voxels intersected but not allocated by the rays around the measurement m in
// a region comprised between m +/- band.
//  - allocationList output list of keys corresponding to voxel blocks to be allocated
//  - reserved allocated size of allocationList
//  - volume indexing structure used to index voxel blocks
//  - pose camera extrinsics matrix
//  - K camera intrinsics matrix
//  - depthmap input depth map
//  - imageSize dimensions of depthmap
//  - band maximum extent of the allocating region, per ray
template <typename FieldType, template <typename> class OctreeT, typename HashType>
unsigned int
buildAllocationList(HashType *allocationList, size_t reserved,
                    OctreeT<FieldType> &volume, const Eigen::Matrix4f &pose,
                    const float *depthmap, const Eigen::Vector2i &imageSize,
                    const float band) {

  const float voxelSize = volume.dim() / volume.size();
  const float inverseVoxelSize = 1 / voxelSize;
  const unsigned int size = volume.size();
  const unsigned block_scale =
      log2(size) - se::math::log2_const(se::VoxelBlock<FieldType>::side);

#ifdef _OPENMP
  std::atomic<unsigned int> voxelCount;
#else
  unsigned int voxelCount;
#endif

  const Eigen::Vector3f camera = pose.topRightCorner<3, 1>();
  const int numSteps = std::ceil(band * inverseVoxelSize);
  voxelCount = 0;
#pragma omp parallel for
  for (int y = 0; y < imageSize.y(); ++y) {
    for (int x = 0; x < imageSize.x(); ++x) {
      const float depth = depthmap[x + y * imageSize.x()];
      if (depth <= 0)
        continue;
      Eigen::Vector3f worldVertex =
          (pose *
           Eigen::Vector3f((x + 0.5f) * depth, (y + 0.5f) * depth, depth)
               .homogeneous())
              .head<3>();

      Eigen::Vector3f direction = (camera - worldVertex).normalized();
      const Eigen::Vector3f origin = worldVertex - (band * 0.5f) * direction;
      const Eigen::Vector3f step = (direction * band) / numSteps;

      Eigen::Vector3i voxel;
      Eigen::Vector3f voxelPos = origin;
      for (int i = 0; i < numSteps; i++) {
        Eigen::Vector3f voxelScaled =
            (voxelPos * inverseVoxelSize).array().floor();
        if ((voxelScaled.x() < size) && (voxelScaled.y() < size) &&
            (voxelScaled.z() < size) && (voxelScaled.x() >= 0) &&
            (voxelScaled.y() >= 0) && (voxelScaled.z() >= 0)) {
          voxel = voxelScaled.cast<int>();
          se::VoxelBlock<FieldType> *n =
              volume.fetch(voxel.x(), voxel.y(), voxel.z());
          if (!n) {
            HashType k =
                volume.hash(voxel.x(), voxel.y(), voxel.z(), block_scale);
            unsigned int idx = voxelCount++;
            if (idx < reserved) {
              allocationList[idx] = k;
            } else
              break;
          } else {
            n->active(true);
          }
        }
        voxelPos += step;
      }
    }
  }
  const unsigned int written = voxelCount;
  return written >= reserved ? reserved : written;
}

}

#endif
