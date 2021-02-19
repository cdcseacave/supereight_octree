#ifndef SE_VOLUME_SDF_IMPL_H
#define SE_VOLUME_SDF_IMPL_H
#include "alloc_impl.hpp"
#include "mapping_impl.hpp"
#include "rendering_impl.hpp"
#include "../../functors/projective_functor.hpp"
#include "../volume.hpp"

namespace se {

using Volume = TVolume<BayesianFusion>;

template <>
void Volume::Integrate(const Eigen::Matrix4f &K, const Eigen::Matrix4f &_pose,
                       const float *depthMap, float timestamp, float mu) {
  const Eigen::Matrix4f pose = _pose * globalTranslation.inverse();
  const float voxelSize = volume.dim() / volume.size();
  const int num_vox_per_pix =
      volume.dim() / ((se::VoxelBlock<BayesianFusion>::side) * voxelSize);
  allocations.reserve(num_vox_per_pix * imageSize.x() * imageSize.y());

  unsigned allocated = buildOctantList(allocations.data(), allocations.capacity(),
      volume, (K * pose).inverse(), depthMap, imageSize,
      compute_stepsize, step_to_depth, mu);
  volume.allocate(allocations.data(), allocated);

  struct bfusion_update funct(depthMap, imageSize, mu, timestamp, voxelSize);
  functor::projective_map(volume, pose, K, imageSize, funct);
}

template <>
void Volume::ExportMesh(std::vector<Eigen::Vector3f>& vertices,
                        std::vector<Eigen::Vector3i>& faces) const {
  ExportMeshImpl<true>(vertices, faces);
}

}

#endif
