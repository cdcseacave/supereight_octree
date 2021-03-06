#ifndef SE_VOLUME_SDF_IMPL_H
#define SE_VOLUME_SDF_IMPL_H
#include "alloc_impl.hpp"
#include "mapping_impl.hpp"
#include "rendering_impl.hpp"
#include "../../functors/projective_functor.hpp"
#include "../volume.hpp"

namespace se {

using Volume = TVolume<SDF>;

template <>
void Volume::Integrate(const Eigen::Matrix4f &K, const Eigen::Matrix4f &_pose,
                       const float *depthMap, float /*timestamp*/, float mu) {
  const Eigen::Matrix4f pose = _pose * globalTranslation.inverse();
  const float voxelSize = volume.dim() / volume.size();
  const int num_vox_per_pix =
      volume.dim() / ((se::VoxelBlock<SDF>::side) * voxelSize);
  allocations.reserve(num_vox_per_pix * imageSize.x() * imageSize.y());

  unsigned allocated =
      buildAllocationList(allocations.data(), allocations.capacity(), volume,
                          (K * pose).inverse(), depthMap, imageSize, 2 * mu);
  volume.allocate(allocations.data(), allocated);

  struct sdf_update funct(depthMap, imageSize, mu, 100);
  functor::projective_map(volume, pose, K, imageSize, funct);
}

template <>
void Volume::ExportMesh(std::vector<Eigen::Vector3f>& vertices,
                        std::vector<Eigen::Vector3i>& faces) const {
  ExportMeshImpl<false>(vertices, faces);
}

}

#endif
