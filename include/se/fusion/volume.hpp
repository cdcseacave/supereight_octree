#ifndef SE_VOLUME_H
#define SE_VOLUME_H
#include "../utils/math_utils.h"
#include "../voxel_traits.hpp"
#include "../octree.hpp"
#include "../ray_iterator.hpp"
#include "../image/image.hpp"
#include "../algorithms/meshing.hpp"
#include "raycast_kernel.hpp"

namespace se {

template <typename FieldType>
class Volume {
public:
  // Initializes the octree attributes
  //  - size number of voxels per side of the cube
  //  - dim cube extension per side, in meter
  void Init(const Eigen::Vector2i &_imageSize, const Eigen::Vector3f &trans,
            int size, float dim);

  void Integrate(const Eigen::Matrix4f &K, const Eigen::Matrix4f &pose,
                 const float *depthmap, float timestamp = 0, float mu = 0.1f);
  void Render(const Eigen::Matrix4f &K, const Eigen::Matrix4f &pose,
              float mu = 0.1f);

  void ExportMesh(std::vector<Eigen::Vector3f>& vertices,
                  std::vector<Eigen::Vector3i>& faces) const;

private:
  template <bool invert>
  void ExportMeshImpl(std::vector<Eigen::Vector3f>& vertices,
                      std::vector<Eigen::Vector3i>& faces) const;

public:
  Eigen::Vector2i imageSize;
  Eigen::Matrix4f globalTranslation;

  se::Octree<FieldType> volume;
  std::vector<se::key_t> allocations;

  se::Image<Eigen::Vector3f> vertices;
  se::Image<Eigen::Vector3f> normals;
};


template <typename FieldType>
void Volume<FieldType>::Init(const Eigen::Vector2i &_imageSize, const Eigen::Vector3f &trans,
                             int size, float dim) {
  imageSize = _imageSize;
  globalTranslation = Eigen::Matrix4f::Identity();
  globalTranslation.topRightCorner<3, 1>() = trans;
  const int numVoxels = std::pow(2, std::round(std::log2(size)));
  volume.init(numVoxels, dim);
  vertices.resize(imageSize.x(), imageSize.y());
  normals.resize(imageSize.x(), imageSize.y());
}

template <typename FieldType>
void Volume<FieldType>::Render(const Eigen::Matrix4f &K, const Eigen::Matrix4f &_pose, float mu) {
  const Eigen::Matrix4f pose = _pose * globalTranslation.inverse();
  raycastKernel(volume, vertices, normals, (K * pose).inverse(),
      SE_NEARPLANE, SE_FARPLANE, mu);
}

template <typename FieldType>
template <bool invert>
void Volume<FieldType>::ExportMeshImpl(std::vector<Eigen::Vector3f>& vertices,
                                       std::vector<Eigen::Vector3i>& faces) const {
  struct Triangle {
    Eigen::Vector3f vertexes[3];
  };
  std::vector<Triangle> triangles;
  auto inside = [](const se::Octree<FieldType>::value_type &val) {
    // meshing::status code;
    // if(val.y == 0.f)
    //   code = meshing::status::UNKNOWN;
    // else
    //   code = val.x < 0.f ? meshing::status::INSIDE :
    //   meshing::status::OUTSIDE;
    // return code;
    return val.x < 0.f;
  };

  auto select = [](const se::Octree<FieldType>::value_type &val) {
    return val.x;
  };

  se::algorithms::marching_cube(volume, select, inside, triangles);

  std::unordered_map<Eigen::Vector3f, uint32_t> mapVertices;
  for (const Triangle &t : triangles) {
    Eigen::Vector3i face;
    for (int i = 0; i < 3; ++i) {
      auto ret = mapVertices.emplace(t.vertexes[i], vertices.size());
      if (ret.second)
        vertices.emplace_back(t.vertexes[i]);
      face[invert ? 2-i : i] = ret.first->second;
    }
    faces.emplace_back(face);
  }
}

}

#endif
