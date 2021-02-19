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

#ifndef PROJECTIVE_FUNCTOR_HPP
#define PROJECTIVE_FUNCTOR_HPP
#include <functional>
#include <vector>

#include "../utils/math_utils.h"
#include "../algorithms/filter.hpp"
#include "../node.hpp"
#include "../functors/data_handler.hpp"

namespace se {

namespace functor {
  template <typename FieldType, template <typename FieldT> class MapT, typename UpdateF>
  class projective_functor {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    projective_functor(MapT<FieldType>& map, UpdateF f, const Eigen::Matrix4f& pose, 
        const Eigen::Matrix4f& K, const Eigen::Vector2i framesize) : 
      _map(map), _function(f), _pose(pose), _K(K), _frame_size(framesize) {
    } 

    void apply() {
      const float voxel_size = _map.dim()/_map.size();
      {
      /* Predicates definition */
      auto in_frustum_predicate = 
        std::bind(algorithms::in_frustum<se::VoxelBlock<FieldType>>, std::placeholders::_1, 
            voxel_size, _K*_pose, _frame_size); 
      auto is_active_predicate = [](const se::VoxelBlock<FieldType>* b) {
        return b->active();
      };

      /* Retrieve the active list */ 
      std::vector<se::VoxelBlock<FieldType>*> active_list;
      algorithms::filter(active_list, _map.getBlockBuffer(), is_active_predicate,
          in_frustum_predicate);

      const size_t list_size = active_list.size();
#pragma omp parallel for
      for(int64_t i = 0; i < list_size; ++i){
        update_block(active_list[i], voxel_size);
      }
      }
      {
      auto& nodes_list = _map.getNodesBuffer();
      const size_t list_size = nodes_list.size();
#pragma omp parallel for
      for(int64_t i = 0; i < list_size; ++i){
        update_node(nodes_list[i], voxel_size);
      }
      }
    }

    void update_block(se::VoxelBlock<FieldType> * block, const float voxel_size) {
      const Eigen::Vector3i blockCoord = block->coordinates();
      const Eigen::Vector3f delta = _pose.topLeftCorner<3,3>() * Eigen::Vector3f(voxel_size, 0, 0);
      const Eigen::Vector3f cameraDelta = _K.topLeftCorner<3,3>() * delta;

      const unsigned blockSide = se::VoxelBlock<FieldType>::side;
      const unsigned ylast = blockCoord(1) + blockSide;
      const unsigned zlast = blockCoord(2) + blockSide;

      bool is_visible = false;
      for(unsigned z = blockCoord(2); z < zlast; ++z)
        for (unsigned y = blockCoord(1); y < ylast; ++y){
          Eigen::Vector3i pix = Eigen::Vector3i(blockCoord(0), y, z);
          const Eigen::Vector3f start = (_pose * (pix.cast<float>() * voxel_size).homogeneous()).topLeftCorner<3, 1>();
          const Eigen::Vector3f camerastart = _K.topLeftCorner<3,3>() * start;
          for (unsigned x = 0; x < blockSide; ++x){
            const Eigen::Vector3f pos = start + (x*delta);
            if (pos(2) < 0.0001f)
                continue;
            const Eigen::Vector3f camera_voxel = camerastart + (x*cameraDelta);
            const float inverse_depth = 1.f / camera_voxel(2);
            const Eigen::Vector2f pixel = Eigen::Vector2f(
                camera_voxel(0) * inverse_depth + 0.5f,
                camera_voxel(1) * inverse_depth + 0.5f);
            if (pixel(0) < 0.5f || pixel(0) > _frame_size(0) - 1.5f || 
                pixel(1) < 0.5f || pixel(1) > _frame_size(1) - 1.5f)
                continue;
            is_visible = true;
            pix(0) = x + blockCoord(0); 
            VoxelBlockHandler<FieldType> handler = {block, pix};
            _function(handler, pos, pixel);
          }
        }
      block->active(is_visible);
    }

    void update_node(se::Node<FieldType>* node, const float voxel_size) { 
      const Eigen::Vector3i voxel = Eigen::Vector3i(unpack_morton(node->code_));
      const Eigen::Vector3f delta = _pose.topLeftCorner<3,3>() * Eigen::Vector3f::Constant(0.5f * voxel_size * node->side_);
      const Eigen::Vector3f delta_c = _K.topLeftCorner<3,3>() * delta;
      const Eigen::Vector3f base_cam = (_pose * (voxel_size * voxel.cast<float>()).homogeneous()).topLeftCorner<3, 1>();
      const Eigen::Vector3f basepix_hom = _K.topLeftCorner<3,3>() * base_cam;

      for(int i = 0; i < 8; ++i) {
        const Eigen::Vector3i dir =  Eigen::Vector3i((i & 1) > 0, (i & 2) > 0, (i & 4) > 0);
        const Eigen::Vector3f vox_cam = base_cam + dir.cast<float>().cwiseProduct(delta); 
        if (vox_cam(2) < 0.0001f)
          continue;
        const Eigen::Vector3f pix_hom = basepix_hom + dir.cast<float>().cwiseProduct(delta_c); 
        const float inverse_depth = 1.f / pix_hom(2);
        const Eigen::Vector2f pixel = Eigen::Vector2f(
            pix_hom(0) * inverse_depth + 0.5f,
            pix_hom(1) * inverse_depth + 0.5f);
        if (pixel(0) < 0.5f || pixel(0) > _frame_size(0) - 1.5f || 
            pixel(1) < 0.5f || pixel(1) > _frame_size(1) - 1.5f)
          continue;
        NodeHandler<FieldType> handler = {node, i};
        _function(handler, vox_cam, pixel);
      }
    }

  private:
    MapT<FieldType>& _map; 
    UpdateF _function; 
    Eigen::Matrix4f _pose;
    Eigen::Matrix4f _K;
    Eigen::Vector2i _frame_size;
  };

  template <typename FieldType, template <typename FieldT> class MapT, typename UpdateF>
  void projective_map(MapT<FieldType>& map, const Eigen::Matrix4f& pose, 
          const Eigen::Matrix4f& K, const Eigen::Vector2i framesize,
          UpdateF funct) {
    projective_functor<FieldType, MapT, UpdateF>(map, funct, pose, K, framesize).apply();
  }
}

}

#endif
