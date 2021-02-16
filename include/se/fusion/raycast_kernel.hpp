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
#ifndef FUSION_SDF_REYCASTKERNEL_HPP
#define FUSION_SDF_REYCASTKERNEL_HPP
#include <se/utils/math_utils.h> 
#include <type_traits>

namespace se {

template <typename T>
void raycastKernel(const se::Octree<T> &volume,
                   se::Image<Eigen::Vector3f> &vertex,
                   se::Image<Eigen::Vector3f> &normal,
                   const Eigen::Matrix4f &view, const float nearPlane,
                   const float farPlane, const float mu) {
  const float INVALID = -2.f;
  const float step = volume.dim() / volume.size();
  const float inverseVoxelSize = 1.f / step;
  const float largestep = step * BLOCK_SIDE;
  int y;
#pragma omp parallel for shared(normal, vertex), private(y)
  for (y = 0; y < vertex.height(); y++)
    for (int x = 0; x < vertex.width(); x++) {

      Eigen::Vector2i pos(x, y);
      const Eigen::Vector3f dir =
          (view.topLeftCorner<3, 3>() * Eigen::Vector3f(x, y, 1.f))
              .normalized();
      const Eigen::Vector3f transl = view.topRightCorner<3, 1>();
      se::ray_iterator<T> ray(volume, transl, dir, nearPlane,
                              farPlane);
      ray.next();
      const float t_min =
          ray.tcmin(); /* Get distance to the first intersected block */
      const Eigen::Vector4f hit = t_min > 0.f
                                      ? raycast(volume, transl, dir, t_min,
                                                ray.tmax(), mu, step, largestep)
                                      : Eigen::Vector4f::Constant(0.f);
      if (hit.w() > 0.0) {
        vertex[x + y * vertex.width()] = hit.head<3>();
        Eigen::Vector3f surfNorm =
            volume.grad(inverseVoxelSize * hit.head<3>(),
                        [](const auto &val) { return val.x; });
        if (surfNorm.norm() == 0) {
          // normal[pos] = normalize(surfNorm); // APN added
          normal[pos.x() + pos.y() * normal.width()] =
              Eigen::Vector3f(INVALID, 0, 0);
        } else {
          // Invert normals if SDF
          normal[pos.x() + pos.y() * normal.width()] =
              std::is_same<T, SDF>::value ? (-1.f * surfNorm).normalized()
                                          : surfNorm.normalized();
        }
      } else {
        vertex[pos.x() + pos.y() * vertex.width()] =
            Eigen::Vector3f::Constant(0);
        normal[pos.x() + pos.y() * normal.width()] =
            Eigen::Vector3f(INVALID, 0, 0);
      }
    }
}

}

#endif
