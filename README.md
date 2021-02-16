# supereight-octree: a fast octree library
This is the core library of [supereight](https://github.com/emanuelev/supereight), simplified and completed with volume integration code for both TSDF and Bayesian fusion.

For more details on the library please refer to the author's paper 
[Efficient Octree-Based Volumetric SLAM Supporting Signed-Distance and
Occupancy Mapping.](https://spiral.imperial.ac.uk/bitstream/10044/1/55715/2/EVespaRAL_final.pdf)

If you publish work that relates to this software,
please cite the paper as:

`@ARTICLE{VespaRAL18, 
author={E. Vespa and N. Nikolov and M. Grimm and L. Nardi and P. H. J. Kelly
and S. Leutenegger}, 
journal={IEEE Robotics and Automation Letters}, 
title={Efficient Octree-Based Volumetric SLAM Supporting Signed-Distance and
Occupancy Mapping}, year={2018}, volume={3}, number={2}, pages={1144-1151}, 
doi={10.1109/LRA.2018.2792537}, ISSN={}, month={April}}`

# Licence
The core library is released under the BSD 3-clause Licence. There are part of
the this software that are released under MIT licence, see individual headers
for which licence applies.

# Dependencies
The library is header only, and the following packages are used:
* Eigen3 
* OpenMP (optional)

# Usage example
Given a list of camera poses and the corresponding depth-maps, a mesh can be generated as:
```
#include <se/fusion/bayesian/volume_impl.hpp>
int main() {
  Eigen::AlignedBox3f aabb;
  //TODO: compute scene bounding box
  const Point3f size = aabb.sizes();
  const float dim = std::max(size.x(), std::max(size.y(), size.z()));
  const Eigen::Vector2i depthMapSize(256, 144);
  se::Volume<se::BayesianFusion> volume;
  volume.Init(depthMapSize, -aabb.min(), 512, dim);
  for (int i=0; i<numCameras; ++i) {
    volume.Integrate(cameras[i].K, cameras[i].pose, depthMaps[i].ptr<float>(), cameras[i].timestamp);
  }
  std::vector<Eigen::Vector3f> vertices;
  std::vector<Eigen::Vector3i> faces;
  volume.ExportMesh(vertices, faces);
}
```
