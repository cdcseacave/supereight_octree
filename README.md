# supereight-octree: a fast octree library
This is the core library of [supereight](https://github.com/emanuelev/supereight).

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
The following packages are required to build the library:
* CMake >= 3.10
* Eigen3 
* Sophus
* OpenMP (optional)
