#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <iostream>
#include <cmath>
template <typename Point> bool check() {
  for (int run=0; run<4; ++run) {
    typename pcl::PointCloud<Point>::Ptr input(new pcl::PointCloud<Point>());
    typename pcl::PointCloud<Point>::Ptr output(new pcl::PointCloud<Point>());
    for(int i=0;i<10000+run*1000;++i) {
      Point p{};p.x=i*.01f;p.y=i%31;p.z=i%23;input->push_back(p);
    }
    pcl::VoxelGrid<Point> filter;
    filter.setInputCloud(input);filter.setLeafSize(.1f,.1f,.1f);filter.filter(*output);
    if(output->empty()) return false;
    for(const auto& p:*output) if(!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.z)) return false;
    // Allocation occurs in the prebuilt PCL library; release occurs here.
    // This is the boundary that crashed with the former native-AVX build.
    output.reset();
  }
  return true;
}
int main() {
  std::cout<<"Eigen alignment="<<EIGEN_MAX_ALIGN_BYTES
           <<" malloc_aligned="<<EIGEN_MALLOC_ALREADY_ALIGNED<<std::endl;
  if(!check<pcl::PointXYZINormal>()||!check<pcl::PointXYZRGB>())return 1;
  std::cout<<"PCL allocation/filter/release regression PASS"<<std::endl;
  return 0;
}
