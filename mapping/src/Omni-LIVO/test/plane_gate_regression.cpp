#include "voxel_map.h"
#include <iostream>
#include <stdexcept>
int main(int argc,char** argv) {
  ros::init(argc,argv,"plane_gate_regression");ros::Time::init();
  for(V3D n:{V3D(0,0,1),V3D(1,0,0),V3D(.2,.6,.7).normalized().eval()}) {
    VoxelMapConfig cfg{};cfg.max_layer_=0;cfg.sigma_num_=3.;
    std::unordered_map<VOXEL_LOCATION,VoxelOctoTree*> map;
    VoxelMapManager manager(cfg,map);VoxelOctoTree voxel(0,0,5,100,.01);
    auto& plane=*voxel.plane_ptr_;plane.is_plane_=true;plane.center_=V3D(2,3,4);
    plane.normal_=n;plane.d_=-n.dot(plane.center_);plane.radius_=10.;plane.plane_var_.setZero();
    auto accepts=[&](double distance,double variance,double cap) {
      manager.config_setting_.max_point_plane_distance_=cap;
      pointWithVar p;p.point_w=plane.center_+distance*n+0.5*n.unitOrthogonal();p.point_b=p.point_w;
      p.var=M3D::Identity()*variance;p.body_var=p.var;
      PointToPlane match;bool ok=false;double prob=0;
      manager.build_single_residual(p,&voxel,0,ok,prob,match);return ok;
    };
    if(!accepts(1.,1.,0.) || accepts(1.,1.,.2))throw std::runtime_error("large uncertainty admitted another surface");
    if(!accepts(.03,1.,.2))throw std::runtime_error("valid near-surface measurement rejected");
    if(accepts(.03,1e-6,.2))throw std::runtime_error("statistical gate was bypassed");
  }
  std::cout<<"PLANE_GATE cross-surface rejection, valid matches and statistical gate PASS\n";
}
