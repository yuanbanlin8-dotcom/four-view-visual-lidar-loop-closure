#include <vikit/seeker_omni_camera.h>
#include <opencv2/ccalib/omnidir.hpp>
#include <iostream>
#include <algorithm>
#include "factory_params.h"
int main() {
 int failed=0;
 for(int id=0;id<4;++id) for(double scale: {1.0,0.5}) {
  const double* a=factory[id];
  vk::SeekerOmniCamera c(a[0],a[1],scale,a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],a[10]);
  cv::Matx33d K(a[3]*scale,0,(a[5]+0.5)*scale-0.5,0,a[4]*scale,(a[6]+0.5)*scale-0.5,0,0,1);
  cv::Vec4d D(a[7],a[8],a[9],a[10]);
  int valid=0,invalid=0,back=0; double roundtrip=0,derivative=0,reference=0;
  for(int y=0;y<c.height();y+=8) for(int x=0;x<c.width();x+=8) {
   Eigen::Vector2d pixel(x,y); auto ray=c.cam2world(pixel);
   if(!ray.allFinite()) {++invalid; continue;}
   ++valid; if(ray.z()<0) ++back;
   auto p=Eigen::Vector3d(ray*3.2); auto projected=c.world2cam(p);
   roundtrip=std::max(roundtrip,(projected-pixel).norm());
   auto analytic=c.projectionJacobian(p);
   if(!projected.allFinite() || !analytic.allFinite()) { ++failed; continue; }
   for(int j=0;j<3;++j) {
    Eigen::Vector3d plus=p,minus=p; plus[j]+=1e-6; minus[j]-=1e-6;
    Eigen::Vector2d numeric=(c.world2cam(plus)-c.world2cam(minus))/(2e-6);
    if(!numeric.allFinite()) ++failed;
    derivative=std::max(derivative,(numeric-analytic.col(j)).norm());
   }
   std::vector<cv::Point3d> obj={cv::Point3d(p.x(),p.y(),p.z())};
   std::vector<cv::Point2d> out;
   cv::omnidir::projectPoints(obj,out,cv::Vec3d(0,0,0),cv::Vec3d(0,0,0),K,a[2],D);
   reference=std::max(reference,std::hypot(out[0].x-projected.x(),out[0].y-projected.y()));
  }
  std::cout<<"cam="<<id<<" scale="<<scale<<" valid="<<valid<<" invalid="<<invalid
   <<" negative_z="<<back<<" roundtrip_px="<<roundtrip<<" jacobian_error="<<derivative
   <<" opencv_error_px="<<reference<<std::endl;
  if(valid<100 || roundtrip>1e-6 || derivative>1e-4 || reference>1e-8) ++failed;
  if(c.world2cam(Eigen::Vector3d(0,0,0)).allFinite()) ++failed;
  if(c.world2cam(Eigen::Vector3d(0,0,-1)).allFinite()) ++failed;
 }
 // Pinhole limit is independently predictable, with exact analytical derivatives.
 vk::SeekerOmniCamera c(640,480,1,0,300,310,320,240,0,0,0,0);
 Eigen::Vector3d p(1,2,4); Eigen::Vector2d expected(395,395);
 if((c.world2cam(p)-expected).norm()>1e-10) ++failed;
 std::cout<<"FAILURES="<<failed<<std::endl;
 return failed?1:0;
}
