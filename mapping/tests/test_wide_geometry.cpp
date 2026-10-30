#include "camera_geometry.h"
#include "factory_params.h"
#include <iostream>
#include <Eigen/Geometry>
int main() {
 const double* a=factory[0]; const double* b=factory[1];
 vk::SeekerOmniCamera c(a[0],a[1],0.5,a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],a[10]);
 vk::SeekerOmniCamera target(b[0],b[1],0.5,b[2],b[3],b[4],b[5],b[6],b[7],b[8],b[9],b[10]);
 int fail=0,count=0,back=0,cross=0; double max_identity=0,max_warp=0;
 cv::Mat img(c.height(),c.width(),CV_8UC1);
 for(int y=0;y<img.rows;++y) for(int x=0;x<img.cols;++x) img.at<uchar>(y,x)=x%200;
 for(int y=20;y<c.height()-20;y+=13) for(int x=20;x<c.width()-20;x+=13) {
  Eigen::Vector2d px(x+0.25,y+0.3);
  if(!c.validPixel(px,4) || x%200<20 || x%200>180) continue;
  auto ray=c.cam2world(px); Eigen::Vector3d p=ray*3;
  if(!ray.allFinite()) {++fail;continue;}
  ++count; if(p.z()<0) ++back;
  if(!camera_geometry::validRay(c,p)) ++fail;
  if(std::abs(camera_geometry::depth(c,p)-3)>1e-10) ++fail;
  if((camera_geometry::atDepth(c,ray,3)-p).norm()>1e-10) ++fail;
  Eigen::Matrix2d A;
  // Both normal signs must produce the same physical intersection, even with Z<0.
  for(double sign:{-1.0,1.0}) {
   if(!camera_geometry::planeAffine(c,c,px,p,sign*ray,Eigen::Matrix3d::Identity(),Eigen::Vector3d::Zero(),1,A)) {++fail;continue;}
   max_identity=std::max(max_identity,(A-Eigen::Matrix2d::Identity()).norm());
   std::vector<float> patch;
   if(!camera_geometry::planePatch(c,c,img,px,p,sign*ray,Eigen::Matrix3d::Identity(),Eigen::Vector3d::Zero(),2,2,patch)) {++fail;continue;}
   for(int l=0;l<2;++l) for(int iy=0;iy<4;++iy) for(int ix=0;ix<4;++ix) {
    if(!std::isfinite(patch[l*16])) { if(l==0) ++fail; continue; }
    double expected=std::fmod(px.x()+(ix-2)*(1<<l),200);
    max_warp=std::max(max_warp,std::abs(expected-patch[l*16+iy*4+ix]));
   }
  }
  Eigen::Matrix3d R=Eigen::AngleAxisd(0.03,Eigen::Vector3d::UnitZ()).toRotationMatrix();
  Eigen::Vector3d t(0.015,-0.01,0.02), dst=R*p+t;
  Eigen::Vector2d q=target.world2cam(dst);
  if(!camera_geometry::validPixel(target,q,4)) continue;
  Eigen::Matrix2d forward,reverse;
  if(!camera_geometry::planeAffine(c,target,px,p,ray,R,t,0.01,forward)) {++fail;continue;}
  if(!camera_geometry::planeAffine(target,c,q,dst,R*ray,R.transpose(),-R.transpose()*t,0.01,reverse)) {++fail;continue;}
  if((reverse*forward-Eigen::Matrix2d::Identity()).norm()>1e-5) ++fail;
  ++cross;
 }
 // Visible near the 90-degree ray must stay finite and keep its physical range.
 Eigen::Vector3d horizon(1,0,0);
 if(!camera_geometry::validRay(c,horizon) || !c.world2cam(horizon).allFinite() ||
    !c.projectionJacobian(horizon).allFinite() || !camera_geometry::atDepth(c,horizon,2).allFinite()) ++fail;
 Eigen::Vector3d hit;
 if(camera_geometry::intersect(Eigen::Vector3d::Zero(),Eigen::Vector3d::UnitX(),Eigen::Vector3d::UnitZ(),Eigen::Vector3d::UnitZ(),hit)) ++fail;
 if(camera_geometry::intersect(Eigen::Vector3d::Zero(),-Eigen::Vector3d::UnitZ(),Eigen::Vector3d::UnitZ(),Eigen::Vector3d::UnitZ(),hit)) ++fail;
 if(camera_geometry::validPixel(c,Eigen::Vector2d::Constant(std::numeric_limits<double>::quiet_NaN()),2)) ++fail;
 if(c.validPixel(Eigen::Vector2d(0,0),3)) ++fail;
 if(!count || !back || !cross || max_identity>1e-6 || max_warp>1e-4) ++fail;
 std::cout<<"points="<<count<<" negative_Z="<<back<<" cross_camera="<<cross
          <<" identity_error="<<max_identity<<" warp_intensity_error="<<max_warp<<" FAILURES="<<fail<<std::endl;
 return fail?1:0;
}
