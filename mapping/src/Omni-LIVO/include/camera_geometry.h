#pragma once
#include <vikit/seeker_omni_camera.h>
#include <opencv2/core.hpp>

namespace camera_geometry {
inline const vk::SeekerOmniCamera* raw(const vk::AbstractCamera& c) {
  return dynamic_cast<const vk::SeekerOmniCamera*>(&c);
}
inline bool validRay(const vk::AbstractCamera& c,const Eigen::Vector3d& p) {
  if(auto o=raw(c)) return o->validRay(p);
  return p.allFinite() && p.z()>1e-6;
}
inline bool validPixel(const vk::AbstractCamera& c,const Eigen::Vector2d& p,int margin=0) {
  if(auto o=raw(c)) return o->validPixel(p,margin);
  return p.allFinite() && p.x()>=margin && p.y()>=margin &&
         p.x()<c.width()-margin && p.y()<c.height()-margin;
}
inline double depth(const vk::AbstractCamera& c,const Eigen::Vector3d& p) {
  return raw(c)?p.norm():p.z();
}
inline Eigen::Vector3d atDepth(const vk::AbstractCamera& c,const Eigen::Vector3d& ray,double d) {
  if(!ray.allFinite() || !std::isfinite(d) || d<=0 || (!raw(c) && ray.z()<=1e-9))
    return Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
  return ray*(raw(c)?d:d/ray.z());
}
// Intersect a forward ray with a plane in the same coordinate frame.
inline bool intersect(const Eigen::Vector3d& origin,const Eigen::Vector3d& ray,
                      const Eigen::Vector3d& point,const Eigen::Vector3d& normal,
                      Eigen::Vector3d& result) {
  if(!origin.allFinite() || !ray.allFinite() || !point.allFinite() || !normal.allFinite()) return false;
  const double den=normal.dot(ray);
  if(std::abs(den)<1e-8*normal.norm()*ray.norm() || normal.norm()<1e-12) return false;
  const double distance=normal.dot(point-origin)/den;
  if(!std::isfinite(distance) || distance<=1e-8) return false;
  result=origin+distance*ray;
  return result.allFinite();
}
// Local affine derivative from source pixels to target pixels on a 3D plane.
// Rays are intersected explicitly; no division by camera Z or signed homography scaling.
inline bool planeAffine(const vk::AbstractCamera& source,const vk::AbstractCamera& target,
                        const Eigen::Vector2d& pixel,const Eigen::Vector3d& point,
                        const Eigen::Vector3d& normal,const Eigen::Matrix3d& R,
                        const Eigen::Vector3d& t,double step,Eigen::Matrix2d& A) {
  if(!std::isfinite(step) || step<=0) return false;
  for(int axis=0;axis<2;++axis) {
    Eigen::Vector2d projected[2];
    for(int side=0;side<2;++side) {
      Eigen::Vector2d q=pixel; q[axis]+=(side?1:-1)*step;
      if(!validPixel(source,q)) return false;
      Eigen::Vector3d p;
      if(!intersect(Eigen::Vector3d::Zero(),source.cam2world(q),point,normal,p)) return false;
      Eigen::Vector3d dst=R*p+t;
      if(!validRay(target,dst)) return false;
      projected[side]=target.world2cam(dst);
      if(!validPixel(target,projected[side])) return false;
    }
    A.col(axis)=(projected[1]-projected[0])/(2*step);
  }
  return A.allFinite() && std::abs(A.determinant())>1e-8;
}
// Exact planar warp of a target patch into the source image, preserving wide-FOV geometry.
inline bool planePatch(const vk::AbstractCamera& source,const vk::AbstractCamera& target,
                       const cv::Mat& image,const Eigen::Vector2d& target_pixel,
                       const Eigen::Vector3d& point,const Eigen::Vector3d& normal,
                       const Eigen::Matrix3d& R,const Eigen::Vector3d& t,
                       int half,int levels,std::vector<float>& patch) {
  if(image.type()!=CV_8UC1 || image.cols!=source.width() || image.rows!=source.height() ||
     half<=0 || levels<1 || levels>8) return false;
  const Eigen::Matrix3d inverse=R.transpose();
  const Eigen::Vector3d origin=-inverse*t;
  const int size=2*half;
  const float missing=std::numeric_limits<float>::quiet_NaN();
  patch.assign(size*size*levels,missing);
  for(int level=0;level<levels;++level) {
    int scale=1<<level;
    bool good=validPixel(target,target_pixel,(half+2)*scale);
    for(int y=0;y<size && good;++y) for(int x=0;x<size;++x) {
      Eigen::Vector2d q=target_pixel+Eigen::Vector2d((x-half)*scale,(y-half)*scale);
      Eigen::Vector3d p;
      if(!intersect(origin,inverse*target.cam2world(q),point,normal,p) || !validRay(source,p)) {good=false;break;}
      const Eigen::Vector2d s=source.world2cam(p);
      if(!validPixel(source,s,1)) {good=false;break;}
      const int ix=int(std::floor(s.x())),iy=int(std::floor(s.y()));
      const double dx=s.x()-ix,dy=s.y()-iy;
      patch[level*size*size+y*size+x]=(1-dy)*((1-dx)*image.at<uchar>(iy,ix)+dx*image.at<uchar>(iy,ix+1))
        +dy*((1-dx)*image.at<uchar>(iy+1,ix)+dx*image.at<uchar>(iy+1,ix+1));
    }
    if(!good) {
      std::fill(patch.begin()+level*size*size,patch.begin()+(level+1)*size*size,missing);
      if(level==0) return false;
    }
  }
  return true;
}
}
