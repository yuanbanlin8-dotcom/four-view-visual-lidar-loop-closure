#pragma once
#include <vikit/abstract_camera.h>
#include <Eigen/LU>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace vk {
// Unified omnidirectional projection with Brown radial/tangential distortion.
// Parameter order follows Kalibr: xi, fx, fy, cx, cy; k1,k2,p1,p2.
class SeekerOmniCamera : public AbstractCamera {
  std::vector<int> invalid_integral_;
  double xi_, fx_, fy_, cx_, cy_, k1_, k2_, p1_, p2_;
  static double nan() { return std::numeric_limits<double>::quiet_NaN(); }
  Vector2d distort(const Vector2d& q, Matrix2d* J=nullptr) const {
    const double x=q.x(), y=q.y(), r2=q.squaredNorm();
    const double a=1+k1_*r2+k2_*r2*r2;
    if(J) {
      const double dx=2*x*(k1_+2*k2_*r2), dy=2*y*(k1_+2*k2_*r2);
      (*J) << a+x*dx+2*p1_*y+6*p2_*x, x*dy+2*p1_*x+2*p2_*y,
              y*dx+2*p1_*x+2*p2_*y, a+y*dy+6*p1_*y+2*p2_*x;
    }
    return Vector2d(x*a+2*p1_*x*y+p2_*(r2+2*x*x),
                    y*a+p1_*(r2+2*y*y)+2*p2_*x*y);
  }
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SeekerOmniCamera(int w,int h,double s,double xi,double fx,double fy,
                   double cx,double cy,double k1,double k2,double p1,double p2)
    : AbstractCamera(int(w*s),int(h*s),s), xi_(xi), fx_(fx*s), fy_(fy*s),
      cx_((cx+0.5)*s-0.5),cy_((cy+0.5)*s-0.5),k1_(k1),k2_(k2),p1_(p1),p2_(p2) {
    if(w<=0 || h<=0 || !std::isfinite(s) || s<=0 || !std::isfinite(xi) || xi<0 ||
       !std::isfinite(fx_) || !std::isfinite(fy_) || fx_<=0 || fy_<=0 ||
       !std::isfinite(cx_) || !std::isfinite(cy_) ||
       !std::isfinite(k1_) || !std::isfinite(k2_) ||
       !std::isfinite(p1_) || !std::isfinite(p2_) || width_<1 || height_<1)
      throw std::invalid_argument("Invalid Seeker omnidirectional calibration");
    invalid_integral_.assign((width_+1)*(height_+1),0);
    for(int y=0;y<height_;++y) {
      int row=0;
      for(int x=0;x<width_;++x) {
        row += !cam2world(double(x),double(y)).allFinite();
        invalid_integral_[(y+1)*(width_+1)+x+1]=invalid_integral_[y*(width_+1)+x+1]+row;
      }
    }

  }
  // Restrict to the injective branch of the unified model, including xi>1.
  bool validRay(const Vector3d& p) const {
    const double n=p.norm(), a=xi_<=1 ? xi_ : 1/xi_;
    return p.allFinite() && n>1e-12 && p.z()>-a*n+1e-12*n;
  }
  Vector2d world2cam(const Vector3d& p) const override {
    if(!validRay(p)) return Vector2d::Constant(nan());
    const double den=p.z()+xi_*p.norm();
    const Vector2d d=distort(p.head<2>()/den);
    return Vector2d(fx_*d.x()+cx_,fy_*d.y()+cy_);
  }
  Vector2d world2cam(const Vector2d& q) const override {
    return world2cam(Vector3d(q.x(),q.y(),1));
  }
  Vector3d cam2world(const double& u,const double& v) const override {
    const Vector2d target((u-cx_)/fx_,(v-cy_)/fy_);
    if(!target.allFinite()) return Vector3d::Constant(nan());
    Vector2d q=target;
    bool converged=false;
    for(int i=0;i<40;++i) {
      Matrix2d J;
      const Vector2d e=distort(q,&J)-target;
      if(e.norm()<1e-12) { converged=true; break; }
      if(!J.allFinite() || std::abs(J.determinant())<1e-14) break;
      const Vector2d step=J.fullPivLu().solve(e);
      double alpha=1;
      while(alpha>1e-6 && (distort(q-alpha*step)-target).squaredNorm()>=e.squaredNorm()) alpha*=0.5;
      if(alpha<=1e-6) break;
      q-=alpha*step;
    }
    if(!converged && (distort(q)-target).norm()>1e-10) return Vector3d::Constant(nan());
    const double r2=q.squaredNorm(), disc=1+(1-xi_*xi_)*r2;
    if(!std::isfinite(disc) || disc<=1e-12) return Vector3d::Constant(nan());
    const double lambda=(xi_+std::sqrt(disc))/(1+r2);
    const Vector3d ray(lambda*q.x(),lambda*q.y(),lambda-xi_);
    if(!validRay(ray)) return Vector3d::Constant(nan());
    return ray.normalized();
  }
  Vector3d cam2world(const Vector2d& p) const override { return cam2world(p.x(),p.y()); }
  Eigen::Matrix<double,2,3> projectionJacobian(const Vector3d& p) const {
    Eigen::Matrix<double,2,3> J;
    if(!validRay(p)) { J.setConstant(nan()); return J; }
    const double n=p.norm(), den=p.z()+xi_*n;
    const Vector3d dd=Vector3d(0,0,1)+xi_*p/n;
    J.row(0)=(Vector3d(1,0,0)/den-p.x()*dd/(den*den)).transpose();
    J.row(1)=(Vector3d(0,1,0)/den-p.y()*dd/(den*den)).transpose();
    Matrix2d D;
    distort(p.head<2>()/den,&D);
    Matrix2d K=Matrix2d::Zero(); K(0,0)=fx_; K(1,1)=fy_;
    return K*D*J;
  }
  // Integral mask checks every pixel supporting the patch, including bilinear neighbors.
  bool validPixel(const Vector2d& p,int margin=0) const {
    if(!p.allFinite() || margin<0 || p.x()<margin || p.y()<margin ||
       p.x()>=width_-margin-1 || p.y()>=height_-margin-1) return false;
    int x0=int(std::floor(p.x()))-margin, y0=int(std::floor(p.y()))-margin;
    int x1=int(std::floor(p.x()))+margin+2, y1=int(std::floor(p.y()))+margin+2;
    const int w=width_+1;
    return invalid_integral_[y1*w+x1]-invalid_integral_[y0*w+x1]
         -invalid_integral_[y1*w+x0]+invalid_integral_[y0*w+x0]==0;
  }
  double errorMultiplier2() const override { return fx_/(1+xi_); }
  double errorMultiplier() const override { return 4*fx_*fy_/((1+xi_)*(1+xi_)); }
  double fx() const override { return fx_; }
  double fy() const override { return fy_; }
  double cx() const override { return cx_; }
  double cy() const override { return cy_; }
};
}
