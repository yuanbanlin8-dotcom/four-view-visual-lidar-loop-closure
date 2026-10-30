#include "LIVMapper.h"
#include "camera_geometry.h"
#include <iostream>
int main(int argc,char** argv) {
 ros::init(argc,argv,"laserMapping"); ros::NodeHandle nh;
 LIVMapper mapper(nh); auto v=mapper.vio_manager;
 std::vector<cv::Mat> images;
 for(auto* c:v->cams) {
  cv::Mat im(c->height(),c->width(),CV_8UC3,cv::Scalar(0,0,0));
  for(int y=0;y<im.rows;++y) for(int x=0;x<im.cols;++x) {
   auto r=c->cam2world(double(x),double(y)); if(!r.allFinite()) continue;
   uchar value=cv::saturate_cast<uchar>(128+45*std::sin(40*r.x())+45*std::sin(35*r.y())+25*std::sin(50*r.z()));
   im.at<cv::Vec3b>(y,x)=cv::Vec3b(value,value,value);
  }
  images.push_back(im);
 }
 std::vector<pointWithVar> points;
 auto* c=v->cams[0];
 for(int y=5;y<c->height()-5;y+=5) for(int x=5;x<c->width()-5;x+=5) {
  Eigen::Vector2d px(x,y); if(!camera_geometry::validPixel(*c,px,4)) continue;
  auto ray=c->cam2world(px); if(!ray.allFinite() || ray.z()>=-0.005) continue;
  pointWithVar p; p.point_w=ray*3; p.point_b=p.point_w; p.point_i=p.point_w;
  p.normal=ray; p.var=Eigen::Matrix3d::Identity()*1e-4; points.push_back(p);
 }
 if(points.empty()) return 2;
 v->processFrame(images,points,mapper.voxel_map,100.0);
 int inserted=0;
 for(auto& voxel:v->feat_map) for(auto* p:voxel.second->voxel_points) {
  p->normal_=p->pos_.normalized(); p->is_normal_initialized_=true; ++inserted;
 }
 v->processFrame(images,points,mapper.voxel_map,100.1);
 int tracked=v->total_points;
 mapper.last_timestamp_lidar=100;
 for(int i=0;i<4;++i) {
  sensor_msgs::ImagePtr msg=cv_bridge::CvImage(std_msgs::Header(),"bgr8",images[i]).toImageMsg();
  msg->header.stamp=ros::Time(i==1?99.9:100.0);
  mapper.img_cbk(msg,i); mapper.img_cbk(msg,i);
  msg->header.stamp=ros::Time(99.0);mapper.img_cbk(msg,i);
  if(mapper.img_buffers[i].size()!=1) return 4;
 }
 sensor_msgs::ImagePtr invalid(new sensor_msgs::Image);
 mapper.img_cbk(invalid,99);
 sensor_msgs::ImagePtr burst=cv_bridge::CvImage(std_msgs::Header(),"bgr8",images[0]).toImageMsg();
 for(int i=0;i<30;++i) {
  burst->header.stamp=ros::Time(101.0+0.01*i);
  mapper.img_cbk(burst,0);
 }
 if(mapper.img_buffers[0].size()>20 || mapper.img_time_buffers[0].size()>20) return 5;
 std::cout<<"CALLBACK_TEST per_camera_order_duplicate_filter_and_buffer_cap=PASS"<<std::endl;

 bool finite=mapper._state.rot_end.allFinite() && mapper._state.pos_end.allFinite() && mapper._state.cov.allFinite();
 std::cout<<"SYNTHETIC_ONLY input_negative_Z="<<points.size()<<" inserted="<<inserted
          <<" tracked="<<tracked<<" finite_state="<<finite<<" position_norm="<<mapper._state.pos_end.norm()<<std::endl;
 return inserted>0 && tracked>0 && finite ? 0:3;
}
