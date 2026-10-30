#include "LIVMapper.h"
#include <iostream>
#include <cmath>
int main(int argc,char** argv) {
 ros::init(argc,argv,"laserMapping");ros::NodeHandle nh;LIVMapper mapper(nh);
 mapper.slam_mode_=LIVO;mapper.img_en=1;mapper.imu_en=false;
 mapper.img_time_offset=0;mapper.exposure_time_init=0;
 for(int round=0;round<2;++round) {
  const double base=100.+round*10.;LidarMeasureGroup meas;
  mapper.last_timestamp_lidar=base;
  meas.last_lio_update_time=base;
  mapper.lid_raw_data_buffer.clear();mapper.lid_header_time_buffer.clear();mapper.imu_buffer.clear();
  PointCloudXYZI::Ptr cloud(new PointCloudXYZI());
  PointType p{};p.x=2.;p.curvature=0.;cloud->push_back(p);p.curvature=100.;cloud->push_back(p);
  mapper.lid_raw_data_buffer.push_back(cloud);mapper.lid_header_time_buffer.push_back(base);
  for(int c=0;c<mapper.num_of_cameras;++c) {
   mapper.img_buffers[c].clear();mapper.img_time_buffers[c].clear();
   mapper.img_buffers[c].push_back(cv::Mat(2,2,CV_8UC3,cv::Scalar(11+c,0,0)));
   mapper.img_time_buffers[c].push_back(base+.05);
  }
  if(!mapper.sync_packages(meas)||meas.lio_vio_flg!=LIO)return 1;
  const double selected=meas.measures.back().lio_time;meas.last_lio_update_time=selected;
  if(round==0) {
   for(int n=1;n<=30;++n) {
    for(int c=0;c<mapper.num_of_cameras;++c) {
     sensor_msgs::ImagePtr msg=cv_bridge::CvImage(std_msgs::Header(),"bgr8",cv::Mat(2,2,CV_8UC3,cv::Scalar(99,0,0))).toImageMsg();
     msg->header.stamp=ros::Time(base+.05+n*.05);
     mapper.img_cbk(msg,c);
    }
   }
  } else {
   for(int c=0;c<mapper.num_of_cameras;++c) {mapper.img_buffers[c].clear();mapper.img_time_buffers[c].clear();}
  }
  const auto remaining=mapper.img_buffers[0].size();
  if(!mapper.sync_packages(meas)||meas.lio_vio_flg!=VIO)return 2;
  const auto& m=meas.measures.back();
  if(std::abs(m.vio_time-selected)>1e-9||std::abs(m.lio_time-selected)>1e-9)return 3;
  for(int c=0;c<mapper.num_of_cameras;++c) if(m.imgs[c].at<cv::Vec3b>(0,0)[0]!=11+c)return 4;
  if(mapper.img_buffers[0].size()!=remaining)return 5;
 }
 std::cout<<"SYNC_PAIR backlog_drop_and_empty_input PASS"<<std::endl;
}
