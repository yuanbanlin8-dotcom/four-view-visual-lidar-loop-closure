#include "IMU_Processing.h"
#include <iostream>
#include <stdexcept>

static sensor_msgs::ImuPtr sample(double t, double wz=0.) {
  sensor_msgs::ImuPtr m(new sensor_msgs::Imu());
  m->header.stamp.fromSec(t);m->linear_acceleration.z=1.;m->angular_velocity.z=wz;return m;
}
static void check(bool ok,const char* why) {if(!ok)throw std::runtime_error(why);}
int main(int argc,char** argv) {
  ros::init(argc,argv,"imu_init_regression");ros::Time::init();
  for(double epoch:{1000.,1790071106.}) for(int groups:{1,3}) {
    ImuProcess imu;imu.set_imu_init_frame_num(groups*20-1);
    StatesGroup state;LidarMeasureGroup lidar;lidar.lio_vio_flg=LIO;
    PointCloudXYZI::Ptr output(new PointCloudXYZI());
    for(int group=0;group<groups;++group) {
      MeasureGroup m;m.lio_time=epoch+(group+1)*.1;
      for(int i=1;i<=20;++i)m.imu.push_back(sample(epoch+group*.1+i*.005));
      lidar.measures.clear();lidar.measures.push_back(m);imu.Process2(lidar,state,output);
      check(std::abs(lidar.last_lio_update_time-m.lio_time)<1e-6,"initialization did not advance the measurement boundary");
    }
    check(!imu.imu_need_init,"initialization did not complete");
    const double begin=epoch+groups*.1;
    MeasureGroup m;m.lio_time=begin+.1;
    for(int i=1;i<=20;++i)m.imu.push_back(sample(begin+i*.005,.1));
    lidar.measures.clear();lidar.measures.push_back(m);
    for(float ms:{1.f,50.f,100.f}) {PointType p{};p.x=3.;p.curvature=ms;lidar.pcl_proc_cur->push_back(p);}
    imu.Process2(lidar,state,output);
    check(state.pos_end.norm()<1e-5,"stationary position drifted at initialization boundary");
    check(std::abs(std::atan2(state.rot_end(1,0),state.rot_end(0,0))-.00975)<1e-5,"first angular propagation interval is incorrect");
    check(output->size()==3,"first cloud lost points");
    check(std::abs(output->front().y+3.*std::sin(.00975))<1e-4,"first point deskew has the wrong time origin");
  }
  std::cout<<"IMU_INIT boundary, first propagation, deskew and epoch invariance PASS\n";
}
