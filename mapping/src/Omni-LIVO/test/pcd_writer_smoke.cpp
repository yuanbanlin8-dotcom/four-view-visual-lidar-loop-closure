#include "LIVMapper.h"

#include <pcl/io/pcd_io.h>
#include <cstdio>
#include <iostream>

int main(int argc, char **argv) {
    ros::init(argc, argv, "pcd_writer_smoke");
    ros::NodeHandle nh;
    LIVMapper mapper(nh);

    pcl::PointXYZRGB first;
    first.x = 1.0f;
    first.y = 2.0f;
    first.z = 3.0f;
    first.r = 12;
    first.g = 34;
    first.b = 56;
    mapper.pcl_wait_save->push_back(first);

    pcl::PointXYZRGB second = first;
    second.x = 1.20f;
    second.r = 78;
    second.g = 90;
    second.b = 123;
    mapper.pcl_wait_save->push_back(second);
    mapper.savePCD();

    const std::string raw_path = std::string(ROOT_DIR) + "Log/PCD/all_raw_points.pcd";
    const std::string downsampled_path = std::string(ROOT_DIR) + "Log/PCD/all_downsampled_points.pcd";
    const std::string scans_pos_path = std::string(ROOT_DIR) + "Log/PCD/scans_pos.json";
    pcl::PointCloud<pcl::PointXYZRGB> raw;
    pcl::PointCloud<pcl::PointXYZRGB> downsampled;
    const bool readable = pcl::io::loadPCDFile(raw_path, raw) == 0 &&
                          pcl::io::loadPCDFile(downsampled_path, downsampled) == 0;
    const bool colors_ok = readable && raw.size() == 2 &&
                           raw[0].r == 12 && raw[0].g == 34 && raw[0].b == 56 &&
                           raw[1].r == 78 && raw[1].g == 90 && raw[1].b == 123;
    const bool downsample_ok = readable && downsampled.size() == 2;

    std::remove(raw_path.c_str());
    std::remove(downsampled_path.c_str());
    std::remove(scans_pos_path.c_str());
    std::cout << "PCD_STREAM readable=" << readable
              << " colors=" << colors_ok
              << " raw_points=" << raw.size()
              << " downsampled_points=" << downsampled.size() << std::endl;
    return colors_ok && downsample_ok ? 0 : 1;
}
