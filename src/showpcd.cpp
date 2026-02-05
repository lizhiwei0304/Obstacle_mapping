#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/PointCloud2.h>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "point_cloud_publisher");
  ros::NodeHandle nh;

  // 发布点云数据
  ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2>("point_cloud", 1);

  // 加载 PCD 文件
  pcl::PointCloud<pcl::PointXYZ> cloud;
  std::string filepath;
  double resolution;
  nh.param<std::string>("filepath", filepath, "/home/wxr/ivrc/6t/slam_ws/maps/map0511driving01.pcd");
  nh.param<double>("resolution", resolution, 0.2);
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(filepath, cloud) == -1)
  {
    PCL_ERROR("Couldn't read file\n");
    return -1;
  }

  // 创建下采样对象
  pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
  pcl::PointCloud<pcl::PointXYZ> cloud_filtered;

  // 设置下采样的体素大小
  voxel_filter.setInputCloud(cloud.makeShared());
  voxel_filter.setLeafSize(resolution, resolution, resolution); // 0.05 分辨率
  voxel_filter.filter(cloud_filtered);

  // 转换为 ROS 消息
  sensor_msgs::PointCloud2 output;
  pcl::toROSMsg(cloud_filtered, output);
  output.header.frame_id = "map"; // 设置坐标系
  output.header.stamp = ros::Time::now();

  ros::Rate rate(1); // 1 Hz
  while (ros::ok())
  {
    pub.publish(output);
    rate.sleep();
  }

  return 0;
}