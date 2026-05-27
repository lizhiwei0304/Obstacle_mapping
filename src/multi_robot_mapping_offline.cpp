#include "multi_robot_mapping.h"

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <grid_map_msgs/GridMap.h>

#include <algorithm>
#include <deque>
#include <string>
#include <vector>

namespace
{
  bool topicMatches(const std::string &bag_topic, const std::string &configured)
  {
    if (bag_topic == configured)
    {
      return true;
    }

    if (!configured.empty() && configured[0] != '/')
    {
      if (bag_topic == "/" + configured)
      {
        return true;
      }
    }

    if (!bag_topic.empty() && bag_topic[0] != '/')
    {
      if (configured == "/" + bag_topic)
      {
        return true;
      }
    }

    return false;
  }
} // namespace

int main(int argc, char **argv)
{
  ros::init(argc, argv, "multi_robot_mapping_offline_node");
  ros::NodeHandle pnh("~");
  ros::NodeHandle nh;

  std::string input_bag_path;
  std::string output_bag_path;
  std::string scan_topic;
  std::string odom_topic;
  std::string output_topic;
  double sync_tolerance_sec;

  nh.param<std::string>("input_bag_path", input_bag_path, std::string());
  nh.param<std::string>("output_bag_path", output_bag_path, std::string());
  nh.param<std::string>("scan_topic", scan_topic, std::string("/vehicle0/registered_scan"));
  nh.param<std::string>("odom_topic", odom_topic, std::string("/vehicle0/state_estimation"));
  nh.param<std::string>("output_map_topic", output_topic, std::string("/trav_map_offline"));
  nh.param<double>("sync_tolerance", sync_tolerance_sec, 0.02);
  sync_tolerance_sec = std::max(0.0, sync_tolerance_sec);

  if (input_bag_path.empty())
  {
    ROS_FATAL("Parameter input_bag_path is required for offline processing");
    return 1;
  }

  if (output_bag_path.empty())
  {
    ROS_FATAL("Parameter output_bag_path is required for offline processing");
    return 1;
  }

  nh.setParam("skip_old_messages", false);

  Mapping mapping(nh, pnh);

  rosbag::Bag input_bag;
  try
  {
    input_bag.open(input_bag_path, rosbag::bagmode::Read);
  }
  catch (const rosbag::BagException &ex)
  {
    ROS_FATAL("Failed to open input bag %s: %s", input_bag_path.c_str(), ex.what());
    return 1;
  }

  rosbag::Bag output_bag;
  try
  {
    output_bag.open(output_bag_path, rosbag::bagmode::Write);
  }
  catch (const rosbag::BagException &ex)
  {
    ROS_FATAL("Failed to open output bag %s: %s", output_bag_path.c_str(), ex.what());
    input_bag.close();
    return 1;
  }

  std::vector<std::string> topics;
  topics.push_back(scan_topic);
  topics.push_back(odom_topic);
  rosbag::View view(input_bag, rosbag::TopicQuery(topics));

  std::deque<sensor_msgs::PointCloud2ConstPtr> cloud_buffer;
  std::deque<nav_msgs::OdometryConstPtr> odom_buffer;
  const ros::Duration tolerance(sync_tolerance_sec);

  size_t processed_pairs = 0;
  size_t dropped_cloud = 0;
  size_t dropped_odom = 0;

  auto tryProcessPairs = [&]()
  {
    while (!cloud_buffer.empty() && !odom_buffer.empty())
    {
      const auto &cloud_msg = cloud_buffer.front();
      const auto &odom_msg = odom_buffer.front();

      ros::Duration delta = cloud_msg->header.stamp - odom_msg->header.stamp;

      if (delta > tolerance)
      {
        // Odometry message is too early relative to cloud, drop it and continue
        dropped_odom++;
        odom_buffer.pop_front();
        continue;
      }

      if (delta < -tolerance)
      {
        // Point cloud is too early relative to odometry, drop it and continue
        dropped_cloud++;
        cloud_buffer.pop_front();
        continue;
      }

      mapping.processSynchronizedMessages(cloud_msg, odom_msg);
      mapping.waitUntilIdle();

      grid_map_msgs::GridMap grid_msg;
      if (mapping.exportGridMap(grid_msg))
      {
        ros::Time stamp = grid_msg.info.header.stamp;
        if (stamp.isZero())
        {
          stamp = cloud_msg->header.stamp;
          grid_msg.info.header.stamp = stamp;
        }
        output_bag.write(output_topic, stamp, grid_msg);
      }
      else
      {
        ROS_WARN_THROTTLE(5.0, "Failed to export grid map message after processing pair %zu", processed_pairs);
      }

      processed_pairs++;
      cloud_buffer.pop_front();
      odom_buffer.pop_front();
    }
  };

  for (const rosbag::MessageInstance &msg : view)
  {
    if (!ros::ok())
    {
      break;
    }

    const std::string &topic = msg.getTopic();
    if (topicMatches(topic, scan_topic))
    {
      auto cloud_msg = msg.instantiate<sensor_msgs::PointCloud2>();
      if (cloud_msg)
      {
        cloud_buffer.push_back(cloud_msg);
      }
    }
    else if (topicMatches(topic, odom_topic))
    {
      auto odom_msg = msg.instantiate<nav_msgs::Odometry>();
      if (odom_msg)
      {
        odom_buffer.push_back(odom_msg);
      }
    }

    tryProcessPairs();
  }

  tryProcessPairs();
  mapping.waitUntilIdle();

  input_bag.close();
  output_bag.close();

  ROS_INFO("Offline processing finished: %zu synchronized pairs processed, %zu cloud drops, %zu odom drops",
           processed_pairs, dropped_cloud, dropped_odom);

  return 0;
}
