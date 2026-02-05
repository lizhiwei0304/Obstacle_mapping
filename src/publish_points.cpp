#include <ros/ros.h>
#include <std_msgs/String.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <vector>
#include <sstream>

class GoalMarkerPublisher
{
public:
  GoalMarkerPublisher()
  {
    // 初始化节点句柄
    nh_ = ros::NodeHandle();
    nh_.param<double>("/robot_scale", scale_, 0.8);
    nh_.param<double>("/pred_alpha", pred_alpha_, 0.8);
    nh_.param<double>("/vehicle_height", vehicle_height_, 0.8);
    nh_.param<double>("/point_scale", point_scale_, 0.8);
    // 订阅 /recorded_goals 话题
    goals_sub_ = nh_.subscribe("/recorded_goals", 10, &GoalMarkerPublisher::goalsCallback, this);

    // 发布标记数组
    marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/visualization_marker_array", 10);
    curmarker_pub_ = nh_.advertise<visualization_msgs::Marker>("/cur_marker", 10);
  }

  void goalsCallback(const std_msgs::String::ConstPtr &msg)
  {
    // 解析记录的目标并生成标记
    std::vector<geometry_msgs::PoseStamped> poses = parseGoals(msg->data);
    publishMarkers(poses);
  }

  std::vector<geometry_msgs::PoseStamped> parseGoals(const std::string &data)
  {
    std::vector<geometry_msgs::PoseStamped> poses;
    std::istringstream ss(data);
    std::string line;

    geometry_msgs::PoseStamped pose; // 在循环外定义 pose

    // 解析每一行并提取位置和姿态
    while (std::getline(ss, line))
    {
      if (line.find("Position:") != std::string::npos)
      {
        sscanf(line.c_str(), "Position: (%lf, %lf, %lf)",
               &pose.pose.position.x,
               &pose.pose.position.y,
               &pose.pose.position.z);
      }
      if (line.find("Orientation:") != std::string::npos)
      {
        sscanf(line.c_str(), "Orientation: (%lf, %lf, %lf, %lf)",
               &pose.pose.orientation.x,
               &pose.pose.orientation.y,
               &pose.pose.orientation.z,
               &pose.pose.orientation.w);
        poses.push_back(pose);
      }
    }
    return poses;
  }

  void publishMarkers(const std::vector<geometry_msgs::PoseStamped> &poses)
  {
    visualization_msgs::Marker curmarker;
    curmarker.header.frame_id = "map";
    curmarker.header.stamp = ros::Time::now();
    curmarker.ns = "cur";
    curmarker.id = 0;
    curmarker.type = visualization_msgs::Marker::MESH_RESOURCE;
    curmarker.mesh_resource = "file:///home/wxr/ivrc/6t/catkin_marker/src/base_link.dae";
    curmarker.action = visualization_msgs::Marker::ADD;
    curmarker.pose.position.x = 0;
    curmarker.pose.position.y = 0;
    curmarker.pose.position.z = 0;
    curmarker.pose.orientation.x = 0.0;
    curmarker.pose.orientation.y = 0.0;
    curmarker.pose.orientation.z = sqrt(2) / 2;
    curmarker.pose.orientation.w = sqrt(2) / 2;
    curmarker.scale.x = scale_;
    curmarker.scale.y = scale_;
    curmarker.scale.z = scale_;
    curmarker.color.r = 0.0f;
    curmarker.color.g = 0.5f;
    curmarker.color.b = 0.0f;
    curmarker.color.a = 1.0;
    curmarker_pub_.publish(curmarker);

    visualization_msgs::MarkerArray marker_array;

    for (size_t i = 0; i < poses.size(); ++i)
    {
      const auto &pose = poses[i];
      // 打印位置和姿态信息
      ROS_INFO("Pose %zu: Position: (%f, %f, %f), Orientation: (%f, %f, %f, %f)",
               i,
               pose.pose.position.x,
               pose.pose.position.y,
               pose.pose.position.z,
               pose.pose.orientation.x,
               pose.pose.orientation.y,
               pose.pose.orientation.z,
               pose.pose.orientation.w);
      visualization_msgs::Marker marker_point;
      marker_point.header.frame_id = "map";
      marker_point.header.stamp = ros::Time::now();
      marker_point.ns = "clicked_points";
      marker_point.id = static_cast<int>(i);                  // 确保每个标记有唯一 ID
      marker_point.type = visualization_msgs::Marker::SPHERE; // 使用球体标记
      marker_point.action = visualization_msgs::Marker::ADD;
      marker_point.pose = pose.pose;
      marker_point.pose.position.z = vehicle_height_;
      marker_point.scale.x = point_scale_;
      marker_point.scale.y = point_scale_;
      marker_point.scale.z = point_scale_;
      marker_point.color.r = 1.0f;
      marker_point.color.g = 0.0f;
      marker_point.color.b = 0.0f;
      marker_point.color.a = 1.0;
      marker_point.lifetime = ros::Duration(0); // 持续时间
      marker_array.markers.push_back(marker_point);

      visualization_msgs::Marker marker_6t;
      marker_6t.header.frame_id = "map";
      marker_6t.header.stamp = ros::Time::now();
      marker_6t.ns = "pred";
      marker_6t.id = static_cast<int>(i); // 确保每个标记有唯一 ID
      marker_6t.type = visualization_msgs::Marker::MESH_RESOURCE;
      marker_6t.mesh_resource = "file:///home/wxr/ivrc/6t/catkin_marker/src/base_link.dae";
      marker_6t.action = visualization_msgs::Marker::ADD;
      marker_6t.pose = pose.pose;
      marker_6t.pose.position.z = vehicle_height_;
      marker_6t.scale.x = scale_;
      marker_6t.scale.y = scale_;
      marker_6t.scale.z = scale_;
      marker_6t.color.r = 0.0f;
      marker_6t.color.g = 0.4f;
      marker_6t.color.b = 0.0f;
      marker_6t.color.a = pred_alpha_;
      marker_6t.lifetime = ros::Duration(0); // 持续时间

      marker_array.markers.push_back(marker_6t);
    }

    // 发布标记数组
    marker_pub_.publish(marker_array);
  }

private:
  ros::NodeHandle nh_;
  ros::Subscriber goals_sub_;
  ros::Publisher marker_pub_;
  ros::Publisher curmarker_pub_;

  double scale_;
  double pred_alpha_;
  double vehicle_height_;
  double point_scale_;
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "goal_marker_publisher");
  GoalMarkerPublisher goal_marker_publisher;

  ros::spin();
  return 0;
}