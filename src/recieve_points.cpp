#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/String.h>
#include <vector>
#include <sstream>

class GoalRecorder
{
public:
  GoalRecorder()
  {
    // 初始化节点句柄
    nh_ = ros::NodeHandle();

    // 订阅 /move_base_simple/goal 话题
    goal_sub_ = nh_.subscribe("/move_base_simple/goal", 10, &GoalRecorder::goalCallback, this);

    // 发布记录的目标位置和姿态
    recorded_goals_pub_ = nh_.advertise<std_msgs::String>("/recorded_goals", 10);
  }

  void goalCallback(const geometry_msgs::PoseStamped::ConstPtr &msg)
  {
    // 记录接收到的目标位置和姿态
    goals_.push_back(*msg);

    // 发布记录的目标位置和姿态
    publishGoals();
  }

  void publishGoals()
  {
    std_msgs::String msg;
    std::stringstream ss;

    ss << "Recorded Goals:\n";
    for (const auto &goal : goals_)
    {
      ss << "Position: (" << goal.pose.position.x << ", "
         << goal.pose.position.y << ", "
         << goal.pose.position.z << ")\n";
      ss << "Orientation: (" << goal.pose.orientation.x << ", "
         << goal.pose.orientation.y << ", "
         << goal.pose.orientation.z << ", "
         << goal.pose.orientation.w << ")\n";
    }

    msg.data = ss.str();
    recorded_goals_pub_.publish(msg);
  }

private:
  ros::NodeHandle nh_;
  ros::Subscriber goal_sub_;
  ros::Publisher recorded_goals_pub_;
  std::vector<geometry_msgs::PoseStamped> goals_; // 存储所有记录的位置和姿态
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "goal_recorder");
  GoalRecorder goal_recorder;

  ros::spin();
  return 0;
}