/**
 * * * @description:
 * * * @filename: show_vehicle_node.cpp
 * * * @author: wangxurui
 * * * @date: 2025-05-13 17:40:12
 **/

#include <ros/ros.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

class PointSubscriber
{
public:
    PointSubscriber()
    {
        nh_ = ros::NodeHandle("~");

        point_sub_ = nh_.subscribe("/move_base_simple/goal", 10, &PointSubscriber::pointCallback, this);

        test_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/move_base_simple/goal", 10);
        marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/visualization_marker_array", 10);
        cur_marker_pub_ = nh_.advertise<visualization_msgs::Marker>("/cur_marker", 10);
        // normal_marker_pub_ = nh_.advertise<visualization_msgs::Marker>("/normal_marker", 10);
        nh_.param<double>("/robot_scale", scale_, 0.8);
        nh_.param<double>("/pred_alpha", pred_alpha_, 0.8);
        nh_.param<double>("/vehicle_height", vehicle_height_, 0.8);
        nh_.param<double>("/vehicle_radius", vehicle_radius_, 0.8);
        geometry_msgs::PoseStamped goal = geometry_msgs::PoseStamped();
        test_pub_.publish(goal);

        visualization_msgs::Marker marker_6t_;
        marker_6t_.header.frame_id = "base_link";
        marker_6t_.header.stamp = ros::Time::now();
        marker_6t_.ns = "cur";
        marker_6t_.id = 0;
        marker_6t_.type = visualization_msgs::Marker::MESH_RESOURCE;
        marker_6t_.mesh_resource = "file:///home/wxr/ivrc/6t/catkin_marker/src/base_link.dae";
        // marker_6t_.mesh_resource = "file:///home/wxr/ivrc/6t/catkin_marker/src/mumaren_model2/meshes/base_link.STL";
        marker_6t_.action = visualization_msgs::Marker::ADD;
        marker_6t_.pose.position.x = 0;
        marker_6t_.pose.position.y = 0;
        marker_6t_.pose.position.z = 0;
        marker_6t_.pose.orientation.x = 0.0;
        marker_6t_.pose.orientation.y = 0.0;
        marker_6t_.pose.orientation.z = sqrt(2) / 2;
        marker_6t_.pose.orientation.w = sqrt(2) / 2;
        marker_6t_.scale.x = scale_;
        marker_6t_.scale.y = scale_;
        marker_6t_.scale.z = scale_;
        marker_6t_.color.r = 0.0f;
        marker_6t_.color.g = 0.5f;
        marker_6t_.color.b = 0.0f;
        marker_6t_.color.a = 1.0;
        marker_6t_.lifetime = ros::Duration();
        marker_array_.markers.push_back(marker_6t_);

        ros::Rate r(5);
        while (ros::ok())
        {
            while (marker_pub_.getNumSubscribers() < 1)
            {
                if (!ros::ok())
                {
                    return;
                }
                ROS_WARN_ONCE("Please create a subscriber to the marker");
                sleep(1);
            }
            marker_pub_.publish(marker_array_);
            // cur_marker_pub_.publish(marker_6t_);
            ros::spinOnce();
            r.sleep();
        }
    }
    void pointCallback(const geometry_msgs::PoseStamped::ConstPtr &msg)
    {
        geometry_msgs::PoseStamped goal = *msg;

        // marker_point_.header.frame_id = "base_link_init";
        // marker_point_.header.stamp = ros::Time::now();
        // marker_point_.ns = "clicked_points";
        // marker_point_.id = 0;                                    // 确保每个标记有唯一 ID
        // marker_point_.type = visualization_msgs::Marker::SPHERE; // 使用球体标记
        // marker_point_.action = visualization_msgs::Marker::ADD;
        // marker_point_.pose = goal.pose;
        // marker_point_.scale.x = 1.0;
        // marker_point_.scale.y = 1.0;
        // marker_point_.scale.z = 1.0;
        // marker_point_.color.r = 1.0f;
        // marker_point_.color.g = 0.0f;
        // marker_point_.color.b = 0.0f;
        // marker_point_.color.a = 1.0;
        // marker_point_.lifetime = ros::Duration(0); // 持续时间
        // marker_array_.markers.push_back(marker_point_);

        marker_6t_.header.frame_id = "base_link";
        marker_6t_.header.stamp = ros::Time::now();
        marker_6t_.ns = "pred";
        marker_6t_.id += 1; // 确保每个标记有唯一 ID
        marker_6t_.type = visualization_msgs::Marker::MESH_RESOURCE;
        marker_6t_.mesh_resource = "file:///home/wxr/ivrc/6t/catkin_marker/src/base_link.dae";
        marker_6t_.action = visualization_msgs::Marker::ADD;
        marker_6t_.pose = goal.pose;
        marker_6t_.scale.x = scale_;
        marker_6t_.scale.y = scale_;
        marker_6t_.scale.z = scale_;
        marker_6t_.color.r = 0.0f;
        marker_6t_.color.g = 0.7f;
        marker_6t_.color.b = 0.0f;
        marker_6t_.color.a = pred_alpha_;
        marker_6t_.lifetime = ros::Duration(0); // 持续时间

        marker_array_.markers.push_back(marker_6t_);

        visualization_msgs::Marker normal_marker;
        normal_marker.header.frame_id = "base_link_init";
        normal_marker.header.stamp = ros::Time::now();
        normal_marker.ns = "normal";
        normal_marker.id = 0;
        normal_marker.type = visualization_msgs::Marker::TRIANGLE_LIST;
        normal_marker.action = visualization_msgs::Marker::ADD;

        normal_marker.pose.position.x = 0;
        normal_marker.pose.position.y = 0;
        normal_marker.pose.position.z = 0;
        normal_marker.pose.orientation.w = 1.0;
        normal_marker.scale.x = 1.0; // Not used for triangles
        normal_marker.scale.y = 1.0; // Not used for triangles
        normal_marker.scale.z = 1.0; // Not used for triangles
        normal_marker.color.r = 0.0;
        normal_marker.color.g = 0.0;
        normal_marker.color.b = 0.8;
        normal_marker.color.a = 0.8; // 透明度

        // 定义椭圆平面的顶点
        int num_points = 100;
        float a = 2.2; // 长轴
        float b = 1.6; // 短轴
        for (int i = 0; i < num_points; i++)
        {
            float angle1 = i * 2 * M_PI / num_points;
            float angle2 = (i + 1) * 2 * M_PI / num_points;

            geometry_msgs::Point p1, p2, p3;
            p1.x = a * cos(angle1);
            p1.y = b * sin(angle1);
            p1.z = 0;

            p2.x = a * cos(angle2);
            p2.y = b * sin(angle2);
            p2.z = 0;

            p3.x = 0; // 中心点
            p3.y = 0;
            p3.z = 0;

            normal_marker.points.push_back(p1);
            normal_marker.points.push_back(p2);
            normal_marker.points.push_back(p3);
        }
        marker_array_.markers.push_back(normal_marker);

        visualization_msgs::Marker arrow_nv;
        arrow_nv.header.frame_id = "base_link_init1"; // Set your frame_id
        arrow_nv.header.stamp = ros::Time::now();
        arrow_nv.ns = "arrow_nv";
        arrow_nv.id = 0;
        arrow_nv.type = visualization_msgs::Marker::ARROW;
        arrow_nv.action = visualization_msgs::Marker::ADD;

        // Set the pose of the marker
        arrow_nv.pose.position.x = 0.0;
        arrow_nv.pose.position.y = 0.0;
        arrow_nv.pose.position.z = 0.0;
        arrow_nv.pose.orientation.x = 0.0;
        arrow_nv.pose.orientation.y = 0.0;
        arrow_nv.pose.orientation.z = 0.0;
        arrow_nv.pose.orientation.w = 1.0;

        // Set the scale of the marker
        arrow_nv.scale.x = 0.1; // Shaft width
        arrow_nv.scale.y = 0.2; // Head width
        arrow_nv.scale.z = 0.2; // Arrow length

        // Set the color of the marker
        arrow_nv.color.r = 0.0f;
        arrow_nv.color.g = 0.0f; // Green color
        arrow_nv.color.b = 1.0f;
        arrow_nv.color.a = 0.8; // Alpha channel

        geometry_msgs::Point start_point;
        start_point.x = 0.0;
        start_point.y = 0.0;
        start_point.z = 0.0;

        geometry_msgs::Point end_point;
        end_point.x = 0.0;
        end_point.y = 0.0;
        end_point.z = 2.5;

        arrow_nv.points.push_back(start_point);
        arrow_nv.points.push_back(end_point);

        marker_array_.markers.push_back(arrow_nv);

        visualization_msgs::Marker arrow_robot;
        arrow_robot.header.frame_id = "base_link"; // Set your frame_id
        arrow_robot.header.stamp = ros::Time::now();
        arrow_robot.ns = "arrow_namespace";
        arrow_robot.id = 0;
        arrow_robot.type = visualization_msgs::Marker::ARROW;
        arrow_robot.action = visualization_msgs::Marker::ADD;

        // Set the pose of the marker
        arrow_robot.pose.position.x = 0.0;
        arrow_robot.pose.position.y = 0.0;
        arrow_robot.pose.position.z = 0.0;
        arrow_robot.pose.orientation.x = 0.0;
        arrow_robot.pose.orientation.y = 0.0;
        arrow_robot.pose.orientation.z = 0.0;
        arrow_robot.pose.orientation.w = 1.0;

        // Set the scale of the marker
        arrow_robot.scale.x = 0.1; // Shaft width
        arrow_robot.scale.y = 0.2; // Head width
        arrow_robot.scale.z = 0.2; // Arrow length

        // Set the color of the marker
        arrow_robot.color.r = 0.0f;
        arrow_robot.color.g = 1.0f; // Green color
        arrow_robot.color.b = 0.0f;
        arrow_robot.color.a = 0.8; // Alpha channel

        geometry_msgs::Point start_point1;
        start_point1.x = 0.0;
        start_point1.y = 0.0;
        start_point1.z = 0.0;

        geometry_msgs::Point end_point1;
        end_point1.x = 0.0;
        end_point1.y = 0.0;
        end_point1.z = 1;

        arrow_robot.points.push_back(start_point1);
        arrow_robot.points.push_back(end_point1);

        marker_array_.markers.push_back(arrow_robot);
        // marker_pub_.publish(marker_array_);
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber point_sub_;
    ros::Publisher marker_pub_;
    ros::Publisher test_pub_;
    ros::Publisher cur_marker_pub_;
    ros::Publisher normal_marker_pub_;
    visualization_msgs::Marker marker_6t_;
    visualization_msgs::Marker marker_point_;
    visualization_msgs::MarkerArray marker_array_;

    double scale_;
    double pred_alpha_;
    double vehicle_height_;
    double vehicle_radius_;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "clicked_point_subscriber");
    PointSubscriber point_subscriber;
    return 0;
}