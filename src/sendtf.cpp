#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "tf_publisher_own");
    ros::NodeHandle nh;

    // 从参数服务器获取 x, y, yaw 值
    double x, y, yaw;
    nh.param("query_pos_x", x, 0.0); // 默认值为 0.0
    nh.param("query_pos_y", y, 0.0); // 默认值为 0.0
    nh.param("query_pos_z", yaw, 0.0); // 默认值为 0.0

    // 创建一个 TransformBroadcaster
    tf::TransformBroadcaster broadcaster;

    // 设置循环频率
    ros::Rate loop_rate(10); // 10 Hz
    while (ros::ok())
    {
        // 创建变换
        geometry_msgs::TransformStamped transform;
        transform.header.stamp = ros::Time::now();
        transform.header.frame_id = "map";   // 父坐标系
        transform.child_frame_id = "submap"; // 子坐标系

        // 设置平移
        transform.transform.translation.x = x;
        transform.transform.translation.y = y;
        transform.transform.translation.z = 0.0; // Z 轴不变

        // 设置旋转
        tf::Quaternion q;
        q.setRPY(0, 0, yaw); // Roll、Pitch、Yaw
        transform.transform.rotation.x = q.x();
        transform.transform.rotation.y = q.y();
        transform.transform.rotation.z = q.z();
        transform.transform.rotation.w = q.w();

        // 发布变换
        broadcaster.sendTransform(transform);

        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}