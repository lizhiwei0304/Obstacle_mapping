/**
 * * * @description:
 * * * @filename: obstacle_mapping_node.cpp
 * * * @author: wangxurui
 * * * @date: 2025-03-14 18:54:07
 **/

#include <ros/ros.h>
#include "obstacle_mapping.h"

int main(int argc, char *argv[])
{
    /****************************
    ros配置
    ****************************/
    ros::init(argc, argv, "obstacle_detection_node");
    ros::NodeHandle nh("~");

    obstacle_mapping om(nh);

    ros::spin();


    return 0;
}
