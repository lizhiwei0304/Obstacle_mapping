#include <ros/ros.h>
#include <string>
#include <grid_map_msgs/GridMap.h>
#include <grid_map_ros/grid_map_ros.hpp>
#include <std_msgs/Bool.h>
#include <fstream>
#include <sstream>

grid_map::GridMap current_grid_map; // 全局变量，用于保存当前的 grid_map
int save_count = 0;                 // 用于跟踪保存次数

void gridMapCallback(const grid_map_msgs::GridMap &msg)
{
    // 将接收到的 GridMap 消息转换为 grid_map 对象
    grid_map::GridMapRosConverter::fromMessage(msg, current_grid_map);
    ROS_INFO("Received grid map.");
}

void saveGridMap(const std_msgs::Bool::ConstPtr &msg)
{
    if (msg->data)
    { // 检查消息是否为 true
        // 生成文件名
        save_count++;
        std::stringstream file_name;
        file_name << "/home/wxr/proj/terrain/ws_trav/src/obstacle_mapping/test_map/elevation" << save_count << ".csv";

        // 保存特定层数据为 CSV 文件
        std::ofstream file(file_name.str());
        if (!file.is_open())
        {
            ROS_ERROR("Could not open the file for writing.");
            return;
        }

        const std::string layer_name = "elevation_BGK"; // 需要保存的层
        if (current_grid_map.exists(layer_name))
        {
            const auto &height_data = current_grid_map[layer_name];

            // 获取 grid_map 的行列数
            Eigen::ArrayXXf data_array = height_data; // 假设 height_data 是 Eigen 类型
            size_t rows = data_array.rows();
            size_t cols = data_array.cols();
            ROS_INFO("行数: %zu, 列数: %zu", rows, cols);
            for (size_t row = 0; row < rows; ++row)
            {
                for (size_t col = 0; col < cols; ++col)
                {
                    file << data_array(row, col);
                    if (col < cols - 1)
                    {
                        file << ","; // 行内元素之间用逗号分隔
                    }
                }
                file << "\n"; // 每行结束后换行
            }
        }
        else
        {
            ROS_WARN("Layer '%s' not found in the grid map.", layer_name.c_str());
        }

        file.close();
    }
    else
    {
        ROS_INFO("Received false, not saving grid map.");
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "gridmap_saver");
    ros::NodeHandle nh;

    // 创建 GridMap 订阅器
    ros::Subscriber map_sub = nh.subscribe("/localmap_filtered", 10, gridMapCallback);

    // 创建 Bool 话题的订阅器
    ros::Subscriber bool_sub = nh.subscribe("/key", 10, saveGridMap);

    ros::spin();
    return 0;
}