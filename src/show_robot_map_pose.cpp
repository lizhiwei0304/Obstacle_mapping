#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <Eigen/Dense>
#include <ros/ros.h>
#include <ros/package.h>
#include <grid_map_msgs/GridMap.h>
#include <grid_map_core/grid_map_core.hpp>
#include <grid_map_ros/grid_map_ros.hpp>

#include <opencv2/opencv.hpp>
#include <visualization_msgs/Marker.h>
#include <std_msgs/Bool.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>

using namespace std;
struct HeightGrid
{
    Eigen::MatrixXd X_;
    Eigen::MatrixXd Y_;
    Eigen::MatrixXd Z_;
    Eigen::MatrixXd Gap_;
};

void loadCsvToMatrix(const string &filename,
                     HeightGrid &map,
                     int rows,
                     int cols,
                     double cell_resolution,
                     double csv_base_resolution)
{
    std::ifstream inputFile(filename);
    if (!inputFile.is_open())
    {
        std::cerr << "Failed to open the file:" << filename.c_str() << std::endl;
        return;
    }

    std::string line;
    std::string token;

    map.X_.resize(rows, cols);
    map.Y_.resize(rows, cols);
    map.Z_.resize(rows, cols);
    map.Gap_.resize(rows, cols);

    inputFile.clear();                 // 清除错误状态
    inputFile.seekg(0, std::ios::beg); // 将文件指针设置为文件的起始位置

    int rowIndex = 0;
    while (getline(inputFile, line))
    {
        int columnIndex = 0;
        std::istringstream tokenStream(line);
        std::string token;
        while (getline(tokenStream, token, ','))
        {
            map.X_(rowIndex, columnIndex) = rowIndex * cell_resolution + cell_resolution / 2;
            map.Y_(rowIndex, columnIndex) = columnIndex * cell_resolution + cell_resolution / 2;
            map.Z_(rowIndex, columnIndex) = std::stod(token) * cell_resolution / csv_base_resolution;
            map.Gap_(rowIndex, columnIndex) = 0;
            ++columnIndex;
        }
        ++rowIndex;
    }

    inputFile.close();

    return;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "show_pose_map");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    // Parameters for robot model visualization
    std::string default_csv_path = ros::package::getPath("obstacle_mapping") + "/test_map/vehicles/6t_hight.csv";
    std::string robot_csv_path;
    pnh.param<std::string>("robot_csv_path", robot_csv_path, default_csv_path);

    int robot_rows = 12;
    int robot_cols = 21;
    pnh.param("robot_rows", robot_rows, robot_rows);
    pnh.param("robot_cols", robot_cols, robot_cols);

    double cell_resolution = 0.2;
    double csv_base_resolution = 0.2;
    pnh.param("robot_cell_resolution", cell_resolution, cell_resolution);
    pnh.param("csv_base_resolution", csv_base_resolution, csv_base_resolution);

    std::string default_matrix_path = ros::package::getPath("obstacle_mapping") + "/poses/T_matrix_init.txt";
    std::string transform_matrix_path;
    pnh.param<std::string>("transform_matrix_path", transform_matrix_path, default_matrix_path);

    std::string parent_frame = "map";
    std::string robot_frame = "robot_init";
    pnh.param<std::string>("parent_frame", parent_frame, parent_frame);
    pnh.param<std::string>("robot_frame", robot_frame, robot_frame);

    double tf_height_offset = 5.0;
    pnh.param("tf_height_offset", tf_height_offset, tf_height_offset);

    std::string publish_topic = "/robot_map_init";
    pnh.param<std::string>("robot_map_topic", publish_topic, publish_topic);

    double publish_rate_hz = 1.0;
    pnh.param("publish_rate", publish_rate_hz, publish_rate_hz);
    if (publish_rate_hz <= 0.0)
    {
        ROS_WARN("publish_rate must be positive, defaulting to 1.0 Hz");
        publish_rate_hz = 1.0;
    }

    ros::Publisher robot_gap_map_pub_ = nh.advertise<grid_map_msgs::GridMap>(publish_topic, 10);

    // Load initial transform matrix
    Eigen::Matrix4d T_matrix;
    std::ifstream infile(transform_matrix_path);
    if (!infile.is_open())
    {
        ROS_ERROR_STREAM("Error opening transform matrix file: " << transform_matrix_path);
        return 1;
    }

    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            infile >> T_matrix(i, j);
        }
    }
    infile.close();

    ROS_INFO_STREAM("Loaded transform matrix from " << transform_matrix_path << '\n' << T_matrix);

    tf::Transform transform;

    Eigen::Matrix3d rotation = T_matrix.block<3, 3>(0, 0);
    Eigen::Quaterniond q(rotation);

    Eigen::Vector3d translation = T_matrix.block<3, 1>(0, 3);
    translation(2) += tf_height_offset;
    transform.setOrigin(tf::Vector3(translation.x(), translation.y(), translation.z()));
    transform.setRotation(tf::Quaternion(q.x(), q.y(), q.z(), q.w()));
    tf::TransformBroadcaster br_;

    ros::Rate rate(publish_rate_hz);
    while (ros::ok())
    {
        // 发布变换
        br_.sendTransform(tf::StampedTransform(transform, ros::Time::now(), parent_frame, robot_frame));

        grid_map::GridMap robot_gridmap;
        HeightGrid robot_gridmap_;
        loadCsvToMatrix(robot_csv_path, robot_gridmap_, robot_rows, robot_cols, cell_resolution, csv_base_resolution);
        double length_ = cell_resolution * robot_cols;
        double wideth_ = cell_resolution * robot_rows;
        grid_map::Length mapLength(wideth_, length_);
        robot_gridmap.setFrameId(robot_frame);
        robot_gridmap.setGeometry(mapLength, cell_resolution, grid_map::Position(0, 0));
        robot_gridmap.add("elevation", robot_gridmap_.Z_.cast<float>());
        // robot_gridmap.add("elevation", robot_gap_map_.Z_.cast<float>());
        grid_map_msgs::GridMap robot_gridmap_msg;
        robot_gridmap_msg.info.header.stamp = ros::Time::now();
        robot_gridmap_msg.info.header.frame_id = robot_frame;
        grid_map::GridMapRosConverter::toMessage(robot_gridmap, robot_gridmap_msg);

        robot_gap_map_pub_.publish(robot_gridmap_msg);
        rate.sleep();
    }

    return 0;
}