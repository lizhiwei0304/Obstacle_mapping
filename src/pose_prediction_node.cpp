/**
 * * * @description:
 * * * @filename: pose_prediction.cpp
 * * * @author: wangxurui
 * * * @date: 2025-04-07 17:51:04
 **/

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

class pose_prediction
{
private:
    // 机器人参数
    // 底盘高度图位置
    // std::string ROBOTFILENAME_ = ros::package::getPath("obstacle_mapping") + "/test_map/vehicles/" + "mumaren.csv";
    std::string ROBOTFILENAME_ = ros::package::getPath("obstacle_mapping") + "/test_map/vehicles/" + "6t_hight.csv";
    std::string TERRIANFILENAME_;
    // 底盘高度图分辨率
    double RESOLUTION_ = 0.2;
    HeightGrid robot_gridmap_; // 机器人底盘高度图
    HeightGrid robot_gap_map_; // 全局坐标系下高度图和间隙图
    double d_touch_gap_ = 0.1;
    double collision_gap_ = 0.1;
    int vehicle_type_ = 1;
    // int rows_ = 9;
    int rows_ = 13;
    // int cols_ = 16;
    int cols_ = 21;
    double length_ = RESOLUTION_ * cols_;
    double wideth_ = RESOLUTION_ * rows_;
    double center_x_ = wideth_ / 2;
    double center_y_ = length_ / 2;
    double center_z_ = 0.0;
    double local_min_z_;
    double lidar_z_;
    double map_length_X_;
    double map_length_Y_;
    double nv_scale_;
    int interation_;

    std::vector<Eigen::Vector4d> touch_points_;
    std::vector<cv::Point2f> local_2D_touch_points_;
    std::vector<Eigen::Vector3d> touch_poly_;
    std::vector<cv::Point2f> local_2D_touch_poly_;
    std::array<Eigen::Vector2d, 4> foot_print_;
    Eigen::ParametrizedLine<double, 3> rotation_line_;

    Eigen::Vector3d pos_;
    Eigen::Vector3d T_vector_;
    Eigen::Matrix3d R_matrix_;
    Eigen::Matrix4d T_matrix_;
    Eigen::Matrix4d T_matrix_init_;
    Eigen::Matrix4d Rotate_with_rotation_line_;
    std::vector<cv::Point2f> touch_points_2D_;

    // 以车辆为中心的局部地图中提取的submap，认为这个submap中心点与车辆几何中心重合
    grid_map::GridMap terrain_sub_gridmap_;
    grid_map::GridMap terrain_sub_gridmap_nv_;
    grid_map::Length submap_length_;
    grid_map::Length submap_length_nv_;
    grid_map::GridMap terrain_gridmap_; // 地形图
    HeightGrid terrain_gridmap_HG_;     // 地形图HG格式
    Eigen::MatrixXf terrainmatrix_;

    bool stable_;
    bool collision_;
    bool save_pose_;
    bool touch_points_change_;
    bool key_pushed_ = true;
    int show_init_ = 0;

    ros::NodeHandle nh_;
    ros::Timer maptimer_;
    ros::Subscriber map_sub_;
    ros::Subscriber key_sub_;
    ros::Publisher marker_pub_;
    ros::Publisher marker_non_pub_;
    ros::Publisher terrain_map_pub_;
    ros::Publisher terrain_submap_pub_;
    ros::Publisher terrain_submap_nv_pub_;
    ros::Publisher robot_gap_map_pub_;
    ros::Publisher normal_vector_pub_;
    tf::TransformBroadcaster br_;

public:
    pose_prediction(ros::NodeHandle &nh);
    ~pose_prediction();
    void loadCsvToMatrix(const string &FILENAME, HeightGrid &map, int rows, int cols);
    void mapcallback(const grid_map_msgs::GridMap::ConstPtr msg);
    void KeyCallback(const std_msgs::Bool::ConstPtr key);
    void calRotationLine();
    bool ifInPoly(const Eigen::Vector3d &intersection);
    double calRotationAngle();
    void predict(Eigen::Vector3d pos);
    bool check_wheel(int i, int j);
    void showPoseInfo();
    void showPose();
    void iterateOnce();
    void pubishmap(const ros::TimerEvent &event);
    void setCenterXYZ(HeightGrid &heightmap, const double &startX, const double &startY, const double &startZ);
    Eigen::ParametrizedLine<double, 3> getRotationLine(const Eigen::Vector3d &intersection, const Eigen::Vector3d &projection, const Eigen::Vector3d &F);
    Eigen::Vector3d fitplane(vector<Eigen::Vector3d> points);
    Eigen::Vector3d getProjectePoint(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2, const Eigen::Vector3d &a);
};

pose_prediction::pose_prediction(ros::NodeHandle &nh) : nh_(nh)
{
    nh_.param<int>("vehicle_type", vehicle_type_, 1);
    nh_.param<int>("show_init", show_init_, 0);
    nh_.param<bool>("save_pose", save_pose_, true);
    nh_.param<double>("lidar_z", lidar_z_, 0.0);
    nh_.param<double>("d_touch_gap", d_touch_gap_, 0.1);
    nh_.param<double>("nv_scale", nv_scale_, 0.1);
    nh_.param<string>("terrainmap_file", TERRIANFILENAME_, ros::package::getPath("obstacle_mapping") + "/test_map/" + "elevation1.csv");
    nh_.param<double>("map_length_X", map_length_X_, 30);
    nh_.param<double>("map_length_Y", map_length_Y_, 30);
    if (vehicle_type_ == 2)
    {
        ROBOTFILENAME_ = ros::package::getPath("obstacle_mapping") + "/test_map/vehicles/" + "mumaren.csv";
        rows_ = 9;
        cols_ = 16;
        length_ = RESOLUTION_ * cols_;
        wideth_ = RESOLUTION_ * rows_;
        center_x_ = wideth_ / 2;
        center_y_ = length_ / 2;
    }
    loadCsvToMatrix(ROBOTFILENAME_, robot_gridmap_, rows_, cols_);
    setCenterXYZ(robot_gridmap_, center_x_, center_y_, center_z_);
    int size_X = map_length_X_ / RESOLUTION_;
    int size_Y = map_length_Y_ / RESOLUTION_;
    loadCsvToMatrix(TERRIANFILENAME_, terrain_gridmap_HG_, size_X, size_Y);
    setCenterXYZ(terrain_gridmap_HG_, 0, 0, -lidar_z_);
    grid_map::Length mapLength(map_length_X_, map_length_Y_);
    terrain_gridmap_.setFrameId("map");
    terrain_gridmap_.setGeometry(mapLength, 0.2, grid_map::Position(0.0, 0.0));
    terrain_gridmap_.add("elevation", terrain_gridmap_HG_.Z_.cast<float>());

    // map_sub_ = nh_.subscribe<grid_map_msgs::GridMap>("/localmap", 10, boost::bind(&pose_prediction::mapcallback, this, _1));
    // key_sub_ = nh_.subscribe<std_msgs::Bool>("/key", 10, boost::bind(&pose_prediction::KeyCallback, this, _1));
    marker_pub_ = nh_.advertise<visualization_msgs::Marker>("/contactable", 10);
    marker_non_pub_ = nh_.advertise<visualization_msgs::Marker>("/noncontactable", 10);
    terrain_map_pub_ = nh_.advertise<grid_map_msgs::GridMap>("/terrain_map", 10);
    terrain_submap_nv_pub_ = nh_.advertise<grid_map_msgs::GridMap>("/terrain_submap_nv", 10);
    terrain_submap_pub_ = nh_.advertise<grid_map_msgs::GridMap>("/terrain_submap", 10);
    robot_gap_map_pub_ = nh_.advertise<grid_map_msgs::GridMap>("/robot_map", 10);
    normal_vector_pub_ = nh_.advertise<grid_map_msgs::GridMap>("/normal_vector", 10);
    maptimer_ = nh_.createTimer(ros::Duration(0.1), boost::bind(&pose_prediction::pubishmap, this, _1));
}

pose_prediction::~pose_prediction()
{
}

void pose_prediction::loadCsvToMatrix(const string &FILENAME, HeightGrid &map, int rows, int cols)
{
    std::ifstream inputFile(FILENAME);
    if (!inputFile.is_open())
    {
        std::cerr << "Failed to open the file:" << FILENAME.c_str() << std::endl;
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
            map.X_(rowIndex, columnIndex) = rowIndex * RESOLUTION_ + RESOLUTION_ / 2;
            map.Y_(rowIndex, columnIndex) = columnIndex * RESOLUTION_ + RESOLUTION_ / 2;
            map.Z_(rowIndex, columnIndex) = std::stod(token) * RESOLUTION_ / 0.2;
            map.Gap_(rowIndex, columnIndex) = 0;
            ++columnIndex;
        }
        ++rowIndex;
    }

    inputFile.close();

    return;
}

void pose_prediction::setCenterXYZ(HeightGrid &heightmap, const double &startX, const double &startY, const double &startZ)
{
    for (auto x = heightmap.X_.data(); x < heightmap.X_.data() + heightmap.X_.size(); ++x)
    {
        *x -= startX;
        *x = -*x;
    }
    for (auto y = heightmap.Y_.data(); y < heightmap.Y_.data() + heightmap.Y_.size(); ++y)
    {
        *y -= startY;
        *y = -*y;
    }
    for (auto z = heightmap.Z_.data(); z < heightmap.Z_.data() + heightmap.Z_.size(); ++z)
    {
        *z -= startZ;
    }
}

void pose_prediction::mapcallback(const grid_map_msgs::GridMap::ConstPtr msg)
{
    grid_map::GridMapRosConverter::fromMessage(*msg, terrain_gridmap_);
}

/*!
 * 预测位姿
 * @param pos 需要估计的车辆局部坐标系下的查询位置
 * @return:
 */
void pose_prediction::predict(Eigen::Vector3d pos)
{
    pos_ = pos;
    ROS_INFO("Predicting Pose on X:%.2f, Y:%.2f at Yaw: %.2f pi ......", pos_.x(), pos_.y(), pos_.z());
    touch_points_.clear();
    local_2D_touch_points_.clear();
    touch_poly_.clear();
    local_2D_touch_poly_.clear();

    stable_ = false;
    collision_ = false;
    touch_points_change_ = true;

    double radius = sqrt(wideth_ * wideth_ + length_ * length_);
    double radius_nv = nv_scale_;
    submap_length_ = grid_map::Length(radius, radius);
    submap_length_nv_ = grid_map::Length(radius_nv, radius_nv);
    // 获取查询位置的子图
    bool success_submap = false;
    // grid_map::Position submap_pos(-0.2, 13.3);
    terrain_sub_gridmap_nv_ = terrain_gridmap_.getSubmap(pos_.head<2>(), submap_length_nv_, success_submap);
    terrain_sub_gridmap_ = terrain_gridmap_.getSubmap(pos_.head<2>(), submap_length_, success_submap);
    if (success_submap == false)
    {
        ROS_WARN("cound not get full submap!");
        return;
    }
    /****************************
    先根据法向量估计大致姿态
    ****************************/
    vector<Eigen::Vector3d> terrain_map_points;
    for (grid_map::GridMapIterator it(terrain_sub_gridmap_nv_); !it.isPastEnd(); ++it)
    {
        Eigen::Vector3d point;
        if (terrain_sub_gridmap_nv_.getPosition3("elevation", *it, point))
        {
            terrain_map_points.push_back(point);
        }
    }
    Eigen::Vector3d normal_vec = fitplane(terrain_map_points);

    Eigen::Matrix3d rotationMatrix;

    Eigen::Vector3d zAxis(0, 0, 1);
    Eigen::Vector3d rotationAxis = zAxis.cross(normal_vec);
    double angle = acos(zAxis.dot(normal_vec));

    rotationMatrix = Eigen::AngleAxisd(angle, rotationAxis.normalized()).toRotationMatrix();

    // 考虑yaw 角度
    Eigen::Matrix3d yawRotation;
    double yaw_angle = pos_.z() / 180 * M_PI;
    yawRotation = Eigen::AngleAxisd(yaw_angle, zAxis).toRotationMatrix();
    rotationMatrix = rotationMatrix * yawRotation;

    T_matrix_ = Eigen::Matrix4d::Zero();
    T_matrix_.block<3, 3>(0, 0) = rotationMatrix;
    T_matrix_.block<4, 1>(0, 3) = Eigen::Vector4d(pos_(0), pos_(1), 0.0, 1.0);

    // debug
    // std::ofstream outfile("/home/wxr/proj/terrain/ws_trav/src/obstacle_mapping/poses/T_matrix_init.txt");
    // if (outfile.is_open())
    // {
    //     outfile << T_matrix_ << std::endl;
    //     outfile.close();
    //     std::cout << "Matrix exported to T_matrix_init.txt" << std::endl;
    // }
    // else
    // {
    //     std::cerr << "Error opening file for writing." << std::endl;
    // }

    /****************************
    迭代初始化
    ****************************/
    robot_gap_map_ = robot_gridmap_;
    terrainmatrix_ = terrain_sub_gridmap_["elevation"];
    double min_gap = 100;
    double min_local_z = 0;

    // 找到最小间距
    for (int i = 0; i < rows_; i++)
    {
        for (int j = 0; j < cols_; j++)
        {
            Eigen::Vector4d point4(robot_gridmap_.X_(i, j), robot_gridmap_.Y_(i, j), robot_gridmap_.Z_(i, j), 1);
            point4 = T_matrix_ * point4;
            grid_map::Index terrain_idx;
            bool insubmap = terrain_sub_gridmap_.getIndex(grid_map::Position(point4.x(), point4.y()), terrain_idx);
            robot_gap_map_.X_(i, j) = point4.x();
            robot_gap_map_.Y_(i, j) = point4.y();
            robot_gap_map_.Z_(i, j) = point4.z();
            if (insubmap)
            {
                double gap = point4.z() - terrainmatrix_(terrain_idx(0), terrain_idx(1));
                robot_gap_map_.Gap_(i, j) = gap;
                min_gap = min(min_gap, gap);
            }

            min_local_z = min(min_local_z, robot_gridmap_.Z_(i, j));
        }
    }
    local_min_z_ = min_local_z;
    T_matrix_(2, 3) -= min_gap;
    T_vector_ = T_matrix_.block(0, 3, 3, 1);
    T_matrix_init_ = T_matrix_;

    // 整体下移动最小间距，然后判断接触点
    for (int i = 0; i < rows_; i++)
    {
        for (int j = 0; j < cols_; j++)
        {
            robot_gap_map_.Z_(i, j) -= min_gap;
            robot_gap_map_.Gap_(i, j) -= min_gap;
            if (robot_gap_map_.Gap_(i, j) < d_touch_gap_)
            {
                touch_points_2D_.push_back(cv::Point2f(robot_gridmap_.X_(i, j), robot_gridmap_.Y_(i, j)));
                if (check_wheel(i, j) == false && robot_gap_map_.Gap_(i, j) < collision_gap_)
                {
                    collision_ = true;
                    touch_points_.push_back(Eigen::Vector4d(robot_gap_map_.X_(i, j), robot_gap_map_.Y_(i, j), robot_gap_map_.Z_(i, j), 1));
                }
                else
                {
                    touch_points_.push_back(Eigen::Vector4d(robot_gap_map_.X_(i, j), robot_gap_map_.Y_(i, j), robot_gap_map_.Z_(i, j), 0));
                }
            }
        }
    }
    string gapfile = "/home/wxr/proj/terrain/ws_trav/src/obstacle_mapping/test_map/gap.csv";
    std::ofstream file(gapfile);

    if (!file.is_open())
    {
        std::cerr << "Unable to open file: " << gapfile << std::endl;
        return;
    }

    // 设置格式
    Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ",", "\n");

    // 写入矩阵到文件
    file << robot_gap_map_.Gap_.format(CSVFormat);
    file.close();
    ros::Rate r(20);
    showPose();
    showPoseInfo();
    // 提取旋转部分
    Eigen::Matrix3d rotation = T_matrix_.block<3, 3>(0, 0);

    double pitch = std::atan2(rotation(2, 1), rotation(2, 2)) * (180.0 / M_PI);
    double roll = std::asin(-rotation(2, 0)) * (180.0 / M_PI);
    roll = abs(roll);
    pitch = abs(pitch);
    // 输出结果
    if (show_init_ == 1)
    {
        while (nh_.ok())
        {
            // 输出结果
            std::cout << "Init Roll: " << roll << " degrees" << std::endl;
            std::cout << "Init Pitch: " << pitch << " degrees" << std::endl;
            showPose();
            showPoseInfo();
            ros::spinOnce();
            r.sleep();
        }
    }

    interation_ = 1;
    while (nh_.ok())
    {
        if (collision_ == true)
        {
            ROS_WARN_STREAM_ONCE("Robot collide with the terrain!!");
            // 提取旋转部分
            rotation = T_matrix_.block<3, 3>(0, 0);
            pitch = std::atan2(rotation(2, 1), rotation(2, 2)) * (180.0 / M_PI);
            roll = std::asin(-rotation(2, 0)) * (180.0 / M_PI);
            roll = abs(roll);
            pitch = abs(pitch);
            // 输出结果
            std::cout << "Final Roll: " << roll << " degrees" << std::endl;
            std::cout << "Final Pitch: " << pitch << " degrees" << std::endl;
            showPose();
            showPoseInfo();
        }
        else if (!stable_)
        {
            if (interation_ > 100)
            {
                std::cout << "The interation is more than 888, exit." << std::endl;
                return;
            }

            iterateOnce();
            interation_++;

            touch_points_.clear();
            touch_points_2D_.clear();

            min_gap = 100;
            // 每次迭代后更新robot_gap_map，并判断新的接触点
            for (int i = 0; i < robot_gridmap_.X_.rows(); i++)
            {
                for (int j = 0; j < robot_gridmap_.X_.cols(); j++)
                {
                    Eigen::Vector4d point4(robot_gridmap_.X_(i, j), robot_gridmap_.Y_(i, j), robot_gridmap_.Z_(i, j), 1);
                    point4 = T_matrix_ * point4;
                    grid_map::Index terrain_idx;
                    bool insubmap = terrain_sub_gridmap_.getIndex(grid_map::Position(point4.x(), point4.y()), terrain_idx);
                    robot_gap_map_.X_(i, j) = point4.x();
                    robot_gap_map_.Y_(i, j) = point4.y();
                    robot_gap_map_.Z_(i, j) = point4.z();
                    if (insubmap)
                    {
                        double gap = point4.z() - terrainmatrix_(terrain_idx(0), terrain_idx(1));
                        robot_gap_map_.Gap_(i, j) = gap;
                        min_gap = min(min_gap, gap);

                        // bug 第二次迭代，位姿态更新后，没有发现接触点？
                        if (robot_gap_map_.Gap_(i, j) < d_touch_gap_)
                        {
                            // touch_points_.push_back(Eigen::Vector3d(robot_gap_map_.X_(i, j), robot_gap_map_.Y_(i, j), robot_gap_map_.Z_(i, j)));
                            touch_points_2D_.push_back(cv::Point2f(robot_gridmap_.X_(i, j), robot_gridmap_.Y_(i, j)));
                            if (check_wheel(i, j) == false && robot_gap_map_.Gap_(i, j) < collision_gap_)
                            {
                                collision_ = true;
                                touch_points_.push_back(Eigen::Vector4d(robot_gap_map_.X_(i, j), robot_gap_map_.Y_(i, j), robot_gap_map_.Z_(i, j), 1));
                            }
                            else
                            {
                                touch_points_.push_back(Eigen::Vector4d(robot_gap_map_.X_(i, j), robot_gap_map_.Y_(i, j), robot_gap_map_.Z_(i, j), 0));
                            }
                        }
                    }
                }
            }

            if (touch_points_.size() == 0)
            {
                ROS_INFO("no touch points found! failing min_gap!");
                T_matrix_(2, 3) -= min_gap;
                T_vector_ = T_matrix_.block(0, 3, 3, 1);

                // 如果没有接触点，整体下移动最小间距，然后判断接触点
                for (int i = 0; i < rows_; i++)
                {
                    for (int j = 0; j < cols_; j++)
                    {
                        robot_gap_map_.Z_(i, j) -= min_gap;
                        robot_gap_map_.Gap_(i, j) -= min_gap;
                        if (robot_gap_map_.Gap_(i, j) < d_touch_gap_)
                        {
                            // touch_points_.push_back(Eigen::Vector3d(robot_gap_map_.X_(i, j), robot_gap_map_.Y_(i, j), robot_gap_map_.Z_(i, j)));
                            touch_points_2D_.push_back(cv::Point2f(robot_gridmap_.X_(i, j), robot_gridmap_.Y_(i, j)));
                            if (check_wheel(i, j) == false && robot_gap_map_.Gap_(i, j) < collision_gap_)
                            {
                                collision_ = true;
                                touch_points_.push_back(Eigen::Vector4d(robot_gap_map_.X_(i, j), robot_gap_map_.Y_(i, j), robot_gap_map_.Z_(i, j), 1));
                            }
                            else
                            {
                                touch_points_.push_back(Eigen::Vector4d(robot_gap_map_.X_(i, j), robot_gap_map_.Y_(i, j), robot_gap_map_.Z_(i, j), 0));
                            }
                        }
                    }
                }
            }
        }
        else
        {
            ROS_INFO_STREAM_ONCE("Pose is stable now~");
            if (save_pose_)
            {
                // 提取旋转部分
                rotation = T_matrix_.block<3, 3>(0, 0);

                // double roll = atan2(rotation(2, 1), rotation(2, 2)) * (180.0 / M_PI);                                                            // 计算 roll
                // double pitch = atan2(-rotation(2, 0), sqrt(rotation(2, 1) * rotation(2, 1) + rotation(2, 2) * rotation(2, 2))) * (180.0 / M_PI); // 计算 pitch
                pitch = std::atan2(rotation(2, 1), rotation(2, 2)) * (180.0 / M_PI);
                roll = std::asin(-rotation(2, 0)) * (180.0 / M_PI);
                roll = abs(roll);
                pitch = abs(pitch);
                // 输出结果
                std::cout << "Final Roll: " << roll << " degrees" << std::endl;
                std::cout << "Final Pitch: " << pitch << " degrees" << std::endl;
                // 打开文件并记录数据
                std::ofstream outfile("/home/wxr/proj/terrain/ws_trav/src/obstacle_mapping/poses/T_matrix_iterated.txt", std::ios::app);
                if (outfile.is_open())
                {
                    outfile << std::fixed << std::setprecision(2) // 设置输出精度
                            << pos_(0) << ","
                            << pos_(1) << ","
                            << pos_(2) << ","
                            << roll << ","
                            << pitch << std::endl;
                    outfile.close();
                    std::cout << "Data exported to T_matrix_iterated.txt" << std::endl;
                    save_pose_ = false;
                }
                else
                {
                    std::cerr << "Error opening file for writing." << std::endl;
                }
            }
        }

        if (show_init_ == interation_)
        {
            while (nh_.ok())
            {
                showPose();
                showPoseInfo();
                ros::spinOnce();
                r.sleep();
            }
        }

        showPose();
        showPoseInfo();
        ros::spinOnce();

        r.sleep();
    }
    return;
}

void pose_prediction::iterateOnce()
{

    if (touch_points_2D_ == local_2D_touch_points_ && !local_2D_touch_points_.empty())
    {
        touch_points_change_ = false;
    }
    else
    {
        touch_points_change_ = true;
    }
    if (touch_points_change_ && key_pushed_)
    {
        touch_poly_.clear();
        local_2D_touch_poly_.clear();
        local_2D_touch_points_ = touch_points_2D_;
        if (local_2D_touch_points_.size() <= 3)
        {
            local_2D_touch_poly_ = local_2D_touch_points_;
        }
        else
        {
            cv::convexHull(local_2D_touch_points_, local_2D_touch_poly_, false);
        }
        for (auto p : local_2D_touch_poly_)
        {
            // 用了强假设，假设车辆中仅有车轮/履带会与地面接触
            touch_poly_.push_back((T_matrix_ * Eigen::Vector4d(p.x, p.y, local_min_z_, 1)).block(0, 0, 3, 1));
        }
        calRotationLine();
        if (stable_)
        {
            return;
        }
        double d_theta = calRotationAngle();
        Eigen::Matrix4d T1 = Eigen::Matrix4d::Identity();
        T1.block<3, 1>(0, 3) = -rotation_line_.origin();
        Eigen::Matrix4d R1 = Eigen::Matrix4d::Identity();
        R1.block<3, 3>(0, 0) = Eigen::Quaterniond().setFromTwoVectors(rotation_line_.direction(), Eigen::Vector3d(0, 0, 1)).toRotationMatrix();
        Eigen::Matrix4d R2 = Eigen::Matrix4d::Identity();
        R2.block<3, 3>(0, 0) = Eigen::AngleAxisd(d_theta, Eigen::Vector3d(0, 0, 1)).toRotationMatrix();
        Eigen::Matrix4d R3 = Eigen::Matrix4d::Identity();
        R3.block<3, 3>(0, 0) = R1.block<3, 3>(0, 0).transpose();
        Eigen::Matrix4d T2 = Eigen::Matrix4d::Identity();
        T2.block<3, 1>(0, 3) = rotation_line_.origin();
        Rotate_with_rotation_line_ = T2 * R3 * R2 * R1 * T1;
        T_matrix_ = Rotate_with_rotation_line_ * T_matrix_;

        T_matrix_.block<2, 1>(0, 3) = pos_.head<2>();
        T_vector_ = T_matrix_.block(0, 3, 3, 1);
        R_matrix_ = T_matrix_.block(0, 0, 3, 3);
        // key_pushed_ = false;
    }
}

Eigen::Vector3d pose_prediction::fitplane(vector<Eigen::Vector3d> points)
{
    Eigen::Vector3d mean_points = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < points.size(); i++)
        mean_points += points[i];

    mean_points /= (double)points.size();

    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (size_t i = 0; i < points.size(); i++)
    {
        Eigen::Vector3d v = points[i] - mean_points;
        cov += v * v.transpose();
    }
    cov /= (double)points.size();
    Eigen::EigenSolver<Eigen::Matrix3d> es(cov);
    Eigen::Matrix<double, 3, 1> D = es.pseudoEigenvalueMatrix().diagonal();
    Eigen::Matrix3d V = es.pseudoEigenvectors();
    Eigen::MatrixXd::Index evalsMax;
    D.minCoeff(&evalsMax);
    Eigen::Matrix<double, 3, 1> n = V.col(evalsMax);
    n.normalize();
    if (n(2, 0) < 0.0)
        n = -n;

    double sigma = D(evalsMax) / D.sum() * 3.0;
    if (isnan(sigma))
    {
        sigma = 1.0;
        n = Eigen::Vector3d(1.0, 0.0, 0.0);
    }

    return n;
}

void pose_prediction::calRotationLine()
{
    Eigen::Vector3d F((-Eigen::Vector3d::UnitZ()).normalized()); // 重力
    Eigen::ParametrizedLine<double, 3> F_line(T_vector_, F);
    Eigen::Vector3d z = R_matrix_.block(0, 2, 3, 1);
    Eigen::Hyperplane<double, 3> plane(z, touch_poly_.front());
    Eigen::Vector3d intersection = F_line.intersectionPoint(plane);
    Eigen::Vector3d rotatep1, rotatep2;
    if (touch_poly_.size() == 1)
    {
        if (touch_poly_.front() == intersection)
        {
            stable_ = true;
            return;
        }
        rotation_line_ = getRotationLine(intersection, touch_poly_.front(), F);
    }
    else
    {
        if (touch_poly_.size() == 2)
        {
            Eigen::Vector3d projection = getProjectePoint(touch_poly_.front(), touch_poly_.back(), intersection);
            if (projection == intersection)
            {
                stable_ = true;
                return;
            }
            // 从touch_poly_.front()到touch_poly_.back()的方向不稳定
            if (((touch_poly_.back().x() - touch_poly_.front().x()) * (intersection.y() - touch_poly_.front().y()) -
                 (intersection.x() - touch_poly_.front().x()) * (touch_poly_.back().y() - touch_poly_.front().y())) < 0)
            {
                rotatep1 = touch_poly_.front();
                rotatep2 = touch_poly_.back();
            }
            else
            {
                rotatep1 = touch_poly_.back();
                rotatep2 = touch_poly_.front();
            }
            // rotation_line_ = getRotationLine(intersection, projection, F);
        }
        else
        {
            if (ifInPoly(intersection))
            {
                stable_ = true;
                return;
            }
            else
            {
                // 遍历多边形的每一条边，找到最近的投影点，并计算旋转线
                double min_dis(FLT_MAX);
                Eigen::Vector3d projection;
                Eigen::Vector3d p1, p2;
                for (int i = 0; i < touch_poly_.size(); i++)
                {
                    p1 = touch_poly_[i];
                    if (i == touch_poly_.size() - 1)
                    {
                        p2 = touch_poly_.front();
                    }
                    else
                    {
                        p2 = touch_poly_[i + 1];
                    }
                    if (((p2.x() - p1.x()) * (intersection.y() - p1.y()) - (intersection.x() - p1.x()) * (p2.y() - p1.y())) >= 0)
                    {
                        // 这条边是稳定的，跳过
                        continue;
                    }
                    Eigen::Vector3d projection_p1p2 = getProjectePoint(p1, p2, intersection);
                    double dis = (projection_p1p2 - intersection).norm();
                    if (dis < min_dis)
                    {
                        min_dis = dis;
                        projection = projection_p1p2;
                        rotatep1 = p1;
                        rotatep2 = p2;
                    }
                }
                // rotation_line_ = getRotationLine(intersection, projection, F);
            }
        }

        // 计算方向向量
        Eigen::Vector3d direction = rotatep2 - rotatep1;

        // 创建参数化直线
        rotation_line_ = Eigen::ParametrizedLine<double, 3>(rotatep1, direction.normalized());
    }
}

Eigen::ParametrizedLine<double, 3> pose_prediction::getRotationLine(const Eigen::Vector3d &intersection, const Eigen::Vector3d &projection, const Eigen::Vector3d &F)
{
    Eigen::Vector3d projection2intersection = (intersection - projection).normalized();
    Eigen::Vector3d perpendicular = projection2intersection.cross(F);
    return Eigen::ParametrizedLine<double, 3>(projection, perpendicular);
}

Eigen::Vector3d pose_prediction::getProjectePoint(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2, const Eigen::Vector3d &a)
{
    Eigen::Vector3d p1p2 = p2 - p1;
    Eigen::Vector3d p1a = a - p1;
    double dotProduct = p1a.dot(p1p2);
    double lengthSquared = p1p2.dot(p1p2);
    double t = dotProduct / lengthSquared;
    Eigen::Vector3d projection;
    // if (t < 0.0)
    // {
    //     projection = p1;
    // }
    // else if (t > 1.0)
    // {
    //     projection = p2;
    // }
    // else
    // {
    //     projection = p1 + t * p1p2;
    // }
    projection = p1 + t * p1p2;
    return projection;
}

bool pose_prediction::ifInPoly(const Eigen::Vector3d &intersection)
{
    for (int i = 0; i < touch_poly_.size(); i++)
    {
        Eigen::Vector2d p1(touch_poly_[i].x(), touch_poly_[i].y());
        Eigen::Vector2d p2;
        Eigen::Vector2d intersect(intersection.x(), intersection.y());
        if (i == touch_poly_.size() - 1)
        {
            p2.x() = touch_poly_.front().x();
            p2.y() = touch_poly_.front().y();
        }
        else
        {
            p2.x() = touch_poly_[i + 1].x();
            p2.y() = touch_poly_[i + 1].y();
        }
        Eigen::Vector2d p1p2 = p2 - p1;
        Eigen::Vector2d p1intersect = intersect - p1;

        if ((p1p2.x() * p1intersect.y() - p1intersect.x() * p1p2.y()) < 0)
        {
            return false;
        }
    }
    return true;
}

double pose_prediction::calRotationAngle()
{
    double d_theta = FLT_MAX;
    for (int i = 0; i < robot_gap_map_.X_.rows(); i++)
    {
        for (int j = 0; j < robot_gap_map_.X_.cols(); j++)
        {
            Eigen::Vector2d rotation_line_origin2cell(robot_gap_map_.X_(i, j) - rotation_line_.origin().x(), robot_gap_map_.Y_(i, j) - rotation_line_.origin().y());
            Eigen::Vector2d rotation_line_direction(rotation_line_.direction()(0), rotation_line_.direction()(1));
            if (rotation_line_origin2cell.x() * rotation_line_direction.y() - rotation_line_origin2cell.y() * rotation_line_direction.x() > 0 &&
                robot_gap_map_.Gap_(i, j) > d_touch_gap_)
            {
                Eigen::Vector2d proj_vec = rotation_line_origin2cell.dot(rotation_line_direction.normalized()) * rotation_line_direction.normalized();
                double distance = (rotation_line_origin2cell - proj_vec).norm();
                d_theta = std::min(d_theta, atan2(robot_gap_map_.Gap_(i, j), distance));
            }
        }
    }
    return d_theta;
}

bool pose_prediction::check_wheel(int i, int j)
{
    if (vehicle_type_ == 1)
    {
        if (i >= 2 && i <= 10)
        {
            return false;
        }
        else
        {
            if (j < 2 || j > 19)
            {
                return false;
            }
        }
    }
    if (vehicle_type_ == 2)
    {
        if (abs(robot_gridmap_.Z_(i, j) - 0.35) <= 0.01)
        {
            return false;
        }
        if (abs(robot_gridmap_.Z_(i, j) - 0.3) <= 0.01)
        {
            return false;
        }
    }
    return true;
}

void pose_prediction::showPoseInfo()
{
    visualization_msgs::Marker touch_points_marker_C;
    touch_points_marker_C.header.frame_id = "map";
    touch_points_marker_C.type = visualization_msgs::Marker::POINTS;
    touch_points_marker_C.action = visualization_msgs::Marker::ADD;
    touch_points_marker_C.scale.x = 0.2;
    touch_points_marker_C.scale.y = 0.2;
    touch_points_marker_C.scale.z = 0.2;
    touch_points_marker_C.ns = "touch_points_Con";
    ROS_INFO("contact_touch_points size: %zu", touch_points_.size());
    for (size_t i = 0; i < touch_points_.size(); ++i)
    {
        geometry_msgs::Point p;
        p.x = touch_points_[i].x();
        p.y = touch_points_[i].y();
        p.z = touch_points_[i].z();
        if (touch_points_[i].w() == 0)
        {
            touch_points_marker_C.points.push_back(p);
        }
    }
    touch_points_marker_C.color = std_msgs::ColorRGBA();
    touch_points_marker_C.color.r = 1.0;
    touch_points_marker_C.color.g = 0.0;
    touch_points_marker_C.color.b = 0.0;
    touch_points_marker_C.color.a = 1.0;

    touch_points_marker_C.pose.orientation.x = 0;
    touch_points_marker_C.pose.orientation.y = 0;
    touch_points_marker_C.pose.orientation.z = 0;
    touch_points_marker_C.pose.orientation.w = 1;

    visualization_msgs::Marker touch_points_marker_N;
    touch_points_marker_N.header.frame_id = "map";
    touch_points_marker_N.type = visualization_msgs::Marker::POINTS;
    touch_points_marker_N.action = visualization_msgs::Marker::ADD;
    touch_points_marker_N.scale.x = 0.2;
    touch_points_marker_N.scale.y = 0.2;
    touch_points_marker_N.scale.z = 0.2;
    touch_points_marker_N.ns = "touch_points_Con";
    // ROS_INFO("non_contact_touch_points size: %d", touch_points_.size());
    for (size_t i = 0; i < touch_points_.size(); ++i)
    {
        geometry_msgs::Point p;
        p.x = touch_points_[i].x();
        p.y = touch_points_[i].y();
        p.z = touch_points_[i].z();
        if (touch_points_[i].w() == 1)
        {
            touch_points_marker_N.points.push_back(p);
        }
    }
    touch_points_marker_N.color = std_msgs::ColorRGBA();
    touch_points_marker_N.color.r = 0.0;
    touch_points_marker_N.color.g = 0.0;
    touch_points_marker_N.color.b = 1.0;
    touch_points_marker_N.color.a = 1.0;

    touch_points_marker_N.pose.orientation.x = 0;
    touch_points_marker_N.pose.orientation.y = 0;
    touch_points_marker_N.pose.orientation.z = 0;
    touch_points_marker_N.pose.orientation.w = 1;

    visualization_msgs::Marker line_marker;
    line_marker.header.frame_id = "map";
    line_marker.header.stamp = ros::Time::now();
    line_marker.ns = "line_visualization";
    line_marker.type = visualization_msgs::Marker::LINE_STRIP;
    line_marker.action = visualization_msgs::Marker::ADD;

    line_marker.color = std_msgs::ColorRGBA();
    line_marker.color.r = 1.0;
    line_marker.color.g = 0.0;
    line_marker.color.b = 1.0;
    line_marker.color.a = 1.0;

    line_marker.scale.x = 0.1;

    // 计算起点和终点
    Eigen::Vector3d point_on_line = rotation_line_.origin();
    Eigen::Vector3d direction = rotation_line_.direction();
    Eigen::Vector3d endpoint = point_on_line + direction * 2.0;

    // 添加起点和终点
    geometry_msgs::Point start;
    start.x = point_on_line.x();
    start.y = point_on_line.y();
    start.z = point_on_line.z();
    line_marker.points.push_back(start);

    geometry_msgs::Point end;
    end.x = endpoint.x();
    end.y = endpoint.y();
    end.z = endpoint.z();
    line_marker.points.push_back(end);

    marker_pub_.publish(touch_points_marker_C);
    marker_non_pub_.publish(touch_points_marker_N);
    return;
}

void pose_prediction::showPose()
{
    tf::Transform transform, transform0, transform_init;

    // 提取旋转部分
    Eigen::Matrix3d rotation = T_matrix_.block<3, 3>(0, 0);
    Eigen::Matrix3d rotation0 = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d rotation_init = T_matrix_init_.block<3, 3>(0, 0);
    Eigen::Quaterniond q(rotation);
    Eigen::Quaterniond q0(rotation0);
    Eigen::Quaterniond q_init(rotation_init);

    // 提取平移部分
    Eigen::Vector3d translation = T_matrix_.block<3, 1>(0, 3);
    Eigen::Vector3d translation_init = T_matrix_init_.block<3, 1>(0, 3);

    // 设置变换
    transform.setOrigin(tf::Vector3(translation.x(), translation.y(), translation.z()));
    transform_init.setOrigin(tf::Vector3(translation_init.x(), translation_init.y(), translation_init.z()));
    transform0.setOrigin(tf::Vector3(translation_init.x(), translation_init.y(), translation_init.z()));
    transform.setRotation(tf::Quaternion(q.x(), q.y(), q.z(), q.w()));
    transform0.setRotation(tf::Quaternion(q0.x(), q0.y(), q0.z(), q0.w()));
    transform_init.setRotation(tf::Quaternion(q_init.x(), q_init.y(), q_init.z(), q_init.w()));

    // 发布变换
    br_.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "map", "robot"));
    br_.sendTransform(tf::StampedTransform(transform0, ros::Time::now(), "map", "robot0"));
    br_.sendTransform(tf::StampedTransform(transform_init, ros::Time::now(), "map", "robot_init"));
    return;
}

void pose_prediction::KeyCallback(const std_msgs::Bool::ConstPtr key)
{
    if (key->data == true)
    {
        key_pushed_ = true;
    }
}

void pose_prediction::pubishmap(const ros::TimerEvent &event)
{
    grid_map_msgs::GridMap grid_map_msg, grid_map_sub_msg, grid_map_sub_nv_msg, robot_gridmap_msg, robot_gridmap_init_msg;

    grid_map_msg.info.header.stamp = ros::Time::now();
    grid_map::Index idx(2, 10);
    grid_map::GridMapRosConverter::toMessage(terrain_gridmap_, grid_map_msg);
    terrain_map_pub_.publish(grid_map_msg);

    grid_map_sub_msg.info.header.stamp = ros::Time::now();
    grid_map::GridMapRosConverter::toMessage(terrain_sub_gridmap_, grid_map_sub_msg);
    terrain_submap_pub_.publish(grid_map_sub_msg);

    grid_map_sub_nv_msg.info.header.stamp = ros::Time::now();
    grid_map::GridMapRosConverter::toMessage(terrain_sub_gridmap_nv_, grid_map_sub_nv_msg);
    terrain_submap_nv_pub_.publish(grid_map_sub_nv_msg);

    grid_map::GridMap robot_gridmap;
    grid_map::Length mapLength(wideth_, length_);
    robot_gridmap.setFrameId("robot");
    robot_gridmap.setGeometry(mapLength, 0.2, grid_map::Position(0, 0));
    robot_gridmap.add("elevation", robot_gridmap_.Z_.cast<float>());
    robot_gridmap.add("gap", robot_gap_map_.Gap_.cast<float>());
    robot_gridmap.add("wheel_show", 0.0);
    auto &wheel_showMatrix = robot_gridmap["wheel_show"];
    for (grid_map::GridMapIterator it(robot_gridmap); !it.isPastEnd(); ++it)
    {
        grid_map::Index idx = *it;
        if (check_wheel(idx(0), idx(1)))
        {
            wheel_showMatrix(idx(0), idx(1)) = 1;
        }
    }
    robot_gridmap_msg.info.header.stamp = ros::Time::now();
    grid_map::GridMapRosConverter::toMessage(robot_gridmap, robot_gridmap_msg);
    robot_gap_map_pub_.publish(robot_gridmap_msg);

    grid_map::GridMap robot_gridmap_init;
    robot_gridmap_init.setFrameId("robot_init");
    robot_gridmap_init.setGeometry(mapLength, 0.2, grid_map::Position(0, 0));
    robot_gridmap_init.add("elevation", robot_gridmap_.Z_.cast<float>());
    // robot_gridmap_init.add("gap", robot_gap_map_.Gap_.cast<float>());
    robot_gridmap_init.add("wheel_show", 0.0);
    robot_gridmap_init_msg.info.header.stamp = ros::Time::now();
    grid_map::GridMapRosConverter::toMessage(robot_gridmap_init, robot_gridmap_init_msg);
    normal_vector_pub_.publish(robot_gridmap_init_msg);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "pose_prediction");
    ros::NodeHandle nh;

    Eigen::Vector3d pos;
    nh.param<double>("query_pos_x", pos.x(), 5.0);
    nh.param<double>("query_pos_y", pos.y(), 0.0);
    nh.param<double>("query_pos_z", pos.z(), 0.0);
    pose_prediction pose_predictor(nh);
    pose_predictor.predict(pos);
}
