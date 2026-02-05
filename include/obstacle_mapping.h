/**
 * * * @description:
 * * * @filename: obstacle_mapping.h
 * * * @author: wangxurui
 * * * @date: 2025-03-14 18:56:22
 **/
#include <iostream>
#include <deque>
#include <tbb/task_scheduler_init.h>
#include <tbb/tbb.h>
// #include <cuda_runtime.h>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>

#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/io.h>
#include <pcl/surface/mls.h>
#include <pcl/common/transforms.h> // For pcl::transformPointCloud
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>

#include <grid_map_msgs/GridMap.h>
#include <grid_map_core/grid_map_core.hpp>
#include <grid_map_ros/grid_map_ros.hpp>

using namespace std;

class obstacle_mapping
{
public:
    obstacle_mapping(ros::NodeHandle &nh);
    ~obstacle_mapping();

private:
    ros::NodeHandle nh_;
    ros::Subscriber lidar_sub_;
    ros::Subscriber odom_sub_;
    ros::Publisher local_map_pub_;
    ros::Publisher local_fused_map_pub_;
    ros::Publisher global_map_pub_;
    ros::Publisher pcd_pub_;
    ros::Publisher local_obstacle_map_pub_;
    ros::Publisher pointcloud_pub_;
    tf::TransformListener listener_;
    ros::Time map_time_;

    grid_map::GridMap local_map_;
    grid_map::GridMap fused_local_map_;
    grid_map::GridMap global_map_;
    grid_map::GridMap global_submap_;
    grid_map::Position center_robot_; // 机器人在全局地图中的位置
    grid_map::Index center_robot_index_;

    std::deque<nav_msgs::Odometry> odom_deque_;
    nav_msgs::Odometry cur_odom_;
    nav_msgs::Odometry last_odom_;

    string lidar_topic_;
    string odom_topic_;
    string map_frame_;
    string robot_frame_;
    string lidar_frame_;
    string pcd_path_;
    string elevation_BGK_savepath_;
    string critical_savepath_;
    string traversability_savepath_;
    string obstacle_savepath_;
    string gridmap_loadpath_;

    double minZ_;
    double maxZ_;
    double lidar_z_;

    double local_map_resolution_;
    double global_map_resolution_;
    double global_map_length_x_;
    double global_map_length_y_;
    double local_map_length_x_;
    double local_map_length_y_;
    double global_submap_length_x_;
    double global_submap_length_y_;

    double maxsensorRangeLimit_;
    double minsensorRangeLimit_;
    double predictionKernalSize_;
    double normal_estimationRadius_;
    double stepRadius_;

    double heightdiffthreshold_;
    double cntratiothreshold_;
    double slope_crit_;
    double roughness_crit_;
    double step_crit_;
    double traversability_crit_;
    double slope_crit_fp_;
    double slope_crit_fp_up_;
    double roughness_crit_fp_;
    double roughness_crit_fp_up_;
    double step_crit_fp_;
    double step_crit_fp_up_;
    double traversability_crit_fp_;
    double traversability_crit_fp_up_;

    int running_mode_;
    int cntthreshold_;
    int mapping_mode_;
    int threadCount_;
    int pose_mode_;

    bool first_map_init_ = true;
    bool save_map_;
    // bool debug_ = false;

    Eigen::Vector3d robot_pose_;

    void init();
    void initgridmap(grid_map::GridMap &map, string frame, double height_map_resolution, double maplength_x, double maplength_y);

    void StoreOdom(const nav_msgs::OdometryConstPtr &odom);

    void Mapping(const sensor_msgs::PointCloud2ConstPtr &msg);
    // void MappingPCD(const pcl::PointCloud<pcl::PointXYZ> &cloud);

    void LocalMapping(const pcl::PointCloud<pcl::PointXYZ> &localcloud);
    void FuseMap();
    void GlobalMapping(const pcl::PointCloud<pcl::PointXYZ> &globalcloud);

    void DenseMapping(grid_map::GridMap &map);
    void FeatureMapping(grid_map::GridMap &map);
    void StepMapping(grid_map::GridMap &map, const grid_map::Index &index, const Eigen::Vector3d &f_point);
    void TraversabilityMapping(grid_map::GridMap &map);
    void FeatureMerge();

    void simpleobstacledetection(grid_map::GridMap &map);
    void cloudobstacledetection(const pcl::PointCloud<pcl::PointXYZ> &localcloud, grid_map::GridMap &map);
    void mapobstacledetection(grid_map::GridMap &map);
    void criticalfootprintsfilter(grid_map::GridMap &map);

    void updateHeightStats(float &height, float &variance, float n, float new_height);
    void areaSingleNormalComputation(grid_map::GridMap &map, const grid_map::Index &index);

    void dist(const Eigen::MatrixXf &xStar, const Eigen::MatrixXf &xTrain, Eigen::MatrixXf &d) const;
    void covSparse(const Eigen::MatrixXf &xStar, const Eigen::MatrixXf &xTrain, Eigen::MatrixXf &Kxz) const;
    void saveMatrixToCSV(const Eigen::MatrixXf &matrix, const std::string &filename);
    Eigen::MatrixXf loadMatrixFromCSV(const std::string &filename);
    int TimestampMatch(double fixtimestamp, std::deque<nav_msgs::Odometry> target_deque);
    double BayesUpdator(double value_update, double value_observe);

    void publocalmap();
    void pubfusedlocalmap();
    void pubglobalmap();

    void pointcloudinterpolation(pcl::PointCloud<pcl::PointXYZ>::Ptr &input);
    void visualizepointcloud();
};
