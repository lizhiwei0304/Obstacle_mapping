/**
 * @description: 多机器人建图类
 * @filename: multi_robot_mapping.h
 * @author: wangxurui
 * @date: 2026-01-28
 **/

#ifndef MULTI_ROBOT_MAPPING_H
#define MULTI_ROBOT_MAPPING_H

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PointStamped.h>
#include <nav_msgs/Odometry.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h> // removeNaNFromPointCloud

#include <grid_map_core/grid_map_core.hpp>
#include <grid_map_ros/grid_map_ros.hpp>
#include <Eigen/Dense>
#include <tf/transform_datatypes.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <geometry_msgs/PointStamped.h>
#include <fstream>
#include <sstream>
#include <mutex>
#include <thread>
#include <queue>
#include <condition_variable>
#include <unordered_set>
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

class Mapping
{
public:
    /**
     * @brief 构造函数
     * @param nh: ROS 节点句柄
     */
    Mapping(ros::NodeHandle &nh, ros::NodeHandle &pnh);

    /**
     * @brief 析构函数
     */
    ~Mapping();

    /**
     * @brief 获取当前点云
     */
    pcl::PointCloud<pcl::PointXYZ>::Ptr getPointCloud() const;

    /**
     * @brief 处理一对同步的点云与里程计消息（可用于离线回放）
     */
    void processSynchronizedMessages(const sensor_msgs::PointCloud2ConstPtr &cloud_msg,
                                     const nav_msgs::OdometryConstPtr &odom_msg);

    /**
     * @brief 阻塞等待，直到队列中的点云全部处理完成
     */
    void waitUntilIdle();

    /**
     * @brief 立即触发一次栅格地图发布
     */
    void publishGridMapOnce();

    /**
     * @brief 将当前局部栅格地图导出为 ROS 消息，但不发布
     */
    bool exportGridMap(grid_map_msgs::GridMap &message);

private:
    // ROS 句柄
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    ros::Subscriber registered_scan_sub_;
    ros::Subscriber odometry_sub_;
    ros::Publisher cloud_pub_;
    ros::Publisher gridmap_pub_;
    ros::Publisher viewpoint_origin_publisher;
    ros::Timer publish_timer_; // 用于固定频率发布的定时器

    // // 使用 message_filters 进行时间同步
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, nav_msgs::Odometry> SyncPolicy;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> cloud_filter_sub_;
    std::shared_ptr<message_filters::Subscriber<nav_msgs::Odometry>> odom_filter_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    // 发布线程控制
    std::thread publish_thread_;
    bool should_exit_;
    std::mutex gridmap_publish_mutex_; // 发布时保护栅格地图访问

    // 处理线程控制
    std::thread processing_thread_;
    std::mutex processing_queue_mutex_;
    std::condition_variable processing_cv_;
    struct ProcessingTask
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
        Eigen::Vector3d position;
        Eigen::Quaterniond orientation;
        ros::Time timestamp;
        ros::WallTime enqueue_wall_time;
        double ros_to_pcl_ms;
    };
    std::queue<ProcessingTask> processing_queue_;
    bool skip_old_messages_;   // 若为 true，当新消息到来时跳过队列中的旧消息
    size_t active_task_count_; // 当前正在处理的任务数量

    struct RuntimeCycleMetrics
    {
        std::size_t input_point_count;
        std::size_t filtered_point_count;
        bool cycle_valid;
        std::string status;
        double ros_to_pcl_ms;
        double queue_wait_ms;
        double task_setup_ms;
        double preprocessing_ms;
        double height_mapping_ms;
        double bgk_mapping_ms;
        double startup_hole_fill_ms;
        double geometric_mapping_ms;
        double step_mapping_ms;
        double traversability_mapping_ms;
        double fine_traversability_mapping_ms;
        std::uint64_t fine_pose_call_count;
        std::uint64_t fine_pose_iteration_total;
        int fine_pose_iteration_max;
        std::uint64_t fine_pose_iteration_limit_count;
        double viewpoint_update_ms;
        double map_publication_ms;

        RuntimeCycleMetrics();
    };

    bool enable_runtime_csv_;
    std::string runtime_csv_output_dir_;
    std::string runtime_csv_path_;
    std::ofstream runtime_csv_file_;
    std::uint64_t runtime_cycle_index_;
    ros::WallTime runtime_log_start_wall_time_;
    RuntimeCycleMetrics current_runtime_metrics_;

    // 点云存储
    pcl::PointCloud<pcl::PointXYZ>::Ptr current_cloud_;

    // 车辆在全局坐标系下的位姿（由互斥锁保护）
    std::mutex vehicle_pose_mutex_;
    Eigen::Vector3d vehicle_position_;       // 全局坐标系下的位置 x, y, z
    Eigen::Quaterniond vehicle_orientation_; // 全局坐标系下的四元数姿态
    bool vehicle_pose_initialized_;          // 标记是否已接收到位姿

    // 用于高度建图的 GridMap
    grid_map::GridMap height_map_;

    // 参数：基础设置
    std::string scan_topic_;
    std::string odom_topic_;
    std::string frame_id_;
    int queue_size_;
    bool debug_mode_;

    // 参数：传感器量程
    double max_range_;
    double min_range_;
    double min_z_;
    double max_z_;

    // 参数：地图设置
    double map_resolution_;
    double map_length_x_;     // 全局地图尺寸 X（例如 1000 m）
    double map_length_y_;     // 全局地图尺寸 Y（例如 1000 m）
    double local_map_size_x_; // 局部可视化地图尺寸 X（例如 40 m）
    double local_map_size_y_; // 局部可视化地图尺寸 Y（例如 40 m）

    // 参数：BGK 推断核大小
    double bgk_kernel_size_;

    // 参数：可通行性阈值
    double slope_threshold_;
    double roughness_threshold_;
    double step_threshold_;
    double init_trav_threshold_;

    // 参数：法向估计与特征检测
    double normal_estimation_radius_;
    double step_radius_;
    int min_normal_points_;
    int thread_count_;

    // 机器人位姿
    Eigen::Vector3d robot_pose_;

    // 车辆模型参数
    struct HeightGrid
    {
        Eigen::MatrixXd X_;
        Eigen::MatrixXd Y_;
        Eigen::MatrixXd Z_;
        Eigen::MatrixXd Gap_;
    };

    struct PrecomputedVehicleModel
    {
        Eigen::Matrix3d heading_rotation = Eigen::Matrix3d::Identity();
        std::vector<Eigen::Vector4d> points;
        std::vector<Eigen::Vector2d> support_xy;
        std::vector<unsigned char> collision_sensitive;
    };

    std::string wheeled_model_path_;
    std::string tracked_model_path_;
    HeightGrid wheeled_model_;
    HeightGrid tracked_model_;
    bool wheeled_model_loaded_;
    bool tracked_model_loaded_;
    int robot_rows_;
    int robot_cols_;
    double robot_model_resolution_;
    double touch_gap_threshold_;
    double collision_gap_threshold_;
    int max_iterations_;
    bool enable_fine_traversability_; // 是否启用细粒度可通行性计算
    bool enable_incremental_geom_;    // 是否启用增量几何建图
    bool enable_incremental_step_;    // 是否启用增量台阶建图
    bool enable_incremental_trav_;    // 是否启用增量可通行性建图

    // 参数：细粒度可通行性范围过滤
    double fine_trav_min_;            // 进行细粒度计算的最小可通行性值
    double fine_trav_max_;            // 进行细粒度计算的最大可通行性值
    double fine_slope_min_;           // 进行细粒度计算的最小坡度值
    double fine_slope_max_;           // 进行细粒度计算的最大坡度值
    double fine_roughness_min_;       // 进行细粒度计算的最小粗糙度值
    double fine_roughness_max_;       // 进行细粒度计算的最大粗糙度值
    double fine_roll_threshold_deg_;  // 横滚角归一化阈值（单位：度）
    double fine_pitch_threshold_deg_; // 俯仰角归一化阈值（单位：度）

    // cell counts
    int viewpoint_number_x_ = 80;
    int viewpoint_number_y_ = 80;
    int viewpoint_number_z_ = 40;

    // cell resolution (meters)
    double viewpoint_resolution_x_ = 0.5;
    double viewpoint_resolution_y_ = 0.5;
    double viewpoint_resolution_z_ = 0.5;

    // derived: physical size (meters)
    double viewpoint_grid_size_x_ = 0.0;
    double viewpoint_grid_size_y_ = 0.0;
    double viewpoint_grid_size_z_ = 0.0;

    // =========================
    // viewpoint vis cloud cache (latest frame)
    // =========================
    ros::Subscriber viewpoint_vis_cloud_sub_;
    std::string viewpoint_vis_topic_;

    mutable std::mutex viewpoint_vis_mutex_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr latest_viewpoint_vis_cloud_{new pcl::PointCloud<pcl::PointXYZI>()};
    ros::Time latest_viewpoint_vis_stamp_;
    std::string latest_viewpoint_vis_frame_id_;
    std::atomic<bool> has_viewpoint_vis_{false};

    ros::Subscriber origin_sub_;
    mutable std::mutex origin_mutex_;
    Eigen::Vector3d latest_origin_ = Eigen::Vector3d::Zero();
    ros::Time latest_origin_stamp_;
    std::atomic<bool> has_origin_{false};

    std::string origin_topic_; // 可从参数读取

    // ...
    void originCallback(const geometry_msgs::PointStampedConstPtr &msg);

    // =========================
    // viewpoint vis cloud callback
    // =========================
    void viewpointVisCloudCallback(const sensor_msgs::PointCloud2ConstPtr &msg);

    /**
     * @brief 同步点云与里程计订阅的回调函数
     */
    void synchronizedCloudOdomCallback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg,
                                       const nav_msgs::OdometryConstPtr &odom_msg);
    // void synchronizedCloudOdomCallback(
    //     const sensor_msgs::PointCloud2ConstPtr &scan_msg,
    //     const nav_msgs::OdometryConstPtr &odom_msg,
    //     const sensor_msgs::PointCloud2ConstPtr &terrain_msg,
    //     const sensor_msgs::PointCloud2ConstPtr &terrain_ext_msg);

    /**
     * @brief 处理线程函数：异步处理队列中的点云
     */
    void processingThreadFunc();

    /**
     * @brief 加载 ROS 参数
     */
    void loadParameters();

    /**
     * @brief 初始化逐周期模块耗时 CSV（每次启动覆盖旧文件）
     */
    void initializeRuntimeCSV();

    /**
     * @brief 写入当前点云处理周期的模块耗时
     */
    void writeRuntimeCycleToCSV(const ros::Time &source_stamp,
                                double cycle_wall_ms);

    /**
     * @brief 处理输入点云
     */
    void processPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, Eigen::Vector3d current_pos);

    /**
     * @brief 高度建图：将点云转换为包含高度层的栅格地图
     */
    void height_mapping(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);

    /**
     * @brief BGK 建图：使用稀疏核推断对高度图进行插值
     */
    void bgk_mapping();

    /**
     * @brief 几何建图：由高度计算坡度、粗糙度与台阶等特征
     */
    void geometric_mapping();

    /**
     * @brief 台阶建图：计算台阶特征
     */
    void step_mapping();

    /**
     * @brief 可通行性建图：计算初始可通行性代价
     */
    void traversability_mapping();

    /**
     * @brief 使用车辆模型进行细粒度可通行性建图
     */
    void finegrained_traversability_mapping();

    /**
     * @brief 从 CSV 文件加载机器人模型
     */
    void loadRobotModels();
    bool loadModelFromCsv(const std::string &path, HeightGrid &model_storage);

    /**
     * @brief 使用迭代方法在地形上预测机器人姿态
     * @param index: 需要预测的栅格索引
     * @param roll: 输出横滚角（单位：度）
     * @param pitch: 输出俯仰角（单位：度）
     * @param contact_points: 输出接触点数量
     * @param stable: 输出稳定性状态（1 表示稳定，0 表示不稳定）
     * @param yaw_rotation: 预先计算的偏航旋转矩阵，用于保证一致性
     * @return: 0 表示失败，1 表示发生碰撞，2 表示成功且稳定
     */
    int predictRobotPose(const grid_map::Index &index, double &roll, double &pitch,
                         int &contact_points, int &stable,
                         const PrecomputedVehicleModel &vehicle_model,
                         int vehicle_type, int &iterations_used,
                         std::string &exit_reason);

    /**
     * @brief 判断某个栅格点是否处于可接触区域（例如轮子区域）
     * @param i, j: 车辆模型中的栅格索引
     * @return: false 表示可接触（轮子），true 表示不可接触（车体底盘）
     */
    bool checkWheel(int vehicle_type, int i, int j) const;

    /**
     * @brief 使用最小二乘从点集拟合平面
     */
    Eigen::Vector3d fitPlane(const std::vector<Eigen::Vector3d> &points);

    /**
     * @brief 初始化栅格地图并创建图层
     */
    void initGridMap();

    /**
     * @brief 计算单个栅格单元的法向量
     */
    bool areaSingleNormalComputation(const grid_map::Index &index);

    /**
     * @brief 计算单个栅格单元的台阶特征
     */
    bool areaSingleStepComputation(const grid_map::Index &index);

    /**
     * @brief 计算两组点集之间的距离矩阵
     */
    void dist(const Eigen::MatrixXf &xStar, const Eigen::MatrixXf &xTrain, Eigen::MatrixXf &d) const;

    /**
     * @brief 计算稀疏协方差核矩阵
     */
    void covSparse(const Eigen::MatrixXf &xStar, const Eigen::MatrixXf &xTrain, Eigen::MatrixXf &Kxz) const;

    /**
     * @brief 更新高度统计量
     */
    void updateHeightStats(float &height, float &variance, float n, float new_height);

    /**
     * @brief 将栅格地图发布为 ROS 消息
     */
    void publishGridMap();

    /**
     * @brief 构建局部栅格地图消息
     */
    bool buildGridMapMessage(grid_map_msgs::GridMap &message);

    bool buildGridMapFromViewpointCloud(grid_map_msgs::GridMap &message);

    /**
     * @brief 固定频率发布栅格地图的定时器回调
     */
    void publishTimerCallback(const ros::TimerEvent &event);
};

#endif // MULTI_ROBOT_MAPPING_H
/**
 * @description: 多机器人建图类
 * @filename: multi_robot_mapping.h
 * @author: wangxurui
 * @date: 2026-01-28
 **/

#ifndef MULTI_ROBOT_MAPPING_H
#define MULTI_ROBOT_MAPPING_H

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h> // removeNaNFromPointCloud

#include <grid_map_core/grid_map_core.hpp>
#include <grid_map_ros/grid_map_ros.hpp>
#include <Eigen/Dense>
#include <tf/transform_datatypes.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <geometry_msgs/PointStamped.h>
#include <fstream>
#include <sstream>
#include <mutex>
#include <thread>
#include <queue>
#include <condition_variable>
#include <unordered_set>
#include <atomic>

class Mapping
{
public:
    /**
     * @brief 构造函数
     * @param nh: ROS 节点句柄
     */
    Mapping(ros::NodeHandle &nh, ros::NodeHandle &pnh);

    /**
     * @brief 析构函数
     */
    ~Mapping();

    /**
     * @brief 获取当前点云
     */
    pcl::PointCloud<pcl::PointXYZ>::Ptr getPointCloud() const;

    /**
     * @brief 处理一对同步的点云与里程计消息（可用于离线回放）
     */
    void processSynchronizedMessages(const sensor_msgs::PointCloud2ConstPtr &cloud_msg,
                                     const nav_msgs::OdometryConstPtr &odom_msg);

    /**
     * @brief 阻塞等待，直到队列中的点云全部处理完成
     */
    void waitUntilIdle();

    /**
     * @brief 立即触发一次栅格地图发布
     */
    void publishGridMapOnce();

    /**
     * @brief 将当前局部栅格地图导出为 ROS 消息，但不发布
     */
    bool exportGridMap(grid_map_msgs::GridMap &message);

private:
    // ROS 句柄
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    ros::Subscriber registered_scan_sub_;
    ros::Subscriber odometry_sub_;
    ros::Publisher cloud_pub_;
    ros::Publisher gridmap_pub_;
    ros::Timer publish_timer_; // 用于固定频率发布的定时器

    // // 使用 message_filters 进行时间同步
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, nav_msgs::Odometry> SyncPolicy;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> cloud_filter_sub_;
    std::shared_ptr<message_filters::Subscriber<nav_msgs::Odometry>> odom_filter_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    // // 4路 ApproximateTime policy（顺序必须和 callback 参数一致）
    // using SyncPolicy = message_filters::sync_policies::ApproximateTime<
    //     sensor_msgs::PointCloud2, nav_msgs::Odometry,
    //     sensor_msgs::PointCloud2, sensor_msgs::PointCloud2>;

    // std::shared_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> cloud_filter_sub_;
    // std::shared_ptr<message_filters::Subscriber<nav_msgs::Odometry>> odom_filter_sub_;

    // // // 新增两个点云订阅
    // std::shared_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> terrain_filter_sub_;
    // std::shared_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> terrain_ext_filter_sub_;

    // std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    // 发布线程控制
    std::thread publish_thread_;
    bool should_exit_;
    std::mutex gridmap_publish_mutex_; // 发布时保护栅格地图访问

    // 处理线程控制
    std::thread processing_thread_;
    std::mutex processing_queue_mutex_;
    std::condition_variable processing_cv_;
    struct ProcessingTask
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
        Eigen::Vector3d position;
        Eigen::Quaterniond orientation;
        ros::Time timestamp;
    };
    std::queue<ProcessingTask> processing_queue_;
    bool skip_old_messages_;   // 若为 true，当新消息到来时跳过队列中的旧消息
    size_t active_task_count_; // 当前正在处理的任务数量

    // 点云存储
    pcl::PointCloud<pcl::PointXYZ>::Ptr current_cloud_;

    // 车辆在全局坐标系下的位姿（由互斥锁保护）
    std::mutex vehicle_pose_mutex_;
    Eigen::Vector3d vehicle_position_;       // 全局坐标系下的位置 x, y, z
    Eigen::Quaterniond vehicle_orientation_; // 全局坐标系下的四元数姿态
    bool vehicle_pose_initialized_;          // 标记是否已接收到位姿

    // 用于高度建图的 GridMap
    grid_map::GridMap height_map_;

    // 参数：基础设置
    std::string scan_topic_;
    std::string frame_id_;
    int queue_size_;
    bool debug_mode_;

    // 参数：传感器量程
    double max_range_;
    double min_range_;
    double min_z_;
    double max_z_;

    // 参数：地图设置
    double map_resolution_;
    double map_length_x_;     // 全局地图尺寸 X（例如 1000 m）
    double map_length_y_;     // 全局地图尺寸 Y（例如 1000 m）
    double local_map_size_x_; // 局部可视化地图尺寸 X（例如 40 m）
    double local_map_size_y_; // 局部可视化地图尺寸 Y（例如 40 m）

    // 参数：BGK 推断核大小
    double bgk_kernel_size_;

    // 参数：可通行性阈值
    double slope_threshold_;
    double roughness_threshold_;
    double step_threshold_;
    double init_trav_threshold_;

    // 参数：法向估计与特征检测
    double normal_estimation_radius_;
    double step_radius_;
    int min_normal_points_;
    int thread_count_;

    // 机器人位姿
    Eigen::Vector3d robot_pose_;

    // 车辆模型参数
    struct HeightGrid
    {
        Eigen::MatrixXd X_;
        Eigen::MatrixXd Y_;
        Eigen::MatrixXd Z_;
        Eigen::MatrixXd Gap_;
    };

    std::string wheeled_model_path_;
    std::string tracked_model_path_;
    HeightGrid wheeled_model_;
    HeightGrid tracked_model_;
    bool wheeled_model_loaded_;
    bool tracked_model_loaded_;
    int robot_rows_;
    int robot_cols_;
    double robot_model_resolution_;
    double touch_gap_threshold_;
    double collision_gap_threshold_;
    int max_iterations_;
    bool enable_fine_traversability_; // 是否启用细粒度可通行性计算
    bool enable_incremental_geom_;    // 是否启用增量几何建图
    bool enable_incremental_step_;    // 是否启用增量台阶建图
    bool enable_incremental_trav_;    // 是否启用增量可通行性建图

    // 参数：细粒度可通行性范围过滤
    double fine_trav_min_;            // 进行细粒度计算的最小可通行性值
    double fine_trav_max_;            // 进行细粒度计算的最大可通行性值
    double fine_slope_min_;           // 进行细粒度计算的最小坡度值
    double fine_slope_max_;           // 进行细粒度计算的最大坡度值
    double fine_roughness_min_;       // 进行细粒度计算的最小粗糙度值
    double fine_roughness_max_;       // 进行细粒度计算的最大粗糙度值
    double fine_roll_threshold_deg_;  // 横滚角归一化阈值（单位：度）
    double fine_pitch_threshold_deg_; // 俯仰角归一化阈值（单位：度）

    // cell counts
    int viewpoint_number_x_ = 80;
    int viewpoint_number_y_ = 80;
    int viewpoint_number_z_ = 40;

    // cell resolution (meters)
    double viewpoint_resolution_x_ = 0.5;
    double viewpoint_resolution_y_ = 0.5;
    double viewpoint_resolution_z_ = 0.5;

    // derived: physical size (meters)
    double viewpoint_grid_size_x_ = 0.0;
    double viewpoint_grid_size_y_ = 0.0;
    double viewpoint_grid_size_z_ = 0.0;

    // =========================
    // viewpoint vis cloud cache (latest frame)
    // =========================
    ros::Subscriber viewpoint_vis_cloud_sub_;
    std::string viewpoint_vis_topic_;

    mutable std::mutex viewpoint_vis_mutex_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr latest_viewpoint_vis_cloud_{new pcl::PointCloud<pcl::PointXYZI>()};
    ros::Time latest_viewpoint_vis_stamp_;
    std::string latest_viewpoint_vis_frame_id_;
    std::atomic<bool> has_viewpoint_vis_{false};

    ros::Subscriber origin_sub_;
    mutable std::mutex origin_mutex_;
    Eigen::Vector3d latest_origin_ = Eigen::Vector3d::Zero();
    ros::Time latest_origin_stamp_;
    std::atomic<bool> has_origin_{false};

    std::string origin_topic_; // 可从参数读取

    // ...
    void originCallback(const geometry_msgs::PointStampedConstPtr &msg);

    // =========================
    // viewpoint vis cloud callback
    // =========================
    void viewpointVisCloudCallback(const sensor_msgs::PointCloud2ConstPtr &msg);

    /**
     * @brief 同步点云与里程计订阅的回调函数
     */
    void synchronizedCloudOdomCallback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg,
                                       const nav_msgs::OdometryConstPtr &odom_msg);
    // void synchronizedCloudOdomCallback(
    //     const sensor_msgs::PointCloud2ConstPtr &scan_msg,
    //     const nav_msgs::OdometryConstPtr &odom_msg,
    //     const sensor_msgs::PointCloud2ConstPtr &terrain_msg,
    //     const sensor_msgs::PointCloud2ConstPtr &terrain_ext_msg);

    /**
     * @brief 处理线程函数：异步处理队列中的点云
     */
    void processingThreadFunc();

    /**
     * @brief 加载 ROS 参数
     */
    void loadParameters();

    /**
     * @brief 处理输入点云
     */
    void processPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, Eigen::Vector3d current_pos);

    /**
     * @brief 高度建图：将点云转换为包含高度层的栅格地图
     */
    void height_mapping(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);

    /**
     * @brief BGK 建图：使用稀疏核推断对高度图进行插值
     */
    void bgk_mapping();

    /**
     * @brief 几何建图：由高度计算坡度、粗糙度与台阶等特征
     */
    void geometric_mapping();

    /**
     * @brief 台阶建图：计算台阶特征
     */
    void step_mapping();

    /**
     * @brief 可通行性建图：计算初始可通行性代价
     */
    void traversability_mapping();

    /**
     * @brief 按轮式车辆能力阈值计算平台相关粗略可通行性
     */
    void wheeled_coarse_traversability_mapping();

    /**
     * @brief 按履带式车辆能力阈值计算平台相关粗略可通行性
     */
    void tracked_coarse_traversability_mapping();

    /**
     * @brief 使用车辆模型进行细粒度可通行性建图
     */
    void finegrained_traversability_mapping();

    /**
     * @brief 从 CSV 文件加载机器人模型
     */
    void loadRobotModels();
    bool loadModelFromCsv(const std::string &path, HeightGrid &model_storage);

    /**
     * @brief 使用迭代方法在地形上预测机器人姿态
     * @param index: 需要预测的栅格索引
     * @param roll: 输出横滚角（单位：度）
     * @param pitch: 输出俯仰角（单位：度）
     * @param contact_points: 输出接触点数量
     * @param stable: 输出稳定性状态（1 表示稳定，0 表示不稳定）
     * @param yaw_rotation: 预先计算的偏航旋转矩阵，用于保证一致性
     * @return: 0 表示失败，1 表示发生碰撞，2 表示成功且稳定
     */
    int predictRobotPose(const grid_map::Index &index, double &roll, double &pitch,
                         int &contact_points, int &stable, const Eigen::Matrix3d &yaw_rotation,
                         const HeightGrid &vehicle_model, int vehicle_type);

    /**
     * @brief 判断某个栅格点是否处于可接触区域（例如轮子区域）
     * @param i, j: 车辆模型中的栅格索引
     * @return: false 表示可接触（轮子），true 表示不可接触（车体底盘）
     */
    bool checkWheel(int vehicle_type, int i, int j) const;

    /**
     * @brief 使用最小二乘从点集拟合平面
     */
    Eigen::Vector3d fitPlane(const std::vector<Eigen::Vector3d> &points);

    /**
     * @brief 初始化栅格地图并创建图层
     */
    void initGridMap();

    /**
     * @brief 计算单个栅格单元的法向量
     */
    bool areaSingleNormalComputation(const grid_map::Index &index);

    /**
     * @brief 计算单个栅格单元的台阶特征
     */
    bool areaSingleStepComputation(const grid_map::Index &index);

    /**
     * @brief 计算两组点集之间的距离矩阵
     */
    void dist(const Eigen::MatrixXf &xStar, const Eigen::MatrixXf &xTrain, Eigen::MatrixXf &d) const;

    /**
     * @brief 计算稀疏协方差核矩阵
     */
    void covSparse(const Eigen::MatrixXf &xStar, const Eigen::MatrixXf &xTrain, Eigen::MatrixXf &Kxz) const;

    /**
     * @brief 更新高度统计量
     */
    void updateHeightStats(float &height, float &variance, float n, float new_height);

    /**
     * @brief 将栅格地图发布为 ROS 消息
     */
    void publishGridMap();

    /**
     * @brief 构建局部栅格地图消息
     */
    bool buildGridMapMessage(grid_map_msgs::GridMap &message);

    bool buildGridMapFromViewpointCloud(grid_map_msgs::GridMap &message);

    /**
     * @brief 固定频率发布栅格地图的定时器回调
     */
    void publishTimerCallback(const ros::TimerEvent &event);
};

#endif // MULTI_ROBOT_MAPPING_H
