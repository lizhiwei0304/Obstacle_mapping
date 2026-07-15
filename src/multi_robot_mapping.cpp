/**
 * @description: Multi-robot mapping node implementation
 * @filename: multi_robot_mapping.cpp
 * @author: wangxurui
 * @date: 2026-01-28
 **/

#include "multi_robot_mapping.h"
#include <ros/ros.h>
#include <ros/package.h>
#include <queue>
#include <algorithm>
#include <limits>
#include <array>
#include <vector>
#include <set>
#include <utility>
#include <stdexcept>

#ifndef MULTI_ROBOT_MAPPING_NO_MAIN
/**
 * @brief Main function - ROS node entry point
 */
int main(int argc, char **argv)
{
  ros::init(argc, argv, "multi_robot_mapping_node");
  ros::NodeHandle pnh("~");
  ros::NodeHandle nh;

  ROS_INFO("Starting multi_robot_mapping_node...");

  Mapping mapping(nh, pnh);

  ROS_INFO("multi_robot_mapping_node is running");

  ros::spin();

  ROS_INFO("Shutting down multi_robot_mapping_node");
  return 0;
}
#endif

// ==================== Mapping Class Implementation ==================== //
Mapping::Mapping(ros::NodeHandle &nh, ros::NodeHandle &pnh)
    : nh_(nh),
      pnh_(pnh),
      current_cloud_(new pcl::PointCloud<pcl::PointXYZ>()),
      scan_topic_("registered_scan"),
      frame_id_("map"),
      queue_size_(10),
      debug_mode_(false),
      max_range_(20.0),
      min_range_(0.1),
      min_z_(-1.0),
      max_z_(0.1),
      map_resolution_(0.2),
      map_length_x_(1000.0),
      map_length_y_(1000.0),
      bgk_kernel_size_(0.8),
      slope_threshold_(0.3),
      roughness_threshold_(0.1),
      step_threshold_(0.2),
      init_trav_threshold_(0.5),
      normal_estimation_radius_(0.5),
      step_radius_(0.3),
      min_normal_points_(6),
      thread_count_(4),
      robot_pose_(Eigen::Vector3d::Zero()),
      robot_rows_(9),
      robot_cols_(11),
      robot_model_resolution_(0.2),
      touch_gap_threshold_(0.05),
      collision_gap_threshold_(0.1),
      wheeled_model_loaded_(false),
      tracked_model_loaded_(false),
      max_iterations_(50),
      enable_fine_traversability_(true),
      enable_incremental_geom_(true),
      enable_incremental_step_(true),
      enable_incremental_trav_(true),
      fine_trav_min_(0.4),
      fine_trav_max_(0.9),
      fine_slope_min_(0.2),
      fine_slope_max_(0.8),
      fine_roughness_min_(0.5),
      fine_roughness_max_(1.0),
      fine_roll_threshold_deg_(30.0),
      fine_pitch_threshold_deg_(30.0),
      vehicle_position_(Eigen::Vector3d::Zero()),
      vehicle_orientation_(Eigen::Quaterniond::Identity()),
      vehicle_pose_initialized_(false),
      local_map_size_x_(40.0),
      local_map_size_y_(40.0),
      should_exit_(false),
      active_task_count_(0)
{
  ROS_INFO("Mapping constructor called");

  ROS_INFO("Mapping::init() - Initializing mapping node");

  loadParameters();

  loadRobotModels();

  initGridMap();

  // ROS_INFO("Subscribed to origin topic: %s", origin_topic_.c_str());
  // origin_sub_ = nh_.subscribe<geometry_msgs::PointStamped>(origin_topic_, 10, &Mapping::originCallback, this);

  // ROS_INFO("Subscribed to viewpoint_vis topic: %s", viewpoint_vis_topic_.c_str());
  // viewpoint_vis_cloud_sub_ = nh_.subscribe(viewpoint_vis_topic_, 5, &Mapping::viewpointVisCloudCallback, this);

  // Subscribe to point cloud topics
  ROS_INFO("Subscribing to point cloud and odometry topics with time synchronization...");

  // Create message filters for synchronized subscription
  cloud_filter_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::PointCloud2>>(
      nh_, scan_topic_, queue_size_);

  odom_filter_sub_ = std::make_shared<message_filters::Subscriber<nav_msgs::Odometry>>(
      nh_, "/vehicle0/state_estimation", queue_size_);

  // Create approximate time synchronizer (allows small time differences)
  // Queue size of 10 means it will try to synchronize the last 10 messages from each topic
  sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
      SyncPolicy(queue_size_), *cloud_filter_sub_, *odom_filter_sub_);

  sync_->registerCallback(boost::bind(&Mapping::synchronizedCloudOdomCallback, this, _1, _2));

  // Create publisher for grid map
  gridmap_pub_ = nh_.advertise<grid_map_msgs::GridMap>("trav_map", 10);

  ROS_INFO("Successfully subscribed to: %s (synchronized)", scan_topic_.c_str());
  ROS_INFO("Successfully subscribed to: state_estimation (synchronized)");
  ROS_INFO("Publishing grid map on topic: trav_map");

  // Start processing thread for asynchronous point cloud processing
  should_exit_ = false;
  processing_thread_ = std::thread(&Mapping::processingThreadFunc, this);
  ROS_INFO("Started asynchronous processing thread (skip_old_messages: %s)",
           skip_old_messages_ ? "true" : "false");

  ROS_INFO("Mapping initialization complete");
  ROS_INFO("Subscribed to topic: %s", scan_topic_.c_str());
  ROS_INFO("Grid map will be published after each processed synchronized message pair");
}

Mapping::~Mapping()
{
  // Stop the publishing timer
  publish_timer_.stop();

  // Signal the processing thread to exit
  should_exit_ = true;

  // Notify processing thread to wake up and exit if it's waiting
  processing_cv_.notify_one();

  // Wait for processing thread to finish
  if (processing_thread_.joinable())
  {
    ROS_INFO("Waiting for processing thread to finish...");
    processing_thread_.join();
    ROS_INFO("Processing thread joined successfully");
  }

  ROS_INFO("Mapping destructor called");
}

void Mapping::loadParameters()
{
  ROS_INFO("Loading ROS parameters...");

  // Load topic settings
  pnh_.param<std::string>("scan_topic", scan_topic_, "registered_scan");
  pnh_.param<std::string>("origin_topic", origin_topic_, "tare_planner_node/viewpoint_origin");
  pnh_.param<std::string>("viewpoint_vis_topic", viewpoint_vis_topic_, "viewpoint_vis_cloud");
  pnh_.param<std::string>("frame_id", frame_id_, "map");
  pnh_.param<int>("queue_size", queue_size_, 200);

  // Load vehicle and debug settings
  pnh_.param<bool>("debug_mode", debug_mode_, false);

  const std::string package_path = ros::package::getPath("obstacle_mapping");
  const std::string vehicles_dir = package_path + "/test_map/vehicles";
  const std::string default_wheeled_model_path = vehicles_dir + "/wheeled_car.csv";
  const std::string default_tracked_model_path = vehicles_dir + "/tracked_car.csv";

  pnh_.param<std::string>("wheeled_model_path", wheeled_model_path_, default_wheeled_model_path);
  pnh_.param<std::string>("tracked_model_path", tracked_model_path_, default_tracked_model_path);

  // Load sensor range parameters
  pnh_.param<double>("max_range", max_range_, 100.0);
  pnh_.param<double>("min_range", min_range_, 0.1);

  // Load point cloud Z range parameters
  pnh_.param<double>("min_z", min_z_, -1.0);
  pnh_.param<double>("max_z", max_z_, 0.1);

  // Load map parameters
  pnh_.param<double>("map_resolution", map_resolution_, 0.2);
  pnh_.param<double>("map_length_x", map_length_x_, 1000.0);
  pnh_.param<double>("map_length_y", map_length_y_, 1000.0);
  pnh_.param<double>("local_map_size_x", local_map_size_x_, 40.0);
  pnh_.param<double>("local_map_size_y", local_map_size_y_, 40.0);

  // Load BGK inference kernel size
  pnh_.param<double>("bgk_kernel_size", bgk_kernel_size_, 0.8);

  // Load traversability thresholds
  pnh_.param<double>("slope_threshold", slope_threshold_, 0.3);
  pnh_.param<double>("roughness_threshold", roughness_threshold_, 0.1);
  pnh_.param<double>("step_threshold", step_threshold_, 0.2);
  pnh_.param<double>("init_trav_threshold", init_trav_threshold_, 0.5);

  // Load feature mapping parameters
  pnh_.param<double>("normal_estimation_radius", normal_estimation_radius_, 0.5);
  pnh_.param<double>("step_radius", step_radius_, 0.3);
  pnh_.param<int>("min_normal_points", min_normal_points_, 6);
  pnh_.param<int>("thread_count", thread_count_, 4);

  // Load vehicle model parameters
  pnh_.param<int>("robot_rows", robot_rows_, 9);
  pnh_.param<int>("robot_cols", robot_cols_, 11);
  pnh_.param<double>("robot_model_resolution", robot_model_resolution_, 0.2);
  pnh_.param<double>("touch_gap_threshold", touch_gap_threshold_, 0.05);
  pnh_.param<double>("collision_gap_threshold", collision_gap_threshold_, 0.1);
  pnh_.param<int>("max_iterations", max_iterations_, 50);
  pnh_.param<bool>("enable_fine_traversability", enable_fine_traversability_, false);

  // Load fine-grained traversability range parameters
  pnh_.param<double>("fine_trav_min", fine_trav_min_, 0.4);
  pnh_.param<double>("fine_trav_max", fine_trav_max_, 0.9);
  pnh_.param<double>("fine_slope_min", fine_slope_min_, 0.2);
  pnh_.param<double>("fine_slope_max", fine_slope_max_, 0.8);
  pnh_.param<double>("fine_roughness_min", fine_roughness_min_, 0.5);
  pnh_.param<double>("fine_roughness_max", fine_roughness_max_, 1.0);
  pnh_.param<double>("fine_roll_threshold_deg", fine_roll_threshold_deg_, 30.0);
  pnh_.param<double>("fine_pitch_threshold_deg", fine_pitch_threshold_deg_, 30.0);

  // Load incremental mapping parameters (default to enabled)
  pnh_.param<bool>("enable_incremental_geom", enable_incremental_geom_, true);
  pnh_.param<bool>("enable_incremental_step", enable_incremental_step_, true);
  pnh_.param<bool>("enable_incremental_trav", enable_incremental_trav_, true);

  // Load asynchronous processing parameters
  pnh_.param<bool>("skip_old_messages", skip_old_messages_, false);

  // // Load local grid map parameters
  // int nx, ny, nz;
  // double rx, ry, rz;

  // ros::Rate rate(10); // 10 Hz
  // while (ros::ok())
  // {
  //   std::string ns = ros::this_node::getNamespace();
  //   const std::string vp_prefix = ns + "/tare_planner_node/viewpoint_manager/";
  //   const bool ok =
  //       nh_.getParam(vp_prefix + "number_x", nx) &&
  //       nh_.getParam(vp_prefix + "number_y", ny) &&
  //       nh_.getParam(vp_prefix + "number_z", nz) &&
  //       nh_.getParam(vp_prefix + "resolution_x", rx) &&
  //       nh_.getParam(vp_prefix + "resolution_y", ry) &&
  //       nh_.getParam(vp_prefix + "resolution_z", rz);

  //   if (ok)
  //     break;

  //   ROS_WARN_THROTTLE(2.0,
  //                     "Waiting for viewpoint_manager params... "
  //                     "(number_x/y/z, resolution_x/y/z)");
  //   rate.sleep();
  // }

  // // ros::ok() 变 false 时就直接返回，避免继续用未定义值
  // if (!ros::ok())
  //   return;

  // // 赋值到成员变量
  // viewpoint_number_x_ = nx;
  // viewpoint_number_y_ = ny;
  // viewpoint_number_z_ = nz;
  // viewpoint_resolution_x_ = rx;
  // viewpoint_resolution_y_ = ry;
  // viewpoint_resolution_z_ = rz;

  // viewpoint_grid_size_x_ = viewpoint_number_x_ * viewpoint_resolution_x_;
  // viewpoint_grid_size_y_ = viewpoint_number_y_ * viewpoint_resolution_y_;
  // viewpoint_grid_size_z_ = viewpoint_number_z_ * viewpoint_resolution_z_;

  // Log loaded parameters
  ROS_INFO("Parameters loaded:");
  ROS_INFO("  - scan_topic: %s", scan_topic_.c_str());
  ROS_INFO("  - frame_id: %s", frame_id_.c_str());
  ROS_INFO("  - queue_size: %d", queue_size_);
  ROS_INFO("  - debug_mode: %s", debug_mode_ ? "true" : "false");
  ROS_INFO("  - Sensor range - max: %.2f, min: %.2f", max_range_, min_range_);
  ROS_INFO("  - Point cloud Z range - min: %.2f, max: %.2f", min_z_, max_z_);
  ROS_INFO("  - Map resolution: %.4f", map_resolution_);
  ROS_INFO("  - Global map size - length_x: %.2f, length_y: %.2f", map_length_x_, map_length_y_);
  ROS_INFO("  - Local map size - length_x: %.2f, length_y: %.2f", local_map_size_x_, local_map_size_y_);
  ROS_INFO("  - BGK kernel size: %.4f", bgk_kernel_size_);
  ROS_INFO("  - Slope threshold: %.4f", slope_threshold_);
  ROS_INFO("  - Roughness threshold: %.4f", roughness_threshold_);
  ROS_INFO("  - Step threshold: %.4f", step_threshold_);
  ROS_INFO("  - Init traversability threshold: %.4f", init_trav_threshold_);
  ROS_INFO("  - Normal estimation radius: %.4f", normal_estimation_radius_);
  ROS_INFO("  - Step radius: %.4f", step_radius_);
  ROS_INFO("  - Min normal points: %d", min_normal_points_);
  ROS_INFO("  - Thread count: %d", thread_count_);
  ROS_INFO("  - Incremental geom enabled: %s", enable_incremental_geom_ ? "true" : "false");
  ROS_INFO("  - Incremental step enabled: %s", enable_incremental_step_ ? "true" : "false");
  ROS_INFO("  - Incremental trav enabled: %s", enable_incremental_trav_ ? "true" : "false");
  ROS_INFO("  - Fine roll threshold: %.2f deg", fine_roll_threshold_deg_);
  ROS_INFO("  - Fine pitch threshold: %.2f deg", fine_pitch_threshold_deg_);
  ROS_INFO("  - Fine slope range: [%.2f, %.2f]", fine_slope_min_, fine_slope_max_);
  ROS_INFO("  - Wheeled model path: %s", wheeled_model_path_.c_str());
  ROS_INFO("  - Tracked model path: %s", tracked_model_path_.c_str());

  ROS_INFO("  - Viewpoint grid:");
  ROS_INFO("    * number (cells): x=%d, y=%d, z=%d",
           viewpoint_number_x_, viewpoint_number_y_, viewpoint_number_z_);
  ROS_INFO("    * resolution (m): x=%.4f, y=%.4f, z=%.4f",
           viewpoint_resolution_x_, viewpoint_resolution_y_, viewpoint_resolution_z_);
  ROS_INFO("    * size (m): x=%.3f, y=%.3f, z=%.3f",
           viewpoint_grid_size_x_, viewpoint_grid_size_y_, viewpoint_grid_size_z_);
  ROS_INFO("    * total cells: %d",
           viewpoint_number_x_ * viewpoint_number_y_ * viewpoint_number_z_);
}

// ==================== Vehicle Model Loading ====================
void Mapping::loadRobotModels()
{
  wheeled_model_loaded_ = loadModelFromCsv(wheeled_model_path_, wheeled_model_);
  tracked_model_loaded_ = loadModelFromCsv(tracked_model_path_, tracked_model_);

  if (!wheeled_model_loaded_ && !tracked_model_loaded_)
  {
    ROS_FATAL("Failed to load both wheeled and tracked vehicle models (%s, %s)",
              wheeled_model_path_.c_str(), tracked_model_path_.c_str());
    throw std::runtime_error("Missing vehicle models");
  }

  if (!wheeled_model_loaded_)
  {
    ROS_WARN("Wheeled vehicle model unavailable; traversability_fine_wheeled layer will be left unchanged");
  }
  if (!tracked_model_loaded_)
  {
    ROS_WARN("Tracked vehicle model unavailable; traversability_fine_tracked layer will be left unchanged");
  }
}

bool Mapping::loadModelFromCsv(const std::string &path, HeightGrid &model_storage)
{
  ROS_INFO("Loading robot model from: %s", path.c_str());

  std::ifstream file(path);
  if (!file.is_open())
  {
    ROS_ERROR("Failed to open robot model file: %s", path.c_str());
    return false;
  }

  std::vector<std::vector<double>> data;
  std::string line;

  while (std::getline(file, line))
  {
    std::stringstream ss(line);
    std::string value;
    std::vector<double> row;

    while (std::getline(ss, value, ','))
    {
      row.push_back(std::stod(value));
    }
    data.push_back(row);
  }

  file.close();

  model_storage.X_.resize(robot_rows_, robot_cols_);
  model_storage.Y_.resize(robot_rows_, robot_cols_);
  model_storage.Z_.resize(robot_rows_, robot_cols_);
  model_storage.Gap_.resize(robot_rows_, robot_cols_);

  for (int i = 0; i < robot_rows_ && i < static_cast<int>(data.size()); ++i)
  {
    for (int j = 0; j < robot_cols_ && j < static_cast<int>(data[i].size()); ++j)
    {
      double resolution = robot_model_resolution_;
      model_storage.X_(i, j) = (i * resolution + resolution / 2.0) - (robot_rows_ * resolution / 2.0);
      model_storage.Y_(i, j) = (j * resolution + resolution / 2.0) - (robot_cols_ * resolution / 2.0);
      model_storage.Z_(i, j) = data[i][j] * resolution / 0.2;
      model_storage.Gap_(i, j) = 0.0;
    }
  }

  ROS_INFO("Robot model loaded: %d x %d cells, resolution %.2f",
           robot_rows_, robot_cols_, robot_model_resolution_);
  return true;
}

// void Mapping::originCallback(const geometry_msgs::PointStampedConstPtr &msg)
// {
//   std::lock_guard<std::mutex> lock(origin_mutex_);

//   latest_origin_.x() = msg->point.x;
//   latest_origin_.y() = msg->point.y;
//   latest_origin_.z() = msg->point.z;
//   latest_origin_stamp_ = msg->header.stamp;

//   has_origin_.store(true, std::memory_order_release);

//   if (debug_mode_)
//   {
//     ROS_DEBUG("Received origin: [%.3f %.3f %.3f] stamp=%.3f",
//               latest_origin_.x(), latest_origin_.y(), latest_origin_.z(),
//               latest_origin_stamp_.toSec());
//   }
// }

// void Mapping::viewpointVisCloudCallback(const sensor_msgs::PointCloud2ConstPtr &msg)
// {
//   // 转成 PCL（按你的需求：PointXYZI）
//   pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
//   try
//   {
//     pcl::fromROSMsg(*msg, *cloud);
//   }
//   catch (const std::exception &e)
//   {
//     ROS_WARN_STREAM("viewpointVisCloudCallback: fromROSMsg failed: " << e.what());
//     return;
//   }

//   // 防御：去掉 NaN/Inf（可选但推荐）
//   std::vector<int> idx;
//   pcl::removeNaNFromPointCloud(*cloud, *cloud, idx);

//   // 缓存最新一帧（线程安全）
//   {
//     std::lock_guard<std::mutex> lk(viewpoint_vis_mutex_);
//     *latest_viewpoint_vis_cloud_ = *cloud; // 深拷贝到缓存
//     latest_viewpoint_vis_stamp_ = msg->header.stamp;
//     latest_viewpoint_vis_frame_id_ = msg->header.frame_id;
//     has_viewpoint_vis_.store(true, std::memory_order_release);
//   }

//   ROS_INFO("Received synchronized point cloud.");
// }

void Mapping::synchronizedCloudOdomCallback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg,
                                            const nav_msgs::OdometryConstPtr &odom_msg)
{
  processSynchronizedMessages(cloud_msg, odom_msg);
}

void Mapping::processSynchronizedMessages(const sensor_msgs::PointCloud2ConstPtr &cloud_msg,
                                          const nav_msgs::OdometryConstPtr &odom_msg)
{
  // ===========================
  // [A] 日志：确认同步数据到达
  // ===========================
  // 注意：cloud_msg->width * cloud_msg->height 是 PointCloud2 的点数估计（organized cloud 时更典型）。
  // 对于非 organized cloud，height 常为 1，width = 点数。
  ROS_INFO("Received synchronized point cloud (%u points) at time: %u.%09u",
           static_cast<unsigned int>(cloud_msg->width * cloud_msg->height),
           static_cast<unsigned int>(cloud_msg->header.stamp.sec),
           static_cast<unsigned int>(cloud_msg->header.stamp.nsec));

  // debug_mode_ 开启时输出更详细的 frame 信息
  // frame_id 用于确认点云在哪个坐标系、里程计在哪个坐标系
  if (debug_mode_)
  {
    ROS_DEBUG("Synchronized messages - Cloud frame: %s, Odom frame: %s",
              cloud_msg->header.frame_id.c_str(),
              odom_msg->header.frame_id.c_str());
  }

  // ==========================================
  // [B] 从里程计提取车辆位姿，并更新共享缓存
  // ==========================================
  // position / orientation 是该回调对应的位姿快照（局部变量）
  // 后面会被打包进 task，保证点云处理使用“当时的位姿”，而不是未来被覆盖的新位姿
  Eigen::Vector3d position;
  Eigen::Quaterniond orientation;

  {
    // vehicle_pose_mutex_：保护共享成员变量
    // vehicle_position_ / vehicle_orientation_ / vehicle_pose_initialized_
    // 避免处理线程或其它回调同时读取/写入造成数据竞争
    std::lock_guard<std::mutex> lock(vehicle_pose_mutex_);

    // ----- 提取 position -----
    position.x() = odom_msg->pose.pose.position.x;
    position.y() = odom_msg->pose.pose.position.y;
    position.z() = odom_msg->pose.pose.position.z;

    // ----- 提取 orientation (四元数) -----
    // 注意：Eigen::Quaterniond 内部顺序是 (w, x, y, z)，这里赋值是正确的
    orientation.w() = odom_msg->pose.pose.orientation.w;
    orientation.x() = odom_msg->pose.pose.orientation.x;
    orientation.y() = odom_msg->pose.pose.orientation.y;
    orientation.z() = odom_msg->pose.pose.orientation.z;

    // 将本次回调的位姿写入成员变量（供其它模块读取）
    vehicle_position_ = position;
    vehicle_orientation_ = orientation;

    // 标记位姿已经初始化（通常其它模块会用这个 flag 判断能否开始建图/融合）
    vehicle_pose_initialized_ = true;
  }

  // debug：打印本次更新的位姿快照
  if (debug_mode_)
  {
    ROS_DEBUG("Updated vehicle pose: pos=(%.2f, %.2f, %.2f)",
              position.x(), position.y(), position.z());
  }

  // =====================================
  // [C] 点云格式转换：ROS -> PCL
  // =====================================
  // 这里使用 pcl::PointXYZ，意味着只保留 xyz 坐标
  // 如果原始点云带 intensity/ring/time 等字段，这里会被丢弃
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::fromROSMsg(*cloud_msg, *cloud);

  // =====================================
  // [D] 构造异步任务 ProcessingTask
  // =====================================
  // 将“点云 + 位姿快照 + 时间戳”打包成一个任务
  // 供后台 processing thread 消费处理
  ProcessingTask task;
  task.cloud = cloud;
  task.position = position;
  task.orientation = orientation;
  task.timestamp = cloud_msg->header.stamp;

  // ==========================================
  // [E] 入队：把任务放进 processing_queue_
  // ==========================================
  {
    // processing_queue_mutex_：保护任务队列（生产者=ROS回调线程，消费者=处理线程）
    std::lock_guard<std::mutex> lock(processing_queue_mutex_);

    // 可选策略：skip_old_messages_ 为 true 时，若处理线程跟不上，就丢掉旧任务
    // 目的：保证系统“实时性”，不让延迟无限增大
    if (skip_old_messages_ && !processing_queue_.empty())
    {
      size_t queue_size = processing_queue_.size();

      // 队列积压超过 2，认为处理端跟不上，清空旧任务
      // 注意：你这里是“全清空”，相当于只保留最新到来的这一帧
      if (queue_size > 2)
      {
        ROS_WARN("Processing queue has %zu tasks backed up. Clearing old tasks (skip_old_messages=true)",
                 queue_size);

        // 用空队列交换，达到快速清空队列的效果（O(1) swap）
        std::queue<ProcessingTask> empty_queue;
        processing_queue_.swap(empty_queue);
      }
    }

    // 把本次任务压入队尾
    processing_queue_.push(task);
  }

  // ==========================================
  // [F] 通知处理线程：队列里有新任务了
  // ==========================================
  // processing_cv_ 是 condition_variable
  // 处理线程通常会 wait 在 cv 上，直到被 notify 唤醒
  processing_cv_.notify_one();

  if (debug_mode_)
  {
    ROS_DEBUG("Queued processing task for cloud with %lu points", cloud->size());
  }
}

// ==================== Thread Implementation ==================== //
void Mapping::processingThreadFunc()
{
  ROS_INFO("Processing thread started");

  // 处理线程主循环：不断从队列取任务并处理，直到 should_exit_ 为 true
  // should_exit_ 通常由析构函数/stop() 设置，用于让线程安全退出
  while (!should_exit_)
  {
    ProcessingTask task;   // 本轮要处理的任务
    bool has_task = false; // 标记本轮是否成功取到了任务（避免空处理）

    // ==========================================
    // [A] 等待任务：条件变量 + 队列锁
    // ==========================================
    {
      // unique_lock 用于配合 condition_variable::wait
      // wait 内部会：1) 原子释放锁并睡眠 2) 被唤醒后重新加锁
      std::unique_lock<std::mutex> lock(processing_queue_mutex_);

      // wait(lock, predicate) 的语义：
      // - 如果 predicate 一开始就为 true，则不会睡眠，直接返回
      // - 否则释放锁并睡眠，直到被 notify 唤醒（或假唤醒），再重新加锁并检查 predicate
      //
      // 这里 predicate 是：队列非空 或者 should_exit_ 为 true
      // 目的：有任务就处理；收到退出信号也要能醒来退出
      processing_cv_.wait(lock, [this]()
                          { return !processing_queue_.empty() || should_exit_; });

      // 被唤醒后，仍然在锁保护范围内，安全检查退出条件
      // 如果已经要退出，并且队列也空了，那就安全退出线程
      if (should_exit_ && processing_queue_.empty())
      {
        ROS_INFO("Processing thread exiting (should_exit=true, queue empty)");
        break;
      }

      // 如果队列非空，从队首取一个任务（FIFO），然后 pop 掉
      if (!processing_queue_.empty())
      {
        task = processing_queue_.front();
        processing_queue_.pop();
        has_task = true;

        // active_task_count_：表示“正在处理中的任务数”
        // 这里取出任务后就+1，表示线程现在手上有活了（但还没处理完）
        active_task_count_++;

        // 仅用于日志：还有多少任务积压
        size_t remaining = processing_queue_.size();
        if (remaining > 0)
        {
          ROS_INFO("Processing cloud from queue (remaining tasks: %zu)", remaining);
        }
      }

      // 离开这个作用域时 lock 自动释放
      // 关键设计：只在锁内做“取任务”，把重计算放到锁外
    }

    // ==========================================
    // [B] 处理任务：在锁外做重活，避免阻塞生产者/其它线程
    // ==========================================
    if (has_task)
    {
      if (debug_mode_)
      {
        ROS_DEBUG("Processing dequeued task with %lu points", task.cloud->size());
      }

      // ------------------------------------------
      // [B1] 将 task 自带的位姿写入 vehicle_*（共享状态）
      // ------------------------------------------
      // 注意：这里的写入其实“功能上未必必须”
      // 因为 task 本身已经包含 position/orientation
      // 但如果 processPointCloud/publishGridMapOnce 内部依赖 vehicle_* 作为当前位姿，
      // 那就需要在这里更新，保证处理使用的是“与该点云对应的位姿快照”

      Eigen::Vector3d vehicle_position;
      Eigen::Quaterniond vehicle_orientation;
      {
        std::lock_guard<std::mutex> lock(vehicle_pose_mutex_);
        vehicle_position_ = task.position;
        vehicle_orientation_ = task.orientation;
        vehicle_position = task.position;
        vehicle_orientation = task.orientation;
      }

      // ------------------------------------------
      // [B2] 真正处理点云（重计算：滤波/配准/栅格更新等）
      // ------------------------------------------
      processPointCloud(task.cloud, vehicle_position);

      // ------------------------------------------
      // [B3] 处理完立刻发布一次栅格地图
      // ------------------------------------------
      // 注意：如果 publishGridMapOnce 里也用到共享资源，要确保内部已正确加锁
      publishGridMapOnce();

      // ==========================================
      // [C] 任务结束：更新 active_task_count_ 并通知等待 idle 的线程
      // ==========================================
      {
        // 这里重新加 processing_queue_mutex_ 的锁，保护 active_task_count_ 与队列状态
        std::lock_guard<std::mutex> queue_lock(processing_queue_mutex_);

        // active_task_count_-- 表示“一个任务处理完成”
        if (active_task_count_ > 0)
        {
          active_task_count_--;
        }

        // 如果队列空 + 没有正在处理的任务，说明系统处于 idle
        // 这时 notify_all() 是为了唤醒 waitUntilIdle() 里等待的线程
        if (processing_queue_.empty() && active_task_count_ == 0)
        {
          processing_cv_.notify_all();
        }
      }
    }
  }

  ROS_INFO("Processing thread finished");
}

void Mapping::waitUntilIdle()
{
  // 外部线程（比如主线程/析构/某个同步逻辑）调用此函数：
  // 阻塞等待直到“队列空 && 没有正在处理的任务”
  std::unique_lock<std::mutex> queue_lock(processing_queue_mutex_);

  // 等待条件：processing_queue_ 为空，并且 active_task_count_==0
  // - 队列空：没有排队任务
  // - active_task_count_==0：处理线程也没有正在处理中的任务（否则队列空但仍在算）
  processing_cv_.wait(queue_lock, [this]()
                      { return processing_queue_.empty() && active_task_count_ == 0; });
}

void Mapping::processPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, Eigen::Vector3d current_pos)
{
  // =========================
  // [0] 输入检查：空点云直接返回
  // =========================
  if (cloud->empty())
  {
    ROS_WARN("Received empty point cloud");
    return;
  }

  // debug 模式下输出点数
  if (debug_mode_)
  {
    ROS_DEBUG("Processing point cloud with %lu points", cloud->size());
  }

  // =====================================================
  // [2] 点云预过滤：Z 过滤 + 最大观测范围过滤
  // =====================================================
  // filtered_cloud 用于存储过滤后的点云
  // reserve(cloud->size())：预分配容量，减少 push_back 时反复扩容，提高效率
  auto filtered_cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  filtered_cloud->reserve(cloud->size());

  int out_of_range = 0; // 统计超出 max_range_ 的点数（仅用于日志/调试）

  for (const auto &point : cloud->points)
  {
    // -----------------------
    // [2.1] Z 范围过滤
    // -----------------------
    // 只保留 z 在 [min_z_, max_z_] 的点
    // 常用于剔除异常高点/地面以下噪声/天花板等
    double dz = point.z - current_pos.z(); // 或 odom 里的 z
    if (dz < min_z_ || dz > max_z_)
      continue;

    // -----------------------
    // [2.2] 最大观测距离过滤（平面距离）
    // -----------------------
    // 这里计算的是点到“当前车辆位置”的 XY 平面距离
    // 距离 <= max_range_ 才保留
    // 此处只用 XY，不考虑 Z（很多地形栅格化只关心平面距离）
    double dx = point.x - current_pos.x();
    double dy = point.y - current_pos.y();

    // 欧氏距离（平面）
    double distance = std::sqrt(dx * dx + dy * dy);

    // 距离在可接受范围内则保留
    if (distance <= max_range_)
    {
      filtered_cloud->push_back(point);
    }
    else
    {
      out_of_range++;
    }
  }

  // =====================================================
  // [3] 过滤结果检查：若全被过滤掉则返回
  // =====================================================
  if (filtered_cloud->empty())
  {
    ROS_WARN("No points in valid Z range [%.2f, %.2f] and observation range [%.2f m]",
             min_z_, max_z_, max_range_);
    return;
  }

  // =====================================================
  // [4] 输出信息：过滤效果 + 进入建图管线
  // =====================================================
  // 这要求：输入 cloud 的 frame 已经是 map/world，且点坐标也是全局坐标
  ROS_INFO("Processing %lu points (filtered from %lu original points)",
           filtered_cloud->size(), cloud->size());
  ROS_INFO("Starting mapping pipeline...");

  // =====================================================
  // [5] Mapping Pipeline：逐步执行并计时
  // =====================================================
  // 你用 ros::Time::now().toSec() 计时，能用但精度一般
  // 如果你想更稳定的 profiling，推荐用 ros::WallTime 或 std::chrono
  double time1 = ros::Time::now().toSec();
  double time2 = 0.0;

  // -----------------------
  // Step 1: Height mapping
  // -----------------------
  // 将点云栅格化，生成高度/统计量（min/max/mean/var 等）
  time1 = ros::Time::now().toSec();
  height_mapping(filtered_cloud);
  time2 = ros::Time::now().toSec();
  ROS_INFO("Height mapping: %.4f seconds", time2 - time1);

  // -----------------------
  // Step 2: BGK mapping
  // -----------------------
  // 使用 BGK（Bayesian Generalized Kernel 或类似稀疏核推断）
  // 对空栅格进行插值/推断，使地图更稠密平滑
  time1 = ros::Time::now().toSec();
  bgk_mapping();
  time2 = ros::Time::now().toSec();
  ROS_INFO("BGK mapping: %.4f seconds", time2 - time1);

  // -----------------------
  // Step 3: Geometric mapping
  // -----------------------
  // 基于高度图计算几何特征，如：
  // - slope（坡度）
  // - roughness（粗糙度）
  time1 = ros::Time::now().toSec();
  geometric_mapping();
  time2 = ros::Time::now().toSec();
  ROS_INFO("Geometric mapping: %.4f seconds", time2 - time1);

  // -----------------------
  // Step 4: Step mapping
  // -----------------------
  // 计算台阶/突变特征（例如邻域最大高度差）
  time1 = ros::Time::now().toSec();
  step_mapping();
  time2 = ros::Time::now().toSec();
  ROS_INFO("Step mapping: %.4f seconds", time2 - time1);

  // -----------------------
  // Step 5: Traversability mapping
  // -----------------------
  // 将各类地形特征融合为初始可通行性代价/概率（0~1 或 cost）
  time1 = ros::Time::now().toSec();
  traversability_mapping();
  time2 = ros::Time::now().toSec();
  ROS_INFO("Traversability mapping: %.4f seconds", time2 - time1);

  // -----------------------
  // Step 6: Fine-grained Traversability
  // -----------------------
  // 可选：结合车辆模型（轮距、离地间隙、最大爬坡等）
  // 做更细粒度的可通行性评估
  if (enable_fine_traversability_)
  {
    time1 = ros::Time::now().toSec();
    finegrained_traversability_mapping();
    time2 = ros::Time::now().toSec();
    ROS_INFO("Fine-grained traversability mapping: %.4f seconds", time2 - time1);
  }
  else
  {
    ROS_DEBUG("Fine-grained traversability mapping disabled");
  }

  ROS_INFO("Mapping pipeline completed");
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Mapping::getPointCloud() const
{
  return current_cloud_;
}

void Mapping::publishGridMapOnce()
{
  publishGridMap();
}

bool Mapping::exportGridMap(grid_map_msgs::GridMap &message)
{
  return buildGridMapMessage(message);
}

// ==================== Grid Map Initialization ====================
void Mapping::initGridMap()
{
  ROS_INFO("Initializing grid map layers...");

  grid_map::Length map_length(map_length_x_, map_length_y_);
  height_map_.setFrameId(frame_id_);
  height_map_.setGeometry(map_length, map_resolution_, grid_map::Position(0.0, 0.0));

  // Add layers
  height_map_.add("elevation");                   // Height of observed points
  height_map_.add("elevation_BGK");               // Interpolated height using BGK
  height_map_.add("variance");                    // Height variance
  height_map_.add("min_elevation");               // Minimum elevation in cell
  height_map_.add("max_elevation");               // Maximum elevation in cell
  height_map_.add("n_points");                    // Number of points in cell
  height_map_.add("normal_x");                    // Surface normal X component
  height_map_.add("normal_y");                    // Surface normal Y component
  height_map_.add("normal_z");                    // Surface normal Z component
  height_map_.add("slope");                       // Slope feature
  height_map_.add("roughness");                   // Roughness feature
  height_map_.add("step");                        // Step feature
  height_map_.add("traversability");              // Traversability cost
  height_map_.add("traversability_fine_wheeled"); // Fine-grained traversability for wheeled car
  height_map_.add("traversability_fine_tracked"); // Fine-grained traversability for tracked car
  height_map_.add("critical");                    // Flag: cells evaluated by fine-grained mapping
  height_map_.add("interpolated");                // Flag: whether this cell has been interpolated by BGK
  height_map_.add("incremental_geom_computed");   // Flag: whether geometric mapping has been computed incrementally
  height_map_.add("incremental_step_computed");   // Flag: whether step mapping has been computed incrementally
  height_map_.add("incremental_trav_computed");   // Flag: whether traversability mapping has been computed incrementally

  // Initialize n_points layer to 0
  height_map_["n_points"].setConstant(0);

  // Initialize flag layers to 0 (not computed/interpolated)
  height_map_["interpolated"].setConstant(0);
  height_map_["incremental_geom_computed"].setConstant(0);
  height_map_["incremental_step_computed"].setConstant(0);
  height_map_["incremental_trav_computed"].setConstant(0);
  height_map_["critical"].setConstant(0);

  ROS_INFO("Global grid map initialized:");
  ROS_INFO("  - Size: [%.2f x %.2f] meters (%.0f x %.0f cells)", map_length_x_, map_length_y_,
           map_length_x_ / map_resolution_, map_length_y_ / map_resolution_);
  ROS_INFO("  - Resolution: %.4f m/cell", map_resolution_);
  ROS_INFO("  - Layers: %lu", height_map_.getLayers().size());
}

// ==================== Height Mapping ====================
void Mapping::height_mapping(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
  // 输出日志：本帧点云点数
  ROS_INFO("Starting height mapping from %lu points...", cloud->size());

  // gridmap_publish_mutex_：保护 height_map_ 的写操作，避免与发布线程/其它更新线程并发读写冲突
  std::lock_guard<std::mutex> lock(gridmap_publish_mutex_);

  // 从 grid_map 中取出不同 layer 的引用（Matrix）
  // 这些 layer 是二维栅格，每个 cell 存一个值
  auto &elevation = height_map_["elevation"];         // 平均高度（或当前估计高度）
  auto &variance = height_map_["variance"];           // 高度方差（用于不确定性）
  auto &min_elevation = height_map_["min_elevation"]; // cell 内最小高度
  auto &max_elevation = height_map_["max_elevation"]; // cell 内最大高度
  auto &n_points = height_map_["n_points"];           // cell 内累计观测点数（用于统计更新）

  int cell_count = 0;    // 统计：成功落在地图范围内并更新过的点数（这里命名是 cell_count，但实际上是“点更新次数”）
  int out_of_bounds = 0; // 统计：落在地图外的点数

  // elevation_BGK：用于 BGK 插值/预测后的高度层
  // 你后面会对“新观测到的格子”用真实观测高度去覆盖 BGK 的值，避免 BGK 继续占用这些格子
  auto &elevation_bgk = height_map_["elevation_BGK"];

  // newly_observed_cells：记录本帧中“从未观测（npts==0）变为有观测（npts>0）”的格子索引
  // 用 set 去重，确保每个格子只记录一次
  std::set<std::pair<int, int>> newly_observed_cells;

  // ==========================
  // [A] 遍历点云：逐点更新对应栅格的统计量
  // ==========================
  for (const auto &point : cloud->points)
  {
    // 将点的 XY 转成 grid_map 的 Position（二维位置）
    grid_map::Position pos(point.x, point.y);
    grid_map::Index index; // 栅格索引 (i, j)

    // getIndex：把连续坐标 pos 映射到栅格 index
    // 若 pos 不在地图范围内，返回 false
    if (!height_map_.getIndex(pos, index))
    {
      out_of_bounds++;
      continue;
    }

    // 统计“成功映射到地图内部的点更新次数”
    cell_count++;

    // 取出该 cell 的各层数据引用（引用方式能避免重复索引开销）
    auto &height = elevation(index(0), index(1));    // 当前高度估计（均值）
    auto &var = variance(index(0), index(1));        // 当前方差
    auto &npts = n_points(index(0), index(1));       // 当前累计点数
    auto &min_h = min_elevation(index(0), index(1)); // 当前最小高度
    auto &max_h = max_elevation(index(0), index(1)); // 当前最大高度

    // 判断这个 cell 在本次更新之前是否未被观测过
    // npts==0 => 未观测（高度可能来自 BGK 或默认值）
    bool was_unobserved = (npts == 0);

    if (npts == 0)
    {
      // ------------------------------------------
      // [A1] cell 第一次被观测：初始化统计量
      // ------------------------------------------
      // 第一个点直接作为均值
      height = point.z;

      // 方差初始化为 0（也可以考虑给一个小先验，如 eps）
      var = 0.0f;

      // min/max 都初始化为该点高度
      min_h = point.z;
      max_h = point.z;

      // 点数置为 1
      npts = 1;
    }
    else
    {
      // ------------------------------------------
      // [A2] cell 已有观测：增量更新统计量
      // ------------------------------------------
      // 点数 +1
      npts += 1;

      // updateHeightStats：通常是在线均值/方差更新（Welford 或类似）
      // 输入：当前均值 height、方差 var、样本数 npts、新样本 point.z
      // 输出：更新后的均值/方差
      updateHeightStats(height, var, npts, point.z);

      // 更新 min/max
      min_h = std::min(min_h, point.z);
      max_h = std::max(max_h, point.z);
    }

    // ------------------------------------------
    // [A3] 记录“从未观测变为已观测”的格子
    // ------------------------------------------
    // 如果之前 npts==0，说明该 cell 的 BGK 高度现在应该被真实观测高度覆盖
    // 但这里不立刻覆盖 BGK，而是先记录下来，循环结束后再统一更新（减少重复写）
    if (was_unobserved)
    {
      newly_observed_cells.insert(std::make_pair(index(0), index(1)));
    }
  }

  // ==========================
  // [B] 循环结束后：对新观测格子覆盖 BGK 高度层
  // ==========================
  // 这样做的含义：
  // - elevation 存的是“真实观测统计得到的高度均值”
  // - elevation_BGK 是“BGK 插值得到的高度”
  // 对于本帧第一次被真实观测到的格子，用真实观测均值覆盖 BGK 值
  int bgk_updated_cells = 0;

  for (const auto &cell_idx : newly_observed_cells)
  {
    // cell_idx 是 pair<int,int> 对应 (i,j)
    elevation_bgk(cell_idx.first, cell_idx.second) =
        elevation(cell_idx.first, cell_idx.second);

    bgk_updated_cells++;
  }

  // 输出统计信息：映射到地图内的点更新次数、以及 BGK 被覆盖的格子数
  // 注意：cell_count 实际上是“点数”，不是 unique cell 数
  ROS_INFO("Height mapping: processed %d cells, updated %d BGK cells",
           cell_count, bgk_updated_cells);
}

void Mapping::updateHeightStats(float &height, float &variance, float n, float new_height)
{
  const float delta = (new_height - height);
  const float delta_n = delta / n;

  // Update mean
  height += delta_n;

  // Update variance
  variance += delta * (new_height - height);

  if (n > 1)
  {
    variance = variance / (n - 1);
  }
}

// ==================== BGK Mapping (Interpolation) ====================
void Mapping::bgk_mapping()
{
  ROS_INFO("Starting BGK interpolation...");

  // 保护 height_map_ 的读写，避免和发布线程/其它更新线程并发冲突
  // 注意：你这里锁住整个插值过程，如果 max_range_ 覆盖很多格子，会长期持锁，影响发布/其它模块
  std::lock_guard<std::mutex> lock(gridmap_publish_mutex_);

  // 取出 grid_map 的 layer 引用（每个 layer 是二维矩阵）
  auto &elevation_bgk = height_map_["elevation_BGK"]; // BGK 插值结果高度（用于填补空格）
  auto &elevation = height_map_["elevation"];         // 观测得到的高度均值（训练数据来自这里）
  auto &n_points = height_map_["n_points"];           // 每格观测点数（n>0 表示该格被观测到）
  auto &interpolated = height_map_["interpolated"];   // 标记位：该格是否已经被插值过（避免重复插值）

  // =====================================================
  // [A] 获取车辆当前位置：用于“限制插值范围”
  // =====================================================
  // 只在车周围 max_range_ 半径内做 BGK 插值，避免全图插值导致计算量爆炸
  Eigen::Vector3d current_pos;
  {
    std::lock_guard<std::mutex> lock(vehicle_pose_mutex_);
    current_pos = vehicle_position_;
  }

  int interpolated_cells = 0;

  // =====================================================
  // [B] 遍历“车周围 max_range_ 圆形区域内”的所有格子
  // =====================================================
  // grid_map::CircleIterator 会遍历地图中位于指定圆形范围内的 grid 索引
  // 这样 BGK 插值只在局部区域发生（局部更新）
  for (grid_map::CircleIterator it(
           height_map_,
           grid_map::Position(current_pos.x(), current_pos.y()),
           max_range_);
       !it.isPastEnd();
       ++it)
  {
    // 当前待插值格子的 (i, j) 索引
    grid_map::Index index = *it;

    // -------------------------------------------------
    // [B1] 跳过“已观测格子”
    // -------------------------------------------------
    // n_points != 0 => 该格已经有真实点云观测，不需要用 BGK 插值
    // 这里 n_points layer 是 float，所以用 != 0.0f
    if (n_points(index(0), index(1)) != 0.0f)
    {
      continue;
    }

    // -------------------------------------------------
    // [B2] 跳过“已经插值过”的格子
    // -------------------------------------------------
    // interpolated != 0 => 之前已经算过 BGK，高度已经填过，不重复计算
    if (interpolated(index(0), index(1)) != 0.0f)
    {
      continue;
    }

    // -------------------------------------------------
    // [B3] 将格子 index 转换为连续坐标 test_pos (x, y)
    // -------------------------------------------------
    // 有些情况下 index->position 转换可能失败（理论上不常见，但你这里做了保护）
    grid_map::Position test_pos;
    if (!height_map_.getPosition(index, test_pos))
    {
      continue;
    }

    // =====================================================
    // [C] 收集训练数据：从 test_pos 周围 bgk_kernel_size_ 半径内找邻居格子
    // =====================================================
    // x_train_vec：训练样本的 (x, y) 坐标，按 [x0,y0,x1,y1,...] 方式存
    // y_train_vec：训练样本的高度 z 值，对应 elevation layer
    std::vector<float> x_train_vec;
    std::vector<float> y_train_vec;

    // test_pos 作为中心，在 bgk_kernel_size_ 半径内遍历邻居格子
    grid_map::Position center = test_pos;
    for (grid_map::CircleIterator cit(height_map_, center, bgk_kernel_size_);
         !cit.isPastEnd();
         ++cit)
    {
      grid_map::Index cidx = *cit;

      // 跳过未观测格子：训练样本必须来自真实观测格子 (n_points>0)
      if (n_points(cidx(0), cidx(1)) == 0.0f)
      {
        continue;
      }

      // 把邻居格子的连续坐标作为训练输入，把该格的 elevation 作为训练输出
      grid_map::Position train_pos;
      if (height_map_.getPosition(cidx, train_pos))
      {
        x_train_vec.push_back(train_pos.x());
        x_train_vec.push_back(train_pos.y());
        y_train_vec.push_back(elevation(cidx(0), cidx(1)));
      }
    }

    // -------------------------------------------------
    // [C1] 训练样本数不足则跳过插值
    // -------------------------------------------------
    // 你这里判断 x_train_vec.size() < 3
    // 注意：x_train_vec 每个样本占 2 个 float（x,y），所以严格说应该判断样本数 < 3：
    //   (x_train_vec.size() / 2) < 3
    // 目前写成 <3 意味着只有当 x_train_vec.size() 是 0,1,2 才跳过，可能会误放行非常少的样本。
    // （不过后面矩阵构造用 x_train_vec.size()/2，所以极端情况下仍可能不稳）
    if (x_train_vec.size() < 3)
    {
      continue;
    }

    // =====================================================
    // [D] 将训练数据从 vector 映射成 Eigen 矩阵（不拷贝，Map 视图）
    // =====================================================
    // x_train: (N, 2) 每行是一个训练点的 (x, y)
    // y_train: (N, 1) 每行是训练点的高度 z
    Eigen::MatrixXf x_train = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(
        x_train_vec.data(), x_train_vec.size() / 2, 2);

    Eigen::MatrixXf y_train = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(
        y_train_vec.data(), y_train_vec.size(), 1);

    // test point：形状 (1,2)
    Eigen::MatrixXf x_test(1, 2);
    x_test << test_pos.x(), test_pos.y();

    // =====================================================
    // [E] 计算稀疏协方差核 K（测试点 vs 训练点）
    // =====================================================
    // covSparse(x_test, x_train, K)：
    // - 输入：x_test (1,2)，x_train (N,2)
    // - 输出：K (1,N)，表示测试点到每个训练点的核权重
    // 实际上这里做的是一种“核加权平均”的插值
    Eigen::MatrixXf K;
    covSparse(x_test, x_train, K);

    // 用权重对 y_train 做加权求和：y_pred = sum_i K_i * y_i
    Eigen::MatrixXf y_pred = (K * y_train).array();

    // 权重和：k_sum = sum_i K_i
    Eigen::MatrixXf k_sum = K.rowwise().sum().array();

    // =====================================================
    // [F] 写回插值结果，并做标记
    // =====================================================
    // 若权重和为正，且预测值不是 NaN，则写入：
    //   elevation_BGK = y_pred / sum(K)
    // 并将 interpolated 标记为 1（表示该格已插值过）
    if (k_sum(0, 0) > 0 && !std::isnan(y_pred(0, 0)))
    {
      elevation_bgk(index(0), index(1)) = y_pred(0, 0) / k_sum(0, 0);
      interpolated(index(0), index(1)) = 1.0f;
      interpolated_cells++;
    }
  }

  ROS_INFO("BGK interpolation completed - interpolated %d cells", interpolated_cells);
}

void Mapping::dist(const Eigen::MatrixXf &xStar, const Eigen::MatrixXf &xTrain, Eigen::MatrixXf &d) const
{
  d = Eigen::MatrixXf::Zero(xStar.rows(), xTrain.rows());
  for (int i = 0; i < xStar.rows(); ++i)
  {
    d.row(i) = (xTrain.rowwise() - xStar.row(i)).rowwise().norm();
  }
}

void Mapping::covSparse(const Eigen::MatrixXf &xStar, const Eigen::MatrixXf &xTrain, Eigen::MatrixXf &Kxz) const
{
  dist(xStar / (bgk_kernel_size_ + 0.1), xTrain / (bgk_kernel_size_ + 0.1), Kxz);
  Kxz = (((2.0f + (Kxz * 2.0f * 3.1415926f).array().cos()) * (1.0f - Kxz.array()) / 3.0f) +
         (Kxz * 2.0f * 3.1415926f).array().sin() / (2.0f * 3.1415926f))
            .matrix() *
        1.0f;

  // Clean up negative values
  for (int i = 0; i < Kxz.rows(); ++i)
  {
    for (int j = 0; j < Kxz.cols(); ++j)
    {
      if (Kxz(i, j) < 0)
      {
        Kxz(i, j) = 0;
      }
    }
  }
}

// ==================== Geometric Mapping ====================
void Mapping::geometric_mapping()
{
  // =====================================================
  // [0] 加锁：保护 height_map_ 的读写（尤其是 layer 写入）
  // =====================================================
  // 这里锁住整个函数，意味着几何特征计算期间地图发布/其它写线程会被阻塞。
  // 如果 areaSingleNormalComputation 很耗时，锁粒度会比较大（后续可优化：锁外计算、锁内写回）。
  std::lock_guard<std::mutex> lock(gridmap_publish_mutex_);

  // incremental_geom_computed：用于增量模式的标记层
  // 约定：>0 表示该 cell 的几何特征（如法向/坡度/粗糙度）已经算过了，
  //       下次就不重复计算，从而节省时间
  auto &incremental_geom_computed = height_map_["incremental_geom_computed"];

  // =====================================================
  // [1] 获取车辆当前位置：用于限制处理范围
  // =====================================================
  // 只计算车辆周围 max_range_ 半径内的格子，避免全图遍历，减少计算量。
  Eigen::Vector3d current_pos;
  {
    // vehicle_position_ 是共享变量，读取时加锁拿到一个快照
    std::lock_guard<std::mutex> pose_lock(vehicle_pose_mutex_);
    current_pos = vehicle_position_;
  }

  // =====================================================
  // [2] 遍历车辆周围 max_range_ 圆形区域内的所有 grid cell
  // =====================================================
  // CircleIterator：遍历地图中位于指定圆形范围内的栅格索引。
  for (grid_map::CircleIterator it(
           height_map_,
           grid_map::Position(current_pos.x(), current_pos.y()), // 圆心：车辆当前位置（map 坐标）
           max_range_);                                          // 半径：max_range_
       !it.isPastEnd();
       ++it)
  {
    // 当前 cell 的索引 (i,j)
    grid_map::Index index = *it;

    // =================================================
    // [2.1] 增量模式：跳过已经计算过的 cell
    // =================================================
    // enable_incremental_geom_ 为 true 时，表示你不想每帧都重复算坡度/法向等几何特征，
    // 而是对每个 cell 只算一次（或按某种策略重算）。
    if (enable_incremental_geom_)
    {
      // incremental_geom_computed layer 用 float 存标记值 >0.1f 就认为“已经计算过”
      if (incremental_geom_computed(index(0), index(1)) > 0.1f)
      {
        continue; // 直接跳过，节省计算
      }
    }

    // =================================================
    // [2.2] 计算该 cell 的几何特征（核心计算）
    // =================================================
    // areaSingleNormalComputation(index)：
    // - 通常会在该 cell 的邻域里取高度数据（比如 elevation 或 elevation_BGK）
    // - 计算局部平面/法向量（normal）
    // - 进一步得到 slope（坡度）、roughness（粗糙度）等几何指标
    // 返回值 computed 表示是否计算成功：
    // - true：该 cell 有足够数据/邻域有效，成功算出几何特征
    // - false：数据不足、邻域空、或某些条件不满足，无法计算
    bool computed = areaSingleNormalComputation(index);

    // =================================================
    // [2.3] 若开启增量模式，且计算成功，则标记该 cell 已计算
    // =================================================
    // 下次进入 geometric_mapping() 时会跳过它，避免重复计算
    if (enable_incremental_geom_ && computed)
    {
      incremental_geom_computed(index(0), index(1)) = 1.0f;
    }
  }
}

bool Mapping::areaSingleNormalComputation(const grid_map::Index &index)
{
  // =====================================================
  // [0] 将当前 cell 的栅格索引 index 转换成连续坐标 center=(x,y)
  // =====================================================
  // grid_map 里每个 cell 都对应一个中心位置 (x,y)（在 map/world 坐标系下）
  grid_map::Position center;
  if (!height_map_.getPosition(index, center))
  {
    // 如果索引无法映射成位置，说明 index 非法或地图内部状态异常
    return false;
  }

  // =====================================================
  // [1] 检查法向估计半径是否足够大
  // =====================================================
  // normal_estimation_radius_：你要在多大邻域内取点来拟合局部平面
  // map_resolution_：栅格分辨率（每个格子的边长）
  //
  // 半径太小会导致邻域里只有很少甚至只有一个格子，无法拟合平面（法向不可靠）
  const double min_allowed_radius = 0.5 * map_resolution_;
  if (normal_estimation_radius_ < min_allowed_radius)
  {
    ROS_DEBUG("Normal estimation radius too small");
    return false;
  }

  // =====================================================
  // [2] 在 center 周围的圆形邻域内收集 3D 点 (x,y,z)
  // =====================================================
  // 这里的“点”不是原始点云，而是“每个邻域格子的中心点 + 该格子的高度 elevation_BGK”
  // 也就是把高度图看成一个“稀疏/规则采样”的 3D 点集。
  size_t n_pts = 0;                                      // 邻域有效点数
  Eigen::Vector3d sum = Eigen::Vector3d::Zero();         // Σ p_i，用于计算均值
  Eigen::Matrix3d sum_squared = Eigen::Matrix3d::Zero(); // Σ (p_i p_i^T)，用于快速算协方差

  // CircleIterator：遍历以 center 为圆心、normal_estimation_radius_ 为半径的所有栅格
  for (grid_map::CircleIterator cit(height_map_, center, normal_estimation_radius_);
       !cit.isPastEnd(); ++cit)
  {
    // 将邻域 cell 的 index 转成连续坐标 pos=(x,y)
    grid_map::Position pos;
    if (!height_map_.getPosition(*cit, pos))
    {
      continue;
    }

    // 取该邻域 cell 的高度（来自 elevation_BGK 层）
    // elevation_BGK：你前面用 BGK 插值填补了空格，所以这里能尽量拿到连续高度
    float z = height_map_.at("elevation_BGK", *cit);

    // 跳过无效高度（NaN）
    if (std::isnan(z))
    {
      continue;
    }

    // 构造一个 3D 点：该栅格中心 (x,y) + 高度 z
    Eigen::Vector3d point(pos.x(), pos.y(), z);

    // 累计统计量：用于后面构造协方差矩阵
    n_pts++;
    sum += point;

    // sum_squared += point * point^T
    // noalias()：告诉 Eigen 这里没有别名（避免额外临时矩阵），小幅提速
    sum_squared.noalias() += point * point.transpose();
  }

  // =====================================================
  // [3] 有效点数检查：点太少无法可靠估计法向
  // =====================================================
  Eigen::Vector3d normal = Eigen::Vector3d::UnitZ(); // 默认法向：Z 轴向上
  const int required_points = std::max(4, min_normal_points_);
  if (static_cast<int>(n_pts) < required_points)
  {
    // 邻域里有效点不够（高度缺失太多或邻域太小），放弃计算
    return false;
  }

  // =====================================================
  // [4] 计算邻域点云的均值和协方差矩阵
  // =====================================================
  // mean = (1/N) Σ p_i
  Eigen::Vector3d mean = sum / n_pts;

  // cov = (1/N) Σ(p_i p_i^T) - mean mean^T
  // 这是协方差矩阵的常用计算形式（数值稳定性还不错，且不用逐点减均值）
  Eigen::Matrix3d cov = sum_squared / n_pts - mean * mean.transpose();

  // =====================================================
  // [5] 对协方差矩阵做特征分解：用 PCA 求法向
  // =====================================================
  // 对称矩阵 cov 的特征分解：cov = V Λ V^T
  // SelfAdjointEigenSolver 更快更稳定（因为 cov 对称）
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);

  // 这里你用 solver.eigenvalues()(1) > 1e-8 做一个“退化”判断：
  // - 如果邻域点几乎共线/退化（协方差某些方向很小），法向估计可能不稳定
  // - 这句判断并不完美，但起到了基本的过滤作用
  if (solver.eigenvalues()(1) > 1e-8)
  {
    // PCA 原理：最小特征值对应的特征向量，是点云“最薄方向”
    // 对局部平面而言，最薄方向就是“平面法向”
    normal = solver.eigenvectors().col(0); // col(0) 对应最小特征值（特征值从小到大排序）

    // 统一法向朝上：如果法向与 +Z 夹角>90°，就翻转
    // 避免同一片地面法向有时朝上有时朝下，影响 slope 计算与可视化
    if (normal.dot(Eigen::Vector3d::UnitZ()) < 0.0)
    {
      normal = -normal;
    }
  }

  // =====================================================
  // [6] 由特征值/法向计算几何特征：roughness 与 slope
  // =====================================================

  // roughness：粗糙度/起伏程度（你这里是一个归一化指标）
  // eigenvalues()(0)：最小特征值，代表“沿法向方向的方差”（点到平面的离散程度）
  // eigenvalues().sum()：总方差（所有方向）
  // 所以 λ0 / (λ0+λ1+λ2) 越大，说明点云越不“扁平”，地面更粗糙
  float roughness = solver.eigenvalues()(0) / solver.eigenvalues().sum();

  // slope：坡度（由法向的 z 分量决定）
  // normal.z() = cos(theta)（theta 是法向与竖直方向的夹角）
  // acos(normal.z()) 得到 theta（0 表示水平面，越大表示越陡）
  //
  // 你后面除以 (slope_threshold_ * pi) 是一种人为归一化：
  // - slope_threshold_ 可能是一个阈值比例（例如 0.5 表示 90°*0.5=45°）
  // - 最终 slope 是一个无量纲比值，用于后续 traversability 代价融合
  float slope = std::acos(std::min(1.0, std::max(-1.0, normal.z()))) / (slope_threshold_ * M_PI);

  // =====================================================
  // [7] 将计算结果写回 grid_map 的 layer
  // =====================================================
  // 将法向向量分量保存到 normal_x/y/z
  height_map_.at("normal_x", index) = normal.x();
  height_map_.at("normal_y", index) = normal.y();
  height_map_.at("normal_z", index) = normal.z();

  // 保存 slope 与 roughness（你还做了阈值归一化）
  height_map_.at("slope", index) = slope;
  height_map_.at("roughness", index) = roughness / roughness_threshold_;

  return true;
}

// ==================== Step Mapping ====================
void Mapping::step_mapping()
{
  // =====================================================
  // [0] 加锁：保护 height_map_ 的读写
  // =====================================================
  // step 计算会读取 elevation_BGK、写入 step layer、写入 incremental_step_computed 标记层。
  // 若同时有发布线程或其它线程访问 height_map_，需要加锁避免并发读写冲突。
  std::lock_guard<std::mutex> lock(gridmap_publish_mutex_);

  // 增量模式标记层：某个 cell 的 step 是否已经计算过
  // 约定：>0 表示已计算，避免重复计算
  auto &incremental_step_computed = height_map_["incremental_step_computed"];

  // =====================================================
  // [1] 获取车辆当前位置：用于限制处理范围
  // =====================================================
  // 只在车周围 max_range_ 圆形范围内计算 step，减少全图计算量。
  Eigen::Vector3d current_pos;
  {
    std::lock_guard<std::mutex> pose_lock(vehicle_pose_mutex_);
    current_pos = vehicle_position_;
  }

  // =====================================================
  // [2] 遍历车辆周围 max_range_ 圆形区域内的所有 cell
  // =====================================================
  for (grid_map::CircleIterator it(
           height_map_,
           grid_map::Position(current_pos.x(), current_pos.y()), // 圆心：车辆当前位置（map坐标）
           max_range_);                                          // 半径：最大处理范围
       !it.isPastEnd();
       ++it)
  {
    // 当前 cell 的索引 (i, j)
    grid_map::Index index = *it;

    // =================================================
    // [2.1] 增量模式：跳过已经算过 step 的 cell
    // =================================================
    // enable_incremental_step_ 开启时，step 对每个 cell 只算一次（或按其它策略算一次）
    // incremental_step_computed > 0.1f 认为已经算过
    if (enable_incremental_step_)
    {
      if (incremental_step_computed(index(0), index(1)) > 0.1f)
      {
        continue;
      }
    }

    // =================================================
    // [2.2] 对单个 cell 计算 step 特征（核心计算在 areaSingleStepComputation）
    // =================================================
    bool computed = areaSingleStepComputation(index);

    // =================================================
    // [2.3] 若开启增量模式且计算成功，则标记该 cell 已计算
    // =================================================
    // 下次 step_mapping() 会跳过它，节省时间
    if (enable_incremental_step_ && computed)
    {
      incremental_step_computed(index(0), index(1)) = 1.0f;
    }
  }
}

bool Mapping::areaSingleStepComputation(const grid_map::Index &index)
{
  // =====================================================
  // [0] 将当前 cell 的 index 转换成连续坐标 center_pos=(x,y)
  // =====================================================
  grid_map::Position center_pos;
  if (!height_map_.getPosition(index, center_pos))
  {
    // index 无法映射到位置（通常是非法索引或地图状态异常）
    return false;
  }

  // =====================================================
  // [1] 取当前 cell 的高度（用 elevation_BGK 作为高度来源）
  // =====================================================
  // elevation_BGK：包含观测高度 + 插值补全高度
  // step 特征通常希望在缺测区域也能计算，所以你用 BGK 层更“连续”
  float center_z = height_map_.at("elevation_BGK", index);

  // 若当前 cell 的高度无效（NaN），无法计算 step
  if (std::isnan(center_z))
  {
    return false;
  }

  // =====================================================
  // [2] 在 step_radius_ 邻域内遍历邻居格子，计算最大高度差（max_step）
  // =====================================================
  // step 的直觉：如果周围存在“台阶/突变”，那么中心格与某些邻居格的高度差会很大
  // 你采用的定义：max_step = max_{neighbor in radius} |z_center - z_neighbor|
  float max_step = -1.0f; // 初始化为 -1 表示尚未找到任何有效邻居

  for (grid_map::CircleIterator sit(height_map_, center_pos, step_radius_);
       !sit.isPastEnd();
       ++sit)
  {
    // 取邻居格子的高度
    float neighbor_z = height_map_.at("elevation_BGK", *sit);

    // 只用有效高度参与计算
    if (!std::isnan(neighbor_z))
    {
      // 当前邻居与中心格的高度差（绝对值）
      float step = std::abs(center_z - neighbor_z);

      // 更新最大高度差
      max_step = std::max(step, max_step);
    }
  }

  // 如果 max_step 仍然 < 0，说明邻域里没有任何有效高度，无法计算
  if (max_step < 0)
  {
    return false;
  }

  // =====================================================
  // [3] step 阈值检查：用于归一化（防止除零）
  // =====================================================
  // 你最后存的是归一化 step：step / step_threshold_
  // step_threshold_ 必须 > 0，否则无法归一化
  if (step_threshold_ <= 0.0)
  {
    // ROS_WARN_THROTTLE：每 5 秒最多打印一次，避免日志刷屏
    ROS_WARN_THROTTLE(5.0, "Step threshold must be positive to compute normalized step values");
    return false;
  }

  // =====================================================
  // [4] 写回 step layer：归一化后的台阶指标
  // =====================================================
  // 归一化后：
  // - step ≈ 1 表示接近阈值大小的台阶
  // - step > 1 表示超过阈值的突变（可认为更不可通行）
  height_map_.at("step", index) = max_step / step_threshold_;

  return true;
}

// ==================== Traversability Mapping ====================
void Mapping::traversability_mapping()
{
  // =====================================================
  // [0] 加锁：保护 height_map_ 各层的读写
  // =====================================================
  // 该函数会读取 slope/step/roughness，写入 traversability、incremental_trav_computed。
  // 加锁避免发布线程或其它线程同时访问造成数据竞争。
  std::lock_guard<std::mutex> lock(gridmap_publish_mutex_);

  // =====================================================
  // [1] 获取各层引用（grid_map layer）
  // =====================================================
  auto &slope = height_map_["slope"];                                         // 归一化坡度指标（>=1 表示超过阈值）
  auto &step = height_map_["step"];                                           // 归一化台阶指标（>=1 表示超过阈值）
  auto &roughness = height_map_["roughness"];                                 // 归一化粗糙度指标（>=1 表示超过阈值）
  auto &traversability = height_map_["traversability"];                       // 输出层：可通行性代价 [0,1]
  auto &incremental_trav_computed = height_map_["incremental_trav_computed"]; // 增量模式标记层

  // =====================================================
  // [2] 获取车辆当前位置：用于限制处理范围
  // =====================================================
  // 只在车周围 max_range_ 圆形范围内融合 traversability，避免全图遍历。
  Eigen::Vector3d current_pos;
  {
    std::lock_guard<std::mutex> pose_lock(vehicle_pose_mutex_);
    current_pos = vehicle_position_;
  }

  // =====================================================
  // [3] 遍历车辆周围 max_range_ 圆形范围内的 cell
  // =====================================================
  for (grid_map::CircleIterator it(
           height_map_,
           grid_map::Position(current_pos.x(), current_pos.y()), // 圆心：车位置（map坐标）
           max_range_);                                          // 半径：处理范围
       !it.isPastEnd();
       ++it)
  {
    // 当前 cell 索引 (i,j)
    grid_map::Index idx = *it;

    // -------------------------------------------------
    // [3.1] 增量模式：跳过已经融合过的 cell
    // -------------------------------------------------
    // enable_incremental_trav_ 开启时，遍历时不会重复更新同一个 cell，
    // 以节省计算开销。
    if (enable_incremental_trav_)
    {
      if (incremental_trav_computed(idx(0), idx(1)) > 0.1f)
      {
        continue;
      }
    }

    // -------------------------------------------------
    // [3.2] 读取三个几何特征（已经归一化）
    // -------------------------------------------------
    float slope_val = slope(idx(0), idx(1));
    float step_val = step(idx(0), idx(1));
    float roughness_val = roughness(idx(0), idx(1));

    // -------------------------------------------------
    // [3.3] 有效性检查：有任何 NaN 则跳过
    // -------------------------------------------------
    // NaN 可能来自：
    // - BGK/高度缺失导致法向/粗糙度无法计算
    // - step 邻域无有效高度
    if (std::isnan(slope_val) || std::isnan(step_val) || std::isnan(roughness_val))
    {
      continue;
    }

    // =================================================
    // [4] 融合策略：阈值截断 + 平均
    // =================================================
    // 你的设计是“硬阈值”：
    // - 任一特征达到/超过 1.0（即超过阈值） -> 直接判定不可通行（cost=1）
    // - 否则对三者取平均，得到 traversability_cost
    float traversability_cost;

    // -------------------------------------------------
    // [4.1] 硬判不可通行：只要任一指标 >= 1.0
    // -------------------------------------------------
    // 因为 slope/step/roughness 你前面都做了阈值归一化：
    //   feature_norm = feature_raw / threshold
    // 所以 feature_norm >= 1 表示超阈值
    if (slope_val >= 1.0f || step_val >= 1.0f || roughness_val >= 1.0f)
    {
      traversability_cost = 1.0f; // 不可通行
    }
    else
    {
      // -------------------------------------------------
      // [4.2] 可通行区域：三者平均
      // -------------------------------------------------
      // 这是一种非常简单的融合：把三个风险源等权看待
      traversability_cost = (slope_val + step_val + roughness_val) / 3.0f;

      // 安全起见：再 clamp 到 [0,1]
      traversability_cost = std::min(1.0f, std::max(0.0f, traversability_cost));
    }

    // =================================================
    // [5] 写回 traversability layer
    // =================================================
    // 约定：0 -> 最可通行，1 -> 最不可通行
    traversability(idx(0), idx(1)) = traversability_cost;

    // =================================================
    // [6] 增量模式标记：本格子已经融合过
    // =================================================
    if (enable_incremental_trav_)
    {
      incremental_trav_computed(idx(0), idx(1)) = 1.0f;
    }
  }
}

// ==================== Fine-grained Traversability Mapping ====================
void Mapping::finegrained_traversability_mapping()
{
  // =====================================================
  // [0] 地图互斥锁：保护 height_map_ 的 layer 读写
  // =====================================================
  // 注意：整个函数都在锁内执行，如果 predictRobotPose 很慢，会阻塞发布/其它线程
  std::lock_guard<std::mutex> lock(gridmap_publish_mutex_);

  // -----------------------------------------------------
  // [1] 获取层引用
  // -----------------------------------------------------
  auto &traversability = height_map_["traversability"]; // 粗粒度 traversability（0~1）
  auto &slope_layer = height_map_["slope"];             // 坡度层（通常是归一化后的）
  auto &critical_layer = height_map_["critical"];       // 标记层：哪些 cell 做了 fine 检查

  // 每次重新计算 fine 时，把 critical 全图清零
  // 这一步是 O(地图总cell数) 的，会比较贵（后面我会说如何优化）
  critical_layer.setConstant(0.0f);

  // =====================================================
  // [2] 定义一个“车辆评估上下文”，用同一套循环处理多个车型
  // =====================================================
  struct VehicleEvalContext
  {
    int id;                  // 车辆类型 ID（传给 predictRobotPose 用）
    const HeightGrid *model; // 指向该车型的高度/足迹模型（轮式/履带）
    grid_map::Matrix *layer; // 输出层指针（traversability_fine_*）
    std::string name;        // 用于日志

    // 统计信息
    int count_success = 0;
    int count_collision = 0;
    int count_failure = 0;
  };

  std::vector<VehicleEvalContext> vehicles;
  vehicles.reserve(2);

  // -----------------------------------------------------
  // [3] 注册车辆：检查模型是否加载、layer 是否存在，然后初始化输出层
  // -----------------------------------------------------
  auto registerVehicle = [&](int id, const HeightGrid &model, bool loaded,
                             const std::string &layer_name, const std::string &name)
  {
    // 模型没加载（比如文件不存在/参数未配置）就跳过该车型
    if (!loaded)
    {
      return;
    }

    // 输出 layer 不存在就跳过（避免 height_map_["xxx"] 抛错或创建失败）
    if (!height_map_.exists(layer_name))
    {
      ROS_WARN("Grid map layer %s is missing; skipping fine-grained output for %s",
               layer_name.c_str(), name.c_str());
      return;
    }

    VehicleEvalContext ctx;
    ctx.id = id;
    ctx.model = &model;

    // 注意：这里把 layer 的矩阵指针保存下来，后面直接写矩阵
    ctx.layer = &height_map_[layer_name];

    // 初始化：先把 coarse traversability 拷贝到 fine layer
    // 这是一整张矩阵的拷贝，成本 O(全图 cell 数)
    *(ctx.layer) = traversability;

    ctx.name = name;
    vehicles.push_back(std::move(ctx));
  };

  // 注册两种车型
  registerVehicle(1, wheeled_model_, wheeled_model_loaded_, "traversability_fine_wheeled", "wheeled car");
  registerVehicle(2, tracked_model_, tracked_model_loaded_, "traversability_fine_tracked", "tracked car");

  // 如果一个车型都没注册成功，则不做 fine
  if (vehicles.empty())
  {
    ROS_WARN_THROTTLE(5.0, "Fine-grained traversability skipped: no vehicle models available");
    return;
  }

  // =====================================================
  // [4] 获取车辆当前位姿快照（位置 + 朝向），用于确定评估区域和 yaw
  // =====================================================
  Eigen::Vector3d current_pos;
  Eigen::Quaterniond current_orientation;
  {
    std::lock_guard<std::mutex> pose_lock(vehicle_pose_mutex_);
    current_pos = vehicle_position_;
    current_orientation = vehicle_orientation_; // 拿一个一致的姿态快照
  }

  // -----------------------------------------------------
  // [5] 由 orientation 提取 yaw，并构造 yaw 旋转矩阵
  // -----------------------------------------------------
  // 目的：让所有 cell 的评估都使用“同一个 yaw”，保证一致性
  tf::Quaternion q_vehicle(current_orientation.x(), current_orientation.y(),
                           current_orientation.z(), current_orientation.w());
  double yaw_angle = tf::getYaw(q_vehicle); // 车体航向角（弧度）

  // 只绕 Z 轴旋转的旋转矩阵（yaw）
  Eigen::Vector3d z_axis(0, 0, 1);
  Eigen::Matrix3d yaw_rotation = Eigen::AngleAxisd(yaw_angle, z_axis).toRotationMatrix();

  int count_skipped = 0;

  // =====================================================
  // [6] 遍历车周围 max_range_ 圆形区域的所有 cell
  // =====================================================
  for (grid_map::CircleIterator it(height_map_, grid_map::Position(current_pos.x(), current_pos.y()), max_range_);
       !it.isPastEnd(); ++it)
  {
    grid_map::Index idx = *it;

    // ---------------------------------------------------
    // [6.1] slope 过滤：只对“足够陡/值得细算”的区域评估
    // ---------------------------------------------------
    // fine_slope_min_：你人为设定的阈值（通常是归一化 slope 的下限）
    // 低于这个阈值认为地形太平坦，不需要做 expensive 的车辆姿态预测
    float slope_val = slope_layer(idx(0), idx(1));
    if (std::isnan(slope_val) || slope_val < fine_slope_min_)
    {
      count_skipped++;
      continue;
    }

    // 获取这个 idx 对应的连续坐标 pos（这里实际没被后续使用，可能是遗留代码）
    grid_map::Position pos;
    if (!height_map_.getPosition(idx, pos))
    {
      continue;
    }

    // 标记该 cell 被纳入了 fine 检查（critical=1）
    critical_layer(idx(0), idx(1)) = 1.0f;

    // ===================================================
    // [6.2] 对每一种车型做“落脚姿态预测”
    // ===================================================
    for (auto &vehicle : vehicles)
    {
      double roll = 0.0;
      double pitch = 0.0;
      int contact_points = 0;
      int is_stable = 1; // 1稳定，0不稳定（由 predictRobotPose 写回）

      // -------------------------------------------------
      // predictRobotPose 的语义（从用法推断）：
      // 输入：cell idx、车辆 yaw_rotation、车辆模型（足迹/离散采样点）、车型id
      // 输出：roll/pitch、contact_points、is_stable、返回 pose_status
      //
      // pose_status 你这里使用约定：
      //   1 => collision / 无法放置（直接 cost=1）
      //   2 => 成功预测并可用
      // 其它 => 失败（跳过，保留 coarse 值）
      // -------------------------------------------------
      int pose_status = predictRobotPose(idx, roll, pitch, contact_points, is_stable,
                                         yaw_rotation, *vehicle.model, vehicle.id);

      // [A] 碰撞：直接不可通行
      if (pose_status == 1)
      {
        vehicle.count_collision++;
        (*(vehicle.layer))(idx(0), idx(1)) = 1.0f;
        continue;
      }

      // [B] 预测失败：不更新该 cell（维持初始化时拷贝的 coarse traversability）
      if (pose_status != 2)
      {
        vehicle.count_failure++;
        continue;
      }

      // [C] 成功：用 roll/pitch 计算 fine 代价
      vehicle.count_success++;

      // -------------------------------------------------
      // 归一化：roll/pitch 超过阈值就趋向 1.0
      // 注意：变量名 fine_roll_threshold_deg_ 暗示阈值单位是“度”
      //       但 roll/pitch 是不是度取决于 predictRobotPose 的实现
      //       如果 predictRobotPose 输出是“弧度”，这里就会单位错配！
      // -------------------------------------------------
      const double roll_norm = std::max(1e-3, fine_roll_threshold_deg_);
      const double pitch_norm = std::max(1e-3, fine_pitch_threshold_deg_);

      double roll_cost = std::min(1.0, std::fabs(roll) / roll_norm);
      double pitch_cost = std::min(1.0, std::fabs(pitch) / pitch_norm);

      // 细粒度代价：roll 与 pitch 代价的均值
      float fine_traversability = (roll_cost + pitch_cost) / 2.0f;

      // 如果判定不稳定（比如支撑点不足、翻车风险等），直接置为不可通行
      if (is_stable == 0)
      {
        fine_traversability = 1.0f;
      }

      // 写回该车型对应的 fine 层
      (*(vehicle.layer))(idx(0), idx(1)) = fine_traversability;
    }
  }

  // =====================================================
  // [7] 打印统计信息：看看有多少 cell 被跳过、成功/碰撞/失败各多少
  // =====================================================
  ROS_INFO("Fine-grained traversability mapping statistics:");
  ROS_INFO("  - Skipped (filter criteria not met): %d cells", count_skipped);
  for (const auto &vehicle : vehicles)
  {
    ROS_INFO("  - %s => success: %d, collision: %d, failure: %d",
             vehicle.name.c_str(), vehicle.count_success, vehicle.count_collision, vehicle.count_failure);
  }
}

// ==================== Robot Pose Prediction ====================
int Mapping::predictRobotPose(const grid_map::Index &center_idx, double &roll, double &pitch,
                              int &contact_points, int &stable, const Eigen::Matrix3d &yaw_rotation,
                              const HeightGrid &vehicle_model, int vehicle_type)
{
  // -----------------------------------------
  // [A1] 将中心 cell 索引转成连续坐标 (x,y)
  // -----------------------------------------
  // center_pos 是地图坐标系中的平面位置，用来获取子地图并作为车辆平移中心
  grid_map::Position center_pos;
  if (!height_map_.getPosition(center_idx, center_pos))
  {
    return 0; // Failed
  }

  // -----------------------------------------
  // [A2] 计算一个足够覆盖车辆的子地图范围（正方形）
  // -----------------------------------------
  // vehicle_model 是 robot_rows_ x robot_cols_ 的离散采样网格（车辆足迹/底盘采样点）
  // robot_model_resolution_ 是模型网格的分辨率
  //
  // 这里把车辆占地宽/长估计出来，然后取对角线长度作为子地图边长（submap_side）
  double vehicle_width = robot_rows_ * robot_model_resolution_;
  double vehicle_length = robot_cols_ * robot_model_resolution_;
  double submap_side = std::sqrt(vehicle_width * vehicle_width + vehicle_length * vehicle_length);
  grid_map::Length submap_length(submap_side, submap_side);

  bool success;
  grid_map::GridMap submap = height_map_.getSubmap(center_pos, submap_length, success);

  if (!success)
  {
    return 0; // Failed
  }

  // -----------------------------------------
  // [B1] 收集子地图内所有有效地形点，用于拟合一个局部平面
  // -----------------------------------------
  // terrain_points 用于 fitPlane()：得到局部地形法向 normal
  std::vector<Eigen::Vector3d> terrain_points;
  for (grid_map::GridMapIterator it(submap); !it.isPastEnd(); ++it)
  {
    grid_map::Index idx = *it;
    float z = submap.at("elevation_BGK", idx);

    // elevation_BGK 有 NaN 的格子跳过
    if (!std::isnan(z))
    {
      grid_map::Position pos;
      submap.getPosition(idx, pos);
      terrain_points.push_back(Eigen::Vector3d(pos.x(), pos.y(), z));
    }
  }

  // 点太少无法拟合平面
  if (terrain_points.size() < 4)
  {
    return 0; // Failed
  }

  // -----------------------------------------
  // [B2] 拟合平面得到法向 normal（fitPlane 内部一般就是 PCA/最小二乘）
  // -----------------------------------------
  Eigen::Vector3d normal = fitPlane(terrain_points);

  // -----------------------------------------
  // [B3] 用 normal 计算“让车辆 z轴对齐到 normal 的旋转”= 地形坡度姿态（roll/pitch）
  // -----------------------------------------
  // z_axis = 世界竖直方向
  Eigen::Vector3d z_axis(0, 0, 1);

  // rotation_axis = z_axis x normal：把 z 转到 normal 的旋转轴
  Eigen::Vector3d rotation_axis = z_axis.cross(normal);

  Eigen::Matrix3d terrain_rotation = Eigen::Matrix3d::Identity();
  if (rotation_axis.norm() > 1e-6)
  {
    // angle = arccos(z · normal)：夹角
    double angle = std::acos(std::min(1.0, std::max(-1.0, z_axis.dot(normal))));

    // terrain_rotation：绕 rotation_axis 旋转 angle
    Eigen::AngleAxisd aa(angle, rotation_axis.normalized());
    terrain_rotation = aa.toRotationMatrix();
  }

  // -----------------------------------------
  // [B4] 合成最终初始姿态：R = yaw * (roll/pitch)
  // -----------------------------------------
  // yaw_rotation 是外部传入的“车辆航向角”旋转（绕 Z）
  // terrain_rotation 是“坡面倾斜”旋转（把 z 对齐到 normal）
  // 注意：矩阵乘法顺序很重要
  // R = yaw * terrain ：表示先对齐坡面，再施加车辆航向
  Eigen::Matrix3d R_matrix = yaw_rotation * terrain_rotation;

  // -----------------------------------------
  // [B5] 构造齐次变换 T：初始把车辆模型放在 (center_x, center_y, z=0) 上
  // -----------------------------------------
  Eigen::Matrix4d T_matrix = Eigen::Matrix4d::Identity();
  T_matrix.block<3, 3>(0, 0) = R_matrix;
  T_matrix.block<3, 1>(0, 3) = Eigen::Vector3d(center_pos.x(), center_pos.y(), 0.0);

  // ==================== Iterative Pose Refinement ====================
  // min_gap：车辆模型点到地形的最小“垂直间隙”（global_point.z - terrain_z）
  double min_gap = 100.0;
  std::vector<Eigen::Vector3d> touch_points;
  std::vector<Eigen::Vector3d> touch_poly;
  bool is_stable = false;
  double local_min_z = vehicle_model.Z_.minCoeff();

  contact_points = 0;
  bool collision = false;

  // --------- helper: 从当前 T_matrix 提取 roll/pitch（单位：deg）---------
  auto updateRollPitchFromT = [&](const Eigen::Matrix4d &T)
  {
    Eigen::Matrix3d R = T.block<3, 3>(0, 0);

    // roll = atan2(R21, R22)
    roll = std::atan2(R(2, 1), R(2, 2)) * 180.0 / M_PI;

    // pitch = asin(-R20)，做一下数值夹紧避免 asin 输入略超 [-1,1]
    double s = -R(2, 0);
    s = std::max(-1.0, std::min(1.0, s));
    pitch = std::asin(s) * 180.0 / M_PI;
  };

  // ==================== Iterative Rotation ====================
  int iteration = 0;
  while (iteration < max_iterations_)
  {
    // Step 0: Check collision and stability
    if (collision)
    {
      stable = 0;
      return 1;
    }

    if (is_stable)
    {
      stable = 1;
      updateRollPitchFromT(T_matrix);
      return 2;
    }

    // -----------------------------------------
    // [C1] 计算 gap_map，并找最小间隙 min_gap
    // -----------------------------------------
    // gap_map(i,j) 表示：车辆模型网格点 (i,j) 变换到世界后，与地形高度的差值
    // gap = z_vehicle_point - z_terrain
    // gap>0: 车辆在地形上方（悬空）
    // gap≈0: 接触
    // gap<0: 穿透/碰撞（这里用 collision_gap_threshold_ 判定）
    min_gap = 100.0;
    Eigen::MatrixXd gap_map(robot_rows_, robot_cols_);
    gap_map.setConstant(std::numeric_limits<double>::quiet_NaN());

    for (int i = 0; i < robot_rows_; ++i)
    {
      for (int j = 0; j < robot_cols_; ++j)
      {
        Eigen::Vector4d robot_point(vehicle_model.X_(i, j), vehicle_model.Y_(i, j), vehicle_model.Z_(i, j), 1.0);
        Eigen::Vector4d global_point = T_matrix * robot_point;

        grid_map::Index terrain_idx;
        if (submap.getIndex(grid_map::Position(global_point.x(), global_point.y()), terrain_idx))
        {
          float terrain_z = submap.at("elevation_BGK", terrain_idx);
          if (!std::isnan(terrain_z))
          {
            double gap = global_point.z() - terrain_z;
            gap_map(i, j) = gap;
            min_gap = std::min(min_gap, gap);
          }
        }
      }
    }

    // -----------------------------------------
    // [C2] “落地”：整体向下平移 min_gap，让最接近地面的点恰好接触地面
    // -----------------------------------------
    // 如果 min_gap 是正数，说明所有点都在地面之上，向下移 min_gap 就会有最小点接触地面
    // 如果 min_gap 是负数，说明已经有点低于地面，T_matrix(2,3)-=min_gap 会把车往上抬
    if (min_gap < 100.0)
    {
      T_matrix(2, 3) -= min_gap;
    }

    // -----------------------------------------
    // [D1] 找接触点：gap < touch_gap_threshold_ 的点当作“支撑接触”
    // -----------------------------------------
    touch_points.clear();
    std::vector<Eigen::Vector2d> touch_points_2d;
    contact_points = 0;
    collision = false;

    for (int i = 0; i < robot_rows_; ++i)
    {
      for (int j = 0; j < robot_cols_; ++j)
      {
        Eigen::Vector4d robot_point(vehicle_model.X_(i, j), vehicle_model.Y_(i, j), vehicle_model.Z_(i, j), 1.0);
        Eigen::Vector4d global_point = T_matrix * robot_point;

        grid_map::Index terrain_idx;
        if (submap.getIndex(grid_map::Position(global_point.x(), global_point.y()), terrain_idx))
        {
          float terrain_z = submap.at("elevation_BGK", terrain_idx);
          if (!std::isnan(terrain_z))
          {
            double gap = global_point.z() - terrain_z;
            gap_map(i, j) = gap;

            if (gap < touch_gap_threshold_)
            {
              contact_points++;
              touch_points.push_back(Eigen::Vector3d(global_point.x(), global_point.y(), global_point.z()));
              touch_points_2d.push_back(Eigen::Vector2d(vehicle_model.X_(i, j), vehicle_model.Y_(i, j)));

              if (checkWheel(vehicle_type, i, j) && gap < collision_gap_threshold_)
              {
                collision = true;
              }
            }
          }
        }
      }
    }

    if (collision)
    {
      stable = 0;
      return 1;
    }

    // 没有任何接触点：说明模型还没“落到地形上”或地形缺失
    // 直接进入下一次迭代
    if (touch_points.empty())
    {
      iteration++;
      continue;
    }

    // Step 4: 支撑多边形 + 重力线投影
    // cross2d: 计算二维叉积符号，用于凸包判向
    auto cross2d = [](const Eigen::Vector2d &O, const Eigen::Vector2d &A, const Eigen::Vector2d &B)
    {
      return (A.x() - O.x()) * (B.y() - O.y()) - (A.y() - O.y()) * (B.x() - O.x());
    };

    // computeHull: 输入一堆 2D 点，输出凸包点序列（逆/顺时针）
    auto computeHull = [&](std::vector<Eigen::Vector2d> pts)
    {
      if (pts.size() <= 3)
      {
        return pts;
      }
      std::sort(pts.begin(), pts.end(), [](const Eigen::Vector2d &a, const Eigen::Vector2d &b)
                {
                if (a.x() == b.x())
                {
                    return a.y() < b.y();
                }
                return a.x() < b.x(); });

      std::vector<Eigen::Vector2d> hull;
      for (const auto &p : pts)
      {
        while (hull.size() >= 2 && cross2d(hull[hull.size() - 2], hull.back(), p) <= 0)
        {
          hull.pop_back();
        }
        hull.push_back(p);
      }
      size_t lower_size = hull.size();
      for (int i = (int)pts.size() - 2; i >= 0; --i)
      {
        const auto &p = pts[i];
        while (hull.size() > lower_size && cross2d(hull[hull.size() - 2], hull.back(), p) <= 0)
        {
          hull.pop_back();
        }
        hull.push_back(p);
      }
      if (!hull.empty())
      {
        hull.pop_back();
      }
      return hull;
    };

    // pointInPoly: 判断点 p 是否在凸多边形 poly 内（假设 poly 点序有一致方向）
    auto pointInPoly = [&](const std::vector<Eigen::Vector3d> &poly, const Eigen::Vector3d &p)
    {
      for (size_t i = 0; i < poly.size(); ++i)
      {
        const auto &p1 = poly[i];
        const auto &p2 = poly[(i + 1) % poly.size()];
        Eigen::Vector2d v1(p2.x() - p1.x(), p2.y() - p1.y());
        Eigen::Vector2d v2(p.x() - p1.x(), p.y() - p1.y());
        if (v1.x() * v2.y() - v2.x() * v1.y() < 0)
        {
          return false;
        }
      }
      return true;
    };

    // 计算支撑凸包（模型系 2D）
    std::vector<Eigen::Vector2d> hull_2d = computeHull(touch_points_2d);

    // 将凸包点提升到 3D：z 用模型底面最小 z（local_min_z），再用 T_matrix 变换到世界系
    touch_poly.clear();
    for (const auto &p : hull_2d)
    {
      Eigen::Vector4d p4(p.x(), p.y(), local_min_z, 1.0);
      touch_poly.push_back((T_matrix * p4).block<3, 1>(0, 0));
    }

    if (touch_poly.empty())
    {
      iteration++;
      continue;
    }

    // 重力方向（世界系向下）
    Eigen::Vector3d gravity(0, 0, -1);
    // 车辆当前平移位置（世界系）
    Eigen::Vector3d T_pos = T_matrix.block<3, 1>(0, 3);
    // 从车辆位置沿重力方向作一条直线
    Eigen::ParametrizedLine<double, 3> gravity_line(T_pos, gravity);
    // contact_plane：接触平面，法向取车辆当前姿态的 z 轴（R 的第三列）
    Eigen::Vector3d plane_normal = T_matrix.block<3, 3>(0, 0).col(2);
    Eigen::Hyperplane<double, 3> contact_plane(plane_normal, touch_poly.front());
    // 重力线与接触平面的交点：可理解为“重心沿重力方向投影到接触平面的位置”
    Eigen::Vector3d intersection = gravity_line.intersectionPoint(contact_plane);

    Eigen::Vector3d rotatep1, rotatep2;
    bool has_rotation_line = false;

    if (touch_poly.size() == 1)
    {
      // 只有一个支撑点：投影若就在支撑点上 -> 稳定
      if ((touch_poly.front() - intersection).norm() < 1e-6)
      {
        is_stable = true;
      }
      else
      {
        // 不稳定：构造一个“旋转轴线”（这里用触点 + 某方向）
        rotatep1 = touch_poly.front();
        rotatep2 = touch_poly.front() + (intersection - touch_poly.front()).normalized().cross(gravity);
        has_rotation_line = true;
      }
    }
    else if (touch_poly.size() == 2)
    {
      // 两个支撑点：形成一条支撑边
      Eigen::Vector3d p1 = touch_poly[0];
      Eigen::Vector3d p2 = touch_poly[1];

      // 投影点 intersection 到边 p1p2 的投影 projection
      Eigen::Vector3d p1p2 = p2 - p1;
      Eigen::Vector3d p1a = intersection - p1;
      double t = p1a.dot(p1p2) / p1p2.dot(p1p2);
      Eigen::Vector3d projection = p1 + t * p1p2;

      // 如果投影点就在支撑边上 -> 稳定
      if ((projection - intersection).norm() < 1e-6)
      {
        is_stable = true;
      }
      else
      {
        // 不稳定：绕这条支撑边旋转
        rotatep1 = p1;
        rotatep2 = p2;
        has_rotation_line = true;
      }
    }
    else
    {
      // 多于两个支撑点：形成支撑多边形
      if (pointInPoly(touch_poly, intersection))
      {
        // 投影在多边形内：稳定
        is_stable = true;
      }
      else
      {
        // 投影在多边形外：需要找到一条“倾覆边”作为旋转轴
        double min_dis = std::numeric_limits<double>::max();

        for (size_t i = 0; i < touch_poly.size(); ++i)
        {
          const auto &p1 = touch_poly[i];
          const auto &p2 = touch_poly[(i + 1) % touch_poly.size()];

          // 先用 2D 叉积判断 intersection 在该边的哪一侧
          Eigen::Vector2d v1(p2.x() - p1.x(), p2.y() - p1.y());
          Eigen::Vector2d v2(intersection.x() - p1.x(), intersection.y() - p1.y());
          if (v1.x() * v2.y() - v2.x() * v1.y() >= 0)
          {
            continue; // 在“内侧”，不是倾覆边候选
          }

          // 计算 intersection 到边 p1p2 的距离 dis，选最近的那条边作为旋转轴
          Eigen::Vector3d p1p2 = p2 - p1;
          Eigen::Vector3d p1a = intersection - p1;
          double t = p1a.dot(p1p2) / p1p2.dot(p1p2);
          Eigen::Vector3d projection = p1 + t * p1p2;
          double dis = (projection - intersection).norm();

          if (dis < min_dis)
          {
            min_dis = dis;
            rotatep1 = p1;
            rotatep2 = p2;
          }
        }

        has_rotation_line = true;
      }
    }

    // 如果稳定了，就进入下一轮循环开头，开头会 return 2
    if (is_stable)
    {
      stable = 1;
      updateRollPitchFromT(T_matrix);
      return 2;
    }

    if (has_rotation_line)
    {
      // 旋转轴的方向与参数化直线
      Eigen::Vector3d direction = (rotatep2 - rotatep1).normalized();
      Eigen::ParametrizedLine<double, 3> rotation_line(rotatep1, direction);

      // d_theta：要绕旋转轴转的角度（选一个最小可行值）
      double d_theta = std::numeric_limits<double>::max();

      // 为了计算点在旋转轴左/右侧，这里投影到 XY 平面做 2D 判断
      Eigen::Vector2d rot_origin(rotation_line.origin().x(), rotation_line.origin().y());
      Eigen::Vector2d rot_dir(rotation_line.direction().x(), rotation_line.direction().y());
      Eigen::Vector2d rot_dir_norm = rot_dir.normalized();

      // 遍历所有非接触点（gap > touch_gap_threshold_），估计需要转多少角
      for (int i = 0; i < robot_rows_; ++i)
      {
        for (int j = 0; j < robot_cols_; ++j)
        {
          double gap = gap_map(i, j);

          // NaN 或已经接触的点跳过
          if (std::isnan(gap) || gap <= touch_gap_threshold_)
          {
            continue;
          }

          // 点变换到世界系
          Eigen::Vector4d robot_point(vehicle_model.X_(i, j), vehicle_model.Y_(i, j), vehicle_model.Z_(i, j), 1.0);
          Eigen::Vector4d global_point = T_matrix * robot_point;

          // 计算点相对旋转轴起点的 2D 向量
          Eigen::Vector2d origin_to_cell(global_point.x() - rot_origin.x(), global_point.y() - rot_origin.y());

          // 判断点在旋转轴的哪一侧（叉积符号）
          // 只对某一侧的点计算 d_theta（避免两边一起压导致矛盾）
          if (origin_to_cell.x() * rot_dir_norm.y() - origin_to_cell.y() * rot_dir_norm.x() > 0)
          {
            // 计算点到旋转轴的垂直距离 distance（在 XY 平面）
            Eigen::Vector2d proj_vec = origin_to_cell.dot(rot_dir_norm) * rot_dir_norm;
            double distance = (origin_to_cell - proj_vec).norm();

            // atan2(gap, distance) 是一个“需要转的角度”的几何估计：
            // gap 越大、distance 越小 -> 需要更大角度把它压下来
            d_theta = std::min(d_theta, std::atan2(gap, distance));
          }
        }
      }

      // 如果找到了有效 d_theta，就绕 rotation_line 旋转更新 T_matrix
      if (d_theta < std::numeric_limits<double>::max())
      {
        // 下面是一套标准的“绕任意轴旋转”的齐次矩阵构造：
        // 1) 平移到轴原点
        // 2) 旋转坐标系，让旋转轴对齐到 z 轴
        // 3) 绕 z 轴旋转 d_theta
        // 4) 旋转回去
        // 5) 平移回去

        Eigen::Matrix4d T1 = Eigen::Matrix4d::Identity();
        T1.block<3, 1>(0, 3) = -rotation_line.origin();

        Eigen::Matrix4d R1 = Eigen::Matrix4d::Identity();
        R1.block<3, 3>(0, 0) =
            Eigen::Quaterniond().setFromTwoVectors(rotation_line.direction(), Eigen::Vector3d(0, 0, 1)).toRotationMatrix();

        Eigen::Matrix4d R2 = Eigen::Matrix4d::Identity();
        R2.block<3, 3>(0, 0) = Eigen::AngleAxisd(d_theta, Eigen::Vector3d(0, 0, 1)).toRotationMatrix();

        Eigen::Matrix4d R3 = Eigen::Matrix4d::Identity();
        R3.block<3, 3>(0, 0) = R1.block<3, 3>(0, 0).transpose();

        Eigen::Matrix4d T2 = Eigen::Matrix4d::Identity();
        T2.block<3, 1>(0, 3) = rotation_line.origin();

        Eigen::Matrix4d rotate_with_rotation_line = T2 * R3 * R2 * R1 * T1;

        // 更新车辆位姿
        T_matrix = rotate_with_rotation_line * T_matrix;

        // 旋转可能引起 xy 平移漂移，这里强制把 xy 拉回 center_pos
        T_matrix.block<2, 1>(0, 3) = Eigen::Vector2d(center_pos.x(), center_pos.y());
      }
    }

    iteration++;
  }

  // -----------------------------------------
  // [H1] 从最终旋转矩阵提取 roll/pitch（并转成度）
  // -----------------------------------------
  Eigen::Matrix3d R = T_matrix.block<3, 3>(0, 0);

  // 这里是标准欧拉角提取（假设 R = Rz*Ry*Rx 或类似约定）
  roll = std::atan2(R(2, 1), R(2, 2)) * 180.0 / M_PI;
  pitch = std::asin(-R(2, 0)) * 180.0 / M_PI;

  ROS_INFO("Pose result (vehicle %d) at cell (%.2f, %.2f): roll=%.3f, pitch=%.3f, stable=%d, collision=%d",
           vehicle_type, center_pos.x(), center_pos.y(), roll, pitch, is_stable ? 1 : 0, collision ? 1 : 0);

  stable = is_stable ? 1 : 0;

  // -----------------------------------------
  // [H2] 返回码约定
  // -----------------------------------------
  // 2 = 成功且稳定
  // 1 = 碰撞（上面已经提前 return 1）
  // 0 = 未稳定或失败
  return is_stable ? 2 : 0;
}

// ==================== Plane Fitting ====================
bool Mapping::checkWheel(int vehicle_type, int i, int j) const
{
  // 说明：i/j 是车辆模型离散网格的行列索引（从 0 开始）
  // 该函数用于区分“允许接触地面的区域(轮/履带)” 和 “不该接触的区域(底盘/车体)”

  if (vehicle_type == 1)
  {
    // -------------------------
    // 车型 1：轮式车
    // -------------------------
    // wheel_rows / wheel_cols 描述轮子接触斑块所在的行/列集合
    // 注意：这里只是“离散网格上的近似轮子区域”，不是连续几何

    static const std::array<int, 4> wheel_rows = {0, 1, 7, 8};    // 轮子所在的行范围（前后两排）
    static const std::array<int, 5> wheel_cols = {1, 2, 3, 8, 9}; // 轮子所在的列范围（左右两侧）

    // row_matches: i 是否落在轮子行集合
    bool row_matches = std::find(wheel_rows.begin(), wheel_rows.end(), i) != wheel_rows.end();

    // col_matches: j 是否落在轮子列集合
    bool col_matches = std::find(wheel_cols.begin(), wheel_cols.end(), j) != wheel_cols.end();

    if (row_matches && col_matches)
    {
      // (i,j) 同时落在轮子行集合 & 轮子列集合 → 认为这是“轮子接触区域”
      // 返回 false：表示“这是允许接触的点”，不要用它做 collision 判定
      return false;
    }

    // 其它位置：认为属于底盘/车体区域（不应该穿透/接触）
    // 返回 true：表示“这是敏感点”，如果 gap 很小/为负 → 判 collision
    return true;
  }

  if (vehicle_type == 2)
  {
    // -------------------------
    // 车型 2：履带车
    // -------------------------
    // 这里用 “列集合” 表示履带接触面（例如左右两条履带所在的列带）
    static const std::array<int, 4> wheel_cols = {0, 1, 7, 8};

    bool col_matches = std::find(wheel_cols.begin(), wheel_cols.end(), j) != wheel_cols.end();
    if (col_matches)
    {
      // 落在履带接触列 → 允许接触 → 返回 false（不作为 collision 敏感点）
      return false;
    }

    // 中间区域：视为车体/底盘 → 返回 true（敏感点）
    return true;
  }

  // 其它 vehicle_type：兜底当作底盘敏感点（一般不应该走到这）
  return true;
}

Eigen::Vector3d Mapping::fitPlane(const std::vector<Eigen::Vector3d> &points)
{
  // -----------------------------------------
  // [0] 输入点数不足：无法拟合平面
  // -----------------------------------------
  // 平面拟合至少需要 3 个不共线点；这里给一个默认法向 (0,0,1)
  if (points.size() < 3)
  {
    return Eigen::Vector3d(0, 0, 1); // 默认向上法向
  }

  // -----------------------------------------
  // [1] 计算点云质心 centroid
  // -----------------------------------------
  // centroid = (1/N) * sum(p_i)
  // 用质心把点云平移到“零均值”，后面算协方差会更稳定、也符合 PCA 定义
  Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
  for (const auto &p : points)
  {
    centroid += p;
  }
  centroid /= points.size();

  // -----------------------------------------
  // [2] 计算协方差矩阵 cov
  // -----------------------------------------
  // 对每个点：diff = p_i - centroid
  // cov = (1/N) * sum(diff * diff^T)
  //
  // cov 是 3x3 对称正半定矩阵，描述点云在 xyz 三个方向上的“离散程度”
  // 平面点云会在两个方向离散较大，在法向方向离散最小
  Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
  for (const auto &p : points)
  {
    Eigen::Vector3d diff = p - centroid;
    cov += diff * diff.transpose();
  }
  cov /= points.size();

  // -----------------------------------------
  // [3] 对协方差矩阵做特征值分解（PCA）
  // -----------------------------------------
  // solver.eigenvalues(): λ0 ≤ λ1 ≤ λ2
  // solver.eigenvectors(): 对应特征向量 v0, v1, v2
  //
  // 对称矩阵用 SelfAdjointEigenSolver 非常合适，数值稳定且返回正交基
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);

  // -----------------------------------------
  // [4] 平面法向 = 最小特征值对应的特征向量
  // -----------------------------------------
  // 直觉：平面点云在“法向方向”的方差最小（点都贴在一个面上）
  // 因此 cov 最小特征值方向就是法向方向
  Eigen::Vector3d normal = solver.eigenvectors().col(0);

  // -----------------------------------------
  // [5] 统一法向朝向：保证朝上（z 正方向）
  // -----------------------------------------
  // 平面法向有两个等价方向 n 和 -n，为了统一输出，让它与 (0,0,1) 点积为正
  if (normal.dot(Eigen::Vector3d(0, 0, 1)) < 0.0)
  {
    normal = -normal;
  }

  // -----------------------------------------
  // [6] 输出单位法向
  // -----------------------------------------
  return normal.normalized();
}

// ==================== Grid Map Publishing ====================
void Mapping::publishGridMap()
{
  grid_map_msgs::GridMap message;
  if (!buildGridMapMessage(message))
  // if (!buildGridMapFromViewpointCloud(message))
  {
    return;
  }
  gridmap_pub_.publish(message);
}

bool Mapping::buildGridMapMessage(grid_map_msgs::GridMap &message)
{
  // ==========================================
  // [1] 加锁：保护 height_map_ 的读操作
  // ==========================================
  // gridmap_publish_mutex_ 主要目的是避免：
  // - 另一个线程正在更新 height_map_（写）
  // - 本线程同时在读取/裁剪/删除 layer
  // 从而造成数据竞争或崩溃
  std::lock_guard<std::mutex> lock(gridmap_publish_mutex_);

  // ==========================================
  // [2] 读取车辆当前位置（快照）
  // ==========================================
  // 这里单独对 vehicle_pose_mutex_ 加锁，是因为 vehicle_position_
  // 由里程计回调/处理线程更新，需要线程安全
  Eigen::Vector3d current_pos;
  {
    std::lock_guard<std::mutex> pose_lock(vehicle_pose_mutex_);
    current_pos = vehicle_position_;
  }

  // ==========================================
  // [3] 从全局 height_map_ 裁剪局部 submap
  // ==========================================
  // local_map_size_x_/y_ 决定局部地图窗口的长宽（单位：m）
  // getSubmap 会复制出一个新的 GridMap（local_map），通常包括数据矩阵
  grid_map::Length submap_length(local_map_size_x_, local_map_size_y_);
  bool success;

  // 以车辆当前位置为中心裁剪一块局部区域
  // 注意：grid_map::Position 的坐标单位是 meters（世界系或 map frame）
  grid_map::GridMap local_map = height_map_.getSubmap(
      grid_map::Position(current_pos.x(), current_pos.y()),
      submap_length,
      success);

  // 如果裁剪失败，返回 false（例如：当前位置不在地图范围内）
  if (!success)
  {
    return false;
  }

  // ==========================================
  // [4] 只保留白名单层（减少消息体积）
  // ==========================================
  // 发布层白名单：只传这些层，避免把很多内部计算层发出去导致消息巨大
  static const std::vector<std::string> layers_to_publish = {
      "elevation",
      "elevation_BGK",
      "slope",
      "roughness",
      "step",
      "traversability",
      "traversability_fine_wheeled",
      "traversability_fine_tracked",
      "critical"};

  // local_map.getLayers() 返回当前 local_map 中实际存在的 layer 名称列表
  const auto existing_layers = local_map.getLayers();

  // 遍历 local_map 的所有 layer，如果不在白名单里就删掉
  // erase(layer) 会把对应 Matrix 从 local_map 里移除
  for (const auto &layer : existing_layers)
  {
    if (std::find(layers_to_publish.begin(), layers_to_publish.end(), layer) == layers_to_publish.end())
    {
      local_map.erase(layer);
    }
  }

  // ==========================================
  // [5] 转 ROS 消息
  // ==========================================
  // GridMapRosConverter 会把 local_map 的 metadata（尺寸、分辨率、坐标系）
  // 和各 layer 的 matrix 数据打包成 grid_map_msgs::GridMap 消息
  grid_map::GridMapRosConverter::toMessage(local_map, message);

  return true;
}

// bool Mapping::buildGridMapMessage(grid_map_msgs::GridMap &message)
// {
//   // ==========================================
//   // [0] 检查 origin 是否已收到
//   // ==========================================
//   if (!has_origin_.load(std::memory_order_acquire))
//   {
//     ROS_WARN_THROTTLE(2.0, "No viewpoint origin received yet, skip publishing.");
//     return false;
//   }

//   // 把 origin 的快照取出来（左下角）
//   Eigen::Vector3d origin;
//   ros::Time origin_stamp;
//   {
//     std::lock_guard<std::mutex> lk(origin_mutex_);
//     origin = latest_origin_;
//     origin_stamp = latest_origin_stamp_;
//   }

//   // ==========================================
//   // [1] 加锁：保护 height_map_ 的读操作
//   // ==========================================
//   std::lock_guard<std::mutex> lock(gridmap_publish_mutex_);

//   // ==========================================
//   // [2] 检查 size 合法性（用 ViewPoint 的窗口大小）
//   // ==========================================
//   if (viewpoint_grid_size_x_ <= 0.0 || viewpoint_grid_size_y_ <= 0.0)
//   {
//     ROS_WARN_THROTTLE(2.0, "Invalid viewpoint grid size: x=%.3f, y=%.3f",
//                       viewpoint_grid_size_x_, viewpoint_grid_size_y_);
//     return false;
//   }

//   // submap 的长宽（单位：m）
//   grid_map::Length submap_length(viewpoint_grid_size_x_, viewpoint_grid_size_y_);

//   // ==========================================
//   // [3] origin(左下角) -> submap center(中心点)
//   // ==========================================
//   grid_map::Position submap_center(origin.x() + 0.5 * viewpoint_grid_size_x_,
//                                    origin.y() + 0.5 * viewpoint_grid_size_y_);

//   // ==========================================
//   // [4] 可选但强烈建议：clamp center，防止窗口越界导致 getSubmap 失败
//   // ==========================================
//   const grid_map::Position map_center = height_map_.getPosition();
//   const grid_map::Length map_len = height_map_.getLength();

//   // 如果窗口比全图还大，直接失败（否则怎么裁都裁不出来）
//   if (submap_length.x() > map_len.x() || submap_length.y() > map_len.y())
//   {
//     ROS_WARN_THROTTLE(2.0,
//                       "Submap larger than height_map. submap=[%.2f %.2f], map=[%.2f %.2f]",
//                       submap_length.x(), submap_length.y(), map_len.x(), map_len.y());
//     return false;
//   }

//   const double map_min_x = map_center.x() - 0.5 * map_len.x();
//   const double map_max_x = map_center.x() + 0.5 * map_len.x();
//   const double map_min_y = map_center.y() - 0.5 * map_len.y();
//   const double map_max_y = map_center.y() + 0.5 * map_len.y();

//   const double half_x = 0.5 * submap_length.x();
//   const double half_y = 0.5 * submap_length.y();

//   // clamp center 到合法范围
//   submap_center.x() = std::min(std::max(submap_center.x(), map_min_x + half_x), map_max_x - half_x);
//   submap_center.y() = std::min(std::max(submap_center.y(), map_min_y + half_y), map_max_y - half_y);

//   // ==========================================
//   // [5] 从全局 height_map_ 裁剪局部 submap（按 origin+size）
//   // ==========================================
//   bool success = false;
//   grid_map::GridMap local_map = height_map_.getSubmap(submap_center, submap_length, success);

//   if (!success)
//   {
//     ROS_WARN_THROTTLE(1.0,
//                       "getSubmap failed. origin=[%.2f %.2f], center=[%.2f %.2f], size=[%.2f %.2f]",
//                       origin.x(), origin.y(),
//                       submap_center.x(), submap_center.y(),
//                       submap_length.x(), submap_length.y());
//     return false;
//   }

//   // 时间戳对齐 origin（可选）
//   local_map.setTimestamp(origin_stamp.toNSec());

//   // ==========================================
//   // [6] 只保留白名单层（减少消息体积）
//   // ==========================================
//   static const std::vector<std::string> layers_to_publish = {
//       "elevation",
//       "elevation_BGK",
//       "slope",
//       "roughness",
//       "step",
//       "traversability",
//       "traversability_fine_wheeled",
//       "traversability_fine_tracked",
//       "critical"};

//   const auto existing_layers = local_map.getLayers();
//   for (const auto &layer : existing_layers)
//   {
//     if (std::find(layers_to_publish.begin(), layers_to_publish.end(), layer) == layers_to_publish.end())
//     {
//       local_map.erase(layer);
//     }
//   }

//   // ==========================================
//   // [7] 转 ROS 消息
//   // ==========================================
//   grid_map::GridMapRosConverter::toMessage(local_map, message);
//   return true;
// }

// bool Mapping::buildGridMapMessage(grid_map_msgs::GridMap &message)
// {
//   // ==========================================================
//   // [0] 获取最新 origin（ViewPoint 左下角），没有就不发布
//   // ==========================================================
//   if (!has_origin_.load(std::memory_order_acquire))
//   {
//     ROS_WARN_THROTTLE(2.0, "No viewpoint origin yet, skip grid map publish.");
//     return false;
//   }

//   Eigen::Vector3d origin;
//   ros::Time origin_stamp;
//   {
//     std::lock_guard<std::mutex> lk(origin_mutex_);
//     origin = latest_origin_;
//     origin_stamp = latest_origin_stamp_;
//   }

//   // ==========================================================
//   // [1] 加锁保护 height_map_（读）
//   // ==========================================================
//   std::lock_guard<std::mutex> lock(gridmap_publish_mutex_);

//   // ==========================================================
//   // [2] 输出地图几何：严格对齐 ViewPoint（res + origin）
//   //     grid_map 只有一个二维 resolution，所以用 viewpoint_resolution_x_
//   // ==========================================================
//   const double vp_res_x = viewpoint_resolution_x_;
//   const double vp_res_y = viewpoint_resolution_y_;

//   double out_res = vp_res_x;
//   if (std::fabs(vp_res_x - vp_res_y) > 1e-9)
//   {
//     ROS_WARN_THROTTLE(2.0,
//                       "ViewPoint res x!=y (%.4f vs %.4f). grid_map uses single res, using res_x.",
//                       vp_res_x, vp_res_y);
//   }

//   if (out_res <= 0.0 || viewpoint_number_x_ <= 0 || viewpoint_number_y_ <= 0)
//   {
//     ROS_WARN_THROTTLE(2.0, "Invalid viewpoint params: res=%.4f, nx=%d, ny=%d",
//                       out_res, viewpoint_number_x_, viewpoint_number_y_);
//     return false;
//   }

//   // 窗口大小（米），严格用 number * ViewPoint res
//   const double size_x = static_cast<double>(viewpoint_number_x_) * out_res;
//   const double size_y = static_cast<double>(viewpoint_number_y_) * out_res;

//   // 让输出 map 的 min corner = origin（左下角）
//   const grid_map::Position out_center(origin.x() + 0.5 * size_x,
//                                       origin.y() + 0.5 * size_y);

//   grid_map::GridMap out_map;
//   out_map.setFrameId(height_map_.getFrameId()); // 若你希望固定 "map"，也可以写 "map"
//   out_map.setTimestamp(origin_stamp.toNSec());
//   out_map.setGeometry(grid_map::Length(size_x, size_y), out_res, out_center);

//   // ==========================================================
//   // [3] 选择发布层 + 每层的 pooling 策略
//   //     你要求 traversability 系列由 MIN 改为 MAX
//   // ==========================================================
//   enum class PoolType
//   {
//     MIN,
//     MAX,
//     MEAN
//   };

//   struct LayerPolicy
//   {
//     std::string name;
//     PoolType type;
//   };

//   const std::vector<LayerPolicy> policies = {
//       {"elevation", PoolType::MEAN},
//       {"elevation_BGK", PoolType::MEAN},
//       {"slope", PoolType::MEAN},
//       {"roughness", PoolType::MEAN},
//       {"step", PoolType::MAX},
//       {"traversability", PoolType::MAX},
//       {"traversability_fine_wheeled", PoolType::MAX},
//       {"traversability_fine_tracked", PoolType::MAX},
//       {"critical", PoolType::MAX},
//   };

//   // 只添加 height_map_ 中实际存在的层
//   std::vector<LayerPolicy> used;
//   used.reserve(policies.size());
//   for (const auto &p : policies)
//   {
//     if (height_map_.exists(p.name))
//     {
//       out_map.add(p.name, std::numeric_limits<float>::quiet_NaN());
//       used.push_back(p);
//     }
//   }

//   if (used.empty())
//   {
//     ROS_WARN_THROTTLE(2.0, "None of publish layers exist in height_map_.");
//     return false;
//   }

//   // ==========================================================
//   // [4] 块聚合 pooling：每个 coarse cell 覆盖 out_res × out_res 的窗口
//   //     在高分辨率 height_map_ 上聚合得到一个值
//   // ==========================================================
//   auto isFinite = [](float v) -> bool
//   { return std::isfinite(v); };

//   auto poolLayerOnSubmap = [&](const std::string &layer,
//                                const grid_map::SubmapGeometry &subgeom,
//                                PoolType type) -> float
//   {
//     grid_map::SubmapIterator it(subgeom);

//     if (type == PoolType::MEAN)
//     {
//       double sum = 0.0;
//       int cnt = 0;
//       for (; !it.isPastEnd(); ++it)
//       {
//         const grid_map::Index idx = *it;
//         const float v = height_map_.at(layer, idx);
//         if (!isFinite(v))
//           continue;
//         sum += v;
//         cnt++;
//       }
//       return (cnt > 0) ? static_cast<float>(sum / cnt)
//                        : std::numeric_limits<float>::quiet_NaN();
//     }
//     else if (type == PoolType::MIN)
//     {
//       float best = std::numeric_limits<float>::infinity();
//       bool has = false;
//       for (; !it.isPastEnd(); ++it)
//       {
//         const grid_map::Index idx = *it;
//         const float v = height_map_.at(layer, idx);
//         if (!isFinite(v))
//           continue;
//         best = std::min(best, v);
//         has = true;
//       }
//       return has ? best : std::numeric_limits<float>::quiet_NaN();
//     }
//     else // MAX
//     {
//       float best = -std::numeric_limits<float>::infinity();
//       bool has = false;
//       for (; !it.isPastEnd(); ++it)
//       {
//         const grid_map::Index idx = *it;
//         const float v = height_map_.at(layer, idx);
//         if (!isFinite(v))
//           continue;
//         best = std::max(best, v);
//         has = true;
//       }
//       return has ? best : std::numeric_limits<float>::quiet_NaN();
//     }
//   };

//   const grid_map::Length cell_window(out_res, out_res);

//   for (grid_map::GridMapIterator it(out_map); !it.isPastEnd(); ++it)
//   {
//     const grid_map::Index idx_out(*it);

//     // coarse cell 中心点坐标
//     grid_map::Position cell_center;
//     out_map.getPosition(idx_out, cell_center);

//     // 如果高分辨率图不覆盖该区域，保持 NaN
//     if (!height_map_.isInside(cell_center))
//       continue;

//     bool ok = false;
//     grid_map::SubmapGeometry subgeom(height_map_, cell_center, cell_window, ok);
//     if (!ok)
//       continue;

//     for (const auto &p : used)
//     {
//       out_map.at(p.name, idx_out) = poolLayerOnSubmap(p.name, subgeom, p.type);
//     }
//   }

//   // ==========================================================
//   // [5] 转 ROS 消息
//   // ==========================================================
//   grid_map::GridMapRosConverter::toMessage(out_map, message);
//   return true;
// }

bool Mapping::buildGridMapFromViewpointCloud(grid_map_msgs::GridMap &message)
{
  // ==========================================================
  // [0] 拿最新缓存点云（只用 x,y）
  // ==========================================================
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_copy(new pcl::PointCloud<pcl::PointXYZI>());
  ros::Time cloud_stamp;
  std::string cloud_frame;

  {
    if (!has_viewpoint_vis_.load(std::memory_order_acquire))
    {
      ROS_WARN_THROTTLE(2.0, "No viewpoint_vis_cloud yet, skip grid map publish.");
      return false;
    }

    std::lock_guard<std::mutex> lk(viewpoint_vis_mutex_);
    *cloud_copy = *latest_viewpoint_vis_cloud_;
    cloud_stamp = latest_viewpoint_vis_stamp_;
    cloud_frame = latest_viewpoint_vis_frame_id_;
  }

  if (cloud_copy->empty())
  {
    ROS_WARN_THROTTLE(2.0, "Latest viewpoint_vis_cloud is empty, skip grid map publish.");
    return false;
  }

  // ==========================================================
  // [1] 加锁保护 height_map_（读）
  // ==========================================================
  std::lock_guard<std::mutex> lock(gridmap_publish_mutex_);

  // ==========================================================
  // [2] 输出分辨率：仍然用 viewpoint_resolution_x_（保持不变）
  // ==========================================================
  const double vp_res_x = viewpoint_resolution_x_;
  const double vp_res_y = viewpoint_resolution_y_;

  double out_res = vp_res_x;
  if (std::fabs(vp_res_x - vp_res_y) > 1e-9)
  {
    ROS_WARN_THROTTLE(2.0,
                      "ViewPoint res x!=y (%.4f vs %.4f). grid_map uses single res, using res_x.",
                      vp_res_x, vp_res_y);
  }
  if (out_res <= 0.0)
  {
    ROS_WARN_THROTTLE(2.0, "Invalid out_res=%.6f", out_res);
    return false;
  }

  // ==========================================================
  // [3] 用点云 (x,y) 决定输出几何 bbox（仅用于 setGeometry）
  //     注意：后面不会填 bbox 内所有格子，只填点命中的格子
  // ==========================================================
  double min_x = std::numeric_limits<double>::infinity();
  double min_y = std::numeric_limits<double>::infinity();
  double max_x = -std::numeric_limits<double>::infinity();
  double max_y = -std::numeric_limits<double>::infinity();

  int valid_xy_cnt = 0;
  for (const auto &p : cloud_copy->points)
  {
    if (!std::isfinite(p.x) || !std::isfinite(p.y))
      continue;

    min_x = std::min(min_x, static_cast<double>(p.x));
    min_y = std::min(min_y, static_cast<double>(p.y));
    max_x = std::max(max_x, static_cast<double>(p.x));
    max_y = std::max(max_y, static_cast<double>(p.y));
    valid_xy_cnt++;
  }

  if (valid_xy_cnt == 0)
  {
    ROS_WARN_THROTTLE(2.0, "No valid (x,y) points in viewpoint_vis_cloud, skip publish.");
    return false;
  }

  // bbox 对齐到栅格线
  auto align_down = [&](double v)
  { return std::floor(v / out_res) * out_res; };
  auto align_up = [&](double v)
  { return std::ceil(v / out_res) * out_res; };

  // 给 bbox 加一圈 margin（1格）
  const double margin = out_res;

  const double aligned_min_x = align_down(min_x - margin);
  const double aligned_min_y = align_down(min_y - margin);
  const double aligned_max_x = align_up(max_x + margin);
  const double aligned_max_y = align_up(max_y + margin);

  const double size_x = std::max(out_res, aligned_max_x - aligned_min_x);
  const double size_y = std::max(out_res, aligned_max_y - aligned_min_y);

  const grid_map::Position out_center(aligned_min_x + 0.5 * size_x,
                                      aligned_min_y + 0.5 * size_y);

  grid_map::GridMap out_map;
  out_map.setFrameId(height_map_.getFrameId().empty() ? cloud_frame : height_map_.getFrameId());
  out_map.setTimestamp(cloud_stamp.toNSec());
  out_map.setGeometry(grid_map::Length(size_x, size_y), out_res, out_center);

  // ==========================================================
  // [4] 层与 pooling 策略（保持不变）
  // ==========================================================
  enum class PoolType
  {
    MIN,
    MAX,
    MEAN
  };

  struct LayerPolicy
  {
    std::string name;
    PoolType type;
  };

  const std::vector<LayerPolicy> policies = {
      {"elevation", PoolType::MEAN},
      {"elevation_BGK", PoolType::MEAN},
      {"slope", PoolType::MEAN},
      {"roughness", PoolType::MEAN},
      {"step", PoolType::MAX},
      {"traversability", PoolType::MAX},
      {"traversability_fine_wheeled", PoolType::MAX},
      {"traversability_fine_tracked", PoolType::MAX},
      {"critical", PoolType::MAX},
  };

  std::vector<LayerPolicy> used;
  used.reserve(policies.size());
  for (const auto &p : policies)
  {
    if (height_map_.exists(p.name))
    {
      out_map.add(p.name, std::numeric_limits<float>::quiet_NaN());
      used.push_back(p);
    }
  }

  if (used.empty())
  {
    ROS_WARN_THROTTLE(2.0, "None of publish layers exist in height_map_.");
    return false;
  }

  // ==========================================================
  // [5] pooling（保持不变）
  // ==========================================================
  auto isFinite = [](float v) -> bool
  { return std::isfinite(v); };

  auto poolLayerOnSubmap = [&](const std::string &layer,
                               const grid_map::SubmapGeometry &subgeom,
                               PoolType type) -> float
  {
    grid_map::SubmapIterator it(subgeom);

    if (type == PoolType::MEAN)
    {
      double sum = 0.0;
      int cnt = 0;
      for (; !it.isPastEnd(); ++it)
      {
        const grid_map::Index idx = *it;
        const float v = height_map_.at(layer, idx);
        if (!isFinite(v))
          continue;
        sum += v;
        cnt++;
      }
      return (cnt > 0) ? static_cast<float>(sum / cnt)
                       : std::numeric_limits<float>::quiet_NaN();
    }
    else if (type == PoolType::MIN)
    {
      float best = std::numeric_limits<float>::infinity();
      bool has = false;
      for (; !it.isPastEnd(); ++it)
      {
        const grid_map::Index idx = *it;
        const float v = height_map_.at(layer, idx);
        if (!isFinite(v))
          continue;
        best = std::min(best, v);
        has = true;
      }
      return has ? best : std::numeric_limits<float>::quiet_NaN();
    }
    else // MAX
    {
      float best = -std::numeric_limits<float>::infinity();
      bool has = false;
      for (; !it.isPastEnd(); ++it)
      {
        const grid_map::Index idx = *it;
        const float v = height_map_.at(layer, idx);
        if (!isFinite(v))
          continue;
        best = std::max(best, v);
        has = true;
      }
      return has ? best : std::numeric_limits<float>::quiet_NaN();
    }
  };

  const grid_map::Length cell_window(out_res, out_res);

  // ==========================================================
  // [6] 关键变化：只填 “点命中的 cell”
  // ==========================================================
  std::unordered_set<uint64_t> visited;
  visited.reserve(static_cast<size_t>(cloud_copy->size()));

  auto packIndex = [](const grid_map::Index &idx) -> uint64_t
  {
    return (static_cast<uint64_t>(static_cast<uint32_t>(idx.x())) << 32) |
           (static_cast<uint64_t>(static_cast<uint32_t>(idx.y())));
  };

  int filled_cells = 0;

  for (const auto &p : cloud_copy->points)
  {
    if (!std::isfinite(p.x) || !std::isfinite(p.y))
      continue;

    const grid_map::Position pos(static_cast<double>(p.x), static_cast<double>(p.y));

    if (!out_map.isInside(pos))
      continue;

    grid_map::Index idx_out;
    if (!out_map.getIndex(pos, idx_out))
      continue;

    const uint64_t key = packIndex(idx_out);
    if (!visited.insert(key).second)
      continue; // 已经填过这个 cell

    // 用 cell 中心点在 height_map_ 上做 pooling（保持原逻辑）
    grid_map::Position cell_center;
    out_map.getPosition(idx_out, cell_center);

    if (!height_map_.isInside(cell_center))
      continue;

    bool ok = false;
    grid_map::SubmapGeometry subgeom(height_map_, cell_center, cell_window, ok);
    if (!ok)
      continue;

    for (const auto &pol : used)
    {
      out_map.at(pol.name, idx_out) = poolLayerOnSubmap(pol.name, subgeom, pol.type);
    }

    filled_cells++;
  }

  ROS_INFO_STREAM_THROTTLE(1.0,
                           "GridMap filled cells by points: " << filled_cells
                                                              << " / unique cells=" << visited.size());

  // ==========================================================
  // [7] 转 ROS 消息
  // ==========================================================
  grid_map::GridMapRosConverter::toMessage(out_map, message);
  return true;
}

// ==================== Fixed-Rate Publishing ====================
void Mapping::publishTimerCallback(const ros::TimerEvent &event)
{
  // Publish grid map at fixed frequency
  publishGridMap();
}
