/*
 * @Author: lee lizw_0304@163.com
 * @Date: 2026-07-15 21:21:31
 * @LastEditors: lee lizw_0304@163.com
 * @LastEditTime: 2026-08-04 15:59:09
 * @FilePath: /src/obstacle_mapping/src/multi_robot_mapping.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
/**
 * @description: Multi-robot mapping node with conditional startup blind-spot completion
 * @filename: multi_robot_mapping.cpp
 * @author: wangxurui
 * @date: 2026-01-28
 **/

#include "multi_robot_mapping.h"
#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <queue>
#include <algorithm>
#include <limits>
#include <array>
#include <vector>
#include <set>
#include <utility>
#include <stdexcept>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <functional>
#include <cstdint>
#include <mutex>
#include <thread>
#include <condition_variable>

namespace
{
  constexpr int kPoseFailure = 0;
  constexpr int kPoseCollision = 1;
  constexpr int kPoseSuccess = 2;
  constexpr int kPoseUnstable = 3;

  enum class StartupHoleFillResult
  {
    kNoHole,
    kFilled,
    kNotClosed,
    kTooLarge,
    kInsufficientBoundary,
    kInvalidMap
  };

  // predictRobotPose() 会对同一格子按“车型 x 航向”连续调用。
  // 以精细化周期编号作为缓存世代，保证同一周期内复用地形
  // 子图，下一周期地图更新后不会沿用过期缓存。
  thread_local std::size_t fine_cycle_generation = 0;

  // Local XYZI terrain-map configuration for the original local_planner.
  // The planner subscribes to /terrain_map and classifies a point as an
  // obstacle when intensity > 0.2.  We therefore publish exactly two values:
  // 0.2 for traversable terrain and 1.0 for non-traversable terrain.
  bool publish_local_trav_cloud = true;
  std::string local_trav_vehicle_type = "wheeled";
  std::string local_trav_layer = "traversability_fine_wheeled";
  std::string local_trav_cloud_topic = "/terrain_map";
  double terrain_map_cost_threshold = 0.98;
  // Preserve the original FitPlane / local_planner Z convention.  After the
  // planner subtracts vehicleZ, traversable points are at -0.1 m and obstacle
  // points are at +0.6 m.
  double terrain_map_traversable_z_offset = -0.1;
  double terrain_map_obstacle_z_offset = 0.6;
  // FitPlane compatibility: its unknown occupancy value (-1) is published as
  // a traversable point.  Keep this configurable because treating unobserved
  // terrain as free is less conservative than omitting it.
  bool terrain_map_unknown_as_traversable = true;
  ros::Publisher local_trav_cloud_publisher;

  // Height cloud aligned one-to-one with ViewPointManager's rolling XY grid.
  // Keep the original absolute /terrain_map publisher above unchanged.  This
  // relative topic resolves inside each vehicle namespace, e.g.
  // /vehicle0/terrain_map, and is used to provide terrain height to viewpoints.
  bool publish_viewpoint_aligned_terrain_cloud = true;
  std::string viewpoint_aligned_terrain_cloud_topic = "terrain_map_ext";
  std::string viewpoint_aligned_height_layer = "elevation_BGK";
  ros::Publisher viewpoint_aligned_terrain_cloud_publisher;

  /**
   * Geometry shared with tare_planner's ViewPointManager.
   *
   * The mapping node intentionally does not subscribe to viewpoint_origin.
   * Instead, it loads the same parameters and applies the same initialization
   * and rollover equations to the odometry sample synchronized with the scan.
   */
  struct ViewpointGridConfig
  {
    int number_x = 0;
    int number_y = 0;
    int number_z = 0;
    double resolution_x = 0.0;
    double resolution_y = 0.0;
    double resolution_z = 0.0;
    int rollover_cells_x = 0;
    int rollover_cells_y = 0;
    int grid_world_x_num = 0;
    int grid_world_y_num = 0;
    int nearby_grid_num = 0;
    double grid_world_cell_size = 0.0;
  };

  struct ViewpointGridSnapshot
  {
    ViewpointGridConfig config;
    Eigen::Vector3d origin = Eigen::Vector3d::Zero();
    Eigen::Vector3d robot_position = Eigen::Vector3d::Zero();
    ros::Time stamp;
    std::size_t generation = 0;
  };

  static inline double QuantizeViewpointOrigin(double value)
  {
    // Keep bit-for-bit the same decimal-grid rule as ViewPointManager::Quantize01.
    return static_cast<double>(std::llround(value * 10.0)) / 10.0;
  }

  class ViewpointGridAlignment
  {
  public:
    void configure(const ViewpointGridConfig &config)
    {
      if (config.number_x <= 0 || config.number_y <= 0 ||
          config.resolution_x <= 0.0 || config.resolution_y <= 0.0 ||
          config.rollover_cells_x <= 0 || config.rollover_cells_y <= 0 ||
          config.grid_world_x_num <= 0 || config.grid_world_y_num <= 0 ||
          config.nearby_grid_num <= 0 || config.grid_world_cell_size <= 0.0)
      {
        throw std::runtime_error("Invalid viewpoint/grid_world geometry parameters");
      }

      if (std::fabs(config.resolution_x - config.resolution_y) > 1e-9)
      {
        throw std::runtime_error(
            "grid_map requires viewpoint resolution_x == resolution_y");
      }

      const double rollover_distance_y =
          static_cast<double>(config.rollover_cells_y) * config.resolution_y;
      if (std::fabs(rollover_distance_y - config.grid_world_cell_size) > 1e-9)
      {
        throw std::runtime_error(
            "Viewpoint X/Y physical rollover distances must equal GridWorld cell size");
      }

      std::lock_guard<std::mutex> lock(mutex_);
      config_ = config;
      configured_ = true;
      initialized_ = false;
      generation_ = 0;
      origin_.setZero();
      robot_position_.setZero();
      stamp_ = ros::Time(0);
    }

    bool update(const Eigen::Vector3d &robot_position, const ros::Time &stamp)
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!configured_)
      {
        return false;
      }

      robot_position_ = robot_position;
      stamp_ = stamp;

      bool rolled = false;
      if (!initialized_)
      {
        initializeOrigin(robot_position);
        initialized_ = true;
        rolled = true;

        ROS_INFO("Inferred initial viewpoint origin: [%.3f, %.3f], robot=[%.3f, %.3f]",
                 origin_.x(), origin_.y(), robot_position.x(), robot_position.y());
      }

      const Eigen::Vector2d old_origin(origin_.x(), origin_.y());
      updateAxis(robot_position.x(), config_.number_x,
                 config_.resolution_x, config_.rollover_cells_x, origin_.x());
      updateAxis(robot_position.y(), config_.number_y,
                 config_.resolution_y, config_.rollover_cells_y, origin_.y());

      if (std::fabs(origin_.x() - old_origin.x()) > 1e-9 ||
          std::fabs(origin_.y() - old_origin.y()) > 1e-9)
      {
        rolled = true;
        ROS_INFO("Inferred viewpoint rollover: origin [%.3f, %.3f] -> [%.3f, %.3f], robot=[%.3f, %.3f]",
                 old_origin.x(), old_origin.y(), origin_.x(), origin_.y(),
                 robot_position.x(), robot_position.y());
      }

      if (rolled)
      {
        ++generation_;
      }
      return rolled;
    }

    bool snapshot(ViewpointGridSnapshot &snapshot) const
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!configured_ || !initialized_)
      {
        return false;
      }

      snapshot.config = config_;
      snapshot.origin = origin_;
      snapshot.robot_position = robot_position_;
      snapshot.stamp = stamp_;
      snapshot.generation = generation_;
      return true;
    }

  private:
    void initializeOrigin(const Eigen::Vector3d &robot_position)
    {
      // Same GridWorld origin used in GridWorld::UpdateRobotPosition().
      const double grid_origin_x =
          -config_.grid_world_cell_size * config_.grid_world_x_num / 2.0;
      const double grid_origin_y =
          -config_.grid_world_cell_size * config_.grid_world_y_num / 2.0;

      const int robot_sub_x = static_cast<int>(
          std::floor((robot_position.x() - grid_origin_x) /
                     config_.grid_world_cell_size));
      const int robot_sub_y = static_cast<int>(
          std::floor((robot_position.y() - grid_origin_y) /
                     config_.grid_world_cell_size));

      const int neighbor_half = config_.nearby_grid_num / 2;
      int start_sub_x = robot_sub_x - neighbor_half;
      int start_sub_y = robot_sub_y - neighbor_half;

      // Match GridWorld::UpdateNeighborCells(): clamp the start cell itself.
      start_sub_x = std::max(
          0, std::min(start_sub_x, config_.grid_world_x_num - 1));
      start_sub_y = std::max(
          0, std::min(start_sub_y, config_.grid_world_y_num - 1));

      origin_.x() = QuantizeViewpointOrigin(
          grid_origin_x + start_sub_x * config_.grid_world_cell_size);
      origin_.y() = QuantizeViewpointOrigin(
          grid_origin_y + start_sub_y * config_.grid_world_cell_size);
      origin_.z() = 0.0; // ViewPointManager is configured as dimension_=2.
    }

    static void updateAxis(double robot_coordinate,
                           int number,
                           double resolution,
                           int rollover_cells,
                           double &origin_coordinate)
    {
      // Exact 2-D counterpart of ViewPointManager::UpdateRobotPosition().
      const double diff = robot_coordinate - origin_coordinate;
      const double rollover_distance =
          static_cast<double>(rollover_cells) * resolution;
      const int robot_grid_sub =
          diff > 0.0 ? static_cast<int>(diff / rollover_distance) : -1;
      const int target_grid_sub = (number / rollover_cells) / 2;
      const int sub_diff = target_grid_sub - robot_grid_sub;
      const int rollover_step = rollover_cells * sub_diff;

      origin_coordinate -= static_cast<double>(rollover_step) * resolution;
      origin_coordinate = QuantizeViewpointOrigin(origin_coordinate);
    }

    mutable std::mutex mutex_;
    ViewpointGridConfig config_;
    bool configured_ = false;
    bool initialized_ = false;
    Eigen::Vector3d origin_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d robot_position_ = Eigen::Vector3d::Zero();
    ros::Time stamp_;
    std::size_t generation_ = 0;
  };

  ViewpointGridAlignment &viewpointGridAlignment()
  {
    static ViewpointGridAlignment alignment;
    return alignment;
  }

  void ConfigureViewpointGridAlignment(ros::NodeHandle &nh)
  {
    ViewpointGridConfig config;
    std::string planner_prefix;
    std::string viewpoint_prefix;

    // 与参考代码加载 viewpoint_manager 参数的方式保持一致：
    // 由当前机器人命名空间拼出 tare_planner_node 的私有参数路径，
    // 并在规划节点完成参数初始化之前以 10 Hz 持续等待。
    ros::Rate rate(10);
    bool loaded = false;
    while (ros::ok())
    {
      const std::string ns = ros::this_node::getNamespace();
      planner_prefix = ns + "/tare_planner_node/";
      viewpoint_prefix = planner_prefix + "viewpoint_manager/";

      loaded = nh.getParam(viewpoint_prefix + "number_x", config.number_x) &&
               nh.getParam(viewpoint_prefix + "number_y", config.number_y) &&
               nh.getParam(viewpoint_prefix + "number_z", config.number_z) &&
               nh.getParam(viewpoint_prefix + "resolution_x", config.resolution_x) &&
               nh.getParam(viewpoint_prefix + "resolution_y", config.resolution_y) &&
               nh.getParam(viewpoint_prefix + "resolution_z", config.resolution_z);

      if (loaded)
        break;

      ROS_WARN_THROTTLE(
          2.0,
          "Waiting for viewpoint_manager params... "
          "(number_x/y/z, resolution_x/y/z)");
      rate.sleep();
    }

    // 与参考代码一致，ros::ok() 变 false 后不使用未初始化参数。
    if (!ros::ok())
      return;

    // 与 GridWorld::ReadParameters() 相同：这些值从 ROS 参数服务器
    // 读取，并保留 GridWorld 源码中的缺省值 121、121、5。
    nh.param<int>(planner_prefix + "kGridWorldXNum",
                  config.grid_world_x_num, 121);
    nh.param<int>(planner_prefix + "kGridWorldYNum",
                  config.grid_world_y_num, 121);
    nh.param<int>(planner_prefix + "kGridWorldNearbyGridNum",
                  config.nearby_grid_num, 5);

    // These two values are derived, not ROS parameters, in the original
    // ViewPointManager::ReadParameters() and GridWorld::ReadParameters().
    config.rollover_cells_x = config.number_x / 5;
    config.rollover_cells_y = config.number_y / 5;
    // Same formula as GridWorld::ReadParameters().
    config.grid_world_cell_size =
        static_cast<double>(config.number_x) * config.resolution_x / 5.0;

    viewpointGridAlignment().configure(config);

    ROS_INFO("Viewpoint-aligned trav_map: params=%s, cells=%dx%d, resolution=%.3f, size=%.3fx%.3f m, rollover=%dx%d cells (%.3fx%.3f m), GridWorld=%dx%d nearby=%d",
             viewpoint_prefix.c_str(), config.number_x, config.number_y,
             config.resolution_x,
             config.number_x * config.resolution_x,
             config.number_y * config.resolution_y,
             config.rollover_cells_x, config.rollover_cells_y,
             config.rollover_cells_x * config.resolution_x,
             config.rollover_cells_y * config.resolution_y,
             config.grid_world_x_num, config.grid_world_y_num,
             config.nearby_grid_num);
  }

  enum class ViewpointPoolType
  {
    kMax,
    kMean
  };

  struct ViewpointLayerPolicy
  {
    const char *name;
    ViewpointPoolType pool_type;
  };

  /**
   * Publish the already-resampled output map as an XYZI cloud for viewpoint
   * height assignment.  Each published point uses the output grid cell center
   * as its real map-frame x/y and elevation_BGK as z.
   */
  void PublishViewpointAlignedTerrainCloud(
      const grid_map::GridMap &output_map,
      const ViewpointGridSnapshot &snapshot)
  {
    if (!publish_viewpoint_aligned_terrain_cloud)
    {
      return;
    }

    if (!output_map.exists(viewpoint_aligned_height_layer))
    {
      ROS_WARN_THROTTLE(
          2.0,
          "Cannot publish viewpoint-aligned terrain cloud: height layer missing (%s)",
          viewpoint_aligned_height_layer.c_str());
      return;
    }

    const bool has_traversability = output_map.exists(local_trav_layer);
    if (!has_traversability)
    {
      ROS_WARN_THROTTLE(
          2.0,
          "Viewpoint-aligned terrain cloud has no %s layer; intensity defaults to 0.2",
          local_trav_layer.c_str());
    }

    pcl::PointCloud<pcl::PointXYZI> terrain_cloud;
    const grid_map::Size output_size = output_map.getSize();
    terrain_cloud.points.reserve(
        static_cast<std::size_t>(output_size(0)) *
        static_cast<std::size_t>(output_size(1)));

    std::size_t invalid_height_count = 0;
    std::size_t traversable_count = 0;
    std::size_t obstacle_count = 0;

    for (grid_map::GridMapIterator it(output_map); !it.isPastEnd(); ++it)
    {
      const float height = output_map.at(viewpoint_aligned_height_layer, *it);
      if (!std::isfinite(height))
      {
        ++invalid_height_count;
        continue;
      }

      grid_map::Position position;
      if (!output_map.getPosition(*it, position))
      {
        continue;
      }

      bool is_obstacle = false;
      if (has_traversability)
      {
        const float traversability = output_map.at(local_trav_layer, *it);
        is_obstacle = std::isfinite(traversability) &&
                      traversability >
                          static_cast<float>(terrain_map_cost_threshold);
      }

      pcl::PointXYZI point;
      point.x = static_cast<float>(position.x());
      point.y = static_cast<float>(position.y());
      point.z = height;
      point.intensity = is_obstacle ? 1.0f : 0.2f;
      terrain_cloud.points.push_back(point);

      if (is_obstacle)
      {
        ++obstacle_count;
      }
      else
      {
        ++traversable_count;
      }
    }

    terrain_cloud.width =
        static_cast<std::uint32_t>(terrain_cloud.points.size());
    terrain_cloud.height = 1;
    terrain_cloud.is_dense = true;

    sensor_msgs::PointCloud2 terrain_message;
    pcl::toROSMsg(terrain_cloud, terrain_message);
    terrain_message.header.frame_id = output_map.getFrameId();
    terrain_message.header.stamp = snapshot.stamp.isZero()
                                       ? ros::Time::now()
                                       : snapshot.stamp;
    viewpoint_aligned_terrain_cloud_publisher.publish(terrain_message);

    ROS_INFO_THROTTLE(
        1.0,
        "Published viewpoint-aligned terrain cloud %s: grid=%dx%d, valid_height=%zu, invalid_height=%zu, traversable=%zu, obstacle=%zu",
        viewpoint_aligned_terrain_cloud_publisher.getTopic().c_str(),
        output_size(0), output_size(1), terrain_cloud.points.size(),
        invalid_height_count, traversable_count, obstacle_count);
  }

  /**
   * Downsample the persistent high-resolution map into a GridMap whose lower
   * corner, cell count and resolution are identical to ViewPointManager.
   * The caller owns the height_map read lock.
   */
  bool BuildViewpointAlignedGridMap(const grid_map::GridMap &height_map,
                                    const std::string &fallback_frame_id,
                                    grid_map_msgs::GridMap &message)
  {
    ViewpointGridSnapshot snapshot;
    if (!viewpointGridAlignment().snapshot(snapshot))
    {
      ROS_WARN_THROTTLE(
          2.0, "Viewpoint geometry has no synchronized pose yet; skip aligned trav_map.");
      return false;
    }

    const ViewpointGridConfig &config = snapshot.config;
    const double output_resolution = config.resolution_x;
    const double size_x = static_cast<double>(config.number_x) * config.resolution_x;
    const double size_y = static_cast<double>(config.number_y) * config.resolution_y;
    const grid_map::Position output_center(snapshot.origin.x() + 0.5 * size_x,
                                           snapshot.origin.y() + 0.5 * size_y);

    grid_map::GridMap output_map;
    output_map.setFrameId(height_map.getFrameId().empty() ? fallback_frame_id : height_map.getFrameId());
    output_map.setTimestamp(snapshot.stamp.toNSec());
    output_map.setGeometry(grid_map::Length(size_x, size_y),
                           output_resolution, output_center);

    const grid_map::Size output_size = output_map.getSize();
    if (output_size(0) != config.number_x ||
        output_size(1) != config.number_y)
    {
      ROS_ERROR_THROTTLE(2.0,
                         "Viewpoint-aligned grid size mismatch: expected=%dx%d actual=%dx%d",
                         config.number_x, config.number_y, output_size(0), output_size(1));
      return false;
    }

    static const std::array<ViewpointLayerPolicy, 14> policies = {{
        {"elevation", ViewpointPoolType::kMean},
        {"elevation_BGK", ViewpointPoolType::kMean},
        {"slope", ViewpointPoolType::kMean},
        {"roughness", ViewpointPoolType::kMean},
        {"step", ViewpointPoolType::kMax},
        {"slope_deg", ViewpointPoolType::kMax},
        {"roughness_raw", ViewpointPoolType::kMax},
        {"step_height", ViewpointPoolType::kMax},
        {"traversability", ViewpointPoolType::kMax},
        {"traversability_coarse_wheeled", ViewpointPoolType::kMax},
        {"traversability_coarse_tracked", ViewpointPoolType::kMax},
        {"traversability_fine_wheeled", ViewpointPoolType::kMax},
        {"traversability_fine_tracked", ViewpointPoolType::kMax},
        {"critical", ViewpointPoolType::kMax},
    }};

    std::vector<ViewpointLayerPolicy> used_policies;
    used_policies.reserve(policies.size());
    for (const auto &policy : policies)
    {
      if (height_map.exists(policy.name))
      {
        output_map.add(policy.name, std::numeric_limits<float>::quiet_NaN());

        used_policies.push_back(policy);
      }
    }

    if (used_policies.empty())
    {
      ROS_WARN_THROTTLE(2.0, "No traversability-related layer exists for aligned trav_map.");
      return false;
    }

    const grid_map::Length source_window(output_resolution,
                                         output_resolution);
    std::size_t sampled_cells = 0;

    for (grid_map::GridMapIterator output_it(output_map);
         !output_it.isPastEnd(); ++output_it)
    {
      const grid_map::Index output_index(*output_it);
      grid_map::Position cell_center;
      if (!output_map.getPosition(output_index, cell_center) ||
          !height_map.isInside(cell_center))
      {
        continue;
      }

      bool submap_ok = false;
      grid_map::SubmapGeometry source_geometry(
          height_map, cell_center, source_window, submap_ok);
      if (!submap_ok)
      {
        continue;
      }

      for (const auto &policy : used_policies)
      {
        double sum = 0.0;
        std::size_t valid_count = 0;
        float maximum = -std::numeric_limits<float>::infinity();

        for (grid_map::SubmapIterator source_it(source_geometry);
             !source_it.isPastEnd(); ++source_it)
        {
          const float value = height_map.at(policy.name, *source_it);
          if (!std::isfinite(value))
          {
            continue;
          }

          sum += static_cast<double>(value);
          maximum = std::max(maximum, value);
          ++valid_count;
        }

        if (valid_count == 0)
        {
          continue;
        }

        output_map.at(policy.name, output_index) =
            policy.pool_type == ViewpointPoolType::kMax
                ? maximum
                : static_cast<float>(sum / static_cast<double>(valid_count));
      }

      ++sampled_cells;
    }

    grid_map::GridMapRosConverter::toMessage(output_map, message);
    PublishViewpointAlignedTerrainCloud(output_map, snapshot);
    ROS_INFO_STREAM_THROTTLE(1.0, "Published viewpoint-aligned trav_map: origin=["
                                      << snapshot.origin.x() << ", " << snapshot.origin.y()
                                      << "], size=" << config.number_x << "x" << config.number_y
                                      << ", resolution=" << output_resolution
                                      << ", generation=" << snapshot.generation
                                      << ", sampled_cells=" << sampled_cells);
    return true;
  }

  // Parameters shared by step-height extraction and platform-specific
  // coarse traversability.  Keeping them in this translation unit avoids
  // adding class members solely for this feature.
  struct StepEvaluationConfig
  {
    // Quantile of plane-compensated neighbor residuals used as step height.
    double robust_quantile = 0.90;
    // Minimum number of valid neighboring cells required for a result.
    int min_valid_neighbors = 3;
    // Measurement/interpolation tolerance before a platform step limit is
    // turned into a hard non-traversable decision.
    double noise_margin_m = 0.05;
  };

  const StepEvaluationConfig &stepEvaluationConfig(ros::NodeHandle &pnh)
  {
    static StepEvaluationConfig config;
    static bool configured = false;
    if (!configured)
    {
      pnh.param<double>("step_robust_quantile",
                        config.robust_quantile, 0.90);
      pnh.param<int>("step_min_valid_neighbors",
                     config.min_valid_neighbors, 3);
      pnh.param<double>("step_noise_margin_m",
                        config.noise_margin_m, 0.05);

      config.robust_quantile =
          std::max(0.50, std::min(1.0, config.robust_quantile));
      config.min_valid_neighbors =
          std::max(1, config.min_valid_neighbors);
      config.noise_margin_m =
          std::max(0.0, config.noise_margin_m);
      configured = true;

      ROS_INFO("Step evaluation: quantile=%.2f, min_neighbors=%d, noise_margin=%.3f m",
               config.robust_quantile,
               config.min_valid_neighbors,
               config.noise_margin_m);
    }
    return config;
  }

  /**
   * @brief Single-slot background executor used by global_trav_map.
   *
   * At most one task can be running and one latest task can be pending.  A new
   * request replaces the pending one instead of extending a FIFO queue.  This
   * keeps global publication best-effort and prevents it from accumulating
   * latency behind the real-time local map pipeline.
   */
  class LatestGlobalPublishWorker
  {
  public:
    LatestGlobalPublishWorker()
        : stop_requested_(false), pending_(false), running_(false),
          worker_(&LatestGlobalPublishWorker::run, this)
    {
    }

    ~LatestGlobalPublishWorker()
    {
      shutdown();
    }

    void requestLatest(std::function<void()> task)
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stop_requested_)
      {
        return;
      }

      // Assignment intentionally replaces an older task that has not started.
      pending_task_ = std::move(task);
      pending_ = true;
      cv_.notify_one();
    }

    void shutdown()
    {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stop_requested_)
        {
          return;
        }
        stop_requested_ = true;
        pending_ = false;
        pending_task_ = std::function<void()>();
      }
      cv_.notify_all();

      if (worker_.joinable())
      {
        worker_.join();
      }
    }

    bool hasPendingTask()
    {
      std::lock_guard<std::mutex> lock(mutex_);
      return pending_;
    }

  private:
    void run()
    {
      while (true)
      {
        std::function<void()> task;
        {
          std::unique_lock<std::mutex> lock(mutex_);
          cv_.wait(lock, [this]()
                   { return stop_requested_ || pending_; });

          if (stop_requested_)
          {
            break;
          }

          task = std::move(pending_task_);
          pending_task_ = std::function<void()>();
          pending_ = false;
          running_ = true;
        }

        try
        {
          if (task)
          {
            task();
          }
        }
        catch (const std::exception &e)
        {
          ROS_ERROR("Asynchronous global map publication failed: %s",
                    e.what());
        }
        catch (...)
        {
          ROS_ERROR("Asynchronous global map publication failed with an unknown exception");
        }

        {
          std::lock_guard<std::mutex> lock(mutex_);
          running_ = false;
        }
      }
    }

    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_requested_;
    bool pending_;
    bool running_;
    std::function<void()> pending_task_;
    std::thread worker_;
  };

  LatestGlobalPublishWorker &globalPublishWorker()
  {
    static LatestGlobalPublishWorker worker;
    return worker;
  }

  /**
   * @brief 补全初始位置附近的激光雷达近场盲区。
   *
   * 不再要求 unknown 区域构成严格封闭连通域。只在 fill_radius
   * 范围内查找无效 elevation_BGK 栅格，并使用盲区边缘附近的
   * 有效地形高度拟合局部平面。
   *
   * 仅修改 elevation_BGK 和 interpolated，n_points 仍然保持为0，
   * 表示这些高度属于推断结果，而不是真实激光观测。
   */
  StartupHoleFillResult fillStartupBlindArea(
      grid_map::GridMap &map,
      const grid_map::Position &initial_position,
      double fill_radius,
      double boundary_width,
      std::size_t max_fill_cells,
      std::size_t min_boundary_cells,
      double recompute_radius,
      std::size_t &filled_cell_count)
  {
    filled_cell_count = 0;

    if (!map.exists("elevation_BGK") ||
        !map.exists("interpolated") ||
        fill_radius <= map.getResolution() ||
        boundary_width < map.getResolution())
    {
      return StartupHoleFillResult::kInvalidMap;
    }

    grid_map::Index initial_index;
    if (!map.getIndex(initial_position, initial_index))
    {
      return StartupHoleFillResult::kInvalidMap;
    }

    auto &elevation_bgk = map["elevation_BGK"];
    auto &interpolated = map["interpolated"];

    // 需要补全的范围。
    const double boundary_inner_radius =
        std::max(0.0, fill_radius - boundary_width);

    const double boundary_outer_radius =
        fill_radius + boundary_width;

    // 只保存索引，避免依赖unknown连通性。
    std::vector<std::pair<int, int>> fill_cells;
    std::vector<std::pair<int, int>> boundary_cells;

    // 同时扫描盲区和边界带。
    for (grid_map::CircleIterator it(
             map, initial_position, boundary_outer_radius);
         !it.isPastEnd(); ++it)
    {
      const grid_map::Index index = *it;

      grid_map::Position position;
      if (!map.getPosition(index, position))
      {
        continue;
      }

      const double distance =
          (position - initial_position).norm();

      const float height =
          elevation_bgk(index(0), index(1));

      // 只补填fill_radius内部的NaN。
      if (distance <= fill_radius &&
          !std::isfinite(height))
      {
        fill_cells.emplace_back(index(0), index(1));
      }

      // 从盲区边缘内外一定宽度内收集有效高度。
      if (distance >= boundary_inner_radius &&
          distance <= boundary_outer_radius &&
          std::isfinite(height))
      {
        boundary_cells.emplace_back(index(0), index(1));
      }
    }

    ROS_INFO_THROTTLE(
        1.0,
        "Startup blind-area candidate: fill_cells=%zu, "
        "boundary_cells=%zu, radius=%.2f, boundary_width=%.2f",
        fill_cells.size(),
        boundary_cells.size(),
        fill_radius,
        boundary_width);

    // 整个限定区域已经有有效高度，无需再补。
    if (fill_cells.empty())
    {
      return StartupHoleFillResult::kNoHole;
    }

    // 防止参数设置错误导致补全区域过大。
    if (fill_cells.size() > max_fill_cells)
    {
      ROS_WARN_THROTTLE(
          1.0,
          "Startup blind area has %zu invalid cells, "
          "exceeding max_fill_cells=%zu",
          fill_cells.size(),
          max_fill_cells);

      return StartupHoleFillResult::kTooLarge;
    }

    // 等待雷达在盲区外围形成足够的有效地面高度。
    if (boundary_cells.size() < min_boundary_cells)
    {
      ROS_WARN_THROTTLE(
          1.0,
          "Startup blind area has insufficient boundary: "
          "%zu < %zu",
          boundary_cells.size(),
          min_boundary_cells);

      return StartupHoleFillResult::kInsufficientBoundary;
    }

    // 使用边界有效高度拟合：
    // z = ax + by + c
    Eigen::MatrixXd A(boundary_cells.size(), 3);
    Eigen::VectorXd z(boundary_cells.size());

    double min_boundary_z =
        std::numeric_limits<double>::infinity();

    double max_boundary_z =
        -std::numeric_limits<double>::infinity();

    for (std::size_t i = 0;
         i < boundary_cells.size(); ++i)
    {
      const grid_map::Index index(
          boundary_cells[i].first,
          boundary_cells[i].second);

      grid_map::Position position;
      if (!map.getPosition(index, position))
      {
        return StartupHoleFillResult::kInvalidMap;
      }

      const double height =
          static_cast<double>(
              elevation_bgk(index(0), index(1)));

      A.row(static_cast<Eigen::Index>(i))
          << position.x(),
          position.y(), 1.0;

      z(static_cast<Eigen::Index>(i)) = height;

      min_boundary_z =
          std::min(min_boundary_z, height);

      max_boundary_z =
          std::max(max_boundary_z, height);
    }

    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(A);

    // 边界点分布退化，例如全部集中在一条直线上。
    if (qr.rank() < 3)
    {
      ROS_WARN_THROTTLE(
          1.0,
          "Startup blind-area boundary geometry is degenerate");

      return StartupHoleFillResult::kInsufficientBoundary;
    }

    const Eigen::Vector3d plane = qr.solve(z);

    if (!plane.allFinite())
    {
      return StartupHoleFillResult::kInvalidMap;
    }

    // 只修改限定半径内的无效高度。
    for (const auto &cell : fill_cells)
    {
      const grid_map::Index index(
          cell.first, cell.second);

      grid_map::Position position;
      if (!map.getPosition(index, position))
      {
        continue;
      }

      double predicted_z =
          plane.x() * position.x() +
          plane.y() * position.y() +
          plane.z();

      // 防止拟合结果产生异常凸起或深坑。
      predicted_z = std::max(
          min_boundary_z,
          std::min(max_boundary_z, predicted_z));

      elevation_bgk(index(0), index(1)) =
          static_cast<float>(predicted_z);

      interpolated(index(0), index(1)) = 1.0f;

      ++filled_cell_count;
    }

    // 补全会影响附近坡度、台阶、粗糙度和精细化可通行性，
    // 因此将整个受影响范围标记为需要重新计算。
    const std::array<const char *, 4> computed_layers = {{"incremental_geom_computed",
                                                          "incremental_step_computed",
                                                          "incremental_trav_computed",
                                                          "incremental_fine_computed"}};

    const double dirty_radius =
        fill_radius + recompute_radius;

    for (grid_map::CircleIterator it(
             map, initial_position, dirty_radius);
         !it.isPastEnd(); ++it)
    {
      for (const char *layer : computed_layers)
      {
        if (map.exists(layer))
        {
          map.at(layer, *it) = 0.0f;
        }
      }
    }

    ROS_INFO(
        "Startup blind area filled: cells=%zu, "
        "boundary=%zu, radius=%.2f m",
        filled_cell_count,
        boundary_cells.size(),
        fill_radius);

    return StartupHoleFillResult::kFilled;
  }
} // namespace

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
      odom_topic_("state_estimation"),
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
      fine_trav_min_(0.55),
      fine_trav_max_(0.9),
      fine_slope_min_(0.50),
      fine_slope_max_(0.8),
      fine_roughness_min_(0.40),
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

  // Load the exact ViewPointManager/GridWorld geometry before any scan is
  // processed.  The mapping node then infers the same origin from synchronized
  // odometry and does not wait for viewpoint_origin.
  ConfigureViewpointGridAlignment(nh_);

  loadRobotModels();

  initGridMap();

  // Subscribe to point cloud topics
  ROS_INFO("Subscribing to point cloud and odometry topics with time synchronization...");

  // Create message filters for synchronized subscription
  cloud_filter_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::PointCloud2>>(
      nh_, scan_topic_, queue_size_);

  odom_filter_sub_ = std::make_shared<message_filters::Subscriber<nav_msgs::Odometry>>(
      nh_, odom_topic_, queue_size_);

  // Create approximate time synchronizer (allows small time differences)
  // Queue size of 10 means it will try to synchronize the last 10 messages from each topic
  sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
      SyncPolicy(queue_size_), *cloud_filter_sub_, *odom_filter_sub_);

  sync_->registerCallback(boost::bind(&Mapping::synchronizedCloudOdomCallback, this, _1, _2));

  // Create publisher for grid map
  gridmap_pub_ = nh_.advertise<grid_map_msgs::GridMap>("trav_map", 10);

  viewpoint_origin_publisher = nh_.advertise<geometry_msgs::PointStamped>(origin_topic_, 1, true);

  ROS_INFO("Publishing inferred viewpoint origin on: %s",
           viewpoint_origin_publisher.getTopic().c_str());

  const std::string vehicle_namespace = nh_.getNamespace();

  const std::string vehicle_type_param =
      (vehicle_namespace == "/")
          ? "/vehicle_type"
          : vehicle_namespace + "/vehicle_type";

  ros::param::param<std::string>(
      vehicle_type_param,
      local_trav_vehicle_type,
      "wheeled");

  ROS_INFO("vehicle_type parameter path: %s",
           vehicle_type_param.c_str());

  ROS_INFO("local_trav_vehicle_type: %s",
           local_trav_vehicle_type.c_str());

  // Create the platform-specific XYZI cloud publisher at startup.  Keeping
  // this outside buildGridMapMessage() makes the topic immediately visible in
  // `rostopic list` and avoids lazy initialization being mistaken for a
  // missing publisher.
  pnh_.param<bool>("publish_traversability_cloud",
                   publish_local_trav_cloud, true);
  // nh_.param<std::string>("vehicle_type",
  //                        local_trav_vehicle_type, "wheeled");
  pnh_.param<std::string>("traversability_cloud_topic",
                          local_trav_cloud_topic,
                          "/local_traversability_cloud");
  pnh_.param<double>("terrain_map_cost_threshold",
                     terrain_map_cost_threshold, 0.98);
  terrain_map_cost_threshold = std::max(
      0.0, std::min(1.0, terrain_map_cost_threshold));
  pnh_.param<double>("terrain_map_traversable_z_offset",
                     terrain_map_traversable_z_offset, -0.1);
  pnh_.param<double>("terrain_map_obstacle_z_offset",
                     terrain_map_obstacle_z_offset, 0.6);
  pnh_.param<bool>("terrain_map_unknown_as_traversable",
                   terrain_map_unknown_as_traversable, true);

  if (local_trav_vehicle_type == "tracked")
  {
    local_trav_layer = "traversability_fine_tracked";
  }
  else if (local_trav_vehicle_type == "wheeled")
  {
    local_trav_layer = "traversability_fine_wheeled";
  }
  else
  {
    ROS_WARN("Unknown vehicle_type '%s'; falling back to wheeled",
             local_trav_vehicle_type.c_str());
    local_trav_vehicle_type = "wheeled";
    local_trav_layer = "traversability_fine_wheeled";
  }

  if (publish_local_trav_cloud)
  {
    local_trav_cloud_publisher =
        nh_.advertise<sensor_msgs::PointCloud2>(
            local_trav_cloud_topic, 1, false);
    ROS_INFO("Publishing %s binary terrain map on %s using layer %s (cost threshold %.3f, intensities 0.2/1.0, z offsets %.2f/%.2f m, unknown_as_traversable=%s)",
             local_trav_vehicle_type.c_str(),
             local_trav_cloud_publisher.getTopic().c_str(),
             local_trav_layer.c_str(),
             terrain_map_cost_threshold,
             terrain_map_traversable_z_offset,
             terrain_map_obstacle_z_offset,
             terrain_map_unknown_as_traversable ? "true" : "false");
  }
  else
  {
    ROS_INFO("Local traversability cloud publication disabled");
  }

  // Publish the resampled elevation map on a relative topic so every robot
  // gets an independent height cloud matching its own viewpoint rolling grid.
  pnh_.param<bool>("publish_viewpoint_aligned_terrain_cloud",
                   publish_viewpoint_aligned_terrain_cloud, true);
  pnh_.param<std::string>("viewpoint_aligned_terrain_cloud_topic",
                          viewpoint_aligned_terrain_cloud_topic,
                          "terrain_map_ext");
  pnh_.param<std::string>("viewpoint_aligned_height_layer",
                          viewpoint_aligned_height_layer,
                          "elevation_BGK");

  if (publish_viewpoint_aligned_terrain_cloud)
  {
    viewpoint_aligned_terrain_cloud_publisher =
        nh_.advertise<sensor_msgs::PointCloud2>(
            viewpoint_aligned_terrain_cloud_topic, 1, false);
    ROS_INFO("Publishing viewpoint-aligned terrain height cloud on %s using layer %s",
             viewpoint_aligned_terrain_cloud_publisher.getTopic().c_str(),
             viewpoint_aligned_height_layer.c_str());
  }
  else
  {
    ROS_INFO("Viewpoint-aligned terrain height cloud publication disabled");
  }

  ROS_INFO("Successfully subscribed to: %s (synchronized)", scan_topic_.c_str());
  ROS_INFO("Successfully subscribed to: %s (synchronized)", odom_topic_.c_str());
  ROS_INFO("Publishing grid map on topic: trav_map");

  // Start processing thread for asynchronous point cloud processing
  should_exit_ = false;
  processing_thread_ = std::thread(&Mapping::processingThreadFunc, this);
  ROS_INFO("Started asynchronous processing thread (skip_old_messages: %s)",
           skip_old_messages_ ? "true" : "false");

  ROS_INFO("Mapping initialization complete");
  ROS_INFO("Subscribed to topic: %s", scan_topic_.c_str());
  ROS_INFO("Subscribed to topic: %s", odom_topic_.c_str());
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

  // No task may retain `this` after Mapping starts destruction.
  globalPublishWorker().shutdown();

  ROS_INFO("Mapping destructor called");
}

void Mapping::loadParameters()
{
  ROS_INFO("Loading ROS parameters...");

  // Load topic settings
  pnh_.param<std::string>("scan_topic", scan_topic_, "registered_scan");
  pnh_.param<std::string>("odom_topic", odom_topic_, "state_estimation");
  pnh_.param<std::string>("origin_topic", origin_topic_, "traverability_viewpoint_origin");

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
  // 精细化校核只用于中高风险不确定区。提高下限可避免
  // 在明显平坦区域反复执行昂贵的整车碰撞/支撑姿态求解。
  pnh_.param<double>("fine_trav_min", fine_trav_min_, 0.55);
  pnh_.param<double>("fine_trav_max", fine_trav_max_, 0.9);
  pnh_.param<double>("fine_slope_min", fine_slope_min_, 0.50);
  pnh_.param<double>("fine_slope_max", fine_slope_max_, 0.8);
  pnh_.param<double>("fine_roughness_min", fine_roughness_min_, 0.40);
  pnh_.param<double>("fine_roughness_max", fine_roughness_max_, 1.0);
  pnh_.param<double>("fine_roll_threshold_deg", fine_roll_threshold_deg_, 30.0);
  pnh_.param<double>("fine_pitch_threshold_deg", fine_pitch_threshold_deg_, 30.0);

  // Load incremental mapping parameters (default to enabled)
  pnh_.param<bool>("enable_incremental_geom", enable_incremental_geom_, true);
  pnh_.param<bool>("enable_incremental_step", enable_incremental_step_, true);
  pnh_.param<bool>("enable_incremental_trav", enable_incremental_trav_, true);

  // Load asynchronous processing parameters
  // Mapping is more expensive than a scan callback.  Keep only the newest
  // waiting scan so terrain-cloud latency cannot grow without bound.
  pnh_.param<bool>("skip_old_messages", skip_old_messages_, true);

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
  ROS_INFO("  - odom_topic: %s", odom_topic_.c_str());
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
    ROS_WARN("Wheeled vehicle model unavailable; wheeled coarse/fine traversability layers will be left unchanged");
  }
  if (!tracked_model_loaded_)
  {
    ROS_WARN("Tracked vehicle model unavailable; tracked coarse/fine traversability layers will be left unchanged");
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

  try
  {
    while (std::getline(file, line))
    {
      if (line.find_first_not_of(" \t\r\n") == std::string::npos)
      {
        continue;
      }

      std::stringstream ss(line);
      std::string value;
      std::vector<double> row;

      while (std::getline(ss, value, ','))
      {
        const std::size_t first = value.find_first_not_of(" \t\r\n");
        const std::size_t last = value.find_last_not_of(" \t\r\n");
        if (first == std::string::npos)
        {
          ROS_ERROR("Empty vehicle model value in %s", path.c_str());
          return false;
        }
        const std::string token = value.substr(first, last - first + 1);
        std::size_t parsed = 0;
        const double parsed_value = std::stod(token, &parsed);
        if (parsed != token.size() || !std::isfinite(parsed_value))
        {
          ROS_ERROR("Invalid vehicle model value '%s' in %s",
                    token.c_str(), path.c_str());
          return false;
        }
        row.push_back(parsed_value);
      }
      data.push_back(std::move(row));
    }
  }
  catch (const std::exception &e)
  {
    ROS_ERROR("Failed to parse vehicle model %s: %s",
              path.c_str(), e.what());
    return false;
  }

  file.close();

  if (data.size() != static_cast<std::size_t>(robot_rows_))
  {
    ROS_ERROR("Vehicle model %s must contain exactly %d rows, but got %zu",
              path.c_str(), robot_rows_, data.size());
    return false;
  }

  for (int i = 0; i < robot_rows_; ++i)
  {
    if (data[i].size() != static_cast<std::size_t>(robot_cols_))
    {
      ROS_ERROR("Vehicle model %s row %d must contain exactly %d columns, but got %zu",
                path.c_str(), i, robot_cols_, data[i].size());
      return false;
    }
  }

  model_storage.X_.resize(robot_rows_, robot_cols_);
  model_storage.Y_.resize(robot_rows_, robot_cols_);
  model_storage.Z_.resize(robot_rows_, robot_cols_);
  model_storage.Gap_.resize(robot_rows_, robot_cols_);

  model_storage.X_.setZero();
  model_storage.Y_.setZero();
  model_storage.Z_.setZero();
  model_storage.Gap_.setZero();

  for (int i = 0; i < robot_rows_; ++i)
  {
    for (int j = 0; j < robot_cols_; ++j)
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

    // 回调只记录位姿已可用。真正用于建图的 vehicle_position_
    // 由处理线程在取出 ProcessingTask 后更新。这样新里程计回调
    // 不会在旧点云处理到一半时覆盖其位姿。
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

    // Latest-wins：处理端跟不上时，未开始的旧点云已经
    // 失去实时价值。每次都清空等待队列，仅保留最新任务，
    // 防止 trav_map 的延迟随运行时间持续增长。
    if (skip_old_messages_ && !processing_queue_.empty())
    {
      const size_t dropped = processing_queue_.size();
      std::queue<ProcessingTask> empty_queue;
      processing_queue_.swap(empty_queue);
      ROS_WARN_THROTTLE(2.0,
                        "Dropped %zu stale mapping task(s); keeping the newest scan",
                        dropped);
    }

    // 把本次任务压入队尾
    processing_queue_.push(std::move(task));
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

      // 使局部子图继承与当前点云一致的时间戳，而不是 0
      // 或发布时刻。这对下游的时序同步和延迟监测很重要。
      {
        std::lock_guard<std::mutex> map_lock(gridmap_publish_mutex_);
        height_map_.setTimestamp(task.timestamp.toNSec());
      }

      // ------------------------------------------
      // [B2] 真正处理点云（重计算：滤波/配准/栅格更新等）
      // ------------------------------------------
      processPointCloud(task.cloud, vehicle_position);

      // Use the pose carried by this exact processing task.  This reproduces
      // ViewPointManager's initial origin and rollover before the map produced
      // from the same scan is published, without waiting for viewpoint_origin.
      viewpointGridAlignment().update(task.position, task.timestamp);

      // 每处理一组同步点云和里程计，都发布一次当前viewpoint原点。
      ViewpointGridSnapshot origin_snapshot;

      if (viewpointGridAlignment().snapshot(origin_snapshot))
      {
        geometry_msgs::PointStamped origin_msg;

        // 与当前处理的点云和里程计使用相同时间戳。
        origin_msg.header.stamp = task.timestamp;
        origin_msg.header.frame_id = frame_id_;

        origin_msg.point.x = origin_snapshot.origin.x();
        origin_msg.point.y = origin_snapshot.origin.y();
        origin_msg.point.z = origin_snapshot.origin.z();

        viewpoint_origin_publisher.publish(origin_msg);

        ROS_INFO_STREAM_THROTTLE(1.0, "Published viewpoint origin: ["
                                          << origin_msg.point.x << ", "
                                          << origin_msg.point.y << ", "
                                          << origin_msg.point.z << "]");
      }

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

  // -------------------------------------------------------------------------
  // 步骤 2.5：初始位置限定范围盲区补全
  // -------------------------------------------------------------------------
  // 不再要求unknown区域完全封闭。只补初始位置固定半径内的
  // elevation_BGK无效栅格，并从盲区边缘有效地形拟合局部平面。
  static bool startup_fill_configured = false;
  static bool startup_fill_enabled = true;
  static bool startup_position_recorded = false;
  static bool startup_fill_finished = false;

  static grid_map::Position startup_position;

  // 雷达高度0.72 m、最低俯角约-15°，
  // 理论盲区半径约为2.69 m，因此取3.3 m并保留余量。
  static double startup_hole_radius = 3.3;

  // 在盲区半径内外各0.9 m范围收集有效地面高度。
  static double startup_hole_boundary_width = 0.9;

  // 0.3 m分辨率下，3.3 m圆形区域总共约380个栅格。
  static int startup_hole_max_cells = 800;

  static int startup_hole_min_boundary_cells = 12;
  static int startup_hole_max_attempts = 100;
  static int startup_hole_attempts = 0;

  if (!startup_fill_configured)
  {
    pnh_.param<bool>(
        "enable_startup_hole_fill",
        startup_fill_enabled,
        true);

    pnh_.param<double>(
        "startup_hole_radius",
        startup_hole_radius,
        3.3);

    pnh_.param<double>(
        "startup_hole_boundary_width",
        startup_hole_boundary_width,
        0.9);

    pnh_.param<int>(
        "startup_hole_max_cells",
        startup_hole_max_cells,
        800);

    pnh_.param<int>(
        "startup_hole_min_boundary_cells",
        startup_hole_min_boundary_cells,
        12);

    pnh_.param<int>(
        "startup_hole_max_attempts",
        startup_hole_max_attempts,
        100);

    startup_hole_radius =
        std::max(map_resolution_ * 2.0,
                 startup_hole_radius);

    startup_hole_boundary_width =
        std::max(map_resolution_,
                 startup_hole_boundary_width);

    startup_hole_max_cells =
        std::max(1, startup_hole_max_cells);

    startup_hole_min_boundary_cells =
        std::max(3, startup_hole_min_boundary_cells);

    startup_hole_max_attempts =
        std::max(1, startup_hole_max_attempts);

    startup_fill_configured = true;

    ROS_INFO(
        "Startup blind-area filling: %s, "
        "radius=%.2f m, boundary_width=%.2f m, "
        "max_cells=%d, min_boundary=%d, max_attempts=%d",
        startup_fill_enabled ? "enabled" : "disabled",
        startup_hole_radius,
        startup_hole_boundary_width,
        startup_hole_max_cells,
        startup_hole_min_boundary_cells,
        startup_hole_max_attempts);
  }

  if (startup_fill_enabled &&
      !startup_fill_finished)
  {
    // 固定记录第一帧同步点云对应的车辆位置，
    // 后续重试时不会跟随车辆移动。
    if (!startup_position_recorded)
    {
      startup_position =
          grid_map::Position(
              current_pos.x(),
              current_pos.y());

      startup_position_recorded = true;

      ROS_INFO(
          "Recorded startup blind-area center: "
          "x=%.3f, y=%.3f",
          startup_position.x(),
          startup_position.y());
    }

    std::size_t filled_cells = 0;
    StartupHoleFillResult fill_result;

    {
      std::lock_guard<std::mutex> lock(
          gridmap_publish_mutex_);

      fill_result = fillStartupBlindArea(
          height_map_,
          startup_position,
          startup_hole_radius,
          startup_hole_boundary_width,
          static_cast<std::size_t>(
              startup_hole_max_cells),
          static_cast<std::size_t>(
              startup_hole_min_boundary_cells),
          std::max(
              normal_estimation_radius_,
              step_radius_) +
              map_resolution_,
          filled_cells);
    }

    if (fill_result ==
        StartupHoleFillResult::kFilled)
    {
      startup_fill_finished = true;

      ROS_INFO(
          "Completed startup blind-area filling: "
          "%zu cells",
          filled_cells);
    }
    else if (fill_result ==
             StartupHoleFillResult::kNoHole)
    {
      startup_fill_finished = true;

      ROS_INFO(
          "Startup area already has complete "
          "elevation coverage");
    }
    else
    {
      ++startup_hole_attempts;

      const char *failure_reason = "unknown";

      switch (fill_result)
      {
      case StartupHoleFillResult::kTooLarge:
        failure_reason = "too_many_invalid_cells";
        break;

      case StartupHoleFillResult::kInsufficientBoundary:
        failure_reason = "insufficient_valid_boundary";
        break;

      case StartupHoleFillResult::kInvalidMap:
        failure_reason = "invalid_map_or_parameters";
        break;

      case StartupHoleFillResult::kNotClosed:
        // 新逻辑不会返回该结果，保留只是兼容原枚举。
        failure_reason = "not_closed";
        break;

      default:
        break;
      }

      ROS_WARN_THROTTLE(
          1.0,
          "Startup blind-area filling attempt "
          "%d/%d failed: %s",
          startup_hole_attempts,
          startup_hole_max_attempts,
          failure_reason);

      if (startup_hole_attempts >=
          startup_hole_max_attempts)
      {
        startup_fill_finished = true;

        ROS_WARN(
            "Startup blind-area filling stopped "
            "after %d attempts",
            startup_hole_attempts);
      }
    }
  }

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
  // 计算台阶/突变特征（局部平面补偿后的稳健高度残差）
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
  height_map_.add("elevation");                     // Height of observed points
  height_map_.add("elevation_BGK");                 // Interpolated height using BGK
  height_map_.add("variance");                      // Height variance
  height_map_.add("min_elevation");                 // Minimum elevation in cell
  height_map_.add("max_elevation");                 // Maximum elevation in cell
  height_map_.add("n_points");                      // Number of points in cell
  height_map_.add("normal_x");                      // Surface normal X component
  height_map_.add("normal_y");                      // Surface normal Y component
  height_map_.add("normal_z");                      // Surface normal Z component
  height_map_.add("slope");                         // Slope feature
  height_map_.add("roughness");                     // Roughness feature
  height_map_.add("step");                          // Step feature
  height_map_.add("slope_deg");                     // Raw slope angle in degrees
  height_map_.add("roughness_raw");                 // Raw PCA roughness ratio
  height_map_.add("step_height");                   // Raw local step height in meters
  height_map_.add("traversability");                // Traversability cost
  height_map_.add("traversability_coarse_wheeled"); // Platform-specific coarse traversability for wheeled car
  height_map_.add("traversability_coarse_tracked"); // Platform-specific coarse traversability for tracked car
  height_map_.add("traversability_fine_wheeled");   // Fine-grained traversability for wheeled car
  height_map_.add("traversability_fine_tracked");   // Fine-grained traversability for tracked car
  height_map_.add("critical");                      // Flag: cells evaluated by fine-grained mapping
  height_map_.add("interpolated");                  // Flag: whether this cell has been interpolated by BGK
  height_map_.add("incremental_geom_computed");     // Flag: whether geometric mapping has been computed incrementally
  height_map_.add("incremental_step_computed");     // Flag: whether step mapping has been computed incrementally
  height_map_.add("incremental_trav_computed");     // Flag: whether traversability mapping has been computed incrementally
  height_map_.add("incremental_fine_computed");     // Flag: whether fine traversability is up to date

  // Initialize n_points layer to 0
  height_map_["n_points"].setConstant(0);

  // Initialize flag layers to 0 (not computed/interpolated)
  height_map_["interpolated"].setConstant(0);
  height_map_["incremental_geom_computed"].setConstant(0);
  height_map_["incremental_step_computed"].setConstant(0);
  height_map_["incremental_trav_computed"].setConstant(0);
  height_map_["incremental_fine_computed"].setConstant(0);
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
  auto &incremental_geom = height_map_["incremental_geom_computed"];
  auto &incremental_step = height_map_["incremental_step_computed"];
  auto &incremental_trav = height_map_["incremental_trav_computed"];
  auto &incremental_fine = height_map_["incremental_fine_computed"];

  // 只有高程均值发生足够大的变化才重算周边特征和精细化姿态。
  // 这个阈值可通过 ~fine_height_change_threshold 调整。
  static bool dirty_configured = false;
  static double fine_height_change_threshold = 0.02;
  static double fine_recompute_radius = -1.0;
  if (!dirty_configured)
  {
    pnh_.param<double>("fine_height_change_threshold",
                       fine_height_change_threshold, 0.02);
    pnh_.param<double>("fine_recompute_radius",
                       fine_recompute_radius, -1.0);
    fine_height_change_threshold =
        std::max(0.0, fine_height_change_threshold);

    if (fine_recompute_radius <= 0.0)
    {
      const double vehicle_width =
          static_cast<double>(robot_rows_) * robot_model_resolution_;
      const double vehicle_length =
          static_cast<double>(robot_cols_) * robot_model_resolution_;
      fine_recompute_radius =
          0.5 * std::hypot(vehicle_width, vehicle_length) +
          std::max(normal_estimation_radius_, step_radius_) +
          map_resolution_;
    }
    fine_recompute_radius =
        std::max(map_resolution_, fine_recompute_radius);
    dirty_configured = true;

    ROS_INFO("Incremental fine recomputation: height_delta=%.3f m, radius=%.2f m",
             fine_height_change_threshold, fine_recompute_radius);
  }

  int cell_count = 0;    // 统计：成功落在地图范围内并更新过的点数（这里命名是 cell_count，但实际上是“点更新次数”）
  int out_of_bounds = 0; // 统计：落在地图外的点数

  // elevation_BGK：用于 BGK 插值/预测后的高度层
  // 你后面会对“新观测到的格子”用真实观测高度去覆盖 BGK 的值，避免 BGK 继续占用这些格子
  auto &elevation_bgk = height_map_["elevation_BGK"];

  // newly_observed_cells：记录本帧中“从未观测（npts==0）变为有观测（npts>0）”的格子索引
  // 用 set 去重，确保每个格子只记录一次
  std::set<std::pair<int, int>> newly_observed_cells;
  std::set<std::pair<int, int>> changed_height_cells;

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
    const float old_height = height;

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
      changed_height_cells.insert(std::make_pair(index(0), index(1)));
    }
    else if (std::isfinite(old_height) &&
             std::fabs(height - old_height) >=
                 fine_height_change_threshold)
    {
      changed_height_cells.insert(std::make_pair(index(0), index(1)));
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

  for (const auto &cell_idx : changed_height_cells)
  {
    // 真实观测发生明显变化时，同步更新 BGK 高程。
    elevation_bgk(cell_idx.first, cell_idx.second) =
        elevation(cell_idx.first, cell_idx.second);
    bgk_updated_cells++;
  }

  // 一个高程变化会影响周边的法向、台阶以及整车支撑姿态。
  // 仅把这些依赖区域标记为 dirty，其他格子沿用已有结果。
  std::size_t invalidated_cells = 0;
  for (const auto &cell_idx : changed_height_cells)
  {
    grid_map::Position changed_position;
    if (!height_map_.getPosition(
            grid_map::Index(cell_idx.first, cell_idx.second),
            changed_position))
    {
      continue;
    }

    for (grid_map::CircleIterator it(height_map_, changed_position,
                                     fine_recompute_radius);
         !it.isPastEnd(); ++it)
    {
      incremental_geom((*it)(0), (*it)(1)) = 0.0f;
      incremental_step((*it)(0), (*it)(1)) = 0.0f;
      incremental_trav((*it)(0), (*it)(1)) = 0.0f;
      incremental_fine((*it)(0), (*it)(1)) = 0.0f;
      ++invalidated_cells;
    }
  }

  // 输出统计信息：映射到地图内的点更新次数、以及 BGK 被覆盖的格子数
  // 注意：cell_count 实际上是“点数”，不是 unique cell 数
  ROS_INFO("Height mapping: processed %d points, new=%zu, changed=%zu, "
           "updated_BGK=%d, invalidated=%zu",
           cell_count, newly_observed_cells.size(),
           changed_height_cells.size(), bgk_updated_cells,
           invalidated_cells);
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
    if ((x_train_vec.size() / 2) < 3)
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
  if (solver.info() != Eigen::Success ||
      !solver.eigenvalues().allFinite())
  {
    return false;
  }

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
  const double eigenvalue_sum = solver.eigenvalues().sum();
  if (!std::isfinite(eigenvalue_sum) || eigenvalue_sum <= 1e-12)
  {
    return false;
  }
  const float roughness = static_cast<float>(std::max(
      0.0, solver.eigenvalues()(0) / eigenvalue_sum));

  // slope：坡度（由法向的 z 分量决定）
  // normal.z() = cos(theta)（theta 是法向与竖直方向的夹角）
  // acos(normal.z()) 得到 theta（0 表示水平面，越大表示越陡）
  //
  // 你后面除以 (slope_threshold_ * pi) 是一种人为归一化：
  // - slope_threshold_ 可能是一个阈值比例（例如 0.5 表示 90°*0.5=45°）
  // - 最终 slope 是一个无量纲比值，用于后续 traversability 代价融合
  const double slope_angle_rad =
      std::acos(std::min(1.0, std::max(-1.0, normal.z())));
  const float slope_deg = static_cast<float>(
      slope_angle_rad * 180.0 / M_PI);
  if (slope_threshold_ <= 0.0 || roughness_threshold_ <= 0.0)
  {
    return false;
  }
  const float slope = static_cast<float>(
      slope_angle_rad / (slope_threshold_ * M_PI));

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
  height_map_.at("slope_deg", index) = slope_deg;
  height_map_.at("roughness_raw", index) = roughness;

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
  // A cell may be recomputed after a height update.  Clear its previous
  // values first so a failed recomputation cannot leave stale step data.
  height_map_.at("step", index) =
      std::numeric_limits<float>::quiet_NaN();
  height_map_.at("step_height", index) =
      std::numeric_limits<float>::quiet_NaN();

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

  // 若当前 cell 的高度无效，无法计算 step
  if (!std::isfinite(center_z))
  {
    return false;
  }

  // =====================================================
  // [2] 去除连续坡面高度变化，收集邻域台阶残差
  // =====================================================
  // 原始 |z_center-z_neighbor| 会把正常斜坡当作台阶。利用几何计算
  // 已得到的中心法向预测邻居的平面高度，真正统计的是邻居高度相对
  // 该平面的残差。
  const StepEvaluationConfig &step_config =
      stepEvaluationConfig(pnh_);

  const float normal_x = height_map_.at("normal_x", index);
  const float normal_y = height_map_.at("normal_y", index);
  const float normal_z = height_map_.at("normal_z", index);
  const bool use_plane_compensation =
      std::isfinite(normal_x) &&
      std::isfinite(normal_y) &&
      std::isfinite(normal_z) &&
      std::abs(normal_z) > 0.20f;

  std::vector<float> height_residuals;

  for (grid_map::CircleIterator sit(height_map_, center_pos, step_radius_);
       !sit.isPastEnd();
       ++sit)
  {
    const grid_map::Index neighbor_index = *sit;
    if (neighbor_index(0) == index(0) &&
        neighbor_index(1) == index(1))
    {
      continue;
    }

    // 取邻居格子的高度
    const float neighbor_z =
        height_map_.at("elevation_BGK", neighbor_index);

    // 只用有效高度参与计算
    if (std::isfinite(neighbor_z))
    {
      double expected_z = static_cast<double>(center_z);
      if (use_plane_compensation)
      {
        grid_map::Position neighbor_pos;
        if (!height_map_.getPosition(neighbor_index, neighbor_pos))
        {
          continue;
        }

        const double dx = neighbor_pos.x() - center_pos.x();
        const double dy = neighbor_pos.y() - center_pos.y();
        expected_z -=
            (static_cast<double>(normal_x) * dx +
             static_cast<double>(normal_y) * dy) /
            static_cast<double>(normal_z);
      }

      const double residual =
          std::abs(static_cast<double>(neighbor_z) - expected_z);
      if (std::isfinite(residual))
      {
        height_residuals.push_back(static_cast<float>(residual));
      }
    }
  }

  if (height_residuals.size() <
      static_cast<std::size_t>(step_config.min_valid_neighbors))
  {
    return false;
  }

  std::sort(height_residuals.begin(), height_residuals.end());
  // round() keeps the maximum when the existing neighborhood contains only
  // 3-4 cells, while a denser neighborhood can discard one isolated maximum.
  const std::size_t quantile_index = std::min(
      height_residuals.size() - 1,
      static_cast<std::size_t>(std::llround(
          step_config.robust_quantile *
          static_cast<double>(height_residuals.size() - 1))));
  const float robust_step = height_residuals[quantile_index];

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
  // - step ≈ 1 表示稳健台阶残差接近通用阈值
  // - step > 1 表示超过阈值的突变（可认为更不可通行）
  height_map_.at("step", index) = robust_step / step_threshold_;
  height_map_.at("step_height", index) = robust_step;

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
  std::lock_guard<std::mutex> lock(gridmap_publish_mutex_);
  ++fine_cycle_generation;

  const StepEvaluationConfig &step_config =
      stepEvaluationConfig(pnh_);

  auto &traversability = height_map_["traversability"];
  auto &slope_layer = height_map_["slope"];
  auto &roughness_layer = height_map_["roughness"];
  auto &step_layer = height_map_["step"];
  auto &slope_deg_layer = height_map_["slope_deg"];
  auto &roughness_raw_layer = height_map_["roughness_raw"];
  auto &step_height_layer = height_map_["step_height"];
  auto &n_points_layer = height_map_["n_points"];
  auto &interpolated_layer = height_map_["interpolated"];
  auto &critical_layer = height_map_["critical"];
  auto &incremental_fine_computed =
      height_map_["incremental_fine_computed"];

  struct VehicleCapability
  {
    double slope_limit_deg;
    double roughness_limit;
    double step_limit_m;
    double roll_limit_deg;
    double pitch_limit_deg;
    double max_pose_tilt_deg;
    double touch_gap_threshold;
    double collision_clearance;
    double geo_slope_weight;
    double geo_roughness_weight;
    double geo_step_weight;
    // Signed motion-efficiency correction:
    //   +flat_penalty * (1-d): inefficient operation on flat terrain
    //   +rough_penalty * d:    extra risk on rough terrain
    //   -rough_bonus * d:      platform capability benefit on rough terrain
    double efficiency_flat_penalty;
    double efficiency_rough_penalty;
    double efficiency_rough_bonus;
  };

  static bool capability_configured = false;
  static VehicleCapability wheeled_capability = {
      25.0, 0.04, 0.18, 25.0, 30.0, 40.0,
      0.05, 0.10, 0.35, 0.25, 0.40,
      0.00, 0.15, 0.00};
  static VehicleCapability tracked_capability = {
      35.0, 0.08, 0.30, 30.0, 40.0, 45.0,
      0.05, 0.08, 0.30, 0.25, 0.45,
      0.04, 0.00, 0.12};

  static double pose_angle_weight = 0.65;
  static double pose_support_weight = 0.20;
  static double pose_heading_weight = 0.15;
  static double efficiency_weight = 1.00;

  // 地形严重度使用同一组参考值，保证两种平台的效率项可直接比较。
  static double efficiency_slope_reference_deg = 35.0;
  static double efficiency_roughness_reference = 0.08;
  static double efficiency_step_reference_m = 0.30;
  static double severity_slope_weight = 0.35;
  static double severity_roughness_weight = 0.25;
  static double severity_step_weight = 0.40;

  // 这些范围作用于“按车型能力阈值归一化后的特征”。
  static double fine_step_min = 0.40;
  static double fine_step_max = 1.00;

  auto clamp01 = [](double value)
  {
    return std::max(0.0, std::min(1.0, value));
  };

  auto normalizeThreeWeights = [](double &a, double &b, double &c,
                                  double da, double db, double dc)
  {
    a = std::max(0.0, a);
    b = std::max(0.0, b);
    c = std::max(0.0, c);
    double sum = a + b + c;
    if (sum < 1e-9)
    {
      a = da;
      b = db;
      c = dc;
      sum = a + b + c;
    }
    a /= sum;
    b /= sum;
    c /= sum;
  };

  if (!capability_configured)
  {
    auto loadCapability = [&](const std::string &prefix,
                              VehicleCapability &capability)
    {
      pnh_.param<double>(prefix + "slope_limit_deg",
                         capability.slope_limit_deg,
                         capability.slope_limit_deg);
      pnh_.param<double>(prefix + "roughness_limit",
                         capability.roughness_limit,
                         capability.roughness_limit);
      pnh_.param<double>(prefix + "step_limit_m",
                         capability.step_limit_m,
                         capability.step_limit_m);
      pnh_.param<double>(prefix + "roll_limit_deg",
                         capability.roll_limit_deg,
                         capability.roll_limit_deg);
      pnh_.param<double>(prefix + "pitch_limit_deg",
                         capability.pitch_limit_deg,
                         capability.pitch_limit_deg);
      pnh_.param<double>(prefix + "max_pose_tilt_deg",
                         capability.max_pose_tilt_deg,
                         capability.max_pose_tilt_deg);
      pnh_.param<double>(prefix + "touch_gap_threshold",
                         capability.touch_gap_threshold,
                         capability.touch_gap_threshold);
      pnh_.param<double>(prefix + "collision_clearance",
                         capability.collision_clearance,
                         capability.collision_clearance);
      pnh_.param<double>(prefix + "geo_slope_weight",
                         capability.geo_slope_weight,
                         capability.geo_slope_weight);
      pnh_.param<double>(prefix + "geo_roughness_weight",
                         capability.geo_roughness_weight,
                         capability.geo_roughness_weight);
      pnh_.param<double>(prefix + "geo_step_weight",
                         capability.geo_step_weight,
                         capability.geo_step_weight);
      capability.slope_limit_deg =
          std::max(1e-3, capability.slope_limit_deg);
      capability.roughness_limit =
          std::max(1e-6, capability.roughness_limit);
      capability.step_limit_m =
          std::max(1e-3, capability.step_limit_m);
      capability.roll_limit_deg =
          std::max(1e-3, capability.roll_limit_deg);
      capability.pitch_limit_deg =
          std::max(1e-3, capability.pitch_limit_deg);
      capability.max_pose_tilt_deg =
          std::max(1.0, std::min(89.0, capability.max_pose_tilt_deg));
      capability.touch_gap_threshold =
          std::max(1e-4, capability.touch_gap_threshold);
      capability.collision_clearance =
          std::max(0.0, capability.collision_clearance);
      normalizeThreeWeights(capability.geo_slope_weight,
                            capability.geo_roughness_weight,
                            capability.geo_step_weight,
                            0.35, 0.25, 0.40);
    };

    loadCapability("wheeled_", wheeled_capability);
    loadCapability("tracked_", tracked_capability);

    // Explicit parameter names are used here so every launch parameter can be
    // found directly in this source file without relying on prefix assembly.
    pnh_.param<double>("wheeled_efficiency_flat_penalty",
                       wheeled_capability.efficiency_flat_penalty, 0.00);
    pnh_.param<double>("wheeled_efficiency_rough_penalty",
                       wheeled_capability.efficiency_rough_penalty, 0.15);
    pnh_.param<double>("wheeled_efficiency_rough_bonus",
                       wheeled_capability.efficiency_rough_bonus, 0.00);
    pnh_.param<double>("tracked_efficiency_flat_penalty",
                       tracked_capability.efficiency_flat_penalty, 0.04);
    pnh_.param<double>("tracked_efficiency_rough_penalty",
                       tracked_capability.efficiency_rough_penalty, 0.00);
    pnh_.param<double>("tracked_efficiency_rough_bonus",
                       tracked_capability.efficiency_rough_bonus, 0.12);

    wheeled_capability.efficiency_flat_penalty =
        clamp01(wheeled_capability.efficiency_flat_penalty);
    wheeled_capability.efficiency_rough_penalty =
        clamp01(wheeled_capability.efficiency_rough_penalty);
    wheeled_capability.efficiency_rough_bonus =
        clamp01(wheeled_capability.efficiency_rough_bonus);
    tracked_capability.efficiency_flat_penalty =
        clamp01(tracked_capability.efficiency_flat_penalty);
    tracked_capability.efficiency_rough_penalty =
        clamp01(tracked_capability.efficiency_rough_penalty);
    tracked_capability.efficiency_rough_bonus =
        clamp01(tracked_capability.efficiency_rough_bonus);

    pnh_.param<double>("pose_angle_weight",
                       pose_angle_weight, 0.65);
    pnh_.param<double>("pose_support_weight",
                       pose_support_weight, 0.20);
    pnh_.param<double>("pose_heading_weight",
                       pose_heading_weight, 0.15);
    pnh_.param<double>("mobility_efficiency_weight",
                       efficiency_weight, 1.00);

    pnh_.param<double>("efficiency_slope_reference_deg",
                       efficiency_slope_reference_deg, 35.0);
    pnh_.param<double>("efficiency_roughness_reference",
                       efficiency_roughness_reference, 0.08);
    pnh_.param<double>("efficiency_step_reference_m",
                       efficiency_step_reference_m, 0.30);
    pnh_.param<double>("severity_slope_weight",
                       severity_slope_weight, 0.35);
    pnh_.param<double>("severity_roughness_weight",
                       severity_roughness_weight, 0.25);
    pnh_.param<double>("severity_step_weight",
                       severity_step_weight, 0.40);
    pnh_.param<double>("fine_step_min", fine_step_min, 0.40);
    pnh_.param<double>("fine_step_max", fine_step_max, 1.00);

    normalizeThreeWeights(pose_angle_weight,
                          pose_support_weight,
                          pose_heading_weight,
                          0.65, 0.20, 0.15);
    normalizeThreeWeights(severity_slope_weight,
                          severity_roughness_weight,
                          severity_step_weight,
                          0.35, 0.25, 0.40);

    efficiency_weight = clamp01(efficiency_weight);
    efficiency_slope_reference_deg =
        std::max(1e-3, efficiency_slope_reference_deg);
    efficiency_roughness_reference =
        std::max(1e-6, efficiency_roughness_reference);
    efficiency_step_reference_m =
        std::max(1e-3, efficiency_step_reference_m);
    fine_step_min = std::max(0.0, fine_step_min);
    fine_step_max = std::max(fine_step_min, fine_step_max);

    capability_configured = true;

    ROS_INFO("Platform-dependent traversability configured");
    ROS_INFO("  Wheeled limits: slope=%.1f deg, roughness=%.4f, step=%.3f m, roll=%.1f deg, pitch=%.1f deg",
             wheeled_capability.slope_limit_deg,
             wheeled_capability.roughness_limit,
             wheeled_capability.step_limit_m,
             wheeled_capability.roll_limit_deg,
             wheeled_capability.pitch_limit_deg);
    ROS_INFO("  Tracked limits: slope=%.1f deg, roughness=%.4f, step=%.3f m, roll=%.1f deg, pitch=%.1f deg",
             tracked_capability.slope_limit_deg,
             tracked_capability.roughness_limit,
             tracked_capability.step_limit_m,
             tracked_capability.roll_limit_deg,
             tracked_capability.pitch_limit_deg);
    ROS_INFO("  Efficiency correction: weight=%.2f", efficiency_weight);
    ROS_INFO("    Wheeled: +%.3f*(1-d) + %.3f*d - %.3f*d",
             wheeled_capability.efficiency_flat_penalty,
             wheeled_capability.efficiency_rough_penalty,
             wheeled_capability.efficiency_rough_bonus);
    ROS_INFO("    Tracked: +%.3f*(1-d) + %.3f*d - %.3f*d",
             tracked_capability.efficiency_flat_penalty,
             tracked_capability.efficiency_rough_penalty,
             tracked_capability.efficiency_rough_bonus);
  }

  struct VehicleEvalContext
  {
    int id = 0;
    const HeightGrid *model = nullptr;
    grid_map::Matrix *coarse_layer = nullptr;
    grid_map::Matrix *fine_layer = nullptr;
    const VehicleCapability *capability = nullptr;
    std::string name;
    int nominal_contact_points = 1;
    int count_success = 0;
    int count_collision = 0;
    int count_failure = 0;
  };

  std::vector<VehicleEvalContext> vehicles;
  vehicles.reserve(2);

  auto registerVehicle = [&](int id,
                             const HeightGrid &model,
                             bool loaded,
                             const std::string &coarse_layer_name,
                             const std::string &fine_layer_name,
                             const std::string &name,
                             const VehicleCapability &capability)
  {
    if (!loaded)
    {
      return;
    }
    if (!height_map_.exists(coarse_layer_name) ||
        !height_map_.exists(fine_layer_name))
    {
      ROS_WARN("Grid map layer is missing for %s (coarse=%s, fine=%s); skipping this vehicle",
               name.c_str(), coarse_layer_name.c_str(),
               fine_layer_name.c_str());
      return;
    }

    VehicleEvalContext context;
    context.id = id;
    context.model = &model;
    context.coarse_layer = &height_map_[coarse_layer_name];
    context.fine_layer = &height_map_[fine_layer_name];
    context.capability = &capability;
    context.name = name;

    double min_support_z = std::numeric_limits<double>::infinity();
    for (int i = 0; i < robot_rows_; ++i)
    {
      for (int j = 0; j < robot_cols_; ++j)
      {
        if (!checkWheel(id, i, j) && std::isfinite(model.Z_(i, j)))
        {
          min_support_z = std::min(min_support_z, model.Z_(i, j));
        }
      }
    }

    int nominal_contacts = 0;
    if (std::isfinite(min_support_z))
    {
      for (int i = 0; i < robot_rows_; ++i)
      {
        for (int j = 0; j < robot_cols_; ++j)
        {
          if (!checkWheel(id, i, j) &&
              std::isfinite(model.Z_(i, j)) &&
              model.Z_(i, j) <=
                  min_support_z + capability.touch_gap_threshold)
          {
            ++nominal_contacts;
          }
        }
      }
    }
    context.nominal_contact_points = std::max(1, nominal_contacts);
    vehicles.push_back(context);
  };

  registerVehicle(1, wheeled_model_, wheeled_model_loaded_,
                  "traversability_coarse_wheeled",
                  "traversability_fine_wheeled", "wheeled car",
                  wheeled_capability);
  registerVehicle(2, tracked_model_, tracked_model_loaded_,
                  "traversability_coarse_tracked",
                  "traversability_fine_tracked", "tracked car",
                  tracked_capability);

  if (vehicles.empty())
  {
    ROS_WARN_THROTTLE(5.0,
                      "Fine traversability skipped: no vehicle models");
    return;
  }

  static bool csv_configured = false;
  static bool csv_enabled = false;
  static bool csv_append = false;
  static int csv_max_cycles = 50;
  static std::string csv_path;
  static std::ofstream csv_stream;
  static std::size_t csv_cycle = 0;

  if (!csv_configured)
  {
    pnh_.param<bool>("enable_fine_debug_csv", csv_enabled, false);
    pnh_.param<bool>("fine_debug_csv_append", csv_append, false);
    pnh_.param<int>("fine_debug_csv_max_cycles", csv_max_cycles, 50);

    const std::string package_path =
        ros::package::getPath("obstacle_mapping");
    const std::string default_csv_path =
        package_path + "/fine_traversability_debug.csv";
    pnh_.param<std::string>("fine_debug_csv_path", csv_path,
                            default_csv_path);
    if (csv_path.empty())
    {
      csv_path = default_csv_path;
    }
    csv_max_cycles = std::max(1, csv_max_cycles);

    if (csv_enabled)
    {
      bool write_header = true;
      if (csv_append)
      {
        std::ifstream existing(csv_path);
        write_header = !existing.good() ||
                       existing.peek() == std::ifstream::traits_type::eof();
      }

      const std::ios_base::openmode mode =
          std::ios::out |
          (csv_append ? std::ios::app : std::ios::trunc);
      csv_stream.open(csv_path, mode);

      if (!csv_stream.is_open())
      {
        ROS_ERROR("Failed to open fine traversability CSV: %s",
                  csv_path.c_str());
        csv_enabled = false;
      }
      else
      {
        if (write_header)
        {
          csv_stream
              << "record_type,stamp_sec,cycle,vehicle_id,vehicle,"
              << "cell_x,cell_y,slope,roughness,step,coarse_cost,"
              << "n_points,interpolated,heading_deg,pose_status,status,"
              << "roll_deg,pitch_deg,contact_points,stable,"
              << "raw_slope_deg,raw_roughness,raw_step_m,"
              << "platform_geo_cost,pose_cost,support_ratio,"
              << "success_headings,terrain_severity,efficiency_adjustment,"
              << "final_cost,success_count,collision_count,"
              << "failure_count,skipped_cells\n";
        }
        csv_stream << std::setprecision(9);
        ROS_INFO("Fine traversability CSV: %s (max_cycles=%d)",
                 csv_path.c_str(), csv_max_cycles);
      }
    }
    csv_configured = true;
  }

  ++csv_cycle;
  const bool record_this_cycle =
      csv_enabled && csv_stream.is_open() &&
      csv_cycle <= static_cast<std::size_t>(csv_max_cycles);
  const double csv_stamp = ros::Time::now().toSec();

  auto writeCellRecord = [&](const VehicleEvalContext &vehicle,
                             const grid_map::Position &position,
                             float slope_value,
                             float roughness_value,
                             float step_value,
                             float coarse_value,
                             float n_points_value,
                             float interpolated_value,
                             double heading_deg,
                             int pose_status,
                             const char *status,
                             double roll_deg,
                             double pitch_deg,
                             int contact_points,
                             int stable,
                             double raw_slope_deg,
                             double raw_roughness,
                             double raw_step_m,
                             double platform_geo_cost,
                             double pose_cost,
                             double support_ratio,
                             int success_headings,
                             double terrain_severity,
                             double efficiency_adjustment,
                             double final_cost)
  {
    if (!record_this_cycle)
    {
      return;
    }

    csv_stream << "cell," << csv_stamp << ',' << csv_cycle << ','
               << vehicle.id << ',' << vehicle.name << ','
               << position.x() << ',' << position.y() << ','
               << slope_value << ',' << roughness_value << ','
               << step_value << ',' << coarse_value << ','
               << n_points_value << ',' << interpolated_value << ','
               << heading_deg << ',' << pose_status << ',' << status << ','
               << roll_deg << ',' << pitch_deg << ','
               << contact_points << ',' << stable << ','
               << raw_slope_deg << ',' << raw_roughness << ','
               << raw_step_m << ',' << platform_geo_cost << ','
               << pose_cost << ',' << support_ratio << ','
               << success_headings << ',' << terrain_severity << ','
               << efficiency_adjustment << ',' << final_cost << ",,,,\n";
  };

  Eigen::Vector3d current_pos;
  Eigen::Quaterniond current_orientation;
  {
    std::lock_guard<std::mutex> pose_lock(vehicle_pose_mutex_);
    current_pos = vehicle_position_;
    current_orientation = vehicle_orientation_;
  }

  tf::Quaternion q_vehicle(current_orientation.x(),
                           current_orientation.y(),
                           current_orientation.z(),
                           current_orientation.w());
  const double yaw_angle = tf::getYaw(q_vehicle);

  static bool fine_pose_configured = false;
  static int fine_heading_samples = 4;
  if (!fine_pose_configured)
  {
    pnh_.param<int>("fine_heading_samples", fine_heading_samples, 4);
    fine_heading_samples = std::max(1, fine_heading_samples);
    fine_pose_configured = true;
    ROS_INFO("Fine traversability heading samples: %d",
             fine_heading_samples);
  }

  const Eigen::Vector3d z_axis(0.0, 0.0, 1.0);
  std::vector<double> fine_headings(
      static_cast<std::size_t>(fine_heading_samples));
  std::vector<Eigen::Matrix3d> fine_heading_rotations(
      static_cast<std::size_t>(fine_heading_samples));
  for (int heading_index = 0;
       heading_index < fine_heading_samples; ++heading_index)
  {
    const double heading =
        fine_heading_samples == 1
            ? yaw_angle
            : M_PI * static_cast<double>(heading_index) /
                  static_cast<double>(fine_heading_samples);
    fine_headings[static_cast<std::size_t>(heading_index)] = heading;
    fine_heading_rotations[static_cast<std::size_t>(heading_index)] =
        Eigen::AngleAxisd(heading, z_axis).toRotationMatrix();
  }

  int count_skipped = 0;
  int count_cached = 0;

  struct PlatformCellMetrics
  {
    double slope_ratio = 0.0;
    double roughness_ratio = 0.0;
    double step_ratio = 0.0;
    double geo_cost = 0.0;
    double efficiency_adjustment = 0.0;
    double base_cost = 0.0;
    bool critical = false;
  };

  auto inBand = [](double value, double minimum, double maximum)
  {
    return value >= minimum && value <= maximum;
  };

  for (grid_map::CircleIterator it(
           height_map_,
           grid_map::Position(current_pos.x(), current_pos.y()),
           max_range_);
       !it.isPastEnd(); ++it)
  {
    const grid_map::Index idx = *it;

    // 高程及其车辆覆盖邻域没有变化时，直接沿用上一次
    // 精细化校核结果。这是保持结果一致的增量优化，
    // 不是简单的降频或丢弃关键区。
    if (incremental_fine_computed(idx(0), idx(1)) > 0.5f)
    {
      ++count_cached;
      continue;
    }

    critical_layer(idx(0), idx(1)) = 0.0f;

    const float coarse_value = traversability(idx(0), idx(1));
    const float slope_value = slope_layer(idx(0), idx(1));
    const float roughness_value = roughness_layer(idx(0), idx(1));
    const float step_value = step_layer(idx(0), idx(1));
    const float raw_slope_deg = slope_deg_layer(idx(0), idx(1));
    const float raw_roughness = roughness_raw_layer(idx(0), idx(1));
    const float raw_step_m = step_height_layer(idx(0), idx(1));
    const float n_points_value = n_points_layer(idx(0), idx(1));
    const float interpolated_value = interpolated_layer(idx(0), idx(1));

    if (!std::isfinite(coarse_value) ||
        !std::isfinite(slope_value) ||
        !std::isfinite(roughness_value) ||
        !std::isfinite(step_value) ||
        !std::isfinite(raw_slope_deg) ||
        !std::isfinite(raw_roughness) ||
        !std::isfinite(raw_step_m))
    {
      for (auto &vehicle : vehicles)
      {
        (*(vehicle.coarse_layer))(idx(0), idx(1)) =
            std::numeric_limits<float>::quiet_NaN();
        (*(vehicle.fine_layer))(idx(0), idx(1)) =
            std::numeric_limits<float>::quiet_NaN();
      }
      ++count_skipped;
      continue;
    }

    // Remove the configurable measurement/interpolation tolerance before
    // calculating continuous platform step risk.  The raw value is retained
    // for diagnostics and for the hard capability comparison below.
    const double effective_step_m = std::max(
        0.0,
        static_cast<double>(raw_step_m) - step_config.noise_margin_m);

    const double slope_severity =
        clamp01(raw_slope_deg / efficiency_slope_reference_deg);
    const double roughness_severity =
        clamp01(raw_roughness / efficiency_roughness_reference);
    const double step_severity =
        clamp01(effective_step_m / efficiency_step_reference_m);
    const double terrain_severity = clamp01(
        severity_slope_weight * slope_severity +
        severity_roughness_weight * roughness_severity +
        severity_step_weight * step_severity);

    // 最多只有轮式和履带式两种平台，用栈上定长数组
    // 避免对每个格子进行一次 vector 堆分配。
    std::array<PlatformCellMetrics, 2> metrics;
    bool any_platform_critical = false;

    for (std::size_t vehicle_index = 0;
         vehicle_index < vehicles.size(); ++vehicle_index)
    {
      const VehicleCapability &capability =
          *vehicles[vehicle_index].capability;
      PlatformCellMetrics &cell_metrics = metrics[vehicle_index];

      cell_metrics.slope_ratio =
          raw_slope_deg / capability.slope_limit_deg;
      cell_metrics.roughness_ratio =
          raw_roughness / capability.roughness_limit;
      cell_metrics.step_ratio =
          effective_step_m / capability.step_limit_m;

      const bool step_capability_exceeded =
          static_cast<double>(raw_step_m) >=
          capability.step_limit_m + step_config.noise_margin_m;

      const bool capability_exceeded =
          cell_metrics.slope_ratio >= 1.0 ||
          cell_metrics.roughness_ratio >= 1.0 ||
          step_capability_exceeded;

      if (capability_exceeded)
      {
        cell_metrics.geo_cost = 1.0;
      }
      else
      {
        cell_metrics.geo_cost = clamp01(
            capability.geo_slope_weight *
                clamp01(cell_metrics.slope_ratio) +
            capability.geo_roughness_weight *
                clamp01(cell_metrics.roughness_ratio) +
            capability.geo_step_weight *
                clamp01(cell_metrics.step_ratio));
      }

      // d=0 表示平坦地形，d=1 表示达到效率参考上限的复杂地形。
      // 修正量允许为负：轮式在复杂地形加罚，履带在复杂地形获益。
      // 但最终结果始终不低于平台几何代价，因此不会降低硬障碍。
      cell_metrics.efficiency_adjustment = efficiency_weight * (capability.efficiency_flat_penalty *
                                                                    (1.0 - terrain_severity) +
                                                                capability.efficiency_rough_penalty * terrain_severity -
                                                                capability.efficiency_rough_bonus * terrain_severity);

      const double adjusted_geometry_cost = clamp01(
          cell_metrics.geo_cost +
          cell_metrics.efficiency_adjustment);
      cell_metrics.base_cost =
          cell_metrics.geo_cost >= 1.0
              ? 1.0
              : std::max(cell_metrics.geo_cost,
                         adjusted_geometry_cost);

      cell_metrics.critical =
          inBand(cell_metrics.geo_cost,
                 fine_trav_min_, fine_trav_max_) ||
          inBand(cell_metrics.slope_ratio,
                 fine_slope_min_, fine_slope_max_) ||
          inBand(cell_metrics.roughness_ratio,
                 fine_roughness_min_, fine_roughness_max_) ||
          inBand(cell_metrics.step_ratio,
                 fine_step_min, fine_step_max);

      // Preserve the platform-specific coarse result before the expensive
      // pose/support/collision verification.  The fine layer starts from the
      // same value and may be overwritten below, while the coarse layer must
      // remain unchanged for comparison and downstream cost selection.
      (*(vehicles[vehicle_index].coarse_layer))(idx(0), idx(1)) =
          static_cast<float>(cell_metrics.base_cost);
      (*(vehicles[vehicle_index].fine_layer))(idx(0), idx(1)) =
          static_cast<float>(cell_metrics.base_cost);
      any_platform_critical =
          any_platform_critical || cell_metrics.critical;
    }

    if (!any_platform_critical)
    {
      ++count_skipped;
      incremental_fine_computed(idx(0), idx(1)) = 1.0f;
      continue;
    }

    grid_map::Position position;
    if (!height_map_.getPosition(idx, position))
    {
      ++count_skipped;
      continue;
    }
    critical_layer(idx(0), idx(1)) = 1.0f;

    // 关键集的并集只用于标记 critical 层。对于某一具体
    // 平台，只有它自身的几何代价落入精细化区间时才执行
    // 昂贵的整车碰撞/支撑校核。非关键平台保留上面已写入的
    // base_cost，不再因另一种平台为 critical 而被连带重算。
    for (std::size_t vehicle_index = 0;
         vehicle_index < vehicles.size(); ++vehicle_index)
    {
      VehicleEvalContext &vehicle = vehicles[vehicle_index];
      const VehicleCapability &capability = *vehicle.capability;
      const PlatformCellMetrics &cell_metrics = metrics[vehicle_index];

      if (!cell_metrics.critical)
      {
        continue;
      }

      bool has_success = false;
      bool has_unstable = false;
      int collision_headings = 0;
      int success_headings = 0;

      double best_partial_pose_cost =
          std::numeric_limits<double>::infinity();
      double best_roll = 0.0;
      double best_pitch = 0.0;
      double best_heading_deg = 0.0;
      int best_contact_points = 0;
      int best_stable = 0;
      double best_support_ratio = 0.0;

      double diagnostic_roll = 0.0;
      double diagnostic_pitch = 0.0;
      double diagnostic_heading_deg = 0.0;
      int diagnostic_contacts = 0;
      int diagnostic_stable = 0;
      int diagnostic_status = kPoseFailure;

      for (int heading_index = 0;
           heading_index < fine_heading_samples; ++heading_index)
      {
        const double heading =
            fine_headings[static_cast<std::size_t>(heading_index)];
        const double heading_deg = heading * 180.0 / M_PI;
        const Eigen::Matrix3d &heading_rotation =
            fine_heading_rotations[static_cast<std::size_t>(heading_index)];

        double roll = 0.0;
        double pitch = 0.0;
        int contact_points = 0;
        int is_stable = 0;
        const int pose_status = predictRobotPose(
            idx, roll, pitch, contact_points, is_stable,
            heading_rotation, *vehicle.model, vehicle.id);

        if (pose_status == kPoseSuccess && is_stable != 0 &&
            std::isfinite(roll) && std::isfinite(pitch))
        {
          // 横滚或俯仰超过该车型能力时，该方向不可行，不能继续记为success。
          if (std::fabs(roll) > capability.roll_limit_deg ||
              std::fabs(pitch) > capability.pitch_limit_deg)
          {
            has_unstable = true;
            diagnostic_roll = roll;
            diagnostic_pitch = pitch;
            diagnostic_heading_deg = heading_deg;
            diagnostic_contacts = contact_points;
            diagnostic_stable = 0;
            diagnostic_status = kPoseUnstable;
            continue;
          }

          ++success_headings;
          const double roll_cost =
              clamp01(std::fabs(roll) / capability.roll_limit_deg);
          const double pitch_cost =
              clamp01(std::fabs(pitch) / capability.pitch_limit_deg);
          const double angle_cost = std::max(roll_cost, pitch_cost);
          const double support_ratio = clamp01(
              static_cast<double>(contact_points) /
              static_cast<double>(vehicle.nominal_contact_points));
          const double support_cost = 1.0 - support_ratio;
          const double partial_pose_cost =
              pose_angle_weight * angle_cost +
              pose_support_weight * support_cost;

          if (!has_success ||
              partial_pose_cost < best_partial_pose_cost)
          {
            has_success = true;
            best_partial_pose_cost = partial_pose_cost;
            best_roll = roll;
            best_pitch = pitch;
            best_heading_deg = heading_deg;
            best_contact_points = contact_points;
            best_stable = is_stable;
            best_support_ratio = support_ratio;
          }
        }
        else if (pose_status == kPoseCollision)
        {
          ++collision_headings;
          if (!has_unstable)
          {
            diagnostic_roll = roll;
            diagnostic_pitch = pitch;
            diagnostic_heading_deg = heading_deg;
            diagnostic_contacts = contact_points;
            diagnostic_stable = is_stable;
            diagnostic_status = pose_status;
          }
        }
        else if (pose_status == kPoseUnstable)
        {
          has_unstable = true;
          diagnostic_roll = roll;
          diagnostic_pitch = pitch;
          diagnostic_heading_deg = heading_deg;
          diagnostic_contacts = contact_points;
          diagnostic_stable = is_stable;
          diagnostic_status = pose_status;
        }
        else if (!has_unstable)
        {
          diagnostic_roll = roll;
          diagnostic_pitch = pitch;
          diagnostic_heading_deg = heading_deg;
          diagnostic_contacts = contact_points;
          diagnostic_stable = is_stable;
          diagnostic_status = kPoseFailure;
        }
      }

      if (has_success)
      {
        ++vehicle.count_success;

        const double heading_cost =
            1.0 -
            clamp01(
                static_cast<double>(success_headings) /
                static_cast<double>(fine_heading_samples));

        const double pose_cost =
            clamp01(
                best_partial_pose_cost +
                pose_heading_weight * heading_cost);

        // 精细姿态校核成功后覆盖车型粗略代价。
        // 但明确超过车辆坡度、粗糙度或台阶能力时仍保持不可通行。
        const double final_cost =
            cell_metrics.geo_cost >= 1.0
                ? 1.0
                : clamp01(
                      pose_cost +
                      cell_metrics.efficiency_adjustment);

        (*(vehicle.fine_layer))(idx(0), idx(1)) =
            static_cast<float>(final_cost);

        writeCellRecord(
            vehicle, position,
            slope_value, roughness_value, step_value, coarse_value,
            n_points_value, interpolated_value,
            best_heading_deg, kPoseSuccess, "success",
            best_roll, best_pitch, best_contact_points, best_stable,
            raw_slope_deg, raw_roughness, raw_step_m,
            cell_metrics.geo_cost, pose_cost, best_support_ratio,
            success_headings, terrain_severity,
            cell_metrics.efficiency_adjustment, final_cost);
      }
      else if (collision_headings == fine_heading_samples)
      {
        ++vehicle.count_collision;
        (*(vehicle.fine_layer))(idx(0), idx(1)) = 1.0f;
        const double support_ratio = clamp01(
            static_cast<double>(diagnostic_contacts) /
            static_cast<double>(vehicle.nominal_contact_points));

        writeCellRecord(
            vehicle, position,
            slope_value, roughness_value, step_value, coarse_value,
            n_points_value, interpolated_value,
            diagnostic_heading_deg, kPoseCollision,
            "collision_all_headings",
            diagnostic_roll, diagnostic_pitch,
            diagnostic_contacts, diagnostic_stable,
            raw_slope_deg, raw_roughness, raw_step_m,
            cell_metrics.geo_cost, 1.0, support_ratio, 0,
            terrain_severity, cell_metrics.efficiency_adjustment, 1.0);
      }
      else if (has_unstable)
      {
        ++vehicle.count_failure;
        (*(vehicle.fine_layer))(idx(0), idx(1)) = 1.0f;
        const double support_ratio = clamp01(
            static_cast<double>(diagnostic_contacts) /
            static_cast<double>(vehicle.nominal_contact_points));

        writeCellRecord(
            vehicle, position,
            slope_value, roughness_value, step_value, coarse_value,
            n_points_value, interpolated_value,
            diagnostic_heading_deg, kPoseUnstable, "unstable",
            diagnostic_roll, diagnostic_pitch,
            diagnostic_contacts, 0,
            raw_slope_deg, raw_roughness, raw_step_m,
            cell_metrics.geo_cost, 1.0, support_ratio,
            success_headings, terrain_severity,
            cell_metrics.efficiency_adjustment, 1.0);
      }
      else
      {
        ++vehicle.count_failure;
        (*(vehicle.fine_layer))(idx(0), idx(1)) =
            static_cast<float>(cell_metrics.base_cost);
        const double support_ratio = clamp01(
            static_cast<double>(diagnostic_contacts) /
            static_cast<double>(vehicle.nominal_contact_points));
        const double no_pose =
            std::numeric_limits<double>::quiet_NaN();

        writeCellRecord(
            vehicle, position,
            slope_value, roughness_value, step_value, coarse_value,
            n_points_value, interpolated_value,
            diagnostic_heading_deg, diagnostic_status,
            "failure_keep_platform_base",
            diagnostic_roll, diagnostic_pitch,
            diagnostic_contacts, diagnostic_stable,
            raw_slope_deg, raw_roughness, raw_step_m,
            cell_metrics.geo_cost, no_pose, support_ratio,
            success_headings, terrain_severity,
            cell_metrics.efficiency_adjustment, cell_metrics.base_cost);
      }
    }

    incremental_fine_computed(idx(0), idx(1)) = 1.0f;
  }

  ROS_INFO("Platform traversability statistics:");
  ROS_INFO("  Skipped cells: %d", count_skipped);
  ROS_INFO("  Reused cached fine cells: %d", count_cached);
  for (const auto &vehicle : vehicles)
  {
    ROS_INFO("  %s => success=%d, collision=%d, failure=%d, nominal_contacts=%d",
             vehicle.name.c_str(),
             vehicle.count_success,
             vehicle.count_collision,
             vehicle.count_failure,
             vehicle.nominal_contact_points);

    if (record_this_cycle)
    {
      csv_stream << "summary," << csv_stamp << ',' << csv_cycle << ','
                 << vehicle.id << ',' << vehicle.name;
      // Fields 6--30 are cell-level values.
      for (int field = 6; field <= 30; ++field)
      {
        csv_stream << ",nan";
      }
      csv_stream << ',' << vehicle.count_success
                 << ',' << vehicle.count_collision
                 << ',' << vehicle.count_failure
                 << ',' << count_skipped << '\n';
    }
  }

  if (record_this_cycle)
  {
    csv_stream.flush();
    if (csv_cycle == static_cast<std::size_t>(csv_max_cycles))
    {
      ROS_INFO("Fine traversability CSV reached %d cycles: %s",
               csv_max_cycles, csv_path.c_str());
    }
  }
}

// ==================== Robot Pose Prediction ====================
int Mapping::predictRobotPose(const grid_map::Index &center_idx,
                              double &roll,
                              double &pitch,
                              int &contact_points,
                              int &stable,
                              const Eigen::Matrix3d &yaw_rotation,
                              const HeightGrid &vehicle_model,
                              int vehicle_type)
{
  roll = 0.0;
  pitch = 0.0;
  contact_points = 0;
  stable = 0;

  static bool pose_limits_configured = false;
  static int fine_min_support_points = 3;
  static double fine_max_rotation_step_deg = 5.0;
  static double wheeled_max_pose_tilt_deg = 40.0;
  static double tracked_max_pose_tilt_deg = 45.0;
  static double wheeled_touch_gap_threshold = 0.05;
  static double tracked_touch_gap_threshold = 0.05;
  static double wheeled_collision_clearance = 0.10;
  static double tracked_collision_clearance = 0.08;

  if (!pose_limits_configured)
  {
    pnh_.param<int>("fine_min_support_points",
                    fine_min_support_points, 3);

    pnh_.param<double>("fine_max_rotation_step_deg",
                       fine_max_rotation_step_deg, 5.0);

    pnh_.param<double>("wheeled_max_pose_tilt_deg",
                       wheeled_max_pose_tilt_deg, 40.0);
    pnh_.param<double>("tracked_max_pose_tilt_deg",
                       tracked_max_pose_tilt_deg, 45.0);
    pnh_.param<double>("wheeled_touch_gap_threshold",
                       wheeled_touch_gap_threshold, touch_gap_threshold_);
    pnh_.param<double>("tracked_touch_gap_threshold",
                       tracked_touch_gap_threshold, touch_gap_threshold_);
    pnh_.param<double>("wheeled_collision_clearance",
                       wheeled_collision_clearance, collision_gap_threshold_);
    pnh_.param<double>("tracked_collision_clearance",
                       tracked_collision_clearance, collision_gap_threshold_);

    fine_min_support_points =
        std::max(3, fine_min_support_points);

    fine_max_rotation_step_deg =
        std::max(0.1,
                 std::min(30.0,
                          fine_max_rotation_step_deg));

    wheeled_max_pose_tilt_deg =
        std::max(1.0, std::min(89.0, wheeled_max_pose_tilt_deg));
    tracked_max_pose_tilt_deg =
        std::max(1.0, std::min(89.0, tracked_max_pose_tilt_deg));
    wheeled_touch_gap_threshold =
        std::max(1e-4, wheeled_touch_gap_threshold);
    tracked_touch_gap_threshold =
        std::max(1e-4, tracked_touch_gap_threshold);
    wheeled_collision_clearance =
        std::max(0.0, wheeled_collision_clearance);
    tracked_collision_clearance =
        std::max(0.0, tracked_collision_clearance);

    pose_limits_configured = true;
  }

  const double pose_max_tilt_deg =
      vehicle_type == 2 ? tracked_max_pose_tilt_deg
                        : wheeled_max_pose_tilt_deg;
  const double pose_touch_gap_threshold =
      vehicle_type == 2 ? tracked_touch_gap_threshold
                        : wheeled_touch_gap_threshold;
  const double pose_collision_clearance =
      vehicle_type == 2 ? tracked_collision_clearance
                        : wheeled_collision_clearance;

  const double vehicle_width =
      static_cast<double>(robot_rows_) *
      robot_model_resolution_;

  const double vehicle_length =
      static_cast<double>(robot_cols_) *
      robot_model_resolution_;

  const double submap_side =
      std::sqrt(vehicle_width * vehicle_width +
                vehicle_length * vehicle_length);

  struct TerrainPatchCache
  {
    const Mapping *owner = nullptr;
    std::size_t generation = 0;
    grid_map::Index center_idx;
    grid_map::Position center_pos;
    grid_map::GridMap submap;
    Eigen::Vector3d terrain_normal = Eigen::Vector3d::UnitZ();
    bool valid = false;
  };

  static thread_local TerrainPatchCache terrain_cache;

  const bool cache_hit =
      terrain_cache.valid &&
      terrain_cache.owner == this &&
      terrain_cache.generation == fine_cycle_generation &&
      terrain_cache.center_idx(0) == center_idx(0) &&
      terrain_cache.center_idx(1) == center_idx(1);

  if (!cache_hit)
  {
    terrain_cache.valid = false;
    terrain_cache.owner = this;
    terrain_cache.generation = fine_cycle_generation;
    terrain_cache.center_idx = center_idx;

    if (!height_map_.getPosition(center_idx,
                                 terrain_cache.center_pos))
    {
      return kPoseFailure;
    }

    bool submap_ok = false;
    terrain_cache.submap = height_map_.getSubmap(
        terrain_cache.center_pos,
        grid_map::Length(submap_side, submap_side),
        submap_ok);

    if (!submap_ok)
    {
      return kPoseFailure;
    }

    // 该子图与初始地形平面只与中心格子有关，与车型和
    // 航向无关，因此在 2 种车型 x 4 个航向之间共享。
    std::vector<Eigen::Vector3d> terrain_points;
    terrain_points.reserve(64);

    for (grid_map::GridMapIterator it(terrain_cache.submap);
         !it.isPastEnd(); ++it)
    {
      const grid_map::Index idx = *it;
      const float z =
          terrain_cache.submap.at("elevation_BGK", idx);

      if (!std::isfinite(z))
      {
        continue;
      }

      grid_map::Position pos;
      if (terrain_cache.submap.getPosition(idx, pos))
      {
        terrain_points.emplace_back(
            pos.x(), pos.y(), static_cast<double>(z));
      }
    }

    if (terrain_points.size() < 4)
    {
      return kPoseFailure;
    }

    terrain_cache.terrain_normal = fitPlane(terrain_points);
    if (!terrain_cache.terrain_normal.allFinite() ||
        terrain_cache.terrain_normal.norm() < 1e-6)
    {
      return kPoseFailure;
    }

    terrain_cache.terrain_normal.normalize();
    if (terrain_cache.terrain_normal.z() < 0.0)
    {
      terrain_cache.terrain_normal =
          -terrain_cache.terrain_normal;
    }

    terrain_cache.valid = true;
  }

  const grid_map::Position &center_pos =
      terrain_cache.center_pos;
  const grid_map::GridMap &submap = terrain_cache.submap;
  const Eigen::Vector3d &terrain_normal =
      terrain_cache.terrain_normal;

  const Eigen::Vector3d z_axis(0.0, 0.0, 1.0);
  const Eigen::Vector3d gravity(0.0, 0.0, -1.0);

  Eigen::Matrix3d terrain_rotation =
      Eigen::Matrix3d::Identity();

  const Eigen::Vector3d terrain_axis =
      z_axis.cross(terrain_normal);

  if (terrain_axis.norm() > 1e-6)
  {
    const double cosine =
        std::max(
            -1.0,
            std::min(
                1.0,
                z_axis.dot(terrain_normal)));

    terrain_rotation =
        Eigen::AngleAxisd(
            std::acos(cosine),
            terrain_axis.normalized())
            .toRotationMatrix();
  }

  // =====================================================
  // 2. 初始化车辆齐次变换
  // =====================================================
  Eigen::Matrix4d T_matrix =
      Eigen::Matrix4d::Identity();

  T_matrix.block<3, 3>(0, 0) =
      terrain_rotation * yaw_rotation;

  T_matrix.block<3, 1>(0, 3) =
      Eigen::Vector3d(
          center_pos.x(),
          center_pos.y(),
          0.0);

  auto updateRollPitchFromT =
      [&](const Eigen::Matrix4d &T)
  {
    const Eigen::Matrix3d R =
        T.block<3, 3>(0, 0);

    roll =
        std::atan2(R(2, 1), R(2, 2)) *
        180.0 / M_PI;

    const double value =
        std::max(
            -1.0,
            std::min(1.0, -R(2, 0)));

    pitch =
        std::asin(value) *
        180.0 / M_PI;
  };

  auto exceedsTiltLimit =
      [&](const Eigen::Matrix4d &T)
  {
    if (!T.allFinite())
    {
      return true;
    }

    const Eigen::Vector3d body_up =
        T.block<3, 3>(0, 0).col(2);

    if (!body_up.allFinite() ||
        body_up.norm() < 1e-6)
    {
      return true;
    }

    const double max_tilt_rad =
        pose_max_tilt_deg *
        M_PI / 180.0;

    return body_up.normalized().z() <
           std::cos(max_tilt_rad);
  };

  auto cross2d =
      [](const Eigen::Vector2d &origin,
         const Eigen::Vector2d &a,
         const Eigen::Vector2d &b)
  {
    return (a.x() - origin.x()) *
               (b.y() - origin.y()) -
           (a.y() - origin.y()) *
               (b.x() - origin.x());
  };

  auto computeHull =
      [&](std::vector<Eigen::Vector2d> points)
  {
    std::sort(
        points.begin(),
        points.end(),
        [](const Eigen::Vector2d &a,
           const Eigen::Vector2d &b)
        {
          if (std::fabs(a.x() - b.x()) < 1e-12)
          {
            return a.y() < b.y();
          }

          return a.x() < b.x();
        });

    points.erase(
        std::unique(
            points.begin(),
            points.end(),
            [](const Eigen::Vector2d &a,
               const Eigen::Vector2d &b)
            {
              return (a - b).squaredNorm() <
                     1e-12;
            }),
        points.end());

    if (points.size() <= 1)
    {
      return points;
    }

    std::vector<Eigen::Vector2d> hull;

    for (const auto &point : points)
    {
      while (
          hull.size() >= 2 &&
          cross2d(
              hull[hull.size() - 2],
              hull.back(),
              point) <= 1e-12)
      {
        hull.pop_back();
      }

      hull.push_back(point);
    }

    const std::size_t lower_size =
        hull.size();

    for (int i =
             static_cast<int>(points.size()) - 2;
         i >= 0;
         --i)
    {
      const auto &point =
          points[static_cast<std::size_t>(i)];

      while (
          hull.size() > lower_size &&
          cross2d(
              hull[hull.size() - 2],
              hull.back(),
              point) <= 1e-12)
      {
        hull.pop_back();
      }

      hull.push_back(point);
    }

    if (!hull.empty())
    {
      hull.pop_back();
    }

    return hull;
  };

  auto pointInConvexHull =
      [&](const std::vector<Eigen::Vector2d> &hull,
          const Eigen::Vector2d &point)
  {
    if (hull.size() < 3)
    {
      return false;
    }

    for (std::size_t i = 0;
         i < hull.size();
         ++i)
    {
      if (cross2d(
              hull[i],
              hull[(i + 1) % hull.size()],
              point) < -1e-9)
      {
        return false;
      }
    }

    return true;
  };

  // =====================================================
  // 3. 支撑姿态迭代
  // =====================================================
  for (int iteration = 0;
       iteration < max_iterations_;
       ++iteration)
  {
    Eigen::MatrixXd gap_map(
        robot_rows_,
        robot_cols_);

    gap_map.setConstant(
        std::numeric_limits<double>::quiet_NaN());

    double min_gap =
        std::numeric_limits<double>::infinity();

    // ---------------------------------------------------
    // 3.1 计算当前车辆模型所有点到地形的间隙
    // ---------------------------------------------------
    for (int i = 0;
         i < robot_rows_;
         ++i)
    {
      for (int j = 0;
           j < robot_cols_;
           ++j)
      {
        const Eigen::Vector4d model_point(
            vehicle_model.X_(i, j),
            vehicle_model.Y_(i, j),
            vehicle_model.Z_(i, j),
            1.0);

        const Eigen::Vector4d world_point =
            T_matrix * model_point;

        grid_map::Index terrain_idx;

        if (!submap.getIndex(
                grid_map::Position(
                    world_point.x(),
                    world_point.y()),
                terrain_idx))
        {
          continue;
        }

        const float terrain_z =
            submap.at(
                "elevation_BGK",
                terrain_idx);

        if (!std::isfinite(terrain_z))
        {
          continue;
        }

        const double gap =
            world_point.z() -
            static_cast<double>(terrain_z);

        gap_map(i, j) = gap;
        min_gap = std::min(min_gap, gap);
      }
    }

    if (!std::isfinite(min_gap))
    {
      return kPoseFailure;
    }

    // 整车竖直落地，使最低模型点接触地形。
    T_matrix(2, 3) -= min_gap;

    // ---------------------------------------------------
    // 3.2 落地后重新查找支撑点和底盘碰撞
    // ---------------------------------------------------
    std::vector<Eigen::Vector3d>
        support_points_world;

    std::vector<Eigen::Vector2d>
        support_points_model;

    contact_points = 0;
    bool collision = false;

    for (int i = 0;
         i < robot_rows_;
         ++i)
    {
      for (int j = 0;
           j < robot_cols_;
           ++j)
      {
        const Eigen::Vector4d model_point(
            vehicle_model.X_(i, j),
            vehicle_model.Y_(i, j),
            vehicle_model.Z_(i, j),
            1.0);

        const Eigen::Vector4d world_point =
            T_matrix * model_point;

        grid_map::Index terrain_idx;

        if (!submap.getIndex(
                grid_map::Position(
                    world_point.x(),
                    world_point.y()),
                terrain_idx))
        {
          continue;
        }

        const float terrain_z =
            submap.at(
                "elevation_BGK",
                terrain_idx);

        if (!std::isfinite(terrain_z))
        {
          continue;
        }

        const double gap =
            world_point.z() -
            static_cast<double>(terrain_z);

        gap_map(i, j) = gap;

        // false：车轮或履带点
        // true ：底盘碰撞敏感点
        const bool collision_sensitive =
            checkWheel(vehicle_type, i, j);

        if (!collision_sensitive &&
            gap <= pose_touch_gap_threshold)
        {
          ++contact_points;

          // 必须使用模型点的实际世界位置作为支撑轴点。
          support_points_world.emplace_back(
              world_point.x(),
              world_point.y(),
              world_point.z());

          support_points_model.emplace_back(
              vehicle_model.X_(i, j),
              vehicle_model.Y_(i, j));
        }

        if (collision_sensitive &&
            gap < pose_collision_clearance)
        {
          collision = true;
        }
      }
    }

    if (collision)
    {
      stable = 0;
      updateRollPitchFromT(T_matrix);
      return kPoseCollision;
    }

    if (support_points_world.empty())
    {
      return kPoseFailure;
    }

    // ---------------------------------------------------
    // 3.3 根据接触点形成点、线或面支撑
    // ---------------------------------------------------
    const std::vector<Eigen::Vector2d>
        hull_model =
            computeHull(support_points_model);

    if (hull_model.empty())
    {
      return kPoseFailure;
    }

    std::vector<Eigen::Vector3d> hull_world;
    hull_world.reserve(hull_model.size());

    for (const auto &hull_point : hull_model)
    {
      bool found = false;

      for (std::size_t k = 0;
           k < support_points_model.size();
           ++k)
      {
        if ((support_points_model[k] -
             hull_point)
                .squaredNorm() < 1e-12)
        {
          hull_world.push_back(
              support_points_world[k]);

          found = true;
          break;
        }
      }

      if (!found)
      {
        return kPoseFailure;
      }
    }

    Eigen::Vector3d rotate_point_1 =
        Eigen::Vector3d::Zero();

    Eigen::Vector3d rotate_point_2 =
        Eigen::Vector3d::Zero();

    bool has_rotation_axis = false;

    double hull_area_twice = 0.0;

    if (hull_model.size() >= 3)
    {
      for (std::size_t i = 0;
           i < hull_model.size();
           ++i)
      {
        const Eigen::Vector2d &p1 =
            hull_model[i];

        const Eigen::Vector2d &p2 =
            hull_model[(i + 1) %
                       hull_model.size()];

        hull_area_twice +=
            p1.x() * p2.y() -
            p2.x() * p1.y();
      }
    }

    const bool has_support_area =
        hull_model.size() >= 3 &&
        std::isfinite(hull_area_twice) &&
        std::fabs(hull_area_twice) >= 2e-4;

    // ===================================================
    // 4. 点支撑或者线支撑
    // ===================================================
    if (!has_support_area)
    {
      // 点支撑：围绕经过接触点的水平轴转动。
      if (hull_world.size() == 1)
      {
        const Eigen::Vector3d center_world =
            T_matrix.block<3, 1>(0, 3);

        Eigen::Vector3d contact_to_center(
            center_world.x() -
                hull_world[0].x(),
            center_world.y() -
                hull_world[0].y(),
            0.0);

        if (!contact_to_center.allFinite() ||
            contact_to_center.norm() < 1e-6)
        {
          stable = 0;
          return kPoseUnstable;
        }

        const Eigen::Vector3d point_axis =
            contact_to_center.cross(gravity);

        if (!point_axis.allFinite() ||
            point_axis.norm() < 1e-6)
        {
          stable = 0;
          return kPoseUnstable;
        }

        rotate_point_1 = hull_world[0];

        rotate_point_2 =
            rotate_point_1 +
            point_axis.normalized();

        has_rotation_axis = true;
      }
      else
      {
        // 两点或多个共线点：
        // 使用距离最远的两个点构造支撑轴。
        double max_distance_squared = -1.0;

        for (std::size_t i = 0;
             i < hull_world.size();
             ++i)
        {
          for (std::size_t j = i + 1;
               j < hull_world.size();
               ++j)
          {
            const double distance_squared =
                (hull_world[j] -
                 hull_world[i])
                    .squaredNorm();

            if (std::isfinite(distance_squared) &&
                distance_squared >
                    max_distance_squared)
            {
              max_distance_squared =
                  distance_squared;

              rotate_point_1 =
                  hull_world[i];

              rotate_point_2 =
                  hull_world[j];
            }
          }
        }

        if (max_distance_squared < 1e-12)
        {
          return kPoseFailure;
        }

        has_rotation_axis = true;
      }
    }
    // ===================================================
    // 5. 面支撑
    // ===================================================
    else
    {
      Eigen::Vector3d support_normal =
          fitPlane(hull_world);

      if (!support_normal.allFinite() ||
          support_normal.norm() < 1e-6)
      {
        return kPoseFailure;
      }

      support_normal.normalize();

      if (support_normal.z() < 0.0)
      {
        support_normal = -support_normal;
      }

      if (std::fabs(
              support_normal.dot(gravity)) <
          1e-6)
      {
        stable = 0;
        return kPoseUnstable;
      }

      const Eigen::Vector3d center_world =
          T_matrix.block<3, 1>(0, 3);

      const Eigen::ParametrizedLine<double, 3>
          gravity_line(
              center_world,
              gravity);

      const Eigen::Hyperplane<double, 3>
          support_plane(
              support_normal,
              hull_world.front());

      const Eigen::Vector3d
          gravity_projection_world =
              gravity_line.intersectionPoint(
                  support_plane);

      if (!gravity_projection_world.allFinite())
      {
        return kPoseFailure;
      }

      // 支撑凸包在车辆模型坐标系中建立，
      // 因此重力投影也必须转换到模型坐标系判断。
      const Eigen::Vector4d projection_world_h(
          gravity_projection_world.x(),
          gravity_projection_world.y(),
          gravity_projection_world.z(),
          1.0);

      const Eigen::Vector4d projection_model_h =
          T_matrix.inverse() *
          projection_world_h;

      if (!projection_model_h.allFinite())
      {
        return kPoseFailure;
      }

      const Eigen::Vector2d projection_model(
          projection_model_h.x(),
          projection_model_h.y());

      if (contact_points >=
              fine_min_support_points &&
          pointInConvexHull(
              hull_model,
              projection_model))
      {
        updateRollPitchFromT(T_matrix);

        if (exceedsTiltLimit(T_matrix))
        {
          stable = 0;
          return kPoseUnstable;
        }

        stable = 1;
        return kPoseSuccess;
      }

      // 重力投影位于支撑面外：
      // 查找距离投影最近的外侧凸包边。
      double nearest_distance_squared =
          std::numeric_limits<double>::infinity();

      for (std::size_t i = 0;
           i < hull_model.size();
           ++i)
      {
        const std::size_t next =
            (i + 1) %
            hull_model.size();

        const Eigen::Vector2d edge =
            hull_model[next] -
            hull_model[i];

        const double edge_length_squared =
            edge.squaredNorm();

        if (edge_length_squared < 1e-12)
        {
          continue;
        }

        // 凸包为逆时针顺序。
        // 叉积为负表示投影位于该边外侧。
        if (cross2d(
                hull_model[i],
                hull_model[next],
                projection_model) >= -1e-9)
        {
          continue;
        }

        double t =
            (projection_model -
             hull_model[i])
                .dot(edge) /
            edge_length_squared;

        t = std::max(
            0.0,
            std::min(1.0, t));

        const Eigen::Vector2d nearest =
            hull_model[i] +
            t * edge;

        const double distance_squared =
            (projection_model - nearest)
                .squaredNorm();

        if (distance_squared <
            nearest_distance_squared)
        {
          nearest_distance_squared =
              distance_squared;

          rotate_point_1 =
              hull_world[i];

          rotate_point_2 =
              hull_world[next];

          has_rotation_axis = true;
        }
      }
    }

    if (!has_rotation_axis)
    {
      return kPoseFailure;
    }

    // ===================================================
    // 6. 根据重力力矩确定旋转方向
    // ===================================================
    const Eigen::Vector3d axis =
        rotate_point_2 -
        rotate_point_1;

    if (!axis.allFinite() ||
        axis.norm() < 1e-6)
    {
      return kPoseFailure;
    }

    const Eigen::Vector3d axis_direction =
        axis.normalized();

    const Eigen::Vector3d center_world =
        T_matrix.block<3, 1>(0, 3);

    const double gravity_moment =
        axis_direction.dot(
            (center_world -
             rotate_point_1)
                .cross(gravity));

    if (!std::isfinite(gravity_moment) ||
        std::fabs(gravity_moment) < 1e-9)
    {
      stable = 0;
      updateRollPitchFromT(T_matrix);
      return kPoseUnstable;
    }

    const double rotation_sign =
        gravity_moment > 0.0
            ? 1.0
            : -1.0;

    // ===================================================
    // 7. 从车轮/履带区域中寻找下一个接地点
    // ===================================================
    double rotation_step =
        std::numeric_limits<double>::infinity();

    for (int i = 0;
         i < robot_rows_;
         ++i)
    {
      for (int j = 0;
           j < robot_cols_;
           ++j)
      {
        // checkWheel=true 表示底盘点。
        // 底盘点不能作为新的支撑接地点。
        if (checkWheel(vehicle_type, i, j))
        {
          continue;
        }

        const double gap =
            gap_map(i, j);

        if (!std::isfinite(gap) ||
            gap <= pose_touch_gap_threshold)
        {
          continue;
        }

        const Eigen::Vector4d model_point(
            vehicle_model.X_(i, j),
            vehicle_model.Y_(i, j),
            vehicle_model.Z_(i, j),
            1.0);

        const Eigen::Vector3d world_point =
            (T_matrix * model_point)
                .head<3>();

        const Eigen::Vector3d radius =
            world_point -
            rotate_point_1;

        // 对绕轴旋转：
        // dP/dtheta = axis × radius
        //
        // z方向速度小于0，说明这个点会向地面运动。
        const double vertical_rate =
            (rotation_sign *
             axis_direction.cross(radius))
                .z();

        if (!std::isfinite(vertical_rate) ||
            vertical_rate >= -1e-8)
        {
          continue;
        }

        const double remaining_gap =
            gap -
            pose_touch_gap_threshold;

        const double candidate_step =
            remaining_gap /
            (-vertical_rate);

        if (std::isfinite(candidate_step) &&
            candidate_step > 1e-8)
        {
          rotation_step =
              std::min(
                  rotation_step,
                  candidate_step);
        }
      }
    }

    if (!std::isfinite(rotation_step))
    {
      return kPoseFailure;
    }

    const double max_rotation_step =
        fine_max_rotation_step_deg *
        M_PI / 180.0;

    rotation_step =
        std::min(
            rotation_step,
            max_rotation_step);

    if (rotation_step < 1e-8)
    {
      return kPoseFailure;
    }

    // ===================================================
    // 8. 绕世界坐标系中的支撑轴旋转车辆
    // ===================================================
    const double signed_step =
        rotation_sign *
        rotation_step;

    const Eigen::Matrix3d incremental_rotation =
        Eigen::AngleAxisd(
            signed_step,
            axis_direction)
            .toRotationMatrix();

    Eigen::Matrix4d pivot_rotation =
        Eigen::Matrix4d::Identity();

    pivot_rotation.block<3, 3>(0, 0) =
        incremental_rotation;

    pivot_rotation.block<3, 1>(0, 3) =
        rotate_point_1 -
        incremental_rotation *
            rotate_point_1;

    T_matrix =
        pivot_rotation *
        T_matrix;

    // 不要执行下面这种旧逻辑：
    //
    // T_matrix.block<2, 1>(0, 3) =
    //     Eigen::Vector2d(center_pos.x(), center_pos.y());
    //
    // 因为它会破坏绕支撑轴旋转的刚体约束。

    if (!T_matrix.allFinite())
    {
      return kPoseFailure;
    }

    if (exceedsTiltLimit(T_matrix))
    {
      stable = 0;
      updateRollPitchFromT(T_matrix);
      return kPoseUnstable;
    }
  }

  // 达到最大迭代次数仍未形成稳定支撑面。
  updateRollPitchFromT(T_matrix);
  stable = 0;
  return kPoseFailure;
}

bool Mapping::checkWheel(int vehicle_type,
                         int i,
                         int j) const
{
  // 返回 false：车轮或履带接触区域。
  // 返回 true ：底盘区域，需要检查碰撞。

  if (vehicle_type == 1)
  {
    // 轮式车：左右边缘行与前后轮列的交集。
    static const std::array<int, 4>
        wheel_rows = {
            0, 1, 7, 8};

    static const std::array<int, 5>
        wheel_cols = {
            1, 2, 3, 8, 9};

    const bool in_wheel_row =
        std::find(
            wheel_rows.begin(),
            wheel_rows.end(),
            i) != wheel_rows.end();

    const bool in_wheel_col =
        std::find(
            wheel_cols.begin(),
            wheel_cols.end(),
            j) != wheel_cols.end();

    return !(in_wheel_row &&
             in_wheel_col);
  }

  if (vehicle_type == 2)
  {
    // 履带车：履带位于车辆宽度方向两侧，
    // 并沿车辆长度方向连续分布。
    //
    // 必须使用行 i 判断，不能使用列 j。
    static const std::array<int, 4>
        track_rows = {
            0, 1, 7, 8};

    const bool in_track_row =
        std::find(
            track_rows.begin(),
            track_rows.end(),
            i) != track_rows.end();

    return !in_track_row;
  }

  return true;
}

// int Mapping::predictRobotPose(const grid_map::Index &center_idx, double &roll, double &pitch,
//                               int &contact_points, int &stable, const Eigen::Matrix3d &yaw_rotation,
//                               const HeightGrid &vehicle_model, int vehicle_type)
// {
//   roll = 0.0;
//   pitch = 0.0;
//   contact_points = 0;
//   stable = 0;

//   static bool pose_limits_configured = false;
//   static int fine_min_support_points = 3;
//   static double fine_max_rotation_step_deg = 5.0;
//   static double fine_max_pose_tilt_deg = 60.0;
//   if (!pose_limits_configured)
//   {
//     pnh_.param<int>("fine_min_support_points", fine_min_support_points, 3);
//     pnh_.param<double>("fine_max_rotation_step_deg", fine_max_rotation_step_deg, 5.0);
//     pnh_.param<double>("fine_max_pose_tilt_deg", fine_max_pose_tilt_deg, 60.0);

//     fine_min_support_points = std::max(3, fine_min_support_points);
//     fine_max_rotation_step_deg = std::max(0.1, std::min(30.0, fine_max_rotation_step_deg));
//     fine_max_pose_tilt_deg = std::max(1.0, std::min(89.0, fine_max_pose_tilt_deg));
//     pose_limits_configured = true;
//   }

//   // -----------------------------------------
//   // [A1] 将中心 cell 索引转成连续坐标 (x,y)
//   // -----------------------------------------
//   // center_pos 是地图坐标系中的平面位置，用来获取子地图并作为车辆平移中心
//   grid_map::Position center_pos;
//   if (!height_map_.getPosition(center_idx, center_pos))
//   {
//     return kPoseFailure;
//   }

//   // -----------------------------------------
//   // [A2] 计算一个足够覆盖车辆的子地图范围（正方形）
//   // -----------------------------------------
//   // vehicle_model 是 robot_rows_ x robot_cols_ 的离散采样网格（车辆足迹/底盘采样点）
//   // robot_model_resolution_ 是模型网格的分辨率
//   //
//   // 这里把车辆占地宽/长估计出来，然后取对角线长度作为子地图边长（submap_side）
//   double vehicle_width = robot_rows_ * robot_model_resolution_;
//   double vehicle_length = robot_cols_ * robot_model_resolution_;
//   double submap_side = std::sqrt(vehicle_width * vehicle_width + vehicle_length * vehicle_length);
//   grid_map::Length submap_length(submap_side, submap_side);

//   bool success;
//   grid_map::GridMap submap = height_map_.getSubmap(center_pos, submap_length, success);

//   if (!success)
//   {
//     return kPoseFailure;
//   }

//   // -----------------------------------------
//   // [B1] 收集子地图内所有有效地形点，用于拟合一个局部平面
//   // -----------------------------------------
//   // terrain_points 用于 fitPlane()：得到局部地形法向 normal
//   std::vector<Eigen::Vector3d> terrain_points;
//   for (grid_map::GridMapIterator it(submap); !it.isPastEnd(); ++it)
//   {
//     grid_map::Index idx = *it;
//     float z = submap.at("elevation_BGK", idx);

//     // elevation_BGK 有 NaN 的格子跳过
//     if (!std::isnan(z))
//     {
//       grid_map::Position pos;
//       submap.getPosition(idx, pos);
//       terrain_points.push_back(Eigen::Vector3d(pos.x(), pos.y(), z));
//     }
//   }

//   // 点太少无法拟合平面
//   if (terrain_points.size() < 4)
//   {
//     return kPoseFailure;
//   }

//   // -----------------------------------------
//   // [B2] 拟合平面得到法向 normal（fitPlane 内部一般就是 PCA/最小二乘）
//   // -----------------------------------------
//   Eigen::Vector3d normal = fitPlane(terrain_points);

//   // -----------------------------------------
//   // [B3] 用 normal 计算“让车辆 z轴对齐到 normal 的旋转”= 地形坡度姿态（roll/pitch）
//   // -----------------------------------------
//   // z_axis = 世界竖直方向
//   Eigen::Vector3d z_axis(0, 0, 1);

//   // rotation_axis = z_axis x normal：把 z 转到 normal 的旋转轴
//   Eigen::Vector3d rotation_axis = z_axis.cross(normal);

//   Eigen::Matrix3d terrain_rotation = Eigen::Matrix3d::Identity();
//   if (rotation_axis.norm() > 1e-6)
//   {
//     // angle = arccos(z · normal)：夹角
//     double angle = std::acos(std::min(1.0, std::max(-1.0, z_axis.dot(normal))));

//     // terrain_rotation：绕 rotation_axis 旋转 angle
//     Eigen::AngleAxisd aa(angle, rotation_axis.normalized());
//     terrain_rotation = aa.toRotationMatrix();
//   }

//   // -----------------------------------------
//   // [B4] 合成最终初始姿态：R = yaw * (roll/pitch)
//   // -----------------------------------------
//   // yaw_rotation 是外部传入的“车辆航向角”旋转（绕 Z）
//   // terrain_rotation 是“坡面倾斜”旋转（把 z 对齐到 normal）
//   // 注意：矩阵乘法顺序很重要
//   // R = yaw * terrain ：表示先对齐坡面，再施加车辆航向
//   // Apply heading first, then tilt the vehicle so its local Z axis remains
//   // aligned with the world-frame terrain normal.
//   Eigen::Matrix3d R_matrix = terrain_rotation * yaw_rotation;

//   // -----------------------------------------
//   // [B5] 构造齐次变换 T：初始把车辆模型放在 (center_x, center_y, z=0) 上
//   // -----------------------------------------
//   Eigen::Matrix4d T_matrix = Eigen::Matrix4d::Identity();
//   T_matrix.block<3, 3>(0, 0) = R_matrix;
//   T_matrix.block<3, 1>(0, 3) = Eigen::Vector3d(center_pos.x(), center_pos.y(), 0.0);

//   // ==================== Iterative Pose Refinement ====================
//   // min_gap：车辆模型点到地形的最小“垂直间隙”（global_point.z - terrain_z）
//   double min_gap = 100.0;
//   std::vector<Eigen::Vector3d> touch_points;
//   std::vector<Eigen::Vector3d> touch_poly;
//   bool is_stable = false;
//   contact_points = 0;
//   bool collision = false;

//   // --------- helper: 从当前 T_matrix 提取 roll/pitch（单位：deg）---------
//   auto updateRollPitchFromT = [&](const Eigen::Matrix4d &T)
//   {
//     Eigen::Matrix3d R = T.block<3, 3>(0, 0);

//     // roll = atan2(R21, R22)
//     roll = std::atan2(R(2, 1), R(2, 2)) * 180.0 / M_PI;

//     // pitch = asin(-R20)，做一下数值夹紧避免 asin 输入略超 [-1,1]
//     double s = -R(2, 0);
//     s = std::max(-1.0, std::min(1.0, s));
//     pitch = std::asin(s) * 180.0 / M_PI;
//   };

//   auto exceedsTiltLimit = [&](const Eigen::Matrix4d &T)
//   {
//     if (!T.allFinite())
//     {
//       return true;
//     }
//     const Eigen::Vector3d body_up = T.block<3, 3>(0, 0).col(2);
//     if (!body_up.allFinite() || body_up.norm() < 1e-6)
//     {
//       return true;
//     }
//     const double max_tilt_rad =
//         fine_max_pose_tilt_deg * M_PI / 180.0;
//     return body_up.normalized().z() < std::cos(max_tilt_rad);
//   };

//   // ==================== Iterative Rotation ====================
//   int iteration = 0;
//   while (iteration < max_iterations_)
//   {
//     // Step 0: Check collision and stability
//     if (collision)
//     {
//       stable = 0;
//       return kPoseCollision;
//     }

//     if (is_stable)
//     {
//       updateRollPitchFromT(T_matrix);
//       if (exceedsTiltLimit(T_matrix))
//       {
//         stable = 0;
//         return kPoseUnstable;
//       }
//       stable = 1;
//       return kPoseSuccess;
//     }

//     // -----------------------------------------
//     // [C1] 计算 gap_map，并找最小间隙 min_gap
//     // -----------------------------------------
//     // gap_map(i,j) 表示：车辆模型网格点 (i,j) 变换到世界后，与地形高度的差值
//     // gap = z_vehicle_point - z_terrain
//     // gap>0: 车辆在地形上方（悬空）
//     // gap≈0: 接触
//     // gap<0: 穿透/碰撞（这里用 collision_gap_threshold_ 判定）
//     min_gap = 100.0;
//     Eigen::MatrixXd gap_map(robot_rows_, robot_cols_);
//     gap_map.setConstant(std::numeric_limits<double>::quiet_NaN());

//     for (int i = 0; i < robot_rows_; ++i)
//     {
//       for (int j = 0; j < robot_cols_; ++j)
//       {
//         Eigen::Vector4d robot_point(vehicle_model.X_(i, j), vehicle_model.Y_(i, j), vehicle_model.Z_(i, j), 1.0);
//         Eigen::Vector4d global_point = T_matrix * robot_point;

//         grid_map::Index terrain_idx;
//         if (submap.getIndex(grid_map::Position(global_point.x(), global_point.y()), terrain_idx))
//         {
//           float terrain_z = submap.at("elevation_BGK", terrain_idx);
//           if (!std::isnan(terrain_z))
//           {
//             double gap = global_point.z() - terrain_z;
//             gap_map(i, j) = gap;
//             min_gap = std::min(min_gap, gap);
//           }
//         }
//       }
//     }

//     // -----------------------------------------
//     // [C2] “落地”：整体向下平移 min_gap，让最接近地面的点恰好接触地面
//     // -----------------------------------------
//     // 如果 min_gap 是正数，说明所有点都在地面之上，向下移 min_gap 就会有最小点接触地面
//     // 如果 min_gap 是负数，说明已经有点低于地面，T_matrix(2,3)-=min_gap 会把车往上抬
//     if (min_gap < 100.0)
//     {
//       T_matrix(2, 3) -= min_gap;
//     }

//     // -----------------------------------------
//     // [D1] 找接触点：gap < touch_gap_threshold_ 的点当作“支撑接触”
//     // -----------------------------------------
//     touch_points.clear();
//     std::vector<Eigen::Vector2d> touch_points_2d;
//     contact_points = 0;
//     collision = false;

//     for (int i = 0; i < robot_rows_; ++i)
//     {
//       for (int j = 0; j < robot_cols_; ++j)
//       {
//         Eigen::Vector4d robot_point(vehicle_model.X_(i, j), vehicle_model.Y_(i, j), vehicle_model.Z_(i, j), 1.0);
//         Eigen::Vector4d global_point = T_matrix * robot_point;

//         grid_map::Index terrain_idx;
//         if (submap.getIndex(grid_map::Position(global_point.x(), global_point.y()), terrain_idx))
//         {
//           float terrain_z = submap.at("elevation_BGK", terrain_idx);
//           if (!std::isnan(terrain_z))
//           {
//             double gap = global_point.z() - terrain_z;
//             gap_map(i, j) = gap;

//             const bool collision_sensitive =
//                 checkWheel(vehicle_type, i, j);

//             // Only wheel/track cells are support contacts.  Chassis cells are
//             // evaluated independently against the configured body clearance.
//             if (!collision_sensitive && gap < touch_gap_threshold_)
//             {
//               contact_points++;
//               touch_points.push_back(Eigen::Vector3d(global_point.x(), global_point.y(), global_point.z()));
//               touch_points_2d.push_back(Eigen::Vector2d(vehicle_model.X_(i, j), vehicle_model.Y_(i, j)));
//             }

//             if (collision_sensitive && gap < collision_gap_threshold_)
//             {
//               collision = true;
//             }
//           }
//         }
//       }
//     }

//     if (collision)
//     {
//       stable = 0;
//       return kPoseCollision;
//     }

//     // A valid support polygon requires at least three non-collinear permitted
//     // wheel/track contacts.  One- or two-point support must never be reported
//     // as a stable vehicle pose.
//     if (touch_points.size() <
//         static_cast<std::size_t>(fine_min_support_points))
//     {
//       return kPoseFailure;
//     }

//     // Step 4: 支撑多边形 + 重力线投影
//     // cross2d: 计算二维叉积符号，用于凸包判向
//     auto cross2d = [](const Eigen::Vector2d &O, const Eigen::Vector2d &A, const Eigen::Vector2d &B)
//     {
//       return (A.x() - O.x()) * (B.y() - O.y()) - (A.y() - O.y()) * (B.x() - O.x());
//     };

//     // computeHull: 输入一堆 2D 点，输出凸包点序列（逆/顺时针）
//     auto computeHull = [&](std::vector<Eigen::Vector2d> pts)
//     {
//       if (pts.size() <= 1)
//       {
//         return pts;
//       }
//       std::sort(pts.begin(), pts.end(), [](const Eigen::Vector2d &a, const Eigen::Vector2d &b)
//                 {
//                 if (a.x() == b.x())
//                 {
//                     return a.y() < b.y();
//                 }
//                 return a.x() < b.x(); });

//       pts.erase(std::unique(pts.begin(), pts.end(),
//                             [](const Eigen::Vector2d &a,
//                                const Eigen::Vector2d &b)
//                             {
//                               return (a - b).squaredNorm() < 1e-12;
//                             }),
//                 pts.end());

//       if (pts.size() <= 1)
//       {
//         return pts;
//       }

//       std::vector<Eigen::Vector2d> hull;
//       for (const auto &p : pts)
//       {
//         while (hull.size() >= 2 && cross2d(hull[hull.size() - 2], hull.back(), p) <= 0)
//         {
//           hull.pop_back();
//         }
//         hull.push_back(p);
//       }
//       size_t lower_size = hull.size();
//       for (int i = (int)pts.size() - 2; i >= 0; --i)
//       {
//         const auto &p = pts[i];
//         while (hull.size() > lower_size && cross2d(hull[hull.size() - 2], hull.back(), p) <= 0)
//         {
//           hull.pop_back();
//         }
//         hull.push_back(p);
//       }
//       if (!hull.empty())
//       {
//         hull.pop_back();
//       }
//       return hull;
//     };

//     // pointInPoly: 判断点 p 是否在凸多边形 poly 内（假设 poly 点序有一致方向）
//     auto pointInPoly = [&](const std::vector<Eigen::Vector3d> &poly, const Eigen::Vector3d &p)
//     {
//       for (size_t i = 0; i < poly.size(); ++i)
//       {
//         const auto &p1 = poly[i];
//         const auto &p2 = poly[(i + 1) % poly.size()];
//         Eigen::Vector2d v1(p2.x() - p1.x(), p2.y() - p1.y());
//         Eigen::Vector2d v2(p.x() - p1.x(), p.y() - p1.y());
//         if (v1.x() * v2.y() - v2.x() * v1.y() < 0)
//         {
//           return false;
//         }
//       }
//       return true;
//     };

//     // 计算支撑凸包（模型系 2D）
//     std::vector<Eigen::Vector2d> hull_2d = computeHull(touch_points_2d);

//     // Keep the actual transformed height of each support sample.  Using one
//     // global minimum model height for all hull vertices is incorrect for the
//     // raised front/rear portions of the tracked model.
//     touch_poly.clear();
//     for (const auto &p : hull_2d)
//     {
//       for (std::size_t k = 0; k < touch_points_2d.size(); ++k)
//       {
//         if ((touch_points_2d[k] - p).squaredNorm() < 1e-12)
//         {
//           touch_poly.push_back(touch_points[k]);
//           break;
//         }
//       }
//     }

//     if (touch_poly.size() < 3 || touch_poly.size() != hull_2d.size())
//     {
//       return kPoseFailure;
//     }

//     double hull_area_twice = 0.0;
//     for (std::size_t i = 0; i < hull_2d.size(); ++i)
//     {
//       const auto &p1 = hull_2d[i];
//       const auto &p2 = hull_2d[(i + 1) % hull_2d.size()];
//       hull_area_twice += p1.x() * p2.y() - p2.x() * p1.y();
//     }
//     if (!std::isfinite(hull_area_twice) ||
//         std::fabs(hull_area_twice) < 2e-4)
//     {
//       return kPoseFailure;
//     }

//     // 重力方向（世界系向下）
//     Eigen::Vector3d gravity(0, 0, -1);
//     // 车辆当前平移位置（世界系）
//     Eigen::Vector3d T_pos = T_matrix.block<3, 1>(0, 3);
//     // 从车辆位置沿重力方向作一条直线
//     Eigen::ParametrizedLine<double, 3> gravity_line(T_pos, gravity);
//     // Fit the support plane from the actual permitted contact points.
//     Eigen::Vector3d plane_normal = fitPlane(touch_poly);
//     if (!plane_normal.allFinite() || plane_normal.norm() < 1e-6 ||
//         std::fabs(plane_normal.dot(gravity)) < 1e-6)
//     {
//       return kPoseFailure;
//     }
//     Eigen::Hyperplane<double, 3> contact_plane(plane_normal, touch_poly.front());
//     // 重力线与接触平面的交点：可理解为“重心沿重力方向投影到接触平面的位置”
//     Eigen::Vector3d intersection = gravity_line.intersectionPoint(contact_plane);
//     if (!intersection.allFinite())
//     {
//       return kPoseFailure;
//     }

//     Eigen::Vector3d rotatep1, rotatep2;
//     bool has_rotation_line = false;

//     // At this point the hull has at least three non-collinear support points.
//     if (pointInPoly(touch_poly, intersection))
//     {
//       is_stable = true;
//     }
//     else
//     {
//       // Projection is outside the support polygon.  Find the nearest exterior
//       // hull edge and use it as the candidate rotation axis.
//       double min_dis = std::numeric_limits<double>::max();
//       bool found_rotation_edge = false;

//       for (size_t i = 0; i < touch_poly.size(); ++i)
//       {
//         const auto &p1 = touch_poly[i];
//         const auto &p2 = touch_poly[(i + 1) % touch_poly.size()];

//         Eigen::Vector2d v1(p2.x() - p1.x(), p2.y() - p1.y());
//         Eigen::Vector2d v2(intersection.x() - p1.x(),
//                            intersection.y() - p1.y());
//         if (v1.x() * v2.y() - v2.x() * v1.y() >= 0)
//         {
//           continue;
//         }

//         const Eigen::Vector3d p1p2 = p2 - p1;
//         const double edge_norm_squared = p1p2.squaredNorm();
//         if (!std::isfinite(edge_norm_squared) || edge_norm_squared < 1e-12)
//         {
//           continue;
//         }
//         const Eigen::Vector3d p1a = intersection - p1;
//         const double t = p1a.dot(p1p2) / edge_norm_squared;
//         const Eigen::Vector3d projection = p1 + t * p1p2;
//         const double dis = (projection - intersection).norm();

//         if (std::isfinite(dis) && dis < min_dis)
//         {
//           min_dis = dis;
//           rotatep1 = p1;
//           rotatep2 = p2;
//           found_rotation_edge = true;
//         }
//       }

//       has_rotation_line = found_rotation_edge;
//     }

//     // 如果稳定了，就进入下一轮循环开头，开头会 return 2
//     if (is_stable)
//     {
//       updateRollPitchFromT(T_matrix);
//       if (exceedsTiltLimit(T_matrix))
//       {
//         stable = 0;
//         return kPoseUnstable;
//       }
//       stable = 1;
//       return kPoseSuccess;
//     }

//     if (has_rotation_line)
//     {
//       // 旋转轴的方向与参数化直线
//       const Eigen::Vector3d axis = rotatep2 - rotatep1;
//       if (!axis.allFinite() || axis.norm() < 1e-6)
//       {
//         return kPoseFailure;
//       }
//       Eigen::Vector3d direction = axis.normalized();
//       Eigen::ParametrizedLine<double, 3> rotation_line(rotatep1, direction);

//       // d_theta：要绕旋转轴转的角度（选一个最小可行值）
//       double d_theta = std::numeric_limits<double>::max();

//       // 为了计算点在旋转轴左/右侧，这里投影到 XY 平面做 2D 判断
//       Eigen::Vector2d rot_origin(rotation_line.origin().x(), rotation_line.origin().y());
//       Eigen::Vector2d rot_dir(rotation_line.direction().x(), rotation_line.direction().y());
//       if (!rot_dir.allFinite() || rot_dir.norm() < 1e-6)
//       {
//         return kPoseFailure;
//       }
//       Eigen::Vector2d rot_dir_norm = rot_dir.normalized();

//       // 遍历所有非接触点（gap > touch_gap_threshold_），估计需要转多少角
//       for (int i = 0; i < robot_rows_; ++i)
//       {
//         for (int j = 0; j < robot_cols_; ++j)
//         {
//           double gap = gap_map(i, j);

//           // NaN 或已经接触的点跳过
//           if (std::isnan(gap) || gap <= touch_gap_threshold_)
//           {
//             continue;
//           }

//           // 点变换到世界系
//           Eigen::Vector4d robot_point(vehicle_model.X_(i, j), vehicle_model.Y_(i, j), vehicle_model.Z_(i, j), 1.0);
//           Eigen::Vector4d global_point = T_matrix * robot_point;

//           // 计算点相对旋转轴起点的 2D 向量
//           Eigen::Vector2d origin_to_cell(global_point.x() - rot_origin.x(), global_point.y() - rot_origin.y());

//           // 判断点在旋转轴的哪一侧（叉积符号）
//           // 只对某一侧的点计算 d_theta（避免两边一起压导致矛盾）
//           if (origin_to_cell.x() * rot_dir_norm.y() - origin_to_cell.y() * rot_dir_norm.x() > 0)
//           {
//             // 计算点到旋转轴的垂直距离 distance（在 XY 平面）
//             Eigen::Vector2d proj_vec = origin_to_cell.dot(rot_dir_norm) * rot_dir_norm;
//             double distance = (origin_to_cell - proj_vec).norm();

//             // atan2(gap, distance) 是一个“需要转的角度”的几何估计：
//             // gap 越大、distance 越小 -> 需要更大角度把它压下来
//             d_theta = std::min(d_theta, std::atan2(gap, distance));
//           }
//         }
//       }

//       // 如果找到了有效 d_theta，就绕 rotation_line 旋转更新 T_matrix
//       if (d_theta < std::numeric_limits<double>::max())
//       {
//         const double max_rotation_step =
//             fine_max_rotation_step_deg * M_PI / 180.0;
//         d_theta = std::max(0.0, std::min(d_theta, max_rotation_step));

//         // 下面是一套标准的“绕任意轴旋转”的齐次矩阵构造：
//         // 1) 平移到轴原点
//         // 2) 旋转坐标系，让旋转轴对齐到 z 轴
//         // 3) 绕 z 轴旋转 d_theta
//         // 4) 旋转回去
//         // 5) 平移回去

//         Eigen::Matrix4d T1 = Eigen::Matrix4d::Identity();
//         T1.block<3, 1>(0, 3) = -rotation_line.origin();

//         Eigen::Matrix4d R1 = Eigen::Matrix4d::Identity();
//         R1.block<3, 3>(0, 0) =
//             Eigen::Quaterniond().setFromTwoVectors(rotation_line.direction(), Eigen::Vector3d(0, 0, 1)).toRotationMatrix();

//         Eigen::Matrix4d R2 = Eigen::Matrix4d::Identity();
//         R2.block<3, 3>(0, 0) = Eigen::AngleAxisd(d_theta, Eigen::Vector3d(0, 0, 1)).toRotationMatrix();

//         Eigen::Matrix4d R3 = Eigen::Matrix4d::Identity();
//         R3.block<3, 3>(0, 0) = R1.block<3, 3>(0, 0).transpose();

//         Eigen::Matrix4d T2 = Eigen::Matrix4d::Identity();
//         T2.block<3, 1>(0, 3) = rotation_line.origin();

//         Eigen::Matrix4d rotate_with_rotation_line = T2 * R3 * R2 * R1 * T1;

//         // 更新车辆位姿
//         T_matrix = rotate_with_rotation_line * T_matrix;

//         // 旋转可能引起 xy 平移漂移，这里强制把 xy 拉回 center_pos
//         T_matrix.block<2, 1>(0, 3) = Eigen::Vector2d(center_pos.x(), center_pos.y());

//         if (!T_matrix.allFinite())
//         {
//           return kPoseFailure;
//         }

//         if (exceedsTiltLimit(T_matrix))
//         {
//           stable = 0;
//           updateRollPitchFromT(T_matrix);
//           return kPoseUnstable;
//         }
//       }
//       else
//       {
//         return kPoseFailure;
//       }
//     }
//     else
//     {
//       return kPoseFailure;
//     }

//     iteration++;
//   }

//   // -----------------------------------------
//   // [H1] 从最终旋转矩阵提取 roll/pitch（并转成度）
//   // -----------------------------------------
//   updateRollPitchFromT(T_matrix);

//   ROS_INFO("Pose result (vehicle %d) at cell (%.2f, %.2f): roll=%.3f, pitch=%.3f, stable=%d, collision=%d",
//            vehicle_type, center_pos.x(), center_pos.y(), roll, pitch, is_stable ? 1 : 0, collision ? 1 : 0);

//   stable = is_stable ? 1 : 0;

//   // -----------------------------------------
//   // [H2] 返回码约定
//   // -----------------------------------------
//   // 2 = 成功且稳定
//   // 1 = 碰撞（上面已经提前 return 1）
//   // 3 = 姿态超过物理倾角限制
//   // 0 = 未稳定或失败
//   return is_stable ? kPoseSuccess : kPoseFailure;
// }

// bool Mapping::checkWheel(int vehicle_type, int i, int j) const
// {
//   // false：车轮/履带接触区域
//   // true：底盘区域，需要检查碰撞
//   if (vehicle_type == 1)
//   {
//     static const std::array<int, 4> wheel_rows = {0, 1, 7, 8};

//     static const std::array<int, 5> wheel_cols = {1, 2, 3, 8, 9};

//     const bool row_matches = std::find(wheel_rows.begin(), wheel_rows.end(), i) != wheel_rows.end();

//     const bool col_matches = std::find(wheel_cols.begin(), wheel_cols.end(), j) != wheel_cols.end();

//     return !(row_matches && col_matches);
//   }

//   if (vehicle_type == 2)
//   {
//     static const std::array<int, 4> track_rows = {0, 1, 7, 8};

//     const bool is_track = std::find(track_rows.begin(), track_rows.end(), i) != track_rows.end();

//     return !is_track;
//   }

//   return true;
// }

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
  if (solver.info() != Eigen::Success ||
      !solver.eigenvalues().allFinite())
  {
    return Eigen::Vector3d::UnitZ();
  }

  // -----------------------------------------
  // [4] 平面法向 = 最小特征值对应的特征向量
  // -----------------------------------------
  // 直觉：平面点云在“法向方向”的方差最小（点都贴在一个面上）
  // 因此 cov 最小特征值方向就是法向方向
  Eigen::Vector3d normal = solver.eigenvectors().col(0);
  if (!normal.allFinite() || normal.norm() < 1e-9)
  {
    return Eigen::Vector3d::UnitZ();
  }

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
  // Keep the original robot-centred local map output.  This topic is intended
  // for consumers that need a small, frequently updated map.
  grid_map_msgs::GridMap message;
  if (buildGridMapMessage(message))
  // if (!buildGridMapFromViewpointCloud(message))
  {
    gridmap_pub_.publish(message);
  }

  // -----------------------------------------------------------------------
  // Global explored map output
  // -----------------------------------------------------------------------
  // height_map_ is already the persistent global map.  Do not allocate and
  // maintain a second 1000 m x 1000 m map here: that would duplicate all
  // layers and consume a very large amount of memory.  Instead, publish a
  // submap whose bounding box contains every cell that has ever received a
  // real point observation (n_points > 0).  Cells inside the bounding box that
  // have not been observed remain NaN/unknown.
  //
  // The publisher is latched so a newly started RViz/planner immediately gets
  // the latest global explored map.  Publishing is deliberately decimated
  // because a global GridMap message can be much larger than the local map.
  static ros::Publisher global_gridmap_pub;
  static bool global_publisher_initialized = false;
  static int global_publish_every_n = 50;
  static int global_publish_counter = 0;
  static bool global_swept_bounds_initialized = false;
  static double global_swept_min_x = 0.0;
  static double global_swept_min_y = 0.0;
  static double global_swept_max_x = 0.0;
  static double global_swept_max_y = 0.0;

  if (!global_publisher_initialized)
  {
    std::string global_gridmap_topic;
    pnh_.param<std::string>("global_gridmap_topic", global_gridmap_topic, "global_trav_map");

    pnh_.param<int>("global_publish_every_n", global_publish_every_n, 50);

    global_publish_every_n = std::max(1, global_publish_every_n);

    global_gridmap_pub = nh_.advertise<grid_map_msgs::GridMap>(global_gridmap_topic, 1, true);
    global_publisher_initialized = true;

    ROS_INFO("Global explored grid map will be published on %s every %d local publications",
             global_gridmap_pub.getTopic().c_str(),
             global_publish_every_n);
  }

  // 每次局部发布时用当前位姿增量扩展“已扫过范围”。
  // 所有进入地图的点都已经通过 max_range_ 过滤，因此该包围框
  // 保证覆盖所有真实观测。与每次扫描 400 万个 n_points 格子
  // 相比，这里每帧只需 O(1) 更新。
  Eigen::Vector3d global_pose_snapshot;
  {
    std::lock_guard<std::mutex> pose_lock(vehicle_pose_mutex_);
    global_pose_snapshot = vehicle_position_;
  }

  const double pose_min_x = global_pose_snapshot.x() - max_range_;
  const double pose_min_y = global_pose_snapshot.y() - max_range_;
  const double pose_max_x = global_pose_snapshot.x() + max_range_;
  const double pose_max_y = global_pose_snapshot.y() + max_range_;
  if (!global_swept_bounds_initialized)
  {
    global_swept_min_x = pose_min_x;
    global_swept_min_y = pose_min_y;
    global_swept_max_x = pose_max_x;
    global_swept_max_y = pose_max_y;
    global_swept_bounds_initialized = true;
  }
  else
  {
    global_swept_min_x = std::min(global_swept_min_x, pose_min_x);
    global_swept_min_y = std::min(global_swept_min_y, pose_min_y);
    global_swept_max_x = std::max(global_swept_max_x, pose_max_x);
    global_swept_max_y = std::max(global_swept_max_y, pose_max_y);
  }

  ++global_publish_counter;
  if (global_publish_counter < global_publish_every_n)
  {
    return;
  }
  global_publish_counter = 0;

  // Capture only immutable request data.  If the worker is already busy this
  // task replaces the older pending request; requests never form a FIFO.
  const double requested_min_x = global_swept_min_x;
  const double requested_min_y = global_swept_min_y;
  const double requested_max_x = global_swept_max_x;
  const double requested_max_y = global_swept_max_y;
  const ros::Publisher requested_publisher = global_gridmap_pub;

  globalPublishWorker().requestLatest(
      [this, requested_min_x, requested_min_y,
       requested_max_x, requested_max_y,
       requested_publisher]() mutable
      {
        const ros::WallTime total_start = ros::WallTime::now();
        grid_map::GridMap global_explored_map;

        // Global publication is lower priority than point-cloud processing.
        // Wait until the processing queue is idle.  If a newer global request
        // arrived while waiting, abandon this older request; the worker will
        // execute the replacement next.
        waitUntilIdle();
        if (globalPublishWorker().hasPendingTask())
        {
          return;
        }

        // The lock is held only while taking a coherent GridMap snapshot.
        // Layer filtering, ROS serialization and publication happen after the
        // lock is released and therefore cannot block local map updates.
        const ros::WallTime snapshot_start = ros::WallTime::now();
        {
          std::lock_guard<std::mutex> lock(gridmap_publish_mutex_);

          const grid_map::Position map_center = height_map_.getPosition();
          const grid_map::Length map_length = height_map_.getLength();
          const double resolution = height_map_.getResolution();
          const double map_min_x =
              map_center.x() - 0.5 * map_length.x();
          const double map_min_y =
              map_center.y() - 0.5 * map_length.y();
          const double map_max_x =
              map_center.x() + 0.5 * map_length.x();
          const double map_max_y =
              map_center.y() + 0.5 * map_length.y();

          const double min_x = std::max(
              map_min_x, requested_min_x - resolution);
          const double min_y = std::max(
              map_min_y, requested_min_y - resolution);
          const double max_x = std::min(
              map_max_x, requested_max_x + resolution);
          const double max_y = std::min(
              map_max_y, requested_max_y + resolution);

          const double length_x =
              std::max(resolution, max_x - min_x);
          const double length_y =
              std::max(resolution, max_y - min_y);
          const grid_map::Position submap_center(
              min_x + 0.5 * length_x,
              min_y + 0.5 * length_y);
          const grid_map::Length submap_length(length_x, length_y);

          bool success = false;
          global_explored_map = height_map_.getSubmap(
              submap_center, submap_length, success);
          if (!success)
          {
            ROS_WARN_THROTTLE(
                2.0, "Failed to crop the asynchronous global map snapshot.");
            return;
          }
        }
        const double snapshot_seconds =
            (ros::WallTime::now() - snapshot_start).toSec();

        static const std::vector<std::string>
            global_layers_to_publish = {
                "elevation",
                "elevation_BGK",
                "slope",
                "roughness",
                "step",
                "slope_deg",
                "roughness_raw",
                "step_height",
                "traversability",
                "traversability_coarse_wheeled",
                "traversability_coarse_tracked",
                "traversability_fine_wheeled",
                "traversability_fine_tracked",
                "critical"};

        const auto global_existing_layers =
            global_explored_map.getLayers();
        for (const auto &layer : global_existing_layers)
        {
          if (std::find(global_layers_to_publish.begin(),
                        global_layers_to_publish.end(),
                        layer) == global_layers_to_publish.end())
          {
            global_explored_map.erase(layer);
          }
        }

        global_explored_map.setTimestamp(
            ros::Time::now().toNSec());

        grid_map_msgs::GridMap global_message;
        grid_map::GridMapRosConverter::toMessage(
            global_explored_map, global_message);
        requested_publisher.publish(global_message);

        const double total_seconds =
            (ros::WallTime::now() - total_start).toSec();
        ROS_INFO_THROTTLE(
            2.0,
            "Published async global map: %.1f x %.1f m, snapshot=%.3f s, total=%.3f s",
            global_explored_map.getLength().x(),
            global_explored_map.getLength().y(),
            snapshot_seconds, total_seconds);
      });
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
  // [4] Publish a local platform-specific XYZI /terrain_map.
  // ==========================================
  // Reuse the same local_map that is already cropped for trav_map.  This
  // avoids a second getSubmap() and keeps the additional publication cheap.
  if (publish_local_trav_cloud)
  {
    if (!local_map.exists(local_trav_layer))
    {
      ROS_WARN_THROTTLE(
          2.0,
          "Cannot publish traversability cloud: required layer missing (%s)",
          local_trav_layer.c_str());
    }
    else
    {
      pcl::PointCloud<pcl::PointXYZI> trav_cloud;
      std::size_t traversable_point_count = 0;
      std::size_t obstacle_point_count = 0;
      std::size_t unknown_point_count = 0;
      const grid_map::Size local_size = local_map.getSize();
      trav_cloud.points.reserve(static_cast<std::size_t>(local_size(0)) * static_cast<std::size_t>(local_size(1)));

      for (grid_map::GridMapIterator it(local_map); !it.isPastEnd(); ++it)
      {
        grid_map::Position position;
        if (!local_map.getPosition(*it, position))
        {
          continue;
        }

        const float traversability = local_map.at(local_trav_layer, *it);

        const bool is_unknown = !std::isfinite(traversability);

        // The original FitPlane publishes occupancy value -1 as a traversable
        // XYZI point.  This switch preserves that behavior by default while
        // allowing a conservative deployment to omit unknown cells.
        if (is_unknown && !terrain_map_unknown_as_traversable)
        {
          continue;
        }

        const bool is_obstacle = !is_unknown && traversability > static_cast<float>(terrain_map_cost_threshold);

        pcl::PointXYZI point;
        point.x = static_cast<float>(position.x());
        point.y = static_cast<float>(position.y());
        point.z = static_cast<float>(current_pos.z() + (is_obstacle ? terrain_map_obstacle_z_offset : terrain_map_traversable_z_offset));

        // Compatibility with the original local_planner:
        //   intensity == 0.2 : traversable (not greater than its 0.2 threshold)
        //   intensity == 1.0 : non-traversable obstacle
        if (!is_obstacle)
        {
          point.intensity = 0.2f;
          if (is_unknown)
          {
            ++unknown_point_count;
          }
          else
          {
            ++traversable_point_count;
          }
        }
        else
        {
          point.intensity = 1.0f;
          ++obstacle_point_count;
        }
        trav_cloud.points.push_back(point);
      }

      trav_cloud.width = static_cast<std::uint32_t>(trav_cloud.points.size());
      trav_cloud.height = 1;
      trav_cloud.is_dense = true;

      sensor_msgs::PointCloud2 trav_cloud_message;
      pcl::toROSMsg(trav_cloud, trav_cloud_message);
      trav_cloud_message.header.frame_id = local_map.getFrameId();

      ros::Time cloud_stamp;
      if (local_map.getTimestamp() > 0)
      {
        cloud_stamp.fromNSec(local_map.getTimestamp());
      }
      else
      {
        cloud_stamp = ros::Time::now();
      }
      trav_cloud_message.header.stamp = cloud_stamp;
      local_trav_cloud_publisher.publish(trav_cloud_message);

      const double message_age =
          std::max(0.0, (ros::Time::now() - cloud_stamp).toSec());
      ROS_INFO_THROTTLE(
          1.0,
          "Published %s: total=%zu, traversable(0.2)=%zu, unknown_as_0.2=%zu, obstacle(1.0)=%zu, source_age=%.3f s",
          local_trav_cloud_publisher.getTopic().c_str(),
          trav_cloud.points.size(),
          traversable_point_count,
          unknown_point_count,
          obstacle_point_count,
          message_age);
    }
  }

  // ==========================================
  // [5] 只保留白名单层（减少消息体积）
  // ==========================================
  // 发布层白名单：只传这些层，避免把很多内部计算层发出去导致消息巨大
  static const std::vector<std::string> layers_to_publish = {
      "elevation",
      "elevation_BGK",
      "slope",
      "roughness",
      "step",
      "slope_deg",
      "roughness_raw",
      "step_height",
      "traversability",
      "traversability_coarse_wheeled",
      "traversability_coarse_tracked",
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
  // [6] 转 ROS 消息
  // ==========================================
  // GridMapRosConverter 会把 local_map 的 metadata（尺寸、分辨率、坐标系）
  // 和各 layer 的 matrix 数据打包成 grid_map_msgs::GridMap 消息
  grid_map::GridMapRosConverter::toMessage(local_map, message);

  // Keep all existing mapping and /terrain_map publication above unchanged.
  // Replace only the outgoing trav_map payload with the ViewPointManager-
  // aligned, one-cell-per-XY-viewpoint resampling of the same height_map_.
  if (!BuildViewpointAlignedGridMap(height_map_, frame_id_, message))
  {
    return false;
  }

  return true;
}

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
      {"traversability_coarse_wheeled", PoolType::MAX},
      {"traversability_coarse_tracked", PoolType::MAX},
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
