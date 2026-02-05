/**
 * * * @description:
 * * * @filename: obstacle_mapping.cpp
 * * * @author: wangxurui
 * * * @date: 2025-03-14 21:05:18
 **/

#include "obstacle_mapping.h"

using namespace std;

obstacle_mapping::obstacle_mapping(ros::NodeHandle &nh) : nh_(nh)
{
  init();
}

obstacle_mapping::~obstacle_mapping()
{
}

void obstacle_mapping::init()
{
  /****************************
  init params
  ****************************/
  nh_.param<bool>("save_map", save_map_, false);
  nh_.param<int>("running_mode", running_mode_, 1);
  nh_.param<int>("mapping_mode", mapping_mode_, 1);
  nh_.param<int>("threadCount", threadCount_, 1);
  nh_.param<int>("pose_mode", pose_mode_, 1);
  nh_.param<int>("cntthreshold", cntthreshold_, 5);
  nh_.param<string>("lidar_topic", lidar_topic_, "/velodyne_points");
  nh_.param<string>("odom_topic", odom_topic_, "/liorf/mapping/odometry");
  nh_.param<string>("frame_id/map", map_frame_, "map");
  nh_.param<string>("frame_id/robot", robot_frame_, "base_link");
  nh_.param<string>("frame_id/lidar", lidar_frame_, "velodyne");
  nh_.param<string>("offline/pcd_path", pcd_path_, "/home/wxr/ivrc/6t/slam_ws/maps/zongpo.pcd");
  nh_.param<string>("offline/elevation_BGK_savepath", elevation_BGK_savepath_,
                    "/home/wxr/proj/terrain/ws_trav/src/obstacle_mapping/test_map/terrain/elevation_BGKmap.csv");
  nh_.param<string>("offline/critical_savepath", critical_savepath_,
                    "/home/wxr/proj/terrain/ws_trav/src/obstacle_mapping/test_map/terrain/criticalmap.csv");
  nh_.param<string>("offline/traversability_savepath", traversability_savepath_,
                    "/home/wxr/proj/terrain/ws_trav/src/obstacle_mapping/test_map/terrain/traversabilitymap.csv");
  nh_.param<string>("offline/obstacle_savepath", obstacle_savepath_,
                    "/home/wxr/proj/terrain/ws_trav/src/obstacle_mapping/test_map/terrain/obstaclemap.csv");
  nh_.param<string>("offline/gridmap_loadpath", gridmap_loadpath_,
                    "/home/wxr/proj/terrain/ws_trav/src/obstacle_mapping/test_map/elevation1.csv");
  nh_.param<double>("local_map/resolution", local_map_resolution_, 0.2);
  nh_.param<double>("local_map/mapLengthX", local_map_length_x_, 40.0);
  nh_.param<double>("local_map/mapLengthY", local_map_length_y_, 60.0);
  nh_.param<double>("global_map/resolution", global_map_resolution_, 0.2);
  nh_.param<double>("global_map/mapLengthX", global_map_length_x_, 1000.0);
  nh_.param<double>("global_map/mapLengthY", global_map_length_y_, 1000.0);
  nh_.param<double>("global_map/submapLengthX", global_submap_length_x_, 100.0);
  nh_.param<double>("global_map/submapLengthY", global_submap_length_y_, 100.0);
  nh_.param<double>("global_map/predictionKernalSize", predictionKernalSize_, 1.0);
  nh_.param<double>("preprocess/minZ", minZ_, -3.0);
  nh_.param<double>("preprocess/maxZ", maxZ_, 3.0);
  nh_.param<double>("preprocess/lidar_Z", lidar_z_, 0);
  nh_.param<double>("heightdiff_threshold", heightdiffthreshold_, 0.6);
  nh_.param<double>("cntratiothreshold", cntratiothreshold_, 0.5);
  nh_.param<double>("normal_estimationRadius", normal_estimationRadius_, 0.5);
  nh_.param<double>("stepRadius", stepRadius_, 0.3);
  nh_.param<double>("maxsensorRangeLimit", maxsensorRangeLimit_, 50.0);
  nh_.param<double>("minsensorRangeLimit", minsensorRangeLimit_, 3.0);
  nh_.param<double>("slope_crit", slope_crit_, 0.2);
  nh_.param<double>("roughness_crit", roughness_crit_, 0.5);
  nh_.param<double>("step_crit", step_crit_, 0.8);
  nh_.param<double>("traversability_crit", traversability_crit_, 0.8);
  nh_.param<double>("slope_crit_fp", slope_crit_fp_, 0.2);
  nh_.param<double>("slope_crit_fp_up", slope_crit_fp_up_, 0.2);
  nh_.param<double>("roughness_crit_fp", roughness_crit_fp_, 0.5);
  nh_.param<double>("roughness_crit_fp_up", roughness_crit_fp_up_, 0.8);
  nh_.param<double>("step_crit_fp", step_crit_fp_, 0.8);
  nh_.param<double>("step_crit_fp_", step_crit_fp_up_, 0.8);
  nh_.param<double>("traversability_crit_fp", traversability_crit_fp_, 0.8);
  nh_.param<double>("traversability_crit_fp_up", traversability_crit_fp_up_, 0.8);

  /****************************  init gridmap  ****************************/
  initgridmap(local_map_, map_frame_, local_map_resolution_, local_map_length_x_, local_map_length_y_);
  initgridmap(fused_local_map_, robot_frame_, local_map_resolution_, local_map_length_x_, local_map_length_y_);
  initgridmap(global_map_, map_frame_, global_map_resolution_, global_map_length_x_, global_map_length_y_);

  local_map_pub_ = nh_.advertise<grid_map_msgs::GridMap>("/localmap", 1);
  local_fused_map_pub_ = nh_.advertise<grid_map_msgs::GridMap>("/localfusedmap", 1);
  global_map_pub_ = nh_.advertise<grid_map_msgs::GridMap>("/globalmap", 1);
  pointcloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/vis_points", 1);

  if (running_mode_ == 1)
  {
    lidar_sub_ = nh_.subscribe<sensor_msgs::PointCloud2>(lidar_topic_, 5, boost::bind(&obstacle_mapping::Mapping, this, _1));
    odom_sub_ = nh_.subscribe<nav_msgs::Odometry>(odom_topic_, 1, boost::bind(&obstacle_mapping::StoreOdom, this, _1));
    ros::spin();
  }
  else if (running_mode_ == 2 || running_mode_ == 3)
  {
    pcd_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/pcd_points", 1);

    ros::Time::init();
    // 创建点云对象
    auto cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    // 加载 PCD 文件
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_path_, *cloud) == -1)
    {
      PCL_ERROR("Couldn't read file %s \n", pcd_path_.c_str());
      return;
    }
    std::cout << "Loaded " << cloud->width * cloud->height << " data points from " << pcd_path_ << std::endl;
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cloud, cloud_msg);
    cloud_msg.header.stamp = ros::Time::now();
    cloud_msg.header.frame_id = "map";

    if (running_mode_ == 2)
    {
      sensor_msgs::PointCloud2ConstPtr cloud_msg_ptr = boost::make_shared<const sensor_msgs::PointCloud2>(cloud_msg);
      Mapping(cloud_msg_ptr);
      if (save_map_)
      {
        saveMatrixToCSV(local_map_["elevation_BGK"], elevation_BGK_savepath_);
        saveMatrixToCSV(local_map_["critical"], critical_savepath_);
        saveMatrixToCSV(local_map_["traversability"], traversability_savepath_);
        saveMatrixToCSV(local_map_["obstacle"], obstacle_savepath_);
      }
    }
    else if (running_mode_ == 3)
    {
      local_map_.add("elevation_BGK", loadMatrixFromCSV(gridmap_loadpath_));
      local_map_.add("elevation", local_map_["elevation_BGK"]);
      FeatureMapping(local_map_);
      TraversabilityMapping(local_map_);
      mapobstacledetection(local_map_);
    }

    ros::Rate loop_rate(10);
    while (ros::ok())
    {
      publocalmap();
      pcd_pub_.publish(cloud_msg);
      ros::spinOnce();
      loop_rate.sleep();
    }
  }
}

void obstacle_mapping::initgridmap(grid_map::GridMap &map, string frame, double height_map_resolution, double mapLength_x, double mapLength_y)
{
  grid_map::Length mapLength(mapLength_x, mapLength_y);
  map.setFrameId(frame);
  map.setGeometry(mapLength, height_map_resolution, grid_map::Position(0.0, 0.0));
  map.add("elevation");
  map.add("elevation_BGK");
  map.add("variance");
  map.add("min_elevation");
  map.add("max_elevation");
  map.add("normal_x");
  map.add("normal_y");
  map.add("normal_z");
  map.add("slope");
  map.add("roughness");
  map.add("step");
  map.add("traversability");
  map.add("n_points", 0);
  map.add("obstacle", 0);
  map.add("obstacle_cnt", 0);
  map.add("critical", 0);
}

void obstacle_mapping::StoreOdom(const nav_msgs::OdometryConstPtr &odom)
{
  odom_deque_.push_back(*odom);
  if (odom_deque_.size() > 50)
    odom_deque_.pop_front();
}

void obstacle_mapping::Mapping(const sensor_msgs::PointCloud2ConstPtr &msg)
{
  map_time_ = msg->header.stamp;
  auto inputcloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  pcl::fromROSMsg(*msg, *inputcloud);
  ROS_INFO("Lidar msg time : %f", map_time_.toSec());

  if (running_mode_ != 1)
  {
    ROS_INFO("Offline Mapping...");

    robot_pose_ = Eigen::Vector3d::Zero();

    auto localcloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    localcloud->clear();
    localcloud->reserve(inputcloud->size());
    for (auto &point : inputcloud->points)
    {
      point.z += lidar_z_;
      if (point.z >= minZ_ && point.z <= maxZ_)
      {
        localcloud->points.push_back(point);
      }
    }

    // pointcloudinterpolation(localcloud);
    LocalMapping(*localcloud);
    cloudobstacledetection(*localcloud, local_map_);
    double time1 = ros::Time::now().toSec();
    DenseMapping(local_map_);
    double time2 = ros::Time::now().toSec();
    FeatureMapping(local_map_);
    double time3 = ros::Time::now().toSec();
    TraversabilityMapping(local_map_);
    criticalfootprintsfilter(local_map_);
    mapobstacledetection(local_map_);
    visualizepointcloud();

    ROS_INFO("DenseMapping takes time : %f s", time2 - time1);
    ROS_INFO("FeatureMapping takes time : %f s", time3 - time2);
    return;
  }

  /****************************
  单帧地图模式
  ****************************/
  if (mapping_mode_ == 1)
  {
    ROS_INFO("Local Mapping...");

    robot_pose_ = Eigen::Vector3d::Zero();

    auto localcloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    localcloud->clear();
    localcloud->reserve(inputcloud->size());
    for (auto &point : inputcloud->points)
    {
      point.z += lidar_z_;
      if (point.z >= minZ_ && point.z <= maxZ_)
      {
        localcloud->points.push_back(point);
      }
    }
    LocalMapping(*localcloud);
    double time1 = ros::Time::now().toSec();
    DenseMapping(local_map_);
    double time2 = ros::Time::now().toSec();
    FeatureMapping(local_map_);
    double time3 = ros::Time::now().toSec();
    TraversabilityMapping(local_map_);

    ROS_INFO("DenseMapping takes time : %f s", time2 - time1);
    ROS_INFO("FeatureMapping takes time : %f s", time3 - time2);
    mapobstacledetection(local_map_);
    publocalmap();
    local_map_.clearAll();
    grid_map::Size local_map_size = local_map_.getSize();
    int rows = local_map_size.x();
    int cols = local_map_size.y();
    Eigen::MatrixXf zeroMatrix = Eigen::MatrixXf::Zero(rows, cols);
    local_map_.add("n_points", zeroMatrix);
    local_map_.add("obstacle", zeroMatrix);
    local_map_.add("obstacle_cnt", zeroMatrix);
    return;
  }

  /****************************
  多帧融合模式
  ****************************/
  else if (mapping_mode_ == 2)
  {
    ROS_INFO("Local Fusion Mapping...");
    auto localcloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    localcloud->clear();
    localcloud->reserve(inputcloud->size());
    for (auto &point : inputcloud->points)
    {
      point.z += lidar_z_;
      if (point.z >= minZ_ && point.z <= maxZ_)
      {
        localcloud->points.push_back(point);
      }
    }
    LocalMapping(*localcloud);
    FuseMap();
    pubfusedlocalmap();
    local_map_.clearAll();
    return;
  }

  /****************************
  全局地图模式
  ****************************/
  else if (mapping_mode_ == 3)
  {

    ROS_INFO("Global Mapping...");
    /****************************
    将点云坐标转到map（全局）坐标系
    ****************************/
    lidar_frame_ = msg->header.frame_id;
    Eigen::Affine3d lmtransformMatrix = Eigen::Affine3d::Identity();
    if (pose_mode_ == 1)
    {
      // 默认TF
      tf::StampedTransform lidarmaptransform;
      try
      {
        // listener_.waitForTransform(map_frame_, lidar_frame_, ros::Time(0), ros::Duration(0.1));
        listener_.lookupTransform(map_frame_, lidar_frame_, ros::Time(0), lidarmaptransform);
      }
      catch (tf::TransformException &ex)
      {
        ROS_WARN("%s", ex.what());
      }
      poseTFToEigen(lidarmaptransform, lmtransformMatrix);
    }
    else if (pose_mode_ == 2)
    {

      // Odometry
      int matchodom_index = TimestampMatch(map_time_.toSec(), odom_deque_);
      cur_odom_ = odom_deque_[matchodom_index];
      double x = cur_odom_.pose.pose.position.x;
      double y = cur_odom_.pose.pose.position.y;
      double z = cur_odom_.pose.pose.position.z;
      double qx = cur_odom_.pose.pose.orientation.x;
      double qy = cur_odom_.pose.pose.orientation.y;
      double qz = cur_odom_.pose.pose.orientation.z;
      double qw = cur_odom_.pose.pose.orientation.w;
      Eigen::Quaterniond quaternion(qw, qx, qy, qz);
      lmtransformMatrix = Eigen::Translation3d(x, y, z) * quaternion;
    }
    else
    {
      ROS_WARN("pose mode %d is unknown!", pose_mode_);
    }

    robot_pose_ = lmtransformMatrix.translation();
    ROS_INFO("robot position: %.2f, %.2f, %.2f", robot_pose_.x(), robot_pose_.y(), robot_pose_.z());

    auto globalcloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    globalcloud->clear();
    globalcloud->reserve(inputcloud->size());
    pcl::transformPointCloud(*inputcloud, *globalcloud, lmtransformMatrix);
    auto globalcloudprocessed = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    globalcloudprocessed->clear();
    globalcloudprocessed->reserve(globalcloud->size());
    for (auto &point : globalcloud->points)
    {
      point.z += lidar_z_;
      if (point.z >= minZ_ && point.z <= maxZ_)
      {
        globalcloudprocessed->points.push_back(point);
      }
    }

    /****************************
    全局建图
    ****************************/
    // GlobalMapping(*globalcloud);
    GlobalMapping(*globalcloudprocessed);
    // 子图中心点
    center_robot_ = robot_pose_.head<2>();
    if (!global_map_.getIndex(center_robot_, center_robot_index_))
    {
      ROS_ERROR("robot is out of global map !");
      return;
    }
    // 提取子图
    bool submap_sucess;
    grid_map::Length submap_length(global_submap_length_x_, global_submap_length_y_);
    global_submap_ = global_map_.getSubmap(center_robot_, submap_length, submap_sucess);
    if (!submap_sucess)
    {
      ROS_WARN("cound not get the global_submap!!!");
      return;
    }
    double time1 = ros::Time::now().toSec();
    DenseMapping(global_submap_);
    double time2 = ros::Time::now().toSec();
    FeatureMapping(global_submap_);
    double time3 = ros::Time::now().toSec();
    TraversabilityMapping(global_submap_);
    mapobstacledetection(global_submap_);

    FeatureMerge();

    ROS_INFO("DenseMapping takes time : %f s", time2 - time1);
    ROS_INFO("FeatureMapping takes time : %f s", time3 - time2);
    pubglobalmap();
    return;
  }
  else
  {
    ROS_ERROR("Mapping mode %d is doesnot exist!!", mapping_mode_);
  }
  return;
}

// void obstacle_mapping::MappingPCD(const pcl::PointCloud<pcl::PointXYZ> &cloud)
// {
//     LocalMapping(cloud);
//     double time1 = ros::Time::now().toSec();
//     DenseMapping(local_map_);
//     double time2 = ros::Time::now().toSec();
//     FeatureMapping(local_map_);
//     double time3 = ros::Time::now().toSec();
//     TraversabilityMapping(local_map_);
//     mapobstacledetection(local_map_);

//     ROS_INFO("DenseMapping takes time : %f s", time2 - time1);
//     ROS_INFO("FeatureMapping takes time : %f s", time3 - time2);
// }

void obstacle_mapping::LocalMapping(const pcl::PointCloud<pcl::PointXYZ> &localcloud)
{
  auto &heightMatrix = local_map_["elevation"];
  auto &varianceMatrix = local_map_["variance"];
  auto &minHeightMatrix = local_map_["min_elevation"];
  auto &maxHeightMatrix = local_map_["max_elevation"];
  auto &npointsMatrix = local_map_["n_points"];
  grid_map::Index PointIndex;
  grid_map::Position PointPosition;

  for (const auto &Point : localcloud)
  {
    PointPosition << Point.x, Point.y;
    if (!local_map_.getIndex(PointPosition, PointIndex))
    {
      continue;
    }
    auto &height = heightMatrix(PointIndex(0), PointIndex(1));
    auto &variance = varianceMatrix(PointIndex(0), PointIndex(1));
    auto &npoints = npointsMatrix(PointIndex(0), PointIndex(1));
    auto &minHeight = minHeightMatrix(PointIndex(0), PointIndex(1));
    auto &maxHeight = maxHeightMatrix(PointIndex(0), PointIndex(1));
    if (!npoints)
    {
      height = Point.z;
      variance = 0.0f;
      minHeight = Point.z;
      maxHeight = Point.z;
      npoints += 1;
    }
    else
    {
      // height = (npoints / (npoints + 1)) * height + 1 / (npoints + 1) * Point.z;
      npoints += 1;
      updateHeightStats(height, variance, npoints, Point.z);
      minHeight = std::min(minHeight, Point.z);
      maxHeight = std::max(maxHeight, Point.z);
      // height = maxHeight;
    }
  }
}

void obstacle_mapping::GlobalMapping(const pcl::PointCloud<pcl::PointXYZ> &globalcloud)
{
  auto &heightMatrix = global_map_["elevation"];
  auto &npointsMatrix = global_map_["n_points"];
  auto &varianceMatrix = global_map_["variance"];
  auto &minHeightMatrix = global_map_["min_elevation"];
  auto &maxHeightMatrix = global_map_["max_elevation"];
  grid_map::Index PointIndex;
  grid_map::Position PointPosition;

  for (const auto &Point : globalcloud)
  {
    PointPosition << Point.x, Point.y;
    if (!global_map_.getIndex(PointPosition, PointIndex))
      continue;
    auto &height = heightMatrix(PointIndex(0), PointIndex(1));
    auto &npoints = npointsMatrix(PointIndex(0), PointIndex(1));
    auto &variance = varianceMatrix(PointIndex(0), PointIndex(1));
    auto &minHeight = minHeightMatrix(PointIndex(0), PointIndex(1));
    auto &maxHeight = maxHeightMatrix(PointIndex(0), PointIndex(1));
    if (!npoints)
    {
      height = Point.z;
      minHeight = Point.z;
      maxHeight = Point.z;
    }
    else
    {
      updateHeightStats(height, variance, npoints, Point.z);
      minHeight = std::min(minHeight, Point.z);
      maxHeight = std::max(maxHeight, Point.z);
    }
    npoints += 1;
  }
}
void obstacle_mapping::DenseMapping(grid_map::GridMap &map)
{
  auto &heightMatrix = map["elevation"];
  auto &npointsMatrix = map["n_points"];
  grid_map::GridMap::Matrix heightMatrix_BGK = heightMatrix;

  for (grid_map::GridMapIterator it(map); !it.isPastEnd(); ++it)
  {
    grid_map::Index index = *it;
    // skip observed point
    if (npointsMatrix(index(0), index(1)) != 0)
    {
      continue;
    }
    Eigen::Vector2d testpoint;
    bool inmap = map.getPosition(index, testpoint);
    if (!inmap)
    {
      continue;
    }
    // skip grids too close
    double pointDistance = (testpoint - robot_pose_.head<2>()).norm();
    if (pointDistance < 3.0 || pointDistance > maxsensorRangeLimit_)
    {
      continue;
    }
    // Training data
    vector<float> xTrainVec;     // training data x and y coordinates
    vector<float> yTrainVecElev; // training data elevation
    vector<float> yTrainVecOccu; // training data occupancy
    grid_map::Position center(testpoint(0), testpoint(1));
    for (grid_map::CircleIterator cit(map, center, predictionKernalSize_); !cit.isPastEnd(); ++cit)
    {
      grid_map::Index cindex = *cit;
      // save only observed grid in this scan 这个迭代器好像有bug
      // if (npointsMatrix(cindex(0), cindex(1)) == 0)
      // {
      //     continue;
      // }
      Eigen::Vector3d trainpoint;
      if (map.getPosition3("elevation", cindex, trainpoint))
      {
        xTrainVec.push_back(trainpoint(0));
        xTrainVec.push_back(trainpoint(1));
        yTrainVecElev.push_back(trainpoint(2));
      }
    }
    // no training data available, continue
    // if (xTrainVec.size() == 0)
    if (xTrainVec.size() < 3)
      continue;
    // convert from vector to eigen
    Eigen::MatrixXf xTrain = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(xTrainVec.data(), xTrainVec.size() / 2, 2);
    Eigen::MatrixXf yTrainElev = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(yTrainVecElev.data(), yTrainVecElev.size(), 1);
    // Test data (current grid)
    vector<float> xTestVec;
    xTestVec.push_back(testpoint(0));
    xTestVec.push_back(testpoint(1));
    Eigen::MatrixXf xTest = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(xTestVec.data(), xTestVec.size() / 2, 2);
    // Predict
    Eigen::MatrixXf Ks;           // covariance matrix
    covSparse(xTest, xTrain, Ks); // sparse kernel

    Eigen::MatrixXf ybarElev = (Ks * yTrainElev).array();
    Eigen::MatrixXf kbar = Ks.rowwise().sum().array();

    // Update Elevation with Prediction
    if (std::isnan(ybarElev(0, 0)) || std::isnan(kbar(0, 0)))
      continue;

    if (kbar(0, 0) == 0)
      continue;

    float elevation = ybarElev(0, 0) / kbar(0, 0);
    heightMatrix_BGK(index(0), index(1)) = elevation;
  }
  map.add("elevation_BGK", heightMatrix_BGK);
}

void obstacle_mapping::StepMapping(grid_map::GridMap &map, const grid_map::Index &index, const Eigen::Vector3d &f_point)
{
  grid_map::Position center = f_point.head<2>();
  double center_h = f_point.z();
  auto &stepMatrix = map["step"];
  // auto &minHeightMatrix = map["min_elevation"];
  // auto &maxHeightMatrix = map["max_elevation"];
  // auto &npointsMatrix = map["n_points"];
  double max_step = -1.0;
  for (grid_map::CircleIterator it(map, center, stepRadius_); !it.isPastEnd(); ++it)
  // for (grid_map::SpiralIterator it(map, center, ); !it.isPastEnd(); ++it)
  {
    grid_map::Index neighbor_idx = *it;
    Eigen::Vector3d point;
    if (map.getPosition3("elevation_BGK", neighbor_idx, point))
    {
      double step = std::abs(center_h - point.z());
      max_step = std::max(step, max_step);
    }
    // grid_map::Index neighbor_idx = *it;
    // if (npointsMatrix(neighbor_idx(0), neighbor_idx(1)) == 0)
    // {
    //     continue;
    // }
  }
  if (max_step >= 0)
  {
    stepMatrix(index(0), index(1)) = max_step / step_crit_;
  }
}

void obstacle_mapping::FeatureMapping(grid_map::GridMap &map)
{
  grid_map::Size gridMapSize = map.getSize();
  int robot_range = center_robot_index_(0) * gridMapSize(1) + center_robot_index_(1);
  // Set number of thread to use for parallel programming.
  std::unique_ptr<tbb::task_scheduler_init> TBBInitPtr;
  if (threadCount_ != -1)
  {
    TBBInitPtr.reset(new tbb::task_scheduler_init(threadCount_));
  }
  // Parallelized iteration through the map.
  tbb::parallel_for(0, gridMapSize(0) * gridMapSize(1), [&](int range)
                    {
    // Recover Cell index from range iterator.
    const grid_map::Index index(range / gridMapSize(1), range % gridMapSize(1));

    Eigen::Vector3d f_point = Eigen::Vector3d::Zero();
    if (map.getPosition3("elevation_BGK", index, f_point)) 
    {
        StepMapping(map, index, f_point);
    } 
    areaSingleNormalComputation(map, index); });
}

void obstacle_mapping::areaSingleNormalComputation(grid_map::GridMap &map, const grid_map::Index &index)
{
  // Requested position (center) of circle in map.
  grid_map::Position center;
  if (!map.getPosition(index, center))
  {
    ROS_ERROR("In file %s at line %d : The index is not in the map !", __FILE__, __LINE__);
    return;
  }
  // Prepare data computation. Check if area is bigger than cell.
  const double minAllowedEstimationRadius = 0.5 * map.getResolution();
  if (normal_estimationRadius_ <= minAllowedEstimationRadius)
  {
    ROS_DEBUG("Estimation radius is smaller than allowed by the map resolution (%f < %f)", normal_estimationRadius_, minAllowedEstimationRadius);
  }

  // Gather surrounding data.
  size_t nPoints = 0;
  grid_map::Position3 sum = grid_map::Position3::Zero();
  Eigen::Matrix3d sumSquared = Eigen::Matrix3d::Zero();
  for (grid_map::CircleIterator circleIterator(map, center, normal_estimationRadius_); !circleIterator.isPastEnd(); ++circleIterator)
  {
    grid_map::Position3 point;
    if (!map.getPosition3("elevation", *circleIterator, point))
    {
      continue;
    }
    nPoints++;
    sum += point;
    sumSquared.noalias() += point * point.transpose();
  }

  Eigen::Vector3d unitaryNormalVector = Eigen::Vector3d::Zero();
  if (nPoints < 4)
  {
    ROS_DEBUG("Not enough points to establish normal direction (nPoints = %i)", static_cast<int>(nPoints));
    unitaryNormalVector = Eigen::Vector3d::UnitZ();
  }
  else
  {
    const grid_map::Position3 mean = sum / nPoints;
    const Eigen::Matrix3d covarianceMatrix = sumSquared / nPoints - mean * mean.transpose();

    // Compute Eigenvectors.
    // Eigenvalues are ordered small to large.
    // Worst case bound for zero eigenvalue from : https://eigen.tuxfamily.org/dox/classEigen_1_1SelfAdjointEigenSolver.html
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver;
    solver.computeDirect(covarianceMatrix, Eigen::DecompositionOptions::ComputeEigenvectors);
    if (solver.eigenvalues()(1) > 1e-8)
    {
      unitaryNormalVector = solver.eigenvectors().col(0);
    }
    else
    { // If second eigenvalue is zero, the normal is not defined.
      ROS_DEBUG("Covariance matrix needed for eigen decomposition is degenerated.");
      ROS_DEBUG("Expected cause: data is on a straight line (nPoints = %i)", static_cast<int>(nPoints));
      unitaryNormalVector = Eigen::Vector3d::UnitZ();
    }
    // Check direction of the normal vector and flip the sign towards the user defined direction.
    if (unitaryNormalVector.dot(Eigen::Vector3d::UnitZ()) < 0.0)
    {
      unitaryNormalVector = -unitaryNormalVector;
    }

    float roughness = solver.eigenvalues()(0) / solver.eigenvalues().sum();

    map.at("normal_x", index) = unitaryNormalVector.x();
    map.at("normal_y", index) = unitaryNormalVector.y();
    map.at("normal_z", index) = unitaryNormalVector.z();
    map.at("slope", index) = std::acos(unitaryNormalVector.z()) / (slope_crit_ * M_PI);
    map.at("roughness", index) = roughness / roughness_crit_;
  }
}

void obstacle_mapping::simpleobstacledetection(grid_map::GridMap &map)
{
  auto &minHeightMatrix = map["min_elevation"];
  auto &maxHeightMatrix = map["max_elevation"];
  auto &npointsMatrix = map["n_points"];
  auto &obstacleMatrix = map["obstacle"];
  grid_map::Index index;

  for (grid_map::GridMapIterator it(map); !it.isPastEnd(); ++it)
  {
    index = *it;
    auto &npoints = npointsMatrix(index(0), index(1));
    auto &minHeight = minHeightMatrix(index(0), index(1));
    auto &maxHeight = maxHeightMatrix(index(0), index(1));
    auto &obstacle = obstacleMatrix(index(0), index(1));
    if (!npoints)
    {
      continue;
    }
    else
    {
      if (maxHeight - minHeight >= heightdiffthreshold_)
      {
        obstacle = 1;
      }
    }
  }
}

void obstacle_mapping::cloudobstacledetection(const pcl::PointCloud<pcl::PointXYZ> &localcloud, grid_map::GridMap &map)
{
  auto &minHeightMatrix = map["min_elevation"];
  auto &npointsMatrix = map["n_points"];
  auto &obstaclecntMatrix = map["obstacle_cnt"];
  auto &obstacleMatrix = map["obstacle"];

  for (const auto &point : localcloud.points)
  {
    grid_map::Index index;
    grid_map::Position pos2d(point.x, point.y);
    if (map.getIndex(pos2d, index))
    {
      auto &minHeight = minHeightMatrix(index(0), index(1));
      auto &obstacle_cnt = obstaclecntMatrix(index(0), index(1));
      if (point.z - minHeight > heightdiffthreshold_)
      {
        obstacle_cnt += 1;
      }
    }
  }

  for (grid_map::GridMapIterator it(map); !it.isPastEnd(); ++it)
  {
    grid_map::Index index = *it;
    auto &npoints = npointsMatrix(index(0), index(1));
    auto &obstacle_cnt = obstaclecntMatrix(index(0), index(1));
    auto &obstacle = obstacleMatrix(index(0), index(1));
    if (obstacle_cnt)
    {
      double cntratio = obstacle_cnt / npoints;
      if (cntratio >= cntratiothreshold_ ||
          obstacle_cnt > cntthreshold_)
      {
        obstacle = 1;
      }
    }
  }
}

void obstacle_mapping::mapobstacledetection(grid_map::GridMap &map)
{
  auto &heightMatrix = map["elevation"];
  auto &heightBGKMatrix = map["elevation_BGK"];
  auto &minHeightMatrix = map["min_elevation"];
  auto &maxHeightMatrix = map["max_elevation"];
  auto &slopeMatrix = map["slope"];
  auto &stepMatrix = map["step"];
  auto &roughnessMatrix = map["roughness"];
  auto &traversabilityMatrix = map["traversability"];
  auto &npointsMatrix = map["n_points"];
  auto &obstacleMatrix = map["obstacle"];
  grid_map::Index index;

  for (grid_map::GridMapIterator it(map); !it.isPastEnd(); ++it)
  {
    index = *it;
    auto &height = heightMatrix(index(0), index(1));
    auto &height_BGK = heightBGKMatrix(index(0), index(1));
    auto &minHeight = minHeightMatrix(index(0), index(1));
    auto &maxHeight = maxHeightMatrix(index(0), index(1));
    auto &slope = slopeMatrix(index(0), index(1));
    auto &step = stepMatrix(index(0), index(1));
    auto &roughness = roughnessMatrix(index(0), index(1));
    auto &traversability = traversabilityMatrix(index(0), index(1));
    auto &npoints = npointsMatrix(index(0), index(1));
    auto &obstacle = obstacleMatrix(index(0), index(1));
    // if (npoints)
    // {
    //     if (maxHeight - minHeight >= heightdiffthreshold_)
    //     {
    //         obstacle = 1;
    //     }
    // }
    if ((!std::isnan(slope) && slope >= 1) ||
        (!std::isnan(roughness) && roughness >= 1) ||
        (!std::isnan(step) && step >= 1) ||
        (!std::isnan(traversability) && traversability >= traversability_crit_))
    {
      obstacle = 1;
      traversability = 1;
      Eigen::Vector2d position = Eigen::Vector2d::Zero();
      if (map.getPosition(index, position))
      {
        if (position.norm() < minsensorRangeLimit_)
        {
          obstacle = 0;
          traversability = 0.5;
        }
      }
    }
  }
}

void obstacle_mapping::FeatureMerge()
{
  grid_map::Index center_in_submap;
  if (!global_submap_.getIndex(center_robot_, center_in_submap))
  {
    ROS_ERROR("In file %s at line %d : The position is not in the map !", __FILE__, __LINE__);
    return;
  }
  grid_map::Index submap_start_idx = center_robot_index_ - center_in_submap;

  auto &slopeMatrix = global_map_["slope"];
  auto &roughnessMatrix = global_map_["roughness"];
  auto &stepMatrix = global_map_["step"];
  auto &elevation_BGKMatrix = global_map_["elevation_BGK"];

  auto &slopeMatrix_sub = global_submap_["slope"];
  auto &roughnessMatrix_sub = global_submap_["roughness"];
  auto &stepMatrix_sub = global_submap_["step"];
  auto &elevation_BGKMatrix_sub = global_submap_["elevation_BGK"];

  for (grid_map::SubmapIterator subiter(global_map_, submap_start_idx, global_submap_.getSize()); !subiter.isPastEnd(); ++subiter)
  {
    grid_map::Index idx = *subiter;
    grid_map::Index subidx = idx - submap_start_idx;
    slopeMatrix(idx(0), idx(1)) = slopeMatrix_sub(subidx(0), subidx(1));
    roughnessMatrix(idx(0), idx(1)) = roughnessMatrix_sub(subidx(0), subidx(1));
    stepMatrix(idx(0), idx(1)) = stepMatrix_sub(subidx(0), subidx(1));
    elevation_BGKMatrix(idx(0), idx(1)) = elevation_BGKMatrix_sub(subidx(0), subidx(1));
  }
  return;
}

void obstacle_mapping::TraversabilityMapping(grid_map::GridMap &map)
{
  auto &slopeMatrix = map["slope"];
  auto &stepMatrix = map["step"];
  auto &roughnessMatrix = map["roughness"];
  auto &traversabilityMatrix = map["traversability"];
  for (grid_map::GridMapIterator it(map); !it.isPastEnd(); ++it)
  {
    grid_map::Index idx = *it;
    if (std::isnan(slopeMatrix(idx(0), idx(1))) || std::isnan(stepMatrix(idx(0), idx(1))) || std::isnan(roughnessMatrix(idx(0), idx(1))))
    {
      continue;
    }
    // if (std::isnan(slopeMatrix(idx(0), idx(1))) || std::isnan(roughnessMatrix(idx(0), idx(1))))
    // {
    //     continue;
    // }
    // traversabilityMatrix(idx(0), idx(1)) = (slopeMatrix(idx(0), idx(1)) + roughnessMatrix(idx(0), idx(1))) / 2;
    traversabilityMatrix(idx(0), idx(1)) = (slopeMatrix(idx(0), idx(1)) + stepMatrix(idx(0), idx(1)) + roughnessMatrix(idx(0), idx(1))) / 3;
  }
}

void obstacle_mapping::FuseMap()
{
  int matchodom_index = TimestampMatch(map_time_.toSec(), odom_deque_);
  cur_odom_ = odom_deque_[matchodom_index];
  if (first_map_init_)
  {
    first_map_init_ = false;
    fused_local_map_ = local_map_;
    last_odom_ = cur_odom_;
    return;
  }

  Eigen::Affine3f transform = Eigen::Affine3f::Identity();
  pcl::PointCloud<pcl::PointXYZ> historymap4transform, map4transform;
  // 计算当前里程计方向角yaw
  double odom_x = cur_odom_.pose.pose.orientation.x;
  double odom_y = cur_odom_.pose.pose.orientation.y;
  double odom_z = cur_odom_.pose.pose.orientation.z;
  double odom_w = cur_odom_.pose.pose.orientation.w;
  double siny_cosp = 2 * (odom_w * odom_z + odom_x * odom_y);
  double cosy_cosp = 1 - 2 * (odom_y * odom_y + odom_z * odom_z);
  double yaw_odom = atan2(siny_cosp, cosy_cosp);
  // 计算历史地图里程计方向角yaw
  double map_x = last_odom_.pose.pose.orientation.x;
  double map_y = last_odom_.pose.pose.orientation.y;
  double map_z = last_odom_.pose.pose.orientation.z;
  double map_w = last_odom_.pose.pose.orientation.w;
  double siny_cosp_map = 2 * (map_w * map_z + map_x * map_y);
  double cosy_cosp_map = 1 - 2 * (map_y * map_y + map_z * map_z);
  double yaw_map = atan2(siny_cosp_map, cosy_cosp_map);

  transform.translation() << -last_odom_.pose.pose.position.x + cur_odom_.pose.pose.position.x,
      -last_odom_.pose.pose.position.y + cur_odom_.pose.pose.position.y,
      -last_odom_.pose.pose.position.z + cur_odom_.pose.pose.position.z;
  transform.rotate(Eigen::AngleAxisf(yaw_odom - yaw_map, Eigen::Vector3f::UnitZ()));

  grid_map::Index index_his, index_cur;

  pcl::transformPointCloud(historymap4transform, map4transform, transform); // 通过转换矩阵transform将historymap4transform转换后存到map4transform

  for (grid_map::GridMapIterator it(fused_local_map_); !it.isPastEnd(); ++it)
  {
    index_his = *it;
    double height = fused_local_map_.at("elevation", index_his);
    double npoints = fused_local_map_.at("n_points", index_his);
    double obstacle = fused_local_map_.at("obstacle", index_his);
    double probability = fused_local_map_.at("probability", index_his);

    if (!std::isnan(height))
    {
      Eigen::Vector2d position;
      if (!fused_local_map_.getPosition(index_his, position))
      {
        ROS_ERROR("In file %s at line %d : The index is not in the map !", __FILE__, __LINE__);
        continue;
        ;
      }

      Eigen::Vector3f point(position.x(), position.y(), 0.0);
      Eigen::Vector3f transformed_point;
      pcl::transformPoint(point, transformed_point, transform);

      Eigen::Vector2d transformed_position(transformed_point(0), transformed_point(1));
      bool inside_map = local_map_.getIndex(transformed_position, index_cur);
      if (!inside_map)
      {
        continue;
      }
      double height_raw = local_map_.at("elevation", index_cur);
      double npoints_raw = local_map_.at("n_points", index_cur);
      // 高度更新
      local_map_.at("elevation", index_cur) = (npoints * height + npoints_raw * height_raw) / (npoints + npoints_raw);
      // 点数量更新
      local_map_.at("n_points", index_cur) += npoints;
      // 障碍物更新
      if (local_map_.at("obstacle", index_cur) == 1.0)
      {
        local_map_.at("probability", index_cur) = BayesUpdator(probability, 0.7);
      }
      else
      {
        local_map_.at("probability", index_cur) = BayesUpdator(probability, 0.2);
      }
    }
  }
  fused_local_map_ = local_map_;
  return;
}

double obstacle_mapping::BayesUpdator(double value_update, double value_observe)
{
  double tem_value = 0;
  if (value_update <= 0)
  {
    tem_value = value_observe;
    return tem_value;
  }
  double tem_odds = (value_update * value_observe) / (1 - value_observe - value_update + value_observe * value_update);
  tem_value = tem_odds / (1 + tem_odds);
  if (tem_value > 0.9)
    tem_value = 0.9;
  if (tem_value < 0.2)
    tem_value = 0.2;
  return tem_value;
}

int obstacle_mapping::TimestampMatch(double fixtimestamp, std::deque<nav_msgs::Odometry> target_deque)
{
  int index = 0;
  double min_diff = 100;
  for (int i = 0; i < target_deque.size(); i++)
  {
    double timediff = fabs(target_deque[i].header.stamp.toSec() - fixtimestamp);
    if (timediff < min_diff)
    {
      min_diff = timediff;
      index = i;
    }
  }
  if (min_diff > 1.0)
  {
    ROS_WARN("Time delay between lidar and pose is over 1 sec!");
  }
  return index;
}

void obstacle_mapping::updateHeightStats(float &height, float &variance, float n, float new_height)
{

  const float delta = (new_height - height);
  const float delta_n = delta / n;

  // Update mean
  height += delta_n;

  // Update variance using the more stable algorithm
  // M2 = M2 + delta * (new_value - updated_mean)
  // where M2 is the sum of squared differences from the mean
  variance += delta * (new_height - height);

  // Convert M2 to variance
  if (n > 1)
  {
    variance = variance / (n - 1); // Use (n-1) for sample variance
  }
}

void obstacle_mapping::dist(const Eigen::MatrixXf &xStar, const Eigen::MatrixXf &xTrain, Eigen::MatrixXf &d) const
{
  d = Eigen::MatrixXf::Zero(xStar.rows(), xTrain.rows());
  for (int i = 0; i < xStar.rows(); ++i)
  {
    d.row(i) = (xTrain.rowwise() - xStar.row(i)).rowwise().norm();
  }
}

void obstacle_mapping::covSparse(const Eigen::MatrixXf &xStar, const Eigen::MatrixXf &xTrain, Eigen::MatrixXf &Kxz) const
{
  dist(xStar / (predictionKernalSize_ + 0.1), xTrain / (predictionKernalSize_ + 0.1), Kxz);
  Kxz = (((2.0f + (Kxz * 2.0f * 3.1415926f).array().cos()) * (1.0f - Kxz.array()) / 3.0f) +
         (Kxz * 2.0f * 3.1415926f).array().sin() / (2.0f * 3.1415926f))
            .matrix() *
        1.0f;
  // Clean up for values with distance outside length scale, possible because Kxz <= 0 when dist >= predictionKernalSize
  for (int i = 0; i < Kxz.rows(); ++i)
    for (int j = 0; j < Kxz.cols(); ++j)
      if (Kxz(i, j) < 0)
        Kxz(i, j) = 0;
}

void obstacle_mapping::saveMatrixToCSV(const Eigen::MatrixXf &matrix, const std::string &filename)
{
  std::ofstream file(filename);
  if (!file.is_open())
  {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return;
  }

  for (int i = 0; i < matrix.rows(); ++i)
  {
    for (int j = 0; j < matrix.cols(); ++j)
    {
      file << matrix(i, j);
      if (j < matrix.cols() - 1)
      {
        file << ","; // 行内各元素间添加逗号
      }
    }
    file << std::endl; // 换行
  }

  file.close();
  std::cout << "Matrix saved to " << filename << std::endl;
}

Eigen::MatrixXf obstacle_mapping::loadMatrixFromCSV(const std::string &filename)
{
  std::ifstream file(filename);
  if (!file.is_open())
  {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  std::vector<std::vector<float>> data;
  std::string line;

  // 逐行读取文件
  while (std::getline(file, line))
  {
    std::stringstream ss(line);
    std::string value;
    std::vector<float> row;

    // 逐个读取列值
    while (std::getline(ss, value, ','))
    {
      row.push_back(std::stof(value));
    }
    data.push_back(row);
  }

  file.close();

  // 将数据转换为 Eigen 矩阵
  Eigen::MatrixXf matrix(data.size(), data[0].size());
  for (size_t i = 0; i < data.size(); ++i)
  {
    for (size_t j = 0; j < data[i].size(); ++j)
    {
      matrix(i, j) = data[i][j];
    }
  }

  return matrix;
}

void obstacle_mapping::criticalfootprintsfilter(grid_map::GridMap &map)
{
  auto &heightMatrix = map["elevation"];
  auto &heightBGKMatrix = map["elevation_BGK"];
  auto &minHeightMatrix = map["min_elevation"];
  auto &maxHeightMatrix = map["max_elevation"];
  auto &slopeMatrix = map["slope"];
  auto &stepMatrix = map["step"];
  auto &roughnessMatrix = map["roughness"];
  auto &traversabilityMatrix = map["traversability"];
  auto &obstacleMatrix = map["obstacle"];
  auto &criticalMatrix = map["critical"];
  grid_map::Index index;
  grid_map::Size size = map.getSize();

  for (grid_map::GridMapIterator it(map); !it.isPastEnd(); ++it)
  {
    index = *it;
    auto &height = heightMatrix(index(0), index(1));
    auto &height_BGK = heightBGKMatrix(index(0), index(1));
    auto &minHeight = minHeightMatrix(index(0), index(1));
    auto &maxHeight = maxHeightMatrix(index(0), index(1));
    auto &slope = slopeMatrix(index(0), index(1));
    auto &step = stepMatrix(index(0), index(1));
    auto &roughness = roughnessMatrix(index(0), index(1));
    auto &traversability = traversabilityMatrix(index(0), index(1));
    auto &obstacle = obstacleMatrix(index(0), index(1));
    auto &critical = criticalMatrix(index(0), index(1));

    // if (((!std::isnan(slope) && slope >= slope_crit_fp_ && slope <= 1.0) ||
    //     (!std::isnan(roughness) && roughness >= roughness_crit_fp_ && roughness <= 1.0) ||
    //     (!std::isnan(step) && step >= step_crit_fp_ && step <= 1.0)
    //     // (!std::isnan(traversability) && traversability >= traversability_crit_fp_)
    //     )
    //     // traversability != 1
    //     )
    // if (traversability >= traversability_crit_fp_ && traversability <= traversability_crit_fp_up_ &&
    //     step >= step_crit_fp_ && obstacle != 1 && roughness >= roughness_crit_fp_)
    if (
        (roughness >= roughness_crit_fp_ &&
         roughness <= roughness_crit_fp_up_) ||
        (slope >= slope_crit_fp_ && slope <= slope_crit_fp_up_)
            // (step >= step_crit_fp_ && step <= step_crit_fp_up_)
            // || step >= step_crit_fp_ && step <= step_crit_fp_up_
            // && traversability >= traversability_crit_fp_ && traversability <= traversability_crit_fp_up_
            && obstacle != 1)
    {
      int nannum = 0;
      bool out = false;
      for (int i = index(0) - 5; i < index(0) + 5; i++)
      {
        for (int j = index(1) - 5; j < index(1) + 5; j++)
        {
          // grid_map::Index ridx(i, j);
          if (i < 0 || i >= size.x() || j < 0 || j >= size.y())
          {
            out = true;
          }
          else
          {
            if (std::isnan(heightBGKMatrix(i, j)))
            {
              nannum++;
            }
          }
        }
      }
      if (out || nannum > 3)
      {
        continue;
      }
      if (index(0) > 120)
      {
        continue;
      }
      critical = 1;
      // traversability = 1;
    }
  }
}

void obstacle_mapping::pointcloudinterpolation(pcl::PointCloud<pcl::PointXYZ>::Ptr &input)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr inputcloudPtr(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ> inputcloud;

  for (auto it = input->begin(); it != input->end(); it++)
  {
    pcl::PointXYZ pt;
    pt.x = it->x;
    pt.y = it->y;
    pt.z = it->z;
    inputcloud.push_back(pt);
  }
  *inputcloudPtr = inputcloud;

  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr dense_points(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr dense_points_converted(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ> mls;

  double search_radius = 0.5;   // 0.03
  double sampling_radius = 0.2; // 0.005
  double step_size = 0.1;       // 0.005
  double gauss_param = (double)std::pow(search_radius, 2);
  int pol_order = 5;

  mls.setComputeNormals(true);
  mls.setInputCloud(inputcloudPtr);
  mls.setPolynomialFit(true);
  mls.setSearchMethod(tree);
  mls.setSearchRadius(search_radius);
  mls.setUpsamplingMethod(pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ>::RANDOM_UNIFORM_DENSITY);
  mls.setUpsamplingRadius(sampling_radius);
  mls.setUpsamplingStepSize(step_size);
  mls.setPolynomialOrder(pol_order);
  mls.setSqrGaussParam(gauss_param); // (the square of the search radius works best in general)
  // mls.setDilationVoxelSize();//Used only in the VOXEL_GRID_DILATION upsampling method
  mls.setPointDensity(1000); // 15
  mls.process(*dense_points);

  pcl::copyPointCloud(*dense_points, *dense_points_converted);
  *input += *dense_points_converted;
  pcl::io::savePCDFile("/home/wxr/proj/terrain/ws_trav/data/dense_pcd/trench00.pcd", *input);
}

void obstacle_mapping::publocalmap()
{
  grid_map_msgs::GridMap msg;
  msg.info.header.stamp = map_time_;
  grid_map::GridMapRosConverter::toMessage(local_map_, msg);
  local_map_pub_.publish(msg);
}

void obstacle_mapping::pubfusedlocalmap()
{
  grid_map_msgs::GridMap msg;
  msg.info.header.stamp = map_time_;
  grid_map::GridMapRosConverter::toMessage(fused_local_map_, msg);
  local_fused_map_pub_.publish(msg);
}

void obstacle_mapping::pubglobalmap()
{
  grid_map_msgs::GridMap msg;
  msg.info.header.stamp = map_time_;
  grid_map::GridMapRosConverter::toMessage(global_submap_, msg);
  global_map_pub_.publish(msg);
}

void obstacle_mapping::visualizepointcloud()
{
  sensor_msgs::PointCloud2 cloud;
  pcl::PointCloud<pcl::PointXYZI> pcl_cloud;
  grid_map::Index index;
  for (grid_map::GridMapIterator it(local_map_); !it.isPastEnd(); ++it)
  {
    index = *it;
    grid_map::Position position;
    local_map_.getPosition(index, position);

    float traversability = local_map_.at("traversability", index);
    float obstacle = local_map_.at("obstacle", index);
    float intensity = (traversability == 1.0 || obstacle == 1.0) ? 1.0 : 0.0;

    pcl::PointXYZI point;
    point.x = position.x();
    point.y = position.y();
    point.z = local_map_.at("elevation_BGK", index);
    ; // Assuming a 2D map, set height to 0
    point.intensity = intensity;

    pcl_cloud.points.push_back(point);
  }

  pcl_cloud.width = pcl_cloud.points.size();
  pcl_cloud.height = 1; // Unorganized point cloud
  pcl_cloud.is_dense = false;

  pcl::toROSMsg(pcl_cloud, cloud);
  cloud.header.frame_id = "map"; // Change to your frame id
  cloud.header.stamp = ros::Time::now();

  pointcloud_pub_.publish(cloud);
}