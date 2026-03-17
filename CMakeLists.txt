cmake_minimum_required(VERSION 3.0.2)
project(obstacle_mapping)
## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)
set(CMAKE_BUILD_TYPE RELEASE)
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
# set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O0")
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  std_msgs
  nav_msgs
  rosbag
  tf
  tf_conversions
  grid_map_core
  grid_map_ros
  grid_map_msgs
  geometry_msgs
  visualization_msgs
  message_filters
  pcl_ros
  pcl_conversions
)

find_package(PCL REQUIRED)

find_package(Boost REQUIRED COMPONENTS system)
find_package(Eigen3 REQUIRED)

pkg_check_modules(TBB "tbb")
if (TBB_FOUND)
  add_definitions(
    ${TBB_CFLAGS}
    ${TBB_CFLAGS_OTHER}
  )
else()
  message([FATAL_ERROR] "tbb module not found")
endif ()

catkin_package(
  INCLUDE_DIRS 
    include
    ${TBB_INCLUDE_DIRS}
  LIBRARIES obstacle_mapping_lib
  CATKIN_DEPENDS 
    roscpp 
    sensor_msgs 
    std_msgs
    rosbag
  DEPENDS 
    system_lib
)

catkin_install_python(PROGRAMS
  scripts/cloud_merger_and_accumulator.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${TBB_INCLUDE_DIRS}
)

# Declare a C++ library
add_library(obstacle_mapping_lib
  src/obstacle_mapping.cpp
)
target_link_libraries(obstacle_mapping_lib ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${TBB_LIBRARIES})

add_executable(obstacle_mapping_node src/obstacle_mapping_node.cpp)
add_dependencies(obstacle_mapping_node obstacle_mapping_lib)
target_link_libraries(obstacle_mapping_node ${catkin_LIBRARIES} obstacle_mapping_lib)

add_executable(pose_prediction_node src/pose_prediction_node.cpp)
target_link_libraries(pose_prediction_node ${catkin_LIBRARIES})

add_executable(pose_prediction src/pose_prediction.cpp)
target_link_libraries(pose_prediction ${catkin_LIBRARIES})

add_executable(key_pub_node src/key_pub_node.cpp)
target_link_libraries(key_pub_node ${catkin_LIBRARIES})

add_executable(gridmap_saver_node src/gridmap_saver.cpp)
target_link_libraries(gridmap_saver_node ${catkin_LIBRARIES})

add_executable(show_vehicle_node src/show_vehicle_node.cpp)
target_link_libraries(show_vehicle_node ${catkin_LIBRARIES})

add_executable(recieve_points src/recieve_points.cpp)
target_link_libraries(recieve_points ${catkin_LIBRARIES})

add_executable(publish_points src/publish_points.cpp)
target_link_libraries(publish_points ${catkin_LIBRARIES})

add_executable(showpcd src/showpcd.cpp)
target_link_libraries(showpcd ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(show_robot_map_pose src/show_robot_map_pose.cpp)
target_link_libraries(show_robot_map_pose ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(multi_robot_mapping_node src/multi_robot_mapping.cpp)
target_link_libraries(multi_robot_mapping_node ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(multi_robot_mapping_offline_node
  src/multi_robot_mapping_offline.cpp
  src/multi_robot_mapping.cpp)
target_compile_definitions(multi_robot_mapping_offline_node PRIVATE MULTI_ROBOT_MAPPING_NO_MAIN)
target_link_libraries(multi_robot_mapping_offline_node ${catkin_LIBRARIES} ${PCL_LIBRARIES})
