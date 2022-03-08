cmake_minimum_required(VERSION 2.8.3)
project(apriltags_rgbd)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  apriltag_msgs
  cv_bridge
  geometry_msgs
  message_filters
  rospy
  sensor_msgs
  std_msgs
  tf2_ros
  visualization_msgs
)
