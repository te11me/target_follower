cmake_minimum_required(VERSION 3.0.2)
project(target_follower)

set(OpenCV_DIR "/opt/opencv-4.5.5/lib/cmake/opencv4")
#静态链接OpenCV4.5.5
set(OpenCV_USE_STATIC_LIBS TRUE)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  image_transport
  cv_bridge
  dynamic_reconfigure
  tf
)

# 必须先声明动态重配置参数再调用catkin_package
generate_dynamic_reconfigure_options(
  cfg/TargetFollower.cfg
)

# 设置TensorRT路径
set(TensorRT_INCLUDE_DIRS "/usr/include/aarch64-linux-gnu")
set(TensorRT_LIBRARIES 
    "/usr/lib/aarch64-linux-gnu/libnvinfer.so"
    "/usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so"
    "/usr/lib/aarch64-linux-gnu/libnvonnxparser.so"
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES target_follower
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs image_transport cv_bridge dynamic_reconfigure
)

find_package(OpenCV 4.5.5 REQUIRED
COMPONENTS  
# core
imgproc
# video
tracking  # 必须添加tracking模块
cudaimgproc
highgui
objdetect
)

# 打印调试信息
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

# 关键修改点：
# #   COMPONENTS 
#     core 
#     imgproc 
#     # video 
#     tracking  # OpenCV3的tracking模块名称
# )

find_package(CUDA REQUIRED)

find_path(TensorRT_INCLUDE NvInfer.h
    PATHS ${TensorRT_INCLUDE_DIRS}
    REQUIRED
)

find_library(TensorRT_LIBRARY nvinfer
    PATHS /usr/lib/aarch64-linux-gnu
    REQUIRED
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${TensorRT_INCLUDE_DIRS}
  # 强制优先使用自定义OpenCV头文件
  /opt/opencv-4.5.5/include/opencv4  # 添加此行
  ${OpenCV_INCLUDE_DIRS}
  # /home/wheeltec/opencv-3.4.5/modules/video/include  # 添加OpenCV视频模块头文件路径
  ${PROJECT_SOURCE_DIR}/../devel/include  # 添加这一行确保包含生成的头文件路径
)

link_directories(
  /usr/lib/aarch64-linux-gnu
  ${CUDA_LIBRARIES}
  ${TensorRT_LIBRARY}  # 添加TensorRT库路径
)

add_executable(target_follower src/target_follower.cpp)
add_dependencies(target_follower target_follower_gencfg) 
target_link_libraries(target_follower
  # opencv_tracking  # 先链接tracking
  ${OpenCV_LIBS}       # 先链接OpenCV4.5.5的库
  ${catkin_LIBRARIES}  # 后链接ROS相关库
  nvinfer
  nvonnxparser
  cuda
  cudart
)

install(DIRECTORY launch cfg models
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

find_package(Threads REQUIRED)
target_link_libraries(target_follower 
    ${catkin_LIBRARIES} 
    Threads::Threads
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")