cmake_minimum_required(VERSION 3.10)
project(SplineTrajectoryExamples)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)  

# Fallback for systems without proper Eigen3 CMake config
if(NOT TARGET Eigen3::Eigen)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(EIGEN3 REQUIRED eigen3)
    include_directories(${EIGEN3_INCLUDE_DIRS})
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/perception_tool)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/frontend_tool)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/backend_tool)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

#test
add_executable(test_grid_map src/test_grid_map.cpp)
target_link_libraries(test_grid_map Eigen3::Eigen  ${OpenCV_LIBS})

add_executable(test_astar src/test_astar.cpp)
target_link_libraries(test_astar Eigen3::Eigen  ${OpenCV_LIBS})

add_executable(test_opt src/test_opt.cpp)
target_link_libraries(test_opt Eigen3::Eigen  ${OpenCV_LIBS})

