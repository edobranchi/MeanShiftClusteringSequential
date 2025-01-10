cmake_minimum_required(VERSION 3.10)
project(MeanShiftClusterSeq)

# Find OpenCV package
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Add include directories for OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})

# Add the executable target
add_executable(MeanShiftClusterSeq main.cpp)

# Link OpenCV libraries
target_link_libraries(MeanShiftClusterSeq ${OpenCV_LIBS})

