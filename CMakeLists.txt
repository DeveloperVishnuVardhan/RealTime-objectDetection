cmake_minimum_required(VERSION 2.8)
project(main.cpp)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(main.cpp main.cpp filters.cpp filters.h)
add_executable(imgDisplay.cpp imgDisplay.cpp filters.cpp filters.h csv_util.cpp csv_util.h distance_metrics.cpp distance_metrics.h)
target_link_libraries(main.cpp ${OpenCV_LIBS})
target_link_libraries(imgDisplay.cpp ${OpenCV_LIBS})

