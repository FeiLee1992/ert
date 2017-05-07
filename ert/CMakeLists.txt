project(ert)
cmake_minimum_required(VERSION 2.8.12)

set(OpenCV_DIR ../opencv/share/OpenCV)

find_package( OpenCV REQUIRED )



include_directories(./code/inc)


set(source ./code/src/utilis.cpp ./code/src/regression.cpp ./code/src/gbdt.cpp ./code/src/main.cpp)


add_executable(ert ${source})



SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")
target_link_libraries(ert ${OpenCV_LIBS})



