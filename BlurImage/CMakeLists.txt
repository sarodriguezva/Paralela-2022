cmake_minimum_required(VERSION 3.0.0)
set(CMAKE_CXX_STANDARD 14)
project(BlurVideo VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(BlurVideo src/main_mpi.cpp)

target_link_libraries(BlurVideo ${OpenCV_LIBS} OpenMP::OpenMP_CXX MPI::MPI_CXX)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
