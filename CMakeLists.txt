cmake_minimum_required(VERSION 3.10)
project(ncnn_retinaface)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


find_package(OpenCV REQUIRED)
find_package(ncnn REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} ${ncnn_INCLUDE_DIRS} $ENV{CONDA_PREFIX}/include)
link_directories( $ENV{CONDA_PREFIX}/lib)

# Các executable hiện có
add_executable(retinaface src/main.cpp)
add_executable(embedding src/embedding.cpp)
add_executable(compare_embedding src/compare_embedding.cpp)
add_executable(capture src/capture.cpp)
add_executable(recognition src/recognition.cpp)

target_link_libraries(retinaface ${OpenCV_LIBS} ncnn)
target_link_libraries(embedding ${OpenCV_LIBS} ncnn)
target_link_libraries(compare_embedding ${OpenCV_LIBS} ncnn)
target_link_libraries(capture ${OpenCV_LIBS} ncnn)
target_link_libraries(recognition ${OpenCV_LIBS} ncnn)

