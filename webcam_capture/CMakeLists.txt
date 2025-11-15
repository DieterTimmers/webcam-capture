cmake_minimum_required(VERSION 3.16)
project(webcam_capture)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add OpenCV - Updated path for external build
set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../install/opencv/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

# Create library
add_library(webcam_capture
    src/webcam_capture.cpp
)

target_link_libraries(webcam_capture ${OpenCV_LIBS})
target_include_directories(webcam_capture PUBLIC 
    include
    ${OpenCV_INCLUDE_DIRS}
)

# Optional: Create example executable
add_executable(webcam_example examples/main.cpp)
target_link_libraries(webcam_example webcam_capture)