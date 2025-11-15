cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(webcam_capture)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add OpenCV
set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../install/opencv/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

# Add LibTorch - Manual approach (no FetchContent)
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

message(STATUS "Found LibTorch: ${Torch_VERSION}")
message(STATUS "CUDA Available: ${TORCH_CUDA_VERSION}")

# Create library
add_library(webcam_capture 
    src/webcam_capture.cpp
    src/torch_inference.cpp
)

target_link_libraries(webcam_capture 
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
)

target_include_directories(webcam_capture PUBLIC 
    include
    ${OpenCV_INCLUDE_DIRS}
)

# Optional: Create example executable
add_executable(webcam_example examples/main.cpp)
target_link_libraries(webcam_example webcam_capture)

# Copy torch libraries on Windows (optional on Linux)
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET webcam_example
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:webcam_example>)
endif (MSVC)