cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
add_compile_options(-std=c++11)
project(taget)
list(APPEND CMAKE_PREFIX_PATH "/home/pinjin/all_libtorch/libtorch_1.7.1")#libtorch路径
list(APPEND CMAKE_PREFIX_PATH "~/opencv-4.5.2/release")#opencv make路径
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Python3 COMPONENTS Development NumPy)
find_package(PythonLibs 3.8)#乱写的

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories("/home/pinjin/download/matplotlib-cpp/") #路径需要根据自己路径做调整	
include_directories("/usr/local/lib/python3.8/dist-packages/numpy/core/include")#numpy
include_directories(taget PRIVATE ${PYTHON_INCLUDE_DIRS})
add_executable(taget taget.cpp)
target_link_libraries(taget "${TORCH_LIBRARIES}")
target_link_libraries(taget "${OpenCV_LIBS}")	
target_link_libraries(taget ${PYTHON_LIBRARIES})
set_property(TARGET taget PROPERTY CXX_STANDARD 14)
