cmake_minimum_required(VERSION 2.8)

project(CubemapSLAM)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native")

add_definitions(-DWITH_BOW2)

find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV Found")
endif(OpenCV_FOUND)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/ThirdParty/DBoW2/
    ${PROJECT_SOURCE_DIR}/ThirdParty/g2o/
    ${Pangolin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_library(${PROJECT_NAME} SHARED
    src/CamModelGeneral.cpp
    src/Frame.cpp 
    src/System.cpp
    src/Tracking.cpp
    src/ORBExtractor.cpp
    src/Initializer.cpp
    src/ORBMatcher.cpp
    src/Map.cpp
    src/MapPoint.cpp
    src/KeyFrame.cpp
    src/LocalMapping.cpp
    src/Viewer.cpp
    src/MapDrawer.cpp
    src/FrameDrawer.cpp
    src/Optimizer.cpp
    src/g2o_cubemap_vertices_edges.cpp
    src/Converter.cpp
    src/KeyFrameDatabase.cpp
    src/LoopClosing.cpp
    src/Sim3Solver.cpp
    src/PnPsolver.cpp
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS} 
    ${Pangolin_LIBRARIES}
    ${PROJECT_SOURCE_DIR}/ThirdParty/g2o/lib/libg2o.so
    ${PROJECT_SOURCE_DIR}/ThirdParty/DBoW2/lib/libDBoW2.so
)


##for fangshan dataset
add_executable(cubemap_fangshan 
    Examples/cubemap_fangshan.cpp)
target_link_libraries(cubemap_fangshan ${PROJECT_NAME})


##for lafida dataset
add_executable(cubemap_lafida
    Examples/cubemap_lafida.cpp)
target_link_libraries(cubemap_lafida ${PROJECT_NAME})

