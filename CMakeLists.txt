cmake_minimum_required (VERSION 3.10.2)
SET(CMAKE_CXX_COMPILER icpc)
SET(INTEL_MKL_LIBS    "-lmkl_rt")
include_directories(${CMAKE_SOURCE_DIR}/include)

project (test_corr_2d)
file(GLOB SOURCES ${CMAKE_SOURCE_DIR}/src/*corr*.cpp)
add_executable (${PROJECT_NAME} "${SOURCES}")
target_link_libraries (${PROJECT_NAME} ${INTEL_MKL_LIBS})

project (test_conv_2d)
file(GLOB SOURCES ${CMAKE_SOURCE_DIR}/src/*conv*.cpp)
add_executable (${PROJECT_NAME} "${SOURCES}")
target_link_libraries (${PROJECT_NAME} ${INTEL_MKL_LIBS})




