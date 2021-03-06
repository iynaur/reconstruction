cmake_minimum_required(VERSION 3.2)

project(sfm)


set(CMAKE_AUTOMOC ON) # For meta object compiler
set(CMAKE_AUTORCC ON) # Resource files
set(CMAKE_AUTOUIC ON) # UI files

# Find the QtWidgets library
find_package(Qt5 COMPONENTS
             Core
             Widgets
             Gui
             OpenGL)

find_package (VTK REQUIRED)
find_package (PCL 1.8 REQUIRED)

# Fix a compilation bug under ubuntu 16.04 (Xenial)
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

find_package(OpenCV 3.4 REQUIRED)

#PCL
include_directories (${PCL_INCLUDE_DIRS})
link_directories    (${PCL_LIBRARY_DIRS})
add_definitions     (${PCL_DEFINITIONS})

find_package(LibXml2 REQUIRED)
include_directories(${LIBXML2_INCLUDE_DIR})



add_definitions(-std=c++11)
set(CMAKE_CXX_STANDARD 14)
ADD_DEFINITIONS (${QT_DEFINITIONS})

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


set(CMAKE_EXE_LINKER_FLAGS "-Wl,-rpath=.")

### LIBMV LIGHT DEFINITIONS ###
find_package(Ceres)
include_directories (libmv_light)
set(LIBMV_LIGHT_INCLUDES
  "${CMAKE_CURRENT_LIST_DIR}/src/libmv_light"
  "${OpenCV_SOURCE_DIR}/include/opencv"
  "${GLOG_INCLUDE_DIRS}"
  "${GFLAGS_INCLUDE_DIRS}"
)

set(LIBMV_LIGHT_LIBS
  correspondence
  multiview
  numeric
  ${GLOG_LIBRARIES}
  ${GFLAGS_LIBRARIES}
)

if(Ceres_FOUND)
  add_definitions("-DCERES_FOUND=1")
  list(APPEND LIBMV_LIGHT_LIBS simple_pipeline)
  list(APPEND LIBMV_LIGHT_INCLUDES "${CERES_INCLUDE_DIR}")
else()
  add_definitions("-DCERES_FOUND=0")
  message(STATUS "CERES support is disabled. Ceres Solver for reconstruction API is required.")
endif()

### COMPILE WITH C++11 IF CERES WAS COMPILED WITH C++11

if(Ceres_FOUND)
  list (FIND CERES_COMPILED_COMPONENTS "C++11" _index)
  if (${_index} GREATER -1)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  endif()
endif()
# build libmv_light
if(NOT CMAKE_VERSION VERSION_LESS 2.8.11) # See ocv_target_include_directories() implementation
  if(TARGET ${the_module})
    get_target_property(__include_dirs ${the_module} INCLUDE_DIRECTORIES)
    include_directories(${__include_dirs})
  endif()
endif()
include_directories(${OCV_TARGET_INCLUDE_DIRS_${the_module}})
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/libmv_light" "${CMAKE_CURRENT_BINARY_DIR}/src/libmv")

#ocv_target_link_libraries(${the_module} ${LIBMV_LIGHT_LIBS})

set(CPPS
  example_sfm.cpp
)

set(HEADS


)

include_directories(${PROJECT_SOURCE_DIR}/common/include)


include_directories(
${PROJECT_SOURCE_DIR}/include)



IF( "${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    set(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/Release")
    set(LIBRARY_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/Release")
    MESSAGE( STATUS "CMAKE_BUILD_TYPE set ${CMAKE_BUILD_TYPE}" )
ELSEIF( "${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/Debug")
    set(LIBRARY_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/Debug")
    MESSAGE( STATUS "CMAKE_BUILD_TYPE set ${CMAKE_BUILD_TYPE}" )
ELSE()
    set(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")
    set(LIBRARY_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")
    MESSAGE( STATUS "CMAKE_BUILD_TYPE not set yet ${CMAKE_BUILD_TYPE}" )
ENDIF()



add_executable(${PROJECT_NAME}
    ${RESOURCES} ${CPPS} ${HEADERS} ${FORMS}
    ${DEPRECATED} ${HARDWARE} ${CAM_LIST} ${HANDEYE_LIST}  ${COMMON_CPP})

target_link_libraries(  ${PROJECT_NAME} ${QT_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS}
    libceres.so
    -lglog
    -lgflags
    mycorrespondence
    mymultiview
    mynumeric
    mysimple_pipeline
    )

#add_executable(leaf leaf.cpp )
#target_link_libraries(leaf PUBLIC  common)





qt5_use_modules(${PROJECT_NAME} Core Widgets Gui OpenGL )


