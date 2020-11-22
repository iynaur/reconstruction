# define the source files
SET(MULTIVIEW_SRC conditioning.cc
                  euclidean_resection.cc
                  fundamental.cc
                  fundamental_kernel.cc
                  homography.cc
                  panography.cc
                  panography_kernel.cc
                  projection.cc
                  robust_estimation.cc
                  robust_fundamental.cc
                  robust_resection.cc
                  triangulation.cc
                  twoviewtriangulation.cc)

# define the header files (make the headers appear in IDEs.)
FILE(GLOB MULTIVIEW_HDRS *.h)

ADD_LIBRARY(mymultiview SHARED ${MULTIVIEW_SRC} ${MULTIVIEW_HDRS})
TARGET_LINK_LIBRARIES(mymultiview  ${GLOG_LIBRARY} mynumeric)
IF(TARGET Eigen3::Eigen)
  TARGET_LINK_LIBRARIES(mymultiview  Eigen3::Eigen)
ENDIF()

LIBMV_INSTALL_LIB(multiview)
