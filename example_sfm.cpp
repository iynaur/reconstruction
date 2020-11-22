#define CERES_FOUND true

#include <opencv2/sfm.hpp>
#include <iostream>
#include <fstream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

//#include "precomp.hpp"

//#if CERES_FOUND

// Eigen
#include <Eigen/Core>

// OpenCV
#include <opencv2/sfm.hpp>

#include <iostream>

using namespace cv;
//using namespace cv::sfm;
using namespace std;

namespace cv
{
namespace mysfm
{
using namespace cv;

/** @brief Reconstruct 3d points from 2d correspondences while performing autocalibration.
  @param points2d Input vector of vectors of 2d points (the inner vector is per image).
  @param Ps Output vector with the 3x4 projections matrices of each image.
  @param points3d Output array with estimated 3d points.
  @param K Input/Output camera matrix \f$K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$. Input parameters used as initial guess.
  @param is_projective if true, the cameras are supposed to be projective.

  This method calls below signature and extracts projection matrices from estimated K, R and t.

   @note
    - Tracks must be as precise as possible. It does not handle outliers and is very sensible to them.
*/
CV_EXPORTS
void
reconstruct(InputArrayOfArrays points2d, OutputArray Ps, OutputArray points3d, InputOutputArray K,
            bool is_projective = false);

/** @brief Reconstruct 3d points from 2d correspondences while performing autocalibration.
  @param points2d Input vector of vectors of 2d points (the inner vector is per image).
  @param Rs Output vector of 3x3 rotations of the camera.
  @param Ts Output vector of 3x1 translations of the camera.
  @param points3d Output array with estimated 3d points.
  @param K Input/Output camera matrix \f$K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$. Input parameters used as initial guess.
  @param is_projective if true, the cameras are supposed to be projective.

  Internally calls libmv simple pipeline routine with some default parameters by instatiating SFMLibmvEuclideanReconstruction class.

  @note
    - Tracks must be as precise as possible. It does not handle outliers and is very sensible to them.
    - To see a working example for camera motion reconstruction, check the following tutorial: @ref tutorial_sfm_trajectory_estimation.
*/
CV_EXPORTS
void
reconstruct(InputArrayOfArrays points2d, OutputArray Rs, OutputArray Ts, InputOutputArray K,
            OutputArray points3d, bool is_projective = false);

/** @brief Reconstruct 3d points from 2d images while performing autocalibration.
  @param images a vector of string with the images paths.
  @param Ps Output vector with the 3x4 projections matrices of each image.
  @param points3d Output array with estimated 3d points.
  @param K Input/Output camera matrix \f$K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$. Input parameters used as initial guess.
  @param is_projective if true, the cameras are supposed to be projective.

  This method calls below signature and extracts projection matrices from estimated K, R and t.

   @note
    - The images must be ordered as they were an image sequence. Additionally, each frame should be as close as posible to the previous and posterior.
    - For now DAISY features are used in order to compute the 2d points tracks and it only works for 3-4 images.
*/
CV_EXPORTS
void
reconstruct(const std::vector<String> images, OutputArray Ps, OutputArray points3d,
            InputOutputArray K, bool is_projective = false);

/** @brief Reconstruct 3d points from 2d images while performing autocalibration.
  @param images a vector of string with the images paths.
  @param Rs Output vector of 3x3 rotations of the camera.
  @param Ts Output vector of 3x1 translations of the camera.
  @param points3d Output array with estimated 3d points.
  @param K Input/Output camera matrix \f$K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$. Input parameters used as initial guess.
  @param is_projective if true, the cameras are supposed to be projective.

  Internally calls libmv simple pipeline routine with some default parameters by instatiating SFMLibmvEuclideanReconstruction class.

   @note
    - The images must be ordered as they were an image sequence. Additionally, each frame should be as close as posible to the previous and posterior.
    - For now DAISY features are used in order to compute the 2d points tracks and it only works for 3-4 images.
    - To see a working example for scene reconstruction, check the following tutorial: @ref tutorial_sfm_scene_reconstruction.
*/
CV_EXPORTS
void
reconstruct(const std::vector<String> images, OutputArray Rs, OutputArray Ts,
            InputOutputArray K, OutputArray points3d, bool is_projective = false);

  template<class T>
  void
  reconstruct_(const T &input, OutputArray Rs, OutputArray Ts, InputOutputArray K, OutputArray points3d, const bool refinement=true)
  {
    // Initial reconstruction
    const int keyframe1 = 1, keyframe2 = 2;
    const int select_keyframes = 1; // enable automatic keyframes selection
    const int verbosity_level = -1; // mute libmv logs

    // Refinement parameters
    const int refine_intrinsics = ( !refinement ) ? 0 :
        sfm::SFM_REFINE_FOCAL_LENGTH |
                                                    sfm::SFM_REFINE_PRINCIPAL_POINT |
                                                    sfm::SFM_REFINE_RADIAL_DISTORTION_K1 |
                                                    sfm::SFM_REFINE_RADIAL_DISTORTION_K2;

    // Camera data
    Matx33d Ka = K.getMat();
    const double focal_length = Ka(0,0);
    const double principal_x = Ka(0,2), principal_y = Ka(1,2), k1 = 0, k2 = 0, k3 = 0;

    // Set reconstruction options
    sfm::libmv_ReconstructionOptions reconstruction_options(keyframe1, keyframe2, refine_intrinsics, select_keyframes, verbosity_level);

    sfm::libmv_CameraIntrinsicsOptions camera_instrinsic_options =
      sfm::libmv_CameraIntrinsicsOptions(sfm::SFM_DISTORTION_MODEL_POLYNOMIAL,
                                    focal_length, principal_x, principal_y,
                                    k1, k2, k3);

    //-- Instantiate reconstruction pipeline
    Ptr<sfm::BaseSFM> reconstruction =
      sfm::SFMLibmvEuclideanReconstruction::create(camera_instrinsic_options, reconstruction_options);

    //-- Run reconstruction pipeline
    reconstruction->run(input, K, Rs, Ts, points3d);

  }


  //  Reconstruction function for API
  void
  reconstruct(InputArrayOfArrays points2d, OutputArray Ps, OutputArray points3d, InputOutputArray K,
              bool is_projective)
  {
    const int nviews = points2d.total();
    CV_Assert( nviews >= 2 );

    // OpenCV data types
    std::vector<Mat> pts2d;
    points2d.getMatVector(pts2d);
    const int depth = pts2d[0].depth();

    Matx33d Ka = K.getMat();

    // Projective reconstruction

    if (is_projective)
    {

      if ( nviews == 2 )
      {
        // Get Projection matrices
        Matx33d F;
        Matx34d P, Pp;

        sfm::normalizedEightPointSolver(pts2d[0], pts2d[1], F);
        sfm::projectionsFromFundamental(F, P, Pp);
        Ps.create(2, 1, depth);
        Mat(P).copyTo(Ps.getMatRef(0));
        Mat(Pp).copyTo(Ps.getMatRef(1));

        // Triangulate and find 3D points using inliers
        sfm::triangulatePoints(points2d, Ps, points3d);
      }
      else
      {
        std::vector<Mat> Rs, Ts;
        reconstruct(points2d, Rs, Ts, Ka, points3d, is_projective);

        // From Rs and Ts, extract Ps
        const int nviews = Rs.size();
        Ps.create(nviews, 1, depth);

        Matx34d P;
        for (size_t i = 0; i < nviews; ++i)
        {
          sfm::projectionFromKRt(Ka, Rs[i], Vec3d(Ts[i]), P);
          Mat(P).copyTo(Ps.getMatRef(i));
        }

        Mat(Ka).copyTo(K.getMat());
      }

    }


    // Affine reconstruction

    else
    {
      // TODO: implement me
      CV_Error(Error::StsNotImplemented, "Affine reconstruction not yet implemented");
    }

  }


  void
  reconstruct(InputArrayOfArrays points2d, OutputArray Rs, OutputArray Ts, InputOutputArray K,
              OutputArray points3d, bool is_projective)
  {
    const int nviews = points2d.total();
    CV_Assert( nviews >= 2 );


    // Projective reconstruction

    if (is_projective)
    {

      // calls simple pipeline
      reconstruct_(points2d, Rs, Ts, K, points3d);

    }

    // Affine reconstruction

    else
    {
      // TODO: implement me
      CV_Error(Error::StsNotImplemented, "Affine reconstruction not yet implemented");
    }

  }


  void
  reconstruct(const std::vector<cv::String> images, OutputArray Ps, OutputArray points3d,
              InputOutputArray K, bool is_projective)
  {
    const int nviews = static_cast<int>(images.size());
    CV_Assert( nviews >= 2 );

    Matx33d Ka = K.getMat();
    const int depth = Mat(Ka).depth();

    // Projective reconstruction

    if ( is_projective )
    {
      std::vector<Mat> Rs, Ts;
      reconstruct(images, Rs, Ts, Ka, points3d, is_projective);

      // From Rs and Ts, extract Ps

      const int nviews_est = Rs.size();
      Ps.create(nviews_est, 1, depth);

      Matx34d P;
      for (size_t i = 0; i < nviews_est; ++i)
      {
        sfm::projectionFromKRt(Ka, Rs[i], Vec3d(Ts[i]), P);
        Mat(P).copyTo(Ps.getMatRef(i));
      }

      Mat(Ka).copyTo(K.getMat());
      }


    // Affine reconstruction

    else
    {
      // TODO: implement me
      CV_Error(Error::StsNotImplemented, "Affine reconstruction not yet implemented");
    }

  }


  void
  reconstruct(const std::vector<cv::String> images, OutputArray Rs, OutputArray Ts,
              InputOutputArray K, OutputArray points3d, bool is_projective)
  {
    const int nviews = static_cast<int>(images.size());
    CV_Assert( nviews >= 2 );

    // Projective reconstruction

    if ( is_projective )
    {
      reconstruct_(images, Rs, Ts, K, points3d);
    }


    // Affine reconstruction

    else
    {
      // TODO: implement me
      CV_Error(Error::StsNotImplemented, "Affine reconstruction not yet implemented");
    }

  }

} // namespace sfm
} // namespace cv

//#endif /* HAVE_CERES */

using namespace std;
using namespace cv;
using namespace cv::sfm;

static void help() {
  cout
      << "\n------------------------------------------------------------------------------------\n"
      << " This program shows the multiview reconstruction capabilities in the \n"
      << " OpenCV Structure From Motion (SFM) module.\n"
      << " It reconstruct a scene from a set of 2D images \n"
      << " Usage:\n"
      << "        example_sfm_scene_reconstruction <path_to_file> <f> <cx> <cy>\n"
      << " where: path_to_file is the file absolute path into your system which contains\n"
      << "        the list of images to use for reconstruction. \n"
      << "        f  is the focal lenght in pixels. \n"
      << "        cx is the image principal point x coordinates in pixels. \n"
      << "        cy is the image principal point y coordinates in pixels. \n"
      << "------------------------------------------------------------------------------------\n\n"
      << endl;
}

int getdir(const string _filename, vector<String> &files)
{
  ifstream myfile(_filename.c_str());
  if (!myfile.is_open()) {
    cout << "Unable to read file: " << _filename << endl;
    exit(0);
  } else {;
    size_t found = _filename.find_last_of("/\\");
    string line_str, path_to_file = _filename.substr(0, found);
    while ( getline(myfile, line_str) ) {
      cout << line_str << endl;
      files.push_back(String(line_str));
    }
  }
  return 1;
}

int main(int argc, char* argv[])
{
  // Read input parameters
  if ( argc != 5 )
  {
    help();
    exit(0);
  }

  // Parse the image paths
  vector<String> images_paths;
  getdir(argv[1], images_paths);

  // Build instrinsics
  float f  = atof(argv[2]),
        cx = atof(argv[3]), cy = atof(argv[4]);
  Matx33d K = Matx33d( f, 0, cx,
                       0, f, cy,
                       0, 0,  1);
  bool is_projective = true;
  vector<Mat> Rs_est, ts_est, points3d_estimated;
  mysfm::reconstruct(images_paths, Rs_est, ts_est, K, points3d_estimated, is_projective);

  // Print output
  cout << "\n----------------------------\n" << endl;
  cout << "Reconstruction: " << endl;
  cout << "============================" << endl;
  cout << "Estimated 3D points: " << points3d_estimated.size() << endl;
  cout << "Estimated cameras: " << Rs_est.size() << endl;
  cout << "Refined intrinsics: " << endl << K << endl << endl;
  cout << "============================" << endl;

  // recover estimated points3d
  ofstream points_file;
  cv::MatIterator_<double> mat_it;
  points_file.open("points.txt");
  points_file.precision(std::numeric_limits<double>::digits10);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudp (new pcl::PointCloud<pcl::PointXYZ>);
  cloudp->resize(points3d_estimated.size());
  for (int i = 0; i < points3d_estimated.size(); ++i) {
    cout << points3d_estimated[i] << endl;
    int id = 0;
    for(mat_it = points3d_estimated[i].begin<double>(); mat_it != points3d_estimated[i].end<double>(); mat_it++) {
      points_file << *mat_it << " ";
      cloudp->points[i].data[id++] = *mat_it;
    }
    points_file << "\n";
  }
  pcl::io::savePCDFile("ans.pcd", *cloudp);

  cout << "Done. Points saved to points.txt" << endl;
  points_file.close();

  return 0;
}
