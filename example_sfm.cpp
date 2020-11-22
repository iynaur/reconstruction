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
#include <opencv2/sfm/conditioning.hpp>
#include <opencv2/sfm/fundamental.hpp>
#include <opencv2/sfm/io.hpp>
#include <opencv2/sfm/numeric.hpp>
#include <opencv2/sfm/projection.hpp>
#include <opencv2/sfm/triangulation.hpp>

#include <opencv2/sfm/reconstruct.hpp>
#include <opencv2/sfm/simple_pipeline.hpp>

#include <iostream>

using namespace cv;
//using namespace cv::sfm;
using namespace std;

#include <Eigen/Core>

#include <opencv2/core/eigen.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "libmv_light/libmv_capi.h"

using namespace std;

namespace cv
{
namespace sfm
{

/* Parses a given array of 2d points into the libmv tracks structure
 */

static void
parser_2D_tracks( const std::vector<Mat> &points2d, libmv::Tracks &tracks )
{
  const int nframes = static_cast<int>(points2d.size());
  for (int frame = 0; frame < nframes; ++frame) {
    const int ntracks = points2d[frame].cols;
    for (int track = 0; track < ntracks; ++track) {
      const Vec2d track_pt = points2d[frame].col(track);
      if ( track_pt[0] > 0 && track_pt[1] > 0 )
        tracks.Insert(frame, track, track_pt[0], track_pt[1]);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////

/* Parses a given set of matches into the libmv tracks structure
 */

static void
parser_2D_tracks( const libmv::Matches &matches, libmv::Tracks &tracks )
{
  std::set<Matches::ImageID>::const_iterator iter_image =
    matches.get_images().begin();

  bool is_first_time = true;

  for (; iter_image != matches.get_images().end(); ++iter_image) {
    // Exports points
    Matches::Features<PointFeature> pfeatures =
      matches.InImage<PointFeature>(*iter_image);

    while(pfeatures) {

      double x = pfeatures.feature()->x(),
             y = pfeatures.feature()->y();

      // valid marker
      if ( x > 0 && y > 0 )
      {
        tracks.Insert(*iter_image, pfeatures.track(), x, y);

        if ( is_first_time )
          is_first_time = false;
      }

      // lost track
      else if ( x < 0 && y < 0 )
      {
        is_first_time = true;
      }

      pfeatures.operator++();
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////

/* Computes the 2d features matches between a given set of images and call the
 * reconstruction pipeline.
 */

static libmv_Reconstruction *libmv_solveReconstructionImpl(
  const std::vector<String> &images,
  const libmv_CameraIntrinsicsOptions* libmv_camera_intrinsics_options,
  libmv_ReconstructionOptions* libmv_reconstruction_options)
{
  Ptr<Feature2D> edetector = ORB::create(10000);
  Ptr<Feature2D> edescriber = xfeatures2d::DAISY::create();
  //Ptr<Feature2D> edescriber = xfeatures2d::LATCH::create(64, true, 4);
  std::vector<std::string> sImages;
  for (int i=0;i<images.size();i++)
      sImages.push_back(images[i].c_str());
  cout << "Initialize nViewMatcher ... ";
  libmv::correspondence::nRobustViewMatching nViewMatcher(edetector, edescriber);

  cout << "OK" << endl << "Performing Cross Matching ... ";
  nViewMatcher.computeCrossMatch(sImages); cout << "OK" << endl;

  // Building tracks
  libmv::Tracks tracks;
  libmv::Matches matches = nViewMatcher.getMatches();
  parser_2D_tracks( matches, tracks );

  // Perform reconstruction
  return libmv_solveReconstruction(tracks,
                                   libmv_camera_intrinsics_options,
                                   libmv_reconstruction_options);
}

///////////////////////////////////////////////////////////////////////////////////////////////

template <class T>
class SFMLibmvReconstructionImpl : public T
{
public:
  SFMLibmvReconstructionImpl(const libmv_CameraIntrinsicsOptions &camera_instrinsic_options,
                             const libmv_ReconstructionOptions &reconstruction_options) :
    libmv_reconstruction_options_(reconstruction_options),
    libmv_camera_intrinsics_options_(camera_instrinsic_options) {}

  /* Run the pipeline given 2d points
   */

  virtual void run(InputArrayOfArrays _points2d)
  {
    std::vector<Mat> points2d;
    _points2d.getMatVector(points2d);
    CV_Assert( _points2d.total() >= 2 );

    // Parse 2d points to Tracks
    Tracks tracks;
    parser_2D_tracks(points2d, tracks);

    // Set libmv logs level
    libmv_initLogging("");

    if (libmv_reconstruction_options_.verbosity_level >= 0)
    {
      libmv_startDebugLogging();
      libmv_setLoggingVerbosity(
        libmv_reconstruction_options_.verbosity_level);
    }

    // Perform reconstruction
    libmv_reconstruction_ =
      *libmv_solveReconstruction(tracks,
                                 &libmv_camera_intrinsics_options_,
                                 &libmv_reconstruction_options_);
  }

  virtual void run(InputArrayOfArrays points2d, InputOutputArray K, OutputArray Rs,
                   OutputArray Ts, OutputArray points3d)
  {
    // Run the pipeline
    run(points2d);

    // Extract Data
    extractLibmvReconstructionData(K, Rs, Ts, points3d);
  }


  /* Run the pipeline given a set of images
   */

  virtual void run(const std::vector <String> &images)
  {
    // Set libmv logs level
    libmv_initLogging("");

    if (libmv_reconstruction_options_.verbosity_level >= 0)
    {
      libmv_startDebugLogging();
      libmv_setLoggingVerbosity(
        libmv_reconstruction_options_.verbosity_level);
    }

    // Perform reconstruction

    libmv_reconstruction_ =
      *libmv_solveReconstructionImpl(images,
                                     &libmv_camera_intrinsics_options_,
                                     &libmv_reconstruction_options_);
  }


  virtual void run(const std::vector <String> &images, InputOutputArray K, OutputArray Rs,
                   OutputArray Ts, OutputArray points3d)
  {
    // Run the pipeline
    run(images);

    // Extract Data
    extractLibmvReconstructionData(K, Rs, Ts, points3d);
  }

  virtual double getError() const { return libmv_reconstruction_.error; }

  virtual void
  getPoints(OutputArray points3d) {
    const size_t n_points =
      libmv_reconstruction_.reconstruction.AllPoints().size();

    points3d.create(n_points, 1, CV_64F);

    Vec3d point3d;
    for ( size_t i = 0; i < n_points; ++i )
    {
      for ( int j = 0; j < 3; ++j )
        point3d[j] =
          libmv_reconstruction_.reconstruction.AllPoints()[i].X[j];
      Mat(point3d).copyTo(points3d.getMatRef(i));
    }

  }

  virtual cv::Mat getIntrinsics() const {
    Mat K;
    eigen2cv(libmv_reconstruction_.intrinsics->K(), K);
    return K;
  }

  virtual void
  getCameras(OutputArray Rs, OutputArray Ts) {
    const size_t n_views =
      libmv_reconstruction_.reconstruction.AllCameras().size();

    Rs.create(n_views, 1, CV_64F);
    Ts.create(n_views, 1, CV_64F);

    Matx33d R;
    Vec3d t;
    for(size_t i = 0; i < n_views; ++i)
    {
      eigen2cv(libmv_reconstruction_.reconstruction.AllCameras()[i].R, R);
      eigen2cv(libmv_reconstruction_.reconstruction.AllCameras()[i].t, t);
      Mat(R).copyTo(Rs.getMatRef(i));
      Mat(t).copyTo(Ts.getMatRef(i));
    }
  }

  virtual void setReconstructionOptions(
    const libmv_ReconstructionOptions &libmv_reconstruction_options) {
      libmv_reconstruction_options_ = libmv_reconstruction_options;
  }

  virtual void setCameraIntrinsicOptions(
    const libmv_CameraIntrinsicsOptions &libmv_camera_intrinsics_options) {
      libmv_camera_intrinsics_options_ = libmv_camera_intrinsics_options;
  }

private:

  void
  extractLibmvReconstructionData(InputOutputArray K,
                                 OutputArray Rs,
                                 OutputArray Ts,
                                 OutputArray points3d)
  {
    getCameras(Rs, Ts);
    getPoints(points3d);
    getIntrinsics().copyTo(K.getMat());
  }

  libmv_Reconstruction libmv_reconstruction_;
  libmv_ReconstructionOptions libmv_reconstruction_options_;
  libmv_CameraIntrinsicsOptions libmv_camera_intrinsics_options_;
};

///////////////////////////////////////////////////////////////////////////////////////////////

Ptr<SFMLibmvEuclideanReconstruction>
SFMLibmvEuclideanReconstruction::create(const libmv_CameraIntrinsicsOptions &camera_instrinsic_options,
                                        const libmv_ReconstructionOptions &reconstruction_options)
{
  return makePtr<SFMLibmvReconstructionImpl<SFMLibmvEuclideanReconstruction> >(camera_instrinsic_options,reconstruction_options);
}

///////////////////////////////////////////////////////////////////////////////////////////////

} /* namespace cv */
} /* namespace sfm */

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

int getdir(const string _filename, std::vector<String> &files)
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

int main()
{
  int argc = 5;
  char* argv[5];
  argv[1] = "dinoR_good_silhouette_images.txt";
  argv[2] = "400";
  argv[3] = "320";
  argv[4] = "240";
  // Read input parameters
  if ( argc != 5 )
  {
    help();
    exit(0);
  }

  // Parse the image paths
  std::vector<String> images_paths;
  getdir(argv[1], images_paths);

  // Build instrinsics
  float f  = atof(argv[2]),
        cx = atof(argv[3]), cy = atof(argv[4]);
  Matx33d K = Matx33d( f, 0, cx,
                       0, f, cy,
                       0, 0,  1);
  bool is_projective = true;
  std::vector<cv::Mat> Rs_est, ts_est, points3d_estimated;
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
    //cout << points3d_estimated[i] << endl;
    int id = 0;
    for(mat_it = points3d_estimated[i].begin<double>(); mat_it != points3d_estimated[i].end<double>(); mat_it++) {
      points_file << *mat_it << " ";
      cloudp->points[i].data[id++] = *mat_it;
    }
    points_file << "\n";
  }
  auto& pts= cloudp->points;
  std::sort(pts.begin(), pts.end(), [](const auto &p1, const auto&p2){
      return fabs(p1.x) + fabs(p1.y) + fabs(p1.z) <
              fabs(p2.x) + fabs(p2.y) + fabs(p2.z);
  });
  cloudp->points.resize(max(1.0, cloudp->points.size()*0.8));
  cloudp->resize(cloudp->points.size());
  pcl::io::savePCDFile("ans.pcd", *cloudp);

  cout << "Done. Points saved to points.txt" << endl;
  points_file.close();

  return 0;
}
