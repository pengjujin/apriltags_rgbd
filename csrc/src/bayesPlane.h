#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "plane.h"
using namespace std;
using namespace Eigen;

namespace BayesPlane {
class BayesPlane{
  public:
    BayesPlane(Plane mean, Matrix<float, 3, 3> cov);
    vector<float> point_probability(vector<Vector3f> points, vector<float> cov, int numSamples = 100);
    static BayesPlane fit_plane_bayes(data_points, cov);

  private:
    Plane mean;
    Matrix<float, 3, 3> cov;
};
}