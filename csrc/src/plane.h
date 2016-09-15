#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

namespace Plane3d {
class Plane3d{
  public:
    Plane3d(Vector3f n, double d);
    int getDim();
    Vector4f vectorize();
    vector<Vector3f> basis();
    vector<Vector3f> project(vector<Vector3f> points);
    vector<float> distance(vector<Vector3f> points);
    vector<float> point_probability(vector<Vector3f> points, vector<float> cov);
    MatrixXd sample(MatrixXd points); 
  private:
    Vector3f n;
    double d;
    int dim;
};
}