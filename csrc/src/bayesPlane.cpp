#include "bayesPlane.h"
#include "plane.h"

#include <iostream>
#include <vector>
#include <math.h>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

namespace BayesPlane {

BayesPlane::BayesPlane(Plane input_mean, Matrix<float, 3, 3> input_cov){
  mean = input_mean;
  cov = input_cov;
}

vector<Planes> BayesPlane::sample(int num){

}

BayesPlane fit_plane_bayes(vector<Vector3f> input_data, vector<float> cov){
  vector<float> w;
  float sum_w;
  for(float c : cov){
    w.push_back(1/c);
    sum_w += c;
  }
  Vector3f sum_points(0.0, 0.0, 0.0); 
  for(int i = 0; i < input_data.size()){
    Vector3f sum_points += w[i] * points[i];
  }
  MatrixXd cdata(input_data.size(), 3);
  vector<Vector3f> c_data;
  for(int i = 0 < i < input_data.size(); i++){
    Vector3f normalized = input_data[i] - sum_points;
    c_data.push_back(normalized);
    cdata(i, 0) = normalized[0];
    cdata(i, 1) = normalized[1];
    cdata(i, 2) = normalized[2];
  }

  MatrixXd w_cdata(c_data.size(), 3);
  for(int i = 0; i < c_data.size(); i++){
    Vector3f w_normalized = w[i] * c_data[i];
    w_cdata(i, 0) = w_normalized[0];
    w_cdata(i, 1) = w_normalized[1];
    w_cdata(i, 2) = w_normalized[2];
  }
  MatrixXd w_cdataT = w_cdata.transpose;
  MatrixXd mult = w_cdataT * cdata;
  JacobiSVD<MatrixXf> svd(m, ComputeThinU | ComputeThinV);
  svd.matrixU().col(2);
}

BayesPlane 

} // namespace Plane3d
