#include "plane.h"
#include <iostream>
#include <vector>
#include <math.h>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

namespace Plane3d {

Plane3d::Plane3d(Vector3f n_in, double d_in){
  n = n_in;
  n.normalize();
  d = d_in;
}

Vector4f Plane3d::vectorize(){
  Vector4f v(n[0], n[1], n[2], d);
  return v;
}

vector<Vector3f> Plane3d::basis(){
  Matrix<float, 4, 3> A;
  A << n[0], n[1], n[2], 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
  Matrix<float, 3, 4> A_trans = A.transpose();
  HouseholderQR<Matrix<float,3,4>> qr(A_trans);
  Matrix<float, 3, 3> Q = qr.householderQ(); 
  Vector3f basis1 = Q.col(1);
  Vector3f basis2 = Q.col(2);
  vector<Vector3f> all_basis;
  all_basis.push_back(basis1);
  all_basis.push_back(basis2);
  return all_basis;
}

vector<Vector3f> Plane3d::project(vector<Vector3f> points){
  vector<Vector3f> projected_points; 
  for(Vector3f point : points){
    float dist = 0;
    for(int i = 0; i < 2; i++){
        dist += n[i] * (point[i] - d*(n[i])); 
    } 
    Vector3f projected_point = point - dist * n; 
    projected_points.push_back(projected_point);
  }
}

vector<float> Plane3d::distance(vector<Vector3f> points){
  vector<float> distances;
  for (Vector3f point : points){
    float distance = point.dot(n) - d;
    distances.push_back(distance);
  }
  return distances; 
}

vector<float> Plane3d::point_probability(vector<Vector3f> points, 
                                         vector<float> cov){
  vector<float> all_point_probability;
  for(int i = 0; i < points.size(); i++){
    Vector3f point = points[i];
    float current_cov = cov[i];
    float k = d / point.dot(n);
    float dr = std::abs(1 - k) * point.norm();
    float prob = 1 - erf(dr / sqrt(2.0 * current_cov));
    all_point_probability.push_back(prob);
  } 
}

vector<Vector3f> Plane3d::sample(int M){
  vector<Vector3f> basis = basis();
  
}

} // namespace Plane3d


int main(){
  Vector3f n(1.0,1.0,1.0);
  double d = 1.0;
  //Plane3d::Plane3d p1(n, d);
  return 0;
}