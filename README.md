## apriltags_rgbd_node
#### Individual tag detection process:
+ Subscribe to apriltag detections
+ Extract 2D pixels that are within the convex hull defined by tag corners
+ Map pixels to 3D points based on depth data
+ Fit bayes plane to isolated pixels
+ Compute average 3D position of pixels - **This is used as the tag's position**
+ Compute average point between corner 1 and 2, then subtract tag position -
this corresponds to the general direction of the tag's x-axis
+ Project the point onto the bayes plane to create an orthogonal vector
+ Compute cross product between computed vector and normal vector to create a second orthogonal vector
+ Normalize vectors
+ Create rotation matrix from vectors (vectors are rotation matrix columns)
+ Convert from rotation matrix to quaternion **This is used as the tag's orientation**

#### Body pose estimation process:
+ Using config file, pre-compute theoretical point positions in body frame on startup
+ For each detected body that has recent tag detections, extract detected corner positions for all tags in body. Only consider tags detected within the last `TF_TIMEOUT` seconds
+ Based on tag detections, extract corresponding points in pre-computed set of points in body frame
+ Use ICP to estimate body pose. Assumes 1:1 correspondence between body points and detected points **This result is used as the body's transform**

#### Filtering process:
+ On each iteration, keep running list of corner detections. Use parameter `WINDOW_SIZE` for number of points to keep
+ Exclude outliers from current set based on standard deviation value (note: these aren't deleted, just excluded from the computation this iteration)
+ Compute average of the remaining corners. Use this for computations this iteration

## Dependencies:
+ Python
  + numpy
  + pandas(?)
  + opencv
  + sklearn

+ ROS
  + See `package.xml` for full list

## Usage:
+ Ensure ROS driver for kinect is running
+ `roslaunch apriltags_rgbd bringup.launch`


## TODO:
+ Parameterize global vars
+ Improve usage of corners - use more pixels to estimate corner positions in case corners are not easily detected
