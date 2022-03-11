## apriltags_rgbd_node
Process:
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
