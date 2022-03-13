# This message is the same as the PointCloud message, but adds a label
# for distinguishing between messages sent to the same topic.

# Time of sensor data acquisition, coordinate frame ID.
Header header

# Label for categorizing PointCloud
std_msgs/String label

# Array of 3d points. Each Point32 should be interpreted as a 3d point
# in the frame given in the header.
geometry_msgs/Point32[] points
