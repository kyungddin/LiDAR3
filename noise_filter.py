#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np
from std_msgs.msg import Header
import open3d as o3d

pub = None

def remove_noise_with_open3d(msg):
    # PointCloud2 → numpy 변환
    points_list = []
    for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
        x, y, z = p
        points_list.append([x, y, z])

    if len(points_list) == 0:
        rospy.loginfo("No points received.")
        return None

    points_np = np.array(points_list, dtype=np.float32)

    # numpy → Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)

    # ✅ 노이즈 제거 (Statistical Outlier Removal)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd_filtered = pcd.select_by_index(ind)

    # Open3D → numpy
    filtered_np = np.asarray(pcd_filtered.points)

    # numpy → PointCloud2 메시지
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = msg.header.frame_id

    processed_msg = pc2.create_cloud_xyz32(header, filtered_np)
    return processed_msg

def callback(msg):
    global pub
    processed_msg = remove_noise_with_open3d(msg)
    if processed_msg:
        pub.publish(processed_msg)

def listener():
    global pub
    rospy.init_node('ouster_noise_filtered', anonymous=True)
    rospy.Subscriber('/ouster/points', PointCloud2, callback)
    pub = rospy.Publisher('/processed_point', PointCloud2, queue_size=1)
    rospy.loginfo("Subscribed to /ouster/points, publishing noise-filtered point cloud")
    rospy.spin()

if __name__ == '__main__':
    listener()