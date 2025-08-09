#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import open3d as o3d

marker_pub = None

def publish_cube(center, extent, frame_id="ouster", marker_id=0):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "bounding_box"
    marker.id = marker_id
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.pose.position.x = center[0]
    marker.pose.position.y = center[1]
    marker.pose.position.z = center[2]
    marker.pose.orientation.x = 0
    marker.pose.orientation.y = 0
    marker.pose.orientation.z = 0
    marker.pose.orientation.w = 1
    marker.scale.x = extent[0]
    marker.scale.y = extent[1]
    marker.scale.z = extent[2]
    marker.color = ColorRGBA(1.0, 0.0, 0.0, 0.5)  # 반투명 빨강
    marker.lifetime = rospy.Duration(0)  # 영구
    marker_pub.publish(marker)

def callback(msg):
    points = []
    for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
        points.append([p[0], p[1], p[2]])

    if len(points) == 0:
        rospy.logwarn("PointCloud 비어 있음")
        return

    np_points = np.array(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_points)

    # DBSCAN 클러스터링
    labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=30, print_progress=False))
    max_label = labels.max()
    rospy.loginfo(f"클러스터 수: {max_label + 1}")

    plane_id = 0
    for i in range(max_label + 1):
        cluster_indices = np.where(labels == i)[0]
        cluster_pcd = pcd.select_by_index(cluster_indices)

        # 클러스터에 평면 분할 시도
        if len(cluster_pcd.points) < 100:
            continue

        plane_model, inliers = cluster_pcd.segment_plane(
            distance_threshold=0.02,
            ransac_n=3,
            num_iterations=1000
        )
        if len(inliers) < 100:
            continue

        inlier_cloud = cluster_pcd.select_by_index(inliers)
        aabb = inlier_cloud.get_axis_aligned_bounding_box()
        center = aabb.get_center()
        extent = aabb.get_extent()

        rospy.loginfo(f"[Plane {plane_id}] Center: {center}, Extent: {extent}, Inliers: {len(inliers)}")
        publish_cube(center, extent, msg.header.frame_id, plane_id)
        plane_id += 1

def main():
    global marker_pub
    rospy.init_node('plane_bbox_publisher', anonymous=True)
    rospy.Subscriber('/ouster/points', PointCloud2, callback)
    marker_pub = rospy.Publisher('/bounding_box_marker', Marker, queue_size=10)
    rospy.loginfo("▶ 바운딩 박스 퍼블리셔 노드 시작됨 (/ouster/points → /bounding_box_marker)")
    rospy.spin()

if __name__ == '__main__':
    main()
