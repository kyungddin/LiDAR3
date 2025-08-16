#!/usr/bin/env python3

# 노이즈 제거 & 바운딩박스 및 텍스트 생성 완성판 (노이즈 제거 비활성화)

import rospy
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import open3d as o3d

marker_pub = None
cloud_pub = None  # ✅ 노이즈 제거 후 포인트 퍼블리셔

#####################################################################################################################

def o3d_to_pointcloud2(o3d_cloud, frame_id="ouster"):
    """Open3D PointCloud → ROS PointCloud2 변환"""
    points = np.asarray(o3d_cloud.points)
    header = rospy.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id
    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)
    ]
    return pc2.create_cloud(header, fields, points)

#####################################################################################################################

def publish_cube(center, extent, frame_id="ouster", marker_id=0):

    # 바운딩박스
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
    marker.color = ColorRGBA(0.0, 0.0, 1.0, 0.5)  # 반투명 파랑
    marker.lifetime = rospy.Duration(0) # 바운딩 박스의 Life Time
    marker_pub.publish(marker)

    # 텍스트 ("Mirror")
    text_marker = Marker()
    text_marker.header.frame_id = frame_id
    text_marker.header.stamp = rospy.Time.now()
    text_marker.ns = "bounding_box_text"
    text_marker.id = marker_id + 1000  # ID 충돌 방지
    text_marker.type = Marker.TEXT_VIEW_FACING
    text_marker.action = Marker.ADD
    text_marker.pose.position.x = center[0]
    text_marker.pose.position.y = center[1]
    text_marker.pose.position.z = center[2] + extent[2] / 2 + 0.1  # 박스 위쪽에 표시
    text_marker.pose.orientation.x = 0
    text_marker.pose.orientation.y = 0
    text_marker.pose.orientation.z = 0
    text_marker.pose.orientation.w = 1
    text_marker.scale.z = 0.2  # 글씨 크기 (m 단위)
    text_marker.color = ColorRGBA(0.0, 0.0, 1.0, 1.0)  # 파란색, 불투명
    text_marker.text = "Mirror"
    text_marker.lifetime = rospy.Duration(0)
    marker_pub.publish(text_marker)

#####################################################################################################################

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

    # <<< MODIFIED: 노이즈 제거 로직 삭제 >>>
    # pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.5)
    # rospy.loginfo(f"노이즈 제거 후 포인트 수: {len(pcd.points)}")

    # ✅ 퍼블리시 (이제 원본 포인트 클라우드를 발행)
    cloud_pub.publish(o3d_to_pointcloud2(pcd, frame_id=msg.header.frame_id))

    # 클러스터링
    labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=30, print_progress=False))
    max_label = labels.max()
    rospy.loginfo(f"클러스터 수: {max_label + 1}")

    plane_id = 0
    for i in range(max_label + 1):
        cluster_indices = np.where(labels == i)[0]
        cluster_pcd = pcd.select_by_index(cluster_indices)

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

        if extent[2] < 0.15:
            rospy.loginfo(f"[Plane {plane_id}] z축 높이 {extent[2]:.2f}m → 너무 낮아서 제거")
            continue

        half_box_size = 0.1
        points_np = np.asarray(inlier_cloud.points)
        mask = (
            (points_np[:,0] > center[0] - half_box_size) & (points_np[:,0] < center[0] + half_box_size) &
            (points_np[:,1] > center[1] - half_box_size) & (points_np[:,1] < center[1] + half_box_size) &
            (points_np[:,2] > center[2] - half_box_size) & (points_np[:,2] < center[2] + half_box_size)
        )
        center_points_count = np.sum(mask)

        if center_points_count > 10:
            rospy.loginfo(f"[Plane {plane_id}] 중심 영역 점 개수 {center_points_count} → 꽉 찬 평면으로 제거")
            continue

        rospy.loginfo(f"[Plane {plane_id}] 중심 영역 점 개수 {center_points_count} → 뚫린 평면으로 유지")
        publish_cube(center, extent, msg.header.frame_id, plane_id)
        plane_id += 1

#####################################################################################################################

def main():
    global marker_pub, cloud_pub
    rospy.init_node('plane_bbox_publisher', anonymous=True)
    rospy.Subscriber('/ouster/points', PointCloud2, callback)
    marker_pub = rospy.Publisher('/bounding_box_marker', Marker, queue_size=10)
    cloud_pub = rospy.Publisher('/filtered_points', PointCloud2, queue_size=10)
    rospy.loginfo("▶ 바운딩 박스 + 텍스트 퍼블리셔 시작")
    rospy.spin()

if __name__ == '__main__':
    main()
