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


def publish_bounding_box(center, extent, frame_id="ouster"):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "bounding_box"
    marker.id = 0
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.scale.x = 0.03  # 선 두께
    marker.color = ColorRGBA(1.0, 0.0, 0.0, 0.8)  # 불투명 빨간색
    marker.lifetime = rospy.Duration(5)  # 영구

    # 바운딩박스 8개 꼭짓점 계산
    dx, dy, dz = extent[0] / 2, extent[1] / 2, extent[2] / 2
    cx, cy, cz = center

    corners = [
        [cx - dx, cy - dy, cz - dz],
        [cx + dx, cy - dy, cz - dz],
        [cx + dx, cy + dy, cz - dz],
        [cx - dx, cy + dy, cz - dz],
        [cx - dx, cy - dy, cz + dz],
        [cx + dx, cy - dy, cz + dz],
        [cx + dx, cy + dy, cz + dz],
        [cx - dx, cy + dy, cz + dz],
    ]

    # 12개의 선을 이루는 점쌍
    lines = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # 아래 면
        (4, 5), (5, 6), (6, 7), (7, 4),  # 위 면
        (0, 4), (1, 5), (2, 6), (3, 7)   # 옆면 연결
    ]

    for start, end in lines:
        p1 = Point(*corners[start])
        p2 = Point(*corners[end])
        marker.points.append(p1)
        marker.points.append(p2)

    marker_pub.publish(marker)


def callback(msg):
    # PointCloud2 → numpy 변환
    points = []
    for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
        points.append([p[0], p[1], p[2]])

    if len(points) == 0:
        rospy.logwarn("PointCloud 비어 있음")
        return

    np_points = np.array(points)

    # numpy → Open3D 포맷
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_points)

    # 평면 분할 (RANSAC)
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.02,
        ransac_n=3,
        num_iterations=1000
    )

    if len(inliers) < 100:
        rospy.logwarn(f"충분한 평면을 찾지 못함 (inliers 수: {len(inliers)})")
        return

    inlier_cloud = pcd.select_by_index(inliers)

    # Axis-aligned 바운딩 박스
    aabb = inlier_cloud.get_axis_aligned_bounding_box()
    center = aabb.get_center()
    extent = aabb.get_extent()

    # 크기 조절: 50% 축소
    scale_factor = 0.5
    extent = extent * scale_factor

    rospy.loginfo(f"[BoundingBox] Center: {center}, Extent: {extent}, Inliers: {len(inliers)}")

    publish_bounding_box(center, extent, msg.header.frame_id)


def main():
    global marker_pub
    rospy.init_node('plane_bbox_publisher', anonymous=True)
    rospy.Subscriber('/ouster/points', PointCloud2, callback)
    marker_pub = rospy.Publisher('/bounding_box_marker', Marker, queue_size=1)
    rospy.loginfo("▶ 바운딩 박스 퍼블리셔 노드 시작됨 (/ouster/points → /bounding_box_marker)")
    rospy.spin()


if __name__ == '__main__':
    main()
