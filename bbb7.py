#!/usr/bin/env python3

# 거울 탐지 알고리즘 (v2.0 - 2nd Return 우선 탐색)
# 거울 탐지 알고리즘만 개선한 코드

import rospy
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import open3d as o3d
from threading import Lock

marker_pub = None
cloud_pub = None
points2_pub = None

last_points2 = None
points2_lock = Lock()

MAX_PLANE_THICKNESS = 0.1  # (meters) 10cm


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
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "bounding_box"
    marker.id = marker_id
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.pose.position = Point(*center)
    marker.pose.orientation = Quaternion(0, 0, 0, 1)
    marker.scale = Point(*extent)
    marker.color = ColorRGBA(0.0, 0.0, 1.0, 0.5)
    marker.lifetime = rospy.Duration(1)
    marker_pub.publish(marker)

    text_marker = Marker()
    text_marker.header.frame_id = frame_id
    text_marker.header.stamp = rospy.Time.now()
    text_marker.ns = "bounding_box_text"
    text_marker.id = marker_id + 1000
    text_marker.type = Marker.TEXT_VIEW_FACING
    text_marker.action = Marker.ADD
    text_marker.pose.position.x = center[0]
    text_marker.pose.position.y = center[1]
    text_marker.pose.position.z = center[2] + extent[2] / 2 + 0.1
    text_marker.pose.orientation.w = 1
    text_marker.scale.z = 0.2
    text_marker.color = ColorRGBA(0.0, 0.0, 1.0, 1.0)
    text_marker.text = "Mirror"
    text_marker.lifetime = rospy.Duration(1)
    marker_pub.publish(text_marker)


#####################################################################################################################

def callback(msg):
    global last_points2

    points1 = []
    for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
        points1.append([p[0], p[1], p[2]])

    if not points1:
        rospy.logwarn("Points1 (1st return) PointCloud 비어 있음")
        return

    pcd1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(points1)))
    pcd1 = pcd1.voxel_down_sample(voxel_size=0.005)
    cloud_pub.publish(o3d_to_pointcloud2(pcd1, frame_id=msg.header.frame_id))

    with points2_lock:
        if last_points2 is None or len(last_points2) == 0:
            rospy.logwarn_throttle(5, "Points2 (2nd return) 데이터가 아직 수신되지 않았습니다.")
            return
        pcd2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(last_points2))

    # <<< MODIFIED: 2nd Return 우선 탐색 로직 >>>

    # 1단계: 2nd Return 포인트를 클러스터링하여 '반투명 영역' 후보를 찾음
    labels2 = np.array(pcd2.cluster_dbscan(eps=0.02, min_points=10, print_progress=False))
    if labels2.max() < 0:
        rospy.loginfo("2nd Return 클러스터 없음")
        return

    pcd1_tree = o3d.geometry.KDTreeFlann(pcd1)
    mirror_count = 0

    # 2단계: 각 '반투명 영역' 후보에 대해 RANSAC 수행
    for i in range(labels2.max() + 1):
        cluster2_indices = np.where(labels2 == i)[0]
        if len(cluster2_indices) < 5: continue

        cluster2_pcd = pcd2.select_by_index(cluster2_indices)
        center2 = cluster2_pcd.get_center()

        # 3단계: '반투명 영역' 중심 주변의 1st Return 포인트들을 수집
        [k, idx1, _] = pcd1_tree.search_radius_vector_3d(center2, radius=0.3)
        if k < 100: continue

        nearby_pcd1 = pcd1.select_by_index(idx1)

        # 4단계: 수집된 1st Return 포인트들로 평면 탐색
        plane_model, inliers = nearby_pcd1.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=500)
        if len(inliers) < 100: continue

        # 5단계: 최종 필터링 및 시각화
        mirror_candidate_pcd = nearby_pcd1.select_by_index(inliers)
        aabb = mirror_candidate_pcd.get_axis_aligned_bounding_box()
        center, extent = aabb.get_center(), aabb.get_extent()

        if extent[2] < 0.15: continue
        if np.min(extent) > MAX_PLANE_THICKNESS: continue

        rospy.loginfo(f"거울 탐지됨! (2nd Return Cluster {i} 기반)")
        publish_cube(center, extent, msg.header.frame_id, mirror_count)
        mirror_count += 1


def callback2(msg):
    global last_points2
    points = []
    for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
        points.append([p[0], p[1], p[2]])
    with points2_lock:
        last_points2 = np.array(points)
        rospy.loginfo_once("첫 번째 points2 데이터 수신 완료.")

    if points2_pub is not None:
        points2_pub.publish(msg)


#####################################################################################################################

def main():
    global marker_pub, cloud_pub, points2_pub
    rospy.init_node('plane_bbox_publisher', anonymous=True)
    rospy.Subscriber('/ouster/points', PointCloud2, callback, queue_size=1)
    rospy.Subscriber('/ouster/points2', PointCloud2, callback2, queue_size=1)
    marker_pub = rospy.Publisher('/bounding_box_marker', Marker, queue_size=50)
    cloud_pub = rospy.Publisher('/filtered_points', PointCloud2, queue_size=10)
    points2_pub = rospy.Publisher('/republished_points2', PointCloud2, queue_size=10)
    rospy.loginfo("▶ 2nd Return 우선 탐색 노드 시작")
    rospy.spin()


if __name__ == '__main__':
    main()
