#!/usr/bin/env python3

# 거울 탐지 및 앞면 판단 알고리즘

import rospy
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import open3d as o3d
from threading import Lock

marker_pub = None
cloud_pub = None
points2_pub = None  # Dual return 포인트를 발행하기 위한 퍼블리셔

last_points2 = None
points2_lock = Lock()

# <<< ADDED: Front 평면 안정화를 위한 전역 변수 >>>
last_front_face_center = None
last_front_face_normal = None
last_switch_time = 0
switch_interval = 3.0
min_move_dist = 0.1

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
    # <<< MODIFIED: 모든 박스를 파란색으로 통일 >>>
    marker.color = ColorRGBA(0.0, 0.0, 1.0, 0.5)  # 반투명 파랑
    marker.lifetime = rospy.Duration(1)
    marker_pub.publish(marker)

    # 텍스트 ("Mirror")
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
    # <<< MODIFIED: 모든 텍스트를 "Mirror"로 통일 >>>
    text_marker.color = ColorRGBA(0.0, 0.0, 1.0, 1.0)
    text_marker.text = "Mirror"
    text_marker.lifetime = rospy.Duration(1)
    marker_pub.publish(text_marker)


# <<< ADDED: Front 텍스트를 별도로 발행하는 함수 >>>
def publish_front_text(center, normal, frame_id="ouster"):
    # RANSAC 법선 벡터 방향이 항상 센서를 향하도록 강제
    view_vector = center - np.array([0, 0, 0])
    if np.dot(normal, view_vector) > 0:
        normal = -normal

    # 법선 벡터는 센서를 향하므로, 텍스트를 살짝 앞으로 표시하려면 빼줘야 함
    front_pos = center + normal * 0.15

    text_marker = Marker()
    text_marker.header.frame_id = frame_id
    text_marker.header.stamp = rospy.Time.now()
    text_marker.ns = "front_text"
    text_marker.id = 9999  # 고유 ID
    text_marker.type = Marker.TEXT_VIEW_FACING
    text_marker.action = Marker.ADD
    text_marker.pose.position.x = front_pos[0]
    text_marker.pose.position.y = front_pos[1]
    text_marker.pose.position.z = front_pos[2]
    text_marker.pose.orientation.w = 1
    text_marker.scale.z = 0.25
    text_marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)
    text_marker.text = "Front"
    text_marker.lifetime = rospy.Duration(1)
    marker_pub.publish(text_marker)


#####################################################################################################################

def callback(msg):
    global last_points2, last_front_face_center, last_front_face_normal, last_switch_time

    points = []
    for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
        points.append([p[0], p[1], p[2]])

    if len(points) == 0:
        rospy.logwarn("PointCloud 비어 있음")
        return

    np_points = np.array(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_points)

    cloud_pub.publish(o3d_to_pointcloud2(pcd, frame_id=msg.header.frame_id))

    labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=30, print_progress=False))
    max_label = labels.max()
    rospy.loginfo(f"클러스터 수: {max_label + 1}")

    candidate_faces = []
    with points2_lock:
        if last_points2 is None or len(last_points2) == 0:
            rospy.logwarn_throttle(5, "points2 데이터가 아직 수신되지 않았습니다.")
            return
        pcd2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(last_points2))
        pcd2_tree = o3d.geometry.KDTreeFlann(pcd2)

    for i in range(max_label + 1):
        cluster_indices = np.where(labels == i)[0]
        cluster_pcd = pcd.select_by_index(cluster_indices)

        if len(cluster_pcd.points) < 100:
            continue

        plane_model, inliers = cluster_pcd.segment_plane(
            distance_threshold=0.01, ransac_n=3, num_iterations=1000
        )
        if len(inliers) < 200:
            continue

        inlier_cloud = cluster_pcd.select_by_index(inliers)
        aabb = inlier_cloud.get_axis_aligned_bounding_box()
        center = aabb.get_center()
        extent = aabb.get_extent()

        if extent[2] < 0.15:
            continue

        thickness = np.min(extent)
        if thickness > MAX_PLANE_THICKNESS:
            continue

        search_radius = 0.1
        [k, idx, _] = pcd2_tree.search_radius_vector_3d(center, search_radius)

        if k > 0:
            normal = np.array(plane_model[:3]) / np.linalg.norm(plane_model[:3])
            candidate_faces.append({
                'id': i, 'center': center, 'extent': extent, 'normal': normal,
                'area': extent[0] * extent[1]
            })

    if not candidate_faces:
        rospy.logwarn("유효한 거울 후보 없음")
        return

    # <<< MODIFIED: 가장 유력한 Front 평면 선택 및 시각화 로직 변경 >>>
    largest_area = max(f['area'] for f in candidate_faces)
    largest_faces = [f for f in candidate_faces if abs(f['area'] - largest_area) < 1e-6]
    closest_face = min(largest_faces, key=lambda f: np.linalg.norm(f['center']))

    # 모든 후보는 파란색 박스와 "Mirror" 텍스트로 시각화
    for face in candidate_faces:
        publish_cube(face['center'], face['extent'], msg.header.frame_id, face['id'])

    # 최종 선택된 Front 평면에만 빨간색 "Front" 텍스트를 추가로 표시
    publish_front_text(closest_face['center'], closest_face['normal'], msg.header.frame_id)


def callback2(msg):
    """/ouster/points2 토픽을 받아서 전역 변수에 저장"""
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
    marker_pub = rospy.Publisher('/bounding_box_marker', Marker, queue_size=50)  # 큐 사이즈 증가
    cloud_pub = rospy.Publisher('/filtered_points', PointCloud2, queue_size=10)
    points2_pub = rospy.Publisher('/republished_points2', PointCloud2, queue_size=10)
    rospy.loginfo("▶ Dual Return 거울 탐지 노드 시작")
    rospy.spin()


if __name__ == '__main__':
    main()
