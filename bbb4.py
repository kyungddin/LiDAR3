#!/usr/bin/env python3

# 거울 탐지 및 그림자 영역 투사 알고리즘 (v2)

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


def publish_front_text(center, normal, frame_id="ouster"):
    view_vector = center - np.array([0, 0, 0])
    if np.dot(normal, view_vector) > 0:
        normal = -normal
    front_pos = center - normal * 0.15

    text_marker = Marker()
    text_marker.header.frame_id = frame_id
    text_marker.header.stamp = rospy.Time.now()
    text_marker.ns = "front_text"
    text_marker.id = 9999
    text_marker.type = Marker.TEXT_VIEW_FACING
    text_marker.action = Marker.ADD
    text_marker.pose.position = Point(*front_pos)
    text_marker.pose.orientation = Quaternion(0, 0, 0, 1)
    text_marker.scale = Vector3(0, 0, 0.25)
    text_marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)
    text_marker.text = "Front"
    text_marker.lifetime = rospy.Duration(1)
    marker_pub.publish(text_marker)


def publish_shadow_box(center, extent, frame_id="ouster"):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "shadow_box"
    marker.id = 8888  # 고유 ID
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.pose.position = Point(*center)
    marker.pose.orientation = Quaternion(0, 0, 0, 1)
    marker.scale = Point(*extent)
    marker.color = ColorRGBA(0.0, 1.0, 0.0, 0.4)  # 반투명 초록
    marker.lifetime = rospy.Duration(1)
    marker_pub.publish(marker)


#####################################################################################################################

def callback(msg):
    global last_front_face_center, last_front_face_normal, last_switch_time

    points = []
    for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
        points.append([p[0], p[1], p[2]])

    if len(points) == 0:
        rospy.logwarn("PointCloud 비어 있음")
        return

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(points)))
    pcd = pcd.voxel_down_sample(voxel_size=0.005)
    cloud_pub.publish(o3d_to_pointcloud2(pcd, frame_id=msg.header.frame_id))

    labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=30, print_progress=False))
    if labels.max() < 0: return

    candidate_faces = []
    for i in range(labels.max() + 1):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) < 100: continue
        cluster_pcd = pcd.select_by_index(cluster_indices)
        plane_model, inliers = cluster_pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        if len(inliers) < 200: continue

        inlier_cloud = cluster_pcd.select_by_index(inliers)
        aabb = inlier_cloud.get_axis_aligned_bounding_box()
        center, extent = aabb.get_center(), aabb.get_extent()

        if extent[2] < 0.15: continue
        if np.min(extent) > MAX_PLANE_THICKNESS: continue

        publish_cube(center, extent, msg.header.frame_id, i)
        normal = np.array(plane_model[:3]) / np.linalg.norm(plane_model[:3])
        candidate_faces.append(
            {'id': i, 'center': center, 'normal': normal, 'area': extent[0] * extent[1], 'aabb': aabb})

    if not candidate_faces: return

    largest_area = max(f['area'] for f in candidate_faces)
    largest_faces = [f for f in candidate_faces if abs(f['area'] - largest_area) < 1e-6]
    closest_face = min(largest_faces, key=lambda f: np.linalg.norm(f['center']))

    now = rospy.get_time()
    if last_front_face_center is None or ((now - last_switch_time) > switch_interval and np.linalg.norm(
            closest_face['center'] - last_front_face_center) > min_move_dist):
        last_front_face_center, last_front_face_normal = closest_face['center'], closest_face['normal']
        last_switch_time = now

    if last_front_face_center is None: return

    publish_front_text(closest_face['center'], closest_face['normal'], msg.header.frame_id)

    # <<< MODIFIED: 투영 기하학을 이용한 그림자 영역 탐지 로직 >>>

    # 1. 법선 벡터 방향을 센서 쪽으로 고정
    normal_vec = last_front_face_normal
    view_vector = last_front_face_center - np.array([0, 0, 0])
    if np.dot(normal_vec, view_vector) > 0:
        normal_vec = -normal_vec

    # 2. 거울 뒤에 있는 모든 점들을 1차 필터링
    all_points = np.asarray(pcd.points)
    dists_from_plane = np.dot(all_points - last_front_face_center, normal_vec)
    behind_indices = np.where(dists_from_plane < -0.05)[0]

    if len(behind_indices) == 0: return

    points_behind = all_points[behind_indices]

    # 3. 각 점을 거울 평면에 투영하여 그림자 영역 내부인지 검증 (Vectorized)
    p0 = last_front_face_center
    n = normal_vec
    plane_dist = np.dot(p0, n)

    # 방향 벡터 D는 점 P 자신이 됨 (센서 원점이 0,0,0 이므로)
    dot_dn = np.dot(points_behind, n)

    # 0으로 나누는 것을 방지
    valid_mask = np.abs(dot_dn) > 1e-6
    t = np.full_like(dot_dn, np.inf)
    t[valid_mask] = plane_dist / dot_dn[valid_mask]

    # 교차점 I = t * P
    intersection_points = t[:, np.newaxis] * points_behind

    # 교차점이 거울의 AABB 내부에 있는지 확인
    mirror_aabb = closest_face['aabb']
    min_b, max_b = mirror_aabb.get_min_bound(), mirror_aabb.get_max_bound()
    inside_mask = np.all((intersection_points >= min_b) & (intersection_points <= max_b), axis=1)

    final_mask = valid_mask & inside_mask

    # 최종 그림자 영역에 속하는 점들의 원본 인덱스
    shadow_indices_global = behind_indices[final_mask]

    if len(shadow_indices_global) > 0:
        rospy.loginfo(f"그림자 영역에서 {len(shadow_indices_global)}개의 점 발견")
        pcd_shadow = pcd.select_by_index(shadow_indices_global)
        shadow_aabb = pcd_shadow.get_axis_aligned_bounding_box()
        publish_shadow_box(shadow_aabb.get_center(), shadow_aabb.get_extent(), msg.header.frame_id)


#####################################################################################################################

def main():
    global marker_pub, cloud_pub
    rospy.init_node('plane_bbox_publisher', anonymous=True)
    rospy.Subscriber('/ouster/points', PointCloud2, callback, queue_size=1)
    marker_pub = rospy.Publisher('/bounding_box_marker', Marker, queue_size=50)
    cloud_pub = rospy.Publisher('/filtered_points', PointCloud2, queue_size=10)
    rospy.loginfo("▶ Shadow Projection 거울 탐지 노드 시작")
    rospy.spin()


if __name__ == '__main__':
    main()
