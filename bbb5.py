#!/usr/bin/env python3

# 거울 탐지 및 그림자 영역 투사 알고리즘 (v4 - Perspective Projection)
# 4점투시 바운딩 박스까지 생성 완료

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


# <<< MODIFIED: 이제 여러 개의 그림자 박스를 발행하는 함수 >>>
def publish_shadow_projection(mirror_center, mirror_extent, frame_id="ouster"):
    num_boxes = 5
    max_depth = 2.0

    dist_to_mirror = np.linalg.norm(mirror_center)
    if dist_to_mirror < 1e-6: return

    direction_vector = mirror_center / dist_to_mirror

    # 투영 방향에 따른 깊이 축 결정
    depth_axis = np.argmax(np.abs(direction_vector))

    step_depth = max_depth / num_boxes

    for i in range(num_boxes):
        # 각 박스(세그먼트)의 중심까지의 거리
        center_dist_from_mirror = (i + 0.5) * step_depth

        # 원근감에 따른 크기 계산
        scale_factor = (dist_to_mirror + center_dist_from_mirror) / dist_to_mirror

        # 박스의 크기(가로, 세로, 깊이) 계산
        new_extent = np.array(mirror_extent) * scale_factor
        # <<< MODIFIED: 깊이를 세그먼트의 길이로 설정하여 빈 공간 제거 >>>
        new_extent[depth_axis] = step_depth

        # 박스의 중심 위치 계산
        new_center = mirror_center + direction_vector * center_dist_from_mirror

        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "shadow_projection"
        marker.id = 8000 + i
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position = Point(*new_center)
        marker.pose.orientation = Quaternion(0, 0, 0, 1)
        marker.scale = Point(*new_extent)
        marker.color = ColorRGBA(0.0, 1.0, 0.0, 0.2)
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

    # <<< MODIFIED: 투영된 그림자 공간 시각화 로직 호출 >>>
    mirror_aabb = closest_face['aabb']
    publish_shadow_projection(mirror_aabb.get_center(), mirror_aabb.get_extent(), msg.header.frame_id)


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
