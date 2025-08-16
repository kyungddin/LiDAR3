#!/usr/bin/env python3

# 거울 탐지 및 그림자 영역 투사 알고리즘 (v5 - Final)

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
reflected_pub = None  # 반사된 포인트를 발행할 퍼블리셔

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
    if o3d_cloud.has_colors():
        colors = (np.asarray(o3d_cloud.colors) * 255).astype(np.uint8)
        rgb_int = (colors[:, 0].astype(np.uint32) << 16) | \
                  (colors[:, 1].astype(np.uint32) << 8) | \
                  (colors[:, 2].astype(np.uint32))
        packed_points = np.zeros(len(points), dtype=[
            ('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.uint32)
        ])
        packed_points['x'], packed_points['y'], packed_points['z'] = points[:, 0], points[:, 1], points[:, 2]
        packed_points['rgb'] = rgb_int.view(np.uint32)
        fields.append(PointField('rgb', 12, PointField.UINT32, 1))
        return pc2.create_cloud(header, fields, packed_points)
    return pc2.create_cloud(header, fields, points)


def reflect_point_cloud_across_plane(pcd, plane_center, plane_normal):
    points = np.asarray(pcd.points)
    dist_to_plane = np.dot(points - plane_center, plane_normal)
    reflected_points = points - 2 * dist_to_plane[:, np.newaxis] * plane_normal
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(reflected_points))


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

    depth_axis_index = np.argmax(np.abs(direction_vector))

    step_depth = max_depth / num_boxes

    # <<< MODIFIED: 첫번째와 마지막 박스를 제외하고 생성 >>>
    for i in range(1, num_boxes - 1):
        center_dist_from_mirror = (i + 0.5) * step_depth

        scale_factor = (dist_to_mirror + center_dist_from_mirror) / dist_to_mirror

        new_extent = np.array(mirror_extent) * scale_factor
        new_extent[depth_axis_index] = step_depth

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

    # <<< ADDED: 복원할 포인트가 속할 전체 영역(Frustum)을 계산하여 반환 >>>
    start_dist = step_depth
    end_dist = max_depth - step_depth

    start_scale = (dist_to_mirror + start_dist) / dist_to_mirror
    end_scale = (dist_to_mirror + end_dist) / dist_to_mirror

    start_extent = np.array(mirror_extent) * start_scale
    end_extent = np.array(mirror_extent) * end_scale

    start_center = mirror_center + direction_vector * start_dist
    end_center = mirror_center + direction_vector * end_dist

    # Frustum의 8개 꼭짓점 계산
    frustum_corners = []
    for sx in [-0.5, 0.5]:
        for sy in [-0.5, 0.5]:
            for sz in [-0.5, 0.5]:
                start_corner = start_center + np.multiply([sx, sy, sz], start_extent)
                end_corner = end_center + np.multiply([sx, sy, sz], end_extent)
                frustum_corners.append(start_corner)
                frustum_corners.append(end_corner)

    return o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(frustum_corners))


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

    mirror_aabb = closest_face['aabb']
    # <<< MODIFIED: publish_shadow_projection 함수가 이제 복원할 영역(frustum)을 반환 >>>
    reflection_frustum = publish_shadow_projection(mirror_aabb.get_center(), mirror_aabb.get_extent(),
                                                   msg.header.frame_id)

    # <<< ADDED: Frustum 내부의 점들을 찾아 반사시키는 로직 >>>
    if reflection_frustum:
        # 거울 뒤에 있는 모든 점들을 1차 필터링
        all_points = np.asarray(pcd.points)
        normal_vec = last_front_face_normal
        view_vector = last_front_face_center - np.array([0, 0, 0])
        if np.dot(normal_vec, view_vector) > 0:
            normal_vec = -normal_vec

        dists_from_plane = np.dot(all_points - last_front_face_center, normal_vec)
        behind_indices = np.where(dists_from_plane < -0.05)[0]

        if len(behind_indices) > 0:
            pcd_behind = pcd.select_by_index(behind_indices)

            # Frustum 내부에 있는 점들만 최종 선택
            indices_in_frustum = reflection_frustum.get_point_indices_within_bounding_box(pcd_behind.points)

            if len(indices_in_frustum) > 0:
                pcd_virtual = pcd_behind.select_by_index(indices_in_frustum)

                # 반사 변환 적용
                pcd_real = reflect_point_cloud_across_plane(pcd_virtual, last_front_face_center, last_front_face_normal)
                pcd_real.paint_uniform_color([1.0, 0.0, 1.0])  # Magenta

                # reflected_pub가 정의되어 있는지 확인 후 발행
                if reflected_pub:
                    reflected_pub.publish(o3d_to_pointcloud2(pcd_real, frame_id=msg.header.frame_id))


#####################################################################################################################

def main():
    global marker_pub, cloud_pub, reflected_pub
    rospy.init_node('plane_bbox_publisher', anonymous=True)
    rospy.Subscriber('/ouster/points', PointCloud2, callback, queue_size=1)
    marker_pub = rospy.Publisher('/bounding_box_marker', Marker, queue_size=50)
    cloud_pub = rospy.Publisher('/filtered_points', PointCloud2, queue_size=10)
    reflected_pub = rospy.Publisher('/reflected_cloud', PointCloud2, queue_size=10)
    rospy.loginfo("▶ Shadow Projection 거울 탐지 노드 시작")
    rospy.spin()


if __name__ == '__main__':
    main()
