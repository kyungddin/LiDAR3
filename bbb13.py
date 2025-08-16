#!/usr/bin/env python3

# 거울 탐지 알고리즘 (v3.20 - Correct Small Box Filter)
# 여기까진 꽤 잘 해

import rospy
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point, Pose, Quaternion
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import open3d as o3d
from threading import Lock
from scipy.spatial.transform import Rotation

# 전역 변수 초기화
marker_pub = None
cloud_pub = None
points2_pub = None
denoised_points2_pub = None
restored_points_pub = None

last_points2 = None
pcd1_global = None
points_lock = Lock()

MAX_PLANE_THICKNESS = 0.1


def o3d_to_pointcloud2(o3d_cloud, frame_id="ouster"):
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


def publish_cube(center, extent, orientation_q, frame_id="ouster", marker_id=0, normal_vector=None):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "bounding_box"
    marker.id = marker_id
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.pose.position = Point(*center)
    marker.pose.orientation = Quaternion(*orientation_q)
    marker.scale.x = extent[0]
    marker.scale.y = extent[1]
    marker.scale.z = 0.01
    marker.color = ColorRGBA(0.0, 0.0, 1.0, 0.5)
    marker.lifetime = rospy.Duration(1)
    marker_pub.publish(marker)
    if normal_vector is not None:
        front_text = Marker()
        front_text.header.frame_id = frame_id
        front_text.header.stamp = rospy.Time.now()
        front_text.ns = "front_back_text"
        front_text.id = marker_id * 2
        front_text.type = Marker.TEXT_VIEW_FACING
        front_text.action = Marker.ADD
        front_pos = center - normal_vector * 0.15
        front_text.pose.position = Point(*front_pos)
        front_text.pose.orientation.w = 1.0
        front_text.scale.z = 0.15
        front_text.color = ColorRGBA(0.0, 1.0, 0.0, 0.8)
        front_text.text = "Front"
        front_text.lifetime = rospy.Duration(1)
        marker_pub.publish(front_text)
        back_text = Marker()
        back_text.header.frame_id = frame_id
        back_text.header.stamp = rospy.Time.now()
        back_text.ns = "front_back_text"
        back_text.id = marker_id * 2 + 1
        back_text.type = Marker.TEXT_VIEW_FACING
        back_text.action = Marker.ADD
        back_pos = center + normal_vector * 0.15
        back_text.pose.position = Point(*back_pos)
        back_text.pose.orientation.w = 1.0
        back_text.scale.z = 0.15
        back_text.color = ColorRGBA(1.0, 0.0, 0.0, 0.8)
        back_text.text = "Back"
        back_text.lifetime = rospy.Duration(1)
        marker_pub.publish(back_text)


def publish_search_boxes(search_obbs, frame_id="ouster"):
    for i, obb in enumerate(search_obbs):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "search_frustum"
        marker.id = i
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position = Point(*obb.center)
        r = Rotation.from_matrix(obb.R.copy())
        q = r.as_quat()
        marker.pose.orientation = Quaternion(*q)
        marker.scale = Point(*obb.extent)
        marker.color = ColorRGBA(0.0, 0.7, 0.3, 0.4)
        marker.lifetime = rospy.Duration(1)
        marker_pub.publish(marker)


def callback_points1(msg):
    global pcd1_global
    points = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
    if points.shape[0] == 0:
        return
    with points_lock:
        pcd1_global = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd1_down = pcd1_global.voxel_down_sample(voxel_size=0.05)
    cloud_pub.publish(o3d_to_pointcloud2(pcd1_down, frame_id=msg.header.frame_id))
    process_mirror_detection(msg.header.frame_id)


def callback_points2(msg):
    global last_points2
    points = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
    with points_lock:
        last_points2 = points


def process_mirror_detection(frame_id):
    with points_lock:
        if last_points2 is None or pcd1_global is None or len(last_points2) == 0:
            return
        pcd2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(last_points2))

    if not pcd2.has_points():
        return

    labels = np.array(pcd2.cluster_dbscan(eps=0.15, min_points=100, print_progress=False))
    denoised_indices = np.where(labels != -1)[0]

    if len(denoised_indices) > 0:
        pcd2_denoised = pcd2.select_by_index(denoised_indices)
        pcd2_denoised.paint_uniform_color([0, 1, 0])
        denoised_points2_pub.publish(o3d_to_pointcloud2(pcd2_denoised, frame_id=frame_id))
    else:
        denoised_points2_pub.publish(o3d_to_pointcloud2(o3d.geometry.PointCloud(), frame_id=frame_id))
        restored_points_pub.publish(o3d_to_pointcloud2(o3d.geometry.PointCloud(), frame_id=frame_id))
        return

    unique_labels = np.unique(labels)
    mirror_count = 0
    all_restored_points = o3d.geometry.PointCloud()

    for label in unique_labels:
        if label == -1: continue

        cluster_indices = np.where(labels == label)[0]
        cluster_pcd = pcd2.select_by_index(cluster_indices)

        plane_model, inliers = cluster_pcd.segment_plane(distance_threshold=0.005, ransac_n=3, num_iterations=1000)
        if len(inliers) < 15: continue

        mirror_candidate_pcd = cluster_pcd.select_by_index(inliers)
        center = mirror_candidate_pcd.get_center()

        normal_vec = plane_model[:3] / np.linalg.norm(plane_model[:3])
        z_axis = normal_vec
        if np.abs(np.dot(z_axis, [0, 0, 1])) > 0.95:
            ref_vec = np.array([1, 0, 0])
        else:
            ref_vec = np.array([0, 0, 1])
        y_axis = np.cross(z_axis, ref_vec);
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis);
        x_axis /= np.linalg.norm(x_axis)
        rotation_matrix = np.stack([x_axis, y_axis, z_axis], axis=1)

        points_np = np.asarray(mirror_candidate_pcd.points)
        transformed_points = np.dot(points_np - center, rotation_matrix)
        extent = np.max(transformed_points, axis=0) - np.min(transformed_points, axis=0)

        face1_center = center + z_axis * (extent[2] / 2.0)
        face2_center = center - z_axis * (extent[2] / 2.0)
        if np.linalg.norm(face1_center) < np.linalg.norm(face2_center):
            z_axis = -z_axis

        # --- 수정된 코드: 작은 '면적'의 거울을 제거 ---
        # 거울의 너비/높이가 최소 30cm는 되어야 의미있는 거울로 판단
        min_mirror_side_length = 0.3

        # extent를 크기순으로 정렬 -> [두께, 짧은 변, 긴 변]
        sorted_extent = np.sort(extent)

        # 두 번째로 작은 변(너비/높이 중 짧은 쪽)이 기준보다 작으면 무시
        if sorted_extent[1] < min_mirror_side_length:
            rospy.logwarn(f"Too small mirror detected, ignoring. Size: {[round(e, 2) for e in extent]}")
            continue
        # ----------------------------------------------------

        quat = Rotation.from_matrix(rotation_matrix).as_quat()
        publish_cube(center, extent, quat, frame_id, mirror_count, normal_vector=z_axis)

        search_obbs = []
        num_boxes, step_distance, scale_factor, height_reduction_factor = 5, 0.7, 0.5, 1.5
        for i in range(num_boxes):
            distance_to_center = step_distance * i + (step_distance / 2.0)
            box_center = center + z_axis * distance_to_center
            size_multiplier = 1 + (scale_factor * i)
            box_extent = np.array([extent[0] * height_reduction_factor, extent[1] * size_multiplier, step_distance])
            search_obb_rot = np.stack([x_axis, y_axis, z_axis], axis=1)
            search_obbs.append(o3d.geometry.OrientedBoundingBox(box_center, search_obb_rot, box_extent))

        publish_search_boxes(search_obbs, frame_id)

        reflected_indices = []
        for obb in search_obbs:
            indices = obb.get_point_indices_within_bounding_box(pcd1_global.points)
            reflected_indices.extend(indices)

        if not reflected_indices: continue

        unique_reflected_indices = np.unique(reflected_indices)
        points_to_restore_pcd = pcd1_global.select_by_index(unique_reflected_indices)
        points_to_restore = np.asarray(points_to_restore_pcd.points)

        a, b, c, d = plane_model
        t = (a * points_to_restore[:, 0] + b * points_to_restore[:, 1] + c * points_to_restore[:, 2] + d) / (
                    a ** 2 + b ** 2 + c ** 2)
        restored_coords = points_to_restore - 2 * t[:, np.newaxis] * np.array([a, b, c])

        restored_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(restored_coords))
        all_restored_points += restored_pcd

        mirror_count += 1

    if all_restored_points.has_points():
        all_restored_points.paint_uniform_color([1, 0, 1])
        restored_points_pub.publish(o3d_to_pointcloud2(all_restored_points, frame_id))


def main():
    global marker_pub, cloud_pub, points2_pub, denoised_points2_pub, restored_points_pub

    rospy.init_node('mirror_detector_node', anonymous=True)

    rospy.Subscriber('/ouster/points', PointCloud2, callback_points1, queue_size=1, buff_size=2 ** 24)
    rospy.Subscriber('/ouster/points2', PointCloud2, callback_points2, queue_size=1, buff_size=2 ** 24)

    marker_pub = rospy.Publisher('/mirror_bounding_box', Marker, queue_size=30)
    cloud_pub = rospy.Publisher('/filtered_points1', PointCloud2, queue_size=2)
    points2_pub = rospy.Publisher('/republished_points2', PointCloud2, queue_size=2)
    denoised_points2_pub = rospy.Publisher('/points2_denoised', PointCloud2, queue_size=2)
    restored_points_pub = rospy.Publisher('/restored_points', PointCloud2, queue_size=2)

    rospy.loginfo("▶ 거울 탐지 및 복원 노드 시작 (v3.20 - Correct Small Box Filter)")
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass