#!/usr/bin/env python3

# 거울 탐지 알고리즘 (v3.53 - Ghost Point Removal)

import rospy
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point, Pose, Quaternion
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import open3d as o3d
from threading import Lock
from scipy.spatial.transform import Rotation, Slerp

# 전역 변수 초기화
marker_pub = None
cloud_pub = None
points2_pub = None
denoised_points2_pub = None
restored_points_pub = None
final_cloud_pub = None  # <-- 추가
last_points2 = None
pcd1_global = None
points_lock = Lock()
last_mirror_state = None
frames_since_detection = 0

# --- 스무딩 파라미터 ---
DETECTION_TTL = 20
SMOOTHING_FACTOR = 0.15
# --- 게이트(Gate) 파라미터 ---
MAX_TRANSLATIONAL_VELOCITY = 0.5
MAX_ROTATIONAL_VELOCITY_DEG = 15.0

MANUAL_YAW_CORRECTION_DEGREES = 0.0
MAX_PLANE_THICKNESS = 0.1

# --- ▼▼▼ 추가/수정된 파라미터 ▼▼▼ ---
Z_CORRECTION_FACTOR = 1.2
CULLING_DISTANCE_FROM_MIRROR = 0.2


def o3d_to_pointcloud2(o3d_cloud, frame_id="ouster"):
    # ... (Unchanged)
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
    # ... (Unchanged)
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
    marker.lifetime = rospy.Duration(0.5)
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
        front_text.lifetime = rospy.Duration(0.5)
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
        back_text.lifetime = rospy.Duration(0.5)
        marker_pub.publish(back_text)


def clear_all_markers():
    # ... (Unchanged)
    marker = Marker()
    marker.action = Marker.DELETEALL
    marker_pub.publish(marker)


def publish_search_boxes(search_obbs, frame_id="ouster"):
    # ... (Unchanged)
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
        marker.lifetime = rospy.Duration(0.5)
        marker_pub.publish(marker)


def callback_points1(msg):
    global pcd1_global
    points = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
    if points.shape[0] == 0: return
    with points_lock:
        pcd1_global = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd1_down = pcd1_global.voxel_down_sample(voxel_size=0.05)
    cloud_pub.publish(o3d_to_pointcloud2(pcd1_down, frame_id=msg.header.frame_id))
    # --- ▼▼▼ 수정된 부분 ▼▼▼ ---
    if final_cloud_pub:
        final_cloud_pub.publish(o3d_to_pointcloud2(pcd1_down, frame_id=msg.header.frame_id))
    process_mirror_detection(msg.header.frame_id)


def callback_points2(msg):
    # ... (Unchanged)
    global last_points2
    points = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
    with points_lock:
        last_points2 = points


def process_mirror_detection(frame_id):
    global last_mirror_state, frames_since_detection

    with points_lock:
        if last_points2 is None or pcd1_global is None or len(last_points2) == 0: return
        pcd2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(last_points2))

    if not pcd1_global.has_points() or not pcd2.has_points(): return

    mirror_found_in_this_frame = False

    labels = np.array(pcd2.cluster_dbscan(eps=0.15, min_points=70, print_progress=False))
    unique_labels = np.unique(labels[labels != -1])

    if len(unique_labels) > 0:
        denoised_indices = np.where(labels != -1)[0]
        pcd2_denoised = pcd2.select_by_index(denoised_indices)
        pcd2_denoised.paint_uniform_color([0, 1, 0])
        denoised_points2_pub.publish(o3d_to_pointcloud2(pcd2_denoised, frame_id=frame_id))

        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            cluster_pcd = pcd2.select_by_index(cluster_indices)

            ### PCA 기반 평면성(Planarity) 검사 로직 ###
            points = np.asarray(cluster_pcd.points)
            if len(points) < 50:
                continue

            mean = np.mean(points, axis=0)
            cov_matrix = np.cov(points - mean, rowvar=False)
            eigenvalues, _ = np.linalg.eig(cov_matrix)

            sorted_eigenvalues = np.sort(eigenvalues)

            sum_eigenvalues = np.sum(sorted_eigenvalues)
            if sum_eigenvalues < 1e-6:
                continue

            planarity = sorted_eigenvalues[0] / sum_eigenvalues

            if planarity > 0.02:
                continue

            plane_model, inliers = cluster_pcd.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)

            if len(inliers) < 15: continue

            mirror_candidate_pcd = cluster_pcd.select_by_index(inliers)
            center = mirror_candidate_pcd.get_center()
            normal_vec = plane_model[:3] / np.linalg.norm(plane_model[:3])
            temp_z_axis = normal_vec
            if np.abs(np.dot(temp_z_axis, [0, 0, 1])) > 0.95:
                ref_vec = np.array([1, 0, 0])
            else:
                ref_vec = np.array([0, 0, 1])
            temp_y_axis = np.cross(temp_z_axis, ref_vec);
            temp_y_axis /= np.linalg.norm(temp_y_axis)
            temp_x_axis = np.cross(temp_y_axis, temp_z_axis);
            temp_x_axis /= np.linalg.norm(temp_x_axis)
            temp_rotation_matrix = np.stack([temp_x_axis, temp_y_axis, temp_z_axis], axis=1)
            points_np = np.asarray(mirror_candidate_pcd.points)
            transformed_points = np.dot(points_np - center, temp_rotation_matrix)
            extent = np.max(transformed_points, axis=0) - np.min(transformed_points, axis=0)
            z_axis = normal_vec
            face1_center = center + z_axis * (extent[2] / 2.0)
            face2_center = center - z_axis * (extent[2] / 2.0)
            if np.linalg.norm(face1_center) < np.linalg.norm(face2_center): z_axis = -z_axis
            if np.abs(np.dot(z_axis, [0, 0, 1])) > 0.95:
                ref_vec = np.array([1, 0, 0])
            else:
                ref_vec = np.array([0, 0, 1])
            y_axis = np.cross(z_axis, ref_vec);
            y_axis /= np.linalg.norm(y_axis)
            x_axis = np.cross(y_axis, z_axis);
            x_axis /= np.linalg.norm(x_axis)
            rotation_matrix = np.stack([x_axis, y_axis, z_axis], axis=1)

            min_mirror_side_length = 0.3
            if np.sort(extent)[1] < min_mirror_side_length: continue

            mirror_found_in_this_frame = True
            frames_since_detection = 0

            if last_mirror_state is None:
                last_mirror_state = {"center": center, "extent": extent,
                                     "rotation": Rotation.from_matrix(rotation_matrix), "plane_model": plane_model}
            else:
                dist_diff = np.linalg.norm(center - last_mirror_state["center"])

                last_rot = last_mirror_state["rotation"]
                current_rot = Rotation.from_matrix(rotation_matrix)
                angular_diff = (last_rot.inv() * current_rot).as_rotvec()
                angle_diff_deg = np.linalg.norm(angular_diff) * 180 / np.pi

                if dist_diff > MAX_TRANSLATIONAL_VELOCITY or angle_diff_deg > MAX_ROTATIONAL_VELOCITY_DEG:
                    rospy.logwarn(
                        f"Outlier detected. Ignoring update. Dist: {dist_diff:.2f}m, Angle: {angle_diff_deg:.2f}deg")
                    mirror_found_in_this_frame = False
                    break

                alpha = SMOOTHING_FACTOR
                last_mirror_state["center"] = alpha * center + (1 - alpha) * last_mirror_state["center"]
                last_mirror_state["extent"] = alpha * extent + (1 - alpha) * last_mirror_state["extent"]
                last_mirror_state["plane_model"] = alpha * plane_model + (1 - alpha) * last_mirror_state["plane_model"]

                key_rotations = Rotation.from_matrix([last_mirror_state["rotation"].as_matrix(), rotation_matrix])
                slerp = Slerp([0, 1], key_rotations)
                last_mirror_state["rotation"] = slerp([alpha])[0]

            break

    if not mirror_found_in_this_frame:
        frames_since_detection += 1

    if frames_since_detection < DETECTION_TTL and last_mirror_state is not None:
        state = last_mirror_state
        center, extent, rotation = state["center"], state["extent"], state["rotation"]

        if MANUAL_YAW_CORRECTION_DEGREES != 0.0:
            correction_rot = Rotation.from_euler('z', MANUAL_YAW_CORRECTION_DEGREES, degrees=True)
            corrected_rotation = rotation * correction_rot
        else:
            corrected_rotation = rotation

        mirror_rotation_matrix = corrected_rotation.as_matrix()
        mirror_z_axis = mirror_rotation_matrix[:, 2]
        mirror_y_axis = mirror_rotation_matrix[:, 1]

        publish_cube(center, extent, corrected_rotation.as_quat(), frame_id, 0, normal_vector=mirror_z_axis)

        vec_for_yaw = center / (np.linalg.norm(center) + np.finfo(float).eps)
        vec_for_pitch = mirror_z_axis

        yaw_horiz_direction = np.array([vec_for_yaw[0], vec_for_yaw[1], 0.0])
        yaw_horiz_direction /= (np.linalg.norm(yaw_horiz_direction) + np.finfo(float).eps)

        pitch_horiz_magnitude = np.sqrt(vec_for_pitch[0] ** 2 + vec_for_pitch[1] ** 2)
        pitch_vert_magnitude = vec_for_pitch[2]

        final_z_axis = yaw_horiz_direction * pitch_horiz_magnitude + np.array([0, 0, pitch_vert_magnitude])
        final_z_axis /= (np.linalg.norm(final_z_axis) + np.finfo(float).eps)

        final_x_axis = np.cross(mirror_y_axis, final_z_axis)
        final_x_axis /= (np.linalg.norm(final_x_axis) + np.finfo(float).eps)

        final_y_axis = np.cross(final_z_axis, final_x_axis)
        final_y_axis /= (np.linalg.norm(final_y_axis) + np.finfo(float).eps)

        search_box_rotation_matrix = np.stack([final_x_axis, final_y_axis, final_z_axis], axis=1)

        search_obbs = []
        num_boxes = 1
        step_distance = 1.5
        height_reduction_factor = 1.5
        base_width_scale = 1.5
        width_scale_increment = 0.6

        for i in range(num_boxes):
            box_center = center + final_z_axis * (step_distance * i + (step_distance / 2.0))
            size_multiplier = base_width_scale + (width_scale_increment * i)
            box_extent = np.array([extent[0] * height_reduction_factor, extent[1] * size_multiplier, step_distance])

            search_obbs.append(o3d.geometry.OrientedBoundingBox(box_center, search_box_rotation_matrix, box_extent))

        publish_search_boxes(search_obbs, frame_id)

        reflected_indices = []
        for obb in search_obbs:
            indices = obb.get_point_indices_within_bounding_box(pcd1_global.points)
            reflected_indices.extend(indices)

        if reflected_indices:
            initial_indices = np.unique(reflected_indices)
            pcd_tree = o3d.geometry.KDTreeFlann(pcd1_global)
            neighbor_indices = []
            search_radius = 0.15
            for index in initial_indices:
                [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd1_global.points[index], search_radius)
                neighbor_indices.extend(idx)
            final_indices_to_restore = np.unique(np.concatenate((initial_indices, neighbor_indices)))
            points_to_restore = np.asarray(pcd1_global.select_by_index(final_indices_to_restore).points)

            ### 추가된 컬링(Culling) 로직 ###
            if CULLING_DISTANCE_FROM_MIRROR > 0.0 and len(points_to_restore) > 0:
                plane_center = state["center"]
                plane_normal = mirror_z_axis
                distances = np.dot(points_to_restore - plane_center, plane_normal)
                points_to_restore = points_to_restore[distances > CULLING_DISTANCE_FROM_MIRROR]

            if len(points_to_restore) == 0:
                restored_pcd = o3d.geometry.PointCloud()
                restored_points_pub.publish(o3d_to_pointcloud2(restored_pcd, frame_id))
            else:
                ### 하우스홀더 변환 기반 반사 로직으로 복구 ###
                estimated_glass_thickness = 0.005
                reflection_plane_center = center + mirror_z_axis * estimated_glass_thickness

                n = mirror_z_axis
                Q = reflection_plane_center

                n_col = n.reshape(3, 1)
                reflection_matrix = np.identity(3) - 2 * (n_col @ n_col.T)
                points_translated = points_to_restore - Q
                points_reflected_translated = points_translated @ reflection_matrix.T
                restored_coords = points_reflected_translated + Q

                ### 각도 보정 로직 ###
                if Z_CORRECTION_FACTOR != 0.0:
                    mirror_z_slope = mirror_z_axis[2]
                    z_correction = -mirror_z_slope * Z_CORRECTION_FACTOR
                    restored_coords[:, 2] += z_correction

                restored_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(restored_coords))
                restored_pcd.paint_uniform_color([1, 0, 1])
                restored_points_pub.publish(o3d_to_pointcloud2(restored_pcd, frame_id))

                ### ▼▼▼ 추가된 최종 클라우드 생성 로직 ▼▼▼ ###
                pcd1_cleaned = pcd1_global.select_by_index(final_indices_to_restore, invert=True)
                final_pcd = pcd1_cleaned + restored_pcd
                final_pcd_down = final_pcd.voxel_down_sample(voxel_size=0.05)
                if final_cloud_pub:
                    final_cloud_pub.publish(o3d_to_pointcloud2(final_pcd_down, frame_id))
        else:
            restored_pcd = o3d.geometry.PointCloud()
            restored_points_pub.publish(o3d_to_pointcloud2(restored_pcd, frame_id))
    else:
        if frames_since_detection >= DETECTION_TTL:
            last_mirror_state = None
            clear_all_markers()
            denoised_points2_pub.publish(o3d_to_pointcloud2(o3d.geometry.PointCloud(), frame_id))


def main():
    global marker_pub, cloud_pub, points2_pub, denoised_points2_pub, restored_points_pub, final_cloud_pub
    rospy.init_node('mirror_detector_node', anonymous=True)
    rospy.Subscriber('/ouster/points', PointCloud2, callback_points1, queue_size=1, buff_size=2 ** 24)
    rospy.Subscriber('/ouster/points2', PointCloud2, callback_points2, queue_size=1, buff_size=2 ** 24)
    marker_pub = rospy.Publisher('/mirror_bounding_box', Marker, queue_size=30)
    cloud_pub = rospy.Publisher('/filtered_points1', PointCloud2, queue_size=2)
    points2_pub = rospy.Publisher('/republished_points2', PointCloud2, queue_size=2)
    denoised_points2_pub = rospy.Publisher('/points2_denoised', PointCloud2, queue_size=2)
    restored_points_pub = rospy.Publisher('/restored_points', PointCloud2, queue_size=2)
    # --- ▼▼▼ 수정된 부분 ▼▼▼ ---
    final_cloud_pub = rospy.Publisher('/final_cloud', PointCloud2, queue_size=2)
    rospy.loginfo("▶ 거울 탐지 및 복원 노드 시작 (v3.53 - Ghost Point Removal)")
    rospy.spin()


if __name__ == '__main__':
    # ... (Unchanged)
    try:
        main()
    except rospy.ROSInterruptException:
        pass