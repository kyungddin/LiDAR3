#!/usr/bin/env python3

# 250809 final + wide raycasting (Full Version)

import rospy
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA, Header
from geometry_msgs.msg import Point
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import open3d as o3d
import math

######################################################################################################################
# Global Variables & Configuration
######################################################################################################################

marker_pub = None
cloud_pub = None

last_bounding_boxes = []

last_front_face_center = None
last_front_face_normal = None
last_switch_time = 0
switch_interval = 3.0
min_move_dist = 0.1

# 광선 다발(시야) 설정
RAY_FOV_H = 30.0  # (degrees) 수평 시야각
RAY_FOV_V = 10.0  # (degrees) 수직 시야각
RAY_GRID_W = 7  # (count) 수평 방향으로 쏠 광선 개수
RAY_GRID_H = 3  # (count) 수직 방향으로 쏠 광선 개수


######################################################################################################################
# Utility Functions
######################################################################################################################

def o3d_to_pointcloud2(o3d_cloud, frame_id="ouster"):
    """Converts an Open3D PointCloud object to a ROS PointCloud2 message."""
    points = np.asarray(o3d_cloud.points)
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id
    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)
    ]
    return pc2.create_cloud(header, fields, points)


def normal_to_angles(normal):
    """Converts a normal vector to human-readable angles."""
    nx, ny, nz = normal
    angle_with_z = math.degrees(math.acos(abs(nz)))
    azimuth = math.degrees(math.atan2(ny, nx))
    if azimuth < 0:
        azimuth += 360
    return angle_with_z, azimuth


def is_similar_box(center, extent, threshold_dist=0.05, threshold_scale=0.05):
    """Checks if a new bounding box is too similar to one already detected."""
    global last_bounding_boxes
    for (c, e) in last_bounding_boxes:
        dist = np.linalg.norm(center - c)
        scale_diff = np.linalg.norm(extent - e)
        if dist < threshold_dist and scale_diff < threshold_scale:
            return True
    return False


######################################################################################################################
# Marker Publishing Functions
######################################################################################################################

def publish_cube(center, extent, frame_id="ouster", marker_id=0):
    """Publishes a CUBE and a TEXT marker for a detected 'Mirror' plane."""
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
    marker.pose.orientation.w = 1.0
    marker.scale.x = extent[0]
    marker.scale.y = extent[1]
    marker.scale.z = extent[2]
    marker.color = ColorRGBA(0.0, 0.0, 1.0, 0.5)
    marker.lifetime = rospy.Duration(0.5)
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
    text_marker.pose.orientation.w = 1.0
    text_marker.scale.z = 0.2
    text_marker.color = ColorRGBA(0.0, 0.0, 1.0, 1.0)
    text_marker.text = "Mirror"
    text_marker.lifetime = rospy.Duration(0.5)
    marker_pub.publish(text_marker)


def publish_front_text(center, normal, frame_id="ouster", marker_id=9999):
    """Publishes a TEXT_VIEW_FACING marker for the 'Front' plane."""
    front_pos = center + normal * 0.15
    front_marker = Marker()
    front_marker.header.frame_id = frame_id
    front_marker.header.stamp = rospy.Time.now()
    front_marker.ns = "front_text"
    front_marker.id = marker_id
    front_marker.type = Marker.TEXT_VIEW_FACING
    front_marker.action = Marker.ADD
    front_marker.pose.position.x = front_pos[0]
    front_marker.pose.position.y = front_pos[1]
    front_marker.pose.position.z = front_pos[2]
    front_marker.pose.orientation.w = 1.0
    front_marker.scale.z = 0.25
    front_marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)
    front_marker.text = "Front"
    front_marker.lifetime = rospy.Duration(1)
    marker_pub.publish(front_marker)


def publish_ray_marker(start_point, end_point, hit, frame_id="ouster", marker_id=8888):
    """Visualizes the raycasting result with an ARROW marker."""
    ray_marker = Marker()
    ray_marker.header.frame_id = frame_id
    ray_marker.header.stamp = rospy.Time.now()
    ray_marker.ns = "raycast"
    ray_marker.id = marker_id
    ray_marker.type = Marker.ARROW
    ray_marker.action = Marker.ADD
    ray_marker.points.append(Point(start_point[0], start_point[1], start_point[2]))
    ray_marker.points.append(Point(end_point[0], end_point[1], end_point[2]))
    ray_marker.scale.x = 0.02
    ray_marker.scale.y = 0.04
    if hit:
        ray_marker.color = ColorRGBA(0.0, 1.0, 0.0, 0.8)
    else:
        ray_marker.color = ColorRGBA(1.0, 1.0, 0.0, 0.8)
    ray_marker.lifetime = rospy.Duration(0.5)
    marker_pub.publish(ray_marker)


######################################################################################################################
# Main Callback Function
######################################################################################################################

def callback(msg):
    global last_front_face_center, last_front_face_normal, last_switch_time, last_bounding_boxes

    # 1. Convert ROS PointCloud2 to Open3D PointCloud
    points = [p for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)]
    if not points:
        rospy.logwarn("Received an empty point cloud.")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))

    # 2. Pre-processing
    pcd = pcd.voxel_down_sample(voxel_size=0.005)
    cloud_pub.publish(o3d_to_pointcloud2(pcd, frame_id=msg.header.frame_id))

    # 3. Setup Raycasting Scene
    scene = o3d.t.geometry.RaycastingScene()
    legacy_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=0.1)
    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(legacy_mesh)
    scene.add_triangles(tensor_mesh)

    # 4. Clustering and Plane Detection
    labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=30, print_progress=False))
    max_label = labels.max()

    candidate_faces = []
    last_bounding_boxes = []

    for i in range(max_label + 1):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) < 100:
            continue

        cluster_pcd = pcd.select_by_index(cluster_indices)

        plane_model, inliers = cluster_pcd.segment_plane(
            distance_threshold=0.01, ransac_n=3, num_iterations=1000
        )
        if len(inliers) < 100:
            continue

        inlier_cloud = cluster_pcd.select_by_index(inliers)
        aabb = inlier_cloud.get_axis_aligned_bounding_box()
        center = aabb.get_center()
        extent = aabb.get_extent()

        if extent[2] < 0.15:
            continue

        points_in_center = inlier_cloud.crop(
            o3d.geometry.AxisAlignedBoundingBox(min_bound=center - 0.11, max_bound=center + 0.11)
        )
        if len(points_in_center.points) > 10:
            continue

        if is_similar_box(center, extent):
            continue

        publish_cube(center, extent, msg.header.frame_id, i)
        last_bounding_boxes.append((center, extent))

        normal = np.array(plane_model[:3])
        normal /= np.linalg.norm(normal)
        area = extent[0] * extent[1]
        candidate_faces.append({'center': center, 'normal': normal, 'area': area})

    # 5. Select the "Front" face and stabilize its marker
    if not candidate_faces:
        return

    largest_area = max([f['area'] for f in candidate_faces])
    largest_faces = [f for f in candidate_faces if abs(f['area'] - largest_area) < 1e-6]
    closest_face = min(largest_faces, key=lambda face: np.linalg.norm(face['center']))

    current_front_face_center = closest_face['center']
    current_front_face_normal = closest_face['normal']
    now = rospy.get_time()

    if last_front_face_center is None:
        last_front_face_center = current_front_face_center
        last_front_face_normal = current_front_face_normal
        last_switch_time = now
    else:
        dist_moved = np.linalg.norm(current_front_face_center - last_front_face_center)
        if (now - last_switch_time) > switch_interval and dist_moved > min_move_dist:
            last_front_face_center = current_front_face_center
            last_front_face_normal = current_front_face_normal
            last_switch_time = now
            angle_z, azimuth = normal_to_angles(last_front_face_normal)
            rospy.loginfo(f"New Front Face Detected - Angle with Z: {angle_z:.2f}°, Azimuth: {azimuth:.2f}°")

    # 6. Generate Ray Bundle and Perform Raycasting
    if last_front_face_center is not None:
        publish_front_text(last_front_face_center, last_front_face_normal, frame_id=msg.header.frame_id)

        forward_vec = last_front_face_normal
        ray_origin = last_front_face_center

        global_up = np.array([0.0, 0.0, 1.0])
        right_vec = np.cross(forward_vec, global_up)
        if np.linalg.norm(right_vec) < 1e-6:
            right_vec = np.cross(forward_vec, np.array([0.0, 1.0, 0.0]))
        right_vec /= np.linalg.norm(right_vec)
        up_vec = np.cross(right_vec, forward_vec)

        ray_directions = []

        for j in range(RAY_GRID_H):
            theta_v = np.deg2rad(-RAY_FOV_V / 2.0 + j * (RAY_FOV_V / (RAY_GRID_H - 1))) if RAY_GRID_H > 1 else 0
            for i in range(RAY_GRID_W):
                theta_h = np.deg2rad(-RAY_FOV_H / 2.0 + i * (RAY_FOV_H / (RAY_GRID_W - 1))) if RAY_GRID_W > 1 else 0

                rot_h = o3d.geometry.get_rotation_matrix_from_axis_angle(up_vec * theta_h)
                rot_v = o3d.geometry.get_rotation_matrix_from_axis_angle(right_vec * theta_v)

                direction = rot_v @ rot_h @ forward_vec
                ray_directions.append(direction)

        rays_o3d = o3d.core.Tensor(
            np.hstack([np.tile(ray_origin, (len(ray_directions), 1)), np.array(ray_directions)]),
            dtype=o3d.core.Dtype.Float32
        )

        ans = scene.cast_rays(rays_o3d)

        for i, direction in enumerate(ray_directions):
            hit_distance = ans['t_hit'][i].item()
            is_hit = np.isfinite(hit_distance) and hit_distance > 0

            if is_hit:
                ray_end_point = ray_origin + direction * hit_distance
            else:
                ray_end_point = ray_origin + direction * 5.0

            publish_ray_marker(ray_origin, ray_end_point, is_hit, msg.header.frame_id, marker_id=8000 + i)


######################################################################################################################
# Main Execution
######################################################################################################################

def main():
    global marker_pub, cloud_pub
    rospy.init_node("mirror_plane_detector", anonymous=True)
    marker_pub = rospy.Publisher("/mirror_markers", Marker, queue_size=100)
    cloud_pub = rospy.Publisher("/mirror_filtered_cloud", PointCloud2, queue_size=5)
    rospy.Subscriber("/ouster/points", PointCloud2, callback, queue_size=1, buff_size=2 ** 24)
    rospy.loginfo("Mirror plane detector node started with wide raycasting.")
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass