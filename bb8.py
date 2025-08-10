#!/usr/bin/env python3

# 250809 final + raycasting (API usage corrected)

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
# Global Variables
######################################################################################################################

marker_pub = None
cloud_pub = None

# Stores (center, extent) of bounding boxes detected in the current frame to avoid duplicates
last_bounding_boxes = []

# Variables for stabilizing the "Front" text marker
last_front_face_center = None
last_front_face_normal = None
last_switch_time = 0
switch_interval = 3.0  # (seconds) Minimum time to keep the "Front" text at one position
min_move_dist = 0.1  # (meters) Minimum distance change required to update the "Front" text position


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
    """Converts a normal vector to human-readable angles (angle with Z-axis and azimuth)."""
    nx, ny, nz = normal
    angle_with_z = math.degrees(math.acos(abs(nz)))
    azimuth = math.degrees(math.atan2(ny, nx))
    if azimuth < 0:
        azimuth += 360
    return angle_with_z, azimuth


def is_similar_box(center, extent, threshold_dist=0.05, threshold_scale=0.05):
    """Checks if a new bounding box is too similar to one already detected in the same frame."""
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
    # Cube Marker
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
    marker.color = ColorRGBA(0.0, 0.0, 1.0, 0.5)  # Blue, semi-transparent
    marker.lifetime = rospy.Duration(1)
    marker_pub.publish(marker)

    # Text Marker ("Mirror")
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
    text_marker.color = ColorRGBA(0.0, 0.0, 1.0, 1.0)  # Blue
    text_marker.text = "Mirror"
    text_marker.lifetime = rospy.Duration(1)
    marker_pub.publish(text_marker)


def publish_front_text(center, normal, frame_id="ouster", marker_id=9999):
    """Publishes a TEXT_VIEW_FACING marker for the 'Front' plane."""
    front_pos = center + normal * 0.15  # Offset the text slightly in front of the plane

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
    front_marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)  # Red
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

    # Set the start and end points of the arrow
    ray_marker.points.append(Point(start_point[0], start_point[1], start_point[2]))
    ray_marker.points.append(Point(end_point[0], end_point[1], end_point[2]))

    # Arrow dimensions
    ray_marker.scale.x = 0.03  # Shaft diameter
    ray_marker.scale.y = 0.06  # Head diameter
    ray_marker.scale.z = 0.0  # Head length (0 for default)

    # Color depends on whether the ray hit an object
    if hit:
        ray_marker.color = ColorRGBA(0.0, 1.0, 0.0, 0.8)  # Green for hit
    else:
        ray_marker.color = ColorRGBA(1.0, 1.0, 0.0, 0.8)  # Yellow for no hit

    ray_marker.lifetime = rospy.Duration(1)
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
    last_bounding_boxes = []  # Reset for current frame

    for i in range(max_label + 1):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) < 100:
            continue

        cluster_pcd = pcd.select_by_index(cluster_indices)

        # RANSAC for plane segmentation
        plane_model, inliers = cluster_pcd.segment_plane(
            distance_threshold=0.01, ransac_n=3, num_iterations=1000
        )
        if len(inliers) < 100:
            continue

        inlier_cloud = cluster_pcd.select_by_index(inliers)
        aabb = inlier_cloud.get_axis_aligned_bounding_box()
        center = aabb.get_center()
        extent = aabb.get_extent()

        # Filtering candidate planes
        if extent[2] < 0.15:  # Filter out floors/small horizontal surfaces
            continue

        # Filter out solid objects by checking point density at the center
        # This helps find surfaces like walls/mirrors
        points_in_center = inlier_cloud.crop(
            o3d.geometry.AxisAlignedBoundingBox(min_bound=center - 0.11, max_bound=center + 0.11)
        )
        if len(points_in_center.points) > 10:
            continue

        if is_similar_box(center, extent):  # Avoid duplicate markers
            continue

        # This candidate is good, publish its marker and save it
        publish_cube(center, extent, msg.header.frame_id, i)
        last_bounding_boxes.append((center, extent))

        normal = np.array(plane_model[:3])
        normal /= np.linalg.norm(normal)
        area = extent[0] * extent[1]
        candidate_faces.append({'center': center, 'normal': normal, 'area': area})

    # 5. Select the "Front" face and stabilize its marker
    if not candidate_faces:
        return  # No valid planes found

    # Find the plane with the largest area, then closest to origin
    largest_area = max([f['area'] for f in candidate_faces])
    largest_faces = [f for f in candidate_faces if abs(f['area'] - largest_area) < 1e-6]
    closest_face = min(largest_faces, key=lambda face: np.linalg.norm(face['center']))

    current_front_face_center = closest_face['center']
    current_front_face_normal = closest_face['normal']
    now = rospy.get_time()

    # Update logic for the stable 'Front' marker
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
            # Log when the front face is updated
            angle_z, azimuth = normal_to_angles(last_front_face_normal)
            rospy.loginfo(f"New Front Face Detected - Angle with Z: {angle_z:.2f}°, Azimuth: {azimuth:.2f}°")

    # 6. Perform Raycasting and Publish All Markers
    if last_front_face_center is not None:
        # Publish the stable "Front" text
        publish_front_text(last_front_face_center, last_front_face_normal, frame_id=msg.header.frame_id)

        # Define ray for Open3D tensor API
        ray_origin = last_front_face_center.astype(np.float32)
        ray_direction = last_front_face_normal.astype(np.float32)
        rays = o3d.core.Tensor([np.hstack([ray_origin, ray_direction])], dtype=o3d.core.Dtype.Float32)

        # Cast the ray
        ans = scene.cast_rays(rays)

        hit_distance = ans['t_hit'][0].item()
        is_hit = np.isfinite(hit_distance) and hit_distance > 0

        # Determine ray's end point for visualization
        if is_hit:
            ray_end_point = ray_origin + ray_direction * hit_distance
        else:
            ray_end_point = ray_origin + ray_direction * 5.0  # Draw a 5m ray if no hit

        # Publish the ray marker
        publish_ray_marker(ray_origin, ray_end_point, is_hit, frame_id=msg.header.frame_id)


######################################################################################################################
# Main Execution
######################################################################################################################

def main():
    global marker_pub, cloud_pub
    rospy.init_node("mirror_plane_detector", anonymous=True)

    # Publishers
    marker_pub = rospy.Publisher("/mirror_markers", Marker, queue_size=50)
    cloud_pub = rospy.Publisher("/mirror_filtered_cloud", PointCloud2, queue_size=5)

    # Subscriber
    rospy.Subscriber("/ouster/points", PointCloud2, callback, queue_size=1, buff_size=2 ** 24)

    rospy.loginfo("Mirror plane detector node started.")
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass