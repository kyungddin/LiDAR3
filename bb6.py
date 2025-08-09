#!/usr/bin/env python3

# 250809 final

import rospy
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import open3d as o3d

######################################################################################################################

marker_pub = None
cloud_pub = None  # 노이즈 제거 후 퍼블리시용

last_bounding_boxes = []  # (center, extent) 리스트 저장

######################################################################################################################

last_front_face_center = None
last_front_face_normal = None
last_switch_time = 0
switch_interval = 3.0  # 초 단위, 텍스트 위치 최소 유지 시간
min_move_dist = 0.1    # m 단위, 위치 변화 임계값

######################################################################################################################

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
    return pc2.create_cloud(header, fields, points)

######################################################################################################################

def publish_cube(center, extent, frame_id="ouster", marker_id=0):
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
    marker.pose.orientation.w = 1
    marker.scale.x = extent[0]
    marker.scale.y = extent[1]
    marker.scale.z = extent[2]
    marker.color = ColorRGBA(0.0, 0.0, 1.0, 0.5)
    marker.lifetime = rospy.Duration(10)
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
    text_marker.lifetime = rospy.Duration(10)
    marker_pub.publish(text_marker)

######################################################################################################################

def publish_front_text(center, normal, frame_id="ouster", marker_id=9999):
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
    front_marker.pose.orientation.w = 1
    front_marker.scale.z = 0.25
    front_marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)
    front_marker.text = "Front"
    front_marker.lifetime = rospy.Duration(10)
    marker_pub.publish(front_marker)

######################################################################################################################

def is_similar_box(center, extent, threshold_dist=0.05, threshold_scale=0.05):
    global last_bounding_boxes
    for (c, e) in last_bounding_boxes:
        dist = np.linalg.norm(center - c)
        scale_diff = np.linalg.norm(extent - e)
        if dist < threshold_dist and scale_diff < threshold_scale:
            return True
    return False

######################################################################################################################

def callback(msg):
    global last_front_face_center, last_front_face_normal, last_switch_time, last_bounding_boxes

    points = []
    for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
        points.append([p[0], p[1], p[2]])

    if len(points) == 0:
        rospy.logwarn("PointCloud 비어 있음")
        return

    np_points = np.array(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_points)

    # 다운샘플링 (voxel size 0.005m)
    voxel_size = 0.005
    pcd = pcd.voxel_down_sample(voxel_size)

    # 노이즈 제거 부분 삭제

    cloud_pub.publish(o3d_to_pointcloud2(pcd, frame_id=msg.header.frame_id))

    labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=30, print_progress=False))
    max_label = labels.max()

    candidate_faces = []

    last_bounding_boxes = []  # 이번 콜백에서 감지된 바운딩 박스 초기화

    for i in range(max_label + 1):
        cluster_indices = np.where(labels == i)[0]
        cluster_pcd = pcd.select_by_index(cluster_indices)

        if len(cluster_pcd.points) < 100:
            continue

        # RANSAC Parameter
        plane_model, inliers = cluster_pcd.segment_plane(
            distance_threshold=0.01,
            ransac_n=3,
            num_iterations=1000
        )
        if len(inliers) < 100:
            continue

        inlier_cloud = cluster_pcd.select_by_index(inliers)
        aabb = inlier_cloud.get_axis_aligned_bounding_box()
        center = aabb.get_center()
        extent = aabb.get_extent()

        if extent[2] < 0.15:
            continue

        # Bounding Box Center Parameter
        half_box_size = 0.11
        points_np = np.asarray(inlier_cloud.points)
        mask = (
            (points_np[:,0] > center[0] - half_box_size) & (points_np[:,0] < center[0] + half_box_size) &
            (points_np[:,1] > center[1] - half_box_size) & (points_np[:,1] < center[1] + half_box_size) &
            (points_np[:,2] > center[2] - half_box_size) & (points_np[:,2] < center[2] + half_box_size)
        )
        center_points_count = np.sum(mask)

        if center_points_count > 10:
            continue

        # 중복 박스인지 확인
        if is_similar_box(center, extent):
            continue

        publish_cube(center, extent, msg.header.frame_id, i)
        last_bounding_boxes.append((center, extent))

        normal = np.array(plane_model[:3])
        normal /= np.linalg.norm(normal)

        area = extent[0] * extent[1]
        candidate_faces.append((i, center, normal, area))

    if len(candidate_faces) == 0:
        rospy.logwarn("유효한 평면 없음")
        return

    largest_area = max([f[3] for f in candidate_faces])
    largest_faces = [f for f in candidate_faces if abs(f[3] - largest_area) < 1e-6 or f[3] == largest_area]

    def dist_to_origin(face):
        return np.linalg.norm(face[1])

    closest_face = min(largest_faces, key=dist_to_origin)

    now = rospy.get_time()

    if last_front_face_center is None:
        last_front_face_center = closest_face[1]
        last_front_face_normal = closest_face[2]
        last_switch_time = now
        publish_front_text(closest_face[1], closest_face[2], frame_id=msg.header.frame_id)
        return

    dist_move = np.linalg.norm(closest_face[1] - last_front_face_center)

    if (now - last_switch_time) > switch_interval and dist_move > min_move_dist:
        last_front_face_center = closest_face[1]
        last_front_face_normal = closest_face[2]
        last_switch_time = now
        publish_front_text(closest_face[1], closest_face[2], frame_id=msg.header.frame_id)
    else:
        publish_front_text(last_front_face_center, last_front_face_normal, frame_id=msg.header.frame_id)

######################################################################################################################

def main():
    global marker_pub, cloud_pub
    rospy.init_node('plane_bbox_publisher', anonymous=True)
    marker_pub = rospy.Publisher('/bounding_box_marker', Marker, queue_size=10)
    cloud_pub = rospy.Publisher('/denoised_cloud', PointCloud2, queue_size=10)
    rospy.Subscriber('/ouster/points', PointCloud2, callback)
    rospy.loginfo("plane_bbox_publisher 노드 시작")
    rospy.spin()

if __name__ == '__main__':
    main()
