#!/usr/bin/env python3

# 250809 final + reflection (v23, The True Logic, Full Code)

import rospy
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA, Header
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import open3d as o3d
import math

######################################################################################################################
# Global Variables & Configuration
######################################################################################################################

marker_pub = None
cloud_pub = None
reflected_pub = None

last_front_face_center = None
last_front_face_normal = None
last_switch_time = 0
switch_interval = 3.0
min_move_dist = 0.1

MIN_VIRTUAL_DISTANCE = 0.05
MAX_VIRTUAL_DISTANCE = 2.0


######################################################################################################################
# Utility & Marker Functions
######################################################################################################################

def o3d_to_pointcloud2(o3d_cloud, frame_id="ouster"):
    points = np.asarray(o3d_cloud.points)
    header = Header(stamp=rospy.Time.now(), frame_id=frame_id)
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
              PointField('y', 4, PointField.FLOAT32, 1),
              PointField('z', 8, PointField.FLOAT32, 1)]
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


def reflect_point_cloud_across_plane(pcd_virtual, plane_center, plane_normal):
    points_virtual = np.asarray(pcd_virtual.points)
    dist_to_plane = np.dot(points_virtual - plane_center, plane_normal)
    points_real = points_virtual - 2 * dist_to_plane[:, np.newaxis] * plane_normal
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_real))


def publish_cube(center, extent, frame_id, marker_id):
    marker = Marker(header=Header(stamp=rospy.Time.now(), frame_id=frame_id),
                    ns="bounding_box", id=marker_id, type=Marker.CUBE, action=Marker.ADD,
                    pose=Pose(Point(*center), Quaternion(0, 0, 0, 1)),
                    scale=Point(*extent), color=ColorRGBA(0.0, 0.0, 1.0, 0.5),
                    lifetime=rospy.Duration(0.5))
    marker_pub.publish(marker)


def publish_front_text(center, normal, frame_id):
    front_pos = center - normal * 0.15
    marker = Marker(header=Header(stamp=rospy.Time.now(), frame_id=frame_id),
                    ns="front_text", id=9999, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD,
                    pose=Pose(Point(*front_pos), Quaternion(0, 0, 0, 1)),
                    scale=Vector3(0, 0, 0.25), color=ColorRGBA(1.0, 0.0, 0.0, 1.0),
                    text="Front", lifetime=rospy.Duration(1))
    marker_pub.publish(marker)


######################################################################################################################
# Main Callback Function
######################################################################################################################

def callback(msg):
    global last_front_face_center, last_front_face_normal, last_switch_time

    # 1. 데이터 변환 및 전처리
    points = [p for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)]
    if not points: return
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(points)))
    pcd = pcd.voxel_down_sample(voxel_size=0.005)
    cloud_pub.publish(o3d_to_pointcloud2(pcd, frame_id=msg.header.frame_id))

    # 2. 클러스터링 및 평면 후보 탐색
    labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=30, print_progress=False))
    if labels.max() < 0: return

    candidate_faces = []
    for i in range(labels.max() + 1):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) < 100: continue
        cluster_pcd = pcd.select_by_index(cluster_indices)
        plane_model, inliers = cluster_pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        if len(inliers) < 100: continue

        inlier_cloud = cluster_pcd.select_by_index(inliers)
        aabb = inlier_cloud.get_axis_aligned_bounding_box()
        center, extent = aabb.get_center(), aabb.get_extent()
        if extent[2] < 0.15: continue

        center_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=center - 0.11, max_bound=center + 0.11)
        if len(inlier_cloud.crop(center_box).points) > 10: continue

        publish_cube(center, extent, msg.header.frame_id, i)
        normal = np.array(plane_model[:3]) / np.linalg.norm(plane_model[:3])
        candidate_faces.append({'id': i, 'center': center, 'normal': normal, 'area': extent[0] * extent[1]})

    # 3. 'Front' 평면 결정 및 안정화
    if not candidate_faces: return
    largest_faces = [f for f in candidate_faces if abs(f['area'] - max(c['area'] for c in candidate_faces)) < 1e-6]
    closest_face = min(largest_faces, key=lambda f: np.linalg.norm(f['center']))

    now = rospy.get_time()
    if last_front_face_center is None or ((now - last_switch_time) > switch_interval and np.linalg.norm(
            closest_face['center'] - last_front_face_center) > min_move_dist):
        last_front_face_center, last_front_face_normal = closest_face['center'], closest_face['normal']
        last_switch_time = now

    if last_front_face_center is None: return

    # RANSAC 법선 벡터 방향이 항상 센서를 향하도록 강제
    view_vector = last_front_face_center - np.array([0, 0, 0])
    if np.dot(last_front_face_normal, view_vector) > 0:
        last_front_face_normal = -last_front_face_normal

    publish_front_text(last_front_face_center, last_front_face_normal, msg.header.frame_id)

    # 4. <<< MODIFIED: 새로운 직접 반사 로직 (레이캐스팅 제거) >>>
    all_points = np.asarray(pcd.points)

    # 각 점들이 평면의 '뒤쪽'(가상)에 있는지 확인
    # 법선 벡터는 센서를 향하므로(예: -X 방향), 거울 뒤의 점(예: +X 위치)과의 내적은 음수가 나옴
    dists_from_plane = np.dot(all_points - last_front_face_center, last_front_face_normal)

    # 거울 뒤 특정 "영역"에 있는 점들만 필터링
    virtual_point_indices = np.where(
        (dists_from_plane < -MIN_VIRTUAL_DISTANCE) &
        (dists_from_plane > -MAX_VIRTUAL_DISTANCE)
    )[0]

    if len(virtual_point_indices) > 0:
        pcd_virtual = pcd.select_by_index(virtual_point_indices)

        # 가상 객체를 실제 위치로 반사
        pcd_real = reflect_point_cloud_across_plane(pcd_virtual, last_front_face_center, last_front_face_normal)

        pcd_real.paint_uniform_color([1.0, 0.0, 1.0])  # Magenta
        reflected_pub.publish(o3d_to_pointcloud2(pcd_real, frame_id=msg.header.frame_id))


######################################################################################################################
# Main Execution
######################################################################################################################

def main():
    global marker_pub, cloud_pub, reflected_pub
    rospy.init_node("mirror_plane_detector", anonymous=True)

    marker_pub = rospy.Publisher("/mirror_markers", Marker, queue_size=100)
    cloud_pub = rospy.Publisher("/mirror_filtered_cloud", PointCloud2, queue_size=10)
    reflected_pub = rospy.Publisher("/reflected_cloud", PointCloud2, queue_size=10)

    rospy.Subscriber("/ouster/points", PointCloud2, callback, queue_size=1, buff_size=2 ** 24)

    rospy.loginfo("The True Logic of Mirror Reflection Node is Running.")
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass