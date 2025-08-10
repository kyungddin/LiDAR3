#!/usr/bin/env python3

# 250809 final + reflection (v33, Neighborhood Restoration, Full Code)

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

RAY_FOV_H = 30.0
RAY_FOV_V = 15.0
RAY_GRID_W = 11
RAY_GRID_H = 7
MIN_REFLECTION_DISTANCE = 0.1
MAX_REFLECTION_DISTANCE = 1.5

# <<< TUNING: 히트 지점 주변을 탐색할 반경(radius)
NEIGHBORHOOD_RADIUS = 0.3  # (meters) 10cm


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


def reflect_point_cloud_across_plane(pcd, plane_center, plane_normal):
    points = np.asarray(pcd.points)
    dist_to_plane = np.dot(points - plane_center, plane_normal)
    reflected_points = points - 2 * dist_to_plane[:, np.newaxis] * plane_normal
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(reflected_points))


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


def publish_ray_marker(start, end, hit, frame_id, marker_id):
    marker = Marker(header=Header(stamp=rospy.Time.now(), frame_id=frame_id),
                    ns="raycast", id=marker_id, type=Marker.ARROW, action=Marker.ADD,
                    points=[Point(*start), Point(*end)],
                    scale=Vector3(0.02, 0.04, 0),
                    color=ColorRGBA(0.0, 1.0, 0.0, 0.8) if hit else ColorRGBA(1.0, 1.0, 0.0, 0.8),
                    lifetime=rospy.Duration(0.5))
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
    closest_face_id = closest_face['id']

    now = rospy.get_time()
    if last_front_face_center is None or ((now - last_switch_time) > switch_interval and np.linalg.norm(
            closest_face['center'] - last_front_face_center) > min_move_dist):
        last_front_face_center, last_front_face_normal = closest_face['center'], closest_face['normal']
        last_switch_time = now

    if last_front_face_center is None: return

    view_vector = last_front_face_center - np.array([0, 0, 0])
    if np.dot(last_front_face_normal, view_vector) > 0:
        last_front_face_normal = -last_front_face_normal

    publish_front_text(last_front_face_center, last_front_face_normal, msg.header.frame_id)

    # 4. <<< MODIFIED: "레이캐스트에 맞은 점 주변 영역"을 반사시키는 최종 로직 >>>

    # 4-1. '가상 세계' 지도 생성 (거울 제외 모든 점)
    other_points_indices = np.where(labels != closest_face_id)[0]
    if len(other_points_indices) == 0: return
    pcd_virtual_candidates = pcd.select_by_index(other_points_indices)
    virtual_object_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_virtual_candidates, 0.15)

    # 4-2. 레이캐스팅 발사
    forward_vec, ray_origin = last_front_face_normal, last_front_face_center
    global_up, right_vec, up_vec = np.array([0., 0., 1.]), None, None
    right_vec = np.cross(global_up, forward_vec)
    if np.linalg.norm(right_vec) < 1e-6: right_vec = np.cross(np.array([0., 1., 0.]), forward_vec)
    right_vec /= np.linalg.norm(right_vec)
    up_vec = np.cross(right_vec, -forward_vec)
    ray_directions = []
    for j in range(RAY_GRID_H):
        theta_v = np.deg2rad(-RAY_FOV_V / 2 + j * (RAY_FOV_V / (RAY_GRID_H - 1))) if RAY_GRID_H > 1 else 0
        for i in range(RAY_GRID_W):
            theta_h = np.deg2rad(-RAY_FOV_H / 2 + i * (RAY_FOV_H / (RAY_GRID_W - 1))) if RAY_GRID_W > 1 else 0
            rot_h, rot_v = o3d.geometry.get_rotation_matrix_from_axis_angle(
                up_vec * theta_h), o3d.geometry.get_rotation_matrix_from_axis_angle(right_vec * theta_v)
            ray_directions.append(rot_v @ rot_h @ -forward_vec)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(virtual_object_mesh))
    rays_o3d = o3d.core.Tensor(np.hstack([np.tile(ray_origin, (len(ray_directions), 1)), ray_directions]),
                               dtype=o3d.core.Dtype.Float32)
    ans = scene.cast_rays(rays_o3d)

    # 4-3. 유효한 히트 지점 수집
    hit_points = []
    for i, t_tensor in enumerate(ans['t_hit']):
        distance = t_tensor.item()
        is_hit = np.isfinite(distance)
        if is_hit and MIN_REFLECTION_DISTANCE < distance < MAX_REFLECTION_DISTANCE:
            hit_points.append(ray_origin + ray_directions[i] * distance)

        end_point = ray_origin + ray_directions[i] * distance if is_hit else ray_origin + ray_directions[i] * 5.0
        publish_ray_marker(ray_origin, end_point, is_hit, msg.header.frame_id, 8000 + i)

    # 4-4. 히트 지점 주변 영역 탐색 및 반사
    if hit_points:
        rospy.loginfo_throttle(1.0, f"SUCCESS! Found {len(hit_points)} virtual points. Searching neighborhood...")

        # '가상 세계' 전체에 대한 KD-Tree를 한번만 생성
        virtual_kdtree = o3d.geometry.KDTreeFlann(pcd_virtual_candidates)

        # 모든 히트 지점 주변의 이웃 포인트 인덱스를 수집 (중복 제거를 위해 set 사용)
        indices_to_reflect = set()
        for hp in hit_points:
            # 각 히트 지점마다 반경 내 이웃을 검색
            [k, idx, _] = virtual_kdtree.search_radius_vector_3d(hp, NEIGHBORHOOD_RADIUS)
            if k > 0:
                indices_to_reflect.update(idx)

        if indices_to_reflect:
            # 최종적으로 선택된 '가상 포인트 클라우드' 영역
            pcd_virtual_area = pcd_virtual_candidates.select_by_index(list(indices_to_reflect))

            pcd_real = reflect_point_cloud_across_plane(pcd_virtual_area, last_front_face_center,
                                                        last_front_face_normal)
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

    rospy.loginfo("Neighborhood Restoration Version of Mirror Reflection Node is Running.")
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass