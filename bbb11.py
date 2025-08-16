#!/usr/bin/env python3

# 거울 탐지 알고리즘 (v3.14 - Corrected Frustum Scaling)

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

last_points2 = None
points2_lock = Lock()

MAX_PLANE_THICKNESS = 0.1  # (meters) 평면으로 인식할 최대 두께


def o3d_to_pointcloud2(o3d_cloud, frame_id="ouster"):
    """
    Open3D PointCloud를 ROS PointCloud2 메시지로 변환합니다.
    """
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
    """
    탐지된 객체 위치에 2D 평면처럼 보이는 얇은 CUBE 마커와 앞/뒤면 텍스트를 발행합니다.
    """
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


def publish_search_boxes(center, base_extent, orientation_q, normal_vector, frame_id="ouster"):
    """
    거울 뒷면에 원근감을 가진 여러 개의 탐색용 바운딩 박스를 발행합니다.
    (수정: 상하/좌우 스케일링 축을 바로잡음)
    """
    # --- 조절 가능한 파라미터 ---
    num_boxes = 5  # 생성할 박스 개수
    step_distance = 0.4  # 박스 한 개의 Z축 방향 길이 (및 박스 간 간격)
    scale_factor = 0.4  # 거리에 따라 '좌우 길이'가 커지는 비율
    height_reduction_factor = 0.7  # 기존 거울 높이 대비 탐색 박스의 높이 비율 (0.5 = 50%)
    # --------------------------

    for i in range(num_boxes):
        # 1. 위치 계산
        distance_to_center = step_distance * i + (step_distance / 2.0)
        box_center = center + normal_vector * distance_to_center

        # 2. 크기 계산 (수정된 부분)
        size_multiplier = 1 + (scale_factor * i)
        box_scale = Point()

        # --- 축 교환 ---
        # 로컬 X축이 '상하 길이'에 해당 -> 줄어든 상태로 고정
        box_scale.x = base_extent[0] * height_reduction_factor
        # 로컬 Y축이 '좌우 길이'에 해당 -> 거리에 따라 점차 늘어남
        box_scale.y = base_extent[1] * size_multiplier
        # Z축(두께)은 틈새 없이 고정
        box_scale.z = step_distance

        # 3. 마커 생성
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "search_frustum"
        marker.id = i
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position = Point(*box_center)
        marker.pose.orientation = Quaternion(*orientation_q)
        marker.scale = box_scale
        marker.color = ColorRGBA(0.0, 0.7, 0.3, 0.4)  # 반투명 초록색
        marker.lifetime = rospy.Duration(1)

        marker_pub.publish(marker)


def callback(msg):
    """
    메인 콜백 함수. 1차/2차 반사 데이터를 받아 거울을 탐지합니다.
    """
    global last_points2

    pcd1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
        np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
    ))
    if pcd1.has_points():
        pcd1_down = pcd1.voxel_down_sample(voxel_size=0.05)
        cloud_pub.publish(o3d_to_pointcloud2(pcd1_down, frame_id=msg.header.frame_id))

    with points2_lock:
        if last_points2 is None or len(last_points2) == 0:
            return
        pcd2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(last_points2))

    if not pcd2.has_points():
        return

    # 1단계: 2nd Return 포인트 클라우드 클러스터링
    labels = np.array(pcd2.cluster_dbscan(eps=0.15, min_points=100, print_progress=False))

    denoised_indices = np.where(labels != -1)[0]
    if len(denoised_indices) > 0:
        pcd2_denoised = pcd2.select_by_index(denoised_indices)
        pcd2_denoised.paint_uniform_color([0, 1, 0])
        denoised_points2_pub.publish(o3d_to_pointcloud2(pcd2_denoised, frame_id=msg.header.frame_id))
    else:
        denoised_points2_pub.publish(o3d_to_pointcloud2(o3d.geometry.PointCloud(), frame_id=msg.header.frame_id))
        return

    # 2단계: 각 클러스터가 평면인지 검사
    unique_labels = np.unique(labels)
    mirror_count = 0
    for label in unique_labels:
        if label == -1:
            continue

        cluster_indices = np.where(labels == label)[0]
        cluster_pcd = pcd2.select_by_index(cluster_indices)

        # 3단계: 클러스터에서 RANSAC으로 평면 탐지
        plane_model, inliers = cluster_pcd.segment_plane(
            distance_threshold=0.01, ransac_n=3, num_iterations=1000
        )

        if len(inliers) < 15:
            continue

        mirror_candidate_pcd = cluster_pcd.select_by_index(inliers)

        # 4단계: 바운딩 박스의 두 면과 센서의 거리를 비교하여 앞/뒷면 판단
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

        dist1_to_origin = np.linalg.norm(face1_center)
        dist2_to_origin = np.linalg.norm(face2_center)

        if dist1_to_origin < dist2_to_origin:
            front_face_center = face1_center
            z_axis = -z_axis
        else:
            front_face_center = face2_center

        rospy.loginfo(
            f"거울 앞면 좌표: x={front_face_center[0]:.2f}, y={front_face_center[1]:.2f}, z={front_face_center[2]:.2f}")

        # 5. 필터링 로직
        if np.min(extent) > MAX_PLANE_THICKNESS:
            continue
        sorted_extent = sorted(extent, reverse=True)
        if sorted_extent[0] < 0.3 or sorted_extent[1] < 0.3:
            continue

        # 6. ROS Marker 생성
        quat = Rotation.from_matrix(rotation_matrix).as_quat()
        publish_cube(center, extent, quat, msg.header.frame_id, mirror_count, normal_vector=z_axis)

        publish_search_boxes(center, extent, quat, z_axis, msg.header.frame_id)

        mirror_count += 1


def callback2(msg):
    """
    2차 반사(/ouster/points2) 데이터를 수신하여 전역 변수에 저장합니다.
    """
    global last_points2
    points = []
    for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
        points.append([p[0], p[1], p[2]])
    with points2_lock:
        last_points2 = np.array(points)

    if points2_pub is not None:
        points2_pub.publish(msg)


def main():
    """
    ROS 노드를 초기화하고, Subscriber와 Publisher를 설정합니다.
    """
    global marker_pub, cloud_pub, points2_pub, denoised_points2_pub

    rospy.init_node('mirror_detector_node', anonymous=True)

    # Subscribers
    rospy.Subscriber('/ouster/points', PointCloud2, callback, queue_size=1, buff_size=2 ** 24)
    rospy.Subscriber('/ouster/points2', PointCloud2, callback2, queue_size=1, buff_size=2 ** 24)

    # Publishers
    marker_pub = rospy.Publisher('/mirror_bounding_box', Marker, queue_size=30)
    cloud_pub = rospy.Publisher('/filtered_points1', PointCloud2, queue_size=2)
    points2_pub = rospy.Publisher('/republished_points2', PointCloud2, queue_size=2)
    denoised_points2_pub = rospy.Publisher('/points2_denoised', PointCloud2, queue_size=2)

    rospy.loginfo("▶ 거울 탐지 노드 시작 (v3.14 - Corrected Frustum Scaling)")
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass