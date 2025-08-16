#!/usr/bin/env python3

# 거울 탐지 알고리즘 (v3.11 - Front/Back Text Fix)

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
    # 큐브 마커 (얇은 평면 시각화)
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
    marker.scale.z = 0.01  # 얇은 두께
    marker.color = ColorRGBA(0.0, 0.0, 1.0, 0.5)
    marker.lifetime = rospy.Duration(1)
    marker_pub.publish(marker)

    # normal_vector가 제대로 전달된 경우에만 "Front", "Back" 텍스트 발행
    if normal_vector is not None:
        # 앞면 텍스트 마커 ("Front")
        front_text = Marker()
        front_text.header.frame_id = frame_id
        front_text.header.stamp = rospy.Time.now()
        front_text.ns = "front_back_text"
        front_text.id = marker_id * 2
        front_text.type = Marker.TEXT_VIEW_FACING
        front_text.action = Marker.ADD
        # 앞면 위치 계산: normal_vector의 반대 방향으로 약간 이동
        front_pos = center - normal_vector * 0.15
        front_text.pose.position = Point(*front_pos)
        front_text.pose.orientation.w = 1.0
        front_text.scale.z = 0.15
        front_text.color = ColorRGBA(0.0, 1.0, 0.0, 0.8)  # 녹색
        front_text.text = "Front"
        front_text.lifetime = rospy.Duration(1)
        marker_pub.publish(front_text)

        # 뒷면 텍스트 마커 ("Back")
        back_text = Marker()
        back_text.header.frame_id = frame_id
        back_text.header.stamp = rospy.Time.now()
        back_text.ns = "front_back_text"
        back_text.id = marker_id * 2 + 1
        back_text.type = Marker.TEXT_VIEW_FACING
        back_text.action = Marker.ADD
        # 뒷면 위치 계산: normal_vector 방향으로 약간 이동
        back_pos = center + normal_vector * 0.15
        back_text.pose.position = Point(*back_pos)
        back_text.pose.orientation.w = 1.0
        back_text.scale.z = 0.15
        back_text.color = ColorRGBA(1.0, 0.0, 0.0, 0.8)  # 빨간색
        back_text.text = "Back"
        back_text.lifetime = rospy.Duration(1)
        marker_pub.publish(back_text)


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
            distance_threshold=0.03, ransac_n=3, num_iterations=1000
        )

        if len(inliers) < 15:
            continue

        mirror_candidate_pcd = cluster_pcd.select_by_index(inliers)

        # 4단계: 바운딩 박스의 두 면과 센서의 거리를 비교하여 앞/뒷면 판단
        center = mirror_candidate_pcd.get_center()

        # 4-1. 바운딩 박스의 방향과 크기 계산
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

        # 4-2. 두 개의 넓은 면의 중심 좌표 계산
        face1_center = center + z_axis * (extent[2] / 2.0)
        face2_center = center - z_axis * (extent[2] / 2.0)

        # 4-3. 각 면의 중심과 센서(원점) 사이의 거리 계산
        dist1_to_origin = np.linalg.norm(face1_center)
        dist2_to_origin = np.linalg.norm(face2_center)

        # 4-4. 거리가 더 짧은 쪽을 '앞면'으로 판단하고, normal_vector 방향을 통일
        if dist1_to_origin < dist2_to_origin:
            front_face_center = face1_center
            # z_axis가 앞면을 향하고 있으므로, 뒷면을 향하도록 뒤집어줌
            z_axis = -z_axis
        else:
            front_face_center = face2_center
            # z_axis는 이미 뒷면을 향하고 있으므로 그대로 둠

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
    marker_pub = rospy.Publisher('/mirror_bounding_box', Marker, queue_size=20)  # 큐 사이즈 증가
    cloud_pub = rospy.Publisher('/filtered_points1', PointCloud2, queue_size=2)
    points2_pub = rospy.Publisher('/republished_points2', PointCloud2, queue_size=2)
    denoised_points2_pub = rospy.Publisher('/points2_denoised', PointCloud2, queue_size=2)

    rospy.loginfo("▶ 거울 탐지 노드 시작 (v3.11 - Front/Back Text Fix)")
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass