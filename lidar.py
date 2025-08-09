import rospy
import numpy as np
from visualization_msgs.msg import Marker
from std_msgs.msg import Header
from geometry_msgs.msg import Point

marker_pub = None

def publish_cube(position, scale=(0.05, 0.05, 0.05), frame_id="map", marker_id=0, color=(0,1,0,0.8)):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "raycast"
    marker.id = marker_id
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.pose.position.x = position[0]
    marker.pose.position.y = position[1]
    marker.pose.position.z = position[2]
    marker.pose.orientation.w = 1.0
    marker.scale.x = scale[0]
    marker.scale.y = scale[1]
    marker.scale.z = scale[2]
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]
    marker.lifetime = rospy.Duration(0)
    marker_pub.publish(marker)

def raycast_plane(ray_origin, ray_dir, plane_point, plane_normal):
    denom = np.dot(plane_normal, ray_dir)
    if abs(denom) < 1e-6:
        return None
    t = np.dot(plane_point - ray_origin, plane_normal) / denom
    if t < 0:
        return None
    intersection = ray_origin + t * ray_dir
    return intersection

def callback(msg):
    global marker_pub

    # (예시) 평면 후보 리스트: (id, center, normal, area)
    candidate_faces = [
        (1, np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), 1.0),
        (2, np.array([2.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), 1.2),
        # 실제 데이터로 교체
    ]

    # last_front_face_center, last_front_face_normal 는 너가 실제로 사용하는 값으로 바꿔줘
    last_front_face_center = np.array([0.0, 0.0, 0.0])
    last_front_face_normal = np.array([0.0, 0.0, 1.0])
    last_front_face_normal = last_front_face_normal / np.linalg.norm(last_front_face_normal)

    ray_origin = last_front_face_center
    ray_dir = last_front_face_normal

    # 후보 평면 중 레이와 충돌 검사 후 첫 충돌점에 큐브 마커 생성
    for i, center, normal, area in candidate_faces:
        normal = normal / np.linalg.norm(normal)

        # 자기 자신 평면 무시
        if np.allclose(center, ray_origin) and np.allclose(normal, ray_dir):
            continue

        hit_point = raycast_plane(ray_origin, ray_dir, center, normal)
        if hit_point is not None:
            publish_cube(hit_point, frame_id=msg.header.frame_id, marker_id=20000 + i)
            break

if __name__ == '__main__':
    rospy.init_node('raycast_visualization_node')
    marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)

    rospy.Subscriber('your_input_topic', YourMsgType, callback)  # 네 메시지 타입, 토픽명으로 수정해줘

    rospy.spin()
