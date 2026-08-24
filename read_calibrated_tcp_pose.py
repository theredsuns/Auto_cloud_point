#!/usr/bin/env python3
"""Print one calibrated AGX TCP PoseStamped sample as a JSON line."""
import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=0.35)
    args = parser.parse_args()
    import rclpy
    from geometry_msgs.msg import PoseStamped

    rclpy.init()
    node = rclpy.create_node("read_calibrated_tcp_pose")
    sample = []

    def callback(message):
        if not sample:
            sample.append(message)

    node.create_subscription(PoseStamped, "/feedback/tcp_pose", callback, 10)
    rclpy.spin_once(node, timeout_sec=max(args.timeout, 0.05))
    if sample:
        pose = sample[0]
        payload = {
            "frame_id": pose.header.frame_id,
            "position_m": [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z],
            "quaternion_xyzw": [pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w],
        }
        print("TCP_POSE_JSON=" + json.dumps(payload, separators=(",", ":")))
    else:
        print("TCP_POSE_JSON=null")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
