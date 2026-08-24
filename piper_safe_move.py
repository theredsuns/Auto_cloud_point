#!/usr/bin/env python3
"""Guarded PiPER-L joint move, including only a safe zero-pose recovery."""
import argparse
import sys
import time

SDK_PATH = "/home/skki/zed_code/arm_control/vendor/piper_sdk_runtime"
if SDK_PATH not in sys.path:
    sys.path.insert(0, SDK_PATH)
from piper_sdk import C_PiperInterface_V2

MAX_SPEED_PERCENT = 10
CONFIRM_TEXT = "ARM_CLEAR"
JOINT_LIMITS_DEG = ((-150.0, 150.0), (0.0, 180.0), (-170.0, 0.0),
                    (-100.0, 100.0), (-70.0, 70.0), (-120.0, 120.0))


def connect(channel):
    arm = C_PiperInterface_V2(can_name=channel, judge_flag=False,
                              start_sdk_joint_limit=True, start_sdk_gripper_limit=True)
    arm.ConnectPort()
    return arm


def current_joints(arm, wait_seconds=2.0):
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        feedback = arm.GetArmJointMsgs()
        if feedback.Hz > 0:
            state = feedback.joint_state
            return [state.joint_1 / 1000.0, state.joint_2 / 1000.0, state.joint_3 / 1000.0,
                    state.joint_4 / 1000.0, state.joint_5 / 1000.0, state.joint_6 / 1000.0]
        time.sleep(0.05)
    return None


def main():
    parser = argparse.ArgumentParser(description="Guarded PiPER-L joint movement")
    parser.add_argument("--can", default="can1")
    parser.add_argument("--joint-deg", type=float, nargs=6, required=True,
                        metavar=("J1", "J2", "J3", "J4", "J5", "J6"))
    parser.add_argument("--speed", type=int, default=5)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()
    if not 1 <= args.speed <= MAX_SPEED_PERCENT:
        sys.exit(f"Refusing: --speed must be 1..{MAX_SPEED_PERCENT}")
    if not args.execute or args.confirm != CONFIRM_TEXT:
        sys.exit("Dry run only. Add --execute --confirm ARM_CLEAR after clearing the workspace.")
    target = list(args.joint_deg)
    for index, (value, limits) in enumerate(zip(target, JOINT_LIMITS_DEG), 1):
        if not limits[0] <= value <= limits[1]:
            sys.exit(f"Refusing: J{index} target {value} outside manufacturer limit {limits}")
    arm = connect(args.can)
    time.sleep(0.5)
    status = arm.GetArmStatus().arm_status
    # 0x4 only records that the *previous* Cartesian target was unreachable.
    # Recover solely by the explicitly requested, manufacturer-safe all-zero
    # joint pose. All other abnormal controller/CAN states remain blocked.
    is_zero_pose = max(abs(value) for value in target) < 1e-6
    recoverable_limit_rejection = "TARGET_POS_EXCEEDS_LIMIT" in str(status.arm_status)
    if status.arm_status != 0 and not (recoverable_limit_rejection and is_zero_pose):
        sys.exit(f"Refusing: arm status is not normal: {status.arm_status}")
    current = current_joints(arm)
    if current is None:
        sys.exit("Refusing: no live joint-angle feedback; no motion command was sent")
    print("[armed] current(deg)=", [round(x, 2) for x in current])
    print("[armed] target(deg)=", target, "speed=", args.speed)
    command = [round(value * 1000.0) for value in target]
    for _ in range(100):
        arm.MotionCtrl_2(0x01, 0x01, args.speed, 0x00)
        arm.JointCtrl(*command)
        time.sleep(0.01)
    print("[sent] guarded joint target for 1.0 s")


if __name__ == "__main__":
    main()
