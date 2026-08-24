#!/usr/bin/env python3
"""Guarded PiPER-L Cartesian command in the arm-base/flange frame."""
import argparse
import sys
import time

SDK_PATH = "/home/skki/zed_code/arm_control/vendor/piper_sdk_runtime"
if SDK_PATH not in sys.path:
    sys.path.insert(0, SDK_PATH)
from piper_sdk import C_PiperInterface_V2

CONFIRM_TEXT = "ARM_CLEAR"
MAX_SPEED_PERCENT = 5
# UI Cartesian nudge is 50 mm.  Keep a 5 mm feedback/odometry tolerance so
# a nominal +50 mm command is not rejected by sub-millimetre sensor jitter.
MAX_POSITION_DELTA_MM = 55.0
MAX_ANGLE_DELTA_DEG = 10.0
POSITION_TOLERANCE_MM = 2.0
ANGLE_TOLERANCE_DEG = 0.5
SETTLE_TIMEOUT_SECONDS = 12.0
TRANSLATION_WAYPOINT_MM = 10.0
ORIENTATION_WAYPOINT_DEG = 1.0


def angle_error_deg(target, actual):
    """Smallest signed Euler-axis difference, wrapped at +/-180 degrees."""
    return abs((target - actual + 180.0) % 360.0 - 180.0)


def send_pose(arm, values, cycles=100):
    encoded = [round(value * 1000.0) for value in values]
    for _ in range(cycles):
        # Firmware S-V1.8-8 requires the installation position for Cartesian
        # IK. 0x01 is horizontal/upright; 0x00 is invalid.
        arm.MotionCtrl_2(0x01, 0x00, MAX_SPEED_PERCENT, 0x00, 0x00, 0x01)
        arm.EndPoseCtrl(*encoded)
        time.sleep(0.01)


def wait_for_pose(arm, values, require_orientation=True):
    deadline = time.monotonic() + SETTLE_TIMEOUT_SECONDS
    final = None
    position_error = float("inf")
    angle_error = float("inf")
    while time.monotonic() < deadline:
        final_raw = arm.GetArmEndPoseMsgs().end_pose
        final = [final_raw.X_axis / 1000.0, final_raw.Y_axis / 1000.0,
                 final_raw.Z_axis / 1000.0, final_raw.RX_axis / 1000.0,
                 final_raw.RY_axis / 1000.0, final_raw.RZ_axis / 1000.0]
        position_error = max(abs(a - b) for a, b in zip(values[:3], final[:3]))
        angle_error = max(angle_error_deg(a, b) for a, b in zip(values[3:], final[3:]))
        if position_error <= POSITION_TOLERANCE_MM and (
            not require_orientation or angle_error <= ANGLE_TOLERANCE_DEG
        ):
            break
        time.sleep(0.1)
    return final, position_error, angle_error


def main():
    parser = argparse.ArgumentParser(description="Guarded PiPER Cartesian endpoint movement")
    parser.add_argument("--can", default="can1")
    parser.add_argument("--pose", type=float, nargs=6, required=True,
                        metavar=("X_MM", "Y_MM", "Z_MM", "RX_DEG", "RY_DEG", "RZ_DEG"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument(
        "--continuous", action="store_true",
        help="Direct continuous target: no software distance/angle limit; hardware IK and joint limits remain active.",
    )
    args = parser.parse_args()
    if not args.execute or args.confirm != CONFIRM_TEXT:
        sys.exit("Dry run only. Add --execute --confirm ARM_CLEAR after clearing the workspace.")

    arm = C_PiperInterface_V2(can_name=args.can, judge_flag=False,
                              start_sdk_joint_limit=True, start_sdk_gripper_limit=True)
    arm.ConnectPort()
    time.sleep(0.4)
    arm_state = arm.GetArmStatus().arm_status.arm_status
    # The firmware reports 0x4 after a rejected Cartesian target.  It is a
    # recoverable planning rejection, not a motor/CAN fault; permit only a
    # newly validated small target to clear it.  Every other arm state stays
    # blocked.
    recoverable_limit_rejection = "TARGET_POS_EXCEEDS_LIMIT" in str(arm_state)
    if arm_state != 0 and not recoverable_limit_rejection:
        sys.exit(f"Refusing: arm status is not normal ({arm_state})")
    if arm.GetArmJointMsgs().Hz <= 0 or arm.GetArmEndPoseMsgs().Hz <= 0:
        sys.exit("Refusing: no live joint/end-pose feedback")

    current_raw = arm.GetArmEndPoseMsgs().end_pose
    current = [current_raw.X_axis / 1000.0, current_raw.Y_axis / 1000.0,
               current_raw.Z_axis / 1000.0, current_raw.RX_axis / 1000.0,
               current_raw.RY_axis / 1000.0, current_raw.RZ_axis / 1000.0]
    position_delta = max(abs(target - source) for target, source in zip(args.pose[:3], current[:3]))
    angle_delta = max(abs(target - source) for target, source in zip(args.pose[3:], current[3:]))
    if not args.continuous and (position_delta > MAX_POSITION_DELTA_MM or angle_delta > MAX_ANGLE_DELTA_DEG):
        sys.exit(f"Refusing: target too far from current pose (max {position_delta:.1f} mm, {angle_delta:.1f} deg; limits {MAX_POSITION_DELTA_MM} mm, {MAX_ANGLE_DELTA_DEG} deg)")

    print("[armed] current base/flange=", [round(value, 2) for value in current])
    print("[armed] target base/flange=", [round(value, 2) for value in args.pose])
    if args.continuous:
        # Continuous UI deliberately has no software distance or orientation
        # cap.  Send the target directly at 5% speed and let firmware IK plus
        # the mechanical/soft limits stop an unreachable target.  No later
        # waypoint is sent after an error, so the final feedback reports the
        # pose at which the arm actually stopped.
        send_pose(arm, args.pose)
        final, position_error, angle_error = wait_for_pose(arm, args.pose)
        print("[feedback] final base/flange=", [round(value, 3) for value in final])
        if position_error > POSITION_TOLERANCE_MM or angle_error > ANGLE_TOLERANCE_DEG:
            current_state = arm.GetArmStatus().arm_status.arm_status
            sys.exit(
                "Continuous target stopped or was rejected by firmware/limits: "
                f"position error {position_error:.3f} mm, angle error {angle_error:.3f} deg; "
                f"arm status {current_state}"
            )
        print("[sent] continuous Cartesian target reached")
        return

    # First settle the translation while holding the measured current
    # orientation. Then solve and execute the requested orientation at that
    # fixed XYZ. This avoids Cartesian IK coupling translation and Euler axes.
    translation_target = list(args.pose[:3]) + list(current[3:])
    translation_span = max(abs(target - source) for target, source in zip(translation_target[:3], current[:3]))
    waypoint_count = max(1, int((translation_span + TRANSLATION_WAYPOINT_MM - 1) // TRANSLATION_WAYPOINT_MM))
    for waypoint_index in range(1, waypoint_count + 1):
        progress = waypoint_index / waypoint_count
        waypoint = [
            source + (target - source) * progress
            for source, target in zip(current[:3], translation_target[:3])
        ] + list(current[3:])
        print(f"[translation] waypoint {waypoint_index}/{waypoint_count}:", [round(value, 2) for value in waypoint[:3]])
        send_pose(arm, waypoint, cycles=30)
        _, translation_error, _ = wait_for_pose(arm, waypoint, require_orientation=False)
        if translation_error > POSITION_TOLERANCE_MM:
            sys.exit(
                f"Translation waypoint {waypoint_index}/{waypoint_count} not reached: "
                f"max position error {translation_error:.3f} mm"
            )
    send_pose(arm, args.pose)
    final, position_error, angle_error = wait_for_pose(arm, args.pose)
    print("[feedback] final base/flange=", [round(value, 3) for value in final])
    if position_error > POSITION_TOLERANCE_MM or angle_error > ANGLE_TOLERANCE_DEG:
        sys.exit(
            "Target not reached: max position error "
            f"{position_error:.3f} mm, max angle error {angle_error:.3f} deg"
        )
    print("[sent] guarded Cartesian target reached")


if __name__ == "__main__":
    main()
