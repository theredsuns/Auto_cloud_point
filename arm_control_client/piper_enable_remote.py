#!/usr/bin/env python3
"""Explicitly enable PiPER motors without commanding any movement."""
import argparse
import sys
import time

SDK_PATH = "/home/skki/zed_code/arm_control/vendor/piper_sdk_runtime"
if SDK_PATH not in sys.path:
    sys.path.insert(0, SDK_PATH)
from piper_sdk import C_PiperInterface_V2


def main():
    parser = argparse.ArgumentParser(description="Enable PiPER motors; does not move joints")
    parser.add_argument("--can", default="can1")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()
    if not args.execute or args.confirm != "ARM_CLEAR":
        raise SystemExit("Refusing: add --execute --confirm ARM_CLEAR after clearing the workspace.")
    arm = C_PiperInterface_V2(can_name=args.can, judge_flag=False,
                              start_sdk_joint_limit=True, start_sdk_gripper_limit=True)
    arm.ConnectPort(); time.sleep(.7)
    if arm.GetArmJointMsgs().Hz <= 0:
        raise SystemExit("Refusing: no live joint feedback; motors were not enabled.")
    for _ in range(12):
        if arm.EnablePiper():
            print("[enabled] PiPER motors enabled; no joint target was sent.")
            return
        time.sleep(.2)
    raise SystemExit("Could not enable PiPER motors. Check physical emergency-stop and power.")


if __name__ == "__main__":
    main()
