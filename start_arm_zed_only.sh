#!/usr/bin/env bash
# Start only the remote PiPER CAN-control and ZED stack.  It sends no motion.
set -euo pipefail

REMOTE_HOST="skki@192.168.50.55"

exec ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_HOST" \
  'bash ~/zed_code/arm_control/start_remote_agx_zed_stack.sh'
