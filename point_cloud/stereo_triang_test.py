#!/usr/bin/env python3
"""
Stereo triangulation + absolute orientation for AprilTag relative pose.
Compares triangulation-based Z vs PnP-based Z.
"""
import os, re, time
import numpy as np
import cv2
import pyzed.sl as sl

TAG_SZ = {0:0.06, 1:0.06, 2:0.06}
D01_TRUE = np.array([0.1587, 0.0, 0.0])
CALIB_FILE = "/home/nkk/coordate_change/ros2_ws/src/apriltag_zed_visp/config/zed_calibration.yaml"

# Load calibrated intrinsics
with open(CALIB_FILE) as f:
    text = f.read()
blocks = text.split('camera_matrix:')
ms = list(re.finditer(r'data:\s*\[([^\]]+)\]', blocks[1]))
K_calib = np.array([float(x) for x in ms[0].group(1).split(',')]).reshape(3,3)

# ZED camera
zed = sl.Camera()
init = sl.InitParameters()
init.camera_resolution = sl.RESOLUTION.HD720
init.depth_mode = sl.DEPTH_MODE.NEURAL
init.camera_fps = 30
if zed.open(init) != sl.ERROR_CODE.SUCCESS:
    print("ZED open failed"); exit()

info = zed.get_camera_information()
calib = info.camera_configuration.calibration_parameters

# Left camera
Kl = np.array([[calib.left_cam.fx, 0, calib.left_cam.cx],
                [0, calib.left_cam.fy, calib.left_cam.cy], [0,0,1]])
Dl = np.array(calib.left_cam.disto[:5])
# Right camera
Kr = np.array([[calib.right_cam.fx, 0, calib.right_cam.cx],
                [0, calib.right_cam.fy, calib.right_cam.cy], [0,0,1]])
Dr = np.array(calib.right_cam.disto[:5])
# Stereo extrinsics (rotation vector → matrix, translation → numpy)
st = calib.stereo_transform
rv_stereo = np.array(st.get_rotation_vector(), dtype=np.float64)
R_stereo, _ = cv2.Rodrigues(rv_stereo)
T_stereo = np.array(st.get_translation().get(), dtype=np.float64) / 1000.0  # mm → m
baseline = np.linalg.norm(T_stereo)

print(f"ZED Stereo: baseline={baseline*1000:.0f}mm")
print(f"Left  fx={Kl[0,0]:.1f} fy={Kl[1,1]:.1f}")
print(f"Right fx={Kr[0,0]:.1f} fy={Kr[1,1]:.1f}")
print(f"Calib fx={K_calib[0,0]:.1f} fy={K_calib[1,1]:.1f}")

# Projection matrices for triangulation (using calibrated K)
# Left camera at origin, right camera at [R|T]
Pl = K_calib @ np.hstack([np.eye(3), np.zeros((3,1))])
Pr = K_calib @ np.hstack([R_stereo, T_stereo.reshape(3,1)])

# Undistortion maps
map_lx, map_ly = cv2.initUndistortRectifyMap(Kl, Dl, None, K_calib, (1280,720), cv2.CV_16SC2)
map_rx, map_ry = cv2.initUndistortRectifyMap(Kr, Dr, None, K_calib, (1280,720), cv2.CV_16SC2)

D_zero = np.zeros(5)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
aruco_p = cv2.aruco.DetectorParameters_create()
aruco_p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
aruco_p.cornerRefinementMaxIterations = 50

zed_l = sl.Mat(); zed_r = sl.Mat()
cv2.namedWindow("Stereo vs PnP", cv2.WINDOW_NORMAL)

def detect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=aruco_p)
    r = {}
    if ids is not None:
        for i, c in zip(ids.flatten(), corners):
            r[int(i)] = c[0]
    return r

def triangulate(c_l, c_r):
    """Triangulate 4 corners from stereo pair → 4x3 array"""
    pts = []
    for i in range(4):
        p4d = cv2.triangulatePoints(Pl, Pr, c_l[i], c_r[i])
        pts.append(p4d[:3].flatten() / p4d[3])
    return np.array(pts)

def fit_frame(pts3d):
    """Fit tag frame from 4 corner 3D points (ArUco order: TL,TR,BR,BL)"""
    c = np.mean(pts3d, axis=0)
    x = pts3d[1] - pts3d[0]; x /= np.linalg.norm(x)
    y = pts3d[3] - pts3d[0]; y /= np.linalg.norm(y)
    z = np.cross(x, y); z /= np.linalg.norm(z)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    if np.linalg.det(R) < 0: R[:,2] *= -1
    return c, R

def pnp(corners, sz):
    h = sz/2
    obj = np.array([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]], dtype=np.float32)
    ok, rv, tv = cv2.solvePnP(obj, corners, K_calib, D_zero, flags=cv2.SOLVEPNP_IPPE)
    if ok:
        R,_ = cv2.Rodrigues(rv); return R, tv.flatten()
    return None, None

def rel(R0, t0, R2, t2):
    R_rel = R0.T @ R2; t_rel = R0.T @ (t2 - t0)
    r_rel,_ = cv2.Rodrigues(R_rel)
    return r_rel.flatten(), t_rel

ema_s, ema_p = None, None
ema_a = 0.08
fc = 0
last_display = 0

while True:
    if zed.grab() != sl.ERROR_CODE.SUCCESS:
        continue
    zed.retrieve_image(zed_l, sl.VIEW.LEFT)
    zed.retrieve_image(zed_r, sl.VIEW.RIGHT)

    frame_l = cv2.cvtColor(zed_l.get_data(), cv2.COLOR_BGRA2BGR)
    frame_r = cv2.cvtColor(zed_r.get_data(), cv2.COLOR_BGRA2BGR)
    frame_l = cv2.remap(frame_l, map_lx, map_ly, cv2.INTER_LINEAR)
    frame_r = cv2.remap(frame_r, map_rx, map_ry, cv2.INTER_LINEAR)

    dl = detect(frame_l); dr = detect(frame_r)
    fc += 1

    t_stereo, t_pnp = None, None

    if 0 in dl and 2 in dl and 0 in dr and 2 in dr:
        # Stereo triangulation
        p0 = triangulate(dl[0], dr[0])
        p2 = triangulate(dl[2], dr[2])
        c0, R0s = fit_frame(p0)
        c2, R2s = fit_frame(p2)
        _, t_stereo = rel(R0s, c0, R2s, c2)

        if ema_s is None: ema_s = t_stereo
        else: ema_s = ema_a*t_stereo + (1-ema_a)*ema_s

        # PnP for comparison
        R0p, t0p = pnp(dl[0], TAG_SZ[0])
        R2p, t2p = pnp(dl[2], TAG_SZ[2])
        if R0p is not None and R2p is not None:
            _, t_pnp = rel(R0p, t0p, R2p, t2p)
            if ema_p is None: ema_p = t_pnp
            else: ema_p = ema_a*t_pnp + (1-ema_a)*ema_p

    now = time.time()
    if now - last_display > 0.05:  # 20Hz display
        last_display = now
        os.system('clear')
        print("=" * 58)
        print("  Stereo Triangulation  vs  PnP  (Z对比)")
        print("=" * 58)
        if ema_s is not None:
            sd = np.linalg.norm(ema_s)*1000
            print(f"  立体三角化:  X={ema_s[0]*1000:7.1f}  Y={ema_s[1]*1000:7.1f}  Z={ema_s[2]*1000:7.1f}  dist={sd:.1f}mm")
            if ema_p is not None:
                pd = np.linalg.norm(ema_p)*1000
                print(f"  PnP (对比):  X={ema_p[0]*1000:7.1f}  Y={ema_p[1]*1000:7.1f}  Z={ema_p[2]*1000:7.1f}  dist={pd:.1f}mm")
                dz = abs(ema_s[2]-ema_p[2])*1000
                print(f"  ΔZ = {dz:.1f}mm  {'✅ 接近' if dz<2 else '⚠️ 偏差大'}")
            print()
            print(f"  EMA α={ema_a}  帧数={fc}  基线={baseline*1000:.0f}mm")
        else:
            print("  等待 ID0 + ID2 标签...")
        print("=" * 58)

    # Draw
    for tid, c in dl.items():
        clr = {0:(255,0,0), 1:(255,0,255), 2:(0,0,255)}.get(tid, (0,255,0))
        cv2.polylines(frame_l, [c.astype(int)], True, clr, 2)
        cv2.putText(frame_l, f"ID{tid}", (int(c[0][0]), int(c[0][1])-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)

    cv2.imshow("Stereo vs PnP", frame_l)
    if cv2.waitKey(1) == 27:
        break

zed.close()
cv2.destroyAllWindows()
