# teleop_xarm.py
import time
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from threading import Event

from xarm.wrapper import XArmAPI
from vive_teleop import ArmTeleop
from rokoko_teleop.const import LINUX_IP, VIVE_PORT  # your network constants

# -------------------- CONFIG --------------------
ROLE = "right_elbow"
CTRL_FREQ_HZ = 20.0                # 20 Hz
DT = 1.0 / CTRL_FREQ_HZ

# Scale from hand motion to robot TCP motion
POS_GAIN = 1.0                     # 1.0 => 1 m hand -> 1000 mm robot
ROT_GAIN = 1.0                     # 1.0 => apply tracker relative rotation as-is

# Map from ArmTeleop (PyBullet frame) to robot base frame (tweak if axes differ)
# Suppose teleop gives pos_rel_pb (X right, Y forward, Z up)
# XArm base: X forward, Y left, Z up   (often!)
PB_TO_ROBOT_R_POS = np.array([
    [ 0, 1, 0],   # teleop.Y → robot.X
    [-1, 0, 0],   # teleop.X → robot.-Y
    [ 0, 0, 1],   # teleop.Z → robot.Z
], dtype=np.float32)
PB_TO_ROBOT_R_ORI = np.array([
    [ 0,  1, 0],   # teleop.Y → robot.X
    [ 1,  0, 0],   # teleop.X → robot.Y
    [ 0,  0, -1],  # teleop.Z → robot.-Z
], dtype=np.float32)

# Safety (per-step clamp & emergency thresholds)
STEP_TRANS_CLAMP_MM = 20.0         # clamp any single update to <= 20 mm step
STEP_ROT_CLAMP_DEG = 10.0          # clamp single update to <= 10 deg step
EMERGENCY_TRANS_JUMP_MM = 120.0    # if a step tries to jump >120 mm -> abort
EMERGENCY_ROT_JUMP_DEG = 45.0      # if a step tries to jump >45 deg  -> abort

# Workspace soft limits (None = skip). Tuple of (min, max) in robot base frame.
X_LIMIT_MM = (100.0, 700.0)
Y_LIMIT_MM = (-350.0, 350.0)
Z_LIMIT_MM = (  10.0, 500.0)

# Motion params
XARM_SPEED = 150.0                 # mm/s
# ------------------------------------------------


def clamp(val, lo, hi):
    return min(max(val, lo), hi)


def clamp_vec3_mm(vec, step_clamp_mm):
    n = float(np.linalg.norm(vec))
    if n <= step_clamp_mm or n == 0.0:
        return vec
    return vec * (step_clamp_mm / n)


def enforce_workspace_mm(p_mm):
    x, y, z = p_mm
    if X_LIMIT_MM is not None:
        x = clamp(x, X_LIMIT_MM[0], X_LIMIT_MM[1])
    if Y_LIMIT_MM is not None:
        y = clamp(y, Y_LIMIT_MM[0], Y_LIMIT_MM[1])
    if Z_LIMIT_MM is not None:
        z = clamp(z, Z_LIMIT_MM[0], Z_LIMIT_MM[1])
    return np.array([x, y, z], dtype=np.float32)


def main():
    stop_evt = Event()

    # ---------------- Robot init ----------------
    ip = "192.168.1.239"
    arm = XArmAPI(ip, is_radian=True)
    time.sleep(1) # Give arm time to initialize

    # Safety features from your snippet
    # (Keep your own SAFETY_BOUNDS if you like.)
    arm.set_reduced_tcp_boundary([650, 100, 400, -400, 400, 10])  # if desired
    arm.set_reduced_mode(True)
    time.sleep(1) # Give arm time to initialize

    arm.motion_enable(enable=True)
    arm.set_mode(7)
    arm.set_state(state=0)

    # Go to a reasonable base pose (or comment if you already placed it there)
    # Units: mm + rad
    ret = arm.set_position(
        x=350, y=0, z=150,
        roll=3.14, pitch=0.0, yaw=0.0,
        speed=XARM_SPEED, is_radian=True
    )
    if ret != 0:
        print("[WARN] set_position to base returned code:", ret)
    time.sleep(1.0)

    # Read back as BASE pose (mm, rad)
    code, pose = arm.get_position(is_radian=True)  # pose: [x, y, z, roll, pitch, yaw]
    if code != 0 or pose is None:
        # fallback to the commanded base if get_position not available
        base_pos_mm = np.array([350.0, 0.0, 150.0], dtype=np.float32)
        base_rpy_rad = np.array([3.14, 0.0, 0.0], dtype=np.float32)
        print("[WARN] get_position failed, using commanded base pose.")
    else:
        base_pos_mm = np.array(pose[:3], dtype=np.float32)
        base_rpy_rad = np.array(pose[3:6], dtype=np.float32)
    base_R = Rotation.from_euler('xyz', base_rpy_rad, degrees=False)

    # --------------- VIVE teleop init ---------------
    tele = ArmTeleop(LINUX_IP, VIVE_PORT, role=ROLE, visualize=False)
    tele.start()

    # --------------- Control loop -------------------
    print("[INFO] Teleop running at %.1f Hz. Ctrl-C to stop." % CTRL_FREQ_HZ)

    last_cmd_pos_mm = base_pos_mm.copy()
    last_cmd_R = base_R
    first = True

    try:
        while not stop_evt.is_set():
            t0 = time.time()

            pos_rel_m, quat_rel_wxyz = tele.get_latest_pose()
            if pos_rel_m is None:
                # no fresh data, hold
                time.sleep(DT)
                continue

            # ---- Build desired absolute pose in ROBOT frame ----
            # Position
            #  - ArmTeleop returns PB-frame meters; map -> robot & scale
            dp_robot_m = PB_TO_ROBOT_R_POS @ pos_rel_m
            dp_robot_mm = (POS_GAIN * 1000.0) * dp_robot_m
            target_pos_mm = base_pos_mm + dp_robot_mm

            # Orientation
            R_rel_pb = Rotation.from_quat([  # ArmTeleop returns [w,x,y,z]
                quat_rel_wxyz[1], quat_rel_wxyz[2], quat_rel_wxyz[3], quat_rel_wxyz[0]
            ])
            # If PB frame ≠ robot frame, lift R_rel into robot frame via PB_TO_ROBOT_R_ORI
            R_pb2rb = Rotation.from_matrix(PB_TO_ROBOT_R_ORI)
            R_rel_robot = R_pb2rb * R_rel_pb * R_pb2rb.inv()
            if ROT_GAIN != 1.0:  # interpolate only if scaling rotation
                slerp = Slerp([0, 1], Rotation.concatenate([Rotation.identity(), R_rel_robot]))
                R_rel_robot = slerp([ROT_GAIN])[0]
            target_R = base_R * R_rel_robot
            target_rpy_rad = target_R.as_euler('xyz', degrees=False)

            # ---- Safety: per-step clamp & emergency stop ----
            step_dp = target_pos_mm - last_cmd_pos_mm
            step_dn = float(np.linalg.norm(step_dp))
            if step_dn > EMERGENCY_TRANS_JUMP_MM:
                print("[EMERGENCY] Translation jump (%.1f mm) too large. Stopping." % step_dn)
                break
            step_dp_clamped = clamp_vec3_mm(step_dp, STEP_TRANS_CLAMP_MM)

            # Angular step check (use geodesic angle)
            dR = last_cmd_R.inv() * target_R
            step_deg = np.degrees(np.linalg.norm(dR.as_rotvec()))
            if step_deg > EMERGENCY_ROT_JUMP_DEG:
                print("[EMERGENCY] Rotation jump (%.1f deg) too large. Stopping." % step_deg)
                break
            # Clamp rotation per-step
            if step_deg > STEP_ROT_CLAMP_DEG and step_deg > 1e-6:
                scale = STEP_ROT_CLAMP_DEG / step_deg
                dR = Rotation.from_rotvec(dR.as_rotvec() * scale)
                target_R = last_cmd_R * dR
                target_rpy_rad = target_R.as_euler('xyz', degrees=False)

            # Apply position clamp after safety
            target_pos_mm = last_cmd_pos_mm + step_dp_clamped

            # Workspace soft limits
            target_pos_mm = enforce_workspace_mm(target_pos_mm)

            # ---- Command robot ----
            ret = arm.set_position(
                x=float(target_pos_mm[0]),
                y=float(target_pos_mm[1]),
                z=float(target_pos_mm[2]),
                roll=float(target_rpy_rad[0]),
                pitch=float(target_rpy_rad[1]),
                yaw=float(target_rpy_rad[2]),
                speed=XARM_SPEED,
                is_radian=True
            )
            if ret != 0:
                print("[WARN] set_position returned code:", ret)

            # Update last commanded
            last_cmd_pos_mm = target_pos_mm
            last_cmd_R = target_R

            # Sleep to hold 20 Hz
            elapsed = time.time() - t0
            remain = DT - elapsed
            if remain > 0:
                time.sleep(remain)

            # (Optional) print first target for sanity
            if first:
                print("[INFO] Base mm:", base_pos_mm, "  Base rpy rad:", base_rpy_rad)
                print("[INFO] First target mm:", target_pos_mm, "  rpy:", target_rpy_rad)
                first = False

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt. Stopping.")
    finally:
        # tidy
        try:
            arm.set_state(0)
        except Exception:
            pass
        try:
            arm.disconnect()
        except Exception:
            pass
        tele.stop()
        print("[INFO] Teleop stopped.")


if __name__ == "__main__":
    main()
