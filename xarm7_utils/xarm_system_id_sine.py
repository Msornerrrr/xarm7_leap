import os
import sys
import time
import math
import csv
import signal
import threading
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt

from xarm.wrapper import XArmAPI


# ==============================
# User Config (edit safely)
# ==============================
IP = "192.168.1.239"
DT = 0.05                    # control/measurement period (s) -> 20 Hz
SPEED = 100                  # xArm linear speed (mm/s) for set_position
ACC = None                   # keep default accel (None) or set a value
DURATION = 30.0              # total trajectory time (s)
START_POSE = [400, 0, 150, math.pi, 0.0, 0.0]  # x,y,z(mm), r,p,y (rad)

# Safety boundary (RTH table example); adjust to your setup
# Order: [x_max, x_min, y_max, y_min, z_max, z_min] in mm
SAFETY_BOUNDS = [650, 200, 250, -250, 350, -50]

# Sine parameters per axis (x,y,z,roll,pitch,yaw)
# amplitudes in mm for xyz, radians for rpy
# frequencies in Hz
# phases in radians
# TRAJ_AMPLITUDES = [50.0, 50.0, 50.0, math.pi/6, math.pi/6, math.pi/6]
TRAJ_AMPLITUDES = [50.0, 50.0, 50.0, 0, 0, 0]
# TRAJ_AMPLITUDES = [0.0, 5.0, 0.0, math.pi/6, math.pi/6, math.pi/6]
TRAJ_FREQS     = [0.10, 0.12, 0.08, 0.10, 0.12, 0.15]
TRAJ_PHASES    = [0.0,  math.pi/3, -math.pi/4, 0.0, math.pi/6, -math.pi/8]

# Logging
LOG_DIR = "./logs"
LOG_BASENAME = "xarm_system_id"
PLOT_FIG = True


# ==============================
# Small utilities
# ==============================
def now_ms() -> float:
    return time.time() * 1000.0


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def angle_diff(a: float, b: float) -> float:
    """Return the signed minimal difference between two angles in radians."""
    d = (a - b + math.pi) % (2*math.pi) - math.pi
    return d


@dataclass
class Trajectory:
    t: np.ndarray           # shape (T,)
    cmd: np.ndarray         # shape (T, 6) -> [x,y,z,r,p,y]
    base: np.ndarray        # base pose used to build cmd (6,)


@dataclass
class ReplayResult:
    t: List[float]          # absolute timestamps (s since epoch) at sample
    t_rel: List[float]      # relative time since start (s)
    cmd: List[List[float]]  # commanded [x,y,z,r,p,y]
    meas: List[List[float]] # measured [x,y,z,r,p,y]
    codes: List[int]        # return codes from set_position (for debugging)


# ==============================
# Trajectory generation
# ==============================
def generate_sine_trajectory(
    duration: float,
    dt: float,
    base_pose: List[float],
    amplitudes: List[float],
    freqs: List[float],
    phases: List[float],
    safety_bounds: List[float] = None
) -> Trajectory:
    """
    Build a 6-DoF sine trajectory around base_pose.
    base_pose: [x,y,z,r,p,y] where xyz in mm, rpy in rad
    amplitudes: same units per axis
    freqs: Hz per axis
    phases: rad per axis
    """
    assert len(base_pose) == 6
    assert len(amplitudes) == 6
    assert len(freqs) == 6
    assert len(phases) == 6

    t = np.arange(0.0, duration + 1e-9, dt)
    base = np.array(base_pose, dtype=np.float64)
    cmd = np.zeros((len(t), 6), dtype=np.float64)

    for i, ti in enumerate(t):
        for d in range(6):
            cmd[i, d] = base[d] + amplitudes[d] * math.sin(2.0 * math.pi * freqs[d] * ti + phases[d])

        # Optional cartesian clamp into reduced safety bounds
        if safety_bounds is not None:
            x_max, x_min, y_max, y_min, z_max, z_min = safety_bounds
            cmd[i, 0] = clamp(cmd[i, 0], x_min, x_max)
            cmd[i, 1] = clamp(cmd[i, 1], y_min, y_max)
            cmd[i, 2] = clamp(cmd[i, 2], z_min, z_max)

    return Trajectory(t=t, cmd=cmd, base=base)


# ==============================
# Arm helpers
# ==============================
def prep_arm(arm: XArmAPI, ip: str):
    """Connect, clear, set reduced mode, enable, set Mode 7."""
    print(f"[INFO] Connecting to xArm at {ip} ...")
    arm.connect(ip)
    time.sleep(0.5)

    # Clean residual warnings/errors
    arm.clean_warn()
    arm.clean_error()

    # Reduced mode + boundary
    print("[INFO] Enabling reduced mode & TCP boundary.")
    arm.set_reduced_tcp_boundary(SAFETY_BOUNDS)
    arm.set_reduced_mode(True)

    time.sleep(0.25)

    print("[INFO] Enabling motion, setting mode/state.")
    arm.motion_enable(enable=True)
    arm.set_mode(7)   # position mode (linear planning)
    arm.set_state(0)

    # Optional: set TCP offset if you have a tool mounted
    # arm.set_tcp_offset([0, 0, 120, 0, 0, 0], is_radian=True)


def move_to_start(arm: XArmAPI, start_pose: List[float], speed: float = SPEED,
                  pos_tol_mm: float = 2.0, rpy_tol_rad: float = 0.02,
                  max_wait_s: float = 15.0, check_bounds: List[float] = SAFETY_BOUNDS):
    """Command start pose and poll until within tolerance or timeout."""
    print(f"[INFO] Moving to start pose: {start_pose}")

    t0 = time.time()
    while True:
        code = arm.set_position(
            x=float(start_pose[0]), y=float(start_pose[1]), z=float(start_pose[2]),
            roll=float(start_pose[3]), pitch=float(start_pose[4]), yaw=float(start_pose[5]),
            speed=float(speed), is_radian=True
        )
        if code != 0:
            # show controller error/warn if available
            try:
                ec, wc = arm.get_err_warn_code()
                print(f"[DEBUG] get_err_warn_code -> error={ec}, warn={wc}")
            except Exception:
                pass
            raise RuntimeError(f"set_position() failed immediately, code={code}")

        time.sleep(0.05)

        m_code, m_pose = arm.get_position(is_radian=True)
        if m_code != 0 or m_pose is None:
            # print diagnostic when polling fails
            try:
                ec, wc = arm.get_err_warn_code()
                print(f"[WARN] get_position code={m_code}, controller err={ec}, warn={wc}")
            except Exception:
                print(f"[WARN] get_position code={m_code}")
        else:
            # check tolerances
            dx = abs(m_pose[0] - start_pose[0])
            dy = abs(m_pose[1] - start_pose[1])
            dz = abs(m_pose[2] - start_pose[2])
            dr = abs(angle_diff(m_pose[3], start_pose[3]))
            dp = abs(angle_diff(m_pose[4], start_pose[4]))
            dyaw = abs(angle_diff(m_pose[5], start_pose[5]))

            if (dx <= pos_tol_mm and dy <= pos_tol_mm and dz <= pos_tol_mm and
                dr <= rpy_tol_rad and dp <= rpy_tol_rad and dyaw <= rpy_tol_rad):
                print("[INFO] Reached start pose within tolerance.")
                return

        if time.time() - t0 > max_wait_s:
            # give detailed diagnostics before failing
            try:
                ec, wc = arm.get_err_warn_code()
                state = arm.state
                mode = arm.mode
                print(f"[DEBUG] After timeout: state={state}, mode={mode}, err={ec}, warn={wc}")
            except Exception:
                pass
            raise TimeoutError(f"Timed out ({max_wait_s}s) reaching start pose (avoid SDK wait timeout=100).")


def get_pose(arm: XArmAPI) -> Tuple[int, List[float]]:
    """Get current TCP pose [x,y,z,r,p,y] in mm/rad."""
    code, pos = arm.get_position(is_radian=True)
    # pos is typically [x, y, z, roll, pitch, yaw]
    return code, pos


# ==============================
# Trajectory replay (poll + stream)
# ==============================
def replay_trajectory(
    arm: XArmAPI,
    traj: Trajectory,
    dt: float,
    speed: float = SPEED,
    acc: float = None
) -> ReplayResult:
    """
    Stream set_position at fixed dt and log measured pose.
    """
    result = ReplayResult(t=[], t_rel=[], cmd=[], meas=[], codes=[])
    t0 = time.time()
    next_tick = t0

    print("[INFO] Starting trajectory replay ... (Ctrl+C to stop)")
    try:
        for i in range(len(traj.t)):
            # Time sync
            now = time.time()
            if now < next_tick:
                time.sleep(next_tick - now)
            now = time.time()  # update after sleep
            t_rel = now - t0
            next_tick = now + dt

            # Command
            cx, cy, cz, cr, cp, cyaw = traj.cmd[i]
            kwargs = dict(
                x=float(cx), y=float(cy), z=float(cz),
                roll=float(cr), pitch=float(cp), yaw=float(cyaw),
                speed=float(speed), is_radian=True, wait=False
            )
            if acc is not None:
                kwargs["acc"] = float(acc)
            code = arm.set_position(**kwargs)

            # Measure right after sending
            m_code, m_pose = get_pose(arm)

            # Log
            result.t.append(now)
            result.t_rel.append(t_rel)
            result.codes.append(code)
            result.cmd.append([cx, cy, cz, cr, cp, cyaw])
            result.meas.append(m_pose if m_pose is not None else [np.nan]*6)

            # Early exit if SDK reports a serious issue
            if code not in (0,):  # treat non-zero as issue to inspect
                print(f"[WARN] set_position returned {code} at i={i}")
                # not breaking immediately; you can choose to break here
    except KeyboardInterrupt:
        print("\n[INFO] Trajectory interrupted by user.")
    finally:
        print("[INFO] Replay finished.")
    return result


# ==============================
# Logging + plotting
# ==============================
def save_log_csv(path: str, result: ReplayResult):
    header = [
        "t_abs_s", "t_rel_s",
        "cmd_x", "cmd_y", "cmd_z", "cmd_r", "cmd_p", "cmd_yaw",
        "meas_x", "meas_y", "meas_z", "meas_r", "meas_p", "meas_yaw",
        "sdk_code"
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(len(result.t)):
            row = [
                result.t[i], result.t_rel[i],
                *result.cmd[i],
                *result.meas[i],
                result.codes[i],
            ]
            writer.writerow(row)
    print(f"[INFO] Log saved to {path}")


def plot_cmd_vs_meas(result: ReplayResult, title_suffix: str = ""):
    t = np.array(result.t_rel)
    cmd = np.array(result.cmd)
    meas = np.array(result.meas)

    labels = ["x (mm)", "y (mm)", "z (mm)", "roll (rad)", "pitch (rad)", "yaw (rad)"]

    # Command vs. measured
    fig1, axs = [], []
    for d in range(6):
        plt.figure()
        plt.plot(t, cmd[:, d], label="commanded")
        plt.plot(t, meas[:, d], label="measured", linestyle="--")
        plt.xlabel("time (s)")
        plt.ylabel(labels[d])
        plt.title(f"{labels[d]} — commanded vs measured{title_suffix}")
        plt.legend()
        plt.grid(True)

    # Error plots
    err = meas - cmd
    for d in range(6):
        plt.figure()
        plt.plot(t, err[:, d])
        plt.xlabel("time (s)")
        plt.ylabel(f"error {labels[d]}")
        plt.title(f"Tracking error — {labels[d]}{title_suffix}")
        plt.grid(True)

    plt.show()


# ==============================
# Main
# ==============================
def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    arm = XArmAPI(IP, is_radian=True)

    # graceful shutdown on Ctrl+C
    stop_event = threading.Event()

    def handle_sigint(sig, frame):
        stop_event.set()
        print("\n[INFO] Caught SIGINT, attempting clean stop ...")
        exit(0)
    signal.signal(signal.SIGINT, handle_sigint)

    try:
        prep_arm(arm, IP)

        # Option A: use configured START_POSE
        # Option B: use current pose as base:
        # _, cur_pose = get_pose(arm); base_pose = cur_pose
        base_pose = START_POSE

        move_to_start(arm, base_pose, speed=SPEED)

        traj = generate_sine_trajectory(
            duration=DURATION,
            dt=DT,
            base_pose=base_pose,
            amplitudes=TRAJ_AMPLITUDES,
            freqs=TRAJ_FREQS,
            phases=TRAJ_PHASES,
            safety_bounds=SAFETY_BOUNDS
        )

        result = replay_trajectory(
            arm=arm,
            traj=traj,
            dt=DT,
            speed=SPEED,
            acc=ACC
        )

        # Save CSV
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(LOG_DIR, f"{LOG_BASENAME}_{timestamp}.csv")
        save_log_csv(csv_path, result)

        # Plot
        if PLOT_FIG:
            plot_cmd_vs_meas(result, title_suffix=f" (dt={DT}s)")
    finally:
        try:
            arm.set_state(0)
        except Exception:
            pass
        try:
            arm.disconnect()
        except Exception:
            pass
        print("[INFO] Disconnected.")


if __name__ == "__main__":
    main()
