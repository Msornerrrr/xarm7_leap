import time

from leap_python.main import LeapNode
from xarm.wrapper import XArmAPI
from leap_utils.leap_teleop_thread import leap_teleop_thread


def main():
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

    hand = LeapNode()

