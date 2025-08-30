import time
import threading

from leap_python.main import LeapNode
from xarm.wrapper import XArmAPI
from leap_utils.leap_teleop import leap_teleop_thread
from xarm7_utils.xarm_teleop import xarm_teleop_thread


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

    # Start xArm teleop thread
    xarm_thread = threading.Thread(target=xarm_teleop_thread, args=(arm,))
    xarm_thread.start()

    # Start Leap teleop thread
    leap_thread = threading.Thread(target=leap_teleop_thread, args=(hand,))
    leap_thread.start()

    input("Press Enter to stop...\n")
    leap_thread.join()
    xarm_thread.join()
    print("Exiting...")
    time.sleep(1)
    
if __name__ == "__main__":
    main()
