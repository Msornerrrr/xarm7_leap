# teleop_xarm.py
import time
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from threading import Event

from xarm.wrapper import XArmAPI
from vive_teleop import ArmTeleop
from rokoko_teleop.const import LINUX_IP, VIVE_PORT  # your network constants
from xarm_teleop_thread import arm_teleop_thread


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

arm_teleop_thread(arm)
