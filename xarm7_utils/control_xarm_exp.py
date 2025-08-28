import numpy as np
import time
import signal 
import threading 
import sys 
import time
import os

from xarm.wrapper import XArmAPI


### Global &  Constant Variables
ip="192.168.1.239"
speed = 100
dt = 0.05
SAFETY_BOUNDS = [650, 100, 300, -300, 400, 10] # TBD
ee_origin_default_offset = [0, 0, 120, 0, 0, 0] # xyz (mm), rpy (rad)


### Sample Code
arm = XArmAPI(ip, is_radian=True)

time.sleep(1) # Give arm time to initialize

# arm.set_tcp_offset(ee_origin_default_offset, is_radian=True)

# add safety checks
arm.set_reduced_tcp_boundary(SAFETY_BOUNDS)
arm.set_reduced_mode(True)

time.sleep(1) # Give arm time to initialize

arm.motion_enable(enable=True)
arm.set_mode(7)
arm.set_state(state=0)

arm.set_position(x=350, y=0, z=150, roll=3.14, pitch=0, yaw=0, speed=speed, is_radian=True)
