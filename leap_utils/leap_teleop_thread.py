from rokoko_teleop import AllegroHandTeleop
from leap_python.main import LeapNode
from mapper import map_teleop_to_leap
import time


def leap_teleop_thread(leap: LeapNode):
    teleop = AllegroHandTeleop(visualize=False, kinematic_rescale=False, teleop_wrist=False)
    teleop.start(frequency=30)

    try:
        while True:
            allegro_q = teleop.get_latest_joint_positions()
            if allegro_q is not None:
                leap_q = map_teleop_to_leap(allegro_q)
                leap.set_leap(leap_q)
            time.sleep(1/30)
    except KeyboardInterrupt:
        teleop.stop()
