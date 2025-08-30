from rokoko_teleop import AllegroHandTeleop
from leap_python.main import LeapNode
from mapper import map_teleop_to_leap
from leap_teleop_thread import leap_teleop_thread


leap = LeapNode()
leap_teleop_thread(leap)
