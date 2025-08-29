# mapper.py
import numpy as np
from leap_python import leap_hand_utils as lhu


def map_teleop_to_leap(latest_joint_positions):
    """
    Combine all retargeting to convert Allegro-style joints from
    AllegroHandTeleop.get_latest_joint_positions() -> LEAP real-hand angles,
    ready for LeapNode.set_leap(...).

    Steps:
      1) Reorder joints to [8..11, 4..7, 0..3, 12..15] (your calc_teleop_targets order)
      2) Apply the two LEAP-specific swaps you had in code
      3) Convert Allegro -> LEAP real angles
      4) Safety-clip to LEAP hardware limits
    """
    if latest_joint_positions is None:
        raise ValueError("No joint data from teleop.")

    # 1) Reorder like your calc_teleop_targets()
    idx = np.array([8, 9,10,11,   4, 5, 6, 7,   0, 1, 2, 3,  12,13,14,15], dtype=int)
    x = np.asarray(latest_joint_positions, dtype=np.float32)[idx].copy()

    # 2) LEAP-specific swaps (mirror your in-place logic exactly)
    # swap slices [0:3] <-> [8:11]
    tmp = x[0:3].copy()
    x[0:3] = x[8:11]
    x[8:11] = tmp

    # swap elements [0,4,8] <-> [1,5,9]
    tmp = x[[0, 4, 8]].copy()
    x[[0, 4, 8]] = x[[1, 5, 9]]
    x[[1, 5, 9]] = tmp

    # 3) Convert Allegro -> LEAP real angles (do NOT zero MCPs)
    leap_q = lhu.allegro_to_LEAPhand(x, zeros=False)

    # 4) Safety-clip to real LEAP limits
    leap_q = lhu.angle_safety_clip(leap_q)
    return leap_q
