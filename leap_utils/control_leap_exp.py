import numpy as np
import time

from leap_python import LeapNode


def main():
    # Initialize Leap Hand
    leap_hand = LeapNode()

    # Define some test poses
    HOME_POSE = np.zeros(16)  # fully open Allegro pose, converted to LEAP
    CLOSED_POSE = HOME_POSE + np.array([
        0.2, 0.3, 0.3, 0.3,   # index
        0.2, 0.3, 0.3, 0.3,   # middle
        0.2, 0.3, 0.3, 0.3,   # ring
        0.4, 0.3, 0.3, 0.3    # thumb (slightly stronger close)
    ])

    print("Starting LEAP hand test...")
    print("Moving to home position.")
    leap_hand.set_leap(HOME_POSE)
    time.sleep(2.0)

    while True:
        # Close hand
        print("Closing hand...")
        leap_hand.set_leap(CLOSED_POSE)

        time.sleep(2.0)
        print("Current pos:", leap_hand.read_pos())

        # Open hand
        print("Opening hand...")
        leap_hand.set_leap(HOME_POSE)
        time.sleep(2.0)
        print("Current pos:", leap_hand.read_pos())


if __name__ == "__main__":
    main()
