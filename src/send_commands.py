#!/usr/bin/env python3
from __future__ import print_function

import time
import numpy as np

import robobo
import cv2
import sys
import signal
import prey


class Direction:
    LEFT = (15, 30, 300)  # Action: 0
    RIGHT = (30, 15, 300)  # Action: 1
    FORWARD = (20, 20, 300)  # Action: 2
    RRIGHT = (-15, -30, 300)  # Action: 3
    LLEFT = (-30, -15, 300)  # Action: 4


def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)


def main():
    signal.signal(signal.SIGINT, terminate_program)
    rob = robobo.SimulationRobobo().connect(address='192.168.1.3', port=19997)

    rob.play_simulation()

    image = rob.get_image_front()
    cv2.imwrite("test_pictures.png", image)

    # Following code moves the robot
    for i in range(1):
        # rob.move(-10, -10, 50)
        # rob.move(-25, 25, 300)
        # rob.move(25, 25, 300)
        # rob.move(30, 30, 100)
        rob.move(15, 30, 300)
        # rob.move(-100, -25, 100)



if __name__ == "__main__":
    main()
