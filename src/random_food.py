from __future__ import print_function

import time
import numpy as np
import vrep
import robobo
import cv2
import sys
import signal
import prey
import vrep
import random

ClientID = '192.168.0.185'


def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)


def main():
    signal.signal(signal.SIGINT, terminate_program)

    rob = robobo.SimulationRobobo().connect(address= ClientID, port=19997)
    _, handlewallXone = vrep.simxGetObjectHandle(rob._clientID, '80cmHighWall200cm', vrep.simx_opmode_blocking)
    _, handlewallXtwo = vrep.simxGetObjectHandle(rob._clientID,'80cmHighWall200cm4', vrep.simx_opmode_blocking)
    _, posXone = vrep.simxGetObjectPosition(rob._clientID,handlewallXone, -1, vrep.simx_opmode_blocking)
    _, posXtwo = vrep.simxGetObjectPosition(rob._clientID,handlewallXtwo, -1, vrep.simx_opmode_blocking)
    _, handlewallYone = vrep.simxGetObjectHandle(rob._clientID,'80cmHighWall200cm3', vrep.simx_opmode_blocking)
    _, handlewallYtwo = vrep.simxGetObjectHandle(rob._clientID,'80cmHighWall200cm7', vrep.simx_opmode_blocking)
    _, posYone = vrep.simxGetObjectPosition(rob._clientID,handlewallYone, -1, vrep.simx_opmode_blocking)
    _, posYtwo = vrep.simxGetObjectPosition(rob._clientID,handlewallYtwo, -1, vrep.simx_opmode_blocking)

    for i in range(0,9):
        _, handle = vrep.simxGetObjectHandle(rob._clientID, 'Food'+str(i), vrep.simx_opmode_blocking)
        xpos = random.uniform(posXone[0] + 0.3, posXtwo[0] - 0.3)
        ypos = random.uniform(posYone[1] - 0.3, posYtwo[1] + 0.3)
        vrep.simxSetObjectPosition(rob._clientID, handle, -1, [xpos, ypos, 0.04], vrep.simx_opmode_oneshot)



    print("robobo is at {}".format(rob.position()))

    for i in range(3):
        print("robobo is at {}".format(rob.position()))
        rob.move(10, 10, 2000)


    time.sleep(0.1)

    # IR reading

    # pause the simulation and read the collected food

    # Stopping the simualtion resets the environment
    rob.stop_world()


if __name__ == "__main__":
    main()